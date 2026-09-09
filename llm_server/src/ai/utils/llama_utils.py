"""Python helpers for the llama.cpp C++ bridge.

This module is the Python-visible ABI and small utilities used by the local
LLM stack:

- Resolve Homebrew's ``libllama.dylib`` so ``ctypes.CDLL`` can load llama.cpp.
- Declare ``argtypes`` / ``restype`` for every ``llama_bridge_*`` symbol so
  pointers are not truncated to 32-bit ``c_int``.
- Initialize the llama.cpp backend once per process.
- Slice a JSON object out of model text that may include a ``<think>`` block.

Nothing here loads GGUF weights or runs a forward pass. ``inference.py``
owns ``CDLL`` and the model pointer; this file only prepares and assists.
"""

import subprocess  # run `brew --prefix llama.cpp` without a shell
from pathlib import Path  # join the keg prefix with lib/libllama.dylib
import ctypes  # C type objects used in argtypes / restype declarations
import json  # parse the sliced {...} object in extract_json


def find_llama_library() -> str:
    """Return the absolute path to Homebrew's ``libllama.dylib``.

    Runs ``brew --prefix llama.cpp`` and appends ``lib/libllama.dylib``.
    Called once at import time from ``config.config`` to set
    ``LLAMA_LIBRARY_PATH``. Does not ``dlopen`` the library.

    Returns:
        Filesystem path suitable for ``ctypes.CDLL``.

    Raises:
        FileNotFoundError: ``brew`` is not on ``PATH``.
        subprocess.CalledProcessError: the ``llama.cpp`` formula is not installed.
    """
    brew_prefix = subprocess.run(  # ask Homebrew where the llama.cpp keg lives
        ["brew", "--prefix", "llama.cpp"],  # argv: no shell, formula name is literal
        capture_output=True,  # keep stdout/stderr in the CompletedProcess
        text=True,  # decode stdout as str instead of bytes
        check=True,  # raise if brew exits non-zero (formula missing)
    )
    return str(  # CDLL requires a str path, not a Path
        Path(brew_prefix.stdout.strip())  # drop the trailing newline from brew
        / "lib"  # conventional dylib directory inside the keg
        / "libllama.dylib"  # shared library inference.py will dlopen
    )


def define_llama_bridge_signatures(bridge: ctypes.CDLL) -> ctypes.CDLL:
    """Attach C argument and return types to every ``llama_bridge_*`` symbol.

    ctypes defaults to ``c_int`` returns. That truncates 64-bit pointers from
    ``llama_bridge_create_context`` and ``llama_bridge_model_load``. This
    function mutates ``bridge`` in place and returns the same object so
    ``localLLM.__init__`` can assign it in one line.

    Args:
        bridge: ``CDLL`` handle for ``libllama_bridge.dylib``, already loaded.

    Returns:
        The same ``bridge`` object, now with signatures set.
    """
    bridge.llama_bridge_init.argtypes = []  # void llama_bridge_init(void)
    bridge.llama_bridge_init.restype = None  # no return; calls llama_backend_init

    bridge.llama_bridge_model_load.argtypes = [ctypes.c_char_p]  # const char *path
    bridge.llama_bridge_model_load.restype = ctypes.c_void_p  # llama_model *; must be 64-bit

    bridge.llama_bridge_model_free.argtypes = [ctypes.c_void_p]  # llama_model * to release
    bridge.llama_bridge_model_free.restype = None  # void

    bridge.llama_bridge_model_size.argtypes = [ctypes.c_void_p]  # llama_model *
    bridge.llama_bridge_model_size.restype = ctypes.c_uint64  # weight size in bytes

    bridge.llama_bridge_model_n_params.argtypes = [ctypes.c_void_p]  # llama_model *
    bridge.llama_bridge_model_n_params.restype = ctypes.c_uint64  # parameter count (can exceed 2^31)

    bridge.llama_bridge_shutdown.argtypes = []  # void llama_bridge_shutdown(void)
    bridge.llama_bridge_shutdown.restype = None  # calls llama_backend_free

    bridge.llama_bridge_tokenize.argtypes = [
        ctypes.c_void_p,  # llama_model * (vocab lives on the model)
        ctypes.c_char_p,  # UTF-8 prompt text
        ctypes.POINTER(ctypes.c_int),  # output ids; Python None becomes NULL for the probe call
        ctypes.c_int,  # capacity of that array, or 0 when probing
    ]
    bridge.llama_bridge_tokenize.restype = ctypes.c_int  # +n written, or -n if the buffer is too small

    bridge.llama_bridge_create_context.argtypes = [
        ctypes.c_void_p,  # llama_model * the context belongs to
        ctypes.c_int,  # n_ctx / n_batch capacity in tokens
    ]
    bridge.llama_bridge_create_context.restype = ctypes.c_void_p  # llama_context *; empty until decode

    bridge.llama_bridge_free_context.argtypes = [
        ctypes.c_void_p,  # llama_context * allocated by create_context
    ]
    bridge.llama_bridge_free_context.restype = None  # void; safe no-op on NULL in C++

    bridge.llama_bridge_decode_prompt.argtypes = [
        ctypes.c_void_p,  # llama_context * mutated in place (KV cache)
        ctypes.POINTER(ctypes.c_int),  # token id array from tokenize
        ctypes.c_int,  # how many tokens to decode
    ]
    bridge.llama_bridge_decode_prompt.restype = ctypes.c_int  # 0 success; not a token id

    bridge.llama_bridge_sample_greedy.argtypes = [
        ctypes.c_void_p,  # llama_context * whose last-position logits are read
    ]
    bridge.llama_bridge_sample_greedy.restype = ctypes.c_int  # next token id (argmax), not a status

    bridge.llama_bridge_token_to_piece.argtypes = [
        ctypes.c_void_p,  # llama_model * (vocab)
        ctypes.c_int,  # token id to detokenize
        ctypes.c_char_p,  # destination UTF-8 buffer
        ctypes.c_int,  # buffer capacity in bytes
    ]
    bridge.llama_bridge_token_to_piece.restype = ctypes.c_int  # bytes written, or -needed if too small

    bridge.llama_bridge_decode_one.argtypes = [
        ctypes.c_void_p,  # llama_context * to append one KV position
        ctypes.c_int,  # token id just sampled
    ]
    bridge.llama_bridge_decode_one.restype = ctypes.c_int  # 0 success; mutates ctx, does not return text

    bridge.llama_bridge_is_eog.argtypes = [
        ctypes.c_void_p,  # llama_model * (vocab)
        ctypes.c_int,  # candidate token id
    ]
    bridge.llama_bridge_is_eog.restype = ctypes.c_bool  # True if this id ends generation

    return bridge  # same object; signatures are attributes on the CDLL


def init_llama(bridge: ctypes.CDLL) -> ctypes.CDLL:
    """Call ``llama_bridge_init`` so llama.cpp's global backend is ready.

    Must run after ``define_llama_bridge_signatures`` and before
    ``llama_bridge_model_load``. Pair with ``llama_bridge_shutdown`` in
    ``localLLM.close``.

    Args:
        bridge: Signed ``CDLL`` handle for the C++ bridge.

    Returns:
        The same ``bridge`` object, after ``llama_backend_init`` has run.
    """
    bridge.llama_bridge_init()  # native: llama_backend_init(); no GGUF I/O
    return bridge  # caller rebinds self.llama_bridge to this same handle


def extract_json(response: str) -> dict:
    """Parse the first-to-last ``{...}`` object in a model response.

    Qwen may emit a ``<think>`` block before the JSON the agent loop
    understands. ``json.loads(response)`` would fail on the leading ``<``.
    This helper slices from the first ``{`` to the last ``}`` and parses that.

    Naive: extra braces in the think text can produce an invalid slice.

    Args:
        response: Raw string from ``localLLM.generate``.

    Returns:
        Decoded object, expected to contain ``action`` and related fields.

    Raises:
        RuntimeError: no ``{`` or no ``}`` in ``response``.
        json.JSONDecodeError: the slice is not valid JSON.
    """
    start = response.find("{")  # index of the first opening brace, or -1
    end = response.rfind("}")  # index of the last closing brace, or -1

    if start == -1 or end == -1:  # model never opened or never closed an object
        raise RuntimeError(
            f"No JSON object found in model response:\n{response}"
        )

    json_text = response[start:end + 1]  # inclusive slice of the candidate object

    return json.loads(json_text)  # dict / list; run_agent expects a dict with "action"
