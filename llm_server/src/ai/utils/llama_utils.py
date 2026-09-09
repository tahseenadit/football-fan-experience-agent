import subprocess
from pathlib import Path
import ctypes


def find_llama_library() -> str:
    brew_prefix = subprocess.run(
        ["brew", "--prefix", "llama.cpp"],
        capture_output=True,
        text=True,
        check=True,
    )
    return str(Path(brew_prefix.stdout.strip()) / "lib" / "libllama.dylib")


def define_llama_bridge_signatures(bridge: ctypes.CDLL) -> ctypes.CDLL:
    bridge.llama_bridge_init.argtypes = []
    bridge.llama_bridge_init.restype = None

    bridge.llama_bridge_model_load.argtypes = [ctypes.c_char_p]
    bridge.llama_bridge_model_load.restype = ctypes.c_void_p

    bridge.llama_bridge_model_free.argtypes = [ctypes.c_void_p]
    bridge.llama_bridge_model_free.restype = None

    bridge.llama_bridge_model_size.argtypes = [ctypes.c_void_p]
    bridge.llama_bridge_model_size.restype = ctypes.c_uint64

    bridge.llama_bridge_model_n_params.argtypes = [ctypes.c_void_p]
    bridge.llama_bridge_model_n_params.restype = ctypes.c_uint64

    bridge.llama_bridge_shutdown.argtypes = []
    bridge.llama_bridge_shutdown.restype = None

    bridge.llama_bridge_tokenize.argtypes = [
        ctypes.c_void_p,          # model
        ctypes.c_char_p,          # text
        ctypes.POINTER(ctypes.c_int),  # output token array
        ctypes.c_int,             # max tokens
    ]

    bridge.llama_bridge_tokenize.restype = ctypes.c_int

    return bridge


def init_llama(bridge: ctypes.CDLL) -> ctypes.CDLL:
    bridge.llama_bridge_init()
    return bridge
