# Chapter 03 — ctypes FFI and `llama_utils.py`

Source: `llm_server/src/ai/utils/llama_utils.py`

This file is the **Python-visible ABI** of the C++ bridge. If a signature is wrong, later calls look like they “work” and then crash or return garbage. If a signature is missing, ctypes guesses (`c_int` returns), which is fatal for pointer-returning functions.

---

## 1. Concepts

### 1.1 Foreign function interface

Python and C++ do not share objects. They share:

- integer-sized values
- pointers (addresses)
- C-layout arrays
- C strings (`char *`, usually UTF-8 bytes)

`ctypes.CDLL` maps a dylib and lets you write `bridge.llama_bridge_decode_one(ctx, token)` as if it were Python.

### 1.2 Why `argtypes` / `restype` exist

Without declarations, ctypes:

- converts Python `int` → `c_int`
- treats the return value as `c_int`

`llama_bridge_create_context` returns a **pointer**. On 64-bit macOS a pointer is 64 bits. Truncating it to 32 bits produces a wrong address. The next native call writes into unmapped memory.

So every function that returns `llama_model *` or `llama_context *` **must** have:

```python
bridge.fn.restype = ctypes.c_void_p
```

### 1.3 `c_void_p` is the Python spelling of “opaque pointer”

We never define a ctypes `Structure` for `llama_context`. We pass the address through and let C++ dereference it. This matches the PDF: Python holds `0x12345678`, native memory holds the KV cache.

---

## 2. File header

```python
import subprocess
from pathlib import Path
import ctypes
import json
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `import subprocess` | Bind process-spawning API | Run `brew --prefix` | `find_llama_library` | Import | N/A | Tiny |
| `from pathlib import Path` | Bind `Path` | Join `lib/libllama.dylib` portably | `find_llama_library` | Import | N/A | Tiny |
| `import ctypes` | Bind FFI | Type objects and later `CDLL` (used here only as types) | `define_llama_bridge_signatures` | Import | N/A | Tiny |
| `import json` | Bind JSON parser | `extract_json` | `run_agent` | Import | N/A | Tiny |

---

## 3. `find_llama_library`

```python
def find_llama_library() -> str:
    brew_prefix = subprocess.run(
        ["brew", "--prefix", "llama.cpp"],
        capture_output=True,
        text=True,
        check=True,
    )
    return str(Path(brew_prefix.stdout.strip()) / "lib" / "libllama.dylib")
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `def find_llama_library() -> str:` | Declare a function returning a path string | Single place to resolve Homebrew llama.cpp | `config.LLAMA_LIBRARY_PATH` |
| `subprocess.run([...], capture_output=True, text=True, check=True)` | Run Homebrew, keep stdout as `str`, raise on non-zero exit | Discover the keg prefix (`/opt/homebrew/opt/llama.cpp` or similar) | The `return` line |
| `brew_prefix.stdout.strip()` | Remove trailing newline | `Path` must not contain `\n` | `Path(...)` |
| `Path(...) / "lib" / "libllama.dylib"` | Join segments | Conventional layout of the formula | `CDLL` in `localLLM.__init__` |
| `return str(...)` | Convert `Path` → `str` | `CDLL` wants a string path | `config.py` assignment |

**Called by:** `config.py` at import: `LLAMA_LIBRARY_PATH = find_llama_library()`.

**When:** Once per process that imports `config.config`. Not per token, not per generate.

**What stops it:**

- User never starts the agent.
- `brew` not installed → `FileNotFoundError`.
- `llama.cpp` not installed → `check=True` raises `CalledProcessError`.
- Ctrl+C while `brew` runs.

**Memory / CPU:** One child process. Output is a short string. No dylib is loaded here.

**Syntax:** `capture_output=True` is `stdout=PIPE, stderr=PIPE`. `text=True` decodes bytes as the locale encoding. `check=True` is the difference between a silent empty path and a crash at import.

---

## 4. `define_llama_bridge_signatures`

```python
def define_llama_bridge_signatures(bridge: ctypes.CDLL) -> ctypes.CDLL:
    ...
    return bridge
```

**Called by:** `localLLM.__init__` after `self.llama_bridge = ctypes.CDLL(LLAMA_BRIDGE_PATH)`.

**When:** Once per `localLLM()` construction. Today that is once per process.

**What stops it:** Constructor exception before this line (failed first `CDLL`). After it returns, signatures stay on the `CDLL` object until process exit.

**Memory / CPU:** Writes a handful of attributes on the `CDLL`. No native calls. Negligible.

**Who needs it:** Every subsequent `self.llama_bridge.llama_bridge_*` call in `inference.py`.

The function **mutates** the passed object and also returns it. The caller assigns the return value, but it is the same object.

### 4.1 Init / shutdown / model lifecycle

| Assignment | C++ function | argtypes | restype | Why this restype |
|---|---|---|---|---|
| `llama_bridge_init` | `void llama_bridge_init()` | `[]` | `None` | No return |
| `llama_bridge_model_load` | `llama_model *(const char *)` | `[c_char_p]` | `c_void_p` | Pointer. Must not be `c_int`. |
| `llama_bridge_model_free` | `void (llama_model *)` | `[c_void_p]` | `None` | |
| `llama_bridge_model_size` | `unsigned long long (llama_model *)` | `[c_void_p]` | `c_uint64` | 64-bit size in bytes |
| `llama_bridge_model_n_params` | `unsigned long long (llama_model *)` | `[c_void_p]` | `c_uint64` | Parameter count can exceed 2³¹ |
| `llama_bridge_shutdown` | `void ()` | `[]` | `None` | |

`c_char_p` accepts a Python `bytes` object (`str(MODEL_PATH).encode("utf-8")`). Passing a `str` can work on some platforms via implicit encoding; this codebase encodes explicitly.

### 4.2 Tokenize

```python
bridge.llama_bridge_tokenize.argtypes = [
    ctypes.c_void_p,               # model
    ctypes.c_char_p,               # text
    ctypes.POINTER(ctypes.c_int),  # output token array
    ctypes.c_int,                  # max tokens
]
bridge.llama_bridge_tokenize.restype = ctypes.c_int
```

The third argument may be Python `None`. ctypes converts `None` for a pointer type into a C `NULL`. That is the probe call:

```text
tokenize(model, text, NULL, 0) → -N
```

`restype = c_int` is required because llama.cpp uses **negative** counts as “need N slots.”

### 4.3 Context

| Function | argtypes | restype |
|---|---|---|
| `llama_bridge_create_context` | `c_void_p, c_int` | `c_void_p` |
| `llama_bridge_free_context` | `c_void_p` | `None` |

`create_context` returning `c_void_p` is the line the PDF spends a chapter on: `ctx` is a pointer.

### 4.4 Decode / sample / detokenize / EOG

| Function | argtypes | restype | Notes |
|---|---|---|---|
| `llama_bridge_decode_prompt` | `void_p, POINTER(c_int), c_int` | `c_int` | Status only |
| `llama_bridge_decode_one` | `void_p, c_int` | `c_int` | Status only |
| `llama_bridge_sample_greedy` | `void_p` | `c_int` | Token **id**, not status |
| `llama_bridge_token_to_piece` | `void_p, c_int, c_char_p, c_int` | `c_int` | Bytes written, or `-needed` |
| `llama_bridge_is_eog` | `void_p, c_int` | `c_bool` | End-of-generation? |

Mixing these restypes up is a common bug:

- Treating `sample_greedy` as status → you would think token `0` means success.
- Treating `decode_*` as a token → you would feed `0` back into the model.

### 4.5 `return bridge`

Who needs it: `localLLM.__init__` assignment. Purpose: fluent style; not required for correctness because mutation already happened.

---

## 5. `init_llama`

```python
def init_llama(bridge: ctypes.CDLL) -> ctypes.CDLL:
    bridge.llama_bridge_init()
    return bridge
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `bridge.llama_bridge_init()` | Call `llama_backend_init()` in C++ | One-time llama.cpp global setup (backend, allocators) | All later model/context calls |
| `return bridge` | Same object | Caller rebinds `self.llama_bridge` | `__init__` |

**Called by:** `localLLM.__init__`, once.

**When:** After signatures are set, before `model_load`.

**What stops it:** Failed `CDLL` or missing symbol (`AttributeError` / `OSError`).

**Memory / CPU:** Small native init. Not the GGUF. Must be paired with `llama_bridge_shutdown` in `close()`.

---

## 6. `extract_json`

```python
def extract_json(response: str) -> dict:
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(...)
    json_text = response[start:end + 1]
    return json.loads(json_text)
```

This is **not** FFI. It lives here because the agent needed a shared helper after Qwen started wrapping JSON in `<think>` (PDF Chapter 5–6).

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `response.find("{")` | Index of first `{` or `-1` | Skip `<think>...` and any preamble | Slice below |
| `response.rfind("}")` | Index of last `}` | Include the whole outermost-looking object | Slice below |
| `if start == -1 or end == -1` | Detect missing braces | Fail with the raw model text | `run_agent` (exception) |
| `response[start:end + 1]` | Inclusive slice | Feed `json.loads` a candidate | Parser |
| `json.loads(json_text)` | Parse to `dict` | Produce `decision["action"]` etc. | `run_agent` |

**Called by:** `run_agent`, once per agent turn, after `llm.generate`.

**When:** After each generation, before tool dispatch.

**What stops it:**

- `generate()` raises first.
- User Ctrl+C during generate.
- No `{` / `}` in the output → `RuntimeError`.
- Braces present but invalid JSON → `json.JSONDecodeError`.
- Nested extra `}` in thinking text can slice wrong (known limitation).

**Memory / CPU:** O(length of response). Responses are at most ~64 tokens of new text plus whatever the model wrote. Microseconds. Allocates one substring and one dict.

**Syntax:** `find` vs `rfind` is intentional. First opening, last closing. Naive but enough for one JSON object after a think block.

---

## 7. What this file does *not* do

- It does not `CDLL` either library. `inference.py` does.
- It does not declare llama.cpp symbols from `libllama.dylib`. Only our `llama_bridge_*` wrappers.
- `self.llama_lib = ctypes.CDLL(LLAMA_LIBRARY_PATH)` exists so the OS loader resolves `libllama` dependencies when the bridge is used; this file never touches `llama_lib`.

## 8. Failure modes that look like “inference bugs”

| Symptom | Likely signature mistake |
|---|---|
| Crash immediately after `create_context` | `restype` not `c_void_p` |
| `decode` always “fails” with a huge integer | `restype` left as default, pointer bits interpreted as `int` |
| Tokenize probe returns a huge positive number | `None` not accepted because argtype is not a pointer |
| `is_eog` always true / false | `restype` not `c_bool` (integer truncation) |
