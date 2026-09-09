# Chapter 05 — `inference.py` Line by Line

Source: `llm_server/src/ai/engine/llama/inference.py`

`localLLM` is the Python object the agent talks to. It owns the two dylibs and the model pointer. Each `generate(prompt)` creates a fresh native context, fills it, samples up to `MAX_NEW_TOKENS`, frees the context, and returns a string.

---

## 1. Imports and `sys.path` bootstrap

```python
import ctypes
import sys
from pathlib import Path

_AI_DIR = Path(__file__).resolve().parents[2]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from config.config import LLAMA_LIBRARY_PATH, LLAMA_BRIDGE_PATH, MODEL_PATH
from config.model_config.llama import MAX_TOKENS, CONTEXT_SIZE, MAX_NEW_TOKENS
from utils.llama_utils import define_llama_bridge_signatures, init_llama
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `import ctypes` | Load FFI | `CDLL`, arrays, buffers | Entire class | Import | N/A | Tiny |
| `import sys` | Load interpreter module | Mutate `sys.path` | Bootstrap | Import | N/A | Tiny |
| `from pathlib import Path` | Path arithmetic | Compute `ai/` from `__file__` | `_AI_DIR` | Import | N/A | Tiny |
| `_AI_DIR = Path(__file__).resolve().parents[2]` | `inference.py` → `llama/` → `engine/` → `ai/` | Imports are `config.*`, `utils.*` under `ai/` | The `if` below | Import | Wrong parents index → import fail | One `Path` |
| `if str(_AI_DIR) not in sys.path:` | Avoid duplicate entries | Idempotent when imported as a module *or* run oddly | Subsequent imports | Import | N/A | Tiny |
| `sys.path.insert(0, ...)` | Prepend `ai/` | `import config.config` resolves | Config / utils imports | Import | User could have a colliding `config` package earlier on `path` | Tiny |
| `from config.config import ...` | Run `config.py` (including `brew`) | Concrete filesystem paths | `__init__` | Import | `brew` fail | `brew` subprocess |
| `from config.model_config.llama import ...` | Bind three ints | Limits in `generate` | `generate` | Import | N/A | Tiny |
| `from utils.llama_utils import ...` | Bind two helpers | Signatures + backend init | `__init__` | Import | N/A | Tiny |

`MAX_TOKENS` is imported and **unused**. `MODEL_PATH` is used only in `__init__`.

**Who imports this file:** `simple_agent.py` (`from engine.llama.inference import localLLM`).

**When the class body runs:** At import. Methods do not run yet.

---

## 2. `class localLLM`

A single instance is constructed in `main()`. The name is `localLLM` (lowercase L), not `LocalLLM`.

### 2.1 `__init__` — load libraries and weights

**Called by:** `main()` → `llm = localLLM()`.

**When:** Once per process.

**What stops it:** Import failure; `CDLL` `OSError`; null model; Ctrl+C during load. If it raises, `main`’s `try` never starts, so `close()` is not called.

#### Load `libllama` then the bridge

```python
print(f"llama.cpp library: {LLAMA_LIBRARY_PATH}")
self.llama_lib = ctypes.CDLL(LLAMA_LIBRARY_PATH)
print("Successfully loaded llama.cpp into Python.")

self.llama_bridge = ctypes.CDLL(LLAMA_BRIDGE_PATH)
print("Successfully loaded llama.cpp bridge into Python.")
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| First `print` | Show the Homebrew dylib path | Debug “wrong bottle” issues | The operator | I/O |
| `CDLL(LLAMA_LIBRARY_PATH)` | `dlopen` llama.cpp | Resolve symbols the bridge is linked against | Bridge calls into llama | Maps the library; does not load GGUF |
| Success print | Confirm `dlopen` | Operator | Humans | I/O |
| `CDLL(LLAMA_BRIDGE_PATH)` | `dlopen` our wrapper | All `llama_bridge_*` calls | Rest of the class | Maps a small dylib |
| Success print | Confirm | Operator | Humans | I/O |

`self.llama_lib` is not called after load. Keeping the handle alive prevents premature unload of `libllama` while the bridge still needs it.

**User stop:** Ctrl+C during `CDLL` is unlikely to land mid-`dlopen` in a useful way; the next line either runs or the process dies.

#### Signatures and backend

```python
self.llama_bridge = define_llama_bridge_signatures(self.llama_bridge)
self.llama_bridge = init_llama(self.llama_bridge)
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `define_llama_bridge_signatures` | Set `argtypes` / `restype` | Correct pointer widths | Every native call | ~0 |
| `init_llama` | `llama_backend_init()` | Global llama.cpp setup | `model_load` and after | Small |

#### Load and inspect the GGUF

```python
print("Loading model:")
print(MODEL_PATH)
self.model = self.llama_bridge.llama_bridge_model_load(
    str(MODEL_PATH).encode("utf-8")
)
if not self.model:
    raise RuntimeError("Failed to load model")
print("\nModel loaded successfully.")

size_bytes = self.llama_bridge.llama_bridge_model_size(self.model)
parameter_count = self.llama_bridge.llama_bridge_model_n_params(self.model)
print(f"Model size: {size_bytes / 1024**2:.2f} MiB")
print(f"Parameters: {parameter_count:,}")
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| Prints of path | Show which file is opening | Debug missing GGUF | Operator | I/O |
| `encode("utf-8")` | Python `str` → `bytes` / `c_char_p` | C API is bytes | `model_load` | Tiny |
| `llama_bridge_model_load` | Read weights | `self.model` handle | `generate`, `close` | **Seconds, ~527 MiB** |
| `if not self.model` | Null check | Fail loud | Caller | Tiny |
| `raise RuntimeError` | Abort constructor | Prevent later use of null | `main` | — |
| Success print | Confirm | Operator | Humans | I/O |
| `model_size` / `n_params` | Query metadata | Observability | Prints only | Tiny |
| `1024**2` | Bytes → MiB | Readable | Print | Tiny |
| `:,` format | Thousands separators | Readable | Print | Tiny |

**Who needs `self.model`:** tokenize, create_context, is_eog, token_to_piece, model_free. Decode functions take `ctx`, which was created from this model.

---

## 3. `generate(self, prompt: str) -> str`

**Called by:** `run_agent` → `response = llm.generate(conversation)` on every agent turn.

**When:** At least once per `run_agent`. Again after each tool call, because the conversation string grew.

**What stops a call that has already started:**

| Stop | Where | User / program action |
|---|---|---|
| Unexpected tokenize result | After probe | Should not happen; raises |
| Tokenize fail on second pass | After fill | Raises |
| Context too small | Before create | Prompt + 64 > 1024 |
| Context create null | After create | Raises |
| Prompt decode ≠ 0 | After decode_prompt | Raises |
| Piece buffer too small | In loop | Raises |
| Token decode ≠ 0 | In loop | Raises |
| EOG | In loop | `break` — **normal** |
| 64 tokens | Loop end | **normal** |
| Ctrl+C | Any Python line / after a native return | User |
| Process kill | Immediate | User |

**Return value:** Concatenated detokenized pieces. May include `<think>...</think>` and JSON. Does not parse JSON.

**Memory shape of one call:**

```text
token_buffer     required * 4 bytes     (freed after return)
ctx              KV + scratch           (explicitly freed)
piece_buffer     256 bytes × iterations (GC)
generated_text   Python str growing     (returned)
```

### 3.1 Two-pass tokenize

```python
required = self.llama_bridge.llama_bridge_tokenize(
    self.model,
    prompt.encode("utf-8"),
    None,
    0,
)
if required >= 0:
    raise RuntimeError(f"Unexpected tokenization result: {required}")
required = -required
print(f"Required token capacity: {required}")

token_buffer = (ctypes.c_int * required)()
token_count = self.llama_bridge.llama_bridge_tokenize(
    self.model,
    prompt.encode("utf-8"),
    token_buffer,
    required,
)
if token_count < 0:
    raise RuntimeError(
        f"Tokenization failed. Required {-token_count} tokens."
    )
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `prompt.encode("utf-8")` | Bytes for C | Bridge uses `strlen` | Both tokenize calls | Allocates a bytes object |
| First `tokenize(..., None, 0)` | Probe | Learn `N` without guessing `MAX_TOKENS` | Allocation size | ms |
| `if required >= 0: raise` | Probe must be negative | Detect API / signature mistakes | Safety | ~0 |
| `required = -required` | `-(-N) → N` | Positive capacity | Array constructor | ~0 |
| Print capacity | Debug agent-prompt growth | Operator | Humans | I/O |
| `(ctypes.c_int * required)()` | Allocate `N` C ints, zeroed | Output buffer | Second tokenize + decode_prompt | `4N` bytes |
| Second `tokenize` | Write token IDs | IDs the transformer will see | `decode_prompt` | ms |
| `if token_count < 0: raise` | Second pass must fit | Race-free in this single-threaded program; still defensive | Safety | ~0 |

Theory and the old `MAX_TOKENS = 128` crash: Chapter 07 and PDF Chapter 4.

`token_buffer` is passed to C as `int *`. Python also *could* build `tokens = [token_buffer[i] for i in range(token_count)]` (older code did, for printing). The current file does not copy to a list.

### 3.2 Context size check and create

```python
context_size = CONTEXT_SIZE
required_context_size = token_count + MAX_NEW_TOKENS
if required_context_size > CONTEXT_SIZE:
    raise RuntimeError(
        f"Context too small. "
        f"Prompt={token_count}, "
        f"generation={MAX_NEW_TOKENS}, "
        f"required={required_context_size}, "
        f"configured={CONTEXT_SIZE}"
    )
ctx = self.llama_bridge.llama_bridge_create_context(
    self.model,
    context_size,
)
if not ctx:
    raise RuntimeError("Failed to create llama context")
print("\nContext created.")
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `context_size = CONTEXT_SIZE` | Local alias `1024` | Passed to C | `create_context` | ~0 |
| `token_count + MAX_NEW_TOKENS` | Worst-case sequence length | Prevent decode past `n_ctx` | The `if` | ~0 |
| `if ... > CONTEXT_SIZE: raise` | Refuse to start | Avoid native overflow / wrap | Operator (clear error) | ~0 |
| `create_context` | Alloc empty `llama_context` | KV home for this generate | All decode/sample | **Allocates KV** |
| `if not ctx` | Null check | Fail loud | Caller | ~0 |
| Print | Confirm | Operator | Humans | I/O |

`context_size` is **not** raised to `required_context_size`. The PDF suggested `max(CONTEXT_SIZE, token_count + MAX_NEW_TOKENS)` as an alternative. This build **raises** instead of growing. Safer for RAM; stricter for long agent histories.

After several tool turns, `conversation` can exceed `1024 - 64` tokens and `generate` will start failing here.

### 3.3 Prompt decode

```python
decode_result = self.llama_bridge.llama_bridge_decode_prompt(
    ctx,
    token_buffer,
    token_count,
)
if decode_result != 0:
    raise RuntimeError(f"llama_decode failed: {decode_result}")
print("Prompt decoded by neural network.")
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `llama_bridge_decode_prompt` | Forward pass over all prompt tokens; mutate `ctx` | Logits for the first sample | Loop | **Dominant** |
| `decode_result != 0` | Status check | `ctx` may be unusable on failure | Safety | ~0 |
| Print | Confirm | Operator | Humans | I/O |

`decode_result` is **not** assigned into `ctx`. `ctx` was already the pointer. Chapter 06.

If this raises, `ctx` is **not** freed in the current code.

### 3.4 Autoregressive loop

```python
generated_text = ""
for _ in range(MAX_NEW_TOKENS):
    next_token = self.llama_bridge.llama_bridge_sample_greedy(ctx)
    if self.llama_bridge.llama_bridge_is_eog(self.model, next_token):
        break
    piece_buffer = ctypes.create_string_buffer(256)
    piece_length = self.llama_bridge.llama_bridge_token_to_piece(
        self.model, next_token, piece_buffer, len(piece_buffer),
    )
    if piece_length < 0:
        raise RuntimeError(f"Piece buffer too small: {-piece_length}")
    piece = piece_buffer.raw[:piece_length].decode("utf-8", errors="replace")
    generated_text += piece
    print(piece, end="", flush=True)
    decode_result = self.llama_bridge.llama_bridge_decode_one(ctx, next_token)
    if decode_result != 0:
        raise RuntimeError(f"llama_decode failed: {decode_result}")
```

`for _ in range(...)`: `_` means “index unused.” The cap is the only reason the loop exists.

| Step | Line(s) | What it does | Purpose | Who needs the result | Cost |
|---|---|---|---|---|---|
| 1 | `sample_greedy(ctx)` | Argmax last logits | Pick token id | EOG, piece, decode_one | Vocab scan + sampler alloc |
| 2 | `is_eog` / `break` | Stop on EOS | Do not print or decode EOS | Loop control | Tiny |
| 3 | `create_string_buffer(256)` | 256 zero bytes | Detokenize dest | `token_to_piece` | 256 B |
| 4 | `token_to_piece` | Fill buffer | Human-readable chunk | `generated_text`, stdout | Tiny |
| 5 | `piece_length < 0` | Capacity error | Surface `-needed` | Operator | — |
| 6 | `.raw[:n].decode(...)` | Bytes → `str` | Python string | Concat | Tiny |
| 7 | `generated_text += piece` | Accumulate | Return value for the agent | `run_agent` | String alloc (CPython may copy) |
| 8 | `print(..., end="", flush=True)` | Stream tokens live | UX | Operator | I/O; `flush` so the terminal updates per token |
| 9 | `decode_one` | Append token to KV; new logits | Next sample | Next iteration | **Per-token matmuls** |
| 10 | status check | Fail loud | Safety | — |

**Order is load-bearing:**

```text
sample  →  maybe stop  →  detokenize/print  →  decode_one
```

If you `decode_one` before printing, you still get the same text, but EOG must stay *before* decode so EOS is not written into the KV cache. If you sample after decode without having decoded the chosen token, you would resample the same logits forever.

**String concatenation:** 64 tiny `+=` operations are fine. This is not a hot Python path compared with `decode_one`.

### 3.5 Teardown of one generate

```python
print("\n\nGeneration complete.")
print(f"Generated text: {generated_text!r}")
self.llama_bridge.llama_bridge_free_context(ctx)
return generated_text
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| Prints | Show the full string with `repr` | Debug hidden characters / think tags | Operator | I/O |
| `free_context(ctx)` | `llama_free` | Return KV RAM | Next `generate` / process | Free |
| `return generated_text` | Hand text to the agent | `extract_json` | `run_agent` | Caller owns the `str` |

`!r` uses `repr()`, so you see `\n` and quotes. That is how `<think>` first became visible in logs.

---

## 4. `close`

```python
def close(self):
    self.llama_bridge.llama_bridge_model_free(self.model)
    self.llama_bridge.llama_bridge_shutdown()
    print("\nModel unloaded.")
```

**Called by:** `main` `finally`.

**When:** After `run_agent` returns or raises (if `localLLM()` succeeded).

**What stops it:** `localLLM()` raised; `kill -9`; calling process never entered `try`.

**Who needs it:** The OS and the next run (GPU memory). Python GC will not call `llama_model_free`.

**Cost:** Frees ~527 MiB plus backend. The print is why tracebacks appear *after* `Model unloaded.`

There is no `atexit` hook and no context-manager (`__enter__` / `__exit__`). `main` must keep the `try/finally`.

---

## 5. Lifecycle diagram

```text
localLLM()
    load libs, init backend, load model          [long-lived]
        generate(prompt1)
            ctx1 create → decode → sample* → free ctx1
        generate(prompt2)
            ctx2 create → decode → sample* → free ctx2
    close()
        free model, backend_free
```

The model is **not** reloaded between agent turns. Only `ctx` is.

---

## 6. CPU timeline of one `generate`

```text
tokenize probe     ████                                                 ~ms
tokenize fill      ████                                                 ~ms
create context     ████████                                             alloc
decode_prompt      ████████████████████████████████████████████████     spike
sample/decode×k    ██ ██ ██ ██ ██ ██ ██ ██                              k ≤ 64
free context       ██
```

Wall time is usually “prompt decode + (tokens × decode_one).” Prints and JSON are invisible on this scale.
