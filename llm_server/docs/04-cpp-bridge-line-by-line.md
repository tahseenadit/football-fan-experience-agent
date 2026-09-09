# Chapter 04 — `bridge.cpp` Line by Line

Source: `llm_server/src/ai/engine/llama/bridge.cpp`

This file is the only C++ we own. It does not implement a transformer. It wraps llama.cpp so Python can call a stable, unmangled C ABI.

Compiled by `build_bridge.sh` into `libllama_bridge.dylib` (Chapter 12).

---

## 1. Theory: why a bridge exists

Python ctypes can call C functions. llama.cpp’s headers are C-compatible, but:

- We want a **small, named surface** (`llama_bridge_*`) that we control.
- We want default parameters filled in C++ (`n_gpu_layers = -1`, `n_ctx`, `n_batch`).
- We want `extern "C"` so symbol names stay exactly `llama_bridge_decode_one`, not `_Z23llama_bridge_decode_one...`.

```text
Python  --ctypes-->  llama_bridge_*  --llama.h-->  libllama
```

Every function below is **called from Python** (`inference.py`) except that they also call into llama.cpp. Nothing in C++ calls back into Python.

---

## 2. File header

```cpp
#include <llama.h>
#include <cstring>

extern "C" {
```

| Line | What it does | Purpose | Who needs it | When compiled / run | Stop | Cost |
|---|---|---|---|---|---|---|
| `#include <llama.h>` | Insert llama.cpp API declarations | Types (`llama_model`, `llama_context`, `llama_batch`) and functions | Every function in this file | Compile time | Missing Homebrew include path → build fails | 0 at runtime |
| `#include <cstring>` | Declare `strlen` | Tokenize needs UTF-8 byte length | `llama_bridge_tokenize` | Compile time | Build fail | 0 |
| `extern "C" {` | Emit C linkage for the block | ctypes looks up unmangled names | `CDLL` attribute access | Link time | If omitted, `AttributeError: llama_bridge_init` | 0 |

The closing `}` of `extern "C"` is the last line of the file.

---

## 3. `llama_bridge_init`

```cpp
void llama_bridge_init() {
    llama_backend_init();
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `void llama_bridge_init()` | Export a no-arg, no-return symbol | Python has one obvious “start llama.cpp” call | `init_llama` → `localLLM.__init__` |
| `llama_backend_init()` | Initialize llama.cpp global backend | Allocators, GPU backend hooks | Every later llama.cpp call |

**Called by:** `init_llama()` once.

**When:** After both dylibs are loaded and signatures are set; before model load.

**What stops it from running:** Constructor never reached; symbol missing; process killed during `CDLL`.

**What stops it from being useful later:** Calling `llama_bridge_shutdown` (`llama_backend_free`). A second init after shutdown is not part of this program’s design.

**Memory / CPU:** Small. No weights. Must not be called in a tight loop (it is not).

---

## 4. `llama_bridge_model_load`

```cpp
llama_model * llama_bridge_model_load(const char * path) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    return llama_model_load_from_file(path, params);
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `llama_model * llama_bridge_model_load(const char * path)` | Load GGUF at `path`, return pointer or null | Give Python an opaque model handle | `localLLM.__init__` stores `self.model` |
| `llama_model_default_params()` | Fill a params struct with llama.cpp defaults | Avoid uninitialized fields | The load call |
| `params.n_gpu_layers = -1` | Request “as many layers as possible” on GPU | Use Metal/CUDA if llama.cpp was built with it | Throughput of every later decode |
| `return llama_model_load_from_file(path, params)` | Read the file, allocate tensors | The actual ~527 MiB load | All tokenize / decode / sample / EOG / piece functions |

**Called by:** `self.llama_bridge.llama_bridge_model_load(str(MODEL_PATH).encode("utf-8"))`.

**When:** Once per process (once per `localLLM()`).

**What stops it:**

- Missing file → returns `nullptr`; Python raises `Failed to load model`.
- User Ctrl+C during load (I/O may continue briefly).
- Insufficient RAM / GPU memory → typically null or abort inside llama.cpp.

**Memory:** Dominant allocation of the process: Q4_0 weights ≈ **526.50 MiB** on disk; runtime may be similar plus scratch. If GPU offload succeeds, a large fraction lives in GPU RAM and a mapping may remain on the CPU side.

**CPU:** Seconds of decode/dequant setup. Highest startup cost.

Python `if not self.model` treats a null pointer as failure. `restype = c_void_p` is mandatory here.

---

## 5. `llama_bridge_model_free`

```cpp
void llama_bridge_model_free(llama_model * model) {
    if (model != nullptr) {
        llama_model_free(model);
    }
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `if (model != nullptr)` | Skip null | Safe if load failed and someone still called free | `close()` |
| `llama_model_free(model)` | Release weight tensors | Return ~527 MiB | OS / next program |

**Called by:** `localLLM.close()` from `main`’s `finally`.

**When:** Process teardown of the agent, or any exception after `localLLM()` succeeded.

**What stops it:** `kill -9`; `localLLM()` never returned so `try/finally` never started; double-free if `close()` were called twice (current `main` calls it once).

**Memory:** Frees the model. Does not unload `libllama.dylib` from the process.

**CPU:** Proportional to tearing down tensor allocations; usually much faster than load.

---

## 6. `llama_bridge_model_size` and `llama_bridge_model_n_params`

```cpp
unsigned long long llama_bridge_model_size(llama_model * model) {
    return llama_model_size(model);
}

unsigned long long llama_bridge_model_n_params(llama_model * model) {
    return llama_model_n_params(model);
}
```

| Function | Returns | Purpose | Who needs it |
|---|---|---|---|
| `llama_bridge_model_size` | Byte size of the model | Human-readable `Model size: 526.50 MiB` | `localLLM.__init__` print only |
| `llama_bridge_model_n_params` | Parameter count | Print `Parameters: 752,393,024` | Same |

**Called by:** `__init__` after a successful load. Not on the generate path.

**When:** Once.

**What stops the prints:** Load failure raises before these lines.

**Memory / CPU:** Two integer queries. Negligible. They do not copy weights.

`unsigned long long` matches Python `c_uint64`.

---

## 7. `llama_bridge_shutdown`

```cpp
void llama_bridge_shutdown() {
    llama_backend_free();
}
```

**Called by:** `localLLM.close()`, after `model_free`.

**Purpose:** Inverse of `llama_backend_init`.

**Who needs it:** Process-global llama.cpp state. Not needed by another function in this file after close; the process is about to exit.

**What stops it:** Same as `model_free`. Order matters: free the model first, then the backend.

**Cost:** Small.

---

## 8. `llama_bridge_tokenize`

```cpp
int llama_bridge_tokenize(
    llama_model * model,
    const char * prompt_text,
    int * tokens,
    int max_tokens
) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int text_len = static_cast<int>(strlen(prompt_text));
    return llama_tokenize(
        vocab,
        prompt_text,
        text_len,
        tokens,
        max_tokens,
        true,
        true
    );
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `llama_model_get_vocab(model)` | Pointer to the model’s vocabulary | Tokenize is vocab-specific (Qwen BPE / Unigram, special tokens) | `llama_tokenize` |
| `strlen(prompt_text)` | Byte length of the UTF-8 C string | llama.cpp does not take a Python `str` | `llama_tokenize` |
| `static_cast<int>(...)` | `size_t` → `int` | Match API; prompts here are tiny vs `INT_MAX` | API |
| `llama_tokenize(..., true, true)` | Last two `true`s: add special tokens, parse special tokens | Match chat/special-token behavior the model was trained with | Downstream decode (token IDs must be the ones the model expects) |
| `return` | `+n` written, or `-n` if buffer too small / probe | Two-pass API in Python | `generate()` |

**Called by:** `generate()`, **twice** per call:

1. `tokens = NULL`, `max_tokens = 0` → expect negative required count.
2. Real buffer, `max_tokens = required`.

**When:** Every `llm.generate(...)`, including every agent turn.

**What stops it:**

- `generate` never entered.
- `prompt_text` null (Python always passes `encode("utf-8")`, which is non-null).
- User abort during generate (tokenize is fast; abort usually hits the next decode).

**Memory:** Does not allocate the output buffer. Python allocates `required * sizeof(int)` (4 bytes × N). Vocab is inside the already-loaded model.

**CPU:** Milliseconds. No matmuls. Cost scales with prompt byte length.

**Null buffer:** When Python passes `None`, `tokens` is `nullptr` and `max_tokens` is `0`. llama.cpp returns `-N`. That is not a failure. Chapter 07.

---

## 9. `llama_bridge_create_context`

```cpp
llama_context * llama_bridge_create_context(
    llama_model * model,
    int context_size
) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = context_size;
    params.n_batch = context_size;
    return llama_init_from_model(model, params);
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `llama_context_default_params()` | Safe defaults | Uninitialized params would be UB | `llama_init_from_model` |
| `params.n_ctx = context_size` | Max sequence length for this context | KV cache capacity (conceptually 0..n_ctx-1) | Every `llama_decode` |
| `params.n_batch = context_size` | Max tokens per decode **call** | Lets `decode_prompt` send the whole prompt in one batch | First decode |
| `llama_init_from_model` | Allocate empty context | Working memory for one `generate()` | `ctx` in Python |

**Called by:** `generate()` after the context-size check, once per generate.

**When:** Each agent turn’s `generate()`.

**Empty context:** Yes. KV slots are allocated (or reserved) but **no prompt tokens have been decoded**. PDF Chapter 2.

```text
model  = Qwen weights, vocab, architecture
ctx    = empty KV + buffers, capacity = context_size
```

**What stops it:** Fit check raise; load never happened; `llama_init_from_model` returns null → Python raises.

**What frees it:** `llama_bridge_free_context` at the end of `generate` (success or after the generation loop). If Python raises **before** `free_context` (e.g. decode_prompt failure), **this context can leak** until process exit. The current `generate()` does not use `try/finally` around `ctx`.

**Memory:** Second largest allocation after the model. Scales with `n_ctx` (here 1024). See Chapter 11.

**CPU:** Allocation and zeroing; tens to hundreds of ms possible, still far below a long prompt decode.

---

## 10. `llama_bridge_free_context`

```cpp
void llama_bridge_free_context(llama_context * context) {
    if (context != nullptr) {
        llama_free(context);
    }
}
```

**Called by:** end of `generate()`, after printing generated text.

**When:** Every successful path through the generation loop (including EOG or max-token stop). **Not** called if `create_context` returned null (Python raises first) or if an exception occurs between create and this line.

**Who needs it:** The process, to return KV/scratch RAM before the next `generate()` allocates another context.

**User abort:** Ctrl+C between create and free → `finally` in `main` calls `model_free` / `shutdown` but **does not** call `free_context`. The dying process drops the mapping. If you later add a long-lived server, this leak matters.

**Cost:** Frees native context; faster than create.

---

## 11. `llama_bridge_decode_prompt`

```cpp
int llama_bridge_decode_prompt(
    llama_context * ctx,
    int * tokens,
    int token_count
) {
    llama_batch batch = llama_batch_get_one(tokens, token_count);
    return llama_decode(ctx, batch);
}
```

This is the **first transformer forward pass** of a generate.

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `llama_batch_get_one(tokens, token_count)` | Build a temporary batch describing those token IDs | `llama_decode` consumes a batch, not a raw array | The decode call |
| `llama_decode(ctx, batch)` | Run the model over the batch; **mutate `ctx`** | Fill KV cache positions `0 .. token_count-1`; produce logits at the last position | `sample_greedy` next |
| `return` | `0` ok, else error | Python `if decode_result != 0` | `generate()` |

The batch is **not** stored inside `ctx`. After return, `batch` is gone. Persistence is the KV cache. PDF Chapter 1.

**Called by:** `generate()`, once, after context create.

**When:** Each generate, after tokenize.

**What stops it:** Failed context create; user kill; `n_batch` too small (we set `n_batch = n_ctx` to avoid that for a prompt that already passed the fit check).

**Memory:** Uses context scratch + writes `token_count` KV positions. Peak compute buffers live in `ctx`.

**CPU / GPU:** **Hottest line in the program per generate**, unless the prompt is short and you then generate 64 tokens (then the sum of `decode_one` can exceed it). Cost ≈ O(layers × prompt_tokens × d_model) with KV write.

---

## 12. `llama_bridge_decode_one`

```cpp
int llama_bridge_decode_one(
    llama_context * ctx,
    int token
) {
    llama_token t = token;
    llama_batch batch = llama_batch_get_one(&t, 1);
    return llama_decode(ctx, batch);
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `llama_token t = token` | Local copy of the Python `int` | `llama_batch_get_one` wants `llama_token *` | Batch |
| `llama_batch_get_one(&t, 1)` | Batch of length 1 | One new position | `llama_decode` |
| `llama_decode(ctx, batch)` | Forward pass for that token; append K/V | Extend the sequence so the *next* sample sees updated logits | Next loop iteration |
| `return` | Status | Python error check | `generate()` |

**How “append” works:** not `ctx.text += piece`. The new token’s K and V are written at the next position. PDF Chapter 1.

**Called by:** `generate()` once per sampled non-EOG token, after detokenize/print.

**When:** Up to `MAX_NEW_TOKENS` times per generate.

**What stops further calls:**

- EOG: `break` **before** `decode_one` (the EOG token is not fed back).
- `MAX_NEW_TOKENS` reached.
- `decode_one` returns non-zero → raise.
- User abort.

**Memory:** One new KV position. Batch `t` is stack memory.

**CPU / GPU:** One-token decode with full attention over cached prefix. Cheaper than the prompt pass, still the per-token bottleneck.

---

## 13. `llama_bridge_sample_greedy`

```cpp
int llama_bridge_sample_greedy(llama_context * ctx) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_token token = llama_sampler_sample(sampler, ctx, -1);
    llama_sampler_free(sampler);
    return token;
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| Default chain params | Start a sampler chain | llama.cpp’s sampling API is chain-based | Init |
| `llama_sampler_chain_init` | Allocate a chain | Holder for the greedy sampler | Add / sample / free |
| `llama_sampler_chain_add(..., llama_sampler_init_greedy())` | Argmax policy | Deterministic next token = `argmax(logits)` | Reproducible agent JSON |
| `llama_sampler_sample(sampler, ctx, -1)` | Read logits from `ctx` at index `-1` (last position) | Choose the token id | Python `next_token` |
| `llama_sampler_free(sampler)` | Destroy the chain | Avoid a leak **every token** | Process |
| `return token` | Integer id | `is_eog`, `token_to_piece`, `decode_one` | `generate()` loop |

**Called by:** `generate()` at the start of every generation iteration, **before** EOG check.

**When:** 1..64 times per generate.

**What stops it:** Loop not entered (decode_prompt failed); previous iteration `break`/raise; max iterations done.

**Memory / CPU:** Sampling itself is a scan of the vocab (Qwen vocab is tens of thousands of logits) — cheap vs decode. **Allocating and freeing a sampler every token is unnecessary overhead.** Correct, but not optimal. A long-lived sampler on `localLLM` would be better.

**`-1` index:** “logits of the last decoded position.” After `decode_prompt`, that is the last prompt token. After `decode_one`, that is the token just appended.

Greedy means temperature is not applied. Same prompt → same tokens (bitwise, ignoring threading/GPU nondeterminism).

---

## 14. `llama_bridge_token_to_piece`

```cpp
int llama_bridge_token_to_piece(
    llama_model * model,
    int token,
    char * buffer,
    int buffer_size
) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_token_to_piece(
        vocab, token, buffer, buffer_size, 0, true
    );
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `llama_model_get_vocab` | Same vocab as tokenize | Detokenize must match tokenize | `llama_token_to_piece` |
| `llama_token_to_piece(..., 0, true)` | Write UTF-8 piece into `buffer`; `lstrip` offset 0; special-token rendering on | Convert id → bytes | Python `piece` string |
| `return` | Bytes written, or negative needed size | Python 256-byte buffer check | `generate()` |

**Called by:** `generate()` for every non-EOG token.

**When:** After sample, after EOG check passed.

**What stops it:** EOG `break`; loop end; raise on previous decode.

**Memory:** Writes into a 256-byte Python `create_string_buffer`. Typical piece is a few bytes (`"{"`, `"action"`, `" Paris"`). If a piece needed more than 256 bytes, llama.cpp returns `-needed` and Python raises.

**CPU:** Table lookup + memcpy. Negligible vs decode.

Python then does `piece_buffer.raw[:piece_length].decode("utf-8", errors="replace")`. `errors="replace"` means invalid UTF-8 becomes `` rather than crashing — important because tokens can be mid-character fragments that only become valid after concatenation.

---

## 15. `llama_bridge_is_eog`

```cpp
bool llama_bridge_is_eog(
    llama_model * model,
    int token
) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_vocab_is_eog(vocab, token);
}
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `llama_vocab_is_eog` | True if `token` is an end-of-generation id for this vocab | Stop the loop without detokenizing or decoding EOS | `generate()` `if ...: break` |

**Called by:** `generate()`, every iteration, immediately after sample.

**When:** Up to 64 times per generate.

**What stops further checks:** `break` itself; raise; loop exhausts.

**Memory / CPU:** Boolean lookup. Negligible.

If this were omitted, the model could emit EOS as text and keep going until `MAX_NEW_TOKENS`, wasting decode calls and polluting `generated_text`.

---

## 16. Call matrix (Python → this file → llama.cpp)

| Python | Bridge | llama.cpp |
|---|---|---|
| `init_llama` | `llama_bridge_init` | `llama_backend_init` |
| `__init__` load | `llama_bridge_model_load` | `llama_model_load_from_file` |
| `__init__` print | `llama_bridge_model_size` / `_n_params` | `llama_model_size` / `_n_params` |
| `generate` tokenize | `llama_bridge_tokenize` | `llama_model_get_vocab`, `llama_tokenize` |
| `generate` ctx | `llama_bridge_create_context` | `llama_init_from_model` |
| `generate` prompt | `llama_bridge_decode_prompt` | `llama_batch_get_one`, `llama_decode` |
| loop sample | `llama_bridge_sample_greedy` | sampler chain + `llama_sampler_sample` |
| loop stop | `llama_bridge_is_eog` | `llama_vocab_is_eog` |
| loop text | `llama_bridge_token_to_piece` | `llama_token_to_piece` |
| loop append | `llama_bridge_decode_one` | `llama_batch_get_one`, `llama_decode` |
| `generate` end | `llama_bridge_free_context` | `llama_free` |
| `close` | `llama_bridge_model_free` | `llama_model_free` |
| `close` | `llama_bridge_shutdown` | `llama_backend_free` |

## 17. What never happens in this file

- No threads.
- No tool calling.
- No JSON.
- No `<think>` handling.
- No Python objects.
- No persistent conversation state (that is the Python `conversation` string + a *new* context each generate).
