# Chapter 06 — Mutable Context, `llama_decode`, and the KV Cache

This chapter is the theory core of [`local_llm_to_agent_textbook.pdf`](local_llm_to_agent_textbook.pdf) Chapters 1–2, written against the current `bridge.cpp` and `inference.py`.

If you only remember one thing: **`decode_result` is a status code. The model state lives in native memory at `ctx`.**

---

## 1. Concept: two channels of `llama_decode`

```text
llama_decode(ctx, batch)
        │
        ├── SIDE EFFECT
        │     mutate the llama_context
        │     write new K/V
        │     produce logits for the last batch position
        │
        └── RETURN VALUE  (int)
              0   success
              ≠0  failure
```

Python:

```python
decode_result = self.llama_bridge.llama_bridge_decode_prompt(ctx, token_buffer, token_count)
if decode_result != 0:
    raise RuntimeError(...)
```

Nobody writes `ctx = decode_result`. That would throw away the pointer and store `0`.

C analogy from the PDF:

```c
void increment(int *x) { *x = *x + 1; }
int number = 5;
increment(&number);   /* number is now 6; return value was void */
```

`llama_decode` is `increment` plus a status `int`.

---

## 2. Concept: `ctx` is a pointer

```python
bridge.llama_bridge_create_context.restype = ctypes.c_void_p
ctx = bridge.llama_bridge_create_context(model, context_size)
```

```text
Python
  ctx  =  0x12345678
              │
              ▼
──────────────── native memory ────────────────
│ llama_context                                │
│   KV cache                                   │
│   logits / output buffer                     │
│   sequence position                          │
│   compute scratch                            │
───────────────────────────────────────────────
```

Python does not hold a dict of tokens. It holds an address. Passing `ctx` into C++ is passing that address. C++ receives `llama_context *`.

**Who needs this pointer:** `decode_prompt`, `sample_greedy`, `decode_one`, `free_context`.

**Who does not need it:** tokenize, token_to_piece, is_eog (those use `model` + a token id).

**When it exists:** From `create_context` to `free_context` inside one `generate()`.

**What stops the pointer from being valid:** `free_context`; process exit; never created.

**Memory:** The pointer is 8 bytes in Python. The object it points to is the KV allocation.

---

## 3. Concept: empty context

`llama_bridge_create_context` calls:

```cpp
llama_context_params params = llama_context_default_params();
params.n_ctx = context_size;
params.n_batch = context_size;
return llama_init_from_model(model, params);
```

Immediately afterward:

```text
model : weights, vocab, architecture     (full)
ctx   : capacity 1024, no tokens decoded (empty)
```

Conceptual slots:

```text
position  0   empty
position  1   empty
...
position  1023 empty
```

llama.cpp may not literally allocate 1024 distinct empty rows up front in the way this diagram suggests, but **no prompt has been processed**. Sampling before `decode_prompt` would be meaningless.

**Purpose of creating it empty:** Separate “allocate working memory” from “run the transformer.” The agent can still fail the fit check *before* paying for allocation (the check is actually just before create).

**Who needs an empty ctx:** `decode_prompt`, as a destination for the first KV writes.

---

## 4. Concept: batch is temporary, KV is persistent

### 4.1 Prompt path

```cpp
llama_batch batch = llama_batch_get_one(tokens, token_count);
return llama_decode(ctx, batch);
```

Suppose tokens are `[The, capital, of, France, is]`.

```text
batch (temporary)
  pos0 The
  pos1 capital
  pos2 of
  pos3 France
  pos4 is
        │
        ▼ llama_decode
ctx KV (persistent)
  pos0 K0 V0
  pos1 K1 V1
  pos2 K2 V2
  pos3 K3 V3
  pos4 K4 V4
  logits at pos4  →  ready to sample
```

After return, `batch` is a C++ local. It is not appended to `ctx`.

### 4.2 One-token path

```cpp
llama_token t = token;                     // e.g. id for " Paris"
llama_batch batch = llama_batch_get_one(&t, 1);
return llama_decode(ctx, batch);
```

```text
Before:  KV[0..4] = prompt
batch:   [Paris]
After:   KV[0..5] = prompt + Paris
         logits at pos5
```

Attention for `" Paris"`:

```text
Paris → Q5, K5, V5
Q5 attends to K0..K5
K5,V5 stored for the future
```

That store **is** the append.

### 4.3 Why we do not resend the whole string

```text
Call 1  batch = [The, capital, of, France, is]   → KV 0..4
Call 2  batch = [Paris]                          → KV 0..5
Call 3  batch = [.]                              → KV 0..6
```

Without a KV cache, every new token would re-encode the entire prefix. That is why autoregressive generation is affordable.

**CPU:** Prompt decode ~ O(n). Each extra token ~ O(n) attention over growing n, plus the usual FFN cost, with previous K/V reused.

---

## 5. Concept: model vs context (multi-user future)

```text
llama_model     = what Qwen knows          (read-mostly weights)
llama_context   = what this session saw    (KV, position)
```

```text
        Qwen3.5-0.8B
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
   ctx A    ctx B    ctx C
   User A   User B   User C
```

This repo does **not** implement A/B/C. It implements:

```text
        Qwen3.5-0.8B
              │
         ctx (one, temporary)
```

**Purpose of the distinction even now:** `close()` frees the model once; `generate()` frees many contexts. You do not reload 527 MiB per agent turn.

---

## 6. Mapping theory onto exact calls in this repo

| Moment | Code | ctx contents (conceptual) |
|---|---|---|
| After `create_context` | `inference.py` | Empty, capacity 1024 |
| After `decode_prompt` | `llama_bridge_decode_prompt` | KV for the full `conversation` string |
| After first `sample_greedy` | logits from last prompt token | Still same KV; no new position yet |
| After `decode_one(next_token)` | one more KV row | Prefix + that token |
| After `free_context` | — | Invalid pointer; do not reuse |

**Who calls `llama_decode`:** only the two bridge wrappers. Python never links `llama_decode` directly.

**When `llama_decode` runs:** once per prompt, then once per emitted non-EOG token.

**What user action stops further `llama_decode`:** Ctrl+C after the current call returns; process kill; agent `return` so no more `generate()`; EOG so the loop does not call `decode_one`; exception.

---

## 7. Logits and greedy sampling

After a successful decode, `ctx` holds a vector of scores, one per vocabulary entry. `llama_sampler_sample(..., ctx, -1)` reads the last position.

```text
decode_prompt("... ASSISTANT:\n")
        │
        ▼
   logits over vocab
        │
        ▼ greedy
   next_token  (often the start of <think> or { )
        │
        ▼ decode_one
   logits for the following token
        │
        ▼ ...
```

Sampling does **not** mutate the KV cache. Only `llama_decode` does.

That is why the loop is sample-then-decode, not decode-then-sample of the same token.

---

## 8. Memory picture during generation

```text
Process RSS
├── Python + imports
├── libllama + libllama_bridge
├── llama_model  (~527 MiB Q4_0)
└── llama_context (this generate only)
      ├── KV cache  (grows in use up to n_ctx)
      └── scratch / compute
```

`token_buffer` is extra `4 * token_count` bytes on the Python heap, live until `generate` returns.

`generated_text` is a Python `str` and is *not* the KV cache. You could discard the string and the model could still continue — if you still had the same `ctx`. We throw `ctx` away instead and, on the next agent turn, re-tokenize the whole conversation and decode it from scratch.

**Implication:** conversation memory for the *agent* is the Python string. Conversation memory for the *transformer* is rebuilt every `generate()`. That is simpler and more CPU-expensive than a persistent `ctx`.

---

## 9. CPU picture

| Call | Mutates ctx? | Compute |
|---|---|---|
| `tokenize` | No | CPU, no attention |
| `create_context` | Allocates empty | Alloc |
| `decode_prompt` | Yes | Full prompt attention + FFN |
| `sample_greedy` | No (reads logits) | Argmax |
| `token_to_piece` | No | Lookup |
| `decode_one` | Yes | One new position |
| `free_context` | Destroys | Free |

GPU offload (`n_gpu_layers = -1`) moves most matmuls off the CPU when Metal/CUDA is available. The mutation story does not change: the same `ctx` pointer is updated, possibly with GPU-resident KV.

---

## 10. Common misconceptions

| Belief | Reality |
|---|---|
| `decode_result` is the next token | It is `0` or an error code |
| `ctx` is a Python list of tokens | It is a pointer to native state |
| `decode_one` appends to a Python batch stored on `ctx` | It runs a new temporary batch; KV persists |
| We must send the full text every `decode_one` | We send one token; KV holds the past |
| Creating a context loads the model | The model is already loaded; context is session RAM |
| The agent’s `conversation +=` updates the KV cache | It only updates a Python string. The next `generate` rebuilds a new ctx from that string |
