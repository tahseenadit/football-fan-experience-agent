# Chapter 11 — Memory, CPU, GPU, and Lifecycle

This chapter accounts for bytes and compute across one run of `python .../simple_agent.py`. Numbers that come from this repo’s own logs are marked **measured**. Everything else is a scaling model.

Measured on the Qwen3.5-0.8B Q4_0 used here:

| Quantity | Value | Source |
|---|---|---|
| Weight file / `llama_model_size` | **526.50 MiB** | `localLLM.__init__` print |
| Parameter count | **752,393,024** | same |
| Configured `n_ctx` | **1024** | `CONTEXT_SIZE` |
| Max new tokens | **64** | `MAX_NEW_TOKENS` |
| GPU offload request | **all possible layers** | `params.n_gpu_layers = -1` |

---

## 1. Process lifetime and who holds RAM

```text
t0  interpreter starts                    Python + stdlib
t1  imports                               + ai modules; brew subprocess (gone)
t2  CDLL both dylibs                      + mapped libllama, libllama_bridge
t3  llama_backend_init                    + small backend state
t4  model_load                            + ~527 MiB weights  [PEAK long-lived]
t5  generate: tokenize buffers            + 4N bytes (N ≈ prompt tokens)
t6  create_context                        + KV + scratch     [PEAK total]
t7  decode / sample loop                  KV fills; compute buffers reuse
t8  free_context                          − context
t9  maybe generate again                  + another context (sequential, not stacked)
t10 close: model_free + backend_free      − weights
t11 process exit                          OS reclaims the rest
```

**Who needs the long-lived 527 MiB:** every `generate`. **Who frees it:** `close()`, or the OS if the process dies.

**User actions:**

| Action | RAM consequence |
|---|---|
| Wait for clean finish | t10 then t11 |
| Ctrl+C after t4 | `finally` → t10 |
| Ctrl+C during t4 | t10 may be skipped; OS reclaims |
| `kill -9` | jump to t11 |

---

## 2. Allocation owners (table)

| Object | Allocator | Size class | Freed by | Leaked if |
|---|---|---|---|---|
| Python heaps, modules | CPython | tens of MiB | exit | — |
| `libllama.dylib` mapping | `dlopen` | library text/data | exit (not `close`) | process lives after close |
| `libllama_bridge.dylib` | `dlopen` | small | exit | same |
| `llama_model` | `llama_model_load_from_file` | **~527 MiB** + runtime tensors | `llama_model_free` | `close` skipped |
| Backend | `llama_backend_init` | small | `llama_backend_free` | `close` skipped |
| `llama_context` | `llama_init_from_model` | KV + scratch ∝ `n_ctx` | `llama_free` | exception in `generate` after create |
| `token_buffer` | `(c_int * N)()` | `4N` bytes | GC after `generate` | — |
| UTF-8 prompt `bytes` | `encode` | `len(prompt)` | GC | — |
| `piece_buffer` | `create_string_buffer(256)` | 256 B × iterations | GC | — |
| `generated_text` | Python `str` | ~bytes of output | returned to agent, then GC | — |
| `conversation` | Python `str` | grows per tool | `run_agent` return | — |
| Sampler chain | `llama_sampler_chain_init` **per token** | small | `llama_sampler_free` same call | crash mid-sample (rare) |

---

## 3. KV cache scaling (theory)

For a decoder-only transformer, cache size is on the order of:

```text
KV_bytes ≈
    n_layers
  × n_kv_heads
  × head_dim
  × n_ctx
  × 2          (K and V)
  × bytes_per_element
```

`bytes_per_element` is often 2 (FP16/BF16) in the cache, sometimes quantized further depending on llama.cpp build and model metadata.

**Who needs this RAM:** `llama_decode`. **Purpose:** avoid recomputing prefix keys/values.

**When it grows in *use*:** each `decode_prompt` / `decode_one` writes new positions, up to `n_ctx`.

**What stops growth:** `free_context`; hitting `n_ctx` (we error before that); process exit.

This repo does not print KV bytes. To observe them on macOS: Activity Monitor or `sample` / Instruments on the Python PID during `Context created.` vs after `Model loaded`.

**CPU connection:** larger used prefix → each `decode_one` attends over more cached positions. The **allocated** `n_ctx=1024` sets the *maximum*, not the per-token used length.

Scratch / compute buffers inside `ctx` can be comparable to or larger than KV for small `n_ctx`. Treat “context RAM” as “whatever `llama_init_from_model` reserved,” not KV alone.

---

## 4. GPU vs CPU

```cpp
params.n_gpu_layers = -1;
```

| Setting | Meaning |
|---|---|
| `0` | All layers on CPU |
| `k > 0` | Offload `k` layers | 
| `-1` | Offload as many as the backend allows |

On Apple Silicon, llama.cpp typically uses Metal. Weights and much of the matmul move to GPU memory. The process still:

- maps the GGUF on the CPU side
- holds `ctx` bookkeeping
- runs Python on the CPU between native calls

**Who needs GPU RAM:** decode_prompt, decode_one. **Who needs CPU:** tokenize, JSON, agent loop, sampler setup, ctypes marshalling.

**What stops GPU use:** a CPU-only llama.cpp bottle; GPU OOM (load may fail); `n_gpu_layers` changed in source and rebuild.

**User action:** quitting the process returns GPU RAM. `close()` should do that before exit if `finally` runs.

ctypes calls are **blocking**. One core waits in the dylib during decode. Other cores / GPU may run inside llama.cpp.

---

## 5. CPU timeline (one generate)

Let `P` = prompt tokens, `G` = generated tokens actually produced (`≤ 64`).

| Phase | Complexity (order) | Notes |
|---|---|---|
| Tokenize ×2 | O(bytes of prompt) | CPU only |
| create_context | O(1) plus allocator work ∝ n_ctx | |
| decode_prompt | **O(P)** attention + FFN for all prompt positions (implementation may batch) | Largest single kernel sequence |
| Each step i = 1..G | sample O(V) + decode_one **O(P+i)** attention | V = vocab size |
| free_context | allocator | |

**Dominant:** `decode_prompt` + Σ `decode_one`.

**Wasted by think tokens:** each think token is a full `decode_one` that the agent then throws away in `extract_json`.

**Wasted by new ctx per generate:** turn 2 re-runs `decode_prompt` over the system prompt and the first model output instead of continuing the old KV.

**Sampler tax:** construct + destroy greedy chain every token. CPU-visible in profiles, still << decode.

---

## 6. Agent-turn compounding

Two generates in the demo:

```text
Generate 1
  P1 ≈ |SYSTEM + USER + ASSISTANT|
  G1 ≈ think + tool JSON
  free ctx

Generate 2
  P2 ≈ P1 + G1 + tool result + cue     # larger prompt
  G2 ≈ think? + answer JSON
  free ctx
```

**RAM:** still one context at a time. Peak is max(ctx at gen1, ctx at gen2), both `n_ctx=1024`, so **peak native context is the same**. Prompt *used* length is larger on gen2, so **compute** rises.

**What stops a third generate:** `action == "answer"`; or `P + 64 > 1024`.

**User cannot** ask the current program for another turn; `main` has one `run_agent` call.

---

## 7. Line-level cost sheet (hot path only)

| Code | CPU | RAM delta | Called |
|---|---|---|---|
| `CDLL` | ms | library maps | once |
| `llama_bridge_model_load` | seconds | **+527 MiB** | once |
| `llama_bridge_tokenize` probe | ms | 0 extra buffers | / generate |
| `(c_int * N)()` | µs | **+4N** | / generate |
| `llama_bridge_tokenize` fill | ms | writes the 4N | / generate |
| `create_context` | 10–100+ ms | **+ context** | / generate |
| `decode_prompt` | **high** | fills KV[0..P) | / generate |
| `sample_greedy` | low + malloc | tiny | / token |
| `is_eog` | tiny | 0 | / token |
| `token_to_piece` | tiny | 256 B | / token |
| `decode_one` | **high / token** | +1 KV row | / token |
| `free_context` | low | **− context** | / generate |
| `simple_test` | tiny | 0 | / tool |
| `extract_json` | tiny | substring | / turn |
| `model_free` | low–medium | **−527 MiB** | once |

---

## 8. What each user-visible print costs

Prints are I/O. `flush=True` on each piece forces a write per token so you see streaming. That can be milliseconds of syscall overhead, still dwarfed by decode on this model.

`Generated text: ...!r` walks the whole string once.

---

## 9. Leak and double-free notes

| Scenario | Leak? |
|---|---|
| Happy path | No native leak intended |
| Raise after `create_context`, before `free_context` | **Yes, `ctx`** until process exit |
| `close()` not called | **Yes, model** until process exit |
| `close()` called twice | **Possible double-free** (`main` does not) |
| `CDLL` handles after `close` | Libraries remain mapped; calling native after `shutdown` is undefined |

For a future HTTP server, wrap `generate` in `try/finally: free_context` and never shut down the backend per request.

---

## 10. Stop-condition × resource matrix

| Stop | Model RAM | Current ctx | Python conversation |
|---|---|---|---|
| EOG / 64 tokens | kept | freed at end of generate | kept |
| `answer` return | kept until `close` | already freed | discarded when `run_agent` returns |
| Context-too-small raise | kept until `finally` | not created | discarded as stack unwinds |
| Ctrl+C in loop | freed in `finally` | may leak if mid-generate | discarded |
| `kill -9` | OS | OS | OS |

---

## 11. Practical observations for this machine

When you run the agent, expect:

1. A multi-second pause at `Loading model:` — disk + dequant + GPU upload.
2. RSS jump of roughly **half a gigabyte**.
3. A shorter pause at `Context created.` / `Prompt decoded`.
4. Token-by-token prints (think tags, then JSON).
5. A second, similar generate after `TOOL RESULT`.
6. `Model unloaded.` and RSS dropping before the process exits.

If RSS does not drop after unload, leftover is Python + mapped dylibs until exit. That is normal.
