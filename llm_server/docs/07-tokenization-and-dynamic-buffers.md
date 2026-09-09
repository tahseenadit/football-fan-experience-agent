# Chapter 07 — Tokenization and Dynamic Buffers

Source of the bug and the fix: [`local_llm_to_agent_textbook.pdf`](local_llm_to_agent_textbook.pdf) Chapter 4, now implemented in `inference.py`.

---

## 1. What a token is

A token is an integer id in the model’s vocabulary. Text is not what the transformer multiplies.

```text
"Please run the simple test."
        │  llama_tokenize(vocab, ...)
        ▼
[151644, 872, 198, ...]     # example IDs, not the real sequence
```

**Who needs token IDs:** `llama_decode` (via `decode_prompt`).

**Who does not:** `run_agent` (it only holds strings). `extract_json` (strings). `simple_test` (strings).

**When tokenization runs:** Twice per `generate()` (probe + fill).

**What stops it:** `generate` not called; probe unexpected; second pass failure; process abort.

**CPU:** Sub-millisecond to a few milliseconds for a ~200-token agent prompt. **Memory:** `4 * N` bytes for the C int array, plus a UTF-8 `bytes` copy of the prompt.

---

## 2. The failure that forced the two-pass API

Early `generate()` did:

```python
token_buffer = (ctypes.c_int * MAX_TOKENS)()    # MAX_TOKENS = 128
token_count = tokenize(model, prompt, token_buffer, MAX_TOKENS)
if token_count < 0:
    raise RuntimeError("Failed to tokenize prompt")
```

That worked for:

```text
The capital of France is
```

It failed when `run_agent` sent:

```text
SYSTEM_PROMPT + USER + message + ASSISTANT
```

`llama_tokenize` returned a **negative** number, e.g. `-173`.

| Interpretation people used | Actual meaning |
|---|---|
| Tokenization failed | Buffer too small |
| `-173` is an error enum | “I need 173 slots” |

Python then raised, `finally` ran `llm.close()`, and the traceback appeared **after** `Model unloaded.`

---

## 3. llama.cpp’s contract

```text
llama_tokenize(vocab, text, text_len, tokens, n_max_tokens, add_special, parse_special)
```

| Return | Meaning |
|---|---|
| `n >= 0` | Wrote `n` ids into `tokens` |
| `n < 0` | Need `−n` slots; `tokens` is incomplete or unused |

Probe pattern:

```text
tokenize(tokens=NULL, n_max=0)  →  -173
allocate 173 ints
tokenize(buffer, 173)           →  173
```

ctypes:

```python
# POINTER(c_int) + None  =>  C nullptr
required = tokenize(model, text, None, 0)
required = -required
token_buffer = (ctypes.c_int * required)()
token_count = tokenize(model, text, token_buffer, required)
```

**Purpose of pass 1:** Learn `N` without maintaining a magic `MAX_TOKENS`.

**Purpose of pass 2:** Fill exactly `N` integers.

**Who needs pass 1:** The array constructor.

**Who needs pass 2:** `decode_prompt`.

**When:** Every generate, because each conversation string has a different length.

**What stops pass 2:** Pass 1 raising (`required >= 0`).

**Memory:** Peak tokenize RAM is one `bytes` prompt + `4N` ints. `N` here is hundreds, not millions.

---

## 4. Why `if required >= 0` exists on the probe

With `tokens=NULL` and `max_tokens=0`, a healthy llama.cpp returns negative. A non-negative probe means:

- signatures are wrong (`None` not passed as a pointer), or
- the C wrapper ignored the null and wrote somewhere, or
- API behavior changed.

Raising is cheaper than trusting a bogus `N`.

---

## 5. Special-token flags in the bridge

```cpp
return llama_tokenize(vocab, prompt_text, text_len, tokens, max_tokens, true, true);
```

Both trailing `true`s:

1. **Add special tokens** — BOS / role markers if the vocab expects them for raw tokenize.
2. **Parse special tokens** — literal `<think>` in the *prompt* can become a special id instead of character pieces.

**Who needs this:** Alignment with how Qwen was trained. Wrong flags → different IDs → garbage generation.

**CPU:** Same order of magnitude either way.

This is **not** the same as applying the official Qwen **chat template**. The agent still concatenates `"USER:"` / `"ASSISTANT:"` as plain text (Chapter 10). Flags only affect how that raw text is cut into ids.

---

## 6. Context-size arithmetic

After tokenize:

```python
required_context_size = token_count + MAX_NEW_TOKENS
if required_context_size > CONTEXT_SIZE:
    raise RuntimeError(...)
```

Example from the PDF, scaled to current constants:

```text
token_count      = 400
MAX_NEW_TOKENS   = 64
CONTEXT_SIZE     = 1024
required         = 464
464 <= 1024  →  OK, create ctx with 1024
```

Overflow example:

```text
token_count      = 980
MAX_NEW_TOKENS   = 64
required         = 1044
1044 > 1024  →  raise, never create ctx
```

**Purpose:** `n_ctx` is a hard native limit. Decoding past it fails or corrupts.

**Who needs the check:** `create_context` / later `decode_one` calls.

**When:** Every generate, after tokenize, before native alloc.

**What stops `create_context`:** This raise. User-visible when the agent history grows (`conversation +=` tool turns).

**Memory:** The check itself is two integers. It **saves** a large failed allocation.

**Alternative in the PDF (not used):**

```python
context_size = max(CONTEXT_SIZE, token_count + MAX_NEW_TOKENS)
```

That would grow RAM instead of raising. The current code prefers a loud error.

`MAX_TOKENS = 128` is unrelated to this check and is unused.

---

## 7. Detokenization buffer (256 bytes)

Separate from tokenize:

```python
piece_buffer = ctypes.create_string_buffer(256)
piece_length = llama_bridge_token_to_piece(model, next_token, piece_buffer, 256)
if piece_length < 0:
    raise RuntimeError(f"Piece buffer too small: {-piece_length}")
```

`llama_token_to_piece` uses the same “negative means need more” idea, but for **bytes of one token**, not token count.

| Buffer | Unit | Typical size here | Grow strategy |
|---|---|---|---|
| Token id buffer | `int` × N | N ≈ prompt length | Two-pass, exact |
| Piece buffer | `char` × 256 | One token’s UTF-8 | Fixed; raise if short |

**Who needs 256:** Almost every Qwen piece is far smaller. Multi-byte punctuation and special tokens still fit.

**What stops generation:** A hypothetical piece > 256 bytes.

**CPU / memory:** 256 bytes allocated **per loop iteration** (new buffer each time). Could be reused; cost is irrelevant next to `decode_one`.

---

## 8. UTF-8 and `errors="replace"`

```python
piece = piece_buffer.raw[:piece_length].decode("utf-8", errors="replace")
```

| Syntax | Purpose |
|---|---|
| `.raw` | Full buffer bytes, including trailing NULs |
| `[:piece_length]` | Only the bytes llama.cpp wrote |
| `.decode("utf-8")` | Match `prompt.encode("utf-8")` |
| `errors="replace"` | Do not crash on a split codepoint |

A token may be a byte fragment. The *concatenation* `generated_text += piece` can become valid UTF-8 even if a single piece would not. `replace` is conservative: one bad piece becomes `` instead of aborting the agent.

**Who needs this:** Streaming print and the string returned to `extract_json`.

---

## 9. Tokenization vs the agent loop

```text
run_agent builds a STRING
        │
        ▼
generate tokenizes THAT string from scratch
        │
        ▼
decode_prompt writes ALL those tokens into a NEW ctx
```

After a tool call, `conversation` is longer. The next tokenize `N` is larger. The next `decode_prompt` is more expensive. The KV cache of the previous generate was freed; nothing is incremental across agent turns.

**CPU implication:** Turn 2 re-pays attention over the system prompt and the first model output. That is wasted compute versus a persistent context, and it is the current design.

**Memory implication:** Peak KV is still one context of 1024, not two at once (create/free are sequential).

---

## 10. Line-level tokenize path (who / when / stop)

| Step | Function | Called by | When | Stopped by |
|---|---|---|---|---|
| Encode prompt | `str.encode` | `generate` | Each generate, twice | Generate not entered |
| Probe | `llama_bridge_tokenize` | `generate` | Each generate | Ctrl+C; raise |
| Negate | `required = -required` | `generate` | After probe | Probe raise |
| Alloc ids | `c_int * required` | `generate` | After negate | Process OOM (unlikely) |
| Fill | `llama_bridge_tokenize` | `generate` | After alloc | Raise |
| Vocab fetch | `llama_model_get_vocab` | C++ tokenize | Each bridge tokenize | Model already freed (should not happen) |
| `strlen` | C++ | Each bridge tokenize | Each call | Null prompt (does not happen) |
| `llama_tokenize` | llama.cpp | C++ | Each call | Same |

**User action that stops the tokenize lines from ever running:** not starting `simple_agent.py`; crashing in `localLLM()`; `run_agent` returning before a second turn (first turn already tokenized).

**User action that stops them mid-flight:** Ctrl+C (usually between the two passes or after, because each pass is short).
