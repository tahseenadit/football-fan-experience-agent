# Chapter 02 — Config, Paths, Constants, and Prompts

Source files:

- `llm_server/src/ai/config/config.py`
- `llm_server/src/ai/config/model_config/llama.py`
- `llm_server/src/ai/utils/prompts/llama.py`

These modules have **no functions that the agent calls in a loop**. They run at **import time** and leave constants in memory. Every later file reads those names.

---

## A. `config/config.py`

```python
import os
from utils.llama_utils import find_llama_library

AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(AI_DIR, "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_0.gguf")
LLAMA_LIBRARY_PATH = find_llama_library()
LLAMA_BRIDGE_PATH = os.path.join(AI_DIR, "engine/llama/libllama_bridge.dylib")
```

`AI_DIR` is computed from `__file__`. The comment in the source says `llm_server/src/ai/config`, but `dirname` is applied **twice**:

```text
__file__                          .../ai/config/config.py
dirname once                      .../ai/config
dirname twice                     .../ai          ← this is AI_DIR
```

So `MODEL_PATH` and `LLAMA_BRIDGE_PATH` are under `ai/`, which is correct.

### Line-by-line

| Line | What it does | Purpose | Who needs it | Called by | When | What stops it | Memory / CPU |
|---|---|---|---|---|---|---|---|
| `import os` | Bind the stdlib path module | Join and inspect filesystem paths | The three assignments below | Python import machinery | First import of `config.config` | Never “stopped”; already finished after import | ~0 |
| `from utils.llama_utils import find_llama_library` | Load `llama_utils.py` and bind one function | Need Homebrew discovery without duplicating it | `LLAMA_LIBRARY_PATH` | Import of `config.py` | Import time | Import error if `ai/` is not on `sys.path` | Loads `llama_utils` (tiny) |
| `AI_DIR = ...` | Resolve absolute directory of `ai/` | Anchor all other paths | `MODEL_PATH`, `LLAMA_BRIDGE_PATH` | Import | Once | N/A after assignment | One Python `str` |
| `MODEL_PATH = ...` | Build absolute path to the GGUF file | Tell `llama_bridge_model_load` which weights to read | `localLLM.__init__`; also imported (unused) by `simple_agent.py` | Import | Once | Wrong path → later load returns null | One `str`; file not opened yet |
| `LLAMA_LIBRARY_PATH = find_llama_library()` | Run `brew --prefix llama.cpp` and append `lib/libllama.dylib` | `ctypes.CDLL` must have a real dylib path | `localLLM.__init__` | Import of `config.py` | Once per process | `brew` missing → `CalledProcessError`; Homebrew package missing → same | Spawns a short-lived subprocess; result is one `str` |
| `LLAMA_BRIDGE_PATH = ...` | Path to our compiled wrapper | Second `CDLL` | `localLLM.__init__` | Import | Once | Missing file → `OSError` at `CDLL` time, not here | One `str` |

### Who imports this module

```text
engine/llama/inference.py   → LLAMA_LIBRARY_PATH, LLAMA_BRIDGE_PATH, MODEL_PATH
agents/simple_agent.py      → MODEL_PATH   (imported, not used in the current file)
```

### User actions that prevent these lines from running

- Never starting the agent (module never imported).
- Importing `inference` with a broken `sys.path` so `config.config` cannot be found.
- `find_llama_library()` raising: Homebrew not installed, `llama.cpp` formula not installed, or `brew` not on `PATH`.

Ctrl+C during `brew` aborts import. `localLLM` is never constructed. No model memory is allocated.

### Memory / CPU notes

`find_llama_library()` is the only non-trivial cost: one `subprocess.run`. Typical duration is tens to a few hundred milliseconds. It does **not** load llama.cpp into the process; it only prints a prefix path.

---

## B. `config/model_config/llama.py`

```python
MAX_TOKENS = 128
MAX_NEW_TOKENS = 64
CONTEXT_SIZE = 1024
```

Three integers. No functions.

| Name | Value | Meaning | Used by | Still load-bearing? |
|---|---|---|---|---|
| `MAX_TOKENS` | 128 | Historical fixed tokenize-buffer cap | Imported by `inference.py` | **No.** `generate()` now sizes the buffer with a probe call. The name remains imported. |
| `MAX_NEW_TOKENS` | 64 | Upper bound on tokens sampled after the prompt | `generate()` loop; context-size check | **Yes.** Stops generation even if the model never emits EOG. |
| `CONTEXT_SIZE` | 1024 | Configured KV-cache capacity in tokens | `generate()` creates `ctx` with this size; also compared to `token_count + MAX_NEW_TOKENS` | **Yes.** |

### Line-by-line

| Line | What it does | Purpose | Who needs it | When | What stops later use | Memory / CPU |
|---|---|---|---|---|---|---|
| `MAX_TOKENS = 128` | Bind an int | Leftover from the fixed-buffer era (PDF Chapter 4) | Nothing in `generate()` anymore | Import | Unused; changing it has no effect today | 28 bytes of a Python int object (interned small int) |
| `MAX_NEW_TOKENS = 64` | Bind an int | Cap decode-one / sample loop | `for _ in range(MAX_NEW_TOKENS)` and the context-fit check | Read every `generate()` | Lower it to `0` → no tokens generated; raise it → more RAM in the *check*, not in the allocated context (context is still `CONTEXT_SIZE` unless you change that too) | Negligible |
| `CONTEXT_SIZE = 1024` | Bind an int | Size passed to `llama_bridge_create_context` | Native `n_ctx` and `n_batch` | Every `generate()` | If `token_count + 64 > 1024`, `generate` raises and never creates `ctx` | Drives KV-cache allocation; see Chapter 11 |

### Callers

Only `inference.py`:

```python
from config.model_config.llama import MAX_TOKENS, CONTEXT_SIZE, MAX_NEW_TOKENS
```

### Stop conditions that involve these numbers

| Condition | What the user / operator did | Effect |
|---|---|---|
| Prompt tokens + 64 > 1024 | System prompt + history grew (agent loop appends) | `RuntimeError: Context too small` |
| Model never emits EOG | Normal for some completions | Loop stops after 64 new tokens |
| Operator edits `CONTEXT_SIZE` down | Config change | Faster / smaller context; more likely to hit the raise |
| Operator edits `MAX_NEW_TOKENS` up without raising `CONTEXT_SIZE` | Config change | The fit check fails sooner |

### Why 1024 and 64 exist together

```text
required_context_size = token_count + MAX_NEW_TOKENS
if required_context_size > CONTEXT_SIZE: raise
```

The configured context must hold the prompt **and** every token you are willing to generate. The PDF (Chapter 4) derived this after a 512-context, 128-new-token example overflowed.

---

## C. `utils/prompts/llama.py`

```python
prompt = "A goalkeeper is a player who is responsible for protecting the"
SYSTEM_PROMPT = """
You are an agent.
...
Do not output anything outside the JSON.
"""
```

| Name | Role | Used? |
|---|---|---|
| `prompt` | Early completion-style test string from before the agent existed | **Not imported** by the agent. Dead constant. |
| `SYSTEM_PROMPT` | Instructions + tool schema + JSON contract | `run_agent()` interpolates it into `conversation` |

### `SYSTEM_PROMPT` line-by-line (semantic, not every blank line)

| Region | What it does | Purpose | Who needs it |
|---|---|---|---|
| `You are an agent.` | Role text | Bias the model toward tool-or-answer behavior | The model, via tokens |
| `You have access to these tools:` + `simple_test` block | Advertise one tool | Without this, the model has no reason to emit `action: tool` | `run_agent` + the model |
| `You must respond using valid JSON only.` | Format constraint | Soft constraint only. Qwen may still emit `<think>` first (Chapter 10) | `extract_json` depends on a `{...}` existing *somewhere* |
| Example tool JSON | Show the exact schema | Few-shot of the action the Python parser understands | `decision["action"]`, `decision["name"]` |
| Example answer JSON | Show the other legal action | Lets the loop terminate | `return decision["content"]` |
| `Do not output anything outside the JSON.` | Negative instruction | Frequently violated by thinking-mode models | Parser, not the model runtime |

### Who calls / who reads it

```text
simple_agent.run_agent
    conversation = f"""
    {SYSTEM_PROMPT}
    USER:
    {user_message}
    ASSISTANT:
    """
        │
        ▼
localLLM.generate(conversation)   # tokenizes this entire string
```

It is **not** a function. Nothing “calls” `SYSTEM_PROMPT`. `run_agent` reads the global once per invocation (and the string is then mutated by appending tool turns).

### When it is used

- Once at the start of `run_agent` to build `conversation`.
- Indirectly on every later `generate()`, because the system text remains at the front of `conversation`.

### What stops it from being sent

- `main()` never calling `run_agent`.
- An exception in `localLLM()` before `run_agent`.
- After the first `generate()`, the same system text is still in the string; you cannot “unsend” it without rebuilding `conversation`.

### Memory / CPU

The prompt is a few hundred characters. After tokenization it is typically on the order of **100–200 tokens** (exact count printed as `Required token capacity`). That is the bulk of the first `decode_prompt` cost.

`prompt` (the goalkeeper sentence) occupies a few dozen bytes and is never tokenized by the agent path.

### Syntax used here

- Triple-quoted string: preserves newlines, which become tokens (`\n` is not free).
- JSON examples inside the string are **text**, not Python dicts. The model must *emit* JSON later; these braces do not execute.

---

## D. Cross-module contract

```text
config.py           "where are the binaries and the GGUF?"
model_config/llama  "how big may a generate() be?"
prompts/llama.py    "what should the model believe it is?"
```

Change paths → load fails. Change constants → generate raises or truncates. Change `SYSTEM_PROMPT` → the model’s first tokens change, which can change whether you get a tool call, an answer, or a `<think>` wrapper.

There is no reload API. Edit the file and start a new process.
