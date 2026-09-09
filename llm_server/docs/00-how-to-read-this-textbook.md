# Chapter 00 — How to Read This Textbook

## 1. Purpose of this series

This is a reference textbook for the current local-LLM agent, not a tutorial that hides details. Each chapter answers the same questions for every important line of code:

| Question | Meaning |
|---|---|
| **What is this line doing?** | The immediate runtime effect. |
| **What is its purpose?** | Why the program cannot omit it. |
| **Who needs this line?** | Which caller, layer, or later step depends on it. |
| **Which function calls it?** | The Python or C++ caller on the stack. |
| **When is it called?** | Process start, each `generate()`, each token, each agent turn, shutdown. |
| **What stops this line?** | The user action or program condition that prevents it from running again, or that aborts it mid-flight. |
| **Memory / CPU** | Heap, native allocations, and approximate compute cost. |

The PDF [`local_llm_to_agent_textbook.pdf`](local_llm_to_agent_textbook.pdf) is the source conversation. These chapters keep that theory and attach it to the files as they exist on disk now.

## 2. The running program

There is one intended entry point:

```text
python llm_server/src/ai/agents/simple_agent.py
```

That process:

1. Loads `libllama.dylib` and `libllama_bridge.dylib` into the Python address space.
2. Loads the GGUF weights once.
3. Enters `run_agent()`, which may call `localLLM.generate()` more than once.
4. Always unloads the model in `finally: llm.close()`.

`main.py` at `llm_server/src/main.py` does nothing. Do not look there for the agent.

## 3. Syntax legend

The stack uses three languages in one process.

### 3.1 Python

| Syntax | Meaning in this codebase |
|---|---|
| `import x` | Bind a module name. No C code runs yet unless the module has import-time side effects. |
| `from a.b import C` | Bind one name from another module. **Does** run that module's top-level code. |
| `class localLLM:` | Define a type. Methods run only when constructed or called. |
| `def f(self, x: str) -> str:` | Method. `self` is the instance. `: str` and `-> str` are type hints; they do not enforce types at runtime. |
| `f"..."` / `f"""..."""` | Format string. Expressions in `{braces}` are evaluated and inserted. |
| `x = (ctypes.c_int * n)()` | Allocate an array of `n` C `int`s on the Python heap, usable as a pointer. |
| `ctypes.CDLL(path)` | `dlopen` a shared library. Symbols become attributes. |
| `bridge.fn.argtypes = [...]` | Tell ctypes the C parameter types so it converts Python values correctly. |
| `bridge.fn.restype = T` | Tell ctypes the C return type. Wrong `restype` silently corrupts pointers. |
| `if __name__ == "__main__":` | Run `main()` only when this file is the process entry script. |
| `try / finally` | `finally` always runs, including after exceptions. That is why you see `Model unloaded.` before a traceback. |
| `while True:` | Loop until `return` or `raise`. This *is* the agent. |
| `continue` | Skip the rest of the loop body and start the next iteration. |
| `raise RuntimeError(...)` | Abort the current call stack unless a `try` catches it. |

### 3.2 C++ (`bridge.cpp`)

| Syntax | Meaning in this codebase |
|---|---|
| `#include <llama.h>` | Pull llama.cpp public C API declarations. |
| `extern "C" { ... }` | Disable C++ name mangling so Python ctypes can find `llama_bridge_*` by exact name. |
| `llama_model *` | Pointer to native model object. Python sees this as `c_void_p`. |
| `llama_context *` | Pointer to one inference session (KV cache + buffers). |
| `nullptr` | Null pointer. Python treats this as `0` / falsy. |
| `static_cast<int>(x)` | Convert `size_t` from `strlen` to `int` for llama.cpp. |
| `return llama_decode(ctx, batch)` | Forward pass. Return code `0` means success. The interesting output is the mutated `ctx`. |

### 3.3 Bash (`build_bridge.sh`)

| Syntax | Meaning |
|---|---|
| `set -euo pipefail` | Exit on error, unset vars, or failed pipe. |
| `$(brew --prefix llama.cpp)` | Command substitution: Homebrew prefix for headers and libs. |
| `-dynamiclib` | Build a macOS `.dylib` instead of an executable. |
| `-Wl,-rpath,...` | Embed a runtime search path so `dlopen` finds `libllama.dylib`. |

## 4. Vocabulary used in every chapter

### 4.1 Model vs context

```text
llama_model     = long-lived weights ("what Qwen knows")
llama_context   = one generate() working state ("what has been decoded so far")
```

One process currently keeps **one model** and creates a **new context per `generate()`**.

### 4.2 Two output channels of `llama_decode`

```text
llama_decode(ctx, batch)
        │
        ├── SIDE EFFECT: mutate native memory at ctx
        │     KV cache, logits, sequence position
        │
        └── RETURN VALUE: int
              0  → success
              !=0 → failure
```

`decode_result` is not the next token. It is only a status code. See Chapter 06.

### 4.3 Batch vs KV cache

```text
batch     = temporary input for THIS forward pass
KV cache  = persistent attention state across passes
```

Tokens are not appended to a Python list inside `ctx`. They are absorbed as key/value vectors.

### 4.4 Stop-condition classes

When a chapter says “what stops this line,” it means one of these:

| Class | Examples |
|---|---|
| **User abort** | Ctrl+C in the terminal (`KeyboardInterrupt`); `kill` / Activity Monitor force-quit |
| **Normal completion** | `action == "answer"`; EOG token; `MAX_NEW_TOKENS` reached; `main()` returns |
| **Configured limit** | Context too small; token buffer too small (legacy path); piece buffer 256 bytes |
| **Hard failure** | Model load returns null; `llama_decode` ≠ 0; unknown tool; invalid JSON |
| **Process teardown** | `finally: llm.close()`; Python interpreter exit |

Ctrl+C during a native `llama_decode` may not interrupt instantly. The signal is delivered to Python; the C++ call usually finishes the current forward pass first.

## 5. Cost model used in later chapters

Orders of magnitude, not laboratory benchmarks.

| Operation | Typical cost in this project |
|---|---|
| Import Python modules | Microseconds to milliseconds. Negligible. |
| `ctypes.CDLL` | Milliseconds. Maps the dylib; does not load GGUF weights. |
| `llama_bridge_model_load` | Seconds. Reads ~527 MiB of Q4_0 weights. Dominant startup cost. |
| Two-pass tokenize | Milliseconds. CPU, no matrix multiplies. |
| `llama_bridge_create_context` | Tens to hundreds of milliseconds. Allocates KV cache and compute buffers. |
| `llama_bridge_decode_prompt` | **Largest per-generate CPU/GPU cost.** One forward pass over the whole prompt. |
| `llama_bridge_decode_one` | One forward pass over a single new token, using the KV cache. Cheap relative to the prompt pass, still the generation bottleneck. |
| `llama_bridge_sample_greedy` | Cheap argmax over logits, but this build **allocates and frees a sampler each token**. |
| `simple_test()` | Nanoseconds. Returns a constant string. |
| `extract_json` | Microseconds. String scan + `json.loads`. |

Resident set after a successful load is dominated by:

1. Model weights (~527 MiB, plus runtime tensors if not fully memory-mapped).
2. One context's KV cache and scratch buffers (grows with `CONTEXT_SIZE`).
3. Python interpreter and imported modules (tens of MiB).

Exact KV-cache bytes depend on layer count, head count, head dimension, and cache dtype. Chapter 11 gives the scaling formula.

## 6. What this textbook does not claim

- It does not document llama.cpp internals beyond what `bridge.cpp` calls.
- It does not yet document a Qwen chat template. Prompts are still hand-built strings. Chapter 10 explains why that produces `<think>` blocks.
- It does not document `llm_server/src/main.py` beyond noting it is unused.

## 7. Suggested study order for one sitting

1. Chapter 01 — see the whole machine.
2. Chapter 06 — understand `ctx` mutation. Everything else is plumbing around that.
3. Chapter 05 — read `generate()` with the mutation model in mind.
4. Chapter 08 — see why `generate()` is called in a loop.
5. Chapter 11 — account for memory so you know what `close()` is freeing.
