# Chapter 01 — Architecture and Call Graph

## 1. What this system is

The process is a **single-user, single-model, locally hosted agent**.

```text
User types nothing interactively in the current build.
main() hard-codes: "Please run the simple test."

USER MESSAGE (string in main)
        │
        ▼
   run_agent()                  Python control loop
        │
        │  conversation string
        ▼
   localLLM.generate()          Python inference wrapper
        │
        │  ctypes calls
        ▼
   libllama_bridge.dylib        Our C++ ABI
        │
        │  llama.cpp C API
        ▼
   libllama.dylib               Transformer runtime
        │
        ▼
   Qwen3.5-0.8B Q4_0.gguf       Weights on disk, then in RAM/GPU
```

The LLM never calls Python by itself. Python calls the model, reads text, optionally runs a tool, then calls the model again. That loop is the entire difference between “a text generator” and “an agent.”

```text
LLM     = text generator
Agent   = LLM + tools + execution loop + conversation state
```

## 2. Layers and ownership

| Layer | Files | Owns | Lifetime |
|---|---|---|---|
| Entry | `agents/simple_agent.py` | Conversation string, tool dispatch | One process |
| Prompts | `utils/prompts/llama.py` | `SYSTEM_PROMPT` constant | Import time |
| Tools | `skills/tools/tools.py`, `skills/simple_skill.py` | Name → function map | Import time |
| Config | `config/config.py`, `config/model_config/llama.py` | Paths and numeric limits | Import time |
| Inference | `engine/llama/inference.py` | `localLLM` instance: libs, model pointer | Constructed in `main()`, freed in `close()` |
| FFI helpers | `utils/llama_utils.py` | Signatures, Homebrew path, JSON slice | Called from config and inference |
| Native bridge | `engine/llama/bridge.cpp` → `libllama_bridge.dylib` | Thin wrappers | Loaded once via `CDLL` |
| llama.cpp | Homebrew `libllama.dylib` | Weights, contexts, decode | Backend init/shutdown |

## 3. Who imports whom

Import arrows (A → B means A imports B):

```text
simple_agent.py
    ├── config.config                 (MODEL_PATH; currently unused in the agent)
    ├── skills.tools.tools            (TOOLS)
    ├── engine.llama.inference        (localLLM)
    ├── utils.prompts.llama           (SYSTEM_PROMPT)
    └── utils.llama_utils             (extract_json)

inference.py
    ├── config.config                 (LLAMA_LIBRARY_PATH, LLAMA_BRIDGE_PATH, MODEL_PATH)
    ├── config.model_config.llama     (MAX_TOKENS, CONTEXT_SIZE, MAX_NEW_TOKENS)
    └── utils.llama_utils             (define_llama_bridge_signatures, init_llama)

config.py
    └── utils.llama_utils             (find_llama_library)   ← runs `brew` at import

tools.py
    └── skills.simple_skill           (simple_test)

llama_utils.py
    └── (stdlib only: subprocess, pathlib, ctypes, json)
```

**Import-time work that costs something:**

- `config.py` runs `brew --prefix llama.cpp` inside `find_llama_library()`.
- `tools.py` and `simple_agent.py` insert `ai/` onto `sys.path`.
- No GGUF file is read until `localLLM.__init__()`.

## 4. Runtime call graph

### 4.1 Process start

```text
if __name__ == "__main__":          # simple_agent.py
    main()
        localLLM.__init__()         # inference.py
            ctypes.CDLL(libllama)
            ctypes.CDLL(libllama_bridge)
            define_llama_bridge_signatures()
            init_llama()
                llama_bridge_init()
                    llama_backend_init()
            llama_bridge_model_load()
            llama_bridge_model_size()
            llama_bridge_model_n_params()
        run_agent(llm, "Please run the simple test.")
```

### 4.2 Each agent turn

```text
run_agent
    while True:
        llm.generate(conversation)
            llama_bridge_tokenize(..., None, 0)          # size probe
            llama_bridge_tokenize(..., buffer, n)        # fill buffer
            llama_bridge_create_context()
            llama_bridge_decode_prompt()                 # fill KV cache
            for _ in range(MAX_NEW_TOKENS):
                llama_bridge_sample_greedy()
                llama_bridge_is_eog()                    # maybe break
                llama_bridge_token_to_piece()
                llama_bridge_decode_one()                # append one KV position
            llama_bridge_free_context()
        extract_json(response)
        if action == "answer":
            return content                               # leaves the while
        if action == "tool":
            TOOLS[name](**arguments)                     # e.g. simple_test()
            conversation += response + tool result
            continue                                     # next generate()
        raise ValueError                                 # unknown action
```

### 4.3 Process end

```text
main
    try:
        run_agent(...)
    finally:
        llm.close()
            llama_bridge_model_free()
            llama_bridge_shutdown()
                llama_backend_free()
```

`close()` runs on success, on `KeyboardInterrupt`, and on any exception from `run_agent`. That is why a crash still prints `Model unloaded.` before the traceback.

## 5. Object identity across the stack

| Python name | ctypes type | C++ type | Created by | Freed by |
|---|---|---|---|---|
| `self.llama_lib` | `CDLL` | mapped `libllama.dylib` | `CDLL(LLAMA_LIBRARY_PATH)` | process exit (not explicitly closed) |
| `self.llama_bridge` | `CDLL` | mapped `libllama_bridge.dylib` | `CDLL(LLAMA_BRIDGE_PATH)` | process exit |
| `self.model` | `c_void_p` | `llama_model *` | `llama_bridge_model_load` | `llama_bridge_model_free` |
| `ctx` (local in `generate`) | `c_void_p` | `llama_context *` | `llama_bridge_create_context` | `llama_bridge_free_context` |
| `token_buffer` | `c_int * n` | `int *` | Python constructor | garbage collected after `generate` returns |
| `piece_buffer` | 256-byte C string | `char *` | `create_string_buffer` | garbage collected each loop iteration |

Python never owns the bytes of the KV cache. It owns an integer address.

```text
Python ctx = 0x12345678
                │
                ▼
         native llama_context { KV cache, logits, buffers }
```

## 6. What a user can do to stop execution

The current `main()` does not read stdin. The only human actions are at the OS / terminal:

| User action | What stops | What still runs |
|---|---|---|
| Let the program finish | Nothing; `answer` returns, `close()` runs | Clean shutdown |
| Ctrl+C during `__init__` (model load) | Load may finish the current I/O; then `KeyboardInterrupt` | If `localLLM()` has not returned, `try/finally` is **not** entered, so `close()` is **not** called. The process then exits and the OS reclaims mappings. |
| Ctrl+C during `generate` / `run_agent` | Current Python bytecode; native decode usually completes the in-flight pass | `finally: llm.close()` **does** run |
| `kill -9` | Immediate death | No `close()`, no Python `finally`. OS reclaims memory. |
| Activity Monitor “Quit” | Usually SIGTERM; Python may not handle it | Not guaranteed to hit `close()` |

There is no in-app “Stop generating” button.

## 7. Concurrency model

There is none.

- One thread runs Python.
- ctypes calls are blocking.
- One `ctx` exists at a time (created and freed inside `generate`).
- The model pointer is shared across `generate()` calls but never used from two threads.

The PDF’s future picture — one model, many contexts for many users — is **not implemented**. Today:

```text
ONE MODEL
   │
   └── generate #1  ctx #1  (freed)
   └── generate #2  ctx #2  (freed)
```

not:

```text
ONE MODEL
   ├── ctx A  User A     ← not in this repo
   ├── ctx B  User B
   └── ctx C  User C
```

## 8. Memory / CPU at the architecture level

| Phase | RAM | CPU / GPU |
|---|---|---|
| Imports | ~30–80 MiB Python | `brew` subprocess once |
| After `CDLL` | Libraries mapped | Almost idle |
| After model load | **+ ~527 MiB** weights (plus GPU copies if offloaded) | High during load, then idle |
| Each `generate` | **+ context** (KV + scratch), then freed | Prompt decode is the spike; then one-token steps |
| Tool call | Bytes for `"Simple test"` | Negligible |
| After `close` | Weights released; Python process still alive until exit | Idle |

GPU: `bridge.cpp` sets `n_gpu_layers = -1` (“offload as many layers as possible”). On a Mac with Metal, much of the matmul can leave the CPU. The Python process still holds pointers and the GGUF mapping.

## 9. File-to-question index

| If you want to know… | Read |
|---|---|
| Where paths and limits live | Chapter 02 |
| Why ctypes signatures must be declared | Chapter 03 |
| What each native function does | Chapter 04 |
| The full `generate()` loop | Chapter 05 |
| Why `decode_result` is not the model state | Chapter 06 |
| Why the first tokenize call uses `None, 0` | Chapter 07 |
| Why `while True` is the agent | Chapter 08 |
| How a name becomes a Python function | Chapter 09 |
| Why `<think>` appears and JSON parse used to crash | Chapter 10 |
| Byte-level RAM and CPU | Chapter 11 |
| How the dylib is compiled | Chapter 12 |
