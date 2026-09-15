# Local LLM to Tool-Using Agent — Textbook Series

This folder is the written reference for the Python + C++ stack under `llm_server/src/ai`.

The companion transcript is [`local_llm_to_agent_textbook.pdf`](local_llm_to_agent_textbook.pdf). That PDF is a lossless record of the chat that built this stack. These markdown chapters turn that material, plus the current source files, into a textbook you can study file by file.

Version-to-version product history: [`../CHANGELOG.md`](../CHANGELOG.md).

## How to read

Start with [00-how-to-read-this-textbook.md](00-how-to-read-this-textbook.md). Then either:

- **Theory first:** chapters 01, 06, 07, 10, 11, 13, 14
- **Code first:** chapters 02 → 05 → 08 → 09 → 12 → 13 → 14

Every source file has a matching line-by-line chapter. Theory chapters explain *why* those lines exist.

## Chapter map

| Chapter | File | What it covers |
|---|---|---|
| 00 | [00-how-to-read-this-textbook.md](00-how-to-read-this-textbook.md) | Reading method, syntax legend, stop-condition vocabulary |
| 01 | [01-architecture-and-call-graph.md](01-architecture-and-call-graph.md) | Layers, who calls whom, process lifetime |
| 02 | [02-config-paths-and-constants.md](02-config-paths-and-constants.md) | `config.py`, `llama.py` constants, `prompts/llama.py` |
| 03 | [03-ctypes-ffi-and-signatures.md](03-ctypes-ffi-and-signatures.md) | `llama_utils.py`: Homebrew lookup, ctypes signatures, JSON extract |
| 04 | [04-cpp-bridge-line-by-line.md](04-cpp-bridge-line-by-line.md) | `bridge.cpp` native API |
| 05 | [05-inference-engine-line-by-line.md](05-inference-engine-line-by-line.md) | `inference.py` `localLLM` |
| 06 | [06-kv-cache-and-mutable-context.md](06-kv-cache-and-mutable-context.md) | Pointers, `llama_decode`, KV cache, empty context |
| 07 | [07-tokenization-and-dynamic-buffers.md](07-tokenization-and-dynamic-buffers.md) | Two-pass tokenize, context-size math |
| 08 | [08-agent-loop-line-by-line.md](08-agent-loop-line-by-line.md) | `simple_agent.py` |
| 09 | [09-tools-skills-and-registry.md](09-tools-skills-and-registry.md) | `tools.py`, `simple_skill.py` |
| 10 | [10-structured-output-and-think-blocks.md](10-structured-output-and-think-blocks.md) | JSON parsing, `<think>` origin |
| 11 | [11-memory-cpu-and-lifecycle.md](11-memory-cpu-and-lifecycle.md) | RAM, CPU, GPU offload, what frees what |
| 12 | [12-build-system.md](12-build-system.md) | `build_bridge.sh` |
| 13 | [13-agent-runtime-guarantees.md](13-agent-runtime-guarantees.md) | LLM vs runtime authority; `agent_v1` → `v1_0_1` → `v1_0_2` line by line |
| 14 | [14-qwen-chatml-and-one-json-per-turn.md](14-qwen-chatml-and-one-json-per-turn.md) | ChatML `messages`, `<\|im_end\|>` stop, `raw_decode` first JSON, `agent_v2` |

## Source files documented

```
llm_server/src/ai/
├── agents/simple_agent.py          → 08
├── agents/agent_v1.py              → 13
├── agents/agent_v1_0_1.py          → 13
├── agents/agent_v1_0_2.py          → 13
├── agents/agent_v2.py              → 14
├── config/config.py                → 02
├── config/model_config/llama.py    → 02, 14
├── engine/llama/bridge.cpp         → 04
├── engine/llama/build_bridge.sh    → 12
├── engine/llama/inference.py       → 05
├── skills/simple_skill.py          → 09
├── skills/tools/tools.py           → 09
├── utils/llama_utils.py            → 03, 14
├── utils/agent_utils.py            → 14
└── utils/prompts/llama.py          → 02
```

`llm_server/src/main.py` is a stub (`def main(): pass`) and is not part of the running agent path.
