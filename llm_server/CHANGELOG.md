# Changelog

All notable changes to the local LLM agent under `llm_server/` are recorded here.

Versions follow the agent entry files:

| Version | Entry file |
|---|---|
| 0.x | `inference.py` / first `simple_agent` loop (pre-numbered) |
| 1.0.0 | `src/ai/agents/agent_v1.py` |
| 1.0.1 | `src/ai/agents/agent_v1_0_1.py` |
| 1.0.2 | `src/ai/agents/agent_v1_0_2.py` |
| 1.0.3 | `src/ai/agents/agent_v1_0_3.py` |
| 2.0.0 | `src/ai/agents/agent_v2.py` |

Textbook chapters: `docs/00-how-to-read-this-textbook.md` through `docs/14-qwen-chatml-and-one-json-per-turn.md`.

---

## [2.0.0] — 2026-09-16

ChatML conversation format. One assistant turn per `generate`. First JSON object only.

### Added

- `utils/agent_utils.py` → `apply_chat_template(messages, add_generation_prompt, enable_thinking)`: Qwen ChatML (`<|im_start|>role` … `<|im_end|>`). With `enable_thinking=False`, appends an empty `<think>\n\n</think>\n\n` block (thinking off).
- `config/model_config/llama.py` → `ENABLE_THINKING = False`.
- `agent_v2.py`: `messages` list (`system` + `user`) instead of a `USER:` / `ASSISTANT:` string. Each step: `apply_chat_template` → `llm.generate(prompt)`.
- Parse-failure and invalid-answer paths append ChatML `assistant` + `user` messages (no `ASSISTANT:` cue in the string).
- Tool follow-up appends the model `response` as `assistant` and the structured `TOOL RESULT` as `user`. System prompt is **not** rebuilt from scratch.

### Changed

- `inference.py` `generate`: after detokenize, if `piece == "<|im_end|>"`, `break` **before** appending or `decode_one`. ChatML end-of-turn is not treated as EOG by `llama_bridge_is_eog` on this vocab.
- `extract_json`: `json.JSONDecoder().raw_decode` from the first `{`. Parses **one** object. Trailing `<|im_end|>` or a second JSON is ignored.
- `MAX_NEW_TOKENS = 256 * 4`, `CONTEXT_SIZE = 2048 * 2` (already on this line of development).

### Fixed

- Empty `Generated text: ''` from unofficial `USER:` / `ASSISTANT:` prompts (model sampled EOG immediately).
- Two actions in one generate (`parse_text_...` JSON, then `<|im_end|>`, then a second assistant `answer` JSON). OCR never ran; `json.loads` raised `Extra data`; parse-error `continue` looped until Ctrl+C.

### Docs

- `docs/14-qwen-chatml-and-one-json-per-turn.md`
- Chapter 10 note: `rfind("}")` is historical; live parser is `raw_decode`.

---

## [1.0.3] — 2026-09-15

Runtime helpers extracted. Step cap. Parse/answer recovery. Rebuild conversation after each tool (still unofficial chat labels).

### Added

- `utils/agent_utils.py`:
  - `usable_tool_arguments` — drop `{type, description}` schema echoes.
  - `execute_tool` — unknown tool → `{success: false}`; wrap skill exceptions; if `get_user_input` `prompt` is not a `str`, default to `"Enter the image file path: "`; pass through skill dicts that already have `success`.
- `MAX_AGENT_STEPS = 10`.
- `try/except` around `extract_json`; on failure, append “Return exactly one valid JSON object.”
- `action == "answer"` requires `content` to be a `str`; otherwise append a correction and continue.
- Step banner: `========== AGENT STEP n ==========`.
- After a tool, **replace** `conversation` with `SYSTEM_PROMPT` + original user task + `TOOL RESULT` JSON + recovery rules (including WRONG/RIGHT ASSUMPTION about filenames like `test.png`).

### Changed

- `get_user_input_skill.py` returns `{success: True, result: value.strip()}` instead of a bare string.
- `SYSTEM_PROMPT`: one JSON per turn; must use `get_user_input` for human data; never `answer` to ask for a path; never invent paths.

### Removed (from the agent file)

- Inlined `_usable_tool_arguments` / inlined try/except around `tool_function` (moved to `execute_tool`).

---

## [1.0.2] — 2026-09-14

Structured tool results as JSON. Recovery text at the failure site. Empty `agent_state` reserved.

### Added

- `agent_state = {}` at the start of `run_agent` (**not written or read yet**).
- Tool payload field `"tool": tool_name` on success and failure.
- Follow-up prompt: `json.dumps(tool_result, indent=2)`; “If the tool failed because information is missing, use the appropriate tool…”; “Never invent file paths, URIs, user input, or tool results.”

### Changed

- `except Exception as e` → `as exc`.
- `decision.get("arguments", {})` formatting only.

### Docs

- `docs/13-agent-runtime-guarantees.md` (`v1` → `v1_0_1` → `v1_0_2`).

---

## [1.0.1] — 2026-09-14

Tool failures must not crash the process.

### Changed

- Bare `tool_result = tool_function(**tool_arguments)` replaced with:

```python
try:
    result = tool_function(**tool_arguments)
    tool_result = {"success": True, "result": result}
except Exception as e:
    tool_result = {"success": False, "error": str(e)}
```

- OCR URL `ValueError` (`http://` / `https://`) becomes the next LLM turn instead of a traceback after `Model unloaded.`

### Unchanged

- Unofficial `USER:` / `ASSISTANT:` conversation string.
- `{tool_result}` interpolated with Python `repr`, not `json.dumps`.
- Lenient `action` as tool name.

---

## [1.0.0] — 2026-09-14

Numbered agent: Swedish OCR task, allowlist tools, lenient JSON actions.

### Added

- `agent_v1.py`: `run_agent` + `while True`, `_usable_tool_arguments`, empty-response guard, `action == "answer"` return, `tool_name = name if action == "tool" else action`.
- Skills: `get_user_input`, `parse_text_from_image_of_swedish_document` (Tesseract `lang="swe"`, reject `http(s)://`).
- `TOOLS` / `TOOLS_DESCRIPTIONS` with provenance text on `document_image_uri`.
- Demo user message: ask for image URI, then parse text.

### Behavior (limitations)

- Skill exceptions abort `run_agent`.
- Model often skipped `get_user_input` and invented a URL; skill rejected it, then the **process** died.
- Conversation was a growing f-string, not ChatML.

---

## [0.3.0] — 2026-09-09 … 2026-09-14

First tool-using loop and tokenizer/agent prompt work (pre-`agent_v1` filename).

### Added

- `simple_agent.py` / first `run_agent`: `SYSTEM_PROMPT` + JSON `action` `tool` | `answer`, `TOOLS["simple_test"]`.
- `extract_json`: first `{` to last `}` (skip `<think>`).
- Two-pass tokenize (`None, 0` probe, then exact buffer).
- Context fit check: `token_count + MAX_NEW_TOKENS` vs `CONTEXT_SIZE`.
- `MAX_NEW_TOKENS` raised 64 → 256 (then later ×4) after truncated tool JSON.

### Fixed

- `Failed to tokenize prompt` when the agent system prompt exceeded `MAX_TOKENS = 128`.
- `JSONDecodeError` on leading `<think>`.

---

## [0.2.0] — 2026-09-09

`localLLM` wrapper: load once, `generate(prompt)` per call, new native context each generate.

### Added

- `engine/llama/inference.py` `localLLM`: `CDLL` llama.cpp + bridge, signatures, `llama_bridge_init`, GGUF load, tokenize, `create_context`, `decode_prompt`, greedy sample loop, `token_to_piece`, `decode_one`, `free_context`, `close`.
- `utils/llama_utils.py`: Homebrew `libllama.dylib`, ctypes `argtypes`/`restype`.
- `engine/llama/bridge.cpp` + `build_bridge.sh` → `libllama_bridge.dylib`.

---

## [0.1.0] — 2026-09-09

Pointer / KV-cache inference (no agent).

### Added

- Native `llama_decode` mutates `ctx`; `decode_result` is status only.
- Empty context at `create_context`; prompt decode fills KV; `decode_one` appends one position.

See `docs/06-kv-cache-and-mutable-context.md` and `docs/local_llm_to_agent_textbook.pdf`.

---

## Version comparison (agents)

| | 1.0.0 | 1.0.1 | 1.0.2 | 1.0.3 | 2.0.0 |
|---|---|---|---|---|---|
| Prompt format | `USER:` / `ASSISTANT:` | same | same | same | ChatML `messages` |
| Tool error | crash | `{success: false}` repr | JSON + `"tool"` | `execute_tool` | `execute_tool` |
| History after tool | append | append | append | **rebuild** string | **append** messages |
| Step cap | none | none | none | 10 | 10 |
| Parse recovery | no | no | no | yes (string) | yes (messages) |
| Stop at `<\|im_end\|>` | n/a | n/a | n/a | n/a | yes |
| First JSON only | `rfind` | `rfind` | `rfind` | `rfind` | `raw_decode` |
| `agent_state` | no | no | empty dict | no | no |
