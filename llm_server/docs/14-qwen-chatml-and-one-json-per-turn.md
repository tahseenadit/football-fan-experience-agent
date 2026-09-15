# Chapter 14 — Qwen ChatML, One JSON Per Turn, and Chat Stop Tokens

Sources:

- `llm_server/src/ai/utils/agent_utils.py` (`apply_chat_template`)
- `llm_server/src/ai/agents/agent_v2.py`
- `llm_server/src/ai/engine/llama/inference.py` (`piece == "<|im_end|>"`)
- `llm_server/src/ai/utils/llama_utils.py` (`extract_json` / `raw_decode`)
- `llm_server/src/ai/config/model_config/llama.py` (`ENABLE_THINKING`)
- `llm_server/src/ai/utils/prompts/llama.py` (`SYSTEM_PROMPT`)

This chapter records the move from unofficial `USER:` / `ASSISTANT:` strings to Qwen ChatML, why that produced two JSON actions in one generate, and the two runtime fixes that make the parse skill run again.

---

## 1. Why ChatML exists

`agent_v1_0_3.py` (and earlier) built a Python string:

```text
{SYSTEM_PROMPT}

USER:
Ask the user for the local image file path. ...

ASSISTANT:
```

Qwen3.5 was not trained on those English labels. It was trained on ChatML special tokens. With the fake prompt, the first sampled token was often end-of-generation, so `Generated text: ''`.

The intended stack:

```text
[{"role": "system", ...}, {"role": "user", ...}]
        │
        ▼  apply_chat_template
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<think>

</think>


        │
        ▼  tokenize + generate
        ▼  one assistant JSON
```

The empty `<think>` block is Qwen3.5’s **thinking off** switch (`enable_thinking=False`). The model should emit JSON after it, not a long reasoning span.

Python ChatML lives in `agent_utils.apply_chat_template`. This build does **not** call llama.cpp `llama_chat_apply_template` (that C API is a non-Jinja matcher). For this Qwen agent, the Python formatter is the source of truth.

---

## 2. `ENABLE_THINKING` (`model_config/llama.py`)

```python
MAX_TOKENS = 128
MAX_NEW_TOKENS = 256 * 4
CONTEXT_SIZE = 2048 * 2
ENABLE_THINKING = False
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `ENABLE_THINKING = False` | Bind a bool | Empty think block in the template | Intended: `apply_chat_template(..., enable_thinking=ENABLE_THINKING)` | Import | N/A | Tiny |

**Status:** `agent_v2.py` currently passes `enable_thinking=False` **literally**, and does not import this name. The flag is the documented contract; wiring it into the call is a one-argument change later.

If `True`, the template ends with `<|im_start|>assistant\n<think>\n` and the model writes reasoning before JSON. That spends `MAX_NEW_TOKENS` on think tokens (Chapter 10).

---

## 3. `apply_chat_template` line by line (`agent_utils.py`)

```python
def apply_chat_template(
    messages: list,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False
) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(
            f"<|im_start|>{role}\n{content}<|im_end|>\n"
        )

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
        if enable_thinking:
            parts.append("<think>\n")
        else:
            parts.append("<think>\n\n</think>\n\n")

    return "".join(parts)
```

| Line | What it does | Purpose | Who needs it | Called by | When | Stop | Memory / CPU |
|---|---|---|---|---|---|---|---|
| `def apply_chat_template(...)` | Format role/content dicts as ChatML | Model-specific prompt | `llm.generate` | `agent_v2.run_agent` each step | Each agent step | Loop end / return | O(chars of messages) |
| `parts = []` | Accumulator | Join once | Return | Start of call | — | Tiny |
| `for message in messages:` | Walk history | Closed turns first | Template | Each call | Empty list → only generation prompt | Tiny |
| `role = message["role"]` | `system` / `user` / `assistant` | Token after `<\|im_start\|>` | f-string | Each message | KeyError if missing | Tiny |
| `content = message["content"]` | Turn text | Body of the turn | f-string | Each message | KeyError if missing | Tiny |
| `parts.append(f"<\|im_start\|>{role}\n{content}<\|im_end\|>\n")` | **Closed** ChatML turn | History the model already “said” or was told | Join | Each message | — | String alloc |
| `if add_generation_prompt:` | Open a live assistant turn | Cue generation | Qwen | `agent_v2` always `True` | `False` would mean “prompt is complete” | Tiny |
| `parts.append("<\|im_start\|>assistant\n")` | Start assistant | Required ChatML header | Model | When add_generation_prompt | — | Tiny |
| `if enable_thinking: "<think>\n"` | Open reasoning | Model fills think then JSON | Generate | Flag True | — | Tiny |
| `else: "<think>\n\n</think>\n\n"` | Empty think | Thinking **off** | Generate | Flag False (v2) | — | Tiny |
| `return "".join(parts)` | One prompt string | `localLLM.generate` | Agent | End of call | — | One `str` |

**Who does not call this:** llama.cpp. It is pure Python.

**What user action stops it:** not running the agent; exception before `apply_chat_template` in the loop.

Do **not** put `ASSISTANT:` or `USER:` in `content`. Those words are the old protocol. ChatML roles are the `role` field.

Closed turns always include `<|im_end|>`. The **live** assistant turn does **not**: generation is supposed to produce JSON, then the runtime stops at `<|im_end|>` (next section).

---

## 4. Example prompt after implementation (turn 1)

Python state:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Ask the user for the local image file path. Then parse the text from the image and return the extracted text."},
]
prompt = apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
```

Model input (structure):

```text
<|im_start|>system
You are an agent.
...
Do not output anything outside JSON.
<|im_end|>
<|im_start|>user
Ask the user for the local image file path. Then parse the text from the image and return the extracted text.
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

The JSON tool call is **not** in the template. It is sampled after this string.

---

## 5. Example after a tool result (turn 2+)

`agent_v2` does **not** rebuild `SYSTEM_PROMPT` + `USER:` from scratch (that was `v1_0_3` / early `v2`). It **appends**:

```python
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "You have just taken the following action:\nTOOL RESULT:\n" + json.dumps(...) + "\n..."})
```

Then `apply_chat_template` again. The second prompt contains the original system and user turns, then:

```text
<|im_start|>assistant
{"action": "tool", "name": "get_user_input", ...}
<|im_end|>
<|im_start|>user
You have just taken the following action:
TOOL RESULT:
{ ... }
...
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

---

## 6. `agent_v2.py` — how the loop uses the template

| Region | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| Import `apply_chat_template` | Bind formatter | ChatML | Loop | Import | — | Tiny |
| `messages = [system, user]` | Runtime history | Template input | `apply_chat_template` | Start of `run_agent` | Function return | Grows each step |
| `for step in range(MAX_AGENT_STEPS)` | Cap 10 turns | Stop infinite loops | Process | Until return | After 10 → string return | Control |
| `prompt = apply_chat_template(..., True, False)` | ChatML string | `generate` | Each step | — | O(history) |
| `response = llm.generate(prompt)` | Sample tokens | Decision text | `extract_json` | Each step | Ctrl+C; EOG; `<\|im_end\|>` | **Dominant** |
| Empty `response` | Return with last `prompt` | Avoid parse on `''` | `main` | First token stop with no text | Tiny |
| `extract_json` success | Dict | Tool / answer | After generate | `raw_decode` fail → except | Tiny |
| Parse `except` | Append assistant + user error | Recovery **used to** loop forever when two JSONs were sliced as one | Next step | After Extra data | History growth |
| `action == "answer"` + `str` content | `return content` | Halt | `main` | Valid answer | Tiny |
| Bad answer content | Append correction | Another generate | Loop | — | Tiny |
| `tool_name` from `name` or `action` | Lenient protocol | `execute_tool` | Non-answer | Unknown tool → structured error | Tiny |
| `execute_tool` | try/except skill | `{success, tool, result\|error}` | Follow-up user turn | Skill / `input()` | Skill |
| Append assistant `response` + user `TOOL RESULT` | History | Next ChatML | After tool | — | String |

**Who calls `apply_chat_template`:** only `run_agent`, once per step. **Who calls `generate`:** same. **User stop:** Ctrl+C; 10 steps; `answer` return.

`MAX_AGENT_STEPS = 10` is why an Extra-data loop did not run forever in theory; in the log the user interrupted with Ctrl+C during a later generate (`<|im_start|^C`).

---

## 7. Failure: two JSON objects in one generate

Observed `Generated text` (step 2-ish):

```text
{
  "action": "parse_text_from_image_of_swedish_document",
  "arguments": {
    "document_image_uri": "test.png"
  }
}<|im_end|>
<|im_start|>assistant
<think>

</think>

{
  "action": "answer",
  "content": "The text extracted from the image is: 'This is a test file. ...'"
}<|im_end|>
```

### What that means

1. ChatML **worked**. The model spoke ChatML (`<|im_end|>`, `<|im_start|>assistant`).
2. The **first** JSON was the correct next action: run OCR on `test.png`.
3. Generation **did not stop** at the first `<|im_end|>`. `llama_bridge_is_eog` returned false for that token, so the loop detokenized `<|im_end|>` as **text** and kept sampling. The model started a **second** assistant turn and invented an answer without Python running OCR.
4. Old `extract_json` did `find("{")` + `rfind("}")`. That slice contained **two** objects. `json.loads` raised `Extra data: line 6 column 2 (char 118)`.
5. `agent_v2` treated that as a parse failure, appended “Return exactly one valid JSON object.”, and `continue`d. The model repeated two-object output. That is the infinite loop.

`agent_v1_0_3.py` looked “fine” because it never used ChatML, so it never emitted `<|im_end|>` as a continuation cue.

```text
generate kept going past <|im_end|>
        │
        ▼
two JSON objects in `response`
        │
        ▼
rfind("}") → Extra data
        │
        ▼
parse except → continue
        │
        └──► generate again (history polluted)
```

The parse skill never ran: execution happens only **after** a successful `extract_json`.

---

## 8. Fix 1: stop generate on `<|im_end|>` (`inference.py`)

```python
            piece = piece_buffer.raw[:piece_length].decode(
                "utf-8",
                errors="replace",
            )

            if piece == "<|im_end|>":
                break

            generated_text += piece
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `if piece == "<\|im_end\|>":` | ChatML end-of-turn as **text** | `is_eog` missed this id | Rest of generate | Each token after detokenize | Piece is something else | Tiny |
| `break` | Leave the `for MAX_NEW_TOKENS` loop | Do **not** append `<\|im_end\|>`, do **not** `decode_one` it, do **not** start a second turn | `return generated_text` | First `<\|im_end\|>` | — | Skips remaining tokens |

**Who calls it:** `localLLM.generate`, used by `agent_v2` (and any other caller).

**What it does not stop:** a second JSON **before** `<|im_end|>` in the same turn (no end token yet). That is why fix 2 still exists.

**User action that stops this line:** process kill; generate not entered.

**CPU:** one string compare per token, nothing vs `decode_one`.

If `<|im_end|>` were split across two pieces (unusual for a special token), this equality would miss. Then fix 2 still saves the agent.

---

## 9. Fix 2: parse only the first JSON (`extract_json`)

Old:

```python
start = response.find("{")
end = response.rfind("}")
json_text = response[start:end + 1]
return json.loads(json_text)
```

New:

```python
    start = response.find("{")

    if start == -1:
        raise RuntimeError(
            f"No JSON object found in model response:\n{response}"
        )

    obj, _ = json.JSONDecoder().raw_decode(response[start:])
    return obj
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `find("{")` | Skip `<think>` / ChatML prefix | First object | `raw_decode` | Each parse | No `{` → RuntimeError | O(n) |
| `JSONDecoder().raw_decode(response[start:])` | Parse **one** JSON value; ignore trailing text | Two objects / `<\|im_end\|>` after the first `}` | `run_agent` | After generate | Invalid first object → `JSONDecodeError` | Tiny |
| `return obj` | Dict (or list) | `action` / `name` | Agent | Success | — | Tiny |

`raw_decode` returns `(object, index_after_object)`. The index is discarded (`_`). Extra JSON after the first object is **ignored**, not an error.

If generate still overshoots, the agent executes the **first** action (`parse_text_from_image_of_swedish_document` with `test.png`), which is what was skipped before.

Chapter 10 still describes the old `rfind` parser as historical. This chapter is the current contract.

---

## 10. Both fixes together

| Layer | Guarantee |
|---|---|
| ChatML template | Qwen sees trained special tokens + empty think |
| `piece == "<\|im_end\|>"` | One assistant turn per `generate` |
| `raw_decode` | One JSON decision even if generate overshoots |
| `execute_tool` try/except | Skill errors become `success: false`, not a crash (Chapter 13) |
| `MAX_AGENT_STEPS` | Loop cannot run forever |

Order of importance for the log in this chapter: **im_end stop** prevents the second JSON; **raw_decode** prevents Extra data if a second JSON still appears.

---

## 11. Line-level generate stop conditions (after the fix)

Inside `for _ in range(MAX_NEW_TOKENS)`:

| Check | When | Effect |
|---|---|---|
| `llama_bridge_is_eog` | Vocab EOG id | `break` before detokenize |
| `piece == "<\|im_end\|>"` | ChatML end rendered as text | `break` before append / decode_one |
| Loop exhausts | 1024 new tokens (`256 * 4`) | Return whatever was accumulated |
| Ctrl+C | User | `KeyboardInterrupt`; `finally: llm.close()` |

---

## 12. What is not in this build

- C++ `llama_bridge_apply_chat_template` (Python formatter only)
- Tokenize `add_special=False` for ChatML (still default `true, true` in `bridge.cpp`). Extra BOS on top of `<|im_start|>` may still exist; if generation looks odd, that is the next tokenize flag change
- Importing `ENABLE_THINKING` into `agent_v2` (hardcoded `False`)
- Injecting `agent_state` paths into OCR (Chapter 13)

---

## 13. Suggested run check

```text
python .../agents/agent_v2.py
```

Expect:

1. Prompt / tokenize of a string that starts with `<|im_start|>system`
2. One JSON per step in `LLM OUTPUT:` (no second `<|im_start|>assistant` inside `Generated text`)
3. `TOOL RESULT` for `parse_text_from_image_of_swedish_document` when that JSON is first
4. No repeating `Extra data` / `Return exactly one valid JSON object` cycle
