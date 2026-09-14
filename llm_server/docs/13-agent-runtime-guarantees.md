# Chapter 13 — Agent Runtime Guarantees

Sources:

- `llm_server/src/ai/agents/agent_v1.py`
- `llm_server/src/ai/agents/agent_v1_0_1.py`
- `llm_server/src/ai/agents/agent_v1_0_2.py`
- `llm_server/src/ai/skills/swedish_language_skills/skills.py`
- `llm_server/src/ai/skills/tools/tools.py`
- `llm_server/src/ai/skills/get_user_input_skill.py`
- `llm_server/src/ai/utils/prompts/llama.py`

This chapter records a useful failure, the architectural rule it exposes, the runtime changes from `agent_v1.py` → `agent_v1_0_1.py` → `agent_v1_0_2.py`, and a line-by-line reading of all three agent files. Nothing from that lineage is summarized away.

---

## 0. The rule

> **The LLM should decide what capability it needs, but your runtime should enforce whether that action is allowed.**

The 0.8B model understood the overall task:

```text
Need image
→ need OCR
→ call parse_text_from_image_of_swedish_document
```

but it skipped:

```text
get_user_input
```

and hallucinated:

```json
"document_image_uri": "https://example.com/swedish-document.png"
```

The Python skill correctly rejected it. So the skill did its job; the **agent runtime needs to become stronger**.

---

## 1. What happened

You asked:

```text
Ask the user for the image URI input.
Then parse the text in the image.
```

In `agent_v1.py` / `agent_v1_0_1.py` / `agent_v1_0_2.py` `main()` the actual string is:

```python
"Ask the user for the image URI input. Then parse the text in the image and return the text."
```

Ideally:

```text
LLM
 ↓
get_user_input
 ↓
human types path
 ↓
LLM
 ↓
parse_text_from_image...
 ↓
OCR
 ↓
answer
```

But the 0.8B model did:

```text
LLM
 ↓
"I need an image URI"
 ↓
invent URI
 ↓
parse_text_from_image...
```

This is typical small-model behavior. Prompt instructions alone are not enough.

**Do not** solve this by making `SYSTEM_PROMPT` longer and longer.

Instead, start adding real runtime guarantees.

---

## 2. First fix: tool failures must not crash the agent

In `agent_v1.py` the execution line is:

```python
tool_result = tool_function(**tool_arguments)
```

Any `ValueError` from the skill kills the entire agent. `main()`’s `finally: llm.close()` still runs, but `run_agent` never gets a second turn. That is undesirable.

`agent_v1_0_1.py` changes it to:

```python
try:
    result = tool_function(**tool_arguments)

    tool_result = {
        "success": True,
        "result": result,
    }
except Exception as e:
    tool_result = {
        "success": False,
        "error": str(e),
    }
```

`agent_v1_0_2.py` keeps that pattern and adds `"tool": tool_name` on both branches (see §8).

Then that object is fed back to the LLM.

So the current failure becomes a **turn**, not a crash:

```json
{
  "success": false,
  "error": "document_image_uri must be a local file path from get_user_input, not an internet URL."
}
```

Then Qwen gets another chance:

```text
Oh. That path wasn't valid.

I need get_user_input.
```

and hopefully produces:

```json
{
  "action": "tool",
  "name": "get_user_input",
  "arguments": {
    "prompt": "Enter the image file path:"
  }
}
```

This alone makes the agent significantly more robust.

---

## 3. The deeper problem: the LLM as a copy machine

Suppose the model does call:

```python
get_user_input()
```

and the human enters:

```text
/Users/farzananitol/Documents/swedish_page.png
```

The conversation becomes something like:

```text
TOOL RESULT:
/Users/farzananitol/Documents/swedish_page.png
```

Then Qwen has to produce:

```json
{
  "action": "tool",
  "name": "parse_text_from_image_of_swedish_document",
  "arguments": {
    "document_image_uri": "/Users/farzananitol/Documents/swedish_page.png"
  }
}
```

That means you are relying on the LLM to:

1. understand the tool result,
2. remember the exact path,
3. copy it correctly,
4. not invent another path.

That will work sometimes, but we are building an **agent runtime**, so we can do better.

---

## 4. Introduce agent state

Until `agent_v1_0_1.py`, state is effectively just:

```python
conversation
```

`agent_v1_0_2.py` introduces:

```python
agent_state = {}
```

Eventually:

```python
agent_state = {
    "user_inputs": {},
    "tool_results": {},
}
```

For example after asking for an image:

```python
agent_state = {
    "image_path": "/Users/farzananitol/Documents/swedish_page.png"
}
```

Now **Python knows the path**, not just the LLM.

This is an important distinction:

```text
LLM context
=
what the model has been told


Agent state
=
what the runtime actually knows
```

Production agents normally need both.

**Status in `agent_v1_0_2.py`:** `agent_state = {}` is created at the start of `run_agent` and is **not yet written or read**. It is the foundation, not the finished store. After `get_user_input` succeeds, the runtime should later do something like `agent_state["image_path"] = result`. After OCR, `agent_state["ocr_text"] = result`. The LLM would still see tool results as text; Python would own the canonical values.

---

## 5. Tools should have preconditions

The OCR tool has a real requirement:

```text
parse_text_from_image_of_swedish_document

requires:
    valid local path
```

That requirement should be enforced by Python.

`skills/swedish_language_skills/skills.py` already does this in part:

```python
if document_image_uri.startswith(
    ("http://", "https://")
):
    raise ValueError(...)
```

Good. That is what caught `https://example.com/swedish-document.png`.

The next validation (not yet in the skill file) is existence and file-ness:

```python
from pathlib import Path


def parse_text_from_image_of_swedish_document(
    document_image_uri: str,
) -> str:

    if document_image_uri.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            "document_image_uri must be a local file path."
        )

    image_path = Path(
        document_image_uri
    ).expanduser()

    if not image_path.exists():
        raise ValueError(
            f"Image does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Not a file: {image_path}"
        )

    image = Image.open(image_path)

    text = ocr_image(image)

    return clean_text(text)
```

Now even if Qwen invents:

```text
/tmp/swedish.png
```

Python catches it. Combined with `agent_v1_0_1.py` / `agent_v1_0_2.py` try/except, that `ValueError` becomes `success: false` JSON instead of a process crash.

**Status in the skill today:** URL prefix check is present. `Path.exists()` / `is_file()` / `expanduser()` are **not** in the file yet. `Image.open(document_image_uri)` will still raise if the path is missing; `v1_0_1`/`v1_0_2` will catch that `FileNotFoundError` as `success: false`. Explicit `exists` checks give a clearer error string.

Current skill body, line by line:

| Line | Code | What it does | Purpose | Who needs it | When | What stops it | Memory / CPU |
|---|---|---|---|---|---|---|---|
| 26 | `def parse_text_from_image_of_swedish_document(document_image_uri: str) -> str:` | Define the OCR skill | One capability: image path → Swedish text | `TOOLS` registry; `run_agent` via `tool_function(**kwargs)` | When the model names this tool | Unknown name → not called | 0 until called |
| 27–29 | docstring | Human description | Readers | Import | — | Tiny |
| 30 | `if document_image_uri.startswith(("http://", "https://")):` | Reject invented URLs | Runtime guarantee the model cannot bypass | The `raise` | Every call | Non-http paths skip this | Tiny |
| 31–34 | `raise ValueError(...)` | Fail the skill | `v1` crashes the agent; `v1_0_1`/`v1_0_2` turn it into `success: false` | Agent loop except | On URL | — |
| 36 | `text_from_image = ""` | Init | Later overwritten | OCR/clean | Every call | — | Empty str |
| 38 | `ocr_image(Image.open(document_image_uri))` | Open file + Tesseract `lang="swe"` | The actual capability | Return value | After URL check | Missing file / bad image → exception | Disk + OCR CPU |
| 40 | `clean_text(...)` | Drop blank lines | Usable OCR dump | Return | After OCR | — | Tiny |
| 42 | `return text_from_image` | Hand text to the agent | Next LLM turn | After clean | — | The string |

`ocr_image` (lines 10–14) is `pytesseract.image_to_string(image, lang="swe")`. `clean_text` (lines 16–24) joins stripped non-empty lines.

---

## 6. Do not let the model invent arguments that require external information

The tool metadata in `tools.py` currently includes provenance language:

```python
"document_image_uri": {
    "type": "string",
    "description": (
        "A real local file path that must already have been "
        "provided by the human through get_user_input. "
        "Never invent this value."
    ),
}
```

That tells the LLM not to invent a string. Helpful, but **prompting is secondary to runtime enforcement**.

`get_user_input` is described as: ask the human; never guess. Arguments: optional `prompt` string.

`TOOLS` maps:

| Name | Function |
|---|---|
| `simple_test` | `simple_test` |
| `get_user_input` | `get_user_input` |
| `parse_text_from_image_of_swedish_document` | `parse_text_from_image_of_swedish_document` |

`get_all_tools_description()` is `json.dumps(TOOLS_DESCRIPTIONS, indent=2, default=str)`, interpolated into `SYSTEM_PROMPT`.

`get_user_input` itself:

| Line | Code | What it does |
|---|---|---|
| 1 | `def get_user_input(prompt: str = "Enter your input: ") -> str:` | Terminal read; default prompt if the model omits `prompt` or copies a schema dict that `_usable_tool_arguments` strips |
| 5 | `return input(prompt)` | Blocks the agent loop until the human types Enter |

---

## 7. Action-compatibility code: useful now, tighten later

All three agent files support:

```json
{
  "action": "tool",
  "name": "get_user_input"
}
```

and:

```json
{
  "action": "get_user_input"
}
```

through:

```python
tool_name = (
    decision.get("name")
    if action == "tool"
    else action
)
```

That is smart for getting a tiny model running.

But notice what happened:

```json
{
  "action": "parse_text_from_image_of_swedish_document"
}
```

The model ignored the protocol and **the runtime accepted it anyway**.

For experimentation that is okay.

Eventually:

```text
Model output
      ↓
strict parser
      ↓
schema validation
      ↓
only then execute
```

Not:

```text
Model output
      ↓
"close enough"
      ↓
execute something
```

`_usable_tool_arguments` is the same kind of “close enough” helper: if `prompt` is `{"type": "string", "description": "..."}` copied from `TOOLS_DESCRIPTIONS`, that dict is dropped so `get_user_input` uses its default string. Kept in all three versions.

---

## 8. The loop change in `agent_v1_0_2.py`

Keep the architecture almost the same, but change the execution section. `agent_v1_0_2.py` does:

```python
if tool_function:

    tool_arguments = _usable_tool_arguments(
        decision.get(
            "arguments",
            {},
        ),
    )

    try:

        result = tool_function(
            **tool_arguments
        )

        tool_result = {
            "success": True,
            "tool": tool_name,
            "result": result,
        }

    except Exception as exc:

        tool_result = {
            "success": False,
            "tool": tool_name,
            "error": str(exc),
        }

    conversation += f"""

        Previous LLM response:
        {response}

        TOOL RESULT:
        {json.dumps(tool_result, indent=2)}

        Use the tool result above.

        If the tool failed because information is missing,
        use the appropriate tool to obtain the missing information.

        Never invent file paths, URIs, user input, or tool results.

        If you now have enough information,
        answer the user.

        ASSISTANT:
        """

    continue
```

Now the exact failed run should behave more like:

```text
Qwen:
parse_text...(example.com)

        ↓

Python:
reject URL

        ↓

TOOL RESULT:
{
  "success": false,
  "tool": "parse_text_from_image_of_swedish_document",
  "error": "must be a local file path..."
}

        ↓

Qwen:
get_user_input

        ↓

terminal:
Enter image file path:

        ↓

human:
/Users/.../page.png

        ↓

Qwen:
parse_text_from_image...

        ↓

Python OCR

        ↓

Swedish text
```

That is already much more agent-like.

`json.dumps(tool_result, indent=2)` matters: `v1_0_1` interpolates the dict with `{tool_result}`, which is Python `repr` (`{'success': False, 'error': '...'}`). `v1_0_2` feeds **JSON**, which matches the model’s JSON-only contract.

---

## 9. Next architectural improvement: a validation layer

Current architecture (all three files):

```text
LLM
 ↓
tool name + arbitrary arguments
 ↓
Python executes
```

Evolve toward:

```text
                        AGENT RUNTIME

LLM
 │
 │ proposes intent
 ▼
┌───────────────────────┐
│ Tool validation       │
│                       │
│ Does tool exist?      │
│ Are args valid?       │
│ Are prerequisites met?│
│ Are paths real?       │
└───────────┬───────────┘
            │
            ▼
       execute tool
            │
            ▼
       structured result
            │
            ▼
       update state
            │
            ▼
           LLM
```

That middle layer becomes important once skills include:

```text
read document
write file
run SQL
open URL
deploy resource
create BigQuery dataset
delete resource
```

You do not want a 0.8B model to have unrestricted authority simply because it emitted valid JSON.

And one small naming change: change

```python
parse_text_from_image_of_swedish_document
```

to something more composable like:

```python
ocr_swedish_document_image
```

because skills should generally represent **one clear capability**. Then later:

```text
ocr_swedish_document_image
extract_swedish_document_fields
summarize_swedish_document
translate_swedish_document
classify_swedish_document
```

**Status:** the registry still uses `parse_text_from_image_of_swedish_document`. Rename is a later, coordinated change (`TOOLS`, `TOOLS_DESCRIPTIONS`, skill function, prompt examples).

This is a good point in the project to build the **agent runtime foundation first**, before adding dozens of skills.

---

## 10. Version map

| File | Role |
|---|---|
| `agent_v1.py` | Baseline loop: conversation string, lenient action, **bare tool call**, crash on skill error |
| `agent_v1_0_1.py` | Same loop + **try/except** + `{success, result\|error}` |
| `agent_v1_0_2.py` | `0.1` + empty **`agent_state`**, `"tool"` in the result, **`json.dumps`**, recovery instructions in the follow-up prompt |

Unchanged across all three (except whitespace around the tool-arguments call in `0.2`):

- `sys.path` bootstrap
- `_usable_tool_arguments`
- empty-response early return
- `action == "answer"` → `return decision["content"]`
- `tool_name = decision.get("name") if action == "tool" else action`
- `raise ValueError(f"Invalid action: {action}")` for unknown actions
- `main()` user message
- `try/finally: llm.close()`

`v1_0_2` still uses a **conversation string**, not the ChatML `messages` list from an earlier experiment. `print(conversation)` then `llm.generate(conversation)` is the generate path.

---

## 11. Shared helper: `_usable_tool_arguments` (all three files, lines 18–27)

```python
def _usable_tool_arguments(arguments) -> dict:
    """Drop schema fragments the model copied from TOOLS_DESCRIPTIONS."""
    if not isinstance(arguments, dict):
        return {}
    usable = {}
    for key, value in arguments.items():
        if isinstance(value, dict) and "type" in value and "description" in value:
            continue
        usable[key] = value
    return usable
```

| Line | What it does | Purpose | Who needs it | Called by | When | Stop | Cost |
|---|---|---|---|---|---|---|---|
| 18 | Declare helper, `dict` return | Sanitize model kwargs | `tool_function(**...)` | `run_agent` | Each tool turn | — | Tiny |
| 19 | Docstring | Why schema dicts are dropped | Readers | — | — | — |
| 20–21 | Non-dict → `{}` | Model may omit or emit a list | Call site | Each tool turn | — | Tiny |
| 22 | Empty `usable` | Accumulator | Loop | Each call | — | Tiny |
| 23 | Iterate kwargs | Inspect each value | Filter | Each call | — | Tiny |
| 24–25 | Skip `{type, description}` dicts | Model echoed `TOOLS_DESCRIPTIONS` | `get_user_input(prompt=str)` | When schema copied | — | Tiny |
| 26 | Keep real values | Path strings, prompt strings | Skill | — | — | Tiny |
| 27 | Return filtered dict | `**tool_arguments` | `tool_function` | After loop | — | Tiny |

**Who calls it:** `run_agent` after resolving `tool_function`. **What stops it:** answer branch; unknown action; empty LLM response. **Memory:** a small dict. **CPU:** negligible vs `generate`.

---

## 12. `agent_v1.py` line by line

```1:115:llm_server/src/ai/agents/agent_v1.py
import subprocess
import sys
import json
from pathlib import Path
# ... through if __name__ == "__main__": main()
```

| Line | Code | What it does | Purpose | Who needs it | When | Stop | Memory / CPU |
|---|---|---|---|---|---|---|---|
| 1 | `import subprocess` | Bind subprocess | **Unused** in this file | — | Import | — | Tiny |
| 2 | `import sys` | Bind sys | `sys.path` | Bootstrap | Import | — | Tiny |
| 3 | `import json` | Bind json | **Unused** in v1 (used in v1_0_2) | — | Import | — | Tiny |
| 4 | `from pathlib import Path` | Bind Path | `_AI_DIR` | Bootstrap | Import | — | Tiny |
| 6 | comment | Dual invocation | Humans | — | — | — |
| 7 | `_AI_DIR = Path(__file__).resolve().parents[1]` | `agents/` → `ai/` | Imports `config`, `skills`, `engine`, `utils` | Insert | Import | Wrong parent → ImportError | Tiny |
| 8–9 | `if ... not in sys.path: insert(0, ...)` | Prepend `ai/` | `python agents/agent_v1.py` | Later imports | Import | Already on path → skip | Tiny |
| 11 | `from config.config import MODEL_PATH` | Import config (runs `brew`) | **MODEL_PATH unused**; side effect loads paths | Import | brew fail | brew |
| 12 | `from skills.tools.tools import TOOLS` | Registry | Lookup by name | Tool branch | Import | — | Tiny |
| 13 | `from engine.llama.inference import localLLM` | Inference class | `main` constructs it | Import | Missing dylib later | Import cost |
| 14 | `from utils.prompts.llama import SYSTEM_PROMPT` | System text + tool JSON | First `conversation` | `run_agent` | Import | — | Tiny |
| 15 | `from utils.llama_utils import extract_json` | `{...}` slice + `json.loads` | Each turn | After generate | Import | — | Tiny |
| 18–27 | `_usable_tool_arguments` | §11 | Tool kwargs | Each tool turn | — | Tiny |
| 30–32 | `def run_agent(llm, user_message) -> str:` | Agent loop | `main` | Once per process | Constructor fail | — |
| 34–41 | `conversation = f"""..."""` | SYSTEM + USER + `ASSISTANT:` | Prompt the unofficial chat format | `llm.generate` | Start of `run_agent` | Never if `run_agent` not called | Few KB |
| 43 | `while True:` | Infinite turns | Tool then generate again | Heart of the agent | Until return/raise/Ctrl+C | Control |
| 47 | `print(conversation)` | Dump prompt | Debug | Operator | Each turn | — | I/O |
| 48 | `response = llm.generate(conversation)` | Tokenize, decode, sample | Model text | Parser | Each turn | Ctrl+C; raise inside generate | **Seconds, ~527 MiB model already loaded** |
| 50–51 | print LLM OUTPUT | Show raw text | Debug | Operator | Each turn | — | I/O |
| 54–55 | `if not response: return ...` | Empty generation guard | Avoid `extract_json` on `''` | `main` print | First token EOG | — | Tiny |
| 57 | `decision = extract_json(response)` | Parse JSON object | `action` / `name` / `arguments` | Branches | After generate | No `{` / truncated / invalid JSON | Tiny |
| 58 | `action = decision.get("action")` | Read action, missing → `None` | Branches | After parse | — | Tiny |
| 64–65 | `if action == "answer": return decision["content"]` | Clean exit | Final string | `main` | Model chose answer | Missing `content` → KeyError | Tiny |
| 70–71 | comments | Protocol vs small-model | Readers | — | — | — |
| 73 | `tool_name = decision.get("name") if action == "tool" else action` | Canonical or action-as-name | `TOOLS.get` | Tool branch | — | Tiny |
| 74 | `tool_function = TOOLS.get(tool_name)` | Name → callable or `None` | `if tool_function` | Each candidate tool | Unknown name → None | Tiny |
| 75 | `if tool_function:` | Only execute allowlisted tools | Safety | Dispatch | — | — |
| 76–78 | `_usable_tool_arguments(decision.get("arguments", {}))` | Sanitize kwargs | `**` call | Inside if | — | Tiny |
| 80 | `tool_result = tool_function(**tool_arguments)` | **Run the skill** | Side effects / OCR / `input()` | Follow-up prompt | Skill `ValueError` **kills agent** | Skill-dependent |
| 85–94 | `conversation += f"""..."""` | Append response + raw `tool_result` + “decide next” + `ASSISTANT:` | Next generate sees history | Loop | After success only | String growth |
| 95 | `continue` | Next `while` iteration | Skip invalid-action raise | Loop | — | — |
| 97 | `raise ValueError(f"Invalid action: {action}")` | Unknown action | Operator | Not answer, not in TOOLS | — | — |
| 99–114 | `main` / `if __name__` | Load model, run agent, always `close()` | Process | Script entry | Ctrl+C after construct still closes | Model load/free |

**v1 tool-call stop conditions:** skill exception (no recovery); `answer`; empty response; invalid action; Ctrl+C; `kill`.

**v1 memory:** conversation string grows; no `agent_state`; tool result interpolated as Python `repr` of a `str` (OCR text or path).

---

## 13. `agent_v1.py` → `agent_v1_0_1.py` (every difference)

Identical except the tool execution block.

### 13.1 Removed (v1 line 80)

```python
tool_result = tool_function(**tool_arguments)
```

Bare call. `ValueError` from URL rejection, `FileNotFoundError` from `Image.open`, `TypeError` from bad kwargs, all unwind to `main` → `close` → traceback.

### 13.2 Added (v1_0_1 lines 80–91)

```python
try:
    result = tool_function(**tool_arguments)

    tool_result = {
        "success": True,
        "result": result,
    }
except Exception as e:
    tool_result = {
        "success": False,
        "error": str(e),
    }
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| 80 | `try:` | Catch skill failures | Keep `while True` alive | Each tool call | — | — |
| 81 | `result = tool_function(**tool_arguments)` | Same as v1 line 80 | Success payload | Each tool call | Exception → except | Skill |
| 83–86 | `tool_result = {success: True, result}` | Normalize | Follow-up prompt; model sees structure | After return | — | Tiny dict |
| 87 | `except Exception as e:` | Any exception, including `ValueError` | Recovery turn | On failure | `BaseException` (Ctrl+C `KeyboardInterrupt`) is **not** caught — process still aborts, `finally` closes model | Tiny |
| 88–91 | `{success: False, error: str(e)}` | Tell the model why | Next generate | After except | — | Tiny |

The follow-up f-string is **unchanged** from v1:

```
previous LLM response: {response}
TOOL RESULT:
{tool_result}
Now decide what to do next. If you think you have enough information, answer the user.
ASSISTANT:
```

`{tool_result}` is the dict’s **repr**, e.g. `{'success': False, 'error': 'document_image_uri must be a local file path...'}`. The model can read it; it is not JSON.

`except Exception` does **not** catch `KeyboardInterrupt` or `SystemExit`. A user Ctrl+C during `input()` or OCR still kills the agent. That is correct.

`main()`, imports, empty-response guard, answer branch, action compatibility, invalid-action raise: **byte-for-byte the same idea as v1** (v1_0_1 dropped the extra blank line after `_usable_tool_arguments(` that v1 had before `tool_result =`).

---

## 14. `agent_v1_0_1.py` → `agent_v1_0_2.py` (every difference)

### 14.1 New local: `agent_state` (line 34)

```python
agent_state = {}
```

| What | Detail |
|---|---|
| What it does | Bind an empty dict in `run_agent` |
| Purpose | Runtime memory distinct from `conversation` (see §4) |
| Who needs it | **Nothing yet.** No reads, no writes |
| When | Once per `run_agent` |
| Stop | Function return discards it |
| Memory | Empty dict until populated in a later version |
| CPU | ~0 |

This is the foundation from §4. It does not yet store `image_path` after `get_user_input`.

### 14.2 `decision.get("arguments", {})` formatting (lines 77–82)

Same call as 0.1; arguments split across lines. No semantic change.

### 14.3 Success payload includes `"tool"` (lines 90–94)

```python
tool_result = {
    "success": True,
    "tool": tool_name,
    "result": result,
}
```

The model sees **which** tool succeeded. Needed when history has several tools.

### 14.4 Except variable `exc` and `"tool"` on failure (lines 96–102)

```python
except Exception as exc:
    tool_result = {
        "success": False,
        "tool": tool_name,
        "error": str(exc),
    }
```

Same as 0.1 except the name `exc` and the extra field. `str(exc)` is the URL-rejection message or `Image.open` error.

### 14.5 Follow-up prompt (lines 105–124)

v1 / v1_0_1:

```text
previous LLM response: {response}

TOOL RESULT:
{tool_result}

Now decide what to do next. If you think you have enough information, answer the user.

ASSISTANT:
```

v1_0_2:

```text
Previous LLM response:
{response}

TOOL RESULT:
{json.dumps(tool_result, indent=2)}

Use the tool result above.

If the tool failed because information is missing,
use the appropriate tool to obtain the missing information.

Never invent file paths, URIs, user input, or tool results.

If you now have enough information,
answer the user.

ASSISTANT:
```

| Piece | Purpose |
|---|---|
| `Previous LLM response:` | Ground the next turn in what was just attempted |
| `json.dumps(..., indent=2)` | Structured JSON the parser-friendly model already knows | 
| `Use the tool result above.` | Direct attention to `success` |
| `If the tool failed because information is missing, use the appropriate tool...` | Route URL failure → `get_user_input` without a longer **system** prompt |
| `Never invent file paths...` | Runtime-adjacent instruction **at the point of failure**, not only in `SYSTEM_PROMPT` |
| `If you now have enough information, answer the user.` | Allow `action: answer` after OCR |
| `ASSISTANT:` | Cue continuation (still unofficial chat format) |

`json` import (line 3) is **used** here. In v1/v1_0_1 it was unused.

Source indentation inside the f-string adds leading spaces to the conversation. Those spaces become tokens. Harmless for this size; sloppy for a later chat template.

### 14.6 `continue` (line 126)

Same as 0.1: next generate. After a failed OCR, the process does **not** raise.

---

## 15. `agent_v1_0_2.py` line by line (complete file)

| Line | Code | What it does | Purpose | Who needs it | When | Stop | Memory / CPU |
|---|---|---|---|---|---|---|---|
| 1 | `import subprocess` | Unused | — | Import | — | Tiny |
| 2 | `import sys` | `sys.path` | Bootstrap | Import | — | Tiny |
| 3 | `import json` | `json.dumps` of `tool_result` | Follow-up prompt | Tool branch | Import | Tiny |
| 4 | `from pathlib import Path` | `_AI_DIR` | Bootstrap | Import | — | Tiny |
| 6 | comment | Dual launch | Humans | — | — | — |
| 7–9 | `_AI_DIR` / `sys.path` | Same as v1 | Imports | Import | — | Tiny |
| 11 | `MODEL_PATH` import | Unused name; config side effects | `find_llama_library` | Import | brew fail | brew |
| 12 | `TOOLS` | Allowlist | Dispatch | Tool branch | Import | Tiny |
| 13 | `localLLM` | Generate | `main` | Import | — | Tiny |
| 14 | `SYSTEM_PROMPT` | Tool list + JSON contract | `conversation` | Start | Import | Tiny |
| 15 | `extract_json` | Parse model text | Each turn | After generate | Import | Tiny |
| 18–27 | `_usable_tool_arguments` | §11 | kwargs | Tool branch | — | Tiny |
| 30–32 | `run_agent(llm, user_message)` | Loop | `main` | Once | Constructor fail | — |
| 34 | `agent_state = {}` | §4 / §14.1 | Future runtime facts | Start | Unused | Empty dict |
| 35–42 | `conversation = f"""SYSTEM...USER...ASSISTANT:"""` | Unofficial chat string | `generate` | Start | — | Grows each turn |
| 44 | `while True:` | Agent | Turns | Until return/raise/signal | Control |
| 48 | `print(conversation)` | Debug | Operator | Each turn | I/O |
| 49 | `llm.generate(conversation)` | Forward pass + sample | `response` | Each turn | Native decode / Ctrl+C | **Dominant** |
| 51–52 | print output | Debug | Operator | Each turn | I/O |
| 55–56 | empty → return string | Guard EOG-first | `main` | Empty generate | Tiny |
| 58 | `extract_json` | Dict | Branches | After text | Truncation / no JSON | Tiny |
| 59 | `action = decision.get("action")` | Maybe None | ifs | After parse | Tiny |
| 65–66 | `answer` → `return content` | Halt | `main` | Success path | KeyError if no content | Tiny |
| 71–72 | comments | Lenient protocol | Readers | — | — |
| 74 | name from `name` or `action` | §7 | `TOOLS.get` | Tool candidate | Tiny |
| 75 | `TOOLS.get(tool_name)` | Allowlist | `if tool_function` | Unknown → None | Tiny |
| 76 | `if tool_function:` | Execute only known tools | Safety | — | — |
| 77–82 | sanitize arguments | Drop schema echoes | `**kwargs` | Inside if | Tiny |
| 84 | `try:` | Skill may fail | Recovery | Each execution | — |
| 86–88 | `result = tool_function(**tool_arguments)` | Capability | `tool_result` | Blocks on `input()` | Exception / Ctrl+C | Skill |
| 90–94 | success dict with `tool` | Structured result | Prompt + future state | After return | Tiny |
| 96 | `except Exception as exc:` | URL reject, missing file, TypeError | Stay in loop | On error | KeyboardInterrupt not caught | Tiny |
| 98–102 | failure dict with `tool` + `error` | Model can recover | Prompt | After except | Tiny |
| 105–124 | append JSON tool result + recovery instructions | Next generate | Loop | After try/except | String + tokenize later |
| 126 | `continue` | Next turn | Skip line 128 | After append | — |
| 128 | `raise ValueError(Invalid action)` | Not answer, not a tool name | Operator | Bad JSON action | — |
| 130–142 | `main` | Construct, run, close | Process | `__main__` | close always if construct succeeded | Load/free ~527 MiB |
| 133–137 | hard-coded user task | Ask URI then OCR | Demo | Once | — | — |
| 144–145 | `__name__ == "__main__"` | Entry | `main` | Script | Import as module → skip | — |

**Who calls `run_agent`:** `main` only. **Who calls `tool_function`:** this file, never the LLM. **What user action stops a line:** Ctrl+C during generate or `input()`; answering path `return`; process kill.

**CPU:** almost all in `generate`. try/except and `json.dumps` are nothing. `get_user_input` waits on the human (wall time, not CPU). OCR is Tesseract on one image.

**Memory:** model stays loaded across turns; `conversation` grows (re-tokenized every generate); `agent_state` empty; `tool_result` dict discarded after interpolation except as text inside `conversation`.

---

## 16. Side-by-side tool execution

### v1

```python
tool_result = tool_function(**tool_arguments)
# conversation += raw tool_result
```

Crash on URL. No second turn.

### v1_0_1

```python
try:
    result = tool_function(**tool_arguments)
    tool_result = {"success": True, "result": result}
except Exception as e:
    tool_result = {"success": False, "error": str(e)}
# conversation += repr(tool_result)
```

Recovery possible. Payload is Python repr. No `tool` field. No `agent_state`.

### v1_0_2

```python
agent_state = {}  # unused store
try:
    result = tool_function(**tool_arguments)
    tool_result = {"success": True, "tool": tool_name, "result": result}
except Exception as exc:
    tool_result = {"success": False, "tool": tool_name, "error": str(exc)}
# conversation += json.dumps(tool_result) + recovery instructions
```

Same recovery, JSON to the model, named tool, empty state reserved, instructions at the failure site instead of a longer `SYSTEM_PROMPT`.

---

## 17. `SYSTEM_PROMPT` (unchanged across the three agents)

`utils/prompts/llama.py` is the system text all three interpolate. It was **not** lengthened to fix hallucination (the rule in §0). Recovery text lives in the **v1_0_2 follow-up** instead.

| Region | Role |
|---|---|
| `You are an agent.` | Role |
| `get_all_tools_description()` | Live JSON of `TOOLS_DESCRIPTIONS` |
| `valid JSON only` | Soft constraint; `extract_json` still needed |
| Example `action: tool` | Protocol |
| Example `action: answer` | Halt |
| `Do not output anything outside the JSON` | Soft |

f-string braces in examples are `{{` / `}}`.

---

## 18. Implemented vs still to build

| Item from this chapter | Where |
|---|---|
| Skill rejects `http://` / `https://` | `skills.py` |
| Provenance text on `document_image_uri` | `tools.py` |
| Tool errors do not crash the agent | `v1_0_1`, `v1_0_2` |
| Structured `{success, result\|error}` | `v1_0_1`, `v1_0_2` |
| `"tool"` field + `json.dumps` + recovery prompt | `v1_0_2` |
| `agent_state = {}` declared | `v1_0_2` |
| `agent_state` written after `get_user_input` / OCR | **Not yet** |
| Inject `agent_state["image_path"]` into OCR kwargs | **Not yet** |
| `Path.exists` / `is_file` / `expanduser` in skill | **Not yet** (exceptions still caught) |
| Strict parser (reject `action` == tool name) | **Not yet** (lenient on purpose) |
| Rename to `ocr_swedish_document_image` | **Not yet** |
| Validation middleware layer (§9) | **Not yet** |
| ChatML `messages` list | Not in these three files (string `conversation`) |

---

## 19. Call graph for the recovered URL failure (`v1_0_2`)

```text
main
 └─ localLLM()                         load weights
 └─ run_agent(llm, "Ask the user...")
       agent_state = {}
       conversation = SYSTEM + USER + ASSISTANT
       generate #1
            model: parse_text_... https://example.com/...
       extract_json
       TOOLS["parse_text_from_image_of_swedish_document"]
       try: skill → ValueError (http)
       tool_result = {success: false, tool: "...", error: "..."}
       conversation += JSON + "use get_user_input if missing..."
       generate #2
            model: hopefully get_user_input
       try: input()                   human types local path
       tool_result = {success: true, result: "/Users/.../x.png"}
       generate #3
            model: parse_text with that path
       try: OCR
       generate #4
            action: answer
       return content
 └─ close()
```

If generate #2 still invents a URL, Python rejects again; the loop continues until `answer`, empty response, invalid action, or the user aborts.

---

## 20. Stop conditions unique to the runtime upgrade

| Event | v1 | v1_0_1 / v1_0_2 |
|---|---|---|
| OCR `ValueError` (URL) | Process crash after `close` | Next LLM turn with `success: false` |
| Missing file | Crash (`FileNotFoundError`) | Next turn with error string |
| `get_user_input` returns a path | LLM must copy it | Same (state not wired yet) |
| Ctrl+C during `input()` | `KeyboardInterrupt`; `finally` closes | Same (`Exception` does not catch it) |
| `action: answer` | Return | Return |
| Empty generate | Return apology string | Return apology string |

---

## 21. What to read next

- Chapter 08 — original `while True` agent (pre-versioned files)
- Chapter 09 — `TOOLS` allowlist
- Chapter 10 — `extract_json` and why the model’s text is not a type system

The runtime in `agent_v1_0_2.py` is the start of the middle layer in §9: catch, structure, tell the model, do not trust it with authority it has not earned.
