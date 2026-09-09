# Chapter 08 — `simple_agent.py` Line by Line

Source: `llm_server/src/ai/agents/simple_agent.py`

This file is the process entry point and the **agent runtime**. The model only generates text. This file decides whether that text is a final answer or a tool call, executes Python, and loops.

```text
LLM     = text generator
Agent   = this while-True loop + TOOLS + conversation string
```

---

## 1. Imports and path bootstrap

```python
import subprocess
import sys
import json
from pathlib import Path

_AI_DIR = Path(__file__).resolve().parents[1]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from config.config import MODEL_PATH
from skills.tools.tools import TOOLS
from engine.llama.inference import localLLM
from utils.prompts.llama import SYSTEM_PROMPT
from utils.llama_utils import extract_json
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `import subprocess` | Bind subprocess | **Nothing in this file uses it** | — | Import | — | Tiny, unused |
| `import sys` | Bind sys | `sys.path` | Bootstrap | Import | — | Tiny |
| `import json` | Bind json | **Unused** (`extract_json` uses json internally) | — | Import | — | Tiny, unused |
| `from pathlib import Path` | Bind Path | `_AI_DIR` | Bootstrap | Import | — | Tiny |
| `_AI_DIR = Path(__file__).resolve().parents[1]` | `simple_agent.py` → `agents/` → `ai/` | Same as inference’s `parents[2]` but one level closer | `sys.path` | Import | Wrong parent → imports fail | Tiny |
| `if ... not in sys.path` / `insert` | Prepend `ai/` | Allow `python agents/simple_agent.py` and `python -m agents.simple_agent` | All `from config...` imports | Import | Already on path → skip insert | Tiny |
| `from config.config import MODEL_PATH` | Import config (runs `brew`) | **MODEL_PATH unused** in this file | Side effect: config import | Import | brew fail | brew |
| `from skills.tools.tools import TOOLS` | Load registry | `TOOLS.get(name)` | `run_agent` | Import | Import error | Tiny |
| `from engine.llama.inference import localLLM` | Load inference (does **not** load GGUF yet) | Construct later | `main` | Import | Missing dylibs not yet | Imports + brew via config |
| `from utils.prompts.llama import SYSTEM_PROMPT` | Bind instruction string | First `conversation` | `run_agent` | Import | — | Tiny |
| `from utils.llama_utils import extract_json` | Bind parser | Each turn | `run_agent` | Import | — | Tiny |

Comment on lines 6–7 states the two supported invocation styles. Both require `ai/` on `sys.path`.

---

## 2. `run_agent(llm, user_message) -> str`

**Called by:** `main()` with a hard-coded user string.

**When:** Once per process in the current program. Internally loops.

**Returns:** `decision["content"]` when the model chooses `action == "answer"`.

**What stops it returning:** Unknown tool; invalid JSON; invalid action; context overflow inside `generate`; Ctrl+C; infinite tool loop (no max-turns cap).

### 2.1 Build the first conversation

```python
conversation = f"""
{SYSTEM_PROMPT}

USER:
{user_message}

ASSISTANT:
"""
```

| Syntax / line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `f"""..."""` | Interpolate `SYSTEM_PROMPT` and `user_message` | One string the model will complete | `llm.generate` |
| `USER:` / `ASSISTANT:` | Pseudo chat markup | Cue a reply after `ASSISTANT:` | The model (informally) |

This is **not** Qwen’s official chat template. See Chapter 10.

**When:** Once, at the start of `run_agent`. Later turns **append** to the same string.

**Memory:** A few KB of text. After tokenize, ~100–200+ tokens.

**CPU:** String format only. The cost is the later `generate`.

**Who calls this block:** Nobody; it is the first statements of `run_agent`.

**What stops it:** `run_agent` never called (`localLLM()` failed).

### 2.2 `while True:` — the agent

```python
while True:
    response = llm.generate(conversation)
    print("LLM OUTPUT:")
    print(response)
    decision = extract_json(response)
    ...
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `while True:` | Infinite loop | Allow tool → model → tool → … | The definition of this agent | Control only |
| `llm.generate(conversation)` | Full tokenize/decode/sample | Get a text decision | Parser + logs | **Seconds** |
| `print("LLM OUTPUT:")` | Label | Debug | Operator | I/O |
| `print(response)` | Dump raw model text including `<think>` | Debug parse issues | Operator | I/O |
| `extract_json(response)` | Slice `{`..`}` and `json.loads` | Structured `decision` | The `if`s below | µs |

**Called by:** `main` → `run_agent`. `generate` is called by this loop.

**When:** First iteration immediately; later iterations after `continue`.

**What stops the next `generate`:**

| Condition | Mechanism |
|---|---|
| Model answered | `return decision["content"]` leaves `run_agent` |
| Unknown tool | `raise ValueError` |
| Bad action | `raise ValueError` |
| Parse / generate error | Exception to `main` |
| User Ctrl+C | `KeyboardInterrupt` to `main` `finally` |
| User kill | Process dies |

There is **no** `max_iterations`. A model that always requests `simple_test` would loop until context overflow or the user aborts.

### 2.3 Answer branch — the only clean exit

```python
if decision["action"] == "answer":
    return decision["content"]
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `decision["action"] == "answer"` | Read the JSON field | Distinguish halt vs tool | Control flow |
| `return decision["content"]` | Exit `run_agent` with a string | Final user-visible answer | `main` print |

**Who calls this:** The loop, when the model emitted that action.

**When:** Typically the **second** generate in the demo (“please run the simple test” → tool → then answer). Could be the first generate if the model refuses the tool.

**What user action stops this line from running:** The model never emits `answer` (keeps calling tools); user aborts first; parse fails first.

**Memory / CPU:** Two dict lookups. The heavy work already happened in `generate`.

Missing `"content"` → `KeyError` (not handled).

### 2.4 Tool branch

```python
if decision["action"] == "tool":
    tool_name = decision["name"]
    tool_arguments = decision.get("arguments", {})
    tool_function = TOOLS.get(tool_name)
    if not tool_function:
        raise ValueError(f"Tool {tool_name} not found")
    tool_result = tool_function(**tool_arguments)
    print(f"TOOL RESULT: {tool_result}")
    conversation += f"""
    {response}

    TOOL RESULT:
    {tool_result}

    Now decide what to do next.

    ASSISTANT:
    """
    continue
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `if action == "tool"` | Second legal action | Enter execution | Tool body | ~0 |
| `decision["name"]` | Tool key | Index `TOOLS` | Registry | ~0 |
| `decision.get("arguments", {})` | Kwargs or empty dict | `simple_test` takes none; future tools may | `**tool_arguments` | ~0 |
| `TOOLS.get(tool_name)` | Name → Python function | Late binding | Call below | ~0 |
| `if not tool_function` | Missing name | Do not call `None` | Safety | — |
| `tool_function(**tool_arguments)` | **Actually run Python** | Side effects / data | `tool_result` string | **Depends on the tool**; `simple_test` is ~0 |
| Print result | Debug | Operator | I/O |
| `conversation += f"""..."""` | Append model output + tool result + cue | Next `generate` sees the tool outcome | Next loop | String alloc |
| `continue` | Skip the invalid-action raise | Next `while` iteration | Loop | ~0 |

**Who calls `simple_test`:** this block, via `TOOLS["simple_test"]`, not the LLM.

**When:** When the parsed JSON says so. In the canned demo, turn 1.

**What stops the tool from running:**

- Model chose `answer`.
- Name not in `TOOLS`.
- `**arguments` mismatch (`TypeError` if the model invents keys `simple_test` does not accept).
- User abort before this line.

**What stops the `continue` from happening:** Raise in the tool; print is after the call so a raising tool skips the append.

**Indentation inside the appended f-string:** the continuation lines are indented in the source, so the string contains leading spaces. Those spaces become tokens. Harmless for a small model, sloppy for a template.

**Memory:** `conversation` grows by `len(response) + len(tool_result) + cue` each tool turn. That growth is what eventually trips `Context too small`.

**CPU after `continue`:** another full `generate` (re-decode the whole string).

### 2.5 Invalid action

```python
raise ValueError(f"Invalid action: {decision['action']}")
```

Reached only if `action` is neither `"answer"` nor `"tool"`.

**Who needs it:** The operator (crash instead of silent hang).

**When:** Malformed-but-JSON model output, e.g. `"action": "search"`.

**What stops it:** The two `if`s returning or continuing first.

---

## 3. `main`

```python
def main():
    llm = localLLM()
    try:
        result = run_agent(
            llm,
            "Please run the simple test.",
        )
        print("\nFINAL ANSWER:")
        print(result)
    finally:
        llm.close()
```

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `llm = localLLM()` | Load dylibs + GGUF | One model for all turns | `run_agent` | **Seconds, ~527 MiB** |
| `try:` | Pair with `finally` | Always unload | GPU/RAM | — |
| `run_agent(llm, "Please run the simple test.")` | Hard-coded user utterance | Demo the tool path | Prints | 1–N generates |
| `print FINAL ANSWER` / `print(result)` | Show `content` | Operator | I/O |
| `finally: llm.close()` | `model_free` + `backend_free` | Release weights | OS | Free |

**Called by:** `if __name__ == "__main__": main()`.

**When:** Script execution.

**What stops `run_agent` from being called:** `localLLM()` raising. Then `finally` is **not** run because `try` never started.

**What stops `FINAL ANSWER` from printing:** Any exception in `run_agent` (including Ctrl+C). `close()` still runs.

**User actions:**

| Action | Effect |
|---|---|
| Run the script and wait | Tool + answer + unload |
| Ctrl+C after model loaded | `close()`; no final answer |
| Ctrl+C during load | Possibly no `close()` |
| Edit the string in `main` | Different first user message; no other CLI |

There is no `input()`. The “user” is this literal.

---

## 4. Module entry

```python
if __name__ == "__main__":
    main()
```

| Line | What it does | Purpose | Who needs it |
|---|---|---|---|
| `__name__ == "__main__"` | True only when this file is the entry script | Avoid running the agent if someone imports `simple_agent` | `main` |
| `main()` | Start the lifetime | The program | The operator |

**When:** Process start.

**What stops it:** Importing the module from elsewhere; then this block is false.

---

## 5. Happy-path call sequence (demo string)

```text
main
 └─ localLLM.__init__                         load Qwen
 └─ run_agent(..., "Please run the simple test.")
       conversation = SYSTEM + USER + ASSISTANT
       generate #1
            model emits <think> + JSON tool
       extract_json → {action: tool, name: simple_test}
       TOOLS["simple_test"]() → "Simple test"
       conversation += response + TOOL RESULT
       generate #2
            model emits JSON answer
       extract_json → {action: answer, content: "..."}
       return content
 └─ print FINAL ANSWER
 └─ close                                unload Qwen
```

PDF Chapters 3 and 5 describe this exact path.

---

## 6. Memory / CPU of the agent layer itself

The agent layer is cheap. Almost all time and RAM sit under `localLLM`.

| Agent-only object | Size | Lifetime |
|---|---|---|
| `SYSTEM_PROMPT` | hundreds of chars | Process |
| `conversation` | grows each tool turn | `run_agent` |
| `response` | ≤ ~64 new tokens of text, plus think | One turn |
| `decision` | tiny dict | One turn |
| `tool_result` | `"Simple test"` | One turn |

**CPU unique to the agent:** `extract_json`, dict checks, one function call, string append. All dominated by `generate`.

**RAM unique to the agent:** the conversation string is duplicated into UTF-8 `bytes` inside every `generate` tokenize. Two copies exist briefly (Python `str` + `bytes`).

---

## 7. Missing production guards (documented so you see the lines that are *not* there)

| Missing line | What would stop | Why it matters |
|---|---|---|
| `for turn in range(MAX_TURNS):` | Infinite tool loops | A confused model can spin |
| `try/finally` around `ctx` in `generate` | Context leak on raise | Server longevity |
| Chat template apply | Ad-hoc USER/ASSISTANT text | `<think>` / formatting drift |
| Schema validation beyond two `if`s | Extra keys, wrong types | `KeyError` / `TypeError` |
| `input()` | Hard-coded user | Not interactive |

These absences are part of the current textbook snapshot, not implied features.
