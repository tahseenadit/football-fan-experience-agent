# Chapter 09 — Tools, Skills, and the Registry

Source files:

- `llm_server/src/ai/skills/simple_skill.py`
- `llm_server/src/ai/skills/tools/tools.py`

These files contain **no neural network**. They are ordinary Python the agent is allowed to invoke when the model emits a tool action.

---

## 1. Theory: who does what

```text
LLM reasoning capability     ← Qwen (tokens)
Tool definitions             ← these files + SYSTEM_PROMPT text
Tool-selection decision      ← Qwen (JSON)
Tool execution               ← Python (this chapter)
Agent loop                   ← simple_agent.py
```

The model cannot import `simple_test`. It can only write the name `"simple_test"`. Python turns that string into a call.

```text
"simple_test"  →  TOOLS["simple_test"]  →  simple_test()  →  "Simple test"
```

---

## 2. `simple_skill.py`

```python
def simple_test() -> str:
    return "Simple test"
```

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `def simple_test() -> str:` | Define a zero-arg function returning a string | Smallest possible tool | `TOOLS` dict; `run_agent` via `**arguments` | When the registry is imported (definition only) | N/A | 0 until called |
| `return "Simple test"` | Produce a constant | Give the model a fact to talk about on the next turn | `conversation += TOOL RESULT` | When `tool_function(**kwargs)` runs | Never called if model answers directly or name mismatches | Nanoseconds; interned `str` |

**Called by:** `run_agent` → `tool_function(**tool_arguments)` when `tool_name == "simple_test"` and `arguments` is `{}` or omitted.

**Not called by:** the model, ctypes, llama.cpp, `generate`.

**Type hint `-> str`:** documentation only at runtime.

**Arguments:** none. If the model sends `"arguments": {"foo": 1}`, Python raises `TypeError: simple_test() got an unexpected keyword argument 'foo'`. That exception leaves `run_agent` and hits `main`’s `finally`.

**What user action stops this line:**

- Not running the agent.
- Model chooses `answer` first.
- User Ctrl+C during the first `generate` (tool never reached).
- JSON parse failure before the tool branch.

**Memory / CPU:** One pointer to a constant string. No I/O, no allocations beyond the call frame.

**Why it exists:** Prove the loop: select → execute → observe → answer. PDF Chapter 3.

---

## 3. `tools.py`

```python
import sys
from pathlib import Path

_AI_DIR = Path(__file__).resolve().parents[2]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from skills.simple_skill import simple_test

TOOLS = {
    "simple_test": simple_test,
}
```

### 3.1 Bootstrap

`tools.py` lives at `ai/skills/tools/tools.py`, so `parents[2]` is `ai/`:

```text
tools.py → tools/ → skills/ → ai/
```

Same pattern as `inference.py`. Needed if this file is ever executed as a script; when imported from `simple_agent` after `ai/` is already on `path`, the `if` is a no-op.

| Line | What it does | Purpose | Who needs it | Cost |
|---|---|---|---|---|
| `import sys` / `Path` | Enable path fix | Resolve `skills.simple_skill` | The import below | Tiny |
| `_AI_DIR = ... parents[2]` | Find `ai/` | `sys.path` | Import of `simple_test` | Tiny |
| `if not in path: insert` | Idempotent prepend | Dual invocation | Subsequent imports | Tiny |

### 3.2 Import the skill

```python
from skills.simple_skill import simple_test
```

**What it does:** Load `simple_skill.py` and bind the function object.

**Purpose:** The registry must hold a callable, not a string.

**Who needs it:** The `TOOLS` literal.

**When:** Import of `tools.py` (triggered by `simple_agent`).

**Stop:** Import error (file missing). The function body does not run yet.

### 3.3 The registry

```python
TOOLS = {
    "simple_test": simple_test,
}
```

| Syntax | Meaning |
|---|---|
| `"simple_test"` | **Public name** the model must emit. Must match `SYSTEM_PROMPT`. |
| `simple_test` | **Function object** (not called here). |

**Who needs `TOOLS`:** `run_agent` (`TOOLS.get(tool_name)`).

**When it is built:** Import time, once.

**When it is read:** Each tool action.

**What stops a lookup from succeeding:** Typo in model output (`"simpletest"`); prompt and dict out of sync after a rename.

**Memory:** One dict, one str key, one function reference. Bytes.

**CPU:** `dict.get` is O(1) average.

**Growth pattern** (PDF, not yet in repo):

```python
TOOLS = {
    "simple_test": simple_test,
    "read_file": read_file,
    "run_sql": run_sql,
}
```

Each new tool needs: a Python function, a dict entry, and matching text in `SYSTEM_PROMPT`. Forgetting the prompt means the model does not know the tool exists. Forgetting the dict means the model can name it and Python will `ValueError`.

---

## 4. Call graph for one tool invocation

```text
Qwen generates text
        │
        ▼
extract_json
        │
        ▼
decision["name"] == "simple_test"
        │
        ▼
TOOLS.get("simple_test")  →  function simple_test
        │
        ▼
simple_test()             →  "Simple test"
        │
        ▼
conversation += "TOOL RESULT:\nSimple test\n..."
        │
        ▼
generate(conversation)    # model sees the result as TEXT
```

The second generate does **not** receive a Python object. It receives characters. The model cannot call `simple_test` again except by emitting JSON again.

---

## 5. Security / capability boundary

`TOOLS` is an allowlist.

| The model writes | Python does |
|---|---|
| `"simple_test"` | Call that function |
| `"os.system"` | `TOOLS.get` → `None` → `ValueError` |
| arbitrary code in `content` | Not executed; only returned if `action == "answer"` |

**Who needs this allowlist:** anyone who later adds tools that touch disk, network, or SQL. The current tool cannot harm anything.

**CPU / memory of a future dangerous tool** would be that tool’s, not the registry’s. The registry does not sandbox calls.

---

## 6. Stop conditions specific to tools

| Event | What does not run |
|---|---|
| First generate chooses `answer` | `simple_test` body |
| `extract_json` fails | Registry lookup |
| Unknown name | Function body |
| `simple_test(**bad_kwargs)` | Append / second generate |
| Ctrl+C during first generate | Everything in this chapter except imports |
| Ctrl+C during `simple_test` | Unlikely; function is instant. `finally` still closes the model |

There is no user UI to “cancel the tool.” Cancellation is OS-level.

---

## 7. Dual-write consistency checklist

When you add a tool, three places must agree:

1. Function in `skills/`
2. Entry in `TOOLS`
3. Description + JSON example in `SYSTEM_PROMPT`

The runtime does not generate the prompt from the registry. Drift is a human error.

| Field in JSON | Consumer |
|---|---|
| `action` | `run_agent` ifs |
| `name` | `TOOLS.get` |
| `arguments` | `function(**arguments)` |

`simple_test` documents `Arguments: none` in the prompt and implements that in Python. Both must stay true.
