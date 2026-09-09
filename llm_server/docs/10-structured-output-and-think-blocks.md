# Chapter 10 — Structured Output, Parsing, and `<think>` Blocks

Sources: `extract_json` in `llama_utils.py`, `SYSTEM_PROMPT` in `prompts/llama.py`, and PDF Chapters 5–6.

---

## 1. The contract the agent wishes it had

`SYSTEM_PROMPT` tells the model:

```text
You must respond using valid JSON only.
Do not output anything outside the JSON.
```

and shows two shapes:

```json
{"action": "tool", "name": "simple_test", "arguments": {}}
{"action": "answer", "content": "your answer here"}
```

Python originally trusted that:

```python
decision = json.loads(response)
```

**Purpose of that line:** turn text into a dict the `if action` branches can read.

**Who needed it:** `run_agent`.

**What actually arrived:**

```text
<think>
</think>
{
  "action": "tool",
  "name": "simple_test",
  "arguments": {}
}
```

`json.loads` saw `<` at column 0 and raised `JSONDecodeError`. The tool was **never executed**, even though the model had chosen it correctly. Then `finally` unloaded the model.

---

## 2. Concept: probabilistic text vs structured actions

```text
LLM output is probabilistic text
        │
        ▼
Agent runtime must interpret it safely
        │
        ▼
structured action
        │
        ▼
execute tool  or  return content
```

An instruction in the prompt is **not a type system**. A 0.8B model trained with a thinking mode will often violate “JSON only.”

**Who needs a parser more sophisticated than `json.loads(entire_string)`:** any agent that uses instruction-tuned / reasoning models.

**CPU of parsing:** irrelevant. **Cost of not parsing:** a crash after a full generate (the expensive part already paid).

---

## 3. `extract_json` — current parser

```python
def extract_json(response: str) -> dict:
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"No JSON object found in model response:\n{response}")
    json_text = response[start:end + 1]
    return json.loads(json_text)
```

| Line | What it does | Purpose | Who needs it | Failure |
|---|---|---|---|---|
| `find("{")` | First opening brace | Skip `<think>...` | Slice | `-1` if the model never opened JSON |
| `rfind("}")` | Last closing brace | Include the object even if think-text had earlier `}` | Slice | `-1` if never closed |
| `if start == -1 or end == -1` | Detect absence | Readable error with raw text | Operator | Raises |
| `response[start:end+1]` | Inclusive substring | Candidate JSON | `json.loads` | May be too wide if extra braces exist |
| `json.loads` | Parse | `dict` | `run_agent` | `JSONDecodeError` if slice is not JSON |

**Called by:** `run_agent` every turn, after `generate`.

**When:** After tokens are already finalized. Parsing cannot change the KV cache (which was already freed).

**What stops it:** `generate` raising; user abort during generate.

**What user action stops a *future* parse:** answering path `return` (no more turns); kill.

**Memory:** One substring. If `response` is ~1–2 KB, the slice is similar.

**CPU:** Linear scan of a tiny string.

### Known limitation

`find` + `rfind` is not a JSON parser. If `<think>` contains `{` or the model emits two objects, the slice can swallow extra text and `json.loads` fails, or the wrong object wins.

Good enough for: think-wrapper + one object. Not good enough for: JSON inside thoughts, or `content` strings with nested objects as the only braces.

---

## 4. Where `<think>` comes from

**Not from** becoming an agent. **Not from** llama.cpp growing a “reasoning runtime.” **Not from** Python inserting tags.

It comes from **tokens the model sampled**, because the **prompt changed**.

### 4.1 Before the agent

Prompt like:

```text
The capital of France is
```

Continuation is usually ` Paris`. No reason to open a thinking span.

### 4.2 After the agent

Prompt like:

```text
You are an agent.
You have access to these tools:
...
You must respond using valid JSON only.
...
USER:
Please run the simple test.
ASSISTANT:
```

That is an instruction / decision problem. Qwen3.5 was trained to emit:

```text
<think> ... </think>
<final answer>
```

for many instruction prompts. Greedy sampling then produces those tokens first:

```text
prompt
  → llama_decode
  → logits
  → greedy
  → token  token  token
  → "<think>"  "\n"  ...  "</think>"  "\n"  "{"  ...
```

**Who “created” `<think>`:** the sampled vocabulary of Qwen, given this prompt.

**Who must tolerate it:** `extract_json` / any future stripper.

**When it appears:** Often on the first tokens of an agent `generate`. Not guaranteed every time; greedy + this prompt made it show up in the PDF.

**What stops the model from emitting it:** a different prompt / chat template / generation config that disables thinking (not implemented here). Stripping in Python does not stop *generation* of those tokens; it only ignores them after the fact. You still **pay decode_one** for every think token, and they count toward `MAX_NEW_TOKENS` (64).

That last point is a resource issue: a long `<think>` can eat the 64-token budget and cut off the JSON.

---

## 5. Why the unofficial chat format matters

Current construction:

```python
conversation = f"""
{SYSTEM_PROMPT}
USER:
{user_message}
ASSISTANT:
"""
```

```text
Python strings with the words USER / ASSISTANT
        │
        ▼
tokenize as raw text
        │
        ▼
Qwen (best-effort)
```

Intended later architecture (PDF, not in tree):

```text
[{"role":"system", ...}, {"role":"user", ...}]
        │
        ▼
Qwen chat template
        │
        ▼
model-specific special tokens
        │
        ▼
tokenize
```

**Purpose of a real template:** control roles, tool-call channels, and thinking mode explicitly instead of hoping the small model infers structure from English labels.

**Who needs the template:** future-you, before adding many tools.

**CPU:** Templating is string work. The win is fewer wasted think tokens and more reliable JSON, which saves **decode** cost.

---

## 6. End-to-end of the first successful tool parse

```text
USER
  │
  ▼
Qwen
  │
  ▼  raw: <think></think>{ "action":"tool", ... }
extract_json
  │
  ▼  dict
TOOLS["simple_test"]
  │
  ▼
simple_test()
```

The PDF’s crash was only at `json.loads`. After `extract_json`, the rest of Chapter 08 runs.

---

## 7. Stop conditions for parsing and thinking tokens

| Event | Think tokens | `extract_json` | Tool / answer |
|---|---|---|---|
| EOG before any `{` | May have been emitted | Raises “No JSON” | No |
| 64 tokens mid-JSON | Cut off | Often `JSONDecodeError` | No |
| Ctrl+C during think tokens | Partial stdout | Not called | No |
| Successful slice | Already paid in generate | Returns dict | Yes |
| User changes prompt to disable thinking | May not appear | Still works if JSON is whole | Yes |

There is no user control to “skip thinking” at runtime.

---

## 8. Memory / CPU of think tokens

Each think token:

- 1 × `sample_greedy` (sampler alloc/free)
- 1 × `token_to_piece`
- 1 × `decode_one` (the real cost)
- a few bytes on `generated_text`

If `<think></think>\n` is 10 tokens, that is 10/64 of the generation budget and ~10 token-decodes of GPU/CPU **before** any JSON the agent can use.

The parser’s own cost does not depend on whether think tags were present, except that `find("{")` starts later in the string.

---

## 9. Syntax recap

| Syntax | Role |
|---|---|
| `str.find` / `str.rfind` | Byte-oriented index in Python 3 (Unicode code points); fine for ASCII `{` `}` |
| Slice `s[i:j]` | New `str`; does not copy the model |
| `json.loads` | Standard parser; strict |
| f-string prompt | Not JSON; only examples of JSON as text |
| `<think>` | Model-emitted text / special tokens; not Python syntax |

---

## 10. Division of responsibility (keep this table)

| Concern | Owner |
|---|---|
| Reasoning text | Qwen |
| Tool list text | `SYSTEM_PROMPT` (us) |
| Decision JSON | Qwen |
| Making JSON usable | `extract_json` (us) |
| Running the function | `run_agent` + `TOOLS` (us) |
| Loop / state | `conversation` (us) |
