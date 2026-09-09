# Chapter 12 — Building `libllama_bridge.dylib`

Source: `llm_server/src/ai/engine/llama/build_bridge.sh`

Python never compiles C++. You (or a prior session) must produce `ai/engine/llama/libllama_bridge.dylib` before `ctypes.CDLL(LLAMA_BRIDGE_PATH)` can succeed.

`config.py` points at that exact path. Homebrew provides `libllama.dylib` separately; the script does not copy it.

---

## 1. The script

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LLAMA_PREFIX="$(brew --prefix llama.cpp)"
GGML_PREFIX="$(brew --prefix ggml)"

c++ -std=c++17 -dynamiclib bridge.cpp \
  -I"$LLAMA_PREFIX/include" \
  -I"$GGML_PREFIX/include" \
  -L"$LLAMA_PREFIX/lib" \
  -L"$GGML_PREFIX/lib" \
  -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA_PREFIX/lib" \
  -Wl,-rpath,"$GGML_PREFIX/lib" \
  -o libllama_bridge.dylib

echo "Built $(pwd)/libllama_bridge.dylib"
```

---

## 2. Line-by-line

| Line | What it does | Purpose | Who needs it | When | Stop | Cost |
|---|---|---|---|---|---|---|
| `#!/usr/bin/env bash` | Find `bash` on `PATH` | Run as `./build_bridge.sh` | The OS exec | Script start | No bash → fail | ~0 |
| `set -euo pipefail` | Exit on error (`-e`), unset var (`-u`), failed pipe (`-o pipefail`) | Do not produce a half-linked dylib after a quiet failure | You | Entire script | Any command non-zero | ~0 |
| `cd "$(dirname "$0")"` | cwd = directory of the script (`engine/llama`) | `-o libllama_bridge.dylib` lands next to `bridge.cpp` | `CDLL` path | Start | Missing script path | ~0 |
| `brew --prefix llama.cpp` | Print keg prefix | Headers (`llama.h`) and `libllama.dylib` | `c++` `-I` / `-L` / rpath | Each build | Formula not installed | subprocess |
| `brew --prefix ggml` | Same for ggml | `ggml` headers/libs llama.cpp depends on | Linker | Each build | Formula not installed | subprocess |
| `c++ -std=c++17` | Invoke the C++17 compiler | `bridge.cpp` uses C++ (`static_cast`, `nullptr`) | Output dylib | Each build | Compile error | seconds |
| `-dynamiclib` | Make a Mach-O dylib, not an executable | `CDLL` needs a shared library | Python | Link | — | — |
| `bridge.cpp` | Single translation unit | Our wrappers | Linker | Compile | Syntax error | — |
| `-I.../include` | Add include paths | `#include <llama.h>` | Preprocessor | Compile | Wrong prefix → file not found | — |
| `-L.../lib` | Add link search paths | Resolve `-lllama` etc. | Linker | Link | missing dylib | — |
| `-lllama -lggml -lggml-base` | Link those libraries | Undefined-symbol errors otherwise | Runtime `dlopen` of the bridge | Link | Version skew | — |
| `-Wl,-rpath,...` | Embed runtime search paths | When Python `dlopen`s the bridge, the dynamic loader still finds Homebrew `libllama` | `CDLL` at agent start | Link / runtime | Wrong rpath → `image not found` | — |
| `-o libllama_bridge.dylib` | Output name | Matches `LLAMA_BRIDGE_PATH` | `inference.py` | Link | Disk full | writes small file |
| `echo Built ...` | Print absolute path | Confirm cwd + success | Operator | After link | `-e` already passed | I/O |

**Called by:** you, in a terminal. **Not** called by `simple_agent.py` or `localLLM`.

**When:** After editing `bridge.cpp`, or after upgrading `llama.cpp` / `ggml` via Homebrew (ABI break).

**What user action stops a build:** Ctrl+C during `c++`; missing Xcode CLT; `brew` missing.

**Memory / CPU of the build:** compiling one ~170-line file is seconds and negligible RAM. It does **not** load the GGUF.

---

## 3. Why rpath exists

Without rpath:

```text
Python CDLL(libllama_bridge.dylib)
        │
        ▼
bridge references libllama.dylib
        │
        ▼
dyld looks in default paths, not necessarily Homebrew
        │
        ▼
OSError: Library not loaded: libllama.dylib
```

With `-Wl,-rpath,"$LLAMA_PREFIX/lib"` the dylib records “also look here.”

`inference.py` still `CDLL`s `libllama` first as belt-and-suspenders. Both mechanisms keep the same keg loaded.

---

## 4. `extern "C"` and the linker

`bridge.cpp` wraps symbols in `extern "C"`. The dylib’s export table then contains:

```text
_llama_bridge_init
_llama_bridge_model_load
...
```

ctypes looks up `llama_bridge_init` (the leading underscore is a Mach-O convention ctypes handles).

If you compiled as C++ without `extern "C"`, `nm` would show mangled names and Python would raise `AttributeError`.

**Who needs the unmangled names:** `define_llama_bridge_signatures` and every `self.llama_bridge.llama_bridge_*` call.

---

## 5. Relationship to runtime stop conditions

Building is a **developer** action. It does not run the model.

| If you skip the build after a C++ edit | Runtime |
|---|---|
| Old dylib still present | Agent runs **old** wrappers |
| Dylib missing | `CDLL` raises at `localLLM()`; no `close()`; no GGUF load |
| Dylib built against newer llama.cpp than installed | Missing symbols at `CDLL` or first call |

**User running the agent never executes `build_bridge.sh`.** Stopping the agent does not stop a build in another terminal.

---

## 6. Syntax used (bash)

| Syntax | Meaning |
|---|---|
| `"$(dirname "$0")"` | Directory of this script, quoted against spaces |
| `$(brew --prefix ...)` | Command substitution, captured into a variable |
| `\` at EOL | Line continuation of one `c++` invocation |
| `-Wl,arg` | Pass `arg` to the linker |
| `-lfoo` | Link `libfoo.dylib` |

`set -u` means a typo in `$LLAMA_PREFIX` aborts instead of passing an empty `-I`.

---

## 7. What this script does not do

- Does not download the GGUF.
- Does not `pip install` anything.
- Does not run tests.
- Does not codesign beyond what `c++` produces locally.
- Does not build llama.cpp itself (Homebrew already did).

After a successful build, the textbook’s runtime chapters apply unchanged.
