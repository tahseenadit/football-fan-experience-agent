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
