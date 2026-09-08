import os
from utils.llama_utils import find_llama_library

AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # llm_server/src/ai/config
MODEL_PATH = os.path.join(AI_DIR, "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_0.gguf")
LLAMA_LIBRARY_PATH = find_llama_library()
LLAMA_BRIDGE_PATH = os.path.join(AI_DIR, "engine/llama/libllama_bridge.dylib")