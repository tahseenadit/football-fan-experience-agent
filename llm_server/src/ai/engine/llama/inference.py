import ctypes
import sys
from pathlib import Path

# ai/engine/llama/inference.py -> parents[2] == ai/
_AI_DIR = Path(__file__).resolve().parents[2]
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from config.config import LLAMA_LIBRARY_PATH, LLAMA_BRIDGE_PATH, MODEL_PATH
from config.model_config.llama import MAX_TOKENS, CONTEXT_SIZE, MAX_NEW_TOKENS
from utils.llama_utils import define_llama_bridge_signatures, init_llama
from utils.prompts.llama import prompt

print(f"llama.cpp library: {LLAMA_LIBRARY_PATH}")

llama_lib = ctypes.CDLL(LLAMA_LIBRARY_PATH)

print("Successfully loaded llama.cpp into Python.")

llama_bridge = ctypes.CDLL(LLAMA_BRIDGE_PATH)

print("Successfully loaded llama.cpp bridge into Python.")

# -------------------------
# Define C function signatures
# -------------------------
llama_bridge = define_llama_bridge_signatures(llama_bridge)

# -------------------------
# Initialize llama.cpp
# -------------------------
llama_bridge = init_llama(llama_bridge)

# -------------------------
# Load model
# -------------------------

print("Loading model:")
print(MODEL_PATH)

model = llama_bridge.llama_bridge_model_load(
    str(MODEL_PATH).encode("utf-8")
)

if not model:
    raise RuntimeError("Failed to load model")

print("\nModel loaded successfully.")

# -------------------------
# Inspect model
# -------------------------

size_bytes = llama_bridge.llama_bridge_model_size(model)
parameter_count = llama_bridge.llama_bridge_model_n_params(model)

print(f"Model size: {size_bytes / 1024**2:.2f} MiB")
print(f"Parameters: {parameter_count:,}")

# -------------------------
# Tokenize prompt
# -------------------------

token_buffer = (ctypes.c_int * MAX_TOKENS)()

token_count = llama_bridge.llama_bridge_tokenize(
    model,
    prompt.encode("utf-8"),
    token_buffer,
    MAX_TOKENS
)

if token_count < 0:
    raise RuntimeError("Failed to tokenize prompt")

tokens = [token_buffer[i] for i in range(token_count)]

print(f"\nText: {prompt}")
print(f"Token count: {token_count}")
print(f"Token IDs: {tokens}")

# -------------------------
# Create inference context
# -------------------------

context_size = CONTEXT_SIZE

ctx = llama_bridge.llama_bridge_create_context(
    model,
    context_size,
)

if not ctx:
    raise RuntimeError("Failed to create llama context")

print("\nContext created.")

# -------------------------
# Run the transformer
# -------------------------

decode_result = llama_bridge.llama_bridge_decode_prompt(
    ctx,
    token_buffer,
    token_count,
)

if decode_result != 0:
    raise RuntimeError(
        f"llama_decode failed: {decode_result}"
    )

print("Prompt decoded by neural network.")

# -------------------------
# Generate multiple tokens
# -------------------------

generated_text = ""

for _ in range(MAX_NEW_TOKENS):

    # 1. Sample from the logits produced
    #    by the most recent llama_decode()
    next_token = llama_bridge.llama_bridge_sample_greedy(
        ctx
    )

    # 2. Stop if model generated an
    #    end-of-generation token
    if llama_bridge.llama_bridge_is_eog(
        model,
        next_token,
    ):
        break

    # 3. Convert token ID → text
    piece_buffer = ctypes.create_string_buffer(256)

    piece_length = llama_bridge.llama_bridge_token_to_piece(
        model,
        next_token,
        piece_buffer,
        len(piece_buffer),
    )

    if piece_length < 0:
        raise RuntimeError(
            f"Piece buffer too small: {-piece_length}"
        )

    piece = piece_buffer.raw[:piece_length].decode(
        "utf-8",
        errors="replace",
    )

    generated_text += piece

    print(
        piece,
        end="",
        flush=True,
    )

    # 4. Feed the generated token back
    #    through the model
    decode_result = llama_bridge.llama_bridge_decode_one(
        ctx,
        next_token,
    )

    if decode_result != 0:
        raise RuntimeError(
            f"llama_decode failed: {decode_result}"
        )

print("\n\nGeneration complete.")
print(f"Generated text: {generated_text!r}")

# -------------------------
# Cleanup
# -------------------------

input("\nPress Enter to unload the model...")

llama_bridge.llama_bridge_free_context(ctx)
llama_bridge.llama_bridge_model_free(model)
llama_bridge.llama_bridge_shutdown()

print("\nModel unloaded.")
