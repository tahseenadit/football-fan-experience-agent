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


class localLLM:
    def __init__(self):
        print(f"llama.cpp library: {LLAMA_LIBRARY_PATH}")
        self.llama_lib = ctypes.CDLL(LLAMA_LIBRARY_PATH)
        print("Successfully loaded llama.cpp into Python.")

        self.llama_bridge = ctypes.CDLL(LLAMA_BRIDGE_PATH)
        print("Successfully loaded llama.cpp bridge into Python.")

        # -------------------------
        # Define C function signatures
        # -------------------------
        self.llama_bridge = define_llama_bridge_signatures(self.llama_bridge)

        # -------------------------
        # Initialize llama.cpp
        # -------------------------
        self.llama_bridge = init_llama(self.llama_bridge)

        # -------------------------
        # Load model
        # -------------------------

        print("Loading model:")
        print(MODEL_PATH)

        self.model = self.llama_bridge.llama_bridge_model_load(
            str(MODEL_PATH).encode("utf-8")
        )

        if not self.model:
            raise RuntimeError("Failed to load model")

        print("\nModel loaded successfully.")

        # -------------------------
        # Inspect model
        # -------------------------

        size_bytes = self.llama_bridge.llama_bridge_model_size(self.model)
        parameter_count = self.llama_bridge.llama_bridge_model_n_params(self.model)

        print(f"Model size: {size_bytes / 1024**2:.2f} MiB")
        print(f"Parameters: {parameter_count:,}")

    def generate(self, prompt: str) -> str:
        # -------------------------
        # Tokenize prompt
        # -------------------------

        # First attempt with no output buffer.
        #
        # llama_tokenize() will return:
        #
        #   -N
        #
        # where N is the number of tokens required.

        required = self.llama_bridge.llama_bridge_tokenize(
            self.model,
            prompt.encode("utf-8"),
            None,
            0,
        )

        if required >= 0:
            raise RuntimeError(
                f"Unexpected tokenization result: {required}"
            )

        required = -required

        print(f"Required token capacity: {required}")


        # Now allocate exactly enough memory.

        token_buffer = (ctypes.c_int * required)()


        token_count = self.llama_bridge.llama_bridge_tokenize(
            self.model,
            prompt.encode("utf-8"),
            token_buffer,
            required,
        )

        if token_count < 0:
            raise RuntimeError(
                f"Tokenization failed. Required {-token_count} tokens."
            )

        # -------------------------
        # Create inference context
        # -------------------------

        context_size = CONTEXT_SIZE

        required_context_size = (
            token_count
            + MAX_NEW_TOKENS
        )

        if required_context_size > CONTEXT_SIZE:
            raise RuntimeError(
                f"Context too small. "
                f"Prompt={token_count}, "
                f"generation={MAX_NEW_TOKENS}, "
                f"required={required_context_size}, "
                f"configured={CONTEXT_SIZE}"
            )

        ctx = self.llama_bridge.llama_bridge_create_context(
            self.model,
            context_size,
        )

        if not ctx:
            raise RuntimeError("Failed to create llama context")

        print("\nContext created.")

        # -------------------------
        # Run the transformer
        # -------------------------

        decode_result = self.llama_bridge.llama_bridge_decode_prompt(
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
            next_token = self.llama_bridge.llama_bridge_sample_greedy(
                ctx
            )

            # 2. Stop if model generated an
            #    end-of-generation token
            if self.llama_bridge.llama_bridge_is_eog(
                self.model,
                next_token,
            ):
                break

            # 3. Convert token ID → text
            piece_buffer = ctypes.create_string_buffer(256)

            piece_length = self.llama_bridge.llama_bridge_token_to_piece(
                self.model,
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
            decode_result = self.llama_bridge.llama_bridge_decode_one(
                ctx,
                next_token,
            )

            if decode_result != 0:
                raise RuntimeError(
                    f"llama_decode failed: {decode_result}"
                )

        print("\n\nGeneration complete.")
        print(f"Generated text: {generated_text!r}")

        self.llama_bridge.llama_bridge_free_context(ctx)
        return generated_text
    
    def close(self):
        # -------------------------
        # Cleanup
        # -------------------------
        self.llama_bridge.llama_bridge_model_free(self.model)
        self.llama_bridge.llama_bridge_shutdown()

        print("\nModel unloaded.")