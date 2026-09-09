#include <llama.h>
#include <cstring>

extern "C" {
    void llama_bridge_init() {
        llama_backend_init();
    }

    llama_model * llama_bridge_model_load(const char * path) {
        llama_model_params params = llama_model_default_params();

        // Negative means: offload as many layers as possible to GPU.
        params.n_gpu_layers = -1;

        return llama_model_load_from_file(path, params);
    }

    void llama_bridge_model_free(llama_model * model) {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }

    unsigned long long llama_bridge_model_size(llama_model * model) {
        return llama_model_size(model);
    }
    
    unsigned long long llama_bridge_model_n_params(llama_model * model) {
        return llama_model_n_params(model);
    }

    void llama_bridge_shutdown() {
        llama_backend_free();
    }

    int llama_bridge_tokenize(
        llama_model * model,
        const char * prompt_text,
        int * tokens,
        int max_tokens
    ) {
        const llama_vocab * vocab = llama_model_get_vocab(model);

        int text_len = static_cast<int>(strlen(prompt_text));

        return llama_tokenize(
            vocab,
            prompt_text,
            text_len,
            tokens,
            max_tokens,
            true,
            true
        );
    }
}