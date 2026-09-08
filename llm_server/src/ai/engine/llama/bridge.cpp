#include <llama.h>

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
}