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

    llama_context * llama_bridge_create_context(
        llama_model * model,
        int context_size
    ) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = context_size;
        params.n_batch = context_size;

        return llama_init_from_model(model, params);
    }

    void llama_bridge_free_context(
        llama_context * context
    ) {
        if (context != nullptr) {
            llama_free(context);
        }
    }

    int llama_bridge_decode_prompt(
        llama_context * ctx,
        int * tokens,
        int token_count
    ) {
        llama_batch batch =
            llama_batch_get_one(
                tokens,
                token_count
            );
    
        return llama_decode(
            ctx,
            batch
        );
    }

    int llama_bridge_sample_greedy(
        llama_context * ctx
    ) {
        llama_sampler_chain_params params =
            llama_sampler_chain_default_params();
    
        llama_sampler * sampler =
            llama_sampler_chain_init(params);
    
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_greedy()
        );
    
        llama_token token =
            llama_sampler_sample(
                sampler,
                ctx,
                -1
            );
    
        llama_sampler_free(sampler);
    
        return token;
    }

    int llama_bridge_token_to_piece(
        llama_model * model,
        int token,
        char * buffer,
        int buffer_size
    ) {
        const llama_vocab * vocab =
            llama_model_get_vocab(model);
    
        return llama_token_to_piece(
            vocab,
            token,
            buffer,
            buffer_size,
            0,
            true
        );
    }
    
}