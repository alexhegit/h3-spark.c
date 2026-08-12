#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_tokenizer {
    char path[4096];
};

static void h3_tok_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
}

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json, char *error,
                                size_t error_size) {
    if (!tokenizer_json || !*tokenizer_json) {
        h3_tok_error(error, error_size, "tokenizer path is required");
        return NULL;
    }
    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        h3_tok_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    snprintf(tokenizer->path, sizeof(tokenizer->path), "%s", tokenizer_json);
    h3_tok_error(error, error_size,
                 "CUDA tokenizer port is not implemented yet; BPE encode/decode "
                 "pending Phase 1");
    free(tokenizer);
    return NULL;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    free(tokenizer);
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    (void)tokenizer;
    (void)utf8;
    (void)pad_empty;
    (void)ids;
    (void)count;
    h3_tok_error(error, error_size, "tokenizer encode is not implemented on CUDA yet");
    return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer, const uint32_t *ids,
                          size_t count, char *error, size_t error_size) {
    (void)tokenizer;
    (void)ids;
    (void)count;
    h3_tok_error(error, error_size, "tokenizer decode is not implemented on CUDA yet");
    return NULL;
}
