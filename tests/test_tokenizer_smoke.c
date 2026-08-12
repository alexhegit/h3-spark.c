#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void check_roundtrip(h3_tokenizer *tokenizer, const char *text) {
    char error[256];
    uint32_t *ids = NULL;
    size_t count = 0;
    CHECK(h3_tokenizer_encode(tokenizer, text, 1, &ids, &count, error,
                              sizeof(error)));
    CHECK(count > 0);
    char *decoded = h3_tokenizer_decode(tokenizer, ids, count, error, sizeof(error));
    CHECK(decoded != NULL);
    CHECK(!strcmp(decoded, text));
    free(decoded);
    h3_tokenizer_ids_free(ids);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
                                  "MiniMax-H3/tokenizer/tokenizer.json";
    FILE *probe = fopen(path, "rb");
    if (!probe) {
        fprintf(stderr, "skip: tokenizer JSON not found at %s\n", path);
        return 0;
    }
    fclose(probe);

    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "skip: cannot load tokenizer at %s: %s\n", path, error);
        return 0;
    }

    check_roundtrip(tokenizer, "hello");

    uint32_t *ids = NULL;
    size_t count = 99;
    CHECK(h3_tokenizer_encode(tokenizer, "", 0, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == 0 && ids == NULL);
    CHECK(h3_tokenizer_encode(tokenizer, "", 1, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == 1 && ids[0] == H3_PAD_TOKEN_ID);
    h3_tokenizer_ids_free(ids);

    h3_tokenizer_free(tokenizer);
    printf("ok: %d tokenizer smoke checks\n", checks);
    return 0;
}
