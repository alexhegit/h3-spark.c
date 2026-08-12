#include "h3_tokenizer.h"

#include "third_party/cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>

typedef struct {
    uint32_t value;
    size_t byte_offset;
    size_t byte_length;
} h3_codepoint;

typedef struct h3_map_node {
    char *key;
    uint32_t value;
    struct h3_map_node *next;
} h3_map_node;

typedef struct {
    h3_map_node **buckets;
    size_t nbuckets;
} h3_u32_map;

typedef struct h3_cache_node {
    char *key;
    uint32_t *ids;
    size_t count;
    struct h3_cache_node *next;
} h3_cache_node;

typedef struct {
    h3_cache_node **buckets;
    size_t nbuckets;
} h3_bpe_cache;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} h3_str_array;

typedef struct {
    uint32_t *items;
    size_t count;
    size_t capacity;
} h3_u32_array;

struct h3_tokenizer {
    h3_u32_map vocab;
    char **inverse_vocab;
    size_t inverse_vocab_size;
    h3_u32_map merge_ranks;
    h3_u32_map added_tokens;
    char **inverse_added;
    size_t inverse_added_size;
    char **added_alternatives;
    size_t added_alternatives_count;
    h3_bpe_cache bpe_cache;
    char *byte_encoder[256];
    int16_t byte_decoder[324];
};

static void h3_tok_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
}

static uint32_t h3_fnv1a(const char *text) {
    uint32_t hash = 2166136261u;
    for (; *text; text++) {
        hash ^= (unsigned char)*text;
        hash *= 16777619u;
    }
    return hash;
}

static char *h3_strdup(const char *text) {
    if (!text) return NULL;
    size_t length = strlen(text);
    char *copy = malloc(length + 1);
    if (!copy) return NULL;
    memcpy(copy, text, length + 1);
    return copy;
}

static int h3_map_init(h3_u32_map *map, size_t nbuckets) {
    map->buckets = calloc(nbuckets, sizeof(*map->buckets));
    if (!map->buckets) return 0;
    map->nbuckets = nbuckets;
    return 1;
}

static void h3_map_free(h3_u32_map *map) {
    if (!map->buckets) return;
    for (size_t index = 0; index < map->nbuckets; index++) {
        h3_map_node *node = map->buckets[index];
        while (node) {
            h3_map_node *next = node->next;
            free(node->key);
            free(node);
            node = next;
        }
    }
    free(map->buckets);
    map->buckets = NULL;
    map->nbuckets = 0;
}

static int h3_map_set(h3_u32_map *map, const char *key, uint32_t value) {
    uint32_t slot = h3_fnv1a(key) % (uint32_t)map->nbuckets;
    for (h3_map_node *node = map->buckets[slot]; node; node = node->next) {
        if (!strcmp(node->key, key)) {
            node->value = value;
            return 1;
        }
    }
    h3_map_node *node = malloc(sizeof(*node));
    if (!node) return 0;
    node->key = h3_strdup(key);
    if (!node->key) {
        free(node);
        return 0;
    }
    node->value = value;
    node->next = map->buckets[slot];
    map->buckets[slot] = node;
    return 1;
}

static int h3_map_get(const h3_u32_map *map, const char *key, uint32_t *value) {
    if (!map->buckets) return 0;
    uint32_t slot = h3_fnv1a(key) % (uint32_t)map->nbuckets;
    for (h3_map_node *node = map->buckets[slot]; node; node = node->next) {
        if (!strcmp(node->key, key)) {
            if (value) *value = node->value;
            return 1;
        }
    }
    return 0;
}

static void h3_cache_free(h3_bpe_cache *cache) {
    if (!cache->buckets) return;
    for (size_t index = 0; index < cache->nbuckets; index++) {
        h3_cache_node *node = cache->buckets[index];
        while (node) {
            h3_cache_node *next = node->next;
            free(node->key);
            free(node->ids);
            free(node);
            node = next;
        }
    }
    free(cache->buckets);
    cache->buckets = NULL;
    cache->nbuckets = 0;
}

static int h3_cache_init(h3_bpe_cache *cache, size_t nbuckets) {
    cache->buckets = calloc(nbuckets, sizeof(*cache->buckets));
    if (!cache->buckets) return 0;
    cache->nbuckets = nbuckets;
    return 1;
}

static int h3_cache_get(const h3_bpe_cache *cache, const char *key,
                        const uint32_t **ids, size_t *count) {
    if (!cache->buckets) return 0;
    uint32_t slot = h3_fnv1a(key) % (uint32_t)cache->nbuckets;
    for (h3_cache_node *node = cache->buckets[slot]; node; node = node->next) {
        if (!strcmp(node->key, key)) {
            if (ids) *ids = node->ids;
            if (count) *count = node->count;
            return 1;
        }
    }
    return 0;
}

static int h3_cache_set(h3_bpe_cache *cache, const char *key,
                        const uint32_t *ids, size_t count) {
    uint32_t slot = h3_fnv1a(key) % (uint32_t)cache->nbuckets;
    for (h3_cache_node *node = cache->buckets[slot]; node; node = node->next) {
        if (!strcmp(node->key, key)) {
            free(node->ids);
            node->ids = malloc(count * sizeof(*node->ids));
            if (!node->ids) return 0;
            memcpy(node->ids, ids, count * sizeof(*node->ids));
            node->count = count;
            return 1;
        }
    }
    h3_cache_node *node = malloc(sizeof(*node));
    if (!node) return 0;
    node->key = h3_strdup(key);
    node->ids = malloc(count * sizeof(*node->ids));
    if (!node->key || !node->ids) {
        free(node->key);
        free(node->ids);
        free(node);
        return 0;
    }
    memcpy(node->ids, ids, count * sizeof(*node->ids));
    node->count = count;
    node->next = cache->buckets[slot];
    cache->buckets[slot] = node;
    return 1;
}

static int h3_str_array_push(h3_str_array *array, char *item) {
    if (array->count == array->capacity) {
        size_t capacity = array->capacity ? array->capacity * 2 : 8;
        char **items = realloc(array->items, capacity * sizeof(*items));
        if (!items) return 0;
        array->items = items;
        array->capacity = capacity;
    }
    array->items[array->count++] = item;
    return 1;
}

static void h3_str_array_free(h3_str_array *array) {
    for (size_t index = 0; index < array->count; index++) free(array->items[index]);
    free(array->items);
    array->items = NULL;
    array->count = 0;
    array->capacity = 0;
}

static int h3_u32_array_push(h3_u32_array *array, uint32_t item) {
    if (array->count == array->capacity) {
        size_t capacity = array->capacity ? array->capacity * 2 : 8;
        uint32_t *items = realloc(array->items, capacity * sizeof(*items));
        if (!items) return 0;
        array->items = items;
        array->capacity = capacity;
    }
    array->items[array->count++] = item;
    return 1;
}

static void h3_u32_array_free(h3_u32_array *array) {
    free(array->items);
    array->items = NULL;
    array->count = 0;
    array->capacity = 0;
}

static int h3_utf8_decode(const char *text, size_t length, size_t offset,
                          uint32_t *value, size_t *units) {
    if (offset >= length) return 0;
    unsigned char first = (unsigned char)text[offset];
    if (first < 0x80) {
        *value = first;
        *units = 1;
        return 1;
    }
    if ((first & 0xe0) == 0xc0) {
        if (offset + 1 >= length) return 0;
        unsigned char second = (unsigned char)text[offset + 1];
        if ((second & 0xc0) != 0x80) return 0;
        *value = ((uint32_t)(first & 0x1f) << 6) | (second & 0x3f);
        if (*value < 0x80) return 0;
        *units = 2;
        return 1;
    }
    if ((first & 0xf0) == 0xe0) {
        if (offset + 2 >= length) return 0;
        unsigned char second = (unsigned char)text[offset + 1];
        unsigned char third = (unsigned char)text[offset + 2];
        if ((second & 0xc0) != 0x80 || (third & 0xc0) != 0x80) return 0;
        *value = ((uint32_t)(first & 0x0f) << 12) |
                 ((uint32_t)(second & 0x3f) << 6) | (third & 0x3f);
        if (*value < 0x800) return 0;
        *units = 3;
        return 1;
    }
    if ((first & 0xf8) == 0xf0) {
        if (offset + 3 >= length) return 0;
        unsigned char second = (unsigned char)text[offset + 1];
        unsigned char third = (unsigned char)text[offset + 2];
        unsigned char fourth = (unsigned char)text[offset + 3];
        if ((second & 0xc0) != 0x80 || (third & 0xc0) != 0x80 ||
            (fourth & 0xc0) != 0x80)
            return 0;
        *value = ((uint32_t)(first & 0x07) << 18) |
                 ((uint32_t)(second & 0x3f) << 12) |
                 ((uint32_t)(third & 0x3f) << 6) | (fourth & 0x3f);
        if (*value < 0x10000 || *value > 0x10ffff) return 0;
        *units = 4;
        return 1;
    }
    return 0;
}

static int h3_valid_utf8(const char *text) {
    size_t length = strlen(text);
    for (size_t offset = 0; offset < length;) {
        uint32_t value;
        size_t units;
        if (!h3_utf8_decode(text, length, offset, &value, &units)) return 0;
        offset += units;
    }
    return 1;
}

static char *h3_codepoint_string(uint32_t value) {
    char buffer[5];
    if (value <= 0x7f) {
        buffer[0] = (char)value;
        buffer[1] = '\0';
    } else if (value <= 0x7ff) {
        buffer[0] = (char)(0xc0 | (value >> 6));
        buffer[1] = (char)(0x80 | (value & 0x3f));
        buffer[2] = '\0';
    } else if (value <= 0xffff) {
        buffer[0] = (char)(0xe0 | (value >> 12));
        buffer[1] = (char)(0x80 | ((value >> 6) & 0x3f));
        buffer[2] = (char)(0x80 | (value & 0x3f));
        buffer[3] = '\0';
    } else {
        buffer[0] = (char)(0xf0 | (value >> 18));
        buffer[1] = (char)(0x80 | ((value >> 12) & 0x3f));
        buffer[2] = (char)(0x80 | ((value >> 6) & 0x3f));
        buffer[3] = (char)(0x80 | (value & 0x3f));
        buffer[4] = '\0';
    }
    return h3_strdup(buffer);
}

static h3_codepoint *h3_codepoints(const char *text, size_t *count) {
    size_t length = strlen(text);
    h3_codepoint *points = calloc(length ? length : 1, sizeof(*points));
    if (!points) return NULL;
    size_t used = 0;
    for (size_t offset = 0; offset < length;) {
        uint32_t value;
        size_t units;
        if (!h3_utf8_decode(text, length, offset, &value, &units)) {
            free(points);
            return NULL;
        }
        points[used++] = (h3_codepoint){value, offset, units};
        offset += units;
    }
    *count = used;
    return points;
}

static int h3_letter(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
           category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
           category == U_OTHER_LETTER;
}

static int h3_number(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
           category == U_OTHER_NUMBER;
}

static int h3_space(uint32_t value) {
    return u_isUWhiteSpace((UChar32)value) || (value >= 0x1c && value <= 0x1f);
}

static char *h3_slice(const char *text, const h3_codepoint *points,
                      size_t start, size_t stop) {
    size_t begin = points[start].byte_offset;
    h3_codepoint last = points[stop - 1];
    size_t end = last.byte_offset + last.byte_length;
    size_t length = end - begin;
    char *slice = malloc(length + 1);
    if (!slice) return NULL;
    memcpy(slice, text + begin, length);
    slice[length] = '\0';
    return slice;
}

static size_t h3_contraction(const h3_codepoint *points, size_t count,
                             size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (points[index].value != '\'') return 0;
    for (size_t item = 0; item < sizeof(values) / sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > count) continue;
        int matches = 1;
        for (size_t offset = 1; offset < length; offset++) {
            uint32_t got = points[index + offset].value;
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][offset]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

static char *h3_nfc_normalize(const char *input) {
    UErrorCode status = U_ZERO_ERROR;
    const UNormalizer2 *nfc = unorm2_getNFCInstance(&status);
    if (U_FAILURE(status)) return NULL;
    int32_t u16_len = 0;
    status = U_ZERO_ERROR;
    u_strFromUTF8(NULL, 0, &u16_len, input, -1, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) return NULL;
    UChar *u16 = malloc((size_t)(u16_len + 1) * sizeof(UChar));
    if (!u16) return NULL;
    status = U_ZERO_ERROR;
    u_strFromUTF8(u16, u16_len + 1, NULL, input, -1, &status);
    if (U_FAILURE(status)) {
        free(u16);
        return NULL;
    }
    status = U_ZERO_ERROR;
    int32_t norm_len = unorm2_normalize(nfc, u16, u16_len, NULL, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(u16);
        return NULL;
    }
    UChar *norm = malloc((size_t)(norm_len + 1) * sizeof(UChar));
    if (!norm) {
        free(u16);
        return NULL;
    }
    status = U_ZERO_ERROR;
    unorm2_normalize(nfc, u16, u16_len, norm, norm_len + 1, &status);
    free(u16);
    if (U_FAILURE(status)) {
        free(norm);
        return NULL;
    }
    int32_t utf8_len = 0;
    status = U_ZERO_ERROR;
    u_strToUTF8(NULL, 0, &utf8_len, norm, norm_len, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(norm);
        return NULL;
    }
    char *text = malloc((size_t)utf8_len + 1);
    if (!text) {
        free(norm);
        return NULL;
    }
    status = U_ZERO_ERROR;
    u_strToUTF8(text, utf8_len + 1, NULL, norm, norm_len, &status);
    free(norm);
    if (U_FAILURE(status)) {
        free(text);
        return NULL;
    }
    return text;
}

static h3_str_array *h3_pretokenize(const char *input) {
    char *text = h3_nfc_normalize(input);
    if (!text) return NULL;
    size_t count = 0;
    h3_codepoint *points = h3_codepoints(text, &count);
    if (!points) {
        free(text);
        return NULL;
    }
    h3_str_array *pieces = calloc(1, sizeof(*pieces));
    if (!pieces) {
        free(points);
        free(text);
        return NULL;
    }
    size_t index = 0;
    while (index < count) {
        size_t contraction = h3_contraction(points, count, index);
        if (contraction) {
            char *piece = h3_slice(text, points, index, index + contraction);
            if (!piece || !h3_str_array_push(pieces, piece)) goto fail;
            index += contraction;
            continue;
        }
        uint32_t value = points[index].value;
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (h3_letter(value)) {
            /* Already at the first letter. */
        } else if (value != '\r' && value != '\n' && !h3_number(value) &&
                   index + 1 < count && h3_letter(points[index + 1].value)) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < count && h3_letter(points[stop].value)) stop++;
            char *piece = h3_slice(text, points, index, stop);
            if (!piece || !h3_str_array_push(pieces, piece)) goto fail;
            index = stop;
            continue;
        }
        if (h3_number(value)) {
            char *piece = h3_slice(text, points, index, index + 1);
            if (!piece || !h3_str_array_push(pieces, piece)) goto fail;
            index++;
            continue;
        }
        size_t punct_start = index +
            (value == ' ' && index + 1 < count &&
             !h3_space(points[index + 1].value) &&
             !h3_letter(points[index + 1].value) &&
             !h3_number(points[index + 1].value));
        size_t stop = punct_start;
        while (stop < count && !h3_space(points[stop].value) &&
               !h3_letter(points[stop].value) &&
               !h3_number(points[stop].value))
            stop++;
        if (stop > punct_start) {
            while (stop < count &&
                   (points[stop].value == '\r' || points[stop].value == '\n'))
                stop++;
            char *piece = h3_slice(text, points, index, stop);
            if (!piece || !h3_str_array_push(pieces, piece)) goto fail;
            index = stop;
            continue;
        }
        if (h3_space(value)) {
            size_t whitespace_end = index + 1;
            while (whitespace_end < count &&
                   h3_space(points[whitespace_end].value))
                whitespace_end++;
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < whitespace_end; cursor++) {
                if (points[cursor].value == '\r' || points[cursor].value == '\n')
                    newline_end = (ptrdiff_t)cursor + 1;
            }
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (whitespace_end == count) piece_end = whitespace_end;
            else if (whitespace_end - index > 1) piece_end = whitespace_end - 1;
            else piece_end = index + 1;
            char *piece = h3_slice(text, points, index, piece_end);
            if (!piece || !h3_str_array_push(pieces, piece)) goto fail;
            index = piece_end;
            continue;
        }
        goto fail;
    }
    free(points);
    free(text);
    return pieces;
fail:
    h3_str_array_free(pieces);
    free(pieces);
    free(points);
    free(text);
    return NULL;
}

static char *h3_pair_key(const char *left, const char *right) {
    static const char separator[] = "\xEF\xBF\xBF"; /* U+FFFF */
    size_t left_len = strlen(left);
    size_t right_len = strlen(right);
    char *key = malloc(left_len + sizeof(separator) - 1 + right_len + 1);
    if (!key) return NULL;
    memcpy(key, left, left_len);
    memcpy(key + left_len, separator, sizeof(separator) - 1);
    memcpy(key + left_len + sizeof(separator) - 1, right, right_len + 1);
    return key;
}

static char *h3_encode_bytes(const h3_tokenizer *tokenizer, const char *piece) {
    size_t length = strlen(piece);
    size_t capacity = length * 4 + 1;
    char *encoded = malloc(capacity);
    if (!encoded) return NULL;
    encoded[0] = '\0';
    size_t used = 0;
    for (size_t index = 0; index < length; index++) {
        const char *symbol = tokenizer->byte_encoder[(unsigned char)piece[index]];
        size_t symbol_len = strlen(symbol);
        if (used + symbol_len + 1 > capacity) {
            capacity = (used + symbol_len + 1) * 2;
            char *grown = realloc(encoded, capacity);
            if (!grown) {
                free(encoded);
                return NULL;
            }
            encoded = grown;
        }
        memcpy(encoded + used, symbol, symbol_len);
        used += symbol_len;
        encoded[used] = '\0';
    }
    return encoded;
}

static int h3_split_codepoints(const char *text, h3_str_array *symbols) {
    size_t length = strlen(text);
    for (size_t offset = 0; offset < length;) {
        uint32_t value;
        size_t units;
        if (!h3_utf8_decode(text, length, offset, &value, &units)) return 0;
        char *symbol = malloc(units + 1);
        if (!symbol) return 0;
        memcpy(symbol, text + offset, units);
        symbol[units] = '\0';
        if (!h3_str_array_push(symbols, symbol)) {
            free(symbol);
            return 0;
        }
        offset += units;
    }
    return 1;
}

static int h3_bpe(const h3_tokenizer *tokenizer, const char *piece,
                   h3_u32_array *output, char *error, size_t error_size) {
    char *encoded = h3_encode_bytes(tokenizer, piece);
    if (!encoded) {
        h3_tok_error(error, error_size, "out of memory during BPE");
        return 0;
    }
    const uint32_t *cached = NULL;
    size_t cached_count = 0;
    if (h3_cache_get(&tokenizer->bpe_cache, encoded, &cached, &cached_count)) {
        for (size_t index = 0; index < cached_count; index++) {
            if (!h3_u32_array_push(output, cached[index])) {
                free(encoded);
                h3_tok_error(error, error_size, "out of memory during BPE");
                return 0;
            }
        }
        free(encoded);
        return 1;
    }
    h3_str_array symbols = {0};
    if (!h3_split_codepoints(encoded, &symbols)) {
        free(encoded);
        h3_str_array_free(&symbols);
        h3_tok_error(error, error_size, "invalid BPE byte encoding");
        return 0;
    }
    while (symbols.count > 1) {
        uint32_t best_rank = UINT32_MAX;
        size_t best = 0;
        int found = 0;
        for (size_t index = 0; index + 1 < symbols.count; index++) {
            char *key = h3_pair_key(symbols.items[index], symbols.items[index + 1]);
            if (!key) {
                free(encoded);
                h3_str_array_free(&symbols);
                h3_tok_error(error, error_size, "out of memory during BPE");
                return 0;
            }
            uint32_t rank;
            if (h3_map_get(&tokenizer->merge_ranks, key, &rank) &&
                (!found || rank < best_rank)) {
                best_rank = rank;
                best = index;
                found = 1;
            }
            free(key);
        }
        if (!found) break;
        char *left = symbols.items[best];
        char *right = symbols.items[best + 1];
        size_t merged_len = strlen(left) + strlen(right);
        char *merged = malloc(merged_len + 1);
        if (!merged) {
            free(encoded);
            h3_str_array_free(&symbols);
            h3_tok_error(error, error_size, "out of memory during BPE");
            return 0;
        }
        memcpy(merged, left, strlen(left));
        memcpy(merged + strlen(left), right, strlen(right) + 1);
        h3_str_array next = {0};
        for (size_t index = 0; index < symbols.count;) {
            if (index + 1 < symbols.count && !strcmp(symbols.items[index], left) &&
                !strcmp(symbols.items[index + 1], right)) {
                if (!h3_str_array_push(&next, h3_strdup(merged))) {
                    free(merged);
                    free(encoded);
                    h3_str_array_free(&symbols);
                    h3_str_array_free(&next);
                    h3_tok_error(error, error_size, "out of memory during BPE");
                    return 0;
                }
                index += 2;
            } else {
                if (!h3_str_array_push(&next, h3_strdup(symbols.items[index]))) {
                    free(merged);
                    free(encoded);
                    h3_str_array_free(&symbols);
                    h3_str_array_free(&next);
                    h3_tok_error(error, error_size, "out of memory during BPE");
                    return 0;
                }
                index++;
            }
        }
        free(merged);
        h3_str_array_free(&symbols);
        symbols = next;
    }
    h3_u32_array ids = {0};
    for (size_t index = 0; index < symbols.count; index++) {
        uint32_t token_id;
        if (!h3_map_get(&tokenizer->vocab, symbols.items[index], &token_id)) {
            h3_str_array_free(&symbols);
            free(encoded);
            h3_u32_array_free(&ids);
            if (error && error_size)
                snprintf(error, error_size,
                         "BPE symbol is absent from vocabulary: %s",
                         symbols.items[index]);
            return 0;
        }
        if (!h3_u32_array_push(&ids, token_id)) {
            h3_str_array_free(&symbols);
            free(encoded);
            h3_u32_array_free(&ids);
            h3_tok_error(error, error_size, "out of memory during BPE");
            return 0;
        }
    }
    h3_str_array_free(&symbols);
    h3_tokenizer *mutable = (h3_tokenizer *)tokenizer;
    if (!h3_cache_set(&mutable->bpe_cache, encoded, ids.items, ids.count)) {
        free(encoded);
        h3_u32_array_free(&ids);
        h3_tok_error(error, error_size, "out of memory during BPE");
        return 0;
    }
    for (size_t index = 0; index < ids.count; index++) {
        if (!h3_u32_array_push(output, ids.items[index])) {
            free(encoded);
            h3_u32_array_free(&ids);
            h3_tok_error(error, error_size, "out of memory during BPE");
            return 0;
        }
    }
    h3_u32_array_free(&ids);
    free(encoded);
    return 1;
}

static int h3_encode_plain(const h3_tokenizer *tokenizer, const char *text,
                           h3_u32_array *output, char *error, size_t error_size) {
    h3_str_array *pieces = h3_pretokenize(text);
    if (!pieces) {
        h3_tok_error(error, error_size, "unable to pre-tokenize input");
        return 0;
    }
    for (size_t index = 0; index < pieces->count; index++) {
        if (!h3_bpe(tokenizer, pieces->items[index], output, error, error_size)) {
            h3_str_array_free(pieces);
            free(pieces);
            return 0;
        }
    }
    h3_str_array_free(pieces);
    free(pieces);
    return 1;
}

static int h3_added_compare(const void *left_raw, const void *right_raw) {
    const char *const *left = left_raw;
    const char *const *right = right_raw;
    size_t left_len = strlen(*left);
    size_t right_len = strlen(*right);
    if (left_len > right_len) return -1;
    if (left_len < right_len) return 1;
    return strcmp(*left, *right);
}

static int h3_added_match(const h3_tokenizer *tokenizer, const char *text,
                          size_t start, size_t *match_start, size_t *match_len,
                          const char **token) {
    size_t text_len = strlen(text);
    int found = 0;
    for (size_t index = 0; index < tokenizer->added_alternatives_count; index++) {
        const char *candidate = tokenizer->added_alternatives[index];
        size_t candidate_len = strlen(candidate);
        if (candidate_len == 0 || start + candidate_len > text_len) continue;
        const char *region = text + start;
        size_t region_len = text_len - start;
        for (size_t offset = 0; offset + candidate_len <= region_len; offset++) {
            if (memcmp(region + offset, candidate, candidate_len) != 0) continue;
            size_t location = start + offset;
            if (!found || location < *match_start ||
                (location == *match_start && candidate_len > *match_len)) {
                *match_start = location;
                *match_len = candidate_len;
                *token = candidate;
                found = 1;
            }
        }
    }
    return found;
}

static char *h3_read_file(const char *path, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        if (error && error_size)
            snprintf(error, error_size, "cannot read tokenizer: %s", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        h3_tok_error(error, error_size, "cannot read tokenizer file");
        return NULL;
    }
    long size = ftell(file);
    if (size < 0) {
        fclose(file);
        h3_tok_error(error, error_size, "cannot read tokenizer file");
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        h3_tok_error(error, error_size, "cannot read tokenizer file");
        return NULL;
    }
    char *data = malloc((size_t)size + 1);
    if (!data) {
        fclose(file);
        h3_tok_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    if (fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        fclose(file);
        h3_tok_error(error, error_size, "cannot read tokenizer file");
        return NULL;
    }
    fclose(file);
    data[size] = '\0';
    return data;
}

static void h3_tokenizer_destroy(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    h3_map_free(&tokenizer->vocab);
    h3_map_free(&tokenizer->merge_ranks);
    h3_map_free(&tokenizer->added_tokens);
    h3_cache_free(&tokenizer->bpe_cache);
    for (size_t index = 0; index < tokenizer->inverse_vocab_size; index++)
        free(tokenizer->inverse_vocab[index]);
    free(tokenizer->inverse_vocab);
    for (size_t index = 0; index < tokenizer->inverse_added_size; index++)
        free(tokenizer->inverse_added[index]);
    free(tokenizer->inverse_added);
    for (size_t index = 0; index < tokenizer->added_alternatives_count; index++)
        free(tokenizer->added_alternatives[index]);
    free(tokenizer->added_alternatives);
    for (size_t index = 0; index < 256; index++) free(tokenizer->byte_encoder[index]);
    free(tokenizer);
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        h3_tok_error(error, error_size, "tokenizer path is required");
        return NULL;
    }
    char *data = h3_read_file(path, error, error_size);
    if (!data) return NULL;
    cJSON *config = cJSON_Parse(data);
    free(data);
    if (!config) {
        h3_tok_error(error, error_size, "invalid tokenizer JSON");
        return NULL;
    }
    cJSON *model = cJSON_GetObjectItemCaseSensitive(config, "model");
    cJSON *normalizer = cJSON_GetObjectItemCaseSensitive(config, "normalizer");
    cJSON *model_type = model ? cJSON_GetObjectItemCaseSensitive(model, "type") : NULL;
    cJSON *unk_token = model ? cJSON_GetObjectItemCaseSensitive(model, "unk_token") : NULL;
    cJSON *normalizer_type =
        normalizer ? cJSON_GetObjectItemCaseSensitive(normalizer, "type") : NULL;
    cJSON *vocab_json = model ? cJSON_GetObjectItemCaseSensitive(model, "vocab") : NULL;
    cJSON *merges_json = model ? cJSON_GetObjectItemCaseSensitive(model, "merges") : NULL;
    if (!cJSON_IsString(model_type) || strcmp(model_type->valuestring, "BPE") != 0 ||
        !cJSON_IsNull(unk_token) ||
        !cJSON_IsString(normalizer_type) ||
        strcmp(normalizer_type->valuestring, "NFC") != 0 ||
        !cJSON_IsObject(vocab_json) || !cJSON_IsArray(merges_json)) {
        cJSON_Delete(config);
        h3_tok_error(error, error_size, "unexpected tokenizer specification");
        return NULL;
    }
    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        cJSON_Delete(config);
        h3_tok_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    if (!h3_map_init(&tokenizer->vocab, 8192) ||
        !h3_map_init(&tokenizer->merge_ranks, 8192) ||
        !h3_map_init(&tokenizer->added_tokens, 256) ||
        !h3_cache_init(&tokenizer->bpe_cache, 256)) {
        cJSON_Delete(config);
        h3_tokenizer_destroy(tokenizer);
        h3_tok_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    size_t maximum_id = 0;
    for (cJSON *entry = vocab_json->child; entry; entry = entry->next) {
        if (!cJSON_IsNumber(entry)) continue;
        uint32_t token_id = (uint32_t)entry->valuedouble;
        if (!h3_map_set(&tokenizer->vocab, entry->string, token_id)) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
        if (token_id > maximum_id) maximum_id = token_id;
    }
    cJSON *added = cJSON_GetObjectItemCaseSensitive(config, "added_tokens");
    if (added) {
        for (cJSON *token = added->child; token; token = token->next) {
            cJSON *identifier = cJSON_GetObjectItemCaseSensitive(token, "id");
            if (cJSON_IsNumber(identifier)) {
                uint32_t token_id = (uint32_t)identifier->valuedouble;
                if (token_id > maximum_id) maximum_id = token_id;
            }
        }
    }
    tokenizer->inverse_vocab_size = maximum_id + 1;
    tokenizer->inverse_vocab = calloc(tokenizer->inverse_vocab_size, sizeof(char *));
    tokenizer->inverse_added_size = maximum_id + 1;
    tokenizer->inverse_added = calloc(tokenizer->inverse_added_size, sizeof(char *));
    if (!tokenizer->inverse_vocab || !tokenizer->inverse_added) {
        cJSON_Delete(config);
        h3_tokenizer_destroy(tokenizer);
        h3_tok_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    for (cJSON *entry = vocab_json->child; entry; entry = entry->next) {
        if (!cJSON_IsNumber(entry)) continue;
        uint32_t token_id = (uint32_t)entry->valuedouble;
        free(tokenizer->inverse_vocab[token_id]);
        tokenizer->inverse_vocab[token_id] = h3_strdup(entry->string);
        if (!tokenizer->inverse_vocab[token_id]) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
    }
    uint32_t rank = 0;
    for (cJSON *entry = merges_json->child; entry; entry = entry->next) {
        const char *left = NULL;
        const char *right = NULL;
        char *owned = NULL;
        if (cJSON_IsString(entry)) {
            const char *space = strchr(entry->valuestring, ' ');
            if (!space) {
                cJSON_Delete(config);
                h3_tokenizer_destroy(tokenizer);
                h3_tok_error(error, error_size, "invalid tokenizer merge");
                return NULL;
            }
            size_t left_len = (size_t)(space - entry->valuestring);
            owned = malloc(left_len + strlen(space + 1) + 2);
            if (!owned) {
                cJSON_Delete(config);
                h3_tokenizer_destroy(tokenizer);
                h3_tok_error(error, error_size, "out of memory loading tokenizer");
                return NULL;
            }
            memcpy(owned, entry->valuestring, left_len);
            owned[left_len] = '\0';
            left = owned;
            right = space + 1;
        } else if (cJSON_IsArray(entry) && cJSON_GetArraySize(entry) == 2) {
            cJSON *left_json = cJSON_GetArrayItem(entry, 0);
            cJSON *right_json = cJSON_GetArrayItem(entry, 1);
            if (!cJSON_IsString(left_json) || !cJSON_IsString(right_json)) {
                cJSON_Delete(config);
                h3_tokenizer_destroy(tokenizer);
                h3_tok_error(error, error_size, "invalid tokenizer merge");
                return NULL;
            }
            left = left_json->valuestring;
            right = right_json->valuestring;
        } else {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "invalid tokenizer merge");
            return NULL;
        }
        char *key = h3_pair_key(left, right);
        free(owned);
        if (!key) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
        if (!h3_map_set(&tokenizer->merge_ranks, key, rank++)) {
            free(key);
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
        free(key);
    }
    h3_str_array alternatives = {0};
    for (cJSON *token = added ? added->child : NULL; token; token = token->next) {
        cJSON *single_word = cJSON_GetObjectItemCaseSensitive(token, "single_word");
        cJSON *lstrip = cJSON_GetObjectItemCaseSensitive(token, "lstrip");
        cJSON *rstrip = cJSON_GetObjectItemCaseSensitive(token, "rstrip");
        cJSON *normalized = cJSON_GetObjectItemCaseSensitive(token, "normalized");
        if (cJSON_IsTrue(single_word) || cJSON_IsTrue(lstrip) || cJSON_IsTrue(rstrip) ||
            cJSON_IsTrue(normalized)) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "unsupported added-token policy");
            return NULL;
        }
        cJSON *content = cJSON_GetObjectItemCaseSensitive(token, "content");
        cJSON *identifier = cJSON_GetObjectItemCaseSensitive(token, "id");
        if (!cJSON_IsString(content) || !cJSON_IsNumber(identifier)) continue;
        uint32_t token_id = (uint32_t)identifier->valuedouble;
        if (!h3_map_set(&tokenizer->added_tokens, content->valuestring, token_id)) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
        free(tokenizer->inverse_added[token_id]);
        tokenizer->inverse_added[token_id] = h3_strdup(content->valuestring);
        if (!tokenizer->inverse_added[token_id]) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
        if (!h3_str_array_push(&alternatives, h3_strdup(content->valuestring))) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
    }
    qsort(alternatives.items, alternatives.count, sizeof(char *), h3_added_compare);
    tokenizer->added_alternatives = alternatives.items;
    tokenizer->added_alternatives_count = alternatives.count;
    for (size_t index = 0; index < 324; index++) tokenizer->byte_decoder[index] = -1;
    unsigned extra = 0;
    for (unsigned byte = 0; byte < 256; byte++) {
        int visible = (byte >= '!' && byte <= '~') ||
                      (byte >= 0xa1 && byte <= 0xac) ||
                      (byte >= 0xae && byte <= 0xff);
        uint32_t codepoint = visible ? byte : 256u + extra++;
        tokenizer->byte_encoder[byte] = h3_codepoint_string(codepoint);
        if (!tokenizer->byte_encoder[byte]) {
            cJSON_Delete(config);
            h3_tokenizer_destroy(tokenizer);
            h3_tok_error(error, error_size, "out of memory loading tokenizer");
            return NULL;
        }
        tokenizer->byte_decoder[codepoint] = (int16_t)byte;
    }
    cJSON_Delete(config);
    return tokenizer;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    h3_tokenizer_destroy(tokenizer);
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || !utf8 || !ids || !count) return 0;
    *ids = NULL;
    *count = 0;
    if (!h3_valid_utf8(utf8)) {
        h3_tok_error(error, error_size, "prompt is not valid UTF-8");
        return 0;
    }
    h3_u32_array output = {0};
    size_t start = 0;
    size_t text_len = strlen(utf8);
    while (start < text_len) {
        size_t match_start = 0;
        size_t match_len = 0;
        const char *added = NULL;
        if (!h3_added_match(tokenizer, utf8, start, &match_start, &match_len, &added))
            break;
        if (match_start > start) {
            char *plain = malloc(match_start - start + 1);
            if (!plain) {
                h3_u32_array_free(&output);
                h3_tok_error(error, error_size, "out of memory encoding prompt");
                return 0;
            }
            memcpy(plain, utf8 + start, match_start - start);
            plain[match_start - start] = '\0';
            int ok = h3_encode_plain(tokenizer, plain, &output, error, error_size);
            free(plain);
            if (!ok) {
                h3_u32_array_free(&output);
                return 0;
            }
        }
        uint32_t token_id;
        if (!h3_map_get(&tokenizer->added_tokens, added, &token_id)) {
            h3_u32_array_free(&output);
            h3_tok_error(error, error_size, "tokenizer failure");
            return 0;
        }
        if (!h3_u32_array_push(&output, token_id)) {
            h3_u32_array_free(&output);
            h3_tok_error(error, error_size, "out of memory encoding prompt");
            return 0;
        }
        start = match_start + match_len;
    }
    if (start < text_len &&
        !h3_encode_plain(tokenizer, utf8 + start, &output, error, error_size)) {
        h3_u32_array_free(&output);
        return 0;
    }
    if (output.count == 0 && pad_empty) {
        if (!h3_u32_array_push(&output, H3_PAD_TOKEN_ID)) {
            h3_u32_array_free(&output);
            h3_tok_error(error, error_size, "out of memory encoding prompt");
            return 0;
        }
    }
    if (output.count) {
        *ids = malloc(output.count * sizeof(**ids));
        if (!*ids) {
            h3_u32_array_free(&output);
            h3_tok_error(error, error_size, "out of memory encoding prompt");
            return 0;
        }
        memcpy(*ids, output.items, output.count * sizeof(**ids));
    }
    *count = output.count;
    h3_u32_array_free(&output);
    return 1;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer, const uint32_t *ids,
                          size_t count, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || (!ids && count)) return NULL;
    char *result = calloc(1, 1);
    unsigned char *bytes = NULL;
    size_t bytes_len = 0;
    size_t bytes_cap = 0;
    if (!result) {
        h3_tok_error(error, error_size, "out of memory decoding tokens");
        return NULL;
    }
    for (size_t index = 0; index < count; index++) {
        uint32_t identifier = ids[index];
        if (identifier >= tokenizer->inverse_vocab_size) {
            free(result);
            free(bytes);
            h3_tok_error(error, error_size, "token ID is out of range");
            return NULL;
        }
        const char *added = tokenizer->inverse_added[identifier];
        if (added) {
            if (bytes_len) {
                char *part = malloc(bytes_len + 1);
                if (!part) goto oom;
                memcpy(part, bytes, bytes_len);
                part[bytes_len] = '\0';
                size_t result_len = strlen(result);
                char *grown = realloc(result, result_len + bytes_len + 1);
                if (!grown) {
                    free(part);
                    goto oom;
                }
                result = grown;
                memcpy(result + result_len, part, bytes_len + 1);
                free(part);
                bytes_len = 0;
            }
            size_t result_len = strlen(result);
            size_t added_len = strlen(added);
            char *grown = realloc(result, result_len + added_len + 1);
            if (!grown) goto oom;
            result = grown;
            memcpy(result + result_len, added, added_len + 1);
            continue;
        }
        const char *symbol = tokenizer->inverse_vocab[identifier];
        if (!symbol) {
            free(result);
            free(bytes);
            h3_tok_error(error, error_size, "unknown token ID");
            return NULL;
        }
        size_t length = strlen(symbol);
        for (size_t offset = 0; offset < length;) {
            uint32_t codepoint;
            size_t units;
            if (!h3_utf8_decode(symbol, length, offset, &codepoint, &units)) {
                free(result);
                free(bytes);
                h3_tok_error(error, error_size, "invalid byte-level token");
                return NULL;
            }
            if (codepoint >= 324 || tokenizer->byte_decoder[codepoint] < 0) {
                free(result);
                free(bytes);
                h3_tok_error(error, error_size, "invalid byte-level token");
                return NULL;
            }
            unsigned char byte = (unsigned char)tokenizer->byte_decoder[codepoint];
            if (bytes_len + 1 > bytes_cap) {
                size_t capacity = bytes_cap ? bytes_cap * 2 : 32;
                unsigned char *grown = realloc(bytes, capacity);
                if (!grown) goto oom;
                bytes = grown;
                bytes_cap = capacity;
            }
            bytes[bytes_len++] = byte;
            offset += units;
        }
    }
    if (bytes_len) {
        char *part = malloc(bytes_len + 1);
        if (!part) goto oom;
        memcpy(part, bytes, bytes_len);
        part[bytes_len] = '\0';
        char *validated = part;
        if (!h3_valid_utf8(part)) {
            validated = h3_strdup("\xef\xbf\xbd");
            free(part);
            if (!validated) goto oom;
        }
        size_t result_len = strlen(result);
        size_t part_len = strlen(validated);
        char *grown = realloc(result, result_len + part_len + 1);
        if (!grown) {
            free(validated);
            goto oom;
        }
        result = grown;
        memcpy(result + result_len, validated, part_len + 1);
        free(validated);
    }
    free(bytes);
    return result;
oom:
    free(result);
    free(bytes);
    h3_tok_error(error, error_size, "out of memory decoding tokens");
    return NULL;
}
