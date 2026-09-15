#pragma once

#include "hash_types.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ETHASH_EPOCH_LENGTH 7500
#define ETHASH_LIGHT_CACHE_ITEM_SIZE 64
#define ETHASH_FULL_DATASET_ITEM_SIZE 128
#define ETHASH_NUM_DATASET_ACCESSES 64

struct ethash_epoch_context {
    const int epoch_number;
    const int light_cache_num_items;
    const union ethash_hash512* const light_cache;
    const uint32_t* const l1_cache;
    const int full_dataset_num_items;
};

struct ethash_epoch_context_full;

struct ethash_result {
    union ethash_hash256 final_hash;
    union ethash_hash256 mix_hash;
};

#ifdef __cplusplus
}
#endif
