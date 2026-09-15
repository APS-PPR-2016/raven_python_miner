#pragma once

#include "ethash.h"
#include "hash_types.hpp"

#include <cstdint>
#include <cstring>
#include <memory>

namespace ethash
{
constexpr auto revision = ETHASH_REVISION;

static constexpr int epoch_length = ETHASH_EPOCH_LENGTH;
static constexpr int light_cache_item_size = ETHASH_LIGHT_CACHE_ITEM_SIZE;
static constexpr int full_dataset_item_size = ETHASH_FULL_DATASET_ITEM_SIZE;
static constexpr int num_dataset_accesses = ETHASH_NUM_DATASET_ACCESSES;

using epoch_context = ethash_epoch_context;
using epoch_context_full = ethash_epoch_context_full;
using result = ethash_result;

inline hash256 hash256_from_bytes(const uint8_t bytes[32]) noexcept
{
    hash256 h;
    std::memcpy(&h, bytes, sizeof(h));
    return h;
}

struct search_result
{
    uint64_t nonce = 0;
    hash256 final_hash = {};
    hash256 mix_hash = {};
    bool solution_found = false;
};

} // namespace ethash
