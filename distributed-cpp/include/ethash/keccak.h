#pragma once

#include "hash_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

union ethash_hash256 ethash_keccak256(const uint8_t* data, size_t size);
union ethash_hash256 ethash_keccak256_32(const uint8_t data[32]);
union ethash_hash512 ethash_keccak512(const uint8_t* data, size_t size);
union ethash_hash512 ethash_keccak512_64(const uint8_t data[64]);

#ifdef __cplusplus
}
#endif
