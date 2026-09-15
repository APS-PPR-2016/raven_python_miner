#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

union ethash_hash256 {
    uint8_t bytes[32];
    uint32_t words[8];
    uint64_t word64s[4];
    uint64_t qwords[4];
};

union ethash_hash512 {
    uint8_t bytes[64];
    uint32_t words[16];
    uint64_t word64s[8];
    uint64_t qwords[8];
};

union ethash_hash1024 {
    uint8_t bytes[128];
    uint32_t words[32];
    uint64_t word64s[16];
    uint64_t qwords[16];
};

#ifdef __cplusplus
}
#endif
