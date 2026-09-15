#pragma once

#include "keccak.h"

namespace ethash {

inline ethash_hash256 keccak256(const uint8_t* data, size_t size) noexcept {
    return ethash_keccak256(data, size);
}

inline ethash_hash256 keccak256(const ethash_hash256& hash) noexcept {
    return ethash_keccak256_32(hash.bytes);
}

inline ethash_hash512 keccak512(const uint8_t* data, size_t size) noexcept {
    return ethash_keccak512(data, size);
}

inline ethash_hash512 keccak512(const ethash_hash512& hash) noexcept {
    return ethash_keccak512_64(hash.bytes);
}

} // namespace ethash
