#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <unordered_map>

namespace kawpow {

constexpr uint32_t FNV_PRIME = 0x01000193u;
constexpr uint32_t FNV_OFFSET_BASIS = 0x811c9dc5u;

inline uint32_t fnv1a(uint32_t u, uint32_t v) {
    return ((u ^ v) * FNV_PRIME);
}

class KISS99 {
public:
    uint32_t z, w, jsr, jcong;

    KISS99(uint32_t z_, uint32_t w_, uint32_t jsr_, uint32_t jcong_)
        : z(z_), w(w_), jsr(jsr_), jcong(jcong_) {}

    uint32_t operator()() {
        z = 36969 * (z & 65535) + (z >> 16);
        w = 18000 * (w & 65535) + (w >> 16);
        uint32_t mwc = ((z << 16) + w);
        jsr ^= (jsr << 17);
        jsr ^= (jsr >> 13);
        jsr ^= (jsr << 5);
        jcong = 69069 * jcong + 1234567;
        return (((mwc ^ jcong) + jsr));
    }
};

// Mathematical expression generators for ProgPOW straight-line unrolling
std::string gen_merge_expr(const std::string& target_expr, const std::string& val_expr, uint32_t selector);
std::string gen_math_expr(const std::string& a_expr, const std::string& b_expr, uint32_t selector);

// Generates the 64-round unrolled ProgPOW kernel source string for a given period
std::pair<std::string, std::string> generate_round_code(uint64_t period);
std::string build_jit_kernel_source(uint64_t period);

// KAWPOW DAG and Cache epoch sizing
bool is_odd_prime(uint32_t n);
uint32_t find_largest_prime(uint32_t n);
uint32_t get_cache_num_items(uint32_t epoch);
uint32_t get_dataset_num_items(uint32_t epoch);

// CUDA Driver / NVRTC runtime abstractions
struct SearchResult {
    uint64_t nonce;
    std::vector<uint8_t> mix_hash;
};

class CudaEngine {
public:
    static bool init();
    static int get_device_count();
    static std::string get_device_name(int device_id);
    static std::pair<int, int> get_compute_capability(int device_id);
};

} // namespace kawpow
