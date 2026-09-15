#include "engine_core.hpp"
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>

namespace kawpow {

std::string gen_merge_expr(const std::string& target_expr, const std::string& val_expr, uint32_t selector) {
    uint32_t s = selector % 4;
    uint32_t x = ((selector >> 16) % 31) + 1;
    if (s == 0) {
        return target_expr + " = (" + target_expr + " * 33) + (" + val_expr + ");";
    } else if (s == 1) {
        return target_expr + " = (" + target_expr + " ^ (" + val_expr + ")) * 33;";
    } else if (s == 2) {
        return target_expr + " = ROTL32(" + target_expr + ", " + std::to_string(x) + ") ^ (" + val_expr + ");";
    } else {
        return target_expr + " = ROTR32(" + target_expr + ", " + std::to_string(x) + ") ^ (" + val_expr + ");";
    }
}

std::string gen_math_expr(const std::string& a_expr, const std::string& b_expr, uint32_t selector) {
    uint32_t s = selector % 11;
    if (s == 0) return "(" + a_expr + " + " + b_expr + ")";
    if (s == 1) return "(" + a_expr + " * " + b_expr + ")";
    if (s == 2) return "(uint32_t)(((uint64_t)(" + a_expr + ") * (uint64_t)(" + b_expr + ")) >> 32)";
    if (s == 3) return "((" + a_expr + " < " + b_expr + ") ? " + a_expr + " : " + b_expr + ")";
    if (s == 4) return "ROTL32(" + a_expr + ", (" + b_expr + ") & 31)";
    if (s == 5) return "ROTR32(" + a_expr + ", (" + b_expr + ") & 31)";
    if (s == 6) return "(" + a_expr + " & " + b_expr + ")";
    if (s == 7) return "(" + a_expr + " | " + b_expr + ")";
    if (s == 8) return "(" + a_expr + " ^ " + b_expr + ")";
    if (s == 9) return "(__clz(" + a_expr + ") + __clz(" + b_expr + "))";
    if (s == 10) return "(__popc(" + a_expr + ") + __popc(" + b_expr + "))";
    return "(" + a_expr + " + " + b_expr + ")";
}

std::pair<std::string, std::string> generate_round_code(uint64_t period) {
    uint32_t p_lo = static_cast<uint32_t>(period & 0xFFFFFFFFULL);
    uint32_t p_hi = static_cast<uint32_t>(period >> 32);
    uint32_t r_z = fnv1a(FNV_OFFSET_BASIS, p_lo);
    uint32_t r_w = fnv1a(r_z, p_hi);
    uint32_t r_jsr = fnv1a(r_w, p_lo);
    uint32_t r_jcong = fnv1a(r_jsr, p_hi);
    KISS99 rng(r_z, r_w, r_jsr, r_jcong);

    std::vector<uint32_t> dst_seq(32);
    std::vector<uint32_t> src_seq(32);
    std::iota(dst_seq.begin(), dst_seq.end(), 0);
    std::iota(src_seq.begin(), src_seq.end(), 0);

    for (int i = 32; i > 1; --i) {
        uint32_t r1 = rng() % i;
        std::swap(dst_seq[i - 1], dst_seq[r1]);
        uint32_t r2 = rng() % i;
        std::swap(src_seq[i - 1], src_seq[r2]);
    }

    size_t dst_counter = 0;
    size_t src_counter = 0;

    std::ostringstream math_oss;
    for (int i = 0; i < 18; ++i) {
        if (i < 11) {
            uint32_t src = src_seq[src_counter % 32]; src_counter++;
            uint32_t dst = dst_seq[dst_counter % 32]; dst_counter++;
            uint32_t sel = rng();
            std::string val = "shared_l1[mix[" + std::to_string(src) + "] % 4096]";
            math_oss << gen_merge_expr("mix[" + std::to_string(dst) + "]", val, sel) << "\n        ";
        }
        if (i < 18) {
            uint32_t src_rnd = rng() % (32 * 31);
            uint32_t src1 = src_rnd % 32;
            uint32_t src2 = src_rnd / 32;
            if (src2 >= src1) src2++;
            uint32_t sel1 = rng();
            uint32_t dst = dst_seq[dst_counter % 32]; dst_counter++;
            uint32_t sel2 = rng();
            std::string math_expr = gen_math_expr("mix[" + std::to_string(src1) + "]", "mix[" + std::to_string(src2) + "]", sel1);
            math_oss << gen_merge_expr("mix[" + std::to_string(dst) + "]", math_expr, sel2) << "\n        ";
        }
    }

    std::vector<uint32_t> dag_dsts(4);
    std::vector<uint32_t> dag_sels(4);
    for (int i = 0; i < 4; ++i) {
        uint32_t d = (i == 0) ? 0 : dst_seq[dst_counter % 32];
        if (i != 0) dst_counter++;
        uint32_t s = rng();
        dag_dsts[i] = d;
        dag_sels[i] = s;
    }

    std::ostringstream dag_oss;
    for (int i = 0; i < 4; ++i) {
        std::string val = "__ldg(&dag[dag_word_offset + " + std::to_string(i) + "])";
        dag_oss << gen_merge_expr("mix[" + std::to_string(dag_dsts[i]) + "]", val, dag_sels[i]);
        if (i < 3) dag_oss << "\n        ";
    }

    return {math_oss.str(), dag_oss.str()};
}

std::string build_jit_kernel_source(uint64_t period) {
    auto [round_math, round_dag] = generate_round_code(period);

    std::ostringstream oss;
    oss << R"(typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))
#define FNV_PRIME 0x01000193u
#define FNV_OFFSET_BASIS 0x811c9dc5u

__device__ __forceinline__ uint32_t fnv1a(uint32_t u, uint32_t v) {
    return (u ^ v) * FNV_PRIME;
}

__constant__ uint32_t RC800[22] = {
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080
};

__constant__ uint8_t ROT800[5][5] = {
    {0, 4, 3, 9, 18},
    {1, 12, 10, 13, 2},
    {30, 6, 11, 15, 29},
    {28, 23, 25, 21, 24},
    {27, 20, 7, 8, 14}
};

__constant__ uint32_t RAVENCOIN_KAWPOW[15] = {
    0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57
};

__device__ void keccak_f800(uint32_t state[25]) {
    uint32_t A[5][5];
    #pragma unroll
    for (int x = 0; x < 5; x++) {
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            A[x][y] = state[x + 5 * y];
        }
    }
    #pragma unroll
    for (int round_idx = 0; round_idx < 22; round_idx++) {
        uint32_t C[5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            C[x] = A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4];
        }
        uint32_t D[5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ROTL32(C[(x + 1) % 5], 1);
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                A[x][y] ^= D[x];
            }
        }
        uint32_t B[5][5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                B[y][(2 * x + 3 * y) % 5] = ROTL32(A[x][y], ROT800[x][y]);
            }
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);
            }
        }
        A[0][0] ^= RC800[round_idx];
    }
    #pragma unroll
    for (int x = 0; x < 5; x++) {
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            state[x + 5 * y] = A[x][y];
        }
    }
}

typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

__device__ __forceinline__ uint32_t kiss99(kiss99_t *st) {
    st->z = 36969 * (st->z & 65535) + (st->z >> 16);
    st->w = 18000 * (st->w & 65535) + (st->w >> 16);
    uint32_t mwc = ((st->z << 16) + st->w);
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    st->jcong = 69069 * st->jcong + 1234567;
    return ((mwc ^ st->jcong) + st->jsr);
}

extern "C" __global__ void kawpow_search_jit(const uint32_t *header_words, uint64_t start_nonce, uint64_t target_high64,
                                             const uint32_t *dag, uint32_t num_dag_items_2048,
                                             uint32_t batch_size, uint64_t *found_nonces, uint8_t *found_mixes,
                                             uint32_t *found_count) {
    __shared__ uint32_t shared_l1[4096];
    for (int idx = threadIdx.x; idx < 4096; idx += blockDim.x) {
        shared_l1[idx] = dag[idx];
    }
    __syncthreads();

    uint32_t warp_lane = threadIdx.x & 31;
    uint32_t subwarp_mask = (warp_lane < 16) ? 0x0000FFFFu : 0xFFFF0000u;
    uint32_t base_lane = (warp_lane < 16) ? 0 : 16;

    uint32_t lane_id = threadIdx.x % 16;
    uint32_t group_id = ((blockIdx.x * blockDim.x) + threadIdx.x) / 16;
    if (group_id >= batch_size) return;
    uint64_t nonce = start_nonce + group_id;

    // 1. Initial Keccak
    uint32_t state[25];
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) state[i] = header_words[i];
        state[8] = (uint32_t)(nonce & 0xFFFFFFFFULL);
        state[9] = (uint32_t)(nonce >> 32);
        #pragma unroll
        for (int i = 10; i < 25; i++) state[i] = RAVENCOIN_KAWPOW[i - 10];
        keccak_f800(state);
    }

    uint32_t state2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state2[i] = __shfl_sync(subwarp_mask, state[i], base_lane + 0);
    }

    // 2. Mix init
    uint32_t z = fnv1a(FNV_OFFSET_BASIS, state2[0]);
    uint32_t w = fnv1a(z, state2[1]);
    uint32_t jsr = fnv1a(w, lane_id);
    uint32_t jcong = fnv1a(jsr, lane_id);
    kiss99_t my_rng = {z, w, jsr, jcong};

    uint32_t mix[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        mix[i] = kiss99(&my_rng);
    }

    // 3. 64-round unrolled ProgPOW loop
    for (int r = 0; r < 64; r++) {
        uint32_t mix_r0 = __shfl_sync(subwarp_mask, mix[0], base_lane + (r % 16));
        uint32_t item_index = mix_r0 % num_dag_items_2048;

        // JIT generated straight-line math & cache ops
        )" << round_math << R"(

        // JIT generated DAG loads
        uint32_t dag_word_offset = (item_index * 64) + (((lane_id ^ r) % 16) * 4);
        )" << round_dag << R"(
    }

    // 4. Reduction
    uint32_t lane_hash = FNV_OFFSET_BASIS;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        lane_hash = fnv1a(lane_hash, mix[i]);
    }

    uint32_t mix_hash[8];
    #pragma unroll
    for (int w_idx = 0; w_idx < 8; w_idx++) {
        uint32_t h0 = __shfl_sync(subwarp_mask, lane_hash, base_lane + w_idx);
        uint32_t h1 = __shfl_sync(subwarp_mask, lane_hash, base_lane + w_idx + 8);
        mix_hash[w_idx] = fnv1a(fnv1a(FNV_OFFSET_BASIS, h0), h1);
    }

    // 5. Final Keccak & target check
    if (lane_id == 0) {
        uint32_t st_final[25];
        #pragma unroll
        for (int i = 0; i < 8; i++) st_final[i] = state2[i];
        #pragma unroll
        for (int i = 8; i < 16; i++) st_final[i] = mix_hash[i - 8];
        #pragma unroll
        for (int i = 16; i < 25; i++) st_final[i] = RAVENCOIN_KAWPOW[i - 16];
        keccak_f800(st_final);

        uint64_t w0 = ((uint64_t)st_final[1] << 32) | (uint64_t)st_final[0];
        uint64_t hash_val_be = (((w0 & 0x00000000000000ffULL) << 56) |
                                ((w0 & 0x000000000000ff00ULL) << 40) |
                                ((w0 & 0x0000000000ff0000ULL) << 24) |
                                ((w0 & 0x00000000ff000000ULL) <<  8) |
                                ((w0 & 0x000000ff00000000ULL) >>  8) |
                                ((w0 & 0x0000ff0000000000ULL) >> 24) |
                                ((w0 & 0x00ff000000000000ULL) >> 40) |
                                ((w0 & 0xff00000000000000ULL) >> 56));

        if (hash_val_be <= target_high64) {
            uint32_t idx = atomicAdd(found_count, 1);
            if (idx < 1024) {
                found_nonces[idx] = nonce;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    ((uint32_t*)found_mixes)[idx * 8 + b] = mix_hash[b];
                }
            }
        }
    }
}
)";

    return oss.str();
}

bool is_odd_prime(uint32_t n) {
    uint32_t d = 3;
    while (d * d <= n) {
        if (n % d == 0) return false;
        d += 2;
    }
    return true;
}

uint32_t find_largest_prime(uint32_t n) {
    if (n < 2) return 0;
    if (n == 2) return 2;
    if (n % 2 == 0) n -= 1;
    while (!is_odd_prime(n)) {
        n -= 2;
    }
    return n;
}

uint32_t get_cache_num_items(uint32_t epoch) {
    uint32_t upper = (1 << 24) / 64 + epoch * ((1 << 17) / 64);
    return find_largest_prime(upper);
}

uint32_t get_dataset_num_items(uint32_t epoch) {
    uint32_t upper = (1 << 30) / 128 + epoch * ((1 << 23) / 128);
    return find_largest_prime(upper);
}

} // namespace kawpow
