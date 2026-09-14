import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc
import numpy as np
import time

CUDA_TEST_SOURCE = r"""
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))
#define ROL64(x, s) (((x) << (s)) | ((x) >> (64 - (s))))
#define FNV_PRIME 0x01000193u
#define FNV_OFFSET_BASIS 0x811c9dc5u

__device__ __forceinline__ uint32_t fnv1(uint32_t u, uint32_t v) {
    return (u * FNV_PRIME) ^ v;
}

__device__ __forceinline__ uint32_t fnv1a(uint32_t u, uint32_t v) {
    return (u ^ v) * FNV_PRIME;
}

// Keccak-f[800] round constants (22 rounds)
__constant__ uint32_t RC800[22] = {
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080
};

// Keccak-f[800] rotation offsets
__constant__ uint8_t ROT800[5][5] = {
    {0, 4, 3, 9, 18},
    {1, 12, 10, 13, 2},
    {30, 6, 11, 15, 29},
    {28, 23, 25, 21, 24},
    {27, 20, 7, 8, 14}
};

// Ravencoin domain string: "RAVENCOINKAWPOW"
__constant__ uint32_t RAVENCOIN_KAWPOW[15] = {
    0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57
};

// Keccak-f[1600] round constants (24 rounds)
__constant__ uint64_t RC1600[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ void keccak_f1600(uint64_t state[25]) {
    uint64_t Aba = state[0],  Abe = state[1],  Abi = state[2],  Abo = state[3],  Abu = state[4];
    uint64_t Aga = state[5],  Age = state[6],  Agi = state[7],  Ago = state[8],  Agu = state[9];
    uint64_t Aka = state[10], Ake = state[11], Aki = state[12], Ako = state[13], Aku = state[14];
    uint64_t Ama = state[15], Ame = state[16], Ami = state[17], Amo = state[18], Amu = state[19];
    uint64_t Asa = state[20], Ase = state[21], Asi = state[22], Aso = state[23], Asu = state[24];

    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;
    uint64_t Ba, Be, Bi, Bo, Bu;
    uint64_t Da, De, Di, Do, Du;

    #pragma unroll
    for (int round = 0; round < 24; round += 2) {
        Ba = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        Be = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        Bi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        Bo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        Bu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        Da = Bu ^ ROL64(Be, 1);
        De = Ba ^ ROL64(Bi, 1);
        Di = Be ^ ROL64(Bo, 1);
        Do = Bi ^ ROL64(Bu, 1);
        Du = Bo ^ ROL64(Ba, 1);

        Ba = Aba ^ Da;
        Be = ROL64(Age ^ De, 44);
        Bi = ROL64(Aki ^ Di, 43);
        Bo = ROL64(Amo ^ Do, 21);
        Bu = ROL64(Asu ^ Du, 14);
        Eba = Ba ^ (~Be & Bi) ^ RC1600[round];
        Ebe = Be ^ (~Bi & Bo);
        Ebi = Bi ^ (~Bo & Bu);
        Ebo = Bo ^ (~Bu & Ba);
        Ebu = Bu ^ (~Ba & Be);

        Ba = ROL64(Abo ^ Do, 28);
        Be = ROL64(Agu ^ Du, 20);
        Bi = ROL64(Aka ^ Da, 3);
        Bo = ROL64(Ame ^ De, 45);
        Bu = ROL64(Asi ^ Di, 61);
        Ega = Ba ^ (~Be & Bi);
        Ege = Be ^ (~Bi & Bo);
        Egi = Bi ^ (~Bo & Bu);
        Ego = Bo ^ (~Bu & Ba);
        Egu = Bu ^ (~Ba & Be);

        Ba = ROL64(Abe ^ De, 1);
        Be = ROL64(Agi ^ Di, 6);
        Bi = ROL64(Ako ^ Do, 25);
        Bo = ROL64(Amu ^ Du, 8);
        Bu = ROL64(Asa ^ Da, 18);
        Eka = Ba ^ (~Be & Bi);
        Eke = Be ^ (~Bi & Bo);
        Eki = Bi ^ (~Bo & Bu);
        Eko = Bo ^ (~Bu & Ba);
        Eku = Bu ^ (~Ba & Be);

        Ba = ROL64(Abu ^ Du, 27);
        Be = ROL64(Aga ^ Da, 36);
        Bi = ROL64(Ake ^ De, 10);
        Bo = ROL64(Ami ^ Di, 15);
        Bu = ROL64(Aso ^ Do, 56);
        Ema = Ba ^ (~Be & Bi);
        Eme = Be ^ (~Bi & Bo);
        Emi = Bi ^ (~Bo & Bu);
        Emo = Bo ^ (~Bu & Ba);
        Emu = Bu ^ (~Ba & Be);

        Ba = ROL64(Abi ^ Di, 62);
        Be = ROL64(Ago ^ Do, 55);
        Bi = ROL64(Aku ^ Du, 39);
        Bo = ROL64(Ama ^ Da, 41);
        Bu = ROL64(Ase ^ De, 2);
        Esa = Ba ^ (~Be & Bi);
        Ese = Be ^ (~Bi & Bo);
        Esi = Bi ^ (~Bo & Bu);
        Eso = Bo ^ (~Bu & Ba);
        Esu = Bu ^ (~Ba & Be);

        Ba = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        Be = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        Bi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        Bo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        Bu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        Da = Bu ^ ROL64(Be, 1);
        De = Ba ^ ROL64(Bi, 1);
        Di = Be ^ ROL64(Bo, 1);
        Do = Bi ^ ROL64(Bu, 1);
        Du = Bo ^ ROL64(Ba, 1);

        Ba = Eba ^ Da;
        Be = ROL64(Ege ^ De, 44);
        Bi = ROL64(Eki ^ Di, 43);
        Bo = ROL64(Emo ^ Do, 21);
        Bu = ROL64(Esu ^ Du, 14);
        Aba = Ba ^ (~Be & Bi) ^ RC1600[round + 1];
        Abe = Be ^ (~Bi & Bo);
        Abi = Bi ^ (~Bo & Bu);
        Abo = Bo ^ (~Bu & Ba);
        Abu = Bu ^ (~Ba & Be);

        Ba = ROL64(Ebo ^ Do, 28);
        Be = ROL64(Egu ^ Du, 20);
        Bi = ROL64(Eka ^ Da, 3);
        Bo = ROL64(Eme ^ De, 45);
        Bu = ROL64(Esi ^ Di, 61);
        Aga = Ba ^ (~Be & Bi);
        Age = Be ^ (~Bi & Bo);
        Agi = Bi ^ (~Bo & Bu);
        Ago = Bo ^ (~Bu & Ba);
        Agu = Bu ^ (~Ba & Be);

        Ba = ROL64(Ebe ^ De, 1);
        Be = ROL64(Egi ^ Di, 6);
        Bi = ROL64(Eko ^ Do, 25);
        Bo = ROL64(Emu ^ Du, 8);
        Bu = ROL64(Esa ^ Da, 18);
        Aka = Ba ^ (~Be & Bi);
        Ake = Be ^ (~Bi & Bo);
        Aki = Bi ^ (~Bo & Bu);
        Ako = Bo ^ (~Bu & Ba);
        Aku = Bu ^ (~Ba & Be);

        Ba = ROL64(Ebu ^ Du, 27);
        Be = ROL64(Ega ^ Da, 36);
        Bi = ROL64(Eke ^ De, 10);
        Bo = ROL64(Emi ^ Di, 15);
        Bu = ROL64(Eso ^ Do, 56);
        Ama = Ba ^ (~Be & Bi);
        Ame = Be ^ (~Bi & Bo);
        Ami = Bi ^ (~Bo & Bu);
        Amo = Bo ^ (~Bu & Ba);
        Amu = Bu ^ (~Ba & Be);

        Ba = ROL64(Ebi ^ Di, 62);
        Be = ROL64(Ego ^ Do, 55);
        Bi = ROL64(Eku ^ Du, 39);
        Bo = ROL64(Ema ^ Da, 41);
        Bu = ROL64(Ese ^ De, 2);
        Asa = Ba ^ (~Be & Bi);
        Ase = Be ^ (~Bi & Bo);
        Asi = Bi ^ (~Bo & Bu);
        Aso = Bo ^ (~Bu & Ba);
        Asu = Bu ^ (~Ba & Be);
    }

    state[0] = Aba;  state[1] = Abe;  state[2] = Abi;  state[3] = Abo;  state[4] = Abu;
    state[5] = Aga;  state[6] = Age;  state[7] = Agi;  state[8] = Ago;  state[9] = Agu;
    state[10] = Aka; state[11] = Ake; state[12] = Aki; state[13] = Ako; state[14] = Aku;
    state[15] = Ama; state[16] = Ame; state[17] = Ami; state[18] = Amo; state[19] = Amu;
    state[20] = Asa; state[21] = Ase; state[22] = Asi; state[23] = Aso; state[24] = Asu;
}

// Keccak-512 for 32-byte input (out is uint64_t[8])
__device__ void keccak512_32(const uint64_t in[4], uint64_t out[8]) {
    uint64_t state[25] = {0};
    state[0] = in[0];
    state[1] = in[1];
    state[2] = in[2];
    state[3] = in[3];
    state[4] = 0x0000000000000001ULL;
    state[8] = 0x8000000000000000ULL;
    keccak_f1600(state);
    for (int i = 0; i < 8; i++) out[i] = state[i];
}

// Keccak-512 for 64-byte input (in-place in uint64_t[8])
__device__ void keccak512_64(uint64_t data[8]) {
    uint64_t state[25] = {0};
    for (int i = 0; i < 8; i++) state[i] = data[i];
    state[8] = 0x8000000000000001ULL;
    keccak_f1600(state);
    for (int i = 0; i < 8; i++) data[i] = state[i];
}

// Build light cache kernel (single block, thread 0 builds sequentially)
extern "C" __global__ void build_light_cache_gpu(uint64_t *cache, uint32_t num_items) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t seed[4] = {0, 0, 0, 0};
    uint64_t item[8];
    keccak512_32(seed, item);
    for (int k = 0; k < 8; k++) cache[k] = item[k];

    for (uint32_t i = 1; i < num_items; i++) {
        keccak512_64(item);
        for (int k = 0; k < 8; k++) cache[i * 8 + k] = item[k];
    }

    for (int q = 0; q < 3; q++) {
        for (uint32_t i = 0; i < num_items; i++) {
            uint32_t t = ((const uint32_t*)(cache + i * 8))[0];
            uint32_t v = t % num_items;
            uint32_t w = (num_items + i - 1) % num_items;

            uint64_t x[8];
            for (int k = 0; k < 8; k++) {
                x[k] = cache[v * 8 + k] ^ cache[w * 8 + k];
            }
            keccak512_64(x);
            for (int k = 0; k < 8; k++) {
                cache[i * 8 + k] = x[k];
            }
        }
    }
}

// True KAWPOW DAG generator kernel using Keccak-512 with 512 parent rounds
extern "C" __global__ void generate_dag_kernel(const uint32_t *light_cache, uint32_t num_cache_items,
                                               uint32_t *dag_out, uint32_t total_half_items) {
    uint32_t gid = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_half_items) return;

    uint64_t mix[8];
    uint32_t cache_offset = (gid % num_cache_items) * 16;
    for (int i = 0; i < 8; i++) {
        mix[i] = ((const uint64_t*)(light_cache + cache_offset))[i];
    }
    ((uint32_t*)mix)[0] ^= gid;
    keccak512_64(mix);

    for (uint32_t j = 0; j < 512; j++) {
        uint32_t t = fnv1(gid ^ j, ((uint32_t*)mix)[j % 16]);
        uint32_t parent_offset = (t % num_cache_items) * 16;
        const uint32_t *parent32 = light_cache + parent_offset;
        uint32_t *mix32 = (uint32_t*)mix;
        for (int k = 0; k < 16; k++) {
            mix32[k] = fnv1(mix32[k], parent32[k]);
        }
    }

    keccak512_64(mix);

    uint32_t out_offset = gid * 16;
    for (int i = 0; i < 8; i++) {
        ((uint64_t*)(dag_out + out_offset))[i] = mix[i];
    }
}

// Keccak-f[800] 25-word permutation
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

// KISS99 RNG
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

__device__ __forceinline__ uint32_t random_math(uint32_t a, uint32_t b, uint32_t selector) {
    switch (selector % 11) {
        default:
        case 0: return a + b;
        case 1: return a * b;
        case 2: return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
        case 3: return (a < b) ? a : b;
        case 4: return ROTL32(a, b & 31);
        case 5: return ROTR32(a, b & 31);
        case 6: return a & b;
        case 7: return a | b;
        case 8: return a ^ b;
        case 9: return __clz(a) + __clz(b);
        case 10: return __popc(a) + __popc(b);
    }
}

__device__ __forceinline__ void random_merge(uint32_t &a, uint32_t b, uint32_t selector) {
    uint32_t x = ((selector >> 16) % 31) + 1;
    switch (selector % 4) {
        case 0: a = (a * 33) + b; break;
        case 1: a = (a ^ b) * 33; break;
        case 2: a = ROTL32(a, x) ^ b; break;
        case 3: a = ROTR32(a, x) ^ b; break;
    }
}

extern "C" __global__ void test_kawpow_single(const uint32_t *header_words, uint64_t nonce, uint64_t period,
                                              const uint32_t *dag, uint32_t num_dag_items_2048,
                                              uint32_t total_dag_words, uint32_t *out_mix, uint32_t *out_final) {
    uint32_t lane_id = threadIdx.x % 16;
    if (blockIdx.x > 0 || threadIdx.x >= 16) return;

    // 1. Lane 0 executes initial Keccak-f[800]
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
        state2[i] = __shfl_sync(0xffff, state[i], 0);
    }

    // 2. Initialize mix across 16 lanes
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

    // 3. Initialize ProgPOW mix_rng_state from period = block_height / 50
    uint32_t p_lo = (uint32_t)(period & 0xFFFFFFFFULL);
    uint32_t p_hi = (uint32_t)(period >> 32);
    uint32_t r_z = fnv1a(FNV_OFFSET_BASIS, p_lo);
    uint32_t r_w = fnv1a(r_z, p_hi);
    uint32_t r_jsr = fnv1a(r_w, p_lo);
    uint32_t r_jcong = fnv1a(r_jsr, p_hi);
    kiss99_t prog_rng_init = {r_z, r_w, r_jsr, r_jcong};

    uint8_t dst_seq_init[32];
    uint8_t src_seq_init[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        dst_seq_init[i] = (uint8_t)i;
        src_seq_init[i] = (uint8_t)i;
    }
    for (int i = 32; i > 1; --i) {
        uint32_t r1 = kiss99(&prog_rng_init) % i;
        uint8_t t1 = dst_seq_init[i - 1]; dst_seq_init[i - 1] = dst_seq_init[r1]; dst_seq_init[r1] = t1;
        uint32_t r2 = kiss99(&prog_rng_init) % i;
        uint8_t t2 = src_seq_init[i - 1]; src_seq_init[i - 1] = src_seq_init[r2]; src_seq_init[r2] = t2;
    }

    // 4. ProgPOW 64-round loop
    for (int r = 0; r < 64; r++) {
        // Broadcast mix[r % 16][0] across 16 lanes
        uint32_t mix_r0 = __shfl_sync(0xffff, mix[0], r % 16);
        uint32_t item_index = mix_r0 % num_dag_items_2048;

        // Reset state per round (matching C++ pass-by-value of mix_rng_state)
        kiss99_t prog_rng = prog_rng_init;
        int dst_counter = 0;
        int src_counter = 0;

        #pragma unroll
        for (int i = 0; i < 18; i++) {
            if (i < 11) {
                uint32_t src = src_seq_init[(src_counter++) % 32];
                uint32_t dst = dst_seq_init[(dst_counter++) % 32];
                uint32_t sel = kiss99(&prog_rng);
                uint32_t offset = mix[src] % 4096;
                random_merge(mix[dst], dag[offset % total_dag_words], sel);
            }
            if (i < 18) {
                uint32_t src_rnd = kiss99(&prog_rng) % (32 * 31);
                uint32_t src1 = src_rnd % 32;
                uint32_t src2 = src_rnd / 32;
                if (src2 >= src1) src2++;

                uint32_t sel1 = kiss99(&prog_rng);
                uint32_t dst = dst_seq_init[(dst_counter++) % 32];
                uint32_t sel2 = kiss99(&prog_rng);

                uint32_t data = random_math(mix[src1], mix[src2], sel1);
                random_merge(mix[dst], data, sel2);
            }
        }

        uint32_t dsts[4];
        uint32_t sels[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            dsts[i] = (i == 0) ? 0 : dst_seq_init[(dst_counter++) % 32];
            sels[i] = kiss99(&prog_rng);
        }

        uint32_t dag_word_offset = (item_index * 64) + (((lane_id ^ r) % 16) * 4);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t word = dag[(dag_word_offset + i) % total_dag_words];
            random_merge(mix[dsts[i]], word, sels[i]);
        }
    }

    // 5. Reduction
    uint32_t lane_hash = FNV_OFFSET_BASIS;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        lane_hash = fnv1a(lane_hash, mix[i]);
    }

    uint32_t mix_hash[8];
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        uint32_t h0 = __shfl_sync(0xffff, lane_hash, w);
        uint32_t h1 = __shfl_sync(0xffff, lane_hash, w + 8);
        mix_hash[w] = fnv1a(fnv1a(FNV_OFFSET_BASIS, h0), h1);
    }

    // 6. Final Keccak & Output
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) out_mix[i] = mix_hash[i];

        uint32_t st_final[25];
        #pragma unroll
        for (int i = 0; i < 8; i++) st_final[i] = state2[i];
        #pragma unroll
        for (int i = 8; i < 16; i++) st_final[i] = mix_hash[i - 8];
        #pragma unroll
        for (int i = 16; i < 25; i++) st_final[i] = RAVENCOIN_KAWPOW[i - 16];
        keccak_f800(st_final);

        #pragma unroll
        for (int i = 0; i < 8; i++) out_final[i] = st_final[i];
    }
}
"""

def main():
    dev = cuda.Device(0)
    cc = dev.compute_capability()
    arch = f"sm_{cc[0]}{cc[1]}"

    print("Compiling CUDA test harness...")
    t0 = time.time()
    err, prog = nvrtc.nvrtcCreateProgram(CUDA_TEST_SOURCE.encode(), b"test.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture={arch}".encode()])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        print("Compile log:", log.decode())
        return

    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    print(f"Compiled in {time.time() - t0:.2f}s")

    k_cache = mod.get_function("build_light_cache_gpu")
    k_dag = mod.get_function("generate_dag_kernel")
    k_test = mod.get_function("test_kawpow_single")

    num_cache_items = 262139
    print(f"Allocating GPU memory for epoch 0 light cache ({num_cache_items * 64} bytes)...")
    cache_gpu = cuda.mem_alloc(num_cache_items * 64)

    print("Running build_light_cache_gpu on RTX 3060...")
    t0 = time.time()
    k_cache(cache_gpu, np.uint32(num_cache_items), block=(1, 1, 1), grid=(1, 1))
    cuda.Context.synchronize()
    print(f"Light cache generated on GPU in {time.time() - t0:.2f}s!")

    # Generate first 64 items (16 KB) for L1 cache plus enough items for Test Case 0
    # For Test Case 0, let's generate 65536 half-items (8192 hash2048 items) or full DAG
    # Let's see: how many half items in epoch 0?
    total_half_items = 8388593 * 2  # 16777186 half-items (1 GB)
    # Let's allocate 1 GB on GPU for epoch 0 DAG
    print("Allocating 1 GB VRAM for Epoch 0 DAG...")
    dag_bytes = total_half_items * 64
    dag_gpu = cuda.mem_alloc(dag_bytes)

    print("Generating 1 GB DAG directly on RTX 3060...")
    t0 = time.time()
    threads_per_block = 256
    blocks = (total_half_items + threads_per_block - 1) // threads_per_block
    k_dag(cache_gpu, np.uint32(num_cache_items), dag_gpu, np.uint32(total_half_items),
          block=(threads_per_block, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()
    print(f"Full 1 GB DAG generated on GPU in {time.time() - t0:.2f}s!")

    test_cases = [
        {
            "block": 0,
            "header": "0000000000000000000000000000000000000000000000000000000000000000",
            "nonce": 0x0,
            "exp_mix": "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a",
            "exp_final": "e601a7257a70dc48fccc97a7330d704d776047623b92883d77111fb36870f3d1"
        },
        {
            "block": 49,
            "header": "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b",
            "nonce": 0x7073c07,
            "exp_mix": "d36f7e815ee09e74eceb9c96993a3d681edf2bf0921fc7bb710364042db99777",
            "exp_final": "e7ced124598fd2500a55ad9f9f48e3569327fe50493c77a4ac9799b96efb9463"
        },
        {
            "block": 50,
            "header": "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922",
            "nonce": 0x76e482e,
            "exp_mix": "d6dc634ae837e2785b347648ea515e25e5d8821ae0b95e1c2a9c2d497e0dcfbd",
            "exp_final": "ab0ad7ef8d8ee317dd12d10310aceed7321d34fb263791c2de5776a6658d177e"
        }
    ]

    header_gpu = cuda.mem_alloc(8 * 4)
    out_mix_gpu = cuda.mem_alloc(8 * 4)
    out_final_gpu = cuda.mem_alloc(8 * 4)
    out_mix = np.zeros(8, dtype=np.uint32)
    out_final = np.zeros(8, dtype=np.uint32)
    num_dag_items_2048 = 8388593 // 2
    total_dag_words = total_half_items * 16

    for tc in test_cases:
        block = tc["block"]
        period = block // 3
        header_bytes = bytes.fromhex(tc["header"])
        header_words = np.frombuffer(header_bytes, dtype=np.uint32)
        cuda.memcpy_htod(header_gpu, header_words)

        k_test(header_gpu, np.uint64(tc["nonce"]), np.uint64(period), dag_gpu,
               np.uint32(num_dag_items_2048), np.uint32(total_dag_words),
               out_mix_gpu, out_final_gpu, block=(16, 1, 1), grid=(1, 1))
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(out_mix, out_mix_gpu)
        cuda.memcpy_dtoh(out_final, out_final_gpu)
        mix_hex = bytes(out_mix).hex()
        final_hex = bytes(out_final).hex()

        print(f"\n--- TEST CASE (block {block}, period {period}) ---")
        print(f"Mix:   {mix_hex} (match: {mix_hex == tc['exp_mix']})")
        print(f"Final: {final_hex} (match: {final_hex == tc['exp_final']})")
        assert mix_hex == tc['exp_mix'] and final_hex == tc['exp_final']

    print("\nALL TEST CASES PASSED WITH 100% BITWISE PARITY!")

if __name__ == '__main__':
    main()
