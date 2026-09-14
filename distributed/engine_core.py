"""
KAWPOW Distributed Engine Core
==============================
Contains the core consensus-identical mathematical components, JIT compiler logic,
Keccak-512 DAG synthesis CUDA kernels, and epoch dimension calculations for Ravencoin.
Extracted cleanly for distributed server/client usage without modifying main.py.
"""

import time
import numpy as np
import pycuda.driver as cuda
from cuda.bindings import nvrtc

FNV_PRIME = 0x01000193
FNV_OFFSET_BASIS = 0x811c9dc5

def fnv1a(u, v):
    return ((u ^ v) * FNV_PRIME) & 0xFFFFFFFF

class KISS99:
    """Consensus-identical KISS99 32-bit PRNG used by ProgPOW/KAWPOW."""
    def __init__(self, z, w, jsr, jcong):
        self.z = z & 0xFFFFFFFF
        self.w = w & 0xFFFFFFFF
        self.jsr = jsr & 0xFFFFFFFF
        self.jcong = jcong & 0xFFFFFFFF

    def __call__(self):
        self.z = (36969 * (self.z & 65535) + (self.z >> 16)) & 0xFFFFFFFF
        self.w = (18000 * (self.w & 65535) + (self.w >> 16)) & 0xFFFFFFFF
        mwc = ((self.z << 16) + self.w) & 0xFFFFFFFF
        self.jsr ^= (self.jsr << 17) & 0xFFFFFFFF
        self.jsr ^= (self.jsr >> 13) & 0xFFFFFFFF
        self.jsr ^= (self.jsr << 5) & 0xFFFFFFFF
        self.jcong = (69069 * self.jcong + 1234567) & 0xFFFFFFFF
        return (((mwc ^ self.jcong) + self.jsr)) & 0xFFFFFFFF

def gen_merge_expr(target_expr, val_expr, selector):
    s = selector % 4
    x = ((selector >> 16) % 31) + 1
    if s == 0:
        return f"{target_expr} = ({target_expr} * 33) + ({val_expr});"
    elif s == 1:
        return f"{target_expr} = ({target_expr} ^ ({val_expr})) * 33;"
    elif s == 2:
        return f"{target_expr} = ROTL32({target_expr}, {x}) ^ ({val_expr});"
    elif s == 3:
        return f"{target_expr} = ROTR32({target_expr}, {x}) ^ ({val_expr});"

def gen_math_expr(a_expr, b_expr, selector):
    s = selector % 11
    if s == 0: return f"({a_expr} + {b_expr})"
    if s == 1: return f"({a_expr} * {b_expr})"
    if s == 2: return f"(uint32_t)(((uint64_t)({a_expr}) * (uint64_t)({b_expr})) >> 32)"
    if s == 3: return f"(({a_expr} < {b_expr}) ? {a_expr} : {b_expr})"
    if s == 4: return f"ROTL32({a_expr}, ({b_expr}) & 31)"
    if s == 5: return f"ROTR32({a_expr}, ({b_expr}) & 31)"
    if s == 6: return f"({a_expr} & {b_expr})"
    if s == 7: return f"({a_expr} | {b_expr})"
    if s == 8: return f"({a_expr} ^ {b_expr})"
    if s == 9: return f"(__clz({a_expr}) + __clz({b_expr}))"
    if s == 10: return f"(__popc({a_expr}) + __popc({b_expr}))"

def generate_round_code(period):
    """Precomputes ProgPOW KISS99 RNG sequence for given period and generates
    unrolled C code for 18 math ops, 11 cache ops, and 4 DAG loads.
    """
    p_lo = period & 0xFFFFFFFF
    p_hi = period >> 32
    r_z = fnv1a(FNV_OFFSET_BASIS, p_lo)
    r_w = fnv1a(r_z, p_hi)
    r_jsr = fnv1a(r_w, p_lo)
    r_jcong = fnv1a(r_jsr, p_hi)
    rng = KISS99(r_z, r_w, r_jsr, r_jcong)

    dst_seq = list(range(32))
    src_seq = list(range(32))
    for i in range(32, 1, -1):
        r1 = rng() % i
        dst_seq[i - 1], dst_seq[r1] = dst_seq[r1], dst_seq[i - 1]
        r2 = rng() % i
        src_seq[i - 1], src_seq[r2] = src_seq[r2], src_seq[i - 1]

    dst_counter = 0
    src_counter = 0

    lines = []
    for i in range(18):
        if i < 11:
            src = src_seq[src_counter % 32]; src_counter += 1
            dst = dst_seq[dst_counter % 32]; dst_counter += 1
            sel = rng()
            val = f"shared_l1[mix[{src}] % 4096]"
            lines.append(gen_merge_expr(f"mix[{dst}]", val, sel))
        if i < 18:
            src_rnd = rng() % (32 * 31)
            src1 = src_rnd % 32
            src2 = src_rnd // 32
            if src2 >= src1: src2 += 1
            sel1 = rng()
            dst = dst_seq[dst_counter % 32]; dst_counter += 1
            sel2 = rng()
            math_expr = gen_math_expr(f"mix[{src1}]", f"mix[{src2}]", sel1)
            lines.append(gen_merge_expr(f"mix[{dst}]", math_expr, sel2))

    dag_dsts = []
    dag_sels = []
    for i in range(4):
        d = 0 if i == 0 else dst_seq[dst_counter % 32]
        if i != 0: dst_counter += 1
        s = rng()
        dag_dsts.append(d)
        dag_sels.append(s)

    dag_code = []
    for i in range(4):
        val = f"__ldg(&dag[dag_word_offset + {i}])"
        dag_code.append(gen_merge_expr(f"mix[{dag_dsts[i]}]", val, dag_sels[i]))

    return "\n        ".join(lines), "\n        ".join(dag_code)

def build_jit_kernel_source(period):
    round_math, round_dag = generate_round_code(period)
    return r"""
typedef unsigned char uint8_t;
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
        """ + round_math + r"""

        // JIT generated DAG loads
        uint32_t dag_word_offset = (item_index * 64) + (((lane_id ^ r) % 16) * 4);
        """ + round_dag + r"""
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
"""

DAG_CUDA_SOURCE = r"""
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define ROL64(x, s) (((x) << (s)) | ((x) >> (64 - (s))))
#define FNV_PRIME 0x01000193u

__device__ __forceinline__ uint32_t fnv1(uint32_t u, uint32_t v) {
    return (u * FNV_PRIME) ^ v;
}

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
    for (int round_idx = 0; round_idx < 24; round_idx += 2) {
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
        Eba = Ba ^ ((~Be) & Bi) ^ RC1600[round_idx];
        Ebe = Be ^ ((~Bi) & Bo);
        Ebi = Bi ^ ((~Bo) & Bu);
        Ebo = Bo ^ ((~Bu) & Ba);
        Ebu = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Abo ^ Do, 28);
        Be = ROL64(Agu ^ Du, 20);
        Bi = ROL64(Aka ^ Da, 3);
        Bo = ROL64(Ame ^ De, 45);
        Bu = ROL64(Asi ^ Di, 61);
        Ega = Ba ^ ((~Be) & Bi);
        Ege = Be ^ ((~Bi) & Bo);
        Egi = Bi ^ ((~Bo) & Bu);
        Ego = Bo ^ ((~Bu) & Ba);
        Egu = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Abe ^ De, 1);
        Be = ROL64(Agi ^ Di, 6);
        Bi = ROL64(Ako ^ Do, 25);
        Bo = ROL64(Amu ^ Du, 8);
        Bu = ROL64(Asa ^ Da, 18);
        Eka = Ba ^ ((~Be) & Bi);
        Eke = Be ^ ((~Bi) & Bo);
        Eki = Bi ^ ((~Bo) & Bu);
        Eko = Bo ^ ((~Bu) & Ba);
        Eku = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Abu ^ Du, 27);
        Be = ROL64(Aga ^ Da, 36);
        Bi = ROL64(Ake ^ De, 10);
        Bo = ROL64(Ami ^ Di, 15);
        Bu = ROL64(Aso ^ Do, 56);
        Ema = Ba ^ ((~Be) & Bi);
        Eme = Be ^ ((~Bi) & Bo);
        Emi = Bi ^ ((~Bo) & Bu);
        Emo = Bo ^ ((~Bu) & Ba);
        Emu = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Abi ^ Di, 62);
        Be = ROL64(Ago ^ Do, 55);
        Bi = ROL64(Aku ^ Du, 39);
        Bo = ROL64(Ama ^ Da, 41);
        Bu = ROL64(Ase ^ De, 2);
        Esa = Ba ^ ((~Be) & Bi);
        Ese = Be ^ ((~Bi) & Bo);
        Esi = Bi ^ ((~Bo) & Bu);
        Eso = Bo ^ ((~Bu) & Ba);
        Esu = Bu ^ ((~Ba) & Be);

        // Second unrolled round
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
        Aba = Ba ^ ((~Be) & Bi) ^ RC1600[round_idx + 1];
        Abe = Be ^ ((~Bi) & Bo);
        Abi = Bi ^ ((~Bo) & Bu);
        Abo = Bo ^ ((~Bu) & Ba);
        Abu = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Ebo ^ Do, 28);
        Be = ROL64(Egu ^ Du, 20);
        Bi = ROL64(Eka ^ Da, 3);
        Bo = ROL64(Eme ^ De, 45);
        Bu = ROL64(Esi ^ Di, 61);
        Aga = Ba ^ ((~Be) & Bi);
        Age = Be ^ ((~Bi) & Bo);
        Agi = Bi ^ ((~Bo) & Bu);
        Ago = Bo ^ ((~Bu) & Ba);
        Agu = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Ebe ^ De, 1);
        Be = ROL64(Egi ^ Di, 6);
        Bi = ROL64(Eko ^ Do, 25);
        Bo = ROL64(Emu ^ Du, 8);
        Bu = ROL64(Esa ^ Da, 18);
        Aka = Ba ^ ((~Be) & Bi);
        Ake = Be ^ ((~Bi) & Bo);
        Aki = Bi ^ ((~Bo) & Bu);
        Ako = Bo ^ ((~Bu) & Ba);
        Aku = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Ebu ^ Du, 27);
        Be = ROL64(Ega ^ Da, 36);
        Bi = ROL64(Eke ^ De, 10);
        Bo = ROL64(Emi ^ Di, 15);
        Bu = ROL64(Eso ^ Do, 56);
        Ama = Ba ^ ((~Be) & Bi);
        Ame = Be ^ ((~Bi) & Bo);
        Ami = Bi ^ ((~Bo) & Bu);
        Amo = Bo ^ ((~Bu) & Ba);
        Amu = Bu ^ ((~Ba) & Be);

        Ba = ROL64(Ebi ^ Di, 62);
        Be = ROL64(Ego ^ Do, 55);
        Bi = ROL64(Eku ^ Du, 39);
        Bo = ROL64(Ema ^ Da, 41);
        Bu = ROL64(Ese ^ De, 2);
        Asa = Ba ^ ((~Be) & Bi);
        Ase = Be ^ ((~Bi) & Bo);
        Asi = Bi ^ ((~Bo) & Bu);
        Aso = Bo ^ ((~Bu) & Ba);
        Asu = Bu ^ ((~Ba) & Be);
    }

    state[0] = Aba; state[1] = Abe; state[2] = Abi; state[3] = Abo; state[4] = Abu;
    state[5] = Aga; state[6] = Age; state[7] = Agi; state[8] = Ago; state[9] = Agu;
    state[10] = Aka; state[11] = Ake; state[12] = Aki; state[13] = Ako; state[14] = Aku;
    state[15] = Ama; state[16] = Ame; state[17] = Ami; state[18] = Amo; state[19] = Amu;
    state[20] = Asa; state[21] = Ase; state[22] = Asi; state[23] = Aso; state[24] = Asu;
}

__device__ void keccak512_hash(const uint8_t *in, int inlen, uint8_t *out) {
    uint64_t state[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) state[i] = 0;

    int r = 72; // rate in bytes for Keccak-512 (576 bits)
    int full_blocks = inlen / r;
    int rem = inlen % r;

    for (int b = 0; b < full_blocks; b++) {
        const uint64_t *in64 = (const uint64_t*)(in + b * r);
        #pragma unroll
        for (int i = 0; i < 9; i++) state[i] ^= in64[i];
        keccak_f1600(state);
    }

    uint8_t block[72];
    #pragma unroll
    for (int i = 0; i < 72; i++) block[i] = 0;
    for (int i = 0; i < rem; i++) block[i] = in[full_blocks * r + i];
    block[rem] ^= 0x01;
    block[r - 1] ^= 0x80;

    const uint64_t *b64 = (const uint64_t*)block;
    #pragma unroll
    for (int i = 0; i < 9; i++) state[i] ^= b64[i];
    keccak_f1600(state);

    uint64_t *out64 = (uint64_t*)out;
    #pragma unroll
    for (int i = 0; i < 8; i++) out64[i] = state[i];
}

extern "C" __global__ void build_light_cache_gpu(const uint8_t *seed, uint8_t *cache, uint32_t num_items) {
    keccak512_hash(seed, 32, cache);
    for (uint32_t i = 1; i < num_items; i++) {
        keccak512_hash(cache + (uint64_t)(i - 1) * 64ULL, 64, cache + (uint64_t)i * 64ULL);
    }

    for (int round_cnt = 0; round_cnt < 3; round_cnt++) {
        for (uint32_t i = 0; i < num_items; i++) {
            uint32_t v = (*(uint32_t*)(cache + (uint64_t)i * 64ULL)) % num_items;
            uint32_t prev_idx = (i == 0) ? (num_items - 1) : (i - 1);
            uint8_t temp[64];
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                temp[k] = cache[(uint64_t)prev_idx * 64ULL + k] ^ cache[(uint64_t)v * 64ULL + k];
            }
            keccak512_hash(temp, 64, cache + (uint64_t)i * 64ULL);
        }
    }
}

extern "C" __global__ void generate_dag_kernel_keccak(const uint8_t *cache, uint32_t num_cache_items,
                                                      uint8_t *dag, uint32_t start_item, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t item_index = start_item + idx;

    uint32_t mix[16];
    uint32_t parent_idx = item_index % num_cache_items;
    const uint32_t *c_ptr = (const uint32_t*)(cache + (uint64_t)parent_idx * 64ULL);
    #pragma unroll
    for (int i = 0; i < 16; i++) mix[i] = c_ptr[i];
    mix[0] ^= item_index;

    uint8_t temp[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) ((uint32_t*)temp)[i] = mix[i];
    uint8_t init_digest[64];
    keccak512_hash(temp, 64, init_digest);
    #pragma unroll
    for (int i = 0; i < 16; i++) mix[i] = ((uint32_t*)init_digest)[i];

    #pragma unroll 16
    for (uint32_t p = 0; p < 512; p++) {
        uint32_t p_cache_idx = fnv1(item_index ^ p, mix[p % 16]) % num_cache_items;
        const uint32_t *p_ptr = (const uint32_t*)(cache + (uint64_t)p_cache_idx * 64ULL);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            mix[i] = fnv1(mix[i], p_ptr[i]);
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) ((uint32_t*)temp)[i] = mix[i];
    keccak512_hash(temp, 64, dag + ((uint64_t)item_index * 64ULL));
}
"""

def is_odd_prime(n):
    d = 3
    while d * d <= n:
        if n % d == 0: return False
        d += 2
    return True

def find_largest_prime(n):
    if n < 2: return 0
    if n == 2: return 2
    if n % 2 == 0: n -= 1
    while not is_odd_prime(n):
        n -= 2
    return n

def get_cache_num_items(epoch):
    upper = (1 << 24) // 64 + epoch * ((1 << 17) // 64)
    return find_largest_prime(upper)

def get_dataset_num_items(epoch):
    upper = (1 << 30) // 128 + epoch * ((1 << 23) // 128)
    return find_largest_prime(upper)

def compile_dag_engine(device):
    cc = device.compute_capability()
    arch = f"sm_{cc[0]}{cc[1]}"

    err, prog = nvrtc.nvrtcCreateProgram(DAG_CUDA_SOURCE.encode(), b"dag_engine.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture={arch}".encode()])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC compilation failed:\n{log.decode().strip()}")

    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    return mod

def compile_jit_kernel(device, period):
    cc = device.compute_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    t0 = time.time()
    src = build_jit_kernel_source(period)
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"jit_search.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture={arch}".encode()])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, sz = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * sz
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC JIT compilation failed for period {period}:\n{log.decode()}")

    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    k_fn = mod.get_function("kawpow_search_jit")
    return k_fn
