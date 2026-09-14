import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc
import numpy as np

CUDA_CODE = r"""
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

extern "C" __global__ void test_keccak_seeds(uint64_t *out_first_items) {
    uint64_t seed[4] = {0, 0, 0, 0};
    uint64_t item0[8];
    keccak512_32(seed, item0);
    for (int i = 0; i < 8; i++) out_first_items[i] = item0[i];

    uint64_t item1[8];
    for (int i = 0; i < 8; i++) item1[i] = item0[i];
    keccak512_64(item1);
    for (int i = 0; i < 8; i++) out_first_items[8 + i] = item1[i];
}
"""

def test():
    dev = cuda.Device(0)
    cc = dev.compute_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    err, prog = nvrtc.nvrtcCreateProgram(CUDA_CODE.encode(), b"test.cu", 0, [], [])
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
    fn = mod.get_function("test_keccak_seeds")

    out_gpu = cuda.mem_alloc(16 * 8)
    fn(out_gpu, block=(1, 1, 1), grid=(1, 1))
    out_cpu = np.zeros(16, dtype=np.uint64)
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    print("Item 0 hex:", bytes(out_cpu[:8]).hex())
    print("Item 1 hex:", bytes(out_cpu[8:]).hex())

if __name__ == '__main__':
    test()
