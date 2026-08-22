#include <cfloat>

static constexpr unsigned long long FLOW2SURF_INVALID_NEIGHBOR =
    0xffffffffffffffffull;

// Encode lexicographic (distance, index) order in one shuffleable value.
__device__ __forceinline__ unsigned long long flow2surf_neighbor_key(
    float distance,
    unsigned int index
) {
    const float canonical_distance = distance == 0.0f ? 0.0f : distance;
    const unsigned int bits = __float_as_uint(canonical_distance);
    const unsigned int ordered = bits ^ (
        (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u
    );
    return (static_cast<unsigned long long>(ordered) << 32) | index;
}

__device__ __forceinline__ unsigned long long flow2surf_warp_min(
    unsigned long long key
) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const unsigned long long candidate = __shfl_down_sync(
            0xffffffffu,
            key,
            offset
        );
        key = candidate < key ? candidate : key;
    }
    return key;
}

__device__ __forceinline__ unsigned long long flow2surf_block_min(
    unsigned long long key,
    unsigned long long* warp_keys
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    key = flow2surf_warp_min(key);
    if (blockDim.x == 32) return __shfl_sync(0xffffffffu, key, 0);

    if (lane == 0) warp_keys[warp] = key;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = blockDim.x >> 5;
        key = lane < warp_count
            ? warp_keys[lane]
            : FLOW2SURF_INVALID_NEIGHBOR;
        key = flow2surf_warp_min(key);
        if (lane == 0) warp_keys[0] = key;
    }
    __syncthreads();
    return warp_keys[0];
}
