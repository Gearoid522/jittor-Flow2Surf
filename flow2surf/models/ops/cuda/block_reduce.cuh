#include <cfloat>

struct Flow2SurfSums {
    float first;
    float second;
};

// Preserve tree-reduction order while completing the warp tail in registers.
__device__ __forceinline__ float flow2surf_block_sum(
    float value,
    float* scratch
) {
    const int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (thread < stride) {
            scratch[thread] += scratch[thread + stride];
        }
        __syncthreads();
    }

    if (thread < 32) {
        value = scratch[thread];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (thread == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ Flow2SurfSums flow2surf_block_sum_pair(
    Flow2SurfSums value,
    float* first_scratch,
    float* second_scratch
) {
    const int thread = threadIdx.x;
    first_scratch[thread] = value.first;
    second_scratch[thread] = value.second;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (thread < stride) {
            first_scratch[thread] += first_scratch[thread + stride];
            second_scratch[thread] += second_scratch[thread + stride];
        }
        __syncthreads();
    }

    if (thread < 32) {
        value = {first_scratch[thread], second_scratch[thread]};
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value.first += __shfl_down_sync(
                0xffffffffu,
                value.first,
                offset
            );
            value.second += __shfl_down_sync(
                0xffffffffu,
                value.second,
                offset
            );
        }
        if (thread == 0) {
            first_scratch[0] = value.first;
            second_scratch[0] = value.second;
        }
    }
    __syncthreads();
    return {first_scratch[0], second_scratch[0]};
}
