__global__ void clear_neighbor_gradients(@ARGS_DEF) {
    @PRECALC
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = in1_shape3;
    const int N = in0_shape2;
    const int neighbor_size = in0_shape0 * C * N;
    if (index < neighbor_size) {
        const int b = index / (C * N);
        const int remainder = index - b * C * N;
        const int channel = remainder / N;
        const int point = remainder - channel * N;
        @out0(b,channel,point) = 0.0f;
    }
    if (index < in3_shape0) {
        @out2(index) = 0.0f;
        @out3(index) = 0.0f;
    }
}

__global__ void neighbor_reduce_grad(@ARGS_DEF) {
    @PRECALC
    const int C = in1_shape3;
    const int N = in0_shape2;
    const int K = in1_shape2;
    const int bn = blockIdx.x;
    const int b = bn / N;
    const int n = bn - b * N;
    const int channel = threadIdx.x;
    __shared__ float sums[256];
    __shared__ float squares[256];
    __shared__ float grad_sums[256];
    __shared__ float grad_norm_sums[256];

    float center_gradient = 0.0f;
    float norm_weight_gradient = 0.0f;
    float norm_bias_gradient = 0.0f;
    for (int edge = 0; edge < K; ++edge) {
        float value = 0.0f;
        if (channel < C) {
            const int neighbor = @in2(b,n,edge);
            value = @in0(b,channel,neighbor)
                + @in0(b,C+channel,n)
                + @in1(b,n,edge,channel);
        }
        const float mean = flow2surf_block_sum(value, sums) / C;
        const float centered = value - mean;
        const float square = channel < C ? centered * centered : 0.0f;
        const float variance = flow2surf_block_sum(square, squares) / C;

        float normalized = 0.0f;
        float upstream = 0.0f;
        float inv_std = 0.0f;
        if (channel < C) {
            inv_std = rsqrtf(variance + 1e-6f);
            normalized = centered * inv_std;
            const float affine = normalized * @in3(channel) + @in4(channel);
            const float activated = affine > 0.0f ? affine : 0.2f * affine;
            const float slope = affine > 0.0f ? 1.0f : 0.2f;
            // Jittor max backward routes gradients to every exact tie.
            if (activated == @in5(b,channel,n)) {
                upstream = @in6(b,channel,n) * slope;
            }
            norm_weight_gradient += upstream * normalized;
            norm_bias_gradient += upstream;
        }
        const float weighted = channel < C ? upstream * @in3(channel) : 0.0f;
        const Flow2SurfSums gradients = flow2surf_block_sum_pair(
            {weighted, weighted * normalized},
            grad_sums,
            grad_norm_sums
        );
        if (channel < C) {
            const float gradient = inv_std * (
                weighted - gradients.first / C
                - normalized * gradients.second / C
            );
            const int neighbor = @in2(b,n,edge);
            @out1(b,n,edge,channel) = gradient;
            atomicAdd(&@out0(b,channel,neighbor), gradient);
            center_gradient += gradient;
        }
    }
    if (channel < C) {
        @out0(b,C+channel,n) = center_gradient;
        atomicAdd(&@out2(channel), norm_weight_gradient);
        atomicAdd(&@out3(channel), norm_bias_gradient);
    }
}

const int neighbor_size = in0_shape0 * in1_shape3 * in0_shape2;
clear_neighbor_gradients<<<(neighbor_size + 255) / 256, 256>>>(@ARGS);
const int threads = in1_shape3 <= 64 ? 64 : (
    in1_shape3 <= 128 ? 128 : 256
);
neighbor_reduce_grad<<<in0_shape0 * in0_shape2, threads>>>(@ARGS);
