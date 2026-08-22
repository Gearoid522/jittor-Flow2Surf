__global__ void clear_hpe_norm_gradients(@ARGS_DEF) {
    @PRECALC
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel < in1_shape0) {
        @out1(channel) = 0.0f;
        @out2(channel) = 0.0f;
    }
}

__global__ void hpe_activation_grad(@ARGS_DEF, const int rows_per_block) {
    @PRECALC
    const int H = in0_shape3;
    const int channel = threadIdx.x;
    const int rows = in0_shape0 * in0_shape1 * in0_shape2;
    __shared__ float sums[256];
    __shared__ float squares[256];

    float weight_gradient = 0.0f;
    float bias_gradient = 0.0f;
    const int first_row = blockIdx.x * rows_per_block;
    const int last_row = min(first_row + rows_per_block, rows);
    for (int row = first_row; row < last_row; ++row) {
        const int offset = row * H + channel;
        const float value = channel < H ? in0_p[offset] : 0.0f;
        const Flow2SurfSums moments = flow2surf_block_sum_pair(
            {value, value * value},
            sums,
            squares
        );

        const float mean = moments.first / H;
        const float raw_variance = moments.second / H - mean * mean;
        const float inv_std = rsqrtf(fmaxf(raw_variance, 0.0f) + 1e-5f);
        float normalized = 0.0f;
        float grad_normalized = 0.0f;
        if (channel < H) {
            normalized = (value - mean) * inv_std;
            const float affine = normalized * @in1(channel) + @in2(channel);
            const float cdf = 0.5f * (
                erff(affine * 0.7071067811865475f) + 1.0f
            );
            const float density = expf(-0.5f * affine * affine)
                * 0.3989422804014327f;
            const float grad_affine = in3_p[offset] * (cdf + affine * density);
            weight_gradient += grad_affine * normalized;
            bias_gradient += grad_affine;
            grad_normalized = grad_affine * @in1(channel);
        }

        const Flow2SurfSums gradients = flow2surf_block_sum_pair(
            {grad_normalized, grad_normalized * normalized},
            sums,
            squares
        );
        if (channel < H) {
            const float variance_gradient = raw_variance > 0.0f
                ? normalized * gradients.second / H
                : 0.0f;
            out0_p[offset] = inv_std * (
                grad_normalized - gradients.first / H - variance_gradient
            );
        }
    }
    if (channel < H) {
        atomicAdd(&@out1(channel), weight_gradient);
        atomicAdd(&@out2(channel), bias_gradient);
    }
}

clear_hpe_norm_gradients<<<(in1_shape0 + 255) / 256, 256>>>(@ARGS);
const int threads = in0_shape3 <= 32 ? 32 : (
    in0_shape3 <= 64 ? 64 : (in0_shape3 <= 128 ? 128 : 256)
);
const int rows = in0_shape0 * in0_shape1 * in0_shape2;
const int rows_per_block = 8;
hpe_activation_grad<<<
    (rows + rows_per_block - 1) / rows_per_block,
    threads
>>>(@ARGS, rows_per_block);
