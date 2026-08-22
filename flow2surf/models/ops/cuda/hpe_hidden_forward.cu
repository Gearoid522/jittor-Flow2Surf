__global__ void hpe_hidden(@ARGS_DEF) {
    @PRECALC
    const int H = out0_shape3;
    const int D = in0_shape3;
    const int row = blockIdx.x;
    const int channel = threadIdx.x;

    __shared__ float sums[256];
    __shared__ float squares[256];

    float value = 0.0f;
    if (channel < H) {
        value = in2_p[channel];
        for (int d = 0; d < D; ++d) {
            value += in0_p[row * D + d] * in1_p[channel * D + d];
        }
    }
    const Flow2SurfSums moments = flow2surf_block_sum_pair(
        {value, value * value},
        sums,
        squares
    );
    if (channel < H) {
        const float mean = moments.first / H;
        const float variance = fmaxf(
            moments.second / H - mean * mean,
            0.0f
        );
        const float normalized = (value - mean) * rsqrtf(variance + 1e-5f);
        const float affine = normalized * in3_p[channel] + in4_p[channel];
        out0_p[row * H + channel] = 0.5f * affine * (
            erff(affine * 0.7071067811865475f) + 1.0f
        );
    }
}

const int threads = out0_shape3 <= 32 ? 32 : (
    out0_shape3 <= 64 ? 64 : (out0_shape3 <= 128 ? 128 : 256)
);
const int rows = out0_shape0 * out0_shape1 * out0_shape2;
hpe_hidden<<<rows, threads>>>(@ARGS);
