__global__ void neighbor_reduce(@ARGS_DEF) {
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

    float best = -FLT_MAX;
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
        if (channel < C) {
            const float normalized = centered * rsqrtf(variance + 1e-6f);
            const float affine = normalized * @in3(channel) + @in4(channel);
            const float activated = affine > 0.0f ? affine : 0.2f * affine;
            if (activated > best) best = activated;
        }
    }
    if (channel < C) @out0(b,channel,n) = best;
}

const int threads = in1_shape3 <= 64 ? 64 : (
    in1_shape3 <= 128 ? 128 : 256
);
neighbor_reduce<<<in0_shape0 * in0_shape2, threads>>>(@ARGS);
