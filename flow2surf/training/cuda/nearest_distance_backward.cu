__global__ void clear_nearest_gradients(@ARGS_DEF) {
    @PRECALC
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int first_size = in0_shape0 * in0_shape1 * in0_shape2;
    const int second_size = in1_shape0 * in1_shape1 * in1_shape2;
    if (index < first_size) out0_p[index] = 0.0f;
    if (index < second_size) out1_p[index] = 0.0f;
}

__global__ void nearest_first_grad(@ARGS_DEF) {
    @PRECALC
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int M = in0_shape1;
    const int count = in0_shape0 * M;
    if (index >= count) return;
    const int b = index / M;
    const int i = index - b * M;
    const int j = @in2(b,i);
    const float scale = 2.0f * @in4(b,i);
    for (int c = 0; c < 3; ++c) {
        const float gradient = scale * (@in0(b,i,c) - @in1(b,j,c));
        atomicAdd(&@out0(b,i,c), gradient);
        atomicAdd(&@out1(b,j,c), -gradient);
    }
}

__global__ void nearest_second_grad(@ARGS_DEF) {
    @PRECALC
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = in1_shape1;
    const int count = in1_shape0 * N;
    if (index >= count) return;
    const int b = index / N;
    const int j = index - b * N;
    const int i = @in3(b,j);
    const float scale = 2.0f * @in5(b,j);
    for (int c = 0; c < 3; ++c) {
        const float gradient = scale * (@in1(b,j,c) - @in0(b,i,c));
        atomicAdd(&@out1(b,j,c), gradient);
        atomicAdd(&@out0(b,i,c), -gradient);
    }
}

const int first_size = in0_shape0 * in0_shape1 * in0_shape2;
const int second_size = in1_shape0 * in1_shape1 * in1_shape2;
const int clear_size = first_size > second_size ? first_size : second_size;
clear_nearest_gradients<<<(clear_size + 255) / 256, 256>>>(@ARGS);
const int first_count = in0_shape0 * in0_shape1;
const int second_count = in1_shape0 * in1_shape1;
nearest_first_grad<<<(first_count + 255) / 256, 256>>>(@ARGS);
nearest_second_grad<<<(second_count + 255) / 256, 256>>>(@ARGS);
