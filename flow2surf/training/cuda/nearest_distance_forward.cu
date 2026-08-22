__global__ void nearest_first(@ARGS_DEF) {
    @PRECALC
    const int M = in0_shape1;
    const int N = in1_shape1;
    const int query_blocks = (M + 255) / 256;
    const int b = blockIdx.x / query_blocks;
    const int i = (blockIdx.x - b * query_blocks) * 256 + threadIdx.x;
    __shared__ float reference_x[256];
    __shared__ float reference_y[256];
    __shared__ float reference_z[256];

    float best = FLT_MAX;
    int best_index = N;
    for (int start = 0; start < N; start += 256) {
        const int loaded = start + threadIdx.x;
        if (loaded < N) {
            reference_x[threadIdx.x] = @in1(b,loaded,0);
            reference_y[threadIdx.x] = @in1(b,loaded,1);
            reference_z[threadIdx.x] = @in1(b,loaded,2);
        }
        __syncthreads();
        if (i < M) {
            const float x = @in0(b,i,0);
            const float y = @in0(b,i,1);
            const float z = @in0(b,i,2);
            const int count = N - start < 256 ? N - start : 256;
            for (int offset = 0; offset < count; ++offset) {
                const float dx = x - reference_x[offset];
                const float dy = y - reference_y[offset];
                const float dz = z - reference_z[offset];
                const float distance = dx * dx + dy * dy + dz * dz;
                const int reference = start + offset;
                if (distance < best
                    || (distance == best && reference < best_index)) {
                    best = distance;
                    best_index = reference;
                }
            }
        }
        __syncthreads();
    }
    if (i < M) {
        @out0(b,i) = best;
        @out1(b,i) = best_index;
    }
}

__global__ void nearest_second(@ARGS_DEF) {
    @PRECALC
    const int M = in0_shape1;
    const int N = in1_shape1;
    const int query_blocks = (N + 255) / 256;
    const int b = blockIdx.x / query_blocks;
    const int j = (blockIdx.x - b * query_blocks) * 256 + threadIdx.x;
    __shared__ float reference_x[256];
    __shared__ float reference_y[256];
    __shared__ float reference_z[256];

    float best = FLT_MAX;
    int best_index = M;
    for (int start = 0; start < M; start += 256) {
        const int loaded = start + threadIdx.x;
        if (loaded < M) {
            reference_x[threadIdx.x] = @in0(b,loaded,0);
            reference_y[threadIdx.x] = @in0(b,loaded,1);
            reference_z[threadIdx.x] = @in0(b,loaded,2);
        }
        __syncthreads();
        if (j < N) {
            const float x = @in1(b,j,0);
            const float y = @in1(b,j,1);
            const float z = @in1(b,j,2);
            const int count = M - start < 256 ? M - start : 256;
            for (int offset = 0; offset < count; ++offset) {
                const float dx = x - reference_x[offset];
                const float dy = y - reference_y[offset];
                const float dz = z - reference_z[offset];
                const float distance = dx * dx + dy * dy + dz * dz;
                const int reference = start + offset;
                if (distance < best
                    || (distance == best && reference < best_index)) {
                    best = distance;
                    best_index = reference;
                }
            }
        }
        __syncthreads();
    }
    if (j < N) {
        @out2(b,j) = best;
        @out3(b,j) = best_index;
    }
}

nearest_first<<<in0_shape0 * ((in0_shape1 + 255) / 256), 256>>>(@ARGS);
nearest_second<<<in1_shape0 * ((in1_shape1 + 255) / 256), 256>>>(@ARGS);
