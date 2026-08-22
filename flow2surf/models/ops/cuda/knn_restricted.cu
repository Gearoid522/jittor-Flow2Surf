__global__ void restricted_knn(@ARGS_DEF) {
    @PRECALC
    __shared__ unsigned long long warp_keys[8];

    const int C = in0_shape1;
    const int N = in0_shape2;
    const int P = in1_shape2;
    const int K = out0_shape2;
    const int bq = blockIdx.x;
    const int b = bq / N;
    const int q = bq - b * N;
    const int thread = threadIdx.x;

    unsigned long long key = FLOW2SURF_INVALID_NEIGHBOR;
    if (thread < P) {
        const int reference = @in1(b,q,thread);
        float query_norm = 0.0f;
        float reference_norm = 0.0f;
        float inner = 0.0f;
        for (int channel = 0; channel < C; ++channel) {
            const float query = @in0(b,channel,q);
            const float value = @in0(b,channel,reference);
            query_norm += query * query;
            reference_norm += value * value;
            inner += query * value;
        }
        const float distance = query_norm + reference_norm - 2.0f * inner;
        key = flow2surf_neighbor_key(distance, thread);
    }

    for (int rank = 0; rank < K; ++rank) {
        const unsigned long long selected = flow2surf_block_min(key, warp_keys);
        if (thread == 0) {
            const int position = static_cast<int>(selected & 0xffffffffu);
            @out0(b,q,rank) = @in1(b,q,position);
        }
        if (key == selected) key = FLOW2SURF_INVALID_NEIGHBOR;
    }
}

int threads = 32;
while (threads < in1_shape2) threads *= 2;
restricted_knn<<<in0_shape0 * in0_shape2, threads>>>(@ARGS);
