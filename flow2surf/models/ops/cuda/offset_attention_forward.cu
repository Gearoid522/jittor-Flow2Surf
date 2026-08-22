const int tile = out2_shape1;
constexpr int threads = 256;
const int batch = in0_shape0;
const int points = in0_shape1;
const int width = in0_shape2;
const int channels = in2_shape2;
cublasHandle_t &handle = cublas_handle;

for (int key_base = 0; key_base < points; key_base += tile) {
    const int active = min(tile, points - key_base);
    // S = Q K for the active key tile. Column-major output keeps each
    // key's query column contiguous for softmax.
    offset_batched_gemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        points,
        active,
        width,
        in0_p,
        width,
        points * width,
        in1_p + key_base,
        points,
        width * points,
        0.0f,
        out2_p,
        points,
        points * tile,
        batch
    );
    normalize_offset_columns<<<batch * active, threads>>>(
        out2_p, points, tile, active
    );

    const float beta = key_base == 0 ? 0.0f : 1.0f;
    // z += P 1 and A += P V are accumulated over key tiles.
    offset_batched_gemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        points,
        1,
        active,
        out2_p,
        points,
        points * tile,
        in3_p,
        active,
        0,
        beta,
        out1_p,
        points,
        points,
        batch
    );
    offset_batched_gemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        channels,
        points,
        active,
        in2_p + key_base * channels,
        channels,
        points * channels,
        out2_p,
        points,
        points * tile,
        beta,
        out0_p,
        channels,
        points * channels,
        batch
    );
}

const int rows = batch * points;
normalize_offset_rows<<<(rows * channels + threads - 1) / threads, threads>>>(
    out0_p, out1_p, rows, channels
);
