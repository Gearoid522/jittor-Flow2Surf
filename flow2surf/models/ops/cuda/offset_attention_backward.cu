const int tile = out3_shape1;
constexpr int threads = 256;
const int batch = in0_shape0;
const int points = in0_shape1;
const int width = in0_shape2;
const int channels = in2_shape2;
cublasHandle_t &handle = cublas_handle;

prepare_offset_gradient<<<batch * points, threads>>>(
    in5_p,
    in3_p,
    in4_p,
    out6_p,
    out5_p,
    channels
);

for (int key_base = 0; key_base < points; key_base += tile) {
    const int active = min(tile, points - key_base);
    // Recompute the active probability tile P.
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
        out3_p,
        points,
        points * tile,
        batch
    );
    normalize_offset_columns<<<batch * active, threads>>>(
        out3_p, points, tile, active
    );

    // dV = P^T (G / z).
    offset_batched_gemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        channels,
        active,
        points,
        out6_p,
        channels,
        points * channels,
        out3_p,
        points,
        points * tile,
        0.0f,
        out2_p + key_base * channels,
        channels,
        points * channels,
        batch
    );
    // dP = (G / z) V^T - <G / z, O>. The following kernel applies
    // the column-softmax Jacobian and overwrites P with dS.
    offset_batched_gemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        points,
        active,
        channels,
        out6_p,
        channels,
        points * channels,
        in2_p + key_base * channels,
        channels,
        points * channels,
        0.0f,
        out4_p,
        points,
        points * tile,
        batch
    );
    differentiate_offset_probabilities<<<batch * active, threads>>>(
        out3_p, out4_p, out5_p, points, tile, active
    );

    const float beta = key_base == 0 ? 0.0f : 1.0f;
    // dQ += dS K^T and dK = Q^T dS.
    offset_batched_gemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        width,
        points,
        active,
        in1_p + key_base,
        points,
        width * points,
        out3_p,
        points,
        points * tile,
        beta,
        out0_p,
        width,
        points * width,
        batch
    );
    offset_batched_gemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        active,
        width,
        points,
        out3_p,
        points,
        points * tile,
        in0_p,
        width,
        points * width,
        0.0f,
        out1_p + key_base,
        points,
        width * points,
        batch
    );
}
