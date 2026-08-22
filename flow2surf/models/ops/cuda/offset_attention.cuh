#pragma once

#include <cfloat>
#include <cmath>

#include "cuda/cublas/inc/cublas_wrapper.h"

static inline void offset_batched_gemm(
    cublasHandle_t &handle,
    cublasOperation_t left_op,
    cublasOperation_t right_op,
    int rows,
    int columns,
    int inner,
    const float *left,
    int left_leading,
    long long left_stride,
    const float *right,
    int right_leading,
    long long right_stride,
    float beta,
    float *output,
    int output_leading,
    long long output_stride,
    int batch
) {
    const float alpha = 1.0f;
    checkCudaErrors(cublasGemmStridedBatchedEx(
        handle,
        left_op,
        right_op,
        rows,
        columns,
        inner,
        &alpha,
        left,
        CUDA_R_32F,
        left_leading,
        left_stride,
        right,
        CUDA_R_32F,
        right_leading,
        right_stride,
        &beta,
        output,
        CUDA_R_32F,
        output_leading,
        output_stride,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

__device__ __forceinline__ float offset_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float offset_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void normalize_offset_columns(
    float *scores,
    int points,
    int tile,
    int active
) {
    const int column = blockIdx.x;
    const int b = column / active;
    const int key = column - b * active;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float warp_values[8];
    __shared__ float column_max;
    __shared__ float inverse_sum;

    float local_max = -FLT_MAX;
    for (int query = threadIdx.x; query < points; query += blockDim.x) {
        const int offset = (b * tile + key) * points + query;
        local_max = fmaxf(local_max, scores[offset]);
    }
    local_max = offset_warp_max(local_max);
    if (lane == 0) warp_values[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? warp_values[lane] : -FLT_MAX;
        value = offset_warp_max(value);
        if (lane == 0) column_max = value;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int query = threadIdx.x; query < points; query += blockDim.x) {
        const int offset = (b * tile + key) * points + query;
        const float probability = expf(scores[offset] - column_max);
        scores[offset] = probability;
        local_sum += probability;
    }
    local_sum = offset_warp_sum(local_sum);
    if (lane == 0) warp_values[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? warp_values[lane] : 0.0f;
        value = offset_warp_sum(value);
        if (lane == 0) inverse_sum = 1.0f / value;
    }
    __syncthreads();

    for (int query = threadIdx.x; query < points; query += blockDim.x) {
        const int offset = (b * tile + key) * points + query;
        scores[offset] *= inverse_sum;
    }
}

__global__ void normalize_offset_rows(
    float *output,
    const float *row_sums,
    int rows,
    int channels
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * channels) {
        output[index] /= row_sums[index / channels] + 1e-9f;
    }
}

__global__ void prepare_offset_gradient(
    const float *gradient,
    const float *output,
    const float *row_sums,
    float *scaled_gradient,
    float *row_inner,
    int channels
) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int warp = thread >> 5;
    __shared__ float warp_values[8];
    __shared__ float inverse_sum;

    if (thread == 0) inverse_sum = 1.0f / (row_sums[row] + 1e-9f);
    __syncthreads();

    float product = 0.0f;
    for (int channel = thread; channel < channels; channel += blockDim.x) {
        const int index = row * channels + channel;
        const float scaled = gradient[index] * inverse_sum;
        scaled_gradient[index] = scaled;
        product += scaled * output[index];
    }
    product = offset_warp_sum(product);
    if (lane == 0) warp_values[warp] = product;
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? warp_values[lane] : 0.0f;
        value = offset_warp_sum(value);
        if (lane == 0) row_inner[row] = value;
    }
}

__global__ void differentiate_offset_probabilities(
    float *probabilities,
    const float *dot_products,
    const float *row_inner,
    int points,
    int tile,
    int active
) {
    const int column = blockIdx.x;
    const int b = column / active;
    const int key = column - b * active;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float warp_values[8];
    __shared__ float column_inner;

    float local_sum = 0.0f;
    for (int query = threadIdx.x; query < points; query += blockDim.x) {
        const int offset = (b * tile + key) * points + query;
        const float grad_probability =
            dot_products[offset] - row_inner[b * points + query];
        local_sum += probabilities[offset] * grad_probability;
    }
    local_sum = offset_warp_sum(local_sum);
    if (lane == 0) warp_values[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < 8 ? warp_values[lane] : 0.0f;
        value = offset_warp_sum(value);
        if (lane == 0) column_inner = value;
    }
    __syncthreads();

    for (int query = threadIdx.x; query < points; query += blockDim.x) {
        const int offset = (b * tile + key) * points + query;
        const float grad_probability =
            dot_products[offset] - row_inner[b * points + query];
        probabilities[offset] *= grad_probability - column_inner;
    }
}
