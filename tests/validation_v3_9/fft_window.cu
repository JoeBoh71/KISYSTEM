#include <cuda_runtime.h>
#include <stdio.h>
#include <math_constants.h>

#define BLOCK_SIZE 256

__device__ float computeHannCoefficient(int n, int N) {
    return 0.5f * (1.0f - cosf(2.0f * CUDART_PI_F * n / (N - 1)));
}

__global__ void hannWindowKernel(float* input, float* output, int N) {
    extern __shared__ float sharedWindow[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load window coefficients into shared memory
    if (tid < N) {
        sharedWindow[tid] = computeHannCoefficient(tid, N);
    }
    __syncthreads();

    // Apply window to input signal
    if (idx < N) {
        output[idx] = input[idx] * sharedWindow[tid];
    }
}

int main() {
    int N = 1024; // Example FFT size
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize input signal with some values
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hannWindowKernel<<<blocksPerGrid, threadsPerBlock, N * sizeof(float)>>>(d_input, d_output, N);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}