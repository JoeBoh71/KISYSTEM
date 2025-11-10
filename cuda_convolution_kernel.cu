#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

#define BLOCK_SIZE 1024
#define FILTER_LENGTH 2048
#define FFT_SIZE 4096
#define OVERLAP FILTER_LENGTH

__global__ void overlap_add_kernel(float* output, float* buffer, cufftComplex* fft_output, int num_blocks) {
    extern __shared__ float shared_buffer[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Load data into shared memory
    if (tid < BLOCK_SIZE) {
        shared_buffer[tid] = buffer[bid * OVERLAP + tid];
    }
    __syncthreads();

    // Copy FFT output to global memory
    if (tid < BLOCK_SIZE) {
        output[bid * BLOCK_SIZE + tid] = fft_output[bid * (FFT_SIZE / 2 + 1)].x;
    }

    // Perform overlap-add
    if (bid > 0 && tid < OVERLAP) {
        atomicAdd(&output[(bid - 1) * BLOCK_SIZE + tid], shared_buffer[tid]);
    }
}

int main() {
    float* h_input = new float[BLOCK_SIZE];
    float* h_filter = new float[FILTER_LENGTH];
    float* h_output = new float[BLOCK_SIZE];

    // Initialize input and filter with some values
    for (int i = 0; i < BLOCK_SIZE; ++i) h_input[i] = 1.0f;
    for (int i = 0; i < FILTER_LENGTH; ++i) h_filter[i] = 1.0f / FILTER_LENGTH;

    // Allocate device memory
    float* d_input, *d_filter, *d_buffer, *d_output;
    cufftComplex* d_fft_input, *d_fft_filter, *d_fft_output;

    cudaMalloc(&d_input, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_filter, FILTER_LENGTH * sizeof(float));
    cudaMalloc(&d_buffer, OVERLAP * sizeof(float) * 2);
    cudaMalloc(&d_output, BLOCK_SIZE * sizeof(float));

    cudaMalloc(&d_fft_input, (FFT_SIZE / 2 + 1) * sizeof(cufftComplex));
    cudaMalloc(&d_fft_filter, (FFT_SIZE / 2 + 1) * sizeof(cufftComplex));
    cudaMalloc(&d_fft_output, (FFT_SIZE / 2 + 1) * sizeof(cufftComplex));

    // Copy data to device
    cudaMemcpy(d_input, h_input, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuFFT plans
    cufftHandle plan_forward_input, plan_forward_filter, plan_inverse;
    cufftPlan1d(&plan_forward_input, FFT_SIZE, CUFFT_R2C, 1);
    cufftPlan1d(&plan_forward_filter, FFT_SIZE, CUFFT_R2C, 1);
    cufftPlan1d(&plan_inverse, FFT_SIZE, CUFFT_C2R, 1);

    // Perform FFT on input and filter
    cufftExecR2C(plan_forward_input, (cufftReal*)d_input, d_fft_input);
    cufftExecR2C(plan_forward_filter, (cufftReal*)d_filter, d_fft_filter);

    // Multiply in frequency domain
    int num_blocks = 1; // Assuming single block for simplicity
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(num_blocks);
    size_t shared_mem_size = OVERLAP * sizeof(float);

    overlap_add_kernel<<<blocks, threads, shared_mem_size>>>(d_output, d_buffer, d_fft_output, num_blocks);
    cudaDeviceSynchronize();

    // Perform IFFT
    cufftExecC2R(plan_inverse, d_fft_output, (cufftReal*)d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_buffer);
    cudaFree(d_output);
    cudaFree(d_fft_input);
    cudaFree(d_fft_filter);
    cudaFree(d_fft_output);

    // Destroy cuFFT plans
    cufftDestroy(plan_forward_input);
    cufftDestroy(plan_forward_filter);
    cufftDestroy(plan_inverse);

    // Free host memory
    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;

    return 0;
}
```

**Note:** The provided code is a simplified version and assumes a single block for simplicity. For real-time processing, you would need to handle multiple blocks and ensure efficient memory management and synchronization. Additionally, error checking should be added for production use.