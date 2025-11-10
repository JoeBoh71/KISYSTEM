#include <iostream>
#include <cuda_runtime.h>

// Define block size and shared memory size
#define BLOCK_SIZE 256
#define FILTER_LENGTH 256
#define SIGNAL_LENGTH 8192

__global__ void firFilterKernel(const float* __restrict__ input, const float* __restrict__ coefficients, float* __restrict__ output) {
    extern __shared__ float sharedCoefficients[];

    // Load filter coefficients into shared memory
    int coeffIndex = threadIdx.x;
    if (coeffIndex < FILTER_LENGTH) {
        sharedCoefficients[coeffIndex] = coefficients[coeffIndex];
    }
    __syncthreads();

    // Calculate the output for each thread
    int signalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;

    if (signalIndex < SIGNAL_LENGTH) {
        for (int i = 0; i < FILTER_LENGTH && (signalIndex - i) >= 0; ++i) {
            result += input[signalIndex - i] * sharedCoefficients[i];
        }
        output[signalIndex] = result;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate host memory
    float* h_input = new float[SIGNAL_LENGTH];
    float* h_coefficients = new float[FILTER_LENGTH];
    float* h_output = new float[SIGNAL_LENGTH];

    // Initialize input and coefficients (example values)
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    for (int i = 0; i < FILTER_LENGTH; ++i) {
        h_coefficients[i] = 1.0f / FILTER_LENGTH;
    }

    // Allocate device memory
    float* d_input, *d_coefficients, *d_output;
    checkCudaError(cudaMalloc((void**)&d_input, SIGNAL_LENGTH * sizeof(float)), "cudaMalloc input");
    checkCudaError(cudaMalloc((void**)&d_coefficients, FILTER_LENGTH * sizeof(float)), "cudaMalloc coefficients");
    checkCudaError(cudaMalloc((void**)&d_output, SIGNAL_LENGTH * sizeof(float)), "cudaMalloc output");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_input, h_input, SIGNAL_LENGTH * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy input");
    checkCudaError(cudaMemcpy(d_coefficients, h_coefficients, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy coefficients");

    // Launch kernel
    int numBlocks = (SIGNAL_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE;
    firFilterKernel<<<numBlocks, BLOCK_SIZE, FILTER_LENGTH * sizeof(float)>>>(d_input, d_coefficients, d_output);
    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy result from device to host
    checkCudaError(cudaMemcpy(h_output, d_output, SIGNAL_LENGTH * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_coefficients);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_coefficients;
    delete[] h_output;

    return 0;
}