#include <cuda_runtime.h>
#include <stdio.h>

__global__ void addKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Initialize host arrays
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2.0f;
    }
    
    // Copy host arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < n; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}