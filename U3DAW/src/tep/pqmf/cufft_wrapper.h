#ifndef CUFFT_WRAPPER_H
#define CUFFT_WRAPPER_H

#include <cufft.h>
#include <stdexcept>

class CUFFTWrapper {
public:
    CUFFTWrapper(int size, cufftType type) : size_(size), type_(type) {
        cufftResult result = cufftPlan1d(&plan_, size_, type_, 1);
        if (result != CUFFT_SUCCESS) {
            throw std::runtime_error("cuFFT plan creation failed");
        }
    }
    
    ~CUFFTWrapper() {
        cufftDestroy(plan_);
    }
    
    // Forward transform (real to complex)
    void executeR2C(const float* input, cufftComplex* output) {
        cufftExecR2C(plan_, const_cast<float*>(input), output);
    }
    
    // Inverse transform (complex to real)
    void executeC2R(const cufftComplex* input, float* output) {
        cufftExecC2R(plan_, const_cast<cufftComplex*>(input), output);
    }
    
    cufftHandle getPlan() const { return plan_; }
    
private:
    cufftHandle plan_;
    int size_;
    cufftType type_;
};

#endif // CUFFT_WRAPPER_H
