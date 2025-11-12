#ifndef OVERLAP_SAVE_H
#define OVERLAP_SAVE_H

#include <vector>

class OverlapSave {
public:
    OverlapSave(int fft_size, int hop_size) 
        : fft_size_(fft_size), hop_size_(hop_size) {
        overlap_buffer_.resize(fft_size - hop_size, 0.0f);
    }
    
    void process(const float* input, float* output, int num_samples) {
        // 1. Prepend overlap from previous frame
        std::vector<float> frame(fft_size_);
        std::copy(overlap_buffer_.begin(), overlap_buffer_.end(), frame.begin());
        std::copy(input, input + hop_size_, frame.begin() + overlap_buffer_.size());
        
        // 2. Process frame (FFT -> Filter -> IFFT)
        // [This would call FFT, apply filters, IFFT]
        
        // 3. Save overlap for next iteration
        std::copy(input + hop_size_ - overlap_buffer_.size(), 
                  input + hop_size_, 
                  overlap_buffer_.begin());
        
        // 4. Output only valid samples (discard overlap region)
        std::copy(frame.begin() + overlap_buffer_.size(), 
                  frame.begin() + overlap_buffer_.size() + hop_size_,
                  output);
    }
    
private:
    int fft_size_;
    int hop_size_;
    std::vector<float> overlap_buffer_;
};

#endif // OVERLAP_SAVE_H
