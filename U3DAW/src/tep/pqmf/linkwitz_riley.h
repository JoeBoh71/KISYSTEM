#ifndef LINKWITZ_RILEY_H
#define LINKWITZ_RILEY_H

#include <cufft.h>

class LinkwitzRileyFilter {
public:
    LinkwitzRileyFilter(int fft_size = 4096);
    void process(const float* input, float* bands[4], int n);
private:
    cufftHandle plan_fwd, plan_inv;
};

#endif
