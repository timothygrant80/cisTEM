#ifndef __INCLUDE_DETAIL_FFT_PROPERTIES_H__
#define __INCLUDE_DETAIL_FFT_PROPERTIES_H__

namespace FastFFT {

typedef struct __align__(8) _FFT_Size {
    // Following Sorensen & Burrus 1993 for clarity
    short N; // N : 1d FFT size
    short L; // L : number of non-zero output/input points
    short P; // P >= L && N % P == 0 : The size of the sub-FFT used to compute the full transform. Currently also must be a power of 2.
    short Q; // Q = N/P : The number of sub-FFTs used to compute the full transform
}

FFT_Size;

typedef struct __align__(8) _Offsets {
    unsigned short shared_input;
    unsigned short shared_output;
    unsigned short physical_x_input;
    unsigned short physical_x_output;
}

Offsets;

typedef struct __align__(64) _LaunchParams {
    int     Q;
    float   twiddle_in;
    dim3    gridDims;
    dim3    threadsPerBlock;
    Offsets mem_offsets;
}

LaunchParams;

} // namespace FastFFT

#endif