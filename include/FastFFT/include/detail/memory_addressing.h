#ifndef __INCLUDE_DETAIL_MEMORY_ADDRESSING_H__
#define __INCLUDE_DETAIL_MEMORY_ADDRESSING_H__

#include <cuda_fp16.h>
#include "../cufftdx/include/detail/system_checks.hpp"

namespace FastFFT {

static constexpr const int XZ_STRIDE = 16;

static constexpr const int          bank_size   = 32;
static constexpr const int          bank_padded = bank_size + 1;
static constexpr const unsigned int ubank_size  = 32;

static constexpr const unsigned int ubank_padded = ubank_size + 1;

__device__ __forceinline__ int GetSharedMemPaddedIndex(const int index) {
    return (index % bank_size) + ((index / bank_size) * bank_padded);
}

__device__ __forceinline__ int GetSharedMemPaddedIndex(const unsigned int index) {
    return (index % ubank_size) + ((index / ubank_size) * ubank_padded);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress(const unsigned int pixel_pitch) {
    return pixel_pitch * (blockIdx.y + blockIdx.z * gridDim.y);
}

// Return the address of the 1D transform index 0. Right now testing for a stride of 2, but this could be modifiable if it works.
static __device__ __forceinline__ unsigned int Return1DFFTAddress_strided_Z(const unsigned int pixel_pitch) {
    // In the current condition, threadIdx.y is either 0 || 1, and gridDim.z = size_z / 2
    // index into a 2D tile in the XZ plane, for output in the ZX transposed plane (for coalsced write.)
    return pixel_pitch * (blockIdx.y + (XZ_STRIDE * blockIdx.z + threadIdx.y) * gridDim.y);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int ReturnZplane(const unsigned int NX, const unsigned int NY) {
    return (blockIdx.z * NX * NY);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_Z(const unsigned int NY) {
    return blockIdx.y + (blockIdx.z * NY);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTColumn_XYZ_transpose(const unsigned int NX) {
    // NX should be size_of<FFT>::value for this method. Should this be templated?
    // presumably the XZ axis is alread transposed on the forward, used to index into this state. Indexs in (ZY)' plane for input, to be transposed and permuted to output.'
    return NX * (XZ_STRIDE * (blockIdx.y + gridDim.y * blockIdx.z) + threadIdx.y);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_XZ_transpose(const unsigned int X) {
    return blockIdx.z + gridDim.z * (blockIdx.y + X * gridDim.y);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_XZ_transpose_strided_Z(const unsigned int IDX) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    // (IDX % XZ_STRIDE) -> transposed x coordinate in tile
    // ((blockIdx.z*XZ_STRIDE) -> tile offest in physical X (with above gives physical X out (transposed Z))
    // (XZ_STRIDE*gridDim.z) -> n elements in physical X (transposed Z)
    // above * blockIdx.y -> offset in physical Y (transposed Y)
    // (IDX / XZ_STRIDE) -> n elements physical Z (transposed X)
    return ((IDX % XZ_STRIDE) + (blockIdx.z * XZ_STRIDE)) + (XZ_STRIDE * gridDim.z) * (blockIdx.y + (IDX / XZ_STRIDE) * gridDim.y);
}

static __device__ __forceinline__ unsigned int Return1DFFTAddress_XZ_transpose_strided_Z(const unsigned int IDX, const unsigned int Q, const unsigned int sub_fft) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    // (IDX % XZ_STRIDE) -> transposed x coordinate in tile
    // ((blockIdx.z*XZ_STRIDE) -> tile offest in physical X (with above gives physical X out (transposed Z))
    // (XZ_STRIDE*gridDim.z) -> n elements in physical X (transposed Z)
    // above * blockIdx.y -> offset in physical Y (transposed Y)
    // (IDX / XZ_STRIDE) -> n elements physical Z (transposed X)
    return ((IDX % XZ_STRIDE) + (blockIdx.z * XZ_STRIDE)) + (XZ_STRIDE * gridDim.z) * (blockIdx.y + ((IDX / XZ_STRIDE) * Q + sub_fft) * gridDim.y);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_YZ_transpose_strided_Z(const unsigned int IDX) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    return ((IDX % XZ_STRIDE) + (blockIdx.y * XZ_STRIDE)) + (gridDim.y * XZ_STRIDE) * (blockIdx.z + (IDX / XZ_STRIDE) * gridDim.z);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTAddress_YZ_transpose_strided_Z(const unsigned int IDX, const unsigned int Q, const unsigned int sub_fft) {
    // return (XZ_STRIDE*blockIdx.z + (X % XZ_STRIDE)) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + (X / XZ_STRIDE) * gridDim.y );
    return ((IDX % XZ_STRIDE) + (blockIdx.y * XZ_STRIDE)) + (gridDim.y * XZ_STRIDE) * (blockIdx.z + ((IDX / XZ_STRIDE) * Q + sub_fft) * gridDim.z);
}

// Return the address of the 1D transform index 0
static __device__ __forceinline__ unsigned int Return1DFFTColumn_XZ_to_XY( ) {
    // return blockIdx.y + gridDim.y * ( blockIdx.z + gridDim.z * X);
    return blockIdx.y + gridDim.y * blockIdx.z;
}

static __device__ __forceinline__ unsigned int Return1DFFTAddress_YX_to_XY( ) {
    return blockIdx.z + gridDim.z * blockIdx.y;
}

static __device__ __forceinline__ unsigned int Return1DFFTAddress_YX( ) {
    return Return1DFFTColumn_XZ_to_XY( );
}
} // namespace FastFFT

#endif