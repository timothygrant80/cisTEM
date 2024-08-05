#ifndef __INCLUDE_DETAIL_DEVICE_HELPERS_H__
#define __INCLUDE_DETAIL_DEVICE_HELPERS_H__

#include <cuda_fp16.h>
#include "../cufftdx/include/cufftdx.hpp"

#include "checks_and_debug.h"

namespace FastFFT {

// GetCudaDeviceArch from https://github.com/mnicely/cufft_examples/blob/master/Common/cuda_helper.h
void GetCudaDeviceProps(DeviceProps& dp);

void CheckSharedMemory(int& memory_requested, DeviceProps& dp);
void CheckSharedMemory(unsigned int& memory_requested, DeviceProps& dp);

template <typename T>
inline bool is_pointer_in_memory_and_registered(T ptr) {
    // FIXME: I don't think this is thread safe, add a mutex as in cistem::GpuImage
    cudaPointerAttributes attr;
    cudaErr(cudaPointerGetAttributes(&attr, ptr));

    if ( attr.type == 1 && attr.devicePointer == attr.hostPointer ) {
        return true;
    }
    else {
        return false;
    }
}

template <typename T>
inline bool is_pointer_in_device_memory(T ptr) {
    // FIXME: I don't think this is thread safe, add a mutex as in cistem::GpuImage
    cudaPointerAttributes attr;
    cudaErr(cudaPointerGetAttributes(&attr, ptr));

    if ( attr.type == 2 || attr.type == 3 ) {
        return true;
    }
    else {
        return false;
    }
}

__device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, short4 img_dims) {
    return ((((int)coords.z * (int)img_dims.y + coords.y) * (int)img_dims.w * 2) + (int)coords.x);
}

} // namespace FastFFT

#endif