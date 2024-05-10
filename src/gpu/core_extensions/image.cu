

#include <cistem_config.h>

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"

using namespace cistem;

// #include "../../include/ieee-754-half/half.hpp"

/** 
 * @brief Allow the cpu image class to use cuda functions that either allocate page locked memory or register
 * a currently (host) allocated memory region as page locked.
 * 
*/

/**
 * @brief Convenience wrapper to Image::Allocate that also allocates page locked memory. It may be more efficient to use cudaHostAlloc, but since
 * we generally assume that function to be compatible with the alignment produced by fftw_malloc used in the Image class method. 
 * 
 * @param wanted_x_size 
 * @param wanted_y_size 
 * @param wanted_z_size 
 * @param is_in_real_space 
 * @param do_fft_planning 
 */

template <typename StorageBaseType>
void Image::AllocatePageLockedMemory(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool is_in_real_space, bool do_fft_planning) {
    Allocate(wanted_x_size, wanted_y_size, wanted_z_size, is_in_real_space, do_fft_planning);

    if constexpr ( std::is_same_v<StorageBaseType, half_float::half> ) {
        // The current methods require the fp32 memory to be allocated prior to allocating the fp16 memory
        Allocate16fBuffer( );
        RegisterPageLockedMemory(real_values_16f);
        SetIsMemoryPageLocked(real_values_16f, true);
    }
    if constexpr ( std::is_same_v<StorageBaseType, float> ) {
        RegisterPageLockedMemory(real_values);
        SetIsMemoryPageLocked(real_values, true);
    }
}

template void Image::AllocatePageLockedMemory<float>(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool is_in_real_space, bool do_fft_planning);
template void Image::AllocatePageLockedMemory<half_float::half>(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool is_in_real_space, bool do_fft_planning);

template <typename StorageBaseType>
void Image::RegisterPageLockedMemory(StorageBaseType* ptr) {
    wxMutexLocker lock(s_mutexProtectingFFTW); // the mutex will be unlocked when this object is destroyed (when it goes out of scope)
    MyDebugAssertTrue(lock.IsOk( ), "Mute locking failed");

    if ( ! IsMemoryPageLocked(ptr) ) {
        if constexpr ( std::is_same_v<StorageBaseType, half_float::half> ) {
            cudaErr(cudaHostRegister(real_values_16f, sizeof(StorageBaseType) * real_memory_allocated, cudaHostRegisterDefault));
        }
        if constexpr ( std::is_same_v<StorageBaseType, float> ) {
            cudaErr(cudaHostRegister(real_values, sizeof(StorageBaseType) * real_memory_allocated, cudaHostRegisterDefault));
        }
        SetIsMemoryPageLocked(ptr, true);
    }
}

template void Image::RegisterPageLockedMemory<float>(float* ptr);
template void Image::RegisterPageLockedMemory<half_float::half>(half_float::half* ptr);

template <typename StorageBaseType>
void Image::UnRegisterPageLockedMemory(StorageBaseType* ptr) {
    MyDebugAssertTrue(IsMemoryAllocated(ptr), "Image is not in memory");

    if ( IsMemoryPageLocked(ptr) ) {
        wxMutexLocker lock(s_mutexProtectingFFTW); // the mutex will be unlocked when this object is destroyed (when it goes out of scope)
        MyDebugAssertTrue(lock.IsOk( ), "Mute locking failed");
        if constexpr ( std::is_same_v<StorageBaseType, half_float::half> ) {
            cudaErr(cudaHostUnregister(real_values_16f));
        }
        if constexpr ( std::is_same_v<StorageBaseType, float> ) {
            cudaErr(cudaHostUnregister(real_values));
        }
        SetIsMemoryPageLocked(ptr, false);
    }
}

//Note: template <> void Image::<type> does not work, while template void Image::<type> does
template void Image::UnRegisterPageLockedMemory<float>(float* ptr);
template void Image::UnRegisterPageLockedMemory<half_float::half>(half_float::half* ptr);
