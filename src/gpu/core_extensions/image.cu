

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
void Image::AllocatePageLockedMemory(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool is_in_real_space, bool do_fft_planning) {
    Allocate(wanted_x_size, wanted_y_size, wanted_z_size, is_in_real_space, do_fft_planning);
    RegisterPageLockedMemory( );
}

void Image::RegisterPageLockedMemory( ) {
    MyDebugAssertTrue(is_in_memory, "Image is not in memory");

    if ( ! page_locked_ptr ) {
        wxMutexLocker lock(s_mutexProtectingFFTW); // the mutex will be unlocked when this object is destroyed (when it goes out of scope)
        MyDebugAssertTrue(lock.IsOk( ), "Mute locking failed");
        cudaErr(cudaHostRegister(real_values, sizeof(float) * real_memory_allocated, cudaHostRegisterDefault));
        cudaErr(cudaHostGetDevicePointer(&page_locked_ptr, real_values, 0));
    }
}

void Image::UnRegisterPageLockedMemory( ) {
    if ( page_locked_ptr ) {
        wxMutexLocker lock(s_mutexProtectingFFTW); // the mutex will be unlocked when this object is destroyed (when it goes out of scope)
        MyDebugAssertTrue(lock.IsOk( ), "Mute locking failed");
        cudaErr(cudaHostUnregister(page_locked_ptr));
        page_locked_ptr = nullptr;
    }
}
