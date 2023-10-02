/*
 * GpuImage.cpp
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

//#include "gpu_core_headers.h"

#include "gpu_core_headers.h"
#include "GpuImage.h"
#include "gpu_indexing_functions.h"
#include "gpu_image_cuFFT_callbacks.h"

// #define USE_ASYNC_MALLOC_FREE
// #define USE_BLOCK_REDUCE
#define USE_FP16_FOR_WHITENPS

__global__ void
MipPixelWiseKernel(cufftReal* mip, const cufftReal* correlation_output, const int4 dims);
__global__ void
MipPixelWiseKernel(cufftReal* mip, cufftReal* other_image, cufftReal* psi, cufftReal* phi, cufftReal* theta,
                   int4 dims, float c_psi, float c_phi, float c_theta);
__global__ void
MipPixelWiseKernel(cufftReal* mip, cufftReal* other_image, cufftReal* psi, cufftReal* phi, cufftReal* theta, cufftReal* defocus, cufftReal* pixel, const int4 dims,
                   float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel);

__global__ void
ClipIntoRealKernel(cufftReal* real_values,
                   cufftReal* other_image_real_values,
                   int4       dims,
                   int4       other_dims,
                   int3       physical_address_of_box_center,
                   int3       other_physical_address_of_box_center,
                   int3       wanted_coordinate_of_box_center,
                   float      wanted_padding_value);

// Inline declarations
__device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index,
                                                int logical_x_dimension,
                                                int physical_address_of_box_center_x);

__device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                                int logical_y_dimension,
                                                int physical_index_of_first_negative_frequency_y);

__device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,
                                                int logical_z_dimension,
                                                int physical_index_of_first_negative_frequency_x);

__device__ __forceinline__ float
d_ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size);

__device__ __forceinline__ void
d_Return3DPhaseFromIndividualDimensions(float phase_x, float phase_y, float phase_z, float2& angles);

__device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, int4 img_dims);

__device__ __forceinline__ int
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_coords, int3 physical_upper_bound_complex);

__device__ __forceinline__ int
d_ReturnFourier1DAddressFromLogicalCoord(int wanted_x_coord, int wanted_y_coord, int wanted_z_coord, const int3& dims, const int3& physical_upper_bound_complex);

__inline__ int
ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                              int logical_y_dimension,
                                              int physical_index_of_first_negative_frequency_y) {
    if ( physical_index >= physical_index_of_first_negative_frequency_y ) {
        return physical_index - logical_y_dimension;
    }
    else
        return physical_index;
};

__inline__ int
ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,
                                              int logical_z_dimension,
                                              int physical_index_of_first_negative_frequency_z) {
    if ( physical_index >= physical_index_of_first_negative_frequency_z ) {
        return physical_index - logical_z_dimension;
    }
    else
        return physical_index;
};

////////////////// For thrust
typedef struct
{
    __host__ __device__ float operator( )(const float& x) const {
        return x * x;
    }
} square;

typedef __align__(4) struct _ValueAndWeight_half {
    static_assert(sizeof(nv_bfloat16) == sizeof(short), "sizeof(__half) != sizeof(short)");
    nv_bfloat16 value;
    short       weight;
} ValueAndWeight_half;

typedef __align__(8) struct _ValueAndWeight {
    static_assert(sizeof(float) == sizeof(int), "sizeof(float) != sizeof(int)");
    float value;
    int   weight;
} ValueAndWeight;

// #define FULL_MASK 0xffffffff

////////////////////////
__inline__ __device__ float warpReduceSum(float val) {
    for ( int offset = warpSize / 2; offset > 0; offset /= 2 )
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int                   lane = threadIdx.x % warpSize;
    int                   wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if ( lane == 0 )
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads( ); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if ( wid == 0 )
        val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__inline__ __device__ float blockReduce2dSum(float val) {
    // This assumes a block size of 32x32 which is the default case  (32,32,1)
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int                     linearIdx = threadIdx.x + threadIdx.y * blockDim.x;
    int                     lane      = linearIdx % warpSize; // lane in warp
    int                     wid       = linearIdx / warpSize; // warp id in block

    val = warpReduceSum(val); // Each warp performs partial reduction

    if ( lane == 0 )
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads( ); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (linearIdx < (blockDim.y * blockDim.x) / warpSize) ? shared[lane] : 0;

    if ( wid == 0 )
        val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {

#pragma unroll 5
    for ( int offset = cistem::gpu::warp_size / 2; offset > 0; offset /= 2 )
        val = gMax(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float shared[32]; // Shared mem for 32 partial sums
    int              lane = threadIdx.x % cistem::gpu::warp_size;
    int              wid  = threadIdx.x / cistem::gpu::warp_size;

    val = warpReduceMax(val); // Each warp performs partial reduction

    if ( lane == 0 )
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads( ); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / cistem::gpu::warp_size) ? shared[lane] : 0;

    if ( wid == 0 )
        val = warpReduceMax(val); //Final reduce within first warp

    return val;
}

__device__ __forceinline__ void warpReduceMax(float& val, int& index) {

    float tmp_val   = val;
    int   tmp_index = index;
    // #pragma unroll 5
    for ( int offset = cistem::gpu::warp_size / 2; offset > 0; offset /= 2 ) {
        tmp_val   = __shfl_xor_sync(0xffffffff, val, offset);
        tmp_index = __shfl_xor_sync(0xffffffff, index, offset);
        if ( tmp_val > val ) {
            val   = tmp_val;
            index = tmp_index;
        }
    }
}

__device__ __forceinline__ void blockReduceMax(float& val, int& index) {

    __shared__ float shared[64]; // Shared mem for 32 partial sums
    int              lane = threadIdx.x % cistem::gpu::warp_size;
    int              wid  = threadIdx.x / cistem::gpu::warp_size;

    warpReduceMax(val, index); // Each warp performs partial reduction

    if ( lane == 0 ) {
        shared[wid]      = val; // Write reduced value to shared memory
        shared[wid + 32] = index; // Write reduced value to shared memory
    }

    __syncthreads( ); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val   = (threadIdx.x < blockDim.x / cistem::gpu::warp_size) ? shared[lane] : 0;
    index = (threadIdx.x < blockDim.x / cistem::gpu::warp_size) ? shared[lane + 32] : 0;

    if ( wid == 0 )
        warpReduceMax(val, index); //Final reduce within first warp

    return;
}

GpuImage::GpuImage( ) {
    is_meta_data_initialized = false;
    SetupInitialValues( );
}

GpuImage::GpuImage(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool is_in_real_space, bool allocate_fp16_buffer) {
    is_meta_data_initialized = false;
    SetupInitialValues( );
    // do_fft_planning is not used in the GPU code path, but is present in the cpu
    Allocate(wanted_x_size, wanted_y_size, wanted_z_size, is_in_real_space, allocate_fp16_buffer);
}

GpuImage::GpuImage(Image& cpu_image) {
    is_meta_data_initialized = false;
    SetupInitialValues( );
    Init(cpu_image);
}

GpuImage::GpuImage(const GpuImage& other_gpu_image) {
    is_meta_data_initialized = false;
    SetupInitialValues( );
    *this = other_gpu_image;
}

GpuImage& GpuImage::operator=(const GpuImage& other_gpu_image) {
    *this = &other_gpu_image;
    return *this;
}

GpuImage& GpuImage::operator=(const GpuImage* other_gpu_image) {
    // Check for self assignment
    if ( this != other_gpu_image ) {

        MyDebugAssertTrue(other_gpu_image->is_in_memory_gpu, "Other image Memory not allocated");
        if ( is_in_memory_gpu == true ) {
            if ( dims.x != other_gpu_image->dims.x || dims.y != other_gpu_image->dims.y || dims.z != other_gpu_image->dims.z ) {
                Deallocate( );
                SetupInitialValues( );
                CopyGpuImageMetaData(other_gpu_image);
                Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
            }
        }
        else {
            SetupInitialValues( );
            CopyGpuImageMetaData(other_gpu_image);
            Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
        }

        precheck;
        cudaErr(cudaMemcpyAsync(real_values, other_gpu_image->real_values, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        postcheck;
    }

    return *this;
}

GpuImage::~GpuImage( ) {
    Deallocate( );
}

/**
 * @brief allocate_real_values defaults to true, because (at the time) we generally want the single precision image buffer in most cases.
 * particularly when interacting with a CPU image, which only supports single precision. At times, deffering allocation may be preferable.
 * @param cpu_image 
 * @param allocate_real_values 
 */
bool GpuImage::Init(Image& cpu_image, bool pin_host_memory, bool allocate_real_values) {
    // Returns true if gpu_memory was changed (alloc/dealloc)
    return InitializeBasedOnCpuImage(cpu_image, pin_host_memory, allocate_real_values);
}

void GpuImage::SetupInitialValues( ) {
    dims  = make_int4(0, 0, 0, 0);
    pitch = 0;
    // FIXME: Tempororay for compatibility with the IMage class
    logical_x_dimension                        = 0;
    logical_y_dimension                        = 0;
    logical_z_dimension                        = 0;
    physical_upper_bound_complex               = make_int3(0, 0, 0);
    physical_address_of_box_center             = make_int3(0, 0, 0);
    physical_index_of_first_negative_frequency = make_int3(0, 0, 0);
    logical_upper_bound_complex                = make_int3(0, 0, 0);
    logical_lower_bound_complex                = make_int3(0, 0, 0);
    logical_upper_bound_real                   = make_int3(0, 0, 0);
    logical_lower_bound_real                   = make_int3(0, 0, 0);

    fourier_voxel_size = make_float3(0.0f, 0.0f, 0.0f);

    number_of_real_space_pixels = 0;

    real_memory_allocated = 0;

    padding_jump_value = 0;

    ft_normalization_factor = 0;

    // weighted_correlation_buffer_size = 0; // TODO: Should this be with b uffer stuff

    real_values    = nullptr; // !<  Real array to hold values for REAL images.
    complex_values = nullptr; // !<  Complex array to hold values for COMP images.

    real_values_16f        = nullptr;
    complex_values_16f     = nullptr;
    ctf_buffer_16f         = nullptr;
    ctf_complex_buffer_16f = nullptr;

    gpu_plan_id = -1;

    insert_into_which_reconstruction = 0;
    host_image_ptr                   = nullptr;

    image_buffer = nullptr;
    mask_CSOS    = nullptr;

    cudaErr(cudaGetDevice(&device_idx));
    cudaErr(cudaDeviceGetAttribute(&number_of_streaming_multiprocessors, cudaDevAttrMultiProcessorCount, device_idx));
    limit_SMs_by_threads = 1;

    set_batch_size = 1;
    AllocateTmpVarsAndEvents( );
    UpdateBoolsToDefault( );
}

void GpuImage::CopyFrom(GpuImage* other_image) {
    *this = other_image;
}

/**
 * @brief Typically used to restore a fixed image in an iterative algo. 
 * TODO: Locking mechanism and safety checks. It is assume you know you are not changing size, or modifying allocations etc.
*/
template <typename StorageTypeBase>
void GpuImage::CopyDataFrom(GpuImage& other_image) {
    MyDebugAssertTrue(dims.x == other_image.dims.x && dims.y == other_image.dims.y && dims.z == other_image.dims.z, "Dimensions do not match");
    MyDebugAssertTrue(real_memory_allocated == other_image.real_memory_allocated, "Memory allocated does not match");
    if constexpr ( std::is_same<StorageTypeBase, float>::value ) {
        MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
        MyDebugAssertTrue(other_image.is_in_memory_gpu, "Other image Memory not allocated");
        cudaErr(cudaMemcpyAsync(real_values, other_image.real_values, sizeof(float) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    else if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {
        MyDebugAssertTrue(is_allocated_16f_buffer, "Memory not allocated");
        MyDebugAssertTrue(other_image.is_allocated_16f_buffer, "Other image Memory not allocated");
        cudaErr(cudaMemcpyAsync(real_values_16f, other_image.real_values_16f, sizeof(__half) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    else {
        MyDebugAssertTrue(false, "Invalid StorageTypeBase");
    }
}

template void GpuImage::CopyDataFrom<float>(GpuImage& other_image);
template void GpuImage::CopyDataFrom<__half>(GpuImage& other_image);

bool GpuImage::InitializeBasedOnCpuImage(Image& cpu_image, bool pin_host_memory, bool allocate_real_values) {
    bool gpu_memory_was_changed = false;
    // First check to see if we have existed before
    if ( is_meta_data_initialized ) {
        //FIXME: i think using host_image_ptr->device_image_ptr == this may be a better check
        // Okay, the GpuImage has existed, see if it is already pointing at the cpu
        if ( real_memory_allocated != cpu_image.real_memory_allocated && cpu_image.real_memory_allocated > 0 ) {
            Deallocate( );
            gpu_memory_was_changed = true;
        }
        return gpu_memory_was_changed;
    }
    else {
        // This is the first time we are initializing the GpuImage
        SetupInitialValues( );
    }

    CopyCpuImageMetaData(cpu_image);

    if ( allocate_real_values ) {
        // This will also check to ensure we are not allocated
        gpu_memory_was_changed = gpu_memory_was_changed || Allocate(dims.x, dims.y, dims.z, is_in_real_space);
    }

    if ( pin_host_memory ) {
        cpu_image.RegisterPageLockedMemory( );
    }

    AllocateTmpVarsAndEvents( );
    return gpu_memory_was_changed;
}

void GpuImage::UpdateCpuFlags( ) {
    MyDebugAssertFalse(host_image_ptr == nullptr, "Host image pointer not set");
    // Call after re-copying. The main image properites are all assumed to be static.
    is_in_real_space         = host_image_ptr->is_in_real_space;
    object_is_centred_in_box = host_image_ptr->object_is_centred_in_box;
    is_fft_centered_in_box   = host_image_ptr->is_fft_centered_in_box;
}

void GpuImage::printVal(std::string msg, int idx) {
    float h_printVal = -9999.0f;
    if ( idx < 0 ) {
        // -1 = last index and counting backward
        idx = real_memory_allocated + idx - 1;
    }

    cudaErr(cudaMemcpy(&h_printVal, &real_values[idx], sizeof(float), cudaMemcpyDeviceToHost));
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    MyDebugAssertFalse(h_printVal == -9999.0f, "Error: printVal failed");
    wxPrintf("%s %6.6e\n", msg, h_printVal);
};

void GpuImage::PrintNppStreamContext( ) {
    MyDebugAssertTrue(is_npp_loaded, "Error: NPP not loaded");

    wxPrintf("NPP stream context:\n");
    wxPrintf("  npp device id %d\n", nppStream.nCudaDeviceId);
    wxPrintf("  npp multi processor count %d\n", nppStream.nMultiProcessorCount);
    wxPrintf("  npp max threads per multi processor %d\n", nppStream.nMaxThreadsPerMultiProcessor);
    wxPrintf("  npp max threads per block %d\n", nppStream.nMaxThreadsPerBlock);
    wxPrintf("  npp max shared memory per block %ld\n", nppStream.nSharedMemPerBlock);
    wxPrintf("  npp compute capability %d.%d\n", nppStream.nCudaDevAttrComputeCapabilityMajor, nppStream.nCudaDevAttrComputeCapabilityMinor);

    wxPrintf("\n NPP ROI:\n");

    wxPrintf("  npp_ROI: %d %d\n", npp_ROI_real_space.width, npp_ROI_real_space.height);

    wxPrintf("  npp_ROI_real_space: %d %d\n", npp_ROI_real_space.width, npp_ROI_real_space.height);
    wxPrintf("  npp_ROI_fourier_space: %d %d\n", npp_ROI_fourier_space.width, npp_ROI_fourier_space.height);
    wxPrintf("  GpuImage.pitch bytes/ elements %ld/ %ld\n", pitch, pitch / sizeof(float));
}

bool GpuImage::HasSameDimensionsAs(Image* other_image) {
    // Functions that call this method also assume these asserts are being called here, so do not remove.
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory, "Other image Memory not allocated");

    if ( dims.x == other_image->logical_x_dimension && dims.y == other_image->logical_y_dimension && dims.z == other_image->logical_z_dimension )
        return true;
    else
        return false;
}

template <typename StorageBaseType>
bool GpuImage::HasSameDimensionsAs<StorageBaseType>(GpuImage& other_image) {
    // Functions that call this method also assume these asserts are being called here, so do not remove.
    if constexpr ( std::is_same<StorageBaseType, __half>::value ) {
        MyDebugAssertTrue(is_allocated_16f_buffer, "FP16 Memory not allocated");
        MyDebugAssertTrue(other_image.is_allocated_16f_buffer, "Other image FP16 Memory not allocated");
    }
    else {
        MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
        MyDebugAssertTrue(other_image.is_in_memory_gpu, "Other image Memory not allocated");
    }

    if ( dims.x == other_image.dims.x && dims.y == other_image.dims.y && dims.z == other_image.dims.z )
        return true;
    else
        return false;
}

template bool GpuImage::HasSameDimensionsAs<float>(GpuImage& other_image);
template bool GpuImage::HasSameDimensionsAs<__half>(GpuImage& other_image);

template <typename StorageType>
__global__ void
MultiplyPixelWiseComplexConjugateKernel(const StorageType* __restrict__ img_complex_values,
                                        const StorageType* __restrict__ ref_complex_values,
                                        StorageType* result_values,
                                        int4         dims) {
    int x = physical_X( );
    if ( x >= dims.w / 2 )
        return;
    int y = physical_Y( );
    if ( y >= dims.y )
        return;

    int address = x + (dims.w / 2) * y;
    int stride  = (dims.w / 2) * dims.y;

    const StorageType ref_val = (StorageType)ref_complex_values[address];
    for ( int k = 0; k < dims.z; k++ ) {
        // In cisTEM translational search is ref * conj(img) (which gives a passive xform, i.e. how the image needs to be shifted back to match the reference)
        result_values[address] = (StorageType)ComplexConjMul(ref_val, (StorageType)img_complex_values[address]);
        address += stride;
    }
}

template <typename StorageType>
__global__ void
MultiplyPixelWiseComplexConjugateKernel(const StorageType* __restrict__ img_complex_values,
                                        const StorageType* __restrict__ ref_complex_values,
                                        StorageType* result_values,
                                        int4         dims,
                                        const int    phase_multiplier) {
    int x = physical_X( );
    if ( x > dims.w / 2 )
        return;
    int y = physical_Y( );
    if ( y > dims.y )
        return;

    int address = x + (dims.w / 2) * y;
    int stride  = (dims.w / 2) * dims.y;

    constexpr float epsilon = 1e-6f;
    // We have one referece image, and this is broadcasted to all images in the stack
    if constexpr ( std::is_same<StorageType, __half2>::value ) {
        const Complex ref_val = (Complex)__half22float2(ref_complex_values[address]);
        Complex       C;
        Complex       phase_shift;

        for ( int k = 0; k < dims.z; k++ ) {
            // the result should be (ref * conj(ref * shift + noise)) = auto correlation of the reference and the phase shifted from the iamge
            C = ComplexConjMul(ref_val, (Complex)__half22float2(img_complex_values[address]));
            // remove the magnitude to get the phase shift (see saxton 1996)
            __sincosf(float(phase_multiplier) * atan2f(C.y, C.x), &phase_shift.y, &phase_shift.x);
            // float amplitude = ComplexModulus(C) + epsilon;
            // ComplexScale(&C, 1.0f / amplitude);
            result_values[address] = __float22half2_rn(ComplexMul(C, phase_shift));

            address += stride;
        }
    }
    else {

        const Complex ref_val = (Complex)ref_complex_values[address];
        Complex       C;
        Complex       phase_shift;

        for ( int k = 0; k < dims.z; k++ ) {
            // the result should be (ref * conj(ref * shift + noise)) = auto correlation of the reference and the phase shifted from the iamge
            C = ComplexConjMul(ref_val, (Complex)img_complex_values[address]);
            // remove the magnitude to get the phase shift (see saxton 1996)
            __sincosf(float(phase_multiplier) * atan2f(C.y, C.x), &phase_shift.y, &phase_shift.x);
            // float amplitude = ComplexModulus(C) + epsilon;
            // ComplexScale(&C, 1.0f / amplitude);
            result_values[address] = (cufftComplex)ComplexMul(C, phase_shift);

            address += stride;
        }
    }
}

template <typename StorageTypeBase>
void GpuImage::MultiplyPixelWiseComplexConjugate<StorageTypeBase>(GpuImage& reference_img, GpuImage& result_image, int phase_multiplier) {
    // FIXME when adding real space complex images
    MyDebugAssertFalse(is_in_real_space, "Image is in real space");
    MyDebugAssertFalse(reference_img.is_in_real_space, "Other image is in real space");
    MyDebugAssertTrue(dims.x == reference_img.dims.x && dims.y == reference_img.dims.y, "Images have different dimensions");
    MyDebugAssertTrue(dims.x == result_image.dims.x && dims.y == result_image.dims.y, "Images have different dimensions");

    //  NppInit();
    //  Conj();
    //  npp_stat = nppiMul_32sc_C1IRSfs((const Npp32sc *)complex_values, 1, (Npp32sc*)reference_img.complex_values, 1, npp_ROI_complex, 0);

    // Multiplier to avoid conditionals in the kernel, if size z == 1 image z is zero resulting in a broadcast of the 2d through the batch.
    // FIXME: somehow this doesn't work.
    ReturnLaunchParameters(dims, false);

    // Override and loop over z in kernel allowing re-use of the image value if broadcast.
    gridDims.z = 1;

    precheck;
    if constexpr ( std::is_same_v<StorageTypeBase, __half> ) {
        MyDebugAssertTrue(is_allocated_16f_buffer, "Memory not allocated");
        MyDebugAssertTrue(reference_img.is_allocated_16f_buffer, "Other image Memory not allocated");
        MyDebugAssertTrue(result_image.is_allocated_16f_buffer, "Result image Memory not allocated");
        if ( phase_multiplier > 0 ) {
            MultiplyPixelWiseComplexConjugateKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_fp16, reference_img.complex_values_fp16, result_image.complex_values_fp16, this->dims, phase_multiplier);
        }
        else {
            MultiplyPixelWiseComplexConjugateKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_fp16, reference_img.complex_values_fp16, result_image.complex_values_fp16, this->dims);
        }
    }
    else {
        MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
        MyDebugAssertTrue(reference_img.is_in_memory_gpu, "Other image Memory not allocated");
        MyDebugAssertTrue(result_image.is_in_memory_gpu, "Result image Memory not allocated");
        if ( phase_multiplier > 0 ) {
            MultiplyPixelWiseComplexConjugateKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values, reference_img.complex_values, result_image.complex_values, this->dims, phase_multiplier);
        }
        else {
            MultiplyPixelWiseComplexConjugateKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values, reference_img.complex_values, result_image.complex_values, this->dims);
        }
    }

    postcheck;
}

template void GpuImage::MultiplyPixelWiseComplexConjugate<__half>(GpuImage& reference_img, GpuImage& result_image, int phase_multiplier);
template void GpuImage::MultiplyPixelWiseComplexConjugate<float>(GpuImage& reference_img, GpuImage& result_image, int phase_multiplier);

__global__ void
ReturnSumOfRealValuesOnEdgesKernel(cufftReal* real_values, int4 dims, int padding_jump_value, float* returnValue);

float GpuImage::ReturnAverageOfRealValuesOnEdges( ) {
    // FIXME to use a masked routing, this is slow af
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(dims.z == 1, "ReturnAverageOfRealValuesOnEdges only implemented in 2d");

    precheck;
    *tmpVal = 5.0f;
    ReturnSumOfRealValuesOnEdgesKernel<<<1, 1, 0, cudaStreamPerThread>>>(real_values, dims, padding_jump_value, tmpVal);
    postcheck;

    // Need to wait on the return value
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    return *tmpVal;
}

__global__ void
ReturnSumOfRealValuesOnEdgesKernel(cufftReal* real_values, int4 dims, int padding_jump_value, float* returnValue) {
    int pixel_counter;
    int line_counter;
    //    int plane_counter;

    double sum              = 0.0;
    int    number_of_pixels = 0;
    int    address          = 0;

    // Two-dimensional image
    // First line
    for ( pixel_counter = 0; pixel_counter < dims.x; pixel_counter++ ) {
        sum += real_values[address];
        address++;
    }
    number_of_pixels += dims.x;
    address += padding_jump_value;

    // Other lines
    for ( line_counter = 1; line_counter < dims.y - 1; line_counter++ ) {
        sum += real_values[address];
        address += dims.x - 1;
        sum += real_values[address];
        address += padding_jump_value + 1;
        number_of_pixels += 2;
    }

    // Last line
    for ( pixel_counter = 0; pixel_counter < dims.x; pixel_counter++ ) {
        sum += real_values[address];
        address++;
    }
    number_of_pixels += dims.x;

    *returnValue = (float)sum / (float)number_of_pixels;
}

//void GpuImage::CublasInit()
//{
//  if ( ! is_cublas_loaded )
//  {
//    cublasCreate(&cublasHandle);
//    is_cublas_loaded = true;
//    cublasSetStream(cublasHandle, cudaStreamPerThread);
//  }
//}

void GpuImage::NppInit( ) {
    if ( ! is_npp_loaded ) {

        int sharedMem;
        // Used for calls to npp buffer functions, but memory alloc/free is synced using cudaStreamPerThread as it does not recognize the nppStreamContext
        nppStream.hStream = cudaStreamPerThread;
        cudaGetDevice(&nppStream.nCudaDeviceId);
        cudaDeviceGetAttribute(&nppStream.nMultiProcessorCount, cudaDevAttrMultiProcessorCount, nppStream.nCudaDeviceId);
        cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, nppStream.nCudaDeviceId);
        cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, nppStream.nCudaDeviceId);
        cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, nppStream.nCudaDeviceId);
        cudaDeviceGetAttribute(&sharedMem, cudaDevAttrMaxSharedMemoryPerBlock, nppStream.nCudaDeviceId);
        nppStream.nSharedMemPerBlock = (size_t)sharedMem;
        cudaDeviceGetAttribute(&nppStream.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, nppStream.nCudaDeviceId);
        cudaDeviceGetAttribute(&nppStream.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, nppStream.nCudaDeviceId);

        //    nppSetStream(cudaStreamPerThread);

        npp_ROI_real_space.width  = dims.x;
        npp_ROI_real_space.height = dims.y * dims.z;

        npp_ROI_fourier_space.width  = dims.w / 2;
        npp_ROI_fourier_space.height = dims.y * dims.z;

        // This is used only in special cases, and is explicit.
        npp_ROI_fourier_with_real_functor.width  = dims.w;
        npp_ROI_fourier_with_real_functor.height = dims.y * dims.z;

        // This is switched appropriately in FFT/iFFT calls
        if ( is_in_real_space ) {
            npp_ROI = npp_ROI_real_space;
        }
        else {
            npp_ROI = npp_ROI_fourier_space;
        }

        is_npp_loaded = true;
    }
}

void GpuImage::BufferInit(BufferType bt, int n_elements) {
    switch ( bt ) {
        case b_image:
            if ( ! is_allocated_image_buffer ) {
                image_buffer              = new GpuImage;
                *image_buffer             = *this;
                is_allocated_image_buffer = true;
            }
            break;

        case b_16f:
            if ( ! is_allocated_16f_buffer ) {
                cudaErr(cudaMallocAsync(&real_values_16f, size_of_half * real_memory_allocated, cudaStreamPerThread));
                complex_values_16f      = (void*)real_values_16f;
                is_allocated_16f_buffer = true;

                real_values_fp16    = reinterpret_cast<__half*>(real_values_16f);
                complex_values_fp16 = reinterpret_cast<__half2*>(complex_values_16f);

                real_values_bf16    = reinterpret_cast<nv_bfloat16*>(real_values_16f);
                complex_values_bf16 = reinterpret_cast<nv_bfloat162*>(complex_values_16f);
            }
            break;

        case b_weighted_correlation: {
            MyDebugAssertTrue(n_elements > 0, "For allocating the weighted_correlation_buffer buffer, you must specify the number of elements");
            if ( is_allocated_weighted_correlation_buffer ) {
                if ( n_elements != weighted_correlation_buffer_size ) {
                    cudaErr(cudaFreeHost(weighted_correlation_buffer));
                    cudaErr(cudaMallocHost(&weighted_correlation_buffer, sizeof(float) * n_elements));
                    weighted_correlation_buffer_size = n_elements;
                }
            }
            else {
                cudaErr(cudaMallocHost(&weighted_correlation_buffer, sizeof(float) * n_elements));
                weighted_correlation_buffer_size         = n_elements;
                is_allocated_weighted_correlation_buffer = true;
            }
            wxPrintf("\n");
            break;
        }

        case b_ctf_16f:
            if ( ! is_allocated_ctf_16f_buffer ) {
                MyDebugAssertTrue(n_elements > 0, "For allocating the ctf_16f buffer, you must specify the number of elements");
                cudaErr(cudaMallocAsync(&ctf_buffer_16f, size_of_half * n_elements, cudaStreamPerThread));

                ctf_complex_buffer_16f      = (void*)ctf_buffer_16f;
                is_allocated_ctf_16f_buffer = true;

                ctf_buffer_fp16         = reinterpret_cast<__half*>(ctf_buffer_16f);
                ctf_complex_buffer_fp16 = reinterpret_cast<__half2*>(ctf_complex_buffer_16f);

                ctf_buffer_bf16         = reinterpret_cast<nv_bfloat16*>(ctf_buffer_16f);
                ctf_complex_buffer_bf16 = reinterpret_cast<nv_bfloat162*>(ctf_complex_buffer_16f);
            }
            break;

        case b_sum:
            if ( ! is_allocated_sum_buffer ) {
                int n_elem;
                nppErr(nppiSumGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->sum_buffer, n_elem, nppStream.hStream));
                is_allocated_sum_buffer = true;
            }
            break;

        case b_min:
            if ( ! is_allocated_min_buffer ) {
                int n_elem;
                nppErr(nppiMinGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->min_buffer, n_elem, nppStream.hStream));

                is_allocated_min_buffer = true;
            }
            break;

        case b_minIDX:
            if ( ! is_allocated_minIDX_buffer ) {
                int n_elem;
                nppErr(nppiMinIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->minIDX_buffer, n_elem, nppStream.hStream));

                is_allocated_minIDX_buffer = true;
            }
            break;

        case b_max:
            if ( ! is_allocated_max_buffer ) {
                int n_elem;
                nppErr(nppiMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->max_buffer, n_elem, nppStream.hStream));

                is_allocated_max_buffer = true;
            }
            break;

        case b_maxIDX:
            if ( ! is_allocated_maxIDX_buffer ) {
                int n_elem;
                nppErr(nppiMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->maxIDX_buffer, n_elem, nppStream.hStream));

                is_allocated_maxIDX_buffer = true;
            }
            break;

        case b_minmax:
            if ( ! is_allocated_minmax_buffer ) {
                int n_elem;
                nppErr(nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->minmax_buffer, n_elem, nppStream.hStream));

                is_allocated_minmax_buffer = true;
            }
            break;

        case b_minmaxIDX:
            if ( ! is_allocated_minmaxIDX_buffer ) {
                int n_elem;
                nppErr(nppiMinMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->minmaxIDX_buffer, n_elem, nppStream.hStream));

                is_allocated_minmaxIDX_buffer = true;
            }
            break;

        case b_mean:
            if ( ! is_allocated_mean_buffer ) {
                int n_elem;
                nppErr(nppiMeanGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->mean_buffer, n_elem, nppStream.hStream));

                is_allocated_mean_buffer = true;
            }
            break;
        case b_meanstddev:
            if ( ! is_allocated_meanstddev_buffer ) {
                int n_elem;
                nppErr(nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaStreamSynchronize(nppStream.hStream));
                cudaErr(cudaMallocAsync(&this->meanstddev_buffer, n_elem, nppStream.hStream));

                is_allocated_meanstddev_buffer = true;
            }
            break;

        case b_countinrange:
            if ( ! is_allocated_countinrange_buffer ) {
                int n_elem;
                nppErr(nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->countinrange_buffer, n_elem, nppStream.hStream));

                is_allocated_countinrange_buffer = true;
            }
            break;

        case b_l2norm:
            if ( ! is_allocated_l2norm_buffer ) {
                int n_elem;
                nppErr(nppiNormL2GetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->l2norm_buffer, n_elem, nppStream.hStream));

                is_allocated_l2norm_buffer = true;
            }
            break;

        case b_dotproduct:
            if ( ! is_allocated_dotproduct_buffer ) {
                int n_elem;
                nppErr(nppiDotProdGetBufferHostSize_32f64f_C1R_Ctx(npp_ROI, &n_elem, nppStream));
                cudaErr(cudaMallocAsync(&this->dotproduct_buffer, n_elem, nppStream.hStream));

                is_allocated_dotproduct_buffer = true;
            }
            break;
    }
}

void GpuImage::FreeFFTPlan( ) {
    if ( set_plan_type != cistem::fft_type::Enum::unset ) {
        cufftErr(cufftDestroy(cuda_plan_inverse));
        cufftErr(cufftDestroy(cuda_plan_forward));
        set_plan_type             = cistem::fft_type::Enum::unset;
        cufft_batch_size          = 1;
        is_set_complexConjMulLoad = false;
    }
}

void GpuImage::BufferDestroy( ) {
    if ( is_allocated_16f_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(real_values_16f, cudaStreamPerThread));
#else
        cudaErr(cudaFree(real_values_16f));
#endif
        is_allocated_16f_buffer = false;
        real_values_16f         = nullptr;
        complex_values_16f      = nullptr;
    }

    if ( is_allocated_weighted_correlation_buffer ) {
        // cudaErr(cudaFreeHost(weighted_correlation_buffer));
        weighted_correlation_buffer_size         = 0;
        is_allocated_weighted_correlation_buffer = false;
    }

    if ( is_allocated_ctf_16f_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(ctf_buffer_16f, cudaStreamPerThread));
#else
        cudaErr(cudaFree(ctf_buffer_16f));
#endif
        is_allocated_ctf_16f_buffer = false;
        ctf_buffer_16f              = nullptr;
        ctf_complex_buffer_16f      = nullptr;
    }

    if ( is_allocated_sum_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->sum_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(sum_buffer));
#endif
        is_allocated_sum_buffer = false;
    }

    if ( is_allocated_min_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->min_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(min_buffer));
#endif

        is_allocated_min_buffer = false;
    }

    if ( is_allocated_minIDX_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->minIDX_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(minIDX_buffer));
#endif
        is_allocated_minIDX_buffer = false;
    }

    if ( is_allocated_max_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->max_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(max_buffer));
#endif
        is_allocated_max_buffer = false;
    }

    if ( is_allocated_maxIDX_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->maxIDX_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(maxIDX_buffer));
#endif

        is_allocated_maxIDX_buffer = false;
    }

    if ( is_allocated_minmax_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->minmax_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(minmax_buffer));
#endif

        is_allocated_minmax_buffer = false;
    }

    if ( is_allocated_minmaxIDX_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->minmaxIDX_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(minmaxIDX_buffer));
#endif
        is_allocated_minmaxIDX_buffer = false;
    }

    if ( is_allocated_mean_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->mean_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(mean_buffer));
#endif
        is_allocated_mean_buffer = false;
    }

    if ( is_allocated_meanstddev_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->meanstddev_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(meanstddev_buffer));
#endif
        is_allocated_meanstddev_buffer = false;
    }

    if ( is_allocated_countinrange_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->countinrange_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(countinrange_buffer));
#endif
        is_allocated_countinrange_buffer = false;
    }

    if ( is_allocated_l2norm_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->l2norm_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(l2norm_buffer));
#endif
        is_allocated_l2norm_buffer = false;
    }

    if ( is_allocated_dotproduct_buffer ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->dotproduct_buffer, nppStream.hStream));
#else
        cudaErr(cudaFree(dotproduct_buffer));
#endif
        is_allocated_dotproduct_buffer = false;
    }

    if ( is_allocated_mask_CSOS ) {
        mask_CSOS->Deallocate( );
        delete mask_CSOS;
    }

    if ( is_allocated_image_buffer ) {
        image_buffer->Deallocate( );
        delete image_buffer;
    }

    if ( is_allocated_clip_into_mask ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(this->clip_into_mask, cudaStreamPerThread));
#else
        cudaErr(cudaFree(clip_into_mask));
#endif
        delete clip_into_mask;
    }
}

float GpuImage::ReturnSumOfSquares( ) {
    // FIXME this assumes padded values are zero which is not strictly true
    MyDebugAssertTrue(is_in_memory_gpu, "Image not allocated");
    MyDebugAssertTrue(is_in_real_space, "This method is for real space, use ReturnSumSquareModulusComplexValues for Fourier space")

            BufferInit(b_l2norm);
    NppInit( );

    nppErr(nppiNorm_L2_32f_C1R_Ctx((Npp32f*)real_values, pitch, npp_ROI,
                                   (Npp64f*)tmpValComplex, (Npp8u*)this->l2norm_buffer, nppStream));

    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    return (float)(*tmpValComplex * *tmpValComplex);

    //    CublasInit();
    //    // With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
    //    cublas_stat = cublasSdot( cublasHandle, real_memory_allocated,
    //                            real_values, 1,
    //                            real_values, 1,
    //                            &returnValue);
    //
    //    if (cublas_stat) {
    //    wxPrintf("Cublas return val %s\n", cublas_stat); }
    //
    //
    //    return returnValue;
}

float GpuImage::ReturnSumSquareModulusComplexValues( ) {
    //
    MyDebugAssertTrue(is_in_memory_gpu, "Image not allocated");
    MyDebugAssertFalse(is_in_real_space, "This method is NOT for real space, use ReturnSumofSquares for realspace") int address   = 0;
    bool                                                                                                                x_is_even = IsEven(dims.x);
    int                                                                                                                 i, j, k, jj, kk;
    const std::complex<float>                                                                                           c1(sqrtf(0.25f), sqrtf(0.25));
    const std::complex<float>                                                                                           c2(sqrtf(0.5f), sqrtf(0.5f)); // original code is pow(abs(Val),2)*0.5
    const std::complex<float>                                                                                           c3(1.0, 1.0);
    const std::complex<float>                                                                                           c4(0.0, 0.0);
    //    float returnValue;

    if ( ! is_allocated_mask_CSOS ) {

        wxPrintf("is mask allocated %d\n", is_allocated_mask_CSOS);
        mask_CSOS              = new GpuImage;
        is_allocated_mask_CSOS = true;
        wxPrintf("is mask allocated %d\n", is_allocated_mask_CSOS);
        // create a mask that can be reproduce the correct weighting from Image::ReturnSumOfSquares on complex images

        wxPrintf("\n\tMaking mask_CSOS\n");
        mask_CSOS->Allocate(dims.x, dims.y, dims.z, true);
        // The mask should always be in real_space, and starts out not centered
        mask_CSOS->is_in_real_space         = false;
        mask_CSOS->object_is_centred_in_box = true;
        // Allocate pinned host memb
        float*               real_buffer;
        std::complex<float>* complex_buffer;
        cudaErr(cudaHostAlloc(&real_buffer, sizeof(float) * real_memory_allocated, cudaHostAllocDefault));
        complex_buffer = (std::complex<float>*)real_buffer;

        for ( k = 0; k <= physical_upper_bound_complex.z; k++ ) {

            kk = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k, dims.z, physical_index_of_first_negative_frequency.z);
            for ( j = 0; j <= physical_upper_bound_complex.y; j++ ) {
                jj = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j, dims.y, physical_index_of_first_negative_frequency.y);
                for ( i = 0; i <= physical_upper_bound_complex.x; i++ ) {
                    if ( (i == 0 || (i == logical_upper_bound_complex.x && x_is_even)) &&
                         (jj == 0 || (jj == logical_lower_bound_complex.y && x_is_even)) &&
                         (kk == 0 || (kk == logical_lower_bound_complex.z && x_is_even)) ) {
                        complex_buffer[address] = c2;
                    }
                    else if ( (i == 0 || (i == logical_upper_bound_complex.x && x_is_even)) && dims.z != 1 ) {
                        complex_buffer[address] = c1;
                    }
                    else if ( (i != 0 && (i != logical_upper_bound_complex.x || ! x_is_even)) || (jj >= 0 && kk >= 0) ) {
                        complex_buffer[address] = c3;
                    }
                    else {
                        complex_buffer[address] = c4;
                    }

                    address++;
                }
            }
        }

        precheck;
        cudaErr(cudaMemcpyAsync(mask_CSOS->real_values, real_buffer, sizeof(float) * real_memory_allocated, cudaMemcpyHostToDevice, cudaStreamPerThread));
        precheck;
        // TODO change this to an event that can then be later checked prior to deleteing
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        cudaErr(cudaFreeHost(real_buffer));

    } // end of mask creation

    BufferInit(b_image);
    precheck;
    cudaErr(cudaMemcpyAsync(image_buffer->real_values, mask_CSOS->real_values, sizeof(float) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    postcheck;

    image_buffer->is_in_real_space = false;
    image_buffer->npp_ROI          = image_buffer->npp_ROI_fourier_space;
    image_buffer->MultiplyPixelWise(*this);

    //    CublasInit();
    // With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
    precheck;

    BufferInit(b_l2norm);
    NppInit( );
    nppErr(nppiNorm_L2_32f_C1R_Ctx((Npp32f*)image_buffer->real_values, pitch, npp_ROI_fourier_with_real_functor,
                                   (Npp64f*)tmpValComplex, (Npp8u*)this->l2norm_buffer, nppStream));

    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    return (float)(*tmpValComplex * *tmpValComplex);

    postcheck;

    //    return (float)(*dotProd * 2.0f);
    //    return (float)(returnValue * 2.0f);
}

void GpuImage::MipPixelWise(GpuImage& other_image) {

    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    precheck;
    ReturnLaunchParameters(dims, true);
    MipPixelWiseKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values, other_image.real_values, this->dims);
    postcheck;
}

__global__ void
MipPixelWiseKernel(cufftReal* mip, const cufftReal* correlation_output, const int4 dims) {

    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.x && coords.y < dims.y && coords.z < dims.z ) {
        int address  = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
        mip[address] = MAX(mip[address], correlation_output[address]);
    }
}

void GpuImage::MipPixelWise(GpuImage& other_image, GpuImage& psi, GpuImage& phi, GpuImage& theta, GpuImage& defocus, GpuImage& pixel,
                            float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel) {

    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    precheck;
    ReturnLaunchParameters(dims, true);
    MipPixelWiseKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values, other_image.real_values,
                                                                              psi.real_values, phi.real_values, theta.real_values, defocus.real_values, pixel.real_values,
                                                                              this->dims, c_psi, c_phi, c_theta, c_defocus, c_pixel);
    postcheck;
}

__global__ void
MipPixelWiseKernel(cufftReal* mip, cufftReal* correlation_output, cufftReal* psi, cufftReal* phi, cufftReal* theta, cufftReal* defocus, cufftReal* pixel, const int4 dims,
                   float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel) {

    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.x && coords.y < dims.y && coords.z < dims.z ) {
        int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
        if ( correlation_output[address] > mip[address] ) {
            mip[address]     = correlation_output[address];
            psi[address]     = c_psi;
            phi[address]     = c_phi;
            theta[address]   = c_theta;
            defocus[address] = c_defocus;
            pixel[address]   = c_pixel;
        }
    }
}

void GpuImage::MipPixelWise(GpuImage& other_image, GpuImage& psi, GpuImage& phi, GpuImage& theta,
                            float c_psi, float c_phi, float c_theta) {

    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    precheck;
    ReturnLaunchParameters(dims, true);
    MipPixelWiseKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values, other_image.real_values,
                                                                              psi.real_values, phi.real_values, theta.real_values,
                                                                              this->dims, c_psi, c_phi, c_theta);
    postcheck;
}

__global__ void
MipPixelWiseKernel(cufftReal* mip, cufftReal* correlation_output, cufftReal* psi, cufftReal* phi, cufftReal* theta, const int4 dims,
                   float c_psi, float c_phi, float c_theta) {

    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.x && coords.y < dims.y && coords.z < dims.z ) {
        int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
        if ( correlation_output[address] > mip[address] ) {
            mip[address]   = correlation_output[address];
            psi[address]   = c_psi;
            phi[address]   = c_phi;
            theta[address] = c_theta;
        }
    }
}

template <typename StorageType>
__global__ void
ApplyBFactorKernel(StorageType* d_input,
                   const int4   dims,
                   const int3   physical_index_of_first_negative_frequency,
                   const int3   physical_upper_bound_complex,
                   float        bfactor) {

    int3 physical_dims = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y,
                                   blockIdx.z);

    if ( physical_dims.x <= physical_upper_bound_complex.x &&
         physical_dims.y <= physical_upper_bound_complex.y &&
         physical_dims.z <= physical_upper_bound_complex.z ) {
        const int address = d_ReturnFourier1DAddressFromPhysicalCoord(physical_dims, physical_upper_bound_complex);
        int       ret_val;
        int       frequency_squared;

        frequency_squared = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(physical_dims.y, dims.y, physical_index_of_first_negative_frequency.y);
        frequency_squared *= frequency_squared;

        ret_val = d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(physical_dims.x, dims.x, physical_index_of_first_negative_frequency.x);
        frequency_squared += ret_val * ret_val;

        ret_val = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(physical_dims.z, dims.z, physical_index_of_first_negative_frequency.z);
        frequency_squared += ret_val * ret_val;

        ComplexScale((StorageType*)&d_input[address], expf(-bfactor * frequency_squared));
    }
}

template <typename StorageTypeBase>
void GpuImage::ApplyBFactor<StorageTypeBase>(float bfactor) {
    MyDebugAssertFalse(is_in_real_space, "This function is only for Fourier space images.");

    precheck;
    ReturnLaunchParameters(dims, false);
    precheck;
    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {
        ApplyBFactorKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_fp16,
                                                                                  dims,
                                                                                  physical_index_of_first_negative_frequency,
                                                                                  physical_upper_bound_complex,
                                                                                  bfactor * 0.25f);
    }
    else {
        ApplyBFactorKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values,
                                                                                  dims,
                                                                                  physical_index_of_first_negative_frequency,
                                                                                  physical_upper_bound_complex,
                                                                                  bfactor);
    }
    postcheck;
}

template void GpuImage::ApplyBFactor<float>(float bfactor);
template void GpuImage::ApplyBFactor<__half>(float bfactor);

template <typename StorageType>
__global__ void
ApplyBFactorKernel(StorageType* d_input,
                   const int    pixel_pitch,
                   const int    NY,
                   const int    physical_index_of_first_negative_frequency_y,
                   const float  fourier_voxel_size_x_sq,
                   const float  fourier_voxel_size_y_sq,
                   float        bfactor,
                   const float  vertical_mask_size,
                   const float  horizontal_mask_size) {
    int x = physical_X_2d_grid( );
    // Only for Fourier Space image s.t. pixel_pitch = dims.w / 2 = physical NX
    if ( x >= pixel_pitch )
        return;
    int y = physical_Y_2d_grid( );
    if ( y >= NY )
        return;

    int address = x + y * pixel_pitch;

    if ( x < horizontal_mask_size || y < vertical_mask_size || y >= NY - vertical_mask_size + 1 ) {
        // TODO: confirm this is correct
        // Mask the central cross
        d_input[address].x = 0.f;
        d_input[address].y = 0.f;
    }
    else {
        y = (y < physical_index_of_first_negative_frequency_y) ? y : y - NY;

        float frequency_squared = float(x * x) * fourier_voxel_size_x_sq +
                                  float(y * y) * fourier_voxel_size_y_sq;

        ComplexScale((StorageType*)&d_input[address], expf(-bfactor * frequency_squared));
    }
}

template <typename StorageTypeBase>
void GpuImage::ApplyBFactor<StorageTypeBase>(float bfactor, const float vertical_mask_size, const float horizontal_mask_size) {
    MyDebugAssertFalse(is_in_real_space, "This function is only for Fourier space images.");
    MyDebugAssertTrue(dims.z == 1, "This function is only for 2D images.");

    precheck;
    ReturnLaunchParameters(dims, false);
    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {

        ApplyBFactorKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_fp16,
                                                                                  dims.w / 2,
                                                                                  dims.y,
                                                                                  physical_index_of_first_negative_frequency.y,
                                                                                  fourier_voxel_size.x * fourier_voxel_size.x,
                                                                                  fourier_voxel_size.y * fourier_voxel_size.y,
                                                                                  bfactor * 0.25f,
                                                                                  vertical_mask_size,
                                                                                  horizontal_mask_size);
    }
    else {
        ApplyBFactorKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values,
                                                                                  dims.w / 2,
                                                                                  dims.y,
                                                                                  physical_index_of_first_negative_frequency.y,
                                                                                  fourier_voxel_size.x * fourier_voxel_size.x,
                                                                                  fourier_voxel_size.y * fourier_voxel_size.y,
                                                                                  bfactor * 0.25f,
                                                                                  vertical_mask_size,
                                                                                  horizontal_mask_size);
    }
    postcheck;
}

template void GpuImage::ApplyBFactor<float>(float bfactor, const float vertical_mask_size, const float horizontal_mask_size);
template void GpuImage::ApplyBFactor<__half>(float bfactor, const float vertical_mask_size, const float horizontal_mask_size);

__global__ void
RotationalAveragePSKernel(const __restrict__ cufftComplex* input_values,
#ifdef USE_FP16_FOR_WHITENPS
                          ValueAndWeight_half* rotational_average_ps,
#else
                          ValueAndWeight* rotational_average_ps,
#endif
                          const int   NX,
                          const int   NY,
                          const int   n_bins,
                          const int   n_bins2,
                          const float resolution_limit) {

    int x = physical_X_2d_grid( );
    if ( x >= NX ) {
        return;
    }
    int y = physical_Y_2d_grid( );
    if ( y >= NY ) {
        return;
    }

    // Organize the shared memory as packed int for count, followed by packed float for rotational average
    extern __shared__ ValueAndWeight shared_rotational_average_ps[];

    // initialize temporary accumulation array in shared memory
    // Every thread in the block writes in, and if the block is < n_bins, the slack will be picked up by the strided loop
    // TODO: LaunchParameters that adjusts the size to reduce thread stalling by reducing the remainder of block size and n_bins
    for ( int i = LinearThreadIdxInBlock_2dGrid( ); i < n_bins; i += BlockDimension_2d( ) ) {
        shared_rotational_average_ps[i].value  = 0.f;
        shared_rotational_average_ps[i].weight = 0;
    }
    __syncthreads( );

    float u, v;

    // First, convert the physical coordinate of the 2d projection to the logical Fourier coordinate (in a natural FFT layout).
    u = float(x);
    // First negative logical fourier component is at NY/2
    if ( y >= NY / 2 ) {
        v = float(y) - NY;
    }
    else {
        v = float(y);
    }

    float abs_value = ComplexModulusSquared(input_values[y * NX + x]);
    int   bin       = int(sqrtf(u * u + v * v) / float(NY) * n_bins2);
    // This check is from Image::Whiten, but should it really be checking a float like this?
    if ( abs_value != 0.0 && bin <= resolution_limit ) {
        atomicAdd(&shared_rotational_average_ps[bin].value, abs_value);
        atomicAdd(&shared_rotational_average_ps[bin].weight, 1);
    }
    __syncthreads( );

    // Now write out the partial PS to global memory
    // All the int values are packed for each block, followed by the float values
#ifdef USE_FP16_FOR_WHITENPS
    ValueAndWeight_half* output_rotational_average_ps = &rotational_average_ps[n_bins * LinearBlockIdx_2dGrid( )];
    ValueAndWeight_half  tmp;
    for ( int i = LinearThreadIdxInBlock_2dGrid( ); i < n_bins; i += BlockDimension_2d( ) ) {
        tmp.value                       = __float2bfloat16(shared_rotational_average_ps[i].value);
        tmp.weight                      = short(shared_rotational_average_ps[i].weight);
        output_rotational_average_ps[i] = tmp;
    }
#else
    ValueAndWeight* output_rotational_average_ps = &rotational_average_ps[n_bins * LinearBlockIdx_2dGrid( )];
    for ( int i = LinearThreadIdxInBlock_2dGrid( ); i < n_bins; i += BlockDimension_2d( ) ) {
        output_rotational_average_ps[i] = shared_rotational_average_ps[i];
    }
#endif
};

__global__ void
WhitenKernel(cufftComplex* input_values,
#ifdef USE_FP16_FOR_WHITENPS
             ValueAndWeight_half* rotational_average_ps,
#else
             ValueAndWeight* rotational_average_ps,
#endif
             const int   NX,
             const int   NY,
             const int   n_bins,
             const int   n_bins2,
             const float resolution_limit) {

    int x = physical_X_2d_grid( );
    if ( x >= NX ) {
        return;
    }
    int y = physical_Y_2d_grid( );
    if ( y >= NY ) {
        return;
    }

    int linear_idx = y * NX + x;

    extern __shared__ ValueAndWeight shared_rotational_average_ps[];

    int offset;
#ifdef USE_FP16_FOR_WHITENPS
    ValueAndWeight_half tmp;
#endif

    // Note this only works if n_threads_per_block > n_bins. Then every thread in the block only accesses one address in shared mem.
    // This allows us to skip initializing the shared mem to zero, accumulate in thread registers (*skipping nBlocks writes to shared mem)
    // and then only call one sync at the end.
    float value  = 0.f;
    int   weight = 0;
    for ( int iBlock = 0; iBlock < GridDimension_2d( ); iBlock++ ) {
        offset = n_bins * iBlock;
        for ( int i = LinearThreadIdxInBlock_2dGrid( ); i < n_bins; i += BlockDimension_2d( ) ) {
#ifdef USE_FP16_FOR_WHITENPS
            tmp = rotational_average_ps[offset + i];
            value += __bfloat162float(tmp.value);
            weight += int(tmp.weight);
#else
            value += rotational_average_ps[i + offset].value;
            weight += rotational_average_ps[i + offset].weight;
#endif
        }
    }

    for ( int i = LinearThreadIdxInBlock_2dGrid( ); i < n_bins; i += BlockDimension_2d( ) ) {
        shared_rotational_average_ps[i].value  = value;
        shared_rotational_average_ps[i].weight = weight;
    }
    __syncthreads( );

    int v = y;

    // First negative logical fourier component is at NY/2
    if ( v >= NY / 2 ) {
        v -= NY;
    }

    int bin = int(sqrtf(float(x * x + v * v) / float(NY) * n_bins2));
    // float min_value = sqrtf(radial_average[1] / float(non_zero_count[1]) * 10e-2f);
    if ( bin <= resolution_limit && shared_rotational_average_ps[bin].weight != 0 ) {
        float norm = sqrtf((shared_rotational_average_ps[bin].weight) / (shared_rotational_average_ps[bin].value));
        if ( isfinite(norm) ) {
            input_values[linear_idx].x *= norm;
            input_values[linear_idx].y *= norm;
        }
        else {
            input_values[linear_idx].x = 0.f;
            input_values[linear_idx].y = 0.f;
        }
    }
    else {
        input_values[linear_idx].x = 0.f;
        input_values[linear_idx].y = 0.f;
    }
}

void GpuImage::Whiten(float resolution_limit) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image not in Fourier space");
    MyDebugAssertTrue(dims.z == 1, "Whitening is only setup to work in 2D");

    // First we need to get the rotationally averaged PS, which we'll store in global memory
    ReturnLaunchParameters<ntds_x_WhitenPS, ntds_y_WhitenPS>(dims, false);

    // assuming square images, otherwise this would be largest dimension
    const int number_of_bins = dims.y / 2 + 1;
    // Extend table to include corners in 3D Fourier space
    const int n_bins  = int(number_of_bins * sqrtf(3.0)) + 1;
    const int n_bins2 = 2 * (number_of_bins - 1);

    MyAssertFalse(n_bins > ntds_x_WhitenPS * ntds_y_WhitenPS, "n_bins is too large and would require an array for local variable storage. The size must be  known at compile time so make a template specialization  of the kernel.");
    // For bin resolution of one pixel, uint16 should be plenty
#ifdef USE_FP16_FOR_WHITENPS
    ValueAndWeight_half* rotational_average_ps;
#else
    ValueAndWeight* rotational_average_ps;
#endif

    const int shared_mem = n_bins * sizeof(ValueAndWeight);

    // For coalescing in global memory (2d grid)
    const int   n_blocks               = gridDims.x * gridDims.y;
    const float resolution_limit_pixel = resolution_limit * dims.x;

#ifdef USE_ASYNC_MALLOC_FREE
    cudaErr(cudaMallocAsync(&rotational_average_ps, shared_mem * n_blocks, cudaStreamPerThread));
#else
    cudaErr(cudaMalloc(&rotational_average_ps, shared_mem * n_blocks));
#endif

    precheck;
    RotationalAveragePSKernel<<<gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>>>(complex_values,
                                                                                              rotational_average_ps,
                                                                                              dims.w / 2,
                                                                                              dims.y,
                                                                                              n_bins,
                                                                                              n_bins2,
                                                                                              resolution_limit_pixel);
    postcheck;

    precheck;
    WhitenKernel<<<gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>>>(complex_values,
                                                                                 rotational_average_ps,
                                                                                 dims.w / 2,
                                                                                 dims.y,
                                                                                 n_bins,
                                                                                 n_bins2,
                                                                                 resolution_limit_pixel);
    postcheck;

    cudaErr(cudaFreeAsync(rotational_average_ps, cudaStreamPerThread));
}

__global__ void
_pre_GetWeightedCorrelationWithImageKernel(const __restrict__ cufftComplex* image_values,
                                           const __restrict__ cufftComplex* projection_values,
                                           cufftReal*                       cross_terms,
                                           cufftReal*                       image_PS,
                                           cufftReal*                       projection_PS,
                                           const int                        output_pitch,
                                           const int                        NX,
                                           const int                        NY,
                                           const float                      filter_radius_low_sq,
                                           const float                      filter_radius_high_sq,
                                           const float                      signed_CC_limit_sq) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= NX ) {
        return;
    }
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y >= NY ) {
        return;
    }

    float u, v;
    // low_limit2 = powf(pixel_size / filter_radius_low, 2);

    // First, convert the physical coordinate of the 2d projection to the logical Fourier coordinate (in a natural FFT layout).
    u = float(x) / float(NX);
    // First negative logical fourier component is at NY/2
    if ( y >= NY / 2 ) {
        v = float(y) / float(NY) - 1;
    }
    else {
        v = float(y) / float(NY);
    }

    u = u * u + v * v;
    // We use a real GpuImage to place the real values of the inputs, which will have padding we need to deal with.
    int real_idx    = y * output_pitch + x;
    int complex_idx = y * NX + x;

#ifdef USE_BLOCK_REDUCE
    float sum1, sum2, sum3;
#endif
    bool set_to_zero = false;
    if ( u >= filter_radius_low_sq && u <= filter_radius_high_sq || x > 1 || y > 1 ) {
        v = RealPartOfComplexConjMul(image_values[complex_idx], projection_values[complex_idx]);
        if ( v == 0.f ) {
            set_to_zero = true;
        }
        else {
#ifdef USE_BLOCK_REDUCE
            sum1 = ComplexModulusSquared(image_values[complex_idx]);
            sum2 = ComplexModulusSquared(projection_values[complex_idx]);
            if ( u > signed_CC_limit_sq ) {
                sum3 = fabsf(v);
            }
            else {
                sum3 = v;
            }
#else
            image_PS[real_idx] = ComplexModulusSquared(image_values[complex_idx]);
            projection_PS[real_idx] = ComplexModulusSquared(projection_values[complex_idx]);
            if ( u > signed_CC_limit_sq ) {
                cross_terms[real_idx] = fabsf(v);
            }
            else {
                cross_terms[real_idx] = v;
            }
#endif
        }
    }
    else {
        set_to_zero = true;
    }

    if ( set_to_zero ) {

#ifdef USE_BLOCK_REDUCE
        sum1 = 0.f;
        sum2 = 0.f;
        sum3 = 0.f;
#else
        cross_terms[real_idx] = 0.f;
        image_PS[real_idx] = 0.f;
        projection_PS[real_idx] = 0.f;
#endif
    }
    // cross_terms[real_idx]   = sum3;
    // image_PS[real_idx]      = sum1;
    // projection_PS[real_idx] = sum2;
#ifdef USE_BLOCK_REDUCE
    // Just simplest test for now
    sum1 = blockReduce2dSum(sum1);
    sum2 = blockReduce2dSum(sum2);
    sum3 = blockReduce2dSum(sum3);

    atomicAdd(&s1, sum1);
    atomicAdd(&s2, sum2);
    atomicAdd(&s3, sum3);
#endif
    // // we know the 0th pixel is already zero for now, because we just wrote it
    // if ( threadIdx.x == 0 ) {
    //     cross_terms[blockIdx.x + blockIdx.y * gridDim.x]   = sum3;
    //     image_PS[blockIdx.x + blockIdx.y * gridDim.x]      = sum1;
    //     projection_PS[blockIdx.x + blockIdx.y * gridDim.x] = sum2;
    // }
}

float GpuImage::GetWeightedCorrelationWithImage(GpuImage& projection_image, GpuImage& cross_terms, GpuImage& image_PS, GpuImage& projection_PS, float filter_radius_low_sq, float filter_radius_high_sq, float signed_CC_limit) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(projection_image.is_in_memory_gpu, "Projection Image Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image not in Fourier space");
    MyDebugAssertFalse(projection_image.is_in_real_space, "Projection Image not in Fourier space");
    MyDebugAssertTrue(dims.z == 1, "Get Weighted Correlation is only setup to work in 2D");
    MyDebugAssertTrue(dims.x == dims.y, "Get Weighted Correlation is only setup to work in square 2D");

    // TODO add checks on buffer images.
    // TODO: add nan check per the cpu method

    /* This is quite different in execution than the cpu method, which first calculates the ab* aa* and bb* in resolution rings 
       applies some logic, and then sums rings, then deals with sums.
       This is unecessary though.
       Condition 1: out of resolution limits = 0.f, same for array and ring
       Condition 2: if ab* is identically zero we don't add to any shells, 
       but in practice this can only happen if a == 0.f || b == 0.f, so we can just multiply set a = 0 if b = 0 or vice versa.
       Condition 3: if sum(bb*) in a ring is identically zero, we don't add to the final sums. This is the "perfect" projection image, so it 
       should only be zero in a shell if we set it that way with a mask, and will be zero everywhere in that shell. (There could be an
       anisotropic SSNR weighting that invalidates this logic)

       So just calculate three temp arrays, fitting this logic, sum independently and then work with that.
    */

    // First we need to get the rotationally averaged PS, which we'll store in global memory
    ReturnLaunchParameters(dims, false);

    cross_terms.Zeros( );
    image_PS.Zeros( );
    projection_PS.Zeros( );

    // float* s1;
    // float* s2;
    // float* s3;
    // cudaErr(cudaMallocManaged(&s1, sizeof(float)));
    // cudaErr(cudaMallocManaged(&s2, sizeof(float)));
    // cudaErr(cudaMallocManaged(&s3, sizeof(float)));
    // *s1 = 0.f;
    // *s2 = 0.f;
    // *s3 = 0.f;

    signed_CC_limit *= signed_CC_limit;
    precheck;
    _pre_GetWeightedCorrelationWithImageKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values,
                                                                                                      projection_image.complex_values,
                                                                                                      cross_terms.real_values,
                                                                                                      image_PS.real_values,
                                                                                                      projection_PS.real_values,
                                                                                                      projection_PS.dims.w,
                                                                                                      dims.w / 2,
                                                                                                      dims.y,
                                                                                                      filter_radius_low_sq,
                                                                                                      filter_radius_high_sq,
                                                                                                      signed_CC_limit);

    // int nBlocks = gridDims.x * gridDims.y;

    // dim3 finalGridDims(1, 1, 1);
    // dim3 finalThreadsPerBlock(1024, 1, 1);
    // precheck;
    // _post_GetWeightedCorrelationWithImageKernel<<<finalGridDims, finalThreadsPerBlock, 0, cudaStreamPerThread>>>(cross_terms.real_values);
    // postcheck;

    float final_sum;
#ifdef USE_BLOCK_REDUCE
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    final_sum = *s3;

    *s1 *= *s2;
    if ( *s1 != 0.0 )
        final_sum /= sqrtf(*s1);
#else
    // These should all be in the same stream so no need to synchronize.
    float sum3 = cross_terms.ReturnSumOfRealValues( );
    float sum1 = image_PS.ReturnSumOfRealValues( );
    float sum2 = projection_PS.ReturnSumOfRealValues( );

    sum1 *= sum2;
    if ( sum1 != 0.0 )
        sum3 /= sqrtf(sum1);
    final_sum = sum3;
#endif

    // wxPrintf("sums %f %f %f\n", sum1, sum2, sum3);
    // wxPrintf("sums %f %f %f atomic\n", *s1, *s2, *s3);
    // wxPrintf("sum3 %f\n", sum3);

    // cudaErr(cudaFree(s1));
    // cudaErr(cudaFree(s2));
    // cudaErr(cudaFree(s3));

    return float(final_sum);
}

void GpuImage::CalculateCrossCorrelationImageWith(GpuImage* other_image) {

    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory_gpu, "Other image memory not allocated");
    MyDebugAssertTrue(is_in_real_space == other_image->is_in_real_space, "Images are in different spaces");
    MyDebugAssertTrue(HasSameDimensionsAs(other_image) == true, "Images are different sizes");

    bool must_fft = false;

    // do we have to fft..

    if ( is_in_real_space == true ) {
        must_fft = true;
        ForwardFFT( );
        other_image->ForwardFFT( );
    }

    // TODO: to get a centered XCF we only want to swap one of these, check euler_search_gpu for reference
    if ( object_is_centred_in_box == true ) {
        object_is_centred_in_box = false;
        SwapRealSpaceQuadrants( );
    }

    if ( other_image->object_is_centred_in_box == true ) {
        other_image->object_is_centred_in_box = false;
        other_image->SwapRealSpaceQuadrants( );
    }

    BackwardFFTAfterComplexConjMul(other_image->complex_values, false);

    if ( must_fft == true )
        other_image->BackwardFFT( );
}

Peak GpuImage::FindPeakWithParabolaFit(float inner_radius_for_peak_search, float outer_radius_for_peak_search) {

    MyDebugAssertTrue(is_in_real_space, "This function is only for real space images.");
    MyDebugAssertTrue(dims.z == 1, "This function is only for 2D images.");
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    constexpr bool should_block_until_complete = true;
    constexpr bool free_gpu_memory             = false;

    std::cerr << "Size x,y,z is : " << dims.x << ", " << dims.y << ", " << dims.z << std::endl;
    Image cpu_buffer = CopyDeviceToNewHost(should_block_until_complete, free_gpu_memory);
    Peak  my_peak    = cpu_buffer.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);
    wxPrintf("Peak found at %f, %f\n", my_peak.x, my_peak.y);

    return my_peak;
}

template <typename T>
__global__ void
FindPeakAtOriginFast2DKernel(const T* __restrict__ real_values,
                             IntegerPeak* device_peak,
                             const int    max_pix_x,
                             const int    max_pix_y,
                             const int    NX,
                             const int    NY,
                             const int    pixel_pitch) {

    // To avoid divergence, rather than returning, just let all threads participate, assigning lowlow to those that would
    // otherwise return.
    int x;
    int y;
    int physical_linear_idx;

    float max_val     = -std::numeric_limits<float>::max( );
    float tmp_max_val = -std::numeric_limits<float>::max( );
    int   my_max_idx  = 0;

    for ( int logical_linear_idx = threadIdx.x; logical_linear_idx < NX * NY; logical_linear_idx += blockDim.x ) {
        x = logical_linear_idx % NX;
        y = logical_linear_idx / NX;

        physical_linear_idx = x + (NY + 2) * y + (pixel_pitch * NY * blockIdx.z);

        // physical_linear_idx = x + pixel_pitch * (y + NY * blockIdx.z);

        if ( (x <= max_pix_x || (x >= NX - max_pix_x - 1 && x < NX)) &&
             (y <= max_pix_y || (y >= NY - max_pix_y - 1 && y < NY)) ) {

            if constexpr ( std::is_same<T, __half>::value ) {
                tmp_max_val = gMax(__half2float(real_values[physical_linear_idx]), max_val);
            }
            else {
                tmp_max_val = gMax(float(real_values[physical_linear_idx]), max_val);
            }
            if ( tmp_max_val > max_val ) {
                max_val    = tmp_max_val;
                my_max_idx = physical_linear_idx;
            }
        }
    }
    __syncthreads( );

    blockReduceMax(max_val, my_max_idx);

    if ( threadIdx.x == 0 ) {
        device_peak[blockIdx.z].value                         = max_val;
        device_peak[blockIdx.z].physical_address_within_image = my_max_idx;
        device_peak[blockIdx.z].z                             = int(blockIdx.z);
    }
    return;

    // Okay, we are in bounds so lets find the max value
}

template <typename T>
__global__ void
FindPeakAtCenterFast2DKernel(const T* __restrict__ real_values,
                             IntegerPeak* device_peak,
                             const int    min_pix_x_y,
                             const int    max_pix_x,
                             const int    max_pix_y,
                             const int    NX,
                             const int    NY,
                             const int    pixel_pitch) {

    // To avoid divergence, rather than returning, just let all threads participate, assigning lowlow to those that would
    // otherwise return.
    int x;
    int y;
    int physical_linear_idx;

    float max_val     = -std::numeric_limits<float>::max( );
    float tmp_max_val = -std::numeric_limits<float>::max( );
    int   my_max_idx  = 0;

    for ( int logical_linear_idx = threadIdx.x; logical_linear_idx < NX * NY; logical_linear_idx += blockDim.x ) {
        x = logical_linear_idx % NX;
        y = logical_linear_idx / NX;

        physical_linear_idx = x + pixel_pitch * (y + NY * blockIdx.z);

        // convert to centered coordinates
        x -= NX / 2;
        y -= NY / 2;

        // I don't like these logicals, it would be better to define a mask that can be applied during conj. multiplications
        if ( (x > min_pix_x_y || x < -min_pix_x_y) && (y > min_pix_x_y || y < -min_pix_x_y) ) {

            if ( x < max_pix_x && x > -max_pix_x && y < max_pix_y && y > -max_pix_y ) {

                if constexpr ( std::is_same<T, __half>::value ) {
                    tmp_max_val = gMax(__half2float(real_values[physical_linear_idx]), max_val);
                }
                else {
                    tmp_max_val = gMax(float(real_values[physical_linear_idx]), max_val);
                }
                if ( tmp_max_val > max_val ) {
                    max_val    = tmp_max_val;
                    my_max_idx = physical_linear_idx;
                }
            }
        }
    }
    __syncthreads( );

    blockReduceMax(max_val, my_max_idx);

    if ( threadIdx.x == 0 ) {
        device_peak[blockIdx.z].value                         = max_val;
        device_peak[blockIdx.z].physical_address_within_image = my_max_idx - (pixel_pitch * 2);
        device_peak[blockIdx.z].z                             = int(blockIdx.z);
    }
    return;

    // Okay, we are in bounds so lets find the max value
}

Peak GpuImage::FindPeakAtCenterFast2d(const BatchedSearch& batch, bool load_half_precision) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
    MyDebugAssertTrue(object_is_centred_in_box, "Peak is not centered in image");
    MyDebugAssertTrue(IsEven(dims.y) && IsEven(dims.x), "Image dimensions must be even");

    MyAssertTrue(long(dims.y) * long(dims.w) * long(batch.n_images_in_this_batch( )) < std::numeric_limits<int>::max( ), "int counters will overflow for this call");

    Peak my_peak;

    int max_pix_x   = batch.max_pixel_radius_x( );
    int max_pix_y   = batch.max_pixel_radius_y( );
    int min_pix_x_y = batch.min_pixel_radius_x_y( );

    const int shrink_max_area = 4;
    if ( max_pix_x > dims.x / 2 - shrink_max_area )
        max_pix_x = dims.x / 2 - shrink_max_area;
    if ( max_pix_y > dims.y / 2 - shrink_max_area )
        max_pix_y = dims.y / 2 - shrink_max_area;

    // we only want one block to keep the reduction simple and to a single kernel
    dim3 gd;
    gd = dim3(1, 1, batch.n_images_in_this_batch( ));
    dim3 tpb;
    tpb = dim3(1024, 1, 1);

    if ( load_half_precision ) {
        precheck;
        FindPeakAtCenterFast2DKernel<<<gd, tpb, 0, cudaStreamPerThread>>>(real_values_fp16, batch._d_peak_buffer, min_pix_x_y, max_pix_x, max_pix_y, dims.x, dims.y, dims.w);
        postcheck;
    }
    else {
        precheck;
        FindPeakAtCenterFast2DKernel<<<gd, tpb, 0, cudaStreamPerThread>>>(real_values, batch._d_peak_buffer, min_pix_x_y, max_pix_x, max_pix_y, dims.x, dims.y, dims.w);
        postcheck;
    }

    cudaErr(cudaMemcpyAsync(batch._peak_buffer, batch._d_peak_buffer, batch.n_images_in_this_batch( ) * sizeof(IntegerPeak), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    float       max_val = -std::numeric_limits<float>::max( );
    IntegerPeak tmp_peak;
    for ( int iPeak = 0; iPeak < batch.n_images_in_this_batch( ); iPeak++ ) {
        if ( batch._peak_buffer[iPeak].value > max_val ) {
            max_val                                = batch._peak_buffer[iPeak].value;
            tmp_peak.value                         = batch._peak_buffer[iPeak].value;
            tmp_peak.physical_address_within_image = batch._peak_buffer[iPeak].physical_address_within_image;
            tmp_peak.z                             = batch._peak_buffer[iPeak].z;
        }
    }

    int offset_for_batch_address_in_2d    = (dims.y) * (dims.w) * (tmp_peak.z);
    my_peak.physical_address_within_image = tmp_peak.physical_address_within_image - offset_for_batch_address_in_2d;
    my_peak.value                         = tmp_peak.value;

    my_peak.x = my_peak.physical_address_within_image % (dims.w) - dims.x / 2;
    my_peak.y = my_peak.physical_address_within_image / (dims.w) - dims.y / 2;
    my_peak.z = tmp_peak.z;

    return my_peak;
}

Peak GpuImage::FindPeakAtOriginFast2D(BatchedSearch* batch, bool load_half_precision) {
    MyDebugAssertTrue(batch->is_initialized( ), "BatchedSearch object is not setup!");

    // batch->SetDeviceBuffer( );
    return FindPeakAtOriginFast2D(batch->max_pixel_radius_x( ),
                                  batch->max_pixel_radius_y( ),
                                  batch->_peak_buffer,
                                  batch->_d_peak_buffer,
                                  batch->n_images_in_this_batch( ),
                                  load_half_precision);
}

Peak GpuImage::FindPeakAtOriginFast2D(int max_pix_x, int max_pix_y, IntegerPeak* pinned_host_buffer, IntegerPeak* device_buffer, int wanted_batch_size, bool load_half_precision) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == true, "Image not in real space");
    MyDebugAssertTrue(! object_is_centred_in_box, "Peak centered in image");
    MyDebugAssertTrue(IsEven(dims.y) && IsEven(dims.x), "Image dimensions must be even");

    MyAssertTrue(long(dims.y) * long(dims.w) * long(wanted_batch_size) < std::numeric_limits<int>::max( ), "int counters will overflow for this call");

    Peak my_peak;

    const int shrink_max_area = 1;
    if ( max_pix_x > dims.x / 2 - shrink_max_area )
        max_pix_x = dims.x / 2 - shrink_max_area;
    if ( max_pix_y > dims.y / 2 - shrink_max_area )
        max_pix_y = dims.y / 2 - shrink_max_area;

    // we only want one block to keep the reduction simple and to a single kernel
    dim3 gd;
    gd = dim3(1, 1, wanted_batch_size);
    dim3 tpb;
    tpb = dim3(1024, 1, 1);

    if ( load_half_precision ) {
        precheck;
        FindPeakAtOriginFast2DKernel<<<gd, tpb, 0, cudaStreamPerThread>>>(real_values_fp16, device_buffer, max_pix_x, max_pix_y, dims.x, dims.y, dims.w);
        postcheck;
    }
    else {
        precheck;
        FindPeakAtOriginFast2DKernel<<<gd, tpb, 0, cudaStreamPerThread>>>(real_values, device_buffer, max_pix_x, max_pix_y, dims.x, dims.y, dims.w);
        postcheck;
    }

    cudaErr(cudaMemcpyAsync(pinned_host_buffer, device_buffer, wanted_batch_size * sizeof(IntegerPeak), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    float       max_val = -std::numeric_limits<float>::max( );
    IntegerPeak tmp_peak;
    for ( int iPeak = 0; iPeak < wanted_batch_size; iPeak++ ) {
        if ( pinned_host_buffer[iPeak].value > max_val ) {
            max_val                                = pinned_host_buffer[iPeak].value;
            tmp_peak.value                         = pinned_host_buffer[iPeak].value;
            tmp_peak.physical_address_within_image = pinned_host_buffer[iPeak].physical_address_within_image;
            tmp_peak.z                             = pinned_host_buffer[iPeak].z;
        }
    }

    int offset_for_batch_address_in_2d    = (dims.y) * (dims.w) * (tmp_peak.z);
    my_peak.physical_address_within_image = tmp_peak.physical_address_within_image - offset_for_batch_address_in_2d;
    my_peak.value                         = tmp_peak.value;

    my_peak.x = my_peak.physical_address_within_image % (dims.y + 2);
    my_peak.y = my_peak.physical_address_within_image / (dims.y + 2);
    my_peak.z = tmp_peak.z;

    if ( my_peak.x > dims.x / 2 )
        my_peak.x -= dims.x;
    if ( my_peak.y > dims.y / 2 )
        my_peak.y -= dims.y;

    return my_peak;
}

void GpuImage::Abs( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    nppErr(nppiAbs_32f_C1IR_Ctx((Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::AbsDiff(GpuImage& other_image) {
    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");

    BufferInit(b_image);
    NppInit( );

    nppErr(nppiAbsDiff_32f_C1R_Ctx((const Npp32f*)real_values, pitch,
                                   (const Npp32f*)other_image.real_values, pitch,
                                   (Npp32f*)this->image_buffer->real_values, pitch, npp_ROI, nppStream));

    precheck;
    cudaErr(cudaMemcpyAsync(real_values, this->image_buffer->real_values, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    postcheck;
}

void GpuImage::AbsDiff(GpuImage& other_image, GpuImage& output_image) {
    // In place abs diff (see overload for out of place)
    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    MyDebugAssertTrue(HasSameDimensionsAs(&output_image), "Images have different dimension.");

    NppInit( );

    nppErr(nppiAbsDiff_32f_C1R_Ctx((const Npp32f*)real_values, pitch,
                                   (const Npp32f*)other_image.real_values, pitch,
                                   (Npp32f*)output_image.real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::Min( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_min);
    nppErr(nppiMin_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, min_buffer, (Npp32f*)&min_value, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MinAndCoords( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_minIDX);
    nppErr(nppiMinIndx_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, minIDX_buffer, (Npp32f*)&min_value, &min_idx.x, &min_idx.y, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::Max( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_max);
    nppErr(nppiMax_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, max_buffer, (Npp32f*)&max_value, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MaxAndCoords( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_maxIDX);
    nppErr(nppiMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, maxIDX_buffer, (Npp32f*)&max_value, &max_idx.x, &max_idx.y, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MinMax( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_minmax);
    nppErr(nppiMinMax_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, (Npp32f*)&min_value, (Npp32f*)&max_value, minmax_buffer, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MinMaxAndCoords( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_minmaxIDX);
    nppErr(nppiMinMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, (Npp32f*)&min_value, (Npp32f*)&max_value, &min_idx, &max_idx, minmax_buffer, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::Mean( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in reall space");

    NppInit( );
    BufferInit(b_mean);
    // // wxPrintf("Pitch, roi: %d, %d, %d\n", pitch, npp_ROI.width, npp_ROI.height);

    PrintNppStreamContext( );

    nppErr(nppiMean_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, mean_buffer, &npp_mean, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    this->img_mean = float(npp_mean);
}

void GpuImage::MeanStdDev( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    MyAssertTrue(false, "This function is currently broken, nppErr returns okay, but illegal mem access");
    NppInit( );
    BufferInit(b_meanstddev);

    nppErr(nppiMean_StdDev_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, meanstddev_buffer, &npp_mean, &npp_stdDev, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    this->img_mean   = float(npp_mean);
    this->img_stdDev = float(npp_stdDev);
}

void GpuImage::ReplaceOutliersWithMean(float mean, float stdDev, float maximum_n_sigmas) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    Npp32f max = mean + maximum_n_sigmas * stdDev;
    Npp32f min = mean - maximum_n_sigmas * stdDev;
    nppErr(nppiThreshold_LTValGTVal_32f_C1IR_Ctx((Npp32f*)real_values, pitch, npp_ROI, min, (Npp32f)mean, max, (Npp32f)mean, nppStream));
}

void GpuImage::ReplaceOutliersWithMean(float maximum_n_sigmas) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    MeanStdDev( );
    ReplaceOutliersWithMean(img_mean, img_stdDev, maximum_n_sigmas);
}

void GpuImage::MultiplyPixelWise(const float& other_array, const int other_array_size) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Not in Fourier space");
    MyDebugAssertTrue(other_array_size == real_memory_allocated / 2, "Array size does not match image size");

    NppInit( );
    nppErr(nppiMul_32fc_C1IR_Ctx((Npp32fc*)&other_array, pitch, (Npp32fc*)complex_values, pitch, npp_ROI, nppStream));
}

void GpuImage::MultiplyPixelWise(GpuImage& other_image) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    NppInit( );
    if ( is_in_real_space ) {
        nppErr(nppiMul_32f_C1IR_Ctx((Npp32f*)other_image.real_values, pitch, (Npp32f*)real_values, pitch, npp_ROI, nppStream));
    }
    else {
        nppErr(nppiMul_32fc_C1IR_Ctx((Npp32fc*)other_image.complex_values, pitch, (Npp32fc*)complex_values, pitch, npp_ROI, nppStream));
    }
    // wxPrintf("Ret val %d\n", ret_val);
    // wxPrintf("ROI %d %d\n", npp_ROI.width, npp_ROI.height);
    // wxPrintf("Pitch %ld\n", pitch);
    // wxPrintf("Is in real space %d\n", is_in_real_space);
    // wxPrintf("Other is in real space %d\n", other_image.is_in_real_space);
    // exit(0);
    // if ( is_in_real_space ) {
    //     nppErr(nppiMul_32f_C1IR_Ctx((Npp32f*)other_image.real_values, pitch, (Npp32f*)real_values, pitch, npp_ROI, nppStream));
    // }
    // else {
    //     nppErr(nppiMul_32fc_C1IR_Ctx((Npp32fc*)other_image.complex_values, pitch, (Npp32fc*)complex_values, pitch, npp_ROI, nppStream));
    // }
}

// Same as above but out of place
void GpuImage::MultiplyPixelWise(GpuImage& other_image, GpuImage& output_image) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    precheck;
    if ( is_in_real_space ) {
        nppErr(nppiMul_32f_C1R_Ctx((Npp32f*)other_image.real_values, pitch,
                                   (Npp32f*)real_values, pitch,
                                   (Npp32f*)output_image.real_values, pitch,
                                   npp_ROI, nppStream));
    }
    else {
        nppErr(nppiMul_32fc_C1R_Ctx((Npp32fc*)other_image.complex_values, pitch,
                                    (Npp32fc*)complex_values, pitch,
                                    (Npp32fc*)output_image.complex_values, pitch,
                                    npp_ROI, nppStream));
    }
    postcheck;
}

void GpuImage::AddConstant(const float add_val) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiAddC_32f_C1IR_Ctx((Npp32f)add_val, (Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::AddConstant(const Npp32fc add_val) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image in real space.")

            NppInit( );
    nppErr(nppiAddC_32fc_C1IR_Ctx((Npp32fc)add_val, (Npp32fc*)complex_values, pitch, npp_ROI, nppStream));
}

void GpuImage::SquareRealValues( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiSqr_32f_C1IR_Ctx((Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::SquareRootRealValues( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiSqrt_32f_C1IR_Ctx((Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::LogarithmRealValues( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    nppErr(nppiLn_32f_C1IR_Ctx((Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::ExponentiateRealValues( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiExp_32f_C1IR_Ctx((Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::CountInRange(float lower_bound, float upper_bound) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiCountInRange_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, &number_of_pixels_in_range,
                                        (Npp32f)lower_bound, (Npp32f)upper_bound, countinrange_buffer, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

float GpuImage::ReturnSumOfRealValues( ) {
    // FIXME assuming padded values are zero
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_sum);
    nppErr(nppiSum_32f_C1R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, sum_buffer, (Npp64f*)tmpValComplex, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    return (float)*tmpValComplex;
}

// float3 GpuImage::ReturnSumOfRealValues3Channel() {
//     // FIXME assuming padded values are zero
//     MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
//     MyDebugAssertTrue(is_in_real_space, "Not in real space");

//     NppInit( );
//     BufferInit(b_sum);
//     nppErr(nppiSum_32f_C3R_Ctx((const Npp32f*)real_values, pitch, npp_ROI, sum_buffer, (Npp64f*)tmpValComplex, nppStream));
//     cudaErr(cudaStreamSynchronize(nppStream.hStream));

//     return (float3)*tmpValComplex;
// }

template <typename StorageType>
__global__ void
AddImageStackKernel(StorageType** stack_ptrs,
                    StorageType*  output,
                    const int     NX,
                    const int     NY,
                    const size_t  NZ,
                    const int     pixel_pitch) {
    int x = physical_X_2d_grid( );
    if ( x >= NX )
        return;
    int y = physical_Y_2d_grid( );
    if ( y >= NY )
        return;

    float sum = 0.0f;
    // Whether we are in real space or not doesn't matter, the padding values would
    // only affect the padding values, st. NX = pixel_pitch

    int address = x + pixel_pitch * y;
    for ( int z = 0; z < NZ; z++ ) {
        if constexpr ( std::is_same<StorageType, __half>::value ) {
            // COnvert
            sum += __half2float(stack_ptrs[z][address]);
        }
        else {
            sum += stack_ptrs[z][address];
        }
    }

    if constexpr ( std::is_same<StorageType, __half>::value ) {
        // COnvert
        output[address] = __float2half(sum);
    }
    else {
        output[address] = sum;
    }
}

template <typename StorageTypeBase>
void GpuImage::AddImageStack(std::vector<GpuImage>& input_stack) {
    AddImageStack<StorageTypeBase>(input_stack, *this);
    return;
}

template void GpuImage::AddImageStack<float>(std::vector<GpuImage>& input_stack);
template void GpuImage::AddImageStack<__half>(std::vector<GpuImage>& input_stack);

/**
 * @brief Sum an image stack and place the results in the output image.
*/
template <typename StorageTypeBase>
void GpuImage::AddImageStack(std::vector<GpuImage>& input_stack, GpuImage& output_image) {
    MyDebugAssertTrue(input_stack[0].HasSameDimensionsAs<StorageTypeBase>(output_image), "Images have different dimensions");

    // Maybe check all the images in the stack?
    MyDebugAssertTrue(input_stack[0].is_in_real_space == output_image.is_in_real_space, "Images not in the same space");

    constexpr bool use_real_space_grid_for_any_type = true;
    this->ReturnLaunchParameters(this->dims, use_real_space_grid_for_any_type);

    precheck;
    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {
        MyDebugAssertTrue(input_stack[0].is_allocated_16f_buffer, "Memory not allocated");
        MyDebugAssertTrue(output_image.is_allocated_16f_buffer, "Output Memory not allocated");
        // No-op if already allocated
        output_image.ptr_array_16f.resize(input_stack.size( ));
        for ( int iPtr = 0; iPtr < input_stack.size( ); iPtr++ ) {
            output_image.ptr_array_16f.SetPointer((__half*)input_stack[iPtr].real_values_16f, iPtr);
        }
        AddImageStackKernel<<<this->gridDims, this->threadsPerBlock, 0, cudaStreamPerThread>>>(output_image.ptr_array_16f.ptr_array,
                                                                                               (__half*)output_image.real_values_16f,
                                                                                               this->dims.x,
                                                                                               this->dims.y,
                                                                                               input_stack.size( ),
                                                                                               this->dims.w);
    }
    else {
        MyDebugAssertTrue(input_stack[0].is_in_memory_gpu, "Memory not allocated");
        MyDebugAssertTrue(output_image.is_in_memory_gpu, "Output Memory not allocated");
        output_image.ptr_array_32f.resize(input_stack.size( ));
        for ( int iPtr = 0; iPtr < input_stack.size( ); iPtr++ ) {
            output_image.ptr_array_32f.SetPointer((float*)input_stack[iPtr].real_values, iPtr);
        }

        AddImageStackKernel<<<this->gridDims, this->threadsPerBlock, 0, cudaStreamPerThread>>>(output_image.ptr_array_32f.ptr_array,
                                                                                               (float*)output_image.real_values,
                                                                                               this->dims.x,
                                                                                               this->dims.y,
                                                                                               input_stack.size( ),
                                                                                               this->dims.w);
    }
    postcheck;
}

//a
template void GpuImage::AddImageStack<float>(std::vector<GpuImage>&, GpuImage&);
template void GpuImage::AddImageStack<__half>(std::vector<GpuImage>&, GpuImage&);

void GpuImage::AddImage(GpuImage& other_image) {
    // Add the real_values into a double array
    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

    NppInit( );
    nppErr(nppiAdd_32f_C1IR_Ctx((const Npp32f*)other_image.real_values, pitch, (Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

template <typename StorageTypeBase>
void GpuImage::SubtractImage(GpuImage& other_image) {
    // Add the real_values into a double array
    MyDebugAssertTrue(HasSameDimensionsAs<StorageTypeBase>(&other_image), "Images have different dimensions");

    NppInit( );

    // I think I can just use the same buffer (even though it is overkill) for fp16

    // TODO: I don't see any reason why we need a special method for complex values.
    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {
        // FIXME: make is_in_real_space a private member such that npp_ROI is set correctly when this is flipped.

        nppErr(nppiSub_16f_C1IR_Ctx((const Npp16f*)other_image.real_values_fp16, pitch / 2, (Npp16f*)real_values_fp16, pitch / 2, npp_ROI_real_space, nppStream));
    }
    else {
        // FIXME: make is_in_real_space a private member such that npp_ROI is set correctly when this is flipped.
        // if ( is_in_real_space ) {
        nppErr(nppiSub_32f_C1IR_Ctx((const Npp32f*)other_image.real_values, pitch, (Npp32f*)real_values, pitch, npp_ROI_real_space, nppStream));
        // }
        // else {
        //     nppErr(nppiSub_32fc_C1IR_Ctx((const Npp32fc*)other_image.complex_values, pitch, (Npp32fc*)complex_values, pitch, npp_ROI_fourier_space, nppStream));
        // }
    }
}

template void GpuImage::SubtractImage<float>(GpuImage& other_image);
template void GpuImage::SubtractImage<__half>(GpuImage& other_image);

void GpuImage::AddSquaredImage(GpuImage& other_image) {
    // Add the real_values into a double array
    MyDebugAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");
    MyDebugAssertTrue(is_in_real_space, "Image is not in real space");

    NppInit( );
    nppErr(nppiAddSquare_32f_C1IR_Ctx((const Npp32f*)other_image.real_values, pitch, (Npp32f*)real_values, pitch, npp_ROI, nppStream));
}

void GpuImage::MultiplyByConstant(float scale_factor) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    if ( is_in_real_space ) {
        nppErr(nppiMulC_32f_C1IR_Ctx((Npp32f)scale_factor, (Npp32f*)real_values, pitch, npp_ROI, nppStream));
    }
    else {
        nppErr(nppiMulC_32f_C1IR_Ctx((Npp32f)scale_factor, (Npp32f*)real_values, pitch, npp_ROI_fourier_with_real_functor, nppStream));
    }
}

void GpuImage::SetToConstant(float scale_factor) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    if ( is_in_real_space ) {
        nppErr(nppiSet_32f_C1R_Ctx((Npp32f)scale_factor, (Npp32f*)real_values, pitch, npp_ROI, nppStream));
    }
    else {
        Npp32fc scale_factor_complex = {scale_factor, scale_factor};
        SetToConstant(scale_factor_complex);
    }
}

void GpuImage::SetToConstant(Npp32fc scale_factor_complex) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    nppErr(nppiSet_32fc_C1R_Ctx((Npp32fc)scale_factor_complex, (Npp32fc*)complex_values, pitch, npp_ROI_fourier_space, nppStream));
}

void GpuImage::Conj( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Conj only supports complex images");

    Npp32fc scale_factor;
    scale_factor.re = 1.0f;
    scale_factor.im = -1.0f;
    NppInit( );
    nppErr(nppiMulC_32fc_C1IR_Ctx((Npp32fc)scale_factor, (Npp32fc*)complex_values, pitch, npp_ROI, nppStream));
}

template <typename StorageTypeBase>
void GpuImage::Zeros<StorageTypeBase>( ) {

    MyDebugAssertFalse(real_memory_allocated == 0, "Host meta data has not been copied");

    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {
        BufferInit(b_16f);
        cudaErr(cudaMemsetAsync(real_values_16f, 0, real_memory_allocated * sizeof(__half), cudaStreamPerThread));
    }

    else {

        if ( ! is_in_memory_gpu ) {
#ifdef USE_ASYNC_MALLOC_FREE
            cudaErr(cudaMallocAsync(&real_values, real_memory_allocated * sizeof(float), cudaStreamPerThread));
#else
            cudaErr(cudaMalloc(&real_values, real_memory_allocated * sizeof(float)));
#endif
            complex_values   = (cufftComplex*)real_values;
            is_in_memory_gpu = true;
        }

        cudaErr(cudaMemsetAsync(real_values, 0, real_memory_allocated * sizeof(float), cudaStreamPerThread));
    }
}

template void GpuImage::Zeros<float>( );
template void GpuImage::Zeros<__half>( );

void GpuImage::CopyHostToDevice(bool should_block_until_complete) {
    MyDebugAssertTrue(host_image_ptr && host_image_ptr->is_in_memory, "Host image not allocated");
    if ( ! is_in_memory_gpu ) {
        Allocate(dims.x, dims.y, dims.z, host_image_ptr->is_in_real_space);
    }

    if ( host_image_ptr->page_locked_ptr != nullptr ) {
        precheck;
        cudaErr(cudaMemcpyAsync(real_values, host_image_ptr->page_locked_ptr, real_memory_allocated * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        postcheck;
    }
    else {
        // FIXME: This is currently needed when copying in a subset of a larger image (batched) where the individual image is not
        // owning (and shows not page locked.) There should probably be a seperate copy function for a sbuset/betch that takes an offset to the owning memory pointer instead.
        precheck;
        cudaErr(cudaMemcpyAsync(real_values, host_image_ptr->real_values, real_memory_allocated * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        postcheck;
    }

    CopyCpuImageMetaData(*host_image_ptr);
    if ( should_block_until_complete ) {
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

void GpuImage::CopyHostToDeviceTextureComplex3d( ) {

    MyDebugAssertTrue(host_image_ptr && host_image_ptr->is_in_memory, "Host memory not allocated");
    MyDebugAssertFalse(is_allocated_texture_cache, "CopyHostToDeviceTextureComplex3d should only be called once");
    MyDebugAssertTrue(dims.x == dims.y && dims.y == dims.z, "CopyHostToDeviceTextureComplex3d only supports cubic 3d host images");
    MyDebugAssertTrue(is_fft_centered_in_box, "CopyHostToDeviceTextureComplex3d only supports fft_centered_in_box");

    int padded_x_dimension = dims.w / 2;
    // We need a temporary host array so we can both de-interlace the real and imaginary parts as well as including x = -1 padding.
    float* host_array_real = new float[padded_x_dimension * dims.y * dims.z];
    float* host_array_imag = new float[padded_x_dimension * dims.y * dims.z];

    for ( int complex_address = 0; complex_address < host_image_ptr->real_memory_allocated / 2; complex_address++ ) {
        host_array_real[complex_address] = real(host_image_ptr->complex_values[complex_address]);
        host_array_imag[complex_address] = imag(host_image_ptr->complex_values[complex_address]);
    }

    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>( );
    cudaExtent            extent      = make_cudaExtent(padded_x_dimension, dims.y, dims.z);

    // Allocate the texture arrays including the padded negative frequency. It seems like this is not part of the Stream ordered allocation yet.
    // TODO: Try fp16 to reduce memory footprint
    cudaErr(cudaMalloc3DArray(&cuArray_real, &channelDesc, extent, cudaArrayDefault));
    cudaErr(cudaMalloc3DArray(&cuArray_imag, &channelDesc, extent, cudaArrayDefault));

    is_allocated_texture_cache = true;

    // Copy over the real array
    cudaMemcpy3DParms p_real = {0};
    p_real.extent            = extent;
    p_real.srcPtr            = make_cudaPitchedPtr(host_array_real, (padded_x_dimension) * sizeof(float), padded_x_dimension, dims.y);
    p_real.dstArray          = cuArray_real;
    p_real.kind              = cudaMemcpyHostToDevice;

    cudaErr(cudaMemcpy3DAsync(&p_real, cudaStreamPerThread));

    // Copy over the real array
    cudaMemcpy3DParms p_imag = {0};
    p_imag.extent            = extent;
    p_imag.srcPtr            = make_cudaPitchedPtr(host_array_imag, (padded_x_dimension) * sizeof(float), padded_x_dimension, dims.y);
    p_imag.dstArray          = cuArray_imag;
    p_imag.kind              = cudaMemcpyHostToDevice;

    cudaErr(cudaMemcpy3DAsync(&p_imag, cudaStreamPerThread));

    // TODO checkout cudaCreateChannelDescHalf https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#sixteen-bit-floating-point-textures
    struct cudaResourceDesc resDesc_real;
    (memset(&resDesc_real, 0, sizeof(resDesc_real)));
    resDesc_real.resType         = cudaResourceTypeArray;
    resDesc_real.res.array.array = cuArray_real;

    struct cudaResourceDesc resDesc_imag;
    (memset(&resDesc_imag, 0, sizeof(resDesc_imag)));
    resDesc_imag.resType         = cudaResourceTypeArray;
    resDesc_imag.res.array.array = cuArray_imag;

    struct cudaTextureDesc texDesc_real;
    (memset(&texDesc_real, 0, sizeof(texDesc_real)));

    texDesc_real.filterMode       = cudaFilterModeLinear;
    texDesc_real.readMode         = cudaReadModeElementType;
    texDesc_real.normalizedCoords = false;
    texDesc_real.addressMode[0]   = cudaAddressModeBorder; //cudaAddressModeClamp;
    texDesc_real.addressMode[1]   = cudaAddressModeBorder; //cudaAddressModeClamp;
    texDesc_real.addressMode[2]   = cudaAddressModeBorder; //cudaAddressModeClamp;

    struct cudaTextureDesc texDesc_imag;
    (memset(&texDesc_imag, 0, sizeof(texDesc_imag)));

    texDesc_imag.filterMode       = cudaFilterModeLinear;
    texDesc_imag.readMode         = cudaReadModeElementType;
    texDesc_imag.normalizedCoords = false;
    texDesc_imag.addressMode[0]   = cudaAddressModeBorder; //cudaAddressModeClamp;
    texDesc_imag.addressMode[1]   = cudaAddressModeBorder; //cudaAddressModeClamp;
    texDesc_imag.addressMode[2]   = cudaAddressModeBorder; //cudaAddressModeClamp;

    cudaErr(cudaCreateTextureObject(&tex_real, &resDesc_real, &texDesc_real, NULL));
    cudaErr(cudaCreateTextureObject(&tex_imag, &resDesc_imag, &texDesc_imag, NULL));

    delete[] host_array_real;
    delete[] host_array_imag;
}

void GpuImage::CopyHostToDevice16f(bool should_block_until_finished) {
    MyDebugAssertTrue(host_image_ptr != nullptr, "Host image not allocated");
    MyDebugAssertTrue(host_image_ptr->is_in_memory_16f, "Host memory not allocated");
    MyDebugAssertFalse(host_image_ptr->is_in_real_space, "CopyHostRealPartToDevice should only be called for complex images");
    MyDebugAssertTrue(host_image_ptr->real_memory_allocated_16f == real_memory_allocated, "Host memory size mismatch");

    CopyCpuImageMetaData(*host_image_ptr);

    BufferInit(b_ctf_16f, real_memory_allocated);
    half_float::half* tmpPinnedPtr;

    // FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning
    cudaErr(cudaHostRegister(host_image_ptr->real_values_16f, sizeof(half_float::half) * real_memory_allocated, cudaHostRegisterDefault));
    cudaErr(cudaHostGetDevicePointer(&tmpPinnedPtr, host_image_ptr->real_values_16f, 0));

    // always unregister the temporary pointer as it is not associated with a GpuImage
    precheck;
    cudaErr(cudaMemcpyAsync((void*)ctf_buffer_16f, tmpPinnedPtr, real_memory_allocated * sizeof(half_float::half), cudaMemcpyHostToDevice, cudaStreamPerThread));
    postcheck;

    if ( should_block_until_finished ) {
        cudaError(cudaStreamSynchronize(cudaStreamPerThread));
    }
    else {
        RecordAndWait( );
    }
    cudaErr(cudaHostUnregister(tmpPinnedPtr));
}

void GpuImage::CopyDeviceToHostAndSynchronize(bool free_gpu_memory, bool unpin_host_memory) {
    CopyDeviceToHost(free_gpu_memory, unpin_host_memory);
    cudaError(cudaStreamSynchronize(cudaStreamPerThread));
}

void GpuImage::CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory) {
    MyDebugAssertTrue(host_image_ptr != nullptr && host_image_ptr->is_in_memory, "Host image not allocated");
    MyDebugAssertTrue(host_image_ptr->page_locked_ptr != nullptr, "Host image not page locked");
    MyDebugAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
    // TODO other asserts on size etc.

    cudaErr(cudaMemcpyAsync(host_image_ptr->page_locked_ptr, real_values, real_memory_allocated * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    //  cudaErr(cudaMemcpyAsync(real_values, real_values, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    // TODO add asserts etc.
    if ( free_gpu_memory ) {
        Deallocate( );
    }
    // If we ran Deallocate, this the memory is already unpinned.
    if ( unpin_host_memory ) {
        host_image_ptr->UnRegisterPageLockedMemory( );
    }
}

void GpuImage::CopyDeviceToHost(Image& cpu_image, bool should_block_until_complete, bool free_gpu_memory, bool unpin_host_memory) {
    MyDebugAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
    MyDebugAssertTrue(cpu_image.is_in_memory, "CPU memory not allocated");
    MyDebugAssertTrue(HasSameDimensionsAs(&cpu_image), "CPU image size mismatch");

    // TODO other asserts on size etc.
    cpu_image.RegisterPageLockedMemory( );

    precheck;
    cudaErr(cudaMemcpyAsync(cpu_image.real_values, real_values, real_memory_allocated * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    postcheck;

    if ( should_block_until_complete )
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    // TODO add asserts etc.
    if ( free_gpu_memory ) {
        Deallocate( );
    }
    if ( unpin_host_memory ) {
        cpu_image.UnRegisterPageLockedMemory( );
    }
}

void GpuImage::CopyDeviceToNewHost(Image& cpu_image, bool should_block_until_complete, bool free_gpu_memory, bool unpin_host_memory) {
    cpu_image.logical_x_dimension = dims.x;
    cpu_image.logical_y_dimension = dims.y;
    cpu_image.logical_z_dimension = dims.z;
    cpu_image.padding_jump_value  = dims.w - dims.x;

    cpu_image.physical_upper_bound_complex_x = physical_upper_bound_complex.x;
    cpu_image.physical_upper_bound_complex_y = physical_upper_bound_complex.y;
    cpu_image.physical_upper_bound_complex_z = physical_upper_bound_complex.z;

    cpu_image.physical_address_of_box_center_x = physical_address_of_box_center.x;
    cpu_image.physical_address_of_box_center_y = physical_address_of_box_center.y;
    cpu_image.physical_address_of_box_center_z = physical_address_of_box_center.z;

    // when copied to GPU image, this is set to 0. not sure if required to be copied.
    //cpu_image.physical_index_of_first_negative_frequency_x = physical_index_of_first_negative_frequency.x;

    cpu_image.physical_index_of_first_negative_frequency_y = physical_index_of_first_negative_frequency.y;
    cpu_image.physical_index_of_first_negative_frequency_z = physical_index_of_first_negative_frequency.z;

    cpu_image.logical_upper_bound_complex_x = logical_upper_bound_complex.x;
    cpu_image.logical_upper_bound_complex_y = logical_upper_bound_complex.y;
    cpu_image.logical_upper_bound_complex_z = logical_upper_bound_complex.z;

    cpu_image.logical_lower_bound_complex_x = logical_lower_bound_complex.x;
    cpu_image.logical_lower_bound_complex_y = logical_lower_bound_complex.y;
    cpu_image.logical_lower_bound_complex_z = logical_lower_bound_complex.z;

    cpu_image.logical_upper_bound_real_x = logical_upper_bound_real.x;
    cpu_image.logical_upper_bound_real_y = logical_upper_bound_real.y;
    cpu_image.logical_upper_bound_real_z = logical_upper_bound_real.z;

    cpu_image.logical_lower_bound_real_x = logical_lower_bound_real.x;
    cpu_image.logical_lower_bound_real_y = logical_lower_bound_real.y;
    cpu_image.logical_lower_bound_real_z = logical_lower_bound_real.z;

    cpu_image.is_in_real_space = is_in_real_space;

    cpu_image.number_of_real_space_pixels = number_of_real_space_pixels;
    cpu_image.object_is_centred_in_box    = object_is_centred_in_box;

    cpu_image.fourier_voxel_size_x = fourier_voxel_size.x;
    cpu_image.fourier_voxel_size_y = fourier_voxel_size.y;
    cpu_image.fourier_voxel_size_z = fourier_voxel_size.z;

    cpu_image.insert_into_which_reconstruction = insert_into_which_reconstruction;

    // cpu_image.real_values = real_values;

    // cpu_image.complex_values = complex_values;

    //cpu_image.padding_jump_value = padding_jump_value;
    cpu_image.image_memory_should_not_be_deallocated = image_memory_should_not_be_deallocated;

    //cpu_image.real_memory_allocated = real_memory_allocated;
    cpu_image.ft_normalization_factor = ft_normalization_factor;

    CopyDeviceToHost(cpu_image, should_block_until_complete, free_gpu_memory, unpin_host_memory);

    cpu_image.is_in_memory     = true;
    cpu_image.is_in_real_space = is_in_real_space;
}

// TODO: should the return type be moved, is that handled already by the compiler?
Image GpuImage::CopyDeviceToNewHost(bool should_block_until_complete, bool free_gpu_memory, bool unpin_host_memory) {
    MyDebugAssertTrue(is_in_memory_gpu, "Image is not in memory on the GPU");

    Image new_cpu_image(dims.x, dims.y, dims.z, true, true);

    //new_cpu_image.Allocate(dims.x,dims.y);
    CopyDeviceToNewHost(new_cpu_image, should_block_until_complete, free_gpu_memory);
    return new_cpu_image;
}

template <class InputType, class OutputType>
void GpuImage::_ForwardFFT( ) {
}

template <>
void GpuImage::_ForwardFFT<float, float2>( ) {
    cufftErr(cufftExecR2C(cuda_plan_forward, (cufftReal*)position_space_ptr, (cufftComplex*)momentum_space_ptr));
}

void GpuImage::ForwardFFTBatched(bool should_scale) {
    MyDebugAssertTrue(dims.y > 1 && dims.x > 1 && dims.z > 1, "ForwardFFTBatched is only implemented for a stack of 2D images");
    MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image alread in Fourier space");

    cufft_batch_size = dims.z;

    SetCufftPlan(cistem::fft_type::Enum::inplace_32f_32f_32f_batched, (void*)real_values, (void*)complex_values);

    // FIXME, this should be a load call back
    if ( should_scale ) {
        this->MultiplyByConstant(ft_normalization_factor * ft_normalization_factor);
    }
    _ForwardFFT<float, float2>( );
    // cudaErr(cufftExecR2C(this->cuda_plan_forward, position_space_ptr, momentum_space_ptr));

    is_in_real_space = false;
    if ( host_image_ptr )
        host_image_ptr->is_in_real_space = false;
    npp_ROI = npp_ROI_fourier_space;
}

void GpuImage::ForwardFFT(bool should_scale) {

    bool is_half_precision = false;

    MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image alread in Fourier space");

    SetCufftPlan(cistem::fft_type::Enum::inplace_32f_32f_32f, (void*)real_values, (void*)complex_values);

    // For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
    if ( is_half_precision && ! is_set_convertInputf16Tof32 ) {
        cufftCallbackLoadR h_ConvertInputf16Tof32Ptr;
        cudaErr(cudaMemcpyFromSymbol(&h_ConvertInputf16Tof32Ptr, d_ConvertInputf16Tof32Ptr, sizeof(h_ConvertInputf16Tof32Ptr)));
        cufftErr(cufftXtSetCallback(cuda_plan_forward, (void**)&h_ConvertInputf16Tof32Ptr, CUFFT_CB_LD_REAL, 0));
        is_set_convertInputf16Tof32 = true;
        //      cudaErr(cudaFree(norm_factor));
        //      this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
    }
    if ( should_scale ) {
        this->MultiplyByConstant(ft_normalization_factor * ft_normalization_factor);
    }

    //    if (should_scale && ! is_set_scaleFFTAndStore)
    //    {
    //
    //        float ft_norm_sq = ft_normalization_factor*ft_normalization_factor;
    //        cudaErr(cudaMalloc((void **)&d_scale_factor, sizeof(float)));
    //        cudaErr(cudaMemcpyAsync(d_scale_factor, &ft_norm_sq, sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
    //        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    //
    //        cufftCallbackStoreC h_scaleFFTAndStorePtr;
    //        cudaErr(cudaMemcpyFromSymbol(&h_scaleFFTAndStorePtr,d_scaleFFTAndStorePtr, sizeof(h_scaleFFTAndStorePtr)));
    //        cudaErr(cufftXtSetCallback(cuda_plan_forward, (void **)&h_scaleFFTAndStorePtr, CUFFT_CB_ST_COMPLEX, (void **)&d_scale_factor));
    //        is_set_scaleFFTAndStore = true;
    //    }

    //    BufferInit(b_image);
    //    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values, (cufftComplex*)image_buffer->complex_values));

    _ForwardFFT<float, float2>( );
    // cudaErr(cufftExecR2C(this->cuda_plan_forward, position_space_ptr, momentum_space_ptr));

    is_in_real_space = false;
    // FIXME: This doesn't really make sense unless we are sure we will copy the image back to the host
    // Instead, we should set the host flag appropriately at that time.
    if ( host_image_ptr )
        host_image_ptr->is_in_real_space = false;
    npp_ROI = npp_ROI_fourier_space;
}

void GpuImage::ForwardFFTAndClipInto(GpuImage& image_to_insert, bool should_scale) {

    MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyDebugAssertTrue(image_to_insert.is_in_memory_gpu, "Gpu memory in image to insert not allocated");
    MyDebugAssertTrue(is_in_real_space, "Image alread in Fourier space");
    MyDebugAssertTrue(image_to_insert.is_in_real_space, "I in image to insert alread in Fourier space");

    SetCufftPlan(cistem::fft_type::Enum::inplace_32f_32f_32f, (void*)real_values, (void*)complex_values);

    // For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
    if ( ! is_set_realLoadAndClipInto ) {

        // We need to make the mask
        image_to_insert.ClipIntoReturnMask(this);

        cufftCallbackLoadR             h_realLoadAndClipInto;
        CB_realLoadAndClipInto_params* d_params;
        CB_realLoadAndClipInto_params  h_params;

        h_params.target = (cufftReal*)image_to_insert.real_values;
        h_params.mask   = (int*)clip_into_mask;
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaMallocAsync((void**)&d_params, sizeof(CB_realLoadAndClipInto_params), cudaStreamPerThread));
#else
        cudaErr(cudaMalloc((void**)&d_params, sizeof(CB_realLoadAndClipInto_params)));
#endif

        cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_realLoadAndClipInto_params), cudaMemcpyHostToDevice, cudaStreamPerThread));
        cudaErr(cudaMemcpyFromSymbol(&h_realLoadAndClipInto, d_realLoadAndClipInto, sizeof(h_realLoadAndClipInto)));
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

        cufftErr(cufftXtSetCallback(cuda_plan_forward, (void**)&h_realLoadAndClipInto, CUFFT_CB_LD_REAL, (void**)&d_params));
        is_set_realLoadAndClipInto = true;

        //      cudaErr(cudaFree(norm_factor));
        //      this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
    }
    if ( should_scale ) {
        this->MultiplyByConstant(ft_normalization_factor * ft_normalization_factor);
    }

    //    BufferInit(b_image);
    //    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values, (cufftComplex*)image_buffer->complex_values));

    _ForwardFFT<float, float2>( );

    // cudaErr(cufftExecR2C(this->cuda_plan_forward, position_space_ptr, momentum_space_ptr));

    is_in_real_space = false;
    if ( host_image_ptr )
        host_image_ptr->is_in_real_space = false;

    npp_ROI = npp_ROI_fourier_space;
}

template <>
void GpuImage::_BackwardFFT<float, float2>( ) {
    cufftErr(cufftExecC2R(cuda_plan_inverse, (cufftComplex*)momentum_space_ptr, (cufftReal*)position_space_ptr));
}

void GpuImage::BackwardFFTBatched(int wanted_batch_size) {
    // MyDebugAssertTrue(dims.y > 1 && dims.x > 1 && dims.z > 1, "ForwardFFTBatched is only implemented for a stack of 2D images");
    MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image is already in real space");

    cufft_batch_size = (wanted_batch_size == 0) ? dims.z : wanted_batch_size;

    SetCufftPlan(cistem::fft_type::Enum::inplace_32f_32f_32f_batched, (void*)real_values, (void*)complex_values);

    _BackwardFFT<float, float2>( );
    // cudaErr(cufftExecC2R(this->cuda_plan_inverse, momentum_space_ptr, position_space_ptr));

    is_in_real_space = true;
    // FIXME: This doesn't really make sense unless we are sure we will copy the image back to the host
    // Instead, we should set the host flag appropriately at that time.
    if ( host_image_ptr )
        host_image_ptr->is_in_real_space = true;

    npp_ROI = npp_ROI_real_space;
}

void GpuImage::BackwardFFT( ) {

    MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image is already in real space");

    SetCufftPlan(cistem::fft_type::Enum::inplace_32f_32f_32f, (void*)real_values, (void*)complex_values);

    //  BufferInit(b_image);
    //  cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values));

    _BackwardFFT<float, float2>( );
    // cudaErr(cufftExecC2R(this->cuda_plan_inverse, momentum_space_ptr, position_space_ptr));

    is_in_real_space = true;
    // FIXME: This doesn't really make sense unless we are sure we will copy the image back to the host
    // Instead, we should set the host flag appropriately at that time.
    if ( host_image_ptr )
        host_image_ptr->is_in_real_space = true;

    npp_ROI = npp_ROI_real_space;
}

template <typename LoadType, typename StoreType>
void GpuImage::BackwardFFTAfterComplexConjMul(LoadType* image_to_multiply, bool load_half_precision) {
    MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image is already in real space");

    if constexpr ( std::is_same<StoreType, __half2>::value ) {
        // We always store the output in the fp16 buffer,
        BufferInit(b_16f);
    }
    else {
        // We would do something else here if it was enabled, but it is not.
        // NOTE: I guess
        static_assert(std::is_same<StoreType, __half2>::value, "GpuImage::BackwardFFTAfterComplexConjMul: StoreType must be __half2");
    }

    SetCufftPlan(cistem::fft_type::Enum::inplace_32f_32f_32f, (void*)real_values, (void*)complex_values);

    if ( ! is_set_complexConjMulLoad ) {
        cufftCallbackStoreC                     h_complexConjMulLoad;
        cufftCallbackStoreR                     h_mipCCGStore;
        CB_complexConjMulLoad_params<LoadType>* d_params;
        CB_complexConjMulLoad_params<LoadType>  h_params;
        h_params.scale  = ft_normalization_factor * ft_normalization_factor;
        h_params.target = (LoadType*)image_to_multiply;
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaMallocAsync((void**)&d_params, sizeof(CB_complexConjMulLoad_params<LoadType>), cudaStreamPerThread));
#else
        cudaErr(cudaMalloc((void**)&d_params, sizeof(CB_complexConjMulLoad_params<LoadType>)));
#endif
        cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_complexConjMulLoad_params<LoadType>), cudaMemcpyHostToDevice, cudaStreamPerThread));
        if ( load_half_precision ) {
            cudaErr(cudaMemcpyFromSymbol(&h_complexConjMulLoad, d_complexConjMulLoad_16f, sizeof(h_complexConjMulLoad)));
        }
        else {
            cudaErr(cudaMemcpyFromSymbol(&h_complexConjMulLoad, d_complexConjMulLoad_32f, sizeof(h_complexConjMulLoad)));
        }

        cudaErr(cudaMemcpyFromSymbol(&h_mipCCGStore, d_mipCCGAndStorePtr, sizeof(h_mipCCGStore)));
        //        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        cufftErr(cufftXtSetCallback(cuda_plan_inverse, (void**)&h_complexConjMulLoad, CUFFT_CB_LD_COMPLEX, (void**)&d_params));
        //        void** fake_params;real_values_16f
        cufftErr(cufftXtSetCallback(cuda_plan_inverse, (void**)&h_mipCCGStore, CUFFT_CB_ST_REAL, (void**)&real_values_16f));

        //        d_complexConjMulLoad;
        is_set_complexConjMulLoad = true;
    }

    //  BufferInit(b_image);
    //  cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values));

    _BackwardFFT<float, float2>( );
    // cudaErr(cufftExecC2R(this->cuda_plan_inverse, momentum_space_ptr, position_space_ptr));

    is_in_real_space = true;
    // FIXME: This doesn't really make sense unless we are sure we will copy the image back to the host
    // Instead, we should set the host flag appropriately at that time.
    if ( host_image_ptr )
        host_image_ptr->is_in_real_space = true;
    npp_ROI = npp_ROI_real_space;
}

template void GpuImage::BackwardFFTAfterComplexConjMul<__half2, __half2>(__half2* image_to_multiply, bool load_half_precision);
template void GpuImage::BackwardFFTAfterComplexConjMul<cufftComplex, __half2>(cufftComplex* image_to_multiply, bool load_half_precision);

void GpuImage::Record( ) {
    MyDebugAssertTrue(is_npp_calc_event_initialized, "NPP event not initialized");
    cudaErr(cudaEventRecord(npp_calc_event, cudaStreamPerThread));
}

void GpuImage::RecordBlocking( ) {
    MyDebugAssertTrue(is_block_host_event_initialized, "block host event not initialized");
    cudaErr(cudaEventRecord(block_host_event, cudaStreamPerThread));
}

void GpuImage::Wait( ) {
    MyDebugAssertTrue(is_npp_calc_event_initialized, "NPP event not initialized");
    cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, npp_calc_event, 0));
    //  cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
}

void GpuImage::WaitBlocking( ) {
    MyDebugAssertTrue(is_block_host_event_initialized, "block host event not initialized");
    cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, block_host_event, 0));
}

void GpuImage::RecordAndWait( ) {
    Record( );
    Wait( );
}

template <typename StorageTypeBase>
void GpuImage::SwapRealSpaceQuadrants( ) {
    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {
        MyDebugAssertTrue(is_allocated_16f_buffer, "Gpu memory not allocated");
        MyDebugAssertFalse(is_in_real_space, "Image is already in real space - SwapRealSpaceQuadrants with fp16 must already have FFTs");
    }
    else {
        MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    }

    bool must_fft = false;

    float x_shift_to_apply;
    float y_shift_to_apply;
    float z_shift_to_apply;

    if ( is_in_real_space == true ) {
        must_fft = true;
        ForwardFFT(true);
    }

    if ( object_is_centred_in_box == true ) {
        x_shift_to_apply = float(physical_address_of_box_center.x);
        y_shift_to_apply = float(physical_address_of_box_center.y);
        z_shift_to_apply = float(physical_address_of_box_center.z);
    }
    else {
        if ( IsEven(dims.x) == true ) {
            x_shift_to_apply = float(physical_address_of_box_center.x);
        }
        else {
            x_shift_to_apply = float(physical_address_of_box_center.x) - 1.0;
        }

        if ( IsEven(dims.y) == true ) {
            y_shift_to_apply = float(physical_address_of_box_center.y);
        }
        else {
            y_shift_to_apply = float(physical_address_of_box_center.y) - 1.0;
        }

        if ( IsEven(dims.z) == true ) {
            z_shift_to_apply = float(physical_address_of_box_center.z);
        }
        else {
            z_shift_to_apply = float(physical_address_of_box_center.z) - 1.0;
        }
    }

    if ( dims.z == 1 ) {
        z_shift_to_apply = 0.0;
    }

    PhaseShift<StorageTypeBase>(x_shift_to_apply, y_shift_to_apply, z_shift_to_apply);

    if ( must_fft == true )
        BackwardFFT( );

    // keep track of center;
    if ( object_is_centred_in_box == true )
        object_is_centred_in_box = false;
    else
        object_is_centred_in_box = true;
}

template void GpuImage::SwapRealSpaceQuadrants<__half>( );
template void GpuImage::SwapRealSpaceQuadrants<float>( );

__global__ void
ZeroCentralPixelKernel(float2* complex_values) {
    complex_values[0].x = 0.0f;
    complex_values[0].y = 0.0f;
}

void GpuImage::ZeroCentralPixel( ) {
    MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyDebugAssertFalse(is_in_real_space, "Image must be in Fourier space");

    ZeroCentralPixelKernel<<<1, 1, 0, cudaStreamPerThread>>>(complex_values);
}

template <typename StorageType>
__global__ void
PhaseShiftKernel(StorageType* d_input,
                 int4 dims, float3 shifts,
                 int3 physical_address_of_box_center,
                 int3 physical_index_of_first_negative_frequency,
                 int3 physical_upper_bound_complex) {
    // FIXME it probably makes sense so just just a linear grid launch and save the extra indexing
    int3 wanted_coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y,
                                   blockIdx.z);

    // FIXME This should probably use cuBlas
    if ( wanted_coords.x <= physical_upper_bound_complex.x &&
         wanted_coords.y <= physical_upper_bound_complex.y &&
         wanted_coords.z <= physical_upper_bound_complex.z ) {

        float2 angles;
        d_Return3DPhaseFromIndividualDimensions(d_ReturnPhaseFromShift(
                                                        shifts.x,
                                                        wanted_coords.x,
                                                        dims.x),
                                                d_ReturnPhaseFromShift(
                                                        shifts.y,
                                                        d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(
                                                                wanted_coords.y,
                                                                dims.y,
                                                                physical_index_of_first_negative_frequency.y),
                                                        dims.y),
                                                d_ReturnPhaseFromShift(
                                                        shifts.z,
                                                        d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(
                                                                wanted_coords.z,
                                                                dims.z,
                                                                physical_index_of_first_negative_frequency.z),
                                                        dims.z),
                                                angles);

        int address      = d_ReturnFourier1DAddressFromPhysicalCoord(wanted_coords, physical_upper_bound_complex);
        d_input[address] = ComplexMul<StorageType, float2>(d_input[address], angles);
    }
}

template <typename StorageTypeBase>
void GpuImage::PhaseShift<StorageTypeBase>(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift) {
    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {
        MyDebugAssertTrue(is_allocated_16f_buffer, "Gpu memory not allocated");
        MyDebugAssertFalse(is_in_real_space, "Image is already in real space - PhaseShift with fp16 must already have FFTs");
    }
    else {
        MyDebugAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    }

    bool need_to_fft = false;
    if ( is_in_real_space == true ) {
        wxPrintf("Doing forward fft in phase shift function\n\n");
        ForwardFFT(true);
        need_to_fft = true;
    }

    float3 shifts = make_float3(wanted_x_shift, wanted_y_shift, wanted_z_shift);
    // TODO set the TPB and inline function for grid

    ReturnLaunchParameters(dims, false);

    precheck;
    if constexpr ( std::is_same<StorageTypeBase, __half>::value ) {

        PhaseShiftKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>((__half2*)complex_values_16f,
                                                                                dims, shifts,
                                                                                physical_address_of_box_center,
                                                                                physical_index_of_first_negative_frequency,
                                                                                physical_upper_bound_complex);
    }
    else {
        PhaseShiftKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>((float2*)complex_values,
                                                                                dims, shifts,
                                                                                physical_address_of_box_center,
                                                                                physical_index_of_first_negative_frequency,
                                                                                physical_upper_bound_complex);
    }

    postcheck;

    if ( need_to_fft == true )
        BackwardFFT( );
}

template void GpuImage::PhaseShift<float>(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift);
template void GpuImage::PhaseShift<__half>(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift);

__device__ __forceinline__ float
d_ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size) {
    return real_space_shift * distance_from_origin * 2.0 * PI / dimension_size;
}

__device__ __forceinline__ void
d_Return3DPhaseFromIndividualDimensions(float phase_x, float phase_y, float phase_z, float2& angles) {
    float temp_phase = -phase_x - phase_y - phase_z;
    __sincosf(temp_phase, &angles.y, &angles.x); // To use as cos.x + i*sin.y
}

__device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index,
                                                int logical_x_dimension,
                                                int physical_address_of_box_center_x) {

    //if (physical_index >= physical_index_of_first_negative_frequency_x)
    if ( physical_index > physical_address_of_box_center_x ) {
        return physical_index - logical_x_dimension;
    }
    else
        return physical_index;
}

__device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                                int logical_y_dimension,
                                                int physical_index_of_first_negative_frequency_y) {

    if ( physical_index >= physical_index_of_first_negative_frequency_y ) {
        return physical_index - logical_y_dimension;
    }
    else
        return physical_index;
}

__device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,

                                                int logical_z_dimension,
                                                int physical_index_of_first_negative_frequency_z) {

    if ( physical_index >= physical_index_of_first_negative_frequency_z ) {
        return physical_index - logical_z_dimension;
    }
    else
        return physical_index;
}

__device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, int4 img_dims) {
    return ((((int)coords.z * (int)img_dims.y + coords.y) * (int)img_dims.w) + (int)coords.x);
}

__device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, int pitch_in_pixels, int NY) {
    return ((((int)coords.z * (int)NY + coords.y) * (int)pitch_in_pixels) + (int)coords.x);
}

__device__ __forceinline__ int
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_coords, int3 physical_upper_bound_complex) {
    return ((int)((physical_upper_bound_complex.y + 1) * wanted_coords.z + wanted_coords.y) *
                    (int)(physical_upper_bound_complex.x + 1) +
            (int)wanted_coords.x);
}

__device__ __forceinline__ int
d_ReturnFourier1DAddressFromLogicalCoord(int wanted_x_coord, int wanted_y_coord, int wanted_z_coord, const int3& dims, const int3& physical_upper_bound_complex) {
    if ( wanted_x_coord >= 0 ) {
        if ( wanted_y_coord < 0 ) {
            wanted_y_coord += dims.y;
        }
        if ( wanted_z_coord < 0 ) {
            wanted_z_coord += dims.z;
        }
    }
    else {
        wanted_x_coord = -wanted_x_coord;

        if ( wanted_y_coord > 0 ) {
            wanted_y_coord = dims.y - wanted_y_coord;
        }
        else {
            wanted_y_coord = -wanted_y_coord;
        }

        if ( wanted_z_coord > 0 ) {
            wanted_z_coord = dims.z - wanted_z_coord;
        }
        else {
            wanted_z_coord = -wanted_z_coord;
        }
    }
    return d_ReturnFourier1DAddressFromPhysicalCoord(make_int3(wanted_x_coord, wanted_y_coord, wanted_z_coord), physical_upper_bound_complex);
}

__global__ void
ClipIntoRealKernel(cufftReal* real_values,
                   cufftReal* other_image_real_values,
                   int4       dims,
                   int4       other_dims,
                   int3       physical_address_of_box_center,
                   int3       other_physical_address_of_box_center,
                   int3       wanted_coordinate_of_box_center,
                   float      wanted_padding_value) {
    int3 other_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    int3 coord = make_int3(0, 0, 0);

    if ( other_coord.x < other_dims.x &&
         other_coord.y < other_dims.y &&
         other_coord.z < other_dims.z ) {

        coord.z = physical_address_of_box_center.z + wanted_coordinate_of_box_center.z +
                  other_coord.z - other_physical_address_of_box_center.z;

        coord.y = physical_address_of_box_center.y + wanted_coordinate_of_box_center.y +
                  other_coord.y - other_physical_address_of_box_center.y;

        coord.x = physical_address_of_box_center.x + wanted_coordinate_of_box_center.x +
                  other_coord.x - other_physical_address_of_box_center.x;

        if ( coord.z < 0 || coord.z >= dims.z ||
             coord.y < 0 || coord.y >= dims.y ||
             coord.x < 0 || coord.x >= dims.x ) {
            other_image_real_values[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = wanted_padding_value;
        }
        else {
            other_image_real_values[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] =
                    real_values[d_ReturnReal1DAddressFromPhysicalCoord(coord, dims)];
        }
    }
}

__global__ void
ClipIntoMaskKernel(
        int*  mask_values,
        int4  dims,
        int4  other_dims,
        int3  physical_address_of_box_center,
        int3  other_physical_address_of_box_center,
        int3  wanted_coordinate_of_box_center,
        float wanted_padding_value) {
    int3 other_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    int3 coord = make_int3(0, 0, 0);

    if ( other_coord.x < other_dims.x &&
         other_coord.y < other_dims.y &&
         other_coord.z < other_dims.z ) {

        coord.z = physical_address_of_box_center.z + wanted_coordinate_of_box_center.z +
                  other_coord.z - other_physical_address_of_box_center.z;

        coord.y = physical_address_of_box_center.y + wanted_coordinate_of_box_center.y +
                  other_coord.y - other_physical_address_of_box_center.y;

        coord.x = physical_address_of_box_center.x + wanted_coordinate_of_box_center.x +
                  other_coord.x - other_physical_address_of_box_center.x;

        if ( coord.z < 0 || coord.z >= dims.z ||
             coord.y < 0 || coord.y >= dims.y ||
             coord.x < 0 || coord.x >= dims.x ) {
            // Assumes that the pixel value at pixel 0 should be zero too.
            mask_values[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = (int)0;
        }
        else {
            mask_values[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = d_ReturnReal1DAddressFromPhysicalCoord(coord, dims);
        }
    }
}

// If you don't want to clip from the center, you can give wanted_coordinate_of_box_center_{x,y,z}. This will define the pixel in the image at which other_image will be centered. (0,0,0) means center of image. This is a dumbed down version that does not fill with noise.
void GpuImage::ClipInto(GpuImage* other_image, float wanted_padding_value,
                        bool fill_with_noise, float wanted_noise_sigma,
                        int wanted_coordinate_of_box_center_x,
                        int wanted_coordinate_of_box_center_y,
                        int wanted_coordinate_of_box_center_z) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");

    if ( ! is_in_real_space ) {
        MyDebugAssertTrue(false, "Call ClipIntoFourierSpace");
    };

    int3 wanted_coordinate_of_box_center = make_int3(wanted_coordinate_of_box_center_x,
                                                     wanted_coordinate_of_box_center_y,
                                                     wanted_coordinate_of_box_center_z);

    other_image->is_in_real_space         = is_in_real_space;
    other_image->object_is_centred_in_box = object_is_centred_in_box;
    other_image->is_fft_centered_in_box   = is_fft_centered_in_box;

    if ( is_in_real_space == true ) {

        MyDebugAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

        ReturnLaunchParameters(other_image->dims, true);

        precheck;
        ClipIntoRealKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values,
                                                                                  other_image->real_values,
                                                                                  dims,
                                                                                  other_image->dims,
                                                                                  physical_address_of_box_center,
                                                                                  other_image->physical_address_of_box_center,
                                                                                  wanted_coordinate_of_box_center,
                                                                                  wanted_padding_value);
        postcheck;
    }
}

void GpuImage::ClipIntoReturnMask(GpuImage* other_image) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");

    int3 wanted_coordinate_of_box_center = make_int3(0, 0, 0);

    other_image->is_in_real_space         = is_in_real_space;
    other_image->object_is_centred_in_box = object_is_centred_in_box;
    other_image->is_fft_centered_in_box   = is_fft_centered_in_box;
#ifdef USE_ASYNC_MALLOC_FREE
    cudaErr(cudaMallocAsync(&other_image->clip_into_mask, sizeof(int) * other_image->real_memory_allocated, cudaStreamPerThread));
#else
    cudaErr(cudaMalloc(&other_image->clip_into_mask, sizeof(int) * other_image->real_memory_allocated));
#endif
    other_image->is_allocated_clip_into_mask = true;

    if ( is_in_real_space == true ) {

        MyDebugAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

        ReturnLaunchParameters(other_image->dims, true);

        precheck;
        ClipIntoMaskKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(other_image->clip_into_mask,
                                                                                  dims,
                                                                                  other_image->dims,
                                                                                  physical_address_of_box_center,
                                                                                  other_image->physical_address_of_box_center,
                                                                                  wanted_coordinate_of_box_center,
                                                                                  0.0f);
        postcheck;
    }
}

void GpuImage::QuickAndDirtyWriteSlices(std::string filename, int first_slice, int last_slice) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    Image buffer_img = CopyDeviceToNewHost(true, false, true);

    bool  OverWriteSlices = true;
    float pixelSize       = 0.0f;

    buffer_img.QuickAndDirtyWriteSlices(filename, first_slice, last_slice, OverWriteSlices, pixelSize);
    buffer_img.Deallocate( );
}

void GpuImage::SetCufftPlan(cistem::fft_type::Enum plan_type, void* input_buffer, void* output_buffer) {

    // We need to set the appropriate pointer types for the requested plan.
    // We also want to record the type of plan requested to see if re-planning is necessary.
    using ft = cistem::fft_type::Enum;

    if ( plan_type == set_plan_type && cufft_batch_size == set_batch_size ) {
        // We are good to go.
        return;
    }
    else {
        if ( set_plan_type != cistem::fft_type::Enum::unset ) {
            // TODO allow for more than one plan, up to some limit, to avoid teh destroy op.
            // Have a simple sort to track most recenetly used plans and evict the oldest if needed.
            cufftErr(cufftDestroy(cuda_plan_inverse));
            cufftErr(cufftDestroy(cuda_plan_forward));
            set_plan_type  = cistem::fft_type::Enum::unset;
            set_batch_size = cufft_batch_size;
        }
        // We need to re-plan.
        switch ( plan_type ) {
            case ft::inplace_32f_32f_32f: {
                is_batched_transform = false;
                position_space_ptr   = reinterpret_cast<cufftReal*>(input_buffer);
                momentum_space_ptr   = reinterpret_cast<cufftComplex*>(output_buffer);
                break;
            }
            case ft::inplace_32f_32f_32f_batched: {
                is_batched_transform = true;
                position_space_ptr   = reinterpret_cast<cufftReal*>(input_buffer);
                momentum_space_ptr   = reinterpret_cast<cufftComplex*>(output_buffer);
                break;
            }
            default: {
                std::cerr << "Want to set plan type " << cistem::fft_type::names[plan_type] << "\n";
                MyDebugAssertTrue(false, "Unsupported plan type");
                break;
            }
        }
        set_plan_type = plan_type;
    }

    int            rank;
    long long int* ifftDims;
    long long int* offtDims;
    long long int* inembed; // input storage (not logical) dimensions
    long long int* onembed;
    long long int  iStride = 1;
    long long int  oStride = 1;
    long long int  iDist;
    long long int  oDist;

    cufftErr(cufftCreate(&cuda_plan_forward));
    cufftErr(cufftCreate(&cuda_plan_inverse));

    cufftErr(cufftSetStream(cuda_plan_forward, cudaStreamPerThread));
    cufftErr(cufftSetStream(cuda_plan_inverse, cudaStreamPerThread));

    if ( dims.z > 1 && ! is_batched_transform ) {
        rank     = 3;
        ifftDims = new long long int[rank];
        offtDims = new long long int[rank];
        inembed  = new long long int[rank];
        onembed  = new long long int[rank];

        ifftDims[0] = dims.z;
        ifftDims[1] = dims.y;
        ifftDims[2] = dims.x;

        inembed[0] = dims.z;
        inembed[1] = dims.y;
        inembed[2] = dims.w; // Storage dimension (padded)

        iDist = inembed[0] * inembed[1] * inembed[2];

        offtDims[0] = dims.z;
        offtDims[1] = dims.y;
        offtDims[2] = dims.w / 2;

        onembed[0] = dims.z;
        onembed[1] = dims.y;
        onembed[2] = dims.w / 2; // Storage dimension (padded)

        oDist = onembed[0] * onembed[1] * onembed[2];
    }
    else if ( dims.y > 1 ) {
        rank     = 2;
        ifftDims = new long long int[rank];
        offtDims = new long long int[rank];
        inembed  = new long long int[rank];
        onembed  = new long long int[rank];

        ifftDims[0] = dims.y;
        ifftDims[1] = dims.x;

        inembed[0] = dims.y;
        inembed[1] = dims.w;

        iDist = inembed[0] * inembed[1];

        offtDims[0] = dims.y;
        offtDims[1] = dims.w / 2;

        onembed[0] = dims.y;
        onembed[1] = dims.w / 2;

        oDist = onembed[0] * onembed[1];
    }
    else {
        rank     = 1;
        ifftDims = new long long int[rank];
        offtDims = new long long int[rank];
        inembed  = new long long int[rank];
        onembed  = new long long int[rank];

        ifftDims[0] = dims.x;
        iDist       = dims.w;

        inembed[0] = dims.w;

        offtDims[0] = dims.w / 2;

        onembed[0] = dims.w / 2;

        oDist = dims.w / 2;
    }

    // As far as I can tell, the padded layout must be assumed and onembed/inembed
    // are not needed. TODO ask John about this.

    // if ( use_half_precision ) {
    //     cudaErr(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
    //                                 NULL, NULL, NULL, CUDA_R_16F,
    //                                 NULL, NULL, NULL, CUDA_C_16F, iBatch, &cuda_plan_worksize_forward, CUDA_C_16F));
    //     cudaErr(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
    //                                 NULL, NULL, NULL, CUDA_C_16F,
    //                                 NULL, NULL, NULL, CUDA_R_16F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_16F));
    // }
    // else {

    cufftErr(cufftXtMakePlanMany(cuda_plan_forward, rank, ifftDims,
                                 inembed, iStride, iDist,
                                 CUDA_R_32F,
                                 onembed, oStride, oDist,
                                 CUDA_C_32F,
                                 cufft_batch_size, &cuda_plan_worksize_forward,
                                 CUDA_C_32F));

    cufftErr(cufftXtMakePlanMany(cuda_plan_inverse, rank, ifftDims,
                                 onembed, oStride, oDist,
                                 CUDA_C_32F,
                                 inembed, iStride, iDist,
                                 CUDA_R_32F,
                                 cufft_batch_size, &cuda_plan_worksize_inverse,
                                 CUDA_C_32F));

    delete[] ifftDims;
    delete[] offtDims;
    delete[] inembed;
    delete[] onembed;
}

void GpuImage::Deallocate( ) {
    if ( host_image_ptr != nullptr && host_image_ptr->page_locked_ptr != nullptr ) {
        host_image_ptr->UnRegisterPageLockedMemory( );
    }

    if ( is_in_memory_gpu ) {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaFreeAsync(real_values, cudaStreamPerThread));
#else
        cudaErr(cudaFree(real_values));
#endif
        is_in_memory_gpu = false;
        real_values      = nullptr;
        complex_values   = nullptr;
    }

    if ( is_in_memory_managed_tmp_vals ) {
        // These are allocated as managed memory which isn't part of the async api TODO: confirm true
        cudaErr(cudaFree(tmpVal));
        cudaErr(cudaFree(tmpValComplex));
        is_in_memory_managed_tmp_vals = false;
    }

    if ( is_npp_calc_event_initialized ) {
        cudaErr(cudaEventDestroy(npp_calc_event));
        is_npp_calc_event_initialized = false;
    }

    if ( is_block_host_event_initialized ) {
        cudaErr(cudaEventDestroy(block_host_event));
        is_block_host_event_initialized = false;
    }

    // Separat method for all the buffer memory spaces, not sure it this makes sense
    BufferDestroy( );

    FreeFFTPlan( );

    //  if (is_cublas_loaded)
    //  {
    //    cudaErr(cublasDestroy(cublasHandle));
    //    is_cublas_loaded = false;
    //  }

    if ( is_allocated_texture_cache ) {
        // In the samples, they check directly if( tex_real ) if(cuArray_real)
        cudaErr(cudaDestroyTextureObject(tex_real));
        cudaErr(cudaDestroyTextureObject(tex_imag));
        cudaErr(cudaFreeArray(cuArray_real));
        cudaErr(cudaFreeArray(cuArray_imag));
        is_allocated_texture_cache = false;
    }
}

__global__ void
CopyFP32toFP16bufferKernelReal(cufftReal* real_32f_values, __half* real_16f_values, int4 dims) {
    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.w && coords.y < dims.y && coords.z < dims.z ) {

        int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);

        real_16f_values[address] = __float2half_rn(real_32f_values[address]);
    }
}

__global__ void
CopyFP32toFP16bufferKernelComplex(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex) {
    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z ) {

        int address = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);

        complex_16f_values[address] = __float22half2_rn(complex_32f_values[address]);
    }
}

__global__ void
CopyFP16buffertoFP32KernelReal(cufftReal* real_32f_values, __half* real_16f_values, int4 dims) {
    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.w && coords.y < dims.y && coords.z < dims.z ) {

        int address              = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
        real_32f_values[address] = __half2float(real_16f_values[address]);
    }
}

__global__ void
CopyFP16buffertoFP32KernelComplex(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex) {
    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z ) {

        int address                 = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);
        complex_32f_values[address] = __half22float2(complex_16f_values[address]);
    }
}

void GpuImage::CopyFP32toFP16buffer(bool deallocate_single_precision) {
    // FIXME when adding real space complex images.
    // FIXME should probably be called COPYorConvert
    MyDebugAssertTrue(is_in_memory_gpu, "Image is in not on the GPU!");

    BufferInit(b_16f);

    precheck;
    if ( is_in_real_space ) {
        ReturnLaunchParameters(dims, true);
        CopyFP32toFP16bufferKernelReal<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values, real_values_fp16, this->dims);
    }
    else {
        ReturnLaunchParameters(dims, false);
        CopyFP32toFP16bufferKernelComplex<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values, complex_values_fp16, this->dims, this->physical_upper_bound_complex);
    }
    postcheck;

    if ( deallocate_single_precision ) {
        cudaErr(cudaFreeAsync(real_values, cudaStreamPerThread));
        FreeFFTPlan( );
        is_in_memory_gpu = false;
    }
}

void GpuImage::CopyFP16buffertoFP32(bool deallocate_half_precision) {
    // FIXME when adding real space complex images.
    // FIXME should probably be called COPYorConvert
    MyDebugAssertTrue(is_allocated_16f_buffer, "fp16 buffer is not allocated is in not on the GPU!");
    if ( ! is_in_memory_gpu )
        Allocate(dims.x, dims.y, dims.z, is_in_real_space, false);
    precheck;
    if ( is_in_real_space ) {
        ReturnLaunchParameters(dims, true);
        CopyFP16buffertoFP32KernelReal<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values, real_values_fp16, this->dims);
    }
    else {
        ReturnLaunchParameters(dims, false);
        CopyFP16buffertoFP32KernelComplex<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values, complex_values_fp16, this->dims, this->physical_upper_bound_complex);
    }
    postcheck;

    if ( deallocate_half_precision ) {
        cudaErr(cudaFreeAsync(real_values_16f, cudaStreamPerThread));
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        is_allocated_16f_buffer = false;
        real_values_16f         = nullptr;
        complex_values_16f      = nullptr;
    }
}

void GpuImage::AllocateTmpVarsAndEvents( ) {
    if ( ! is_in_memory_managed_tmp_vals ) {
        cudaErr(cudaMallocManaged(&tmpVal, sizeof(float)));
        cudaErr(cudaMallocManaged(&tmpValComplex, sizeof(double)));
        is_in_memory_managed_tmp_vals = true;
    }
    if ( ! is_npp_calc_event_initialized ) {
        cudaErr(cudaEventCreateWithFlags(&npp_calc_event, cudaEventDisableTiming));
        is_npp_calc_event_initialized = true;
    }
    if ( ! is_block_host_event_initialized ) {
        cudaErr(cudaEventCreateWithFlags(&block_host_event, cudaEventBlockingSync));
        is_block_host_event_initialized = true;
    }
}

bool GpuImage::Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space, bool allocate_fp16_buffer) {
    MyDebugAssertTrue(wanted_x_size > 0 && wanted_y_size > 0 && wanted_z_size > 0, "Bad dimensions: %i %i %i\n", wanted_x_size, wanted_y_size, wanted_z_size);

    // check to see if we need to do anything?
    bool memory_was_allocated = false;
    if ( is_in_memory_gpu == true ) {
        is_in_real_space = should_be_in_real_space;
        if ( wanted_x_size == dims.x && wanted_y_size == dims.y && wanted_z_size == dims.z ) {
            // everything is already done..
            is_in_real_space = should_be_in_real_space;
            // if not allocating fp16 buffer, then we are done.
            if ( ! allocate_fp16_buffer )
                return false;
        }
        else {
            Deallocate( );
        }
    }

    // Update existing values with the wanted values
    this->is_in_real_space = should_be_in_real_space;
    dims.x                 = wanted_x_size;
    dims.y                 = wanted_y_size;
    dims.z                 = wanted_z_size;

    // For compatibility with CPU image methods
    logical_x_dimension = wanted_x_size;
    logical_y_dimension = wanted_y_size;
    logical_z_dimension = wanted_z_size;

    // first_x_dimension
    if ( IsEven(wanted_x_size) == true )
        real_memory_allocated = wanted_x_size / 2 + 1;
    else
        real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

    real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
    real_memory_allocated *= 2; // room for complex

    // TODO consider option to add host mem here. For now, just do gpu mem.
    //////    real_values = (float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
    //////    complex_values = (std::complex<float>*) real_values;  // Set the complex_values to point at the newly allocated real values;
    //    wxPrintf("\n\n\tAllocating mem\t\n\n");
    if ( allocate_fp16_buffer ) {
        BufferInit(b_16f);
    }
    else {
#ifdef USE_ASYNC_MALLOC_FREE
        cudaErr(cudaMallocAsync(&real_values, real_memory_allocated * sizeof(cufftReal), cudaStreamPerThread));
#else
        cudaErr(cudaMalloc(&real_values, real_memory_allocated * sizeof(cufftReal)));
#endif
        // FIXME: with mixed precision this variable is no longer descriptive enough
        is_in_memory_gpu = true;
    }

    memory_was_allocated = true;
    complex_values       = (cufftComplex*)real_values;

    // Update addresses etc..
    UpdateLoopingAndAddressing(wanted_x_size, wanted_y_size, wanted_z_size);

    if ( IsEven(wanted_x_size) == true )
        padding_jump_value = 2;
    else
        padding_jump_value = 1;

    // record the full length ( pitch / 4 )
    dims.w = dims.x + padding_jump_value;
    pitch  = dims.w * sizeof(float);

    number_of_real_space_pixels = int(dims.x) * int(dims.y) * int(dims.z);
    ft_normalization_factor     = 1.0 / sqrtf(float(number_of_real_space_pixels));

    AllocateTmpVarsAndEvents( );

    return memory_was_allocated;
}

void GpuImage::UpdateBoolsToDefault( ) {
    // The main purpose for all of this meta data is to ensure we don't have any issues with memory allocations on the device.
    // This should only be called on a newly created image.
    MyDebugAssertFalse(is_meta_data_initialized, "GpuImage::UpdateBoolsToDefault() Should not be called on a non-initialized image");

    is_meta_data_initialized        = false;
    is_in_memory_managed_tmp_vals   = false;
    is_npp_calc_event_initialized   = false;
    is_block_host_event_initialized = false;

    is_in_memory                           = false;
    is_in_real_space                       = true;
    object_is_centred_in_box               = true;
    is_fft_centered_in_box                 = false;
    image_memory_should_not_be_deallocated = false;

    is_in_memory_gpu = false;

    // libraries
    set_plan_type    = cistem::fft_type::Enum::unset;
    cufft_batch_size = 1;
    //    is_cublas_loaded = false;
    is_npp_loaded = false;

    // Buffers
    is_allocated_image_buffer = false;
    is_allocated_mask_CSOS    = false;

    is_allocated_sum_buffer          = false;
    is_allocated_min_buffer          = false;
    is_allocated_minIDX_buffer       = false;
    is_allocated_max_buffer          = false;
    is_allocated_maxIDX_buffer       = false;
    is_allocated_minmax_buffer       = false;
    is_allocated_minmaxIDX_buffer    = false;
    is_allocated_mean_buffer         = false;
    is_allocated_meanstddev_buffer   = false;
    is_allocated_countinrange_buffer = false;
    is_allocated_l2norm_buffer       = false;
    is_allocated_dotproduct_buffer   = false;
    is_allocated_16f_buffer          = false;
    is_allocated_ctf_16f_buffer      = false;

    // Texture cache for interpolation
    is_allocated_texture_cache = false;

    // Callbacks
    is_set_convertInputf16Tof32 = false;
    is_set_scaleFFTAndStore     = false;
    is_set_complexConjMulLoad   = false;
    is_allocated_clip_into_mask = false;
    is_set_realLoadAndClipInto  = false;
}

//!>  \brief  Update all properties related to looping & addressing in real & Fourier space, given the current logical dimensions.

void GpuImage::UpdateLoopingAndAddressing(int wanted_x_size, int wanted_y_size, int wanted_z_size) {
    dims.x = wanted_x_size;
    dims.y = wanted_y_size;
    dims.z = wanted_z_size;

    physical_address_of_box_center.x = wanted_x_size / 2;
    physical_address_of_box_center.y = wanted_y_size / 2;
    physical_address_of_box_center.z = wanted_z_size / 2;

    physical_upper_bound_complex.x = wanted_x_size / 2;
    physical_upper_bound_complex.y = wanted_y_size - 1;
    physical_upper_bound_complex.z = wanted_z_size - 1;

    //physical_index_of_first_negative_frequency.x= wanted_x_size / 2 + 1;
    if ( IsEven(wanted_y_size) == true ) {
        physical_index_of_first_negative_frequency.y = wanted_y_size / 2;
    }
    else {
        physical_index_of_first_negative_frequency.y = wanted_y_size / 2 + 1;
    }

    if ( IsEven(wanted_z_size) == true ) {
        physical_index_of_first_negative_frequency.z = wanted_z_size / 2;
    }
    else {
        physical_index_of_first_negative_frequency.z = wanted_z_size / 2 + 1;
    }

    // Update the Fourier voxel size

    fourier_voxel_size.x = 1.0 / double(wanted_x_size);
    fourier_voxel_size.y = 1.0 / double(wanted_y_size);
    fourier_voxel_size.z = 1.0 / double(wanted_z_size);

    // Logical bounds
    if ( IsEven(wanted_x_size) == true ) {
        logical_lower_bound_complex.x = -wanted_x_size / 2;
        logical_upper_bound_complex.x = wanted_x_size / 2;
        logical_lower_bound_real.x    = -wanted_x_size / 2;
        logical_upper_bound_real.x    = wanted_x_size / 2 - 1;
    }
    else {
        logical_lower_bound_complex.x = -(wanted_x_size - 1) / 2;
        logical_upper_bound_complex.x = (wanted_x_size - 1) / 2;
        logical_lower_bound_real.x    = -(wanted_x_size - 1) / 2;
        logical_upper_bound_real.x    = (wanted_x_size - 1) / 2;
    }

    if ( IsEven(wanted_y_size) == true ) {
        logical_lower_bound_complex.y = -wanted_y_size / 2;
        logical_upper_bound_complex.y = wanted_y_size / 2 - 1;
        logical_lower_bound_real.y    = -wanted_y_size / 2;
        logical_upper_bound_real.y    = wanted_y_size / 2 - 1;
    }
    else {
        logical_lower_bound_complex.y = -(wanted_y_size - 1) / 2;
        logical_upper_bound_complex.y = (wanted_y_size - 1) / 2;
        logical_lower_bound_real.y    = -(wanted_y_size - 1) / 2;
        logical_upper_bound_real.y    = (wanted_y_size - 1) / 2;
    }

    if ( IsEven(wanted_z_size) == true ) {
        logical_lower_bound_complex.z = -wanted_z_size / 2;
        logical_upper_bound_complex.z = wanted_z_size / 2 - 1;
        logical_lower_bound_real.z    = -wanted_z_size / 2;
        logical_upper_bound_real.z    = wanted_z_size / 2 - 1;
    }
    else {
        logical_lower_bound_complex.z = -(wanted_z_size - 1) / 2;
        logical_upper_bound_complex.z = (wanted_z_size - 1) / 2;
        logical_lower_bound_real.z    = -(wanted_z_size - 1) / 2;
        logical_upper_bound_real.z    = (wanted_z_size - 1) / 2;
    }
}

/* ///////////////////////////////////////
GPU resize from here:
/////////////////////////////////////// */

// __global__ void ClipIntoRealSpaceKernel(cufftReal* real_values,
//                                         cufftReal* other_image_real_values,
//                                         int4       dims,
//                                         int4       other_dims,
//                                         int3       physical_address_of_box_center,
//                                         int3       other_physical_address_of_box_center,
//                                         int3       wanted_coordinate_of_box_center,
//                                         float      wanted_padding_value);

__global__ void
ClipIntoRealSpaceKernel(cufftReal* real_values,
                        cufftReal* other_image_real_values,
                        int4       dims,
                        int4       other_dims,
                        int3       physical_address_of_box_center,
                        int3       other_physical_address_of_box_center,
                        int3       wanted_coordinate_of_box_center,
                        float      wanted_padding_value) {
    int3 other_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    int3 coord = make_int3(0, 0, 0);

    if ( other_coord.x < other_dims.x &&
         other_coord.y < other_dims.y &&
         other_coord.z < other_dims.z ) {

        coord.z = physical_address_of_box_center.z + wanted_coordinate_of_box_center.z +
                  other_coord.z - other_physical_address_of_box_center.z;

        coord.y = physical_address_of_box_center.y + wanted_coordinate_of_box_center.y +
                  other_coord.y - other_physical_address_of_box_center.y;

        coord.x = physical_address_of_box_center.x + wanted_coordinate_of_box_center.x +
                  other_coord.x - other_physical_address_of_box_center.x;

        if ( coord.z < 0 || coord.z >= dims.z ||
             coord.y < 0 || coord.y >= dims.y ||
             coord.x < 0 || coord.x >= dims.x ) {
            other_image_real_values[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = wanted_padding_value;
        }
        else {
            other_image_real_values[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] =
                    real_values[d_ReturnReal1DAddressFromPhysicalCoord(coord, dims)];
        }
    }
}

__global__ void
SetUniformComplexValueKernel(cufftComplex* complex_values, cufftComplex value, int4 dims);

// __global__ void SetUniformComplexValueKernel(cufftComplex* complex_values, cufftComplex value, int4 dims)
//   {
//       int3 kernel_coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
//       blockIdx.y * blockDim.y + threadIdx.y,
//       blockIdx.z);
//       long position1d = get_1d_position_from_3d_coords(kernel_coords, dims);
//       complex_values[position1d] = value;
//   }

__global__ void
ClipIntoFourierSpaceKernel(cufftComplex* source_complex_values,
                           cufftComplex* destination_complex_values,
                           int4          source_dims,
                           int4          destination_dims,
                           int3          destination_image_physical_index_of_first_negative_frequency,
                           int3          source_logical_lower_bound_complex,
                           int3          source_logical_upper_bound_complex,
                           int3          source_physical_upper_bound_complex,
                           int3          destination_physical_upper_bound_complex,
                           float2        out_of_bounds_value,
                           bool          zero_central_pixel) {
    int3 index_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    if ( index_coord.y >= destination_dims.y ||
         index_coord.z >= destination_dims.z ||
         index_coord.x >= destination_dims.w / 2 ) {
        return;
    }

    int destination_index = d_ReturnFourier1DAddressFromPhysicalCoord(index_coord, destination_physical_upper_bound_complex);

    int3 source_coord = index_coord;

    source_coord.y = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(source_coord.y, destination_dims.y, destination_image_physical_index_of_first_negative_frequency.y);
    source_coord.z = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(source_coord.z, destination_dims.z, destination_image_physical_index_of_first_negative_frequency.z);

    if ( source_coord.y >= destination_image_physical_index_of_first_negative_frequency.y )
        source_coord.y -= destination_dims.y;
    if ( source_coord.z >= destination_image_physical_index_of_first_negative_frequency.z )
        source_coord.z -= destination_dims.z;

    float2 new_value;
    if ( source_coord.x < source_logical_lower_bound_complex.x ||
         source_coord.x > source_logical_upper_bound_complex.x ||
         source_coord.y < source_logical_lower_bound_complex.y ||
         source_coord.y > source_logical_upper_bound_complex.y ||
         source_coord.z < source_logical_lower_bound_complex.z ||
         source_coord.z > source_logical_upper_bound_complex.z ) // these can only be true if the destination image has a dimension bigger than the source
    // consider creating a second kenel for just smaller images clipinto...
    {

        new_value = out_of_bounds_value;
    }
    else {
        int3 physical_address;
        if ( source_coord.x >= 0 ) {
            physical_address.x = source_coord.x;

            if ( source_coord.y >= 0 ) {
                physical_address.y = source_coord.y;
            }
            else {
                physical_address.y = source_dims.y + source_coord.y;
            }

            if ( source_coord.z >= 0 ) {
                physical_address.z = source_coord.z;
            }
            else {
                physical_address.z = source_dims.z + source_coord.z;
            }
        }
        else {
            physical_address.x = -source_coord.x;

            if ( source_coord.y > 0 ) {
                physical_address.y = source_dims.y - source_coord.y;
            }
            else {
                physical_address.y = -source_coord.y;
            }

            if ( source_coord.z > 0 ) {
                physical_address.z = source_dims.z - source_coord.z;
            }
            else {
                physical_address.z = -source_coord.z;
            }
        }

        int source_index = d_ReturnFourier1DAddressFromPhysicalCoord(physical_address, source_physical_upper_bound_complex);
        new_value        = source_complex_values[source_index];
    }

    if ( destination_index == 0 && zero_central_pixel ) {
        destination_complex_values[destination_index].x = 0.0f;
        destination_complex_values[destination_index].y = 0.0f;
    }
    else
        destination_complex_values[destination_index] = new_value;
}

__global__ void
ClipIntoFourierSpaceKernel(__half2* source_complex_values,
                           __half2* destination_complex_values,
                           int4     source_dims,
                           int4     destination_dims,
                           int3     destination_image_physical_index_of_first_negative_frequency,
                           int3     source_logical_lower_bound_complex,
                           int3     source_logical_upper_bound_complex,
                           int3     source_physical_upper_bound_complex,
                           int3     destination_physical_upper_bound_complex,
                           __half2  out_of_bounds_value,
                           bool     zero_central_pixel) {
    int3 index_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    if ( index_coord.y >= destination_dims.y ||
         index_coord.z >= destination_dims.z ||
         index_coord.x >= destination_dims.w / 2 ) {
        return;
    }

    int destination_index = d_ReturnFourier1DAddressFromPhysicalCoord(index_coord, destination_physical_upper_bound_complex);

    int3 source_coord = index_coord;

    source_coord.y = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(source_coord.y, destination_dims.y, destination_image_physical_index_of_first_negative_frequency.y);
    source_coord.z = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(source_coord.z, destination_dims.z, destination_image_physical_index_of_first_negative_frequency.z);

    if ( source_coord.y >= destination_image_physical_index_of_first_negative_frequency.y )
        source_coord.y -= destination_dims.y;
    if ( source_coord.z >= destination_image_physical_index_of_first_negative_frequency.z )
        source_coord.z -= destination_dims.z;

    __half2 new_value;
    if ( source_coord.x < source_logical_lower_bound_complex.x ||
         source_coord.x > source_logical_upper_bound_complex.x ||
         source_coord.y < source_logical_lower_bound_complex.y ||
         source_coord.y > source_logical_upper_bound_complex.y ||
         source_coord.z < source_logical_lower_bound_complex.z ||
         source_coord.z > source_logical_upper_bound_complex.z ) // these can only be true if the destination image has a dimension bigger than the source
    // consider creating a second kenel for just smaller images clipinto...
    {

        new_value = out_of_bounds_value;
    }
    else {
        int3 physical_address;
        if ( source_coord.x >= 0 ) {
            physical_address.x = source_coord.x;

            if ( source_coord.y >= 0 ) {
                physical_address.y = source_coord.y;
            }
            else {
                physical_address.y = source_dims.y + source_coord.y;
            }

            if ( source_coord.z >= 0 ) {
                physical_address.z = source_coord.z;
            }
            else {
                physical_address.z = source_dims.z + source_coord.z;
            }
        }
        else {
            physical_address.x = -source_coord.x;

            if ( source_coord.y > 0 ) {
                physical_address.y = source_dims.y - source_coord.y;
            }
            else {
                physical_address.y = -source_coord.y;
            }

            if ( source_coord.z > 0 ) {
                physical_address.z = source_dims.z - source_coord.z;
            }
            else {
                physical_address.z = -source_coord.z;
            }
        }

        int source_index = d_ReturnFourier1DAddressFromPhysicalCoord(physical_address, source_physical_upper_bound_complex);
        new_value        = source_complex_values[source_index];
    }

    if ( destination_index == 0 && zero_central_pixel ) {
        destination_complex_values[destination_index].x = 0.0f;
        destination_complex_values[destination_index].y = 0.0f;
    }
    else
        destination_complex_values[destination_index] = new_value;
}

void GpuImage::Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value, bool zero_central_pixel) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(wanted_x_dimension != 0 && wanted_y_dimension != 0 && wanted_z_dimension != 0, "Resize dimension is zero");
    MyDebugAssertFalse(zero_central_pixel && is_in_real_space, "Zero central pixel only works in Fourier space");

    if ( dims.x == wanted_x_dimension && dims.y == wanted_y_dimension && dims.z == wanted_z_dimension ) {
        wxPrintf("Wanted dimensions are the same as current dimensions.\n");
        return;
    }

    GpuImage temp_image;

    temp_image.Allocate(wanted_x_dimension, wanted_y_dimension, wanted_z_dimension, is_in_real_space);

    if ( is_in_real_space ) {
        ClipInto(&temp_image, wanted_padding_value, false, 1.0, 0, 0, 0);
    }
    else {
        ClipIntoFourierSpace(&temp_image, wanted_padding_value, zero_central_pixel);
    }

    // wxPrintf("Consuming temp image\n");
    Consume(&temp_image);
}

void GpuImage::CopyCpuImageMetaData(Image& cpu_image) {
    host_image_ptr = &cpu_image;

    dims = make_int4(cpu_image.logical_x_dimension,
                     cpu_image.logical_y_dimension,
                     cpu_image.logical_z_dimension,
                     cpu_image.logical_x_dimension + cpu_image.padding_jump_value);

    logical_x_dimension = cpu_image.logical_x_dimension;
    logical_y_dimension = cpu_image.logical_y_dimension;
    logical_z_dimension = cpu_image.logical_z_dimension;

    pitch = dims.w * sizeof(float);

    physical_upper_bound_complex = make_int3(cpu_image.physical_upper_bound_complex_x,
                                             cpu_image.physical_upper_bound_complex_y,
                                             cpu_image.physical_upper_bound_complex_z);

    physical_address_of_box_center = make_int3(cpu_image.physical_address_of_box_center_x,
                                               cpu_image.physical_address_of_box_center_y,
                                               cpu_image.physical_address_of_box_center_z);

    physical_index_of_first_negative_frequency = make_int3(0,
                                                           cpu_image.physical_index_of_first_negative_frequency_y,
                                                           cpu_image.physical_index_of_first_negative_frequency_z);

    logical_upper_bound_complex = make_int3(cpu_image.logical_upper_bound_complex_x,
                                            cpu_image.logical_upper_bound_complex_y,
                                            cpu_image.logical_upper_bound_complex_z);

    logical_lower_bound_complex = make_int3(cpu_image.logical_lower_bound_complex_x,
                                            cpu_image.logical_lower_bound_complex_y,
                                            cpu_image.logical_lower_bound_complex_z);

    logical_upper_bound_real = make_int3(cpu_image.logical_upper_bound_real_x,
                                         cpu_image.logical_upper_bound_real_y,
                                         cpu_image.logical_upper_bound_real_z);

    logical_lower_bound_real = make_int3(cpu_image.logical_lower_bound_real_x,
                                         cpu_image.logical_lower_bound_real_y,
                                         cpu_image.logical_lower_bound_real_z);

    is_in_real_space            = cpu_image.is_in_real_space;
    number_of_real_space_pixels = cpu_image.number_of_real_space_pixels;
    object_is_centred_in_box    = cpu_image.object_is_centred_in_box;
    is_fft_centered_in_box      = cpu_image.is_fft_centered_in_box;

    fourier_voxel_size = make_float3(cpu_image.fourier_voxel_size_x,
                                     cpu_image.fourier_voxel_size_y,
                                     cpu_image.fourier_voxel_size_z);

    insert_into_which_reconstruction = cpu_image.insert_into_which_reconstruction;

    is_in_memory = cpu_image.is_in_memory;

    padding_jump_value                     = cpu_image.padding_jump_value;
    image_memory_should_not_be_deallocated = cpu_image.image_memory_should_not_be_deallocated; // TODO what is this for?

    // real_values       = NULL; // !<  Real array to hold values for REAL images.
    // complex_values    = NULL; // !<  Complex array to hold values for COMP images.
    // is_in_memory_gpu      = false;
    real_memory_allocated = cpu_image.real_memory_allocated;

    ft_normalization_factor = cpu_image.ft_normalization_factor;

    is_meta_data_initialized = true;
}

void GpuImage::CopyGpuImageMetaData(const GpuImage* other_image) {
    is_in_real_space = other_image->is_in_real_space;

    number_of_real_space_pixels = other_image->number_of_real_space_pixels;

    insert_into_which_reconstruction = other_image->insert_into_which_reconstruction;

    object_is_centred_in_box = other_image->object_is_centred_in_box;
    is_fft_centered_in_box   = other_image->is_fft_centered_in_box;

    dims = other_image->dims;

    // FIXME: temp for comp
    logical_x_dimension = other_image->dims.x;
    logical_y_dimension = other_image->dims.y;
    logical_z_dimension = other_image->dims.z;

    pitch = other_image->pitch;

    physical_upper_bound_complex = other_image->physical_upper_bound_complex;

    physical_address_of_box_center = other_image->physical_address_of_box_center;

    physical_index_of_first_negative_frequency = other_image->physical_index_of_first_negative_frequency;

    fourier_voxel_size = other_image->fourier_voxel_size;

    logical_upper_bound_complex = other_image->logical_upper_bound_complex;

    logical_lower_bound_complex = other_image->logical_lower_bound_complex;

    logical_upper_bound_real = other_image->logical_upper_bound_real;

    logical_lower_bound_real = other_image->logical_lower_bound_real;

    padding_jump_value = other_image->padding_jump_value;

    ft_normalization_factor = other_image->ft_normalization_factor;

    real_memory_allocated = other_image->real_memory_allocated;

    is_in_memory             = other_image->is_in_memory;
    is_meta_data_initialized = other_image->is_meta_data_initialized;
}

/*
 Overwrite current GpuImage with a new image, then deallocate new image.
*/
// copy the parameters then directly steal the memory of another image, leaving it an empty shell
void GpuImage::Consume(GpuImage* other_image) {
    MyDebugAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");

    if ( this == other_image ) { // no need to consume, its the same image.
        return;
    }

    Deallocate( );
    // Normally, we don't want to overwrite existing metadata, but in this case we do, so set this flag to override runtime checks resulting from SetupINitialValues - > UpdateBoolsToDefault
    is_meta_data_initialized = false;
    SetupInitialValues( );
    CopyGpuImageMetaData(other_image);

    real_values      = other_image->real_values;
    complex_values   = other_image->complex_values;
    is_in_memory_gpu = other_image->is_in_memory_gpu;

    cuda_plan_forward = other_image->cuda_plan_forward;
    cuda_plan_inverse = other_image->cuda_plan_inverse;
    set_plan_type     = other_image->set_plan_type;
    cufft_batch_size  = other_image->cufft_batch_size;

    // We neeed to override the other image pointers so that it doesn't deallocate the memory.
    other_image->real_values      = NULL;
    other_image->complex_values   = NULL;
    other_image->is_in_memory_gpu = false;

    other_image->cuda_plan_forward = NULL;
    other_image->cuda_plan_inverse = NULL;
    other_image->set_plan_type     = cistem::fft_type::Enum::unset;

    return;
}

void GpuImage::ClipIntoFourierSpace(GpuImage* destination_image, float wanted_padding_value, bool zero_central_pixel, bool use_fp16) {
    MyDebugAssertTrue(is_in_memory_gpu || (use_fp16 && is_allocated_16f_buffer), "Memory not allocated");
    MyDebugAssertTrue(destination_image->is_in_memory_gpu || (use_fp16 && destination_image->is_allocated_16f_buffer), "Destination image memory not allocated");
    MyDebugAssertTrue(destination_image->object_is_centred_in_box && object_is_centred_in_box, "ClipInto assumes both images are centered at the moment.");
    MyDebugAssertFalse(is_in_real_space && destination_image->is_in_real_space, "ClipIntoFourierSpace assumes both images are in fourier space");

    destination_image->object_is_centred_in_box = object_is_centred_in_box;
    destination_image->is_fft_centered_in_box   = is_fft_centered_in_box;

    ReturnLaunchParameters(destination_image->dims, false);

    // TODO could easily template this, but plan to replace with a straight memcpy anyway
    if ( use_fp16 ) {
        __half2 padding_value = __float2half2_rn(wanted_padding_value);
        precheck;
        ClipIntoFourierSpaceKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_fp16,
                                                                                          destination_image->complex_values_fp16,
                                                                                          dims,
                                                                                          destination_image->dims,
                                                                                          destination_image->physical_index_of_first_negative_frequency,
                                                                                          logical_lower_bound_complex,
                                                                                          logical_upper_bound_complex,
                                                                                          physical_upper_bound_complex,
                                                                                          destination_image->physical_upper_bound_complex,
                                                                                          padding_value,
                                                                                          zero_central_pixel);

        postcheck;
    }
    else {

        float2 padding_value = {wanted_padding_value, wanted_padding_value};
        precheck;
        ClipIntoFourierSpaceKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values,
                                                                                          destination_image->complex_values,
                                                                                          dims,
                                                                                          destination_image->dims,
                                                                                          destination_image->physical_index_of_first_negative_frequency,
                                                                                          logical_lower_bound_complex,
                                                                                          logical_upper_bound_complex,
                                                                                          physical_upper_bound_complex,
                                                                                          destination_image->physical_upper_bound_complex,
                                                                                          padding_value,
                                                                                          zero_central_pixel);

        postcheck;
    }
    cudaStreamSynchronize(cudaStreamPerThread);
}

__global__ void
ExtractSliceKernel(const cudaTextureObject_t tex_real,
                   const cudaTextureObject_t tex_imag,
                   float2*                   outputData,
                   const int                 NX,
                   const int                 NY,
                   const float3              col1,
                   const float3              col2,
                   const float               resolution_limit,
                   const bool                apply_resolution_limit) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= NX ) {
        return;
    }
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y >= NY ) {
        return;
    }

    float u, v, tu, tv, tw;

    // First, convert the physical coordinate of the 2d projection to the logical Fourier coordinate (in a natural FFT layout).
    u = (float)x;
    // First negative logical fourier component is at NY/2
    if ( y >= NY / 2 ) {
        v = (float)y - NY;
    }
    else {
        v = (float)y;
    }
    // logical Fourier z = 0 for a projection

    if ( (apply_resolution_limit && float(v * v + u * u) > resolution_limit * resolution_limit) || (x == 0 && y == 0) ) {
        outputData[y * NX + x] = make_float2(0.f, 0.f);
    }
    else {
        // Based on RotationMatrix.RotateCoords()
        // Based on RotationMatrix.RotateCoords()
        tu = u * col1.x + v * col2.x;
        tv = u * col1.y + v * col2.y;
        tw = u * col1.z + v * col2.z;

        if ( tu < 0 ) {
            // We have only the positive X half of the FFT, re-use variable u here to return the complex conjugate
            u  = -1.f;
            tu = -tu;
            tv = -tv;
            tw = -tw;
        }
        else
            u = 1.f;

        // Now convert the logical Fourier coordinate to the Swapped Fourier *physical* coordinate
        // The logical origin is physically at X = 1, Y = Z = NY/2
        // Also: include the 1/2 pixel offset to account for different conventions between cuda and cisTEM
        tu += 1.5f;
        tv += (float(NY / 2) + 0.5f);
        tw += (float(NY / 2) + 0.5f);

        outputData[y * NX + x] = make_float2(tex3D<float>(tex_real, tu, tv, tw), u * tex3D<float>(tex_imag, tu, tv, tw));
    }
}

__global__ void
ExtractSliceAndWhitenKernel(const cudaTextureObject_t tex_real,
                            const cudaTextureObject_t tex_imag,
                            float2*                   outputData,
                            float2                    shifts,
                            const int                 NX,
                            const int                 NY,
                            const float3              col1,
                            const float3              col2,
                            const float               resolution_limit,
                            const bool                apply_resolution_limit,
                            const int                 n_bins,
                            const int                 n_bins2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= NX ) {
        return;
    }
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y >= NY ) {
        return;
    }

    // Be carefule!! There are a lot of integer/float conversions in this function. If you change anything, make sure it is still correct.
    extern __shared__ int non_zero_count[];
    float*                radial_average = (float*)&non_zero_count[n_bins];

    // initialize temporary accumulation array in shared memory
    //FIXME
    if ( threadIdx.x == 0 ) {
        for ( int i = threadIdx.y; i < n_bins; i += blockDim.y ) {
            radial_average[i] = 0.f;
            non_zero_count[i] = 0;
        }
    }
    __syncthreads( );

    float u, v, tu, tv, tw, frequency_sq;

    // First, convert the physical coordinate of the 2d projection to the logical Fourier coordinate (in a natural FFT layout).
    u = float(x);
    // First negative logical fourier component is at NY/2
    if ( y >= NY / 2 ) {
        v = float(y) - NY;
    }
    else {
        v = float(y);
    }

    // Shifts are already 2*pi*dx/NX
    shifts.x *= u;
    shifts.y *= v;
    // logical Fourier z = 0 for a projection

    frequency_sq = u * u + v * v;

    if ( (apply_resolution_limit && frequency_sq > resolution_limit * resolution_limit) || (x == 0 && y == 0) ) {
        outputData[y * NX + x] = make_float2(0.f, 0.f);
    }
    else {

        // Based on RotationMatrix.RotateCoords()
        // Based on RotationMatrix.RotateCoords()
        tu = u * col1.x + v * col2.x;
        tv = u * col1.y + v * col2.y;
        tw = u * col1.z + v * col2.z;

        if ( tu < 0 ) {
            // We have only the positive X half of the FFT, re-use variable u here to return the complex conjugate
            u  = -1.f;
            tu = -tu;
            tv = -tv;
            tw = -tw;
        }
        else
            u = 1.f;

        // Now convert the logical Fourier coordinate to the Swapped Fourier *physical* coordinate
        // The logical origin is physically at X = 1, Y = Z = NY/2
        // Also: include the 1/2 pixel offset to account for different conventions between cuda and cisTEM
        tu += 1.5f;
        tv += (float(NY / 2) + 0.5f);
        tw += (float(NY / 2) + 0.5f);

        // reuse u and v to grab results
        v = u * tex3D<float>(tex_imag, tu, tv, tw);
        u = tex3D<float>(tex_real, tu, tv, tw);

        // Resuse y to get the address and then x as the bin number
        y = y * NX + x;

        // Get the norm squared for the pixel
        tw = u * u + v * v;
        // Again, assuming NX = NY in real space
        x = int(sqrtf(frequency_sq) / float(NY) * float(n_bins2));
        // This check is from Image::Whiten, but should it really be checking a float like this?
        if ( tw != 0.0 ) {
            if ( x <= resolution_limit ) {
                atomicAdd(&non_zero_count[x], 1);
                atomicAdd(&radial_average[x], tw);
            }
        }
        __syncthreads( );

        if ( x <= resolution_limit ) {
            if ( non_zero_count[x] == 0 ) {
                outputData[y] = make_float2(0.f, 0.f);
            }
            else {
                // Note that this scaling factor is inverted from the CPU code so the second step is multiplication
                // tw = sqrtf(float(non_zero_count[x]) / radial_average[x]);
                tw = rsqrtf(radial_average[x] / non_zero_count[x]);
                // outputData[y] = make_float2(u * tw, v * tw);
                __sincosf(-shifts.x - shifts.y, &tv, &tu);
                outputData[y] = ComplexMulAndScale((Complex)make_float2(tu, tv), (Complex)make_float2(u, v), tw);
                // outputData[y] = make_float2(u / tw, v / tw);
            }
        }
        else {
            outputData[y] = make_float2(0.f, 0.f);
        }
    }
}

void GpuImage::ExtractSlice(GpuImage* volume_to_extract_from, AnglesAndShifts& angles_and_shifts, float pixel_size, float resolution_limit, bool apply_resolution_limit, bool whiten_spectrum) {
    //    MyDebugAssertTrue(image_to_extract.logical_x_dimension == logical_x_dimension && image_to_extract.logical_y_dimension == logical_y_dimension, "Error: Images different sizes");
    MyDebugAssertTrue(dims.z == 1, "Error: attempting to project 3d to 3d");
    MyDebugAssertTrue(volume_to_extract_from->dims.z > 1, "Error: attempting to project 2d to 2d");
    MyDebugAssertTrue(is_in_memory_gpu, "Error: gpu memory not allocated");
    MyDebugAssertTrue(volume_to_extract_from->is_allocated_texture_cache, "3d volume not allocated in the texture cache");
    // MyDebugAssertTrue(IsCubic( ), "Image volume to project is not cubic"); // This is checked on call to CopyHostToDeviceTextureComplex3d
    MyDebugAssertFalse(volume_to_extract_from->object_is_centred_in_box, "Image volume quadrants not swapped");
    MyDebugAssertTrue(volume_to_extract_from->is_fft_centered_in_box, "Image volume Fourier quadrants not swapped as required for texture locality");

    // Get launch params for a complex non-redundant half image
    ReturnLaunchParameters(dims, false);

    /*
    Since we only rotate 2d coords, we only need 6 floats from the rotation matrix. Reduce register pressure.
        tu = u * m[0] + v * m[1]; 
        tv = u * m[3] + v * m[4]; 
        tw = u * m[6] + v * m[7];
    */
    float*       m    = &angles_and_shifts.euler_matrix.m[0][0];
    const float3 col1 = make_float3(m[0], m[3], m[6]);
    const float3 col2 = make_float3(m[1], m[4], m[7]);

    float resolution_limit_pixel = resolution_limit * dims.x;

    if ( whiten_spectrum && ! apply_resolution_limit ) {
        // WE need an over-ride to reproduce the default behavior of Image;:Whiten() when no resolution limit is given
        resolution_limit_pixel = 1.f * dims.x;
    }

    if ( whiten_spectrum ) {
        // assuming square images, otherwise this would be largest dimension
        int number_of_bins = dims.y / 2 + 1;
        // Extend table to include corners in 3D Fourier space
        int n_bins  = int(number_of_bins * sqrtf(3.0)) + 1;
        int n_bins2 = 2 * (number_of_bins - 1);
        // For bin resolution of one pixel, uint16 should be plenty
        int shared_mem = n_bins * (sizeof(float) + sizeof(int));
        // TODO: add check on shared mem from FastFFT
        float2 shifts = make_float2(angles_and_shifts.ReturnShiftX( ), angles_and_shifts.ReturnShiftY( ));
        shifts.x      = shifts.x * pi_v<float> * 2.0f / float(dims.x) / pixel_size;
        shifts.y      = shifts.y * pi_v<float> * 2.0f / float(dims.y) / pixel_size;

        // Image::Whiten() defaults to a res limit of 1.0, so we need to match that in the event we opt to not apply a res li mit

        precheck;
        ExtractSliceAndWhitenKernel<<<gridDims, threadsPerBlock, shared_mem, cudaStreamPerThread>>>(volume_to_extract_from->tex_real,
                                                                                                    volume_to_extract_from->tex_imag,
                                                                                                    (float2*)complex_values,
                                                                                                    shifts,
                                                                                                    dims.w / 2,
                                                                                                    dims.y,
                                                                                                    col1,
                                                                                                    col2,
                                                                                                    resolution_limit_pixel,
                                                                                                    apply_resolution_limit,
                                                                                                    n_bins,
                                                                                                    n_bins2);

        postcheck;
    }
    else {
        precheck;
        ExtractSliceKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(volume_to_extract_from->tex_real,
                                                                                  volume_to_extract_from->tex_imag,
                                                                                  (float2*)complex_values,
                                                                                  dims.w / 2,
                                                                                  dims.y,
                                                                                  col1,
                                                                                  col2,
                                                                                  resolution_limit_pixel,
                                                                                  apply_resolution_limit);

        postcheck;
    }

    object_is_centred_in_box = false;
    is_fft_centered_in_box   = false;
    is_in_real_space         = false;
}

__global__ void
ExtractSliceShiftAndCtfKernel(const cudaTextureObject_t tex_real,
                              const cudaTextureObject_t tex_imag,
                              float2*                   outputData,
                              const __half2* __restrict__ ctf_image,
                              float2       shifts,
                              const int    NX,
                              const int    NY,
                              const float3 col1,
                              const float3 col2,
                              const float  resolution_limit,
                              const bool   apply_resolution_limit,
                              const bool   apply_ctf,
                              const bool   abs_ctf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= NX ) {
        return;
    }
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( y >= NY ) {
        return;
    }

    float u, v, tu, tv, tw, frequency_sq;

    // First, convert the physical coordinate of the 2d projection to the logical Fourier coordinate (in a natural FFT layout).
    u = float(x);
    // First negative logical fourier component is at NY/2
    if ( y >= NY / 2 ) {
        v = float(y) - NY;
    }
    else {
        v = float(y);
    }

    // Shifts are already 2*pi*dx/NX
    shifts.x *= u;
    shifts.y *= v;
    // logical Fourier z = 0 for a projection

    frequency_sq = u * u + v * v;

    if ( (apply_resolution_limit && frequency_sq > resolution_limit * resolution_limit) || (x == 0 && y == 0) ) {
        outputData[y * NX + x] = make_float2(0.f, 0.f);
    }
    else {

        // Based on RotationMatrix.RotateCoords()
        tu = u * col1.x + v * col2.x;
        tv = u * col1.y + v * col2.y;
        tw = u * col1.z + v * col2.z;

        if ( tu < 0 ) {
            // We have only the positive X half of the FFT, re-use variable u here to return the complex conjugate
            u  = -1.f;
            tu = -tu;
            tv = -tv;
            tw = -tw;
        }
        else
            u = 1.f;

        // Now convert the logical Fourier coordinate to the Swapped Fourier *physical* coordinate
        // The logical origin is physically at X = 1, Y = Z = NY/2
        // Also: include the 1/2 pixel offset to account for different conventions between cuda and cisTEM
        tu += 1.5f;
        tv += (float(NY / 2) + 0.5f);
        tw += (float(NY / 2) + 0.5f);

        // reuse u and v to grab results
        v = u * tex3D<float>(tex_imag, tu, tv, tw);
        u = tex3D<float>(tex_real, tu, tv, tw);

        // Resuse y to get the address and then x as the bin number
        y = y * NX + x;

        // outputData[y] = make_float2(u * tw, v * tw);
        __sincosf(-shifts.x - shifts.y, &tv, &tu);

        // reuse tw for our CTF value (assuming it is = RE + i*0)
        if ( apply_ctf && abs_ctf ) {
            __half2 tmp_half2 = ctf_image[y];
            tmp_half2         = __hmul2(tmp_half2, tmp_half2);
            tw                = __low2float(tmp_half2);
            tw += __high2float(tmp_half2);
            tw            = sqrtf(tw);
            outputData[y] = ComplexMulAndScale((Complex)make_float2(tu, tv), (Complex)make_float2(u, v), tw);
        }
        else {
            if ( apply_ctf ) {
                outputData[y] = ComplexMul((Complex)__half22float2(ctf_image[y]),
                                           ComplexMul((Complex)make_float2(tu, tv), (Complex)make_float2(u, v)));
            }
            else {
                outputData[y] = ComplexMul((Complex)make_float2(tu, tv), (Complex)make_float2(u, v));
            }
        }

        // outputData[y] = make_float2(u / tw, v / tw);
    }
}

void GpuImage::ExtractSliceShiftAndCtf(GpuImage* volume_to_extract_from, GpuImage* ctf_image, AnglesAndShifts& angles_and_shifts, float pixel_size, float resolution_limit, bool apply_resolution_limit,
                                       bool swap_quadrants, bool apply_shifts, bool apply_ctf, bool absolute_ctf) {
    MyDebugAssertTrue(dims.z == 1, "Error: attempting to project 3d to 3d");
    MyDebugAssertTrue(volume_to_extract_from->dims.z > 1, "Error: attempting to project 2d to 2d");
    MyDebugAssertTrue(is_in_memory_gpu, "Error: gpu memory not allocated");
    MyDebugAssertTrue(volume_to_extract_from->is_allocated_texture_cache, "3d volume not allocated in the texture cache");
    // MyDebugAssertTrue(IsCubic( ), "Image volume to project is not cubic"); // This is checked on call to CopyHostToDeviceTextureComplex3d
    MyDebugAssertFalse(volume_to_extract_from->object_is_centred_in_box, "Image volume quadrants not swapped");
    MyDebugAssertTrue(volume_to_extract_from->is_fft_centered_in_box, "Image volume Fourier quadrants not swapped as required for texture locality");
    if ( apply_ctf ) {
        MyDebugAssertTrue(ctf_image->is_allocated_ctf_16f_buffer, "Error: ctf fp16 gpu memory not allocated");
    }

    // Get launch params for a complex non-redundant half image
    ReturnLaunchParameters(dims, false);

    /*
    Since we only rotate 2d coords, we only need 6 floats from the rotation matrix. Reduce register pressure.
        tu = u * m[0] + v * m[1]; 
        tv = u * m[3] + v * m[4]; 
        tw = u * m[6] + v * m[7];
    */
    float*       m    = &angles_and_shifts.euler_matrix.m[0][0];
    const float3 col1 = make_float3(m[0], m[3], m[6]);
    const float3 col2 = make_float3(m[1], m[4], m[7]);

    float resolution_limit_pixel = resolution_limit * dims.x;

    float2 shifts = make_float2(angles_and_shifts.ReturnShiftX( ), angles_and_shifts.ReturnShiftY( ));
    if ( ! apply_shifts ) {
        shifts.x = 0.f;
        shifts.y = 0.f;
    }
    if ( swap_quadrants ) {
        // We apply the real space quadrant swap (if any) at the same time as any wanted shifts.
        // Again, we are assuming an even sized image
        shifts.x += float(physical_address_of_box_center.x);
        shifts.y += float(physical_address_of_box_center.y);
    }
    shifts.x = shifts.x * pi_v<float> * 2.0f / float(dims.x) / pixel_size;
    shifts.y = shifts.y * pi_v<float> * 2.0f / float(dims.y) / pixel_size;

    // Image::Whiten() defaults to a res limit of 1.0, so we need to match that in the event we opt to not apply a res li mit

    precheck;
    ExtractSliceShiftAndCtfKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(volume_to_extract_from->tex_real,
                                                                                         volume_to_extract_from->tex_imag,
                                                                                         (float2*)complex_values,
                                                                                         (__half2*)ctf_image->ctf_complex_buffer_16f,
                                                                                         shifts,
                                                                                         dims.w / 2,
                                                                                         dims.y,
                                                                                         col1,
                                                                                         col2,
                                                                                         resolution_limit_pixel,
                                                                                         apply_resolution_limit,
                                                                                         apply_ctf,
                                                                                         absolute_ctf);

    postcheck;

    if ( swap_quadrants )
        object_is_centred_in_box = true;
    else
        object_is_centred_in_box = false;

    is_fft_centered_in_box = false;
    is_in_real_space       = false;
}
