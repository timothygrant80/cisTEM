/*
 * GpuImage.cpp
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

//#include "gpu_core_headers.h"

#include "gpu_core_headers.h"

__global__ void ConvertToHalfPrecisionKernelComplex(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex);
__global__ void ConvertToHalfPrecisionKernelReal(cufftReal* real_32f_values, __half* real_16f_values, int4 dims);

__global__ void MultiplyPixelWiseComplexConjugateKernel(cufftComplex* ref_complex_values, cufftComplex* img_complex_values, int4 dims, int3 physical_upper_bound_complex);
__global__ void MipPixelWiseKernel(cufftReal* mip, const cufftReal* correlation_output, const int4 dims);
__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal* other_image, cufftReal* psi, cufftReal* phi, cufftReal* theta,
                                   int4 dims, float c_psi, float c_phi, float c_theta);
__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal* other_image, cufftReal* psi, cufftReal* phi, cufftReal* theta, cufftReal* defocus, cufftReal* pixel, const int4 dims,
                                   float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel);

__global__ void ApplyBFactorKernel(cufftComplex* d_input,
                                   const int4    dims,
                                   const int3    physical_index_of_first_negative_frequency,
                                   const int3    physical_upper_bound_complex,
                                   float         bfactor);

__global__ void ApplyBFactorKernel(cufftComplex* d_input,
                                   const int4    dims,
                                   const int3    physical_index_of_first_negative_frequency,
                                   const int3    physical_upper_bound_complex,
                                   float         bfactor,
                                   const float   vertical_mask_size,
                                   const float   horizontal_mask_size); // Specialization for unblur refinement, merges MaskCentralCross()

__global__ void PhaseShiftKernel(cufftComplex* d_input,
                                 int4 dims, float3 shifts,
                                 int3 physical_address_of_box_center,
                                 int3 physical_index_of_first_negative_frequency,
                                 int3 physical_upper_bound_complex);

__global__ void ClipIntoRealKernel(cufftReal* real_values_gpu,
                                   cufftReal* other_image_real_values_gpu,
                                   int4       dims,
                                   int4       other_dims,
                                   int3       physical_address_of_box_center,
                                   int3       other_physical_address_of_box_center,
                                   int3       wanted_coordinate_of_box_center,
                                   float      wanted_padding_value);

// cuFFT callbacks
__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {

    const __half element = ((__half*)dataIn)[offset];
    return (cufftReal)(__half2float(element));
}

__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr);

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr) {
    float scale_factor = *(float*)callerInfo;

    ((cufftComplex*)dataOut)[offset] = (cufftComplex)ComplexScale(element, scale_factor);
}

//__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;
__device__ cufftCallbackStoreC d_scaleFFTAndStorePtr = CB_scaleFFTAndStore;

__device__ void CB_mipCCGAndStore(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPtr);

__device__ void CB_mipCCGAndStore(void* dataOut, size_t offset, cufftReal element, void* callerInfo, void* sharedPtr) {

    __half* data_out_half = (__half*)callerInfo;
    //	data_out_half[offset] = __float2half(element);
    //	((cufftReal *)dataOut)[offset] = element;

#ifdef DISABLECACHEHINTS
    ((__half*)data_out_half)[offset] = __float2half(element);
#else
    __stcs(&data_out_half[offset], __float2half(element));
#endif

    //	)_(dataOut)[offset] = __float2half(element);
    //
}

__device__ cufftCallbackStoreR d_mipCCGAndStorePtr = CB_mipCCGAndStore;

template <typename T>
struct CB_complexConjMulLoad_params {
    T*    target;
    float scale;
};

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);
static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {
    CB_complexConjMulLoad_params<cufftComplex>* my_params = (CB_complexConjMulLoad_params<cufftComplex>*)callerInfo;
    return (cufftComplex)ComplexConjMulAndScale(my_params->target[offset], ((Complex*)dataIn)[offset], my_params->scale);
    //    return (cufftComplex)make_float2(my_params->scale * __fmaf_rn(my_params->target[offset].x ,  ((Complex *)dataIn)[offset].x, my_params->target[offset].y * ((Complex *)dataIn)[offset].y),
    //    		my_params->scale * __fmaf_rn(my_params->target[offset].y , -((Complex *)dataIn)[offset].x, my_params->target[offset].x * ((Complex *)dataIn)[offset].y));
}

static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {
    CB_complexConjMulLoad_params<__half2>* my_params = (CB_complexConjMulLoad_params<__half2>*)callerInfo;
    return (cufftComplex)ComplexConjMulAndScale((cufftComplex)__half22float2(my_params->target[offset]), ((Complex*)dataIn)[offset], my_params->scale);
}

__device__ cufftCallbackLoadC d_complexConjMulLoad_32f = CB_complexConjMulLoad_32f;
__device__ cufftCallbackLoadC d_complexConjMulLoad_16f = CB_complexConjMulLoad_16f;

typedef struct _CB_realLoadAndClipInto_params {
    int*       mask;
    cufftReal* target;

} CB_realLoadAndClipInto_params;

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {

    CB_realLoadAndClipInto_params* my_params = (CB_realLoadAndClipInto_params*)callerInfo;
    int                            idx       = my_params->mask[offset];
    if ( idx == 0 ) {
        return 0.0f;
    }
    else {
        return my_params->target[idx];
    }
}

__device__ cufftCallbackLoadR d_realLoadAndClipInto = CB_realLoadAndClipInto;

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
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_dims, int3 physical_upper_bound_complex);

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

////////////////////////

GpuImage::GpuImage( ) {
    SetupInitialValues( );
}

GpuImage::GpuImage(Image& cpu_image) {

    SetupInitialValues( );
    Init(cpu_image);
}

GpuImage::GpuImage(const GpuImage& other_gpu_image) // copy constructor
{

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

        MyAssertTrue(other_gpu_image->is_in_memory_gpu, "Other image Memory not allocated");

        if ( is_in_memory_gpu == true ) {

            if ( dims.x != other_gpu_image->dims.x || dims.y != other_gpu_image->dims.y || dims.z != other_gpu_image->dims.z ) {
                Deallocate( );
                Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
            }
        }
        else {
            Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
        }

        // by here the memory allocation should be ok..

        is_in_real_space         = other_gpu_image->is_in_real_space;
        object_is_centred_in_box = other_gpu_image->object_is_centred_in_box;

        precheck
                cudaErr(cudaMemcpyAsync(real_values_gpu, other_gpu_image->real_values_gpu, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        postcheck
    }

    return *this;
}

GpuImage::~GpuImage( ) {
    Deallocate( );
    // cudaErr(cudaFree(tmpVal));
    // cudaErr(cudaFree(tmpValComplex));
}

void GpuImage::Init(Image& cpu_image) {
    CopyFromCpuImage(cpu_image);
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

    real_values    = NULL;
    complex_values = NULL;

    real_memory_allocated = 0;

    padding_jump_value = 0;

    ft_normalization_factor = 0;

    real_values_gpu    = NULL; // !<  Real array to hold values for REAL images.
    complex_values_gpu = NULL; // !<  Complex array to hold values for COMP images.

    gpu_plan_id = -1;

    insert_into_which_reconstruction = 0;
    hostImage                        = NULL;

    cudaErr(cudaEventCreateWithFlags(&nppCalcEvent, cudaEventDisableTiming));

            cudaErr(cudaGetDevice(&device_idx));
    cudaErr(cudaDeviceGetAttribute(&number_of_streaming_multiprocessors, cudaDevAttrMultiProcessorCount, device_idx));
    limit_SMs_by_threads = 1;

    UpdateBoolsToDefault( );
}

void GpuImage::CopyFrom(GpuImage* other_image) {
    *this = other_image;
}

void GpuImage::CopyFromCpuImage(Image& cpu_image) {

    UpdateBoolsToDefault( );

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

    fourier_voxel_size = make_float3(cpu_image.fourier_voxel_size_x,
                                     cpu_image.fourier_voxel_size_y,
                                     cpu_image.fourier_voxel_size_z);

    insert_into_which_reconstruction = cpu_image.insert_into_which_reconstruction;
    real_values                      = cpu_image.real_values;
    complex_values                   = cpu_image.complex_values;

    is_in_memory = cpu_image.is_in_memory;

    padding_jump_value                     = cpu_image.padding_jump_value;
    image_memory_should_not_be_deallocated = cpu_image.image_memory_should_not_be_deallocated; // TODO what is this for?

    real_values_gpu       = NULL; // !<  Real array to hold values for REAL images.
    complex_values_gpu    = NULL; // !<  Complex array to hold values for COMP images.
    is_in_memory_gpu      = false;
    real_memory_allocated = cpu_image.real_memory_allocated;

    ft_normalization_factor = cpu_image.ft_normalization_factor;

    // FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning
    cudaHostRegister(real_values, sizeof(float) * real_memory_allocated, cudaHostRegisterDefault);
    is_host_memory_pinned    = true;
    is_meta_data_initialized = true;
    cudaHostGetDevicePointer(&pinnedPtr, real_values, 0);

    cudaMallocManaged(&tmpVal, sizeof(float));
    cudaMallocManaged(&tmpValComplex, sizeof(double));

    hostImage = &cpu_image;
}

void GpuImage::UpdateCpuFlags( ) {

    // Call after re-copying. The main image properites are all assumed to be static.
    is_in_real_space         = hostImage->is_in_real_space;
    object_is_centred_in_box = hostImage->object_is_centred_in_box;
}

void GpuImage::printVal(std::string msg, int idx) {

    float h_printVal = -9999.0f;

    cudaErr(cudaMemcpy(&h_printVal, &real_values_gpu[idx], sizeof(float), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(cudaStreamPerThread);
    wxPrintf("%s %6.6e\n", msg, h_printVal);
};

bool GpuImage::HasSameDimensionsAs(GpuImage* other_image) {
    // Functions that call this method also assume these asserts are being called here, so do not remove.
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
    // end of dependent asserts.

    if ( dims.x == other_image->dims.x && dims.y == other_image->dims.y && dims.z == other_image->dims.z )
        return true;
    else
        return false;
}

void GpuImage::MultiplyPixelWiseComplexConjugate(GpuImage& other_image) {
    // FIXME when adding real space complex images
    MyAssertFalse(is_in_real_space, "Image is in real space");
    MyAssertFalse(other_image.is_in_real_space, "Other image is in real space");
    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

    //  NppInit();
    //  Conj();
    //  npp_stat = nppiMul_32sc_C1IRSfs((const Npp32sc *)complex_values_gpu, 1, (Npp32sc*)other_image.complex_values_gpu, 1, npp_ROI_complex, 0);

    precheck
            ReturnLaunchParamters(dims, false);
    MultiplyPixelWiseComplexConjugateKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_gpu, other_image.complex_values_gpu, this->dims, this->physical_upper_bound_complex);
    postcheck
}

__global__ void ReturnSumOfRealValuesOnEdgesKernel(cufftReal* real_values_gpu, int4 dims, int padding_jump_value, float* returnValue);

float GpuImage::ReturnAverageOfRealValuesOnEdges( ) {
    // FIXME to use a masked routing, this is slow af
    MyAssertTrue(is_in_memory, "Memory not allocated");
    MyAssertTrue(dims.z == 1, "ReturnAverageOfRealValuesOnEdges only implemented in 2d");

    precheck* tmpVal = 5.0f;
    ReturnSumOfRealValuesOnEdgesKernel<<<1, 1, 0, cudaStreamPerThread>>>(real_values_gpu, dims, padding_jump_value, tmpVal);
    postcheck

            // Need to wait on the return value
            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    return *tmpVal;
}

__global__ void ReturnSumOfRealValuesOnEdgesKernel(cufftReal* real_values_gpu, int4 dims, int padding_jump_value, float* returnValue) {

    int pixel_counter;
    int line_counter;
    //	int plane_counter;

    double sum              = 0.0;
    int    number_of_pixels = 0;
    int    address          = 0;

    // Two-dimensional image
    // First line
    for ( pixel_counter = 0; pixel_counter < dims.x; pixel_counter++ ) {
        sum += real_values_gpu[address];
        address++;
    }
    number_of_pixels += dims.x;
    address += padding_jump_value;

    // Other lines
    for ( line_counter = 1; line_counter < dims.y - 1; line_counter++ ) {
        sum += real_values_gpu[address];
        address += dims.x - 1;
        sum += real_values_gpu[address];
        address += padding_jump_value + 1;
        number_of_pixels += 2;
    }

    // Last line
    for ( pixel_counter = 0; pixel_counter < dims.x; pixel_counter++ ) {
        sum += real_values_gpu[address];
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
        nppStream.hStream = cudaStreamPerThread; // FIXME to use member stream
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

void GpuImage::BufferInit(BufferType bt) {

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
                cudaErr(cudaMalloc(&real_values_16f, sizeof(__half) * real_memory_allocated));
                complex_values_16f      = (__half2*)real_values_16f;
                is_allocated_16f_buffer = true;
            }
            break;

        case b_sum:
            if ( ! is_allocated_sum_buffer ) {
                int n_elem;
                nppiSumGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->sum_buffer, n_elem));
                is_allocated_sum_buffer = true;
            }
            break;

        case b_min:
            if ( ! is_allocated_min_buffer ) {
                int n_elem;
                nppiMinGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->min_buffer, n_elem));
                is_allocated_min_buffer = true;
            }
            break;

        case b_minIDX:
            if ( ! is_allocated_minIDX_buffer ) {
                int n_elem;
                nppiMinIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->minIDX_buffer, n_elem));
                is_allocated_minIDX_buffer = true;
            }
            break;

        case b_max:
            if ( ! is_allocated_max_buffer ) {
                int n_elem;
                nppiMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->max_buffer, n_elem));
                is_allocated_max_buffer = true;
            }
            break;

        case b_maxIDX:
            if ( ! is_allocated_maxIDX_buffer ) {
                int n_elem;
                nppiMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->maxIDX_buffer, n_elem));
                is_allocated_maxIDX_buffer = true;
            }
            break;

        case b_minmax:
            if ( ! is_allocated_minmax_buffer ) {
                int n_elem;
                nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->minmax_buffer, n_elem));
                is_allocated_minmax_buffer = true;
            }
            break;

        case b_minmaxIDX:
            if ( ! is_allocated_minmaxIDX_buffer ) {
                int n_elem;
                nppiMinMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->minmaxIDX_buffer, n_elem));
                is_allocated_minmaxIDX_buffer = true;
            }
            break;

        case b_mean:
            if ( ! is_allocated_mean_buffer ) {
                int n_elem;
                nppiMeanGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->mean_buffer, n_elem));
                is_allocated_mean_buffer = true;
            }
            break;

        case b_meanstddev:
            if ( ! is_allocated_meanstddev_buffer ) {
                int n_elem;
                nppiMeanGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->meanstddev_buffer, n_elem));
                is_allocated_meanstddev_buffer = true;
            }
            break;

        case b_countinrange:
            if ( ! is_allocated_countinrange_buffer ) {
                int n_elem;
                nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->countinrange_buffer, n_elem));
                is_allocated_countinrange_buffer = true;
            }
            break;

        case b_l2norm:
            if ( ! is_allocated_l2norm_buffer ) {
                int n_elem;
                nppiNormL2GetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->l2norm_buffer, n_elem));

                is_allocated_l2norm_buffer = true;
            }
            break;

        case b_dotproduct:
            if ( ! is_allocated_dotproduct_buffer ) {
                int n_elem;
                nppiDotProdGetBufferHostSize_32f64f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
                cudaErr(cudaMalloc(&this->dotproduct_buffer, n_elem));
                is_allocated_dotproduct_buffer = true;
            }
            break;
    }
}

void GpuImage::BufferDestroy( ) {

    if ( is_allocated_image_buffer ) {
        image_buffer->Deallocate( );
        cudaErr(cudaFree(image_buffer));
        is_allocated_image_buffer = false;
    }

    if ( is_allocated_16f_buffer ) {
        cudaErr(cudaFree(real_values_16f));
        is_allocated_16f_buffer = false;
    }

    if ( is_allocated_sum_buffer ) {
        cudaErr(cudaFree(sum_buffer));
        is_allocated_sum_buffer = false;
    }

    if ( is_allocated_min_buffer ) {
        cudaErr(cudaFree(min_buffer));
        is_allocated_min_buffer = false;
    }

    if ( is_allocated_minIDX_buffer ) {
        cudaErr(cudaFree(minIDX_buffer));
        is_allocated_minIDX_buffer = false;
    }

    if ( is_allocated_max_buffer ) {
        cudaErr(cudaFree(max_buffer));
        is_allocated_max_buffer = false;
    }

    if ( is_allocated_maxIDX_buffer ) {
        cudaErr(cudaFree(maxIDX_buffer));
        is_allocated_maxIDX_buffer = false;
    }

    if ( is_allocated_minmax_buffer ) {
        cudaErr(cudaFree(minmax_buffer));
        is_allocated_minmax_buffer = false;
    }

    if ( is_allocated_minmaxIDX_buffer ) {
        cudaErr(cudaFree(minmaxIDX_buffer));
        is_allocated_minmaxIDX_buffer = false;
    }

    if ( is_allocated_mean_buffer ) {
        cudaErr(cudaFree(mean_buffer));
        is_allocated_mean_buffer = false;
    }

    if ( is_allocated_meanstddev_buffer ) {
        cudaErr(cudaFree(meanstddev_buffer));
        is_allocated_meanstddev_buffer = false;
    }

    if ( is_allocated_countinrange_buffer ) {
        cudaErr(cudaFree(countinrange_buffer));
        is_allocated_countinrange_buffer = false;
    }

    if ( is_allocated_l2norm_buffer ) {
        cudaErr(cudaFree(l2norm_buffer));
        is_allocated_l2norm_buffer = false;
    }

    if ( is_allocated_dotproduct_buffer ) {
        cudaErr(cudaFree(dotproduct_buffer));
        is_allocated_dotproduct_buffer = false;
    }
}

float GpuImage::ReturnSumOfSquares( ) {

    // FIXME this assumes padded values are zero which is not strictly true
    MyAssertTrue(is_in_memory_gpu, "Image not allocated");
    MyAssertTrue(is_in_real_space, "This method is for real space, use ReturnSumSquareModulusComplexValues for Fourier space")

            BufferInit(b_l2norm);
    NppInit( );

    nppErr(nppiNorm_L2_32f_C1R_Ctx((Npp32f*)real_values_gpu, pitch, npp_ROI,
                                   (Npp64f*)tmpValComplex, (Npp8u*)this->l2norm_buffer, nppStream));

    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    return (float)(*tmpValComplex * *tmpValComplex);

    //	CublasInit();
    //	// With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
    //	cublas_stat = cublasSdot( cublasHandle, real_memory_allocated,
    //							real_values_gpu, 1,
    //							real_values_gpu, 1,
    //							&returnValue);
    //
    //	if (cublas_stat) {
    //	wxPrintf("Cublas return val %s\n", cublas_stat); }
    //
    //
    //	return returnValue;
}

float GpuImage::ReturnSumSquareModulusComplexValues( ) {

    //
    MyAssertTrue(is_in_memory_gpu, "Image not allocated");
    MyAssertFalse(is_in_real_space, "This method is NOT for real space, use ReturnSumofSquares for realspace") int address   = 0;
    bool                                                                                                           x_is_even = IsEven(dims.x);
    int                                                                                                            i, j, k, jj, kk;
    const std::complex<float>                                                                                      c1(sqrtf(0.25f), sqrtf(0.25));
    const std::complex<float>                                                                                      c2(sqrtf(0.5f), sqrtf(0.5f)); // original code is pow(abs(Val),2)*0.5
    const std::complex<float>                                                                                      c3(1.0, 1.0);
    const std::complex<float>                                                                                      c4(0.0, 0.0);
    //	float returnValue;

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
        cudaErr(cudaHostAlloc(&mask_CSOS->real_values, sizeof(float) * real_memory_allocated, cudaHostAllocDefault));
        mask_CSOS->complex_values = (std::complex<float>*)mask_CSOS->real_values;

        for ( k = 0; k <= physical_upper_bound_complex.z; k++ ) {

            kk = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k, dims.z, physical_index_of_first_negative_frequency.z);
            for ( j = 0; j <= physical_upper_bound_complex.y; j++ ) {
                jj = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j, dims.y, physical_index_of_first_negative_frequency.y);
                for ( i = 0; i <= physical_upper_bound_complex.x; i++ ) {
                    if ( (i == 0 || (i == logical_upper_bound_complex.x && x_is_even)) &&
                         (jj == 0 || (jj == logical_lower_bound_complex.y && x_is_even)) &&
                         (kk == 0 || (kk == logical_lower_bound_complex.z && x_is_even)) ) {
                        mask_CSOS->complex_values[address] = c2;
                    }
                    else if ( (i == 0 || (i == logical_upper_bound_complex.x && x_is_even)) && dims.z != 1 ) {
                        mask_CSOS->complex_values[address] = c1;
                    }
                    else if ( (i != 0 && (i != logical_upper_bound_complex.x || ! x_is_even)) || (jj >= 0 && kk >= 0) ) {
                        mask_CSOS->complex_values[address] = c3;
                    }
                    else {
                        mask_CSOS->complex_values[address] = c4;
                    }

                    address++;
                }
            }
        }

        precheck
                cudaErr(cudaMemcpyAsync(mask_CSOS->real_values_gpu, mask_CSOS->real_values, sizeof(float) * real_memory_allocated, cudaMemcpyHostToDevice, cudaStreamPerThread));
        precheck
                // TODO change this to an event that can then be later checked prior to deleteing
                cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        cudaErr(cudaFreeHost(mask_CSOS->real_values));

    } // end of mask creation

    BufferInit(b_image);
    precheck
            cudaErr(cudaMemcpyAsync(image_buffer->real_values_gpu, mask_CSOS->real_values_gpu, sizeof(float) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    postcheck

            image_buffer->is_in_real_space = false;
    image_buffer->npp_ROI                  = image_buffer->npp_ROI_fourier_space;
    image_buffer->MultiplyPixelWise(*this);

    //	CublasInit();
    // With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
    precheck

            BufferInit(b_l2norm);
    NppInit( );
    nppErr(nppiNorm_L2_32f_C1R_Ctx((Npp32f*)image_buffer->real_values_gpu, pitch, npp_ROI_fourier_with_real_functor,
                                   (Npp64f*)tmpValComplex, (Npp8u*)this->l2norm_buffer, nppStream));

    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    return (float)(*tmpValComplex * *tmpValComplex);

    //	cublasSdot( cublasHandle, real_memory_allocated,
    //			  image_buffer->real_values_gpu, 1,
    //			  image_buffer->real_values_gpu, 1,
    //			  &returnValue);

    postcheck

    //	return (float)(*dotProd * 2.0f);
    //	return (float)(returnValue * 2.0f);
}

__global__ void MultiplyPixelWiseComplexConjugateKernel(cufftComplex* ref_complex_values, cufftComplex* img_complex_values, int4 dims, int3 physical_upper_bound_complex) {
    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z ) {

        int address = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);

        ref_complex_values[address] = (cufftComplex)ComplexConjMul((Complex)img_complex_values[address], (Complex)ref_complex_values[address]);
    }
}

void GpuImage::MipPixelWise(GpuImage& other_image) {

    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    precheck
            ReturnLaunchParamters(dims, true);
    MipPixelWiseKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values_gpu, other_image.real_values_gpu, this->dims);
    postcheck
}

__global__ void MipPixelWiseKernel(cufftReal* mip, const cufftReal* correlation_output, const int4 dims) {

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

    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    precheck
            ReturnLaunchParamters(dims, true);
    MipPixelWiseKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values_gpu, other_image.real_values_gpu,
                                                                              psi.real_values_gpu, phi.real_values_gpu, theta.real_values_gpu, defocus.real_values_gpu, pixel.real_values_gpu,
                                                                              this->dims, c_psi, c_phi, c_theta, c_defocus, c_pixel);
    postcheck
}

__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal* correlation_output, cufftReal* psi, cufftReal* phi, cufftReal* theta, cufftReal* defocus, cufftReal* pixel, const int4 dims,
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

    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    precheck
            ReturnLaunchParamters(dims, true);
    MipPixelWiseKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values_gpu, other_image.real_values_gpu,
                                                                              psi.real_values_gpu, phi.real_values_gpu, theta.real_values_gpu,
                                                                              this->dims, c_psi, c_phi, c_theta);
    postcheck
}

__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal* correlation_output, cufftReal* psi, cufftReal* phi, cufftReal* theta, const int4 dims,
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

void GpuImage::ApplyBFactor(float bfactor) {

    MyAssertFalse(is_in_real_space, "This function is only for Fourier space images.");

    precheck
            ReturnLaunchParamters(dims, false);
    ApplyBFactorKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_gpu,
                                                                              dims,
                                                                              physical_index_of_first_negative_frequency,
                                                                              physical_upper_bound_complex,
                                                                              bfactor * 0.25f);
    postcheck
}

__global__ void ApplyBFactorKernel(cufftComplex* d_input,
                                   const int4    dims,
                                   const int3    physical_index_of_first_negative_frequency,
                                   const int3    physical_upper_bound_complex,
                                   float         bfactor) {

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

        ComplexScale((Complex*)&d_input[address], expf(-bfactor * frequency_squared));
    }
}

void GpuImage::ApplyBFactor(float bfactor, const float vertical_mask_size, const float horizontal_mask_size) {

    MyDebugAssertFalse(is_in_real_space, "This function is only for Fourier space images.");
    MyDebugAssertTrue(dims.z == 1, "This function is only for 2D images.");
    MyDebugAssertTrue(vertical_mask_size > 0 && horizontal_mask_size > 0, "Half width must be greater than 0");

    precheck
            ReturnLaunchParamters(dims, false);
    ApplyBFactorKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_gpu,
                                                                              dims,
                                                                              physical_index_of_first_negative_frequency,
                                                                              physical_upper_bound_complex,
                                                                              bfactor * 0.25f,
                                                                              vertical_mask_size,
                                                                              horizontal_mask_size);
    postcheck
}

__global__ void ApplyBFactorKernel(cufftComplex* d_input,
                                   const int4    dims,
                                   const int3    physical_index_of_first_negative_frequency,
                                   const int3    physical_upper_bound_complex,
                                   float         bfactor,
                                   const float   vertical_mask_size,
                                   const float   horizontal_mask_size) {

    int3 physical_dims = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y,
                                   blockIdx.z);

    if ( physical_dims.x <= physical_upper_bound_complex.x &&
         physical_dims.y <= physical_upper_bound_complex.y ) {
        int address = d_ReturnFourier1DAddressFromPhysicalCoord(physical_dims, physical_upper_bound_complex);

        if ( physical_dims.x <= horizontal_mask_size - 1 ||
             physical_dims.y <= vertical_mask_size - 1 ||
             physical_dims.y >= physical_upper_bound_complex.y - vertical_mask_size ) {
            // TODO: confirm this is correct
            // Mask the central cross
            d_input[address].x = 0.f;
            d_input[address].y = 0.f;
        }
        else {
            int x2, y2;

            y2 = d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(physical_dims.y, dims.y, physical_index_of_first_negative_frequency.y);
            y2 *= y2;

            x2 = d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(physical_dims.x, dims.x, physical_index_of_first_negative_frequency.x);
            x2 *= x2;

            ComplexScale((Complex*)&d_input[address], expf(-bfactor * (x2 + y2)));
        }
    }
}

void GpuImage::CalculateCrossCorrelationImageWith(GpuImage* other_image) {

    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(is_in_real_space == other_image->is_in_real_space, "Images are in different spaces");
    MyDebugAssertTrue(HasSameDimensionsAs(other_image) == true, "Images are different sizes");

    bool must_fft = false;

    // do we have to fft..

    if ( is_in_real_space == true ) {
        must_fft = true;
        ForwardFFT( );
        other_image->ForwardFFT( );
    }

    if ( object_is_centred_in_box == true ) {
        object_is_centred_in_box = false;
        SwapRealSpaceQuadrants( );
    }

    if ( other_image->object_is_centred_in_box == true ) {
        other_image->object_is_centred_in_box = false;
        other_image->SwapRealSpaceQuadrants( );
    }

    BackwardFFTAfterComplexConjMul(other_image->complex_values_gpu, false);

    if ( must_fft == true )
        other_image->BackwardFFT( );
}

Peak GpuImage::FindPeakWithParabolaFit(float inner_radius_for_peak_search, float outer_radius_for_peak_search) {

    MyDebugAssertTrue(is_in_real_space, "This function is only for real space images.");
    MyDebugAssertTrue(dims.z == 1, "This function is only for 2D images.");

    Peak     my_peak;
    GpuImage buffer;
    int      size = myroundint(outer_radius_for_peak_search * 2.0f) + 1;
    if ( size % 2 != 0 )
        size++;
    buffer.Allocate(size, size, true);
    ClipIntoRealSpace(&buffer, 0.f, false, 0.f, 0, 0, 0);
    // Since we free the GPU memory after the copy, we have to make sure the kernel is complete.
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    bool  should_block_until_complete = true;
    bool  free_gpu_memory             = true;
    Image cpu_buffer                  = buffer.CopyDeviceToNewHost(should_block_until_complete, free_gpu_memory);
    my_peak                           = cpu_buffer.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);
    wxPrintf("Peak found at %f, %f\n", my_peak.x, my_peak.y);

    return my_peak;
}

void GpuImage::Abs( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    nppErr(nppiAbs_32f_C1IR_Ctx((Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::AbsDiff(GpuImage& other_image) {
    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");

    BufferInit(b_image);
    NppInit( );

    nppErr(nppiAbsDiff_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch,
                                   (const Npp32f*)other_image.real_values_gpu, pitch,
                                   (Npp32f*)this->image_buffer->real_values_gpu, pitch, npp_ROI, nppStream));

    precheck
            cudaErr(cudaMemcpyAsync(real_values_gpu, this->image_buffer->real_values_gpu, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    postcheck
}

void GpuImage::AbsDiff(GpuImage& other_image, GpuImage& output_image) {
    // In place abs diff (see overload for out of place)
    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
    MyAssertTrue(HasSameDimensionsAs(&output_image), "Images have different dimension.");

    NppInit( );

    nppErr(nppiAbsDiff_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch,
                                   (const Npp32f*)other_image.real_values_gpu, pitch,
                                   (Npp32f*)output_image.real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::Min( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_min);
    nppErr(nppiMin_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, min_buffer, (Npp32f*)&min_value, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MinAndCoords( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_minIDX);
    nppErr(nppiMinIndx_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, minIDX_buffer, (Npp32f*)&min_value, &min_idx.x, &min_idx.y, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::Max( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_max);
    nppErr(nppiMax_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, max_buffer, (Npp32f*)&max_value, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MaxAndCoords( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_maxIDX);
    nppErr(nppiMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, maxIDX_buffer, (Npp32f*)&max_value, &max_idx.x, &max_idx.y, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MinMax( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_minmax);
    nppErr(nppiMinMax_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, (Npp32f*)&min_value, (Npp32f*)&max_value, minmax_buffer, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::MinMaxAndCoords( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_minmaxIDX);
    nppErr(nppiMinMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, (Npp32f*)&min_value, (Npp32f*)&max_value, &min_idx, &max_idx, minmax_buffer, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

void GpuImage::Mean( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_mean);
    nppErr(nppiMean_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, mean_buffer, npp_mean, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    this->img_mean = (float)*npp_mean;
}

void GpuImage::MeanStdDev( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_meanstddev);
    nppErr(nppiMean_StdDev_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, meanstddev_buffer, npp_mean, npp_stdDev, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    this->img_mean   = (float)*npp_mean;
    this->img_stdDev = (float)*npp_stdDev;
}

void GpuImage::MultiplyPixelWise(GpuImage& other_image) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    if ( is_in_real_space ) {
        nppErr(nppiMul_32f_C1IR_Ctx((Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
    }
    else {
        nppErr(nppiMul_32fc_C1IR_Ctx((Npp32fc*)other_image.complex_values_gpu, pitch, (Npp32fc*)complex_values_gpu, pitch, npp_ROI, nppStream));
    }
}

void GpuImage::AddConstant(const float add_val) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiAddC_32f_C1IR_Ctx((Npp32f)add_val, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::AddConstant(const Npp32fc add_val) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Image in real space.")

            NppInit( );
    nppErr(nppiAddC_32fc_C1IR_Ctx((Npp32fc)add_val, (Npp32fc*)complex_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::SquareRealValues( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiSqr_32f_C1IR_Ctx((Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::SquareRootRealValues( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiSqrt_32f_C1IR_Ctx((Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::LogarithmRealValues( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    nppErr(nppiLn_32f_C1IR_Ctx((Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::ExponentiateRealValues( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiExp_32f_C1IR_Ctx((Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::CountInRange(float lower_bound, float upper_bound) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    nppErr(nppiCountInRange_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, &number_of_pixels_in_range,
                                        (Npp32f)lower_bound, (Npp32f)upper_bound, countinrange_buffer, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));
}

float GpuImage::ReturnSumOfRealValues( ) {
    // FIXME assuming padded values are zero
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Not in real space");

    NppInit( );
    BufferInit(b_sum);
    nppErr(nppiSum_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, sum_buffer, (Npp64f*)tmpValComplex, nppStream));
    cudaErr(cudaStreamSynchronize(nppStream.hStream));

    return (float)*tmpValComplex;
}

void GpuImage::AddImage(GpuImage& other_image) {
    // Add the real_values_gpu into a double array
    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

    NppInit( );
    nppErr(nppiAdd_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::SubtractImage(GpuImage& other_image) {
    // Add the real_values_gpu into a double array
    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

    NppInit( );
    nppErr(nppiSub_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::AddSquaredImage(GpuImage& other_image) {
    // Add the real_values_gpu into a double array
    MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");
    MyAssertTrue(is_in_real_space, "Image is not in real space");

    NppInit( );
    nppErr(nppiAddSquare_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::MultiplyByConstant(float scale_factor) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    if ( is_in_real_space ) {
        nppErr(nppiMulC_32f_C1IR_Ctx((Npp32f)scale_factor, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
    }
    else {
        nppErr(nppiMulC_32f_C1IR_Ctx((Npp32f)scale_factor, (Npp32f*)real_values_gpu, pitch, npp_ROI_fourier_with_real_functor, nppStream));
    }
}

void GpuImage::SetToConstant(float scale_factor) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    if ( is_in_real_space ) {
        nppErr(nppiSet_32f_C1R_Ctx((Npp32f)scale_factor, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));
    }
    else {
        Npp32fc scale_factor_complex = {scale_factor, scale_factor};
        SetToConstant(scale_factor_complex);
    }
}

void GpuImage::SetToConstant(Npp32fc scale_factor_complex) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");

    NppInit( );
    nppErr(nppiSet_32fc_C1R_Ctx((Npp32fc)scale_factor_complex, (Npp32fc*)complex_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::Conj( ) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertFalse(is_in_real_space, "Conj only supports complex images");

    Npp32fc scale_factor;
    scale_factor.re = 1.0f;
    scale_factor.im = -1.0f;
    NppInit( );
    nppErr(nppiMulC_32fc_C1IR_Ctx((Npp32fc)scale_factor, (Npp32fc*)complex_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::Zeros( ) {

    MyAssertTrue(is_meta_data_initialized, "Host meta data has not been copied");

    if ( ! is_in_memory_gpu ) {
        cudaErr(cudaMalloc(&real_values_gpu, real_memory_allocated * sizeof(float)));
        complex_values_gpu = (cufftComplex*)real_values_gpu;
        is_in_memory_gpu   = true;
    }

    precheck
            cudaErr(cudaMemsetAsync(real_values_gpu, 0, real_memory_allocated * sizeof(float), cudaStreamPerThread));
    postcheck
}

void GpuImage::CopyHostToDevice( ) {

    MyAssertTrue(is_in_memory, "Host memory not allocated");

    if ( ! is_in_memory_gpu ) {
        cudaErr(cudaMalloc(&real_values_gpu, real_memory_allocated * sizeof(float)));
        complex_values_gpu = (cufftComplex*)real_values_gpu;
        is_in_memory_gpu   = true;
    }

    precheck
            cudaErr(cudaMemcpyAsync(real_values_gpu, pinnedPtr, real_memory_allocated * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
    postcheck

    UpdateCpuFlags( );
}

void GpuImage::CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory) {

    MyAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
    // TODO other asserts on size etc.
    precheck
            cudaErr(cudaMemcpyAsync(pinnedPtr, real_values_gpu, real_memory_allocated * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    postcheck
            //  cudaErr(cudaMemcpyAsync(real_values, real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
            // TODO add asserts etc.
            if ( free_gpu_memory ) {
        Deallocate( );
    }
    if ( unpin_host_memory && is_host_memory_pinned ) {
        cudaHostUnregister(real_values);
        is_host_memory_pinned = false;
    }
}

void GpuImage::CopyDeviceToHost(Image& cpu_image, bool should_block_until_complete, bool free_gpu_memory) {

    MyAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
    // TODO other asserts on size etc.

    float* tmpPinnedPtr;
    // FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning
    cudaHostRegister(cpu_image.real_values, sizeof(float) * real_memory_allocated, cudaHostRegisterDefault);
    cudaHostGetDevicePointer(&tmpPinnedPtr, cpu_image.real_values, 0);

    precheck
            cudaErr(cudaMemcpyAsync(tmpPinnedPtr, real_values_gpu, real_memory_allocated * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    postcheck

            if ( should_block_until_complete ) cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    // TODO add asserts etc.
    if ( free_gpu_memory ) {
        cudaFree(real_values_gpu);
    } // FIXME what about the other structures

    cudaHostUnregister(tmpPinnedPtr);
}

void GpuImage::CopyDeviceToNewHost(Image& cpu_image, bool should_block_until_complete, bool free_gpu_memory) {
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

    // cpu_image.real_values = real_values_gpu;

    // cpu_image.complex_values = complex_values_gpu;

    cpu_image.is_in_memory = is_in_memory;
    //cpu_image.padding_jump_value = padding_jump_value;
    cpu_image.image_memory_should_not_be_deallocated = image_memory_should_not_be_deallocated;

    //cpu_image.real_memory_allocated = real_memory_allocated;
    cpu_image.ft_normalization_factor = ft_normalization_factor;

    CopyDeviceToHost(cpu_image, should_block_until_complete, free_gpu_memory);

    cpu_image.is_in_memory     = true;
    cpu_image.is_in_real_space = is_in_real_space;
}

Image GpuImage::CopyDeviceToNewHost(bool should_block_until_complete, bool free_gpu_memory) {
    Image new_cpu_image;
    new_cpu_image.Allocate(dims.x, dims.y, dims.z, true, false);

    //new_cpu_image.Allocate(dims.x,dims.y);
    CopyDeviceToNewHost(new_cpu_image, should_block_until_complete, free_gpu_memory);
    return new_cpu_image;
}

void GpuImage::CopyVolumeHostToDevice( ) {

    // FIXME not working
    bool is_working = false;
    MyAssertTrue(is_working, "CopyVolumeHostToDevice is not properly worked out");

    d_pitchedPtr = {0};
    d_extent     = make_cudaExtent(dims.x * sizeof(float), dims.y, dims.z);
    cudaErr(cudaMalloc3D(&d_pitchedPtr, d_extent)); // Complex values need to be pointed
    this->real_values_gpu = (cufftReal*)d_pitchedPtr.ptr; // Set the values here

    d_3dparams        = {0};
    d_3dparams.srcPtr = make_cudaPitchedPtr((void*)real_values, dims.x * sizeof(float), dims.x, dims.y);
    d_3dparams.dstPtr = d_pitchedPtr;
    d_3dparams.extent = d_extent;
    d_3dparams.kind   = cudaMemcpyHostToDevice;
    cudaErr(cudaMemcpy3D(&d_3dparams));
}

void GpuImage::CopyVolumeDeviceToHost(bool free_gpu_memory, bool unpin_host_memory) {

    // FIXME not working
    bool is_working = false;
    MyAssertTrue(is_working, "CopyVolumeDeviceToHost is not properly worked out");

    if ( ! is_in_memory ) {
        cudaErr(cudaMallocHost(&real_values, real_memory_allocated * sizeof(float)));
    }
    h_pitchedPtr      = make_cudaPitchedPtr((void*)real_values, dims.x * sizeof(float), dims.x, dims.y);
    h_extent          = make_cudaExtent(dims.x * sizeof(float), dims.y, dims.z);
    h_3dparams        = {0};
    h_3dparams.srcPtr = d_pitchedPtr;
    h_3dparams.dstPtr = h_pitchedPtr;
    h_3dparams.extent = h_extent;
    h_3dparams.kind   = cudaMemcpyDeviceToHost;
    cudaErr(cudaMemcpy3D(&h_3dparams));

    is_in_memory = true;

    // TODO add asserts etc.
    if ( free_gpu_memory ) {
        cudaFree(d_pitchedPtr.ptr);
    } // FIXME what about the other structures
    if ( unpin_host_memory && is_host_memory_pinned ) {
        cudaHostUnregister(real_values);
        is_host_memory_pinned = false;
    }
}

void GpuImage::ForwardFFT(bool should_scale) {

    bool is_half_precision = false;

    MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyAssertTrue(is_in_real_space, "Image alread in Fourier space");

    if ( ! is_fft_planned ) {
        SetCufftPlan( );
    }

    // For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
    if ( is_half_precision && ! is_set_convertInputf16Tof32 ) {
        cufftCallbackLoadR h_ConvertInputf16Tof32Ptr;
        cudaErr(cudaMemcpyFromSymbol(&h_ConvertInputf16Tof32Ptr, d_ConvertInputf16Tof32Ptr, sizeof(h_ConvertInputf16Tof32Ptr)));
        cudaErr(cufftXtSetCallback(cuda_plan_forward, (void**)&h_ConvertInputf16Tof32Ptr, CUFFT_CB_LD_REAL, 0));
        is_set_convertInputf16Tof32 = true;
        //	  cudaErr(cudaFree(norm_factor));
        //	  this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
    }
    if ( should_scale ) {
        this->MultiplyByConstant(ft_normalization_factor * ft_normalization_factor);
    }

    //	if (should_scale && ! is_set_scaleFFTAndStore)
    //	{
    //
    //		float ft_norm_sq = ft_normalization_factor*ft_normalization_factor;
    //		cudaErr(cudaMalloc((void **)&d_scale_factor, sizeof(float)));
    //		cudaErr(cudaMemcpyAsync(d_scale_factor, &ft_norm_sq, sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
    //		cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    //
    //		cufftCallbackStoreC h_scaleFFTAndStorePtr;
    //		cudaErr(cudaMemcpyFromSymbol(&h_scaleFFTAndStorePtr,d_scaleFFTAndStorePtr, sizeof(h_scaleFFTAndStorePtr)));
    //		cudaErr(cufftXtSetCallback(cuda_plan_forward, (void **)&h_scaleFFTAndStorePtr, CUFFT_CB_ST_COMPLEX, (void **)&d_scale_factor));
    //		is_set_scaleFFTAndStore = true;
    //	}

    //	BufferInit(b_image);
    //    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)image_buffer->complex_values));

    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)complex_values_gpu));

    is_in_real_space = false;
    npp_ROI          = npp_ROI_fourier_space;
}

void GpuImage::ForwardFFTAndClipInto(GpuImage& image_to_insert, bool should_scale) {

    MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyAssertTrue(image_to_insert.is_in_memory_gpu, "Gpu memory in image to insert not allocated");
    MyAssertTrue(is_in_real_space, "Image alread in Fourier space");
    MyAssertTrue(image_to_insert.is_in_real_space, "I in image to insert alread in Fourier space");

    if ( ! is_fft_planned ) {
        SetCufftPlan( );
    }

    // For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
    if ( ! is_set_realLoadAndClipInto ) {

        // We need to make the mask
        image_to_insert.ClipIntoReturnMask(this);

        cufftCallbackLoadR             h_realLoadAndClipInto;
        CB_realLoadAndClipInto_params* d_params;
        CB_realLoadAndClipInto_params  h_params;

        h_params.target = (cufftReal*)image_to_insert.real_values_gpu;
        h_params.mask   = (int*)clip_into_mask;

        cudaErr(cudaMalloc((void**)&d_params, sizeof(CB_realLoadAndClipInto_params)));
        cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_realLoadAndClipInto_params), cudaMemcpyHostToDevice, cudaStreamPerThread));
        cudaErr(cudaMemcpyFromSymbol(&h_realLoadAndClipInto, d_realLoadAndClipInto, sizeof(h_realLoadAndClipInto)));
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

        cudaErr(cufftXtSetCallback(cuda_plan_forward, (void**)&h_realLoadAndClipInto, CUFFT_CB_LD_REAL, (void**)&d_params));
        is_set_realLoadAndClipInto = true;

        //	  cudaErr(cudaFree(norm_factor));
        //	  this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
    }
    if ( should_scale ) {
        this->MultiplyByConstant(ft_normalization_factor * ft_normalization_factor);
    }

    //	BufferInit(b_image);
    //    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)image_buffer->complex_values));

    cudaErr(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)complex_values_gpu));

    is_in_real_space = false;
    npp_ROI          = npp_ROI_fourier_space;
}

void GpuImage::BackwardFFT( ) {

    MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyAssertFalse(is_in_real_space, "Image is already in real space");

    if ( ! is_fft_planned ) {
        SetCufftPlan( );
    }

    //  BufferInit(b_image);
    //  cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values_gpu));

    cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)complex_values_gpu, (cufftReal*)real_values_gpu));

    is_in_real_space = true;
    npp_ROI          = npp_ROI_real_space;
}

template <typename T>
void GpuImage::BackwardFFTAfterComplexConjMul(T* image_to_multiply, bool load_half_precision) {

    MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
    MyAssertFalse(is_in_real_space, "Image is already in real space");
    MyAssertTrue(load_half_precision ? is_allocated_16f_buffer : true, "FP16 memory is not allocated, but is requested.");

    if ( ! is_fft_planned ) {
        SetCufftPlan( );
    }
    if ( ! is_set_complexConjMulLoad ) {
        cufftCallbackStoreC              h_complexConjMulLoad;
        cufftCallbackStoreR              h_mipCCGStore;
        CB_complexConjMulLoad_params<T>* d_params;
        CB_complexConjMulLoad_params<T>  h_params;
        h_params.scale  = ft_normalization_factor * ft_normalization_factor;
        h_params.target = (T*)image_to_multiply;
        cudaErr(cudaMalloc((void**)&d_params, sizeof(CB_complexConjMulLoad_params<T>)));
        cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_complexConjMulLoad_params<T>), cudaMemcpyHostToDevice, cudaStreamPerThread));
        if ( load_half_precision ) {
            cudaErr(cudaMemcpyFromSymbol(&h_complexConjMulLoad, d_complexConjMulLoad_16f, sizeof(h_complexConjMulLoad)));
        }
        else {
            cudaErr(cudaMemcpyFromSymbol(&h_complexConjMulLoad, d_complexConjMulLoad_32f, sizeof(h_complexConjMulLoad)));
        }

        cudaErr(cudaMemcpyFromSymbol(&h_mipCCGStore, d_mipCCGAndStorePtr, sizeof(h_mipCCGStore)));
        cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
        cudaErr(cufftXtSetCallback(cuda_plan_inverse, (void**)&h_complexConjMulLoad, CUFFT_CB_LD_COMPLEX, (void**)&d_params));
        //		void** fake_params;real_values_16f
        cudaErr(cufftXtSetCallback(cuda_plan_inverse, (void**)&h_mipCCGStore, CUFFT_CB_ST_REAL, (void**)&real_values_16f));

        //		d_complexConjMulLoad;
        is_set_complexConjMulLoad = true;
    }

    //  BufferInit(b_image);
    //  cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values_gpu));

    cudaErr(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)complex_values_gpu, (cufftReal*)real_values_gpu));

    is_in_real_space = true;
    npp_ROI          = npp_ROI_real_space;
}

template void GpuImage::BackwardFFTAfterComplexConjMul(__half2* image_to_multiply, bool load_half_precision);
template void GpuImage::BackwardFFTAfterComplexConjMul(cufftComplex* image_to_multiply, bool load_half_precision);

void GpuImage::Record( ) {
    cudaErr(cudaEventRecord(nppCalcEvent, cudaStreamPerThread));
}

void GpuImage::Wait( ) {
    cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, nppCalcEvent, 0));
    //  cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
}

void GpuImage::RecordAndWait( ) {
    Record( );
    Wait( );
}

void GpuImage::SwapRealSpaceQuadrants( ) {

    MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");

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

    PhaseShift(x_shift_to_apply, y_shift_to_apply, z_shift_to_apply);

    if ( must_fft == true )
        BackwardFFT( );

    // keep track of center;
    if ( object_is_centred_in_box == true )
        object_is_centred_in_box = false;
    else
        object_is_centred_in_box = true;
}

void GpuImage::PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift) {

    MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");

    bool need_to_fft = false;
    if ( is_in_real_space == true ) {
        wxPrintf("Doing forward fft in phase shift function\n\n");
        ForwardFFT(true);
        need_to_fft = true;
    }

    float3 shifts = make_float3(wanted_x_shift, wanted_y_shift, wanted_z_shift);
    // TODO set the TPB and inline function for grid

    dim3 threadsPerBlock(32, 32, 1);
    dim3 gridDims((dims.w / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y, dims.z);

    precheck
            PhaseShiftKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_gpu,
                                                                                    dims, shifts,
                                                                                    physical_address_of_box_center,
                                                                                    physical_index_of_first_negative_frequency,
                                                                                    physical_upper_bound_complex);

    postcheck

            if ( need_to_fft == true ) BackwardFFT( );
}

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
    //	MyAssertTrue(is_in_memory, "Memory not allocated");
    //	MyAssertTrue(physical_index <= physical_upper_bound_complex_x, "index out of bounds");

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
    //	MyAssertTrue(is_in_memory, "Memory not allocated");
    //	MyAssertTrue(physical_index <= physical_upper_bound_complex_y, "index out of bounds");

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
    //	MyAssertTrue(is_in_memory, "Memory not allocated");
    //	MyAssertTrue(physical_index <= physical_upper_bound_complex_z, "index out of bounds");

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
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_dims, int3 physical_upper_bound_complex) {
    return ((int)((physical_upper_bound_complex.y + 1) * wanted_dims.z + wanted_dims.y) *
                    (int)(physical_upper_bound_complex.x + 1) +
            (int)wanted_dims.x);
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

__global__ void ClipIntoRealKernel(cufftReal* real_values_gpu,
                                   cufftReal* other_image_real_values_gpu,
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
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = wanted_padding_value;
        }
        else {
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] =
                    real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(coord, dims)];
        }
    }
}

__global__ void ClipIntoMaskKernel(
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

__global__ void PhaseShiftKernel(cufftComplex* d_input,
                                 int4 dims, float3 shifts,
                                 int3 physical_address_of_box_center,
                                 int3 physical_index_of_first_negative_frequency,
                                 int3 physical_upper_bound_complex) {

    // FIXME it probably makes sense so just just a linear grid launch and save the extra indexing
    int3 wanted_dims = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    float2 init_vals;
    float2 angles;

    // FIXME This should probably use cuBlas
    if ( wanted_dims.x <= physical_upper_bound_complex.x &&
         wanted_dims.y <= physical_upper_bound_complex.y &&
         wanted_dims.z <= physical_upper_bound_complex.z ) {

        d_Return3DPhaseFromIndividualDimensions(d_ReturnPhaseFromShift(
                                                        shifts.x,
                                                        wanted_dims.x,
                                                        dims.x),
                                                d_ReturnPhaseFromShift(
                                                        shifts.y,
                                                        d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(
                                                                wanted_dims.y,
                                                                dims.y,
                                                                physical_index_of_first_negative_frequency.y),
                                                        dims.y),
                                                d_ReturnPhaseFromShift(
                                                        shifts.z,
                                                        d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(
                                                                wanted_dims.z,
                                                                dims.z,
                                                                physical_index_of_first_negative_frequency.z),
                                                        dims.z),
                                                angles);

        int address        = d_ReturnFourier1DAddressFromPhysicalCoord(wanted_dims, physical_upper_bound_complex);
        init_vals.x        = d_input[address].x;
        init_vals.y        = d_input[address].y;
        d_input[address].x = init_vals.x * angles.x - init_vals.y * angles.y;
        d_input[address].y = init_vals.x * angles.y + init_vals.y * angles.x;
    }
}

// If you don't want to clip from the center, you can give wanted_coordinate_of_box_center_{x,y,z}. This will define the pixel in the image at which other_image will be centered. (0,0,0) means center of image. This is a dumbed down version that does not fill with noise.
void GpuImage::ClipInto(GpuImage* other_image, float wanted_padding_value,
                        bool fill_with_noise, float wanted_noise_sigma,
                        int wanted_coordinate_of_box_center_x,
                        int wanted_coordinate_of_box_center_y,
                        int wanted_coordinate_of_box_center_z) {

    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
    MyAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");

    int3 wanted_coordinate_of_box_center = make_int3(wanted_coordinate_of_box_center_x,
                                                     wanted_coordinate_of_box_center_y,
                                                     wanted_coordinate_of_box_center_z);

    other_image->is_in_real_space         = is_in_real_space;
    other_image->object_is_centred_in_box = object_is_centred_in_box;

    if ( is_in_real_space == true ) {

        MyAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

        ReturnLaunchParamters(other_image->dims, true);

        precheck
                ClipIntoRealKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values_gpu,
                                                                                          other_image->real_values_gpu,
                                                                                          dims,
                                                                                          other_image->dims,
                                                                                          physical_address_of_box_center,
                                                                                          other_image->physical_address_of_box_center,
                                                                                          wanted_coordinate_of_box_center,
                                                                                          wanted_padding_value);
        postcheck
    }
}

void GpuImage::ClipIntoReturnMask(GpuImage* other_image) {

    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");

    int3 wanted_coordinate_of_box_center = make_int3(0, 0, 0);

    other_image->is_in_real_space         = is_in_real_space;
    other_image->object_is_centred_in_box = object_is_centred_in_box;

    cudaErr(cudaMalloc(&other_image->clip_into_mask, sizeof(int) * other_image->real_memory_allocated));
    other_image->is_allocated_clip_into_mask = true;

    if ( is_in_real_space == true ) {

        MyAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

        ReturnLaunchParamters(other_image->dims, true);

        precheck
                ClipIntoMaskKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(other_image->clip_into_mask,
                                                                                          dims,
                                                                                          other_image->dims,
                                                                                          physical_address_of_box_center,
                                                                                          other_image->physical_address_of_box_center,
                                                                                          wanted_coordinate_of_box_center,
                                                                                          0.0f);
        postcheck
    }
}

void GpuImage::QuickAndDirtyWriteSlices(std::string filename, int first_slice, int last_slice) {

    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    Image buffer_img;
    buffer_img.Allocate(dims.x, dims.y, dims.z, true);

    buffer_img.is_in_real_space         = is_in_real_space;
    buffer_img.object_is_centred_in_box = object_is_centred_in_box;
    // Implicitly waiting on work to finish since copy is queued in the stream
    cudaErr(cudaMemcpy((void*)buffer_img.real_values, (const void*)real_values_gpu, real_memory_allocated * sizeof(float), cudaMemcpyDeviceToHost));
    bool  OverWriteSlices = true;
    float pixelSize       = 0.0f;

    buffer_img.QuickAndDirtyWriteSlices(filename, first_slice, last_slice, OverWriteSlices, pixelSize);
    buffer_img.Deallocate( );
}

void GpuImage::SetCufftPlan(bool use_half_precision) {

    int            rank;
    long long int* fftDims;
    long long int* inembed;
    long long int* onembed;

    cudaErr(cufftCreate(&cuda_plan_forward));
    cudaErr(cufftCreate(&cuda_plan_inverse));

    cudaErr(cufftSetStream(cuda_plan_forward, cudaStreamPerThread));
    cudaErr(cufftSetStream(cuda_plan_inverse, cudaStreamPerThread));

    if ( dims.z > 1 ) {
        rank    = 3;
        fftDims = new long long int[rank];
        inembed = new long long int[rank];
        onembed = new long long int[rank];

        fftDims[0] = dims.z;
        fftDims[1] = dims.y;
        fftDims[2] = dims.x;

        inembed[0] = dims.z;
        inembed[1] = dims.y;
        inembed[2] = dims.w; // Storage dimension (padded)

        onembed[0] = dims.z;
        onembed[1] = dims.y;
        onembed[2] = dims.w / 2; // Storage dimension (padded)
    }
    else if ( dims.y > 1 ) {
        rank    = 2;
        fftDims = new long long int[rank];
        inembed = new long long int[rank];
        onembed = new long long int[rank];

        fftDims[0] = dims.y;
        fftDims[1] = dims.x;

        inembed[0] = dims.y;
        inembed[1] = dims.w;

        onembed[0] = dims.y;
        onembed[1] = dims.w / 2;
    }
    else {
        rank    = 1;
        fftDims = new long long int[rank];
        inembed = new long long int[rank];
        onembed = new long long int[rank];

        fftDims[0] = dims.x;

        inembed[0] = dims.w;
        onembed[0] = dims.w / 2;
    }

    int iBatch = 1;

    // As far as I can tell, the padded layout must be assumed and onembed/inembed
    // are not needed. TODO ask John about this.

    if ( use_half_precision ) {
        cudaErr(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
                                    NULL, NULL, NULL, CUDA_R_16F,
                                    NULL, NULL, NULL, CUDA_C_16F, iBatch, &cuda_plan_worksize_forward, CUDA_C_16F));
        cudaErr(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
                                    NULL, NULL, NULL, CUDA_C_16F,
                                    NULL, NULL, NULL, CUDA_R_16F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_16F));
    }
    else {
        cudaErr(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
                                    NULL, NULL, NULL, CUDA_R_32F,
                                    NULL, NULL, NULL, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F));
        cudaErr(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
                                    NULL, NULL, NULL, CUDA_C_32F,
                                    NULL, NULL, NULL, CUDA_R_32F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_32F));
    }

    //    cufftPlanMany(&dims.cuda_plan_forward, rank, fftDims,
    //                  inembed, iStride, iDist,
    //                  onembed, oStride, oDist, CUFFT_R2C, iBatch);
    //    cufftPlanMany(&dims.cuda_plan_inverse, rank, fftDims,
    //                  onembed, oStride, oDist,
    //                  inembed, iStride, iDist, CUFFT_C2R, iBatch);

    delete[] fftDims;
    delete[] inembed;
    delete[] onembed;

    is_fft_planned = true;
}

void GpuImage::Deallocate( ) {

    if ( is_host_memory_pinned ) {
        cudaErr(cudaHostUnregister(real_values));
        is_host_memory_pinned = false;
    }
    if ( is_in_memory_gpu ) {
        cudaErr(cudaFree(real_values_gpu));
        cudaErr(cudaFree(tmpVal));
        cudaErr(cudaFree(tmpValComplex));
        is_in_memory_gpu = false;
    }

    BufferDestroy( );

    if ( is_fft_planned ) {
        cudaErr(cufftDestroy(cuda_plan_inverse));
        cudaErr(cufftDestroy(cuda_plan_forward));
        is_fft_planned            = false;
        is_set_complexConjMulLoad = false;
    }

    //  if (is_cublas_loaded)
    //  {
    //    cudaErr(cublasDestroy(cublasHandle));
    //    is_cublas_loaded = false;
    //  }

    if ( is_allocated_mask_CSOS ) {
        mask_CSOS->Deallocate( );
        delete mask_CSOS;
    }

    if ( is_allocated_image_buffer ) {
        image_buffer->Deallocate( );
        delete image_buffer;
    }

    if ( is_allocated_clip_into_mask ) {
        cudaErr(cudaFree(clip_into_mask));
        delete clip_into_mask;
    }
}

void GpuImage::ConvertToHalfPrecision(bool deallocate_single_precision) {

    // FIXME when adding real space complex images.
    // FIXME should probably be called COPYorConvert
    MyAssertTrue(is_in_memory_gpu, "Image is in not on the GPU!");

    BufferInit(b_16f);

    precheck if ( is_in_real_space ) {
        ReturnLaunchParamters(dims, true);
        ConvertToHalfPrecisionKernelReal<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values_gpu, real_values_16f, this->dims);
    }
    else {
        ReturnLaunchParamters(dims, false);
        ConvertToHalfPrecisionKernelComplex<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_gpu, complex_values_16f, this->dims, this->physical_upper_bound_complex);
    }
    postcheck

            if ( deallocate_single_precision ) {
        cudaErr(cudaFree(real_values_gpu));
        is_in_memory_gpu = false;
    }
}

__global__ void ConvertToHalfPrecisionKernelReal(cufftReal* real_32f_values, __half* real_16f_values, int4 dims) {

    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.w && coords.y < dims.y && coords.z < dims.z ) {

        int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);

        real_16f_values[address] = __float2half_rn(real_32f_values[address]);
    }
}

__global__ void ConvertToHalfPrecisionKernelComplex(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex) {

    int3 coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y,
                            blockIdx.z);

    if ( coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z ) {

        int address = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);

        complex_16f_values[address] = __float22half2_rn(complex_32f_values[address]);
    }
}

void GpuImage::Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space) {

    MyAssertTrue(wanted_x_size > 0 && wanted_y_size > 0 && wanted_z_size > 0, "Bad dimensions: %i %i %i\n", wanted_x_size, wanted_y_size, wanted_z_size);

    // check to see if we need to do anything?

    if ( is_in_memory_gpu == true ) {
        is_in_real_space = should_be_in_real_space;
        if ( wanted_x_size == dims.x && wanted_y_size == dims.y && wanted_z_size == dims.z ) {
            // everything is already done..
            is_in_real_space = should_be_in_real_space;
            //			wxPrintf("returning\n");
            return;
        }
        else {
            Deallocate( );
        }
    }

    SetupInitialValues( );
    this->is_in_real_space = should_be_in_real_space;
    dims.x                 = wanted_x_size;
    dims.y                 = wanted_y_size;
    dims.z                 = wanted_z_size;

    // if we got here we need to do allocation..

    // first_x_dimension
    if ( IsEven(wanted_x_size) == true )
        real_memory_allocated = wanted_x_size / 2 + 1;
    else
        real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

    real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
    real_memory_allocated *= 2; // room for complex

    // TODO consider option to add host mem here. For now, just do gpu mem.
    //////	real_values = (float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
    //////	complex_values = (std::complex<float>*) real_values;  // Set the complex_values to point at the newly allocated real values;
    //	wxPrintf("\n\n\tAllocating mem\t\n\n");
    cudaErr(cudaMalloc(&real_values_gpu, real_memory_allocated * sizeof(cufftReal)));
    complex_values_gpu = (cufftComplex*)real_values_gpu;
    is_in_memory_gpu   = true;

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

    // Set other gpu vals

    is_host_memory_pinned    = false;
    is_meta_data_initialized = true;
}

void GpuImage::UpdateBoolsToDefault( ) {

    is_meta_data_initialized = false;

    is_in_memory                           = false;
    is_in_real_space                       = true;
    object_is_centred_in_box               = true;
    image_memory_should_not_be_deallocated = false;

    is_in_memory_gpu      = false;
    is_host_memory_pinned = false;

    // libraries
    is_fft_planned = false;
    //	is_cublas_loaded = false;
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

__global__ void ClipIntoRealSpaceKernel(cufftReal* real_values_gpu,
                                        cufftReal* other_image_real_values_gpu,
                                        int4       dims,
                                        int4       other_dims,
                                        int3       physical_address_of_box_center,
                                        int3       other_physical_address_of_box_center,
                                        int3       wanted_coordinate_of_box_center,
                                        float      wanted_padding_value);

__global__ void ClipIntoRealSpaceKernel(cufftReal* real_values_gpu,
                                        cufftReal* other_image_real_values_gpu,
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
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = wanted_padding_value;
        }
        else {
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] =
                    real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(coord, dims)];
        }
    }
}

__global__ void SetUniformComplexValueKernel(cufftComplex* complex_values_gpu, cufftComplex value, int4 dims);

// __global__ void SetUniformComplexValueKernel(cufftComplex* complex_values_gpu, cufftComplex value, int4 dims)
//   {
//       int3 kernel_coords = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
//       blockIdx.y * blockDim.y + threadIdx.y,
//       blockIdx.z);
//       long position1d = get_1d_position_from_3d_coords(kernel_coords, dims);
//       complex_values_gpu[position1d] = value;
//   }

__global__ void ClipIntoFourierSpaceKernel(cufftComplex* source_complex_values_gpu,
                                           cufftComplex* destination_complex_values_gpu,
                                           int4          source_dims,
                                           int4          destination_dims,
                                           int3          destination_image_physical_index_of_first_negative_frequency,
                                           int3          source_logical_lower_bound_complex,
                                           int3          source_logical_upper_bound_complex,
                                           int3          source_physical_upper_bound_complex,
                                           int3          destination_physical_upper_bound_complex,
                                           cufftComplex  out_of_bounds_value);

__global__ void ClipIntoFourierSpaceKernel(cufftComplex* source_complex_values_gpu,
                                           cufftComplex* destination_complex_values_gpu,
                                           int4          source_dims,
                                           int4          destination_dims,
                                           int3          destination_image_physical_index_of_first_negative_frequency,
                                           int3          source_logical_lower_bound_complex,
                                           int3          source_logical_upper_bound_complex,
                                           int3          source_physical_upper_bound_complex,
                                           int3          destination_physical_upper_bound_complex,
                                           cufftComplex  out_of_bounds_value) {
    int3 index_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z);

    if ( index_coord.y > destination_dims.y ||
         index_coord.z > destination_dims.z ||
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

    cufftComplex new_value;
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
        new_value        = source_complex_values_gpu[source_index];
    }

    destination_complex_values_gpu[destination_index] = new_value;
}

void GpuImage::Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value) {
    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(wanted_x_dimension != 0 && wanted_y_dimension != 0 && wanted_z_dimension != 0, "Resize dimension is zero");

    // this is never called for some reason?
    MyDebugAssertTrue(is_in_real_space, "Resize is only set up for real space on the gpu currently");

    if ( dims.x == wanted_x_dimension && dims.y == wanted_y_dimension && dims.z == wanted_z_dimension ) {
        wxPrintf("Wanted dimensions are the same as current dimensions.\n");
        return;
    }

    // wxPrintf("Init temp GPUImage.\n");
    GpuImage temp_image;

    // wxPrintf("Allocating memory.\n");
    temp_image.Allocate(wanted_x_dimension, wanted_y_dimension, wanted_z_dimension, is_in_real_space);

    // wxPrintf("Clipping temp image.\n");
    precheck if ( is_in_real_space ) {
        // wxPrintf("Clipping real space.\n");
        ClipIntoRealSpace(&temp_image, wanted_padding_value, false, 1.0, 0, 0, 0);
    }
    else {
        // wxPrintf("Clipping Fourier space.\n");
        ClipIntoFourierSpace(&temp_image, wanted_padding_value);
    }
    //ClipInto(&temp_image, wanted_padding_value, false, 1.0, 0, 0, 0);
    postcheck

            // wxPrintf("Consuming temp image\n");
            Consume(temp_image);
}

/*
 Overwrite current GpuImage with a new image, then deallocate new image.
*/
void GpuImage::Consume(GpuImage& temp_image) // copy the parameters then directly steal the memory of another image, leaving it an empty shell
{
    MyDebugAssertTrue(temp_image.is_in_memory_gpu, "Other image Memory not allocated");

    if ( this == &temp_image ) { // no need to consume, its the same image.
        return;
    }
    UpdateBoolsToDefault( );

    is_in_real_space = temp_image.is_in_real_space;

    number_of_real_space_pixels = temp_image.number_of_real_space_pixels;

    insert_into_which_reconstruction = temp_image.insert_into_which_reconstruction;

    object_is_centred_in_box = temp_image.object_is_centred_in_box;

    dims = temp_image.dims;

    // FIXME: temp for comp
    logical_x_dimension = temp_image.dims.x;
    logical_y_dimension = temp_image.dims.y;
    logical_z_dimension = temp_image.dims.z;

    pitch = temp_image.pitch;

    physical_upper_bound_complex = temp_image.physical_upper_bound_complex;

    physical_address_of_box_center = temp_image.physical_address_of_box_center;

    physical_index_of_first_negative_frequency = temp_image.physical_index_of_first_negative_frequency;

    fourier_voxel_size = temp_image.fourier_voxel_size;

    logical_upper_bound_complex = temp_image.logical_upper_bound_complex;

    logical_lower_bound_complex = temp_image.logical_lower_bound_complex;

    logical_upper_bound_real = temp_image.logical_upper_bound_real;

    logical_lower_bound_real = temp_image.logical_lower_bound_real;

    padding_jump_value = temp_image.padding_jump_value;

    ft_normalization_factor = temp_image.ft_normalization_factor;

    real_values    = NULL;
    complex_values = NULL;
    //   cudaErr(cudaFree(real_values_gpu));
    //   cudaErr(cudaFree(complex_values_gpu));
    real_memory_allocated = temp_image.real_memory_allocated;

    is_in_memory             = false; //temp_image.is_in_memory;
    is_meta_data_initialized = true;
    precheck
    Deallocate( );
    postcheck
            //Allocate(dims.x, dims.y, dims.z, is_in_real_space);

            // temp_image.printVal("Value 1:", 1);

            if ( is_in_real_space ) {
        precheck
        Zeros( );
        postcheck

                precheck
                        cudaErr(cudaMemcpyAsync(real_values_gpu, temp_image.real_values_gpu, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        postcheck
    }
    else {
        precheck
                // why doesn't this work?? ---> //cudaErr(cudaMemcpyAsync(complex_values_gpu, temp_image.complex_values_gpu, sizeof(cufftComplex) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                cudaErr(cudaMemcpyAsync(complex_values_gpu, temp_image.complex_values_gpu, sizeof(cufftReal) * real_memory_allocated, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        postcheck
    }

    precheck
            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    postcheck

            // this results in core dump...
            //wxPrintf(" After copy [0] = %s %s\n", std::to_string(real_values_gpu[0]), std::to_string(temp_image.real_values_gpu[0]));

            precheck
                    is_in_memory_gpu = true;
    postcheck

            // complex values are not dealt with

            precheck
                    cudaMallocManaged(&tmpVal, sizeof(float));
    cudaMallocManaged(&tmpValComplex, sizeof(double));
    postcheck

            precheck
                    temp_image.Deallocate( );
    postcheck
}

void GpuImage::ClipIntoRealSpace(GpuImage* other_image, float wanted_padding_value,
                                 bool fill_with_noise, float wanted_noise_sigma,
                                 int wanted_coordinate_of_box_center_x,
                                 int wanted_coordinate_of_box_center_y,
                                 int wanted_coordinate_of_box_center_z) {

    MyDebugAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyDebugAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
    MyDebugAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");

    int3 wanted_coordinate_of_box_center = make_int3(wanted_coordinate_of_box_center_x,
                                                     wanted_coordinate_of_box_center_y,
                                                     wanted_coordinate_of_box_center_z);

    other_image->is_in_real_space         = is_in_real_space;
    other_image->object_is_centred_in_box = object_is_centred_in_box;

    MyDebugAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

    ReturnLaunchParamters(other_image->dims, true);

    precheck
            ClipIntoRealKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(real_values_gpu,
                                                                                      other_image->real_values_gpu,
                                                                                      dims,
                                                                                      other_image->dims,
                                                                                      physical_address_of_box_center,
                                                                                      other_image->physical_address_of_box_center,
                                                                                      wanted_coordinate_of_box_center,
                                                                                      wanted_padding_value);
    postcheck
}

void GpuImage::ClipIntoFourierSpace(GpuImage* destination_image, float wanted_padding_value) {
    MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
    MyAssertTrue(destination_image->is_in_memory_gpu, "Destination image memory not allocated");
    MyAssertTrue(destination_image->object_is_centred_in_box && object_is_centred_in_box, "ClipInto assumes both images are centered at the moment.");
    MyAssertFalse(is_in_real_space && destination_image->is_in_real_space, "ClipIntoFourierSpace assumes both images are in fourier space");

    destination_image->object_is_centred_in_box = object_is_centred_in_box;

    ReturnLaunchParamters(destination_image->dims, false);

    precheck
            ClipIntoFourierSpaceKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(complex_values_gpu,
                                                                                              destination_image->complex_values_gpu,
                                                                                              dims,
                                                                                              destination_image->dims,
                                                                                              destination_image->physical_index_of_first_negative_frequency,
                                                                                              logical_lower_bound_complex,
                                                                                              logical_upper_bound_complex,
                                                                                              physical_upper_bound_complex,
                                                                                              destination_image->physical_upper_bound_complex,
                                                                                              make_cuComplex(wanted_padding_value, 0.0));

    postcheck
            cudaStreamSynchronize(cudaStreamPerThread);
}
