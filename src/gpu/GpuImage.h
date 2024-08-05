/*
 * GpuImage.h
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

#ifndef GPUIMAGE_H_
#define GPUIMAGE_H_

#include "../constants/constants.h"
// #include "TensorManager.h"
#include "../programs/refine3d/batched_search.h"
#include "core_extensions/data_views/pointers.h"

class BatchedSearch;

class GpuImage {

  private:
    using tmp_val_idx = cistem::gpu::tmp_val::Enum;

  public:
    GpuImage( );
    GpuImage(int wanted_x_size, int wanted_y_size, int wanted_z_size = 1, bool is_in_real_space = true, bool allocate_fp16_buffer = false);
    GpuImage(const GpuImage& other_gpu_image); // copy constructor
    GpuImage(Image& cpu_image);
    ~GpuImage( );

    GpuImage& operator=(const GpuImage& t);
    GpuImage& operator=(const GpuImage* t);

    // FIXME: move constructor?

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // START MEMBER VARIABLES FROM THE cpu IMAGE CLASS

    // TODO: These should mostly be made private since they are properties of the data and should not be modified unless a method modifies the data.
    int4 dims;
    // FIXME: Temporary for compatibility with the image class.
    int  logical_x_dimension, logical_y_dimension, logical_z_dimension;
    bool is_in_real_space; // !< Whether the image is in real or Fourier space
    bool object_is_centred_in_box; //!<  Whether the object or region of interest is near the center of the box (as opposed to near the corners and wrapped around). This refers to real space and is meaningless in Fourier space.
    bool is_fft_centered_in_box;
    int3 physical_upper_bound_complex;
    int3 physical_address_of_box_center;
    int3 physical_index_of_first_negative_frequency;
    int3 logical_upper_bound_complex;
    int3 logical_lower_bound_complex;
    int3 logical_upper_bound_real;
    int3 logical_lower_bound_real;

    int device_idx;
    int number_of_streaming_multiprocessors;
    int limit_SMs_by_threads;

    float3 fourier_voxel_size;

    int real_memory_allocated; // !<  Number of floats allocated in real space;
    int padding_jump_value; // !<  The FFTW padding value, if odd this is 2, if even it is 1.  It is used in loops etc over real space.
    int insert_into_which_reconstruction; // !<  Determines which reconstruction the image will be inserted into (for FSC calculation).

    int   number_of_real_space_pixels; // !<	Total number of pixels in real space
    float ft_normalization_factor; // !<	Normalization factor for the Fourier transform (1/sqrt(N), where N is the number of pixels in real space)
    // Arrays to hold voxel values

    // These arrays can be used to pass pointers for an image stack into a kernel. Initialize each type empty and explicitly
    // since the GpuImage class itself is not templated. TODO: use template parameters when re-writing
    DevicePointerArray<__half>  ptr_array_16f;
    DevicePointerArray<__half2> ptr_array_16fc;
    DevicePointerArray<float>   ptr_array_32f;
    DevicePointerArray<float2>  ptr_array_32fc;

    bool is_in_memory; // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array.
    bool image_memory_should_not_be_deallocated; // !< Don't deallocate the memory, generally should only be used when doing something funky with the pointers
    int  gpu_plan_id;

    // end  MEMBER VARIABLES FROM THE cpu IMAGE CLASS
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float*  real_values; // !<  Real array to hold values for REAL images.
    float2* complex_values; // !<  Complex array to hold values for COMP images.

    // To make it easier to switch between different types of FFT plans we have void pointers for them here
    void* position_space_ptr;
    void* momentum_space_ptr;

    // The half precision buffers may be used as fp16 or bfloat16 and it is up to the user to track what is what.
    // This of course assumes they have the same size, which they should.
    static constexpr size_t size_of_half = sizeof(__half);
    static_assert(size_of_half == sizeof(nv_bfloat16), "it is assumed sizeof(fp16) == sizeof(bfloat16)");
    static_assert(size_of_half * 2 == sizeof(nv_bfloat162), "it is assumed sizeof(fp16) == sizeof(bfloat16)");
    static_assert(size_of_half == sizeof(half_float::half), "GPU and CPU half precision types must be the same size");

    void* real_values_16f;
    void* complex_values_16f;
    void* ctf_buffer_16f;
    void* ctf_complex_buffer_16f;

    __half*  real_values_fp16;
    __half2* complex_values_fp16;
    __half*  ctf_buffer_fp16;
    __half2* ctf_complex_buffer_fp16;

    nv_bfloat16*  real_values_bf16;
    nv_bfloat162* complex_values_bf16;
    nv_bfloat16*  ctf_buffer_bf16;
    nv_bfloat162* ctf_complex_buffer_bf16;

    // We want to be able to re-use the texture object, so only set it up once.
    cudaTextureObject_t tex_real;

    cudaTextureObject_t tex_imag;
    cudaArray*          cuArray_real = 0;
    cudaArray*          cuArray_imag = 0;

    bool is_allocated_texture_cache;

    // TODO: This is currently unused. What was the thinking?
    enum ImageType : size_t { real16f    = sizeof(__half),
                              complex16f = sizeof(__half2),
                              real32f    = sizeof(float),
                              complex32f = sizeof(float2),
                              real64f    = sizeof(double),
                              complex64f = sizeof(double2) };

    ImageType img_type;

    bool is_in_memory_gpu; // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array. Default = .FALSE.

    size_t pitch;

    dim3 threadsPerBlock;
    dim3 gridDims;

    bool is_meta_data_initialized;

    double* tmpValComplex;
    bool    is_in_memory_managed_tmp_vals;

    ////////////////////////////////////////////////////////

    cudaEvent_t npp_calc_event;
    cudaEvent_t block_host_event;
    cudaEvent_t return_sum_of_squares_event;
    bool        is_npp_calc_event_initialized;
    bool        is_block_host_event_initialized;
    bool        is_return_sum_of_squares_event_initialized;
    //	cublasHandle_t cublasHandle;

    cufftHandle cuda_plan_forward;
    cufftHandle cuda_plan_inverse;
    size_t      cuda_plan_worksize_forward;
    size_t      cuda_plan_worksize_inverse;

    long long cufft_batch_size;

    //Stream for asynchronous command execution
    cudaStream_t     calcStream;
    cudaStream_t     copyStream;
    NppStreamContext nppStream;

    void PrintNppStreamContext( );

    //	bool is_cublas_loaded;
    bool is_npp_loaded;
    //	cublasStatus_t cublas_stat;
    NppStatus npp_stat;

    // For the full image set width/height, otherwise set on function call.
    NppiSize npp_ROI;
    NppiSize npp_ROI_real_space;
    NppiSize npp_ROI_fourier_space;
    NppiSize npp_ROI_fourier_with_real_functor;

    ////////////////////////////////////////////////////////////////////////
    ///// Methods that should behave as their counterpart in the Image class
    ///// have /**CPU_eq**/
    ////////////////////////////////////////////////////////////////////////

    void QuickAndDirtyWriteSlices(std::string filename, int first_slice, int last_slice); /**CPU_eq**/

    void QuickAndDirtyWriteSlice(std::string filename, int first_slice) { QuickAndDirtyWriteSlices(filename, first_slice, first_slice); }; /**CPU_eq**/

    void ZeroCentralPixel( ); /**CPU_eq**/
    template <typename StorageTypeBase = float>
    void PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift); /**CPU_eq**/

    void MultiplyByConstant(float scale_factor); /**CPU_eq**/
    void MultiplyByConstant16f(const float scale_factor, int n_slices = 1);
    void MultiplyByConstant16f(__half* input_ptr, const float scale_factor, int n_slices = 1);

    void SetToConstant(float val);
    void SetToConstant(Npp32fc val);
    void Conj( ); // FIXME
    void MultiplyPixelWise(const float& other_array, const int other_array_size); // dose filter for example
    void MultiplyPixelWise(GpuImage& other_image); /**CPU_eq**/
    void MultiplyPixelWise(GpuImage& other_image, GpuImage& output_image); /**CPU_eq**/

    template <typename StorageTypeBase = float>
    void MultiplyPixelWiseComplexConjugate(GpuImage& other_image, GpuImage& result_image, int phase_multiplier);

    template <typename StorageTypeBase = float>
    void MultiplyPixelWiseComplexConjugate(GpuImage& other_image, GpuImage& result_image) { MultiplyPixelWiseComplexConjugate<StorageTypeBase>(other_image, result_image, 0); };

    template <typename StorageTypeBase = float>
    void SwapRealSpaceQuadrants( ); /**CPU_eq**/
    void ClipInto(GpuImage* other_image,
                  float     wanted_padding_value              = 0.f, /**CPU_eq**/
                  bool      fill_with_noise                   = false,
                  float     wanted_noise_sigma                = 1.f,
                  int       wanted_coordinate_of_box_center_x = 0,
                  int       wanted_coordinate_of_box_center_y = 0,
                  int       wanted_coordinate_of_box_center_z = 0);

    void ClipIntoFourierSpace(GpuImage* destination_image, float wanted_padding_value, bool zero_central_pixel = false, bool use_fp16 = false);

    void ClipIntoReturnMask(GpuImage* other_image);

    // Used with explicit specializtion
    template <class InputType, class OutputType>
    void _ForwardFFT( );

    template <class InputType, class OutputType>
    void _BackwardFFT( );

    void ForwardFFT(bool should_scale = true); /**CPU_eq**/
    void ForwardFFTBatched(bool should_scale = true);

    void BackwardFFT( ); /**CPU_eq**/
    void BackwardFFTBatched(int wanted_batch_size = 0); // if zero, defaults to dims.z

    void ForwardFFTAndClipInto(GpuImage& image_to_insert, bool should_scale);
    template <typename LoadType, typename StoreType = __half>
    void BackwardFFTAfterComplexConjMul(LoadType* image_to_multiply, bool load_half_precision, StoreType* output_ptr = nullptr);

    void Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value, bool zero_central_pixel = false);
    void Consume(GpuImage* other_image);
    void CopyCpuImageMetaData(Image& cpu_image);
    bool InitializeBasedOnCpuImage(Image& cpu_image, bool pin_host_memory, bool allocate_real_values);
    void CopyGpuImageMetaData(const GpuImage* other_image);
    void CopyLoopingAndAddressingFrom(GpuImage* other_image);

    void  L2Norm( );
    float ReturnSumOfSquares( );

    void NormalizeRealSpaceStdDeviation(float additional_scalar, float pre_calculated_avg, float average_on_edge);
    void NormalizeRealSpaceStdDeviationAndCastToFp16(float additional_scalar, float pre_calculated_avg, float average_on_edge);

    float ReturnAverageOfRealValuesOnEdges( );
    void  Deallocate( );

    void CopyFrom(GpuImage* other_image);

    template <typename StorageTypeBase>
    void CopyDataFrom(GpuImage& other_image);
    void CopyFP32toFP16buffer(const float* __restrict__ real_32f_values, __half* __restrict__real_16f_values, int n_elements);
    void CopyFP32toFP16buffer(const float2* __restrict__ complex_32f_values, __half2* __restrict__ complex_16f_values, int n_elements);

    void CopyFP32toFP16buffer(bool deallocate_single_precision = true);
    void CopyFP16buffertoFP32(bool deallocate_half_precision = true);
    void CopyFP32toFP16bufferAndScale(float scalar);

    void AllocateTmpVarsAndEvents( );
    // If we allocate the fp16 buffer, we will not allocate fp32, will leave it alone if the same size, and will remove it if different.
    bool Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space, bool allocate_fp16_buffer = false);

    bool Allocate(int wanted_x_size, int wanted_y_size, bool should_be_in_real_space, bool allocate_fp16_buffer = false) {
        return Allocate(wanted_x_size, wanted_y_size, 1, should_be_in_real_space, allocate_fp16_buffer);
    };

    // Combines this and UpdatePhysicalAddressOfBoxCenter and SetLogicalDimensions
    void UpdateLoopingAndAddressing(int wanted_x_size, int wanted_y_size, int wanted_z_size);

    ////////////////////////////////////////////////////////////////////////
    ///// Methods that do not have a counterpart in the image class
    ////////////////////////////////////////////////////////////////////////

    void CopyHostToDevice(Image& host_image, bool should_block_until_complete = false, bool pin_host_memory = true, cudaStream_t stream = cudaStreamPerThread);

    void CopyHostToDeviceAndSynchronize(Image& host_image, bool pin_host_memory = true) { CopyHostToDevice(host_image, true, pin_host_memory); };

    void CopyHostToDeviceTextureComplex3d(Image& host_image);
    void CopyHostToDeviceTextureComplex2d(Image& host_image);

    void CopyHostToDevice16f(Image& host_image, bool should_block_until_finished = false); // CTF images in the ImageClass are stored as complex, even if they only have a real part. This is a waste of memory bandwidth on the GPU
    void CopyDeviceToHostAndSynchronize(Image& cpu_image, bool unpin_host_memory = true);
    void CopyDeviceToHost(Image& host_image, bool unpin_host_memory = true);

    void  CopyDeviceToNewHost(Image& cpu_image, bool should_block_until_complete, bool free_gpu_memory, bool unpin_host_memory = true);
    Image CopyDeviceToNewHost(bool should_block_until_complete, bool free_gpu_memory, bool unpin_host_memory = true);

    // Synchronize the full stream.
    void Record( );
    void RecordBlocking( );
    void Wait( );
    void WaitBlocking( );
    void RecordAndWait( );
    // Maximum intensity projection
    void MipPixelWise(GpuImage& other_image);
    void MipPixelWise(GpuImage& other_image, GpuImage& psi, GpuImage& phi, GpuImage& theta,
                      float c_psi, float c_phi, float c_theta);
    void MipPixelWise(GpuImage& other_image, GpuImage& psi, GpuImage& phi, GpuImage& theta, GpuImage& defocus, GpuImage& pixel,
                      float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel);

    // FIXME: These are added for the unblur refinement but are untested.
    template <typename StorageTypeBase = float>
    void ApplyBFactor(float bfactor);
    template <typename StorageTypeBase = float>
    void ApplyBFactor(float bfactor, float vertical_mask_size, float horizontal_mask_size); // Specialization for unblur refinement, merges MaskCentralCross()
    void Whiten(float resolution_limit = 1.f);

    float GetWeightedCorrelationWithImage(GpuImage& projection_image, GpuImage& cross_terms, GpuImage& image_PS, GpuImage& projection_PS, float filter_radius_low_sq, float filter_radius_high_sq, float signed_CC_limit);

    inline void MaskCentralCross(float vertical_mask_size, float horizontal_mask_size) { return; }; // noop

    void CalculateCrossCorrelationImageWith(GpuImage* other_image);
    Peak FindPeakWithParabolaFit(float inner_radius_for_peak_search, float outer_radius_for_peak_search);
    Peak FindPeakAtOriginFast2D(int wanted_max_pix_x, int wanted_max_pix_y, IntegerPeak* pinned_host_buffer, IntegerPeak* device_buffer, int wanted_batch_size, bool load_half_precision = false);
    Peak FindPeakAtOriginFast2D(BatchedSearch* batch, bool load_half_precision = false);
    Peak FindPeakAtCenterFast2d(const BatchedSearch& batch, bool load_half_precision = false);

    bool Init(Image& cpu_image, bool pin_host_memory = true, bool allocate_real_values = true);
    void SetupInitialValues( );
    void UpdateBoolsToDefault( );
    void SetCufftPlan(cistem::fft_type::Enum plan_type, void* input_buffer, void* output_buffer);

    cistem::fft_type::Enum set_plan_type;
    long long              set_batch_size;
    bool                   is_batched_transform;

    template <int ntds_x = 32, int ntds_y = 32>
    __inline__ void ReturnLaunchParameters(int4 input_dims, bool real_space) {
        static_assert(ntds_x % cistem::gpu::warp_size == 0);
        static_assert(ntds_x * ntds_y <= cistem::gpu::max_threads_per_block);
        int div = 1;
        if ( ! real_space )
            div++;

        threadsPerBlock = dim3(ntds_x, ntds_y, 1);
        gridDims        = dim3((input_dims.w / div + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (input_dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                               input_dims.z);
    };

    __inline__ void ReturnLaunchParameters1d_X(const int4 input_dims, const bool real_space) {
        int div = 1;
        if ( ! real_space )
            div++;

        using namespace cistem::gpu;
        // Note: that second set of parens changes the division!
        threadsPerBlock = dim3(std::max(min_threads_per_block, std::min(max_threads_per_block, warp_size * ((input_dims.w / div + warp_size - 1) / warp_size))), 1, 1);
        gridDims        = dim3((input_dims.w / div + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (input_dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                               input_dims.z);
    };

    __inline__ void ReturnLaunchParameters1d_X_strided_Y(const int4 input_dims, const bool real_space, const int stride_y) {
        int div = 1;
        if ( ! real_space )
            div++;

        using namespace cistem::gpu;
        // Note: that second set of parens changes the division!
        threadsPerBlock = dim3(std::max(min_threads_per_block, std::min(max_threads_per_block / stride_y, warp_size * ((input_dims.w / div + warp_size - 1) / warp_size))), stride_y, 1);
        gridDims        = dim3((input_dims.w / div + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (input_dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                               input_dims.z);
    };

    __inline__ void ReturnLaunchParametersLimitSMs(float N, int M) {
        // This should only be called for kernels with grid stride loops setup. The idea
        // is to limit the number of SMs available for some kernels so that other threads on the device can run in parallel.
        // limit_SMs_by_threads is default 1, so this must be set prior to this call.
        threadsPerBlock = dim3(M, 1, 1);
        gridDims        = dim3(myroundint(N * number_of_streaming_multiprocessors), 1, 1);
    };

    void UpdateFlagsFromHostImage(Image& host_image);
    void UpdateFlagsFromDeviceImage(Image& host_image);

    void printVal(std::string msg, int idx);

    template <typename StorageBaseType = float>
    bool HasSameDimensionsAs(GpuImage& other_image);

    template <typename StorageBaseType = float>
    bool HasSameDimensionsAs(GpuImage* other_image) { return HasSameDimensionsAs<StorageBaseType>(*other_image); };

    bool HasSameDimensionsAs(Image* other_image);

    bool HasSameDimensionsAs(Image& other_image) { return HasSameDimensionsAs(&other_image); };

    template <typename StorageTypeBase = float>
    void Zeros( );

    void ExtractSlice(GpuImage* volume_to_extract_from, AnglesAndShifts& angles_and_shifts, float pixel_size, float resolution_limit = 1.f, bool apply_resolution_limit = true, bool whiten_spectrum = false);

    void ExtractSliceShiftAndCtf(GpuImage* volume_to_extract_from, GpuImage* ctf_image, AnglesAndShifts& angles_and_shifts, float pixel_size, float real_space_binning_factor, float resolution_limit, bool apply_resolution_limit,
                                 bool swap_quadrants, bool apply_shifts, bool apply_ctf, bool absolute_ctf, bool zero_central_pixel = false, cudaStream_t stream = cudaStreamPerThread);

    void Abs( );
    void AbsDiff(GpuImage& other_image); // inplace
    void AbsDiff(GpuImage& other_image, GpuImage& output_image);
    void SquareRealValues( );
    void SquareRootRealValues( );
    void LogarithmRealValues( );
    void ExponentiateRealValues( );
    void AddConstant(const float add_val);
    void AddConstant(const Npp32fc add_val);

    void AddImage(GpuImage& other_image);

    void AddImage(GpuImage* other_image) { AddImage(*other_image); }; // for compatibility with Image class

    template <typename StorageTypeBase>
    void AddImageStack(std::vector<GpuImage>& input_stack);
    template <typename StorageTypeBase>
    void AddImageStack(std::vector<GpuImage>& input_stack, GpuImage& output_image);

    template <typename StorageTypeBase = float>
    void SubtractImage(GpuImage& other_image);

    template <typename StorageTypeBase = float>
    void SubtractImage(GpuImage* other_image) { SubtractImage<StorageTypeBase>(*other_image); }; // for compatibility with Image class

    void AddSquaredImage(GpuImage& other_image);

    void ReplaceOutliersWithMean(float mean, float stdDev, float maximum_n_sigmas);
    void ReplaceOutliersWithMean(float maximum_n_sigmas);

    // Statitical Methods
    float ReturnSumOfRealValues( );
    // float3    ReturnSumOfRealValues3Channel( );
    NppiPoint min_idx;
    NppiPoint max_idx;
    float     min_value;
    float     max_value;
    float     img_mean;
    float     img_stdDev;
    Npp64f    npp_mean;
    Npp64f    npp_stdDev;
    int       number_of_pixels_in_range;
    void      Min( );
    void      MinAndCoords( );
    void      Max( );
    void      MaxAndCoords( );
    void      MinMax( );
    void      MinMaxAndCoords( );
    void      Mean( );
    void      MeanStdDev( );
    void      AverageError(const GpuImage& other_image); // TODO add me
    void      AverageRelativeError(const GpuImage& other_image); // TODO addme
    void      CountInRange(float lower_bound, float upper_bound);
    void      HistogramEvenBins( ); // TODO add me
    void      HistogramDefinedBins( ); // TODO add me

    // TODO
    /*

  Mean, Mean_StdDev 
  */

    ////////////////////////////////////////////////////////////////////////
    ///// Methods for creating or storing masks used for otherwise slow looping operations
    ////////////////////////////////////////////////////////////////////////

    enum BufferType : int { b_image,
                            b_sum,
                            b_min,
                            b_minIDX,
                            b_max,
                            b_maxIDX,
                            b_minmax,
                            b_minmaxIDX,
                            b_mean,
                            b_meanstddev,
                            b_countinrange,
                            b_histogram,
                            b_16f,
                            b_ctf_16f,
                            b_l2norm,
                            b_dotproduct,
                            b_clip_into_mask,
                            b_weighted_correlation };

    //  void CublasInit();
    void NppInit( );
    void BufferInit(BufferType bt, int n_elements = 0);
    void BufferDestroy( );
    void FreeFFTPlan( );

    // Real buffer = size real_values
    GpuImage* image_buffer;
    bool      is_allocated_image_buffer;

    // Npp specific buffers;
    Npp8u* sum_buffer;
    bool   is_allocated_sum_buffer;
    Npp8u* min_buffer;
    bool   is_allocated_min_buffer;
    Npp8u* minIDX_buffer;
    bool   is_allocated_minIDX_buffer;
    Npp8u* max_buffer;
    bool   is_allocated_max_buffer;
    Npp8u* maxIDX_buffer;
    bool   is_allocated_maxIDX_buffer;
    Npp8u* minmax_buffer;
    bool   is_allocated_minmax_buffer;
    Npp8u* minmaxIDX_buffer;
    bool   is_allocated_minmaxIDX_buffer;
    Npp8u* mean_buffer;
    bool   is_allocated_mean_buffer;
    Npp8u* meanstddev_buffer;
    bool   is_allocated_meanstddev_buffer;
    Npp8u* countinrange_buffer;
    bool   is_allocated_countinrange_buffer;
    Npp8u* l2norm_buffer;
    bool   is_allocated_l2norm_buffer;
    Npp8u* dotproduct_buffer;
    bool   is_allocated_dotproduct_buffer;
    bool   is_allocated_16f_buffer;
    bool   is_allocated_ctf_16f_buffer;
    int*   clip_into_mask;
    bool   is_allocated_clip_into_mask;
    bool   is_set_realLoadAndClipInto;
    float* weighted_correlation_buffer;
    bool   is_allocated_weighted_correlation_buffer;
    int    weighted_correlation_buffer_size;

    GpuImage* mask_CSOS;
    bool      is_allocated_mask_CSOS;
    float     ReturnSumSquareModulusComplexValues( );

    // Callback related parameters
    bool is_set_convertInputf16Tof32;
    bool is_set_scaleFFTAndStore;
    bool is_set_complexConjMulLoad;

    /*template void d_MultiplyByScalar<T>(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch);*/
};

#endif /* GPUIMAGE_H_ */
