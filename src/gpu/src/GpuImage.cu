/*
 * GpuImage.cpp
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

#include "gpu_core_headers.h"


//#include <Exceptions.h>
//#include <helper_string.h>
// Kernel declarations



__global__ void ConvertToHalfPrecisionKernel(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex);

__global__ void MultiplyPixelWiseComplexConjugateKernel(cufftComplex* ref_complex_values, cufftComplex* img_complex_values, int4 dims, int3 physical_upper_bound_complex);
__global__ void MipPixelWiseKernel(cufftReal *mip, const cufftReal *correlation_output, const int4 dims);
__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *other_image, cufftReal *psi, cufftReal *phi, cufftReal *theta,
                                   int4 dims,float c_psi, float c_phi, float c_theta);
__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *other_image, cufftReal *psi, cufftReal *phi, cufftReal *theta, cufftReal *defocus, cufftReal *pixel, const int4 dims,
                                   float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel);



__global__ void PhaseShiftKernel(cufftComplex* d_input, 
                                 int4 dims, float3 shifts, 
                                 int3 physical_address_of_box_center, 
                                 int3 physical_index_of_first_negative_frequency,
                                 int3 physical_upper_bound_complex);

__global__ void ClipIntoRealKernel(cufftReal* real_values_gpu,
                                   cufftReal* other_image_real_values_gpu,
                                   int4 dims, 
                                   int4 other_dims,
                                   int3 physical_address_of_box_center, 
                                   int3 other_physical_address_of_box_center, 
                                   int3 wanted_coordinate_of_box_center, 
                                   float wanted_padding_value);

// cuFFT callbacks
__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

__device__ cufftReal CB_ConvertInputf16Tof32(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{

	const __half element = ((__half*)dataIn)[offset];
	return (cufftReal)(__half2float(element));
}
__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr);

__device__ void CB_scaleFFTAndStore(void* dataOut, size_t offset, cufftComplex element, void* callerInfo, void* sharedPtr)
{
	float scale_factor = *(float *)callerInfo;

	((cufftComplex*)dataOut)[offset] = (cufftComplex)ComplexScale(element, scale_factor);
}
//__device__ cufftCallbackLoadR d_ConvertInputf16Tof32Ptr = CB_ConvertInputf16Tof32;
__device__ cufftCallbackStoreC d_scaleFFTAndStorePtr = CB_scaleFFTAndStore;



//static __device__ cufftComplex CB_complexConjMulLoad(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

template<typename T> struct CB_complexConjMulLoad_params
{
	T* 		target;
	float 	scale;

} ;


static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);
static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
 {
	CB_complexConjMulLoad_params<cufftComplex>* my_params = (CB_complexConjMulLoad_params<cufftComplex> *)callerInfo;
	return (cufftComplex)ComplexConjMulAndScale(my_params->target[offset],((Complex *)dataIn)[offset],my_params->scale);
 }
static __device__ cufftComplex CB_complexConjMulLoad_16f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
 {
	CB_complexConjMulLoad_params<__half2>* my_params = (CB_complexConjMulLoad_params<__half2> *)callerInfo;
	return (cufftComplex)ComplexConjMulAndScale((cufftComplex)__half22float2(my_params->target[offset]),((Complex *)dataIn)[offset],my_params->scale);
 }

 __device__ cufftCallbackLoadC d_complexConjMulLoad_32f = CB_complexConjMulLoad_32f;
 __device__ cufftCallbackLoadC d_complexConjMulLoad_16f = CB_complexConjMulLoad_16f;

typedef struct _CB_realLoadAndClipInto_params
{
	cufftReal*	target;
	int i_lower_bound;
	int j_lower_bound;
	int i_upper_bound;
	int j_upper_bound;
	int nx_small;
	int nx_large;

} CB_realLoadAndClipInto_params;


static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{
 CB_realLoadAndClipInto_params* my_params = (CB_realLoadAndClipInto_params *)callerInfo;



 int i = offset % my_params->nx_large;
 if (i < my_params->i_lower_bound || i > my_params->i_upper_bound)
 {
	 return (cufftReal)0.0f;
 }

 int j = (offset-i) / my_params->nx_large;

 if (j < my_params->j_lower_bound || j > my_params->j_upper_bound)
 {
	 return (cufftReal)0.0f;
 }


 // If we are here, we are in bounds.
 int i_small = i - my_params->i_lower_bound;
 int j_small = j - my_params->j_lower_bound ;
 return my_params->target[i_small + j_small*my_params->nx_small];


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
d_Return3DPhaseFromIndividualDimensions( float phase_x, float phase_y, float phase_z, float2 &angles);

 __device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, int4 img_dims);

 __device__ __forceinline__ int
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_dims, int3 physical_upper_bound_complex);

 __inline__ int
ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                              int logical_y_dimension,
                                              int physical_index_of_first_negative_frequency_y)
{
    if (physical_index >= physical_index_of_first_negative_frequency_y)
    {
    	 return physical_index - logical_y_dimension;
    }
    else return physical_index;
};
__inline__ int
ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,
                                              int logical_z_dimension,
                                              int physical_index_of_first_negative_frequency_z)
{
    if (physical_index >= physical_index_of_first_negative_frequency_z)
    {
    	 return physical_index - logical_z_dimension;
    }
    else return physical_index;
};

////////////////// For thrust
typedef struct 
{
  __host__ __device__
    float operator()(const float& x) const {
      return x * x;
    }
} square;
////////////////////////

GpuImage::GpuImage()
{ 
  SetupInitialValues();
}


GpuImage::GpuImage(Image &cpu_image) 
{

  SetupInitialValues();
  Init(cpu_image);
	
}

GpuImage::GpuImage( const GpuImage &other_gpu_image) // copy constructor
{

	SetupInitialValues();
	*this = other_gpu_image;
}

GpuImage & GpuImage::operator = (const GpuImage &other_gpu_image)
{
	*this = &other_gpu_image;
	return *this;
}


GpuImage & GpuImage::operator = (const GpuImage *other_gpu_image)
{
	// Check for self assignment
	if(this != other_gpu_image)
	{

		MyAssertTrue(other_gpu_image->is_in_memory_gpu, "Other image Memory not allocated");

		if (is_in_memory_gpu == true)
		{

			if (dims.x != other_gpu_image->dims.x || dims.y != other_gpu_image->dims.y || dims.z != other_gpu_image->dims.z)
			{
				Deallocate();
				Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
			}
		}
		else
		{
			Allocate(other_gpu_image->dims.x, other_gpu_image->dims.y, other_gpu_image->dims.z, other_gpu_image->is_in_real_space);
		}

		// by here the memory allocation should be ok..

		is_in_real_space = other_gpu_image->is_in_real_space;
		object_is_centred_in_box = other_gpu_image->object_is_centred_in_box;

		pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
		checkCudaErrors(cudaMemcpyAsync(real_values_gpu,other_gpu_image->real_values_gpu,sizeof(cufftReal)*real_memory_allocated,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
		checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	}

   return *this;
}

GpuImage::~GpuImage() 
{
//  Deallocate();
}


void GpuImage::Init(Image &cpu_image)
{
	CopyFromCpuImage(cpu_image);
}

void GpuImage::SetupInitialValues()
{

	dims = make_int4(0, 0, 0, 0); pitch = 0;
	physical_upper_bound_complex = make_int3(0, 0, 0);
	physical_address_of_box_center = make_int3(0, 0, 0);
	physical_index_of_first_negative_frequency = make_int3(0, 0, 0);
	logical_upper_bound_complex = make_int3(0, 0, 0);
	logical_lower_bound_complex = make_int3(0, 0, 0);
	logical_upper_bound_real = make_int3(0, 0, 0);
	logical_lower_bound_real = make_int3(0, 0, 0);

	fourier_voxel_size = make_float3(0.0f, 0.0f, 0.0f);


	number_of_real_space_pixels = 0;


	real_values = NULL;
	complex_values = NULL;

	real_memory_allocated = 0;


//	plan_fwd = NULL;
//	plan_bwd = NULL;
//
//	planned = false;

	padding_jump_value = 0;

	ft_normalization_factor = 0;

	real_values_gpu = NULL;									// !<  Real array to hold values for REAL images.
	complex_values_gpu = NULL;								// !<  Complex array to hold values for COMP images.


	gpu_plan_id = -1;

	insert_into_which_reconstruction = 0;
	hostImage = NULL;

	UpdateBoolsToDefault();

}

void GpuImage::CopyFromCpuImage(Image &cpu_image) 
{

	UpdateBoolsToDefault();

	dims = make_int4(cpu_image.logical_x_dimension,
				   cpu_image.logical_y_dimension,
				   cpu_image.logical_z_dimension,
				   cpu_image.logical_x_dimension + cpu_image.padding_jump_value);

	pitch = dims.w * sizeof(float);

	physical_upper_bound_complex = make_int3(cpu_image.physical_upper_bound_complex_x,
										   cpu_image.physical_upper_bound_complex_y,
										   cpu_image.physical_upper_bound_complex_z);

	physical_address_of_box_center   = make_int3(cpu_image.physical_address_of_box_center_x,
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


	is_in_real_space = cpu_image.is_in_real_space;
	number_of_real_space_pixels = cpu_image.number_of_real_space_pixels;
	object_is_centred_in_box = cpu_image.object_is_centred_in_box;

	fourier_voxel_size = make_float3(cpu_image.fourier_voxel_size_x,
								   cpu_image.fourier_voxel_size_y,
								   cpu_image.fourier_voxel_size_z);


	insert_into_which_reconstruction = cpu_image.insert_into_which_reconstruction;
	real_values = cpu_image.real_values;
	complex_values = cpu_image.complex_values;

	is_in_memory = cpu_image.is_in_memory;

	padding_jump_value = cpu_image.padding_jump_value;
	image_memory_should_not_be_deallocated = cpu_image.image_memory_should_not_be_deallocated; // TODO what is this for?

	real_values_gpu = NULL;									// !<  Real array to hold values for REAL images.
	complex_values_gpu = NULL;								// !<  Complex array to hold values for COMP images.
	is_in_memory_gpu = false;
	real_memory_allocated =  cpu_image.real_memory_allocated;


	ft_normalization_factor = cpu_image.ft_normalization_factor;

	// FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning
	cudaHostRegister(real_values, sizeof(float)*real_memory_allocated, cudaHostRegisterDefault);
	is_host_memory_pinned = true;
	is_meta_data_initialized = true;
	cudaHostGetDevicePointer( &pinnedPtr, real_values, 0);

	cudaMallocManaged(&tmpVal, sizeof(cufftReal));
	cudaMallocManaged(&tmpValComplex, sizeof(cufftComplex));

	hostImage = &cpu_image;
 
}

void GpuImage::UpdateCpuFlags() 
{

  // Call after re-copying. The main image properites are all assumed to be static.
  is_in_real_space = hostImage->is_in_real_space;
  object_is_centred_in_box = hostImage->object_is_centred_in_box;

}

void GpuImage::printVal(std::string msg, int idx)
{

  float h_printVal = -9999.0f;

  checkCudaErrors(cudaMemcpy(&h_printVal, &real_values_gpu[idx], sizeof(float), cudaMemcpyDeviceToHost));
  cudaStreamSynchronize(cudaStreamPerThread);
  wxPrintf("%s %6.6e\n", msg, h_printVal);

};

bool GpuImage::HasSameDimensionsAs(GpuImage *other_image)
{
	// Functions that call this method also assume these asserts are being called here, so do not remove.
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
	// end of dependent asserts.

	if (dims.x == other_image->dims.x && dims.y == other_image->dims.y && dims.z == other_image->dims.z) return true;
	else return false;
}

void GpuImage::MultiplyPixelWiseComplexConjugate(GpuImage &other_image)
{
	// FIXME when adding real space complex images
	MyAssertFalse( is_in_real_space, "Image is in real space");
	MyAssertFalse( other_image.is_in_real_space, "Other image is in real space");
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");


	//  NppInit();
	//  Conj();
	//  npp_stat = nppiMul_32sc_C1IRSfs((const Npp32sc *)complex_values_gpu, 1, (Npp32sc*)other_image.complex_values_gpu, 1, npp_ROI_complex, 0);

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	ReturnLaunchParamters(dims, false);
	MultiplyPixelWiseComplexConjugateKernel<< <gridDims, threadsPerBlock,0, cudaStreamPerThread>> > (complex_values_gpu, other_image.complex_values_gpu,this->dims, this->physical_upper_bound_complex);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}


__global__ void ReturnSumOfRealValuesOnEdgesKernel(cufftReal *real_values_gpu, int4 dims, int padding_jump_value, float* returnValue);

float GpuImage::ReturnAverageOfRealValuesOnEdges()
{
	// FIXME to use a masked routing, this is slow af
	MyAssertTrue(is_in_memory, "Memory not allocated");
	MyAssertTrue(dims.z == 1, "ReturnAverageOfRealValuesOnEdges only implemented in 2d");

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	*tmpVal = 5.0f;
	ReturnSumOfRealValuesOnEdgesKernel<< <1, 1, 0, cudaStreamPerThread>> >(real_values_gpu, dims, padding_jump_value, tmpVal);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

	// Need to wait on the return value
	checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));


  return *tmpVal;
}

__global__ void ReturnSumOfRealValuesOnEdgesKernel(cufftReal *real_values_gpu, int4 dims, int padding_jump_value, float* returnValue)
{

	int pixel_counter;
	int line_counter;
	int plane_counter;

	double sum = 0.0;
	int number_of_pixels = 0;
	int address = 0;


		// Two-dimensional image
		// First line
		for (pixel_counter=0; pixel_counter < dims.x; pixel_counter++)
		{
			sum += real_values_gpu[address];
			address++;
		}
		number_of_pixels += dims.x;
		address += padding_jump_value;

		// Other lines
		for (line_counter=1; line_counter < dims.y-1; line_counter++)
		{
			sum += real_values_gpu[address];
			address += dims.x-1;
			sum += real_values_gpu[address];
			address += padding_jump_value + 1;
			number_of_pixels += 2;
		}

		// Last line
		for (pixel_counter=0; pixel_counter < dims.x; pixel_counter++)
		{
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

void GpuImage::NppInit()
{
  if ( ! is_npp_loaded )
  {

	int sharedMem;
	nppStream.hStream = cudaStreamPerThread; // FIXME to use member stream
	cudaGetDevice(&nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMultiProcessorCount,cudaDevAttrMultiProcessorCount,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerMultiProcessor,cudaDevAttrMaxThreadsPerMultiProcessor,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerBlock,cudaDevAttrMaxThreadsPerBlock,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nMaxThreadsPerMultiProcessor,cudaDevAttrMaxThreadsPerMultiProcessor,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&sharedMem,cudaDevAttrMaxSharedMemoryPerBlock,nppStream.nCudaDeviceId);
	nppStream.nSharedMemPerBlock = (size_t)sharedMem;
	cudaDeviceGetAttribute(&nppStream.nCudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMajor,nppStream.nCudaDeviceId);
	cudaDeviceGetAttribute(&nppStream.nCudaDevAttrComputeCapabilityMinor,cudaDevAttrComputeCapabilityMinor,nppStream.nCudaDeviceId);

//    nppSetStream(cudaStreamPerThread);
    npp_ROI.width  = dims.x;
    npp_ROI.height = dims.y * dims.z;

    npp_ROI_complex.width = dims.w / 2;
    npp_ROI_complex.height = dims.y * dims.z;

    is_npp_loaded = true;

  }
}

void GpuImage::BufferInit(BufferType bt)
{

  switch (bt)
  {
    case b_image :
        if ( ! is_allocated_image_buffer )
        {
          image_buffer = new GpuImage;
          *image_buffer = *this;
          is_allocated_image_buffer = true;
        }     
        break;

    case b_16f :
    	if ( ! is_allocated_16f_buffer )
    	{
    		checkCudaErrors(cudaMalloc(&real_values_16f, sizeof(__half)*real_memory_allocated));
    		complex_values_16f = (__half2 *)real_values_16f;
    		is_allocated_16f_buffer = true;
    	}
    	break;

    case b_sum :
        if ( ! is_allocated_sum_buffer ) 
        {
          int n_elem;
          nppiSumGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
          checkCudaErrors(cudaMalloc(&this->sum_buffer, n_elem));
          is_allocated_sum_buffer = true;
        }     
        break;   

    case b_min :
        if ( ! is_allocated_min_buffer )
        {
          int n_elem;
          nppiMinGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
          checkCudaErrors(cudaMalloc(&this->min_buffer, n_elem));
          is_allocated_min_buffer = true;
        }
        break;

  case b_minIDX :
      if ( ! is_allocated_minIDX_buffer )
      {
        int n_elem;
        nppiMinIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->minIDX_buffer, n_elem));
        is_allocated_minIDX_buffer = true;
      }
      break;

  case b_max :
      if ( ! is_allocated_max_buffer )
      {
        int n_elem;
        nppiMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->max_buffer, n_elem));
        is_allocated_max_buffer = true;
      }
      break;

  case b_maxIDX :
      if ( ! is_allocated_maxIDX_buffer )
      {
        int n_elem;
        nppiMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->maxIDX_buffer, n_elem));
        is_allocated_maxIDX_buffer = true;
      }
      break;

  case b_minmax :
      if ( ! is_allocated_minmax_buffer )
      {
        int n_elem;
        nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->minmax_buffer, n_elem));
        is_allocated_minmax_buffer = true;
      }
      break;

  case b_minmaxIDX :
      if ( ! is_allocated_minmaxIDX_buffer )
      {
        int n_elem;
        nppiMinMaxIndxGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->minmaxIDX_buffer, n_elem));
        is_allocated_minmaxIDX_buffer = true;
      }
      break;

  case b_mean :
      if ( ! is_allocated_mean_buffer )
      {
        int n_elem;
        nppiMeanGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->mean_buffer, n_elem));
        is_allocated_mean_buffer = true;
      }
      break;

  case b_meanstddev :
      if ( ! is_allocated_meanstddev_buffer )
      {
        int n_elem;
        nppiMeanGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->meanstddev_buffer, n_elem));
        is_allocated_meanstddev_buffer = true;
      }
      break;

  case b_countinrange :
      if ( ! is_allocated_countinrange_buffer )
      {
        int n_elem;
        nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
        checkCudaErrors(cudaMalloc(&this->countinrange_buffer, n_elem));
        is_allocated_countinrange_buffer = true;
      }
      break;

  case b_l2norm :
	  if (! is_allocated_l2norm_buffer)
	  {
		  int n_elem;
		  nppiNormL2GetBufferHostSize_32f_C1R_Ctx(npp_ROI, &n_elem,nppStream);
		  checkCudaErrors(cudaMalloc(&this->l2norm_buffer,n_elem));

		  is_allocated_l2norm_buffer = true;
	  }
	  break;

  case b_dotproduct :
	  if ( ! is_allocated_dotproduct_buffer )
	  {
		  int n_elem;
		  nppiDotProdGetBufferHostSize_32f64f_C1R_Ctx(npp_ROI, &n_elem, nppStream);
		  checkCudaErrors(cudaMalloc(&this->dotproduct_buffer, n_elem));
		  is_allocated_dotproduct_buffer = true;
	  }
	  break;


}

}


float GpuImage::ReturnSumOfSquares()
{

	// FIXME this assumes padded values are zero which is not strictly true
	MyAssertTrue(is_in_memory_gpu, "Image not allocated");
	MyAssertTrue(is_in_real_space, "This method is for real space, use ReturnSumSquareModulusComplexValues for Fourier space")


//	float returnValue = 0.0f;
	Npp64f* pNorm;
	checkCudaErrors(cudaMallocManaged(&pNorm, sizeof(Npp64f)));

	BufferInit(b_l2norm);
	NppInit();

	checkNppErrors(nppiNorm_L2_32f_C1R_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI,
									 	   (Npp64f *)pNorm, (Npp8u *)this->l2norm_buffer, nppStream));

	return ((float)*pNorm * (float)*pNorm);

//	CublasInit();complex_6EK0_SSU.pdb
//	// With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
//	cublas_stat = cublasSdot( cublasHandle, real_memory_allocated,
//							real_values_gpu, 1,
//							real_values_gpu, 1,
//							&returnValue);
//
//	if (cublas_stat) {
//	wxPrintf("Cublas return val %s\n", cublas_stat); }


//	return returnValue;

}



float GpuImage::ReturnSumSquareModulusComplexValues()
{

	//
	MyAssertTrue(is_in_memory_gpu, "Image not allocated");
	MyAssertFalse(is_in_real_space, "This method is NOT for real space, use ReturnSumofSquares for realspace")
	int address = 0;
	bool x_is_even = IsEven(dims.x);
	int i,j,k,jj,kk;
	const std::complex<float> c1(sqrtf(0.25f),sqrtf(0.25));
	const std::complex<float> c2(sqrtf(0.5),sqrtf(0.5)); // original code is pow(abs(Val),2)*0.5
	const std::complex<float> c3(1.0,1.0);
	const std::complex<float> c4(0.0,0.0);
	float returnValue;

	if ( ! is_allocated_mask_CSOS )
	{

		wxPrintf("is mask allocated %d\n", is_allocated_mask_CSOS);
		mask_CSOS = new GpuImage;
		is_allocated_mask_CSOS = true;
		wxPrintf("is mask allocated %d\n", is_allocated_mask_CSOS);
		// create a mask that can be reproduce the correct weighting from Image::ReturnSumOfSquares on complex images

		wxPrintf("\n\tMaking mask_CSOS\n");
		mask_CSOS->Allocate(dims.x, dims.y, dims.z, true);
		// The mask should always be in real_space, and starts out not centered
		mask_CSOS->is_in_real_space = true;
		mask_CSOS->object_is_centred_in_box = true;
		// Allocate pinned host memb
		checkCudaErrors(cudaHostAlloc(&mask_CSOS->real_values, sizeof(float)*real_memory_allocated, cudaHostAllocDefault));
		mask_CSOS->complex_values = (std::complex<float>*) mask_CSOS->real_values;
		for (k = 0; k <= physical_upper_bound_complex.z; k++)
		{

			kk = ReturnFourierLogicalCoordGivenPhysicalCoord_Z(k, dims.z, physical_index_of_first_negative_frequency.z);
			for (j = 0; j <= physical_upper_bound_complex.y; j++)
			{
				jj = ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j,dims.y, physical_index_of_first_negative_frequency.y);
				for (i = 0; i <= physical_upper_bound_complex.x; i++)
				{
					if ((i == 0  || (i  == logical_upper_bound_complex.x && x_is_even)) && 
			  (jj == 0 || (jj == logical_lower_bound_complex.y && x_is_even)) &&
			  (kk == 0 || (kk == logical_lower_bound_complex.z && x_is_even)))
		  {
			mask_CSOS->complex_values[address] = c2;

		  }
					else if ((i == 0 || (i == logical_upper_bound_complex.x && x_is_even)) && dims.z != 1) 
		  {
			mask_CSOS->complex_values[address] = c1;
		  }
					else if ((i != 0 && (i != logical_upper_bound_complex.x || ! x_is_even)) || (jj >= 0 && kk >= 0)) 
		  {
			mask_CSOS->complex_values[address] = c3;
		  }
		  else
		  {
			mask_CSOS->complex_values[address] = c4;
		  }

					address++;
				}
			}
		}   

		pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
		checkCudaErrors(cudaMemcpyAsync(mask_CSOS->real_values_gpu, mask_CSOS->real_values,sizeof(float)*real_memory_allocated,cudaMemcpyHostToDevice,cudaStreamPerThread));
		pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
		// TODO change this to an event that can then be later checked prior to deleteing
		checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
		checkCudaErrors(cudaFreeHost(mask_CSOS->real_values));

	} // end of mask creation


	BufferInit(b_image);
	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	checkCudaErrors(cudaMemcpyAsync(image_buffer->real_values_gpu, mask_CSOS->real_values_gpu, sizeof(float)*real_memory_allocated,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	image_buffer->MultiplyPixelWise(*this);

//	CublasInit();
	// With real and complex interleaved, treating as real is equivalent to taking the conj dot prod
	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

	NppInit();
	BufferInit(b_dotproduct);
	Npp64f* dotProd;
	checkCudaErrors(cudaMallocManaged(&dotProd, sizeof(Npp64f)));

	checkNppErrors(nppiDotProd_32f64f_C1R_Ctx((Npp32f *)image_buffer->real_values_gpu, pitch,
											  (Npp32f *)image_buffer->real_values_gpu, pitch,
											  npp_ROI, dotProd, dotproduct_buffer, nppStream));

//	cublasSdot( cublasHandle, real_memory_allocated,
//			  image_buffer->real_values_gpu, 1,
//			  image_buffer->real_values_gpu, 1,
//			  &returnValue);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

	return (float)(*dotProd * 2.0f);

  
}


__global__ void MultiplyPixelWiseComplexConjugateKernel(cufftComplex* ref_complex_values, cufftComplex* img_complex_values, int4 dims, int3 physical_upper_bound_complex)
{
  int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z)
    {

	    int address = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);

	    ref_complex_values[address] = (cufftComplex)ComplexConjMul((Complex)img_complex_values[address],(Complex)ref_complex_values[address]);
    }
    

}

void GpuImage::MipPixelWise(GpuImage &other_image)
{

	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	ReturnLaunchParamters(dims, true);
	MipPixelWiseKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> > (real_values_gpu, other_image.real_values_gpu, this->dims);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}
__global__ void MipPixelWiseKernel(cufftReal *mip, const cufftReal *correlation_output, const int4 dims)
{

    int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.x && coords.y < dims.y && coords.z < dims.z)
    {
	    int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
	    mip[address] = MAX(mip[address], correlation_output[address]);
    }
}

void GpuImage::MipPixelWise(GpuImage &other_image, GpuImage &psi, GpuImage &phi, GpuImage &theta, GpuImage &defocus, GpuImage &pixel,
                            float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel)
{

	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	ReturnLaunchParamters(dims, true);
	MipPixelWiseKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(real_values_gpu, other_image.real_values_gpu,
																	   psi.real_values_gpu,phi.real_values_gpu,theta.real_values_gpu,defocus.real_values_gpu,pixel.real_values_gpu,
																		this->dims, c_psi, c_phi, c_theta, c_defocus, c_pixel);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);


}



__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *correlation_output, cufftReal *psi, cufftReal *phi, cufftReal *theta, cufftReal *defocus, cufftReal *pixel, const int4 dims,
                                   float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel)
{

    int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.x && coords.y < dims.y && coords.z < dims.z)
    {
	    int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
      if (correlation_output[address] > mip[address])
      {
        mip[address] = correlation_output[address];
        psi[address] = c_psi;
        phi[address] = c_phi;
        theta[address] = c_theta;
        defocus[address] = c_defocus;
        pixel[address] = c_pixel;
      }

    }
}

void GpuImage::MipPixelWise(GpuImage &other_image, GpuImage &psi, GpuImage &phi, GpuImage &theta,
                            float c_psi, float c_phi, float c_theta)
{

	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	ReturnLaunchParamters(dims, true);
	MipPixelWiseKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(real_values_gpu, other_image.real_values_gpu,
																	   psi.real_values_gpu,phi.real_values_gpu,theta.real_values_gpu,
																	   this->dims, c_psi, c_phi, c_theta);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}



__global__ void MipPixelWiseKernel(cufftReal* mip, cufftReal *correlation_output, cufftReal *psi, cufftReal *phi, cufftReal *theta, const int4 dims,
                                   float c_psi, float c_phi, float c_theta)
{

    int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z);

    if (coords.x < dims.x && coords.y < dims.y && coords.z < dims.z)
    {
	  int address = d_ReturnReal1DAddressFromPhysicalCoord(coords, dims);
      if (correlation_output[address] > mip[address])
      {
        mip[address] = correlation_output[address];
        psi[address] = c_psi;
        phi[address] = c_phi;
        theta[address] = c_theta;
      }

    }
}


void GpuImage::Abs()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

  NppInit();
  checkNppErrors(nppiAbs_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI,nppStream));
}

void GpuImage::AbsDiff(GpuImage &other_image)
{
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");


	BufferInit(b_image);
	NppInit();

	checkNppErrors(nppiAbsDiff_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch,
									 (const Npp32f *)other_image.real_values_gpu, pitch,
									 (      Npp32f *)this->image_buffer->real_values_gpu, pitch, npp_ROI,nppStream));

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	checkCudaErrors(cudaMemcpyAsync(real_values_gpu,this->image_buffer->real_values_gpu,sizeof(cufftReal)*real_memory_allocated,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}

void GpuImage::AbsDiff(GpuImage &other_image, GpuImage &output_image)
{
  // In place abs diff (see overload for out of place)
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimension.");
	MyAssertTrue(HasSameDimensionsAs(&output_image), "Images have different dimension.");



  NppInit();
  
  checkNppErrors(nppiAbsDiff_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch,
                                     (const Npp32f *)other_image.real_values_gpu, pitch,
                                     (      Npp32f *)output_image.real_values_gpu, pitch, npp_ROI,nppStream));

}

void GpuImage::Min()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");

	NppInit();
	BufferInit(b_min);
	checkNppErrors(nppiMin_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, min_buffer, (Npp32f *)&min_value,nppStream));
}
void GpuImage::MinAndCoords()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_minIDX);
	checkNppErrors(nppiMinIndx_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch, npp_ROI, minIDX_buffer, (Npp32f *)&min_value, &min_idx.x, &min_idx.y,nppStream));
}
void GpuImage::Max()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_max);
	checkNppErrors(nppiMax_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, max_buffer, (Npp32f *)&max_value,nppStream));
}
void GpuImage::MaxAndCoords()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_maxIDX);
	checkNppErrors(nppiMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, maxIDX_buffer, (Npp32f *)&max_value, &max_idx.x, &max_idx.y, nppStream));
}
void GpuImage::MinMax()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_minmax);
	checkNppErrors(nppiMinMax_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, (Npp32f *)&min_value, (Npp32f *)&max_value, minmax_buffer,nppStream));
}
void GpuImage::MinMaxAndCoords()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_minmaxIDX);
	checkNppErrors(nppiMinMaxIndx_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, (Npp32f *)&min_value, (Npp32f *)&max_value,  &min_idx, &max_idx,minmax_buffer,nppStream));
}

void GpuImage::Mean()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");

	NppInit();
	BufferInit(b_mean);
	checkNppErrors(nppiMean_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, mean_buffer, npp_mean, nppStream));
	this->img_mean   = (float)*npp_mean;

}

void GpuImage::MeanStdDev()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	BufferInit(b_meanstddev);
	checkNppErrors(nppiMean_StdDev_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI, meanstddev_buffer, npp_mean, npp_stdDev,nppStream));
	this->img_mean   = (float)*npp_mean;
	this->img_stdDev = (float)*npp_stdDev;
}

void GpuImage::MultiplyPixelWise(GpuImage &other_image)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

	NppInit();
	checkNppErrors(nppiMul_32f_C1IR_Ctx((Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI,nppStream));
}


void GpuImage::AddConstant(const float add_val)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	checkNppErrors(nppiAddC_32f_C1IR_Ctx((Npp32f)add_val, (Npp32f*)real_values_gpu, pitch, npp_ROI,nppStream));
}

void GpuImage::SquareRealValues()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	checkNppErrors(nppiSqr_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::SquareRootRealValues()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	checkNppErrors(nppiSqrt_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::LogarithmRealValues()
{
  MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
  
  NppInit();
  checkNppErrors(nppiLn_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI,nppStream));

}

void GpuImage::ExponentiateRealValues()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	checkNppErrors(nppiExp_32f_C1IR_Ctx((Npp32f *)real_values_gpu, pitch, npp_ROI, nppStream));
}

void GpuImage::CountInRange(float lower_bound, float upper_bound)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	NppInit();
	checkNppErrors(nppiCountInRange_32f_C1R_Ctx((const Npp32f *)real_values_gpu, pitch, npp_ROI, &number_of_pixels_in_range,
											(Npp32f)lower_bound,(Npp32f)upper_bound,countinrange_buffer, nppStream));

}

float GpuImage::ReturnSumOfRealValues()
{
	// FIXME assuming padded values are zero
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(is_in_real_space, "Not in real space");


	Npp64f sum_val;

	NppInit();
	BufferInit(b_sum);
	checkNppErrors(nppiSum_32f_C1R_Ctx((const Npp32f*)real_values_gpu, pitch, npp_ROI,sum_buffer,&sum_val, nppStream));

	return (float)sum_val;
}
void GpuImage::AddImage(GpuImage &other_image)
{
  // Add the real_values_gpu into a double array
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");


	NppInit();
	checkNppErrors(nppiAdd_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu, pitch, (Npp32f*)real_values_gpu, pitch, npp_ROI, nppStream));

} 

void GpuImage::AddSquaredImage(GpuImage &other_image)
{
	// Add the real_values_gpu into a double array
	MyAssertTrue(HasSameDimensionsAs(&other_image), "Images have different dimensions");

	NppInit();
	checkNppErrors(nppiAddSquare_32f_C1IR_Ctx((const Npp32f*)other_image.real_values_gpu,  pitch, (Npp32f*)real_values_gpu,  pitch, npp_ROI, nppStream));
} 

void GpuImage::MultiplyByConstant(float scale_factor)
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");

	NppInit();
	checkNppErrors(nppiMulC_32f_C1IR_Ctx((Npp32f) scale_factor, (Npp32f*)real_values_gpu,  pitch, npp_ROI, nppStream));
}

void GpuImage::Conj()
{
	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertFalse(is_in_real_space, "Conj only supports complex images");

	float scale_factor = -1.0f;
	NppInit();
	checkNppErrors(nppiMulC_32f_C1IR_Ctx((Npp32f) (scale_factor+1), (Npp32f*)real_values_gpu,  dims.w/2*sizeof(float), npp_ROI,nppStream));
	// FIXME make sure that a) there isn't already a function fo rthis, b) you aren't striding out of bounds (mask instead_;
}

void GpuImage::Zeros()
{
 
  MyAssertTrue(is_meta_data_initialized, "Host meta data has not been copied");

  if ( ! is_in_memory_gpu )
  {
    checkCudaErrors(cudaMalloc(&real_values_gpu, real_memory_allocated*sizeof(float)));
    complex_values_gpu = (cufftComplex *)real_values_gpu;
    is_in_memory_gpu = true;
  }

  pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
  checkCudaErrors(cudaMemsetAsync(real_values_gpu, 0, real_memory_allocated*sizeof(float), cudaStreamPerThread));
  checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

}


void GpuImage::CopyHostToDevice()
{
 
	MyAssertTrue(is_in_memory, "Host memory not allocated");

	if ( ! is_in_memory_gpu )
	{
		checkCudaErrors(cudaMalloc(&real_values_gpu, real_memory_allocated*sizeof(float)));
		complex_values_gpu = (cufftComplex *)real_values_gpu;
		is_in_memory_gpu = true;
	}

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	checkCudaErrors(cudaMemcpyAsync( real_values_gpu, pinnedPtr, real_memory_allocated*sizeof(float),cudaMemcpyHostToDevice,cudaStreamPerThread));
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

	UpdateCpuFlags();

}

void GpuImage::CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory)
{
 
	MyAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
	// TODO other asserts on size etc.
	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	checkCudaErrors(cudaMemcpyAsync(pinnedPtr, real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	//  checkCudaErrors(cudaMemcpyAsync(real_values, real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
	// TODO add asserts etc.
	if (free_gpu_memory) { cudaFree(real_values_gpu) ; } // FIXME what about the other structures
	if (unpin_host_memory && is_host_memory_pinned)
	{
		cudaHostUnregister(real_values);
		is_host_memory_pinned = false;
	}

}

void GpuImage::CopyDeviceToHost(Image &cpu_image, bool should_block_until_complete, bool free_gpu_memory)
{

	MyAssertTrue(is_in_memory_gpu, "GPU memory not allocated");
	// TODO other asserts on size etc.


	float* tmpPinnedPtr;
	// FIXME for now always pin the memory - this might be a bad choice for single copy or small images, but is required for asynch xfer and is ~2x as fast after pinning
	cudaHostRegister(cpu_image.real_values, sizeof(float)*real_memory_allocated, cudaHostRegisterDefault);
	cudaHostGetDevicePointer( &tmpPinnedPtr, cpu_image.real_values, 0);

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	checkCudaErrors(cudaMemcpyAsync(tmpPinnedPtr, real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost,cudaStreamPerThread));
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

	if (should_block_until_complete) checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
	// TODO add asserts etc.
	if (free_gpu_memory) { cudaFree(real_values_gpu) ; } // FIXME what about the other structures

	cudaHostUnregister(tmpPinnedPtr);


}

void GpuImage::CopyVolumeHostToDevice()
{


  // FIXME not working
    bool is_working = false;
    MyAssertTrue(is_working, "CopyVolumeHostToDevice is not properly worked out");

		d_pitchedPtr = { 0 };
		d_extent = make_cudaExtent(dims.x * sizeof(float), dims.y, dims.z);
		checkCudaErrors(cudaMalloc3D(&d_pitchedPtr, d_extent)); // Complex values need to be pointed
    this->real_values_gpu = (cufftReal *)d_pitchedPtr.ptr; // Set the values here

		d_3dparams        = { 0 };
		d_3dparams.srcPtr = make_cudaPitchedPtr((void*)real_values, dims.x * sizeof(float), dims.x, dims.y);
		d_3dparams.dstPtr = d_pitchedPtr;
		d_3dparams.extent = d_extent;
		d_3dparams.kind   = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&d_3dparams));

}

void GpuImage::CopyVolumeDeviceToHost(bool free_gpu_memory, bool unpin_host_memory)
{

  // FIXME not working
    bool is_working = false;
    MyAssertTrue(is_working, "CopyVolumeDeviceToHost is not properly worked out");

    if ( ! is_in_memory )
    {
		  checkCudaErrors(cudaMallocHost(&real_values, real_memory_allocated*sizeof(float)));
    }
    h_pitchedPtr = make_cudaPitchedPtr((void*)real_values, dims.x * sizeof(float), dims.x, dims.y);
		h_extent = make_cudaExtent(dims.x * sizeof(float), dims.y, dims.z);
		h_3dparams        = { 0 };
		h_3dparams.srcPtr = d_pitchedPtr;
		h_3dparams.dstPtr = h_pitchedPtr;
		h_3dparams.extent = h_extent;
		h_3dparams.kind   = cudaMemcpyDeviceToHost;
		checkCudaErrors(cudaMemcpy3D(&h_3dparams));

    is_in_memory = true;

    // TODO add asserts etc.
    if (free_gpu_memory) 
    { cudaFree(d_pitchedPtr.ptr) ; } // FIXME what about the other structures
    if (unpin_host_memory && is_host_memory_pinned) 
    {
      cudaHostUnregister(real_values);
      is_host_memory_pinned = false;
    }
}

void GpuImage::ForwardFFT(bool should_scale)
{

	bool is_half_precision = false;

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertTrue(is_in_real_space, "Image alread in Fourier space");

	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}

	// For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
	if ( is_half_precision && ! is_set_convertInputf16Tof32 )
	{
		cufftCallbackLoadR h_ConvertInputf16Tof32Ptr;
		checkCudaErrors(cudaMemcpyFromSymbol(&h_ConvertInputf16Tof32Ptr,d_ConvertInputf16Tof32Ptr, sizeof(h_ConvertInputf16Tof32Ptr)));
		checkCudaErrors(cufftXtSetCallback(cuda_plan_forward, (void **)&h_ConvertInputf16Tof32Ptr, CUFFT_CB_LD_REAL, 0));
		is_set_convertInputf16Tof32 = true;
		//	  checkCudaErrors(cudaFree(norm_factor));
		//	  this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}
	if (should_scale)
	{
		this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}

//	if (should_scale && ! is_set_scaleFFTAndStore)
//	{
//
//		float ft_norm_sq = ft_normalization_factor*ft_normalization_factor;
//		checkCudaErrors(cudaMalloc((void **)&d_scale_factor, sizeof(float)));
//		checkCudaErrors(cudaMemcpyAsync(d_scale_factor, &ft_norm_sq, sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
//		checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
//
//		cufftCallbackStoreC h_scaleFFTAndStorePtr;
//		checkCudaErrors(cudaMemcpyFromSymbol(&h_scaleFFTAndStorePtr,d_scaleFFTAndStorePtr, sizeof(h_scaleFFTAndStorePtr)));
//		checkCudaErrors(cufftXtSetCallback(cuda_plan_forward, (void **)&h_scaleFFTAndStorePtr, CUFFT_CB_ST_COMPLEX, (void **)&d_scale_factor));
//		is_set_scaleFFTAndStore = true;
//	}


//	BufferInit(b_image);
//    checkCudaErrors(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)image_buffer->complex_values));

    checkCudaErrors(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)complex_values_gpu));

    is_in_real_space = false;


}

void GpuImage::ForwardFFTAndClipInto(GpuImage &image_to_insert, bool should_scale)
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertTrue(image_to_insert.is_in_memory_gpu, "Gpu memory in image to insert not allocated");
	MyAssertTrue(is_in_real_space, "Image alread in Fourier space");
	MyAssertTrue(image_to_insert.is_in_real_space, "I in image to insert alread in Fourier space");


	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}

	// For reference to clear cufftXtClearCallback(cufftHandle lan, cufftXtCallbackType type);
	if ( ! is_set_realLoadAndClipInto )
	{
		cufftCallbackLoadR h_realLoadAndClipInto;
		CB_realLoadAndClipInto_params* d_params;
		CB_realLoadAndClipInto_params h_params;


		h_params.target = (cufftReal *)image_to_insert.real_values_gpu;
		h_params.i_lower_bound =  this->physical_address_of_box_center.x - image_to_insert.physical_address_of_box_center.x;
		h_params.j_lower_bound =  this->physical_address_of_box_center.y - image_to_insert.physical_address_of_box_center.y;
		h_params.i_upper_bound = h_params.i_lower_bound + image_to_insert.dims.x - 1;
		h_params.j_upper_bound = h_params.j_lower_bound + image_to_insert.dims.y - 1;
		// For the offset in memory, we need to consider the physical dimensions of the images and not just the logical.
		h_params.nx_small = (image_to_insert.dims.w);
		h_params.nx_large 	  = this->dims.w;

		checkCudaErrors(cudaMalloc((void **)&d_params,sizeof(CB_realLoadAndClipInto_params)));
		checkCudaErrors(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_realLoadAndClipInto_params), cudaMemcpyHostToDevice, cudaStreamPerThread));
		checkCudaErrors(cudaMemcpyFromSymbol(&h_realLoadAndClipInto,d_realLoadAndClipInto, sizeof(h_realLoadAndClipInto)));
		checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));

		checkCudaErrors(cufftXtSetCallback(cuda_plan_forward, (void **)&h_realLoadAndClipInto, CUFFT_CB_LD_REAL, (void **)&d_params));
		is_set_realLoadAndClipInto = true;


		//	  checkCudaErrors(cudaFree(norm_factor));
		//	  this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}
	if (should_scale)
	{
		this->MultiplyByConstant(ft_normalization_factor*ft_normalization_factor);
	}




//	BufferInit(b_image);
//    checkCudaErrors(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)image_buffer->complex_values));

    checkCudaErrors(cufftExecR2C(this->cuda_plan_forward, (cufftReal*)real_values_gpu, (cufftComplex*)complex_values_gpu));

    is_in_real_space = false;


}

void GpuImage::BackwardFFT()
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertFalse(is_in_real_space, "Image is already in real space");

	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}

	//  BufferInit(b_image);
	//  checkCudaErrors(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values_gpu));

	checkCudaErrors(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)complex_values_gpu, (cufftReal*)real_values_gpu));

	is_in_real_space = true;

}

template < typename T > void GpuImage::BackwardFFTAfterComplexConjMul(T* image_to_multiply, bool load_half_precision)
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");
	MyAssertFalse(is_in_real_space, "Image is already in real space");

	if ( ! is_fft_planned )
	{
		SetCufftPlan();
	}

	if ( ! is_set_complexConjMulLoad )
	{
		cufftCallbackStoreC h_complexConjMulLoad;


			CB_complexConjMulLoad_params<T>* d_params;
			CB_complexConjMulLoad_params<T> h_params;
			h_params.scale = ft_normalization_factor*ft_normalization_factor;
			h_params.target = (T *)image_to_multiply;
			checkCudaErrors(cudaMalloc((void **)&d_params,sizeof(CB_complexConjMulLoad_params<T>)));
			checkCudaErrors(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_complexConjMulLoad_params<T>), cudaMemcpyHostToDevice, cudaStreamPerThread));

			if (load_half_precision)
			{
				checkCudaErrors(cudaMemcpyFromSymbol(&h_complexConjMulLoad,d_complexConjMulLoad_16f, sizeof(h_complexConjMulLoad)));
			}
			else
			{
				checkCudaErrors(cudaMemcpyFromSymbol(&h_complexConjMulLoad,d_complexConjMulLoad_32f, sizeof(h_complexConjMulLoad)));
			}

			checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
			checkCudaErrors(cufftXtSetCallback(cuda_plan_inverse, (void **)&h_complexConjMulLoad, CUFFT_CB_LD_COMPLEX, (void **)&d_params));

//		d_complexConjMulLoad;
		is_set_complexConjMulLoad = true;
	}

	//  BufferInit(b_image);
	//  checkCudaErrors(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)image_buffer->complex_values, (cufftReal*)real_values_gpu));

	checkCudaErrors(cufftExecC2R(this->cuda_plan_inverse, (cufftComplex*)complex_values_gpu, (cufftReal*)real_values_gpu));

	is_in_real_space = true;

}
template void GpuImage::BackwardFFTAfterComplexConjMul(__half2* image_to_multiply, bool load_half_precision);
template void GpuImage::BackwardFFTAfterComplexConjMul(cufftComplex* image_to_multiply, bool load_half_precision);



void GpuImage::Wait()
{
  checkCudaErrors(cudaStreamSynchronize(cudaStreamPerThread));
}

void GpuImage::SwapRealSpaceQuadrants()
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");

	bool must_fft = false;

	float x_shift_to_apply;
	float y_shift_to_apply;
	float z_shift_to_apply;

	if (is_in_real_space == true)
	{
		must_fft = true;
		ForwardFFT(true);
	}

	if (object_is_centred_in_box == true)
	{
		x_shift_to_apply = float(physical_address_of_box_center.x);
		y_shift_to_apply = float(physical_address_of_box_center.y);
		z_shift_to_apply = float(physical_address_of_box_center.z);
	}
	else
	{
		if (IsEven(dims.x) == true)
		{
			x_shift_to_apply = float(physical_address_of_box_center.x);
		}
		else
		{
			x_shift_to_apply = float(physical_address_of_box_center.x) - 1.0;
		}

		if (IsEven(dims.y) == true)
		{
			y_shift_to_apply = float(physical_address_of_box_center.y);
		}
		else
		{
			y_shift_to_apply = float(physical_address_of_box_center.y) - 1.0;
		}

		if (IsEven(dims.z) == true)
		{
			z_shift_to_apply = float(physical_address_of_box_center.z);
		}
		else
		{
			z_shift_to_apply = float(physical_address_of_box_center.z) - 1.0;
		}
	}


	if (dims.z == 1)
	{
		z_shift_to_apply = 0.0;
	}

	PhaseShift(x_shift_to_apply, y_shift_to_apply, z_shift_to_apply);

	if (must_fft == true) BackwardFFT();


	// keep track of center;
	if (object_is_centred_in_box == true) object_is_centred_in_box = false;
	else object_is_centred_in_box = true;
}




void GpuImage::PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift)
{

	MyAssertTrue(is_in_memory_gpu, "Gpu memory not allocated");

	bool need_to_fft = false;
	if (is_in_real_space == true)
	{
    wxPrintf("Doing forward fft in phase shift function\n\n");
		ForwardFFT(true);
		need_to_fft = true;
	}

  float3 shifts = make_float3(wanted_x_shift, wanted_y_shift, wanted_z_shift);
  // TODO set the TPB and inline function for grid

  
  dim3 threadsPerBlock(32, 32, 1);
  dim3 gridDims((dims.w/2 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                (dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y, dims.z); 

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	PhaseShiftKernel<< <gridDims, threadsPerBlock,0,cudaStreamPerThread>> >(complex_values_gpu, 
                                                        dims, shifts,
                                                        physical_address_of_box_center,
                                                        physical_index_of_first_negative_frequency,
                                                        physical_upper_bound_complex);
  
    checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

	if (need_to_fft == true) BackwardFFT();

}

__device__ __forceinline__ float
d_ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size)
{
	return real_space_shift * distance_from_origin * 2.0 * PI / dimension_size;
}

__device__ __forceinline__ void
d_Return3DPhaseFromIndividualDimensions( float phase_x, float phase_y, float phase_z, float2 &angles)
{
	float temp_phase = -phase_x-phase_y-phase_z;
	__sincosf(temp_phase, &angles.y, &angles.x); // To use as cos.x + i*sin.y
}


 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index, 
                                                int logical_x_dimension,
                                                int physical_address_of_box_center_x)
{
//	MyAssertTrue(is_in_memory, "Memory not allocated");
//	MyAssertTrue(physical_index <= physical_upper_bound_complex_x, "index out of bounds");

    //if (physical_index >= physical_index_of_first_negative_frequency_x)
    if (physical_index > physical_address_of_box_center_x)
    {
    	 return physical_index - logical_x_dimension;
    }
    else return physical_index;
}


 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index,
                                                int logical_y_dimension,
                                                int physical_index_of_first_negative_frequency_y )
{
//	MyAssertTrue(is_in_memory, "Memory not allocated");
//	MyAssertTrue(physical_index <= physical_upper_bound_complex_y, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_y)
    {
    	 return physical_index - logical_y_dimension;
    }
    else return physical_index;
}


 __device__ __forceinline__ int
d_ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index,

                                                int logical_z_dimension,
                                                int physical_index_of_first_negative_frequency_z )
{
//	MyAssertTrue(is_in_memory, "Memory not allocated");
//	MyAssertTrue(physical_index <= physical_upper_bound_complex_z, "index out of bounds");

    if (physical_index >= physical_index_of_first_negative_frequency_z)
    {
    	 return physical_index - logical_z_dimension;
    }
    else return physical_index;
}

 __device__ __forceinline__ int
d_ReturnReal1DAddressFromPhysicalCoord(int3 coords, int4 img_dims)
{
	return ( (((int)coords.z*(int)img_dims.y + coords.y) * (int)img_dims.w)  + (int)coords.x) ;
}

 __device__ __forceinline__ int
d_ReturnFourier1DAddressFromPhysicalCoord(int3 wanted_dims, int3 physical_upper_bound_complex)
{
	return ( (int)((physical_upper_bound_complex.y + 1) * wanted_dims.z + wanted_dims.y) *
            (int)(physical_upper_bound_complex.x + 1) + (int)wanted_dims.x );
}


__global__ void ClipIntoRealKernel(cufftReal* real_values_gpu,
                                   cufftReal* other_image_real_values_gpu,
                                   int4 dims, 
                                   int4 other_dims,
                                   int3 physical_address_of_box_center, 
                                   int3 other_physical_address_of_box_center,
                                   int3 wanted_coordinate_of_box_center, 
                                   float wanted_padding_value)
{
  int3 other_coord = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                               blockIdx.y*blockDim.y + threadIdx.y,
                               blockIdx.z);

  int3 coord = make_int3(0, 0, 0); 
  
  if (other_coord.x < other_dims.x &&
      other_coord.y < other_dims.y &&
      other_coord.z < other_dims.z)
  {

    coord.z = physical_address_of_box_center.z + wanted_coordinate_of_box_center.z + 
              other_coord.z - other_physical_address_of_box_center.z;

    coord.y = physical_address_of_box_center.y + wanted_coordinate_of_box_center.y + 
              other_coord.y - other_physical_address_of_box_center.y;

    coord.x = physical_address_of_box_center.x + wanted_coordinate_of_box_center.x + 
              other_coord.x - other_physical_address_of_box_center.x;

    if (coord.z < 0 || coord.z >= dims.z || 
        coord.y < 0 || coord.y >= dims.y ||
        coord.x < 0 || coord.x >= dims.x)
    {
      other_image_real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = wanted_padding_value;
    }
    else
    {
      other_image_real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims) ] = 
                  real_values_gpu[ d_ReturnReal1DAddressFromPhysicalCoord(coord, dims) ];
    }





  }
//		for (kk = 0; kk < other_image->logical_z_dimension; kk++)
//		{
//			kk_logi = kk - other_image->physical_address_of_box_center_z;
//			k = physical_address_of_box_center_z + wanted_coordinate_of_box_center_z + kk_logi;

//			for (jj = 0; jj < other_image->logical_y_dimension; jj++)
//			{
//				jj_logi = jj - other_image->physical_address_of_box_center_y;
//				j = physical_address_of_box_center_y + wanted_coordinate_of_box_center_y + jj_logi;

//				for (ii = 0; ii < other_image->logical_x_dimension; ii++)
//				{
//					ii_logi = ii - other_image->physical_address_of_box_center_x;
//					i = physical_address_of_box_center_x + wanted_coordinate_of_box_center_x + ii_logi;

//					if (k < 0 || k >= logical_z_dimension || j < 0 || j >= logical_y_dimension || i < 0 || i >= logical_x_dimension)
//					{
//						other_image->real_values[pixel_counter] = wanted_padding_value;
//					}
//					else
//					{
//						other_image->real_values[pixel_counter] = ReturnRealPixelFromPhysicalCoord(i, j, k);
//					}

//					pixel_counter++;
//				}

//				pixel_counter+=other_image->padding_jump_value;
//			}
//		}
//	}

}
__global__ void PhaseShiftKernel(cufftComplex* d_input, 
                                 int4 dims, float3 shifts, 
                                 int3 physical_address_of_box_center, 
                                 int3 physical_index_of_first_negative_frequency,
                                 int3 physical_upper_bound_complex)
{
	
// FIXME it probably makes sense so just just a linear grid launch and save the extra indexing
  int3 wanted_dims = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
                               blockIdx.y*blockDim.y + threadIdx.y,
                               blockIdx.z);

  float2 init_vals;
  float2 angles;

// FIXME This should probably use cuBlas
  if (wanted_dims.x <= physical_upper_bound_complex.x && 
      wanted_dims.y <= physical_upper_bound_complex.y && 
      wanted_dims.z <= physical_upper_bound_complex.z)
  {
  
    
    d_Return3DPhaseFromIndividualDimensions( d_ReturnPhaseFromShift(
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
    
    int address = d_ReturnFourier1DAddressFromPhysicalCoord(wanted_dims, physical_upper_bound_complex);
    init_vals.x = d_input[ address ].x;
    init_vals.y = d_input[ address ].y;
    d_input[ address ].x = init_vals.x*angles.x - init_vals.y*angles.y;
    d_input[ address ].y = init_vals.x*angles.y + init_vals.y*angles.x;
  }
  

}


// If you don't want to clip from the center, you can give wanted_coordinate_of_box_center_{x,y,z}. This will define the pixel in the image at which other_image will be centered. (0,0,0) means center of image. This is a dumbed down version that does not fill with noise.
void GpuImage::ClipInto(GpuImage *other_image, float wanted_padding_value,                 
                        bool fill_with_noise, float wanted_noise_sigma,
                        int wanted_coordinate_of_box_center_x, 
                        int wanted_coordinate_of_box_center_y, 
                        int wanted_coordinate_of_box_center_z)
{

	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
	MyAssertTrue(other_image->is_in_memory_gpu, "Other image Memory not allocated");
	MyAssertTrue(is_in_real_space, "Clip into is only set up for real space on the gpu currently");


  int3 wanted_coordinate_of_box_center = make_int3(wanted_coordinate_of_box_center_x, 
                                                   wanted_coordinate_of_box_center_y, 
                                                   wanted_coordinate_of_box_center_z);

	other_image->is_in_real_space = is_in_real_space;
	other_image->object_is_centred_in_box = object_is_centred_in_box;

	if (is_in_real_space == true)
	{

		MyAssertTrue(object_is_centred_in_box, "real space image, not centred in box");

    ReturnLaunchParamters(other_image->dims, true);

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
    ClipIntoRealKernel<< <gridDims, threadsPerBlock, 0, cudaStreamPerThread>> >(real_values_gpu,
                                                              other_image->real_values_gpu,
                                                              dims, 
                                                              other_image->dims,
                                                              physical_address_of_box_center,
                                                              other_image->physical_address_of_box_center, 
                                                              wanted_coordinate_of_box_center, 
                                                              wanted_padding_value);
    checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
  }
	else
  {
    // FIXME
    wxPrintf("\n\nClipInto is only setup for real space!\n\n");
    exit(-1);
  }
//	{
//		for (kk = 0; kk <= other_image->physical_upper_bound_complex_z; kk++)
//		{
//			temp_logical_z = other_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Z(kk);

//			//if (temp_logical_z > logical_upper_bound_complex_z || temp_logical_z < logical_lower_bound_complex_z) continue;

//			for (jj = 0; jj <= other_image->physical_upper_bound_complex_y; jj++)
//			{
//				temp_logical_y = other_image->ReturnFourierLogicalCoordGivenPhysicalCoord_Y(jj);

//				//if (temp_logical_y > logical_upper_bound_complex_y || temp_logical_y < logical_lower_bound_complex_y) continue;

//				for (ii = 0; ii <= other_image->physical_upper_bound_complex_x; ii++)
//				{
//					temp_logical_x = ii;

//					//if (temp_logical_x > logical_upper_bound_complex_x || temp_logical_x < logical_lower_bound_complex_x) continue;

//					if (fill_with_noise == false) other_image->complex_values[pixel_counter] = ReturnComplexPixelFromLogicalCoord(temp_logical_x, temp_logical_y, temp_logical_z, wanted_padding_value + I * 0.0f);
//					else
//					{

//						if (temp_logical_x < logical_lower_bound_complex_x || temp_logical_x > logical_upper_bound_complex_x || temp_logical_y < logical_lower_bound_complex_y ||temp_logical_y > logical_upper_bound_complex_y || temp_logical_z < logical_lower_bound_complex_z || temp_logical_z > logical_upper_bound_complex_z)
//						{
//							other_image->complex_values[pixel_counter] = (global_random_number_generator.GetNormalRandom() * wanted_noise_sigma) + (I * global_random_number_generator.GetNormalRandom() * wanted_noise_sigma);
//						}
//						else
//						{
//							other_image->complex_values[pixel_counter] = complex_values[ReturnFourier1DAddressFromLogicalCoord(temp_logical_x,temp_logical_y, temp_logical_z)];

//						}


//					}
//					pixel_counter++;

//				}

//			}
//		}


//		// When we are clipping into a larger volume in Fourier space, there is a half-plane (vol) or half-line (2D image) at Nyquist for which FFTW
//		// does not explicitly tell us the values. We need to fill them in.
//		if (logical_y_dimension < other_image->logical_y_dimension || logical_z_dimension < other_image->logical_z_dimension)
//		{
//			// For a 2D image
//			if (logical_z_dimension == 1)
//			{
//				jj = physical_index_of_first_negative_frequency_y;
//				for (ii = 0; ii <= physical_upper_bound_complex_x; ii++)
//				{
//					other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj,0)] = complex_values[ReturnFourier1DAddressFromPhysicalCoord(ii,jj,0)];
//				}
//			}
//			// For a 3D volume
//			else
//			{

//				// Deal with the positive Nyquist of the 2nd dimension
//				for (kk_logi = logical_lower_bound_complex_z; kk_logi <= logical_upper_bound_complex_z; kk_logi ++)
//				{
//					jj = physical_index_of_first_negative_frequency_y;
//					jj_logi = logical_lower_bound_complex_y;
//					for (ii = 0; ii <= physical_upper_bound_complex_x; ii++)
//					{
//						other_image->complex_values[other_image->ReturnFourier1DAddressFromLogicalCoord(ii,jj,kk_logi)] = complex_values[ReturnFourier1DAddressFromLogicalCoord(ii,jj_logi,kk_logi)];
//					}
//				}


//				// Deal with the positive Nyquist in the 3rd dimension
//				kk = physical_index_of_first_negative_frequency_z;
//				int kk_mirror = other_image->logical_z_dimension - physical_index_of_first_negative_frequency_z;
//				//wxPrintf("\nkk = %i; kk_mirror = %i\n",kk,kk_mirror);
//				int jj_mirror;
//				//wxPrintf("Will loop jj from %i to %i\n",1,physical_index_of_first_negative_frequency_y);
//				for (jj = 1; jj <= physical_index_of_first_negative_frequency_y; jj ++ )
//				{
//					//jj_mirror = other_image->logical_y_dimension - jj;
//					jj_mirror = jj;
//					for (ii = 0; ii <= physical_upper_bound_complex_x; ii++ )
//					{
//						//wxPrintf("(1) ii = %i; jj = %i; kk = %i; jj_mirror = %i; kk_mirror = %i\n",ii,jj,kk,jj_mirror,kk_mirror);
//						other_image->complex_values[other_image-> ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk)] = other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj_mirror,kk_mirror)];
//					}
//				}
//				//wxPrintf("Will loop jj from %i to %i\n", other_image->logical_y_dimension - physical_index_of_first_negative_frequency_y, other_image->logical_y_dimension - 1);
//				for (jj = other_image->logical_y_dimension - physical_index_of_first_negative_frequency_y; jj <= other_image->logical_y_dimension - 1; jj ++)
//				{
//					//jj_mirror = other_image->logical_y_dimension - jj;
//					jj_mirror = jj;
//					for (ii = 0; ii <= physical_upper_bound_complex_x; ii++ )
//					{
//						//wxPrintf("(2) ii = %i; jj = %i; kk = %i; jj_mirror = %i; kk_mirror = %i\n",ii,jj,kk,jj_mirror,kk_mirror);
//						other_image->complex_values[other_image-> ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk)] = other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj_mirror,kk_mirror)];
//					}
//				}
//				jj = 0;
//				for (ii = 0; ii <= physical_upper_bound_complex_x; ii++)
//				{
//					other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk)] = other_image->complex_values[other_image->ReturnFourier1DAddressFromPhysicalCoord(ii,jj,kk_mirror)];
//				}

//			}
//		}


//	}

}

void GpuImage::QuickAndDirtyWriteSlices(std::string filename, int first_slice, int last_slice)
{

	MyAssertTrue(is_in_memory_gpu, "Memory not allocated");
  Image buffer_img;
  buffer_img.Allocate(dims.x, dims.y, dims.z, true);

  buffer_img.is_in_real_space = is_in_real_space;
  buffer_img.object_is_centred_in_box = object_is_centred_in_box;
  // Implicitly waiting on work to finish since copy is queued in the stream
  checkCudaErrors(cudaMemcpy((void*)buffer_img.real_values,(const void*)real_values_gpu, real_memory_allocated*sizeof(float),cudaMemcpyDeviceToHost));
  bool OverWriteSlices = true;
  float pixelSize = 0.0f;

  buffer_img.QuickAndDirtyWriteSlices(filename, first_slice, last_slice, OverWriteSlices, pixelSize);
  buffer_img.Deallocate();
}

void GpuImage::SetCufftPlan(bool use_half_precision)
{

    int rank;
    long long int* fftDims;
    long long int* inembed;
    long long int* onembed;


    checkCudaErrors(cufftCreate(&cuda_plan_forward));
    checkCudaErrors(cufftCreate(&cuda_plan_inverse));

    checkCudaErrors(cufftSetStream(cuda_plan_forward, cudaStreamPerThread));
    checkCudaErrors(cufftSetStream(cuda_plan_forward, cudaStreamPerThread));

    if (dims.z > 1) 
    { 
      rank = 3;
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
      onembed[2] = dims.w/2; // Storage dimension (padded)

      
    }
    else if (dims.y > 1) 
    { 
      wxPrintf("\n\nAllocating a 2d Plan\n\n");
      rank = 2;
      fftDims = new long long int[rank];
      inembed = new long long int[rank];
      onembed = new long long int[rank];

      fftDims[0] = dims.y;
      fftDims[1] = dims.x;

      inembed[0] = dims.y;
      inembed[1] = dims.w;

      onembed[0] = dims.y;
      onembed[1] = dims.w/2;

    }    
    else 
    { 
      rank = 1; 
      fftDims = new long long int[rank];
      inembed = new long long int[rank];
      onembed = new long long int[rank];
  
      fftDims[0] = dims.x;

      inembed[0] = dims.w;
      onembed[0] = dims.w/2;
    }



    int iBatch = 1;

// As far as I can tell, the padded layout must be assumed and onembed/inembed
// are not needed. TODO ask John about this.

    if (use_half_precision)
    {
    	checkCudaErrors(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_R_16F,
    				  NULL, NULL, NULL, CUDA_C_16F, iBatch, &cuda_plan_worksize_forward, CUDA_C_16F));
    	checkCudaErrors(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_C_16F,
    				  NULL, NULL, NULL, CUDA_R_16F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_16F));
    }
    else
    {
    	checkCudaErrors(cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_R_32F,
    				  NULL, NULL, NULL, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F));
    	checkCudaErrors(cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
    				  NULL, NULL, NULL, CUDA_C_32F,
    				  NULL, NULL, NULL, CUDA_R_32F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_32F));
    }

//    cufftPlanMany(&dims.cuda_plan_forward, rank, fftDims, 
//                  inembed, iStride, iDist, 
//                  onembed, oStride, oDist, CUFFT_R2C, iBatch);
//    cufftPlanMany(&dims.cuda_plan_inverse, rank, fftDims, 
//                  onembed, oStride, oDist, 
//                  inembed, iStride, iDist, CUFFT_C2R, iBatch);

 

    delete [] fftDims;
    delete [] inembed;
    delete [] onembed;

	is_fft_planned = true;

  }  



void GpuImage::Deallocate()
{


  if (is_host_memory_pinned) cudaHostUnregister(real_values);

  cudaFree(tmpVal);
  cudaFree(tmpValComplex);

  if (is_fft_planned)
  {
    checkCudaErrors(cufftDestroy(cuda_plan_inverse));
    checkCudaErrors(cufftDestroy(cuda_plan_forward));
    is_fft_planned = false;
  }

//  if (is_cublas_loaded)
//  {
//    checkCudaErrors(cublasDestroy(cublasHandle));
//    is_cublas_loaded = false;
//  }

  if (is_allocated_mask_CSOS)
  {
    mask_CSOS->Deallocate();
  }

  if (is_allocated_image_buffer)
  {
    image_buffer->Deallocate();
  }

  if (is_allocated_sum_buffer) checkCudaErrors(cudaFree(this->sum_buffer)); is_allocated_sum_buffer = false;


}

void GpuImage::ConvertToHalfPrecision(bool deallocate_single_precision)
{

	// FIXME when adding real space complex images
	MyAssertTrue( is_in_memory_gpu, "Image is in not on the GPU!");

	BufferInit(b_16f);

	pre_checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);
	ReturnLaunchParamters(dims, false);
	ConvertToHalfPrecisionKernel<< <gridDims, threadsPerBlock,0, cudaStreamPerThread>> > (complex_values_gpu, complex_values_16f, this->dims, this->physical_upper_bound_complex);
	checkErrorsAndTimingWithSynchronization(cudaStreamPerThread);

	if (deallocate_single_precision)
	{
		checkCudaErrors(cudaFree(real_values_gpu));
		is_in_memory_gpu = false;
	}
}

__global__ void ConvertToHalfPrecisionKernel(cufftComplex* complex_32f_values, __half2* complex_16f_values, int4 dims, int3 physical_upper_bound_complex)
{

	int3 coords = make_int3(blockIdx.x*blockDim.x + threadIdx.x,
						  blockIdx.y*blockDim.y + threadIdx.y,
						  blockIdx.z);

	if (coords.x < dims.w / 2 && coords.y < dims.y && coords.z < dims.z)
	{

		int address = d_ReturnFourier1DAddressFromPhysicalCoord(coords, physical_upper_bound_complex);

		complex_16f_values[address] = __float22half2_rn(complex_32f_values[address]);
	}


}
void GpuImage::Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space)
{

	MyAssertTrue(wanted_x_size > 0 && wanted_y_size > 0 && wanted_z_size > 0,"Bad dimensions: %i %i %i\n",wanted_x_size,wanted_y_size,wanted_z_size);

	// check to see if we need to do anything?

	if (is_in_memory_gpu == true)
	{
		is_in_real_space = should_be_in_real_space;
		if (wanted_x_size == dims.x && wanted_y_size == dims.y && wanted_z_size == dims.z)
		{
			// everything is already done..
			is_in_real_space = should_be_in_real_space;
	//			wxPrintf("returning\n");
			return;
		}
		else
		{
		  Deallocate();
		}
	}

	SetupInitialValues();
	this->is_in_real_space = should_be_in_real_space;
	dims.x = wanted_x_size; dims.y = wanted_y_size; dims.z = wanted_z_size;

	// if we got here we need to do allocation..

	// first_x_dimension
	if (IsEven(wanted_x_size) == true) real_memory_allocated =  wanted_x_size / 2 + 1;
	else real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

	real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
	real_memory_allocated *= 2; // room for complex

	// TODO consider option to add host mem here. For now, just do gpu mem.
	//////	real_values = (float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
	//////	complex_values = (std::complex<float>*) real_values;  // Set the complex_values to point at the newly allocated real values;
//	wxPrintf("\n\n\tAllocating mem\t\n\n");
	checkCudaErrors(cudaMalloc(&real_values_gpu, real_memory_allocated*sizeof(cufftReal)));
	complex_values_gpu = (cufftComplex *)real_values_gpu;
	is_in_memory_gpu = true;

	// Update addresses etc..
	UpdateLoopingAndAddressing(wanted_x_size, wanted_y_size, wanted_z_size);

	if (IsEven(wanted_x_size) == true) padding_jump_value = 2;
	else padding_jump_value = 1;

	// record the full length ( pitch / 4 )
	dims.w = dims.x + padding_jump_value;
	pitch = dims.w * sizeof(float);

	number_of_real_space_pixels = int(dims.x) * int(dims.y) * int(dims.z);
	ft_normalization_factor = 1.0 / sqrtf(float(number_of_real_space_pixels));


	// Set other gpu vals

	is_host_memory_pinned = false;
	is_meta_data_initialized = true;

}

void GpuImage::UpdateBoolsToDefault()
{

	is_meta_data_initialized = false;

	is_in_memory = false;
	is_in_real_space = true;
	object_is_centred_in_box = true;
	image_memory_should_not_be_deallocated = false;

	is_in_memory_gpu = false;
	is_host_memory_pinned = false;

	// libraries
	is_fft_planned = false;
//	is_cublas_loaded = false;
	is_npp_loaded = false;

	// Buffers
	is_allocated_image_buffer = false;
	is_allocated_mask_CSOS = false;

	is_allocated_sum_buffer = false;
	is_allocated_min_buffer = false;
	is_allocated_minIDX_buffer = false;
	is_allocated_max_buffer = false;
	is_allocated_maxIDX_buffer = false;
	is_allocated_minmax_buffer = false;
	is_allocated_minmaxIDX_buffer = false;
	is_allocated_mean_buffer = false;
	is_allocated_meanstddev_buffer = false;
	is_allocated_countinrange_buffer = false;
	is_allocated_l2norm_buffer = false;
	is_allocated_dotproduct_buffer = false;
	is_allocated_16f_buffer = false;

	// Callbacks
	is_set_convertInputf16Tof32 = false;
	is_set_scaleFFTAndStore = false;
	is_set_complexConjMulLoad = false;
	is_set_realLoadAndClipInto = false;

}

//!>  \brief  Update all properties related to looping & addressing in real & Fourier space, given the current logical dimensions.

void GpuImage::UpdateLoopingAndAddressing(int wanted_x_size, int wanted_y_size, int wanted_z_size)
{


	dims.x = wanted_x_size;
	dims.y = wanted_y_size;
	dims.z = wanted_z_size;

	physical_address_of_box_center.x = wanted_x_size / 2;
	physical_address_of_box_center.y= wanted_y_size / 2;
	physical_address_of_box_center.z= wanted_z_size / 2;

	physical_upper_bound_complex.x= wanted_x_size / 2;
	physical_upper_bound_complex.y= wanted_y_size - 1;
	physical_upper_bound_complex.z= wanted_z_size - 1;


	//physical_index_of_first_negative_frequency.x= wanted_x_size / 2 + 1;
	if (IsEven(wanted_y_size) == true)
	{
		physical_index_of_first_negative_frequency.y= wanted_y_size / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency.y= wanted_y_size / 2 + 1;
	}

	if (IsEven(wanted_z_size) == true)
	{
		physical_index_of_first_negative_frequency.z= wanted_z_size / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency.z= wanted_z_size / 2 + 1;
	}


    // Update the Fourier voxel size

	fourier_voxel_size.x= 1.0 / double(wanted_x_size);
	fourier_voxel_size.y= 1.0 / double(wanted_y_size);
	fourier_voxel_size.z= 1.0 / double(wanted_z_size);

	// Logical bounds
	if (IsEven(wanted_x_size) == true)
	{
		logical_lower_bound_complex.x= -wanted_x_size / 2;
		logical_upper_bound_complex.x=  wanted_x_size / 2;
	    logical_lower_bound_real.x   = -wanted_x_size / 2;
	    logical_upper_bound_real.x   =  wanted_x_size / 2 - 1;
	}
	else
	{
		logical_lower_bound_complex.x= -(wanted_x_size-1) / 2;
		logical_upper_bound_complex.x=  (wanted_x_size-1) / 2;
		logical_lower_bound_real.x   = -(wanted_x_size-1) / 2;
		logical_upper_bound_real.x    =  (wanted_x_size-1) / 2;
	}


	if (IsEven(wanted_y_size) == true)
	{
	    logical_lower_bound_complex.y= -wanted_y_size / 2;
	    logical_upper_bound_complex.y=  wanted_y_size / 2 - 1;
	    logical_lower_bound_real.y   = -wanted_y_size / 2;
	    logical_upper_bound_real.y   =  wanted_y_size / 2 - 1;
	}
	else
	{
	    logical_lower_bound_complex.y= -(wanted_y_size-1) / 2;
	    logical_upper_bound_complex.y=  (wanted_y_size-1) / 2;
	    logical_lower_bound_real.y   = -(wanted_y_size-1) / 2;
	    logical_upper_bound_real.y    =  (wanted_y_size-1) / 2;
	}

	if (IsEven(wanted_z_size) == true)
	{
		logical_lower_bound_complex.z= -wanted_z_size / 2;
		logical_upper_bound_complex.z=  wanted_z_size / 2 - 1;
		logical_lower_bound_real.z   = -wanted_z_size / 2;
		logical_upper_bound_real.z   =  wanted_z_size / 2 - 1;

	}
	else
	{
		logical_lower_bound_complex.z= -(wanted_z_size - 1) / 2;
		logical_upper_bound_complex.z=  (wanted_z_size - 1) / 2;
		logical_lower_bound_real.z   = -(wanted_z_size - 1) / 2;
		logical_upper_bound_real.z    =  (wanted_z_size - 1) / 2;
	}
}



