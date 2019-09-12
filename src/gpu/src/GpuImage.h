/*
 * GpuImage.h
 *
 *  Created on: Jul 31, 2019
 *      Author: himesb
 */

#ifndef GPUIMAGE_H_
#define GPUIMAGE_H_


class GpuImage {

public:

	GpuImage();
	GpuImage( const GpuImage &other_gpu_image ); // copy constructor
	GpuImage(Image &cpu_image);
	virtual ~GpuImage();

	GpuImage & operator = (const GpuImage &t);
	GpuImage & operator = (const GpuImage *t);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// START MEMBER VARIABLES FROM THE cpu IMAGE CLASS

	int4 dims;
	bool 		 is_in_real_space;								// !< Whether the image is in real or Fourier space
	bool 		 object_is_centred_in_box;						//!<  Whether the object or region of interest is near the center of the box (as opposed to near the corners and wrapped around). This refers to real space and is meaningless in Fourier space.
	int3 physical_upper_bound_complex;
	int3 physical_address_of_box_center;
	int3 physical_index_of_first_negative_frequency;
	int3 logical_upper_bound_complex;
	int3 logical_lower_bound_complex;
	int3 logical_upper_bound_real;
	int3 logical_lower_bound_real;

	float3 fourier_voxel_size;


	long         real_memory_allocated;							// !<  Number of floats allocated in real space;
	int          padding_jump_value;                            // !<  The FFTW padding value, if odd this is 2, if even it is 1.  It is used in loops etc over real space.
	int			 insert_into_which_reconstruction;				// !<  Determines which reconstruction the image will be inserted into (for FSC calculation).

	long		 number_of_real_space_pixels;					// !<	Total number of pixels in real space
	float		 ft_normalization_factor;						// !<	Normalization factor for the Fourier transform (1/sqrt(N), where N is the number of pixels in real space)
	// Arrays to hold voxel values

	float 	 	 *real_values;									// !<  Real array to hold values for REAL images.
	std::complex<float> *complex_values;								// !<  Complex array to hold values for COMP images.
	bool         is_in_memory;                                  // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array. 
	bool         image_memory_should_not_be_deallocated;	    // !< Don't deallocate the memory, generally should only be used when doing something funky with the pointers
	int          gpu_plan_id;

	// end  MEMBER VARIABLES FROM THE cpu IMAGE CLASS
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Image*        hostImage;

	cufftReal 	 	*real_values_gpu;									// !<  Real array to hold values for REAL images.
	cufftComplex 	*complex_values_gpu;								// !<  Complex array to hold values for COMP images.

	__half* 	real_values_16f;
	__half2*	complex_values_16f;


	enum ImageType : size_t  { real16f = sizeof(__half), complex16f = sizeof(__half2), real32f = sizeof(float), complex32f = sizeof(float2), real64f = sizeof(double), complex64f = sizeof(double2) };
	ImageType img_type;


	bool        	is_in_memory_gpu;                                  // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array. Default = .FALSE.
	bool		 	is_host_memory_pinned;	 							// !<  Is the host memory already page locked (2x bandwith and required for asynchronous xfer);
	float*    pinnedPtr;


	cudaMemcpy3DParms h_3dparams = {0};
	cudaExtent h_extent;
	cudaPos    h_pos;
	cudaPitchedPtr h_pitchedPtr;

	cudaMemcpy3DParms d_3dparams = {0};
	cudaExtent d_extent;
	cudaPos    d_pos;
	cudaPitchedPtr d_pitchedPtr;

	size_t pitch;

	dim3 threadsPerBlock;
	dim3 gridDims;

	bool is_meta_data_initialized;
	float* tmpVal;
	float* tmpValComplex;
  

 ////////////////////////////////////////////////////////


	cudaEvent_t calcEvent, copyEvent;
	cublasHandle_t cublasHandle;

	cufftHandle cuda_plan_forward;
	cufftHandle cuda_plan_inverse;
	size_t	cuda_plan_worksize_forward;
	size_t	cuda_plan_worksize_inverse;

	//Stream for asynchronous command execution
	cudaStream_t calcStream;
	cudaStream_t copyStream;
	NppStreamContext nppStream;

	bool is_fft_planned;
	bool is_cublas_loaded;
	bool is_npp_loaded;
	cublasStatus_t cublas_stat;
	NppStatus npp_stat;

	// For the full image set width/height, otherwise set on function call.
	NppiSize npp_ROI;
	NppiSize npp_ROI_complex;



	////////////////////////////////////////////////////////////////////////
	///// Methods that should behave as their counterpart in the Image class
	///// have /**CPU_eq**/
	////////////////////////////////////////////////////////////////////////

	void QuickAndDirtyWriteSlices(std::string filename, int first_slice, int last_slice); /**CPU_eq**/
	void PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift);    /**CPU_eq**/
	void MultiplyByConstant(float scale_factor);                                          /**CPU_eq**/
	void Conj(); // FIXME
	void MultiplyPixelWise(GpuImage &other_image);                                        /**CPU_eq**/
	void MultiplyPixelWiseComplexConjugate(GpuImage &other_image);

	void SwapRealSpaceQuadrants();                                                        /**CPU_eq**/
	void ClipInto(GpuImage *other_image, float wanted_padding_value,                      /**CPU_eq**/
				bool fill_with_noise, float wanted_noise_sigma,
				int wanted_coordinate_of_box_center_x,
				int wanted_coordinate_of_box_center_y,
				int wanted_coordinate_of_box_center_z);
	void ForwardFFT(bool should_scale = true);                                           /**CPU_eq**/
	void BackwardFFT();                                                                   /**CPU_eq**/
	void ForwardFFTAndClipInto(GpuImage &image_to_insert, bool should_scale);
	template < typename T > void BackwardFFTAfterComplexConjMul(T* image_to_multiply, bool load_half_precision = false);


	float ReturnSumOfSquares();
	float ReturnAverageOfRealValuesOnEdges();
	void Deallocate();
	void ConvertToHalfPrecision(bool deallocate_single_precision = true);
	void Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space);
	// Combines this and UpdatePhysicalAddressOfBoxCenter and SetLogicalDimensions
	void UpdateLoopingAndAddressing(int wanted_x_size, int wanted_y_size, int wanted_z_size);


	////////////////////////////////////////////////////////////////////////
	///// Methods that do not have a counterpart in the image class
	////////////////////////////////////////////////////////////////////////

	void CopyHostToDevice();
	void CopyDeviceToHost(bool free_gpu_memory = true, bool unpin_host_memory = true);
	void CopyDeviceToHost(Image &cpu_image, bool should_block_until_complete = false, bool free_gpu_memory = true);
	// The volume copies with memory coalescing favoring padding are not directly
	// compatible with the memory layout in Image().
	void CopyVolumeHostToDevice();
	void CopyVolumeDeviceToHost(bool free_gpu_memory = true, bool unpin_host_memory = true);
	// Synchronize the full stream.
	void Wait();
	// Maximum intensity projection
	void MipPixelWise(GpuImage &other_image);
	void MipPixelWise(GpuImage &other_image, GpuImage &psi, GpuImage &phi, GpuImage &theta,
					float c_psi, float c_phi, float c_theta);
	void MipPixelWise(GpuImage &other_image, GpuImage &psi, GpuImage &phi, GpuImage &theta, GpuImage &defocus, GpuImage &pixel,
					float c_psi, float c_phi, float c_theta, float c_defocus, float c_pixel);



	void Init(Image &cpu_image);
	void SetCufftPlan(bool use_half_precision = false);
	void SetupInitialValues();
	void UpdateBoolsToDefault();

	__inline__  void ReturnLaunchParamters(int4 input_dims, bool real_space)
	{
	  int div;
	  if (real_space) { div = 1; } else { div = 2; }
	  threadsPerBlock = dim3(32, 32, 1);
	  gridDims = dim3((input_dims.w/div + threadsPerBlock.x - 1) / threadsPerBlock.x,
					  (input_dims.y     + threadsPerBlock.y - 1) / threadsPerBlock.y,
					   input_dims.z);
	};

	void CopyFromCpuImage(Image &cpu_image);
	void UpdateCpuFlags();
	void printVal(std::string msg, int idx);
	bool HasSameDimensionsAs(GpuImage *other_image);
	void Zeros();

	void Abs();
	void AbsDiff(GpuImage &other_image); // inplace
	void AbsDiff(GpuImage &other_image, GpuImage &output_image);
	void SquareRealValues();
	void SquareRootRealValues();
	void LogarithmRealValues();
	void ExponentiateRealValues();
	void AddConstant(const float add_val);
	void AddImage(GpuImage &other_image);
	void AddSquaredImage(GpuImage &other_image);

	// Statitical Methods
	float ReturnSumOfRealValues();
	NppiPoint min_idx; NppiPoint max_idx; float min_value; float max_value;
	float img_mean; float img_stdDev; Npp64f* npp_mean; Npp64f* npp_stdDev;
	int number_of_pixels_in_range;
	void Min();
	void MinAndCoords();
	void Max();
	void MaxAndCoords();
	void MinMax();
	void MinMaxAndCoords();
	void Mean();
	void MeanStdDev();
	void AverageError(const GpuImage &other_image); // TODO add me
	void AverageRelativeError(const GpuImage &other_image); // TODO addme
	void CountInRange(float lower_bound, float upper_bound);
	void HistogramEvenBins(); // TODO add me
	void HistogramDefinedBins(); // TODO add me

  
  // TODO
  /*

  Mean, Mean_StdDev 
  */
  

  ////////////////////////////////////////////////////////////////////////
  ///// Methods for creating or storing masks used for otherwise slow looping operations
  ////////////////////////////////////////////////////////////////////////

  enum BufferType : int  { b_image, b_sum, b_min, b_minIDX, b_max, b_maxIDX, b_minmax, b_minmaxIDX, b_mean, b_meanstddev,
  	  	  	  	  	  	   b_countinrange, b_histogram, b_16f };

  void CublasInit();
  void NppInit();
  void BufferInit(BufferType bt);



  // Real buffer = size real_values
  GpuImage* image_buffer; bool is_allocated_image_buffer;

  // Npp specific buffers;
  Npp8u* sum_buffer; 			bool is_allocated_sum_buffer;
  Npp8u* min_buffer; 			bool is_allocated_min_buffer;
  Npp8u* minIDX_buffer; 		bool is_allocated_minIDX_buffer;
  Npp8u* max_buffer; 			bool is_allocated_max_buffer;
  Npp8u* maxIDX_buffer; 		bool is_allocated_maxIDX_buffer;
  Npp8u* minmax_buffer; 		bool is_allocated_minmax_buffer;
  Npp8u* minmaxIDX_buffer; 		bool is_allocated_minmaxIDX_buffer;
  Npp8u* mean_buffer; 			bool is_allocated_mean_buffer;
  Npp8u* meanstddev_buffer; 	bool is_allocated_meanstddev_buffer;
  Npp8u* countinrange_buffer;	bool is_allocated_countinrange_buffer;
  	  	  	  	  	  	  	  	bool is_allocated_16f_buffer;
  	  	  	  	  	  	  	  	bool is_set_realLoadAndClipInto;

  
  GpuImage* mask_CSOS;   bool is_allocated_mask_CSOS;
  float ReturnSumSquareModulusComplexValues();
  
  // Callback related parameters
  bool is_set_convertInputf16Tof32;
  bool is_set_scaleFFTAndStore;
  bool is_set_complexConjMulLoad;

/*template void d_MultiplyByScalar<T>(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch);*/



private:


};

#endif /* GPUIMAGE_H_ */
