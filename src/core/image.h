/*  \brief  Image class (derived from Fortran images module)

	for information on actual data management / addressing see the image_data_array class..

*/

class ReconstructedVolume;
class EulerSearch;
class ResolutionStatistics;
class RotationMatrix;
class MyApp;

class Image {

public:

	int 		 logical_x_dimension;							// !< Logical (X) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 		 logical_y_dimension;							// !< Logical (Y) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 		 logical_z_dimension;							// !< Logical (Z) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).

	bool 		 is_in_real_space;								// !< Whether the image is in real or Fourier space
	bool 		 object_is_centred_in_box;						//!<  Whether the object or region of interest is near the center of the box (as opposed to near the corners and wrapped around). This refers to real space and is meaningless in Fourier space.

	int			 physical_upper_bound_complex_x;				// !< In each dimension, the upper bound of the complex image's physical addresses
	int			 physical_upper_bound_complex_y;				// !< In each dimension, the upper bound of the complex image's physical addresses
	int			 physical_upper_bound_complex_z;				// !< In each dimension, the upper bound of the complex image's physical addresses

	int      	 physical_address_of_box_center_x;				// !< In each dimension, the address of the pixel at the origin
	int      	 physical_address_of_box_center_y;				// !< In each dimension, the address of the pixel at the origin
	int      	 physical_address_of_box_center_z;				// !< In each dimension, the address of the pixel at the origin

	//int			 physical_index_of_first_negative_frequency_x;	// !<  In each dimension, the physical index of the first pixel which stores negative frequencies
	int			 physical_index_of_first_negative_frequency_y;	// !<  In each dimension, the physical index of the first pixel which stores negative frequencies
	int			 physical_index_of_first_negative_frequency_z;	// !<  In each dimension, the physical index of the first pixel which stores negative frequencies

	float  		 fourier_voxel_size_x;							// !<  Distance from Fourier voxel to Fourier voxel, expressed in reciprocal pixels
	float  		 fourier_voxel_size_y;							// !<  Distance from Fourier voxel to Fourier voxel, expressed in reciprocal pixels
	float  		 fourier_voxel_size_z;							// !<  Distance from Fourier voxel to Fourier voxel, expressed in reciprocal pixels

	int			 logical_upper_bound_complex_x;					// !<  In each dimension, the upper bound of the complex image's logical addresses
	int			 logical_upper_bound_complex_y;					// !<  In each dimension, the upper bound of the complex image's logical addresses
	int			 logical_upper_bound_complex_z;					// !<  In each dimension, the upper bound of the complex image's logical addresses

	int			 logical_lower_bound_complex_x;					// !<  In each dimension, the lower bound of the complex image's logical addresses
	int			 logical_lower_bound_complex_y;					// !<  In each dimension, the lower bound of the complex image's logical addresses
	int			 logical_lower_bound_complex_z;					// !<  In each dimension, the lower bound of the complex image's logical addresses

	int			 logical_upper_bound_real_x;					// !<  In each dimension, the upper bound of the real image's logical addresses
	int			 logical_upper_bound_real_y;					// !<  In each dimension, the upper bound of the real image's logical addresses
	int			 logical_upper_bound_real_z;					// !<  In each dimension, the upper bound of the real image's logical addresses

	int			 logical_lower_bound_real_x;					// !<  In each dimension, the lower bound of the real image's logical addresses
	int			 logical_lower_bound_real_y;					// !<  In each dimension, the lower bound of the real image's logical addresses
	int			 logical_lower_bound_real_z;					// !<  In each dimension, the lower bound of the real image's logical addresses


	long         real_memory_allocated;							// !<  Number of floats allocated in real space;

	int          padding_jump_value;                            // !<  The FFTW padding value, if odd this is 2, if even it is 1.  It is used in loops etc over real space.

	int			 insert_into_which_reconstruction;				// !<  Determines which reconstruction the image will be inserted into (for FSC calculation).

	long		 number_of_real_space_pixels;					// !<	Total number of pixels in real space
	float		 ft_normalization_factor;						// !<	Normalization factor for the Fourier transform (1/sqrt(N), where N is the number of pixels in real space)

	// Arrays to hold voxel values

	float 	 	 *real_values;									// !<  Real array to hold values for REAL images.
	std::complex<float> *complex_values;								// !<  Complex array to hold values for COMP images.
	bool         is_in_memory;                                  // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array. Default = .FALSE.


	// FFTW-specfic

	fftwf_plan 	 plan_fwd;										// !< FFTW plan for the image (fwd)
	fftwf_plan	 plan_bwd;										// !< FFTW plan for the image (bwd)
	bool      	 planned;										// !< Whether the plan has been setup by/for FFTW
	bool         image_memory_should_not_be_deallocated;	    // !< Don't deallocate the memory, generally should only be used when doing something funky with the pointers

	static wxMutex s_mutexProtectingFFTW;

	// Methods

	Image();
	Image( const Image &other_image); // copy constructor
	~Image();

	Image & operator = (const Image &t);
	Image & operator = (const Image *t);

	void SetupInitialValues();

	void Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size = 1, bool is_in_real_space = true, bool do_fft_planning = true);
	void Allocate(int wanted_x_size, int wanted_y_size, bool is_in_real_space = true);
	void Allocate(Image *image_to_copy_size_and_space_from);

	void AllocateAsPointingToSliceIn3D(Image *wanted3d, long wanted_slice);

	void Deallocate();

	int ReturnSmallestLogicalDimension();
	int ReturnLargestLogicalDimension();
	void SampleFFT(Image &sampled_image, int sample_rate);
	float ReturnSumOfSquares(float wanted_mask_radius = 0.0, float wanted_center_x = 0.0, float wanted_center_y = 0.0, float wanted_center_z = 0.0, bool invert_mask = false);
	float ReturnSumOfRealValues();
	float ReturnSigmaNoise();
	float ReturnSigmaNoise(Image &matching_projection, float mask_radius = 0.0);
	float ReturnImageScale(Image &matching_projection, float mask_radius = 0.0);
	float ReturnCorrelationCoefficientUnnormalized(Image &other_image, float wanted_mask_radius = 0.0);
	float ReturnBeamTiltSignificanceScore(Image calculated_beam_tilt);
	float ReturnPixelWiseProduct(Image &other_image);
	float GetWeightedCorrelationWithImage(Image &projection_image, int *bins, float signed_CC_limit);
	void PhaseFlipPixelWise(Image &other_image);
	void MultiplyPixelWiseReal(Image &other_image, bool absolute = false);
	void MultiplyPixelWise(Image &other_image);
	void ConjugateMultiplyPixelWise(Image &other_image);
	void ComputeFSCVectorized(Image *other_image, Image *work_this_image_squared, Image *work_other_image_squared, Image *work_cross_product_image, int number_of_shells, int *shell_number, float *computed_fsc, double *work_sum_of_squares, double *work_sum_of_other_squares, double *work_sum_of_cross_products);
	void ComputeFSC(Image *other_image, int number_of_shells, int *shell_number, float *computed_fsc, double *work_sum_of_squares, double *work_sum_of_other_squares, double *work_sum_of_cross_products);
	void DividePixelWise(Image &other_image);
	void AddGaussianNoise(float wanted_sigma_value = 1.0, RandomNumberGenerator *provided_generator = NULL);
	long ZeroFloat(float wanted_mask_radius = 0.0, bool outsize = false);
	long ZeroFloatAndNormalize(float wanted_sigma_value = 1.0, float wanted_mask_radius = 0.0, bool outside = false);
	long Normalize(float wanted_sigma_value = 1.0, float wanted_mask_radius = 0.0, bool outside = false);
	void NormalizeSumOfSquares();
	void ZeroFloatOutside(float wanted_mask_radius, bool invert_mask = false);
	void ReplaceOutliersWithMean(float maximum_n_sigmas);
	float ReturnVarianceOfRealValues(float wanted_mask_radius = 0.0, float wanted_center_x = 0.0, float wanted_center_y = 0.0, float wanted_center_z = 0.0, bool invert_mask = false);
	EmpiricalDistribution ReturnDistributionOfRealValues(float wanted_mask_radius = 0.0, bool outside = false, float wanted_center_x = 0.0, float wanted_center_y = 0.0, float wanted_center_z = 0.0);
	void UpdateDistributionOfRealValues(EmpiricalDistribution *distribution_to_update, float wanted_mask_radius = 0.0, bool outside = false, float wanted_center_x = 0.0, float wanted_center_y = 0.0, float wanted_center_z = 0.0);
	void ApplySqrtNFilter();
	void Whiten(float resolution_limit = 1.0, Curve *whitening_filter = NULL);
	void OptimalFilterBySNRImage(Image &SNR_image, int include_reference_weighting = 1);
	void MultiplyByWeightsCurve(Curve &weights, float scale_factor = 1.0);
	void WeightBySSNR(Image &ctf_image, float molecular_mass_kDa, float pixel_size, Curve &SSNR, Image &projection_image, bool weight_particle_image, bool weight_projection_image);
	void OptimalFilterSSNR(Curve &SSNR);
	void OptimalFilterFSC(Curve &FSC);
	void OptimalFilterWarp(CTF ctf, float pixel_size_in_angstroms, float ssnr_falloff_fudge_factor = 1.0, float ssnr_scale_fudge_factor = 1.0);
	//float Correct3D(float wanted_mask_radius = 0.0);
	float CorrectSinc(float wanted_mask_radius = 0.0, float padding_factor = 1.0, bool force_background_value = false, float wanted_mask_value = 0.0);
	void MirrorXFourier2D(Image &mirrored_image);
	void MirrorYFourier2D(Image &mirrored_image);
	void RotateQuadrants(Image &rotated_image, int quad_i);
	void Rotate3DByRotationMatrixAndOrApplySymmetry(RotationMatrix &wanted_matrix, float wanted_max_radius_in_pixels = 0.0, wxString wanted_symmetry="C1"); // use identiy matrix to just impose sym
	void Rotate3DByRotationMatrixAndOrApplySymmetryThenShift(RotationMatrix &wanted_matrix, float wanted_x_shift, float wanted_y_shift, float wanted_z_shift, float wanted_max_radius_in_pixels = 0.0, wxString wanted_symmetry = "C1"); // like above but with shift
	void Rotate3DThenShiftThenApplySymmetry(RotationMatrix &wanted_matrix, float wanted_x_shift, float wanted_y_shift, float wanted_z_shift, float wanted_max_radius_in_pixels = 0.0, wxString wanted_symmetry = "C1");
	void GenerateReferenceProjections(Image *projections, EulerSearch &parameters, float resolution);
	void RotateFourier2DGenerateIndex(Kernel2D **&kernel_index, float psi_max, float psi_step, float psi_start, bool invert_angle = false);
	void RotateFourier2DDeleteIndex(Kernel2D **&kernel_index, float psi_max, float psi_step);
	void RotateFourier2DFromIndex(Image &rotated_image, Kernel2D *kernel_index);
	void RotateFourier2DIndex(Kernel2D *kernel_index, AnglesAndShifts &rotation_angle, float resolution_limit = 1.0, float padding_factor = 1.0);
	Kernel2D ReturnLinearInterpolatedFourierKernel2D(float &x, float &y);
	void RotateFourier2D(Image &rotated_image, AnglesAndShifts &rotation_angle, float resolution_limit_in_reciprocal_pixels = 1.0, bool use_nearest_neighbor = false);
	void Rotate2D(Image &rotated_image, AnglesAndShifts &rotation_angle, float mask_radius_in_pixels = 0.0);
	void Rotate2DInPlace(float rotation_in_degrees, float mask_radius_in_pixels = 0.0);
	void Rotate2DInPlaceBy90Degrees(bool rotate_by_positive_90 = true);
	void Rotate2DSample(Image &rotated_image, AnglesAndShifts &rotation_angle, float mask_radius_in_pixels = 0.0);
	float Skew2D(Image &skewed_image, float height_offset, float minimum_height, float skew_axis, float skew_angle, bool adjust_signal = false);
	float ReturnLinearInterpolated2D(float &wanted_physical_x_coordinate, float &wanted_physical_y_coordinate);
	float ReturnNearest2D(float &wanted_physical_x_coordinate, float &wanted_physical_y_coordinate);
	void ExtractSlice(Image &image_to_extract, AnglesAndShifts &angles_and_shifts_of_image, float resolution_limit = 1.0, bool apply_resolution_limit = true);
	void ExtractSliceByRotMatrix(Image &image_to_extract, RotationMatrix &wanted_matrix, float resolution_limit = 1.0, bool apply_resolution_limit = true);
	std::complex<float> ReturnNearestFourier2D(float &x, float &y);
	std::complex<float> ReturnLinearInterpolatedFourier2D(float &x, float &y);
	std::complex<float> ReturnLinearInterpolatedFourier(float &x, float &y, float &z);
	void AddByLinearInterpolationReal(float &wanted_x_coordinate, float &wanted_y_coordinate, float &wanted_z_coordinate, float &wanted_value);
	void AddByLinearInterpolationFourier2D(float &wanted_x_coordinate, float &wanted_y_coordinate, std::complex<float> &wanted_value);
	float CosineRingMask(float wanted_inner_radius, float wanted_outer_radius, float wanted_mask_edge);
	float CosineMask(float wanted_mask_radius, float wanted_mask_edge, bool invert = false, bool force_mask_value = false, float wanted_mask_value = 0.0);
	float CosineRectangularMask(float wanted_mask_radius_x, float wanted_mask_radius_y, float wanted_mask_radius_z, float wanted_mask_edge, bool invert = false, bool force_mask_value = false, float wanted_mask_value = 0.0);
	void ConvertToAutoMask(float pixel_size, float outer_mask_radius_in_angstroms, float filter_resolution_in_angstroms, float rebin_value, bool auto_estimate_initial_bin_value = true, float wanted_initial_bin_value = 0.0f);
	void LocalResSignificanceFilter(float pixel_size, float starting_resolution, float mask_radius_in_angstroms);
	void GaussianLowPassFilter(float sigma);
	void GaussianHighPassFilter(float sigma);
	void ApplyLocalResolutionFilter(Image &local_resolution_map, float pixel_size, int wanted_number_of_levels);
	void CircleMask(float wanted_mask_radius, bool invert = false);
	void CircleMaskWithValue(float wanted_mask_radius, float wanted_mask_value, bool invert = false);
	void SquareMaskWithValue(float wanted_mask_dim, float wanted_mask_value, bool invert = false, int wanted_center_x = 0, int wanted_center_y = 0, int wanted_center_z = 0);
	void TriangleMask(float wanted_triangle_half_base_length);
	void CalculateCTFImage(CTF &ctf_of_image, bool calculate_complex_ctf = false, bool apply_coherence_envelope = false);
	void CalculateBeamTiltImage(CTF &ctf_of_image, bool output_phase_shifts = false);
	bool ContainsBlankEdges(float mask_radius = 0.0);
	void CorrectMagnificationDistortion(float distortion_angle, float distortion_major_axis, float distortion_minor_axis);
	float ApplyMask(Image &mask_file, float cosine_edge_width, float weight_outside_mask, float low_pass_filter_outside, float filter_cosine_edge_width, float outside_mask_value = 0.0, bool use_outside_mask_value = false);
	Peak CenterOfMass(float threshold = 0.0, bool apply_threshold = false);
	Peak StandardDeviationOfMass(float threshold = 0.0, bool apply_threshold = false, bool invert_densities = false);
	float ReturnAverageOfMaxN(int number_of_pixels_to_average = 100, float mask_radius = 0.0);
	float ReturnAverageOfMinN(int number_of_pixels_to_average = 100, float mask_radius = 0.0);

	void AddSlices(Image &sum_of_slices, int first_slice = 0, int last_slice = 0, bool calculate_average = false);

	float FindBeamTilt(CTF &input_ctf, float pixel_size, Image &phase_error_output, Image &beamtilt_output, Image &difference_image, float &beamtilt_x, float &beamtilt_y, float &particle_shift_x, float &particle_shift_y, float phase_multiplier, bool progress_bar, int first_position_to_search = 0, int last_position_to_search = INT_MAX, MyApp *app_for_result = NULL);

	inline long ReturnVolumeInRealSpace()
	{
		return number_of_real_space_pixels;
	};

	inline long ReturnReal1DAddressFromPhysicalCoord(int wanted_x, int wanted_y, int wanted_z)
	{
		MyDebugAssertTrue(wanted_x >= 0 && wanted_x < logical_x_dimension && wanted_y >= 0 && wanted_y < logical_y_dimension && wanted_z >= 0 && wanted_z < logical_z_dimension, "Requested pixel (%i, %i, %i) is outside range (%i-%i, %i-%i, %i-%i)", wanted_x, wanted_y, wanted_z,0,logical_x_dimension-1,0,logical_y_dimension-1,0,logical_z_dimension-1);

		return ((long(logical_x_dimension + padding_jump_value) * long(logical_y_dimension)) * long(wanted_z)) + (long(logical_x_dimension + padding_jump_value) * long(wanted_y)) + long(wanted_x);

	};

	inline float ReturnRealPixelFromPhysicalCoord(int wanted_x, int wanted_y, int wanted_z)
	{
		MyDebugAssertTrue(is_in_real_space == true,  "Requested real pixel, but image is in Fourier space");
		return real_values[ReturnReal1DAddressFromPhysicalCoord(wanted_x, wanted_y, wanted_z)];
	};

	inline long ReturnFourier1DAddressFromPhysicalCoord(int wanted_x, int wanted_y, int wanted_z)
	{
		MyDebugAssertTrue(wanted_x >= 0 && wanted_x <= physical_address_of_box_center_x && wanted_y >= 0 && wanted_y <= physical_upper_bound_complex_y && wanted_z >= 0 && wanted_z <= physical_upper_bound_complex_z, "Address (%i %i %i) out of bounds (%i to %i; %i to %i; %i to %i)!",wanted_x,wanted_y,wanted_z,0,physical_upper_bound_complex_x,0,physical_upper_bound_complex_y,0,physical_upper_bound_complex_z );
		return ((long(physical_upper_bound_complex_x + 1) * long(physical_upper_bound_complex_y + 1)) * long(wanted_z)) + (long(physical_upper_bound_complex_x + 1) * long(wanted_y)) + long(wanted_x);
	};

	inline long ReturnFourier1DAddressFromLogicalCoord(int wanted_x, int wanted_y, int wanted_z)
	{
		MyDebugAssertTrue(wanted_x >= logical_lower_bound_complex_x && wanted_x <=logical_upper_bound_complex_x && wanted_y >= logical_lower_bound_complex_y && wanted_y <= logical_upper_bound_complex_y && wanted_z >= logical_lower_bound_complex_z && wanted_z <= logical_upper_bound_complex_z, "Logical coordinates (%i, %i, %i) are out of bounds (%i to %i ; %i to %i; %i to %i)\n", wanted_x, wanted_y, wanted_z, logical_lower_bound_complex_x, logical_upper_bound_complex_x, logical_lower_bound_complex_y, logical_upper_bound_complex_y, logical_lower_bound_complex_z, logical_upper_bound_complex_z)

		int physical_x_address;
		int physical_y_address;
		int physical_z_address;

		if (wanted_x >= 0)
		{
			physical_x_address = wanted_x;

			if (wanted_y >= 0)
			{
				physical_y_address = wanted_y;
			}
			else
			{
				physical_y_address = logical_y_dimension + wanted_y;
			}

			if (wanted_z >= 0)
			{
				physical_z_address = wanted_z;
			}
			else
			{
				physical_z_address = logical_z_dimension + wanted_z;
			}
		}
		else
		{
			physical_x_address = -wanted_x;

			if (wanted_y > 0)
			{
				physical_y_address = logical_y_dimension - wanted_y;
			}
			else
			{
				physical_y_address = -wanted_y;
			}

			if (wanted_z > 0)
			{
				physical_z_address = logical_z_dimension - wanted_z;
			}
			else
			{
				physical_z_address = -wanted_z;
			}
		}

		return ReturnFourier1DAddressFromPhysicalCoord(physical_x_address, physical_y_address, physical_z_address);
	};

	std::complex<float> ReturnComplexPixelFromLogicalCoord(int wanted_x, int wanted_y, int wanted_z, std::complex<float> out_of_bounds_value)
	{
		if (wanted_x < logical_lower_bound_complex_x || wanted_x > logical_upper_bound_complex_x || wanted_y < logical_lower_bound_complex_y ||wanted_y > logical_upper_bound_complex_y || wanted_z < logical_lower_bound_complex_z || wanted_z > logical_upper_bound_complex_z)
		{

			return out_of_bounds_value;
		}
		else return complex_values[ReturnFourier1DAddressFromLogicalCoord(wanted_x, wanted_y, wanted_z)];
	};

	bool HasSameDimensionsAs(Image *other_image);
	inline bool IsCubic()
	{
		return (logical_x_dimension == logical_y_dimension && logical_x_dimension == logical_z_dimension);
	}
	inline bool IsSquare()
	{
		MyDebugAssertTrue(logical_z_dimension == 1, "Image is three-dimensional");
		return (logical_x_dimension == logical_y_dimension);
	}

	inline void ReturnCosineMaskBandpassResolution(float pixel_size_in_angstrom, float &wanted_cutoff_in_angstrom, float &wanted_falloff_in_number_of_fourier_space_voxels)
	{
		// For example, if you want a cutoff at 2 Angstrom res, with a 14 pixel fall off, and your image is at 1.2 apix, inputs will be 1.2, 2, 14
		// output in the last two args will be the mask_outer_radius, and mask_edge that are passed to CosineRingMask.
		  wanted_cutoff_in_angstrom = 0.5f*(pixel_size_in_angstrom*2.0f/wanted_cutoff_in_angstrom);

		  if (logical_z_dimension > 1)
		  {
			  wanted_falloff_in_number_of_fourier_space_voxels = wanted_falloff_in_number_of_fourier_space_voxels/3.0f*(fourier_voxel_size_x + fourier_voxel_size_y + fourier_voxel_size_z);
		  }
		  else
		  {
			  wanted_falloff_in_number_of_fourier_space_voxels = wanted_falloff_in_number_of_fourier_space_voxels/2.0f*(fourier_voxel_size_x + fourier_voxel_size_y);
		  }

	}

	bool IsBinary();

	void SetLogicalDimensions(int wanted_x_size, int wanted_y_size, int wanted_z_size = 1);
	void UpdateLoopingAndAddressing();
	void UpdatePhysicalAddressOfBoxCenter();

	int ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index);
	int ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index);
	int ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index);

	float ReturnMaximumDiagonalRadius();

	bool FourierComponentHasExplicitHermitianMate(int physical_index_x, int physical_index_y, int physical_index_z);
	bool FourierComponentIsExplicitHermitianMate(int physical_index_x, int physical_index_y, int physical_index_z);


	inline void NormalizeFT() { MultiplyByConstant(ft_normalization_factor); }
	inline void NormalizeFTAndInvertRealValues() { MultiplyByConstant(-ft_normalization_factor); }
	void DivideByConstant(float constant_to_divide_by);
	void MultiplyByConstant(float constant_to_multiply_by);
	void InvertRealValues();
	void TakeReciprocalRealValues(float zeros_become = 0.0);
	void AddConstant(float constant_to_add);
	void MultiplyAddConstant(float constant_to_multiply_by, float constant_to_add);
	void AddMultiplyConstant(float constant_to_add, float constant_to_multiply_by);
	void AddMultiplyAddConstant(float first_constant_to_add, float constant_to_multiply_by, float second_constant_to_add);
	void SquareRealValues();
	void SquareRootRealValues();
	void ExponentiateRealValues();

	long ReturnNumberofNonZeroPixels();

	void ForwardFFT(bool should_scale = true);
	void BackwardFFT();

	void AddFFTWPadding();
	void RemoveFFTWPadding();

	inline void ReadSlice(MRCFile *input_file, long slice_to_read) { MyDebugAssertTrue(slice_to_read > 0, "Start slice is 0, the first slice is 1!");	MyDebugAssertTrue(slice_to_read <= input_file->ReturnNumberOfSlices(), "End slice (%li) is greater than number of slices in the file! (%i)", slice_to_read, input_file->ReturnNumberOfSlices());ReadSlices(input_file, slice_to_read, slice_to_read);}; //!> \brief Read a a slice from disk..(this just calls ReadSlices)
	inline void ReadSlice(DMFile *input_file, long slice_to_read) { MyDebugAssertTrue(slice_to_read > 0, "Start slice is 0, the first slice is 1!");	MyDebugAssertTrue(slice_to_read <= input_file->ReturnNumberOfSlices(), "End slice (%li) is greater than number of slices in the file! (%i)", slice_to_read, input_file->ReturnNumberOfSlices());ReadSlices(input_file, slice_to_read, slice_to_read);}; //!> \brief Read a a slice from disk..(this just calls ReadSlices)
	inline void ReadSlice(EerFile *input_file, long slice_to_read) { MyDebugAssertTrue(slice_to_read > 0, "Start slice is 0, the first slice is 1!");	MyDebugAssertTrue(slice_to_read <= input_file->ReturnNumberOfSlices(), "End slice (%li) is greater than number of slices in the file! (%i)", slice_to_read, input_file->ReturnNumberOfSlices());ReadSlices(input_file, slice_to_read, slice_to_read);}; //!> \brief Read a a slice from disk..(this just calls ReadSlices)
	inline void ReadSlice(ImageFile *input_file, long slice_to_read) { MyDebugAssertTrue(slice_to_read > 0, "Start slice is 0, the first slice is 1!");	MyDebugAssertTrue(slice_to_read <= input_file->ReturnNumberOfSlices(), "End slice (%li) is greater than number of slices in the file! (%i)", slice_to_read, input_file->ReturnNumberOfSlices());ReadSlices(input_file, slice_to_read, slice_to_read);}; //!> \brief Read a a slice from disk..(this just calls ReadSlices)
	void ReadSlices(MRCFile *input_file, long start_slice, long end_slice);
	void ReadSlices(DMFile *input_file, long start_slice, long end_slice);
	void ReadSlices(EerFile *input_file, long start_slice, long end_slice);
	void ReadSlices(ImageFile *input_file, long start_slice, long end_slice);

	inline void WriteSlice(MRCFile *input_file, long slice_to_write) {  MyDebugAssertTrue(slice_to_write > 0, "Start slice is 0, the first slice is 1!"); WriteSlices(input_file, slice_to_write, slice_to_write);}
	void WriteSlices(MRCFile *input_file, long start_slice, long end_slice);
	void WriteSlicesAndFillHeader(std::string wanted_filename, float wanted_pixel_size);

	void QuickAndDirtyWriteSlices(std::string filename, long first_slice_to_write, long last_slice_to_write, bool overwrite = false, float pixel_size = 0.0f);
	void QuickAndDirtyWriteSlice(std::string filename, long slice_to_write, bool overwrite = false, float pixel_size = 0.0f);
	void QuickAndDirtyReadSlice(std::string filename, long slice_to_read);
	void QuickAndDirtyReadSlices(std::string filename, int first_slice_to_read, int last_slice_to_read);

	bool IsConstant(bool compare_to_constant = false, float constant_to_compare = 0.0f);
	bool HasNan();
	bool HasNegativeRealValue();
	void SetToConstant(float wanted_value);
	void ClipIntoLargerRealSpace2D(Image *other_image, float wanted_padding_value = 0);
	void ClipInto(Image *other_image, float wanted_padding_value = 0.0, bool fill_with_noise = false, float wanted_noise_sigma = 1.0,int wanted_coordinate_of_box_center_x=0, int wanted_coordinate_of_box_center_y=0, int wanted_coordinate_of_box_center_z=0);
	void ChangePixelSize(Image *other_image, float wanted_factor, float wanted_tolerance, bool return_fft = false);
	void InsertOtherImageAtSpecifiedPosition(Image *other_image, int wanted_x_coord, int wanted_y_coord, int wanted_z_coord, float threshold_value = -FLT_MAX);
	void Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value = 0);
	void RealSpaceBinning(int bin_x, int bin_y, int bin_z = 1, bool symmetrical = false, bool exclude_incomplete_bins = false);
	float ReturnVarianceOfRealValuesTiled(int bin_x, int bin_y, int bin_z = 1, bool exclude_incomplete_bins = false);
	void CopyFrom(Image *other_image);
	void CopyLoopingAndAddressingFrom(Image *other_image);
	void Consume(Image *other_image);
	void RealSpaceIntegerShift(int wanted_x_shift, int wanted_y_shift, int wanted_z_shift = 0);
	void DilateBinarizedMask(float dilation_radius);
	void ErodeBinarizedMask(float erosion_radius);

	void PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift = 0.0);

	void MakeAbsolute();

	void AddImage(Image *other_image);
	void SubtractImage(Image *other_image);
	void SubtractSquaredImage(Image *other_image);
	void ApplyBFactor(float bfactor);
	void ApplyBFactorAndWhiten(Curve &power_spectrum, float bfactor_low, float bfactor_high, float bfactor_res_limit);
	void CalculateDerivative(float direction_in_x = 0.0f, float direction_in_y = 0.0f, float direction_in_z = 0.0f);
	void SharpenMap(float pixel_size, float resolution_limit,  bool invert_hand = false, float inner_mask_radius = 0.0f, float outer_mask_radius = 100.0f, float start_res_for_whitening = 8.0f, float additional_bfactor_low = 0.0f, float additional_bfactor_high = 0.0f, float filter_edge = 20.0f, bool should_auto_mask = true,  Image *input_mask = NULL, ResolutionStatistics *resolution_statistics = NULL, float statistics_scale_factor = 1.0f, Curve *original_log_plot = NULL, Curve *sharpened_log_plot = NULL);
	void InvertHandedness();
	void ApplyCTFPhaseFlip(CTF ctf_to_apply);
	void ApplyCTF(CTF ctf_to_apply, bool absolute = false, bool apply_beam_tilt = false, bool apply_envelope = false);
	void ApplyCurveFilter(Curve *filter_to_apply, float resolution_limit = 1.0);
	void ApplyCurveFilterUninterpolated(Curve *filter_to_apply, float resolution_limit = 1.0f, float scale = 0.0f);
	void MaskCentralCross(int vertical_half_width = 1, int horizontal_half_width = 1);
	void ZeroCentralPixel();
	void CalculateCrossCorrelationImageWith(Image *other_image);
	void SwapRealSpaceQuadrants();
	void ComputeAmplitudeSpectrumFull2D(Image *other_image, bool calculate_phases = false, float phase_multiplier = 1.0f);
	void ComputeFilteredAmplitudeSpectrumFull2D(Image* average_spectrum_masked, Image* current_power_spectrum, float& average, float& sigma, float minimum_resolution, float maximum_resolution, float pixel_size_for_fitting);
	void ComputeAmplitudeSpectrum(Image *other_image, bool signed_values = false);
	void ComputeHistogramOfRealValuesCurve(Curve *histogram_curve);
	void Compute1DAmplitudeSpectrumCurve(Curve *curve_with_average_power, Curve *curve_with_number_of_values);
	void Compute1DPowerSpectrumCurve(Curve *curve_with_average_power, Curve *curve_with_number_of_values, bool average_amplitudes_not_intensities = false);
	void Compute1DRotationalAverage(Curve &average, Curve &number_of_values, bool fractional_radius_in_real_space = false, bool average_real_parts = false);
	void ComputeSpatialFrequencyAtEveryVoxel();
	void AverageRadially();
	void ComputeLocalMeanAndVarianceMaps(Image *local_mean_map, Image *local_variance_map, Image *mask, long number_of_pixels_within_mask);
	void SpectrumBoxConvolution(Image *output_image, int box_size, float minimum_radius);
	void TaperEdges();
	float ReturnAverageOfRealValues(float wanted_mask_radius = 0.0, bool invert_mask = false);
	float ReturnMedianOfRealValues();
	float ReturnAverageOfRealValuesOnEdges();
	float ReturnAverageOfRealValuesAtRadius(float wanted_mask_radius);
	float ReturnAverageOfRealValuesInRing(float wanted_inner_radius,float wanted_outer_radius);
	float ReturnSigmaOfFourierValuesOnEdges();
	float ReturnSigmaOfFourierValuesOnEdgesAndCorners();
	float ReturnMaximumValue(float minimum_distance_from_center = 0.0, float minimum_distance_from_edge = 0.0);
	float ReturnMinimumValue(float minimum_distance_from_center = 0.0, float minimum_distance_from_edge = 0.0);
	void SetMaximumValue(float new_maximum_value);
	void SetMinimumValue(float new_minimum_value);
	void SetMinimumAndMaximumValues( float new_minimum_value, float new_maximum_value);
	void Binarise(float threshold_value);
	void BinariseInverse(float threshold_value);


	void ComputeAverageAndSigmaOfValuesInSpectrum(float minimum_radius, float maximum_radius, float &average, float &sigma, int cross_half_width = 2);
	void SetMaximumValueOnCentralCross(float maximum_value);
	void ApplyMirrorAlongY();
	void InvertPixelOrder();

	void GetMinMax(float &min_value, float &max_value);

	void RandomisePhases(float wanted_radius_in_reciprocal_pixels);

	float ReturnCorrelationBetweenTwoHorizontalLines(int first_line, int second_line); // for relion repeated line edge detection
	float ReturnCorrelationBetweenTwoVerticalLines(int first_line, int second_line);
	bool  ContainsRepeatedLineEdges();

	float GetCorrelationWithCTF(CTF ctf);
	void SetupQuickCorrelationWithCTF(CTF ctf, int &number_of_values, double &norm_image, double &image_mean, int *addresses, float *spatial_frequency_squared, float *azimuth);
	float QuickCorrelationWithCTF(CTF ctf, int number_of_values, double norm_image, double image_mean, int *addresses, float *spatial_frequency_squared, float *azimuth);
	float ReturnIcinessOfSpectrum(float pixel_size_in_Angstroms);

	// Interpolation
	void GetRealValueByLinearInterpolationNoBoundsCheckImage(float &x, float &y, float &interpolated_value);


	Peak FindPeakAtOriginFast2D(int max_pix_x, int max_pix_y);
	Peak FindPeakWithIntegerCoordinates(float wanted_min_radius = 0.0, float wanted_max_radius = FLT_MAX,  int wanted_min_distance_from_edges = 0);
	Peak FindPeakWithParabolaFit(float wanted_min_radius = 0.0, float wanted_max_radius = FLT_MAX);

	void SubSampleWithNoisyResampling(Image *first_sampled_image, Image *second_sampled_image);
	void SubSampleMask(Image *first_sampled_image, Image *second_sampled_image);
	// Test patterns
	void Sine1D(int number_of_periods);

	// for displaying
	void CreateOrthogonalProjectionsImage(Image *image_to_create, bool include_projections = true, float scale_factor = 1.0f, float mask_radius_in_pixels = 0.0f);
};


class BeamTiltScorer
{
	CTF *pointer_to_ctf_to_use_for_calculation;
	Image *pointer_binarised_phase_difference_spectrum;


	Image beamtilt_spectrum;
	float pixel_size;
	float mask_radius;
	float phase_multiplier;

public:

	double ScoreValues(double []);
	BeamTiltScorer(CTF *pointer_to_wanted_ctf, Image *pointer_to_wanted_binarized_phase_diff_spectrum, float wanted_pixel_size, float wanted_mask_radius, float wanted_phase_multiplier);

	Image temp_image; // to hold experimental beam tilt
};
