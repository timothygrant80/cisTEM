/*  \brief  Image class (derived from Fortran images module)

	for information on actual data management / addressing see the image_data_array class..

*/

class ReconstructedVolume;
class EulerSearch;

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

	// Arrays to hold voxel values

	float 	 	 *real_values;									// !<  Real array to hold values for REAL images.
	fftwf_complex *complex_values;								// !<  Complex array to hold values for COMP images.
	bool         is_in_memory;                                  // !<  Whether image values are in-memory, in other words whether the image has memory space allocated to its data array. Default = .FALSE.


	// FFTW-specfic

	fftwf_plan 	 plan_fwd;										// !< FFTW plan for the image (fwd)
	fftwf_plan	 plan_bwd;										// !< FFTW plan for the image (bwd)
	bool      	 planned;										// !< Whether the plan has been setup by/for FFTW

	// Methods

	Image();
	Image( const Image &other_image); // copy constructor
	~Image();

	Image & operator = (const Image &t);
	Image & operator = (const Image *t);

	void Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size = 1, bool is_in_real_space = true);
	void Allocate(int wanted_x_size, int wanted_y_size, bool is_in_real_space = true);
	void Deallocate();

	int ReturnSmallestLogicalDimension();
	int ReturnLargestLogicalDimension();
	void SampleFFT(Image &sampled_image, int sample_rate);
	float ReturnSumOfSquares();
	float ReturnSigmaNoise(Image &matching_projection, float mask_radius = 0.0);
	float ReturnImageScale(Image &matching_projection, float mask_radius = 0.0);
	float ReturnCorrelationCoefficientUnnormalized(Image &other_image, float wanted_mask_radius = 0.0);
	float GetWeightedCorrelationWithImage(Image &projection_image, int *bins);
	void PhaseFlipPixelWise(Image &other_image);
	void MultiplyPixelWiseReal(Image &other_image);
	void MultiplyPixelWise(Image &other_image);
	void AddGaussianNoise(float wanted_sigma_value = 1.0);
	void Normalize(float wanted_sigma_value = 1.0, float wanted_mask_radius = 0.0);
	void ReplaceOutliersWithMean(float maximum_n_sigmas);
	float ReturnVarianceOfRealValues(float wanted_mask_radius = 0.0, float wanted_center_x = 0.0, float wanted_center_y = 0.0, float wanted_center_z = 0.0);
	void ApplySqrtNFilter();
	void WhitenTwo(Image &other_image);
	void Whiten();
	void OptimalFilterBySNRImage(Image &SNR_image);
	void MultiplyByWeightsCurve(Curve &weights);
	void OptimalFilterSSNR(Curve &SSNR);
	void OptimalFilterFSC(Curve &FSC);
	float Correct3D(float mask_radius = 0.0);
	void MirrorXFourier2D(Image &mirrored_image);
	void MirrorYFourier2D(Image &mirrored_image);
	void RotateQuadrants(Image &rotated_image, int quad_i);
	void GenerateReferenceProjections(Image *projections, EulerSearch &parameters);
	void RotateFourier2DGenerateIndex(Kernel2D **&kernel_index, float psi_max, float psi_step);
	void RotateFourier2DDeleteIndex(Kernel2D **&kernel_index, float psi_max, float psi_step);
	void RotateFourier2DFromIndex(Image &rotated_image, Kernel2D *kernel_index);
	void RotateFourier2DIndex(Kernel2D *kernel_index, AnglesAndShifts &rotation_angle, float resolution_limit = 1.0, float padding_factor = 1.0);
	Kernel2D ReturnLinearInterpolatedFourierKernel2D(float &x, float &y);
	void RotateFourier2D(Image &rotated_image, AnglesAndShifts &rotation_angle, float resolution_limit = 1.0, bool use_nearest_neighbor = false);
	void ExtractSlice(Image &image_to_extract, AnglesAndShifts &angles_and_shifts_of_image, float resolution_limit = 1.0);
	fftwf_complex ReturnNearestFourier2D(float &x, float &y);
	fftwf_complex ReturnLinearInterpolatedFourier2D(float &x, float &y);
	fftwf_complex ReturnLinearInterpolatedFourier(float &x, float &y, float &z);
	void AddByLinearInterpolationReal(float &wanted_x_coordinate, float &wanted_y_coordinate, float &wanted_z_coordinate, float &wanted_value);
	void AddByLinearInterpolationFourier2D(float &wanted_x_coordinate, float &wanted_y_coordinate, fftwf_complex &wanted_value);
	float CosineRingMask(float wanted_inner_radius, float wanted_outer_radius, float wanted_mask_edge);
	float CosineMask(float wanted_mask_radius, float wanted_mask_edge, bool invert = false);
	void CircleMask(float wanted_mask_radius, bool invert = false);
	void CalculateCTFImage(CTF &ctf_of_image);

	inline long ReturnReal1DAddressFromPhysicalCoord(int wanted_x, int wanted_y, int wanted_z)
	{
		MyDebugAssertTrue(wanted_x >= 0 && wanted_x < logical_x_dimension && wanted_y >= 0 && wanted_y < logical_y_dimension && wanted_z >= 0 && wanted_z < logical_z_dimension, "Requested pixel (%i, %i, %i) is outside range (%i-%i, %i-%i, %i-%i)", wanted_x, wanted_y, wanted_z,0,logical_x_dimension-1,0,logical_y_dimension-1,0,logical_z_dimension-1);

		return (((logical_x_dimension + padding_jump_value) * logical_y_dimension) * wanted_z) + ((logical_x_dimension + padding_jump_value) * wanted_y) + wanted_x;

	};

	inline float ReturnRealPixelFromPhysicalCoord(int wanted_x, int wanted_y, int wanted_z)
	{
		MyDebugAssertTrue(is_in_real_space == true,  "Requested real pixel, but image is in Fourier space");
		return real_values[ReturnReal1DAddressFromPhysicalCoord(wanted_x, wanted_y, wanted_z)];
	};


	inline long ReturnFourier1DAddressFromPhysicalCoord(int wanted_x, int wanted_y, int wanted_z)
	{
		MyDebugAssertTrue(wanted_x >= 0 && wanted_x <= physical_address_of_box_center_x && wanted_y >= 0 && wanted_y <= physical_upper_bound_complex_y && wanted_z >= 0 && wanted_z <= physical_upper_bound_complex_z, "Address (%i %i %i) out of bounds (%i to %i; %i to %i; %i to %i)!",wanted_x,wanted_y,wanted_z,0,physical_upper_bound_complex_x,0,physical_upper_bound_complex_y,0,physical_upper_bound_complex_z );
		return (((physical_upper_bound_complex_x + 1) * (physical_upper_bound_complex_y + 1)) * wanted_z) + ((physical_upper_bound_complex_x + 1) * wanted_y) + wanted_x;
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

	fftw_complex ReturnComplexPixelFromLogicalCoord(int wanted_x, int wanted_y, int wanted_z, float out_of_bounds_value)
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

	void SetLogicalDimensions(int wanted_x_size, int wanted_y_size, int wanted_z_size = 1);
	void UpdateLoopingAndAddressing();
	void UpdatePhysicalAddressOfBoxCenter();

	int ReturnFourierLogicalCoordGivenPhysicalCoord_X(int physical_index);
	int ReturnFourierLogicalCoordGivenPhysicalCoord_Y(int physical_index);
	int ReturnFourierLogicalCoordGivenPhysicalCoord_Z(int physical_index);

	int ReturnMaximumDiagonalRadius();

	bool FourierComponentHasExplicitHermitianMate(int physical_index_x, int physical_index_y, int physical_index_z);

	void DivideByConstant(float constant_to_divide_by);
	void MultiplyByConstant(float constant_to_multiply_by);
	void AddConstant(float constant_to_add);

	void ForwardFFT(bool should_scale = true);
	void BackwardFFT();

	void AddFFTWPadding();
	void RemoveFFTWPadding();

	inline void ReadSlice(MRCFile *input_file, long slice_to_read) { MyDebugAssertTrue(slice_to_read > 0, "Start slice is 0, the first slice is 1!");	MyDebugAssertTrue(slice_to_read <= input_file->ReturnNumberOfSlices(), "End slice is greater than number of slices in the file!");ReadSlices(input_file, slice_to_read, slice_to_read);}; //!> \brief Read a a slice from disk..(this just calls ReadSlices)
	void ReadSlices(MRCFile *input_file, long start_slice, long end_slice);
	void ReadSlices(DMFile *input_file, long start_slice, long end_slice);

	inline void WriteSlice(MRCFile *input_file, long slice_to_write) {  MyDebugAssertTrue(slice_to_write > 0, "Start slice is 0, the first slice is 1!"); WriteSlices(input_file, slice_to_write, slice_to_write);}
	void WriteSlices(MRCFile *input_file, long start_slice, long end_slice);

	void QuickAndDirtyWriteSlices(std::string filename, long first_slice_to_write, long last_slice_to_write);
	void QuickAndDirtyWriteSlice(std::string filename, long slice_to_write);
	void QuickAndDirtyReadSlice(std::string filename, long slice_to_read);

	bool IsConstant();
	void SetToConstant(float wanted_value);
	void ClipInto(Image *other_image, float wanted_padding_value = 0);
	void Resize(int wanted_x_dimension, int wanted_y_dimension, int wanted_z_dimension, float wanted_padding_value = 0);
	void CopyFrom(Image *other_image);
	void CopyLoopingAndAddressingFrom(Image *other_image);
	void Consume(Image *other_image);
	void PhaseShift(float wanted_x_shift, float wanted_y_shift, float wanted_z_shift = 0.0);

	void AddImage(Image *other_image);
	void SubtractImage(Image *other_image);
	void ApplyBFactor(float bfactor);
	void ApplyCTFPhaseFlip(CTF ctf_to_apply);
	void ApplyCTF(CTF ctf_to_apply);
	void MaskCentralCross(int vertical_half_width = 1, int horizontal_half_width = 1);
	void CalculateCrossCorrelationImageWith(Image *other_image);
	void SwapRealSpaceQuadrants();
	void ComputeAmplitudeSpectrumFull2D(Image *other_image);
	void ComputeAmplitudeSpectrum(Image *other_image);
	void Compute1DRotationalAverage(double average[], int number_of_bins);
	void SpectrumBoxConvolution(Image *output_image, int box_size, float minimum_radius);
	void TaperEdges();
	float ReturnAverageOfRealValues(float wanted_mask_radius = 0.0);
	float ReturnAverageOfRealValuesOnEdges();
	float ReturnMaximumValue(float inner_radius, float outer_radius);
	void SetMaximumValue(float new_maximum_value);
	void SetMinimumAndMaximumValues( float new_minimum_value, float new_maximum_value);
	void ComputeAverageAndSigmaOfValuesInSpectrum(float minimum_radius, float maximum_radius, float &average, float &sigma, int cross_half_width = 2);
	void SetMaximumValueOnCentralCross(float maximum_value);
	void ApplyMirrorAlongY();

	void GetMinMax(float &min_value, float &max_value);

	float GetCorrelationWithCTF(CTF ctf);

	// Interpolation
	void GetRealValueByLinearInterpolationNoBoundsCheckImage(float &x, float &y, float &interpolated_value);


	Peak FindPeakAtOriginFast2D(int wanted_max_1d_distance);
	Peak FindPeakWithIntegerCoordinates(float wanted_min_radius = 0, float wanted_max_radius = FLT_MAX);
	Peak FindPeakWithParabolaFit(float wanted_min_radius = 0, float wanted_max_radius = FLT_MAX);

	// Test patterns
	void Sine1D(int number_of_periods);
};



