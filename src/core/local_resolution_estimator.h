// Compute local resolution estimates
class LocalResolutionEstimator {

public:

	LocalResolutionEstimator();
	~LocalResolutionEstimator();

	void SetAllUserParameters(	Image *wanted_input_volume_one,
								Image *wanted_input_volume_two,
								Image *wanted_mask_volume,
								int wanted_first_slice,
								int wanted_last_slice,
								int wanted_sampling_step,
								float input_pixel_size_in_Angstroms,
								int wanted_box_size,
								float wanted_threshold_snr, // snr treshold. 0.334 gives the 0.143 criterion
								float wanted_threshold_confidence_n_sigma, // confidence - number of sigmas above estimate we want to be
								bool wanted_use_fixed_fsc_threshold,
								float wanted_fixed_fsc_threshold,
								wxString wanted_symmetry_symbol,
								bool wanted_whiten_half_maps
								);

	void EstimateLocalResolution(Image *local_resolution_volume);


private:
	// Parameters from the user
	int		box_size;
	int		number_of_fsc_shells;
	float	pixel_size_in_Angstroms;
	float 	highest_resolution_expected_in_Angstroms;
	float	maximum_radius_in_Angstroms;
	Image 	*input_volume_one;
	Image 	*input_volume_two;
	Image	*input_volume_mask; // don't bother calculating local resolution outside the mask
	int		first_slice; // don't calculate resolution before first slice (xy plane) of the volume
	int 	last_slice;

	bool 	use_fixed_fsc_threshold;
	float 	fixed_fsc_threshold;
	float 	threshold_confidence_n_sigma; // confidence - number of sigmas above estimate we want to be
	float 	threshold_snr; // snr treshold. 0.334 gives the 0.143 criterion
	float 	resolution_value_before_first_shell; //40.0;//
	float 	resolution_value_where_wont_estimate;

	bool	whiten_half_maps;

	// Internal
	Image	box_one;
	Image	box_two;
	int		*shell_number_lut; // a look-up table: for each voxel in the local_volumes, remember what fsc shell we're in
	bool	shell_number_lut_is_allocated;
	int		sampling_step; // we don't necessarily need to estimate the resolution at every voxel
	int 	symmetry_redundancy;


	// Private methods
	void	AllocateShellNumberLUT();
	void	DeallocateShellNumberLUT();
	void	PopulateShellNumberLUT();
	void    CountIndependentVoxelsPerShell(float number_of_independent_voxels[], float wanted_scaling_factor);
	void 	ComputeFSCThresholdBasedOnUnbiasedSNREstimator(float number_of_independent_voxels[], float fsc_threshold[] );
	void	ComputeLocalFSCAndCompareToThreshold(float fsc_threshold[], Image *local_resolution_volume, Image box_mask);
	void 	AllocateLocalVolumes();
	void 	DeallocateLocalVolumes();
	void 	SetInputVolumes(Image *wanted_input_volume_one, Image *wanted_input_volume_two, Image *wanted_mask_volume);
	inline void SetPixelSize(float wanted_pixel_size_in_Angstroms) {pixel_size_in_Angstroms = wanted_pixel_size_in_Angstroms;};
	inline void SetBoxSize(int wanted_box_size) {box_size = wanted_box_size;};
	float 	ReturnResolutionOfIntersectionBetweenFSCAndThreshold(float resolution_one, float resolution_two, float fsc_one, float fsc_two, float threshold_one, float threshold_two);

	// The key equations
	float 	SigmaZSquaredAuto(float number_of_independent_voxels);
	float 	SigmaZSquared(float rho, float number_of_independent_voxels);
	float	RhoThreshold(float snr_threshold_halfmaps, float number_of_sigmas, float number_of_independent_voxels, int number_of_iterations = 2);

};
