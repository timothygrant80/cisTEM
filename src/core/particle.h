/*  \brief  ReconstructedVolume class */

class ReconstructedVolume;

class Particle {

public:

	int							origin_micrograph;
	float						origin_x_coordinate;	// Particle X coordinate in micrograph, in A
	float						origin_y_coordinate;	// Particle Y coordinate in micrograph, in A
	int							location_in_stack;		// Starts with 0
	float						pixel_size;				// In A
	float						sigma_signal;
	float						sigma_noise;
	float						snr;
	float						logp;					// Log likelihood;
	float						particle_occupancy;		// Particle occupancy within a class
	float						particle_score;
	AnglesAndShifts				alignment_parameters;
	RotationMatrix				*euler_matrix;
/*	float						phi_average;			// In deg
	float						theta_average;			// In deg
	float						psi_average;			// In deg
	float						shift_x_average;		// In A
	float						shift_y_average;		// In A
	float						phi_variance;			// In deg^2
	float						theta_variance;			// In deg^2
	float						psi_variance;			// In deg^2
	float						shift_x_variance;		// In A^2
	float						shift_y_variance;		// In A^2
*/
	float						scaled_noise_variance;
	ParameterConstraints		parameter_constraints;
	CTF							ctf_parameters;
	CTF							current_ctf;
	bool						ctf_is_initialized;
	Image						*ctf_image;
	bool						ctf_image_calculated;
	Image						*particle_image;
	bool						is_normalized;
	float						normalized_sigma;
	bool						is_masked;
	float						mask_radius;			// In A
	float						mask_falloff;			// In A
	float						mask_volume;			// In pixels
	float						molecular_mass_kDa;		// In kilo Daltons
	bool						is_filtered;
	float						filter_radius_low;		// In 1/pixel
	float						filter_radius_high;		// In 1/pixel
	float						filter_falloff;			// In 1/pixel
	float						filter_volume;			// In reciprocal pixels, including the half of the transform not stored by the FFT
	float						signed_CC_limit;		// In 1/pixel, 0.0 = use max resolution
	bool						is_ssnr_filtered;
	bool						is_centered_in_box;
	int							shift_counter;
	bool						insert_even;
	int							number_of_parameters;
	float						target_phase_error;
	float						*temp_float;
	float						*current_parameters;
	float						*parameter_average;
	float						*parameter_variance;
	float						*refined_parameters;
	bool						*parameter_map;
	bool						*constraints_used;
	int							number_of_search_dimensions;
	int							*bin_index;
	float						mask_center_2d_x;
	float						mask_center_2d_y;
	float						mask_center_2d_z;
	float						mask_radius_2d;
	bool						apply_2D_masking;

	Particle();
	Particle(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	~Particle();

	void Init();
	void AllocateImage(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	void AllocateCTFImage(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	void Allocate(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	void ResetImageFlags();
	void PhaseShift();
	void PhaseShiftInverse();
	void Whiten(float resolution_limit = 0.49);
	void ForwardFFT();
	void BackwardFFT();
	void CosineMask(bool invert = false, bool force_mask_value = false, float wanted_mask_value = 0.0);
	void CenterInBox();
	void CenterInCorner();
	void InitCTF(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle);
	void SetDefocus(float defocus_1, float defocus_2, float astigmatism_angle);
	void InitCTFImage(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle);
	void PhaseFlipImage();
	void CTFMultiplyImage();
	void SetIndexForWeightedCorrelation();
	void WeightBySSNR(Curve &SSNR, int include_reference_weighting = 1);
	void WeightBySSNR(Curve &SSNR, Image &projection_image);
	void CalculateProjection(Image &projection_image, ReconstructedVolume &input_3d);
	void GetParameters(float *output_parameters);
	void SetParameters(float *wanted_parameters, bool initialize_scores = false);
	void SetAlignmentParameters(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x = 0.0, float wanted_shift_y = 0.0);
	void SetParameterStatistics(float *wanted_averages, float *wanted_variances);
	void SetParameterConstraints(float wanted_noise_variance);
	float ReturnParameterPenalty(float *parameters);
	float ReturnParameterLogP(float *parameters);
	int MapParameterAccuracy(float *accuracies);
	int MapParametersFromExternal(float *input_parameters, float *mapped_parameters);
	int MapParameters(float *mapped_parameters);
	int UnmapParametersToExternal(float *output_parameters, float *mapped_parameters);
	int UnmapParameters(float *mapped_parameters);
	float ReturnLogLikelihood(ReconstructedVolume &input_3d, ResolutionStatistics &statistics, Image &projection_image, float classification_resolution_limit, float &alpha, float &sigma);
	float ReturnMaskedLogLikelihood(ReconstructedVolume &input_3d, Image &projection_image, float classification_resolution_limit);
	float MLBlur(Image *input_classes_cache, float var_X, Image &cropped_input_image, Image *rotation_cache, Image &blurred_image,
			int current_class, int number_of_rotations, float psi_step, float psi_start, float smoothing_factor, float &max_log_particle, int best_class,
			float best_psi, Image &best_correlation_map, bool calculate_correlation_map_only = false, bool uncrop = true, bool apply_ctf_to_classes = true);
	void EstimateSigmaNoise();
};
