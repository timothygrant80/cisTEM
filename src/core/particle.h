/*  \brief  ReconstructedVolume class */

class ReconstructedVolume;
/*
class ParameterMap {

	bool position_in_stack;
	bool image_is_active;
	bool psi;
	bool theta;
	bool phi;
	bool x_shift;
	bool y_shift;
	bool defocus_1;
	bool defocus_2;
	bool defocus_angle;
	bool phase_shift;
	bool occupancy;
	bool logp;
	bool sigma;
	bool score;
	bool score_change;
	bool pixel_size;
	bool microscope_voltage_kv;
	bool microscope_spherical_aberration_mm;
	bool beam_tilt_x;
	bool beam_tilt_y;
	bool image_shift_x;
	bool image_shift_y;

	ParameterMap();
};*/

class ParameterMap {

public :

	bool phi;
	bool theta;
	bool psi;
	bool x_shift;
	bool y_shift;

	ParameterMap();
	void SetAllTrue();
};

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
	Image						*beamtilt_image;
	bool						beamtilt_image_calculated;
	Image						*particle_image;
	bool						includes_reference_ssnr_weighting;
	bool						is_normalized;
	float						normalized_sigma;
	bool						is_phase_flipped;
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
	//int							number_of_parameters;
	float						target_phase_error;
	//float						*temp_float;
	cisTEMParameterLine			current_parameters;
	cisTEMParameterLine			temp_parameters;
	cisTEMParameterLine			parameter_average;
	cisTEMParameterLine			parameter_variance;
	ParameterMap				parameter_map;
	ParameterMap				constraints_used;
	int							number_of_search_dimensions;
	int							*bin_index;
	float						mask_center_2d_x;
	float						mask_center_2d_y;
	float						mask_center_2d_z;
	float						mask_radius_2d;
	bool						apply_2D_masking;
	bool						no_ctf_weighting;
	bool						complex_ctf;

	Particle();
	Particle(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	~Particle();

	void Init();
	void AllocateImage(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	void AllocateCTFImage(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	void Allocate(int wanted_logical_x_dimension, int wanted_logical_y_dimension);
	void Deallocate();
	void ResetImageFlags();
	void PhaseShift();
	void PhaseShiftInverse();
	void Whiten(float resolution_limit = 0.49);
	void ForwardFFT(bool do_scaling = true);
	void BackwardFFT();
	void CosineMask(bool invert = false, bool force_mask_value = false, float wanted_mask_value = 0.0);
	void CenterInBox();
	void CenterInCorner();
	void InitCTF(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle, float phase_shift, float beam_tilt_x = 0.0f, float beam_tilt_y = 0.0f, float particle_shift_x = 0.0f, float particle_shift_y = 0.0f);
	void SetDefocus(float defocus_1, float defocus_2, float astigmatism_angle, float phase_shift);
	void SetBeamTilt(float beam_tilt_x, float beam_tilt_y, float particle_shift_x = 0.0f, float particle_shift_y = 0.0f);
	void SetLowResolutionContrast(float low_resolution_contrast);
	void InitCTFImage(float voltage_kV, float spherical_aberration_mm, float amplitude_contrast, float defocus_1, float defocus_2, float astigmatism_angle, float phase_shift, float beam_tilt_x = 0.0f, float beam_tilt_y = 0.0f, float particle_shift_x = 0.0f, float particle_shift_y = 0.0f, bool calculate_complex_ctf = false);
	void PhaseFlipImage();
	void CTFMultiplyImage();
	void BeamTiltMultiplyImage();
	void SetIndexForWeightedCorrelation(bool limit_resolution = true);
	void WeightBySSNR(Curve &SSNR, int include_reference_weighting = 1, bool no_ctf = false);
	void WeightBySSNR(Curve &SSNR, Image &projection_image, bool weight_particle_image = true, bool weight_projection_image = true);
	void CalculateProjection(Image &projection_image, ReconstructedVolume &input_3d);
	void GetParameters(cisTEMParameterLine &output_parameters);
	void SetParameters(cisTEMParameterLine &wanted_parameters, bool initialize_scores = false);
	void SetAlignmentParameters(float wanted_euler_phi, float wanted_euler_theta, float wanted_euler_psi, float wanted_shift_x = 0.0, float wanted_shift_y = 0.0);
	void SetParameterStatistics(cisTEMParameterLine &wanted_averages, cisTEMParameterLine &wanted_variances);
	void SetParameterConstraints(float wanted_noise_variance);
	float ReturnParameterPenalty(cisTEMParameterLine &parameters);
	float ReturnParameterLogP(cisTEMParameterLine &parameters);
	int MapParameterAccuracy(float *accuracies);
	int MapParametersFromExternal(cisTEMParameterLine &input_parameters, float *mapped_parameters);
	int MapParameters(float *mapped_parameters);
	int UnmapParametersToExternal(cisTEMParameterLine &output_parameters, float *mapped_parameters);
	int UnmapParameters(float *mapped_parameters);
	void FindBeamTilt(Image &sum_of_phase_differences, CTF &input_ctf, float pixel_size, Image &phase_error_output, Image &beamtilt_output, Image &difference_image, float &beamtilt_x, float &beamtilt_y, float &particle_shift_x, float &particle_shift_y, float phase_multiplier = 20.0f, bool progress_bar = false);
	float ReturnLogLikelihood(Image &input_image, Image &padded_unbinned_image, CTF &input_ctf, ReconstructedVolume &input_3d, ResolutionStatistics &statistics, float classification_resolution_limit, Image *phase_difference = NULL);
	void CalculateMaskedLogLikelihood(Image &projection_image, ReconstructedVolume &input_3d, float classification_resolution_limit);
	float MLBlur(Image *input_classes_cache, float var_X, Image &cropped_input_image, Image *rotation_cache, Image &blurred_image,
			int current_class, int number_of_rotations, float psi_step, float psi_start, float smoothing_factor, float &max_log_particle, int best_class,
			float best_psi, Image &best_correlation_map, bool calculate_correlation_map_only = false, bool uncrop = true, bool apply_ctf_to_classes = true,
			Image *image_to_blur = NULL, Image *diff_image_to_blur = NULL, float max_shift_in_angstroms = FLT_MAX);
	void EstimateSigmaNoise();
};
