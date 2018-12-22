// Facilitate automatic particle finding in micrographs
class ParticleFinder {

public:

	ParticleFinder();
	~ParticleFinder();

	ArrayOfParticlePositionAssets 	ReturnArrayOfParticlePositionAssets();
	void 							DoItAll();
	void							RedoWithNewMaximumRadius();
	void							RedoWithNewTypicalRadius();
	void							RedoWithNewMinimumPeakHeight();
	void							RedoWithNewHighestResolution();
	void							RedoWithNewMinimumDistanceFromEdges();
	void							RedoWithNewAvoidHighVarianceAreas();
	void							RedoWithNewAvoidAbnormalLocalMeanAreas();
	void							RedoWithNewNumberOfBackgroundBoxes();
	void							RedoWithNewAlgorithmToFindBackground();
	wxString						ReturnMicrographFilename() 					{ return micrograph_filename; };
	CTF								ReturnMicrographCTF()						{ return micrograph_ctf; };
	float							ReturnOriginalMicrographPixelSize()			{ return original_micrograph_pixel_size; };
	void							SetAllUserParameters(	wxString			wanted_micrograph_filename,
															float				wanted_original_micrograph_pixel_size,
															float				wanted_acceleration_voltage_in_keV,
															float				wanted_spherical_aberration_in_mm,
															float				wanted_amplitude_contrast,
															float				wanted_additional_phase_shift_in_radians,
															float				wanted_defocus_1_in_angstroms,
															float				wanted_defocus_2_in_angstroms,
															float				wanted_astigmatism_angle_in_degrees,
															bool				wanted_already_have_templates,
															wxString			wanted_templates_filename,
															bool				wanted_average_templates_radially,
															int					wanted_number_of_template_rotations,
															float				wanted_typical_radius_in_angstroms,
															float				wanted_maximum_radius_in_angstroms,
															float				wanted_highest_resolution_to_use,
															wxString			wanted_output_stack_filename,
															int					wanted_output_stack_box_size,
															int					wanted_minimum_distance_from_edges_in_pixels,
															float				wanted_minimum_peak_height_for_candidate_particles,
															bool				wanted_avoid_high_variance_areas,
															bool				wanted_avoid_high_low_mean_areas,
															int					wanted_algorithm_to_find_background,
															int					wanted_number_of_background_boxes,
															bool				wanted_particles_are_white);





	// Todo: To be made private
	// Todo: use more appropriate objects to do this. Maybe ParticlePositionAssets, or wxArrays of floats
	// Use will (mis)use Curve objects to keep track of our results
	Curve results_x_y;
	Curve results_height_template;
	Curve results_rotation;

	bool write_out_plt;


private:
	// Parameters from the user
	wxString 	micrograph_filename;
	wxString 	templates_filename;
	bool 		already_have_templates;
	float 		original_micrograph_pixel_size;
	float 		highest_resolution_to_use;
	float 		maximum_radius_in_angstroms;
	float		typical_radius_in_angstroms;
	bool		average_templates_radially;
	int			algorithm_to_find_background;
	int			number_of_background_boxes;
	int			number_of_template_rotations;
	float		minimum_peak_height_for_candidate_particles;
	int			output_stack_box_size;
	bool		avoid_high_variance_areas;
	bool		avoid_high_low_mean_areas;
	int			minimum_distance_from_edges_in_pixels;
	wxString	output_stack_filename;
	bool		particles_are_white;

	// CTF parameters from the user
	float		acceleration_voltage_in_keV;
	float		spherical_aberration_in_mm;
	float		amplitude_contrast;
	float		additional_phase_shift_in_radians;
	float		defocus_1_in_angstroms;
	float		defocus_2_in_angstroms;
	float		astigmatism_angle_in_degrees;

	// Internal
	int			new_micrograph_dimension_x;
	int 		new_micrograph_dimension_y;
	int 		new_template_dimension;
	float 		pixel_size;
	float 		maximum_radius_in_pixels;
	float		typical_radius_in_pixels;
	int			number_of_background_boxes_to_skip;
	//float 		minimum_distance_between_picks_in_angstroms;
	//float		minimum_distance_between_picks_in_pixels;
	int 		minimum_box_size_for_object_with_psf;
	int 		minimum_box_size_for_picking;
	int			number_of_templates;
	MRCFile		micrograph_file;
	MRCFile		template_file;
	CTF			micrograph_ctf;
	Image		*template_image;
	ProgressBar *my_progress_bar;
	Curve		template_power_spectrum;
	Curve		background_power_spectrum;
	Curve		current_power_spectrum;
	Curve		current_number_of_fourier_elements;
	Curve		temp_curve;
	Image		micrograph;
	Image		micrograph_bp;
	Image		micrograph_whitened;
	float		micrograph_mean;
	Image 		local_mean;
	Image 		local_sigma;
	//Image 		local_sigma_modified;
	float 		local_sigma_mode;
	float 		local_sigma_fwhm;
	float 		local_mean_mode;
	float		local_mean_fwhm;
	Curve 		background_whitening_filter;
	Image		maximum_score;
	Image 		maximum_score_modified;
	Image		template_rotation_giving_maximum_score;
	Image		template_giving_maximum_score;


	// Private methods
	void		UpdateMaximumRadiusInPixels();
	void		UpdateTypicalRadiusInPixels();
	void		CloseImageFiles();
	void		FindPeaksAndExtractParticles();
	void		RemoveHighLowMeanAreasFromTargetFunction();
	void		RemoveHighVarianceAreasFromTargetFunction();
	void		DoTemplateMatching();
	void		WhitenMicrographBackground();
	void		UpdateLocalMeanAndSigma();
	void		BandPassMicrograph();
	void		ReadAndResampleMicrograph();
	void		GenerateATemplate();
	void		SetupCurveObjects();
	void		ReadTemplatesFromDisk();
	void		AllocateTemplateImages();
	void		DeallocateTemplateImages();
	void		UpdateMinimumBoxSize();
	void		OpenMicrographAndUpdateDimensions();
	void		UpdatePixelSizeFromMicrographDimensions();
	void 		OpenTemplatesAndUpdateDimensions();
	void		UpdateCTF();
	void 		ComputeLocalMeanAndStandardDeviation(Image *micrograph, Image *mask_image, float mask_radius_in_pixels, long number_of_pixels_within_mask, Image *micrograph_local_mean, Image *micrograph_local_stdev);
	void 		SetAreaToIgnore(Image &my_image, int central_pixel_address_x, int central_pixel_address_y, Image *box_image, float wanted_value);
	void 		SetCircularAreaToIgnore(Image &my_image, const int central_pixel_address_x, const int central_pixel_address_y, const float wanted_radius, const float wanted_value);
	void 		PrepareTemplateForMatching(Image *template_image, Image &prepared_image, float in_plane_rotation, CTF *micrograph_ctf, Curve *whitening_filter);

};
