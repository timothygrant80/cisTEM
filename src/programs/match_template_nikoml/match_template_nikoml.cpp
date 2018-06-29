#include "../../core/core_headers.h"

// nasty globals to track histogram size

int histogram_number_of_points = 1024;
float histogram_min = -20.0f;
float histogram_max = 50.0f;

class
MatchTemplateMLApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	//void SendProgramDefinedResultToMaster(float *result_array, int array_size, int result_number, int number_of_expected_results); // overidden as i want to do complicated stuff
	void MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results);
	void ProgramSpecificInit();
	// for master collation

	float *collated_data_array;
	int active_result_number;
	int number_of_received_results;
	float total_number_of_ccs;



	private:
};

class ImageProjectionComparison
{
public:
	Particle					*particle;
	ReconstructedVolume			*reference_volume;
	Image						*projection_image;
//	Image						*temp_image;
};

// This is the function which will be minimized
float FrealignObjectiveFunction(void *scoring_parameters, float *array_of_values)
{
	ImageProjectionComparison *comparison_object = reinterpret_cast < ImageProjectionComparison *> (scoring_parameters);
	for (int i = 0; i < comparison_object->particle->number_of_parameters; i++) {comparison_object->particle->temp_float[i] = comparison_object->particle->current_parameters[i];}
	comparison_object->particle->UnmapParameters(array_of_values);

//	comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image, *comparison_object->particle->ctf_image,
//			comparison_object->particle->alignment_parameters, 0.0, 0.0,
//			comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, true);

	if (comparison_object->particle->no_ctf_weighting) comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image,
			*comparison_object->particle->ctf_image, comparison_object->particle->alignment_parameters, 0.0, 0.0,
			comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, false, false, false, false);
	// Case for normal parameter refinement with weighting applied to particle images and 3D reference
	else if (comparison_object->particle->includes_reference_ssnr_weighting) comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image,
			*comparison_object->particle->ctf_image, comparison_object->particle->alignment_parameters, 0.0, 0.0,
			comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, true, true, false, false);
	// Case for normal parameter refinement with weighting applied only to particle images
	else comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image,
			*comparison_object->particle->ctf_image, comparison_object->particle->alignment_parameters, 0.0, 0.0,
			comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, true, false, true, true);

//	if (comparison_object->particle->origin_micrograph < 0) comparison_object->particle->origin_micrograph = 0;
//	comparison_object->particle->origin_micrograph++;
//	for (int i = 0; i < comparison_object->projection_image->real_memory_allocated; i++) {comparison_object->projection_image->real_values[i] *= fabs(comparison_object->projection_image->real_values[i]);}
//	comparison_object->projection_image->ForwardFFT();
//	comparison_object->projection_image->CalculateCrossCorrelationImageWith(comparison_object->particle->particle_image);
//	comparison_object->projection_image->SwapRealSpaceQuadrants();
//	comparison_object->projection_image->BackwardFFT();
//	comparison_object->projection_image->QuickAndDirtyWriteSlice("proj.mrc", comparison_object->particle->origin_micrograph);
//	comparison_object->projection_image->SwapRealSpaceQuadrants();
//	comparison_object->particle->particle_image->SwapRealSpaceQuadrants();
//	comparison_object->particle->particle_image->BackwardFFT();
//	comparison_object->particle->particle_image->QuickAndDirtyWriteSlice("part.mrc", comparison_object->particle->origin_micrograph);
//	comparison_object->particle->particle_image->SwapRealSpaceQuadrants();
//	exit(0);

//	float score =  	- comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image, comparison_object->particle->bin_index,
//			  comparison_object->particle->pixel_size / comparison_object->particle->signed_CC_limit)
//			- comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_float);
//	wxPrintf("psi, theta, phi, x, y, = %g, %g, %g, %g, %g, score = %g\n",
//			comparison_object->particle->alignment_parameters.ReturnPsiAngle(),
//			comparison_object->particle->alignment_parameters.ReturnThetaAngle(),
//			comparison_object->particle->alignment_parameters.ReturnPhiAngle(),
//			comparison_object->particle->alignment_parameters.ReturnShiftX(),
//			comparison_object->particle->alignment_parameters.ReturnShiftY(), score);
//	return score;
//	wxPrintf("sigma_noise, mask_volume, penalty = %g %g %g\n", comparison_object->particle->sigma_noise, comparison_object->particle->mask_volume,
//			comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_float));
	return 	- comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image, comparison_object->particle->bin_index,
			  comparison_object->particle->pixel_size / comparison_object->particle->signed_CC_limit)
			- comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_float);
		// This penalty term assumes a Gaussian x,y distribution that is probably not correct in most cases. It might be better to leave it out.

}

IMPLEMENT_APP(MatchTemplateMLApp)

void MatchTemplateMLApp::ProgramSpecificInit()
{
	collated_data_array = NULL;
	active_result_number = -1;
	number_of_received_results = 0;
}

// override the DoInteractiveUserInput

void MatchTemplateMLApp::DoInteractiveUserInput()
{
	wxString	input_search_images;
	wxString	input_reconstruction;

	wxString    mip_output_file;
	wxString    best_psi_output_file;
	wxString    best_theta_output_file;
	wxString    best_phi_output_file;

	wxString    output_histogram_file;
	wxString    correlation_variance_output_file;
	wxString    correlation_average_output_file;
	wxString    scaled_mip_output_file;

	float		pixel_size = 1.0f;
	float		voltage_kV = 300.0f;
	float		spherical_aberration_mm = 2.7f;
	float		amplitude_contrast = 0.07f;
	float 		defocus1 = 10000.0f;
	float		defocus2 = 10000.0f;;
	float		defocus_angle;
	float 		phase_shift;
	float		low_resolution_limit = 300.0;
	float		high_resolution_limit = 8.0;
	float		angular_step = 5.0;
	int			best_parameters_to_keep = 20;
	float 		defocus_search_range = 500;
	float 		defocus_step = 50;
	float		padding = 1.0;
	bool		ctf_refinement = false;
	float		mask_radius_search = 0.0;
	wxString	my_symmetry = "C1";
	float 		in_plane_angular_step = 0;
	float		randomise_resolution = 0.0;

	/*
	wxString	input_parameter_file;

	wxString	input_reconstruction_statistics;
	bool		use_statistics;
	wxString	ouput_matching_projections;
	wxString	ouput_parameter_file;
	wxString	ouput_shift_file;
	wxString	my_symmetry = "C1";
	int			first_particle = 1;
	int			last_particle = 0;
	float		percent_used = 1.0;

	float		molecular_mass_kDa = 1000.0;
	float		inner_mask_radius = 0.0;
	float		outer_mask_radius = 100.0;
	float		signed_CC_limit = 0.0;
	float		classification_resolution_limit = 0.0;
	float		mask_radius_search = 0.0;
	float		high_resolution_limit_search = 20.0;


	float		max_search_x = 0;
	float		max_search_y = 0;
	float		mask_center_2d_x = 100.0;
	float		mask_center_2d_y = 100.0;
	float		mask_center_2d_z = 100.0;
	float		mask_radius_2d = 100.0;




//	float		filter_constant = 1.0;
	bool		global_search = false;
	bool		local_refinement = true;
	bool		refine_psi = true;
	bool		refine_theta = true;
	bool		refine_phi = true;
	bool		refine_x = true;
	bool		refine_y = true;
	bool		calculate_matching_projections = false;
	bool		apply_2D_masking = false;

	bool		normalize_particles = true;
	bool		invert_contrast = false;
	bool		exclude_blank_edges = true;
	bool		normalize_input_3d = true;
	bool		threshold_input_3d = true;

	*/

	UserInput *my_input = new UserInput("MatchTemplate", 1.00);

	input_search_images = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "my_image_stack.mrc", true);
	input_reconstruction = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	mip_output_file = my_input->GetFilenameFromUser("Output MIP file", "The file for saving the maximum intensity projection image", "my_mip.mrc", false);
	scaled_mip_output_file = my_input->GetFilenameFromUser("Output Scaled MIP file", "The file for saving the maximum intensity projection image divided by correlation variance", "my_mip_scaled.mrc", false);
	best_psi_output_file = my_input->GetFilenameFromUser("Output psi file", "The file for saving the best psi image", "my_psi.mrc", false);
	best_theta_output_file = my_input->GetFilenameFromUser("Output theta file", "The file for saving the best psi image", "my_theta.mrc", false);
	best_phi_output_file = my_input->GetFilenameFromUser("Output phi file", "The file for saving the best psi image", "my_phi.mrc", false);
	correlation_average_output_file = my_input->GetFilenameFromUser("Correlation average value", "The file for saving the average value of all correlation images", "my_corr_average.mrc", false);
	correlation_variance_output_file = my_input->GetFilenameFromUser("Correlation variance output file", "The file for saving the variance of all correlation images", "my_corr_variance.mrc", false);
	output_histogram_file = my_input->GetFilenameFromUser("Output histogram of correlation values", "histogram of all correlation values", "my_histogram.txt", false);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	defocus1 = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
	defocus2 = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
	defocus_angle = my_input->GetFloatFromUser("Defocus Angle (degrees)", "Defocus Angle for the input image", "0.0");
	phase_shift = my_input->GetFloatFromUser("Phase Shift (degrees)", "Additional phase shift in degrees", "0.0");
	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	angular_step = my_input->GetFloatFromUser("Out of plane angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
	in_plane_angular_step = my_input->GetFloatFromUser("In plane angular step (0.0 = set automatically)", "Angular step size for in-plane rotations during the search", "0.0", 0.0);
	best_parameters_to_keep = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
	defocus_search_range = my_input->GetFloatFromUser("Defocus search range (A)", "Search range (-value ... + value) around current defocus", "500.0", 0.0);
	defocus_step = my_input->GetFloatFromUser("Defocus step (A)", "Step size used in the defocus search", "50.0", 0.0);
	padding = my_input->GetFloatFromUser("Tuning parameters: padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
	ctf_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
	mask_radius_search = my_input->GetFloatFromUser("Mask radius for global search (A) (0.0 = max)", "Radius of a circular mask to be applied to the input images during global search", "100.0", 0.0);
	randomise_resolution = my_input->GetFloatFromUser("Randomise resolution for second template (A)", "frequencies higher than this resolution will be randomised", "10.0", 0.0);
	my_symmetry = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");



	int first_search_position = -1;
	int last_search_position = -1;
	int image_number_for_gui = 0;
	int number_of_jobs_per_image_in_gui = 0;

	delete my_input;

	my_current_job.Reset(29);
	my_current_job.ManualSetArguments("ttffffffffffifffbfftttttttftiiiitf",	input_search_images.ToUTF8().data(),
															input_reconstruction.ToUTF8().data(),
															pixel_size,
															voltage_kV,
															spherical_aberration_mm,
															amplitude_contrast,
															defocus1,
															defocus2,
															defocus_angle,
															low_resolution_limit,
															high_resolution_limit,
															angular_step,
															best_parameters_to_keep,
															defocus_search_range,
															defocus_step,
															padding,
															ctf_refinement,
															mask_radius_search,
															phase_shift,
															mip_output_file.ToUTF8().data(),
															best_psi_output_file.ToUTF8().data(),
															best_theta_output_file.ToUTF8().data(),
															best_phi_output_file.ToUTF8().data(),
															scaled_mip_output_file.ToUTF8().data(),
															correlation_variance_output_file.ToUTF8().data(),
															my_symmetry.ToUTF8().data(),
															in_plane_angular_step,
															output_histogram_file.ToUTF8().data(),
															first_search_position,
															last_search_position,
															image_number_for_gui,
															number_of_jobs_per_image_in_gui,
															correlation_average_output_file.ToUTF8().data(),
															randomise_resolution);

	//input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	//input_reconstruction_statistics = my_input->GetFilenameFromUser("Input data statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
	//use_statistics = my_input->GetYesNoFromUser("Use statistics", "Answer No if no statistics are available?", "Yes");
	//ouput_matching_projections = my_input->GetFilenameFromUser("Output matching projections", "The output image stack, containing the matching projections", "my_projection_stack.mrc", false);
	//ouput_parameter_file = my_input->GetFilenameFromUser("Output parameter file", "The output parameter file, containing your refined particle alignment parameters", "my_refined_parameters.par", false);
	//ouput_shift_file = my_input->GetFilenameFromUser("Output parameter changes", "The changes in the alignment parameters compared to the input parameters", "my_parameter_changes.par", false);
	//my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
	//first_particle = my_input->GetIntFromUser("First particle to refine (0 = first in stack)", "The first particle in the stack that should be refined", "1", 0);
	//last_particle = my_input->GetIntFromUser("Last particle to refine (0 = last in stack)", "The last particle in the stack that should be refined", "0", 0);
	//percent_used = my_input->GetFloatFromUser("Percent of particles to use (1 = all)", "The percentage of randomly selected particles that will be refined", "1.0", 0.0, 1.0);
	//molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	//inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the input reconstruction in Angstroms", "0.0", 0.0);
	//outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the input reconstruction and images during refinement, in Angstroms", "100.0", inner_mask_radius);
	//signed_CC_limit = my_input->GetFloatFromUser("Resolution limit for signed CC (A) (0.0 = max)", "The absolute value of the weighted Fourier ring correlation will be used beyond this limit", "0.0", 0.0);
	//classification_resolution_limit = my_input->GetFloatFromUser("Res limit for classification (A) (0.0 = max)", "Resolution limit of the data used for calculating LogP", "0.0", 0.0);
	//mask_radius_search = my_input->GetFloatFromUser("Mask radius for global search (A) (0.0 = max)", "Radius of a circular mask to be applied to the input images during global search", "100.0", 0.0);
	//high_resolution_limit_search = my_input->GetFloatFromUser("Approx. resolution limit for search (A)", "High resolution limit of the data used in the global search in Angstroms", "20.0", 0.0);
	//max_search_x = my_input->GetFloatFromUser("Search range in X (A) (0.0 = 0.5 * mask radius)", "The maximum global peak search distance along X from the particle box center", "0.0", 0.0);
	//max_search_y = my_input->GetFloatFromUser("Search range in Y (A) (0.0 = 0.5 * mask radius)", "The maximum global peak search distance along Y from the particle box center", "0.0", 0.0);
	//mask_center_2d_x = my_input->GetFloatFromUser("2D mask X coordinate (A)", "X coordinate of 2D mask center", "100.0", 0.0);
	//mask_center_2d_y = my_input->GetFloatFromUser("2D mask Y coordinate (A)", "Y coordinate of 2D mask center", "100.0", 0.0);
	//mask_center_2d_z = my_input->GetFloatFromUser("2D mask Z coordinate (A)", "Z coordinate of 2D mask center", "100.0", 0.0);
	//mask_radius_2d = my_input->GetFloatFromUser("2D mask radius (A)", "Radius of a circular mask to be used for likelihood calculation", "100.0", 0.0);
//	filter_constant = my_input->GetFloatFromUser("Tuning parameters: filter constant", "Constant determining how strongly data with small CTF values is suppressed during particle alignment", "1.0", 1.0);
	//global_search = my_input->GetYesNoFromUser("Global search", "Should a global search be performed before local refinement?", "No");
	//local_refinement = my_input->GetYesNoFromUser("Local refinement", "Should a local parameter refinement be performed?", "Yes");
	//refine_psi = my_input->GetYesNoFromUser("Refine Psi", "Should the Psi Euler angle be refined (parameter 1)?", "Yes");
	//refine_theta = my_input->GetYesNoFromUser("Refine Theta", "Should the Theta Euler angle be refined (parameter 2)?", "Yes");
	//refine_phi = my_input->GetYesNoFromUser("Refine Phi", "Should the Phi Euler angle be refined (parameter 3)?", "Yes");
	//refine_x = my_input->GetYesNoFromUser("Refine ShiftX", "Should the X shift be refined (parameter 4)?", "Yes");
	//refine_y = my_input->GetYesNoFromUser("Refine ShiftY", "Should the Y shift be refined (parameter 5)?", "Yes");
	//calculate_matching_projections = my_input->GetYesNoFromUser("Calculate matching projections", "Should matching projections be calculated?", "No");
	//apply_2D_masking = my_input->GetYesNoFromUser("Apply 2D masking", "Should 2D masking be used for the likelihood calculation?", "No");
	//normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
	//invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	//exclude_blank_edges = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "Yes");
	//normalize_input_3d = my_input->GetYesNoFromUser("Normalize input reconstruction", "The input reconstruction should always be normalized unless it was generated by reconstruct3d with normalized particles", "Yes");
	//threshold_input_3d = my_input->GetYesNoFromUser("Threshold input reconstruction", "Should the input reconstruction thresholded to suppress some of the background noise", "No");

//	bool local_global_refine = false;
//	int current_class = 0;
//	bool ignore_input_angles = false;


	// Add phase flip option, normalize option, remove input statistics & use statistics



}

// override the do calculation method which will be what is actually run..

bool MatchTemplateMLApp::DoCalculation()
{
	/*
	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument(); // global
	wxString input_parameter_file 				= my_current_job.arguments[1].ReturnStringArgument(); // not sure
	wxString input_reconstruction				= my_current_job.arguments[2].ReturnStringArgument(); // global
	wxString input_reconstruction_statistics 	= my_current_job.arguments[3].ReturnStringArgument(); // global
	bool	 use_statistics						= my_current_job.arguments[4].ReturnBoolArgument();   // global
	wxString ouput_matching_projections 		= my_current_job.arguments[5].ReturnStringArgument(); // ignore (always false)
	wxString ouput_parameter_file				= my_current_job.arguments[6].ReturnStringArgument(); // not sure par file
	wxString ouput_shift_file					= my_current_job.arguments[7].ReturnStringArgument(); // not sure output
	wxString my_symmetry						= my_current_job.arguments[8].ReturnStringArgument(); // global
	int		 first_particle						= my_current_job.arguments[9].ReturnIntegerArgument(); // local (effectively ignore)
	int		 last_particle						= my_current_job.arguments[10].ReturnIntegerArgument(); // local (effectively ignore)
	float	 percent_used						= my_current_job.arguments[11].ReturnFloatArgument();
	float 	 pixel_size							= my_current_job.arguments[12].ReturnFloatArgument(); // local
	float    voltage_kV							= my_current_job.arguments[13].ReturnFloatArgument(); // local
	float 	 spherical_aberration_mm			= my_current_job.arguments[14].ReturnFloatArgument(); // local
	float    amplitude_contrast					= my_current_job.arguments[15].ReturnFloatArgument(); // local
	float	 molecular_mass_kDa					= my_current_job.arguments[16].ReturnFloatArgument(); // global
	float    inner_mask_radius					= my_current_job.arguments[17].ReturnFloatArgument(); // global
	float    outer_mask_radius					= my_current_job.arguments[18].ReturnFloatArgument(); // global
	float    low_resolution_limit				= my_current_job.arguments[19].ReturnFloatArgument(); // global
	float    high_resolution_limit				= my_current_job.arguments[20].ReturnFloatArgument(); // global
	float	 signed_CC_limit					= my_current_job.arguments[21].ReturnFloatArgument(); // global
	float	 classification_resolution_limit	= my_current_job.arguments[22].ReturnFloatArgument(); // global
	float    mask_radius_search					= my_current_job.arguments[23].ReturnFloatArgument(); // global
	float	 high_resolution_limit_search		= my_current_job.arguments[24].ReturnFloatArgument(); // global
	float	 angular_step						= my_current_job.arguments[25].ReturnFloatArgument(); // global
	int		 best_parameters_to_keep			= my_current_job.arguments[26].ReturnIntegerArgument(); // global
	float	 max_search_x						= my_current_job.arguments[27].ReturnFloatArgument(); // global
	float	 max_search_y						= my_current_job.arguments[28].ReturnFloatArgument(); // global
	refine_particle.mask_center_2d_x			= my_current_job.arguments[29].ReturnFloatArgument(); // global
	refine_particle.mask_center_2d_y			= my_current_job.arguments[30].ReturnFloatArgument(); // global
	refine_particle.mask_center_2d_z			= my_current_job.arguments[31].ReturnFloatArgument(); // global
	refine_particle.mask_radius_2d				= my_current_job.arguments[32].ReturnFloatArgument(); // global
	float	 defocus_search_range				= my_current_job.arguments[33].ReturnFloatArgument(); // global
	float	 defocus_step						= my_current_job.arguments[34].ReturnFloatArgument(); // global
	float	 padding							= my_current_job.arguments[35].ReturnFloatArgument(); // global
//	float	 filter_constant					= my_current_job.arguments[35].ReturnFloatArgument();
	bool	 global_search						= my_current_job.arguments[36].ReturnBoolArgument(); // global
	bool	 local_refinement					= my_current_job.arguments[37].ReturnBoolArgument(); // global
// Psi, Theta, Phi, ShiftX, ShiftY
	refine_particle.parameter_map[3]			= my_current_job.arguments[38].ReturnBoolArgument(); //global
	refine_particle.parameter_map[2]			= my_current_job.arguments[39].ReturnBoolArgument(); //global
	refine_particle.parameter_map[1]			= my_current_job.arguments[40].ReturnBoolArgument(); // global
	refine_particle.parameter_map[4]			= my_current_job.arguments[41].ReturnBoolArgument(); // global
	refine_particle.parameter_map[5]			= my_current_job.arguments[42].ReturnBoolArgument(); // global
	bool 	 calculate_matching_projections		= my_current_job.arguments[43].ReturnBoolArgument(); // global - but ignore
	refine_particle.apply_2D_masking			= my_current_job.arguments[44].ReturnBoolArgument(); // global
	bool	 ctf_refinement						= my_current_job.arguments[45].ReturnBoolArgument(); // global
	bool	 normalize_particles				= my_current_job.arguments[46].ReturnBoolArgument();
	bool	 invert_contrast					= my_current_job.arguments[47].ReturnBoolArgument(); // global - but ignore.
	bool	 exclude_blank_edges				= my_current_job.arguments[48].ReturnBoolArgument();
	bool	 normalize_input_3d					= my_current_job.arguments[49].ReturnBoolArgument();
	bool	 threshold_input_3d					= my_current_job.arguments[50].ReturnBoolArgument();
	bool	 local_global_refine				= my_current_job.arguments[51].ReturnBoolArgument();
	int		 current_class						= my_current_job.arguments[52].ReturnIntegerArgument(); // global - but ignore.
	bool	 ignore_input_angles				= my_current_job.arguments[53].ReturnBoolArgument(); // during global search, ignore the starting parameters (this helps reduce bias)
*/

	wxDateTime start_time = wxDateTime::Now();

	wxString	input_search_images_filename = my_current_job.arguments[0].ReturnStringArgument();
	wxString	input_reconstruction_filename = my_current_job.arguments[1].ReturnStringArgument();
	float		pixel_size = my_current_job.arguments[2].ReturnFloatArgument();
	float		voltage_kV = my_current_job.arguments[3].ReturnFloatArgument();
	float		spherical_aberration_mm = my_current_job.arguments[4].ReturnFloatArgument();
	float		amplitude_contrast = my_current_job.arguments[5].ReturnFloatArgument();
	float 		defocus1 = my_current_job.arguments[6].ReturnFloatArgument();
	float		defocus2 = my_current_job.arguments[7].ReturnFloatArgument();
	float		defocus_angle = my_current_job.arguments[8].ReturnFloatArgument();;
	float		low_resolution_limit = my_current_job.arguments[9].ReturnFloatArgument();
	float		high_resolution_limit_search = my_current_job.arguments[10].ReturnFloatArgument();
	float		angular_step = my_current_job.arguments[11].ReturnFloatArgument();
	int			best_parameters_to_keep = my_current_job.arguments[12].ReturnIntegerArgument();
	float 		defocus_search_range = my_current_job.arguments[13].ReturnFloatArgument();
	float 		defocus_step = my_current_job.arguments[14].ReturnFloatArgument();
	float		padding = my_current_job.arguments[15].ReturnFloatArgument();
	bool		ctf_refinement = my_current_job.arguments[16].ReturnBoolArgument();
	float		mask_radius_search = my_current_job.arguments[17].ReturnFloatArgument();
	float 		phase_shift = my_current_job.arguments[18].ReturnFloatArgument();
	wxString    mip_output_file = my_current_job.arguments[19].ReturnStringArgument();
	wxString    best_psi_output_file = my_current_job.arguments[20].ReturnStringArgument();
	wxString    best_theta_output_file = my_current_job.arguments[21].ReturnStringArgument();
	wxString    best_phi_output_file = my_current_job.arguments[22].ReturnStringArgument();
	wxString    scaled_mip_output_file = my_current_job.arguments[23].ReturnStringArgument();
	wxString    correlation_variance_output_file = my_current_job.arguments[24].ReturnStringArgument();
	wxString 	my_symmetry = my_current_job.arguments[25].ReturnStringArgument();
	float		in_plane_angular_step = my_current_job.arguments[26].ReturnFloatArgument();
	wxString    output_histogram_file = my_current_job.arguments[27].ReturnStringArgument();
	int 		first_search_position = my_current_job.arguments[28].ReturnIntegerArgument();
	int 		last_search_position = my_current_job.arguments[29].ReturnIntegerArgument();
	int 		image_number_for_gui = my_current_job.arguments[30].ReturnIntegerArgument();
	int 		number_of_jobs_per_image_in_gui = my_current_job.arguments[31].ReturnIntegerArgument();
	wxString    correlation_average_output_file = my_current_job.arguments[32].ReturnStringArgument();
	float		randomise_resolution = my_current_job.arguments[33].ReturnFloatArgument();

	/*wxPrintf("input image = %s\n", input_search_images_filename);
	wxPrintf("input reconstruction= %s\n", input_reconstruction_filename);
	wxPrintf("pixel size = %f\n", pixel_size);
	wxPrintf("voltage = %f\n", voltage_kV);
	wxPrintf("Cs = %f\n", spherical_aberration_mm);
	wxPrintf("amp contrast = %f\n", amplitude_contrast);
	wxPrintf("defocus1 = %f\n", defocus1);
	wxPrintf("defocus2 = %f\n", defocus2);
	wxPrintf("defocus_angle = %f\n", defocus_angle);
	wxPrintf("low res limit = %f\n", low_resolution_limit);
	wxPrintf("high res limit = %f\n", high_resolution_limit_search);
	wxPrintf("angular step = %f\n", angular_step);
	wxPrintf("best params to keep = %i\n", best_parameters_to_keep);
	wxPrintf("defocus search range = %f\n", defocus_search_range);
	wxPrintf("defocus step = %f\n", defocus_step);
	wxPrintf("padding = %f\n", padding);
	wxPrintf("ctf_refinement = %i\n", int(ctf_refinement));
	wxPrintf("mask search radius = %f\n", mask_radius_search);
	wxPrintf("phase shift = %f\n", phase_shift);
	wxPrintf("symmetry = %s\n", my_symmetry);
	wxPrintf("in plane step = %f\n", in_plane_angular_step);
	wxPrintf("first location = %i\n", first_search_position);
	wxPrintf("last location = %i\n", last_search_position);
	*/

	bool parameter_map[5]; // needed for euler search init
	for (int i = 0; i < 5; i++) {parameter_map[i] = true;}


	float outer_mask_radius;
	float current_psi;
	float psi_step;
	float psi_max;
	float psi_start;
	float histogram_step;

	float expected_threshold;
	float actual_number_of_ccs_calculated;

	double histogram_min_scaled; // scaled for the x*y scaling which is only applied at the end.
	double histogram_step_scaled;// scaled for the x*y scaling which is only applied at the end.

	long *histogram_data;

	int current_bin;

	float temp_float;
	double temp_double_array[5];

	int number_of_rotations;
	long total_correlation_positions;
	long current_correlation_position;
	long pixel_counter;

	int current_search_position;
	int current_x;
	int current_y;

	EulerSearch	global_euler_search;
	AnglesAndShifts angles;

	ImageFile input_search_image_file;
	ImageFile input_reconstruction_file;

	Curve whitening_filter;
	Curve number_of_terms;

	input_search_image_file.OpenFile(input_search_images_filename.ToStdString(), false);
	input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString(), false);

	Image input_image;
	Image input_reconstruction;

	Image current_projection;
	Image padded_reference;
	Image current_projection_random;
	Image padded_reference_random;

	Image projection_filter;

	Image max_intensity_projection;

	Image best_psi;
	Image best_theta;
	Image best_phi;


	Image correlation_pixel_sum;
	Image correlation_pixel_sum_of_squares;

	input_image.ReadSlice(&input_search_image_file, 1);
	padded_reference.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	padded_reference_random.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	max_intensity_projection.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_psi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_theta.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_phi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	correlation_pixel_sum.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	correlation_pixel_sum_of_squares.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);

	padded_reference.SetToConstant(0.0f);
	padded_reference_random.SetToConstant(0.0f);
	max_intensity_projection.SetToConstant(0.0f);
	best_psi.SetToConstant(0.0f);
	best_theta.SetToConstant(0.0f);
	best_phi.SetToConstant(0.0f);
	correlation_pixel_sum.SetToConstant(0.0f);
	correlation_pixel_sum_of_squares.SetToConstant(0.0f);

	input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices());
	input_reconstruction.ForwardFFT();
	//input_reconstruction.CosineMask(0.1, 0.01, true);
	//input_reconstruction.Whiten();
	//if (first_search_position == 0) input_reconstruction.QuickAndDirtyWriteSlices("/tmp/filter.mrc", 1, input_reconstruction.logical_z_dimension);
	input_reconstruction.ZeroCentralPixel();
	input_reconstruction.SwapRealSpaceQuadrants();

	// setup curve

	histogram_step = (histogram_max - histogram_min) / float(histogram_number_of_points);
	histogram_min_scaled = histogram_min / double(sqrt(input_image.logical_x_dimension * input_image.logical_y_dimension));
	histogram_step_scaled = histogram_step / double(sqrt(input_image.logical_x_dimension * input_image.logical_y_dimension));

	histogram_data = new long[histogram_number_of_points];

	for ( int counter = 0; counter < histogram_number_of_points; counter++ )
	{
		histogram_data[counter] = 0;
	}

	CTF input_ctf;
	input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));

	// assume cube

	current_projection.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);
	projection_filter.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);


	// angular step

	if (angular_step <= 0) angular_step = CalculateAngularStep(high_resolution_limit_search, mask_radius_search);
	if (in_plane_angular_step <= 0)
	{
		psi_step = rad_2_deg(pixel_size / mask_radius_search);
		psi_step = 360.0 / int(360.0 / psi_step + 0.5);
	}
	else
	{
		psi_step = in_plane_angular_step;
	}

	//psi_start = psi_step / 2.0 * global_random_number_generator.GetUniformRandom();
	psi_start = 0.0f;
	psi_max = 360.0f;

	//psi_step = 5;

	//wxPrintf("psi_start = %f, psi_max = %f, psi_step = %f\n", psi_start, psi_max, psi_step);

	// search grid

	global_euler_search.InitGrid(my_symmetry, angular_step, 0.0, 0.0, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);

	if (global_euler_search.test_mirror == true) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
	{
		global_euler_search.theta_max = 180.0f;
		global_euler_search.CalculateGridSearchPositions();
	}


	// for now, I am assuming the MTF has been applied already.
	// work out the filter to just whiten the image..

	whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

	wxDateTime my_time_out;
	wxDateTime my_time_in;

	// remove outliers

	input_image.ReplaceOutliersWithMean(5.0f);
	input_image.ForwardFFT();

	input_image.ZeroCentralPixel();
	input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
	whitening_filter.SquareRoot();
	whitening_filter.Reciprocal();
	whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue());

	//whitening_filter.WriteToFile("/tmp/filter.txt");
	input_image.ApplyCurveFilter(&whitening_filter);
	input_image.ZeroCentralPixel();
	input_image.DivideByConstant(sqrt(input_image.ReturnSumOfSquares()));
	//input_image.QuickAndDirtyWriteSlice("/tmp/white.mrc", 1);
	//exit(-1);

	// make the projection filter, which will be CTF * whitening filter
	projection_filter.CalculateCTFImage(input_ctf);
	projection_filter.ApplyCurveFilter(&whitening_filter);

	// count total searches (lazy)

	total_correlation_positions = 0;
	current_correlation_position = 0;

	// if running locally, search over all of them

	if (is_running_locally == true)
	{
		first_search_position = 0;
		last_search_position = global_euler_search.number_of_search_positions - 1;
	}

	for (current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++)
	{
		//loop over each rotation angle

		for (current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
		{
			total_correlation_positions++;
		}
	}

	number_of_rotations = 0;

	for (current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
	{
		number_of_rotations++;
	}

	ProgressBar *my_progress;

	if (is_running_locally == true)
	{
		//Loop over ever search position

		wxPrintf("\nSearching %i positions on the euler sphere.\n", last_search_position - first_search_position, first_search_position, last_search_position);
		wxPrintf("Searching %i rotations per position.\n", number_of_rotations);
		wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions);

		wxPrintf("Performing Search...\n\n");
		my_progress = new ProgressBar(total_correlation_positions);
	}

//	wxPrintf("Searching %i - %i of %i total positions\n", first_search_position, last_search_position, global_euler_search.number_of_search_positions);
//	wxPrintf("psi_start = %f, psi_max = %f, psi_step = %f\n", psi_start, psi_max, psi_step);

	actual_number_of_ccs_calculated = 0.0;


// REMOVE THIS SECTION

	//Image average_value_image;
	//Image variance_image;

	//average_value_image.QuickAndDirtyReadSlice("/tmp/centre_sums.mrc", 1);
	//variance_image.QuickAndDirtyReadSlice("/tmp/centre_square_sums.mrc", 1);

	//average_value_image.SwapRealSpaceQuadrants();
	//variance_image.SwapRealSpaceQuadrants();


	//Image *correlation_buffers;
	//int current_rotation;
	//correlation_buffers = new Image[number_of_rotations];

	//for (current_rotation = 0; current_rotation < number_of_rotations; current_rotation++)
	//{
		//correlation_buffers[current_rotation].Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	//}


	for (current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++)
	{
		//loop over each rotation angle

		//current_rotation = 0;
		for (current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
		{
			angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);
			//angles.Init(-105.27, 73.04, 134.84, 0.0, 0.0);

			input_reconstruction.ExtractSlice(current_projection, angles, 1.0, false);
			current_projection.SwapRealSpaceQuadrants();
			//if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj_nofilter.mrc", 1);


			current_projection.MultiplyPixelWise(projection_filter);
			current_projection_random.CopyFrom(&current_projection);
			current_projection_random.RandomisePhases(pixel_size / randomise_resolution);


			//if (first_search_position == 0) projection_filter.QuickAndDirtyWriteSlice("/tmp/projection_filter.mrc", 1);
			//if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj_afterfilter.mrc", 1);

			//current_projection.ZeroCentralPixel();
			//current_projection.DivideByConstant(sqrt(current_projection.ReturnSumOfSquares()));
			current_projection.BackwardFFT();
			current_projection_random.BackwardFFT();



			current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges());
			current_projection_random.AddConstant(-current_projection_random.ReturnAverageOfRealValuesOnEdges());

			if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj.mrc", 1);
			if (first_search_position == 0) current_projection_random.QuickAndDirtyWriteSlice("/tmp/small_proj_random.mrc", 1);


			padded_reference.SetToConstant(0.0f);
			padded_reference_random.SetToConstant(0.0f);

			current_projection.ClipIntoLargerRealSpace2D(&padded_reference);
			current_projection_random.ClipIntoLargerRealSpace2D(&padded_reference_random);

			padded_reference.ForwardFFT();
			//padded_reference.ZeroCentralPixel();

			padded_reference_random.ForwardFFT();
			//padded_reference_random.ZeroCentralPixel();

#ifdef MKL
			// Use the MKL
			vmcMulByConj(padded_reference.real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (padded_reference.complex_values),reinterpret_cast <MKL_Complex8 *> (input_image.complex_values),reinterpret_cast <MKL_Complex8 *> (padded_reference.complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
			vmcMulByConj(padded_reference_random.real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (padded_reference_random.complex_values),reinterpret_cast <MKL_Complex8 *> (input_image.complex_values),reinterpret_cast <MKL_Complex8 *> (padded_reference_random.complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
#else
			for (pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated / 2; pixel_counter ++)
			{
				padded_reference.complex_values[pixel_counter] *= conj(input_image.complex_values[pixel_counter]);
				padded_reference_random.complex_values[pixel_counter] *= conj(input_image.complex_values[pixel_counter]);
			}
#endif

			//padded_reference.MultiplyByConstant(sqrtf(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension));
			padded_reference.BackwardFFT();
			padded_reference_random.BackwardFFT();

			padded_reference.SubtractImage(&padded_reference_random);

			pixel_counter = 0;

			for (current_y = 0; current_y < max_intensity_projection.logical_y_dimension; current_y++)
			{
				for (current_x = 0; current_x < max_intensity_projection.logical_x_dimension; current_x++)
				{
					// first mip

					if (padded_reference.real_values[pixel_counter] > max_intensity_projection.real_values[pixel_counter])
					{
						max_intensity_projection.real_values[pixel_counter] = padded_reference.real_values[pixel_counter];
						best_psi.real_values[pixel_counter] = current_psi;
						best_theta.real_values[pixel_counter] = global_euler_search.list_of_search_parameters[current_search_position][1];
						best_phi.real_values[pixel_counter] = global_euler_search.list_of_search_parameters[current_search_position][0];
					}

					// histogram

					current_bin = int(double((padded_reference.real_values[pixel_counter]) - histogram_min_scaled) / histogram_step_scaled);
					//current_bin = int(double((padded_reference.real_values[pixel_counter]) - histogram_min) / histogram_step);

					if (current_bin >= 0 && current_bin <= histogram_number_of_points)
					{
						histogram_data[current_bin] += 1;
					}


					pixel_counter++;
				}

				pixel_counter+=padded_reference.padding_jump_value;
			}


			correlation_pixel_sum.AddImage(&padded_reference);
			padded_reference.SquareRealValues();
			correlation_pixel_sum_of_squares.AddImage(&padded_reference);

			//max_intensity_projection.QuickAndDirtyWriteSlice("/tmp/mip.mrc", 1);

			current_projection.is_in_real_space = false;
			padded_reference.is_in_real_space = true;

			current_correlation_position++;
			if (is_running_locally == true) my_progress->Update(current_correlation_position);

			if (is_running_locally == false)
			{
				actual_number_of_ccs_calculated++;
				temp_float = current_correlation_position;
				JobResult *temp_result = new JobResult;
				temp_result->SetResult(1, &temp_float);
				AddJobToResultQueue(temp_result);
			}
		}
	}

	if (is_running_locally == true)
	{
		delete my_progress;

		// scale images..

		for (pixel_counter = 0; pixel_counter <  correlation_pixel_sum.real_memory_allocated; pixel_counter++)
		{

			correlation_pixel_sum.real_values[pixel_counter] /= float(total_correlation_positions);
			correlation_pixel_sum_of_squares.real_values[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares.real_values[pixel_counter] / float(total_correlation_positions) - powf(correlation_pixel_sum.real_values[pixel_counter], 2)) * sqrtf(correlation_pixel_sum.logical_x_dimension * correlation_pixel_sum.logical_y_dimension);
		}


		max_intensity_projection.MultiplyByConstant(sqrtf(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension));
		correlation_pixel_sum.MultiplyByConstant(sqrtf(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension));
//		correlation_pixel_sum_of_squares.MultiplyByConstant(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension);

		// we need to quadrant swap the images, also shift them, with an extra pixel shift.  This is because I take the conjugate of the input image, not the reference..



		max_intensity_projection.InvertPixelOrder();
		max_intensity_projection.PhaseShift(max_intensity_projection.physical_address_of_box_center_x + 1, max_intensity_projection.physical_address_of_box_center_y + 1, 0);


		best_psi.InvertPixelOrder();
		best_psi.PhaseShift(best_psi.physical_address_of_box_center_x + 1, best_psi.physical_address_of_box_center_y + 1, 0);

		best_theta.InvertPixelOrder();
		best_theta.PhaseShift(best_theta.physical_address_of_box_center_x + 1, best_theta.physical_address_of_box_center_y + 1, 0);

		best_phi.InvertPixelOrder();
		best_phi.PhaseShift(best_phi.physical_address_of_box_center_x + 1, best_phi.physical_address_of_box_center_y + 1, 0);

		correlation_pixel_sum.InvertPixelOrder();
		correlation_pixel_sum.PhaseShift(correlation_pixel_sum.physical_address_of_box_center_x + 1, correlation_pixel_sum.physical_address_of_box_center_y + 1, 0);

		correlation_pixel_sum_of_squares.InvertPixelOrder();
		correlation_pixel_sum_of_squares.PhaseShift(correlation_pixel_sum_of_squares.physical_address_of_box_center_x + 1, correlation_pixel_sum_of_squares.physical_address_of_box_center_y + 1, 0);



		// calculate the expected threshold (from peter's paper)

		expected_threshold = sqrtf(2.0f)*erfcinv((2.0f*(1))/((input_image.logical_x_dimension * input_image.logical_y_dimension * double(total_correlation_positions))));

		// write out images..

		max_intensity_projection.QuickAndDirtyWriteSlice(mip_output_file.ToStdString(), 1);
		max_intensity_projection.SubtractImage(&correlation_pixel_sum);
		max_intensity_projection.DividePixelWise(correlation_pixel_sum_of_squares);
		max_intensity_projection.QuickAndDirtyWriteSlice(scaled_mip_output_file.ToStdString(), 1);

		correlation_pixel_sum.QuickAndDirtyWriteSlice(correlation_average_output_file.ToStdString(), 1);
		correlation_pixel_sum_of_squares.QuickAndDirtyWriteSlice(correlation_variance_output_file.ToStdString(), 1);
		best_psi.QuickAndDirtyWriteSlice(best_psi_output_file.ToStdString(), 1);
		best_theta.QuickAndDirtyWriteSlice(best_theta_output_file.ToStdString(), 1);
		best_phi.QuickAndDirtyWriteSlice(best_phi_output_file.ToStdString(), 1);

		// write out histogram..

		temp_float = histogram_min + (histogram_step / 2.0f); // start position
		NumericTextFile histogram_file(output_histogram_file, OPEN_TO_WRITE, 4);

		double *expected_survival_histogram = new double[histogram_number_of_points];
		double *survival_histogram = new double[histogram_number_of_points];
		ZeroDoubleArray(survival_histogram, histogram_number_of_points);

		for (int line_counter = 0; line_counter <= histogram_number_of_points; line_counter++)
		{
				expected_survival_histogram[line_counter] = (erfc((temp_float + histogram_step * float(line_counter))/sqrtf(2.0f))/2.0f)*(input_image.logical_x_dimension * input_image.logical_y_dimension * float(total_correlation_positions));
		}

		survival_histogram[histogram_number_of_points - 1] = histogram_data[histogram_number_of_points - 1];

		for (int line_counter = histogram_number_of_points - 2; line_counter >= 0 ; line_counter--)
		{
			survival_histogram[line_counter] = survival_histogram[line_counter + 1] + histogram_data[line_counter];
		}

		histogram_file.WriteCommentLine("Expected threshold = %.2f\n", expected_threshold);
		histogram_file.WriteCommentLine("histogram, expected histogram, survival histogram, expected survival histogram");

		for (int line_counter = 0; line_counter < histogram_number_of_points; line_counter++)
		{
			temp_double_array[0] = temp_float + histogram_step * float(line_counter);
			temp_double_array[1] = histogram_data[line_counter];
			temp_double_array[2] = survival_histogram[line_counter];
			temp_double_array[3] = expected_survival_histogram[line_counter];
			histogram_file.WriteLine(temp_double_array);
		}

		histogram_file.Close();

		// memory cleanup

		delete [] survival_histogram;
		delete [] expected_survival_histogram;

	}
	else
	{
		// send back the final images to master (who should merge them, and send to the gui)

		long result_array_counter;
		long number_of_result_floats = 4; // first  is x size, second float y size of images, 3rd is number allocated, 4th  float is number of doubles in the histogram
		long pixel_counter;
		float *pointer_to_histogram_data;

		pointer_to_histogram_data = (float *) histogram_data;

		number_of_result_floats += max_intensity_projection.real_memory_allocated * 6;
		number_of_result_floats += histogram_number_of_points * 2; // long for the y

		float *result = new float[number_of_result_floats];
		result[0] = max_intensity_projection.logical_x_dimension;
		result[1] = max_intensity_projection.logical_y_dimension;
		result[2] = max_intensity_projection.real_memory_allocated;
		result[3] = histogram_number_of_points;
		result[4] = actual_number_of_ccs_calculated;

		result_array_counter = 5;

		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
		{
			result[result_array_counter] = max_intensity_projection.real_values[pixel_counter];
			result_array_counter++;
		}

		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
		{
			result[result_array_counter] = best_psi.real_values[pixel_counter];
			result_array_counter++;
		}


		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
		{
			result[result_array_counter] = best_theta.real_values[pixel_counter];
			result_array_counter++;
		}


		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
		{
			result[result_array_counter] = best_phi.real_values[pixel_counter];
			result_array_counter++;
		}


		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
		{
			result[result_array_counter] = correlation_pixel_sum.real_values[pixel_counter];
			result_array_counter++;
		}


		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
		{
			result[result_array_counter] = correlation_pixel_sum_of_squares.real_values[pixel_counter];
			result_array_counter++;
		}

		for (pixel_counter = 0; pixel_counter < histogram_number_of_points * 2; pixel_counter++)
		{
			result[result_array_counter] = 	pointer_to_histogram_data[pixel_counter];
			result_array_counter++;
		}

		SendProgramDefinedResultToMaster(result, number_of_result_floats, image_number_for_gui, number_of_jobs_per_image_in_gui);
	}

	delete [] histogram_data;

	if (is_running_locally == true)
	{
		wxPrintf("\nMatch Template: Normal termination\n");
		wxDateTime finish_time = wxDateTime::Now();
		wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
	}

	return true;
}

void MatchTemplateMLApp::MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results)
{
	if (collated_data_array == NULL)
	{
		collated_data_array = new float[array_size];
		ZeroFloatArray(collated_data_array, array_size);
		number_of_received_results = 0;
		total_number_of_ccs = 0.0f;

	}

	collated_data_array[0] = result_array[0];
	collated_data_array[1] = result_array[1];
	collated_data_array[2] = result_array[2];
	collated_data_array[3] = result_array[3];

	total_number_of_ccs += result_array[4];


	float *collated_mip_data = &collated_data_array[4];
	float *collated_psi_data = &collated_data_array[4 + int(result_array[2])];
	float *collated_theta_data = &collated_data_array[4 + int(result_array[2]) + int(result_array[2])];
	float *collated_phi_data = &collated_data_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	float *collated_pixel_sums = &collated_data_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	float *collated_pixel_square_sums = &collated_data_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];

	float *result_mip_data = &result_array[5];
	float *result_psi_data = &result_array[5 + int(result_array[2])];
	float *result_theta_data = &result_array[5 + int(result_array[2]) + int(result_array[2])];
	float *result_phi_data = &result_array[5  + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	float *result_pixel_sums = &result_array[5  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	float *result_pixel_square_sums = &result_array[5  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];

	//long *collated_histogram_data = (long *) &collated_data_array[array_size - int(collated_data_array[3] * 2) - 1];
	//long *input_histogram_data = (long *) &result_array[array_size - int(result_array[3] * 2) - 1];

	long *collated_histogram_data = (long *) &collated_data_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	long *input_histogram_data = (long *) &result_array[5  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];

	long pixel_counter;
	long result_array_counter;

	// handle the images..

	for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
	{
		if (result_mip_data[pixel_counter] >  collated_mip_data[pixel_counter])
		{
			collated_mip_data[pixel_counter] = result_mip_data[pixel_counter];
			collated_psi_data[pixel_counter] = result_psi_data[pixel_counter];
			collated_theta_data[pixel_counter] = result_theta_data[pixel_counter];
			collated_phi_data[pixel_counter] = result_phi_data[pixel_counter];

		}
	}

	// sums and sum of squares

	for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
	{
		collated_pixel_sums[pixel_counter] += result_pixel_sums[pixel_counter];
	}

	for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
	{
		collated_pixel_square_sums[pixel_counter] += result_pixel_square_sums[pixel_counter];
	}


	// handle the histogram..


	for (pixel_counter = 0; pixel_counter < histogram_number_of_points; pixel_counter++)
	{
	//	wxPrintf("Adding %li to %li\n", input_histogram_data[pixel_counter], collated_histogram_data[pixel_counter]);
		collated_histogram_data[pixel_counter] += input_histogram_data[pixel_counter];
	}


	// how do I know when I have finished?

	number_of_received_results++;

	//wxPrintf("MASTER _ Got %i of %i\n", number_of_received_results, number_of_expected_results);

	if (number_of_received_results == number_of_expected_results) // we should be done..
	{
		Image temp_image;

		temp_image.Allocate(int(collated_data_array[0]), int(collated_data_array[1]), true);

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_mip_data[pixel_counter];
		}

		temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));
		temp_image.InvertPixelOrder();
		temp_image.PhaseShift(temp_image.physical_address_of_box_center_x + 1, temp_image.physical_address_of_box_center_y + 1, 0);


		temp_image.QuickAndDirtyWriteSlice("/tmp/mip.mrc", 1);

		// psi

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_psi_data[pixel_counter];
		}

		temp_image.InvertPixelOrder();
		temp_image.PhaseShift(temp_image.physical_address_of_box_center_x + 1, temp_image.physical_address_of_box_center_y + 1, 0);
		temp_image.QuickAndDirtyWriteSlice("/tmp/psi.mrc", 1);

		//theta

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_theta_data[pixel_counter];
		}

		temp_image.InvertPixelOrder();
		temp_image.PhaseShift(temp_image.physical_address_of_box_center_x + 1, temp_image.physical_address_of_box_center_y + 1, 0);
		temp_image.QuickAndDirtyWriteSlice("/tmp/theta.mrc", 1);

		// phi

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_phi_data[pixel_counter];
		}

		temp_image.InvertPixelOrder();
		temp_image.PhaseShift(temp_image.physical_address_of_box_center_x + 1, temp_image.physical_address_of_box_center_y + 1, 0);
		temp_image.QuickAndDirtyWriteSlice("/tmp/phi.mrc", 1);

		/*
		// sums

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_pixel_sums[pixel_counter] / total_number_of_ccs;
		}

		//temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));
		temp_image.SwapRealSpaceQuadrants();
		temp_image.QuickAndDirtyWriteSlice("/tmp/sums.mrc", 1);

		// square sums

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = sqrtf(collated_pixel_square_sums[pixel_counter] / total_number_of_ccs - pow(collated_pixel_sums[pixel_counter] / total_number_of_ccs, 2));
		}

		temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));
		temp_image.SwapRealSpaceQuadrants();
		temp_image.QuickAndDirtyWriteSlice("/tmp/square_sums.mrc", 1);
*/
		// do the scaling..

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			//correlation_pixel_sum.real_values[pixel_counter] /= fabsf(float(correlation_pixel_sum_of_squares.real_values[pixel_counter] / float(total_correlation_positions) - powf(correlation_pixel_sum.real_values[pixel_counter] / float(total_correlation_positions), 2)));
			collated_pixel_sums[pixel_counter] /= total_number_of_ccs;
			collated_pixel_square_sums[pixel_counter] = sqrtf(collated_pixel_square_sums[pixel_counter] / total_number_of_ccs - pow(collated_pixel_sums[pixel_counter], 2)) * sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension);
			collated_mip_data[pixel_counter] = (collated_mip_data[pixel_counter] - collated_pixel_sums[pixel_counter]) / collated_pixel_square_sums[pixel_counter];
		}

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_mip_data[pixel_counter];
		}

		temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));
		temp_image.InvertPixelOrder();
		temp_image.PhaseShift(temp_image.physical_address_of_box_center_x + 1, temp_image.physical_address_of_box_center_y + 1, 0);
		temp_image.QuickAndDirtyWriteSlice("/tmp/scaled_mip.mrc", 1);

		// sums

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_pixel_sums[pixel_counter];
		}

		temp_image.InvertPixelOrder();
		temp_image.PhaseShift(temp_image.physical_address_of_box_center_x + 1, temp_image.physical_address_of_box_center_y + 1, 0);
		temp_image.QuickAndDirtyWriteSlice("/tmp/sums.mrc", 1);

		// square sums

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = collated_pixel_square_sums[pixel_counter];
		}

		//temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));

		temp_image.InvertPixelOrder();
		temp_image.PhaseShift(temp_image.physical_address_of_box_center_x + 1, temp_image.physical_address_of_box_center_y + 1, 0);
		temp_image.QuickAndDirtyWriteSlice("/tmp/square_sums.mrc", 1);


		/// histogram

		float histogram_step = (histogram_max - histogram_min) / float(histogram_number_of_points);
		float temp_float = histogram_min + (histogram_step / 2.0f); // start position
		NumericTextFile histogram_file("/tmp/histogram.txt", OPEN_TO_WRITE, 4);

		double *expected_survival_histogram = new double[histogram_number_of_points];
		double *survival_histogram = new double[histogram_number_of_points];

		float expected_threshold;

		double temp_double_array[5];

		ZeroDoubleArray(survival_histogram, histogram_number_of_points);
		survival_histogram[histogram_number_of_points - 1] = collated_histogram_data[histogram_number_of_points - 1];

		for (int line_counter = histogram_number_of_points - 2; line_counter >= 0 ; line_counter--)
		{
			survival_histogram[line_counter] = survival_histogram[line_counter + 1] + collated_histogram_data[line_counter];
		}

		for (int line_counter = 0; line_counter <= histogram_number_of_points; line_counter++)
		{
//			expected_survival_histogram[line_counter] = (erfc((temp_float + histogram_step * float(line_counter))/sqrtf(2.0f))/2.0f)*(double(survival_histogram[0]));
			expected_survival_histogram[line_counter] = (erfc((temp_float + histogram_step * float(line_counter))/sqrtf(2.0f))/2.0f)*(collated_data_array[0] * collated_data_array[1] * total_number_of_ccs);
		}


		expected_threshold = sqrtf(2.0f)*erfcinv((2.0f*(1))/(((collated_data_array[0] * collated_data_array[1] * total_number_of_ccs))));

		histogram_file.WriteCommentLine("Expected threshold = %.2f\n", expected_threshold);
		histogram_file.WriteCommentLine("histogram, expected histogram, survival histogram, expected survival histogram");

		for (int line_counter = 0; line_counter < histogram_number_of_points; line_counter++)
		{
			temp_double_array[0] = temp_float + histogram_step * float(line_counter);
			temp_double_array[1] = collated_histogram_data[line_counter];
			temp_double_array[2] = survival_histogram[line_counter];
			temp_double_array[3] = expected_survival_histogram[line_counter];
			histogram_file.WriteLine(temp_double_array);
		}

		histogram_file.Close();

		delete expected_survival_histogram;
		delete survival_histogram;

		// reset

		ZeroFloatArray(collated_data_array, array_size);
		number_of_received_results = 0;
		active_result_number = -1;

	}


}
