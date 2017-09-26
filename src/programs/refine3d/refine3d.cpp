#include "../../core/core_headers.h"

class
Refine3DApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

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

IMPLEMENT_APP(Refine3DApp)

// override the DoInteractiveUserInput

void Refine3DApp::DoInteractiveUserInput()
{
	wxString	input_particle_images;
	wxString	input_parameter_file;
	wxString	input_reconstruction;
	wxString	input_reconstruction_statistics;
	bool		use_statistics;
	wxString	ouput_matching_projections;
	wxString	ouput_parameter_file;
	wxString	ouput_shift_file;
	wxString	my_symmetry = "C1";
	int			first_particle = 1;
	int			last_particle = 0;
	float		percent_used = 1.0;
	float		pixel_size = 1.0;
	float		voltage_kV = 300.0;
	float		spherical_aberration_mm = 2.7;
	float		amplitude_contrast = 0.07;
	float		molecular_mass_kDa = 1000.0;
	float		inner_mask_radius = 0.0;
	float		outer_mask_radius = 100.0;
	float		low_resolution_limit = 300.0;
	float		high_resolution_limit = 8.0;
	float		signed_CC_limit = 0.0;
	float		classification_resolution_limit = 0.0;
	float		mask_radius_search = 0.0;
	float		high_resolution_limit_search = 20.0;
	float		angular_step = 5.0;
	int			best_parameters_to_keep = 20;
	float		max_search_x = 0;
	float		max_search_y = 0;
	float		mask_center_2d_x = 100.0;
	float		mask_center_2d_y = 100.0;
	float		mask_center_2d_z = 100.0;
	float		mask_radius_2d = 100.0;
	float 		defocus_search_range = 500;
	float 		defocus_step = 50;
	float		padding = 1.0;
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
	bool		ctf_refinement = false;
	bool		normalize_particles = true;
	bool		invert_contrast = false;
	bool		exclude_blank_edges = true;
	bool		normalize_input_3d = true;
	bool		threshold_input_3d = true;

	UserInput *my_input = new UserInput("Refine3D", 1.02);

	input_particle_images = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	input_reconstruction = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	input_reconstruction_statistics = my_input->GetFilenameFromUser("Input data statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
	use_statistics = my_input->GetYesNoFromUser("Use statistics", "Answer No if no statistics are available?", "Yes");
	ouput_matching_projections = my_input->GetFilenameFromUser("Output matching projections", "The output image stack, containing the matching projections", "my_projection_stack.mrc", false);
	ouput_parameter_file = my_input->GetFilenameFromUser("Output parameter file", "The output parameter file, containing your refined particle alignment parameters", "my_refined_parameters.par", false);
	ouput_shift_file = my_input->GetFilenameFromUser("Output parameter changes", "The changes in the alignment parameters compared to the input parameters", "my_parameter_changes.par", false);
	my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
	first_particle = my_input->GetIntFromUser("First particle to refine (0 = first in stack)", "The first particle in the stack that should be refined", "1", 0);
	last_particle = my_input->GetIntFromUser("Last particle to refine (0 = last in stack)", "The last particle in the stack that should be refined", "0", 0);
	percent_used = my_input->GetFloatFromUser("Percent of particles to use (1 = all)", "The percentage of randomly selected particles that will be refined", "1.0", 0.0, 1.0);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the input reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the input reconstruction and images during refinement, in Angstroms", "100.0", inner_mask_radius);
	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	signed_CC_limit = my_input->GetFloatFromUser("Resolution limit for signed CC (A) (0.0 = max)", "The absolute value of the weighted Fourier ring correlation will be used beyond this limit", "0.0", 0.0);
	classification_resolution_limit = my_input->GetFloatFromUser("Res limit for classification (A) (0.0 = max)", "Resolution limit of the data used for calculating LogP", "0.0", 0.0);
	mask_radius_search = my_input->GetFloatFromUser("Mask radius for global search (A) (0.0 = max)", "Radius of a circular mask to be applied to the input images during global search", "100.0", 0.0);
	high_resolution_limit_search = my_input->GetFloatFromUser("Approx. resolution limit for search (A)", "High resolution limit of the data used in the global search in Angstroms", "20.0", 0.0);
	angular_step = my_input->GetFloatFromUser("Angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
	best_parameters_to_keep = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
	max_search_x = my_input->GetFloatFromUser("Search range in X (A) (0.0 = 0.5 * mask radius)", "The maximum global peak search distance along X from the particle box center", "0.0", 0.0);
	max_search_y = my_input->GetFloatFromUser("Search range in Y (A) (0.0 = 0.5 * mask radius)", "The maximum global peak search distance along Y from the particle box center", "0.0", 0.0);
	mask_center_2d_x = my_input->GetFloatFromUser("2D mask X coordinate (A)", "X coordinate of 2D mask center", "100.0", 0.0);
	mask_center_2d_y = my_input->GetFloatFromUser("2D mask Y coordinate (A)", "Y coordinate of 2D mask center", "100.0", 0.0);
	mask_center_2d_z = my_input->GetFloatFromUser("2D mask Z coordinate (A)", "Z coordinate of 2D mask center", "100.0", 0.0);
	mask_radius_2d = my_input->GetFloatFromUser("2D mask radius (A)", "Radius of a circular mask to be used for likelihood calculation", "100.0", 0.0);
	defocus_search_range = my_input->GetFloatFromUser("Defocus search range (A)", "Search range (-value ... + value) around current defocus", "500.0", 0.0);
	defocus_step = my_input->GetFloatFromUser("Defocus step (A)", "Step size used in the defocus search", "50.0", 0.0);
	padding = my_input->GetFloatFromUser("Tuning parameters: padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
//	filter_constant = my_input->GetFloatFromUser("Tuning parameters: filter constant", "Constant determining how strongly data with small CTF values is suppressed during particle alignment", "1.0", 1.0);
	global_search = my_input->GetYesNoFromUser("Global search", "Should a global search be performed before local refinement?", "No");
	local_refinement = my_input->GetYesNoFromUser("Local refinement", "Should a local parameter refinement be performed?", "Yes");
	refine_psi = my_input->GetYesNoFromUser("Refine Psi", "Should the Psi Euler angle be refined (parameter 1)?", "Yes");
	refine_theta = my_input->GetYesNoFromUser("Refine Theta", "Should the Theta Euler angle be refined (parameter 2)?", "Yes");
	refine_phi = my_input->GetYesNoFromUser("Refine Phi", "Should the Phi Euler angle be refined (parameter 3)?", "Yes");
	refine_x = my_input->GetYesNoFromUser("Refine ShiftX", "Should the X shift be refined (parameter 4)?", "Yes");
	refine_y = my_input->GetYesNoFromUser("Refine ShiftY", "Should the Y shift be refined (parameter 5)?", "Yes");
	calculate_matching_projections = my_input->GetYesNoFromUser("Calculate matching projections", "Should matching projections be calculated?", "No");
	apply_2D_masking = my_input->GetYesNoFromUser("Apply 2D masking", "Should 2D masking be used for the likelihood calculation?", "No");
	ctf_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
	normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
	invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	exclude_blank_edges = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "Yes");
	normalize_input_3d = my_input->GetYesNoFromUser("Normalize input reconstruction", "The input reconstruction should always be normalized unless it was generated by reconstruct3d with normalized particles", "Yes");
	threshold_input_3d = my_input->GetYesNoFromUser("Threshold input reconstruction", "Should the input reconstruction thresholded to suppress some of the background noise", "No");
	// Add phase flip option, normalize option, remove input statistics & use statistics

	delete my_input;

	bool local_global_refine = false;
	int current_class = 0;
	bool ignore_input_angles = false;
	my_current_job.Reset(54);
	my_current_job.ManualSetArguments("ttttbttttiifffffffffffffffifffffffffbbbbbbbbbbbbbbbbib",	input_particle_images.ToUTF8().data(),
																								input_parameter_file.ToUTF8().data(),
																								input_reconstruction.ToUTF8().data(),
																								input_reconstruction_statistics.ToUTF8().data(), use_statistics,
																								ouput_matching_projections.ToUTF8().data(),
																								ouput_parameter_file.ToUTF8().data(),
																								ouput_shift_file.ToUTF8().data(),
																								my_symmetry.ToUTF8().data(),
																								first_particle, last_particle, percent_used,
																								pixel_size, voltage_kV, spherical_aberration_mm, amplitude_contrast,
																								molecular_mass_kDa, inner_mask_radius, outer_mask_radius, low_resolution_limit,
																								high_resolution_limit, signed_CC_limit, classification_resolution_limit,
																								mask_radius_search, high_resolution_limit_search, angular_step, best_parameters_to_keep,
																								max_search_x, max_search_y,
																								mask_center_2d_x, mask_center_2d_y, mask_center_2d_z, mask_radius_2d,
																								defocus_search_range, defocus_step, padding,
																								global_search, local_refinement,
																								refine_psi, refine_theta, refine_phi, refine_x, refine_y,
																								calculate_matching_projections, apply_2D_masking, ctf_refinement, normalize_particles,
																								invert_contrast, exclude_blank_edges, normalize_input_3d, threshold_input_3d,
																								local_global_refine, current_class, ignore_input_angles);
}

// override the do calculation method which will be what is actually run..

bool Refine3DApp::DoCalculation()
{
	Particle refine_particle;
	Particle search_particle;

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

	refine_particle.constraints_used[4] = true;		// Constraint for X shifts
	refine_particle.constraints_used[5] = true;		// Constraint for Y shifts

	Image input_image;
//	Image ctf_input_image;
	Image projection_image;
	Image search_projection_image;
	Image unbinned_image;
	Image binned_image;
	Image final_image;
	Image temp_image;
	Image temp_image2;
	Image sum_power;
	Image *projection_cache = NULL;
//	CTF   my_ctf;
	CTF   input_ctf;
	Image snr_image;
	ReconstructedVolume			input_3d;
	ReconstructedVolume			search_reference_3d;
	ImageProjectionComparison	comparison_object;
	ConjugateGradient			conjugate_gradient_minimizer;
	EulerSearch					global_euler_search;
//	Kernel2D					**kernel_index = NULL;
	Curve						noise_power_spectrum;
	Curve						number_of_terms;
	RandomNumberGenerator		random_particle(true);
	ProgressBar					*my_progress;

	JobResult *intermediate_result;

	int i;
	int j;
	int fourier_size_x, fourier_size_y, fourier_size_z;
	int current_image;
	int images_to_process = 0;
	int image_counter = 0;
	int defocus_i;
	int best_defocus_i;
	int search_box_size;
	int result_parameter_counter;
	int number_of_blank_edges;
	int max_samples = 2000;
	int istart;
	float input_parameters[refine_particle.number_of_parameters];
	float output_parameters[refine_particle.number_of_parameters];
	float gui_result_parameters[refine_particle.number_of_parameters];
	float search_parameters[refine_particle.number_of_parameters];
	float parameter_average[refine_particle.number_of_parameters];
	float parameter_variance[refine_particle.number_of_parameters];
	float output_parameter_average[refine_particle.number_of_parameters];
	float output_parameter_change[refine_particle.number_of_parameters];
	float cg_starting_point[refine_particle.number_of_parameters];
	float cg_accuracy[refine_particle.number_of_parameters];
	float binning_factor_refine;
	float binning_factor_search;
	float mask_falloff = 20.0;	// in Angstrom
//	float alpha;
//	float sigma;
	float logp;
	float temp_float;
	float psi;
	float psi_max;
	float psi_step;
	float psi_start;
	float score;
	float best_score;
	float mask_radius_for_noise;
	float percentage;
	float variance;
	float average;
	float average_density_max;
	bool skip_local_refinement = false;
	wxDateTime my_time_in;
//	wxDateTime my_time_out;

	ZeroFloatArray(input_parameters, refine_particle.number_of_parameters);
	ZeroFloatArray(output_parameters, refine_particle.number_of_parameters);
	ZeroFloatArray(search_parameters, refine_particle.number_of_parameters);
	ZeroFloatArray(parameter_average, refine_particle.number_of_parameters);
	ZeroFloatArray(parameter_variance, refine_particle.number_of_parameters);
	ZeroFloatArray(output_parameter_average, refine_particle.number_of_parameters);
	ZeroFloatArray(output_parameter_change, refine_particle.number_of_parameters);
	ZeroFloatArray(cg_starting_point, refine_particle.number_of_parameters);
	ZeroFloatArray(cg_accuracy, refine_particle.number_of_parameters);

	if (! DoesFileExist(input_parameter_file))
	{
		SendError(wxString::Format("Error: Input parameter file %s not found\n", input_parameter_file));
		exit(-1);
	}
	if (! DoesFileExist(input_particle_images))
	{
		SendError(wxString::Format("Error: Input particle stack %s not found\n", input_particle_images));
		exit(-1);
	}
	if (! DoesFileExist(input_reconstruction))
	{
		SendError(wxString::Format("Error: Input reconstruction %s not found\n", input_reconstruction));
		exit(-1);
	}
	//	wxPrintf("\nOpening input file %s.\n", input_parameter_file);
	FrealignParameterFile input_par_file(input_parameter_file, OPEN_TO_READ);
	MRCFile input_stack(input_particle_images.ToStdString(), false);

	input_par_file.ReadFile(false, input_stack.ReturnZSize());
	random_particle.SetSeed(int(10000.0 * fabsf(input_par_file.ReturnAverage(14, true)))%10000);

	MRCFile input_file(input_reconstruction.ToStdString(), false, true);
	MRCFile *output_file;
	if (percent_used < 1.0 && calculate_matching_projections)
	{
		calculate_matching_projections = false;
		wxPrintf("\nPercent of particles used < 1, matching projections not calculated.\n");
	}
	if (calculate_matching_projections) output_file = new MRCFile(ouput_matching_projections.ToStdString(), true);
	FrealignParameterFile my_output_par_file(ouput_parameter_file, OPEN_TO_WRITE);
	FrealignParameterFile my_output_par_shifts_file(ouput_shift_file, OPEN_TO_WRITE, 16);

	if (input_stack.ReturnXSize() != input_stack.ReturnYSize())
	{
		SendError("Error: Particles are not square\n");
		input_stack.PrintInfo();
		exit(-1);
	}
	if ((input_file.ReturnXSize() != input_file.ReturnYSize()) || (input_file.ReturnXSize() != input_file.ReturnZSize()))
	{
		SendError("Error: Input reconstruction is not cubic\n");
		input_file.PrintInfo();
		exit(-1);
	}
	if (input_file.ReturnXSize() != input_stack.ReturnXSize())
	{
		SendError("Error: Dimension of particles and input reconstruction differ\n");
		input_file.PrintInfo();
		input_stack.PrintInfo();
		exit(-1);
	}
	if (last_particle < first_particle && last_particle != 0)
	{
		SendError("Error: Number of last particle to refine smaller than number of first particle to refine\n");
		exit(-1);
	}

	if (last_particle == 0) last_particle = input_stack.ReturnZSize();
	if (first_particle == 0) first_particle = 1;
	if (last_particle > input_stack.ReturnZSize()) last_particle = input_stack.ReturnZSize();

	if (max_search_x == 0.0) max_search_x = mask_radius_search;
	if (max_search_y == 0.0) max_search_y = mask_radius_search;

	my_time_in = wxDateTime::Now();
	my_output_par_file.WriteCommentLine("C Refine3D run date and time:              " + my_time_in.FormatISOCombined(' '));
	my_output_par_file.WriteCommentLine("C Input particle images:                   " + input_particle_images);
	my_output_par_file.WriteCommentLine("C Input Frealign parameter filename:       " + input_parameter_file);
	my_output_par_file.WriteCommentLine("C Input reconstruction:                    " + input_reconstruction);
	my_output_par_file.WriteCommentLine("C Input data statistics:                   " + input_reconstruction_statistics);
	my_output_par_file.WriteCommentLine("C Use statistics:                          " + BoolToYesNo(use_statistics));
	my_output_par_file.WriteCommentLine("C Output matching projections:             " + ouput_matching_projections);
	my_output_par_file.WriteCommentLine("C Output parameter file:                   " + ouput_parameter_file);
	my_output_par_file.WriteCommentLine("C Output parameter changes:                " + ouput_shift_file);
	my_output_par_file.WriteCommentLine("C Particle symmetry:                       " + my_symmetry);
	my_output_par_file.WriteCommentLine("C First particle to refine:                " + wxString::Format("%i", first_particle));
	my_output_par_file.WriteCommentLine("C Last particle to refine:                 " + wxString::Format("%i", last_particle));
	my_output_par_file.WriteCommentLine("C Percent of particles to refine:          " + wxString::Format("%f", percent_used));
	my_output_par_file.WriteCommentLine("C Pixel size of images (A):                " + wxString::Format("%f", pixel_size));
	my_output_par_file.WriteCommentLine("C Beam energy (keV):                       " + wxString::Format("%f", voltage_kV));
	my_output_par_file.WriteCommentLine("C Spherical aberration (mm):               " + wxString::Format("%f", spherical_aberration_mm));
	my_output_par_file.WriteCommentLine("C Amplitude contrast:                      " + wxString::Format("%f", amplitude_contrast));
	my_output_par_file.WriteCommentLine("C Molecular mass of particle (kDa):        " + wxString::Format("%f", molecular_mass_kDa));
	my_output_par_file.WriteCommentLine("C Inner mask radius for refinement (A):    " + wxString::Format("%f", inner_mask_radius));
	my_output_par_file.WriteCommentLine("C Outer mask radius for refinement (A):    " + wxString::Format("%f", outer_mask_radius));
	my_output_par_file.WriteCommentLine("C Low resolution limit (A):                " + wxString::Format("%f", low_resolution_limit));
	my_output_par_file.WriteCommentLine("C High resolution limit (A):               " + wxString::Format("%f", high_resolution_limit));
	my_output_par_file.WriteCommentLine("C Resolution limit for signed CC (A):      " + wxString::Format("%f", signed_CC_limit));
	my_output_par_file.WriteCommentLine("C Res limit for classification (A):        " + wxString::Format("%f", classification_resolution_limit));
	my_output_par_file.WriteCommentLine("C Mask radius for global search (A):       " + wxString::Format("%f", mask_radius_search));
	my_output_par_file.WriteCommentLine("C Approx. resolution limit for search (A): " + wxString::Format("%f", high_resolution_limit_search));
	my_output_par_file.WriteCommentLine("C Angular step:                            " + wxString::Format("%f", angular_step));
	my_output_par_file.WriteCommentLine("C Number of top hits to refine:            " + wxString::Format("%i", best_parameters_to_keep));
	my_output_par_file.WriteCommentLine("C Search range in X (A):                   " + wxString::Format("%f", max_search_x));
	my_output_par_file.WriteCommentLine("C Search range in Y (A):                   " + wxString::Format("%f", max_search_y));
	my_output_par_file.WriteCommentLine("C 2D mask X coordinate (A):                " + wxString::Format("%f", refine_particle.mask_center_2d_x));
	my_output_par_file.WriteCommentLine("C 2D mask Y coordinate (A):                " + wxString::Format("%f", refine_particle.mask_center_2d_y));
	my_output_par_file.WriteCommentLine("C 2D mask Z coordinate (A):                " + wxString::Format("%f", refine_particle.mask_center_2d_z));
	my_output_par_file.WriteCommentLine("C 2D mask radius (A):                      " + wxString::Format("%f", refine_particle.mask_radius_2d));
	my_output_par_file.WriteCommentLine("C Defocus search range (A):                " + wxString::Format("%f", defocus_search_range));
	my_output_par_file.WriteCommentLine("C Defocus step (A):                        " + wxString::Format("%f", defocus_step));
	my_output_par_file.WriteCommentLine("C Padding factor:                          " + wxString::Format("%f", padding));
//	my_output_par_file.WriteCommentLine("C Filter constant:                         " + wxString::Format("%f", filter_constant));
	my_output_par_file.WriteCommentLine("C Global search:                           " + BoolToYesNo(global_search));
	my_output_par_file.WriteCommentLine("C Local refinement:                        " + BoolToYesNo(local_refinement));
	my_output_par_file.WriteCommentLine("C Refine Psi:                              " + BoolToYesNo(refine_particle.parameter_map[3]));
	my_output_par_file.WriteCommentLine("C Refine Theta:                            " + BoolToYesNo(refine_particle.parameter_map[2]));
	my_output_par_file.WriteCommentLine("C Refine Phi:                              " + BoolToYesNo(refine_particle.parameter_map[1]));
	my_output_par_file.WriteCommentLine("C Refine ShiftX:                           " + BoolToYesNo(refine_particle.parameter_map[4]));
	my_output_par_file.WriteCommentLine("C Refine ShiftY:                           " + BoolToYesNo(refine_particle.parameter_map[5]));
	my_output_par_file.WriteCommentLine("C Calculate matching projections:          " + BoolToYesNo(calculate_matching_projections));
	my_output_par_file.WriteCommentLine("C Apply 2D masking:                        " + BoolToYesNo(refine_particle.apply_2D_masking));
	my_output_par_file.WriteCommentLine("C Refine defocus:                          " + BoolToYesNo(ctf_refinement));
	my_output_par_file.WriteCommentLine("C Normalize particles:                     " + BoolToYesNo(normalize_particles));
	my_output_par_file.WriteCommentLine("C Invert particle contrast:                " + BoolToYesNo(invert_contrast));
	my_output_par_file.WriteCommentLine("C Exclude images with blank edges:         " + BoolToYesNo(exclude_blank_edges));
	my_output_par_file.WriteCommentLine("C Normalize input reconstruction:          " + BoolToYesNo(normalize_input_3d));
	my_output_par_file.WriteCommentLine("C Threshold input reconstruction:          " + BoolToYesNo(threshold_input_3d));
	my_output_par_file.WriteCommentLine("C");
//	my_output_par_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP NormSigmaNoise          Score         Change");
	my_output_par_file.WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE");
	fflush(my_output_par_file.parameter_file);
//	my_output_par_shifts_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP NormSigmaNoise          Score");
	my_output_par_shifts_file.WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE");

	if (! refine_particle.parameter_map[1] && ! refine_particle.parameter_map[2] && ! refine_particle.parameter_map[3] && ! refine_particle.parameter_map[4] && ! refine_particle.parameter_map[5])
	{
		local_refinement = false;
		global_search = false;
	}

	if (local_global_refine)
	{
		refine_particle.parameter_map[1] = true;
		refine_particle.parameter_map[2] = true;
		refine_particle.parameter_map[3] = true;
		refine_particle.parameter_map[4] = true;
		refine_particle.parameter_map[5] = true;
		local_refinement = true;
		global_search = true;
	}

	if (high_resolution_limit < 2.0 * pixel_size) high_resolution_limit = 2.0 * pixel_size;
	if (classification_resolution_limit < 2.0 * pixel_size) classification_resolution_limit = 2.0 * pixel_size;
	if (high_resolution_limit_search < 2.0 * pixel_size) high_resolution_limit_search = 2.0 * pixel_size;
	if (signed_CC_limit == 0.0) signed_CC_limit = pixel_size;

	if (outer_mask_radius > float(input_stack.ReturnXSize()) / 2.0 * pixel_size- mask_falloff) outer_mask_radius = float(input_stack.ReturnXSize()) / 2.0 * pixel_size - mask_falloff;
	if (mask_radius_search > float(input_stack.ReturnXSize()) / 2.0 * pixel_size- mask_falloff) mask_radius_search = float(input_stack.ReturnXSize()) / 2.0 * pixel_size - mask_falloff;

	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, my_symmetry);
	input_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	ResolutionStatistics input_statistics(pixel_size, input_3d.density_map.logical_y_dimension);
	ResolutionStatistics search_statistics;
	ResolutionStatistics refine_statistics;
	if (use_statistics)
	{
		if (! DoesFileExist(input_reconstruction_statistics))
		{
			SendError(wxString::Format("Error: Input statistics %s not found\n", input_reconstruction_statistics));
			exit(-1);
		}
		input_statistics.ReadStatisticsFromFile(input_reconstruction_statistics);
	}
	else
	{
		wxPrintf("\nUsing default statistics\n");
		input_statistics.GenerateDefaultStatistics(molecular_mass_kDa);
	}
	refine_statistics = input_statistics;
	input_3d.density_map.ReadSlices(&input_file,1,input_3d.density_map.logical_z_dimension);
//!!! This line is incompatible with ML !!!
//	input_3d.density_map.CosineMask(outer_mask_radius / pixel_size, mask_falloff / pixel_size);
//	input_3d.density_map.AddConstant(- input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
	// Remove masking here to avoid edge artifacts later
	input_3d.density_map.CosineMask(outer_mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
	if (inner_mask_radius > 0.0) input_3d.density_map.CosineMask(inner_mask_radius / pixel_size, mask_falloff / pixel_size, true);
//	for (i = 0; i < input_3d.density_map.real_memory_allocated; i++) if (input_3d.density_map.real_values[i] < 0.0) input_3d.density_map.real_values[i] = -log(-input_3d.density_map.real_values[i] + 1.0);
	if (threshold_input_3d)
	{
		average_density_max = input_3d.density_map.ReturnAverageOfMaxN(100, outer_mask_radius / pixel_size);
		input_3d.density_map.SetMinimumValue(-0.3 * average_density_max);
//		input_3d.density_map.SetMinimumValue(0.0);
	}

	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
//	if (outer_mask_radius > input_image.physical_address_of_box_center_x * pixel_size- mask_falloff) outer_mask_radius = input_image.physical_address_of_box_center_x * pixel_size - mask_falloff;
//	if (mask_radius_search > input_image.physical_address_of_box_center_x * pixel_size- mask_falloff) mask_radius_search = input_image.physical_address_of_box_center_x * pixel_size - mask_falloff;
	input_3d.mask_radius = outer_mask_radius;

	if (global_search)
	{
		if (best_parameters_to_keep == 0) {best_parameters_to_keep = 1; skip_local_refinement = true;}
		// Assume square particles
		search_reference_3d = input_3d;
		search_statistics = input_statistics;
		search_box_size = ReturnClosestFactorizedUpper(myroundint(2.0 * padding * (std::max(max_search_x, max_search_y) + mask_radius_search)), 3, true);
		if (search_box_size > search_reference_3d.density_map.logical_x_dimension) search_box_size = search_reference_3d.density_map.logical_x_dimension;
		if (search_box_size != search_reference_3d.density_map.logical_x_dimension) search_reference_3d.density_map.Resize(search_box_size, search_box_size, search_box_size);
//		search_reference_3d.PrepareForProjections(high_resolution_limit_search, true);
		search_reference_3d.PrepareForProjections(low_resolution_limit, high_resolution_limit_search, true);
//		search_statistics.Init(search_reference_3d.pixel_size, search_reference_3d.density_map.logical_y_dimension / 2 + 1);
		search_particle.Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension);
		search_projection_image.Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension, false);
		temp_image2.Allocate(search_box_size, search_box_size, true);
		binning_factor_search = search_reference_3d.pixel_size / pixel_size;
		//Scale to make projections compatible with images for ML calculation
		search_reference_3d.density_map.MultiplyByConstant(powf(powf(binning_factor_search, 1.0 / 3.0), 2));
		//if (angular_step <= 0) angular_step = 360.0 * high_resolution_limit_search / PI / outer_mask_radius;
		if (angular_step <= 0) angular_step = CalculateAngularStep(high_resolution_limit_search, outer_mask_radius);
		psi_step = rad_2_deg(search_reference_3d.pixel_size / outer_mask_radius);
		psi_step = 360.0 / int(360.0 / psi_step + 0.5);
		psi_start = psi_step / 2.0 * global_random_number_generator.GetUniformRandom();
		psi_max = 0.0;
		if (refine_particle.parameter_map[3]) psi_max = 360.0;
		wxPrintf("\nBox size for search = %i, binning factor = %f, new pixel size = %f, resolution limit = %f\nAngular step size = %f, in-plane = %f\n", search_box_size, binning_factor_search, search_reference_3d.pixel_size, search_reference_3d.pixel_size * 2.0, angular_step, psi_step);
	}

	if (padding != 1.0)
	{
		input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension * padding, input_3d.density_map.logical_y_dimension * padding, input_3d.density_map.logical_z_dimension * padding, input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
		refine_statistics.part_SSNR.ResampleCurve(&refine_statistics.part_SSNR, refine_statistics.part_SSNR.number_of_points * padding);
	}

//	input_3d.PrepareForProjections(high_resolution_limit);
	input_3d.PrepareForProjections(low_resolution_limit, high_resolution_limit);
	binning_factor_refine = input_3d.pixel_size / pixel_size;
	//Scale to make projections compatible with images for ML calculation
//	input_3d.density_map.MultiplyByConstant(binning_factor_refine);
//	input_3d.density_map.MultiplyByConstant(powf(powf(binning_factor_refine, 1.0 / 3.0), 2));
	wxPrintf("\nBinning factor for refinement = %f, new pixel size = %f\n", binning_factor_refine, input_3d.pixel_size);

	temp_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	sum_power.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	refine_particle.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension);
//	ctf_input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	projection_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	unbinned_image.Allocate(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, true);
	if (ctf_refinement) binned_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	final_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);

// Read whole parameter file to work out average values and variances
	j = 0;
	for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
	{
		input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		if (input_parameters[7] >= 0)
		{
			for (i = 0; i < refine_particle.number_of_parameters; i++)
			{
				parameter_average[i] += input_parameters[i];
				parameter_variance[i] += powf(input_parameters[i],2);
			}
			j++;
		}
		if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle) images_to_process++;
	}

	for (i = 0; i < refine_particle.number_of_parameters; i++)
	{
		parameter_average[i] /= j;
		parameter_variance[i] /= j;
		parameter_variance[i] -= powf(parameter_average[i],2);

		if (parameter_variance[i] < 0.001) refine_particle.constraints_used[i] = false;
	}
	refine_particle.SetParameterStatistics(parameter_average, parameter_variance);
	input_par_file.Rewind();

	if (normalize_particles)
	{
		wxPrintf("Calculating noise power spectrum...\n\n");
		percentage = float(max_samples) / float(images_to_process);
		sum_power.SetToConstant(0.0);
		mask_radius_for_noise = outer_mask_radius / pixel_size;
		number_of_blank_edges = 0;
		if (2.0 * mask_radius_for_noise + mask_falloff / pixel_size > 0.95 * input_image.logical_x_dimension)
		{
			mask_radius_for_noise = 0.95 * input_image.logical_x_dimension / 2.0 - mask_falloff / 2.0 / pixel_size;
		}
		noise_power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		my_progress = new ProgressBar(images_to_process);
		for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
		{
			input_par_file.ReadLine(input_parameters);
			if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
			image_counter++;
			my_progress->Update(image_counter);
			if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0 * percentage)) continue;
			input_image.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
			if (exclude_blank_edges && input_image.ContainsBlankEdges(outer_mask_radius / pixel_size)) {number_of_blank_edges++; continue;}
			variance = input_image.ReturnVarianceOfRealValues(outer_mask_radius / pixel_size, 0.0, 0.0, 0.0, true);
			if (variance == 0.0) continue;
			input_image.MultiplyByConstant(1.0 / sqrtf(variance));
			input_image.CosineMask(outer_mask_radius / pixel_size, mask_falloff / pixel_size, true);
			input_image.ForwardFFT();
			temp_image.CopyFrom(&input_image);
			temp_image.ConjugateMultiplyPixelWise(input_image);
			sum_power.AddImage(&temp_image);
		}
		delete my_progress;
		sum_power.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);
		noise_power_spectrum.SquareRoot();
		noise_power_spectrum.Reciprocal();

		input_par_file.Rewind();
		if (exclude_blank_edges)
		{
			wxPrintf("\nImages with blank edges excluded from noise power calculation = %i\n", number_of_blank_edges);
		}
	}

	if (global_search)
	{
		for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.parameter_map[i] = refine_particle.parameter_map[i];}
// Set parameter_map for x,y translations to true since they will always be searched and refined in a global search
// Decided not to do this to honor user request
//		search_particle.parameter_map[4] = true;
//		search_particle.parameter_map[5] = true;
		for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.constraints_used[i] = refine_particle.constraints_used[i];}
		search_particle.SetParameterStatistics(parameter_average, parameter_variance);

// Use projection_cache only if both phi and theta are searched; otherwise calculate projections on the fly
		if (search_particle.parameter_map[1] && search_particle.parameter_map[2])
		{
			global_euler_search.InitGrid(my_symmetry, angular_step, 0.0, 0.0, psi_max, psi_step, psi_start, search_reference_3d.pixel_size / high_resolution_limit_search, search_particle.parameter_map, best_parameters_to_keep);
			projection_cache = new Image [global_euler_search.number_of_search_positions];
			for (i = 0; i < global_euler_search.number_of_search_positions; i++)
			{
				projection_cache[i].Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension, false);
			}
			search_reference_3d.density_map.GenerateReferenceProjections(projection_cache, global_euler_search, search_reference_3d.pixel_size / high_resolution_limit_search);
			wxPrintf("\nNumber of global search views = %i\n", global_euler_search.number_of_search_positions);
		}
//		search_projection_image.RotateFourier2DGenerateIndex(kernel_index, psi_max, psi_step, psi_start);

		if (search_particle.parameter_map[4]) global_euler_search.max_search_x = max_search_x;
		else global_euler_search.max_search_x = 0.0;
		if (search_particle.parameter_map[5]) global_euler_search.max_search_y = max_search_y;
		else global_euler_search.max_search_y = 0.0;
	}

	wxPrintf("\nAverage sigma noise = %f, average LogP = %f\nAverage ShiftX = %f, average ShiftY = %f\nSigma ShiftX = %f, sigma ShiftY = %f\nNumber of particles to refine = %i\n\n",
			parameter_average[14], parameter_average[15], parameter_average[4], parameter_average[5], sqrtf(parameter_variance[4]), sqrtf(parameter_variance[5]), images_to_process);

	image_counter = 0;
	my_progress = new ProgressBar(images_to_process);
	for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
	{
		input_par_file.ReadLine(input_parameters);
		temp_float = random_particle.GetUniformRandom();
		if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
		image_counter++;
		if (temp_float < 1.0 - 2.0 * percent_used)
		{
			input_parameters[7] = -1;//- fabsf(input_parameters[7]);
			input_parameters[16] = 0.0;
			my_output_par_file.WriteLine(input_parameters);

			if (is_running_locally == false) // send results back to the gui..
			{
				intermediate_result = new JobResult;
				intermediate_result->job_number = my_current_job.job_number;

				gui_result_parameters[0] = current_class;
				for (result_parameter_counter = 1; result_parameter_counter < refine_particle.number_of_parameters + 1; result_parameter_counter++)
				{
					gui_result_parameters[result_parameter_counter] = input_parameters[result_parameter_counter - 1];
				}

				intermediate_result->SetResult(refine_particle.number_of_parameters + 1, gui_result_parameters);
				AddJobToResultQueue(intermediate_result);
			}

			for (i = 1; i < refine_particle.number_of_parameters; i++) output_parameters[i] = 0.0;
			output_parameters[0] = input_parameters[0];

			my_output_par_shifts_file.WriteLine(output_parameters);

			my_progress->Update(image_counter);
			continue;
		}
		else
		{
//			input_parameters[7] = fabsf(input_parameters[7]);
			temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		}
		for (i = 0; i < refine_particle.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}

		if (local_global_refine)
		{
			if (input_parameters[7] == 0.0) {local_refinement = false; global_search = true;}
			else {local_refinement = true; global_search = false;}
		}

// Set up Particle object
		refine_particle.ResetImageFlags();
		refine_particle.mask_radius = outer_mask_radius;
		refine_particle.mask_falloff = mask_falloff;
//		refine_particle.filter_radius_low = low_resolution_limit;
		refine_particle.filter_radius_high = high_resolution_limit;
		refine_particle.molecular_mass_kDa = molecular_mass_kDa;
		refine_particle.signed_CC_limit = signed_CC_limit;
		// The following line would allow using particles with different pixel sizes
		refine_particle.pixel_size = input_3d.pixel_size;
		refine_particle.is_normalized = normalize_particles;
		refine_particle.sigma_noise = input_parameters[14] / binning_factor_refine;
//		refine_particle.logp = -std::numeric_limits<float>::max();
		refine_particle.SetParameters(input_parameters);
		refine_particle.MapParameterAccuracy(cg_accuracy);
//		refine_particle.SetIndexForWeightedCorrelation();
		refine_particle.SetParameterConstraints(powf(parameter_average[14],2));

		input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, pixel_size, input_parameters[11]);
//		ctf_input_image.CalculateCTFImage(input_ctf);
//		refine_particle.is_phase_flipped = true;

		input_image.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
		input_image.ReplaceOutliersWithMean(5.0);
		if (invert_contrast) input_image.InvertRealValues();
		if (normalize_particles)
		{
			input_image.ForwardFFT();
			// Whiten noise
			input_image.ApplyCurveFilter(&noise_power_spectrum);
			// Apply cosine filter to reduce ringing
//			input_image.CosineMask(std::max(pixel_size / high_resolution_limit, pixel_size / 7.0f + pixel_size / mask_falloff) - pixel_size / (2.0 * mask_falloff), pixel_size / mask_falloff);
			input_image.BackwardFFT();
			// Normalize background variance and average
			variance = input_image.ReturnVarianceOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
			average = input_image.ReturnAverageOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, true);
			input_image.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
			// At this point, input_image should have white background with a variance of 1. The variance should therefore be about 1/binning_factor^2 after binning.
		}

		// Option to add noise to images to get out of local optima
//		input_image.AddGaussianNoise(sqrtf(2.0 * input_image.ReturnVarianceOfRealValues()));

		input_image.ClipInto(&unbinned_image);
		unbinned_image.ForwardFFT();
		unbinned_image.ClipInto(refine_particle.particle_image);
		// Multiply by binning_factor so variance after binning is close to 1.
//		refine_particle.particle_image->MultiplyByConstant(binning_factor_refine);
		comparison_object.reference_volume = &input_3d;
		comparison_object.projection_image = &projection_image;
		comparison_object.particle = &refine_particle;
		refine_particle.MapParameters(cg_starting_point);
		refine_particle.PhaseShiftInverse();

		if (ctf_refinement && high_resolution_limit <= 20.0)
		{
//			wxPrintf("\nRefining defocus for parameter line %i\n", current_image);
			refine_particle.filter_radius_low = 30.0;
			refine_particle.SetIndexForWeightedCorrelation();
			binned_image.CopyFrom(refine_particle.particle_image);
			refine_particle.InitCTF(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], input_parameters[11]);
			best_score = - std::numeric_limits<float>::max();
			for (defocus_i = - myround(float(defocus_search_range)/float(defocus_step)); defocus_i <= myround(float(defocus_search_range)/float(defocus_step)); defocus_i++)
			{
				refine_particle.SetDefocus(input_parameters[8] + defocus_i * defocus_step, input_parameters[9] + defocus_i * defocus_step, input_parameters[10], input_parameters[11]);
				refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8] + defocus_i * defocus_step, input_parameters[9] + defocus_i * defocus_step, input_parameters[10], input_parameters[11]);
				if (normalize_input_3d) refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 1);
//				// Apply SSNR weighting only to image since input 3D map assumed to be calculated from correctly whitened images
				else refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 0);
				refine_particle.PhaseFlipImage();
//				refine_particle.CosineMask(false, true, 0.0);
				refine_particle.CosineMask();
				refine_particle.PhaseShift();
				refine_particle.CenterInCorner();
//				refine_particle.WeightBySSNR(input_3d.statistics.part_SSNR, 1);

				score = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
				if (score > best_score)
				{
					best_score = score;
					best_defocus_i = defocus_i;
//					wxPrintf("Parameter line = %i, Defocus = %f, score = %g\n", current_image, defocus_i * defocus_step, score);
				}
				refine_particle.particle_image->CopyFrom(&binned_image);
				refine_particle.is_ssnr_filtered = false;
				refine_particle.is_masked = false;
				refine_particle.is_centered_in_box = true;
				refine_particle.shift_counter = 1;
			}
			output_parameters[8] = input_parameters[8] + best_defocus_i * defocus_step;
			output_parameters[9] = input_parameters[9] + best_defocus_i * defocus_step;
			refine_particle.SetDefocus(output_parameters[8], output_parameters[9], input_parameters[10], input_parameters[11]);
			refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, output_parameters[8], output_parameters[9], input_parameters[10], input_parameters[11]);
		}
		else
		{
			refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], input_parameters[11]);
		}
		refine_particle.filter_radius_low = low_resolution_limit;
		refine_particle.SetIndexForWeightedCorrelation();
		if (normalize_input_3d) refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 1);
		// Apply SSNR weighting only to image since input 3D map assumed to be calculated from correctly whitened images
		else refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 0);
		refine_particle.PhaseFlipImage();
//		refine_particle.CosineMask(false, true, 0.0);
		refine_particle.CosineMask();
		refine_particle.PhaseShift();
		refine_particle.CenterInCorner();
//		refine_particle.WeightBySSNR(input_3d.statistics.part_SSNR, 1);

//		input_parameters[15] = 10.0;
		input_parameters[15] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);

		if ((refine_particle.number_of_search_dimensions > 0) && (global_search || local_refinement))
		{
			if (global_search)
			{
//				my_time_in = wxDateTime::UNow();
				search_particle.ResetImageFlags();
				search_particle.pixel_size = search_reference_3d.pixel_size;
				if (mask_radius_search == 0.0)
				{
					search_particle.mask_radius = search_particle.particle_image->logical_x_dimension / 2 * search_particle.pixel_size - mask_falloff;
				}
				else
				{
					search_particle.mask_radius = mask_radius_search;
				}
				search_particle.mask_falloff = mask_falloff;
				search_particle.filter_radius_low = 0.0;
				search_particle.filter_radius_high = high_resolution_limit_search;
				search_particle.molecular_mass_kDa = molecular_mass_kDa;
				search_particle.signed_CC_limit = signed_CC_limit;
				search_particle.sigma_noise = input_parameters[14] / binning_factor_search;
//				search_particle.logp = -std::numeric_limits<float>::max();
				search_particle.SetParameters(input_parameters);
				search_particle.number_of_search_dimensions = refine_particle.number_of_search_dimensions;
				search_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], input_parameters[11]);
				temp_image.CopyFrom(&input_image);
				// Multiply by binning_factor so variance after binning is close to 1.
//				temp_image.MultiplyByConstant(binning_factor_search);
				// Assume square images
				if (search_box_size != temp_image.logical_x_dimension)
				{
					temp_image.ClipInto(&temp_image2);
					temp_image2.ForwardFFT();
					temp_image2.ClipInto(search_particle.particle_image);
				}
				else
				{
					temp_image.ForwardFFT();
					temp_image.ClipInto(search_particle.particle_image);
				}
				search_particle.PhaseShiftInverse();
				// Always apply particle SSNR weighting (i.e. whitening) reference normalization since reference
				// projections will not have SSNR (i.e. CTF-dependent) weighting applied
				search_particle.WeightBySSNR(search_statistics.part_SSNR, 1);
				search_particle.PhaseFlipImage();
//				search_particle.CosineMask(false, true, 0.0);
				search_particle.CosineMask();
				search_particle.PhaseShift();
//				search_particle.CenterInCorner();
//				search_particle.WeightBySSNR(search_reference_3d.statistics.part_SSNR);

				if (search_particle.parameter_map[1] && ! search_particle.parameter_map[2])
				{
					global_euler_search.InitGrid(my_symmetry, angular_step, 0.0, input_parameters[2], psi_max, psi_step, psi_start, search_reference_3d.pixel_size / high_resolution_limit_search, search_particle.parameter_map, best_parameters_to_keep);
					if (! search_particle.parameter_map[3]) global_euler_search.psi_start = 360.0 - input_parameters[3];
					global_euler_search.Run(search_particle, search_reference_3d.density_map, input_parameters + 1, projection_cache);
				}
				else
				if (! search_particle.parameter_map[1] && search_particle.parameter_map[2])
				{
					global_euler_search.InitGrid(my_symmetry, angular_step, input_parameters[1], 0.0, psi_max, psi_step, psi_start, search_reference_3d.pixel_size / high_resolution_limit_search, search_particle.parameter_map, best_parameters_to_keep);
					if (! search_particle.parameter_map[3]) global_euler_search.psi_start = 360.0 - input_parameters[3];
					global_euler_search.Run(search_particle, search_reference_3d.density_map, input_parameters + 1, projection_cache);
				}
				else
				if (search_particle.parameter_map[1] && search_particle.parameter_map[2])
				{
					if (! search_particle.parameter_map[3]) global_euler_search.psi_start = 360.0 - input_parameters[3];
					global_euler_search.Run(search_particle, search_reference_3d.density_map, input_parameters + 1, projection_cache);
				}
				else
				{
					best_parameters_to_keep = 0;
				}

				// Do local refinement of the top hits to determine the best match
				for (i = 0; i < search_particle.number_of_parameters; i++) {search_parameters[i] = input_parameters[i];}
				for (j = 1; j < 6; j++) {global_euler_search.list_of_best_parameters[0][j - 1] = input_parameters[j];}
				search_particle.SetParameterConstraints(powf(parameter_average[14],2));
				comparison_object.reference_volume = &search_reference_3d;
				comparison_object.projection_image = &search_projection_image;
				comparison_object.particle = &search_particle;
				search_particle.CenterInCorner();
				search_particle.SetIndexForWeightedCorrelation();
				search_particle.SetParameters(input_parameters);
				search_particle.MapParameters(cg_starting_point);
				search_particle.mask_radius = outer_mask_radius;
//				output_parameters[15] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
//				if (! local_refinement) input_parameters[15] = output_parameters[15];
				search_particle.UnmapParametersToExternal(output_parameters, cg_starting_point);
				if (ignore_input_angles && best_parameters_to_keep >= 1) istart = 1;
				else istart = 0;
				for (i = istart; i <= best_parameters_to_keep; i++)
				{
					for (j = 1; j < 6; j++) {search_parameters[j] = global_euler_search.list_of_best_parameters[i][j - 1];}
//					wxPrintf("parameters in  = %i %g, %g, %g, %g, %g %g\n", i, search_parameters[3], search_parameters[2],
//							search_parameters[1], search_parameters[4], search_parameters[5], global_euler_search.list_of_best_parameters[i][5]);
					if (! search_particle.parameter_map[4]) search_parameters[4] = input_parameters[4];
					if (! search_particle.parameter_map[5]) search_parameters[5] = input_parameters[5];
					search_particle.SetParameters(search_parameters);
					search_particle.MapParameters(cg_starting_point);
					search_parameters[15] = - 100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, search_particle.number_of_search_dimensions, cg_starting_point, cg_accuracy);
					if (i == istart)
					{
						output_parameters[15] = search_parameters[15];
						if (! local_refinement) input_parameters[15] = output_parameters[15];
					}
					if (skip_local_refinement) temp_float = search_parameters[15];
					else temp_float = - 100.0 * conjugate_gradient_minimizer.Run();
					// Uncomment the following line to skip local refinement.
//					temp_float = search_parameters[15];
//					wxPrintf("best, refine in, out, diff = %i %g %g %g %g\n", i, output_parameters[15], search_parameters[15], temp_float, temp_float - output_parameters[15]);
//					log_diff = output_parameters[15] - temp_float;
//					if (log_diff > log_range) log_diff = log_range;
//					if (log_diff < - log_range) log_diff = - log_range;
					if (temp_float > output_parameters[15])
					// If log_diff >= 0, exp(log_diff) will always be larger than the random number and the search parameters will be kept.
					// If log_diff < 0, there is an increasing chance that the random number is larger than exp(log_diff) and the new
					// (worse) parameters will not be kept.
//					if ((global_random_number_generator.GetUniformRandom() + 1.0) / 2.0 < 1.0 / (1.0 + exp(log_diff)))
					{
//						if (log_diff < 0.0) wxPrintf("log_diff = %g\n", log_diff);
//						wxPrintf("image_counter = %i, i = %i, score = %g\n", image_counter, i, temp_float);
						search_particle.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues());
						output_parameters[15] = temp_float;
//						wxPrintf("parameters out = %g, %g, %g, %g, %g\n", output_parameters[3], output_parameters[2],
//								output_parameters[1], output_parameters[4], output_parameters[5]);
					}
//					wxPrintf("refine in, out, keep = %i %g %g %g\n", i, search_parameters[15], temp_float, output_parameters[15]);
//					wxPrintf("parameters out = %g, %g, %g, %g, %g\n", output_parameters[3], output_parameters[2],
//							output_parameters[1], output_parameters[4], output_parameters[5]);
				}
				refine_particle.SetParameters(output_parameters, true);
//				my_time_out = wxDateTime::UNow(); wxPrintf("global search done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
			}

			if (local_refinement)
			{
//				my_time_in = wxDateTime::UNow();
				comparison_object.reference_volume = &input_3d;
				comparison_object.projection_image = &projection_image;
				comparison_object.particle = &refine_particle;
				refine_particle.MapParameters(cg_starting_point);

				temp_float = - 100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, refine_particle.number_of_search_dimensions, cg_starting_point, cg_accuracy);
//???				if (! global_search) input_parameters[15] = temp_float;
				output_parameters[15] = - 100.0 * conjugate_gradient_minimizer.Run();

				refine_particle.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues());

//				my_time_out = wxDateTime::UNow(); wxPrintf("local refinement done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
			}
//			log_diff = input_parameters[15] - output_parameters[15];
//			wxPrintf("in = %g out = %g log_diff = %g ratio = %g\n", input_parameters[15], output_parameters[15], log_diff, 1.0 / (1.0 + exp(log_diff)));
//			if (log_diff > log_range) log_diff = log_range;
//			if (log_diff < - log_range) log_diff = - log_range;
			// If log_diff >= 0, exp(log_diff) will never be smaller than the random number and the new parameters will be kept.
			// If log_diff < 0 (new parameters give worse likelihood), new parameters will only be kept if random number smaller than exp(log_diff).
//			if ((global_random_number_generator.GetUniformRandom() + 1.0) / 2.0 >= 1.0 / (1.0 + exp(log_diff))) for (i = 0; i < refine_particle.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}
//			else output_parameters[16] = output_parameters[15] - input_parameters[15];
			output_parameters[16] = output_parameters[15] - input_parameters[15];
//			wxPrintf("in, out, diff = %g %g %g\n", input_parameters[15], output_parameters[15], output_parameters[16]);
			if (output_parameters[16] < 0.0) for (i = 0; i < refine_particle.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}
		}
		else
		{
			input_parameters[15] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
			output_parameters[15] = input_parameters[15]; output_parameters[16] = 0.0;
		}

		refine_particle.SetParameters(output_parameters);

		refine_particle.SetAlignmentParameters(output_parameters[1], output_parameters[2], output_parameters[3], 0.0, 0.0);
//		unbinned_image.ClipInto(refine_particle.particle_image);
//		refine_particle.particle_image->MultiplyByConstant(binning_factor_refine);
//		refine_particle.particle_image->QuickAndDirtyWriteSlice("part3.mrc", 1);
//		refine_particle.PhaseFlipImage();
//		refine_particle.CalculateProjection(projection_image, input_3d);
//		projection_image.ClipInto(&unbinned_image);
//		unbinned_image.BackwardFFT();
//		unbinned_image.ClipInto(&final_image);
//		logp = refine_particle.ReturnLogLikelihood(input_image, final_image, pixel_size, classification_resolution_limit, alpha, sigma);

//		logp = refine_particle.ReturnLogLikelihood(input_3d, refine_statistics, classification_resolution_limit);
		output_parameters[13] = refine_particle.ReturnLogLikelihood(input_image, unbinned_image, input_ctf, input_3d, input_statistics, classification_resolution_limit);
//		output_parameters[14] = sigma * binning_factor_refine;

//		refine_particle.CalculateMaskedLogLikelihood(projection_image, input_3d, classification_resolution_limit);
//		output_parameters[13] = refine_particle.logp;
		if (refine_particle.snr > 0.0) output_parameters[14] = sqrtf(1.0 / refine_particle.snr);

//		output_parameters[14] = refine_particle.sigma_noise * binning_factor_refine;
//		wxPrintf("logp, sigma, score = %g %g %g\n", output_parameters[13], output_parameters[14], output_parameters[15]);
//		refine_particle.CalculateProjection(projection_image, input_3d);
//		projection_image.BackwardFFT();
//		wxPrintf("snr = %g mask = %g var_A = %g\n", refine_particle.snr, refine_particle.mask_volume, projection_image.ReturnVarianceOfRealValues());
//		output_parameters[14] = sqrtf(refine_particle.snr * refine_particle.particle_image->number_of_real_space_pixels
//				/ refine_particle.mask_volume / projection_image.ReturnVarianceOfRealValues()) * binning_factor_refine;

		if (calculate_matching_projections)
		{
			refine_particle.CalculateProjection(projection_image, input_3d);
			projection_image.ClipInto(&unbinned_image);
			unbinned_image.BackwardFFT();
			unbinned_image.ClipInto(&final_image);
			final_image.ForwardFFT();
			final_image.PhaseShift(output_parameters[4] / pixel_size, output_parameters[5] / pixel_size);
			final_image.BackwardFFT();
			final_image.WriteSlice(output_file, image_counter);
		}

		temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		temp_float = output_parameters[1]; output_parameters[1] = output_parameters[3]; output_parameters[3] = temp_float;
		for (i = 1; i < refine_particle.number_of_parameters; i++)
		{
			if (isnanf(output_parameters[i]) != 0)
			{
//				MyDebugAssertTrue(false, "NaN value for output parameter encountered");
				output_parameters[i] = input_parameters[i];
			}
		}
		input_parameters[7] = 1;//fabsf(input_parameters[7]);
		output_parameters[7] = input_parameters[7];
		if (output_parameters[15] < 0.0) output_parameters[15] = 0.0;
		my_output_par_file.WriteLine(output_parameters);

		if (is_running_locally == false) // send results back to the gui..
		{
			intermediate_result = new JobResult;
			intermediate_result->job_number = my_current_job.job_number;

			gui_result_parameters[0] = current_class;
			for (result_parameter_counter = 1; result_parameter_counter < refine_particle.number_of_parameters + 1; result_parameter_counter++)
			{
				gui_result_parameters[result_parameter_counter] = output_parameters[result_parameter_counter - 1];
			}

			intermediate_result->SetResult(refine_particle.number_of_parameters + 1, gui_result_parameters);
			AddJobToResultQueue(intermediate_result);
		}

		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_average[i] += output_parameters[i];}
		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameters[i] -= input_parameters[i];}
		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_change[i] += powf(output_parameters[i],2);}
		my_output_par_shifts_file.WriteLine(output_parameters);

		fflush(my_output_par_file.parameter_file);
		fflush(my_output_par_shifts_file.parameter_file);

		my_progress->Update(image_counter);
	}

	for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_average[i] /= float(last_particle - first_particle + 1);}
	for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_change[i] /= float(last_particle - first_particle + 1);}
	for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_change[i] = sqrtf(output_parameter_change[i]);}
	my_output_par_file.WriteLine(output_parameter_average, true);
	my_output_par_shifts_file.WriteLine(output_parameter_change, true);
	my_output_par_file.WriteCommentLine("C  Total particles included, overall score, average occupancy "
			+ wxString::Format("%11i %10.6f %10.6f", last_particle - first_particle + 1, output_parameter_average[15], output_parameter_average[12]));

	delete my_progress;
//	delete global_euler_search;
	if (global_search)
	{
		delete [] projection_cache;
//		search_projection_image.RotateFourier2DDeleteIndex(kernel_index, psi_max, psi_step);
	}
	if (calculate_matching_projections) delete output_file;

	wxPrintf("\nRefine3D: Normal termination\n\n");

	return true;
}
