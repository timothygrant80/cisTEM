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
	comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image, *comparison_object->particle->ctf_image,
			comparison_object->particle->alignment_parameters, comparison_object->particle->mask_radius, comparison_object->particle->mask_falloff,
			comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false, true);

//	for (int i = 0; i < comparison_object->projection_image->real_memory_allocated; i++) {comparison_object->projection_image->real_values[i] *= fabs(comparison_object->projection_image->real_values[i]);}
//	comparison_object->projection_image->ForwardFFT();
//	comparison_object->projection_image->CalculateCrossCorrelationImageWith(comparison_object->particle->particle_image);
//	comparison_object->projection_image->SwapRealSpaceQuadrants();
//	comparison_object->projection_image->BackwardFFT();
//	comparison_object->projection_image->QuickAndDirtyWriteSlice("proj.mrc", 1);
//	comparison_object->particle->particle_image->SwapRealSpaceQuadrants();
//	comparison_object->particle->particle_image->BackwardFFT();
//	comparison_object->particle->particle_image->QuickAndDirtyWriteSlice("part.mrc", 1);
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
	return 	- comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image, comparison_object->particle->bin_index,
			  comparison_object->particle->pixel_size / comparison_object->particle->signed_CC_limit)
			- comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_float);

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
	float		pixel_size = 1.0;
	float		voltage_kV = 300.0;
	float		spherical_aberration_mm = 2.7;
	float		amplitude_contrast = 0.07;
	float		molecular_mass_kDa = 1000.0;
	float		mask_radius = 100.0;
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
	bool		invert_contrast = false;

	UserInput *my_input = new UserInput("Refine3D", 1.01);

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
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius for refinement (A)", "Radius of a circular mask to be applied to the input images during refinement", "100.0", 0.0);
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
	invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");

	delete my_input;

	int current_class = 0;
	my_current_job.Reset(46);
	my_current_job.ManualSetArguments("ttttbttttiifffffffffffffifffffffffbbbbbbbbbbbi",	input_particle_images.ToUTF8().data(),
																						input_parameter_file.ToUTF8().data(),
																						input_reconstruction.ToUTF8().data(),
																						input_reconstruction_statistics.ToUTF8().data(), use_statistics,
																						ouput_matching_projections.ToUTF8().data(),
																						ouput_parameter_file.ToUTF8().data(),
																						ouput_shift_file.ToUTF8().data(),
																						my_symmetry.ToUTF8().data(),
																						first_particle, last_particle,
																						pixel_size, voltage_kV, spherical_aberration_mm, amplitude_contrast,
																						molecular_mass_kDa, mask_radius, low_resolution_limit, high_resolution_limit,
																						signed_CC_limit, classification_resolution_limit,
																						mask_radius_search, high_resolution_limit_search, angular_step, best_parameters_to_keep,
																						max_search_x, max_search_y,
																						mask_center_2d_x, mask_center_2d_y, mask_center_2d_z, mask_radius_2d,
																						defocus_search_range, defocus_step,
																						padding,
																						global_search, local_refinement,
																						refine_psi, refine_theta, refine_phi, refine_x, refine_y,
																						calculate_matching_projections,
																						apply_2D_masking, ctf_refinement, invert_contrast, current_class);
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
	float 	 pixel_size							= my_current_job.arguments[11].ReturnFloatArgument(); // local
	float    voltage_kV							= my_current_job.arguments[12].ReturnFloatArgument(); // local
	float 	 spherical_aberration_mm			= my_current_job.arguments[13].ReturnFloatArgument(); // local
	float    amplitude_contrast					= my_current_job.arguments[14].ReturnFloatArgument(); // local
	float	 molecular_mass_kDa					= my_current_job.arguments[15].ReturnFloatArgument(); // global
	float    mask_radius						= my_current_job.arguments[16].ReturnFloatArgument(); // global
	float    low_resolution_limit				= my_current_job.arguments[17].ReturnFloatArgument(); // global
	float    high_resolution_limit				= my_current_job.arguments[18].ReturnFloatArgument(); // global
	float	 signed_CC_limit					= my_current_job.arguments[19].ReturnFloatArgument(); // global
	float	 classification_resolution_limit	= my_current_job.arguments[20].ReturnFloatArgument(); // global
	float    mask_radius_search					= my_current_job.arguments[21].ReturnFloatArgument(); // global
	float	 high_resolution_limit_search		= my_current_job.arguments[22].ReturnFloatArgument(); // global
	float	 angular_step						= my_current_job.arguments[23].ReturnFloatArgument(); // global
	int		 best_parameters_to_keep			= my_current_job.arguments[24].ReturnIntegerArgument(); // global
	float	 max_search_x						= my_current_job.arguments[25].ReturnFloatArgument(); // global
	float	 max_search_y						= my_current_job.arguments[26].ReturnFloatArgument(); // global
	refine_particle.mask_center_2d_x			= my_current_job.arguments[27].ReturnFloatArgument(); // global
	refine_particle.mask_center_2d_y			= my_current_job.arguments[28].ReturnFloatArgument(); // global
	refine_particle.mask_center_2d_z			= my_current_job.arguments[29].ReturnFloatArgument(); // global
	refine_particle.mask_radius_2d				= my_current_job.arguments[30].ReturnFloatArgument(); // global
	float	 defocus_search_range				= my_current_job.arguments[31].ReturnFloatArgument(); // global
	float	 defocus_step						= my_current_job.arguments[32].ReturnFloatArgument(); // global
	float	 padding							= my_current_job.arguments[33].ReturnFloatArgument(); // global
//	float	 filter_constant					= my_current_job.arguments[34].ReturnFloatArgument();
	bool	 global_search						= my_current_job.arguments[34].ReturnBoolArgument(); // global
	bool	 local_refinement					= my_current_job.arguments[35].ReturnBoolArgument(); // global
// Psi, Theta, Phi, ShiftX, ShiftY
	refine_particle.parameter_map[3]			= my_current_job.arguments[36].ReturnBoolArgument(); //global
	refine_particle.parameter_map[2]			= my_current_job.arguments[37].ReturnBoolArgument(); //global
	refine_particle.parameter_map[1]			= my_current_job.arguments[38].ReturnBoolArgument(); // global
	refine_particle.parameter_map[4]			= my_current_job.arguments[39].ReturnBoolArgument(); // global
	refine_particle.parameter_map[5]			= my_current_job.arguments[40].ReturnBoolArgument(); // global
	bool 	 calculate_matching_projections		= my_current_job.arguments[41].ReturnBoolArgument(); // global - but ignore
	refine_particle.apply_2D_masking			= my_current_job.arguments[42].ReturnBoolArgument(); // global
	bool	 ctf_refinement						= my_current_job.arguments[43].ReturnBoolArgument(); // global
	bool	 invert_contrast					= my_current_job.arguments[44].ReturnBoolArgument(); // global - but ignore.
	int		 current_class						= my_current_job.arguments[45].ReturnIntegerArgument(); // global - but ignore.

	refine_particle.constraints_used[4] = true;		// Constraint for X shifts
	refine_particle.constraints_used[5] = true;		// Constraint for Y shifts

	Image input_image;
	Image ctf_input_image;
	Image projection_image;
	Image search_projection_image;
	Image unbinned_image;
	Image binned_image;
	Image final_image;
	Image temp_image;
	Image temp_image2;
	Image *projection_cache = NULL;
	CTF   my_ctf;
	CTF   my_input_ctf;
	Image snr_image;
	ReconstructedVolume			input_3d;
	ReconstructedVolume			search_reference_3d;
	ImageProjectionComparison	comparison_object;
	ConjugateGradient			conjugate_gradient_minimizer;
	EulerSearch					global_euler_search;
	Kernel2D					**kernel_index = NULL;

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
	float mask_falloff = 20.0;
	float alpha;
	float sigma;
	float logp;
	float temp_float;
	float psi;
	float psi_max;
	float psi_step;
	float psi_start;
	float score;
	float best_score;
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

	wxPrintf("opening input file %s.\n", input_parameter_file);
	FrealignParameterFile my_input_par_file(input_parameter_file, OPEN_TO_READ);
	my_input_par_file.ReadFile();

	MRCFile input_stack(input_particle_images.ToStdString(), false);
	MRCFile input_file(input_reconstruction.ToStdString(), false);
	MRCFile *output_file;
	if (calculate_matching_projections) output_file = new MRCFile(ouput_matching_projections.ToStdString(), true);
	FrealignParameterFile my_output_par_file(ouput_parameter_file, OPEN_TO_WRITE);
	FrealignParameterFile my_output_par_shifts_file(ouput_shift_file, OPEN_TO_WRITE, 15);

	if (input_stack.ReturnXSize() != input_stack.ReturnYSize())
	{
		MyPrintWithDetails("Error: Particles are not square\n");
		SendError("Error: Particles are not square");
		abort();
	}
	if ((input_file.ReturnXSize() != input_file.ReturnYSize()) || (input_file.ReturnXSize() != input_file.ReturnZSize()))
	{
		MyPrintWithDetails("Error: Input reconstruction is not cubic\n");
		SendError("Error: Input reconstruction is not cubic\n");
		abort();
	}
	if (input_file.ReturnXSize() != input_stack.ReturnXSize())
	{
		MyPrintWithDetails("Error: Dimension of particles and input reconstruction differ\n");
		SendError("Error: Dimension of particles and input reconstruction differ");
		abort();
	}
	if (last_particle < first_particle && last_particle != 0)
	{
		MyPrintWithDetails("Error: Number of last particle to refine smaller than number of first particle to refine\n");
		SendError("Error: Number of last particle to refine smaller than number of first particle to refine");
		abort();
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
	my_output_par_file.WriteCommentLine("C Pixel size of images (A):                " + wxString::Format("%f", pixel_size));
	my_output_par_file.WriteCommentLine("C Beam energy (keV):                       " + wxString::Format("%f", voltage_kV));
	my_output_par_file.WriteCommentLine("C Spherical aberration (mm):               " + wxString::Format("%f", spherical_aberration_mm));
	my_output_par_file.WriteCommentLine("C Amplitude contrast:                      " + wxString::Format("%f", amplitude_contrast));
	my_output_par_file.WriteCommentLine("C Mask radius for refinement (A):          " + wxString::Format("%f", mask_radius));
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
	my_output_par_file.WriteCommentLine("C Invert particle contrast:                " + BoolToYesNo(invert_contrast));
	my_output_par_file.WriteCommentLine("C");
//	my_output_par_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP NormSigmaNoise          Score         Change");
	my_output_par_file.WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LogP      SIGMA   SCORE  CHANGE");
	fflush(my_output_par_file.parameter_file);
//	my_output_par_shifts_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP NormSigmaNoise          Score");
	my_output_par_shifts_file.WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LogP      SIGMA   SCORE");


	if (signed_CC_limit == 0.0) signed_CC_limit = pixel_size;

	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, my_symmetry);
	input_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	ResolutionStatistics input_statistics(pixel_size, input_3d.density_map.logical_y_dimension);
	ResolutionStatistics search_statistics;
	if (use_statistics)
	{
		input_statistics.ReadStatisticsFromFile(input_reconstruction_statistics);
	}
	else
	{
		wxPrintf("\nUsing default statistics\n");
		input_statistics.GenerateDefaultStatistics(molecular_mass_kDa);
	}
	input_3d.density_map.ReadSlices(&input_file,1,input_3d.density_map.logical_z_dimension);
	input_3d.density_map.AddConstant(- input_3d.density_map.ReturnAverageOfRealValuesOnEdges());

	if (global_search)
	{
		// Assume square particles
		search_reference_3d = input_3d;
		search_statistics = input_statistics;
		search_box_size = ReturnClosestFactorizedUpper(myroundint(2.0 * padding * (std::max(max_search_x,max_search_y) + mask_radius_search)), 3, true);
		if (search_box_size > search_reference_3d.density_map.logical_x_dimension) search_box_size = search_reference_3d.density_map.logical_x_dimension;
		if (search_box_size != search_reference_3d.density_map.logical_x_dimension) search_reference_3d.density_map.Resize(search_box_size, search_box_size, search_box_size);
		search_reference_3d.PrepareForProjections(high_resolution_limit_search, true);
//		search_statistics.Init(search_reference_3d.pixel_size, search_reference_3d.density_map.logical_y_dimension / 2 + 1);
		search_particle.Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension);
		search_projection_image.Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension, false);
		temp_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
		temp_image2.Allocate(search_box_size, search_box_size, true);
		binning_factor_search = search_reference_3d.pixel_size / pixel_size;
		//if (angular_step <= 0) angular_step = 360.0 * high_resolution_limit_search / PI / mask_radius;
		if (angular_step <= 0) angular_step = CalculateAngularStep(high_resolution_limit_search, mask_radius);
		psi_step = rad_2_deg(search_reference_3d.pixel_size / mask_radius);
		psi_step = 360.0 / int(360.0 / psi_step + 0.5);
		psi_start = psi_step / 2.0 * global_random_number_generator.GetUniformRandom();
		psi_max = 0.0;
		if (refine_particle.parameter_map[3]) psi_max = 360.0;
		wxPrintf("\nBox size for search = %i, binning factor = %f, new pixel size = %f, resolution limit = %f\nAngular step size = %f, in-plane = %f\n", search_box_size, binning_factor_search, search_reference_3d.pixel_size, search_reference_3d.pixel_size * 2.0, angular_step, psi_step);
	}

	if (padding != 1.0)
	{
		input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension * padding, input_3d.density_map.logical_y_dimension * padding, input_3d.density_map.logical_z_dimension * padding, input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
		input_statistics.part_SSNR.ResampleCurve(&input_statistics.part_SSNR, input_statistics.part_SSNR.number_of_points * padding);
	}

	input_3d.PrepareForProjections(high_resolution_limit);
	binning_factor_refine = input_3d.pixel_size / pixel_size;
	wxPrintf("\nBinning factor for refinement = %f, new pixel size = %f\n", binning_factor_refine, input_3d.pixel_size);

	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	refine_particle.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension);
	ctf_input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	projection_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	unbinned_image.Allocate(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, true);
	if (ctf_refinement) binned_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	final_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);

// Read whole parameter file to work out average values and variances
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		for (i = 0; i < refine_particle.number_of_parameters; i++)
		{
			parameter_average[i] += input_parameters[i];
			parameter_variance[i] += powf(input_parameters[i],2);
		}
		if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle) images_to_process++;
	}

	for (i = 0; i < refine_particle.number_of_parameters; i++)
	{
		parameter_average[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] -= powf(parameter_average[i],2);

		if (parameter_variance[i] < 0.001) refine_particle.constraints_used[i] = false;
	}
	refine_particle.SetParameterStatistics(parameter_average, parameter_variance);

	if (global_search)
	{
		for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.parameter_map[i] = refine_particle.parameter_map[i];}
// Set parameter_map for x,y translations to true since they will always be searched and refined in a global search
		search_particle.parameter_map[4] = true;
		search_particle.parameter_map[5] = true;
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
		}
		search_projection_image.RotateFourier2DGenerateIndex(kernel_index, psi_max, psi_step, psi_start);

		global_euler_search.max_search_x = max_search_x;
		global_euler_search.max_search_y = max_search_y;
	}

	wxPrintf("\nAverage sigma noise = %f, average score = %f\nAverage ShiftX = %f, average ShiftY = %f\nSigma ShiftX = %f, sigma ShiftY = %f\nNumber of particles to refine = %i\n\n",
			parameter_average[13], parameter_average[14], parameter_average[4], parameter_average[5], sqrtf(parameter_variance[4]), sqrtf(parameter_variance[5]), images_to_process);
	my_input_par_file.Rewind();

	ProgressBar *my_progress = new ProgressBar(images_to_process);
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
		image_counter++;
		for (i = 0; i < refine_particle.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}

// Set up Particle object
		refine_particle.ResetImageFlags();
		refine_particle.mask_radius = mask_radius;
		refine_particle.mask_falloff = mask_falloff;
		refine_particle.filter_radius_low = low_resolution_limit;
		refine_particle.filter_radius_high = high_resolution_limit;
		refine_particle.molecular_mass_kDa = molecular_mass_kDa;
		refine_particle.signed_CC_limit = signed_CC_limit;
		// The following line would allow using particles with different pixel sizes
		refine_particle.pixel_size = input_3d.pixel_size;
		refine_particle.sigma_noise = input_parameters[13] * sqrtf(pixel_size/refine_particle.pixel_size);
		refine_particle.SetParameters(input_parameters);
		refine_particle.MapParameterAccuracy(cg_accuracy);
		refine_particle.SetIndexForWeightedCorrelation();
		refine_particle.SetParameterConstraints(powf(parameter_average[13],2));

		my_input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, pixel_size, 0.0);
		ctf_input_image.CalculateCTFImage(my_input_ctf);

		input_image.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
		input_image.ReplaceOutliersWithMean(5.0);
		if (invert_contrast) input_image.InvertRealValues();
//		input_image.ZeroFloatOutside(mask_radius / pixel_size);
		input_image.ClipInto(&unbinned_image);
		unbinned_image.ForwardFFT();
		unbinned_image.ClipInto(refine_particle.particle_image);

		comparison_object.reference_volume = &input_3d;
		comparison_object.projection_image = &projection_image;
		comparison_object.particle = &refine_particle;
		refine_particle.MapParameters(cg_starting_point);
		refine_particle.PhaseShiftInverse();

		if (ctf_refinement)
		{
			wxPrintf("\nRefining defocus for parameter line %i\n", current_image);
			binned_image.CopyFrom(refine_particle.particle_image);
			refine_particle.InitCTF(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10]);
			best_score = - std::numeric_limits<float>::max();
			for (defocus_i = - myround(float(defocus_search_range)/float(defocus_step)); defocus_i <= myround(float(defocus_search_range)/float(defocus_step)); defocus_i++)
			{
				refine_particle.SetDefocus(input_parameters[8] + defocus_i * defocus_step, input_parameters[9] + defocus_i * defocus_step, input_parameters[10]);
				refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8] + defocus_i * defocus_step, input_parameters[9] + defocus_i * defocus_step, input_parameters[10]);
				refine_particle.WeightBySSNR(input_statistics.part_SSNR, 1);
				refine_particle.PhaseFlipImage();
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
			refine_particle.SetDefocus(output_parameters[8], output_parameters[9], input_parameters[10]);
			refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, output_parameters[8], output_parameters[9], input_parameters[10]);
		}
		else
		{
			refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10]);
		}
		refine_particle.WeightBySSNR(input_statistics.part_SSNR, 1);
		refine_particle.PhaseFlipImage();
		refine_particle.CosineMask();
		refine_particle.PhaseShift();
		refine_particle.CenterInCorner();
//		refine_particle.WeightBySSNR(input_3d.statistics.part_SSNR, 1);

//		input_parameters[14] = 10.0;
		input_parameters[14] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);

		if ((refine_particle.number_of_search_dimensions > 0) && (global_search || local_refinement))
		{
			if (global_search)
			{
				my_time_in = wxDateTime::UNow();
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
				search_particle.sigma_noise = input_parameters[13] * sqrtf(pixel_size / search_particle.pixel_size);
				search_particle.SetParameters(input_parameters);
				search_particle.number_of_search_dimensions = refine_particle.number_of_search_dimensions;
				search_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10]);
				temp_image.CopyFrom(&input_image);
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
				search_particle.WeightBySSNR(search_statistics.part_SSNR);
				search_particle.PhaseFlipImage();
				search_particle.CosineMask();
				search_particle.PhaseShift();
				search_particle.CenterInCorner();
//				search_particle.WeightBySSNR(search_reference_3d.statistics.part_SSNR);

				if (search_particle.parameter_map[1] && ! search_particle.parameter_map[2])
				{
					global_euler_search.InitGrid(my_symmetry, angular_step, 0.0, input_parameters[2], psi_max, psi_step, psi_start, search_reference_3d.pixel_size / high_resolution_limit_search, search_particle.parameter_map, best_parameters_to_keep);
					global_euler_search.Run(search_particle, search_reference_3d.density_map, input_parameters + 1, projection_cache, kernel_index);
				}
				else
				if (! search_particle.parameter_map[1] && search_particle.parameter_map[2])
				{
					global_euler_search.InitGrid(my_symmetry, angular_step, input_parameters[1], 0.0, psi_max, psi_step, psi_start, search_reference_3d.pixel_size / high_resolution_limit_search, search_particle.parameter_map, best_parameters_to_keep);
					global_euler_search.Run(search_particle, search_reference_3d.density_map, input_parameters + 1, projection_cache, kernel_index);
				}
				else
				if (search_particle.parameter_map[1] && search_particle.parameter_map[2])
				{
					global_euler_search.Run(search_particle, search_reference_3d.density_map, input_parameters + 1, projection_cache, kernel_index);
				}
				else
				{
					best_parameters_to_keep = 0;
				}

				// Do local refinement of the top hits to determine the best match
				for (i = 0; i < search_particle.number_of_parameters; i++) {search_parameters[i] = input_parameters[i];}
				for (j = 1; j <= 6; j++) {global_euler_search.list_of_best_parameters[best_parameters_to_keep][j - 1] = input_parameters[j];}
				search_particle.SetParameterConstraints(powf(parameter_average[13],2));
				comparison_object.reference_volume = &search_reference_3d;
				comparison_object.projection_image = &search_projection_image;
				comparison_object.particle = &search_particle;
				search_particle.SetIndexForWeightedCorrelation();
				search_particle.SetParameters(input_parameters);
				search_particle.MapParameters(cg_starting_point);
				search_particle.mask_radius = mask_radius;
				output_parameters[14] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
				search_particle.UnmapParametersToExternal(output_parameters, cg_starting_point);
				for (i = 0; i <= best_parameters_to_keep; i++)
				{
					for (j = 1; j <= 6; j++) {search_parameters[j] = global_euler_search.list_of_best_parameters[i][j - 1];}
//					wxPrintf("parameters in  = %g, %g, %g, %g, %g\n", search_parameters[3], search_parameters[2],
//							search_parameters[1], search_parameters[4], search_parameters[5]);
					search_particle.SetParameters(search_parameters);
					search_particle.MapParameters(cg_starting_point);
					search_parameters[14] = - 100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, search_particle.number_of_search_dimensions, cg_starting_point, cg_accuracy);
					temp_float = - 100.0 * conjugate_gradient_minimizer.Run();
					if (temp_float > output_parameters[14])
					{
//						wxPrintf("image_counter = %i, i = %i, score = %g\n", image_counter, i, temp_float);
						search_particle.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues());
						output_parameters[14] = temp_float;
//						wxPrintf("parameters out = %g, %g, %g, %g, %g\n", output_parameters[3], output_parameters[2],
//								output_parameters[1], output_parameters[4], output_parameters[5]);
					}
//					wxPrintf("parameters out = %g, %g, %g, %g, %g\n", output_parameters[3], output_parameters[2],
//							output_parameters[1], output_parameters[4], output_parameters[5]);
				}
				refine_particle.SetParameters(output_parameters);
//				my_time_out = wxDateTime::UNow(); wxPrintf("global search done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
			}

			if (local_refinement)
			{
				my_time_in = wxDateTime::UNow();
				comparison_object.reference_volume = &input_3d;
				comparison_object.projection_image = &projection_image;
				comparison_object.particle = &refine_particle;
				refine_particle.MapParameters(cg_starting_point);

				temp_float = - 100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, refine_particle.number_of_search_dimensions, cg_starting_point, cg_accuracy);
				if (! global_search) input_parameters[14] = temp_float;
				output_parameters[14] = - 100.0 * conjugate_gradient_minimizer.Run();

				refine_particle.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues());

//				my_time_out = wxDateTime::UNow(); wxPrintf("local refinement done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
			}
			output_parameters[15] = output_parameters[14] - input_parameters[14];
			if (output_parameters[15] < 0.0) for (i = 0; i < refine_particle.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}
		}
		else
		{
			input_parameters[14] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
			output_parameters[14] = input_parameters[14]; output_parameters[15] = 0.0;
		}

		refine_particle.SetParameters(output_parameters);

		refine_particle.SetAlignmentParameters(output_parameters[1], output_parameters[2], output_parameters[3], 0.0, 0.0);
		unbinned_image.ClipInto(refine_particle.particle_image);
//		refine_particle.PhaseFlipImage();
//		refine_particle.CalculateProjection(projection_image, input_3d);
//		projection_image.ClipInto(&unbinned_image);
//		unbinned_image.BackwardFFT();
//		unbinned_image.ClipInto(&final_image);
//		logp = refine_particle.ReturnLogLikelihood(input_image, final_image, pixel_size, classification_resolution_limit, alpha, sigma);
		logp = refine_particle.ReturnLogLikelihood(input_3d, input_statistics, projection_image, classification_resolution_limit, alpha, sigma);
		output_parameters[12] = logp;
		output_parameters[13] = sigma;

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
				MyDebugPrintWithDetails("NaN value for output parameter encountered\n");
		//		SendError(wxString::Format("Error: NaN value for output parameter encountered (Particle %i) - Exiting", refine_particle.location_in_stack));
		//		MyDebugAssertTrue(false, "NaN value for output parameter encountered");
				output_parameters[i] = input_parameters[i];
			}
		}
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
	//		wxPrintf("REFINE3D :: Calling - Adding Job to result Queue\n");
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
			+ wxString::Format("%11i %10.6f %10.6f", last_particle - first_particle + 1, output_parameter_average[14], output_parameter_average[11]));

	delete my_progress;
//	delete global_euler_search;
	if (global_search)
	{
		delete [] projection_cache;
		search_projection_image.RotateFourier2DDeleteIndex(kernel_index, psi_max, psi_step);
	}
	if (calculate_matching_projections) delete output_file;

	wxPrintf("\nRefine3D: Normal termination\n\n");

	return true;



}
