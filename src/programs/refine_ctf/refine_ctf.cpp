#include "../../core/core_headers.h"

class AggregatedTemplateResult
{
public:


	float image_number;
	int number_of_received_results;
	float total_number_of_ccs;

	float *collated_data_array;
	float *collated_mip_data;
	float *collated_psi_data;
	float *collated_theta_data;
	float *collated_phi_data;
	float *collated_defocus_data;
	float *collated_pixel_sums;
	float *collated_pixel_square_sums;

	AggregatedTemplateResult();
	~AggregatedTemplateResult();
	void AddResult(float *result_array, long array_size, int result_number, int number_of_expected_results);
};

WX_DECLARE_OBJARRAY(AggregatedTemplateResult, ArrayOfAggregatedTemplateResults);
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfAggregatedTemplateResults);

class
RefineCTFApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	//void SendProgramDefinedResultToMaster(float *result_array, int array_size, int result_number, int number_of_expected_results); // overidden as i want to do complicated stuff
	void MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results);
	void ProgramSpecificInit();
	// for master collation

	ArrayOfAggregatedTemplateResults aggregated_results;

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

IMPLEMENT_APP(RefineCTFApp)

void RefineCTFApp::ProgramSpecificInit()
{
}

// override the DoInteractiveUserInput

void RefineCTFApp::DoInteractiveUserInput()
{
	UserInput *my_input = new UserInput("RefineCTF", 1.00);

	int max_threads;

	wxString input_particle_images = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	wxString input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	wxString input_reconstruction = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	wxString input_reconstruction_statistics = my_input->GetFilenameFromUser("Input data statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
	bool use_statistics = my_input->GetYesNoFromUser("Use statistics", "Answer No if no statistics are available?", "Yes");
	wxString ouput_parameter_file = my_input->GetFilenameFromUser("Output parameter file", "The output parameter file, containing your refined particle alignment parameters", "my_refined_parameters.par", false);
	wxString ouput_shift_file = my_input->GetFilenameFromUser("Output parameter changes", "The changes in the alignment parameters compared to the input parameters", "my_parameter_changes.par", false);
	wxString ouput_phase_difference_image = my_input->GetFilenameFromUser("Output phase difference image", "Diagnostic image indicating the average phase difference (x20) between aligned images and matching projections", "my_phase_difference.mrc", false);
	wxString ouput_beamtilt_image = my_input->GetFilenameFromUser("Output beam tilt image", "Diagnostic image indicating phase difference (x20) generated by beam tilt", "my_beamtilt_image.mrc", false);
	wxString ouput_difference_image = my_input->GetFilenameFromUser("Output phase diff - beam tilt ", "Difference between phase difference and matching beam tilt (x20)", "my_difference_image.mrc", false);
	int first_particle = my_input->GetIntFromUser("First particle to refine (0 = first in stack)", "The first particle in the stack that should be refined", "1", 0);
	int last_particle = my_input->GetIntFromUser("Last particle to refine (0 = last in stack)", "The last particle in the stack that should be refined", "0", 0);
	float pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	float voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	float spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	float amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	float molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	float inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the input reconstruction in Angstroms", "0.0", 0.0);
	float outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the input reconstruction and images during refinement, in Angstroms", "100.0", inner_mask_radius);
	float low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	float high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	float defocus_search_range = my_input->GetFloatFromUser("Defocus search range (A)", "Search range (-value ... + value) around current defocus", "500.0", 0.0);
	float defocus_step = my_input->GetFloatFromUser("Defocus step (A)", "Step size used in the defocus search", "50.0", 0.0);
	float padding = my_input->GetFloatFromUser("Tuning parameters: padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
	bool ctf_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
	bool beamtilt_refinement = my_input->GetYesNoFromUser("Refine beamtilt", "Should the beam tilt be refined?", "No");
	bool normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
	bool invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	bool exclude_blank_edges = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "Yes");
	bool normalize_input_3d = my_input->GetYesNoFromUser("Normalize input reconstruction", "The input reconstruction should always be normalized unless it was generated by reconstruct3d with normalized particles", "Yes");
	bool threshold_input_3d = my_input->GetYesNoFromUser("Threshold input reconstruction", "Should the input reconstruction thresholded to suppress some of the background noise", "No");

#ifdef _OPENMP
	max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "when threading, what is the max threads to run", "1", 1);
#else
	max_threads = 1;
#endif

	int image_number_for_gui = 0;
	int number_of_jobs_per_image_in_gui = 0;
	wxString directory_for_results = "/dev/null"; // shouldn't be used in interactive

	delete my_input;

	my_current_job.Reset(35);
	my_current_job.ManualSetArguments("ttttbtttttiiffffffffffffbbbbbbbiiit",
		input_particle_images.ToUTF8().data(),
		input_parameter_file.ToUTF8().data(),
		input_reconstruction.ToUTF8().data(),
		input_reconstruction_statistics.ToUTF8().data(),
		use_statistics,
		ouput_parameter_file.ToUTF8().data(),
		ouput_shift_file.ToUTF8().data(),
		ouput_phase_difference_image.ToUTF8().data(),
		ouput_beamtilt_image.ToUTF8().data(),
		ouput_difference_image.ToUTF8().data(),
		first_particle,
		last_particle,
		pixel_size,
		voltage_kV,
		spherical_aberration_mm,
		amplitude_contrast,
		molecular_mass_kDa,
		inner_mask_radius,
		outer_mask_radius,
		low_resolution_limit,
		high_resolution_limit,
		defocus_search_range,
		defocus_step,
		padding,
		ctf_refinement,
		beamtilt_refinement,
		normalize_particles,
		invert_contrast,
		exclude_blank_edges,
		normalize_input_3d,
		threshold_input_3d,
		image_number_for_gui,
		number_of_jobs_per_image_in_gui,
		max_threads,
		directory_for_results.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool RefineCTFApp::DoCalculation()
{
	wxDateTime start_time = wxDateTime::Now();

	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_parameter_file 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_reconstruction				= my_current_job.arguments[2].ReturnStringArgument();
	wxString input_reconstruction_statistics 	= my_current_job.arguments[3].ReturnStringArgument();
	bool	 use_statistics						= my_current_job.arguments[4].ReturnBoolArgument();
	wxString ouput_parameter_file				= my_current_job.arguments[5].ReturnStringArgument();
	wxString ouput_shift_file					= my_current_job.arguments[6].ReturnStringArgument();
	wxString ouput_phase_difference_image		= my_current_job.arguments[7].ReturnStringArgument();
	wxString ouput_beamtilt_image				= my_current_job.arguments[8].ReturnStringArgument();
	wxString ouput_difference_image				= my_current_job.arguments[9].ReturnStringArgument();
	int		 first_particle						= my_current_job.arguments[10].ReturnIntegerArgument();
	int		 last_particle						= my_current_job.arguments[11].ReturnIntegerArgument();
	float 	 pixel_size							= my_current_job.arguments[12].ReturnFloatArgument();
	float    voltage_kV							= my_current_job.arguments[13].ReturnFloatArgument();
	float 	 spherical_aberration_mm			= my_current_job.arguments[14].ReturnFloatArgument();
	float    amplitude_contrast					= my_current_job.arguments[15].ReturnFloatArgument();
	float	 molecular_mass_kDa					= my_current_job.arguments[16].ReturnFloatArgument();
	float    inner_mask_radius					= my_current_job.arguments[17].ReturnFloatArgument();
	float    outer_mask_radius					= my_current_job.arguments[18].ReturnFloatArgument();
	float    low_resolution_limit				= my_current_job.arguments[19].ReturnFloatArgument();
	float    high_resolution_limit				= my_current_job.arguments[20].ReturnFloatArgument();
	float	 defocus_search_range				= my_current_job.arguments[21].ReturnFloatArgument();
	float	 defocus_step						= my_current_job.arguments[22].ReturnFloatArgument();
	float	 padding							= my_current_job.arguments[23].ReturnFloatArgument();
	bool	 ctf_refinement						= my_current_job.arguments[24].ReturnBoolArgument();
	bool	 beamtilt_refinement				= my_current_job.arguments[25].ReturnBoolArgument();
	bool	 normalize_particles				= my_current_job.arguments[26].ReturnBoolArgument();
	bool	 invert_contrast					= my_current_job.arguments[27].ReturnBoolArgument();
	bool	 exclude_blank_edges				= my_current_job.arguments[28].ReturnBoolArgument();
	bool	 normalize_input_3d					= my_current_job.arguments[29].ReturnBoolArgument();
	bool	 threshold_input_3d					= my_current_job.arguments[30].ReturnBoolArgument();
	int		 image_number_for_gui				= my_current_job.arguments[31].ReturnIntegerArgument();
	int		 number_of_jobs_per_image_in_gui	= my_current_job.arguments[32].ReturnIntegerArgument();
	int		 max_threads						= my_current_job.arguments[33].ReturnIntegerArgument();
	wxString directory_for_results				= my_current_job.arguments[34].ReturnStringArgument();

	Particle refine_particle;

	int i, j;
	int current_image;
	int current_image_local;
	int max_samples = 2000;
	int images_to_process = 0;
	int number_of_blank_edges;
	int number_of_blank_edges_local;
	int image_counter;
	int defocus_i;
	int best_defocus_i;
	int random_reset_counter;
	int random_reset_count = 10;
	float defocus_lower_limit = 15000.0f * sqrtf(voltage_kV / 300.0f);
	float defocus_upper_limit = 25000.0f * sqrtf(voltage_kV / 300.0f);
	float defocus_range_mean2 = defocus_upper_limit + defocus_lower_limit;
	float defocus_range_std = 0.5f * (defocus_upper_limit - defocus_lower_limit);
	float defocus_mean_score = 0.0f;
	float phase_multiplier = 20.0f;
	float beamtilt_x, beamtilt_y;
	float particle_shift_x, particle_shift_y;
	float mask_falloff = 20.0f;	// in Angstrom
	float average_density_max;
	float binning_factor_refine;
	float percentage;
	float temp_float;
	float input_parameters[refine_particle.number_of_parameters];
	float parameter_average[refine_particle.number_of_parameters];
	float parameter_variance[refine_particle.number_of_parameters];
	float output_parameters[refine_particle.number_of_parameters];
	float cg_starting_point[refine_particle.number_of_parameters];
	float cg_accuracy[refine_particle.number_of_parameters];
	float output_parameter_average[refine_particle.number_of_parameters];
	float output_parameter_change[refine_particle.number_of_parameters];
	float output_parameter_average_local[refine_particle.number_of_parameters];
	float output_parameter_change_local[refine_particle.number_of_parameters];
	float mask_radius_for_noise;
	float variance;
	float average;
	float defocus_score;
	float score;
	float best_score;
	bool file_read;
	bool defocus_bias = false;
	wxString symmetry = "C1";

//	refine_particle.constraints_used[4] = true;		// Constraint for X shifts
//	refine_particle.constraints_used[5] = true;		// Constraint for Y shifts

	Image						input_image;
	Image						input_image_local;
	Image						sum_power;
	Image						sum_power_local;
	Image						temp_image;
	Image						temp_image_local;
	Image						unbinned_image;
	Image						binned_image;
	Image						projection_image_local;
	Image						phase_difference_image;
	Image						phase_difference_image_local;
	Image						phase_difference_sum;
	Image						phase_difference_sum_local;
	Image						beamtilt_image;
	CTF   						input_ctf;
	RandomNumberGenerator 		random_particle(true);
	ReconstructedVolume			input_3d;
	ImageProjectionComparison	comparison_object;
	Curve						noise_power_spectrum;
	Curve						number_of_terms;
	ProgressBar 				*my_progress;

	AnglesAndShifts rotation_matrix;

	ZeroFloatArray(input_parameters, refine_particle.number_of_parameters);
	ZeroFloatArray(parameter_average, refine_particle.number_of_parameters);
	ZeroFloatArray(parameter_variance, refine_particle.number_of_parameters);
	ZeroFloatArray(output_parameters, refine_particle.number_of_parameters);
	ZeroFloatArray(cg_starting_point, refine_particle.number_of_parameters);
	ZeroFloatArray(cg_accuracy, refine_particle.number_of_parameters);
	ZeroFloatArray(output_parameter_average, refine_particle.number_of_parameters);
	ZeroFloatArray(output_parameter_change, refine_particle.number_of_parameters);

	if ((is_running_locally && !DoesFileExist(input_parameter_file)) || (!is_running_locally && !DoesFileExistWithWait(input_parameter_file, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input parameter file %s not found\n", input_parameter_file));
	}
	if ((is_running_locally && !DoesFileExist(input_particle_images)) || (!is_running_locally && !DoesFileExistWithWait(input_particle_images, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input particle stack %s not found\n", input_particle_images));
	}
	if ((is_running_locally && !DoesFileExist(input_reconstruction)) || (!is_running_locally && !DoesFileExistWithWait(input_reconstruction, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input reconstruction %s not found\n", input_reconstruction));
	}

	FrealignParameterFile input_par_file(input_parameter_file, OPEN_TO_READ);
	MRCFile input_stack(input_particle_images.ToStdString(), false);

	input_par_file.ReadFile(false, input_stack.ReturnZSize());
	random_particle.SetSeed(int(10000.0 * fabsf(input_par_file.ReturnAverage(14, true)))%10000);
	if (defocus_bias)
	{
		float *buffer_array = new float[input_par_file.number_of_lines];
		for (current_image = 0; current_image < input_par_file.number_of_lines; current_image++)
		{
			buffer_array[current_image] = expf(- powf(0.25 * (fabsf(input_par_file.ReadParameter(current_image, 8)) + fabsf(input_par_file.ReadParameter(current_image, 9)) - defocus_range_mean2) / defocus_range_std, 2.0));
		}
		std::sort(buffer_array, buffer_array + input_par_file.number_of_lines -1);
		defocus_mean_score = buffer_array[input_par_file.number_of_lines / 2];
//		wxPrintf("median = %g\n", defocus_mean_score);
		//		defocus_mean_score /= current_image;
		delete [] buffer_array;
	}

	MRCFile input_file(input_reconstruction.ToStdString(), false, true);
	MRCFile ouput_phase_difference_file(ouput_phase_difference_image.ToStdString(), true);
	MRCFile ouput_beamtilt_file(ouput_beamtilt_image.ToStdString(), true);
	MRCFile ouput_difference_file(ouput_difference_image.ToStdString(), true);
//	MRCFile *output_file;

	FrealignParameterFile my_output_par_file(ouput_parameter_file, OPEN_TO_WRITE);
	FrealignParameterFile my_output_par_shifts_file(ouput_shift_file, OPEN_TO_WRITE, 16);

	if (input_stack.ReturnXSize() != input_stack.ReturnYSize())
	{
		input_stack.PrintInfo();
		SendErrorAndCrash("Error: Particles are not square\n");
	}
	if ((input_file.ReturnXSize() != input_file.ReturnYSize()) || (input_file.ReturnXSize() != input_file.ReturnZSize()))
	{
		input_file.PrintInfo();
		SendErrorAndCrash("Error: Input reconstruction is not cubic\n");
	}
	if (input_file.ReturnXSize() != input_stack.ReturnXSize())
	{
		input_file.PrintInfo();
		input_stack.PrintInfo();
		SendErrorAndCrash("Error: Dimension of particles and input reconstruction differ\n");
	}
	if (last_particle < first_particle && last_particle != 0)
	{
		SendErrorAndCrash("Error: Number of last particle to refine smaller than number of first particle to refine\n");
	}

	if (last_particle == 0) last_particle = input_stack.ReturnZSize();
	if (first_particle == 0) first_particle = 1;
	if (last_particle > input_stack.ReturnZSize()) last_particle = input_stack.ReturnZSize();

	if (is_running_locally == false) max_threads = 1;

	my_output_par_file.WriteCommentLine("C Refine3D run date and time:              " + start_time.FormatISOCombined(' '));
	my_output_par_file.WriteCommentLine("C Input particle images:                   " + input_particle_images);
	my_output_par_file.WriteCommentLine("C Input Frealign parameter filename:       " + input_parameter_file);
	my_output_par_file.WriteCommentLine("C Input reconstruction:                    " + input_reconstruction);
	my_output_par_file.WriteCommentLine("C Input data statistics:                   " + input_reconstruction_statistics);
	my_output_par_file.WriteCommentLine("C Use statistics:                          " + BoolToYesNo(use_statistics));
	my_output_par_file.WriteCommentLine("C Output parameter file:                   " + ouput_parameter_file);
	my_output_par_file.WriteCommentLine("C Output parameter changes:                " + ouput_shift_file);
	my_output_par_file.WriteCommentLine("C First particle to refine:                " + wxString::Format("%i", first_particle));
	my_output_par_file.WriteCommentLine("C Last particle to refine:                 " + wxString::Format("%i", last_particle));
	my_output_par_file.WriteCommentLine("C Pixel size of images (A):                " + wxString::Format("%f", pixel_size));
	my_output_par_file.WriteCommentLine("C Beam energy (keV):                       " + wxString::Format("%f", voltage_kV));
	my_output_par_file.WriteCommentLine("C Spherical aberration (mm):               " + wxString::Format("%f", spherical_aberration_mm));
	my_output_par_file.WriteCommentLine("C Amplitude contrast:                      " + wxString::Format("%f", amplitude_contrast));
	my_output_par_file.WriteCommentLine("C Molecular mass of particle (kDa):        " + wxString::Format("%f", molecular_mass_kDa));
	my_output_par_file.WriteCommentLine("C Inner mask radius for refinement (A):    " + wxString::Format("%f", inner_mask_radius));
	my_output_par_file.WriteCommentLine("C Outer mask radius for refinement (A):    " + wxString::Format("%f", outer_mask_radius));
	my_output_par_file.WriteCommentLine("C Low resolution limit (A):                " + wxString::Format("%f", low_resolution_limit));
	my_output_par_file.WriteCommentLine("C High resolution limit (A):               " + wxString::Format("%f", high_resolution_limit));
	my_output_par_file.WriteCommentLine("C Defocus search range (A):                " + wxString::Format("%f", defocus_search_range));
	my_output_par_file.WriteCommentLine("C Defocus step (A):                        " + wxString::Format("%f", defocus_step));
	my_output_par_file.WriteCommentLine("C Padding factor:                          " + wxString::Format("%f", padding));
	my_output_par_file.WriteCommentLine("C Refine defocus:                          " + BoolToYesNo(ctf_refinement));
	my_output_par_file.WriteCommentLine("C Refine beamtilt:                         " + BoolToYesNo(beamtilt_refinement));
	my_output_par_file.WriteCommentLine("C Normalize particles:                     " + BoolToYesNo(normalize_particles));
	my_output_par_file.WriteCommentLine("C Invert particle contrast:                " + BoolToYesNo(invert_contrast));
	my_output_par_file.WriteCommentLine("C Exclude images with blank edges:         " + BoolToYesNo(exclude_blank_edges));
	my_output_par_file.WriteCommentLine("C Normalize input reconstruction:          " + BoolToYesNo(normalize_input_3d));
	my_output_par_file.WriteCommentLine("C Threshold input reconstruction:          " + BoolToYesNo(threshold_input_3d));
	my_output_par_file.WriteCommentLine("C");
	my_output_par_file.WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE");
	fflush(my_output_par_file.parameter_file);
	my_output_par_shifts_file.WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE");

	if (high_resolution_limit < 2.0 * pixel_size) high_resolution_limit = 2.0 * pixel_size;
	if (outer_mask_radius > float(input_stack.ReturnXSize()) / 2.0 * pixel_size- mask_falloff) outer_mask_radius = float(input_stack.ReturnXSize()) / 2.0 * pixel_size - mask_falloff;

	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, symmetry);
	input_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	ResolutionStatistics input_statistics(pixel_size, input_3d.density_map.logical_y_dimension);
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
	// Remove masking here to avoid edge artifacts later
	input_3d.density_map.CosineMask(outer_mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
	if (inner_mask_radius > 0.0) input_3d.density_map.CosineMask(inner_mask_radius / pixel_size, mask_falloff / pixel_size, true);
	if (threshold_input_3d)
	{
		average_density_max = input_3d.density_map.ReturnAverageOfMaxN(100, outer_mask_radius / pixel_size);
		input_3d.density_map.SetMinimumValue(-0.3 * average_density_max);
	}

	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	input_3d.mask_radius = outer_mask_radius;

	if (padding != 1.0)
	{
		input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension * padding, input_3d.density_map.logical_y_dimension * padding, input_3d.density_map.logical_z_dimension * padding, input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
		refine_statistics.part_SSNR.ResampleCurve(&refine_statistics.part_SSNR, refine_statistics.part_SSNR.number_of_points * padding);
	}

	input_3d.PrepareForProjections(low_resolution_limit, high_resolution_limit);
	binning_factor_refine = input_3d.pixel_size / pixel_size;
//	unbinned_image.Allocate(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, true);
//	binned_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
//	projection_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	phase_difference_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	phase_difference_sum.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	beamtilt_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);

	wxPrintf("\nBinning factor for refinement = %f, new pixel size = %f\n", binning_factor_refine, input_3d.pixel_size);

	temp_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	sum_power.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
//	refine_particle.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension);

// Read whole parameter file to work out average values and variances
	j = 0;
	for (current_image = 0; current_image < input_par_file.number_of_lines; current_image++)
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

		if (parameter_variance[i] < 0.001f) refine_particle.constraints_used[i] = false;
	}
//	refine_particle.SetParameterStatistics(parameter_average, parameter_variance);
	input_par_file.Rewind();

//	image_counter = 0;
//#pragma omp parallel num_threads(max_threads) default(none) shared(input_par_file, input_stack, image_counter) \
//private(current_image, input_image_local, global_random_number_generator)
//{
//		input_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
//
//#pragma omp for schedule(static,1)
//	for (current_image = 0; current_image < input_par_file.number_of_lines; current_image++)
//	{
//#pragma omp critical
//		{
//			image_counter++;
//			input_image_local.ReadSlice(&input_stack, image_counter);
//		}
////		input_image_local.ReadSlice(&input_stack, myroundint(fabsf(global_random_number_generator.GetUniformRandom() * (input_par_file.number_of_lines-1) + 1)));
//	}
//
//	input_image_local.Deallocate();
//}

	if (normalize_particles)
	{
		wxPrintf("Calculating noise power spectrum...\n\n");
		random_reset_count = std::max(random_reset_count, max_threads);
		percentage = float(max_samples) / float(images_to_process) / random_reset_count;
		sum_power.SetToConstant(0.0f);
		mask_radius_for_noise = outer_mask_radius / pixel_size;
		number_of_blank_edges = 0;
		if (2.0 * mask_radius_for_noise + mask_falloff / pixel_size > 0.95f * input_image.logical_x_dimension)
		{
			mask_radius_for_noise = 0.95f * input_image.logical_x_dimension / 2.0f - mask_falloff / 2.0f / pixel_size;
		}
		noise_power_spectrum.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((sum_power.logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
		number_of_terms.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((sum_power.logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
		if (is_running_locally == true) my_progress = new ProgressBar(images_to_process / max_threads);
		current_image = 0;
		random_reset_counter = 0;

		#pragma omp parallel num_threads(max_threads) default(none) shared(input_par_file, first_particle, last_particle, my_progress, percentage, exclude_blank_edges, input_stack, \
			outer_mask_radius, pixel_size, mask_falloff, number_of_blank_edges, sum_power, current_image, global_random_number_generator, random_reset_count, random_reset_counter, \
			mask_radius_for_noise) \
		private(current_image_local, input_parameters, image_counter, number_of_blank_edges_local, variance, temp_image_local, sum_power_local, input_image_local, temp_float, file_read)
		{

		image_counter = 0;
		number_of_blank_edges_local = 0;
		input_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		temp_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		sum_power_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
		sum_power_local.SetToConstant(0.0f);
		file_read = false;

		#pragma omp for schedule(static,1)
		for (current_image_local = 0; current_image_local < input_par_file.number_of_lines; current_image_local++)
		{
			#pragma omp critical
			{
				input_par_file.ReadLine(input_parameters, current_image);
				current_image++;
				if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle)
				{
					file_read = false;
					if (random_reset_counter == 0) temp_float = global_random_number_generator.GetUniformRandom();
					if ((temp_float >= 1.0 - 2.0f * percentage) || (random_reset_counter != 0))
					{
						random_reset_counter++;
						if (random_reset_counter == random_reset_count) random_reset_counter = 0;
//						wxPrintf("reading %i\n", int(input_parameters[0] + 0.5f));
						input_image_local.ReadSlice(&input_stack, int(input_parameters[0] + 0.5f));
						file_read = true;
					}
				}
			}
			if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
			image_counter++;
			if (is_running_locally == true && ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
			if (! file_read) continue;
//			if ((temp_float < 1.0 - 2.0f * percentage) && (random_reset_counter == 0)) continue;
//			if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0f * percentage)) continue;
//			input_image_local.ReadSlice(&input_stack, int(input_parameters[0] + 0.5f));
			if (exclude_blank_edges && input_image_local.ContainsBlankEdges(outer_mask_radius / pixel_size)) {number_of_blank_edges_local++; continue;}
			variance = input_image_local.ReturnVarianceOfRealValues(outer_mask_radius / pixel_size, 0.0f, 0.0f, 0.0f, true);
			if (variance == 0.0f) continue;
			input_image_local.MultiplyByConstant(1.0f / sqrtf(variance));
			input_image_local.CosineMask(mask_radius_for_noise, mask_falloff / pixel_size, true);
			input_image_local.ForwardFFT();
			temp_image_local.CopyFrom(&input_image_local);
			temp_image_local.ConjugateMultiplyPixelWise(input_image_local);
			sum_power_local.AddImage(&temp_image_local);
		}

		#pragma omp critical
		{
			number_of_blank_edges += number_of_blank_edges_local;
			sum_power.AddImage(&sum_power_local);
		}

		input_image_local.Deallocate();
		sum_power_local.Deallocate();
		temp_image_local.Deallocate();

		} // end omp section

		if (is_running_locally == true) delete my_progress;
		sum_power.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);
		noise_power_spectrum.SquareRoot();
		noise_power_spectrum.Reciprocal();

		input_par_file.Rewind();
		if (exclude_blank_edges)
		{
			wxPrintf("\nImages with blank edges excluded from noise power calculation = %i\n", number_of_blank_edges);
		}
	}

	wxPrintf("\nAverage sigma noise = %f, average LogP = %f\nAverage ShiftX = %f, average ShiftY = %f\nSigma ShiftX = %f, sigma ShiftY = %f\nNumber of particles to refine = %i\n\n",
			parameter_average[14], parameter_average[15], parameter_average[4], parameter_average[5], sqrtf(parameter_variance[4]), sqrtf(parameter_variance[5]), images_to_process);

	if (is_running_locally == true) my_progress = new ProgressBar(images_to_process / max_threads);

	phase_difference_sum.SetToConstant(0.0f);
	current_image = 0;

	#pragma omp parallel num_threads(max_threads) default(none) shared(parameter_average, parameter_variance, input_3d, input_par_file, input_stack, phase_difference_sum, max_threads, \
		first_particle, last_particle, invert_contrast, normalize_particles, noise_power_spectrum, padding, ctf_refinement, defocus_search_range, defocus_step, normalize_input_3d, \
		refine_statistics, voltage_kV, spherical_aberration_mm, amplitude_contrast, pixel_size, my_progress, outer_mask_radius, mask_falloff, high_resolution_limit, molecular_mass_kDa, \
		binning_factor_refine, cg_accuracy, low_resolution_limit, input_statistics, my_output_par_file, my_output_par_shifts_file, output_parameter_average, output_parameter_change, current_image) \
	private(i, image_counter, refine_particle, current_image_local, phase_difference_sum_local, input_parameters, temp_float, output_parameters, input_ctf, variance, average, comparison_object, \
		best_score, defocus_i, score, cg_starting_point, output_parameter_average_local, output_parameter_change_local, input_image_local, phase_difference_image_local, unbinned_image, \
		binned_image, projection_image_local, best_defocus_i)
	{ // for omp

	refine_particle.constraints_used[4] = true;		// Constraint for X shifts
	refine_particle.constraints_used[5] = true;		// Constraint for Y shifts
	refine_particle.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension);
	refine_particle.SetParameterStatistics(parameter_average, parameter_variance);

	phase_difference_sum_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	input_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	unbinned_image.Allocate(input_stack.ReturnXSize() * padding, input_stack.ReturnYSize() * padding, true);
	binned_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	phase_difference_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	projection_image_local.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);

	ZeroFloatArray(output_parameter_average_local, refine_particle.number_of_parameters);
	ZeroFloatArray(output_parameter_change_local, refine_particle.number_of_parameters);

	image_counter = 0;
	phase_difference_sum_local.SetToConstant(0.0f);

	#pragma omp for schedule(static,1)
	for (current_image_local = 0; current_image_local < input_par_file.number_of_lines; current_image_local++)
	{
		#pragma omp critical
		{
			input_par_file.ReadLine(input_parameters, current_image);
			current_image++;
			if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle) input_image_local.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
		}
		if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
		image_counter++;
		temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		for (i = 0; i < refine_particle.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}

// Set up Particle object
		refine_particle.ResetImageFlags();
		refine_particle.mask_radius = outer_mask_radius;
		refine_particle.mask_falloff = mask_falloff;
//		refine_particle.filter_radius_low = low_resolution_limit;
		refine_particle.filter_radius_high = high_resolution_limit;
		refine_particle.molecular_mass_kDa = molecular_mass_kDa;
		refine_particle.signed_CC_limit = pixel_size;
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

//		input_image_local.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
		input_image_local.ReplaceOutliersWithMean(5.0);
		if (invert_contrast) input_image_local.InvertRealValues();
		if (normalize_particles)
		{
			input_image_local.ForwardFFT();
			// Whiten noise
			input_image_local.ApplyCurveFilter(&noise_power_spectrum);
			// Apply cosine filter to reduce ringing
//			input_image_local.CosineMask(std::max(pixel_size / high_resolution_limit, pixel_size / 7.0f + pixel_size / mask_falloff) - pixel_size / (2.0 * mask_falloff), pixel_size / mask_falloff);
			input_image_local.BackwardFFT();
			// Normalize background variance and average
			variance = input_image_local.ReturnVarianceOfRealValues(input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
			average = input_image_local.ReturnAverageOfRealValues(input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, true);
			input_image_local.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
			// At this point, input_image should have white background with a variance of 1. The variance should therefore be about 1/binning_factor^2 after binning.
		}

		// Option to add noise to images to get out of local optima
//		input_image.AddGaussianNoise(sqrtf(2.0 * input_image.ReturnVarianceOfRealValues()));

		input_image_local.ClipInto(&unbinned_image);
		unbinned_image.ForwardFFT();
		unbinned_image.ClipInto(refine_particle.particle_image);
		// Multiply by binning_factor so variance after binning is close to 1.
//		refine_particle.particle_image->MultiplyByConstant(binning_factor_refine);
		comparison_object.reference_volume = &input_3d;
		comparison_object.projection_image = &projection_image_local;
		comparison_object.particle = &refine_particle;
		refine_particle.MapParameters(cg_starting_point);
		refine_particle.PhaseShiftInverse();

		refine_particle.filter_radius_low = 30.0;
		refine_particle.SetIndexForWeightedCorrelation();
		binned_image.CopyFrom(refine_particle.particle_image);
		refine_particle.InitCTF(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], input_parameters[11]);
		if (ctf_refinement)
		{
			best_score = - std::numeric_limits<float>::max();
			for (defocus_i = - myround(float(defocus_search_range)/float(defocus_step)); defocus_i <= myround(float(defocus_search_range)/float(defocus_step)); defocus_i++)
			{
				refine_particle.SetDefocus(input_parameters[8] + defocus_i * defocus_step, input_parameters[9] + defocus_i * defocus_step, input_parameters[10], input_parameters[11]);
				refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8] + defocus_i * defocus_step, input_parameters[9] + defocus_i * defocus_step, input_parameters[10], input_parameters[11]);
				if (normalize_input_3d) refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 1);
				// Apply SSNR weighting only to image since input 3D map assumed to be calculated from correctly whitened images
				else refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 0);
				refine_particle.PhaseFlipImage();
	//			refine_particle.CosineMask(false, true, 0.0);
				refine_particle.CosineMask();
				refine_particle.PhaseShift();
				refine_particle.CenterInCorner();
	//			refine_particle.WeightBySSNR(input_3d.statistics.part_SSNR, 1);

				score = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
				if (score > best_score)
				{
					best_score = score;
					best_defocus_i = defocus_i;
				}
				refine_particle.particle_image->CopyFrom(&binned_image);
				refine_particle.is_ssnr_filtered = false;
				refine_particle.is_masked = false;
				refine_particle.is_centered_in_box = true;
				refine_particle.shift_counter = 1;
			}
			output_parameters[8] = input_parameters[8] + best_defocus_i * defocus_step;
			output_parameters[9] = input_parameters[9] + best_defocus_i * defocus_step;
		}
		else
		{
			output_parameters[8] = input_parameters[8];
			output_parameters[9] = input_parameters[9];
		}
		refine_particle.SetDefocus(output_parameters[8], output_parameters[9], input_parameters[10], input_parameters[11]);
		refine_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, output_parameters[8], output_parameters[9], input_parameters[10], input_parameters[11]);

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
		input_parameters[15] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
		output_parameters[15] = input_parameters[15]; output_parameters[16] = 0.0;

		output_parameters[13] = refine_particle.ReturnLogLikelihood(input_image_local, unbinned_image, input_ctf, input_3d, input_statistics, 2.0f * pixel_size, &phase_difference_image_local);
		phase_difference_sum_local.AddImage(&phase_difference_image_local);

		if (refine_particle.snr > 0.0f) output_parameters[14] = sqrtf(1.0f / refine_particle.snr);

		temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		temp_float = output_parameters[1]; output_parameters[1] = output_parameters[3]; output_parameters[3] = temp_float;
		for (i = 1; i < refine_particle.number_of_parameters; i++)
		{
			if (std::isnan(output_parameters[i]) != 0)
			{
				output_parameters[i] = input_parameters[i];
			}
		}
		input_parameters[7] = 1;
		output_parameters[7] = input_parameters[7];
		if (output_parameters[15] < 0.0f) output_parameters[15] = 0.0f;
		// will not work with threading
//		my_output_par_file.WriteLine(output_parameters);

/*		if (is_running_locally == false) // send results back to the gui..
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
		} */

		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_average_local[i] += output_parameters[i];}
		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameters[i] -= input_parameters[i];}
		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_change_local[i] += powf(output_parameters[i],2);}
		// will not work with threading
//		my_output_par_shifts_file.WriteLine(output_parameters);

		if (max_threads < 2)
		{
			fflush(my_output_par_file.parameter_file);
			fflush(my_output_par_shifts_file.parameter_file);
		}

		if (is_running_locally == true && ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
	}

	#pragma omp critical
	{
		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_average[i] += output_parameter_average_local[i];}
		for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_change[i] += output_parameter_change_local[i];}
		phase_difference_sum.AddImage(&phase_difference_sum_local);
	}

	phase_difference_sum_local.Deallocate();
	input_image_local.Deallocate();
	unbinned_image.Deallocate();
	binned_image.Deallocate();
	phase_difference_image_local.Deallocate();
	projection_image_local.Deallocate();
	refine_particle.Deallocate();

	} // end omp section

	if (is_running_locally == true) delete my_progress;

	if (beamtilt_refinement)
	{
		input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, output_parameters[8], output_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, pixel_size, input_parameters[11]);
		refine_particle.FindBeamTilt(phase_difference_sum, input_ctf, pixel_size, temp_image, beamtilt_image, sum_power, beamtilt_x, beamtilt_y, particle_shift_x, particle_shift_y, phase_multiplier, is_running_locally);

		temp_image.WriteSlice(&ouput_phase_difference_file,1);
		sum_power.WriteSlice(&ouput_difference_file,1);
		beamtilt_image.WriteSlice(&ouput_beamtilt_file,1);

		wxPrintf("\nBeam tilt x,y [mrad]   = %10.4f %10.4f\n", 1000.0f * beamtilt_x, 1000.0f * beamtilt_y);
		wxPrintf("Particle shift x,y [A] = %10.4f %10.4f\n", particle_shift_x, particle_shift_y);
	}

	for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_average[i] /= float(last_particle - first_particle + 1);}
	for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_change[i] /= float(last_particle - first_particle + 1);}
	for (i = 1; i < refine_particle.number_of_parameters; i++) {output_parameter_change[i] = sqrtf(output_parameter_change[i]);}
	my_output_par_file.WriteLine(output_parameter_average, true);
	my_output_par_shifts_file.WriteLine(output_parameter_change, true);
	my_output_par_file.WriteCommentLine("C  Total particles included, overall score, average occupancy "
			+ wxString::Format("%11i %10.6f %10.6f", last_particle - first_particle + 1, output_parameter_average[15], output_parameter_average[12]));

	if (is_running_locally == true)
	{
		wxPrintf("\nRefine CTF: Normal termination\n");
		wxDateTime finish_time = wxDateTime::Now();
		wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
	}

	return true;
}
//
//
//
//	float expected_threshold;
//	float actual_number_of_ccs_calculated;
//
//	float temp_float;
//	double temp_double_array[5];
//
//	int number_of_rotations;
//	long total_correlation_positions;
//	long current_correlation_position;
//	long pixel_counter;
//
//	int current_search_position;
//	int current_x;
//	int current_y;
//
//	int defocus_i;
//
//	EulerSearch	global_euler_search;
//	AnglesAndShifts angles;
//
//	ImageFile input_search_image_file;
//	ImageFile input_reconstruction_file;
//
//	Curve whitening_filter;
//	Curve number_of_terms;
//
//	input_search_image_file.OpenFile(input_search_images_filename.ToStdString(), false);
//	input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString(), false);
//
//	Image input_image;
//	Image padded_reference;
//	Image input_reconstruction;
//	Image current_projection;
//	Image padded_projection;
//
//	Image projection_filter;
//
//	Image max_intensity_projection;
//
//	Image best_psi;
//	Image best_theta;
//	Image best_phi;
//	Image best_defocus;
//
//	Image correlation_pixel_sum;
//	Image correlation_pixel_sum_of_squares;
//
//	input_image.ReadSlice(&input_search_image_file, 1);
//	padded_reference.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	max_intensity_projection.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	best_psi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	best_theta.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	best_phi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	best_defocus.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	correlation_pixel_sum.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	correlation_pixel_sum_of_squares.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//
//	padded_reference.SetToConstant(0.0f);
//	max_intensity_projection.SetToConstant(0.0f);
//	best_psi.SetToConstant(0.0f);
//	best_theta.SetToConstant(0.0f);
//	best_phi.SetToConstant(0.0f);
//	best_defocus.SetToConstant(0.0f);
//	correlation_pixel_sum.SetToConstant(0.0f);
//	correlation_pixel_sum_of_squares.SetToConstant(0.0f);
//
//// Some settings for testing
//	padding = 2.0f;
////	ctf_refinement = true;
////	defocus_search_range = 200.0f;
////	defocus_step = 50.0f;
//
//	input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices());
//	if (padding != 1.0f)
//	{
//		input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges());
//	}
//	input_reconstruction.ForwardFFT();
//	//input_reconstruction.CosineMask(0.1, 0.01, true);
//	//input_reconstruction.Whiten();
//	//if (first_search_position == 0) input_reconstruction.QuickAndDirtyWriteSlices("/tmp/filter.mrc", 1, input_reconstruction.logical_z_dimension);
//	input_reconstruction.ZeroCentralPixel();
//	input_reconstruction.SwapRealSpaceQuadrants();
//
//	CTF input_ctf;
//	input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));
//
//	// assume cube
//
//	current_projection.Allocate(input_reconstruction_file.ReturnXSize(), input_reconstruction_file.ReturnXSize(), false);
//	projection_filter.Allocate(input_reconstruction_file.ReturnXSize(), input_reconstruction_file.ReturnXSize(), false);
//	if (padding != 1.0f) padded_projection.Allocate(input_reconstruction_file.ReturnXSize() * padding, input_reconstruction_file.ReturnXSize() * padding, false);
//
//
//	// angular step
//
//	if (angular_step <= 0) angular_step = CalculateAngularStep(high_resolution_limit_search, mask_radius_search);
//	if (in_plane_angular_step <= 0)
//	{
//		psi_step = rad_2_deg(pixel_size / mask_radius_search);
//		psi_step = 360.0 / int(360.0 / psi_step + 0.5);
//	}
//	else
//	{
//		psi_step = in_plane_angular_step;
//	}
//
//	//psi_start = psi_step / 2.0 * global_random_number_generator.GetUniformRandom();
//	psi_start = 0.0f;
//	psi_max = 360.0f;
//
//	//psi_step = 5;
//
//	//wxPrintf("psi_start = %f, psi_max = %f, psi_step = %f\n", psi_start, psi_max, psi_step);
//
//	// search grid
//
//	global_euler_search.InitGrid(my_symmetry, angular_step, 0.0, 0.0, psi_max, psi_step, psi_start, pixel_size / high_resolution_limit_search, parameter_map, best_parameters_to_keep);
//
//	if (global_euler_search.test_mirror == true) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
//	{
//		global_euler_search.theta_max = 180.0f;
//	}
//
//	global_euler_search.CalculateGridSearchPositions(false);
//
//
//	// for now, I am assuming the MTF has been applied already.
//	// work out the filter to just whiten the image..
//
//	whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
//	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
//
//	wxDateTime my_time_out;
//	wxDateTime my_time_in;
//
//	// remove outliers
//
//	input_image.ReplaceOutliersWithMean(5.0f);
//	input_image.ForwardFFT();
//	input_image.SwapRealSpaceQuadrants();
//
//	input_image.ZeroCentralPixel();
//	input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
//	whitening_filter.SquareRoot();
//	whitening_filter.Reciprocal();
//	whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue());
//
//	//whitening_filter.WriteToFile("/tmp/filter.txt");
//	input_image.ApplyCurveFilter(&whitening_filter);
//	input_image.ZeroCentralPixel();
//	input_image.DivideByConstant(sqrt(input_image.ReturnSumOfSquares()));
//	//input_image.QuickAndDirtyWriteSlice("/tmp/white.mrc", 1);
//	//exit(-1);
//
//	// count total searches (lazy)
//
//	total_correlation_positions = 0;
//	current_correlation_position = 0;
//
//	// if running locally, search over all of them
//
//	if (is_running_locally == true)
//	{
//		first_search_position = 0;
//		last_search_position = global_euler_search.number_of_search_positions - 1;
//	}
//
//	for (current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++)
//	{
//		//loop over each rotation angle
//
//		for (current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
//		{
//			total_correlation_positions++;
//		}
//	}
//
//	if (defocus_step <= 0.0)
//	{
//		defocus_search_range = 0.0f;
//		defocus_step = 100.0f;
//	}
//	total_correlation_positions *= (2 * myroundint(float(defocus_search_range)/float(defocus_step)) + 1);
//
//	number_of_rotations = 0;
//
//	for (current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
//	{
//		number_of_rotations++;
//	}
//
//	if (is_running_locally == true)
//	{
//		//Loop over ever search position
//
//		wxPrintf("\nSearching %i positions on the euler sphere.\n", last_search_position - first_search_position, first_search_position, last_search_position);
//		wxPrintf("Searching %i rotations per position.\n", number_of_rotations);
//		wxPrintf("There are %li correlation positions total.\n\n", total_correlation_positions);
//
//		wxPrintf("Performing Search...\n\n");
//		my_progress = new ProgressBar(total_correlation_positions);
//	}
//
////	wxPrintf("Searching %i - %i of %i total positions\n", first_search_position, last_search_position, global_euler_search.number_of_search_positions);
////	wxPrintf("psi_start = %f, psi_max = %f, psi_step = %f\n", psi_start, psi_max, psi_step);
//
//	actual_number_of_ccs_calculated = 0.0;
//
//
//// REMOVE THIS SECTION
//
//	//Image average_value_image;
//	//Image variance_image;
//
//	//average_value_image.QuickAndDirtyReadSlice("/tmp/centre_sums.mrc", 1);
//	//variance_image.QuickAndDirtyReadSlice("/tmp/centre_square_sums.mrc", 1);
//
//	//average_value_image.SwapRealSpaceQuadrants();
//	//variance_image.SwapRealSpaceQuadrants();
//
//
//	//Image *correlation_buffers;
//	//int current_rotation;
//	//correlation_buffers = new Image[number_of_rotations];
//
//	//for (current_rotation = 0; current_rotation < number_of_rotations; current_rotation++)
//	//{
//		//correlation_buffers[current_rotation].Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
//	//}
//
////	last_search_position = first_search_position;
////	psi_max = psi_start;
//
//	for (defocus_i = - myroundint(float(defocus_search_range)/float(defocus_step)); defocus_i <= myroundint(float(defocus_search_range)/float(defocus_step)); defocus_i++)
//	{
//		// make the projection filter, which will be CTF * whitening filter
//		input_ctf.SetDefocus((defocus1 + defocus_i * defocus_step) / pixel_size, (defocus2 + defocus_i * defocus_step) / pixel_size, deg_2_rad(defocus_angle));
//		projection_filter.CalculateCTFImage(input_ctf);
//		projection_filter.ApplyCurveFilter(&whitening_filter);
//
//		for (current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++)
//		{
//			//loop over each rotation angle
//
//			//current_rotation = 0;
//			for (current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step)
//			{
//				angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);
////				angles.Init(67.909943, 132.502991, 298.514923, 0.0, 0.0);
//
//				if (padding != 1.0f)
//				{
//					input_reconstruction.ExtractSlice(padded_projection, angles, 1.0f, false);
//					padded_projection.SwapRealSpaceQuadrants();
//					padded_projection.BackwardFFT();
//					padded_projection.ClipInto(&current_projection);
//					current_projection.ForwardFFT();
//				}
//				else
//				{
//					input_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
//					current_projection.SwapRealSpaceQuadrants();
//				}
//				//if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj_nofilter.mrc", 1);
//
//
//				current_projection.MultiplyPixelWise(projection_filter);
//
//
//				//if (first_search_position == 0) projection_filter.QuickAndDirtyWriteSlice("/tmp/projection_filter.mrc", 1);
//				//if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj_afterfilter.mrc", 1);
//
//				//current_projection.ZeroCentralPixel();
//				//current_projection.DivideByConstant(sqrt(current_projection.ReturnSumOfSquares()));
//				current_projection.BackwardFFT();
//				//current_projection.ReplaceOutliersWithMean(6.0f);
//
//				// find the pixel with the largest absolute density, and shift it to the centre
//
//			/*	pixel_counter = 0;
//				int best_x;
//				int best_y;
//				float max_value = -FLT_MAX;
//
//				for ( int y = 0; y < current_projection.logical_y_dimension; y ++ )
//				{
//					for ( int x = 0; x < current_projection.logical_x_dimension; x ++ )
//					{
//						if (fabsf(current_projection.real_values[pixel_counter]) > max_value)
//						{
//							max_value = fabsf(current_projection.real_values[pixel_counter]);
//							best_x = x - current_projection.physical_address_of_box_center_x;
//							best_y = y - current_projection.physical_address_of_box_center_y;;
//						}
//						pixel_counter++;
//					}
//					pixel_counter += current_projection.padding_jump_value;
//				}
//
//				current_projection.RealSpaceIntegerShift(best_x, best_y, 0);
//*/
//				///
//
//
//				current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges());
////				if (first_search_position == 0) current_projection.QuickAndDirtyWriteSlice("/tmp/small_proj.mrc", 1);
//
//				padded_reference.SetToConstant(0.0f);
//				current_projection.ClipIntoLargerRealSpace2D(&padded_reference);
//
//				padded_reference.ForwardFFT();
//				padded_reference.ZeroCentralPixel();
//				padded_reference.DivideByConstant(sqrtf(padded_reference.ReturnSumOfSquares()));
//
//				//if (first_search_position == 0)  padded_reference.QuickAndDirtyWriteSlice("/tmp/proj.mrc", 1);
//
//#ifdef MKL
//				// Use the MKL
//				vmcMulByConj(padded_reference.real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (input_image.complex_values),reinterpret_cast <MKL_Complex8 *> (padded_reference.complex_values),reinterpret_cast <MKL_Complex8 *> (padded_reference.complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
//#else
//				for (pixel_counter = 0; pixel_counter < padded_reference.real_memory_allocated / 2; pixel_counter ++)
//				{
//					padded_reference.complex_values[pixel_counter] = conj(padded_reference.complex_values[pixel_counter]) * input_image.complex_values[pixel_counter];
//				}
//#endif
//
//				//padded_reference.MultiplyByConstant(sqrtf(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension));
//				padded_reference.BackwardFFT();
//
//				// REMOVE THIS
//				//padded_reference.SubtractImage(&average_value_image);
//			//	padded_reference.DividePixelWise(variance_image);
//
//				//padded_reference.RealSpaceIntegerShift(-best_x, -best_y, 0);
////				if (first_search_position == 0) padded_reference.QuickAndDirtyWriteSlice("/tmp/cc.mrc", 1);
//
//				//exit(-1);
//				//padded_reference.SwapRealSpaceQuadrants();
//
//				// update mip..
//
//				pixel_counter = 0;
//
//				for (current_y = 0; current_y < max_intensity_projection.logical_y_dimension; current_y++)
//				{
//					for (current_x = 0; current_x < max_intensity_projection.logical_x_dimension; current_x++)
//					{
//						// first mip
//
//						if (padded_reference.real_values[pixel_counter] > max_intensity_projection.real_values[pixel_counter])
//						{
//							max_intensity_projection.real_values[pixel_counter] = padded_reference.real_values[pixel_counter];
//							best_psi.real_values[pixel_counter] = current_psi;
//							best_theta.real_values[pixel_counter] = global_euler_search.list_of_search_parameters[current_search_position][1];
//							best_phi.real_values[pixel_counter] = global_euler_search.list_of_search_parameters[current_search_position][0];
//							best_defocus.real_values[pixel_counter] = defocus_i * defocus_step;
//						}
//
//						pixel_counter++;
//					}
//
//					pixel_counter+=padded_reference.padding_jump_value;
//				}
//
//
//				correlation_pixel_sum.AddImage(&padded_reference);
//				padded_reference.SquareRealValues();
//				correlation_pixel_sum_of_squares.AddImage(&padded_reference);
//
//				//max_intensity_projection.QuickAndDirtyWriteSlice("/tmp/mip.mrc", 1);
//
//				current_projection.is_in_real_space = false;
//				padded_reference.is_in_real_space = true;
//
//				current_correlation_position++;
//				if (is_running_locally == true) my_progress->Update(current_correlation_position);
//
//				if (is_running_locally == false)
//				{
//					actual_number_of_ccs_calculated++;
//					temp_float = current_correlation_position;
//					JobResult *temp_result = new JobResult;
//					temp_result->SetResult(1, &temp_float);
//					AddJobToResultQueue(temp_result);
//				}
//			}
//		}
//	}
//
//	if (is_running_locally == true)
//	{
//		delete my_progress;
//
//		// scale images..
//
//		for (pixel_counter = 0; pixel_counter <  correlation_pixel_sum.real_memory_allocated; pixel_counter++)
//		{
//
//			correlation_pixel_sum.real_values[pixel_counter] /= float(total_correlation_positions);
//			correlation_pixel_sum_of_squares.real_values[pixel_counter] = sqrtf(correlation_pixel_sum_of_squares.real_values[pixel_counter] / float(total_correlation_positions) - powf(correlation_pixel_sum.real_values[pixel_counter], 2)) * sqrtf(correlation_pixel_sum.logical_x_dimension * correlation_pixel_sum.logical_y_dimension);
//		}
//
//
//		max_intensity_projection.MultiplyByConstant(sqrtf(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension));
//		correlation_pixel_sum.MultiplyByConstant(sqrtf(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension));
////		correlation_pixel_sum_of_squares.MultiplyByConstant(max_intensity_projection.logical_x_dimension * max_intensity_projection.logical_y_dimension);
//
//		// we need to quadrant swap the images, also shift them, with an extra pixel shift.  This is because I take the conjugate of the input image, not the reference..
//
//		// calculate the expected threshold (from peter's paper)
//
//		expected_threshold = sqrtf(2.0f)*cisTEM_erfcinv((2.0f*(1))/((input_image.logical_x_dimension * input_image.logical_y_dimension * double(total_correlation_positions))));
//
//		// write out images..
//
////		wxPrintf("\nPeak at %g, %g : %g\n", max_intensity_projection.FindPeakWithIntegerCoordinates().x, max_intensity_projection.FindPeakWithIntegerCoordinates().y, max_intensity_projection.FindPeakWithIntegerCoordinates().value);
////		wxPrintf("Sigma = %g, ratio = %g\n", sqrtf(max_intensity_projection.ReturnVarianceOfRealValues()), max_intensity_projection.FindPeakWithIntegerCoordinates().value / sqrtf(max_intensity_projection.ReturnVarianceOfRealValues()));
//
//		max_intensity_projection.QuickAndDirtyWriteSlice(mip_output_file.ToStdString(), 1, true, pixel_size);
//		max_intensity_projection.SubtractImage(&correlation_pixel_sum);
//		max_intensity_projection.DividePixelWise(correlation_pixel_sum_of_squares);
//		max_intensity_projection.QuickAndDirtyWriteSlice(scaled_mip_output_file.ToStdString(), 1, true, pixel_size);
//
//		correlation_pixel_sum.QuickAndDirtyWriteSlice(correlation_average_output_file.ToStdString(), 1, true, pixel_size);
//		correlation_pixel_sum_of_squares.QuickAndDirtyWriteSlice(correlation_variance_output_file.ToStdString(), 1, true, pixel_size);
//		best_psi.QuickAndDirtyWriteSlice(best_psi_output_file.ToStdString(), 1, true, pixel_size);
//		best_theta.QuickAndDirtyWriteSlice(best_theta_output_file.ToStdString(), 1, true, pixel_size);
//		best_phi.QuickAndDirtyWriteSlice(best_phi_output_file.ToStdString(), 1, true, pixel_size);
//		best_defocus.QuickAndDirtyWriteSlice(best_defocus_output_file.ToStdString(), 1, true, pixel_size);
//	}
//	else
//	{
//		// send back the final images to master (who should merge them, and send to the gui)
//
//		long result_array_counter;
//		long number_of_result_floats = 3; // first  is x size, second float y size of images, 3rd is number allocated
//		long pixel_counter;
//
//		number_of_result_floats += max_intensity_projection.real_memory_allocated * 7;
//
//		float *result = new float[number_of_result_floats];
//		result[0] = max_intensity_projection.logical_x_dimension;
//		result[1] = max_intensity_projection.logical_y_dimension;
//		result[2] = max_intensity_projection.real_memory_allocated;
//		result[3] = actual_number_of_ccs_calculated;
//
//		result_array_counter = 4;
//
//		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
//		{
//			result[result_array_counter] = max_intensity_projection.real_values[pixel_counter];
//			result_array_counter++;
//		}
//
//		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
//		{
//			result[result_array_counter] = best_psi.real_values[pixel_counter];
//			result_array_counter++;
//		}
//
//
//		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
//		{
//			result[result_array_counter] = best_theta.real_values[pixel_counter];
//			result_array_counter++;
//		}
//
//
//		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
//		{
//			result[result_array_counter] = best_phi.real_values[pixel_counter];
//			result_array_counter++;
//		}
//
//
//		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
//		{
//			result[result_array_counter] = best_defocus.real_values[pixel_counter];
//			result_array_counter++;
//		}
//
//
//		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
//		{
//			result[result_array_counter] = correlation_pixel_sum.real_values[pixel_counter];
//			result_array_counter++;
//		}
//
//
//		for (pixel_counter = 0; pixel_counter < max_intensity_projection.real_memory_allocated; pixel_counter++)
//		{
//			result[result_array_counter] = correlation_pixel_sum_of_squares.real_values[pixel_counter];
//			result_array_counter++;
//		}
//
//		SendProgramDefinedResultToMaster(result, number_of_result_floats, image_number_for_gui, number_of_jobs_per_image_in_gui);
//	}
//
//	if (is_running_locally == true)
//	{
//		wxPrintf("\nRefine CTF: Normal termination\n");
//		wxDateTime finish_time = wxDateTime::Now();
//		wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
//	}
//
//	return true;
//}
//
void RefineCTFApp::MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results)
{
	// do we have this image number already?

	bool need_a_new_result = true;
	int array_location = -1;
	long pixel_counter;

	wxPrintf("Master Handling result for image %i..", result_number);

	for (int result_counter = 0; result_counter < aggregated_results.GetCount(); result_counter++)
	{
		if (aggregated_results[result_counter].image_number == result_number)
		{
			aggregated_results[result_counter].AddResult(result_array, array_size, result_number, number_of_expected_results);
			need_a_new_result = false;
			array_location = result_counter;
			wxPrintf("Found array location for image %i, at %i\n", result_number, array_location);
			break;
		}
	}

	if (need_a_new_result == true) // we aren't collecting data for this result yet.. start
	{
		AggregatedTemplateResult result_to_add;
		aggregated_results.Add(result_to_add);
		aggregated_results[aggregated_results.GetCount() - 1].image_number = result_number;
		aggregated_results[aggregated_results.GetCount() - 1].AddResult(result_array, array_size, result_number, number_of_expected_results);
		array_location = aggregated_results.GetCount() - 1;
		wxPrintf("Adding new result to array for image %i, at %i\n", result_number, array_location);
	}

	// did this complete a result?

	if (aggregated_results[array_location].number_of_received_results == number_of_expected_results) // we should be done for this image
	{
		// TODO send the result back to the GUI, for now hack mode to save the files to the directory..

		wxString directory_for_writing_results = my_job_package.jobs[0].arguments[34].ReturnStringArgument();

		Image temp_image;
		temp_image.Allocate(int(aggregated_results[array_location].collated_data_array[0]), int(aggregated_results[array_location].collated_data_array[1]), true);

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_mip_data[pixel_counter];
		}

		temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));
		wxPrintf("writing to %s/n", wxString::Format("%s/mip.mrc\n", directory_for_writing_results));
		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/mip.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		// psi

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_psi_data[pixel_counter];
		}

		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/psi.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		//theta

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_theta_data[pixel_counter];
		}

		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/theta.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		// phi

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_phi_data[pixel_counter];
		}

		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/phi.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		// defocus

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_defocus_data[pixel_counter];
		}

		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/defocus.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		// do the scaling..

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			aggregated_results[array_location].collated_pixel_sums[pixel_counter] /= aggregated_results[array_location].total_number_of_ccs;
			aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] = sqrtf(aggregated_results[array_location].collated_pixel_square_sums[pixel_counter] / aggregated_results[array_location].total_number_of_ccs - pow(aggregated_results[array_location].collated_pixel_sums[pixel_counter], 2)) * sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension);
			aggregated_results[array_location].collated_mip_data[pixel_counter] = (aggregated_results[array_location].collated_mip_data[pixel_counter] - aggregated_results[array_location].collated_pixel_sums[pixel_counter]) / aggregated_results[array_location].collated_pixel_square_sums[pixel_counter];
		}

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_mip_data[pixel_counter];
		}

		temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));
		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/scaled_mip.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		// sums

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_sums[pixel_counter];
		}

		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/sums.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		// square sums

		for (pixel_counter = 0; pixel_counter <  int(result_array[2]); pixel_counter++)
		{
			temp_image.real_values[pixel_counter] = aggregated_results[array_location].collated_pixel_square_sums[pixel_counter];
		}

		//temp_image.MultiplyByConstant(sqrtf(temp_image.logical_x_dimension * temp_image.logical_y_dimension));

		temp_image.QuickAndDirtyWriteSlice(wxString::Format("%s/square_sums.mrc", directory_for_writing_results).ToStdString(), aggregated_results[array_location].image_number);

		// this should be done now.. so delete it

		aggregated_results.RemoveAt(array_location);
	}
}

AggregatedTemplateResult::AggregatedTemplateResult()
{
	image_number = -1;
	number_of_received_results = 0;
	total_number_of_ccs = 0.0f;

	collated_data_array = NULL;
	collated_mip_data = NULL;
	collated_psi_data = NULL;
	collated_theta_data = NULL;
	collated_phi_data = NULL;
	collated_defocus_data = NULL;
	collated_pixel_sums = NULL;
	collated_pixel_square_sums = NULL;
}

AggregatedTemplateResult::~AggregatedTemplateResult()
{
	if (collated_data_array != NULL) delete [] collated_data_array;
}

void AggregatedTemplateResult::AddResult(float *result_array, long array_size, int result_number, int number_of_expected_results)
{
	if (collated_data_array == NULL)
	{
		collated_data_array = new float[array_size];
		ZeroFloatArray(collated_data_array, array_size);
		number_of_received_results = 0;
		total_number_of_ccs = 0.0f;

		// nasty..

		collated_mip_data = &collated_data_array[3];
		collated_psi_data = &collated_data_array[3 + int(result_array[2])];
		collated_theta_data = &collated_data_array[3 + int(result_array[2]) + int(result_array[2])];
		collated_phi_data = &collated_data_array[3  + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
		collated_defocus_data = &collated_data_array[3  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
		collated_pixel_sums = &collated_data_array[3  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
		collated_pixel_square_sums = &collated_data_array[3  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];

		collated_data_array[0] = result_array[0];
		collated_data_array[1] = result_array[1];
		collated_data_array[2] = result_array[2];
	}

	total_number_of_ccs += result_array[3];

	float *result_mip_data = &result_array[4];
	float *result_psi_data = &result_array[4 + int(result_array[2])];
	float *result_theta_data = &result_array[4 + int(result_array[2]) + int(result_array[2])];
	float *result_phi_data = &result_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	float *result_defocus_data = &result_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	float *result_pixel_sums = &result_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];
	float *result_pixel_square_sums = &result_array[4  + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2]) + int(result_array[2])];

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
			collated_defocus_data[pixel_counter] = result_defocus_data[pixel_counter];
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

	number_of_received_results++;
	wxPrintf("Received %i of %i results\n", number_of_received_results, number_of_expected_results);
}
