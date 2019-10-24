#include "../../core/core_headers.h"

class
RefineCTFApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	void MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results);
	void ProgramSpecificInit();
	// for master collation

	private:

	// for loading new 3d references on the fly..
	void SetupNewReference3D(wxString wanted_filename, float inner_mask_radius, float outer_mask_radius, float pixel_size, float mask_falloff, bool threshold_input_3d , float padding, bool beamtilt_refinement, float low_resolution_limit, float high_resolution_limit, float molecular_mass_in_kDa);

	MRCFile input_file;
	ReconstructedVolume			input_3d, unbinned_3d;
	Image						sum_for_master;
	long						total_results_for_master;
	int							number_of_received_results;
	float						voltage_for_master;
	float						spherical_aberration_for_master;
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
//	for (int i = 0; i < comparison_object->particle->number_of_parameters; i++) {comparison_object->particle->temp_float[i] = comparison_object->particle->current_parameters[i];}

	comparison_object->particle->temp_parameters = comparison_object->particle->current_parameters;
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
			- comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_parameters);
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
	wxString input_star_filename = my_input->GetFilenameFromUser("Input cisTEM star filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.star", true);
	wxString input_reconstruction = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	wxString input_reconstruction_statistics = my_input->GetFilenameFromUser("Input data statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
	bool use_statistics = my_input->GetYesNoFromUser("Use statistics", "Answer No if no statistics are available?", "Yes");
	wxString ouput_star_filename = my_input->GetFilenameFromUser("Output star file", "The output parameter file, containing your refined particle alignment parameters", "my_refined_parameters.star", false);
	wxString ouput_shift_filename = my_input->GetFilenameFromUser("Output parameter changes", "The changes in the alignment parameters compared to the input parameters", "my_parameter_changes.par", false);
	wxString ouput_phase_difference_image = my_input->GetFilenameFromUser("Output phase difference image", "Diagnostic image indicating the average phase difference (x20) between aligned images and matching projections", "my_phase_difference.mrc", false);
	wxString ouput_beamtilt_image = my_input->GetFilenameFromUser("Output beam tilt image", "Diagnostic image indicating phase difference (x20) generated by beam tilt", "my_beamtilt_image.mrc", false);
	wxString ouput_difference_image = my_input->GetFilenameFromUser("Output phase diff - beam tilt ", "Difference between phase difference and matching beam tilt (x20)", "my_difference_image.mrc", false);
	int first_particle = my_input->GetIntFromUser("First particle to refine (0 = first in stack)", "The first particle in the stack that should be refined", "1", 0);
	int last_particle = my_input->GetIntFromUser("Last particle to refine (0 = last in stack)", "The last particle in the stack that should be refined", "0", 0);
	float pixel_size = my_input->GetFloatFromUser("Pixel size of reconstruction (A)", "Pixel size of input reconstruction in Angstroms", "1.0", 0.0);
	float molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	float inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the input reconstruction in Angstroms", "0.0", 0.0);
	float outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the input reconstruction and images during refinement, in Angstroms", "100.0", inner_mask_radius);
	float low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	float high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	float defocus_search_range = my_input->GetFloatFromUser("Defocus search range (A)", "Search range (-value ... + value) around current defocus", "500.0", 0.0);
	float defocus_step = my_input->GetFloatFromUser("Defocus step (A)", "Step size used in the defocus search", "50.0", 0.0);
	float padding = my_input->GetFloatFromUser("Tuning parameters: padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
	bool ctf_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
	bool beamtilt_refinement = my_input->GetYesNoFromUser("Estimate beamtilt", "Should the beam tilt be estimated?", "No");
	bool normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
	bool invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	bool exclude_blank_edges = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "Yes");
	bool normalize_input_3d = my_input->GetYesNoFromUser("Normalize input reconstruction", "The input reconstruction should always be normalized unless it was generated by reconstruct3d with normalized particles", "Yes");
	bool threshold_input_3d = my_input->GetYesNoFromUser("Threshold input reconstruction", "Should the input reconstruction thresholded to suppress some of the background noise", "No");

#ifdef _OPENMP
	max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
	max_threads = 1;
#endif

	int job_number_from_gui = 0;
	int expected_number_of_results_from_gui = 0;
	bool estimate_phase_difference_image;
	wxString directory_for_results = "/dev/null"; // shouldn't be used in interactive

	delete my_input;

//	my_current_job.Reset(32);
    	my_current_job.ManualSetArguments("ttttbtttttiifffffffffbbbbbbbiii",
		input_particle_images.ToUTF8().data(),
		input_star_filename.ToUTF8().data(),
		input_reconstruction.ToUTF8().data(),
		input_reconstruction_statistics.ToUTF8().data(),
		use_statistics,
		ouput_star_filename.ToUTF8().data(),
		ouput_shift_filename.ToUTF8().data(),
		ouput_phase_difference_image.ToUTF8().data(),
		ouput_beamtilt_image.ToUTF8().data(),
		ouput_difference_image.ToUTF8().data(),
		first_particle,
		last_particle,
		pixel_size,
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
		job_number_from_gui,
		expected_number_of_results_from_gui,
		max_threads);
}

// override the do calculation method which will be what is actually run..

bool RefineCTFApp::DoCalculation()
{
	wxDateTime start_time = wxDateTime::Now();

	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_star_filename 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_reconstruction				= my_current_job.arguments[2].ReturnStringArgument();
	wxString input_reconstruction_statistics 	= my_current_job.arguments[3].ReturnStringArgument();
	bool	 use_statistics						= my_current_job.arguments[4].ReturnBoolArgument();
	wxString output_star_filename				= my_current_job.arguments[5].ReturnStringArgument();
	wxString output_shift_filename				= my_current_job.arguments[6].ReturnStringArgument();
	wxString ouput_phase_difference_image		= my_current_job.arguments[7].ReturnStringArgument();
	wxString ouput_beamtilt_image				= my_current_job.arguments[8].ReturnStringArgument();
	wxString ouput_difference_image				= my_current_job.arguments[9].ReturnStringArgument();
	int		 first_particle						= my_current_job.arguments[10].ReturnIntegerArgument();
	int		 last_particle						= my_current_job.arguments[11].ReturnIntegerArgument();
	float 	 pixel_size							= my_current_job.arguments[12].ReturnFloatArgument();
	float	 molecular_mass_kDa					= my_current_job.arguments[13].ReturnFloatArgument();
	float    inner_mask_radius					= my_current_job.arguments[14].ReturnFloatArgument();
	float    outer_mask_radius					= my_current_job.arguments[15].ReturnFloatArgument();
	float    low_resolution_limit				= my_current_job.arguments[16].ReturnFloatArgument();
	float    high_resolution_limit				= my_current_job.arguments[17].ReturnFloatArgument();
	float	 defocus_search_range				= my_current_job.arguments[18].ReturnFloatArgument();
	float	 defocus_step						= my_current_job.arguments[19].ReturnFloatArgument();
	float	 padding							= my_current_job.arguments[20].ReturnFloatArgument();
	bool	 ctf_refinement						= my_current_job.arguments[21].ReturnBoolArgument();
	bool	 beamtilt_refinement				= my_current_job.arguments[22].ReturnBoolArgument();
	bool	 normalize_particles				= my_current_job.arguments[23].ReturnBoolArgument();
	bool	 invert_contrast					= my_current_job.arguments[24].ReturnBoolArgument();
	bool	 exclude_blank_edges				= my_current_job.arguments[25].ReturnBoolArgument();
	bool	 normalize_input_3d					= my_current_job.arguments[26].ReturnBoolArgument();
	bool	 threshold_input_3d					= my_current_job.arguments[27].ReturnBoolArgument();
	int		 job_number_from_gui				= my_current_job.arguments[28].ReturnIntegerArgument();
	int		 expected_number_of_jobs_from_gui	= my_current_job.arguments[29].ReturnIntegerArgument();
	int		 max_threads						= my_current_job.arguments[30].ReturnIntegerArgument();

	wxString currently_open_3d_filename;
	wxArrayString all_reference_3d_filenames;

	Particle refine_particle;

	int current_line;
	int current_line_local;
	int max_samples = 2000;
	int images_to_process = 0;
	int number_of_blank_edges;
	int number_of_blank_edges_local;
	int image_counter;
	int defocus_i;
	int best_defocus_i;
	int random_reset_counter;
	int random_reset_count = 10;
//	float defocus_lower_limit = 15000.0f * sqrtf(voltage_kV / 300.0f);
//	float defocus_upper_limit = 25000.0f * sqrtf(voltage_kV / 300.0f);
	float voltage_kV = 0.0f;
	float spherical_aberration_mm;
	float phase_multiplier = 1.0f;
	float beamtilt_x, beamtilt_y;
	float particle_shift_x, particle_shift_y;
	float mask_falloff = 20.0f;	// in Angstrom
	float average_density_max;
	float binning_factor_refine;
	float percentage;
	float temp_float;

	int filename_counter;

	cisTEMParameterLine input_parameters;
	cisTEMParameterLine output_parameters;
	cisTEMParameterLine parameter_average;
	cisTEMParameterLine parameter_variance;

	float cg_starting_point[17];
	float cg_accuracy[17];


	float mask_radius_for_noise;
	float variance;
	float average;
	float score;
	float best_score;
	bool file_read;
	bool *image_has_been_processed;
	wxString symmetry = "C1";

// Constraints for phi, theta, psi not yet implemented
	refine_particle.constraints_used.phi = false;
	refine_particle.constraints_used.theta = false;
	refine_particle.constraints_used.psi = false;
	refine_particle.constraints_used.x_shift = true;		// Constraint for X shifts
	refine_particle.constraints_used.y_shift = true;		// Constraint for Y shifts

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
	Image						ctf_image_local;
	Image						padded_projection_image_local;
	CTF   						input_ctf;
//	RandomNumberGenerator 		random_particle(true);
	ReconstructedVolume			input_3d_local, unbinned_3d_local;
	ImageProjectionComparison	comparison_object;
	Curve						noise_power_spectrum;
	Curve						number_of_terms;
	ProgressBar 				*my_progress;

	AnglesAndShifts rotation_matrix;

	ZeroFloatArray(cg_starting_point, 17);
	ZeroFloatArray(cg_accuracy, 17);

	if ((is_running_locally && !DoesFileExist(input_star_filename)) || (!is_running_locally && !DoesFileExistWithWait(input_star_filename, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input parameter file %s not found\n", input_star_filename));
	}
	if ((is_running_locally && !DoesFileExist(input_particle_images)) || (!is_running_locally && !DoesFileExistWithWait(input_particle_images, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input particle stack %s not found\n", input_particle_images));
	}
	if ((is_running_locally && !DoesFileExist(input_reconstruction)) || (!is_running_locally && !DoesFileExistWithWait(input_reconstruction, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input reconstruction %s not found\n", input_reconstruction));
	}

	cisTEMParameters input_star_file;
	input_star_file.ReadFromcisTEMStarFile(input_star_filename);

	if (input_star_file.parameters_that_were_read.reference_3d_filename == true)
	{
		if (is_running_locally == true) MyPrintfCyan("Running with per particle 3D reference from input star file..\n");
		all_reference_3d_filenames.Add(input_star_file.ReturnReference3DFilename(0).Trim(true).Trim(false));

		// get all the filenames..

		bool found_filename;

		for (current_line = 1; current_line < input_star_file.ReturnNumberofLines(); current_line++)
		{
			found_filename = false;

			for (filename_counter = 0; filename_counter < all_reference_3d_filenames.GetCount(); filename_counter++)
			{
				if (all_reference_3d_filenames[filename_counter] == input_star_file.ReturnReference3DFilename(current_line).Trim(true).Trim(false))
				{
					found_filename = true;
					break;
				}
			}

			if (found_filename == false)
			{
				all_reference_3d_filenames.Add(input_star_file.ReturnReference3DFilename(current_line).Trim(true).Trim(false));
			}
		}
	}
	else
	{
			input_star_file.SetAllReference3DFilename(input_reconstruction);
			all_reference_3d_filenames.Add(input_star_file.ReturnReference3DFilename(0).Trim(true).Trim(false));
	}

//	input_star_file.WriteTocisTEMStarFile("/tmp/star_file_with_filename.star");

	wxPrintf("There are are %li 3D references\n", all_reference_3d_filenames.GetCount());
	currently_open_3d_filename = all_reference_3d_filenames[0];

	if (input_star_file.parameters_that_were_read.reference_3d_filename == true)
	{
		if (is_running_locally == true) MyPrintfCyan("Running with per particle 3D reference from input star file..\n");
		all_reference_3d_filenames.Add(input_star_file.ReturnReference3DFilename(0).Trim(true).Trim(false));

		// get all the filenames..

		bool found_filename;

		for (current_line = 1; current_line < input_star_file.ReturnNumberofLines(); current_line++)
		{
			found_filename = false;

			for (filename_counter = 0; filename_counter < all_reference_3d_filenames.GetCount(); filename_counter++)
			{
				if (all_reference_3d_filenames[filename_counter] == input_star_file.ReturnReference3DFilename(current_line).Trim(true).Trim(false))
				{
					found_filename = true;
					break;
				}
			}

			if (found_filename == false)
			{
				all_reference_3d_filenames.Add(input_star_file.ReturnReference3DFilename(current_line).Trim(true).Trim(false));
			}
		}
	}
	else
	{
			input_star_file.SetAllReference3DFilename(input_reconstruction);
			all_reference_3d_filenames.Add(input_star_file.ReturnReference3DFilename(0).Trim(true).Trim(false));
	}

//	input_star_file.WriteTocisTEMStarFile("/tmp/star_file_with_filename.star");

//	wxPrintf("There are %li 3D references\n", all_reference_3d_filenames.GetCount());
	currently_open_3d_filename = all_reference_3d_filenames[0];

	MRCFile input_stack(input_particle_images.ToStdString(), false);

	if (last_particle == 0) last_particle = input_stack.ReturnZSize();
	if (first_particle == 0) first_particle = 1;
	if (last_particle > input_stack.ReturnZSize()) last_particle = input_stack.ReturnZSize();

	for (current_line = 0; current_line < input_star_file.ReturnNumberofLines(); current_line++)
	{
		if (input_star_file.ReturnPositionInStack(current_line) >= first_particle && input_star_file.ReturnPositionInStack(current_line) <= last_particle)
		{
			if (beamtilt_refinement)
			{
				input_parameters = input_star_file.ReturnLine(current_line);
				if (voltage_kV == 0.0f)
				{
					voltage_kV = input_parameters.microscope_voltage_kv;
					spherical_aberration_mm = input_parameters.microscope_spherical_aberration_mm;
				}
				else if (voltage_kV != input_parameters.microscope_voltage_kv)
				{
					SendErrorAndCrash(wxString::Format("Error: Input parameter file %s contains variable MicroscopeVoltagekV, not allowed with beamtilt estimation\n", input_star_filename));
				}
				else if (spherical_aberration_mm != input_parameters.microscope_spherical_aberration_mm)
				{
					SendErrorAndCrash(wxString::Format("Error: Input parameter file %s contains variable MicroscopeCsMM, not allowed with beamtilt estimation\n", input_star_filename));
				}
			}
			images_to_process++;
		}
	}

//	random_particle.SetSeed(int(10000.0 * fabsf(input_star_file.ReturnAverageSigma(true)))%10000);

	input_file.OpenFile(currently_open_3d_filename.ToStdString(), false, true);

	MRCFile ouput_phase_difference_file;
	MRCFile ouput_beamtilt_file(ouput_beamtilt_image.ToStdString(), true);
	MRCFile ouput_difference_file(ouput_difference_image.ToStdString(), true);

	if (is_running_locally == true)
	{
		ouput_phase_difference_file.OpenFile(ouput_phase_difference_image.ToStdString(), true);
		ouput_beamtilt_file.OpenFile(ouput_beamtilt_image.ToStdString(), true);
		ouput_difference_file.OpenFile(ouput_difference_image.ToStdString(), true);
	}


	// Hack to make threading work
//	float *output_par_cache;
//	output_par_cache = new float[input_par_file.records_per_line * input_par_file.number_of_lines];
//	float *output_par_shifts_cache;
//	output_par_shifts_cache = new float[input_par_file.records_per_line * input_par_file.number_of_lines];
//	int cache_pointer, current_line;

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

	cisTEMParameters output_star_file;
	cisTEMParameters output_shifts_file;

	if (is_running_locally == true)
	{
		output_star_file.PreallocateMemoryAndBlank(input_star_file.ReturnNumberofLines());
		output_shifts_file.PreallocateMemoryAndBlank(input_star_file.ReturnNumberofLines());

		output_star_file.AddCommentToHeader("# RefineCTF run date and time:             " + start_time.FormatISOCombined(' '));
		output_star_file.AddCommentToHeader("# Input particle images:                   " + input_particle_images);
		output_star_file.AddCommentToHeader("# Input cisTEM parameter filename:         " + input_star_filename);
		output_star_file.AddCommentToHeader("# Input reconstruction:                    " + input_reconstruction);
		output_star_file.AddCommentToHeader("# Input data statistics:                   " + input_reconstruction_statistics);
		output_star_file.AddCommentToHeader("# Use statistics:                          " + BoolToYesNo(use_statistics));
		output_star_file.AddCommentToHeader("# Output cisTEM star file:                 " + output_star_filename);
		output_star_file.AddCommentToHeader("# Output parameter changes:                " + output_shift_filename);
		output_star_file.AddCommentToHeader("# First particle to refine:                " + wxString::Format("%i", first_particle));
		output_star_file.AddCommentToHeader("# Last particle to refine:                 " + wxString::Format("%i", last_particle));
		output_star_file.AddCommentToHeader("# Pixel size of reconstruction (A):        " + wxString::Format("%f", pixel_size));
		//	output_star_file.AddCommentToHeader("# Beam energy (keV):                       " + wxString::Format("%f", voltage_kV));
		//	output_star_file.AddCommentToHeader("# Spherical aberration (mm):               " + wxString::Format("%f", spherical_aberration_mm));
		//	output_star_file.AddCommentToHeader("# Amplitude contrast:                      " + wxString::Format("%f", amplitude_contrast));
		output_star_file.AddCommentToHeader("# Molecular mass of particle (kDa):        " + wxString::Format("%f", molecular_mass_kDa));
		output_star_file.AddCommentToHeader("# Inner mask radius for refinement (A):    " + wxString::Format("%f", inner_mask_radius));
		output_star_file.AddCommentToHeader("# Outer mask radius for refinement (A):    " + wxString::Format("%f", outer_mask_radius));
		output_star_file.AddCommentToHeader("# Low resolution limit (A):                " + wxString::Format("%f", low_resolution_limit));
		output_star_file.AddCommentToHeader("# High resolution limit (A):               " + wxString::Format("%f", high_resolution_limit));
		output_star_file.AddCommentToHeader("# Defocus search range (A):                " + wxString::Format("%f", defocus_search_range));
		output_star_file.AddCommentToHeader("# Defocus step (A):                        " + wxString::Format("%f", defocus_step));
		output_star_file.AddCommentToHeader("# Padding factor:                          " + wxString::Format("%f", padding));
		output_star_file.AddCommentToHeader("# Refine defocus:                          " + BoolToYesNo(ctf_refinement));
		output_star_file.AddCommentToHeader("# Estimate beamtilt:                       " + BoolToYesNo(beamtilt_refinement));
		output_star_file.AddCommentToHeader("# Normalize particles:                     " + BoolToYesNo(normalize_particles));
		output_star_file.AddCommentToHeader("# Invert particle contrast:                " + BoolToYesNo(invert_contrast));
		output_star_file.AddCommentToHeader("# Exclude images with blank edges:         " + BoolToYesNo(exclude_blank_edges));
		output_star_file.AddCommentToHeader("# Normalize input reconstruction:          " + BoolToYesNo(normalize_input_3d));
		output_star_file.AddCommentToHeader("# Threshold input reconstruction:          " + BoolToYesNo(threshold_input_3d));
		output_star_file.AddCommentToHeader("#");
	}

	if (high_resolution_limit < 2.0 * pixel_size) high_resolution_limit = 2.0 * pixel_size;
	if (outer_mask_radius > float(input_file.ReturnXSize()) / 2.0 * pixel_size- mask_falloff) outer_mask_radius = float(input_file.ReturnXSize()) / 2.0 * pixel_size - mask_falloff;

	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, symmetry);
	input_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	ResolutionStatistics input_statistics(pixel_size, input_3d.density_map->logical_y_dimension);
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
	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	input_3d.mask_radius = outer_mask_radius;

	if (padding != 1.0)
	{
		refine_statistics.part_SSNR.ResampleCurve(&refine_statistics.part_SSNR, refine_statistics.part_SSNR.number_of_points * padding);
	}

	if (beamtilt_refinement)
	{
		phase_difference_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
		phase_difference_sum.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
		beamtilt_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
		phase_difference_sum.SetToConstant(0.0f);
	}

	SetupNewReference3D(currently_open_3d_filename, inner_mask_radius, outer_mask_radius, pixel_size, mask_falloff, threshold_input_3d , padding, beamtilt_refinement, low_resolution_limit, high_resolution_limit, molecular_mass_kDa);

	binning_factor_refine = input_3d.pixel_size / pixel_size;
//	unbinned_image.Allocate(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, true);
//	binned_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);

	wxPrintf("\nBinning factor for refinement = %f, new pixel size = %f\n", binning_factor_refine, input_3d.pixel_size);

	temp_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	sum_power.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
//	refine_particle.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension);


	// Read whole parameter file to work out average values and variances

	parameter_average = input_star_file.ReturnParameterAverages();
	parameter_variance = input_star_file.ReturnParameterVariances();

	if (parameter_variance.phi < 0.001) refine_particle.constraints_used.phi = false;
	if (parameter_variance.theta < 0.001) refine_particle.constraints_used.theta = false;
	if (parameter_variance.psi < 0.001) refine_particle.constraints_used.psi = false;
	if (parameter_variance.x_shift < 0.001) refine_particle.constraints_used.x_shift = false;
	if (parameter_variance.y_shift < 0.001) refine_particle.constraints_used.y_shift = false;

	if (normalize_particles)
	{
		wxPrintf("Calculating noise power spectrum...\n\n");
		random_reset_count = std::max(random_reset_count, max_threads);
		percentage = float(max_samples) / float(images_to_process) / random_reset_count;
		sum_power.SetToConstant(0.0f);
		number_of_blank_edges = 0;
		noise_power_spectrum.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((sum_power.logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
		number_of_terms.SetupXAxis(0.0f, 0.5f * sqrtf(2.0f), int((sum_power.logical_x_dimension / 2.0f + 1.0f) * sqrtf(2.0f) + 1.0f));
		if (is_running_locally == true) my_progress = new ProgressBar(images_to_process / max_threads);
		current_line = 0;
		random_reset_counter = 0;

		#pragma omp parallel num_threads(max_threads) default(none) shared(input_star_file, first_particle, last_particle, my_progress, percentage, exclude_blank_edges, input_stack, \
			outer_mask_radius, mask_falloff, number_of_blank_edges, sum_power, current_line, global_random_number_generator, random_reset_count, random_reset_counter) \
		private(current_line_local, input_parameters, image_counter, number_of_blank_edges_local, variance, temp_image_local, sum_power_local, input_image_local, temp_float, file_read, \
			mask_radius_for_noise)
		{

		image_counter = 0;
		number_of_blank_edges_local = 0;
		input_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		temp_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		sum_power_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
		sum_power_local.SetToConstant(0.0f);
		file_read = false;

		#pragma omp for schedule(static,1)
		for (current_line_local = 0; current_line_local < input_star_file.ReturnNumberofLines(); current_line_local++)
		{
			#pragma omp critical
			{
				input_parameters = input_star_file.ReturnLine(current_line);

				current_line++;
				if (input_parameters.position_in_stack >= first_particle && input_parameters.position_in_stack <= last_particle)
				{
					file_read = false;
					if (random_reset_counter == 0) temp_float = global_random_number_generator.GetUniformRandom();
					if ((temp_float >= 1.0 - 2.0f * percentage) || (random_reset_counter != 0))
					{
						random_reset_counter++;
						if (random_reset_counter == random_reset_count) random_reset_counter = 0;
						input_image_local.ReadSlice(&input_stack, input_parameters.position_in_stack);
						file_read = true;
					}
				}
			}
			if (input_parameters.position_in_stack < first_particle || input_parameters.position_in_stack > last_particle) continue;
			image_counter++;
			if (is_running_locally == true && ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
			if (! file_read) continue;
			mask_radius_for_noise = outer_mask_radius / input_parameters.pixel_size;
			if (2.0 * mask_radius_for_noise + mask_falloff / input_parameters.pixel_size > 0.95f * input_stack.ReturnXSize())
			{
				mask_radius_for_noise = 0.95f * input_stack.ReturnXSize() / 2.0f - mask_falloff / 2.0f / input_parameters.pixel_size;
			}
			if (exclude_blank_edges && input_image_local.ContainsBlankEdges(mask_radius_for_noise)) {number_of_blank_edges_local++; continue;}
			variance = input_image_local.ReturnVarianceOfRealValues(mask_radius_for_noise, 0.0f, 0.0f, 0.0f, true);
			if (variance == 0.0f) continue;
			input_image_local.MultiplyByConstant(1.0f / sqrtf(variance));
			input_image_local.CosineMask(mask_radius_for_noise, mask_falloff / input_parameters.pixel_size, true);
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

		if (exclude_blank_edges)
		{
			wxPrintf("\nImages with blank edges excluded from noise power calculation = %i\n", number_of_blank_edges);
		}
	}

	wxPrintf("\nAverage sigma noise = %f, average LogP = %f\nAverage ShiftX = %f, average ShiftY = %f\nSigma ShiftX = %f, sigma ShiftY = %f\n\nNumber of particles to refine = %i\n\n",
			parameter_average.sigma, parameter_average.score, parameter_average.x_shift, parameter_average.y_shift, sqrtf(parameter_variance.x_shift), sqrtf(parameter_variance.y_shift), images_to_process);

	if (is_running_locally == true) my_progress = new ProgressBar(images_to_process / max_threads);

	// look over all file names..

	for (filename_counter = 0; filename_counter < all_reference_3d_filenames.GetCount(); filename_counter++)
	{

		currently_open_3d_filename = all_reference_3d_filenames[filename_counter];
		SetupNewReference3D(currently_open_3d_filename, inner_mask_radius, outer_mask_radius, pixel_size, mask_falloff, threshold_input_3d , padding, beamtilt_refinement, low_resolution_limit, high_resolution_limit, molecular_mass_kDa);

		#pragma omp parallel num_threads(max_threads) default(none) shared(parameter_average, parameter_variance, input_star_file, input_stack, phase_difference_sum, max_threads, \
		first_particle, last_particle, invert_contrast, normalize_particles, noise_power_spectrum, padding, ctf_refinement, defocus_search_range, defocus_step, normalize_input_3d, \
		refine_statistics, pixel_size, my_progress, outer_mask_radius, mask_falloff, high_resolution_limit, molecular_mass_kDa, \
		binning_factor_refine, cg_accuracy, low_resolution_limit, input_statistics, output_star_file, beamtilt_refinement, currently_open_3d_filename, filename_counter) \
		private(image_counter, refine_particle, current_line_local, phase_difference_sum_local, input_parameters, temp_float, output_parameters, input_ctf, variance, average, comparison_object, \
		best_score, defocus_i, score, cg_starting_point, input_image_local, phase_difference_image_local, unbinned_image, input_3d_local, unbinned_3d_local, \
		binned_image, projection_image_local, best_defocus_i, ctf_image_local, padded_projection_image_local)
		{ // for omp

		CTF ctf_copy;
		Image image_copy;

		refine_particle.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension);
		refine_particle.SetParameterStatistics(parameter_average, parameter_variance);


		input_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);

		if (ctf_refinement)
		{
			input_3d_local.CopyAllButVolume(&input_3d);
			input_3d_local.density_map = input_3d.density_map;

			unbinned_image.Allocate(input_stack.ReturnXSize() * padding, input_stack.ReturnYSize() * padding, true);
			binned_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
			projection_image_local.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
		}

		if (beamtilt_refinement)
		{
//			unbinned_3d_local = unbinned_3d;
			unbinned_3d_local.CopyAllButVolume(&unbinned_3d);
			unbinned_3d_local.density_map = unbinned_3d.density_map;

			ctf_image_local.Allocate(unbinned_3d_local.density_map->logical_x_dimension, unbinned_3d_local.density_map->logical_y_dimension, false);
			padded_projection_image_local.Allocate(unbinned_3d_local.density_map->logical_x_dimension, unbinned_3d_local.density_map->logical_y_dimension, false);
			phase_difference_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
			phase_difference_sum_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
			phase_difference_sum_local.SetToConstant(0.0f);
		}

		image_counter = 0;

		#pragma omp for schedule(dynamic,1)
		for (current_line_local = 0; current_line_local < input_star_file.ReturnNumberofLines(); current_line_local++)
		{

			input_parameters = input_star_file.ReturnLine(current_line_local);
			if (input_parameters.position_in_stack < first_particle || input_parameters.position_in_stack > last_particle ) continue;

			if (input_parameters.reference_3d_filename.Trim(true).Trim(false) != currently_open_3d_filename)
			{
				wxPrintf("%s does not equal %s\n", input_parameters.reference_3d_filename.Trim(true).Trim(false), currently_open_3d_filename );
				continue;
			}

			input_image_local.ReadSlice(&input_stack, input_parameters.position_in_stack);
			image_counter++;

			output_parameters = input_parameters;

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
			refine_particle.sigma_noise = input_parameters.sigma / binning_factor_refine;
			//		refine_particle.logp = -std::numeric_limits<float>::max();
			refine_particle.SetParameters(input_parameters);
			refine_particle.MapParameterAccuracy(cg_accuracy);
			//		refine_particle.SetIndexForWeightedCorrelation();
			refine_particle.SetParameterConstraints(powf(parameter_average.sigma,2));

			input_ctf.Init(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, 0.0, 0.0, 0.0, pixel_size, input_parameters.phase_shift);
			//		ctf_input_image.CalculateCTFImage(input_ctf);
			//		refine_particle.is_phase_flipped = true;

			//		input_image_local.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
			input_image_local.ReplaceOutliersWithMean(5.0);
			if (invert_contrast) input_image_local.InvertRealValues();
			if (normalize_particles)
			{
				input_image_local.ChangePixelSize(&input_image_local, pixel_size / input_parameters.pixel_size, 0.001f, true);
				//			input_image_local.ForwardFFT();
				// Whiten noise
				input_image_local.ApplyCurveFilter(&noise_power_spectrum);
				// Apply cosine filter to reduce ringing
				//			input_image_local.CosineMask(std::max(pixel_size / high_resolution_limit, pixel_size / 7.0f + pixel_size / mask_falloff) - pixel_size / (2.0 * mask_falloff), pixel_size / mask_falloff);
				input_image_local.BackwardFFT();
				// Normalize background variance and average
				variance = input_image_local.ReturnVarianceOfRealValues(input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
				average = input_image_local.ReturnAverageOfRealValues(input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, true);

				if (variance == 0.0f) input_image_local.SetToConstant(0.0f);
				else input_image_local.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
			// At this point, input_image should have white background with a variance of 1. The variance should therefore be about 1/binning_factor^2 after binning.
			}
			else input_image_local.ChangePixelSize(&input_image_local, pixel_size / input_parameters.pixel_size, 0.001f);

			// Option to add noise to images to get out of local optima
			//		input_image.AddGaussianNoise(sqrtf(2.0 * input_image.ReturnVarianceOfRealValues()));

			if (ctf_refinement)
			{

				input_image_local.ClipInto(&unbinned_image);
				unbinned_image.ForwardFFT();
				unbinned_image.ClipInto(refine_particle.particle_image);
				// Multiply by binning_factor so variance after binning is close to 1.
				//			refine_particle.particle_image->MultiplyByConstant(binning_factor_refine);
				comparison_object.reference_volume = &input_3d_local;
				comparison_object.projection_image = &projection_image_local;
				comparison_object.particle = &refine_particle;
				refine_particle.MapParameters(cg_starting_point);
				refine_particle.PhaseShiftInverse();

				refine_particle.filter_radius_low = 30.0;
				refine_particle.SetIndexForWeightedCorrelation();
				binned_image.CopyFrom(refine_particle.particle_image);
				refine_particle.InitCTF(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, input_parameters.image_shift_x, input_parameters.image_shift_y);
				//			refine_particle.InitCTF(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift);

				best_score = - std::numeric_limits<float>::max();
				for (defocus_i = - myround(float(defocus_search_range)/float(defocus_step)); defocus_i <= myround(float(defocus_search_range)/float(defocus_step)); defocus_i++)
				{
					refine_particle.SetDefocus(input_parameters.defocus_1 + defocus_i * defocus_step, input_parameters.defocus_2 + defocus_i * defocus_step, input_parameters.defocus_angle, input_parameters.phase_shift);
					refine_particle.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1 + defocus_i * defocus_step, input_parameters.defocus_2 + defocus_i * defocus_step, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, input_parameters.image_shift_x, input_parameters.image_shift_y);
					//					refine_particle.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1 + defocus_i * defocus_step, input_parameters.defocus_2 + defocus_i * defocus_step, input_parameters.defocus_angle, input_parameters.phase_shift);
					if (normalize_input_3d) refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 1);
					// Apply SSNR weighting only to image since input 3D map assumed to be calculated from correctly whitened images
					else refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 0);
					refine_particle.PhaseFlipImage();
					//					refine_particle.CosineMask(false, true, 0.0);
					refine_particle.CosineMask();
					refine_particle.PhaseShift();
					refine_particle.CenterInCorner();
//					refine_particle.WeightBySSNR(input_3d_local.statistics.part_SSNR, 1);

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

				output_parameters.defocus_1 = input_parameters.defocus_1 + best_defocus_i * defocus_step;
				output_parameters.defocus_2 = input_parameters.defocus_2 + best_defocus_i * defocus_step;

				refine_particle.SetDefocus(output_parameters.defocus_1, output_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift);
				refine_particle.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, output_parameters.defocus_1, output_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift);

				refine_particle.filter_radius_low = low_resolution_limit;
				refine_particle.SetIndexForWeightedCorrelation();
				if (normalize_input_3d) refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 1);
				// Apply SSNR weighting only to image since input 3D map assumed to be calculated from correctly whitened images
				else refine_particle.WeightBySSNR(refine_statistics.part_SSNR, 0);
				refine_particle.PhaseFlipImage();
				//			refine_particle.CosineMask(false, true, 0.0);
				refine_particle.CosineMask();
				refine_particle.PhaseShift();
				refine_particle.CenterInCorner();
				input_parameters.score = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
				output_parameters.score = input_parameters.score;
				output_parameters.score_change = 0.0;


				ctf_copy.Init(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, output_parameters.defocus_1, output_parameters.defocus_2 , input_parameters.defocus_angle, 0.0, 0.0, 0.0, pixel_size, input_parameters.phase_shift);
				image_copy.CopyFrom(&input_image_local);
				output_parameters.logp = refine_particle.ReturnLogLikelihood(image_copy, unbinned_image, ctf_copy, input_3d_local, input_statistics, 2.0f * pixel_size);
			}

			if (beamtilt_refinement)
			{
				input_image_local.ForwardFFT();
				ctf_image_local.CalculateCTFImage(input_ctf);
				if (padding != 1.0)
				{
					unbinned_3d_local.CalculateProjection(padded_projection_image_local, ctf_image_local, refine_particle.alignment_parameters, 0.0, 0.0, 1.0, true, true, false, true, false);
					padded_projection_image_local.ClipInto(&phase_difference_image_local);
				}
				else unbinned_3d_local.CalculateProjection(phase_difference_image_local, ctf_image_local, refine_particle.alignment_parameters, 0.0, 0.0, 1.0, true, true, false, true, false);

			//	phase_difference_image_local.QuickAndDirtyWriteSlice("/tmp/proj.mrc", 1, true);
			//	input_image_local.QuickAndDirtyWriteSlice("/tmp/input.mrc", 1, true);
			//	exit(-1);
				phase_difference_image_local.complex_values[0] = 0.0f + I * 0.0f;
				temp_float = sqrtf(phase_difference_image_local.ReturnSumOfSquares() * input_image_local.ReturnSumOfSquares());
				phase_difference_image_local.ConjugateMultiplyPixelWise(input_image_local);
				phase_difference_image_local.CosineMask(0.95f / 2.0f, 2.0f / phase_difference_image_local.logical_x_dimension);
				phase_difference_image_local.DivideByConstant(temp_float);
				phase_difference_sum_local.AddImage(&phase_difference_image_local);
			}

			if (refine_particle.snr > 0.0f) output_parameters.sigma = sqrtf(1.0f / refine_particle.snr);

			output_parameters.ReplaceNanAndInfWithOther(input_parameters);

			input_parameters.image_is_active = 1;
			output_parameters.image_is_active = input_parameters.image_is_active;
			if (output_parameters.score < 0.0f) output_parameters.score = 0.0f;
			// will not work with threading
			//		my_output_par_file.WriteLine(output_parameters);

			if (is_running_locally == true)
			{
				output_star_file.all_parameters[current_line_local] = output_parameters;
			}
			else
			{

			}

			if (is_running_locally == false) // send results back to the gui..
			{
				JobResult *intermediate_result = new JobResult;
				intermediate_result->job_number = my_current_job.job_number;
				float result_line[5];

				result_line[0] = output_parameters.position_in_stack;
				result_line[1] = output_parameters.defocus_1;
				result_line[2] = output_parameters.defocus_2;
				result_line[3] = output_parameters.logp;
				result_line[4] = output_parameters.score;

				intermediate_result->SetResult(5, result_line);
				AddJobToResultQueue(intermediate_result);
			}
			else if (ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
		}

		if (beamtilt_refinement)
		{
			#pragma omp critical
			{
				phase_difference_sum.AddImage(&phase_difference_sum_local);
			}
		}

		if (beamtilt_refinement)
		{
			ctf_image_local.Deallocate();
			padded_projection_image_local.Deallocate();
			phase_difference_image_local.Deallocate();
			phase_difference_sum_local.Deallocate();
		}
		input_image_local.Deallocate();

		if (ctf_refinement)
		{
			unbinned_image.Deallocate();
			binned_image.Deallocate();
			projection_image_local.Deallocate();
		}

		refine_particle.Deallocate();

		} // end omp section
	}

	if (is_running_locally == true)
	{
		delete my_progress;


		if (beamtilt_refinement)
		{
			////phase_difference_sum.QuickAndDirtyWriteSlice("temp_phase_difference_sum.mrc", 1);
			//phase_difference_sum.QuickAndDirtyReadSlice("temp_phase_difference_sum.mrc", 1);
			//	phase_difference_sum.ForwardFFT();
			phase_difference_sum.DivideByConstant(float(images_to_process));

			//phase_difference_sum.DivideByConstant(float(148436));
			phase_difference_sum.CosineMask(0.45f, pixel_size / mask_falloff);



			#pragma omp parallel num_threads(max_threads) default(none) shared(voltage_kV, spherical_aberration_mm, pixel_size, temp_image, beamtilt_image, sum_power, beamtilt_x, beamtilt_y, particle_shift_x, particle_shift_y, phase_multiplier, max_threads, score, phase_difference_sum) private(input_ctf)
			{
				float score_local = FLT_MAX;
				float beamtilt_x_local;
				float beamtilt_y_local;
				float particle_shift_x_local;
				float particle_shift_y_local;

				Image temp_image_local;
				Image beamtilt_image_local;
				Image sum_power_local;
				Image phase_difference_sum_local;

				temp_image_local.CopyFrom(&temp_image);
				beamtilt_image_local.CopyFrom(&beamtilt_image);
				sum_power_local.CopyFrom(&sum_power);
				phase_difference_sum_local.CopyFrom(&phase_difference_sum);

				const int total_number_of_positions = 290880; // hard coded based on FindBeamTilt, not very nice.

				int first_position_to_search;
				int last_position_to_search;
				int number_of_positions_per_thread = int(ceilf(total_number_of_positions / max_threads));

				first_position_to_search = ReturnThreadNumberOfCurrentThread()*number_of_positions_per_thread;
				last_position_to_search = (ReturnThreadNumberOfCurrentThread() + 1) * number_of_positions_per_thread - 1;

				if (first_position_to_search < 0) first_position_to_search = 0;
				if (last_position_to_search > total_number_of_positions - 1) last_position_to_search = total_number_of_positions - 1;

				input_ctf.Init(voltage_kV, spherical_aberration_mm, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, pixel_size, 0.0f);

				// need to work out which section we should be running

				//wxPrintf("first = %i, last = %i\n", first_position_to_search, last_position_to_search);
				score_local = phase_difference_sum.FindBeamTilt(input_ctf, pixel_size, temp_image_local, beamtilt_image_local, sum_power_local, beamtilt_x_local, beamtilt_y_local, particle_shift_x_local, particle_shift_y_local, phase_multiplier, is_running_locally, first_position_to_search, last_position_to_search);

				#pragma omp critical
				{
					if (score_local > score)
					{
						score = score_local;
						beamtilt_x = beamtilt_x_local;
						beamtilt_y = beamtilt_y_local;
						particle_shift_x = particle_shift_x_local;
						particle_shift_y = particle_shift_y_local;
						sum_power.CopyFrom(&sum_power_local);
						beamtilt_image.CopyFrom(&beamtilt_image_local);
						temp_image.CopyFrom(&temp_image_local);
					}
				}

			} // end omp


			temp_image.WriteSlice(&ouput_phase_difference_file,1);

			wxPrintf("Final score = %f\n", score);
			sum_power.WriteSlice(&ouput_difference_file,1);
			beamtilt_image.WriteSlice(&ouput_beamtilt_file,1);

			if (score > 10.0f)
			{
				wxPrintf("\nBeam tilt x,y [mrad]   = %10.4f %10.4f\n", 1000.0f * beamtilt_x, 1000.0f * beamtilt_y);
				wxPrintf("Particle shift x,y [A] = %10.4f %10.4f\n", particle_shift_x, particle_shift_y);

				for (current_line = 0; current_line < output_star_file.all_parameters.GetCount(); current_line++)
				{
					output_star_file.all_parameters.Item(current_line).beam_tilt_x = 1000.0f * beamtilt_x;
					output_star_file.all_parameters.Item(current_line).beam_tilt_y = 1000.0f * beamtilt_y;
					output_star_file.all_parameters.Item(current_line).image_shift_x = particle_shift_x;
					output_star_file.all_parameters.Item(current_line).image_shift_y = particle_shift_y;
				}
			}
			else
			{
				wxPrintf("\nNo detectable beam tilt, set to zero\n");
			}
		}

		for (current_line = 0; current_line < output_star_file.all_parameters.GetCount(); current_line++)
		{
			output_shifts_file.all_parameters[current_line] = output_star_file.all_parameters[current_line];
			output_shifts_file.all_parameters[current_line].Subtract(input_star_file.all_parameters[current_line]);
			output_shifts_file.all_parameters[current_line].position_in_stack = output_star_file.all_parameters[current_line].position_in_stack;
			output_shifts_file.all_parameters[current_line].score_change = 0.0f;
		}

		output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, first_particle, last_particle);
		output_shifts_file.WriteTocisTEMStarFile(output_shift_filename, -1, -1, first_particle, last_particle);

		wxPrintf("\nRefine CTF: Normal termination\n");
		wxDateTime finish_time = wxDateTime::Now();
		wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
	}
	else // we need to send our section of the beam tilt images back to the master..
	{
		long result_array_counter;
		long number_of_result_floats = 6; // first  is x size, second float y size of images, 3rd is number allocated, 4th  images_to_process, voltage, spherical_aberration
		long pixel_counter;

		number_of_result_floats += phase_difference_sum.real_memory_allocated;

		float *result = new float[number_of_result_floats];
		result[0] = phase_difference_sum.logical_x_dimension;
		result[1] = phase_difference_sum.logical_y_dimension;
		result[2] = phase_difference_sum.real_memory_allocated;
		result[3] = images_to_process;
		result[4] = voltage_kV;
		result[5] = spherical_aberration_mm;

		result_array_counter = 6;

		for (pixel_counter = 0; pixel_counter < phase_difference_sum.real_memory_allocated; pixel_counter++)
		{
			result[result_array_counter] = phase_difference_sum.real_values[pixel_counter];
			result_array_counter++;
		}

		SendProgramDefinedResultToMaster(result, number_of_result_floats, job_number_from_gui, expected_number_of_jobs_from_gui);
	}

	return true;
}

void RefineCTFApp::MasterHandleProgramDefinedResult(float *result_array, long array_size, int result_number, int number_of_expected_results) // my_app.cpp deletes the result array memory
{

	wxPrintf("Master, Received result %i (%i of %i)\n", result_number, number_of_received_results  + 1, number_of_expected_results);

	if (sum_for_master.is_in_memory == false)
	{
		sum_for_master.Allocate(result_array[0], result_array[1], 1, false);
		sum_for_master.SetToConstant(0.0f);
		total_results_for_master = 0;
		number_of_received_results = 0;

		voltage_for_master = result_array[4];
		spherical_aberration_for_master = result_array[5];
	}

	// add this result..
	total_results_for_master += result_array[3];

	for (long pixel_counter = 0; pixel_counter < result_array[2]; pixel_counter++)
	{
		sum_for_master.real_values[pixel_counter] += result_array[pixel_counter + 6];
	}

	// check voltage hasn't changed..

	if (voltage_for_master != result_array[4] || spherical_aberration_for_master != result_array[5])
	{
		MyPrintWithDetails("Error: Mismatched voltage (%f / %f) or Cs (%f / %f)\n\n", voltage_for_master, result_array[4], spherical_aberration_for_master, result_array[5]);
	}

	// did this complete a result?

	number_of_received_results++;

	if (number_of_received_results == number_of_expected_results) // we should be done
	{
		// write out the scaled sum to disk for the gui to run a beam tilt estimation on..
		std::string phase_error_filename = current_job_package.jobs[0].arguments[7].ReturnStringArgument();
		float pixel_size = current_job_package.jobs[0].arguments[12].ReturnFloatArgument();

		sum_for_master.DivideByConstant(total_results_for_master);
		sum_for_master.CosineMask(0.45f, pixel_size / 20.0f);
		sum_for_master.QuickAndDirtyWriteSlice(phase_error_filename, 1, true);

/*		wxPrintf("Estimating Beam Tilt...\n");
		std::string phase_error_filename = current_job_package.jobs[0].arguments[7].ReturnStringArgument();
		std::string found_beamtilt_filename = current_job_package.jobs[0].arguments[8].ReturnStringArgument();
		float pixel_size = current_job_package.jobs[0].arguments[12].ReturnFloatArgument();
		CTF input_ctf;
		float score;

		float found_beamtilt_x;
		float found_beamtilt_y;
		float found_particle_shift_x;
		float found_particle_shift_y;

		Image phase_error;
		Image difference_image;
		Image beamtilt_image;

		phase_error.Allocate(sum_for_master.logical_x_dimension, sum_for_master.logical_y_dimension, 1, true);
		difference_image.Allocate(sum_for_master.logical_x_dimension, sum_for_master.logical_y_dimension, 1, true);
		beamtilt_image.Allocate(sum_for_master.logical_x_dimension, sum_for_master.logical_y_dimension, 1, true);

		sum_for_master.DivideByConstant(total_results_for_master);
		sum_for_master.CosineMask(0.45f, pixel_size / 20.0f);

		input_ctf.Init(voltage_for_master, spherical_aberration_for_master, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, pixel_size, 0.0f);
		score = sum_for_master.FindBeamTilt(input_ctf, pixel_size, phase_error, beamtilt_image, difference_image, found_beamtilt_x, found_beamtilt_y, found_particle_shift_x, found_particle_shift_y, 1.0f, true);

		phase_error.QuickAndDirtyWriteSlice(phase_error_filename, 1, true);
		beamtilt_image.QuickAndDirtyWriteSlice(found_beamtilt_filename, 1, true);

		if (score > 10.0f)
		{
			wxPrintf("\nBeam tilt x,y [mrad]   = %10.4f %10.4f\n", 1000.0f * found_beamtilt_x, 1000.0f * found_beamtilt_y);
			wxPrintf("Particle shift x,y [A] = %10.4f %10.4f\n", found_particle_shift_x, found_particle_shift_y);
		}
		else
		{
			wxPrintf("\nNo beam tilt detected, set to zero\n");
		}
*/

	}
}

void RefineCTFApp::SetupNewReference3D(wxString wanted_filename, float inner_mask_radius, float outer_mask_radius, float pixel_size, float mask_falloff, bool threshold_input_3d , float padding, bool beamtilt_refinement, float low_resolution_limit, float high_resolution_limit, float molecular_mass_kDa)
{
	input_file.OpenFile(wanted_filename.ToStdString(), false, true);

	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, "C1");
	input_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	input_3d.mask_radius = outer_mask_radius;
	input_3d.density_map->ReadSlices(&input_file,1,input_file.ReturnZSize());
	// Remove masking here to avoid edge artifacts later
	input_3d.density_map->CosineMask(outer_mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
	if (inner_mask_radius > 0.0) input_3d.density_map->CosineMask(inner_mask_radius / pixel_size, mask_falloff / pixel_size, true);
	if (threshold_input_3d)
	{
		float average_density_max = input_3d.density_map->ReturnAverageOfMaxN(100, outer_mask_radius / pixel_size);
		input_3d.density_map->SetMinimumValue(-0.3 * average_density_max);
	}
	if (padding != 1.0)
	{
		input_3d.density_map->Resize(input_3d.density_map->logical_x_dimension * padding, input_3d.density_map->logical_y_dimension * padding, input_3d.density_map->logical_z_dimension * padding, input_3d.density_map->ReturnAverageOfRealValuesOnEdges());
	}


	if (beamtilt_refinement)
	{
			unbinned_3d = input_3d;
			unbinned_3d.PrepareForProjections(0.0f, 2.0f / 0.95f * pixel_size, false, false);
	}

	input_3d.PrepareForProjections(low_resolution_limit, high_resolution_limit);


}
