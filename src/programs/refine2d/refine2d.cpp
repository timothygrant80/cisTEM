#include "../../core/core_headers.h"

// TODO : Switch to new parameter file format (star) properly and remove hacks
class
Refine2DApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	void DumpArrays();

	int xy_dimensions;
	int number_of_classes;
	int number_of_nonzero_classes;
	int images_processed;
	int *list_of_nozero_classes;
	float sum_logp_total;
	float sum_snr;
	float pixel_size;
	float mask_radius;
	float mask_falloff;
	float log_range;
	float average_snr;
	float *class_logp;
	Image *class_averages;
	Image *CTF_sums;
	wxString dump_file;

	void SendRefineResult(cisTEMParameterLine *current_params);

	private:
};



IMPLEMENT_APP(Refine2DApp)

// override the DoInteractiveUserInput

void Refine2DApp::DoInteractiveUserInput()
{
	wxString	input_particle_images;
	wxString	input_star_filename;
	wxString	input_class_averages;
	wxString	output_star_filename;
	wxString	ouput_class_averages;
	int			number_of_classes = 0;
	int			first_particle = 1;
	int			last_particle = 0;
	float		percent_used = 1.0;
//	float		voltage_kV = 300.0;
//	float		spherical_aberration_mm = 2.7;
//	float		amplitude_contrast = 0.07;
	float		low_resolution_limit = 300.0;
	float		high_resolution_limit = 8.0;
	float		angular_step = 5.0;
	float		max_search_range = 0;
	float		smoothing_factor = 1.0;
	int			padding_factor = 1;
	bool		normalize_particles = true;
	bool		invert_contrast = false;
	bool		exclude_blank_edges = true;
	bool		dump_arrays = false;
	bool 		auto_mask = false;
	bool		auto_centre = false;
	int			max_threads;

	UserInput *my_input = new UserInput("Refine2D", 1.02);

	input_particle_images = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	input_star_filename = my_input->GetFilenameFromUser("Input cisTEM star file", "The input star file, containing your particle alignment parameters", "my_parameters.star", true);
	input_class_averages = my_input->GetFilenameFromUser("Input class averages", "The 2D references representing the current best estimates of the classes", "my_input_classes.mrc", false);
	output_star_filename = my_input->GetFilenameFromUser("Output cisTEM star file", "The output star file, containing your refined particle alignment parameters", "my_refined_parameters.star", false);
	ouput_class_averages = my_input->GetFilenameFromUser("Output class averages", "The refined 2D class averages", "my_refined_classes.mrc", false);
	number_of_classes = my_input->GetIntFromUser("Number of classes (>0 = initialize classes)", "The number of classes that should be refined; 0 = the number is determined by the stack of input averages", "0", 0);
	first_particle = my_input->GetIntFromUser("First particle to refine (0 = first in stack)", "The first particle in the stack that should be refined", "1", 0);
	last_particle = my_input->GetIntFromUser("Last particle to refine (0 = last in stack)", "The last particle in the stack that should be refined", "0", 0);
	percent_used = my_input->GetFloatFromUser("Percent of particles to use (1 = all)", "The percentage of randomly selected particles that will be used for classification", "1.0", 0.0, 1.0);
	pixel_size = my_input->GetFloatFromUser("Pixel size of class averages (A)", "Pixel size of input class averages in Angstroms", "1.0", 0.0);
//	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
//	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
//	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius (A)", "Radius of a circular mask to be applied to the input class averages", "100.0", 0.0);
	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	angular_step = my_input->GetFloatFromUser("Angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
	max_search_range = my_input->GetFloatFromUser("Search range (A) (0.0 = max)", "The maximum global peak search distance along X and Y from the particle box center", "0.0", 0.0);
	smoothing_factor = my_input->GetFloatFromUser("Tuning parameter: smoothing factor", "Factor for likelihood-weighting; values smaller than 1 will blur results more, larger values will emphasize peaks", "1.0", 0.01);
	padding_factor = my_input->GetIntFromUser("Tuning parameter: padding factor for interpol.", "Factor determining how padding is used to improve interpolation for image rotation", "2", 1);
	normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
	invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	exclude_blank_edges = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "Yes");
	auto_mask = my_input->GetYesNoFromUser("Automatically mask class averages", "Should automatic masking be applied to class averages?", "No");
	auto_centre = my_input->GetYesNoFromUser("Automatically center class averages", "Should class averages be centered to their center of mass automatically?", "No");
	dump_arrays = my_input->GetYesNoFromUser("Dump intermediate arrays (merge later)", "Should the intermediate 2D class sums be dumped to a file for later merging with other jobs", "No");
	dump_file = my_input->GetFilenameFromUser("Output dump filename for intermediate arrays", "The name of the dump file with the intermediate 2D class sums", "dump_file.dat", false);

#ifdef _OPENMP
	max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
	max_threads = 1;
#endif

	delete my_input;

	int current_class = 0;
//	my_current_job.Reset(25);
	my_current_job.ManualSetArguments("tttttiiiffffffffibbbbtbbi",	input_particle_images.ToUTF8().data(),
																	input_star_filename.ToUTF8().data(),
																	input_class_averages.ToUTF8().data(),
																	output_star_filename.ToUTF8().data(),
																	ouput_class_averages.ToUTF8().data(),
																	number_of_classes, first_particle, last_particle, percent_used,
																	pixel_size, mask_radius, low_resolution_limit, high_resolution_limit,
																	angular_step, max_search_range, smoothing_factor,
																	padding_factor, normalize_particles, invert_contrast,
																	exclude_blank_edges, dump_arrays, dump_file.ToUTF8().data(),
																	auto_mask, auto_centre, max_threads);
}

// override the do calculation method which will be what is actually run..

bool Refine2DApp::DoCalculation()
{
	Particle input_particle;
	Particle input_particle_local;

	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_star_filename 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_class_averages				= my_current_job.arguments[2].ReturnStringArgument();
	wxString output_star_filename				= my_current_job.arguments[3].ReturnStringArgument();
	wxString ouput_class_averages 				= my_current_job.arguments[4].ReturnStringArgument();
	number_of_classes							= my_current_job.arguments[5].ReturnIntegerArgument();
	int		 first_particle						= my_current_job.arguments[6].ReturnIntegerArgument();
	int		 last_particle						= my_current_job.arguments[7].ReturnIntegerArgument();
	float	 percent_used						= my_current_job.arguments[8].ReturnFloatArgument();
	pixel_size									= my_current_job.arguments[9].ReturnFloatArgument();
//	float    voltage_kV							= my_current_job.arguments[10].ReturnFloatArgument();
//	float 	 spherical_aberration_mm			= my_current_job.arguments[11].ReturnFloatArgument();
//	float    amplitude_contrast					= my_current_job.arguments[12].ReturnFloatArgument();
	mask_radius									= my_current_job.arguments[10].ReturnFloatArgument();
	float    low_resolution_limit				= my_current_job.arguments[11].ReturnFloatArgument();
	float    high_resolution_limit				= my_current_job.arguments[12].ReturnFloatArgument();
	float	 angular_step						= my_current_job.arguments[13].ReturnFloatArgument();
	float	 max_search_range					= my_current_job.arguments[14].ReturnFloatArgument();
	float	 smoothing_factor					= my_current_job.arguments[15].ReturnFloatArgument();
	int		 padding_factor						= my_current_job.arguments[16].ReturnIntegerArgument();
// Psi, Theta, Phi, ShiftX, ShiftY
	input_particle.parameter_map.psi			= true;
	input_particle.parameter_map.theta			= false;
	input_particle.parameter_map.phi			= false;
	input_particle.parameter_map.x_shift		= true;
	input_particle.parameter_map.y_shift		= true;
	bool	 normalize_particles				= my_current_job.arguments[17].ReturnBoolArgument();
	bool	 invert_contrast					= my_current_job.arguments[18].ReturnBoolArgument();
	bool	 exclude_blank_edges				= my_current_job.arguments[19].ReturnBoolArgument();
	bool	 dump_arrays						= my_current_job.arguments[20].ReturnBoolArgument();
	dump_file	 								= my_current_job.arguments[21].ReturnStringArgument();
	bool auto_mask 								= my_current_job.arguments[22].ReturnBoolArgument();
	bool auto_centre							= my_current_job.arguments[23].ReturnBoolArgument();
	int	 max_threads							= my_current_job.arguments[24].ReturnIntegerArgument();

	input_particle.constraints_used.x_shift = true;		// Constraint for X shifts
	input_particle.constraints_used.y_shift = true;		// Constraint for Y shifts

	Image	input_image, cropped_input_image;
	Image	input_image_local, temp_image_local, sum_power_local;
	Image	sum_power, ctf_input_image, padded_image;
	Image	ctf_input_image_local, cropped_input_image_local;
	Image	best_correlation_map, temp_image;
//	Image	*rotation_cache = NULL;
//	Image	*blurred_images = NULL;
	Image	*input_classes_cache = NULL;
	CTF		input_ctf;
	AnglesAndShifts rotation_angle;
	ProgressBar *my_progress;
	Curve	noise_power_spectrum, number_of_terms;

	int i, j, k;
	int fourier_size;
	int current_class, current_line;
	int current_line_local;
	int number_of_rotations;
	int image_counter, images_to_process, pixel_counter;
	int projection_counter;
	int padded_box_size, cropped_box_size, binned_box_size;
	int best_class;
	int images_processed_local;
//	float input_parameters[17];
//	float output_parameters[17];
//	float parameter_average[17];
//	float parameter_variance[17];
	float binning_factor, binned_pixel_size;
	float temp_float;
	float psi;
//	float psi_max;
	float psi_step;
	float psi_start;
	float average;
	float variance;
	float ssq_X;
	float sum_logp_particle;
	float sum_logp_particle_local;
	float sum_logp_total_local;
	float sum_snr_local;
	float occupancy;
	float filter_constant;
	float mask_radius_for_noise;
	float max_corr, max_logp_particle;
	float percentage;
	float random_shift;
	float low_resolution_contrast = 0.5f;
	int number_of_blank_edges;
	int number_of_blank_edges_local;
	int max_samples = 2000;
	int block_size_max = 100;
	int block_size;
	bool keep_reading;
	int current_block_read_size;
	bool file_read;
	wxDateTime my_time_in;

	cisTEMParameterLine input_parameters;
	cisTEMParameterLine output_parameters;
	cisTEMParameterLine parameter_average;
	cisTEMParameterLine parameter_variance;

//	ZeroFloatArray(input_parameters, 17);
//	ZeroFloatArray(output_parameters, 17);
//	ZeroFloatArray(parameter_average, 17);
//	ZeroFloatArray(parameter_variance, 17);

	if ((is_running_locally && !DoesFileExist(input_star_filename)) || (!is_running_locally && !DoesFileExistWithWait(input_star_filename, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input star file %s not found\n", input_star_filename));
	}
	if ((is_running_locally && !DoesFileExist(input_particle_images)) || (!is_running_locally && !DoesFileExistWithWait(input_particle_images, 90)))
	{
		SendErrorAndCrash(wxString::Format("Error: Input particle stack %s not found\n", input_particle_images));
	}

	cisTEMParameters input_star_file;
	input_star_file.ReadFromcisTEMStarFile(input_star_filename);

	// Read whole parameter file to work out average values and variances

	parameter_average = input_star_file.ReturnParameterAverages();
	parameter_variance = input_star_file.ReturnParameterVariances();

	MRCFile input_stack(input_particle_images.ToStdString(), false);
//	FrealignParameterFile input_par_file(input_parameter_file, OPEN_TO_READ);
//	FrealignParameterFile output_par_file(ouput_parameter_file, OPEN_TO_WRITE);
	MRCFile *input_classes = NULL;
	MRCFile *output_classes = NULL;
	if (! dump_arrays || number_of_classes != 0) output_classes = new MRCFile(ouput_class_averages.ToStdString(), true);

	if (input_stack.ReturnXSize() != input_stack.ReturnYSize())
	{
		SendErrorAndCrash(wxString::Format("Error: Particles are not square\n", input_particle_images));
	}

	if (number_of_classes == 0)
	{
		if (! DoesFileExist(input_class_averages))
		{
			SendErrorAndCrash(wxString::Format("Error: Input class averages %s not found\n", input_class_averages));
		}
		input_classes = new MRCFile (input_class_averages.ToStdString(), false);
		if (input_classes->ReturnXSize() != input_stack.ReturnXSize() || input_classes->ReturnYSize() != input_stack.ReturnYSize() )
		{
			SendErrorAndCrash("Error: Dimension of particles and input classes differ\n");
		}
		number_of_classes = input_classes->ReturnZSize();
	}
	list_of_nozero_classes = new int [number_of_classes];
	int *reverse_list_of_nozero_classes = new int [number_of_classes];

	if (last_particle < first_particle && last_particle != 0)
	{
		SendErrorAndCrash("Error: Number of last particle to refine smaller than number of first particle to refine\n");
	}

	if (last_particle == 0) last_particle = input_stack.ReturnZSize();
	if (first_particle == 0) first_particle = 1;
	if (last_particle > input_stack.ReturnZSize()) last_particle = input_stack.ReturnZSize();

//	input_par_file.ReadFile();

	images_to_process = 0;
	mask_falloff = 20.0;
	log_range = 20.0;
//	image_counter = 0;
	for (current_line = 0; current_line < input_star_file.ReturnNumberofLines(); current_line++)
	{
		if (input_star_file.ReturnPositionInStack(current_line) >= first_particle && input_star_file.ReturnPositionInStack(current_line) <= last_particle) images_to_process++;
	}
//	for (current_line = 1; current_line <= input_par_file.number_of_lines; current_line++)
//	{
//		input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
//		if (input_parameters[7] > 0.0)
//		{
//			for (i = 0; i < 17; i++)
//			{
//					parameter_average[i] += input_parameters[i];
//					parameter_variance[i] += powf(input_parameters[i],2);
//			}
//			image_counter++;
//		}
//		if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle) images_to_process++;
//	}

//	if (image_counter > 0)
//	{
//		for (i = 0; i < 17; i++)
//		{
//			parameter_average[i] /= image_counter;
//			parameter_variance[i] /= image_counter;
//			parameter_variance[i] -= powf(parameter_average[i],2);
//
//			// nasty hack for new file format support in other programs
//
//			if (parameter_variance[i] < 0.001 && i == 1) input_particle.constraints_used.phi = false;
//			if (parameter_variance[i] < 0.001 && i == 2) input_particle.constraints_used.theta = false;
//			if (parameter_variance[i] < 0.001 && i == 3) input_particle.constraints_used.psi = false;
//			if (parameter_variance[i] < 0.001 && i == 4) input_particle.constraints_used.x_shift = false;
//			if (parameter_variance[i] < 0.001 && i == 5) input_particle.constraints_used.y_shift = false;
//		}
//	}
//	else
//	{
//		input_particle.constraints_used.psi 	= false;
//		input_particle.constraints_used.theta 	= false;
//		input_particle.constraints_used.phi 	= false;
//		input_particle.constraints_used.x_shift = false;
//		input_particle.constraints_used.y_shift = false;
//
//		/*for (i = 0; i < 17; i++)
//		{
//			input_particle.constraints_used[i] = false;
//		}*/
//	}

	if (parameter_variance.phi < 0.001) input_particle.constraints_used.phi = false;
	if (parameter_variance.theta < 0.001) input_particle.constraints_used.theta = false;
	if (parameter_variance.psi < 0.001) input_particle.constraints_used.psi = false;
	if (parameter_variance.x_shift < 0.001) input_particle.constraints_used.x_shift = false;
	if (parameter_variance.y_shift < 0.001) input_particle.constraints_used.y_shift = false;

	xy_dimensions = input_stack.ReturnXSize();
	if (parameter_average.sigma < 0.01) parameter_average.sigma = 10.0;
	average_snr = 1.0 / powf(parameter_average.sigma, 2);
// *****
//	average_snr = 0.002;
	wxPrintf("\nShift averages x, y = %g, %g, shift std x, y = %g, %g, average SNR = %g\n", parameter_average.x_shift, parameter_average.y_shift,
			sqrtf(parameter_variance.x_shift), sqrtf(parameter_variance.y_shift), average_snr);

	input_particle.SetParameterStatistics(parameter_average, parameter_variance);
//	input_par_file.Rewind();

	if (max_search_range == 0.0) max_search_range = std::max(input_stack.ReturnXSize(), input_stack.ReturnYSize()) / 2.0 * pixel_size;

	cisTEMParameters output_star_file;

	// allocate memory for output files..

	output_star_file.PreallocateMemoryAndBlank(input_star_file.ReturnNumberofLines());
	output_star_file.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | BEST_2D_CLASS | PSI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | LOGP | SCORE | SCORE_CHANGE | OCCUPANCY | SIGMA | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y);

	my_time_in = wxDateTime::Now();
	output_star_file.AddCommentToHeader("# Refine2D run date and time:              " + my_time_in.FormatISOCombined(' '));
	output_star_file.AddCommentToHeader("# Input particle images:                   " + input_particle_images);
	output_star_file.AddCommentToHeader("# Input cisTEM parameter filename:         " + input_star_filename);
	output_star_file.AddCommentToHeader("# Input class averages:                    " + input_class_averages);
	output_star_file.AddCommentToHeader("# Output cisTEM parameter file:            " + output_star_filename);
	output_star_file.AddCommentToHeader("# Output class averages:                   " + ouput_class_averages);
	output_star_file.AddCommentToHeader("# First particle to refine:                " + wxString::Format("%i", first_particle));
	output_star_file.AddCommentToHeader("# Last particle to refine:                 " + wxString::Format("%i", last_particle));
	output_star_file.AddCommentToHeader("# Percent of particles to use:             " + wxString::Format("%f", percent_used));
	output_star_file.AddCommentToHeader("# Pixel size of class averages (A):        " + wxString::Format("%f", pixel_size));
	output_star_file.AddCommentToHeader("# Mask radius for refinement (A):          " + wxString::Format("%f", mask_radius));
	output_star_file.AddCommentToHeader("# Low resolution limit (A):                " + wxString::Format("%f", low_resolution_limit));
	output_star_file.AddCommentToHeader("# High resolution limit (A):               " + wxString::Format("%f", high_resolution_limit));
	output_star_file.AddCommentToHeader("# Angular step:                            " + wxString::Format("%f", angular_step));
	output_star_file.AddCommentToHeader("# Search range (A):                        " + wxString::Format("%f", max_search_range));
	output_star_file.AddCommentToHeader("# Smoothing factor:                        " + wxString::Format("%f", smoothing_factor));
	output_star_file.AddCommentToHeader("# Padding factor:                          " + wxString::Format("%i", padding_factor));
	output_star_file.AddCommentToHeader("# Normalize particles:                     " + BoolToYesNo(normalize_particles));
	output_star_file.AddCommentToHeader("# Invert particle contrast:                " + BoolToYesNo(invert_contrast));
	output_star_file.AddCommentToHeader("# Exclude images with blank edges:         " + BoolToYesNo(exclude_blank_edges));
	output_star_file.AddCommentToHeader("# Dump intermediate arrays:                " + BoolToYesNo(dump_arrays));
	output_star_file.AddCommentToHeader("# Automatically mask class averages:       " + BoolToYesNo(auto_mask));
	output_star_file.AddCommentToHeader("# Automatically center class averages:     " + BoolToYesNo(auto_centre));
	output_star_file.AddCommentToHeader("# Output dump filename:                    " + dump_file);
	output_star_file.AddCommentToHeader("#");

	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	if (input_classes != NULL)
	{
		number_of_nonzero_classes = 0;
		for (current_class = 0; current_class < number_of_classes; current_class++)
		{
			input_image.ReadSlice(input_classes, current_class + 1);
			if (input_image.ReturnVarianceOfRealValues() == 0.0)
			{
				reverse_list_of_nozero_classes[current_class] = -1;
			}
			else
			{
				list_of_nozero_classes[number_of_nonzero_classes] = current_class;
				reverse_list_of_nozero_classes[current_class] = number_of_nonzero_classes;
				number_of_nonzero_classes++;
			}
		}
		for (current_class = number_of_nonzero_classes; current_class < number_of_classes; current_class++)
		{
			list_of_nozero_classes[current_class] = -1;
		}
	}
	else
	{
		number_of_nonzero_classes = number_of_classes;
		for (current_class = 0; current_class < number_of_classes; current_class++)
		{
			list_of_nozero_classes[current_class] = current_class;
			reverse_list_of_nozero_classes[current_class] = current_class;
		}
	}

	if (high_resolution_limit < 2.0 * pixel_size) high_resolution_limit = 2.0 * pixel_size;

	sum_power.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	temp_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	ctf_input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	cropped_box_size = ReturnClosestFactorizedUpper(myroundint(2.0 * (max_search_range + mask_radius + mask_falloff) / pixel_size), 3, true);
	if (cropped_box_size > input_stack.ReturnXSize()) cropped_box_size = input_stack.ReturnXSize();
	cropped_input_image.Allocate(cropped_box_size, cropped_box_size, true);
	binning_factor = high_resolution_limit / pixel_size / 2.0;
	if (binning_factor < 1.0) binning_factor = 1.0;
	fourier_size = ReturnClosestFactorizedUpper(cropped_box_size / binning_factor, 3, true);
	//fourier_size = ReturnClosestFactorizedUpper((cropped_box_size * 1.6) / binning_factor, 3, true);
	if (fourier_size > cropped_box_size) fourier_size = cropped_box_size;

	//if (fourier_size > cropped_box_size) fourier_size = cropped_box_size;
//	best_correlation_map.Allocate(fourier_size, fourier_size, true);
	binning_factor = float(cropped_box_size) / float(fourier_size);
	binned_pixel_size = pixel_size * binning_factor;
	input_particle.Allocate(fourier_size, fourier_size);
	padded_box_size = int(powf(2.0, float(padding_factor)) + 0.5) * fourier_size;
//	padded_image.Allocate(padded_box_size, padded_box_size, true);

	if (angular_step == 0.0)
	{
		psi_step = rad_2_deg(binned_pixel_size / mask_radius);
	}
	else
	{
		psi_step = angular_step;
	}
	number_of_rotations = int(360.0 / psi_step + 0.5);
	psi_step = 360.0 / number_of_rotations;
	psi_start = psi_step / 2.0 * global_random_number_generator.GetUniformRandom();
//	psi_max = 360.0;

	wxPrintf("\nNumber of classes = %i, nonzero classes = %i, box size = %i, binning factor = %f, new pixel size = %f, resolution limit = %f, angular step size = %f, percent_used = %f\n",
			number_of_classes, number_of_nonzero_classes, fourier_size, binning_factor, binned_pixel_size, binned_pixel_size * 2.0, psi_step, percent_used);

	class_logp = new float [number_of_nonzero_classes];
	float *class_variance_correction = new float [number_of_rotations * number_of_nonzero_classes + 1];
	float *class_variance = new float [number_of_nonzero_classes];
	class_averages = new Image [number_of_nonzero_classes];
	CTF_sums = new Image [number_of_nonzero_classes];
//	blurred_images = new Image [number_of_nonzero_classes];
	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		class_averages[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		CTF_sums[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
//		blurred_images[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		class_averages[i].SetToConstant(0.0);
		CTF_sums[i].SetToConstant(0.0);
		class_logp[i] = - std::numeric_limits<float>::max();
	}
//	float *logp = new float [number_of_nonzero_classes];

//	rotation_cache = new Image [number_of_rotations + 1];
//	for (i = 0; i <= number_of_rotations; i++)
//	{
//		rotation_cache[i].Allocate(fourier_size, fourier_size, false);
//	}

	input_classes_cache = new Image [number_of_nonzero_classes];
	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		input_classes_cache[i].Allocate(fourier_size, fourier_size, true);
	}

	if (input_classes != NULL)
	{
		j = 0;
		Image buffer_image;
		//float original_average_value = -FLT_MAX;
		float average_value;
		float average_of_100_max;
		float threshold_value;
		Image size_image;
		long address;
		Peak center_of_mass;
		Image copy_image;
		Image difference_image;


		float original_average_value = -0.3f;

		for (current_class = 0; current_class < number_of_classes; current_class++)
		{

			input_image.ReadSlice(input_classes, current_class + 1);
			variance = input_image.ReturnVarianceOfRealValues();
			if (variance == 0.0 && input_classes != NULL) continue;

			input_image.MultiplyByConstant(binning_factor);

			if (auto_mask == true)
			{
				copy_image.CopyFrom(&input_image);
				difference_image.CopyFrom(&input_image);
				buffer_image.CopyFrom(&input_image);
				buffer_image.SetMinimumValue(-0.3 * input_image.ReturnMaximumValue());
				buffer_image.ForwardFFT();
				buffer_image.CosineMask(pixel_size / 20.0f, pixel_size / 10.0f);
				buffer_image.BackwardFFT();

				average_value = buffer_image.ReturnAverageOfRealValues(mask_radius / pixel_size, true);
				average_of_100_max = buffer_image.ReturnAverageOfMaxN(500, mask_radius / pixel_size);
				threshold_value = average_value + ((average_of_100_max - average_value) * 0.02);

				buffer_image.CosineMask(mask_radius / pixel_size, 1.0, false, true, -FLT_MAX);
				buffer_image.Binarise(threshold_value);

				if (buffer_image.IsConstant() == false)
				{
					rle3d my_rle3d(buffer_image);
					size_image.Allocate(buffer_image.logical_x_dimension, buffer_image.logical_y_dimension, buffer_image.logical_z_dimension, true);
					my_rle3d.ConnectedSizeDecodeTo(size_image);

					float size_of_biggest_thing = size_image.ReturnMaximumValue();
					size_image.Binarise(size_of_biggest_thing - 1.0f);
					size_image.DilateBinarizedMask(5.0f / pixel_size);
	//				if (first_particle == 1) size_image.QuickAndDirtyWriteSlice("/tmp/size_ori.mrc", current_class + 1);

					for (address = 0; address < input_image.real_memory_allocated; address++)
					{
						if (size_image.real_values[address] == 0.0f) buffer_image.real_values[address] = 1.0f;
						else buffer_image.real_values[address] = 0.0f;
					}

					// look for holes so we can fill them in
					if (buffer_image.IsConstant() == false)
					{
						rle3d my_rle3d2(buffer_image);
						my_rle3d2.ConnectedSizeDecodeTo(buffer_image);

						for (address = 0; address < input_image.real_memory_allocated; address++)
						{
							if (buffer_image.real_values[address] <= size_of_biggest_thing) size_image.real_values[address] = 1.0f;
							else size_image.real_values[address] = 0.0f;

						}
					}

					size_image.CosineMask(mask_radius / pixel_size, 1.0, false, true, 0.0f);


	//				if (first_particle == 1) size_image.QuickAndDirtyWriteSlice("/tmp/size.mrc", current_class + 1);
					input_image.SetMinimumValue(original_average_value);

					for (address = 0; address < input_image.real_memory_allocated; address++)
					{
						if (size_image.real_values[address] <= 0.0f) input_image.real_values[address] = original_average_value;
						//else
						//input_image.real_values[address] += original_average_value;

						//input_image.real_values[address] -= original_average_value;
					}

					input_image.CosineMask(mask_radius / pixel_size, 1.0, false, true, original_average_value);
	//				if (first_particle == 1) input_image.QuickAndDirtyWriteSlice("/tmp/pure_masked", current_class + 1);
					difference_image.CopyFrom(&input_image);
					difference_image.SubtractImage(&copy_image);

	//				if (first_particle == 1) difference_image.QuickAndDirtyWriteSlice("/tmp/difference.mrc", current_class + 1);
					difference_image.MultiplyByConstant(0.5f);
					copy_image.AddImage(&difference_image);
					input_image.CopyFrom(&copy_image);
				}
				else
				{
					input_image.SetMinimumValue(-0.3 * input_image.ReturnMaximumValue());
					input_image.CosineMask(mask_radius / pixel_size, 1.0, false, true, 0.0f);
				}
			}
			else
			{
				input_image.SetMinimumValue(-0.3 * input_image.ReturnMaximumValue());
				input_image.CosineMask(mask_radius / pixel_size, 1.0, false, true, 0.0f);
			}

			if (auto_centre == true)
			{
//				input_image.CenterOfMass();
				center_of_mass = input_image.CenterOfMass(0.0f, true);
				input_image.RealSpaceIntegerShift(myroundint(center_of_mass.x), myroundint(center_of_mass.y), 0);
			}


			input_image.ClipInto(&cropped_input_image);
//			if (first_particle == 1) input_image.QuickAndDirtyWriteSlice("/tmp/masked.mrc", current_class + 1);
			cropped_input_image.ForwardFFT();
			cropped_input_image.ClipInto(&input_classes_cache[j]);
			j++;
		}
	}
	else
	{
		wxPrintf("\nGenerating %i starting class averages...\n\n", number_of_classes);
		my_progress = new ProgressBar(images_to_process);
		number_of_blank_edges = 0;
		image_counter = 0;
		random_shift = mask_radius / pixel_size * 0.2;
		for (current_class = 0; current_class < number_of_classes; current_class++) {list_of_nozero_classes[current_class] = 0;}


		// read in blocks to avoid seeking a lot in large files
		block_size = myroundint(float(images_to_process) / 100.0f);
		if (float(block_size) > float(input_star_file.ReturnNumberofLines()) * percent_used * 0.01f)
		{
			block_size = myroundint(float(input_star_file.ReturnNumberofLines()) * percent_used * 0.01f);
		}
		if (block_size < 1) block_size = 1;

		keep_reading = false;
		current_block_read_size = 0;

		for (current_line = 0; current_line < input_star_file.ReturnNumberofLines(); current_line++)
		{
			input_parameters = input_star_file.ReturnLine(current_line);

			if (input_parameters.position_in_stack < first_particle || input_parameters.position_in_stack > last_particle) continue;

			image_counter++;
			my_progress->Update(image_counter);

			if (keep_reading == false)
			{
				if (global_random_number_generator.GetUniformRandom() < 1.0 - 2.0 * (percent_used / float(block_size)))
				{
					/*if (is_running_locally == false)
					{
						temp_float = current_line;
						JobResult *temp_result = new JobResult;
						temp_result->SetResult(1, &temp_float);
						AddJobToResultQueue(temp_result);
						//wxPrintf("Refine3D : Adding job to job queue..\n");
					} */
					continue;
				}
				else
				{
					keep_reading = true;
					current_block_read_size = 0;
				}
			}

			current_block_read_size++;
			if (current_block_read_size == block_size) keep_reading = false;

			input_image.ReadSlice(&input_stack, input_parameters.position_in_stack);
			input_image.ChangePixelSize(&input_image, pixel_size / input_parameters.pixel_size, 0.001f);
			//if (exclude_blank_edges && input_image.ContainsBlankEdges(mask_radius / pixel_size)) {number_of_blank_edges++; continue;}
			if (normalize_particles)
			{
				variance = input_image.ReturnVarianceOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
				average = input_image.ReturnAverageOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, true);
				if (invert_contrast) input_image.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
				else input_image.AddMultiplyConstant(- average, - 1.0 / sqrtf(variance));
			}
			else
			if (! invert_contrast) input_image.MultiplyByConstant(- 1.0 );
			for (current_class = 0; current_class < number_of_classes; current_class++)
			{
				//if (global_random_number_generator.GetUniformRandom() >= 1.0 - 2.0 * percent_used / number_of_classes)
				if (global_random_number_generator.GetUniformRandom() >= 1.0 - 2.0 / number_of_classes)
				{
					input_image.RealSpaceIntegerShift(myroundint(random_shift * global_random_number_generator.GetUniformRandom()),
							myroundint(random_shift * global_random_number_generator.GetUniformRandom()));
					class_averages[current_class].AddImage(&input_image);
					list_of_nozero_classes[current_class]++;
				}
			}

			if (is_running_locally == false)
			{
				temp_float = current_line;
				JobResult *temp_result = new JobResult;
				temp_result->SetResult(1, &temp_float);
				AddJobToResultQueue(temp_result);
				//wxPrintf("Refine2D : Adding job to job queue..\n");
			}
		}

		for (current_class = 0; current_class < number_of_classes; current_class++)
		{
			if (list_of_nozero_classes[current_class] != 0) class_averages[current_class].MultiplyByConstant(1.0 / list_of_nozero_classes[current_class]);
			class_averages[current_class].WriteSlice(output_classes, current_class + 1);
		}
		delete [] list_of_nozero_classes;
		delete [] reverse_list_of_nozero_classes;
//		delete [] rotation_cache;
		delete [] input_classes_cache;
		delete [] class_averages;
		delete [] CTF_sums;
//		delete [] blurred_images;
		delete [] class_variance_correction;
		delete [] class_variance;
//		delete [] logp;
		delete [] class_logp;
		delete output_classes;
		delete my_progress;
		if (exclude_blank_edges)
		{
			wxPrintf("\nNumber of excluded images with blank edges = %i\n", number_of_blank_edges);
		}
		wxPrintf("\nRefine2D: Normal termination\n\n");
		return true;
	}

	if (normalize_particles)
	{
		wxPrintf("\nCalculating noise power spectrum...\n\n");
		percentage = float(max_samples) / float(images_to_process);
		sum_power.SetToConstant(0.0);
		number_of_blank_edges = 0;
		noise_power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
//		if (is_running_locally == true) my_progress = new ProgressBar(images_to_process);
		if (is_running_locally == true) my_progress = new ProgressBar(images_to_process / max_threads);
//		current_line = 0;

		// read in blocks to avoid seeking a lot in large files
		block_size = block_size_max;
		if (float(block_size) > float(images_to_process) * percentage * 0.1f)
		{
			block_size = myroundint(float(images_to_process) * percentage * 0.1f);
		}
		if (block_size < 1) block_size = 1;

//		image_counter = 0;

		#pragma omp parallel num_threads(max_threads) default(none) shared(input_star_file, first_particle, last_particle, my_progress, percentage, exclude_blank_edges, input_stack, \
			number_of_blank_edges, sum_power, current_line, global_random_number_generator, block_size) \
		private(current_line_local, input_parameters, number_of_blank_edges_local, variance, temp_image_local, sum_power_local, input_image_local, temp_float, file_read, \
			mask_radius_for_noise, image_counter, keep_reading, current_block_read_size)
		{

		image_counter = 0;
		number_of_blank_edges_local = 0;
		keep_reading = false;
		current_block_read_size = 0;
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
				input_parameters = input_star_file.ReturnLine(current_line_local);

//				current_line++;
//				if (input_star_file.ReturnPositionInStack(current_line) < first_particle || input_star_file.ReturnPositionInStack(current_line) > last_particle) continue;
//				image_counter++;
				if (input_parameters.position_in_stack >= first_particle && input_parameters.position_in_stack <= last_particle)
				{
					image_counter++;
					if (is_running_locally == true && ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
//					if (is_running_locally == true) my_progress->Update(image_counter);
					file_read = false;
					if (keep_reading == false)
					{
						if ((global_random_number_generator.GetUniformRandom() >= 1.0 - 2.0 * (percentage / float(block_size))))
						{
							keep_reading = true;
							current_block_read_size = 0;
						}
					}

					current_block_read_size++;
					if (current_block_read_size == block_size) keep_reading = false;

					input_image_local.ReadSlice(&input_stack, input_parameters.position_in_stack);
					file_read = true;
				}
			}

			if (input_parameters.position_in_stack < first_particle || input_parameters.position_in_stack > last_particle) continue;
			if (! file_read) continue;
			mask_radius_for_noise = mask_radius / input_parameters.pixel_size;
			if (2.0 * mask_radius_for_noise + mask_falloff / input_parameters.pixel_size > 0.95 * input_image_local.logical_x_dimension)
			{
				mask_radius_for_noise = 0.95 * input_image_local.logical_x_dimension / 2.0 - mask_falloff / 2.0 / input_parameters.pixel_size;
			}
			if (exclude_blank_edges && input_image_local.ContainsBlankEdges(mask_radius_for_noise)) {number_of_blank_edges_local++; continue;}
//			if (input_image_local.logical_x_dimension != 128 || input_image_local.logical_y_dimension != 128 || pixel_size != 3.32 || input_parameters.pixel_size != 3.32) \
//			wxPrintf("input_image_local.logical_x_dimension, input_image_local.logical_x_dimension, pixel_size, input_parameters.pixel_size = %i %i %g %g\n", input_image_local.logical_x_dimension, input_image_local.logical_x_dimension, pixel_size, input_parameters.pixel_size);
			input_image_local.ChangePixelSize(&input_image_local, pixel_size / input_parameters.pixel_size, 0.001f);
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
//		input_par_file.Rewind();
		sum_power.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);
		noise_power_spectrum.SquareRoot();
		noise_power_spectrum.Reciprocal();

		if (exclude_blank_edges)
		{
			wxPrintf("\nNumber of excluded images with blank edges = %i\n", number_of_blank_edges);
		}
	}

	wxPrintf("\nCalculating new class averages...\n\n");
	number_of_blank_edges = 0;
	images_processed = 0;
	sum_logp_total = - std::numeric_limits<float>::max();
	sum_snr = 0.0f;
	sum_logp_particle = - std::numeric_limits<float>::max();

	if (is_running_locally == true) my_progress = new ProgressBar(images_to_process / max_threads);

	#pragma omp parallel num_threads(max_threads) default(none) shared(input_star_file, first_particle, last_particle, my_progress, percentage, exclude_blank_edges, input_stack, \
		number_of_blank_edges, global_random_number_generator, percent_used, cropped_box_size, low_resolution_limit, high_resolution_limit, binned_pixel_size, invert_contrast, \
		noise_power_spectrum, padded_box_size, psi_step, psi_start, number_of_rotations, reverse_list_of_nozero_classes, smoothing_factor, max_search_range, output_star_file, \
		fourier_size, input_particle, binning_factor, normalize_particles, low_resolution_contrast, input_classes_cache, sum_logp_particle) \
	private(current_line_local, input_parameters, image_counter, number_of_blank_edges_local, variance, temp_image_local, sum_power_local, input_image_local, temp_float, file_read, \
		output_parameters, input_ctf, average, ctf_input_image_local, cropped_input_image_local, psi, i, rotation_angle, current_class, best_class, ssq_X, best_correlation_map, \
		images_processed_local, padded_image, input_particle_local, sum_logp_particle_local, max_logp_particle, sum_logp_total_local, sum_snr_local)
	{

	image_counter = 0;
	number_of_blank_edges_local = 0;
	images_processed_local = 0;
	sum_snr_local = 0.0f;
	sum_logp_total_local = - std::numeric_limits<float>::max();
	input_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	temp_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	ctf_input_image_local.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	cropped_input_image_local.Allocate(cropped_box_size, cropped_box_size, true);
	padded_image.Allocate(padded_box_size, padded_box_size, true);
	best_correlation_map.Allocate(fourier_size, fourier_size, true);

	input_particle_local.CopyAllButImages(&input_particle);
	input_particle_local.Allocate(fourier_size, fourier_size);

	Image	*rotation_cache = NULL;
	rotation_cache = new Image [number_of_rotations + 1];
	for (i = 0; i <= number_of_rotations; i++)
	{
		rotation_cache[i].Allocate(fourier_size, fourier_size, false);
	}
	float *logp = new float [number_of_nonzero_classes];

	Image	*blurred_images = NULL;
	blurred_images = new Image [number_of_nonzero_classes];
	Image	*class_averages_local = NULL;
	class_averages_local = new Image [number_of_nonzero_classes];
	Image	*CTF_sums_local = NULL;
	CTF_sums_local = new Image [number_of_nonzero_classes];
	float *class_logp_local;
	class_logp_local = new float [number_of_nonzero_classes];

	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		class_averages_local[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		CTF_sums_local[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
		blurred_images[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		class_averages_local[i].SetToConstant(0.0);
		CTF_sums_local[i].SetToConstant(0.0);
		class_logp_local[i] = - std::numeric_limits<float>::max();
	}

	#pragma omp for schedule(dynamic,1)
	for (current_line_local = 0; current_line_local < input_star_file.ReturnNumberofLines(); current_line_local++)
	{
		input_parameters = input_star_file.ReturnLine(current_line_local);
		if (input_parameters.position_in_stack < first_particle || input_parameters.position_in_stack > last_particle) continue;
		image_counter++;
		if ((global_random_number_generator.GetUniformRandom() < 1.0f - 2.0f * percent_used))
		{
			input_parameters.best_2d_class = - abs(input_parameters.best_2d_class);
			input_parameters.score_change = 0.0f;
			output_parameters = input_parameters;
//			output_par_file.WriteLine(input_parameters);
			SendRefineResult(&input_parameters);
			output_star_file.all_parameters[current_line_local] = input_parameters;
			if (is_running_locally == true && ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
			continue;
		}

		input_image_local.ReadSlice(&input_stack, input_parameters.position_in_stack);

		if (exclude_blank_edges && input_image_local.ContainsBlankEdges(mask_radius / input_parameters.pixel_size))
		{
			number_of_blank_edges_local++;
			input_parameters.best_2d_class = - abs(input_parameters.best_2d_class);
			input_parameters.score_change = 0.0;
			output_parameters = input_parameters;
			SendRefineResult(&input_parameters);
			output_star_file.all_parameters[current_line_local] = input_parameters;
			if (is_running_locally == true && ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
			continue;
		}
		else
		{
			input_parameters.best_2d_class = fabsf(input_parameters.best_2d_class);
		}
		input_image_local.ChangePixelSize(&input_image_local, pixel_size / input_parameters.pixel_size, 0.001f);

		output_parameters = input_parameters;

// Set up Particle object
		input_particle_local.ResetImageFlags();
		input_particle_local.mask_radius = mask_radius;
		input_particle_local.mask_falloff = mask_falloff;
		input_particle_local.filter_radius_low = low_resolution_limit;
		input_particle_local.filter_radius_high = high_resolution_limit;
		// The following line would allow using particles with different pixel sizes
		input_particle_local.pixel_size = binned_pixel_size;
//		input_particle_local.snr = average_snr * powf(binning_factor, 2);

		input_particle_local.SetParameters(input_parameters);
		input_particle_local.SetParameterConstraints(1.0 / average_snr / powf(binning_factor, 2));

		input_image_local.ReplaceOutliersWithMean(5.0);
		if (invert_contrast) input_image_local.InvertRealValues();
		input_particle_local.InitCTFImage(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, std::min(input_parameters.amplitude_contrast, 0.001f), input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, 0.0f, 0.0f);
		input_particle_local.SetLowResolutionContrast(low_resolution_contrast);
		input_ctf.Init(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, std::min(input_parameters.amplitude_contrast, 0.001f), input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, 0.0, 0.0, 0.0, pixel_size, input_parameters.phase_shift, input_parameters.beam_tilt_x / 1000.0f, input_parameters.beam_tilt_y / 1000.0f, 0.0f, 0.0f);
		input_ctf.SetLowResolutionContrast(low_resolution_contrast);
		ctf_input_image_local.CalculateCTFImage(input_ctf);

		if (normalize_particles)
		{
			input_image_local.ForwardFFT();
			// Whiten noise
			input_image_local.ApplyCurveFilter(&noise_power_spectrum);
			// Apply cosine filter to reduce ringing
			input_image_local.CosineMask(std::max(pixel_size / high_resolution_limit, pixel_size / 7.0f + pixel_size / mask_falloff) - pixel_size / (2.0 * mask_falloff), pixel_size / mask_falloff);
			input_image_local.BackwardFFT();
			// Normalize background variance and average
			variance = input_image_local.ReturnVarianceOfRealValues(input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
			average = input_image_local.ReturnAverageOfRealValues(input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, true);
			input_image_local.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
			// At this point, input_image should have white background with a variance of 1. The variance should therefore be about 1/binning_factor^2 after binning.
		}
		// Multiply by binning_factor so variance after binning is close to 1.
		input_image_local.MultiplyByConstant(binning_factor);
		// Determine sum of squares of corrected image and binned image for ML calculation
		temp_image_local.CopyFrom(&input_image_local);
//		temp_image_local.CosineMask(temp_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size);
		temp_image_local.ClipInto(&cropped_input_image_local);
		cropped_input_image_local.CosineMask(cropped_input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
		cropped_input_image_local.ForwardFFT();
		cropped_input_image_local.ClipInto(input_particle_local.particle_image);
		ssq_X = input_particle_local.particle_image->ReturnSumOfSquares();
		input_particle_local.CTFMultiplyImage();
		variance = input_particle_local.particle_image->ReturnSumOfSquares();
		// Apply CTF
		input_image_local.ForwardFFT();
		input_image_local.MultiplyPixelWiseReal(ctf_input_image_local);
		input_image_local.BackwardFFT();
		// Calculate rotated versions of input image
//		input_particle_local.mask_volume = input_image_local.CosineMask(input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size);
		input_image_local.ClipInto(&cropped_input_image_local);
		input_particle_local.mask_volume = cropped_input_image_local.CosineMask(cropped_input_image_local.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
//		input_particle_local.mask_volume = cropped_input_image_local.number_of_real_space_pixels;
		cropped_input_image_local.ForwardFFT();
		// This ClipInto is needed as a filtering step to set Fourier terms outside the final binned image to zero.
		cropped_input_image_local.ClipInto(input_particle_local.particle_image);
		input_particle_local.particle_image->ClipInto(&padded_image);
		// Pre-correct for real-space interpolation errors
		// Maybe this is not a good idea since this amplifies noise...
//		padded_image.CorrectSinc();
		padded_image.BackwardFFT();
//???		input_particle_local.mask_volume = padded_image.CosineMask(padded_image.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size);

		for (i = 0; i < number_of_rotations; i++)
		{
			psi = i * psi_step + psi_start;
			rotation_angle.GenerateRotationMatrix2D(psi);
			padded_image.Rotate2DSample(rotation_cache[i], rotation_angle);
			// Correct variance for interpolation.
			temp_float = rotation_cache[i].ReturnSumOfSquares();
			rotation_cache[i].MultiplyByConstant(sqrtf(variance / temp_float));
			rotation_cache[i].ForwardFFT();
// *******
//			rotation_cache[i].SwapRealSpaceQuadrants();
		}
		input_particle_local.mask_volume = rotation_cache[0].number_of_real_space_pixels * input_particle_local.mask_volume / cropped_input_image_local.number_of_real_space_pixels;
//		wxPrintf("rot = %li, in = %li, mask = %g\n", rotation_cache[0].number_of_real_space_pixels, cropped_input_image_local.number_of_real_space_pixels, input_particle_local.mask_volume);
		input_particle_local.is_masked = true;

		sum_logp_particle_local = - std::numeric_limits<float>::max();
		max_logp_particle = - std::numeric_limits<float>::max();

		// Calculate score of previous best match, store cc map in best_correlation_map. Matching projection is in last location of projection_cache.
		best_class = abs(input_parameters.best_2d_class) - 1;
		if (best_class < 0 || best_class >= number_of_classes)
		{
			best_class = -1;
		}
		else
		{
			best_class = reverse_list_of_nozero_classes[best_class];
		}
// ********
//		best_class = -1;
		if (best_class >= 0)
		{
			rotation_angle.GenerateRotationMatrix2D(- input_parameters.psi);
			padded_image.Rotate2DSample(rotation_cache[number_of_rotations], rotation_angle);
			temp_float = rotation_cache[number_of_rotations].ReturnSumOfSquares();
			rotation_cache[number_of_rotations].MultiplyByConstant(sqrtf(variance / temp_float));
//			rotation_cache[number_of_rotations].MultiplyByConstant(-1.0);
//			rotation_cache[number_of_rotations].QuickAndDirtyWriteSlice("rot.mrc", 1);
//			wxPrintf("best class = %i angle = %g\n", best_class, input_parameters[3]);
//			exit(0);
			rotation_cache[number_of_rotations].ForwardFFT();
			temp_float = input_particle_local.MLBlur(input_classes_cache, ssq_X, cropped_input_image_local, rotation_cache,
					blurred_images[0], best_class, number_of_rotations, psi_step, psi_start, smoothing_factor,
					max_logp_particle, best_class, input_parameters.psi, best_correlation_map, true, true, true, NULL, NULL, max_search_range);
		}

		for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
		{
			//wxPrintf("Working on class %i\n", current_class);
			logp[current_class] = input_particle_local.MLBlur(input_classes_cache, ssq_X, cropped_input_image_local, rotation_cache,
					blurred_images[current_class], current_class, number_of_rotations, psi_step, psi_start, smoothing_factor,
					max_logp_particle, best_class, input_parameters.psi, best_correlation_map, false, true, true, NULL, NULL, max_search_range);
			// Sum likelihoods of all classes
			sum_logp_particle_local = ReturnSumOfLogP(sum_logp_particle_local, logp[current_class], log_range);
		}
		sum_logp_total_local = ReturnSumOfLogP(sum_logp_total_local, sum_logp_particle_local, log_range);

		if (input_particle_local.current_parameters.sigma != 0.0) sum_snr_local += 1.0 / input_particle_local.current_parameters.sigma;
		for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
		{
			// Divide class likelihoods by summed likelihood; this is a simple subtraction of the log values
			logp[current_class] -= sum_logp_particle_local;
			if (logp[current_class] >= - log_range)
			{
				// Sum the class likelihoods (now divided by the summed likelihood)
				class_logp_local[current_class] = ReturnSumOfLogP(class_logp_local[current_class], logp[current_class], log_range);
				// Apply likelihood weight to blurred image
				if (input_particle_local.current_parameters.sigma > 0.0)
				{
					temp_float = expf(logp[current_class]);
					// Need to divide here by sigma^2; already divided once on input, therefore divide only by sigma here.
					blurred_images[current_class].MultiplyByConstant(temp_float / input_particle_local.current_parameters.sigma);
					// Add weighted image to class average
					class_averages_local[current_class].AddImage(&blurred_images[current_class]);
					// Copy and multiply CTF image
					temp_image_local.CopyFrom(&ctf_input_image_local);
					temp_image_local.MultiplyPixelWiseReal(ctf_input_image_local);
					temp_image_local.MultiplyByConstant(temp_float / input_particle_local.current_parameters.sigma);
					CTF_sums_local[current_class].AddImage(&temp_image_local);
					if (current_class + 1 == input_particle_local.current_parameters.best_2d_class) input_particle_local.current_parameters.occupancy = 100.0 * temp_float;
				}
			}
		}

		input_particle_local.GetParameters(output_parameters);

		output_parameters.best_2d_class = list_of_nozero_classes[output_parameters.best_2d_class - 1] + 1;
		// Multiply measured sigma noise in binned image by binning factor to obtain sigma noise of unbinned image
		output_parameters.sigma *= binning_factor;
		if (output_parameters.sigma > 100.0) output_parameters.sigma = 100.0;
		output_parameters.score_change = output_parameters.score - input_parameters.score;
		SendRefineResult(&output_parameters);
		output_star_file.all_parameters[current_line_local] = output_parameters;

		images_processed_local++;

		if (is_running_locally == true && ReturnThreadNumberOfCurrentThread() == 0) my_progress->Update(image_counter);
	}

	#pragma omp critical
	{
		number_of_blank_edges += number_of_blank_edges_local;
		images_processed += images_processed_local;
		sum_snr += sum_snr_local;
		sum_logp_particle = ReturnSumOfLogP(sum_logp_particle, sum_logp_particle_local, log_range);
		sum_logp_total = ReturnSumOfLogP(sum_logp_total, sum_logp_total_local, log_range);
		for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
		{
			class_logp[current_class] = ReturnSumOfLogP(class_logp[current_class], class_logp_local[current_class], log_range);
			class_averages[current_class].AddImage(&class_averages_local[current_class]);
			CTF_sums[current_class].AddImage(&CTF_sums_local[current_class]);
		}

	}

	input_image_local.Deallocate();
	temp_image_local.Deallocate();
	ctf_input_image_local.Deallocate();
	cropped_input_image_local.Deallocate();
	padded_image.Deallocate();
	best_correlation_map.Deallocate();
	input_particle_local.Deallocate();
	delete [] rotation_cache;
	delete [] logp;
	delete [] blurred_images;
	delete [] class_averages_local;
	delete [] CTF_sums_local;
	delete [] class_logp_local;

	} // end omp section

	if (is_running_locally == true) delete my_progress;

	if (exclude_blank_edges && ! normalize_particles)
	{
		wxPrintf("\nNumber of excluded images with blank edges = %i\n", number_of_blank_edges);
	}

	for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
	{
		class_averages[current_class].ForwardFFT();
//		class_averages[current_class].CosineMask(0.45 / binning_factor, pixel_size / mask_falloff);
//		class_averages[current_class].CosineMask(0.45 / binning_factor, 0.1 / binning_factor);
	}

	output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, first_particle, last_particle);

	if (dump_arrays)
	{
		wxPrintf("\nDumping intermediate files...\n");
		DumpArrays();
	}
	else
	{
		image_counter = 0;
		temp_image.SetToConstant(0.0);
		wxPrintf("\n");
		for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
		{
			if (fabsf(class_logp[current_class]) <= log_range)
			{
				// Divide summed class likelihood by number of images
				occupancy = class_logp[current_class] - logf(images_processed);
				if (occupancy >= - log_range)
				{
					occupancy = expf(occupancy);
				}
				else
				{
					occupancy = 0.0;
				}
				if (occupancy > 0.0)
				{
					filter_constant = occupancy * sum_snr / images_processed;
					for (i = 0; i < class_averages[current_class].real_memory_allocated / 2; i++)
					{
						class_averages[current_class].complex_values[i] /= (abs(CTF_sums[current_class].complex_values[i]) + occupancy);
					}
					class_averages[current_class].BackwardFFT();
				}
				variance = class_averages[current_class].ReturnVarianceOfRealValues();
//				wxPrintf("images_processed = %i, occupancy = %g, variance = %g\n", images_processed, occupancy, variance);
			}
			else
			{
				occupancy = 0.0;
			}

			while (image_counter < list_of_nozero_classes[current_class])
			{
				temp_image.WriteSlice(output_classes, image_counter + 1);
				wxPrintf("Class = %4i, average occupancy = %10.4f\n", image_counter + 1, 0.0);
				image_counter++;
			}
			class_averages[current_class].WriteSlice(output_classes, image_counter + 1);
			wxPrintf("Class = %4i, average occupancy = %10.4f\n", image_counter + 1, 100.0 * occupancy);
			image_counter++;
		}
		while (image_counter < number_of_classes)
		{
			temp_image.WriteSlice(output_classes, image_counter + 1);
			image_counter++;
		}
	}

	wxPrintf("\nTotal logP = %g\n", sum_logp_total);

	delete [] list_of_nozero_classes;
	delete [] reverse_list_of_nozero_classes;
//	delete [] rotation_cache;
	delete [] input_classes_cache;
	delete [] class_averages;
	delete [] CTF_sums;
//	delete [] blurred_images;
	delete [] class_variance_correction;
	delete [] class_variance;
//	delete [] logp;
	delete [] class_logp;
	if (input_classes != NULL) delete input_classes;
	if (output_classes != NULL) delete output_classes;

	wxPrintf("\nRefine2D: Normal termination\n\n");

	return true;
}

void Refine2DApp::DumpArrays()
{
	int i;
	int count = 0;
	char temp_char[4 * sizeof(int) + 6 * sizeof(float)];
	char *char_pointer;

	std::ofstream b_stream(dump_file.c_str(), std::fstream::out | std::fstream::binary);

	char_pointer = (char *) &xy_dimensions;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &number_of_classes;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &number_of_nonzero_classes;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &images_processed;
	for (i = 0; i < sizeof(int); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &sum_logp_total;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &sum_snr;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &pixel_size;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &mask_radius;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &mask_falloff;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	char_pointer = (char *) &log_range;
	for (i = 0; i < sizeof(float); i++) {temp_char[count] = char_pointer[i]; count++;};
	b_stream.write(temp_char, count);

	char_pointer = (char *) list_of_nozero_classes;
	b_stream.write(char_pointer, sizeof(int) * number_of_classes);
	char_pointer = (char *) class_logp;
	b_stream.write(char_pointer, sizeof(float) * number_of_nonzero_classes);

	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		char_pointer = (char *) class_averages[i].real_values;
		b_stream.write(char_pointer, sizeof(float) * class_averages[i].real_memory_allocated);
		char_pointer = (char *) CTF_sums[i].real_values;
		b_stream.write(char_pointer, sizeof(float) * CTF_sums[i].real_memory_allocated);
	}

	b_stream.close();
}

void Refine2DApp::SendRefineResult(cisTEMParameterLine *current_params)
{
	if (is_running_locally == false) // send results back to the gui..
	{
		float gui_result_params[19];

		JobResult *intermediate_result = new JobResult;
		intermediate_result->job_number = my_current_job.job_number;

		gui_result_params[0] = current_params->position_in_stack;
		gui_result_params[1] = current_params->psi;
		gui_result_params[2] = current_params->x_shift;
		gui_result_params[3] = current_params->y_shift;
		gui_result_params[4] = current_params->best_2d_class;
		wxPrintf("best class = %i\n", current_params->best_2d_class);
		gui_result_params[5] = current_params->sigma;
		gui_result_params[6] = current_params->logp;
		gui_result_params[7] = current_params->amplitude_contrast;
		gui_result_params[8] = current_params->pixel_size;
		gui_result_params[9] = current_params->microscope_voltage_kv;
		gui_result_params[10] = current_params->microscope_spherical_aberration_mm;
		gui_result_params[11] = current_params->beam_tilt_x;
		gui_result_params[12] = current_params->beam_tilt_y;
		gui_result_params[13] = current_params->image_shift_x;
		gui_result_params[14] = current_params->image_shift_y;
		gui_result_params[15] = current_params->defocus_1;
		gui_result_params[16] = current_params->defocus_2;
		gui_result_params[17] = current_params->defocus_angle;
		gui_result_params[18] = current_params->phase_shift;



		intermediate_result->SetResult(19, gui_result_params);
		AddJobToResultQueue(intermediate_result);
	}
}
