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

	void SendRefineResult(float *current_params);

	private:
};



IMPLEMENT_APP(Refine2DApp)

// override the DoInteractiveUserInput

void Refine2DApp::DoInteractiveUserInput()
{
	wxString	input_particle_images;
	wxString	input_parameter_file;
	wxString	input_class_averages;
	wxString	ouput_parameter_file;
	wxString	ouput_class_averages;
	int			number_of_classes = 0;
	int			first_particle = 1;
	int			last_particle = 0;
	float		percent_used = 1.0;
	float		voltage_kV = 300.0;
	float		spherical_aberration_mm = 2.7;
	float		amplitude_contrast = 0.07;
	float		low_resolution_limit = 300.0;
	float		high_resolution_limit = 8.0;
	float		angular_step = 5.0;
	float		max_search_x = 0;
	float		max_search_y = 0;
	float		smoothing_factor = 1.0;
	int			padding_factor = 1;
	bool		normalize_particles = true;
	bool		invert_contrast = false;
	bool		exclude_blank_edges = true;
	bool		dump_arrays = false;

	UserInput *my_input = new UserInput("Refine2D", 1.02);

	input_particle_images = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	input_class_averages = my_input->GetFilenameFromUser("Input class averages", "The 2D references representing the current best estimates of the classes", "my_input_classes.mrc", false);
	ouput_parameter_file = my_input->GetFilenameFromUser("Output parameter file", "The output parameter file, containing your refined particle alignment parameters", "my_refined_parameters.par", false);
	ouput_class_averages = my_input->GetFilenameFromUser("Output class averages", "The refined 2D class averages", "my_refined_classes.mrc", false);
	number_of_classes = my_input->GetIntFromUser("Number of classes (>0 = initialize classes)", "The number of classes that should be refined; 0 = the number is determined by the stack of input averages", "0", 0);
	first_particle = my_input->GetIntFromUser("First particle to refine (0 = first in stack)", "The first particle in the stack that should be refined", "1", 0);
	last_particle = my_input->GetIntFromUser("Last particle to refine (0 = last in stack)", "The last particle in the stack that should be refined", "0", 0);
	percent_used = my_input->GetFloatFromUser("Percent of particles to use (1 = all)", "The percentage of randomly selected particles that will be used for classification", "1.0", 0.0, 1.0);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius (A)", "Radius of a circular mask to be applied to the input class averages", "100.0", 0.0);
	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	angular_step = my_input->GetFloatFromUser("Angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
	max_search_x = my_input->GetFloatFromUser("Search range in X (A) (0.0 = max)", "The maximum global peak search distance along X from the particle box center", "0.0", 0.0);
	max_search_y = my_input->GetFloatFromUser("Search range in Y (A) (0.0 = max)", "The maximum global peak search distance along Y from the particle box center", "0.0", 0.0);
	smoothing_factor = my_input->GetFloatFromUser("Tuning parameter: smoothing factor", "Factor for likelihood-weighting; values smaller than 1 will blur results more, larger values will emphasize peaks", "1.0", 0.01);
	padding_factor = my_input->GetIntFromUser("Tuning parameter: padding factor for interpol.", "Factor determining how padding is used to improve interpolation for image rotation", "2", 1);
	normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
	invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	exclude_blank_edges = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "Yes");
	dump_arrays = my_input->GetYesNoFromUser("Dump intermediate arrays (merge later)", "Should the intermediate 2D class sums be dumped to a file for later merging with other jobs", "No");
	dump_file = my_input->GetFilenameFromUser("Output dump filename for intermediate arrays", "The name of the dump file with the intermediate 2D class sums", "dump_file.dat", false);

	delete my_input;

	int current_class = 0;
	my_current_job.Reset(26);
	my_current_job.ManualSetArguments("tttttiiiffffffffffffibbbbt",	input_particle_images.ToUTF8().data(),
																	input_parameter_file.ToUTF8().data(),
																	input_class_averages.ToUTF8().data(),
																	ouput_parameter_file.ToUTF8().data(),
																	ouput_class_averages.ToUTF8().data(),
																	number_of_classes, first_particle, last_particle, percent_used,
																	pixel_size, voltage_kV, spherical_aberration_mm, amplitude_contrast,
																	mask_radius, low_resolution_limit, high_resolution_limit,
																	angular_step, max_search_x, max_search_y, smoothing_factor,
																	padding_factor, normalize_particles, invert_contrast,
																	exclude_blank_edges, dump_arrays, dump_file.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool Refine2DApp::DoCalculation()
{
	Particle input_particle;

	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_parameter_file 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_class_averages				= my_current_job.arguments[2].ReturnStringArgument();
	wxString ouput_parameter_file				= my_current_job.arguments[3].ReturnStringArgument();
	wxString ouput_class_averages 				= my_current_job.arguments[4].ReturnStringArgument();
	number_of_classes							= my_current_job.arguments[5].ReturnIntegerArgument();
	int		 first_particle						= my_current_job.arguments[6].ReturnIntegerArgument();
	int		 last_particle						= my_current_job.arguments[7].ReturnIntegerArgument();
	float	 percent_used						= my_current_job.arguments[8].ReturnFloatArgument();
	pixel_size									= my_current_job.arguments[9].ReturnFloatArgument();
	float    voltage_kV							= my_current_job.arguments[10].ReturnFloatArgument();
	float 	 spherical_aberration_mm			= my_current_job.arguments[11].ReturnFloatArgument();
	float    amplitude_contrast					= my_current_job.arguments[12].ReturnFloatArgument();
	float    mask_radius						= my_current_job.arguments[13].ReturnFloatArgument();
	float    low_resolution_limit				= my_current_job.arguments[14].ReturnFloatArgument();
	float    high_resolution_limit				= my_current_job.arguments[15].ReturnFloatArgument();
	float	 angular_step						= my_current_job.arguments[16].ReturnFloatArgument();
	float	 max_search_x						= my_current_job.arguments[17].ReturnFloatArgument();
	float	 max_search_y						= my_current_job.arguments[18].ReturnFloatArgument();
	float	 smoothing_factor					= my_current_job.arguments[19].ReturnFloatArgument();
	int		 padding_factor						= my_current_job.arguments[20].ReturnIntegerArgument();
// Psi, Theta, Phi, ShiftX, ShiftY
	input_particle.parameter_map.psi			= true;
	input_particle.parameter_map.theta			= false;
	input_particle.parameter_map.phi			= false;
	input_particle.parameter_map.x_shift		= true;
	input_particle.parameter_map.y_shift		= true;
	bool	 normalize_particles				= my_current_job.arguments[21].ReturnBoolArgument();
	bool	 invert_contrast					= my_current_job.arguments[22].ReturnBoolArgument();
	bool	 exclude_blank_edges				= my_current_job.arguments[23].ReturnBoolArgument();
	bool	 dump_arrays						= my_current_job.arguments[24].ReturnBoolArgument();
	dump_file	 								= my_current_job.arguments[25].ReturnStringArgument();

	input_particle.constraints_used.x_shift = true;		// Constraint for X shifts
	input_particle.constraints_used.y_shift = true;		// Constraint for Y shifts

	Image	input_image, cropped_input_image;
	Image	sum_power, ctf_input_image, padded_image;
	Image	best_correlation_map, temp_image;
	Image	*rotation_cache = NULL;
	Image	*blurred_images = NULL;
	Image	*input_classes_cache = NULL;
	CTF		input_ctf;
	AnglesAndShifts rotation_angle;
	ProgressBar *my_progress;
	Curve	noise_power_spectrum, number_of_terms;

	int i, j, k;
	int fourier_size;
	int current_class, current_image;
	int number_of_rotations;
	int image_counter, images_to_process, pixel_counter;
	int projection_counter;
	int padded_box_size, cropped_box_size, binned_box_size;
	int best_class;
	float input_parameters[17];
	float output_parameters[17];
	float parameter_average[17];
	float parameter_variance[17];
	float binning_factor, binned_pixel_size;
	float temp_float;
	float psi;
	float psi_max;
	float psi_step;
	float psi_start;
	float average;
	float variance;
	float ssq_X;
	float sum_logp_particle;
	float occupancy;
	float filter_constant;
	float mask_radius_for_noise;
	float max_corr, max_logp_particle;
	float percentage;
	float random_shift;
	int number_of_blank_edges;
	int max_samples = 2000;
	wxDateTime my_time_in;

	ZeroFloatArray(input_parameters, 17);
	ZeroFloatArray(output_parameters, 17);
	ZeroFloatArray(parameter_average, 17);
	ZeroFloatArray(parameter_variance, 17);

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
	MRCFile input_stack(input_particle_images.ToStdString(), false);
	FrealignParameterFile input_par_file(input_parameter_file, OPEN_TO_READ);
	FrealignParameterFile output_par_file(ouput_parameter_file, OPEN_TO_WRITE);
	MRCFile *input_classes = NULL;
	MRCFile *output_classes = NULL;
	if (! dump_arrays || number_of_classes != 0) output_classes = new MRCFile(ouput_class_averages.ToStdString(), true);

	if (input_stack.ReturnXSize() != input_stack.ReturnYSize())
	{
		SendError("Error: Particles are not square\n");
		exit(-1);
	}

	if (number_of_classes == 0)
	{
		if (! DoesFileExist(input_class_averages))
		{
			SendError(wxString::Format("Error: Input class averages %s not found\n", input_class_averages));
			exit(-1);
		}
		input_classes = new MRCFile (input_class_averages.ToStdString(), false);
		if (input_classes->ReturnXSize() != input_stack.ReturnXSize() || input_classes->ReturnYSize() != input_stack.ReturnYSize() )
		{
			SendError("Error: Dimension of particles and input classes differ\n");
			exit(-1);
		}
		number_of_classes = input_classes->ReturnZSize();
	}
	list_of_nozero_classes = new int [number_of_classes];
	int *reverse_list_of_nozero_classes = new int [number_of_classes];

	if (last_particle < first_particle && last_particle != 0)
	{
		SendError("Error: Number of last particle to refine smaller than number of first particle to refine\n");
		exit(-1);
	}

	if (last_particle == 0) last_particle = input_stack.ReturnZSize();
	if (first_particle == 0) first_particle = 1;
	if (last_particle > input_stack.ReturnZSize()) last_particle = input_stack.ReturnZSize();

	input_par_file.ReadFile();
	// Read whole parameter file to work out average values and variances

	images_to_process = 0;
	mask_falloff = 20.0;
	log_range = 20.0;
	image_counter = 0;
	for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
	{
		input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		if (input_parameters[7] > 0.0)
		{
			for (i = 0; i < 17; i++)
			{
					parameter_average[i] += input_parameters[i];
					parameter_variance[i] += powf(input_parameters[i],2);
			}
			image_counter++;
		}
		if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle) images_to_process++;
	}

	if (image_counter > 0)
	{
		for (i = 0; i < 17; i++)
		{
			parameter_average[i] /= image_counter;
			parameter_variance[i] /= image_counter;
			parameter_variance[i] -= powf(parameter_average[i],2);

			// nasty hack for new file format support in other programs

			if (parameter_variance[i] < 0.001 && i == 1) input_particle.constraints_used.phi = false;
			if (parameter_variance[i] < 0.001 && i == 2) input_particle.constraints_used.theta = false;
			if (parameter_variance[i] < 0.001 && i == 3) input_particle.constraints_used.psi = false;
			if (parameter_variance[i] < 0.001 && i == 4) input_particle.constraints_used.x_shift = false;
			if (parameter_variance[i] < 0.001 && i == 5) input_particle.constraints_used.y_shift = false;
		}
	}
	else
	{
		input_particle.constraints_used.psi 	= false;
		input_particle.constraints_used.theta 	= false;
		input_particle.constraints_used.phi 	= false;
		input_particle.constraints_used.x_shift = false;
		input_particle.constraints_used.y_shift = false;

		/*for (i = 0; i < 17; i++)
		{
			input_particle.constraints_used[i] = false;
		}*/
	}

	xy_dimensions = input_stack.ReturnXSize();
	if (parameter_average[14] < 0.01) parameter_average[14] = 10.0;
	average_snr = 1.0 / powf(parameter_average[14], 2);
// *****
//	average_snr = 0.002;
	wxPrintf("\nShift averages x, y = %g, %g, shift std x, y = %g, %g, average SNR = %g\n", parameter_average[4], parameter_average[5],
			sqrtf(parameter_variance[4]), sqrtf(parameter_variance[5]), average_snr);


	// hack because of changes to new parameter file..

	cisTEMParameterLine parameter_averages;
	cisTEMParameterLine parameter_variances;

	parameter_averages.position_in_stack = parameter_average[0];
	parameter_averages.psi = parameter_average[3];
	parameter_averages.theta = parameter_average[2];
	parameter_averages.phi = parameter_average[1];
	parameter_averages.x_shift = parameter_average[4];
	parameter_averages.y_shift = parameter_average[5];
	parameter_averages.image_is_active = parameter_average[7];
	parameter_averages.defocus_1 = parameter_average[8];
	parameter_averages.defocus_2 = parameter_average[9];
	parameter_averages.defocus_angle = parameter_average[10];
	parameter_averages.phase_shift = parameter_average[11];
	parameter_averages.occupancy = parameter_average[12];
	parameter_averages.logp = parameter_average[13];
	parameter_averages.sigma = parameter_average[14];
	parameter_averages.score = parameter_average[15];
	parameter_averages.score_change = parameter_average[16];

	parameter_variances.position_in_stack = parameter_variance[0];
	parameter_variances.psi = parameter_variance[3];
	parameter_variances.theta = parameter_variance[2];
	parameter_variances.phi = parameter_variance[1];
	parameter_variances.x_shift = parameter_variance[4];
	parameter_variances.y_shift = parameter_variance[5];
	parameter_variances.image_is_active = parameter_variance[7];
	parameter_variances.defocus_1 = parameter_variance[8];
	parameter_variances.defocus_2 = parameter_variance[9];
	parameter_variances.defocus_angle = parameter_variance[10];
	parameter_variances.phase_shift = parameter_variance[11];
	parameter_variances.occupancy = parameter_variance[12];
	parameter_variances.logp = parameter_variance[13];
	parameter_variances.sigma = parameter_variance[14];
	parameter_variances.score = parameter_variance[15];
	parameter_variances.score_change = parameter_variance[16];


	// end of hack

	input_particle.SetParameterStatistics(parameter_averages, parameter_variances);
	input_par_file.Rewind();

	if (max_search_x == 0.0) max_search_x = input_stack.ReturnXSize() / 2.0 * pixel_size;
	if (max_search_y == 0.0) max_search_y = input_stack.ReturnYSize() / 2.0 * pixel_size;

	my_time_in = wxDateTime::Now();
	output_par_file.WriteCommentLine("C Refine2D run date and time:              " + my_time_in.FormatISOCombined(' '));
	output_par_file.WriteCommentLine("C Input particle images:                   " + input_particle_images);
	output_par_file.WriteCommentLine("C Input Frealign parameter filename:       " + input_parameter_file);
	output_par_file.WriteCommentLine("C Input class averages:                    " + input_class_averages);
	output_par_file.WriteCommentLine("C Output parameter file:                   " + ouput_parameter_file);
	output_par_file.WriteCommentLine("C Output class averages:                   " + ouput_class_averages);
	output_par_file.WriteCommentLine("C First particle to refine:                " + wxString::Format("%i", first_particle));
	output_par_file.WriteCommentLine("C Last particle to refine:                 " + wxString::Format("%i", last_particle));
	output_par_file.WriteCommentLine("C Percent of particles to use:             " + wxString::Format("%f", percent_used));
	output_par_file.WriteCommentLine("C Pixel size of images (A):                " + wxString::Format("%f", pixel_size));
	output_par_file.WriteCommentLine("C Beam energy (keV):                       " + wxString::Format("%f", voltage_kV));
	output_par_file.WriteCommentLine("C Spherical aberration (mm):               " + wxString::Format("%f", spherical_aberration_mm));
	output_par_file.WriteCommentLine("C Amplitude contrast:                      " + wxString::Format("%f", amplitude_contrast));
	output_par_file.WriteCommentLine("C Mask radius for refinement (A):          " + wxString::Format("%f", mask_radius));
	output_par_file.WriteCommentLine("C Low resolution limit (A):                " + wxString::Format("%f", low_resolution_limit));
	output_par_file.WriteCommentLine("C High resolution limit (A):               " + wxString::Format("%f", high_resolution_limit));
	output_par_file.WriteCommentLine("C Angular step:                            " + wxString::Format("%f", angular_step));
	output_par_file.WriteCommentLine("C Search range in X (A):                   " + wxString::Format("%f", max_search_x));
	output_par_file.WriteCommentLine("C Search range in Y (A):                   " + wxString::Format("%f", max_search_y));
	output_par_file.WriteCommentLine("C Smoothing factor:                        " + wxString::Format("%f", smoothing_factor));
	output_par_file.WriteCommentLine("C Padding factor:                          " + wxString::Format("%i", padding_factor));
	output_par_file.WriteCommentLine("C Normalize particles:                     " + BoolToYesNo(normalize_particles));
	output_par_file.WriteCommentLine("C Invert particle contrast:                " + BoolToYesNo(invert_contrast));
	output_par_file.WriteCommentLine("C Exclude images with blank edges:         " + BoolToYesNo(exclude_blank_edges));
	output_par_file.WriteCommentLine("C Dump intermediate arrays:                " + BoolToYesNo(dump_arrays));
	output_par_file.WriteCommentLine("C Output dump filename:                    " + dump_file);
	output_par_file.WriteCommentLine("C");
	output_par_file.WriteCommentLine("C           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST  PSHIFT     OCC      LogP      SIGMA   SCORE  CHANGE");
	fflush(output_par_file.parameter_file);

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
	// Prevent division by zero when correcting CTF
	if (amplitude_contrast <= 0.0) amplitude_contrast = 0.001;

	sum_power.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	temp_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	ctf_input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
	cropped_box_size = ReturnClosestFactorizedUpper(myroundint(2.0 * (std::max(max_search_x,max_search_y) + mask_radius + mask_falloff) / pixel_size), 3, true);
	if (cropped_box_size > input_stack.ReturnXSize()) cropped_box_size = input_stack.ReturnXSize();
	cropped_input_image.Allocate(cropped_box_size, cropped_box_size, true);
	binning_factor = high_resolution_limit / pixel_size / 2.0;
	if (binning_factor < 1.0) binning_factor = 1.0;
	fourier_size = ReturnClosestFactorizedUpper(cropped_box_size / binning_factor, 3, true);
	if (fourier_size > cropped_box_size) fourier_size = cropped_box_size;
	best_correlation_map.Allocate(fourier_size, fourier_size, true);
	binning_factor = float(cropped_box_size) / float(fourier_size);
	binned_pixel_size = pixel_size * binning_factor;
	input_particle.Allocate(fourier_size, fourier_size);
	padded_box_size = int(powf(2.0, float(padding_factor)) + 0.5) * fourier_size;
	padded_image.Allocate(padded_box_size, padded_box_size, true);

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
	psi_max = 360.0;

	wxPrintf("\nNumber of classes = %i, nonzero classes = %i, box size = %i, binning factor = %f, new pixel size = %f, resolution limit = %f, angular step size = %f\n",
			number_of_classes, number_of_nonzero_classes, fourier_size, binning_factor, binned_pixel_size, binned_pixel_size * 2.0, psi_step);

	class_logp = new float [number_of_nonzero_classes];
	float *class_variance_correction = new float [number_of_rotations * number_of_nonzero_classes + 1];
	float *class_variance = new float [number_of_nonzero_classes];
	class_averages = new Image [number_of_nonzero_classes];
	CTF_sums = new Image [number_of_nonzero_classes];
	blurred_images = new Image [number_of_nonzero_classes];
	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		class_averages[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		CTF_sums[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
		blurred_images[i].Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
		class_averages[i].SetToConstant(0.0);
		CTF_sums[i].SetToConstant(0.0);
		class_logp[i] = - std::numeric_limits<float>::max();
	}
	float *logp = new float [number_of_nonzero_classes];

	rotation_cache = new Image [number_of_rotations + 1];
	for (i = 0; i <= number_of_rotations; i++)
	{
		rotation_cache[i].Allocate(fourier_size, fourier_size, false);
	}

	input_classes_cache = new Image [number_of_nonzero_classes];
	for (i = 0; i < number_of_nonzero_classes; i++)
	{
		input_classes_cache[i].Allocate(fourier_size, fourier_size, true);
	}

	if (input_classes != NULL)
	{
		j = 0;
		for (current_class = 0; current_class < number_of_classes; current_class++)
		{
			input_image.ReadSlice(input_classes, current_class + 1);
			variance = input_image.ReturnSumOfSquares();
			if (variance == 0.0 && input_classes != NULL) continue;
			input_image.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
			input_image.SetMinimumValue(-0.3 * input_image.ReturnMaximumValue());
			// Not clear if the following is the right thing to do here...
//			input_image.AddConstant(- input_image.ReturnAverageOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, true));
			input_image.MultiplyByConstant(binning_factor);
			input_image.ClipInto(&cropped_input_image);
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
		for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
		{
			input_par_file.ReadLine(input_parameters);
			if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
			image_counter++;
			my_progress->Update(image_counter);

			input_image.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
			if (exclude_blank_edges && input_image.ContainsBlankEdges(mask_radius / pixel_size)) {number_of_blank_edges++; continue;}
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
				if (global_random_number_generator.GetUniformRandom() >= 1.0 - 2.0 * percent_used / number_of_classes)
				{
					input_image.RealSpaceIntegerShift(myroundint(random_shift * global_random_number_generator.GetUniformRandom()),
							myroundint(random_shift * global_random_number_generator.GetUniformRandom()));
					class_averages[current_class].AddImage(&input_image);
					list_of_nozero_classes[current_class]++;
				}
			}

			if (is_running_locally == false)
			{
				temp_float = current_image;
				JobResult *temp_result = new JobResult;
				temp_result->SetResult(1, &temp_float);
				AddJobToResultQueue(temp_result);
				//wxPrintf("Refine3D : Adding job to job queue..\n");
			}
		}
		for (current_class = 0; current_class < number_of_classes; current_class++)
		{
			if (list_of_nozero_classes[current_class] != 0) class_averages[current_class].MultiplyByConstant(1.0 / list_of_nozero_classes[current_class]);
			class_averages[current_class].WriteSlice(output_classes, current_class + 1);
		}
		delete [] list_of_nozero_classes;
		delete [] reverse_list_of_nozero_classes;
		delete [] rotation_cache;
		delete [] input_classes_cache;
		delete [] class_averages;
		delete [] CTF_sums;
		delete [] blurred_images;
		delete [] class_variance_correction;
		delete [] class_variance;
		delete [] logp;
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
		mask_radius_for_noise = mask_radius / pixel_size;
		number_of_blank_edges = 0;
		if (2.0 * mask_radius_for_noise + mask_falloff / pixel_size > 0.95 * input_image.logical_x_dimension)
		{
			mask_radius_for_noise = 0.95 * input_image.logical_x_dimension / 2.0 - mask_falloff / 2.0 / pixel_size;
		}
		noise_power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		image_counter = 0;
		my_progress = new ProgressBar(images_to_process);
		for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
		{
			input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
			if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
			image_counter++;
			my_progress->Update(image_counter);
			if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0 * percentage)) continue;
			input_image.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
			if (exclude_blank_edges && input_image.ContainsBlankEdges(mask_radius / pixel_size)) {number_of_blank_edges++; continue;}
			variance = input_image.ReturnVarianceOfRealValues(mask_radius / pixel_size, 0.0, 0.0, 0.0, true);
			if (variance == 0.0) continue;
			input_image.MultiplyByConstant(1.0 / sqrtf(variance));
			input_image.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size, true);
			input_image.ForwardFFT();
			temp_image.CopyFrom(&input_image);
			temp_image.ConjugateMultiplyPixelWise(input_image);
			sum_power.AddImage(&temp_image);
		}
		delete my_progress;
		input_par_file.Rewind();
		sum_power.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);
		noise_power_spectrum.SquareRoot();
		noise_power_spectrum.Reciprocal();

		if (exclude_blank_edges)
		{
			wxPrintf("\nNumber of excluded images with blank edges = %i\n", number_of_blank_edges);
		}
	}

	wxPrintf("\nCalculating new class averages...\n\n");
	image_counter = 0;
	number_of_blank_edges = 0;
	images_processed = 0;
	sum_logp_total = - std::numeric_limits<float>::max();
	sum_snr = 0.0;
	my_progress = new ProgressBar(images_to_process);
	for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
	{

		input_par_file.ReadLine(input_parameters);
		if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
		image_counter++;
		if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0 * percent_used))
		{
			input_parameters[7] = - fabsf(input_parameters[7]);
			input_parameters[16] = 0.0;
			output_par_file.WriteLine(input_parameters);
			SendRefineResult(input_parameters);
			my_progress->Update(image_counter);
			continue;
		}

		input_image.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));

		if (exclude_blank_edges && input_image.ContainsBlankEdges(mask_radius / pixel_size))
		{
			number_of_blank_edges++;
			input_parameters[7] = - fabsf(input_parameters[7]);
			input_parameters[16] = 0.0;
			output_par_file.WriteLine(input_parameters);
			SendRefineResult(input_parameters);
			my_progress->Update(image_counter);
			continue;
		}
		else
		{
			input_parameters[7] = fabsf(input_parameters[7]);
			temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		}

		for (i = 0; i < 17; i++) {output_parameters[i] = input_parameters[i];}


// Set up Particle object
		input_particle.ResetImageFlags();
		input_particle.mask_radius = mask_radius;
		input_particle.mask_falloff = mask_falloff;
		input_particle.filter_radius_low = low_resolution_limit;
		input_particle.filter_radius_high = high_resolution_limit;
		// The following line would allow using particles with different pixel sizes
		input_particle.pixel_size = binned_pixel_size;
//		input_particle.snr = average_snr * powf(binning_factor, 2);

		// hack due to changes in parameter files

		cisTEMParameterLine input_params_new_format;
		
		input_params_new_format.position_in_stack = input_parameters[0];
		input_params_new_format.psi = input_parameters[3];
		input_params_new_format.theta = input_parameters[2];
		input_params_new_format.phi = input_parameters[1];
		input_params_new_format.x_shift = input_parameters[4];
		input_params_new_format.y_shift = input_parameters[5];
		input_params_new_format.image_is_active = input_parameters[7];
		input_params_new_format.defocus_1 = input_parameters[8];
		input_params_new_format.defocus_2 = input_parameters[9];
		input_params_new_format.defocus_angle = input_parameters[10];
		input_params_new_format.phase_shift = input_parameters[11];
		input_params_new_format.occupancy = input_parameters[12];
		input_params_new_format.logp = input_parameters[13];
		input_params_new_format.sigma = input_parameters[14];
		input_params_new_format.score = input_parameters[15];
		input_params_new_format.score_change = input_parameters[16];


		// end of hack
		
		input_particle.SetParameters(input_params_new_format);
		input_particle.SetParameterConstraints(1.0 / average_snr / powf(binning_factor, 2));

		input_image.ReplaceOutliersWithMean(5.0);
		if (invert_contrast) input_image.InvertRealValues();
		input_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], input_parameters[11]);
		input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, pixel_size, input_parameters[11]);
		ctf_input_image.CalculateCTFImage(input_ctf);

		if (normalize_particles)
		{
			input_image.ForwardFFT();
			// Whiten noise
			input_image.ApplyCurveFilter(&noise_power_spectrum);
			// Apply cosine filter to reduce ringing
			input_image.CosineMask(std::max(pixel_size / high_resolution_limit, pixel_size / 7.0f + pixel_size / mask_falloff) - pixel_size / (2.0 * mask_falloff), pixel_size / mask_falloff);
			input_image.BackwardFFT();
			// Normalize background variance and average
			variance = input_image.ReturnVarianceOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, 0.0, 0.0, 0.0, true);
			average = input_image.ReturnAverageOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, true);
			input_image.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
			// At this point, input_image should have white background with a variance of 1. The variance should therefore be about 1/binning_factor^2 after binning.
		}
		// Multiply by binning_factor so variance after binning is close to 1.
		input_image.MultiplyByConstant(binning_factor);
		// Determine sum of squares of corrected image and binned image for ML calculation
		temp_image.CopyFrom(&input_image);
//		temp_image.CosineMask(temp_image.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size);
		temp_image.ClipInto(&cropped_input_image);
		cropped_input_image.CosineMask(cropped_input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
		cropped_input_image.ForwardFFT();
		cropped_input_image.ClipInto(input_particle.particle_image);
		ssq_X = input_particle.particle_image->ReturnSumOfSquares();
		input_particle.CTFMultiplyImage();
		variance = input_particle.particle_image->ReturnSumOfSquares();
		// Apply CTF
		input_image.ForwardFFT();
		input_image.MultiplyPixelWiseReal(ctf_input_image);
		input_image.BackwardFFT();
		// Calculate rotated versions of input image
//		input_particle.mask_volume = input_image.CosineMask(input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size);
		input_image.ClipInto(&cropped_input_image);
		input_particle.mask_volume = cropped_input_image.CosineMask(cropped_input_image.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size, false, true, 0.0);
//		input_particle.mask_volume = cropped_input_image.number_of_real_space_pixels;
		cropped_input_image.ForwardFFT();
		// This ClipInto is needed as a filtering step to set Fourier terms outside the final binned image to zero.
		cropped_input_image.ClipInto(input_particle.particle_image);
		input_particle.particle_image->ClipInto(&padded_image);
		// Pre-correct for real-space interpolation errors
		// Maybe this is not a good idea since this amplifies noise...
//		padded_image.CorrectSinc();
		padded_image.BackwardFFT();
//???		input_particle.mask_volume = padded_image.CosineMask(padded_image.physical_address_of_box_center_x - mask_falloff / pixel_size, mask_falloff / pixel_size);

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
		input_particle.mask_volume = rotation_cache[0].number_of_real_space_pixels * input_particle.mask_volume / cropped_input_image.number_of_real_space_pixels;
//		wxPrintf("rot = %li, in = %li, mask = %g\n", rotation_cache[0].number_of_real_space_pixels, cropped_input_image.number_of_real_space_pixels, input_particle.mask_volume);
		input_particle.is_masked = true;

		sum_logp_particle = - std::numeric_limits<float>::max();
		max_logp_particle = - std::numeric_limits<float>::max();

		// Calculate score of previous best match, store cc map in best_correlation_map. Matching projection is in last location of projection_cache.
		best_class = myroundint(fabsf(input_parameters[7])) - 1;
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
			rotation_angle.GenerateRotationMatrix2D(- input_parameters[3]);
			padded_image.Rotate2DSample(rotation_cache[number_of_rotations], rotation_angle);
			temp_float = rotation_cache[number_of_rotations].ReturnSumOfSquares();
			rotation_cache[number_of_rotations].MultiplyByConstant(sqrtf(variance / temp_float));
//			rotation_cache[number_of_rotations].MultiplyByConstant(-1.0);
//			rotation_cache[number_of_rotations].QuickAndDirtyWriteSlice("rot.mrc", 1);
//			wxPrintf("best class = %i angle = %g\n", best_class, input_parameters[3]);
//			exit(0);
			rotation_cache[number_of_rotations].ForwardFFT();
			temp_float = input_particle.MLBlur(input_classes_cache, ssq_X, cropped_input_image, rotation_cache,
					blurred_images[0], best_class, number_of_rotations, psi_step, psi_start, smoothing_factor,
					max_logp_particle, best_class, input_parameters[3], best_correlation_map, true);
		}

		for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
		{
			//wxPrintf("Working on class %i\n", current_class);
			logp[current_class] = input_particle.MLBlur(input_classes_cache, ssq_X, cropped_input_image, rotation_cache,
					blurred_images[current_class], current_class, number_of_rotations, psi_step, psi_start, smoothing_factor,
					max_logp_particle, best_class, input_parameters[3], best_correlation_map);
			// Sum likelihoods of all classes
			sum_logp_particle = ReturnSumOfLogP(sum_logp_particle, logp[current_class], log_range);
		}
		sum_logp_total = ReturnSumOfLogP(sum_logp_total, sum_logp_particle, log_range);

		if (input_particle.current_parameters.sigma != 0.0) sum_snr += 1.0 / input_particle.current_parameters.sigma;
		for (current_class = 0; current_class < number_of_nonzero_classes; current_class++)
		{
			// Divide class likelihoods by summed likelihood; this is a simple subtraction of the log values
			logp[current_class] -= sum_logp_particle;
			if (logp[current_class] >= - log_range)
			{
				// Sum the class likelihoods (now divided by the summed likelihood)
				class_logp[current_class] = ReturnSumOfLogP(class_logp[current_class], logp[current_class], log_range);
				// Apply likelihood weight to blurred image
				if (input_particle.current_parameters.sigma > 0.0)
				{
					temp_float = expf(logp[current_class]);
					// Need to divide here by sigma^2; already divided once on input, therefore divide only by sigma here.
					blurred_images[current_class].MultiplyByConstant(temp_float / input_particle.current_parameters.sigma);
					// Add weighted image to class average
					class_averages[current_class].AddImage(&blurred_images[current_class]);
					// Copy and multiply CTF image
					temp_image.CopyFrom(&ctf_input_image);
					temp_image.MultiplyPixelWiseReal(ctf_input_image);
					temp_image.MultiplyByConstant(temp_float / input_particle.current_parameters.sigma);
					CTF_sums[current_class].AddImage(&temp_image);
					if (current_class + 1 == input_particle.current_parameters.image_is_active) input_particle.current_parameters.occupancy = 100.0 * temp_float;
				}
			}
		}

		// hack due to update to new parmaeter format..

		cisTEMParameterLine output_params_new_format;
		input_particle.GetParameters(output_params_new_format);

		output_parameters[0] = output_params_new_format.position_in_stack;
		output_parameters[3] = output_params_new_format.psi;
		output_parameters[2] = output_params_new_format.theta;
		output_parameters[1] = output_params_new_format.phi;
		output_parameters[4] = output_params_new_format.x_shift;
		output_parameters[5] = output_params_new_format.y_shift;
		output_parameters[6] = input_parameters[6];
		output_parameters[7] = output_params_new_format.image_is_active;
		output_parameters[8] = output_params_new_format.defocus_1;
		output_parameters[9] = output_params_new_format.defocus_2;
		output_parameters[10] = output_params_new_format.defocus_angle;
		output_parameters[11] = output_params_new_format.phase_shift;
		output_parameters[12] = output_params_new_format.occupancy;
		output_parameters[13] = output_params_new_format.logp;
		output_parameters[14] = output_params_new_format.sigma;
		output_parameters[15] = output_params_new_format.score;
		output_parameters[16] = output_params_new_format.score_change;

		// end hack

		temp_float = output_parameters[1]; output_parameters[1] = output_parameters[3]; output_parameters[3] = temp_float;
		output_parameters[7] = list_of_nozero_classes[myroundint(output_parameters[7]) - 1] + 1.0;
		// Multiply measured sigma noise in binned image by binning factor to obtain sigma noise of unbinned image
		output_parameters[14] *= binning_factor;
		if (output_parameters[14] > 100.0) output_parameters[14] = 100.0;
		output_parameters[16] = output_parameters[15] - input_parameters[15];
		output_par_file.WriteLine(output_parameters);
		SendRefineResult(output_parameters);
		fflush(output_par_file.parameter_file);

		images_processed++;

		my_progress->Update(image_counter);
	}

	delete my_progress;

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
				variance = class_averages[current_class].ReturnSumOfSquares();
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
	delete [] rotation_cache;
	delete [] input_classes_cache;
	delete [] class_averages;
	delete [] CTF_sums;
	delete [] blurred_images;
	delete [] class_variance_correction;
	delete [] class_variance;
	delete [] logp;
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

void Refine2DApp::SendRefineResult(float *current_params)
{
	if (is_running_locally == false) // send results back to the gui..
	{
		float gui_result_params[7];

		JobResult *intermediate_result = new JobResult;
		intermediate_result->job_number = my_current_job.job_number;

		gui_result_params[0] = current_params[0];
		gui_result_params[1] = current_params[1];
		gui_result_params[2] = current_params[4];
		gui_result_params[3] = current_params[5];
		gui_result_params[4] = current_params[7];
		gui_result_params[5] = current_params[14];
		gui_result_params[6] = current_params[13];

		intermediate_result->SetResult(7, gui_result_params);
		AddJobToResultQueue(intermediate_result);
	}
}
