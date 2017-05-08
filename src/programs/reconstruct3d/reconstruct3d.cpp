#include "../../core/core_headers.h"

class
Reconstruct3DApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

IMPLEMENT_APP(Reconstruct3DApp)

// override the DoInteractiveUserInput

void Reconstruct3DApp::DoInteractiveUserInput()
{
	wxString	input_particle_stack;
	wxString	input_parameter_file;
	wxString	input_reconstruction;
	wxString	output_reconstruction_1;
	wxString	output_reconstruction_2;
	wxString	output_reconstruction_filtered;
	wxString	output_resolution_statistics;
	wxString	my_symmetry = "C1";
	int			first_particle = 1;
	int			last_particle = 0;
	float		pixel_size = 1;
	float		voltage_kV = 300.0;
	float		spherical_aberration_mm = 2.7;
	float		amplitude_contrast = 0.07;
	float		molecular_mass_kDa = 1000.0;
	float		inner_mask_radius = 0.0;
	float		outer_mask_radius = 100.0;
	float		resolution_limit_rec = 0.0;
	float		resolution_limit_ref = 0.0;
	float		score_weight_conversion = 5.0;
	float		score_threshold = 1.0;
	float		smoothing_factor = 1.0;
	float		padding = 1.0;
	bool		normalize_particles = true;
	bool		adjust_scores = true;
	bool		invert_contrast = false;
	bool		exclude_blank_edges = false;
	bool		crop_images = false;
	bool		split_even_odd = true;
	bool		center_mass = false;
	bool		use_input_reconstruction = false;
	bool		threshold_input_3d = true;
	bool		dump_arrays = false;
	wxString	dump_file_1;
	wxString	dump_file_2;

	UserInput *my_input = new UserInput("Reconstruct3D", 1.03);

	input_particle_stack = my_input->GetFilenameFromUser("Input particle images", "The input particle image stack, containing the 2D images for each particle in the dataset", "my_particle_stack.mrc", true);
	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	input_reconstruction = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from the previous refinement cycle to perform likelihood blurring", "my_input_reconstruction.mrc", false);
	output_reconstruction_1 = my_input->GetFilenameFromUser("Output reconstruction 1", "The first output 3D reconstruction, calculated form half the data", "my_reconstruction_1.mrc", false);
	output_reconstruction_2 = my_input->GetFilenameFromUser("Output reconstruction 2", "The second output 3D reconstruction, calculated form half the data", "my_reconstruction_2.mrc", false);
	output_reconstruction_filtered = my_input->GetFilenameFromUser("Output filtered reconstruction", "The final 3D reconstruction, containing from all data and optimally filtered", "my_filtered_reconstruction.mrc", false);
	output_resolution_statistics = my_input->GetFilenameFromUser("Output resolution statistics", "The text file with the resolution statistics for the final reconstruction", "my_statistics.txt", false);
	my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
	first_particle = my_input->GetIntFromUser("First particle to include (0 = first in stack)", "The first particle in the stack that should be included in the reconstruction", "1", 0);
	last_particle = my_input->GetIntFromUser("Last particle to include (0 = last in stack)", "The last particle in the stack that should be included in the reconstruction", "0", 0);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	inner_mask_radius = my_input->GetFloatFromUser("Inner mask radius (A)", "Radius of a circular mask to be applied to the center of the final reconstruction in Angstroms", "0.0", 0.0);
	outer_mask_radius = my_input->GetFloatFromUser("Outer mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", inner_mask_radius);
	resolution_limit_rec = my_input->GetFloatFromUser("Rec. resolution limit (A) (0.0 = max)", "The resolution to which the reconstruction will be calculated, in Angstroms", "0.0", 0.0);
	resolution_limit_ref = my_input->GetFloatFromUser("Ref. resolution limit (A) (0.0 = max)", "The resolution used during refinement, in Angstroms", "0.0", 0.0);
	score_weight_conversion = my_input->GetFloatFromUser("Particle weighting factor (A^2)", "Constant to convert particle scores to resolution-dependent weight in squared Angstroms", "5.0", 0.0);
	score_threshold = my_input->GetFloatFromUser("Score threshold (<= 1 = percentage)", "Minimum score to include a particle in the reconstruction", "1.0");
	smoothing_factor = my_input->GetFloatFromUser("Tuning parameter: smoothing factor", "Factor for likelihood-weighting; values smaller than 1 will blur results more, larger values will emphasize peaks", "1.0", 0.01);
	padding = my_input->GetFloatFromUser("Tuning parameters: padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
	normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "The input particle images should always be normalized unless they were pre-processed", "Yes");
	adjust_scores = my_input->GetYesNoFromUser("Adjust scores for defocus dependence", "Should the particle scores be adjusted internally to reduce their dependence on defocus", "Yes");
	invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	exclude_blank_edges = my_input->GetYesNoFromUser("Exclude images with blank edges", "Should particle images with blank edges be excluded from processing?", "No");
	crop_images = my_input->GetYesNoFromUser("Crop particle images", "Should the particle images be cropped to speed up computation?", "No");
	split_even_odd = my_input->GetYesNoFromUser("FSC calculation with even/odd particles", "Should the FSC half volumes be calulated using even and odd particles?", "Yes");
	center_mass = my_input->GetYesNoFromUser("Center mass", "Should the calculated map be centered in the box according to the center of mass (only for C symmetry)?", "No");
	use_input_reconstruction = my_input->GetYesNoFromUser("Apply likelihood blurring", "Should ML blurring be applied?", "No");
	threshold_input_3d = my_input->GetYesNoFromUser("Threshold input reconstruction", "Should the input reconstruction thresholded to suppress some of the background noise", "No");
	dump_arrays = my_input->GetYesNoFromUser("Dump intermediate arrays (merge later)", "Should the 3D reconstruction arrays be dumped to a file for later merging with other jobs", "No");
	dump_file_1 = my_input->GetFilenameFromUser("Output dump filename for odd particles", "The name of the first dump file with the intermediate reconstruction arrays", "dump_file_1.dat", false);
	dump_file_2 = my_input->GetFilenameFromUser("Output dump filename for even particles", "The name of the second dump file with the intermediate reconstruction arrays", "dump_file_2.dat", false);

	delete my_input;

	my_current_job.Reset(35);
	my_current_job.ManualSetArguments("ttttttttiifffffffffffffbbbbbbbbbbtt",	input_particle_stack.ToUTF8().data(),
																				input_parameter_file.ToUTF8().data(),
																				input_reconstruction.ToUTF8().data(),
																				output_reconstruction_1.ToUTF8().data(),
																				output_reconstruction_2.ToUTF8().data(),
																				output_reconstruction_filtered.ToUTF8().data(),
																				output_resolution_statistics.ToUTF8().data(),
																				my_symmetry.ToUTF8().data(),
																				first_particle, last_particle,
																				pixel_size, voltage_kV, spherical_aberration_mm, amplitude_contrast,
																				molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
																				resolution_limit_rec, resolution_limit_ref, score_weight_conversion, score_threshold,
																				smoothing_factor, padding, normalize_particles, adjust_scores,
																				invert_contrast, exclude_blank_edges, crop_images, split_even_odd, center_mass,
																				use_input_reconstruction, threshold_input_3d, dump_arrays,
																				dump_file_1.ToUTF8().data(),
																				dump_file_2.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool Reconstruct3DApp::DoCalculation()
{
	wxString input_particle_stack 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_parameter_file 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_reconstruction				= my_current_job.arguments[2].ReturnStringArgument();
	wxString output_reconstruction_1			= my_current_job.arguments[3].ReturnStringArgument();
	wxString output_reconstruction_2			= my_current_job.arguments[4].ReturnStringArgument();
	wxString output_reconstruction_filtered		= my_current_job.arguments[5].ReturnStringArgument();
	wxString output_resolution_statistics		= my_current_job.arguments[6].ReturnStringArgument();
	wxString my_symmetry						= my_current_job.arguments[7].ReturnStringArgument();
	int		 first_particle						= my_current_job.arguments[8].ReturnIntegerArgument();
	int		 last_particle						= my_current_job.arguments[9].ReturnIntegerArgument();
	float 	 pixel_size							= my_current_job.arguments[10].ReturnFloatArgument();
	float    voltage_kV							= my_current_job.arguments[11].ReturnFloatArgument();
	float 	 spherical_aberration_mm			= my_current_job.arguments[12].ReturnFloatArgument();
	float    amplitude_contrast					= my_current_job.arguments[13].ReturnFloatArgument();
	float 	 molecular_mass_kDa					= my_current_job.arguments[14].ReturnFloatArgument();
	float    inner_mask_radius					= my_current_job.arguments[15].ReturnFloatArgument();
	float    outer_mask_radius					= my_current_job.arguments[16].ReturnFloatArgument();
	float    resolution_limit_rec				= my_current_job.arguments[17].ReturnFloatArgument();
	float    resolution_limit_ref				= my_current_job.arguments[18].ReturnFloatArgument();
	float    score_weight_conversion			= my_current_job.arguments[19].ReturnFloatArgument();
	float    score_threshold					= my_current_job.arguments[20].ReturnFloatArgument();
	float	 smoothing_factor					= my_current_job.arguments[21].ReturnFloatArgument();
	float    padding							= my_current_job.arguments[22].ReturnFloatArgument();
	bool	 normalize_particles				= my_current_job.arguments[23].ReturnBoolArgument();
	bool	 adjust_scores						= my_current_job.arguments[24].ReturnBoolArgument();
	bool	 invert_contrast					= my_current_job.arguments[25].ReturnBoolArgument();
	bool	 exclude_blank_edges				= my_current_job.arguments[26].ReturnBoolArgument();
	bool	 crop_images						= my_current_job.arguments[27].ReturnBoolArgument();
	bool	 split_even_odd						= my_current_job.arguments[28].ReturnBoolArgument();
	bool	 center_mass						= my_current_job.arguments[29].ReturnBoolArgument();
	bool	 use_input_reconstruction			= my_current_job.arguments[30].ReturnBoolArgument();
	bool	 threshold_input_3d					= my_current_job.arguments[31].ReturnBoolArgument();
	bool	 dump_arrays						= my_current_job.arguments[32].ReturnBoolArgument();
	wxString dump_file_1 						= my_current_job.arguments[33].ReturnStringArgument();
	wxString dump_file_2 						= my_current_job.arguments[34].ReturnStringArgument();

	ReconstructedVolume input_3d(molecular_mass_kDa);
	ReconstructedVolume output_3d(molecular_mass_kDa);
	ReconstructedVolume output_3d1(molecular_mass_kDa);
	ReconstructedVolume output_3d2(molecular_mass_kDa);
	Image 				temp_image;
	Image 				temp2_image;
	Image				temp3_image;
	Image				current_ctf_image;
	Image				projection_image;
	Image				padded_projection_image;
	Image				cropped_projection_image;
	Image				unmasked_image;
	Image				padded_image;
	Image				*rotation_cache = NULL;
	CTF					current_ctf;
	CTF					input_ctf;
	Curve				noise_power_spectrum;
	Curve				number_of_terms;
	AnglesAndShifts 	rotation_angle;
	ProgressBar			*my_progress;

	int i, j;
	int current_image;
	int images_to_process = 0;
	int images_for_noise_power = 0;
	int image_counter = 0;
	int box_size;
	int original_box_size;
	int intermediate_box_size;
	int number_of_blank_edges;
	int max_samples = 2000;
	int random_seed_multiplier = 2;
	int fsc_particle_repeat = 2;
	int padding_2d = 2;
	int padded_box_size;
	int number_of_rotations;
	int min_class;
	int max_class;
	float temp_float;
	float mask_volume_fraction;
	float mask_falloff = 10.0;
	float particle_area_in_pixels;
	float binning_factor = 1.0;
	float variance;
	float average;
	float sigma;
	float original_pixel_size = pixel_size;
	float outer_mask_in_pixels = outer_mask_radius / pixel_size;
	float average_density_max;
	float percentage;
	float ssq_X;
	float psi;
	float psi_step;
//	float psi_max;
	float psi_start;
	float symmetry_weight = 1.0;
	bool rotational_blurring = true;
	wxDateTime my_time_in;

	MRCFile input_file(input_particle_stack.ToStdString(), false);
	MRCFile *input_3d_file;
	if (use_input_reconstruction) input_3d_file = new MRCFile(input_reconstruction.ToStdString(), false);
	FrealignParameterFile input_par_file(input_parameter_file, OPEN_TO_READ);
	input_par_file.ReadFile(true, input_file.ReturnZSize());
/*	input_par_file.ReduceAngles();
	min_class = myroundint(input_par_file.ReturnMin(7));
	max_class = myroundint(input_par_file.ReturnMax(7));
	for (i = min_class; i <= max_class; i++)
	{
		temp_float = input_par_file.ReturnDistributionMax(2, i);
		sigma = input_par_file.ReturnDistributionSigma(2, temp_float, i);
		if (temp_float != 0.0) wxPrintf("theta max, sigma, phi max, sigma = %i %g %g", i, temp_float, sigma);
		input_par_file.SetParameters(2, temp_float, sigma / 2.0, i);
		temp_float = input_par_file.ReturnDistributionMax(3, i);
		sigma = input_par_file.ReturnDistributionSigma(3, temp_float, i);
		if (temp_float != 0.0) wxPrintf(" %g %g\n", temp_float, sigma);
		input_par_file.SetParameters(3, temp_float, sigma / 2.0, i);
	} */
	// sigma values
	input_par_file.RemoveOutliers(14, 2.0);
	// score values
	input_par_file.RemoveOutliers(15, 1.0);
	NumericTextFile output_statistics_file(output_resolution_statistics, OPEN_TO_WRITE, 7);

	if (is_running_locally == false)
	{
		float result;
		my_result.SetResult(1, &result);
	}
	if (input_file.ReturnXSize() != input_file.ReturnYSize())
	{
		MyPrintWithDetails("Error: Particles are not square\n");
		abort();
	}
	if (last_particle < first_particle && last_particle != 0)
	{
		MyPrintWithDetails("Error: Number of last particle to refine smaller than number of first particle to refine\n");
		abort();
	}

	if (last_particle == 0) last_particle = input_file.ReturnZSize();
	if (first_particle == 0) first_particle = 1;
	if (last_particle > input_file.ReturnZSize()) last_particle = input_file.ReturnZSize();

	input_par_file.CalculateDefocusDependence();
	if (adjust_scores) input_par_file.AdjustScores();

	if (score_threshold > 0.0 && score_threshold < 1.0) score_threshold = input_par_file.ReturnThreshold(score_threshold);
	if (score_threshold == 1.0) score_threshold = input_par_file.ReturnMin(15);

	if (! split_even_odd) fsc_particle_repeat = myroundint((input_par_file.ReturnMax(0) - input_par_file.ReturnMin(0) + 1.0) / 100.0);
	if (fsc_particle_repeat % 2 != 0) fsc_particle_repeat++;

	my_time_in = wxDateTime::Now();
	output_statistics_file.WriteCommentLine("C Refine3D run date and time:              " + my_time_in.FormatISOCombined(' '));
	output_statistics_file.WriteCommentLine("C Input particle images:                   " + input_particle_stack);
	output_statistics_file.WriteCommentLine("C Input Frealign parameter filename:       " + input_parameter_file);
	output_statistics_file.WriteCommentLine("C Input reconstruction:                    " + input_reconstruction);
	output_statistics_file.WriteCommentLine("C Output reconstruction 1:                 " + output_reconstruction_1);
	output_statistics_file.WriteCommentLine("C Output reconstruction 2:                 " + output_reconstruction_2);
	output_statistics_file.WriteCommentLine("C Output filtered reconstruction:          " + output_reconstruction_filtered);
	output_statistics_file.WriteCommentLine("C Output resolution statistics:            " + output_resolution_statistics);
	output_statistics_file.WriteCommentLine("C Particle symmetry:                       " + my_symmetry);
	output_statistics_file.WriteCommentLine("C First particle to include:               " + wxString::Format("%i", first_particle));
	output_statistics_file.WriteCommentLine("C Last particle to include:                " + wxString::Format("%i", last_particle));
	output_statistics_file.WriteCommentLine("C Pixel size of images (A):                " + wxString::Format("%f", pixel_size));
	output_statistics_file.WriteCommentLine("C Beam energy (keV):                       " + wxString::Format("%f", voltage_kV));
	output_statistics_file.WriteCommentLine("C Spherical aberration (mm):               " + wxString::Format("%f", spherical_aberration_mm));
	output_statistics_file.WriteCommentLine("C Amplitude contrast:                      " + wxString::Format("%f", amplitude_contrast));
	output_statistics_file.WriteCommentLine("C Molecular mass of particle (kDa):        " + wxString::Format("%f", molecular_mass_kDa));
	output_statistics_file.WriteCommentLine("C Inner mask radius (A):                   " + wxString::Format("%f", inner_mask_radius));
	output_statistics_file.WriteCommentLine("C Outer mask radius (A):                   " + wxString::Format("%f", outer_mask_radius));
	output_statistics_file.WriteCommentLine("C Rec. resolution limit (A):               " + wxString::Format("%f", resolution_limit_rec));
	output_statistics_file.WriteCommentLine("C Ref. resolution limit (A):               " + wxString::Format("%f", resolution_limit_ref));
	output_statistics_file.WriteCommentLine("C Particle weighting factor (A^2):         " + wxString::Format("%f", score_weight_conversion));
	output_statistics_file.WriteCommentLine("C Score threshold:                         " + wxString::Format("%f", score_threshold));
	output_statistics_file.WriteCommentLine("C Smoothing factor:                        " + wxString::Format("%f", smoothing_factor));
	output_statistics_file.WriteCommentLine("C Padding factor:                          " + wxString::Format("%f", padding));
	output_statistics_file.WriteCommentLine("C Normalize particles:                     " + BoolToYesNo(normalize_particles));
	output_statistics_file.WriteCommentLine("C Adjust scores for defocus dependence:    " + BoolToYesNo(adjust_scores));
	output_statistics_file.WriteCommentLine("C Invert particle contrast:                " + BoolToYesNo(invert_contrast));
	output_statistics_file.WriteCommentLine("C Exclude images with blank edges:         " + BoolToYesNo(exclude_blank_edges));
	output_statistics_file.WriteCommentLine("C Crop particle images:                    " + BoolToYesNo(crop_images));
	output_statistics_file.WriteCommentLine("C FSC with even/odd particles:             " + BoolToYesNo(split_even_odd));
	output_statistics_file.WriteCommentLine("C Apply likelihood blurring:               " + BoolToYesNo(use_input_reconstruction));
	output_statistics_file.WriteCommentLine("C Threshold input reconstruction:          " + BoolToYesNo(threshold_input_3d));
	output_statistics_file.WriteCommentLine("C Dump intermediate arrays:                " + BoolToYesNo(dump_arrays));
	output_statistics_file.WriteCommentLine("C Output dump filename for odd particles:  " + dump_file_1);
	output_statistics_file.WriteCommentLine("C Output dump filename for even particles: " + dump_file_2);
	output_statistics_file.WriteCommentLine("C");

	original_box_size = input_file.ReturnXSize();
	// If resolution limit higher that Nyquist, do not do binning
	if (resolution_limit_rec < 2.0 * pixel_size) resolution_limit_rec = 0.0;
	if (resolution_limit_ref < 2.0 * pixel_size) resolution_limit_ref = 0.0;

	// Assume square particles and cubic volumes
	if (resolution_limit_rec != 0.0)
	{
		binning_factor = resolution_limit_rec / pixel_size / 2.0;
		intermediate_box_size = ReturnClosestFactorizedUpper(original_box_size / binning_factor, 3, true);
		if (intermediate_box_size > original_box_size) intermediate_box_size = original_box_size;
		binning_factor = float(original_box_size) / float(intermediate_box_size);
		pixel_size *= binning_factor;
		if (crop_images)
		{
			box_size = ReturnClosestFactorizedUpper(myroundint(3.0 * outer_mask_radius / pixel_size), 3, true);
			if (box_size > intermediate_box_size) box_size = intermediate_box_size;
			if (box_size == intermediate_box_size) crop_images = false;
		}
		else
		{
			box_size = intermediate_box_size;
		}
	}
	else
	{
		intermediate_box_size = 0;
		if (crop_images)
		{
			box_size = ReturnClosestFactorizedUpper(myroundint(3.0 * outer_mask_radius / pixel_size), 3, true);
			if (box_size > original_box_size) box_size = original_box_size;
			if (box_size == original_box_size) crop_images = false;
		}
		else
		{
			box_size = original_box_size;
		}
	}

	Particle input_particle(box_size, box_size);
	input_particle.AllocateCTFImage(box_size, box_size);
	if (use_input_reconstruction) unmasked_image.Allocate(box_size, box_size, true);
	current_ctf_image.Allocate(original_box_size, original_box_size, true);
	temp_image.Allocate(original_box_size, original_box_size, true);
	temp3_image.Allocate(original_box_size, original_box_size, false);
	if (resolution_limit_rec != 0.0 && crop_images) temp2_image.Allocate(intermediate_box_size, intermediate_box_size, true);
	float input_parameters[input_particle.number_of_parameters];
	float parameter_average[input_particle.number_of_parameters];
	float parameter_variance[input_particle.number_of_parameters];
	ZeroFloatArray(input_parameters, input_particle.number_of_parameters);
	ZeroFloatArray(parameter_average, input_particle.number_of_parameters);
	ZeroFloatArray(parameter_variance, input_particle.number_of_parameters);

// Read whole parameter file to work out average image_sigma_noise and average score
	for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
	{
		input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		for (i = 0; i < input_particle.number_of_parameters; i++)
		{
			parameter_average[i] += input_parameters[i];
			parameter_variance[i] += powf(input_parameters[i],2);
		}
		if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle && input_parameters[12] != 0.0 && input_parameters[15] >= score_threshold && input_parameters[7] >= 0.0) images_to_process++;
		if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle) images_for_noise_power++;
	}
	for (i = 0; i < input_particle.number_of_parameters; i++)
	{
		parameter_average[i] /= input_par_file.number_of_lines;
		parameter_variance[i] /= input_par_file.number_of_lines;
		parameter_variance[i] -= powf(parameter_average[i],2);
	}
	input_particle.SetParameterStatistics(parameter_average, parameter_variance);
	input_particle.mask_radius = outer_mask_radius;
	input_particle.mask_falloff = mask_falloff;
	input_par_file.Rewind();

	Reconstruct3D my_reconstruction_1(box_size, box_size, box_size, pixel_size, parameter_average[12], parameter_average[14], score_weight_conversion, my_symmetry);
	Reconstruct3D my_reconstruction_2(box_size, box_size, box_size, pixel_size, parameter_average[12], parameter_average[14], score_weight_conversion, my_symmetry);
	my_reconstruction_1.original_x_dimension = original_box_size;
	my_reconstruction_1.original_y_dimension = original_box_size;
	my_reconstruction_1.original_z_dimension = original_box_size;
	my_reconstruction_1.original_pixel_size = original_pixel_size;
	my_reconstruction_1.center_mass = center_mass;
	my_reconstruction_2.original_x_dimension = original_box_size;
	my_reconstruction_2.original_y_dimension = original_box_size;
	my_reconstruction_2.original_z_dimension = original_box_size;
	my_reconstruction_2.original_pixel_size = original_pixel_size;
	my_reconstruction_2.center_mass = center_mass;

	wxPrintf("\nNumber of particles to reconstruct = %i, average sigma noise = %f, average LogP = %f\n", images_to_process, parameter_average[14], parameter_average[15]);
	wxPrintf("Box size for reconstruction = %i, binning factor = %f\n", box_size, binning_factor);

	if (images_to_process == 0)
	{
		if (! dump_arrays) MyPrintWithDetails("Error: No particles to process\n");
		if (dump_arrays)
		{
			wxPrintf("\nDumping reconstruction arrays...\n");
			my_reconstruction_1.DumpArrays(dump_file_1, false);
			my_reconstruction_2.DumpArrays(dump_file_2, true);

			wxPrintf("\nReconstruct3D: Normal termination\n\n");
		}
		return true;
	}

	if (2.0 * outer_mask_in_pixels + mask_falloff / original_pixel_size > 0.95 * original_box_size)
	{
		outer_mask_in_pixels = 0.95 * original_box_size / 2.0 - mask_falloff / 2.0 / original_pixel_size;
	}

	if (normalize_particles)
	{
		wxPrintf("\nCalculating noise power spectrum...\n\n");
		percentage = float(max_samples) / float(images_to_process);
		temp3_image.SetToConstant(0.0);
		number_of_blank_edges = 0;
		noise_power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((temp3_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((temp3_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
		my_progress = new ProgressBar(images_for_noise_power);
		for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
		{
			input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
			if (input_parameters[0] < first_particle || input_parameters[0] > last_particle) continue;
			image_counter++;
			my_progress->Update(image_counter);
			if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0 * percentage)) continue;
			current_ctf_image.ReadSlice(&input_file, int(input_parameters[0] + 0.5));
			if (exclude_blank_edges && current_ctf_image.ContainsBlankEdges(outer_mask_radius / original_pixel_size)) {number_of_blank_edges++; continue;}
			variance = current_ctf_image.ReturnVarianceOfRealValues(outer_mask_radius / original_pixel_size, 0.0, 0.0, 0.0, true);
			if (variance == 0.0) continue;
			current_ctf_image.MultiplyByConstant(1.0 / sqrtf(variance));
			current_ctf_image.CosineMask(outer_mask_in_pixels, mask_falloff / original_pixel_size, true);
			current_ctf_image.ForwardFFT();
			temp_image.CopyFrom(&current_ctf_image);
			temp_image.ConjugateMultiplyPixelWise(current_ctf_image);
			temp3_image.AddImage(&temp_image);
		}
		delete my_progress;
		temp3_image.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);
		noise_power_spectrum.SquareRoot();
		noise_power_spectrum.Reciprocal();

		input_par_file.Rewind();
		if (exclude_blank_edges)
		{
			wxPrintf("\nImages with blank edges excluded from noise power calculation = %i\n", number_of_blank_edges);
		}
	}

	if (use_input_reconstruction)
	{

		input_3d.InitWithDimensions(original_box_size, original_box_size, original_box_size, pixel_size, my_symmetry);
		projection_image.Allocate(original_box_size, original_box_size, false);
		if (resolution_limit_rec != 0.0 || crop_images) cropped_projection_image.Allocate(box_size, box_size, false);
		input_3d.density_map.ReadSlices(input_3d_file,1,input_3d.density_map.logical_z_dimension);
		// Check that the input 3D map has reasonable density values
//		float *temp_3d = new float [int(input_3d.density_map.number_of_real_space_pixels / 10.0 + input_3d.density_map.logical_x_dimension)];
//		input_3d.density_map.AddConstant(- input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
		// Remove masking here to avoid edge artifacts later
		input_3d.density_map.CosineMask(outer_mask_in_pixels, 3.0 * mask_falloff / pixel_size, false, true, 0.0);
		if (inner_mask_radius > 0.0) input_3d.density_map.CosineMask(inner_mask_radius / pixel_size, mask_falloff / pixel_size, true);
//		for (i = 0; i < input_3d.density_map.real_memory_allocated; i++) if (input_3d.density_map.real_values[i] < 0.0) input_3d.density_map.real_values[i] = -log(-input_3d.density_map.real_values[i] + 1.0);
//		average_density_max = input_3d.density_map.ReturnAverageOfMaxN(100, outer_mask_in_pixels);
/*		if (average_density_max < 0.1 || average_density_max > 25)
		{

			SendInfo(wxString::Format("Input 3D densities out of range (average max = %g). Rescaling...", average_density_max));
			input_3d.density_map.MultiplyByConstant(0.1 / average_density_max);
			average_density_max = 0.1;
		} */
		if (threshold_input_3d)
		{
			// Threshold map to suppress negative noise
			average_density_max = input_3d.density_map.ReturnAverageOfMaxN(100, outer_mask_in_pixels);
			input_3d.density_map.SetMinimumValue(-0.3 * average_density_max);
//			input_3d.density_map.SetMinimumValue(input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
		}
		if (padding != 1.0)
		{
			input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension * padding, input_3d.density_map.logical_y_dimension * padding, input_3d.density_map.logical_z_dimension * padding, input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
			padded_projection_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
		}
		input_3d.mask_radius = outer_mask_radius;
		if (input_3d.mask_radius > input_3d.density_map.physical_address_of_box_center_x * pixel_size- mask_falloff) input_3d.mask_radius = input_3d.density_map.physical_address_of_box_center_x * pixel_size- mask_falloff;
		input_3d.PrepareForProjections(0.0, resolution_limit_ref, false, false);
		// Multiply by binning_factor to scale reference to be compatible with scaled binned image (see below)
		if (binning_factor != 1.0) input_3d.density_map.MultiplyByConstant(binning_factor);

		if (rotational_blurring)
		{
			padded_box_size = int(powf(2.0, float(padding_2d)) + 0.5) * box_size;
			padded_image.Allocate(padded_box_size, padded_box_size, true);

			psi_step = 2.0 * rad_2_deg(pixel_size / outer_mask_radius);
			if (psi_step < 5.0) psi_step = 15.0;
			number_of_rotations = int(360.0 / psi_step + 0.5);
			if (number_of_rotations % 2 == 0) number_of_rotations--;
			psi_step = 360.0 / number_of_rotations;
	//		psi_step = 2.0;
			psi_start = - (number_of_rotations - 1) / 2.0 * psi_step;

			rotation_cache = new Image [number_of_rotations];
			for (i = 0; i < number_of_rotations; i++)
			{
				rotation_cache[i].Allocate(box_size, box_size, false);
			}
			wxPrintf("\nUsing %i rotations for image blurring with angular step %g\n", number_of_rotations, psi_step);
		}
	}

	wxPrintf("\nCalculating reconstruction...\n\n");
	my_progress = new ProgressBar(images_to_process);
	image_counter = 0;
	for (current_image = 1; current_image <= input_par_file.number_of_lines; current_image++)
	{
		input_par_file.ReadLine(input_parameters);
		if (input_parameters[0] < first_particle || input_parameters[0] > last_particle || input_parameters[12] == 0.0 || input_parameters[15] < score_threshold || input_parameters[7] < 0.0) continue;
		image_counter++;
		input_particle.location_in_stack = int(input_parameters[0] + 0.5);
		input_particle.pixel_size = pixel_size;
		input_particle.is_masked = false;

		input_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], input_parameters[11]);

		if (use_input_reconstruction)
		{
			input_particle.alignment_parameters.Init(input_parameters[3], input_parameters[2], input_parameters[1], 0.0, 0.0);
			if (padding != 1.0)
			{
				input_3d.density_map.ExtractSlice(padded_projection_image, input_particle.alignment_parameters);
				padded_projection_image.SwapRealSpaceQuadrants();
				padded_projection_image.BackwardFFT();
				padded_projection_image.ClipInto(&projection_image);
				projection_image.ForwardFFT();
				if (! crop_images) projection_image.SwapRealSpaceQuadrants();
			}
			else
			{
				input_3d.density_map.ExtractSlice(projection_image, input_particle.alignment_parameters);
				if (crop_images) projection_image.SwapRealSpaceQuadrants();
			}
		}
		else
		{
			input_particle.alignment_parameters.Init(input_parameters[3], input_parameters[2], input_parameters[1], input_parameters[4], input_parameters[5]);
		}

		if (crop_images || binning_factor != 1.0)
		{
			input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, original_pixel_size, input_parameters[11]);
			if (input_ctf.IsAlmostEqualTo(&current_ctf, 40.0 / pixel_size) == false)
			// Need to calculate current_ctf_image to be inserted into ctf_reconstruction
			{
				current_ctf = input_ctf;
				current_ctf_image.CalculateCTFImage(current_ctf);
			}
		}
		// Assume square images
		if (crop_images)
		{
			temp_image.ReadSlice(&input_file, input_particle.location_in_stack);
			if (invert_contrast) temp_image.InvertRealValues();
			if (normalize_particles)
			{
				temp_image.ForwardFFT();
				temp_image.ApplyCurveFilter(&noise_power_spectrum);
				if (use_input_reconstruction)
				{
					temp_image.PhaseShift(- input_parameters[4] / original_pixel_size, - input_parameters[5] / original_pixel_size);
					temp_image.PhaseFlipPixelWise(current_ctf_image);
				}
				temp_image.BackwardFFT();
				// Normalize background variance and average
				variance = temp_image.ReturnVarianceOfRealValues(temp_image.physical_address_of_box_center_x - mask_falloff / original_pixel_size, 0.0, 0.0, 0.0, true);
				average = temp_image.ReturnAverageOfRealValues(temp_image.physical_address_of_box_center_x - mask_falloff / original_pixel_size, true);
				temp_image.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
			}
			else
			if (use_input_reconstruction)
			{
				temp_image.ForwardFFT();
				temp_image.PhaseShift(- input_parameters[4] / - original_pixel_size, - input_parameters[5] / - original_pixel_size);
				temp_image.PhaseFlipPixelWise(current_ctf_image);
				if (binning_factor == 1.0) temp_image.BackwardFFT();
			}
			if (binning_factor != 1.0)
			{
				if (use_input_reconstruction)
				{
					if (normalize_particles) temp_image.ForwardFFT();
					temp_image.ClipInto(&temp2_image);
					temp2_image.BackwardFFT();
					temp2_image.ClipInto(input_particle.particle_image);
					unmasked_image.CopyFrom(input_particle.particle_image);
					input_particle.CosineMask(false, true, 0.0);
					input_particle.ForwardFFT();
					unmasked_image.ForwardFFT();
					projection_image.ClipInto(&temp2_image);
					temp2_image.BackwardFFT();
					temp2_image.ClipInto(&cropped_projection_image);
					cropped_projection_image.ForwardFFT();
					cropped_projection_image.SwapRealSpaceQuadrants();
					cropped_projection_image.MultiplyPixelWiseReal(*input_particle.ctf_image, true);
					// Multiply by binning_factor to scale noise variance to 1 in binned image
					input_particle.particle_image->MultiplyByConstant(binning_factor);
					ssq_X = input_particle.particle_image->ReturnSumOfSquares();
					temp_float = - std::numeric_limits<float>::max();
					if (rotational_blurring)
					{
						input_particle.particle_image->ClipInto(&padded_image);
						// Pre-correct for real-space interpolation errors
						padded_image.CorrectSinc();
						padded_image.BackwardFFT();
						variance = input_particle.particle_image->ReturnSumOfSquares();
						j = 0;
						for (i = 0; i < number_of_rotations; i++)
						{
							psi = i * psi_step + psi_start;
							if (fabsf(psi) < psi_step / 2.0) rotation_cache[0].CopyFrom(input_particle.particle_image);
							else
							{
								j++;
								rotation_angle.GenerateRotationMatrix2D(psi);
								padded_image.Rotate2DSample(rotation_cache[j], rotation_angle);
								rotation_cache[j].ForwardFFT();
								// Correct variance for interpolation.
								temp_float = rotation_cache[j].ReturnSumOfSquares();
								rotation_cache[j].MultiplyByConstant(sqrtf(variance / temp_float));
							}
						}
						input_particle.MLBlur(&cropped_projection_image, ssq_X, cropped_projection_image, rotation_cache, *input_particle.particle_image, 0,
								number_of_rotations, psi_step, psi_start, smoothing_factor, temp_float, -1, 0.0, cropped_projection_image, false, false, false, &unmasked_image);
					}
					else
					{
						input_particle.MLBlur(&cropped_projection_image, ssq_X, cropped_projection_image, input_particle.particle_image, *input_particle.particle_image, 0, 1, 0.0, 0.0,
								smoothing_factor, temp_float, -1, 0.0, cropped_projection_image, false, false, false, &unmasked_image);
					}
					input_particle.particle_image->object_is_centred_in_box = false;
				}
				else
				{
					if (normalize_particles) temp_image.ForwardFFT();
					temp_image.ClipInto(&temp2_image);
					temp2_image.BackwardFFT();
					temp2_image.ClipInto(input_particle.particle_image);
					input_particle.ForwardFFT();
					input_particle.PhaseFlipImage();
				}
			}
			else
			{
				if (use_input_reconstruction)
				{
					temp_image.ClipInto(input_particle.particle_image);
					unmasked_image.CopyFrom(input_particle.particle_image);
					input_particle.CosineMask(false, true, 0.0);
					input_particle.ForwardFFT();
					unmasked_image.ForwardFFT();
					projection_image.BackwardFFT();
					projection_image.ClipInto(&cropped_projection_image);
					cropped_projection_image.ForwardFFT();
					cropped_projection_image.SwapRealSpaceQuadrants();
					cropped_projection_image.MultiplyPixelWiseReal(*input_particle.ctf_image, true);
					ssq_X = input_particle.particle_image->ReturnSumOfSquares();
					temp_float = - std::numeric_limits<float>::max();
					if (rotational_blurring)
					{
						input_particle.particle_image->ClipInto(&padded_image);
						// Pre-correct for real-space interpolation errors
						padded_image.CorrectSinc();
						padded_image.BackwardFFT();
						variance = input_particle.particle_image->ReturnSumOfSquares();
						j = 0;
						for (i = 0; i < number_of_rotations; i++)
						{
							psi = i * psi_step + psi_start;
							if (fabsf(psi) < psi_step / 2.0) rotation_cache[0].CopyFrom(input_particle.particle_image);
							else
							{
								j++;
								rotation_angle.GenerateRotationMatrix2D(psi);
								padded_image.Rotate2DSample(rotation_cache[j], rotation_angle);
								rotation_cache[j].ForwardFFT();
								// Correct variance for interpolation.
								temp_float = rotation_cache[j].ReturnSumOfSquares();
								rotation_cache[j].MultiplyByConstant(sqrtf(variance / temp_float));
							}
						}
						input_particle.MLBlur(&cropped_projection_image, ssq_X, cropped_projection_image, rotation_cache, *input_particle.particle_image, 0,
								number_of_rotations, psi_step, psi_start, smoothing_factor, temp_float, -1, 0.0, cropped_projection_image, false, false, false, &unmasked_image);
					}
					else
					{
						input_particle.MLBlur(&cropped_projection_image, ssq_X, cropped_projection_image, input_particle.particle_image, *input_particle.particle_image, 0, 1, 0.0, 0.0,
								smoothing_factor, temp_float, -1, 0.0, cropped_projection_image, false, false, false, &unmasked_image);
					}
					input_particle.particle_image->object_is_centred_in_box = false;
				}
				else
				{
					temp_image.ClipInto(input_particle.particle_image);
					input_particle.ForwardFFT();
					input_particle.PhaseFlipImage();
				}
			}
		}
		else
		{
			if (binning_factor != 1.0)
			{
				temp_image.ReadSlice(&input_file, input_particle.location_in_stack);
				if (invert_contrast) temp_image.InvertRealValues();
				if (normalize_particles)
				{
					temp_image.ForwardFFT();
					temp_image.ApplyCurveFilter(&noise_power_spectrum);
					if (use_input_reconstruction)
					{
						temp_image.PhaseShift(- input_parameters[4] / original_pixel_size, - input_parameters[5] / original_pixel_size);
						temp_image.PhaseFlipPixelWise(current_ctf_image);
					}
					temp_image.BackwardFFT();
					// Normalize background variance and average
					variance = temp_image.ReturnVarianceOfRealValues(temp_image.physical_address_of_box_center_x - mask_falloff / original_pixel_size, 0.0, 0.0, 0.0, true);
					average = temp_image.ReturnAverageOfRealValues(temp_image.physical_address_of_box_center_x - mask_falloff / original_pixel_size, true);
					temp_image.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
				}
				else
				if (use_input_reconstruction)
				{
					temp_image.ForwardFFT();
					temp_image.PhaseShift(- input_parameters[4] / - original_pixel_size, - input_parameters[5] / - original_pixel_size);
					temp_image.PhaseFlipPixelWise(current_ctf_image);
					temp_image.BackwardFFT();
				}
				if (use_input_reconstruction)
				{
					temp3_image.CopyFrom(&temp_image);
					temp_image.CosineMask(outer_mask_radius / original_pixel_size, mask_falloff / original_pixel_size, false, true, 0.0);
					temp_image.ForwardFFT();
					temp3_image.ForwardFFT();
					temp_image.ClipInto(input_particle.particle_image);
					temp3_image.ClipInto(&unmasked_image);
					projection_image.ClipInto(&cropped_projection_image);
					cropped_projection_image.MultiplyPixelWiseReal(*input_particle.ctf_image, true);
					// Multiply by binning_factor to scale noise variance to 1 in binned image
					input_particle.particle_image->MultiplyByConstant(binning_factor);
					ssq_X = input_particle.particle_image->ReturnSumOfSquares();
					temp_float = - std::numeric_limits<float>::max();
					if (rotational_blurring)
					{
						input_particle.particle_image->ClipInto(&padded_image);
						// Pre-correct for real-space interpolation errors
						padded_image.CorrectSinc();
						padded_image.BackwardFFT();
						variance = input_particle.particle_image->ReturnSumOfSquares();
						j = 0;
						for (i = 0; i < number_of_rotations; i++)
						{
							psi = i * psi_step + psi_start;
							if (fabsf(psi) < psi_step / 2.0) rotation_cache[0].CopyFrom(input_particle.particle_image);
							else
							{
								j++;
								rotation_angle.GenerateRotationMatrix2D(psi);
								padded_image.Rotate2DSample(rotation_cache[j], rotation_angle);
								rotation_cache[j].ForwardFFT();
								// Correct variance for interpolation.
								temp_float = rotation_cache[j].ReturnSumOfSquares();
								rotation_cache[j].MultiplyByConstant(sqrtf(variance / temp_float));
							}
						}
						input_particle.MLBlur(&cropped_projection_image, ssq_X, cropped_projection_image, rotation_cache, *input_particle.particle_image, 0,
								number_of_rotations, psi_step, psi_start, smoothing_factor, temp_float, -1, 0.0, cropped_projection_image, false, false, false, &unmasked_image);
					}
					else
					{
						input_particle.MLBlur(&cropped_projection_image, ssq_X, cropped_projection_image, input_particle.particle_image, *input_particle.particle_image, 0, 1, 0.0, 0.0,
								smoothing_factor, temp_float, -1, 0.0, cropped_projection_image, false, false, false, &unmasked_image);
					}
					input_particle.particle_image->object_is_centred_in_box = false;
				}
				else
				{
					temp_image.ForwardFFT();
					temp_image.ClipInto(input_particle.particle_image);
					input_particle.PhaseFlipImage();
				}
			}
			else
			{
				input_particle.particle_image->ReadSlice(&input_file, input_particle.location_in_stack);
				if (invert_contrast) input_particle.particle_image->InvertRealValues();
				if (normalize_particles)
				{
					input_particle.ForwardFFT();
					input_particle.particle_image->ApplyCurveFilter(&noise_power_spectrum);
					if (use_input_reconstruction)
					{
						input_particle.particle_image->PhaseShift(- input_parameters[4] / pixel_size, - input_parameters[5] / pixel_size);
						input_particle.PhaseFlipImage();
					}
					input_particle.BackwardFFT();
					// Normalize background variance and average
					variance = input_particle.particle_image->ReturnVarianceOfRealValues(input_particle.particle_image->physical_address_of_box_center_x - mask_falloff / original_pixel_size, 0.0, 0.0, 0.0, true);
					average = input_particle.particle_image->ReturnAverageOfRealValues(input_particle.particle_image->physical_address_of_box_center_x - mask_falloff / original_pixel_size, true);
					input_particle.particle_image->AddMultiplyConstant(- average, 1.0 / sqrtf(variance));
				}
				else
				if (use_input_reconstruction)
				{
					input_particle.ForwardFFT();
					input_particle.particle_image->PhaseShift(- input_parameters[4] / pixel_size, - input_parameters[5] / pixel_size);
					input_particle.PhaseFlipImage();
					input_particle.BackwardFFT();

				}
				if (use_input_reconstruction)
				{
					unmasked_image.CopyFrom(input_particle.particle_image);
					input_particle.CosineMask(false, true, 0.0);
					input_particle.ForwardFFT();
					unmasked_image.ForwardFFT();
					projection_image.MultiplyPixelWiseReal(*input_particle.ctf_image, true);
					ssq_X = input_particle.particle_image->ReturnSumOfSquares();
					temp_float = - std::numeric_limits<float>::max();
					if (rotational_blurring)
					{
						input_particle.particle_image->ClipInto(&padded_image);
						// Pre-correct for real-space interpolation errors
						padded_image.CorrectSinc();
						padded_image.BackwardFFT();
						variance = input_particle.particle_image->ReturnSumOfSquares();
						j = 0;
						for (i = 0; i < number_of_rotations; i++)
						{
							psi = i * psi_step + psi_start;
							if (fabsf(psi) < psi_step / 2.0) rotation_cache[0].CopyFrom(input_particle.particle_image);
							else
							{
								j++;
								rotation_angle.GenerateRotationMatrix2D(psi);
								padded_image.Rotate2DSample(rotation_cache[j], rotation_angle);
								rotation_cache[j].ForwardFFT();
								// Correct variance for interpolation.
								temp_float = rotation_cache[j].ReturnSumOfSquares();
								rotation_cache[j].MultiplyByConstant(sqrtf(variance / temp_float));
							}
						}
						input_particle.MLBlur(&projection_image, ssq_X, projection_image, rotation_cache, *input_particle.particle_image, 0,
								number_of_rotations, psi_step, psi_start, smoothing_factor, temp_float, -1, 0.0, projection_image, false, false, false, &unmasked_image);
					}
					else
					{
						input_particle.MLBlur(&projection_image, ssq_X, projection_image, input_particle.particle_image, *input_particle.particle_image, 0,
								1, 0.0, 0.0, smoothing_factor, temp_float, -1, 0.0, projection_image, false, false, false, &unmasked_image);
					}
					input_particle.particle_image->object_is_centred_in_box = false;
				}
				else
				{
					input_particle.ForwardFFT();
					input_particle.PhaseFlipImage();
				}
			}
		}
		if (! use_input_reconstruction) input_particle.particle_image->SwapRealSpaceQuadrants();

		input_particle.particle_score = input_parameters[15];
		input_particle.particle_occupancy = input_parameters[12];
		input_particle.sigma_noise = input_parameters[14];
		if (input_particle.sigma_noise <= 0.0) input_particle.sigma_noise = parameter_average[14];

//		if (current_image % 2 == 0)
		if (int(input_parameters[0] + 0.5) % fsc_particle_repeat < fsc_particle_repeat / 2)
		{
			input_particle.insert_even = true;
		}
		else
		{
			input_particle.insert_even = false;
		}

		if (input_particle.insert_even)
		{
			my_reconstruction_2.InsertSliceWithCTF(input_particle, symmetry_weight);
		}
		else
		{
			my_reconstruction_1.InsertSliceWithCTF(input_particle, symmetry_weight);
		}

		if (is_running_locally == false)
		{
			temp_float = input_particle.location_in_stack;
			JobResult *temp_result = new JobResult;
			temp_result->SetResult(1, &temp_float);
			AddJobToResultQueue(temp_result);
			//wxPrintf("Refine3D : Adding job to job queue..\n");
		}
		my_progress->Update(image_counter);
	}
	delete my_progress;

	if (use_input_reconstruction)
	{
		delete input_3d_file;
		if (rotational_blurring) delete [] rotation_cache;
	}

	if (dump_arrays)
	{
		wxPrintf("\nDumping reconstruction arrays...\n");
		my_reconstruction_1.DumpArrays(dump_file_1, false);
		my_reconstruction_2.DumpArrays(dump_file_2, true);

		wxPrintf("\nReconstruct3D: Normal termination\n\n");

		return true;
	}

	output_3d1.FinalizeSimple(my_reconstruction_1, original_box_size, original_pixel_size, pixel_size,
			inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_1);
	output_3d2.FinalizeSimple(my_reconstruction_2, original_box_size, original_pixel_size, pixel_size,
			inner_mask_radius, outer_mask_radius, mask_falloff, output_reconstruction_2);

	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
	my_reconstruction_1 += my_reconstruction_2;
	my_reconstruction_2.FreeMemory();

	output_3d.FinalizeOptimal(my_reconstruction_1, output_3d1.density_map, output_3d2.density_map,
			original_pixel_size, pixel_size, inner_mask_radius, outer_mask_radius, mask_falloff,
			center_mass, output_reconstruction_filtered, output_statistics_file);

	wxPrintf("\nReconstruct3D: Normal termination\n\n");

	return true;
}
