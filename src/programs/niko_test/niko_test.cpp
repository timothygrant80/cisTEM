#include "../../core/core_headers.h"

class
NikoTestApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

class ImageProjectionComparison
{
public:
	Image 						*particle_image;
	Image						*CTF_image;
	ReconstructedVolume			*reference_volume;
	Image						*projection_image;
	AnglesAndShifts 			*particle_parameters;
	ParameterConstraints		*parameter_constraints;
	float						mask_radius;
	float						mask_falloff;
	float						low_resolution_limit;
	float						high_resolution_limit;
	float						*input_parameters;
	float						*temp_float;
	bool						*parameter_map;
};

int MapParameters(float *input_parameters, float *mapped_parameters, bool *parameter_map, int number_of_parameters)
{
	int i;
	int j = 0;

	for (i = 0; i < number_of_parameters; i++)
	{
		if (parameter_map[i])
		{
			mapped_parameters[j] = input_parameters[i];
			j++;
		}
	}

	return j;
}

int UnmapParameters(float *input_parameters, float *mapped_parameters, bool *parameter_map, int number_of_parameters)
{
	int i;
	int j = 0;

	for (i = 0; i < number_of_parameters; i++)
	{
		if (parameter_map[i])
		{
			input_parameters[i] = mapped_parameters[j];
			j++;
		}
	}

	return j;
}

// This is the function which will be minimized
float FrealignObjectiveFunction(void *scoring_parameters, float *array_of_values)
{
	ImageProjectionComparison *comparison_object = reinterpret_cast < ImageProjectionComparison *> (scoring_parameters);
	comparison_object->temp_float[0] = comparison_object->input_parameters[3];
	comparison_object->temp_float[1] = comparison_object->input_parameters[2];
	comparison_object->temp_float[2] = comparison_object->input_parameters[1];
	comparison_object->temp_float[3] = 0.0;
	comparison_object->temp_float[4] = 0.0;
	UnmapParameters(comparison_object->temp_float, array_of_values, comparison_object->parameter_map, 5);
	comparison_object->particle_parameters->Init(comparison_object->temp_float[0], comparison_object->temp_float[1], comparison_object->temp_float[2],
			comparison_object->temp_float[3], comparison_object->temp_float[4]);

	comparison_object->reference_volume->density_map.ExtractSlice(*comparison_object->projection_image, *comparison_object->particle_parameters);
	comparison_object->projection_image->MultiplyPixelWise(*comparison_object->CTF_image);
//	comparison_object->projection_image->SwapRealSpaceQuadrants();
	comparison_object->projection_image->BackwardFFT();
	comparison_object->projection_image->CosineMask(comparison_object->mask_radius, comparison_object->mask_falloff);
	comparison_object->projection_image->ForwardFFT();
	comparison_object->projection_image->PhaseShift(comparison_object->particle_parameters->ReturnShiftX() / comparison_object->reference_volume->pixel_size,
			comparison_object->particle_parameters->ReturnShiftY() / comparison_object->reference_volume->pixel_size);
//	comparison_object->projection_image->SwapRealSpaceQuadrants();
//	wxPrintf("Parameters = %f, %f, %f, %f, %f, score = %f\n", array_of_values[0],array_of_values[1],array_of_values[2], array_of_values[3],array_of_values[4],
//			comparison_object->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image,
//			comparison_object->low_resolution_limit, comparison_object->high_resolution_limit));
//	exit(0);
	// Evaluate the function, return NEGATIVE value since we are using a MINIMIZER
	return 	- comparison_object->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image,
			  comparison_object->low_resolution_limit, comparison_object->high_resolution_limit)
			- comparison_object->parameter_constraints->ReturnShiftXPenalty(comparison_object->particle_parameters->ReturnShiftX())
			- comparison_object->parameter_constraints->ReturnShiftYPenalty(comparison_object->particle_parameters->ReturnShiftY());
}

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput()
{
	wxString	input_parameter_file;
	wxString	input_particle_images;
	wxString	input_reconstruction;
	wxString	input_reconstruction_statistics;
	wxString	ouput_matching_projections;
	wxString	ouput_parameter_file;
	wxString	ouput_shift_file;
	float		pixel_size = 1.0;
	float		voltage_kV = 300.0;
	float		spherical_aberration_mm = 2.7;
	float		amplitude_contrast = 0.07;
	float		mask_radius = 100.0;
	float		low_resolution_limit = 300.0;
	float		high_resolution_limit = 8.0;
	float		padding = 1.0;
	wxString	my_symmetry = "C1";
	bool		refine_psi = true;
	bool		refine_theta = true;
	bool		refine_phi = true;
	bool		refine_x = true;
	bool		refine_y = true;
	bool		calculate_matching_projections = false;


	UserInput *my_input = new UserInput("Refine", 1.0);

	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	input_particle_images = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	input_reconstruction = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	input_reconstruction_statistics = my_input->GetFilenameFromUser("Input reconstruction statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", true);
	ouput_matching_projections = my_input->GetFilenameFromUser("Output matching projections", "The output image stack, containing the matching projections", "my_projection_stack.mrc", false);
	ouput_parameter_file = my_input->GetFilenameFromUser("Output parameter file", "The output parameter file, containing your refined particle alignment parameters", "my_refined_parameters.par", false);
	ouput_shift_file = my_input->GetFilenameFromUser("Output parameter changes", "The changes in the alignment parameters compared to the input parameters", "my_parameter_changes.par", false);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius (A)", "Radius of a circular mask to be applied to the input images", "100.0", 0.0);
	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	padding = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
	my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
	refine_psi = my_input->GetYesNoFromUser("Refine Psi", "Should the Psi Euler angle be refined (parameter 1)?", "Yes");
	refine_theta = my_input->GetYesNoFromUser("Refine Theta", "Should the Theta Euler angle be refined (parameter 2)?", "Yes");
	refine_phi = my_input->GetYesNoFromUser("Refine Phi", "Should the Phi Euler angle be refined (parameter 3)?", "Yes");
	refine_x = my_input->GetYesNoFromUser("Refine ShiftX", "Should the X shift be refined (parameter 4)?", "Yes");
	refine_y = my_input->GetYesNoFromUser("Refine ShiftY", "Should the Y shift be refined (parameter 5)?", "Yes");
	calculate_matching_projections = my_input->GetYesNoFromUser("Calculate matching projections", "Should matching projections be calculated?", "No");

	delete my_input;

	my_current_job.Reset(10);
	my_current_job.ManualSetArguments("tttttttfffffffftbbbbbb",	input_parameter_file.ToUTF8().data(),
																input_particle_images.ToUTF8().data(),
																input_reconstruction.ToUTF8().data(),
																input_reconstruction_statistics.ToUTF8().data(),
																ouput_matching_projections.ToUTF8().data(),
																ouput_parameter_file.ToUTF8().data(),
																ouput_shift_file.ToUTF8().data(),
																pixel_size,
																voltage_kV,
																spherical_aberration_mm,
																amplitude_contrast,
																mask_radius,
																low_resolution_limit,
																high_resolution_limit,
																padding,
																my_symmetry.ToUTF8().data(),
																refine_psi, refine_theta, refine_phi, refine_x, refine_y,
																calculate_matching_projections);
}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation()
{
	wxString input_parameter_file 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_particle_images 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_reconstruction				= my_current_job.arguments[2].ReturnStringArgument();
	wxString input_reconstruction_statistics 	= my_current_job.arguments[3].ReturnStringArgument();
	wxString ouput_matching_projections 		= my_current_job.arguments[4].ReturnStringArgument();
	wxString ouput_parameter_file				= my_current_job.arguments[5].ReturnStringArgument();
	wxString ouput_shift_file					= my_current_job.arguments[6].ReturnStringArgument();
	float 	 pixel_size							= my_current_job.arguments[7].ReturnFloatArgument();
	float    voltage_kV							= my_current_job.arguments[8].ReturnFloatArgument();
	float 	 spherical_aberration_mm			= my_current_job.arguments[9].ReturnFloatArgument();
	float    amplitude_contrast					= my_current_job.arguments[10].ReturnFloatArgument();
	float    mask_radius						= my_current_job.arguments[11].ReturnFloatArgument();
	float    low_resolution_limit				= my_current_job.arguments[12].ReturnFloatArgument();
	float    high_resolution_limit				= my_current_job.arguments[13].ReturnFloatArgument();
	float	 padding							= my_current_job.arguments[14].ReturnFloatArgument();
	wxString my_symmetry						= my_current_job.arguments[15].ReturnStringArgument();
	bool	 parameter_map[5];
// Psi, Theta, Phi, ShiftX, ShiftY
	parameter_map[2]							= my_current_job.arguments[16].ReturnBoolArgument();
	parameter_map[1]							= my_current_job.arguments[17].ReturnBoolArgument();
	parameter_map[0]							= my_current_job.arguments[18].ReturnBoolArgument();
	parameter_map[3]							= my_current_job.arguments[19].ReturnBoolArgument();
	parameter_map[4]							= my_current_job.arguments[20].ReturnBoolArgument();
	bool calculate_matching_projections			= my_current_job.arguments[21].ReturnBoolArgument();

	Image input_image;
	Image particle_image;
	Image ctf_image;
	Image projection_image;
	Image unbinned_image;
	Image final_image;
	CTF   my_ctf;
	Image snr_image;
	ReconstructedVolume			input_3d;
	ImageProjectionComparison	comparison_object;
	ConjugateGradient			conjugate_gradient_minimizer;
	AnglesAndShifts				my_parameters;
	ParameterConstraints		my_constraints;
	float particle_weight;
	float particle_score;
	float image_sigma_noise;
	float particle_occupancy;

	int i;
	int fourier_size_x;
	int fourier_size_y;
	int fourier_size_z;
	int current_image;
	int number_of_search_dimensions;
	int parameters_per_line = 16;
	float input_parameters[parameters_per_line];
	float output_parameters[parameters_per_line];
	float parameter_average[parameters_per_line];
	float parameter_variance[parameters_per_line];
	float binning_factor;
	float cg_starting_point[5];
	float cg_accuracy[5];
	float temp_float[5];
	float target_phase_error = 45.0;
	float mask_falloff;

	ZeroFloatArray(input_parameters, parameters_per_line);
	ZeroFloatArray(output_parameters, parameters_per_line);
	ZeroFloatArray(parameter_average, parameters_per_line);
	ZeroFloatArray(parameter_variance, parameters_per_line);
	NumericTextFile my_input_par_file(input_parameter_file, OPEN_TO_READ);
	MRCFile input_stack(input_particle_images.ToStdString(), false);
	MRCFile input_file(input_reconstruction.ToStdString(), false);
	MRCFile output_file(ouput_matching_projections.ToStdString(), true);
	NumericTextFile my_output_par_file(ouput_parameter_file, OPEN_TO_WRITE, 16);
	my_output_par_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP     SigmaNoise          Score         Change");
	NumericTextFile my_output_par_shifts_file(ouput_shift_file, OPEN_TO_WRITE, 15);
	my_output_par_shifts_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP     SigmaNoise          Score");
	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, my_symmetry);
	input_3d.ReadStatisticsFromFile(input_reconstruction_statistics);
	input_3d.density_map.ReadSlices(&input_file,1,input_3d.density_map.logical_z_dimension);
	MyDebugAssertTrue(input_3d.density_map.IsCubic(), "3D reference not cubic");

	if (padding != 1.0)
	{
		input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension * padding, input_3d.density_map.logical_y_dimension * padding, input_3d.density_map.logical_z_dimension * padding, input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
	}

	input_3d.density_map.Correct3D();
	input_3d.density_map.ForwardFFT();
	binning_factor = high_resolution_limit / pixel_size / 2.0;
	fourier_size_x = int(input_3d.density_map.logical_x_dimension / binning_factor);
	if (! IsEven(fourier_size_x)) fourier_size_x++;
	fourier_size_x += 2;
	fourier_size_y = int(input_3d.density_map.logical_y_dimension / binning_factor);
	if (! IsEven(fourier_size_y)) fourier_size_y++;
	fourier_size_y += 2;
	fourier_size_z = int(input_3d.density_map.logical_z_dimension / binning_factor);
	if (! IsEven(fourier_size_z)) fourier_size_z++;
	fourier_size_z += 2;
// The following line assumes that we have a cubic volume
	binning_factor = float(input_3d.density_map.logical_x_dimension) / float(fourier_size_x);

	if (binning_factor != 1.0 )
	{
		input_3d.density_map.Resize(fourier_size_x, fourier_size_y, fourier_size_z);
		input_3d.pixel_size *= binning_factor;
		pixel_size *= binning_factor;
	}

	wxPrintf("\nBinning factor = %f, new pixel size = %f\n", binning_factor, pixel_size);
	input_3d.density_map.SwapRealSpaceQuadrants();
	comparison_object.reference_volume = &input_3d;
	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	comparison_object.particle_image = &particle_image;
	particle_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	comparison_object.CTF_image = &ctf_image;
	ctf_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	snr_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	comparison_object.projection_image = &projection_image;
	projection_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	comparison_object.particle_parameters = &my_parameters;
	comparison_object.parameter_constraints = &my_constraints;
	comparison_object.input_parameters = input_parameters;
	comparison_object.temp_float = temp_float;
	comparison_object.parameter_map = parameter_map;
	unbinned_image.Allocate(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, true);
	final_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);

// Read whole parameter file to work out average values and variances
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters);
		for (i = 1; i < parameters_per_line; i++)
		{
			parameter_average[i] += input_parameters[i];
			parameter_variance[i] += powf(input_parameters[i],2);
		}
	}
	for (i = 1; i < parameters_per_line; i++)
	{
		parameter_average[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] -= powf(parameter_average[i],2);
	}
	wxPrintf("\nAverage sigma noise = %f, average score = %f\nAverage ShiftX = %f, average ShiftY = %f\nSigma ShiftX = %f, sigma ShiftY = %f\nNumber of particles to refine = %i\n\n",
			parameter_average[13], parameter_average[14], parameter_average[4], parameter_average[5], sqrtf(parameter_variance[4]), sqrtf(parameter_variance[5]), my_input_par_file.number_of_lines);
	my_input_par_file.Rewind();

	// Set up the remainder of the comparison object
	mask_falloff = 20.0 / pixel_size;
	comparison_object.mask_radius = mask_radius / pixel_size;
	comparison_object.mask_falloff = mask_falloff;
	comparison_object.low_resolution_limit = pixel_size / low_resolution_limit;
	comparison_object.high_resolution_limit = pixel_size / high_resolution_limit;

	my_constraints.InitShiftX(parameter_average[4], parameter_variance[4], powf(parameter_average[13],2));
	my_constraints.InitShiftY(parameter_average[5], parameter_variance[5], powf(parameter_average[13],2));

//	cg_accuracy[0] = 1.0;
//	cg_accuracy[1] = 1.0;
//	cg_accuracy[2] = 1.0;
//	cg_accuracy[3] = 0.5;
//	cg_accuracy[4] = 0.5;
	temp_float[0] = target_phase_error / (1.0 / high_resolution_limit * 2.0 * PI * mask_radius);
	temp_float[1] = target_phase_error / (1.0 / high_resolution_limit * 2.0 * PI * mask_radius);
	temp_float[2] = target_phase_error / (1.0 / high_resolution_limit * 2.0 * PI * mask_radius);
	temp_float[3] = deg_2_rad(target_phase_error) / (1.0 / high_resolution_limit * 2.0 * PI * pixel_size);
	temp_float[4] = deg_2_rad(target_phase_error) / (1.0 / high_resolution_limit * 2.0 * PI * pixel_size);
	number_of_search_dimensions = MapParameters(temp_float, cg_accuracy, parameter_map, 5);

	ProgressBar *my_progress = new ProgressBar(my_input_par_file.number_of_lines);
//	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	for (current_image = 1; current_image <= 100; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters);
//		input_parameters[1] += global_random_number_generator.GetNormalRandom() * 5.0;
//		input_parameters[2] += global_random_number_generator.GetNormalRandom() * 5.0;
//		input_parameters[3] += global_random_number_generator.GetNormalRandom() * 5.0;
//		input_parameters[4] += global_random_number_generator.GetNormalRandom() * 5.0;
//		input_parameters[5] += global_random_number_generator.GetNormalRandom() * 5.0;
		for (i = 0; i < parameters_per_line; i++) {output_parameters[i] = input_parameters[i];}

		my_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, pixel_size, 0.0);
		ctf_image.CalculateCTFImage(my_ctf);

//		particle_image.ReadSlice(&input_stack, current_image);
//		particle_image.ForwardFFT();
		input_image.ReadSlice(&input_stack, current_image);
		input_image.ForwardFFT();
		input_image.ClipInto(&particle_image);
//		input_image.QuickAndDirtyWriteSlice("input_image.mrc",1);
//		particle_image.QuickAndDirtyWriteSlice("particle_image.mrc",1);
//		exit(0);
		particle_image.PhaseShift(-input_parameters[4] / pixel_size, -input_parameters[5] / pixel_size);
		particle_image.BackwardFFT();
		particle_image.CosineMask(mask_radius / pixel_size, mask_falloff);
		particle_image.ForwardFFT();
		particle_image.SwapRealSpaceQuadrants();
		particle_image.Whiten();
		snr_image.CopyFrom(&ctf_image);
		for (i = 1; i < snr_image.real_memory_allocated / 2; i++) {snr_image.complex_values[i] *= conjf(snr_image.complex_values[i]);}
		snr_image.MultiplyByWeightsCurve(input_3d.statistics.part_SSNR);
		particle_image.OptimalFilterBySNRImage(snr_image);

		if (number_of_search_dimensions > 0)
		{
			temp_float[0] = input_parameters[3];
			temp_float[1] = input_parameters[2];
			temp_float[2] = input_parameters[1];
			temp_float[3] = 0.0;
			temp_float[4] = 0.0;
			MapParameters(temp_float, cg_starting_point, parameter_map, 5);

			input_parameters[14] = - 100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction,&comparison_object,number_of_search_dimensions,cg_starting_point,cg_accuracy);
			output_parameters[14] = - 100.0 * conjugate_gradient_minimizer.Run();

			temp_float[0] = input_parameters[3];
			temp_float[1] = input_parameters[2];
			temp_float[2] = input_parameters[1];
			temp_float[3] = 0.0;
			temp_float[4] = 0.0;
			UnmapParameters(temp_float, conjugate_gradient_minimizer.GetPointerToBestValues(), parameter_map, 5);

			output_parameters[1] = temp_float[2];
			output_parameters[2] = temp_float[1];
			output_parameters[3] = temp_float[0];
			output_parameters[4] = temp_float[3] + input_parameters[4];
			output_parameters[5] = temp_float[4] + input_parameters[5];
			output_parameters[15] = output_parameters[14] - input_parameters[14];

		}
		else
		{
			temp_float[0] = input_parameters[3];
			temp_float[1] = input_parameters[2];
			temp_float[2] = input_parameters[1];
			temp_float[3] = 0.0;
			temp_float[4] = 0.0;
			input_parameters[14] = - 100.0 * FrealignObjectiveFunction(&comparison_object, temp_float);
			output_parameters[14] = input_parameters[14];
			output_parameters[15] = 0.0;
		}

		my_parameters.Init(output_parameters[3], output_parameters[2], output_parameters[1],
				output_parameters[4], output_parameters[5]);
		my_output_par_file.WriteLine(output_parameters);
		for (i = 1; i < parameters_per_line; i++) {output_parameters[i] -= input_parameters[i];}
		my_output_par_shifts_file.WriteLine(output_parameters);

		if (calculate_matching_projections)
		{
			input_3d.density_map.ExtractSlice(projection_image, my_parameters);
			projection_image.MultiplyPixelWise(ctf_image);
			projection_image.PhaseShift(my_parameters.ReturnShiftX() / pixel_size, my_parameters.ReturnShiftY() / pixel_size);
			projection_image.SwapRealSpaceQuadrants();
			projection_image.ClipInto(&unbinned_image);
			unbinned_image.BackwardFFT();
			unbinned_image.ClipInto(&final_image);
			final_image.WriteSlice(&output_file, current_image);
		}

		my_progress->Update(current_image);
	}
	delete my_progress;

	return true;
}
