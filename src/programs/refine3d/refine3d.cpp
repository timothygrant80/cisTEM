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
	for (int i = 0; i < comparison_object->particle->number_of_parameters; i++) {comparison_object->particle->temp_float[i] = comparison_object->particle->input_parameters[i];}
	comparison_object->particle->UnmapParameters(array_of_values);

	comparison_object->reference_volume->CalculateProjection(*comparison_object->projection_image, *comparison_object->particle->ctf_image,
			comparison_object->particle->alignment_parameters, comparison_object->particle->mask_radius, comparison_object->particle->mask_falloff,
			comparison_object->particle->pixel_size / comparison_object->particle->filter_radius_high, false);

	return 	- comparison_object->particle->particle_image->GetWeightedCorrelationWithImage(*comparison_object->projection_image, comparison_object->particle->bin_index)
			- comparison_object->particle->ReturnParameterPenalty(comparison_object->particle->temp_float);
}

IMPLEMENT_APP(Refine3DApp)

// override the DoInteractiveUserInput

void Refine3DApp::DoInteractiveUserInput()
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
	float		mask_radius_search = 0.0;
	float		high_resolution_limit_search = 20.0;
	float		angular_step = 5.0;
	int			best_parameters_to_keep = 20;
	float		padding = 1.0;
	float		mask_center_2d_x = 100.0;
	float		mask_center_2d_y = 100.0;
	float		mask_center_2d_z = 100.0;
	float		mask_radius_2d = 100.0;
	wxString	my_symmetry = "C1";
	bool		global_search = false;
	bool		local_refinement = true;
	bool		refine_psi = true;
	bool		refine_theta = true;
	bool		refine_phi = true;
	bool		refine_x = true;
	bool		refine_y = true;
	bool		calculate_matching_projections = false;
	bool		apply_2D_masking = false;

	UserInput *my_input = new UserInput("Refine3D", 1.0);

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
	mask_radius = my_input->GetFloatFromUser("Mask radius for refinement (A)", "Radius of a circular mask to be applied to the input images during refinement", "100.0", 0.0);
	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
	mask_radius_search = my_input->GetFloatFromUser("Mask radius for global search (A) (0.0 = max)", "Radius of a circular mask to be applied to the input images during global search", "100.0", 0.0);
	high_resolution_limit_search = my_input->GetFloatFromUser("Approximate high resolution limit for search (A)", "High resolution limit of the data used in the global search in Angstroms", "20.0", 0.0);
	angular_step = my_input->GetFloatFromUser("Angular step (0.0 = set automatically)", "Angular step size for global grid search", "0.0", 0.0);
	best_parameters_to_keep = my_input->GetFloatFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "1", 20);
	padding = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
	my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
	mask_center_2d_x = my_input->GetFloatFromUser("2D mask X coordinate (A)", "X coordinate of 2D mask center", "100.0", 0.0);
	mask_center_2d_y = my_input->GetFloatFromUser("2D mask Y coordinate (A)", "Y coordinate of 2D mask center", "100.0", 0.0);
	mask_center_2d_z = my_input->GetFloatFromUser("2D mask Z coordinate (A)", "Z coordinate of 2D mask center", "100.0", 0.0);
	mask_radius_2d = my_input->GetFloatFromUser("2D mask radius (A)", "Radius of a circular mask to be used for likelihood calculation", "100.0", 0.0);
	global_search = my_input->GetYesNoFromUser("Global search", "Should a global search be performed before local refinement?", "No");
	local_refinement = my_input->GetYesNoFromUser("Local refinement", "Should a local parameter refinement be performed?", "Yes");
	refine_psi = my_input->GetYesNoFromUser("Refine Psi", "Should the Psi Euler angle be refined (parameter 1)?", "Yes");
	refine_theta = my_input->GetYesNoFromUser("Refine Theta", "Should the Theta Euler angle be refined (parameter 2)?", "Yes");
	refine_phi = my_input->GetYesNoFromUser("Refine Phi", "Should the Phi Euler angle be refined (parameter 3)?", "Yes");
	refine_x = my_input->GetYesNoFromUser("Refine ShiftX", "Should the X shift be refined (parameter 4)?", "Yes");
	refine_y = my_input->GetYesNoFromUser("Refine ShiftY", "Should the Y shift be refined (parameter 5)?", "Yes");
	calculate_matching_projections = my_input->GetYesNoFromUser("Calculate matching projections", "Should matching projections be calculated?", "No");
	apply_2D_masking = my_input->GetYesNoFromUser("Apply 2D masking", "Should 2D masking be used for the likelihood calculation?", "No");

	delete my_input;

	my_current_job.Reset(33);
	my_current_job.ManualSetArguments("tttttttffffffffffiftffffbbbbbbbbb",	input_parameter_file.ToUTF8().data(),
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
																			mask_radius, low_resolution_limit, high_resolution_limit,
																			mask_radius_search, high_resolution_limit_search, angular_step, best_parameters_to_keep,
																			padding,
																			my_symmetry.ToUTF8().data(),
																			mask_center_2d_x, mask_center_2d_y, mask_center_2d_z, mask_radius_2d,
																			global_search, local_refinement,
																			refine_psi, refine_theta, refine_phi, refine_x, refine_y,
																			calculate_matching_projections,
																			apply_2D_masking);
}

// override the do calculation method which will be what is actually run..

bool Refine3DApp::DoCalculation()
{
	Particle my_particle;
	Particle search_particle;

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
	float    mask_radius_search					= my_current_job.arguments[14].ReturnFloatArgument();
	float	 high_resolution_limit_search		= my_current_job.arguments[15].ReturnFloatArgument();
	float	 angular_step						= my_current_job.arguments[16].ReturnFloatArgument();
	int		 best_parameters_to_keep			= my_current_job.arguments[17].ReturnIntegerArgument();
	float	 padding							= my_current_job.arguments[18].ReturnFloatArgument();
	wxString my_symmetry						= my_current_job.arguments[19].ReturnStringArgument();
	float	 mask_center_2d_x					= my_current_job.arguments[20].ReturnFloatArgument();
	float	 mask_center_2d_y					= my_current_job.arguments[21].ReturnFloatArgument();
	float	 mask_center_2d_z					= my_current_job.arguments[22].ReturnFloatArgument();
	float	 mask_radius_2d						= my_current_job.arguments[23].ReturnFloatArgument();
	bool	 global_search						= my_current_job.arguments[24].ReturnBoolArgument();
	bool	 local_refinement					= my_current_job.arguments[25].ReturnBoolArgument();
// Psi, Theta, Phi, ShiftX, ShiftY
	my_particle.parameter_map[3]				= my_current_job.arguments[26].ReturnBoolArgument();
	my_particle.parameter_map[2]				= my_current_job.arguments[27].ReturnBoolArgument();
	my_particle.parameter_map[1]				= my_current_job.arguments[28].ReturnBoolArgument();
	my_particle.parameter_map[4]				= my_current_job.arguments[29].ReturnBoolArgument();
	my_particle.parameter_map[5]				= my_current_job.arguments[30].ReturnBoolArgument();
	bool 	 calculate_matching_projections		= my_current_job.arguments[31].ReturnBoolArgument();
	bool	 apply_2D_masking					= my_current_job.arguments[32].ReturnBoolArgument();

	my_particle.constraints_used[4] = true;		// Constraint for X shifts
	my_particle.constraints_used[5] = true;		// Constraint for Y shifts

	Image input_image;
//	Image ctf_image;
	Image ctf_input_image;
	Image projection_image;
	Image search_projection_image;
	Image unbinned_image;
	Image final_image;
	Image temp_image;
//	Image temp_image2;
//	Image reference_3d_search;
	Image *projection_cache = NULL;
	CTF   my_ctf;
	CTF   my_input_ctf;
	Image snr_image;
//	Curve particle_ssnr;
	ReconstructedVolume			input_3d;
	ReconstructedVolume			search_reference_3d;
	ImageProjectionComparison	comparison_object;
	ConjugateGradient			conjugate_gradient_minimizer;
	EulerSearch					global_euler_search;
	Kernel2D					**kernel_index = NULL;
	float particle_weight;
	float particle_score;
	float image_sigma_noise;
	float particle_occupancy;

	int i;
	int j;
	int psi_i;
	int fourier_size_x, fourier_size_y, fourier_size_z;
	int current_image;
//	int parameters_per_line = my_particle.number_of_parameters;
	int number_of_independent_pixels;
	float input_parameters[my_particle.number_of_parameters];
	float output_parameters[my_particle.number_of_parameters];
	float search_parameters[my_particle.number_of_parameters];
	float parameter_average[my_particle.number_of_parameters];
	float parameter_variance[my_particle.number_of_parameters];
	float cg_starting_point[my_particle.number_of_parameters];
	float cg_accuracy[my_particle.number_of_parameters];
	float binning_factor;
	float binning_factor_search;
	float pixel_size_search;
	float mask_falloff = 20.0;
	float alpha;
	float sigma;
	float variance_unfiltered;
	float variance_filtered;
	float variance_difference;
	float logp;
	float pixel_size_binned;
	float rotated_center_x;
	float rotated_center_y;
	float rotated_center_z;
	float temp_float;
	float psi;
	float psi_step;
//	float best_parameters_to_keep = 20;
	wxDateTime my_time_in;
	wxDateTime my_time_out;

	ZeroFloatArray(input_parameters, my_particle.number_of_parameters);
	ZeroFloatArray(output_parameters, my_particle.number_of_parameters);
	ZeroFloatArray(search_parameters, my_particle.number_of_parameters);
	ZeroFloatArray(parameter_average, my_particle.number_of_parameters);
	ZeroFloatArray(parameter_variance, my_particle.number_of_parameters);
	ZeroFloatArray(cg_starting_point, my_particle.number_of_parameters);
	ZeroFloatArray(cg_accuracy, my_particle.number_of_parameters);

	NumericTextFile my_input_par_file(input_parameter_file, OPEN_TO_READ);
	MRCFile input_stack(input_particle_images.ToStdString(), false);
	MRCFile input_file(input_reconstruction.ToStdString(), false);
	MRCFile output_file(ouput_matching_projections.ToStdString(), true);
	NumericTextFile my_output_par_file(ouput_parameter_file, OPEN_TO_WRITE, 16);
	my_output_par_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP NormSigmaNoise          Score         Change");
	NumericTextFile my_output_par_shifts_file(ouput_shift_file, OPEN_TO_WRITE, 15);
	my_output_par_shifts_file.WriteCommentLine("C    Particle#            Psi          Theta            Phi         ShiftX         ShiftY            Mag     Micrograph       Defocus1       Defocus2       AstigAng      Occupancy           LogP NormSigmaNoise          Score");

	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, my_symmetry);
	input_3d.ReadStatisticsFromFile(input_reconstruction_statistics);
	input_3d.density_map.ReadSlices(&input_file,1,input_3d.density_map.logical_z_dimension);
	MyDebugAssertTrue(input_3d.density_map.IsCubic(), "3D reference not cubic");

	if (global_search)
	{
		search_reference_3d.InitWithDimensions(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, input_3d.density_map.logical_z_dimension, pixel_size, my_symmetry);
//		search_reference_3d.density_map.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, input_3d.density_map.logical_z_dimension, false);
		search_reference_3d.density_map.CopyFrom(&input_3d.density_map);
		search_reference_3d.density_map.Correct3D();
		search_reference_3d.density_map.ForwardFFT();
		binning_factor_search = high_resolution_limit_search / pixel_size / 2.0;
		fourier_size_x = ReturnClosestFactorizedLower(search_reference_3d.density_map.logical_x_dimension / binning_factor_search, 3, true);
		fourier_size_y = ReturnClosestFactorizedLower(search_reference_3d.density_map.logical_y_dimension / binning_factor_search, 3, true);
		fourier_size_z = ReturnClosestFactorizedLower(search_reference_3d.density_map.logical_z_dimension / binning_factor_search, 3, true);
	// The following line assumes that we have a cubic volume
		binning_factor_search = float(search_reference_3d.density_map.logical_x_dimension) / float(fourier_size_x);
		if (binning_factor_search != 1.0 )
		{
			search_reference_3d.density_map.Resize(fourier_size_x, fourier_size_y, fourier_size_z);
			search_reference_3d.pixel_size *= binning_factor_search;
		}
		search_reference_3d.density_map.SwapRealSpaceQuadrants();
		search_particle.Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension);
//		for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.parameter_map[i] = my_particle.parameter_map[i];}
//		for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.constraints_used[i] = my_particle.constraints_used[i];}
		search_reference_3d.statistics.part_SSNR = input_3d.statistics.part_SSNR;
		search_projection_image.Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension, false);
		pixel_size_search = pixel_size * binning_factor_search;
		if (angular_step <= 0) angular_step = 360.0 * high_resolution_limit_search / PI / mask_radius;
//		psi_step = angular_step / 5.0;
		psi_step = rad_2_deg(pixel_size_search / mask_radius);
		psi_step = 360.0 / int(360.0 / psi_step);
//		psi_step = 1.0;
		wxPrintf("\nBinning factor for search = %f, new pixel size = %f, resolution limit = %f\nAngular step size = %f, in-plane = %f\n", binning_factor_search, pixel_size_search, pixel_size_search * 2.0, angular_step, psi_step);
	}

	if (padding != 1.0)
	{
		input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension * padding, input_3d.density_map.logical_y_dimension * padding, input_3d.density_map.logical_z_dimension * padding, input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
		input_3d.statistics.part_SSNR.ResampleCurve(&input_3d.statistics.part_SSNR, input_3d.statistics.part_SSNR.number_of_points * padding);
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
	}
	pixel_size_binned = pixel_size * binning_factor;
	wxPrintf("\nBinning factor for refinement = %f, new pixel size = %f\n", binning_factor, pixel_size_binned);

	input_3d.density_map.SwapRealSpaceQuadrants();
//	comparison_object.reference_volume = &input_3d;
	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);
	my_particle.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension);
//	comparison_object.particle = &my_particle;
	ctf_input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);
//	comparison_object.projection_image = &projection_image;
	projection_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	unbinned_image.Allocate(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, true);
	final_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	temp_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
//	temp_image2.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);

	mask_center_2d_x = mask_center_2d_x / pixel_size - input_image.physical_address_of_box_center_x;
	mask_center_2d_y = mask_center_2d_y / pixel_size - input_image.physical_address_of_box_center_y;
	mask_center_2d_z = mask_center_2d_z / pixel_size - input_image.physical_address_of_box_center_z;
	mask_radius_2d = mask_radius_2d / pixel_size;

// Read whole parameter file to work out average values and variances
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		for (i = 0; i < my_particle.number_of_parameters; i++)
		{
			parameter_average[i] += input_parameters[i];
			parameter_variance[i] += powf(input_parameters[i],2);
		}
	}
	for (i = 0; i < my_particle.number_of_parameters; i++)
	{
		parameter_average[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] -= powf(parameter_average[i],2);

		if (parameter_variance[i] < 0.001) my_particle.constraints_used[i] = false;
	}
	my_particle.SetParameterStatistics(parameter_average, parameter_variance);

	if (global_search)
	{
		for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.parameter_map[i] = my_particle.parameter_map[i];}
		for (i = 0; i < search_particle.number_of_parameters; i++) {search_particle.constraints_used[i] = my_particle.constraints_used[i];}
		search_particle.SetParameterStatistics(parameter_average, parameter_variance);

		global_euler_search.InitGrid(angular_step, psi_step, pixel_size_search / high_resolution_limit_search, search_particle.parameter_map, best_parameters_to_keep);
//		my_input_par_file.Rewind();
//		for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
//		{
//			my_input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
//			global_euler_search.list_of_search_parameters[current_image - 1][0] = input_parameters[1];
//			global_euler_search.list_of_search_parameters[current_image - 1][1] = input_parameters[2];
//		}
		projection_cache = new Image [global_euler_search.number_of_search_positions];
		for (i = 0; i < global_euler_search.number_of_search_positions; i++)
		{
			projection_cache[i].Allocate(search_reference_3d.density_map.logical_x_dimension, search_reference_3d.density_map.logical_y_dimension, false);
		}
		search_projection_image.RotateFourier2DGenerateIndex(kernel_index, 360.0, psi_step);
		search_reference_3d.density_map.GenerateReferenceProjections(projection_cache, global_euler_search);
	}

	wxPrintf("\nAverage sigma noise = %f, average score = %f\nAverage ShiftX = %f, average ShiftY = %f\nSigma ShiftX = %f, sigma ShiftY = %f\nNumber of particles to refine = %i\n",
			parameter_average[13], parameter_average[14], parameter_average[4], parameter_average[5], sqrtf(parameter_variance[4]), sqrtf(parameter_variance[5]), my_input_par_file.number_of_lines);
	my_input_par_file.Rewind();

	ProgressBar *my_progress = new ProgressBar(my_input_par_file.number_of_lines);
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
//	for (current_image = 1; current_image <= 10; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		for (i = 0; i < my_particle.number_of_parameters; i++) {output_parameters[i] = input_parameters[i];}

// Set up Particle object
		my_particle.ResetImageFlags();
		my_particle.mask_radius = mask_radius;
		my_particle.mask_falloff = mask_falloff;
		my_particle.filter_radius_low = low_resolution_limit;
		my_particle.filter_radius_high = high_resolution_limit;
		// The following line would allow using particles with different pixel sizes
		my_particle.pixel_size = pixel_size_binned;
		my_particle.sigma_noise = input_parameters[13] * sqrtf(pixel_size_binned / pixel_size);
		my_particle.SetParameters(input_parameters);
//		my_particle.MapParameters(cg_starting_point);
		my_particle.MapParameterAccuracy(cg_accuracy);
		my_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10]);

		my_input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, pixel_size, 0.0);
		ctf_input_image.CalculateCTFImage(my_input_ctf);

		input_image.ReadSlice(&input_stack, int(input_parameters[0] + 0.5));
		input_image.ClipInto(&unbinned_image);
		unbinned_image.ForwardFFT();
		unbinned_image.ClipInto(my_particle.particle_image);

		my_particle.PhaseShiftInverse();
		my_particle.CosineMask();
		my_particle.PhaseShift();
		my_particle.CenterInCorner();
		my_particle.WeightBySSNR(input_3d.statistics.part_SSNR);

		if ((my_particle.number_of_search_dimensions > 0) && (global_search || local_refinement))
		{
			wxPrintf("starting refinement for particle %i\n", current_image);
			if (global_search)
			{
				my_time_in = wxDateTime::UNow();
//				for (i = 0; i < search_particle.number_of_parameters; i++) {search_parameters[i] = input_parameters[i];}
				search_particle.ResetImageFlags();
				if (mask_radius_search == 0.0)
				{
					search_particle.mask_radius = search_particle.particle_image->logical_x_dimension / 2 * pixel_size_search - mask_falloff;
				}
				else
				{
					search_particle.mask_radius = mask_radius_search;
				}
				search_particle.mask_falloff = mask_falloff;
				search_particle.filter_radius_low = low_resolution_limit;
				search_particle.filter_radius_high = high_resolution_limit_search;
				search_particle.pixel_size = pixel_size_search;
				search_particle.sigma_noise = input_parameters[13] * sqrtf(pixel_size_search / pixel_size);
//				search_particle.MapParameters(cg_starting_point);
//				search_particle.MapParameterAccuracy(cg_accuracy);
				search_particle.number_of_search_dimensions = my_particle.number_of_search_dimensions;
				search_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10]);
				temp_image.CopyFrom(&input_image);
				temp_image.ForwardFFT();
				temp_image.ClipInto(search_particle.particle_image);
				search_particle.PhaseShiftInverse();
				search_particle.CosineMask();
				search_particle.PhaseShift();
				search_particle.CenterInCorner();

				search_particle.WeightBySSNR(search_reference_3d.statistics.part_SSNR);
//				search_particle.particle_image->QuickAndDirtyWriteSlice("junk.mrc", 1);
//				exit(0);
				global_euler_search.Run(search_particle, search_reference_3d.density_map, projection_cache, kernel_index);
//				search_particle.UnmapParametersToExternal(output_parameters, global_euler_search.list_of_best_parameters[0]);

//				for (i = 1; i <= 6; i++) {output_parameters[i] = global_euler_search.list_of_best_parameters[0][i - 1];}
//				wxPrintf("%i, %f, %f, %f\n",current_image, output_parameters[3], output_parameters[2], output_parameters[1]);
				for (i = 0; i < search_particle.number_of_parameters; i++) {search_parameters[i] = input_parameters[i];}
				search_particle.SetParameterConstraints(powf(parameter_average[13],2));
				comparison_object.reference_volume = &search_reference_3d;
				comparison_object.projection_image = &search_projection_image;
				comparison_object.particle = &search_particle;
				search_particle.SetIndexForWeightedCorrelation();
				output_parameters[14] = - std::numeric_limits<float>::max();
				for (i = 0; i < best_parameters_to_keep; i++)
				{
					for (j = 1; j <= 6; j++) {search_parameters[j] = global_euler_search.list_of_best_parameters[i][j - 1];}
//					if (i == best_parameters_to_keep - 1) {for (j = 0; j < search_particle.number_of_parameters; j++) {search_parameters[j] = input_parameters[j];};}
					search_particle.SetParameters(search_parameters);
					search_particle.MapParameters(cg_starting_point);
					search_particle.mask_radius = mask_radius;
//					wxPrintf("i = %i, old score = %f, start = %f, %f, %f, %f, %f\n", i, global_euler_search.list_of_best_parameters[i][5], cg_starting_point[0],cg_starting_point[1],cg_starting_point[2],cg_starting_point[3],cg_starting_point[4]);
					search_parameters[14] = - 100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, search_particle.number_of_search_dimensions, cg_starting_point, cg_accuracy);
					temp_float = - 100.0 * conjugate_gradient_minimizer.Run();
//					wxPrintf("i = %i, old score = %f, new score = %f\n\n", i, search_parameters[14], temp_float);
					if (temp_float > output_parameters[14])
					{
						wxPrintf("i = %i, old score = %f, new score = %f\n", i, search_parameters[14], temp_float);
						search_particle.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues());
						output_parameters[14] = temp_float;
					}
				}
				my_particle.SetParameters(output_parameters);
				my_time_out = wxDateTime::UNow(); wxPrintf("global search done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
			}

			if (local_refinement)
			{
				my_time_in = wxDateTime::UNow();
				comparison_object.reference_volume = &input_3d;
				comparison_object.projection_image = &projection_image;
				comparison_object.particle = &my_particle;
				my_particle.SetIndexForWeightedCorrelation();
				my_particle.MapParameters(cg_starting_point);

				my_particle.SetParameterConstraints(powf(parameter_average[13],2));

				input_parameters[14] = - 100.0 * conjugate_gradient_minimizer.Init(&FrealignObjectiveFunction, &comparison_object, my_particle.number_of_search_dimensions, cg_starting_point, cg_accuracy);
				output_parameters[14] = - 100.0 * conjugate_gradient_minimizer.Run();

				my_particle.UnmapParametersToExternal(output_parameters, conjugate_gradient_minimizer.GetPointerToBestValues());

				output_parameters[15] = output_parameters[14] - input_parameters[14];
				my_time_out = wxDateTime::UNow(); wxPrintf("local refinement done: ms taken = %li\n", my_time_out.Subtract(my_time_in).GetMilliseconds());
			}
		}
		else
		{
			input_parameters[14] = - 100.0 * FrealignObjectiveFunction(&comparison_object, cg_starting_point);
			output_parameters[14] = input_parameters[14]; output_parameters[15] = 0.0;
		}

		my_particle.SetAlignmentParameters(output_parameters[1], output_parameters[2], output_parameters[3], output_parameters[4], output_parameters[5]);
		my_particle.CalculateProjection(projection_image, input_3d);
		projection_image.ClipInto(&unbinned_image);
		unbinned_image.BackwardFFT();
		unbinned_image.ClipInto(&final_image);
		alpha = input_image.ReturnImageScale(final_image, mask_radius / pixel_size);
		final_image.MultiplyByConstant(alpha);
//		temp_image2.CopyFrom(&final_image);
//		temp_image2.PhaseFlipPixelWise(ctf_input_image);

		temp_image.CopyFrom(&input_image);
		temp_image.PhaseFlipPixelWise(ctf_input_image);
		temp_image.SubtractImage(&final_image);
// Here one can whiten with an average noise power spectrum that can probably be approximated by the data power spectrum (assuming the signal is weak)
//		input_image.ForwardFFT();
//		temp_image.ForwardFFT();
//		input_image.WhitenTwo(temp_image);
//		input_image.BackwardFFT();
//		temp_image.BackwardFFT();
		if (apply_2D_masking)
		{
			my_particle.euler_matrix->RotateCoords(mask_center_2d_x, mask_center_2d_y, mask_center_2d_z, rotated_center_x, rotated_center_y, rotated_center_z);
			variance_filtered = input_image.ReturnVarianceOfRealValues(mask_radius_2d, rotated_center_x, rotated_center_y, rotated_center_z);
			variance_difference = temp_image.ReturnVarianceOfRealValues(mask_radius_2d, rotated_center_x, rotated_center_y, rotated_center_z);
			sigma = sqrtf(variance_difference / final_image.ReturnVarianceOfRealValues(mask_radius_2d, rotated_center_x, rotated_center_y, rotated_center_z));
		}
		else
		{
			variance_filtered = input_image.ReturnVarianceOfRealValues(mask_radius / pixel_size);
			variance_difference = temp_image.ReturnVarianceOfRealValues(mask_radius / pixel_size);
			sigma = sqrtf(variance_difference / final_image.ReturnVarianceOfRealValues(mask_radius / pixel_size));
		}

		number_of_independent_pixels = my_particle.mask_volume * powf(pixel_size_binned / pixel_size,2);
		logp = - number_of_independent_pixels * variance_difference / variance_filtered / 2.0 + my_particle.ReturnParameterLogP(output_parameters);

		output_parameters[12] = logp;
		output_parameters[13] = sigma;

		temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		temp_float = output_parameters[1]; output_parameters[1] = output_parameters[3]; output_parameters[3] = temp_float;
		my_output_par_file.WriteLine(output_parameters);
		for (i = 1; i < my_particle.number_of_parameters; i++) {output_parameters[i] -= input_parameters[i];}
		my_output_par_shifts_file.WriteLine(output_parameters);

		if (calculate_matching_projections)
		{
			final_image.WriteSlice(&output_file, int(input_parameters[0] + 0.5));
		}

		my_progress->Update(current_image);
	}
	delete my_progress;
//	delete global_euler_search;
	if (global_search)
	{
		delete [] projection_cache;
		search_projection_image.RotateFourier2DDeleteIndex(kernel_index, 360, psi_step);
	}

	return true;
}
