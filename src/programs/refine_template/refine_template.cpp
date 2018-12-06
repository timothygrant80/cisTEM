#include "../../core/core_headers.h"


class
RefineTemplateApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

class TemplateComparisonObject
{
public:
	Image						*input_reconstruction, *windowed_particle, *projection_filter;
	AnglesAndShifts				*angles;
};

// This is the function which will be minimized
Peak TemplateScore(void *scoring_parameters)
{
	TemplateComparisonObject *comparison_object = reinterpret_cast < TemplateComparisonObject *> (scoring_parameters);
	Image current_projection;
	Peak box_peak;

	current_projection.Allocate(comparison_object->windowed_particle->logical_x_dimension, comparison_object->windowed_particle->logical_x_dimension, false);
	if (comparison_object->input_reconstruction->logical_x_dimension != comparison_object->windowed_particle->logical_x_dimension)
	{
		Image padded_projection;
		padded_projection.Allocate(comparison_object->input_reconstruction->logical_x_dimension, comparison_object->input_reconstruction->logical_x_dimension, false);
		comparison_object->input_reconstruction->ExtractSlice(padded_projection, *comparison_object->angles, 1.0f, false);
		padded_projection.SwapRealSpaceQuadrants();
		padded_projection.BackwardFFT();
		padded_projection.ClipInto(&current_projection);
		current_projection.ForwardFFT();
	}
	else
	{
		comparison_object->input_reconstruction->ExtractSlice(current_projection, *comparison_object->angles, 1.0f, false);
		current_projection.SwapRealSpaceQuadrants();
	}

	current_projection.MultiplyPixelWise(*comparison_object->projection_filter);
	current_projection.ZeroCentralPixel();
	current_projection.DivideByConstant(sqrtf(current_projection.ReturnSumOfSquares()));
#ifdef MKL
	// Use the MKL
	vmcMulByConj(current_projection.real_memory_allocated/2,reinterpret_cast <MKL_Complex8 *> (comparison_object->windowed_particle->complex_values),reinterpret_cast <MKL_Complex8 *> (current_projection.complex_values),reinterpret_cast <MKL_Complex8 *> (current_projection.complex_values),VML_EP|VML_FTZDAZ_ON|VML_ERRMODE_IGNORE);
#else
	for (pixel_counter = 0; pixel_counter < current_projection.real_memory_allocated / 2; pixel_counter ++)
	{
		current_projection.complex_values[pixel_counter] = conj(current_projection.complex_values[pixel_counter]) * comparison_object->windowed_particle->complex_values[pixel_counter];
	}
#endif
	current_projection.BackwardFFT();

	return current_projection.FindPeakWithIntegerCoordinates();
//	return box_peak.value;
}

IMPLEMENT_APP(RefineTemplateApp)


// override the DoInteractiveUserInput

void RefineTemplateApp::DoInteractiveUserInput()
{
	wxString	input_search_images;
	wxString	input_reconstruction;

	wxString    mip_input_filename;
	wxString    scaled_mip_input_filename;
	wxString    best_psi_input_filename;
	wxString    best_theta_input_filename;
	wxString    best_phi_input_filename;
	wxString    best_defocus_input_filename;
	wxString    best_psi_output_file;
	wxString    best_theta_output_file;
	wxString    best_phi_output_file;
	wxString    best_defocus_output_file;

	wxString    mip_output_file;
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
	float		angular_range = 2.0;
	float		angular_step = 5.0;
	int			best_parameters_to_keep = 20;
	float 		defocus_search_range = 1000;
	float 		defocus_search_step = 100;
	float 		defocus_refine_step = 5;
	float		padding = 1.0;
	bool		ctf_refinement = false;
	float		mask_radius = 0.0;
	wxString	my_symmetry = "C1";
	float 		in_plane_angular_step = 0;
	float 		wanted_threshold;
	float 		min_peak_radius;

	UserInput *my_input = new UserInput("RefineTemplate", 1.00);

	input_search_images = my_input->GetFilenameFromUser("Input images to be searched", "The input image stack, containing the images that should be searched", "my_image_stack.mrc", true);
	input_reconstruction = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	mip_input_filename = my_input->GetFilenameFromUser("Input MIP file", "The file with the maximum intensity projection image", "in_mip.mrc", false);
	scaled_mip_input_filename = my_input->GetFilenameFromUser("Input scaled MIP file", "The file with the scaled MIP (peak search done on this image)", "in_mip_scaled.mrc", false);
	best_psi_input_filename = my_input->GetFilenameFromUser("Input psi file", "The file with the best psi image", "in_psi.mrc", true);
	best_theta_input_filename = my_input->GetFilenameFromUser("Input theta file", "The file with the best psi image", "in_theta.mrc", true);
	best_phi_input_filename = my_input->GetFilenameFromUser("Input phi file", "The file with the best psi image", "in_phi.mrc", true);
	best_defocus_input_filename = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "in_defocus.mrc", true);
	mip_output_file = my_input->GetFilenameFromUser("Output MIP file", "The file for saving the maximum intensity projection image", "out_mip.mrc", false);
	scaled_mip_output_file = my_input->GetFilenameFromUser("Output Scaled MIP file", "The file for saving the maximum intensity projection image divided by correlation variance", "out_mip_scaled.mrc", false);
	best_psi_output_file = my_input->GetFilenameFromUser("Output psi file", "The file for saving the best psi image", "out_psi.mrc", false);
	best_theta_output_file = my_input->GetFilenameFromUser("Output theta file", "The file for saving the best psi image", "out_theta.mrc", false);
	best_phi_output_file = my_input->GetFilenameFromUser("Output phi file", "The file for saving the best psi image", "out_phi.mrc", false);
	best_defocus_output_file = my_input->GetFilenameFromUser("Output defocus file", "The file for saving the best defocus image", "out_defocus.mrc", false);
	wanted_threshold = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
	min_peak_radius = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 0.0);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7");
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	defocus1 = my_input->GetFloatFromUser("Defocus1 (angstroms)", "Defocus1 for the input image", "10000", 0.0);
	defocus2 = my_input->GetFloatFromUser("Defocus2 (angstroms)", "Defocus2 for the input image", "10000", 0.0);
	defocus_angle = my_input->GetFloatFromUser("Defocus Angle (degrees)", "Defocus Angle for the input image", "0.0");
	phase_shift = my_input->GetFloatFromUser("Phase Shift (degrees)", "Additional phase shift in degrees", "0.0");
//	low_resolution_limit = my_input->GetFloatFromUser("Low resolution limit (A)", "Low resolution limit of the data used for alignment in Angstroms", "300.0", 0.0);
//	high_resolution_limit = my_input->GetFloatFromUser("High resolution limit (A)", "High resolution limit of the data used for alignment in Angstroms", "8.0", 0.0);
//	angular_range = my_input->GetFloatFromUser("Angular refinement range", "AAngular range to refine", "2.0", 0.1);
	angular_step = my_input->GetFloatFromUser("Out of plane angular step", "Angular step size for global grid search", "0.1", 0.01);
	in_plane_angular_step = my_input->GetFloatFromUser("In plane angular step", "Angular step size for in-plane rotations during the search", "0.1", 0.01);
//	best_parameters_to_keep = my_input->GetIntFromUser("Number of top hits to refine", "The number of best global search orientations to refine locally", "20", 1);
	defocus_search_range = my_input->GetFloatFromUser("Defocus search range (A)", "Search range (-value ... + value) around current defocus", "1000.0", 0.0);
	defocus_search_step = my_input->GetFloatFromUser("Defocus search step (A) (0.0 = no search)", "Step size used in the defocus search", "100.0", 0.0);
	defocus_refine_step = my_input->GetFloatFromUser("Defocus refine step (A) (0.0 = no refinement)", "Step size used in the defocus refinement", "5.0", 0.0);
	padding = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "2.0", 1.0);
//	ctf_refinement = my_input->GetYesNoFromUser("Refine defocus", "Should the particle defocus be refined?", "No");
	mask_radius = my_input->GetFloatFromUser("Mask radius (A) (0.0 = no mask)", "Radius of a circular mask to be applied to the input particles during refinement", "0.0", 0.0);
//	my_symmetry = my_input->GetSymmetryFromUser("Template symmetry", "The symmetry of the template reconstruction", "C1");


	int first_search_position = -1;
	int last_search_position = -1;
	int image_number_for_gui = 0;
	int number_of_jobs_per_image_in_gui = 0;

	wxString directory_for_results = "/dev/null"; // shouldn't be used in interactive

	delete my_input;

	my_current_job.Reset(42);
	my_current_job.ManualSetArguments("ttfffffffffffiffffbffttttttttttttfftfiiiit",	input_search_images.ToUTF8().data(),
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
															angular_range,
															angular_step,
															best_parameters_to_keep,
															defocus_search_range,
															defocus_search_step,
															defocus_refine_step,
															padding,
															ctf_refinement,
															mask_radius,
															phase_shift,
															mip_input_filename.ToUTF8().data(),
															scaled_mip_input_filename.ToUTF8().data(),
															best_psi_input_filename.ToUTF8().data(),
															best_theta_input_filename.ToUTF8().data(),
															best_phi_input_filename.ToUTF8().data(),
															best_defocus_input_filename.ToUTF8().data(),
															best_psi_output_file.ToUTF8().data(),
															best_theta_output_file.ToUTF8().data(),
															best_phi_output_file.ToUTF8().data(),
															best_defocus_output_file.ToUTF8().data(),
															mip_output_file.ToUTF8().data(),
															scaled_mip_output_file.ToUTF8().data(),
															wanted_threshold,
															min_peak_radius,
															my_symmetry.ToUTF8().data(),
															in_plane_angular_step,
															first_search_position,
															last_search_position,
															image_number_for_gui,
															number_of_jobs_per_image_in_gui,
															directory_for_results.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool RefineTemplateApp::DoCalculation()
{
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
	float		angular_range = my_current_job.arguments[11].ReturnFloatArgument();
	float		angular_step = my_current_job.arguments[12].ReturnFloatArgument();
	int			best_parameters_to_keep = my_current_job.arguments[13].ReturnIntegerArgument();
	float 		defocus_search_range = my_current_job.arguments[14].ReturnFloatArgument();
	float 		defocus_search_step = my_current_job.arguments[15].ReturnFloatArgument();
	float 		defocus_refine_step = my_current_job.arguments[16].ReturnFloatArgument();
	float		padding = my_current_job.arguments[17].ReturnFloatArgument();
	bool		ctf_refinement = my_current_job.arguments[18].ReturnBoolArgument();
	float		mask_radius = my_current_job.arguments[19].ReturnFloatArgument();
	float 		phase_shift = my_current_job.arguments[20].ReturnFloatArgument();
	wxString    mip_input_filename = my_current_job.arguments[21].ReturnStringArgument();
	wxString    scaled_mip_input_filename = my_current_job.arguments[22].ReturnStringArgument();
	wxString    best_psi_input_filename = my_current_job.arguments[23].ReturnStringArgument();
	wxString    best_theta_input_filename = my_current_job.arguments[24].ReturnStringArgument();
	wxString    best_phi_input_filename = my_current_job.arguments[25].ReturnStringArgument();
	wxString    best_defocus_input_filename = my_current_job.arguments[26].ReturnStringArgument();
	wxString    best_psi_output_file = my_current_job.arguments[27].ReturnStringArgument();
	wxString    best_theta_output_file = my_current_job.arguments[28].ReturnStringArgument();
	wxString    best_phi_output_file = my_current_job.arguments[29].ReturnStringArgument();
	wxString    best_defocus_output_file = my_current_job.arguments[30].ReturnStringArgument();
	wxString    mip_output_file = my_current_job.arguments[31].ReturnStringArgument();
	wxString    scaled_mip_output_file = my_current_job.arguments[32].ReturnStringArgument();
	float		wanted_threshold = my_current_job.arguments[33].ReturnFloatArgument();
	float		min_peak_radius = my_current_job.arguments[34].ReturnFloatArgument();
	wxString 	my_symmetry = my_current_job.arguments[35].ReturnStringArgument();
	float		in_plane_angular_step = my_current_job.arguments[36].ReturnFloatArgument();
	int 		first_search_position = my_current_job.arguments[37].ReturnIntegerArgument();
	int 		last_search_position = my_current_job.arguments[38].ReturnIntegerArgument();
	int 		image_number_for_gui = my_current_job.arguments[39].ReturnIntegerArgument();
	int 		number_of_jobs_per_image_in_gui = my_current_job.arguments[40].ReturnIntegerArgument();
	wxString	directory_for_results = my_current_job.arguments[41].ReturnStringArgument();

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
	wxPrintf("mask search radius = %f\n", mask_radius);
	wxPrintf("phase shift = %f\n", phase_shift);
	wxPrintf("symmetry = %s\n", my_symmetry);
	wxPrintf("in plane step = %f\n", in_plane_angular_step);
	wxPrintf("first location = %i\n", first_search_position);
	wxPrintf("last location = %i\n", last_search_position);
	*/

	bool parameter_map[5]; // needed for euler search init
	for (int i = 0; i < 5; i++) {parameter_map[i] = true;}

	float outer_mask_radius;

	float temp_float;
	double temp_double_array[5];

	int number_of_rotations;
	long total_correlation_positions;
	long current_correlation_position;
	long pixel_counter;

	int current_x;
	int current_y;

	int phi_i;
	int theta_i;
	int psi_i;
	int defocus_i;
	int defocus_is;

	AnglesAndShifts angles;
	TemplateComparisonObject template_object;

	ImageFile input_search_image_file;
	ImageFile mip_input_file;
	ImageFile scaled_mip_input_file;
	ImageFile best_psi_input_file;
	ImageFile best_theta_input_file;
	ImageFile best_phi_input_file;
	ImageFile best_defocus_input_file;
	ImageFile input_reconstruction_file;

	Curve whitening_filter;
	Curve number_of_terms;

	input_search_image_file.OpenFile(input_search_images_filename.ToStdString(), false);
	mip_input_file.OpenFile(mip_input_filename.ToStdString(), false);
	scaled_mip_input_file.OpenFile(scaled_mip_input_filename.ToStdString(), false);
	best_psi_input_file.OpenFile(best_psi_input_filename.ToStdString(), false);
	best_theta_input_file.OpenFile(best_theta_input_filename.ToStdString(), false);
	best_phi_input_file.OpenFile(best_phi_input_filename.ToStdString(), false);
	best_defocus_input_file.OpenFile(best_defocus_input_filename.ToStdString(), false);
	input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString(), false);

	Image input_image;
	Image windowed_particle;
	Image padded_reference;
	Image input_reconstruction;
	Image current_projection;
	Image padded_projection;

	Image projection_filter;

	Image mip_image;
	Image scaled_mip_image;
	Image psi_image;
	Image theta_image;
	Image phi_image;
	Image defocus_image;
	Image best_psi;
	Image best_theta;
	Image best_phi;
	Image best_defocus;
	Image best_mip;
	Image best_scaled_mip;

	Peak current_peak;
	Peak template_peak;
	long current_address;
	long address_offset;

	float current_phi;
	float current_theta;
	float current_psi;
	float current_defocus;
	float best_score;
	float score;

	float best_phi_score;
	float best_theta_score;
	float best_psi_score;
	float best_defocus_score;
	int ii, jj, kk, ll;
	float mult_i;

	int number_of_peaks_found = 0;
	float mask_falloff = 20.0;

	input_image.ReadSlice(&input_search_image_file, 1);
	mip_image.ReadSlice(&mip_input_file, 1);
	scaled_mip_image.ReadSlice(&scaled_mip_input_file, 1);
	psi_image.ReadSlice(&best_psi_input_file, 1);
	theta_image.ReadSlice(&best_theta_input_file, 1);
	phi_image.ReadSlice(&best_phi_input_file, 1);
	defocus_image.ReadSlice(&best_defocus_input_file, 1);
	padded_reference.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_mip.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_psi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_theta.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_phi.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);
	best_defocus.Allocate(input_image.logical_x_dimension, input_image.logical_y_dimension, 1);

	best_psi.SetToConstant(0.0f);
	best_theta.SetToConstant(0.0f);
	best_phi.SetToConstant(0.0f);
	best_defocus.SetToConstant(0.0f);

// Some settings for testing
//	padding = 2.0f;
//	ctf_refinement = true;
//	defocus_search_range = 200.0f;
//	defocus_step = 50.0f;

	input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices());
	if (padding != 1.0f)
	{
		input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges());
	}
	input_reconstruction.ForwardFFT();
	//input_reconstruction.CosineMask(0.1, 0.01, true);
	//input_reconstruction.Whiten();
	//if (first_search_position == 0) input_reconstruction.QuickAndDirtyWriteSlices("/tmp/filter.mrc", 1, input_reconstruction.logical_z_dimension);
	input_reconstruction.ZeroCentralPixel();
	input_reconstruction.SwapRealSpaceQuadrants();

	CTF input_ctf;
	input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, 0.0, 0.0, 0.0, pixel_size, deg_2_rad(phase_shift));

	// assume cube

	windowed_particle.Allocate(input_reconstruction_file.ReturnXSize(), input_reconstruction_file.ReturnXSize(), false);
	current_projection.Allocate(input_reconstruction_file.ReturnXSize(), input_reconstruction_file.ReturnXSize(), false);
	projection_filter.Allocate(input_reconstruction_file.ReturnXSize(), input_reconstruction_file.ReturnXSize(), false);
	if (padding != 1.0f) padded_projection.Allocate(input_reconstruction_file.ReturnXSize() * padding, input_reconstruction_file.ReturnXSize() * padding, false);

	temp_float = (float(input_reconstruction_file.ReturnXSize()) / 2.0f - 1.0f) * pixel_size;
	if (mask_radius > temp_float) mask_radius = temp_float;

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

	input_image.ApplyCurveFilter(&whitening_filter);
	input_image.ZeroCentralPixel();
	input_image.DivideByConstant(sqrt(input_image.ReturnSumOfSquares()));
	input_image.BackwardFFT();

	// count total searches (lazy)

	total_correlation_positions = 0;
	current_correlation_position = 0;

	// if running locally, search over all of them

	best_scaled_mip.CopyFrom(&scaled_mip_image);
	current_peak.value = FLT_MAX;
	wxPrintf("\n");
	while (current_peak.value >= wanted_threshold)
	{
		// look for a peak..

		current_peak = best_scaled_mip.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, input_reconstruction_file.ReturnXSize() / 2 + 1);
		if (current_peak.value < wanted_threshold) break;

		// ok we have peak..

		number_of_peaks_found++;

		// get angles and mask out the local area so it won't be picked again..

		float sq_dist_x, sq_dist_y;
		long address = 0;

		int i,j;

		current_peak.x = current_peak.x + best_scaled_mip.physical_address_of_box_center_x;
		current_peak.y = current_peak.y + best_scaled_mip.physical_address_of_box_center_y;

//		wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

		for ( j = 0; j < best_scaled_mip.logical_y_dimension; j ++ )
		{
			sq_dist_y = float(pow(j-current_peak.y, 2));
			for ( i = 0; i < best_scaled_mip.logical_x_dimension; i ++ )
			{
				sq_dist_x = float(pow(i-current_peak.x,2));

				// The square centered at the pixel
				if ( sq_dist_x + sq_dist_y <= min_peak_radius )
				{
					best_scaled_mip.real_values[address] = -FLT_MAX;
				}

				if (sq_dist_x == 0.0f && sq_dist_y == 0.0f)
				{
					current_phi = phi_image.real_values[address];
					current_theta = theta_image.real_values[address];
					current_psi = psi_image.real_values[address];
					current_defocus = defocus_image.real_values[address];
				}

				address++;
			}
			address += best_scaled_mip.padding_jump_value;
		}

		wxPrintf("Peak %i at x, y, psi, theta, phi, defocus = %f, %f, %f, %f, %f, %f : %f\n", number_of_peaks_found, current_peak.x, current_peak.y, current_psi, current_theta, current_phi, current_defocus, current_peak.value);
	}

	if (defocus_search_step <= 0.0)
	{
		defocus_search_range = 0.0f;
		defocus_search_step = 100.0f;
	}

	number_of_rotations = (2 * myroundint(float(angular_range)/float(angular_step)) + 1) * (2 * myroundint(float(angular_range)/float(angular_step)) + 1) * (2 * myroundint(float(angular_range)/float(in_plane_angular_step)) + 1);
	total_correlation_positions = number_of_peaks_found * number_of_rotations * (2 * myroundint(float(defocus_search_range)/float(defocus_search_step)) + 1);

//	ProgressBar *my_progress;

	if (is_running_locally == true)
	{
		wxPrintf("\nSearching %i positions in the MIP.\n", number_of_peaks_found);
//		wxPrintf("Searching %i rotations per peak.\n", number_of_rotations);
//		wxPrintf("Calculating %li correlation total.\n\n", total_correlation_positions);

		wxPrintf("Performing Refinement...\n\n");
//		my_progress = new ProgressBar(total_correlation_positions);
	}
	current_peak.value = FLT_MAX;
	best_mip.CopyFrom(&mip_image);
	best_scaled_mip.CopyFrom(&scaled_mip_image);
	best_psi.CopyFrom(&psi_image);
	best_theta.CopyFrom(&theta_image);
	best_phi.CopyFrom(&phi_image);
	best_defocus.CopyFrom(&defocus_image);
	number_of_peaks_found = 0;

	template_object.input_reconstruction = &input_reconstruction;
	template_object.windowed_particle = &windowed_particle;
	template_object.projection_filter = &projection_filter;
	template_object.angles = &angles;

	while (current_peak.value >= wanted_threshold)
	{
		// look for a peak..

		current_peak = scaled_mip_image.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, input_reconstruction_file.ReturnXSize() / 2 + 1);
		if (current_peak.value < wanted_threshold) break;

		// ok we have peak..

		padded_reference.CopyFrom(&input_image);
		padded_reference.RealSpaceIntegerShift(current_peak.x, current_peak.y);
		padded_reference.ClipInto(&windowed_particle);
		if (mask_radius > 0.0f) windowed_particle.CosineMask(mask_radius / pixel_size, mask_falloff / pixel_size);
		windowed_particle.ForwardFFT();
		windowed_particle.SwapRealSpaceQuadrants();

		number_of_peaks_found++;

		// get angles and mask out the local area so it won't be picked again..

		float sq_dist_x, sq_dist_y;
		long address = 0;

		int i,j;

		current_peak.x = current_peak.x + scaled_mip_image.physical_address_of_box_center_x;
		current_peak.y = current_peak.y + scaled_mip_image.physical_address_of_box_center_y;

//		wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

		for ( j = 0; j < scaled_mip_image.logical_y_dimension; j ++ )
		{
			sq_dist_y = float(pow(j-current_peak.y, 2));
			for ( i = 0; i < scaled_mip_image.logical_x_dimension; i ++ )
			{
				sq_dist_x = float(pow(i-current_peak.x,2));

				// The square centered at the pixel
				if ( sq_dist_x + sq_dist_y <= min_peak_radius )
				{
					scaled_mip_image.real_values[address] = -FLT_MAX;
				}

				if (sq_dist_x == 0.0f && sq_dist_y == 0.0f)
				{
					current_address = address;
					current_phi = phi_image.real_values[address];
					current_theta = theta_image.real_values[address];
					current_psi = psi_image.real_values[address];
					current_defocus = defocus_image.real_values[address];
					best_score = -FLT_MAX;

					wxPrintf("\nRefining peak %i at x, y = %f, %f\n", number_of_peaks_found, current_peak.x, current_peak.y);

//					for (defocus_is = 0; defocus_is <= myroundint(float(defocus_search_range)/float(defocus_search_step)); defocus_is = defocus_is - myroundint(float(4 * defocus_is - 1) / 2.0f))
/*					for (defocus_is = -myroundint(float(defocus_search_range)/float(defocus_search_step)); defocus_is <= myroundint(float(defocus_search_range)/float(defocus_search_step)); defocus_is++)
					{
						input_ctf.SetDefocus((defocus1 + current_defocus + defocus_i * defocus_refine_step + defocus_is * defocus_search_step) / pixel_size, (defocus2 + current_defocus + defocus_i * defocus_refine_step + defocus_is * defocus_search_step) / pixel_size, deg_2_rad(defocus_angle));
						projection_filter.CalculateCTFImage(input_ctf);
						projection_filter.ApplyCurveFilter(&whitening_filter);

						angles.Init(current_phi, current_theta, current_psi, 0.0, 0.0);

						template_peak = TemplateScore(&template_object);
						score = template_peak.value;
						if (score > best_score)
						{
							best_score = score;
							temp_float = best_scaled_mip.real_values[current_address] * best_score / best_mip.real_values[current_address] * sqrtf(current_projection.logical_x_dimension * current_projection.logical_y_dimension);
							wxPrintf("Value for dpsi, dtheta, dphi, ddefocus = %f, %f, %f, %f : %f\n", 0.0, 0.0, 0.0, defocus_is * defocus_search_step, temp_float);
//							best_psi.real_values[current_address] = current_psi + psi_i * in_plane_angular_step;
//							best_theta.real_values[current_address] = current_theta + theta_i * angular_step;
//							best_phi.real_values[current_address] = current_phi + phi_i * angular_step;
							best_defocus.real_values[current_address] = current_defocus + defocus_is * defocus_search_step;
						}
					}
					if (defocus_search_step > 0.0) defocus_is = myroundint(best_defocus.real_values[current_address] - current_defocus) / defocus_search_step;
					current_defocus = defocus_image.real_values[current_address];
*/
					// Do local search
//					best_score = -FLT_MAX;
					phi_i = 0;
					theta_i = 0;
					psi_i = 0;
					defocus_i = 0;
//					score = best_score;
					for (mult_i = 17.0f; mult_i > 0.5f; mult_i /= 2.0f)
					{
						for (ll = 0; ll < 2; ll = -2 * ll + 1)
						{
							if ((ll != 0) && (defocus_refine_step == 0.0f)) break;
							do
							{
								best_defocus_score = best_score;
								defocus_i += myroundint(mult_i * ll);

								// make the projection filter, which will be CTF * whitening filter
								input_ctf.SetDefocus((defocus1 + current_defocus + defocus_i * defocus_refine_step) / pixel_size, (defocus2 + current_defocus + defocus_i * defocus_refine_step) / pixel_size, deg_2_rad(defocus_angle));
								projection_filter.CalculateCTFImage(input_ctf);
								projection_filter.ApplyCurveFilter(&whitening_filter);

								for (kk = 0; kk < 2; kk = -2 * kk + 1)
								{
									do
									{
										best_phi_score = best_score;
										phi_i += kk;
										for (jj = 0; jj < 2; jj = -2 * jj + 1)
										{
											do
											{
												best_theta_score = best_score;
												theta_i += jj;
												for (ii = 0; ii < 2; ii = -2 * ii + 1)
												{
													do
													{
														best_psi_score = best_score;
														psi_i += ii;

														angles.Init(current_phi + phi_i * angular_step, current_theta + theta_i * angular_step, current_psi + psi_i * in_plane_angular_step, 0.0, 0.0);

														template_peak = TemplateScore(&template_object);
														score = template_peak.value;
														if (score > best_score)
														{
															best_score = score;
															address_offset = (best_scaled_mip.logical_x_dimension + best_scaled_mip.padding_jump_value) * myroundint(template_peak.y) + myroundint(template_peak.x);
															temp_float = best_scaled_mip.real_values[current_address] * best_score / best_mip.real_values[current_address] * sqrtf(current_projection.logical_x_dimension * current_projection.logical_y_dimension);
															wxPrintf("Value for dpsi, dtheta, dphi, ddefocus = %f, %f, %f, %f : %f\n", psi_i * in_plane_angular_step, theta_i * angular_step, phi_i * angular_step, defocus_i * defocus_refine_step + defocus_is * defocus_search_step, temp_float);
															best_psi.real_values[current_address + address_offset] = current_psi + psi_i * in_plane_angular_step;
															best_theta.real_values[current_address + address_offset] = current_theta + theta_i * angular_step;
															best_phi.real_values[current_address + address_offset] = current_phi + phi_i * angular_step;
															best_defocus.real_values[current_address + address_offset] = current_defocus + defocus_i * defocus_refine_step;
														}
													} while (best_score > best_psi_score);
													psi_i -= ii;
												}
											} while (best_score > best_theta_score);
											theta_i -= jj;
										}
									} while (best_score > best_phi_score);
									phi_i -= kk;
								}
							} while (best_score > best_defocus_score);
							defocus_i -= ll;
						}
					}
				}
				address++;
			}
			address += scaled_mip_image.padding_jump_value;
		}
		best_scaled_mip.real_values[current_address + address_offset] *= best_score / best_mip.real_values[current_address] * sqrtf(current_projection.logical_x_dimension * current_projection.logical_y_dimension);
		best_mip.real_values[current_address + address_offset] = best_score * sqrtf(padded_reference.logical_x_dimension * padded_reference.logical_y_dimension);
	}

	if (is_running_locally == true)
	{
		best_mip.QuickAndDirtyWriteSlice(mip_output_file.ToStdString(), 1, true, pixel_size);
		best_scaled_mip.QuickAndDirtyWriteSlice(scaled_mip_output_file.ToStdString(), 1, true, pixel_size);
		best_psi.QuickAndDirtyWriteSlice(best_psi_output_file.ToStdString(), 1, true, pixel_size);
		best_theta.QuickAndDirtyWriteSlice(best_theta_output_file.ToStdString(), 1, true, pixel_size);
		best_phi.QuickAndDirtyWriteSlice(best_phi_output_file.ToStdString(), 1, true, pixel_size);
		best_defocus.QuickAndDirtyWriteSlice(best_defocus_output_file.ToStdString(), 1, true, pixel_size);
	}

	if (is_running_locally == true)
	{
		wxPrintf("\nRefine Template: Normal termination\n");
		wxDateTime finish_time = wxDateTime::Now();
		wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
	}

	return true;
}
