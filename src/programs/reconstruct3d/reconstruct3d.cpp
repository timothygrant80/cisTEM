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
	float		resolution_limit = 0.0;
	float		score_bfactor_conversion = 5.0;
	float		score_threshold = 0.0;
	bool		normalize_particles = true;
	bool		adjust_scores = true;
	bool		invert_contrast = false;
	bool		crop_images = false;
	bool		dump_arrays = false;
	wxString	dump_file_1;
	wxString	dump_file_2;

	UserInput *my_input = new UserInput("Reconstruct3D", 1.02);

	input_particle_stack = my_input->GetFilenameFromUser("Input particle images", "The input particle image stack, containing the 2D images for each particle in the dataset", "my_particle_stack.mrc", true);
	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
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
	resolution_limit = my_input->GetFloatFromUser("Resolution limit (A) (0.0 = max)", "The resolution to which the reconstruction will be calculated, in Angstroms", "0.0", 0.0);
	score_bfactor_conversion = my_input->GetFloatFromUser("Particle weighting factor (A^2)", "Constant to convert particle scores to B-factors in squared Angstroms", "5.0", 0.0);
	score_threshold = my_input->GetFloatFromUser("Score threshold (< 1 = percentage)", "Minimum score to include a particle in the reconstruction", "0.0", 0.0);
	normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "Should the input particle images be normalized to have a constant variance", "Yes");
	adjust_scores = my_input->GetYesNoFromUser("Adjust scores for defocus dependence", "Should the particle scores be adjusted internally to reduce their dependence on defocus", "Yes");
	invert_contrast = my_input->GetYesNoFromUser("Invert particle contrast", "Should the contrast in the particle images be inverted?", "No");
	crop_images = my_input->GetYesNoFromUser("Crop particle images", "Should the particle images be cropped to speed up computation?", "No");
	dump_arrays = my_input->GetYesNoFromUser("Dump intermediate arrays (merge later)", "Should the 3D reconstruction arrays be dumped to a file for later merging with other jobs", "No");
	dump_file_1 = my_input->GetFilenameFromUser("Output dump filename for odd particles", "The name of the first dump file with the intermediate reconstruction arrays", "dump_file_1.dat", false);
	dump_file_2 = my_input->GetFilenameFromUser("Output dump filename for even particles", "The name of the second dump file with the intermediate reconstruction arrays", "dump_file_2.dat", false);

	delete my_input;

	my_current_job.Reset(26);
	my_current_job.ManualSetArguments("tttttttiiffffffffffbbbbbtt",	input_particle_stack.ToUTF8().data(),
																	input_parameter_file.ToUTF8().data(),
																	output_reconstruction_1.ToUTF8().data(),
																	output_reconstruction_2.ToUTF8().data(),
																	output_reconstruction_filtered.ToUTF8().data(),
																	output_resolution_statistics.ToUTF8().data(),
																	my_symmetry.ToUTF8().data(),
																	first_particle, last_particle,
																	pixel_size, voltage_kV, spherical_aberration_mm, amplitude_contrast,
																	molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
																	resolution_limit, score_bfactor_conversion, score_threshold,
																	normalize_particles, adjust_scores,
																	invert_contrast, crop_images, dump_arrays,
																	dump_file_1.ToUTF8().data(),
																	dump_file_2.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool Reconstruct3DApp::DoCalculation()
{
	wxString input_particle_stack 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_parameter_file 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString output_reconstruction_1			= my_current_job.arguments[2].ReturnStringArgument();
	wxString output_reconstruction_2			= my_current_job.arguments[3].ReturnStringArgument();
	wxString output_reconstruction_filtered		= my_current_job.arguments[4].ReturnStringArgument();
	wxString output_resolution_statistics		= my_current_job.arguments[5].ReturnStringArgument();
	wxString my_symmetry						= my_current_job.arguments[6].ReturnStringArgument();
	int		 first_particle						= my_current_job.arguments[7].ReturnIntegerArgument();
	int		 last_particle						= my_current_job.arguments[8].ReturnIntegerArgument();
	float 	 pixel_size							= my_current_job.arguments[9].ReturnFloatArgument();
	float    voltage_kV							= my_current_job.arguments[10].ReturnFloatArgument();
	float 	 spherical_aberration_mm			= my_current_job.arguments[11].ReturnFloatArgument();
	float    amplitude_contrast					= my_current_job.arguments[12].ReturnFloatArgument();
	float 	 molecular_mass_kDa					= my_current_job.arguments[13].ReturnFloatArgument();
	float    inner_mask_radius					= my_current_job.arguments[14].ReturnFloatArgument();
	float    outer_mask_radius					= my_current_job.arguments[15].ReturnFloatArgument();
	float    resolution_limit					= my_current_job.arguments[16].ReturnFloatArgument();
	float    score_bfactor_conversion			= my_current_job.arguments[17].ReturnFloatArgument();
	float    score_threshold					= my_current_job.arguments[18].ReturnFloatArgument();
	bool	 normalize_particles				= my_current_job.arguments[19].ReturnBoolArgument();
	bool	 adjust_scores						= my_current_job.arguments[20].ReturnBoolArgument();
	bool	 invert_contrast					= my_current_job.arguments[21].ReturnBoolArgument();
	bool	 crop_images						= my_current_job.arguments[22].ReturnBoolArgument();
	bool	 dump_arrays						= my_current_job.arguments[23].ReturnBoolArgument();
	wxString dump_file_1 						= my_current_job.arguments[24].ReturnStringArgument();
	wxString dump_file_2 						= my_current_job.arguments[25].ReturnStringArgument();

	ReconstructedVolume output_3d(molecular_mass_kDa);
	ReconstructedVolume output_3d1(molecular_mass_kDa);
	ReconstructedVolume output_3d2(molecular_mass_kDa);
	Image 				temp_image;
	Image 				temp2_image;
	Image				current_ctf_image;
	CTF					current_ctf;
	CTF					input_ctf;

	int i;
	int current_image;
	int images_to_process = 0;
	int image_counter = 0;
	int box_size;
	int original_box_size;
	int intermediate_box_size;
	float temp_float;
	float mask_volume_fraction;
	float mask_falloff = 10.0;
	float particle_area_in_pixels;
	float binning_factor = 1.0;
	float original_pixel_size = pixel_size;
	wxDateTime my_time_in;

	MRCFile input_file(input_particle_stack.ToStdString(), false);
	FrealignParameterFile my_input_par_file(input_parameter_file, OPEN_TO_READ);
	my_input_par_file.ReadFile();
	NumericTextFile output_statistics_file(output_resolution_statistics, OPEN_TO_WRITE, 7);
/*
	if (is_running_locally == false)
	{
		float result;
		my_result.SetResult(1, &result);
	}*/
	if (input_file.ReturnXSize() != input_file.ReturnYSize())
	{
		MyPrintWithDetails("Error: Particles are not square\n");
		SendError("Error: Particles are not square");
		abort();
	}
	if (last_particle < first_particle && last_particle != 0)
	{
		MyPrintWithDetails("Error: Number of last particle to refine smaller than number of first particle to refine\n");
		SendError("Error: Number of last particle to refine smaller than number of first particle to refine");
		abort();
	}

	if (last_particle == 0) last_particle = input_file.ReturnZSize();
	if (first_particle == 0) first_particle = 1;
	if (last_particle > input_file.ReturnZSize()) last_particle = input_file.ReturnZSize();

	my_input_par_file.CalculateDefocusDependence();
	if (adjust_scores) my_input_par_file.AdjustScores();

	if (score_threshold > 0.0 && score_threshold < 1.0) score_threshold = my_input_par_file.ReturnThreshold(score_threshold);

	my_time_in = wxDateTime::Now();
	output_statistics_file.WriteCommentLine("C Refine3D run date and time:              " + my_time_in.FormatISOCombined(' '));
	output_statistics_file.WriteCommentLine("C Input particle images:                   " + input_particle_stack);
	output_statistics_file.WriteCommentLine("C Input Frealign parameter filename:       " + input_parameter_file);
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
	output_statistics_file.WriteCommentLine("C Resolution limit (A):                    " + wxString::Format("%f", resolution_limit));
	output_statistics_file.WriteCommentLine("C Particle weighting factor (A^2):         " + wxString::Format("%f", score_bfactor_conversion));
	output_statistics_file.WriteCommentLine("C Score threshold:                         " + wxString::Format("%f", score_threshold));
	output_statistics_file.WriteCommentLine("C Normalize particles:                     " + BoolToYesNo(normalize_particles));
	output_statistics_file.WriteCommentLine("C Adjust scores for defocus dependence:    " + BoolToYesNo(adjust_scores));
	output_statistics_file.WriteCommentLine("C Invert particle contrast:                " + BoolToYesNo(invert_contrast));
	output_statistics_file.WriteCommentLine("C Crop particle images:                    " + BoolToYesNo(crop_images));
	output_statistics_file.WriteCommentLine("C Dump intermediate arrays:                " + BoolToYesNo(dump_arrays));
	output_statistics_file.WriteCommentLine("C Output dump filename for odd particles:  " + dump_file_1);
	output_statistics_file.WriteCommentLine("C Output dump filename for even particles: " + dump_file_2);
	output_statistics_file.WriteCommentLine("C");

	original_box_size = input_file.ReturnXSize();
	// If resolution limit higher that Nyquist, do not do binning
	if (resolution_limit < 2.0 * pixel_size) resolution_limit = 0.0;

	// Assume square particles and cubic volumes
	if (resolution_limit != 0.0)
	{
		binning_factor = resolution_limit / pixel_size / 2.0;
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
	current_ctf_image.Allocate(original_box_size, original_box_size, 1, false);
	temp_image.Allocate(original_box_size, original_box_size, true);
	if (resolution_limit != 0.0 && crop_images) temp2_image.Allocate(intermediate_box_size, intermediate_box_size, true);
	float input_parameters[input_particle.number_of_parameters];
	float parameter_average[input_particle.number_of_parameters];
	float parameter_variance[input_particle.number_of_parameters];
	ZeroFloatArray(input_parameters, input_particle.number_of_parameters);
	ZeroFloatArray(parameter_average, input_particle.number_of_parameters);
	ZeroFloatArray(parameter_variance, input_particle.number_of_parameters);

// Read whole parameter file to work out average image_sigma_noise and average score
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters); temp_float = input_parameters[1]; input_parameters[1] = input_parameters[3]; input_parameters[3] = temp_float;
		for (i = 0; i < input_particle.number_of_parameters; i++)
		{
			parameter_average[i] += input_parameters[i];
			parameter_variance[i] += powf(input_parameters[i],2);
		}
		if (input_parameters[0] >= first_particle && input_parameters[0] <= last_particle && input_parameters[14] >= score_threshold) images_to_process++;
	}
	for (i = 0; i < input_particle.number_of_parameters; i++)
	{
		parameter_average[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] /= my_input_par_file.number_of_lines;
		parameter_variance[i] -= powf(parameter_average[i],2);

	}
	input_particle.SetParameterStatistics(parameter_average, parameter_variance);

	Reconstruct3D my_reconstruction_1(box_size, box_size, box_size, pixel_size, parameter_average[11], parameter_average[14], score_bfactor_conversion, my_symmetry);
	Reconstruct3D my_reconstruction_2(box_size, box_size, box_size, pixel_size, parameter_average[11], parameter_average[14], score_bfactor_conversion, my_symmetry);
	my_reconstruction_1.original_x_dimension = original_box_size;
	my_reconstruction_1.original_y_dimension = original_box_size;
	my_reconstruction_1.original_z_dimension = original_box_size;
	my_reconstruction_1.original_pixel_size = original_pixel_size;
	my_reconstruction_2.original_x_dimension = original_box_size;
	my_reconstruction_2.original_y_dimension = original_box_size;
	my_reconstruction_2.original_z_dimension = original_box_size;
	my_reconstruction_2.original_pixel_size = original_pixel_size;

	wxPrintf("\nNumber of particles to reconstruct = %i, average sigma noise = %f, average score = %f\n", images_to_process, parameter_average[13], parameter_average[14]);
	wxPrintf("Box size for reconstruction = %i, binning factor = %f\n\n", box_size, binning_factor);
	my_input_par_file.Rewind();

	if (images_to_process == 0)
	{
		MyPrintWithDetails("Error: No particles to process\n");
		SendError("Error: No Particles to process");
		abort();
	}

	ProgressBar *my_progress = new ProgressBar(images_to_process);
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(input_parameters);
		if (input_parameters[0] < first_particle || input_parameters[0] > last_particle || input_parameters[14] <= score_threshold) continue;
		image_counter++;
		input_particle.location_in_stack = int(input_parameters[0] + 0.5);
		input_particle.pixel_size = pixel_size;

//		input_particle.ctf_parameters.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, pixel_size, 0.0);
		input_particle.InitCTFImage(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10]);

		// Assume square images
		if (crop_images)
		{
			input_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, original_pixel_size, 0.0);
			if (input_ctf.IsAlmostEqualTo(&current_ctf, 40.0 / pixel_size) == false)
			// Need to calculate current_ctf_image to be inserted into ctf_reconstruction
			{
				current_ctf = input_ctf;
				current_ctf_image.CalculateCTFImage(current_ctf);
			}
			temp_image.ReadSlice(&input_file, input_particle.location_in_stack);
			if (invert_contrast) temp_image.InvertRealValues();
			if (normalize_particles) temp_image.ZeroFloatAndNormalize(1.0, outer_mask_radius / original_pixel_size, true);
			temp_image.ForwardFFT();
			temp_image.PhaseFlipPixelWise(current_ctf_image);
			if (binning_factor != 1.0)
			{
				temp_image.ClipInto(&temp2_image);
				temp2_image.BackwardFFT();
				temp2_image.ClipInto(input_particle.particle_image);
				input_particle.particle_image->ForwardFFT();
			}
			else
			{
				temp_image.BackwardFFT();
				temp_image.ClipInto(input_particle.particle_image);
				input_particle.particle_image->ForwardFFT();
			}
		}
		else
		{
			if (binning_factor != 1.0)
			{
				temp_image.ReadSlice(&input_file, input_particle.location_in_stack);
				if (invert_contrast) temp_image.InvertRealValues();
				if (normalize_particles) temp_image.ZeroFloatAndNormalize(1.0, outer_mask_radius / pixel_size, true);
				temp_image.ForwardFFT();
				temp_image.ClipInto(input_particle.particle_image);
			}
			else
			{
				input_particle.particle_image->ReadSlice(&input_file, input_particle.location_in_stack);
				if (invert_contrast) input_particle.particle_image->InvertRealValues();
				if (normalize_particles) input_particle.particle_image->ZeroFloatAndNormalize(1.0, outer_mask_radius / pixel_size, true);
			}
			input_particle.PhaseFlipImage();
		}
		input_particle.particle_image->SwapRealSpaceQuadrants();

		input_particle.alignment_parameters.Init(input_parameters[3], input_parameters[2], input_parameters[1], input_parameters[4], input_parameters[5]);

		input_particle.particle_score = input_parameters[14];
		input_particle.particle_occupancy = input_parameters[11];
		input_particle.sigma_noise = input_parameters[13];
		if (current_image % 2 == 0)
		{
			input_particle.insert_even = true;
		}
		else
		{
			input_particle.insert_even = false;
		}
		if (input_particle.sigma_noise <= 0.0) input_particle.sigma_noise = parameter_average[13];

		if (input_particle.insert_even)
		{
			my_reconstruction_2.InsertSliceWithCTF(input_particle);
		}
		else
		{
			my_reconstruction_1.InsertSliceWithCTF(input_particle);
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

	output_3d.FinalizeOptimal(my_reconstruction_1, output_3d1.density_map, output_3d2.density_map,
			original_pixel_size, pixel_size, inner_mask_radius, outer_mask_radius, mask_falloff,
			output_reconstruction_filtered, output_statistics_file);

	wxPrintf("\nReconstruct3D: Normal termination\n\n");

	return true;
}
