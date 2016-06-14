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
	float		score_bfactor_conversion = 5.0;
	float		score_threshold = 0.0;
	bool		normalize_particles = true;
	bool		adjust_scores = true;
	bool		dump_arrays = false;
	wxString	dump_file_1;
	wxString	dump_file_2;

	UserInput *my_input = new UserInput("Reconstruct3D", 1.01);

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
	score_bfactor_conversion = my_input->GetFloatFromUser("Particle weighting factor (A^2)", "Constant to convert particle scores to B-factors in squared Angstroms", "5.0", 0.0);
	score_threshold = my_input->GetFloatFromUser("Score threshold (< 1 = percentage)", "Minimum score to include a particle in the reconstruction", "0.0", 0.0);
	normalize_particles = my_input->GetYesNoFromUser("Normalize particles", "Should the input particle images be normalized to have a constant variance", "Yes");
	adjust_scores = my_input->GetYesNoFromUser("Adjust scores for defocus dependence", "Should the particle scores be adjusted internally to reduce their dependence on defocus", "Yes");
	dump_arrays = my_input->GetYesNoFromUser("Dump intermediate arrays (merge later)", "Should the 3D reconstruction arrays be dumped to a file for later merging with other jobs", "No");
	dump_file_1 = my_input->GetFilenameFromUser("Output dump filename for odd particles", "The name of the first dump file with the intermediate reconstruction arrays", "dump_file_1.dat", false);
	dump_file_2 = my_input->GetFilenameFromUser("Output dump filename for even particles", "The name of the second dump file with the intermediate reconstruction arrays", "dump_file_2.dat", false);

	delete my_input;

	my_current_job.Reset(23);
	my_current_job.ManualSetArguments("tttttttiifffffffffbbbtt",	input_particle_stack.ToUTF8().data(),
																	input_parameter_file.ToUTF8().data(),
																	output_reconstruction_1.ToUTF8().data(),
																	output_reconstruction_2.ToUTF8().data(),
																	output_reconstruction_filtered.ToUTF8().data(),
																	output_resolution_statistics.ToUTF8().data(),
																	my_symmetry.ToUTF8().data(),
																	first_particle, last_particle,
																	pixel_size, voltage_kV, spherical_aberration_mm, amplitude_contrast,
																	molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
																	score_bfactor_conversion, score_threshold,
																	normalize_particles, adjust_scores,
																	dump_arrays,
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
	float    score_bfactor_conversion			= my_current_job.arguments[16].ReturnFloatArgument();
	float    score_threshold					= my_current_job.arguments[17].ReturnFloatArgument();
	bool	 normalize_particles				= my_current_job.arguments[18].ReturnBoolArgument();
	bool	 adjust_scores						= my_current_job.arguments[19].ReturnBoolArgument();
	bool	 dump_arrays						= my_current_job.arguments[20].ReturnBoolArgument();
	wxString dump_file_1 						= my_current_job.arguments[21].ReturnStringArgument();
	wxString dump_file_2 						= my_current_job.arguments[22].ReturnStringArgument();

	ReconstructedVolume output_3d;
	ReconstructedVolume output_3d1;
	ReconstructedVolume output_3d2;

	int current_image;
	int images_to_process = 0;
	int image_counter = 0;
	float temp_float[50];
	float average_score = 0.0;
	float average_sigma = 0.0;
	float average_occupancy = 0.0;
	float mask_volume_fraction;
	float pssnr_scale_factor;
	float mask_falloff = 10.0;
	float particle_area_in_pixels;
	wxDateTime my_time_in;

	MRCFile input_file(input_particle_stack.ToStdString(), false);
	MRCFile output_file;
	FrealignParameterFile my_input_par_file(input_parameter_file, OPEN_TO_READ);
	my_input_par_file.ReadFile();
	NumericTextFile output_statistics_file(output_resolution_statistics, OPEN_TO_WRITE, 7);

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
	output_statistics_file.WriteCommentLine("C Particle weighting factor (A^2):         " + wxString::Format("%f", score_bfactor_conversion));
	output_statistics_file.WriteCommentLine("C Score threshold:                         " + wxString::Format("%f", score_threshold));
	output_statistics_file.WriteCommentLine("C Normalize particles:                     " + BoolToYesNo(normalize_particles));
	output_statistics_file.WriteCommentLine("C Adjust scores for defocus dependence:    " + BoolToYesNo(adjust_scores));
	output_statistics_file.WriteCommentLine("C Dump intermediate arrays:                " + BoolToYesNo(dump_arrays));
	output_statistics_file.WriteCommentLine("C Output dump filename for odd particles:  " + dump_file_1);
	output_statistics_file.WriteCommentLine("C Output dump filename for even particles: " + dump_file_2);
	output_statistics_file.WriteCommentLine("C");

	Particle input_particle(input_file.ReturnXSize(), input_file.ReturnYSize());

// Read whole parameter file to work out average image_sigma_noise and average score
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(temp_float);
		average_occupancy += temp_float[11];
		average_sigma += temp_float[13];
		average_score += temp_float[14];
		if (temp_float[0] >= first_particle && temp_float[0] <= last_particle && temp_float[14] >= score_threshold) images_to_process++;
	}
	average_occupancy /= my_input_par_file.number_of_lines;
	average_sigma /= my_input_par_file.number_of_lines;
	average_score /= my_input_par_file.number_of_lines;

	Reconstruct3D my_reconstruction_1(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnXSize(), pixel_size, average_occupancy, average_score, score_bfactor_conversion, my_symmetry);
	Reconstruct3D my_reconstruction_2(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnXSize(), pixel_size, average_occupancy, average_score, score_bfactor_conversion, my_symmetry);

	wxPrintf("\nNumber of particles to reconstruct = %i, average sigma noise = %f, average score = %f\n\n", images_to_process, average_sigma, average_score);
	my_input_par_file.Rewind();

	if (images_to_process == 0)
	{
		MyPrintWithDetails("Error: No particles to process\n");
		abort();
	}

	ProgressBar *my_progress = new ProgressBar(images_to_process);
	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(temp_float);
		if (temp_float[0] < first_particle || temp_float[0] > last_particle || temp_float[14] < score_threshold) continue;
		image_counter++;
		input_particle.location_in_stack = int(temp_float[0] + 0.5);

		input_particle.particle_image->ReadSlice(&input_file, input_particle.location_in_stack);
//		mask_volume_2d = input_particle.particle_image->CosineMask(float(input_particle.particle_image->physical_address_of_box_center_x) - 6.0, 6.0);
		if (normalize_particles) input_particle.particle_image->ZeroFloatAndNormalize(1.0, outer_mask_radius / pixel_size, true);
		input_particle.pixel_size = pixel_size;

		input_particle.alignment_parameters.Init(temp_float[3], temp_float[2], temp_float[1], temp_float[4], temp_float[5]);
		input_particle.ctf_parameters.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, temp_float[8], temp_float[9], temp_float[10], 0.0, 0.0, 0.0, pixel_size, 0.0);

		input_particle.particle_score = temp_float[14];
		input_particle.particle_occupancy = temp_float[11];
		input_particle.sigma_noise = temp_float[13];
		if (current_image % 2 == 0)
		{
			input_particle.insert_even = true;
		}
		else
		{
			input_particle.insert_even = false;
		}
		if (input_particle.sigma_noise <= 0.0) input_particle.sigma_noise = average_sigma;

		my_progress->Update(image_counter);
		if (input_particle.insert_even)
		{
			my_reconstruction_2.InsertSliceWithCTF(input_particle);
		}
		else
		{
			my_reconstruction_1.InsertSliceWithCTF(input_particle);
		}
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

	output_3d1.InitWithReconstruct3D(my_reconstruction_1, pixel_size);
	output_3d1.Calculate3DSimple(my_reconstruction_1);
	output_3d1.density_map.SwapRealSpaceQuadrants();
	output_3d1.density_map.BackwardFFT();
	pssnr_scale_factor = output_3d1.Correct3D(outer_mask_radius / pixel_size);
	output_3d1.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, mask_falloff / pixel_size);
	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
	output_file.OpenFile(output_reconstruction_1.ToStdString(), true);
	output_3d1.density_map.WriteSlices(&output_file,1,output_3d1.density_map.logical_z_dimension);
	output_file.SetPixelSize(pixel_size);
	output_file.CloseFile();
	output_3d1.density_map.ForwardFFT();

	output_3d2.InitWithReconstruct3D(my_reconstruction_2, pixel_size);
	output_3d2.Calculate3DSimple(my_reconstruction_2);
	output_3d2.density_map.SwapRealSpaceQuadrants();
	output_3d2.density_map.BackwardFFT();
	pssnr_scale_factor = output_3d2.Correct3D(outer_mask_radius / pixel_size);
	output_3d2.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, mask_falloff / pixel_size);
	output_file.OpenFile(output_reconstruction_2.ToStdString(), true);
	output_3d2.density_map.WriteSlices(&output_file,1,output_3d2.density_map.logical_z_dimension);
	output_file.SetPixelSize(pixel_size);
	output_file.CloseFile();
	output_3d2.density_map.ForwardFFT();

	output_3d.InitWithReconstruct3D(my_reconstruction_1, pixel_size);
	output_3d.statistics.CalculateFSC(output_3d1.density_map, output_3d2.density_map);
//	my_reconstruction_1 += my_reconstruction_2;
//	pssnr_scale_factor = my_reconstruction_1.Correct3DCTF(output_3d2.density_map);
//	my_reconstruction_1.image_reconstruction.BackwardFFT();
//	pssnr_scale_factor = my_reconstruction_1.image_reconstruction.Correct3D();
//	my_reconstruction_1.image_reconstruction.ForwardFFT();

	output_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	output_3d.statistics.CalculateParticleFSCandSSNR(output_3d1.mask_volume_in_voxels, molecular_mass_kDa);
	particle_area_in_pixels = PI * powf(3.0 * (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3)) / 4.0 / PI, 2.0 / 3.0);
	mask_volume_fraction = output_3d1.density_map.logical_x_dimension * output_3d1.density_map.logical_y_dimension / particle_area_in_pixels
			* output_3d1.mask_volume_in_voxels / output_3d1.density_map.ReturnVolumeInRealSpace();
	output_3d.statistics.CalculateParticleSSNR(my_reconstruction_1.image_reconstruction, my_reconstruction_1.ctf_reconstruction, mask_volume_fraction);
	output_3d.has_statistics = true;
	output_3d.PrintStatistics();
	output_3d.WriteStatisticsToFile(output_statistics_file);

	my_reconstruction_1 += my_reconstruction_2;
//	~my_reconstruction_2;
	pssnr_scale_factor = output_3d1.density_map.ReturnVolumeInRealSpace() / (kDa_to_Angstrom3(molecular_mass_kDa) / powf(pixel_size,3))
			* particle_area_in_pixels / output_3d1.density_map.logical_x_dimension / output_3d1.density_map.logical_y_dimension;
	output_3d.Calculate3DOptimal(my_reconstruction_1, pssnr_scale_factor);
	output_3d.density_map.SwapRealSpaceQuadrants();
	output_3d.density_map.CosineMask(0.5, 1.0 / 20.0);
	output_3d.density_map.BackwardFFT();
	output_3d.Correct3D();
	output_3d.CosineRingMask(inner_mask_radius / pixel_size, outer_mask_radius / pixel_size, mask_falloff / pixel_size);
	output_file.OpenFile(output_reconstruction_filtered.ToStdString(), true);
	output_3d.density_map.WriteSlices(&output_file,1,output_3d.density_map.logical_z_dimension);
	output_file.SetPixelSize(pixel_size);
	output_file.CloseFile();

	wxPrintf("\nReconstruct3D: Normal termination\n\n");

	return true;
}
