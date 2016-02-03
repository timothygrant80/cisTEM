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
	wxString input_parameter_file;
	wxString input_particle_stack;
	wxString output_reconstruction_1;
	wxString output_reconstruction_2;
	wxString output_reconstruction_filtered;
	float pixel_size = 1;
	float voltage_kV = 300.0;
	float spherical_aberration_mm = 2.7;
	float amplitude_contrast = 0.07;
	float molecular_mass_kDa = 1000.0;
	float mask_radius = 100.0;
	float score_bfactor_conversion = 5.0;
	wxString my_symmetry = "C1";

	UserInput *my_input = new UserInput("Reconstruct3D", 1.0);

	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	input_particle_stack = my_input->GetFilenameFromUser("Input particle stack", "The input particle image stack, containing the 2D images for each particle in the dataset", "my_particle_stack.mrc", true);
	output_reconstruction_1 = my_input->GetFilenameFromUser("Output reconstruction 1", "The first output 3D reconstruction, calculated form half the data", "my_reconstruction_1.mrc", false);
	output_reconstruction_2 = my_input->GetFilenameFromUser("Output reconstruction 2", "The second output 3D reconstruction, calculated form half the data", "my_reconstruction_2.mrc", false);
	output_reconstruction_filtered = my_input->GetFilenameFromUser("Output filtered reconstruction", "The final 3D reconstruction, containing from all data and optimally filtered", "my_filtered_reconstruction.mrc", false);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	molecular_mass_kDa = my_input->GetFloatFromUser("Molecular mass of particle (kDa)", "Total molecular mass of the particle to be reconstructed in kilo Daltons", "1000.0", 0.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius (A)", "Radius of the spherical mask to be applied to the final reconstruction in Angstroms", "100.0", 0.0);
	score_bfactor_conversion = my_input->GetFloatFromUser("Particle weighting factor (A^2)", "Constant to convert particle scores to B-factors in squared Angstroms", "5.0", 0.0);
	my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

	delete my_input;

	my_current_job.Reset(13);
	my_current_job.ManualSetArguments("tttttffffffft",  input_parameter_file.ToUTF8().data(),
														input_particle_stack.ToUTF8().data(),
														output_reconstruction_1.ToUTF8().data(),
														output_reconstruction_2.ToUTF8().data(),
														output_reconstruction_filtered.ToUTF8().data(),
														pixel_size,
														voltage_kV,
														spherical_aberration_mm,
														amplitude_contrast,
														molecular_mass_kDa,
														mask_radius,
														score_bfactor_conversion,
														my_symmetry.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool Reconstruct3DApp::DoCalculation()
{
/*
	float pixel_size = 3.28;
//	float pixel_size = 1.64;
	float voltage_kV = 300.0;
	float spherical_aberration_mm = 2.7;
	float amplitude_contrast = 0.07;
	float molecular_mass_kDa = 3200.0;
	float mask_radius = 170.0;
	float score_bfactor_conversion = 5.0;
	wxString my_symmetry = "C1";
/*
	float pixel_size = 3.4918;
	float voltage_kV = 300.0;
	float spherical_aberration_mm = 2.7;
	float amplitude_contrast = 0.07;
	float molecular_mass_kDa = 47371.0;
	float mask_radius = 365.0;
	float score_bfactor_conversion = 5.0;
	wxString my_symmetry = "I2";

	float pixel_size = 0.82;
	float voltage_kV = 300.0;
	float spherical_aberration_mm = 2.7;
	float amplitude_contrast = 0.07;
	float molecular_mass_kDa = 3200.0;
	float mask_radius = 180.0;
	float score_bfactor_conversion = 5.0;
	wxString my_symmetry = "C1";
*/
	wxString input_parameter_file 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_particle_stack 				= my_current_job.arguments[1].ReturnStringArgument();
	wxString output_reconstruction_1			= my_current_job.arguments[2].ReturnStringArgument();
	wxString output_reconstruction_2			= my_current_job.arguments[3].ReturnStringArgument();
	wxString output_reconstruction_filtered		= my_current_job.arguments[4].ReturnStringArgument();
	float 	 pixel_size							= my_current_job.arguments[5].ReturnFloatArgument();
	float    voltage_kV							= my_current_job.arguments[6].ReturnFloatArgument();
	float 	 spherical_aberration_mm			= my_current_job.arguments[7].ReturnFloatArgument();
	float    amplitude_contrast					= my_current_job.arguments[8].ReturnFloatArgument();
	float 	 molecular_mass_kDa					= my_current_job.arguments[9].ReturnFloatArgument();
	float    mask_radius						= my_current_job.arguments[10].ReturnFloatArgument();
	float    score_bfactor_conversion			= my_current_job.arguments[11].ReturnFloatArgument();
	wxString my_symmetry						= my_current_job.arguments[12].ReturnStringArgument();

	Image input_image;
	ReconstructedVolume output_3d;
	ReconstructedVolume output_3d1;
	ReconstructedVolume output_3d2;
	float particle_weight;
	float particle_score;
	float image_sigma_noise;
	float particle_occupancy;

	int current_image;
	float temp_float[50];
	float average_score = 0.0;
	float average_sigma = 0.0;
	float volume_fraction;
	float pssnr_correction_factor;
	float kDa_to_Angstrom3 = 1000.0 / 0.81;

//	SymmetryMatrix my_symmetry_matrices(my_symmetry);
//	my_symmetry_matrices.PrintMatrices();

	MRCFile input_file(input_particle_stack.ToStdString(), false);
	MRCFile output_file;
	NumericTextFile my_par_file(input_parameter_file, OPEN_TO_READ);
	AnglesAndShifts my_parameters;
	CTF my_ctf;
//	ResolutionStatistics *my_statistics;

	Reconstruct3d my_reconstruction_1(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnXSize(), pixel_size, my_symmetry);
	Reconstruct3d my_reconstruction_2(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnXSize(), pixel_size, my_symmetry);

// Read whole parameter file to work out average image_sigma_noise and average score
	for (current_image = 1; current_image <= input_file.ReturnNumberOfSlices(); current_image++)
	{
		my_par_file.ReadLine(temp_float);
		average_sigma += temp_float[13];
		average_score += temp_float[14];
	}
	average_sigma /= input_file.ReturnNumberOfSlices();
	average_score /= input_file.ReturnNumberOfSlices();
	wxPrintf("\nAverage sigma noise = %f, average score = %f\n\n", average_sigma, average_score);
	my_par_file.Rewind();

	ProgressBar *my_progress = new ProgressBar(input_file.ReturnNumberOfSlices());
	for (current_image = 1; current_image <= input_file.ReturnNumberOfSlices(); current_image++)
//	for (current_image = 1; current_image <= 100; current_image++)
	{
		input_image.ReadSlice(&input_file, current_image);

//		input_image.SetToConstant(0.0);
//		input_image.real_values[input_image.ReturnReal1DAddressFromPhysicalCoord(64,64,0)] = 1.0;

		my_par_file.ReadLine(temp_float);
		my_parameters.Init(temp_float[3], temp_float[2], temp_float[1], temp_float[4], temp_float[5]);
		my_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, temp_float[8], temp_float[9], temp_float[10], 0.0, 0.0, 0.0, pixel_size, 0.0);

		particle_score = temp_float[14];
		particle_occupancy = temp_float[11];
		image_sigma_noise = temp_float[13];
		if (image_sigma_noise <= 0.0) image_sigma_noise = average_sigma;
//		image_sigma_noise = average_sigma;
		particle_weight = particle_occupancy / 100.0 / pow(image_sigma_noise,2);

		input_image.ForwardFFT();

/*		long pixel_counter;
		for (int j = input_image.logical_lower_bound_complex_y; j <= input_image.logical_upper_bound_complex_y; j++)
		{
			for (int i = 1; i <= input_image.logical_upper_bound_complex_x; i++)
			{
				if (input_image.ReturnFourierLogicalCoordGivenPhysicalCoord_X(i)==20 && input_image.ReturnFourierLogicalCoordGivenPhysicalCoord_Y(j)==20)
				{
					pixel_counter = input_image.ReturnFourier1DAddressFromLogicalCoord(i,j,0);
					wxPrintf("counter = %i, coord = %li, image = %g\n", current_image, pixel_counter, cabsf(input_image.complex_values[pixel_counter]));
				}
			}
		}
*/
//		input_image.ApplyCTF(my_ctf);
		input_image.PhaseShift(-temp_float[4] / pixel_size, -temp_float[5] / pixel_size);
		input_image.SwapRealSpaceQuadrants();

//		if (current_image % 100 == 0 || current_image == 1)
//		{
//			wxPrintf("Working on image %i, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", current_image, temp_float[1], temp_float[2], temp_float[3], temp_float[4], temp_float[5], temp_float[8], temp_float[9], temp_float[10], temp_float[11]);
//			fflush(stdout);
//		}
		my_progress->Update(current_image);
		if (current_image % 2 == 0)
		{
			my_reconstruction_2.InsertSlice(input_image, my_ctf, my_parameters, particle_weight, particle_score, average_score, score_bfactor_conversion);
	//		my_reconstruction_2.InsertSlice(input_image, my_parameters);
		}
		else
		{
			my_reconstruction_1.InsertSlice(input_image, my_ctf, my_parameters, particle_weight, particle_score, average_score, score_bfactor_conversion);
	//		my_reconstruction_1.InsertSlice(input_image, my_parameters);
		}
	}
	delete my_progress;

//	input_image.QuickAndDirtyReadSlice("test.mrc", 1);

//	volume_fraction = (4.0 / 3.0 * PI * pow(mask_radius/pixel_size,3)) / input_image.logical_x_dimension / input_image.logical_y_dimension / input_image.logical_x_dimension;
	output_3d1.Init(my_reconstruction_1, pixel_size);
	output_3d1.Calculate3DSimple(my_reconstruction_1);
	output_3d1.density_map.SwapRealSpaceQuadrants();
	output_3d1.density_map.BackwardFFT();
	pssnr_correction_factor = output_3d1.Correct3D(mask_radius);
	output_3d1.CosineMask(mask_radius/pixel_size, 6.0);
	output_3d.mask_volume_in_voxels = output_3d1.mask_volume_in_voxels;
	output_file.OpenFile(output_reconstruction_1.ToStdString(), true);
	output_3d1.density_map.WriteSlices(&output_file,1,output_3d1.density_map.logical_z_dimension);
	output_file.CloseFile();
	output_3d1.density_map.ForwardFFT();

	output_3d2.Init(my_reconstruction_2, pixel_size);
	output_3d2.Calculate3DSimple(my_reconstruction_2);
	output_3d2.density_map.SwapRealSpaceQuadrants();
	output_3d2.density_map.BackwardFFT();
	pssnr_correction_factor = output_3d2.Correct3D(mask_radius);
	output_3d2.CosineMask(mask_radius/pixel_size, 6.0);
	output_file.OpenFile(output_reconstruction_2.ToStdString(), true);
	output_3d2.density_map.WriteSlices(&output_file,1,output_3d2.density_map.logical_z_dimension);
	output_file.CloseFile();
	output_3d2.density_map.ForwardFFT();

	output_3d.Init(my_reconstruction_1, pixel_size);
	output_3d.statistics.CalculateFSC(output_3d1.density_map, output_3d2.density_map);
	output_3d.molecular_mass_in_kDa = molecular_mass_kDa;
	output_3d.statistics.CalculateParticleFSCandSSNR(output_3d1.mask_volume_in_voxels, molecular_mass_kDa, pixel_size);
	output_3d.statistics.CalculateParticleSSNR(my_reconstruction_1.image_reconstruction, my_reconstruction_1.ctf_reconstruction);
	output_3d.has_statistics = true;
	output_3d.PrintStatistics();

	my_reconstruction_1 += my_reconstruction_2;
//	~my_reconstruction_2;
//	volume_fraction = molecular_mass_kDa * kDa_to_Angstrom3 / pow(pixel_size,3) / output_3d1.density_map.logical_x_dimension / output_3d1.density_map.logical_y_dimension / output_3d1.density_map.logical_z_dimension;
	output_3d.Calculate3DOptimal(my_reconstruction_1, pssnr_correction_factor);
	output_3d.density_map.SwapRealSpaceQuadrants();
	output_3d.density_map.BackwardFFT();
	output_3d.Correct3D();
	output_3d.CosineMask(mask_radius/pixel_size, 6.0);
//	wxPrintf("Mask volume = %f\n", output_3d.mask_volume_in_voxels);
	output_file.OpenFile(output_reconstruction_filtered.ToStdString(), true);
	output_3d.density_map.WriteSlices(&output_file,1,output_3d.density_map.logical_z_dimension);
	output_file.CloseFile();

/*	output_3d.density_map.ForwardFFT();
	output_3d.OptimalFilter();
	output_3d.density_map.BackwardFFT();
	output_file.OpenFile("test3d_filtered_r1.mrc", true);
	output_3d.density_map.WriteSlices(&output_file,1,output_3d.density_map.logical_z_dimension);
	output_file.CloseFile();
*/
	return true;
}
