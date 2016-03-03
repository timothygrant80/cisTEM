#include "../../core/core_headers.h"

class
Project3DApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(Project3DApp)

// override the DoInteractiveUserInput

void Project3DApp::DoInteractiveUserInput()
{
	wxString input_parameter_file;
	wxString input_reconstruction;
	wxString ouput_projection_stack;
	float pixel_size = 1;
	float voltage_kV = 300.0;
	float spherical_aberration_mm = 2.7;
	float amplitude_contrast = 0.07;
	float mask_radius = 100.0;
	float padding = 1.0;
	float wanted_SNR = 1.0;
	wxString my_symmetry = "C1";
	bool apply_CTF;
	bool apply_shifts;
	bool apply_mask;
	bool add_noise;

	UserInput *my_input = new UserInput("Project3D", 1.0);

	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	input_reconstruction = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
	ouput_projection_stack = my_input->GetFilenameFromUser("Output projection stack", "The output image stack, containing the 2D projections", "my_projection_stack.mrc", false);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", 0.0);
	wanted_SNR = my_input->GetFloatFromUser("Wanted SNR", "The ratio of signal to noise variance after adding Gaussian noise and before masking", "1.0", 0.0);
	padding = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
	my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
	apply_CTF = my_input->GetYesNoFromUser("Apply CTF", "Should the CTF be applied to the output projections?", "No");
	apply_shifts = my_input->GetYesNoFromUser("Apply shifts", "Should the particle translations be applied to the output projections?", "No");
	apply_mask = my_input->GetYesNoFromUser("Apply mask", "Should the particles be masked with the circular mask?", "No");
	add_noise = my_input->GetYesNoFromUser("Add noise", "Should the Gaussian noise be added?", "No");

	delete my_input;

	my_current_job.Reset(10);
	my_current_job.ManualSetArguments("tttffffffftbbbb",input_parameter_file.ToUTF8().data(),
														input_reconstruction.ToUTF8().data(),
														ouput_projection_stack.ToUTF8().data(),
														pixel_size,
														voltage_kV,
														spherical_aberration_mm,
														amplitude_contrast,
														mask_radius,
														wanted_SNR,
														padding,
														my_symmetry.ToUTF8().data(),
														apply_CTF,
														apply_shifts,
														apply_mask,
														add_noise);
}

// override the do calculation method which will be what is actually run..

bool Project3DApp::DoCalculation()
{
	wxString input_parameter_file 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_reconstruction				= my_current_job.arguments[1].ReturnStringArgument();
	wxString output_projection_stack 			= my_current_job.arguments[2].ReturnStringArgument();
	float 	 pixel_size							= my_current_job.arguments[3].ReturnFloatArgument();
	float    voltage_kV							= my_current_job.arguments[4].ReturnFloatArgument();
	float 	 spherical_aberration_mm			= my_current_job.arguments[5].ReturnFloatArgument();
	float    amplitude_contrast					= my_current_job.arguments[6].ReturnFloatArgument();
	float    mask_radius						= my_current_job.arguments[7].ReturnFloatArgument();
	float	 wanted_SNR							= my_current_job.arguments[8].ReturnFloatArgument();
	float	 padding							= my_current_job.arguments[9].ReturnFloatArgument();
	wxString my_symmetry						= my_current_job.arguments[10].ReturnStringArgument();
	bool	 apply_CTF							= my_current_job.arguments[11].ReturnBoolArgument();
	bool	 apply_shifts						= my_current_job.arguments[12].ReturnBoolArgument();
	bool	 apply_mask							= my_current_job.arguments[13].ReturnBoolArgument();
	bool	 add_noise							= my_current_job.arguments[14].ReturnBoolArgument();

	Image projection_image;
	Image final_image;
	ReconstructedVolume input_3d;

	int current_image;
	float temp_float[50];
	float average_score = 0.0;
	float average_sigma = 0.0;
	float variance;
	float binning_factor = 1.0;

	NumericTextFile my_par_file(input_parameter_file, OPEN_TO_READ);
	MRCFile input_file(input_reconstruction.ToStdString(), false);
	MRCFile output_file(output_projection_stack.ToStdString(), true);
	AnglesAndShifts my_parameters;
	CTF my_ctf;
	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, my_symmetry);
	input_3d.density_map.ReadSlices(&input_file,1,input_3d.density_map.logical_z_dimension);
	if (padding != 1.0)
	{
		input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension * padding, input_3d.density_map.logical_y_dimension * padding, input_3d.density_map.logical_z_dimension * padding, input_3d.density_map.ReturnAverageOfRealValuesOnEdges());
	}
	input_3d.density_map.Correct3D();
	input_3d.density_map.ForwardFFT();
	if (binning_factor != 1.0 ) input_3d.density_map.Resize(input_3d.density_map.logical_x_dimension / binning_factor, input_3d.density_map.logical_y_dimension / binning_factor, input_3d.density_map.logical_z_dimension / binning_factor);
	input_3d.density_map.SwapRealSpaceQuadrants();
	projection_image.Allocate(input_3d.density_map.logical_x_dimension, input_3d.density_map.logical_y_dimension, false);
	final_image.Allocate(input_file.ReturnXSize() / binning_factor, input_file.ReturnYSize() / binning_factor, true);

// Read whole parameter file to work out average image_sigma_noise and average score
	for (current_image = 1; current_image <= my_par_file.number_of_lines; current_image++)
	{
		my_par_file.ReadLine(temp_float);
		average_sigma += temp_float[13];
		average_score += temp_float[14];
	}
	average_sigma /= input_file.ReturnNumberOfSlices();
	average_score /= input_file.ReturnNumberOfSlices();
	wxPrintf("\nAverage sigma noise = %f, average score = %f\nNumber of projections to calculate = %i\n\n", average_sigma, average_score, my_par_file.number_of_lines);
	my_par_file.Rewind();

	ProgressBar *my_progress = new ProgressBar(my_par_file.number_of_lines);
	for (current_image = 1; current_image <= my_par_file.number_of_lines; current_image++)
//	for (current_image = 1; current_image <= 100; current_image++)
	{
		my_par_file.ReadLine(temp_float);
		my_parameters.Init(temp_float[3], temp_float[2], temp_float[1], temp_float[4], temp_float[5]);
		my_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, temp_float[8], temp_float[9], temp_float[10], 0.0, 0.0, 0.0, pixel_size, 0.0);

		input_3d.density_map.ExtractSlice(projection_image, my_parameters);

		if (apply_CTF) projection_image.ApplyCTF(my_ctf);
		if (apply_shifts) projection_image.PhaseShift(temp_float[4] / pixel_size, temp_float[5] / pixel_size);
		projection_image.SwapRealSpaceQuadrants();

		projection_image.BackwardFFT();
		projection_image.ClipInto(&final_image);
		if (add_noise && wanted_SNR != 0.0)
		{
			variance = final_image.ReturnVarianceOfRealValues();
			final_image.AddGaussianNoise(sqrtf(variance / wanted_SNR));
//			wxPrintf("sigma_signal = %f, sigma_noise = %f\n", sqrtf(variance), sqrtf(variance / wanted_SNR));
		}
		if (apply_mask) final_image.CosineMask(mask_radius / pixel_size, 6.0);
		final_image.WriteSlice(&output_file, int(temp_float[0] + 0.5));

		my_progress->Update(current_image);
	}
	delete my_progress;

	return true;
}
