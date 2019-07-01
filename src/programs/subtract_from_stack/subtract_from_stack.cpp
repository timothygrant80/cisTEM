#include "../../core/core_headers.h"

class
SubtractFromStackApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};

IMPLEMENT_APP(SubtractFromStackApp)

// override the DoInteractiveUserInput

void SubtractFromStackApp::DoInteractiveUserInput()
{
	wxString	input_particle_images;
	wxString	input_parameter_file;
	wxString	input_reconstruction;
	wxString	input_reconstruction_statistics = "";
	bool		use_statistics = false;
	wxString	output_subtracted_images;
	float		pixel_size = 1.0;
	float		voltage_kV = 300.0;
	float		spherical_aberration_mm = 2.7;
	float		amplitude_contrast = 0.07;
	bool        use_least_squares_scaling;
	float 		mask_radius;
	int			first_particle;
	int			last_particle;

	UserInput *my_input = new UserInput("SubtractFromStack", 1.00);

	input_particle_images = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	input_parameter_file = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
	input_reconstruction = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
//	input_reconstruction_statistics = my_input->GetFilenameFromUser("Input data statistics", "The table listing FSC, Part_FSC, Part_SSNR and Rec_SSNR", "my_statistics.txt", false);
//	use_statistics = my_input->GetYesNoFromUser("Use statistics", "Answer No if no statistics are available?", "Yes");
	output_subtracted_images = my_input->GetFilenameFromUser("Output stack of subtracted images", "The output image stack, containing the matching projections", "subtracted_stack.mrc", false);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
	use_least_squares_scaling = my_input->GetYesNoFromUser("Use Least Squares Scaling", "Answer yes to scale per particle.", "Yes");
	mask_radius = my_input->GetFloatFromUser("Mask Radius for scaling (A)", "Only consider within this radius for scaling", "100", 0.0);
	first_particle = my_input->GetIntFromUser("First particle to process", "first particle to process", "1", 1);
	last_particle = my_input->GetIntFromUser("Last  particle to process (0 = last in stack)", "last particle to process", "0", 0);



	delete my_input;

	my_current_job.Reset(12);
	my_current_job.ManualSetArguments("ttttbtffffbfii",	input_particle_images.ToUTF8().data(),
														input_parameter_file.ToUTF8().data(),
														input_reconstruction.ToUTF8().data(),
														input_reconstruction_statistics.ToUTF8().data(),
														use_statistics,
														output_subtracted_images.ToUTF8().data(),
														pixel_size,
														voltage_kV,
														spherical_aberration_mm,
														amplitude_contrast,
														use_least_squares_scaling,
														mask_radius,
														first_particle,
														last_particle);
}

// override the do calculation method which will be what is actually run..

bool SubtractFromStackApp::DoCalculation()
{
	Particle refine_particle;
	Particle search_particle;

	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument(); // global
	wxString input_parameter_file 				= my_current_job.arguments[1].ReturnStringArgument(); // not sure
	wxString input_reconstruction_filename		= my_current_job.arguments[2].ReturnStringArgument(); // global
	wxString input_reconstruction_statistics 	= my_current_job.arguments[3].ReturnStringArgument(); // global
	bool	 use_statistics						= my_current_job.arguments[4].ReturnBoolArgument();   // global
	wxString output_subtracted_images 			= my_current_job.arguments[5].ReturnStringArgument(); // ignore (always false)
	float 	 pixel_size							= my_current_job.arguments[6].ReturnFloatArgument(); // local
	float    voltage_kV							= my_current_job.arguments[7].ReturnFloatArgument(); // local
	float 	 spherical_aberration_mm			= my_current_job.arguments[8].ReturnFloatArgument(); // local
	float    amplitude_contrast					= my_current_job.arguments[9].ReturnFloatArgument(); // local
	bool 	 use_least_squares_scaling			= my_current_job.arguments[10].ReturnBoolArgument();   // global
	float    mask_radius						= my_current_job.arguments[11].ReturnFloatArgument(); // local
	int		 first_particle						= my_current_job.arguments[12].ReturnIntegerArgument(); // local
	int		 last_particle						= my_current_job.arguments[13].ReturnIntegerArgument(); // local


	ReconstructedVolume input_3d;
	Image projection_image;
	Image particle_image;
	Image temp_image;
	Image sum_power;

	long current_image;
	long image_counter = 0;
	long pixel_counter;
	long number_of_pixels_in_image;
	float temp_float[50];

	double dot_product;
	double self_dot_product;
	long used_pixels;

	float average_sigma = 0.0f;
	float average_score = 0.0f;
	float scale_factor;

	float mask_radius_for_noise;
	float variance;
	float percentage;

	int image_write_position;
	int number_of_images_to_process;
	int number_of_images_processed;

	Curve noise_power_spectrum;
	Curve number_of_terms;

	ProgressBar *my_progress;

	MRCFile input_stack(input_particle_images.ToStdString(), false);
	FrealignParameterFile my_input_par_file(input_parameter_file, OPEN_TO_READ);
	my_input_par_file.ReadFile();
	MRCFile input_file(input_reconstruction_filename.ToStdString(), false);
	MRCFile output_file(output_subtracted_images.ToStdString(), true);
	AnglesAndShifts my_parameters;
	CTF my_ctf;

	if (last_particle == 0) last_particle = my_input_par_file.number_of_lines;
	number_of_images_to_process = (last_particle - first_particle) + 1;

	if ((input_file.ReturnXSize() != input_file.ReturnYSize()) || (input_file.ReturnXSize() != input_file.ReturnZSize()))
	{
		MyPrintWithDetails("Error: Input reconstruction is not cubic\n");
		DEBUG_ABORT;
	}


	input_3d.InitWithDimensions(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), pixel_size, "C1");
	input_3d.density_map->ReadSlices(&input_file,1,input_3d.density_map->logical_z_dimension);
	input_3d.mask_radius = FLT_MAX;
	input_3d.PrepareForProjections(0.0, 2.0 * pixel_size);

	projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
	sum_power.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
	particle_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), false);

	number_of_pixels_in_image = long(input_stack.ReturnXSize()) * long(input_stack.ReturnYSize());

	wxPrintf("\nCalculating noise power spectrum...\n\n");

	percentage = float(2500) / float(my_input_par_file.number_of_lines);
	sum_power.SetToConstant(0.0);
	mask_radius_for_noise = mask_radius / pixel_size;

	if (2.0 * mask_radius_for_noise + 0.05 / pixel_size > 0.95 * particle_image.logical_x_dimension)
	{
		mask_radius_for_noise = 0.95 * particle_image.logical_x_dimension / 2.0 - 0.05 / 2.0 / pixel_size;
	}

	noise_power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

	my_progress = new ProgressBar(my_input_par_file.number_of_lines);
	image_counter = 0;

	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(temp_float);
		image_counter++;
		my_progress->Update(image_counter);

		if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0 * percentage)) continue;

		particle_image.ReadSlice(&input_stack, int(temp_float[0] + 0.5));
		variance = particle_image.ReturnVarianceOfRealValues(mask_radius / pixel_size, 0.0, 0.0, 0.0, true);
		if (variance == 0.0) continue;
		particle_image.MultiplyByConstant(1.0 / sqrtf(variance));
		particle_image.CosineMask(mask_radius / pixel_size, 5.0 , true);
		particle_image.ForwardFFT();
		temp_image.CopyFrom(&particle_image);
		temp_image.ConjugateMultiplyPixelWise(particle_image);
		sum_power.AddImage(&temp_image);
	}

	delete my_progress;

	sum_power.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);

	noise_power_spectrum.SquareRoot();
	//noise_power_spectrum.Reciprocal();

	my_input_par_file.Rewind();

	wxPrintf("\nSubtracting...\n\n");
	image_counter = 0;
	my_progress = new ProgressBar(number_of_images_to_process);

	image_write_position = 1;
	number_of_images_processed = 0;

	for (current_image = 1; current_image <= my_input_par_file.number_of_lines; current_image++)
	{
		my_input_par_file.ReadLine(temp_float);

		if (temp_float[0] < first_particle || temp_float[0] > last_particle) continue;
		particle_image.ReadSlice(&input_stack, current_image);

		image_counter++;

		my_parameters.Init(temp_float[3], temp_float[2], temp_float[1], temp_float[4], temp_float[5]);
		my_ctf.Init(voltage_kV, spherical_aberration_mm, amplitude_contrast, temp_float[8], temp_float[9], temp_float[10], 0.0, 0.0, 0.0, pixel_size, temp_float[11]);

		input_3d.density_map->ExtractSlice(projection_image, my_parameters);
		projection_image.ApplyCTF(my_ctf);
		projection_image.PhaseShift(temp_float[4] / pixel_size, temp_float[5] / pixel_size);
		projection_image.SwapRealSpaceQuadrants();
//		projection_image.ZeroCentralPixel();
//		projection_image.DivideByConstant(sqrtf(projection_image.ReturnSumOfSquares()));
		projection_image.ApplyCurveFilter(&noise_power_spectrum);
		projection_image.BackwardFFT();

		if (use_least_squares_scaling == true)
		{
			used_pixels = 0;
			dot_product = 0.0;
			self_dot_product = 0.0;
			int x;
			int y;

			float max_radius = powf(mask_radius / pixel_size, 2.0f);
			float x_rad_sq;
			float y_rad_sq;
			float current_radius_squared;

			pixel_counter = 0;

			for ( y = 0; y < particle_image.logical_y_dimension; y ++ )
			{
				y_rad_sq = powf(y - particle_image.physical_address_of_box_center_y, 2.0f);

				for ( x = 0; x < particle_image.logical_x_dimension; x ++ )
				{
					x_rad_sq = powf(x - particle_image.physical_address_of_box_center_x, 2.0f);
					current_radius_squared = x_rad_sq + y_rad_sq;

					if (particle_image.real_values[pixel_counter] != 0. && projection_image.real_values[pixel_counter] != 0. && current_radius_squared < max_radius)
					{
						dot_product += particle_image.real_values[pixel_counter] * projection_image.real_values[pixel_counter];
						self_dot_product += pow(projection_image.real_values[pixel_counter], 2);
						used_pixels++;
					}
					pixel_counter++;

				}

				pixel_counter += particle_image.padding_jump_value;

			}

			scale_factor = dot_product / self_dot_product;
			//scale_factor = 1.0f / scale_factor;
			//wxPrintf("Scale factor = %f\n", scale_factor);
			projection_image.MultiplyByConstant(scale_factor);


		}

		particle_image.SubtractImage(&projection_image);
		particle_image.WriteSlice(&output_file, image_write_position);
		//projection_image.QuickAndDirtyWriteSlice("/tmp/projs.mrc", image_write_position);
		image_write_position++;
		number_of_images_processed++;

		my_progress->Update(number_of_images_processed);
	}
	delete my_progress;
	wxPrintf("\nSubtractFromStack: Normal termination\n\n");

	return true;
}
