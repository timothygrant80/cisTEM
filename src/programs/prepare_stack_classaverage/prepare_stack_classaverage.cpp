#include "../../core/core_headers.h"

class
PrepareStackApp : public MyApp
{
	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	float GetMaxJobWaitTimeInSeconds() {return 180.0f;}

	private:
};



IMPLEMENT_APP(PrepareStackApp)

// override the DoInteractiveUserInput

void PrepareStackApp::DoInteractiveUserInput()
{
	wxString	input_particle_images;
	wxString	output_classaverage_images;
	wxString    input_star_file;
	wxString    input_selection_file;
	int	 wanted_output_box_size;

	bool resample_box;
	bool process_a_subset = false;
	float mask_radius;
	float output_pixel_size;
	int first_classaverage = 0;
	int last_classaverage = 0;

	int number_of_classes;
	int images_per_class;
	float microscope_voltage;
	float microscope_cs;
	float amplitude_contrast;

	bool invert_contrast;

	UserInput *my_input = new UserInput("PrepareStackClassAverage", 1.00);

	input_particle_images = my_input->GetFilenameFromUser("Input particle stack", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	output_classaverage_images = my_input->GetFilenameFromUser("Output particle stack", "The output image stack, containing the prepared particle images", "my_image_stack_prep.mrc", false);
	input_star_file = my_input->GetFilenameFromUser("Input cisTEM star filename", "The input star file, containing your particle alignment parameters", "my_parameters.star", true);
	input_selection_file = my_input->GetFilenameFromUser("Input class selection filename", "text file containing the wanted class averages", "my_selection.txt", true);
	output_pixel_size = my_input->GetFloatFromUser("Wanted Output Pixel Size (Angstroms)", "Pixel Size of the images", "1", 0.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius (Angstroms)", "For calculating noise statistics", "100", 0.0);
	resample_box = my_input->GetYesNoFromUser("Resample output?","If yes you can resample the output image to a specified size", "NO");

	if (resample_box == true) wanted_output_box_size = my_input->GetIntFromUser("Resampled box size", "How big to resample the box size to?", "512", 0.0);
	else wanted_output_box_size = 1;

	number_of_classes = my_input->GetIntFromUser("Number of classes wanted", "", "100", 1);
	images_per_class = my_input->GetIntFromUser("Number of images per class wanted", "", "25", 1);

	invert_contrast = my_input->GetYesNoFromUser("Invert Contrast?","If yes the contrast of the images will be inverted", "YES");


	delete my_input;



	my_current_job.ManualSetArguments("ttttffbiiibbii",	input_particle_images.ToUTF8().data(),
															output_classaverage_images.ToUTF8().data(),
															input_star_file.ToUTF8().data(),
															input_selection_file.ToUTF8().data(),
															output_pixel_size,
															mask_radius,
															resample_box,
															wanted_output_box_size,
															number_of_classes,
															images_per_class,
															invert_contrast,
															process_a_subset,
															first_classaverage,
															last_classaverage);
}

// override the do calculation method which will be what is actually run..

bool PrepareStackApp::DoCalculation()
{


	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString output_classaverage_images 		= my_current_job.arguments[1].ReturnStringArgument();
	wxString input_star_filename 				= my_current_job.arguments[2].ReturnStringArgument();
	wxString input_selection_file 				= my_current_job.arguments[3].ReturnStringArgument();
	float output_pixel_size			          	= my_current_job.arguments[4].ReturnFloatArgument();
	float  mask_radius                          = my_current_job.arguments[5].ReturnFloatArgument();
	bool resample_box							= my_current_job.arguments[6].ReturnBoolArgument();
	int	 wanted_output_box_size 				= my_current_job.arguments[7].ReturnIntegerArgument();
	int number_of_classes 						= my_current_job.arguments[8].ReturnIntegerArgument();
	int images_per_class 						= my_current_job.arguments[9].ReturnIntegerArgument();
	bool invert_contrast						= my_current_job.arguments[10].ReturnBoolArgument();
	bool process_a_subset						= my_current_job.arguments[11].ReturnBoolArgument();
	int  first_classaverage					    = my_current_job.arguments[12].ReturnIntegerArgument();
	int  last_classaverage					    = my_current_job.arguments[13].ReturnIntegerArgument();

	int class_averages_to_process = (last_classaverage - first_classaverage) + 1;

	ProgressBar *my_progress;
	int max_samples = 2000;

	int current_image;
	int image_counter;
	int current_classaverage;
	int line_counter;
	int class_counter;
	int random_image;
	int pixel_counter;
	int output_file_position;

	int total_positions;
	int current_position = 0;

	float input_parameters[17];
	ZeroFloatArray(input_parameters, 17);

	float mask_radius_for_noise = mask_radius / output_pixel_size;
	float mask_falloff = 10.0f;
	float variance;
	float average;

	float bin_factor;

	Image input_image;
	Image sum_image;
	Image ctf_sum_image;
	Image ctf_input_image;
	Image rotated_image;
	Image sum_power;
	Image temp_image;

	wxArrayInt classaverages_to_make;

	cisTEMParameterLine temp_line;
	ArrayOfcisTEMParameterLines class_members;

	CTF current_ctf;
	AnglesAndShifts rotation_angle;

	float temp_float[50];

	if (is_running_locally == false)
	{
		float result;
		my_result.SetResult(1, &result);
	}


	ImageFile input_file(input_particle_images.ToStdString());
	int images_to_process = input_file.ReturnNumberOfSlices();

	bin_factor = input_file.ReturnXSize() / wanted_output_box_size;
	MRCFile *output_file;

	NumericTextFile wanted_class_averages(input_selection_file, OPEN_TO_READ);

	if (process_a_subset == false)
	{
		first_classaverage = 0;
		last_classaverage = wanted_class_averages.number_of_lines - 1;
	}

	cisTEMParameters input_star_file;
	input_star_file.ReadFromcisTEMStarFile(input_star_filename);

	if (is_running_locally == true) output_file = new MRCFile(output_classaverage_images.ToStdString(), true);

	Curve noise_power_spectrum;
	Curve number_of_terms;
	RandomNumberGenerator random_generator;

	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
	sum_power.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1, false);
	sum_power.SetToConstant(0.0);

	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
	sum_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
	ctf_sum_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
	ctf_input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
	rotated_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);


	if (is_running_locally == true) wxPrintf("\nCalculating noise power spectrum...\n\n");

	float percentage = float(max_samples) / float(images_to_process);
	sum_power.SetToConstant(0.0);

	if (2.0 * mask_radius_for_noise + mask_falloff / output_pixel_size > 0.95 * input_image.logical_x_dimension)
	{
		mask_radius_for_noise = 0.95 * input_image.logical_x_dimension / 2.0 - mask_falloff / 2.0 / output_pixel_size;
	}

	noise_power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((sum_power.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

	if (is_running_locally == true) my_progress = new ProgressBar(input_file.ReturnNumberOfSlices());

	for (current_image = 1; current_image <= input_file.ReturnNumberOfSlices(); current_image++)
	{
		if ((global_random_number_generator.GetUniformRandom() < 1.0 - 2.0 * percentage)) continue;

		input_image.ReadSlice(&input_file, current_image);
		input_image.ChangePixelSize(&input_image, output_pixel_size / input_star_file.ReturnPixelSize(current_image - 1), 0.001f);
		variance = input_image.ReturnVarianceOfRealValues(mask_radius_for_noise, 0.0, 0.0, 0.0, true);
		if (variance == 0.0) continue;

		input_image.MultiplyByConstant(1.0 / sqrtf(variance));
		input_image.CosineMask(mask_radius_for_noise, mask_falloff / output_pixel_size, true);
		input_image.ForwardFFT();
		temp_image.CopyFrom(&input_image);
		temp_image.ConjugateMultiplyPixelWise(input_image);
		sum_power.AddImage(&temp_image);

		if (is_running_locally == true) my_progress->Update(current_image);
	}

	sum_power.Compute1DRotationalAverage(noise_power_spectrum, number_of_terms);
	noise_power_spectrum.SquareRoot();
	noise_power_spectrum.Reciprocal();

	if (is_running_locally == true) delete my_progress;

	if (is_running_locally == true)
	{
		wxPrintf("\nPreparing Stack...\n\n");
		total_positions = ((last_classaverage - first_classaverage) + 1) * number_of_classes;
		my_progress = new ProgressBar(total_positions);
	}

	// read the class average file..

	for (current_image = 0; current_image < wanted_class_averages.number_of_lines; current_image++)
	{
		wanted_class_averages.ReadLine(temp_float);
		classaverages_to_make.Add(int(temp_float[0]));
	}

	for (current_classaverage = first_classaverage; current_classaverage <= last_classaverage; current_classaverage++)
	{
		// get all image members of the selected class

		class_members.Clear();

		for (line_counter = 0; line_counter < input_star_file.ReturnNumberofLines(); line_counter++)
		{

			if (input_star_file.ReturnBest2DClass(line_counter) == classaverages_to_make[current_classaverage])
			{
				temp_line = input_star_file.ReturnLine(line_counter);

				class_members.Add(temp_line);
			}
		}

		// ok make it..

		for (class_counter = 0; class_counter < number_of_classes; class_counter++)
		{

			sum_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
			sum_image.SetToConstant(0.0f);
			sum_image.is_in_real_space = false;
			rotated_image.is_in_real_space = false;

			ctf_sum_image.SetToConstant(0.0f);
			ctf_sum_image.is_in_real_space = false;

			for (image_counter = 0; image_counter < images_per_class; image_counter++)
			{
				random_image = myroundint(fabsf(global_random_number_generator.GetUniformRandom() * (class_members.GetCount() - 1)));
				//random_image = image_counter;// + 1;
				//wxPrintf("random = %i\n", random_image);
				input_image.ReadSlice(&input_file, class_members[random_image].position_in_stack);
				input_image.ChangePixelSize(&input_image, output_pixel_size / input_star_file.ReturnPixelSize(current_image - 1), 0.001f);
				if (invert_contrast == true) input_image.InvertRealValues();
				input_image.ForwardFFT();
				input_image.ApplyCurveFilter(&noise_power_spectrum);

				current_ctf.Init(class_members[random_image].microscope_voltage_kv, class_members[random_image].microscope_spherical_aberration_mm, class_members[random_image].amplitude_contrast, class_members[random_image].defocus_1, class_members[random_image].defocus_2, class_members[random_image].defocus_angle, output_pixel_size, class_members[random_image].phase_shift);
				ctf_input_image.CalculateCTFImage(current_ctf);

				input_image.PhaseShift(class_members[random_image].x_shift / output_pixel_size, class_members[random_image].y_shift / output_pixel_size);
				rotation_angle.Init(0.0, 0.0, -class_members[random_image].psi, 0, 0);

				input_image.SwapRealSpaceQuadrants();
				input_image.MultiplyPixelWiseReal(ctf_input_image);

				input_image.RotateFourier2D(rotated_image, rotation_angle);

				sum_image.AddImage(&rotated_image);
				ctf_input_image.MultiplyPixelWiseReal(ctf_input_image);
				ctf_input_image.object_is_centred_in_box = false;
				ctf_input_image.RotateFourier2D(rotated_image, rotation_angle);
				ctf_sum_image.AddImage(&rotated_image);
			}

			for (pixel_counter = 0; pixel_counter < sum_image.real_memory_allocated / 2; pixel_counter++)
			{
				if (abs(ctf_sum_image.complex_values[pixel_counter]) != 0.0f) sum_image.complex_values[pixel_counter] /= (abs(ctf_sum_image.complex_values[pixel_counter]) + images_per_class /2);
			}

			sum_image.SwapRealSpaceQuadrants();
			sum_image.CosineMask(0.45,0.1);

			if (resample_box == true)
			{
				sum_image.Resize(wanted_output_box_size, wanted_output_box_size, 1);
			}

			sum_image.BackwardFFT();
			variance = sum_image.ReturnVarianceOfRealValues(sum_image.physical_address_of_box_center_x - mask_falloff / (output_pixel_size * bin_factor), 0.0, 0.0, 0.0, true);
			average = sum_image.ReturnAverageOfRealValues(sum_image.physical_address_of_box_center_x - mask_falloff / (output_pixel_size * bin_factor), true);
			sum_image.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));

			output_file_position = (number_of_classes * current_classaverage + class_counter) + 1;

			if (is_running_locally == true)
			{
			//	wxPrintf("Writing to %i (%i, %i)\n", output_file_position, current_classaverage, class_counter);
				sum_image.WriteSlice(output_file, output_file_position);
				current_position++;
				my_progress->Update(current_position);
			}
			else
			{
				SendProcessedImageResult(&sum_image, output_file_position, output_classaverage_images);
			}

			}

	}

	if (is_running_locally == true)
	{
		delete my_progress;
		delete output_file;
	}

	if (is_running_locally == true) wxPrintf("\nPrepareStack: Normal termination\n\n");
	//else wxSleep(10); // to make sure we don't die before the image data has been sent over completely (not sure if necessary)

	return true;
}

