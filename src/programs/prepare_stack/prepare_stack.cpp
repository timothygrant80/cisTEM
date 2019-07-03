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
	wxString	output_particle_images;
	wxString 	input_star_file;

	int	 wanted_output_box_size;

	bool resample_box;
	bool process_a_subset = false;
	float mask_radius;
	float pixel_size;
	int first_particle = 0;
	int last_particle = 0;

	UserInput *my_input = new UserInput("PrepareStack", 1.00);

	input_particle_images = my_input->GetFilenameFromUser("Input particle stack", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
	input_star_file = my_input->GetFilenameFromUser("Input cisTEM star file", "The input parameter file, containing your particle parameters", "my_parameters.star", true);
	output_particle_images = my_input->GetFilenameFromUser("Output particle stack", "The output image stack, containing the prepared particle images", "my_image_stack_prep.mrc", false);
	pixel_size = my_input->GetFloatFromUser("Wanted Output Pixel Size (Angstroms)", "Wanted pixel size of the images", "1", 0.0);
	mask_radius = my_input->GetFloatFromUser("Mask radius (Angstroms)", "For calculating noise statistics", "100", 0.0);
	resample_box = my_input->GetYesNoFromUser("Resample output?","If yes you can resample the output image to a specified size", "NO");

	if (resample_box == true) wanted_output_box_size = my_input->GetIntFromUser("Resampled box size", "How big to resample the box size to?", "512", 0.0);
	else wanted_output_box_size = 1;

	delete my_input;


	my_current_job.ManualSetArguments("tttffbibii",	input_particle_images.ToUTF8().data(),
												input_star_file.ToUTF8().data(),
												output_particle_images.ToUTF8().data(),
												pixel_size,
												mask_radius,
												resample_box,
												wanted_output_box_size,
												process_a_subset,
												first_particle,
												last_particle);
}

// override the do calculation method which will be what is actually run..

bool PrepareStackApp::DoCalculation()
{


	wxString input_particle_images 				= my_current_job.arguments[0].ReturnStringArgument();
	wxString input_star_filename				= my_current_job.arguments[1].ReturnStringArgument();
	wxString output_particle_images 			= my_current_job.arguments[2].ReturnStringArgument();
	float output_pixel_size				       	= my_current_job.arguments[3].ReturnFloatArgument();
	float  mask_radius                          = my_current_job.arguments[4].ReturnFloatArgument();
	bool resample_box							= my_current_job.arguments[5].ReturnBoolArgument();
	int	 wanted_output_box_size 				= my_current_job.arguments[6].ReturnIntegerArgument();
	bool process_a_subset						= my_current_job.arguments[7].ReturnBoolArgument();
	int  first_particle						    = my_current_job.arguments[8].ReturnIntegerArgument();
	int  last_particle						    = my_current_job.arguments[9].ReturnIntegerArgument();

	ProgressBar *my_progress;
	int max_samples = 2000;
	int images_to_process = last_particle - first_particle;
	long current_image;
	float mask_radius_for_noise = mask_radius / output_pixel_size;
	float mask_falloff = 10.0f;
	float variance;
	float average;

	if (is_running_locally == false)
	{
		float result;
		my_result.SetResult(1, &result);
	}


	ImageFile input_file(input_particle_images.ToStdString());
	MRCFile *output_file;

	if (is_running_locally == true) output_file = new MRCFile(output_particle_images.ToStdString(), true);

	Image input_image;
	Image sum_power;
	Image temp_image;

	Curve noise_power_spectrum;
	Curve number_of_terms;
	RandomNumberGenerator random_generator;

	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1);
	sum_power.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), 1, false);
	sum_power.SetToConstant(0.0);

	cisTEMParameters input_star_file;
	input_star_file.ReadFromcisTEMStarFile(input_star_filename);

	if (process_a_subset == false)
	{
		first_particle = 1;
		last_particle = input_file.ReturnNumberOfSlices();
	}

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
		if (current_image < first_particle || current_image > last_particle) continue;
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

	if (is_running_locally == true) wxPrintf("Preparing Stack...\n\n");
	if (is_running_locally == true) my_progress = new ProgressBar(input_file.ReturnNumberOfSlices());

	for (current_image = 1; current_image <= input_file.ReturnNumberOfSlices(); current_image++)
	{
		if (current_image < first_particle || current_image > last_particle) continue;

		input_image.ReadSlice(&input_file, current_image);
		input_image.ChangePixelSize(&input_image, output_pixel_size / input_star_file.ReturnPixelSize(current_image - 1), 0.001f);
		input_image.ForwardFFT();
		input_image.ApplyCurveFilter(&noise_power_spectrum);

		if (resample_box == true)
		{
			input_image.Resize(wanted_output_box_size, wanted_output_box_size, 1);
		}

		input_image.BackwardFFT();
		variance = input_image.ReturnVarianceOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / output_pixel_size, 0.0, 0.0, 0.0, true);
		average = input_image.ReturnAverageOfRealValues(input_image.physical_address_of_box_center_x - mask_falloff / output_pixel_size, true);
		input_image.AddMultiplyConstant(- average, 1.0 / sqrtf(variance));

		if (is_running_locally == true)
		{
			input_image.WriteSlice(output_file, current_image);
		}

		if (is_running_locally == true) my_progress->Update(current_image);
		else
		{
			SendProcessedImageResult(&input_image, current_image, output_particle_images);
		}
	}

	if (is_running_locally == true) delete my_progress;


	if (is_running_locally == true) wxPrintf("\nPrepareStack: Normal termination\n\n");
	//else wxSleep(10); // to make sure we don't die before the image data has been sent over completely (not sure if necessary)

	return true;
}

