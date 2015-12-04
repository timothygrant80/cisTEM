#include "../../core/core_headers.h"

class
UnBlurApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

IMPLEMENT_APP(UnBlurApp)

// override the DoInteractiveUserInput

void UnBlurApp::DoInteractiveUserInput()
{
	std::string input_filename;
	std::string output_filename;
	float original_pixel_size = 1;
	float minimum_shift_in_angstroms = 2;
	float maximum_shift_in_angstroms = 80;
	bool should_dose_filter = true;
	bool should_restore_power = true;
	float termination_threshold_in_angstroms = 1;
	int max_iterations = 20;
	float bfactor_in_angstroms = 1500;
	bool should_mask_central_cross = true;
	int horizontal_mask_size = 1;
	int vertical_mask_size = 1;
	float exposure_per_frame = 0.0;
	float acceleration_voltage = 300.0;
	float pre_exposure_amount = 0.0;

	bool set_expert_options;

	 UserInput *my_input = new UserInput("Unblur", 1.0);

	 input_filename = my_input->GetFilenameFromUser("Input stack filename", "The input file, containing your raw movie frames", "my_movie.mrc", true );
	 output_filename = my_input->GetFilenameFromUser("Output aligned sum", "The output file, containing a weighted sum of the aligned input frames", "my_aligned_sum.mrc", false);
	 original_pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	 should_dose_filter = my_input->GetYesNoFromUser("Apply Exposure filter?", "Apply an exposure-dependent filter to frames before summing them", "YES");

	 if (should_dose_filter == true)
	 {
		 acceleration_voltage = my_input->GetFloatFromUser("Acceleration voltage (kV)", "Acceleration voltage during imaging", "300.0");
		 exposure_per_frame = my_input->GetFloatFromUser("Exposure per frame (e/A^2)", "Exposure per frame, in electrons per square Angstrom", "1.0", 0.0);
	 	 pre_exposure_amount = my_input->GetFloatFromUser("Pre-exposure amount (e/A^2)", "Amount of pre-exposure prior to the first frame, in electrons per square Angstrom", "0.0", 0.0);
	 }
	 else
	 {
	 	 exposure_per_frame = 0.0;
	 	 acceleration_voltage = 300.0;
	 	 pre_exposure_amount = 0.0;
	 }

	 set_expert_options = my_input->GetYesNoFromUser("Set Expert Options?", "Set these for more control, hopefully not needed", "NO");

	 if (set_expert_options == true)
	 {
	 	 minimum_shift_in_angstroms = my_input->GetFloatFromUser("Minimum shift for initial search (A)", "Initial search will be limited to between the inner and outer radii.", "2.0", 0.0);
	 	 maximum_shift_in_angstroms = my_input->GetFloatFromUser("Outer radius shift limit (A)", "The maximum shift of each alignment step will be limited to this value.", "80.0", minimum_shift_in_angstroms);
	 	 bfactor_in_angstroms = my_input->GetFloatFromUser("B-factor to apply to images (A^2)", "This B-Factor will be used to filter the reference prior to alignment", "1500", 0.0);
	 	 vertical_mask_size = my_input->GetIntFromUser("Half-width of vertical Fourier mask", "The vertical line mask will be twice this size. The central cross mask helps\nreduce problems by line artefacts from the detector", "1", 1);
	 	 horizontal_mask_size = my_input->GetIntFromUser("Half-width of horizontal Fourier mask", "The horizontal line mask will be twice this size. The central cross mask helps\nreduce problems by line artefacts from the detector", "1", 1);
	 	 termination_threshold_in_angstroms = my_input->GetFloatFromUser("Termination shift threshold (A)", "Alignment will iterate until the maximum shift is below this value", "1", 0.0);
	 	 max_iterations = my_input->GetIntFromUser("Maximum number of iterations", "Alignment will stop at this number, even if the threshold shift is not reached", "20", 0);

	 	 if (should_dose_filter == true)
	 	 {
	 		 should_restore_power = my_input->GetYesNoFromUser("Restore Noise Power?", "Restore the power of the noise to the level it would be without exposure filtering", "YES");
	 	 }
 	 }
 	 else
 	 {
 		 minimum_shift_in_angstroms = original_pixel_size + 0.001;
 		 maximum_shift_in_angstroms = 100.0;
 		 bfactor_in_angstroms = 1500.0;
 		 vertical_mask_size = 1;
 		 horizontal_mask_size = 1;
 		 termination_threshold_in_angstroms = original_pixel_size / 2;
 		 max_iterations = 20;
 		 should_restore_power = true;
 	 }

	 delete my_input;

	 my_current_job.Reset(16);
	 my_current_job.ManualSetArguments("ttfffbbfifbiifff",  input_filename.c_str(),
			 	 	 	 	 	 	 	 	 	 	 	 output_filename.c_str(),
														 original_pixel_size,
														 minimum_shift_in_angstroms,
														 maximum_shift_in_angstroms,
														 should_dose_filter,
														 should_restore_power,
														 termination_threshold_in_angstroms,
														 max_iterations,
														 bfactor_in_angstroms,
														 should_mask_central_cross,
														 horizontal_mask_size,
														 vertical_mask_size,
														 acceleration_voltage,
														 exposure_per_frame,
														 pre_exposure_amount);


}

// overide the do calculation method which will be what is actually run..

bool UnBlurApp::DoCalculation()
{
	int pre_binning_factor;
	long image_counter;
	int pixel_counter;

	float unitless_bfactor;

	float pixel_size;
	float min_shift_in_pixels;
	float max_shift_in_pixels;
	float termination_threshold_in_pixels;

	Image sum_image;

	// get the arguments for this job..

	std::string input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	std::string output_filename 					= my_current_job.arguments[1].ReturnStringArgument();
	float       original_pixel_size					= my_current_job.arguments[2].ReturnFloatArgument();
	float 		minumum_shift_in_angstroms			= my_current_job.arguments[3].ReturnFloatArgument();
	float 		maximum_shift_in_angstroms			= my_current_job.arguments[4].ReturnFloatArgument();
	bool 		should_dose_filter					= my_current_job.arguments[5].ReturnBoolArgument();
	bool        should_restore_power				= my_current_job.arguments[6].ReturnBoolArgument();
	float 		termination_threshold_in_angstoms	= my_current_job.arguments[7].ReturnFloatArgument();
	int         max_iterations						= my_current_job.arguments[8].ReturnIntegerArgument();
	float 		bfactor_in_angstoms					= my_current_job.arguments[9].ReturnFloatArgument();
	bool        should_mask_central_cross			= my_current_job.arguments[10].ReturnBoolArgument();
	int         horizontal_mask_size				= my_current_job.arguments[11].ReturnIntegerArgument();
	int         vertical_mask_size					= my_current_job.arguments[12].ReturnIntegerArgument();
	float       acceleration_voltage				= my_current_job.arguments[13].ReturnFloatArgument();
	float       exposure_per_frame                  = my_current_job.arguments[14].ReturnFloatArgument();
	float       pre_exposure_amount                 = my_current_job.arguments[15].ReturnFloatArgument();

	//my_current_job.PrintAllArguments();

	// The Files

	MRCFile input_file(input_filename, false);
	MRCFile output_file(output_filename, true);

	long number_of_input_images = input_file.ReturnNumberOfSlices();

	// Arrays to hold the input images

	Image *unbinned_image_stack; // We will allocate this later depending on if we are binning or not.
	Image *image_stack = new Image[number_of_input_images];

	// Arrays to hold the shifts..

	float *x_shifts = new float[number_of_input_images];
	float *y_shifts = new float[number_of_input_images];

	// Arrays to hold the 1D dose filter, and 1D restoration filter..

	float *dose_filter;
	float *dose_filter_sum_of_squares;

	// Electron dose object for if dose filtering..

	ElectronDose my_electron_dose(acceleration_voltage, original_pixel_size);

	// some quick checks..

	if (number_of_input_images <= 2)
	{
		SendError(wxString::Format("Error: Movie (%s) contains less than 3 frames.. Terminating.", input_filename));
		ExitMainLoop();
	}

	// Read in and FFT all the images..

	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		image_stack[image_counter].ReadSlice(&input_file, image_counter + 1);
		image_stack[image_counter].ForwardFFT(true);

		x_shifts[image_counter] = 0.0;
		y_shifts[image_counter] = 0.0;

	}

	// if we are binning - choose a binning factor..

	pre_binning_factor = int(myround(5. / original_pixel_size));
	if (pre_binning_factor < 1) pre_binning_factor = 1;

	wxPrintf("Prebinning factor = %i\n", pre_binning_factor);

	// if we are going to be binning, we need to allocate the unbinned array..

	if (pre_binning_factor > 1)
	{
		unbinned_image_stack = new Image[number_of_input_images];
		pixel_size = original_pixel_size * pre_binning_factor;
	}
	else
	{
		pixel_size = original_pixel_size;
	}

	// convert shifts to pixels..

	min_shift_in_pixels = minumum_shift_in_angstroms / pixel_size;
	max_shift_in_pixels = maximum_shift_in_angstroms / pixel_size;
	termination_threshold_in_pixels = termination_threshold_in_angstoms / pixel_size;


	// calculate the bfactor

	unitless_bfactor = bfactor_in_angstoms / pow(pixel_size, 2);

	if (min_shift_in_pixels <= 1.01) min_shift_in_pixels = 1.01;  // we always want to ignore the central peak initially.

	if (pre_binning_factor > 1)
	{
		for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
		{
			unbinned_image_stack[image_counter] = image_stack[image_counter];
			image_stack[image_counter].Resize(unbinned_image_stack[image_counter].logical_x_dimension / pre_binning_factor, unbinned_image_stack[image_counter].logical_y_dimension / pre_binning_factor, 1);
			//image_stack[image_counter].QuickAndDirtyWriteSlice("binned.mrc", image_counter + 1);
		}

		// for the binned images, we don't want to insist on a super low termination factor.

		if (termination_threshold_in_pixels < 1 && pre_binning_factor > 1) termination_threshold_in_pixels = 1;

	}

	// do the initial refinement (only 1 round - with the min shift)

//	unblur_refine_alignment(image_stack, number_of_input_images, 1, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, min_shift_in_pixels, max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, x_shifts, y_shifts);

	// now do the actual refinement..

//	unblur_refine_alignment(image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, x_shifts, y_shifts);


	// if we have been using pre-binning, we need to do a refinment on the unbinned data..

	if (pre_binning_factor > 1)
	{
		// we don't need the binned images anymore..

		delete [] image_stack;
		image_stack = unbinned_image_stack;
		pixel_size = original_pixel_size;

		// Adjust the shifts, then phase shift the original images

		for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
		{
			x_shifts[image_counter] *= pre_binning_factor;
			y_shifts[image_counter] *= pre_binning_factor;

			image_stack[image_counter].PhaseShift(x_shifts[image_counter], y_shifts[image_counter], 0.0);
		}

		// convert parameters to pixels with new pixel size..

		min_shift_in_pixels = minumum_shift_in_angstroms / original_pixel_size;
		max_shift_in_pixels = maximum_shift_in_angstroms / original_pixel_size;
		termination_threshold_in_pixels = termination_threshold_in_angstoms / original_pixel_size;

		// recalculate the bfactor

		unitless_bfactor = bfactor_in_angstoms / pow(original_pixel_size, 2);

		// do the refinement..

//		unblur_refine_alignment(image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, original_pixel_size, x_shifts, y_shifts);

		// if allocated delete the binned stack, and swap the unbinned to image_stack - so that no matter what is happening we can just use image_stack



	}

	// we should be finished with alignment, now we just need to make the final sum..

	if (should_dose_filter == true)
	{
		// allocate arrays for the filter, and the sum of squares..

		dose_filter = new float[image_stack[0].real_memory_allocated / 2];
		dose_filter_sum_of_squares = new float[image_stack[0].real_memory_allocated / 2];

		for (pixel_counter = 0; pixel_counter < image_stack[0].real_memory_allocated / 2; pixel_counter++)
		{
			dose_filter[pixel_counter] = 0.0;
			dose_filter_sum_of_squares[pixel_counter] = 0.0;
		}

		for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
		{
			my_electron_dose.CalculateDoseFilterAs1DArray(&image_stack[image_counter], dose_filter, (image_counter * exposure_per_frame) + pre_exposure_amount, ((image_counter + 1) * exposure_per_frame) + pre_exposure_amount);

			// filter the image, and also calculate the sum of squares..

			for (pixel_counter = 0; pixel_counter < image_stack[image_counter].real_memory_allocated / 2; pixel_counter++)
			{
				image_stack[image_counter].complex_values[pixel_counter] *= dose_filter[pixel_counter];
				dose_filter_sum_of_squares[pixel_counter] += pow(dose_filter[pixel_counter], 2);

				//if (image_counter == 65) wxPrintf("%f\n", dose_filter[pixel_counter]);
			}
		}
	}

	sum_image.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
	sum_image.SetToConstant(0.0);

	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		sum_image.AddImage(&image_stack[image_counter]);
		wxPrintf("#%li = %f, %f\n", image_counter, x_shifts[image_counter] * pixel_size, y_shifts[image_counter] * pixel_size);
	}

	// if we are restoring the power - do it here..

	if (should_dose_filter == true && should_restore_power == true)
	{
		for (pixel_counter = 0; pixel_counter < sum_image.real_memory_allocated / 2; pixel_counter++)
		{
			if (dose_filter_sum_of_squares[pixel_counter] != 0)
			{
				sum_image.complex_values[pixel_counter] /= sqrt(dose_filter_sum_of_squares[pixel_counter]);
			}
		}
	}

	// now we just need to write out the final sum..

	sum_image.WriteSlice(&output_file, 1);

	// fill the result array..

	if (result_array != NULL)
	{
		delete [] result_array;

	}

	result_array = new float[number_of_input_images * 2];
	result_array_size = number_of_input_images * 2;

	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		result_array[image_counter] = x_shifts[image_counter] * original_pixel_size;
		result_array[image_counter + number_of_input_images] = y_shifts[image_counter] * original_pixel_size;

		wxPrintf("image #%li = %f, %f\n", image_counter, result_array[image_counter], result_array[image_counter + number_of_input_images]);
	}


	delete [] x_shifts;
	delete [] y_shifts;
	delete [] image_stack;

	if (should_dose_filter == true)
	{
		delete [] dose_filter;
		delete [] dose_filter_sum_of_squares;
	}

	return true;
}




