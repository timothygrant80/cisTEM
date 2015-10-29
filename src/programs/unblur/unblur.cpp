#include "../../core/core_headers.h"

class
UnBlurApp : public MyApp
{

	public:

	bool DoCalculation();

	private:

};

void unblur_refine_alignment(Image *input_stack, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float pixel_size, float *x_shifts, float *y_shifts);

IMPLEMENT_APP(UnBlurApp)

// overide the do calculation method which will be what is actually run..

bool UnBlurApp::DoCalculation()
{
	int pre_binning_factor;
	long image_counter;

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

	unitless_bfactor = bfactor_in_angstoms / pow(pixel_size, 2);

	// The Files

	MRCFile input_file(input_filename, false);
	MRCFile output_file(output_filename, true);

	long number_of_input_images = input_file.ReturnNumberOfSlices();

	// Arrays to hold the input images

	Image *unbinned_image_stack; // We will allocate this later depending on if we are binning or not.
	Image *image_stack = new Image[number_of_input_images];

	// Arrays to hold the shifts..

	float *x_shifts = new float(number_of_input_images);
	float *y_shifts = new float(number_of_input_images);

	// some quick checks..

	if (number_of_input_images <= 2)
	{
		SendError(wxString::Format("Error: Movie (%s) contains less than 3 frames.. Terminating.", input_filename));
		ExitMainLoop();
	}

	// if we are binning - choose a binning factor..

	pre_binning_factor = int(myround(float(image_stack[0].logical_x_dimension) / 1024.0));
	if (pre_binning_factor < 1) pre_binning_factor = 1;

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

	min_shift_in_pixels = min_shift_in_pixels / pixel_size;
	max_shift_in_pixels = min_shift_in_pixels / pixel_size;
	termination_threshold_in_pixels = termination_threshold_in_angstoms / pixel_size;

	if (min_shift_in_pixels <= 1) min_shift_in_pixels = 1;  // we always want to ignore the central peak initially.

	// Read in and FFT all the images..

	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		image_stack[image_counter].ReadSlice(&input_file, image_counter);
		image_stack[image_counter].ForwardFFT(true);

		if (pre_binning_factor > 1)
		{
			unbinned_image_stack[image_counter] = image_stack[image_counter];
			image_stack[image_counter].Resize(image_stack[image_counter].logical_x_dimension / pre_binning_factor, image_stack[image_counter].logical_y_dimension / pre_binning_factor, 1);

		}

	}

	// do the initial refinement (only 1 round - with the min shift)

	unblur_refine_alignment(image_stack, number_of_input_images, 1, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, min_shift_in_pixels, max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, x_shifts, y_shifts);

	// now do the actual refinement..

	unblur_refine_alignment(image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, x_shifts, y_shifts);

	// if we have been using pre-binning, we need to do a refinment on the unbinned data..

	if (pre_binning_factor > 1)
	{
		// first adjust the shifts, then phase shift the original images

		for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
		{
			x_shifts[image_counter] *= pre_binning_factor;
			y_shifts[image_counter] *= pre_binning_factor;

			unbinned_image_stack[image_counter].PhaseShift(x_shifts[image_counter], y_shifts[image_counter], 1.0);
		}

		// convert shifts to pixels with new pixel size..

		min_shift_in_pixels = min_shift_in_pixels / original_pixel_size;
		max_shift_in_pixels = min_shift_in_pixels / original_pixel_size;
		termination_threshold_in_pixels = termination_threshold_in_angstoms / original_pixel_size;

		// do the refinement..

		unblur_refine_alignment(unbinned_image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, original_pixel_size, x_shifts, y_shifts);

		// if allocated delete the binned stack, and swap the unbinned to image_stack - so that no matter what is happening we can just use image_stack

		delete [] image_stack;
		image_stack = unbinned_image_stack;
		pixel_size = original_pixel_size;

	}

	// we should be finished with alignment, now we just need to make the final sum..

	/*
    ! Dose filtering
    if (apply_dose_filter%value) then
        do image_counter = 1,number_of_frames_per_movie%value
            call my_electron_dose%ApplyDoseFilterToImage(image_stack(image_counter), &
                                                         dose_start=((image_counter-1)*exposure_per_frame%value) &
                                                                    + pre_exposure_amount%value, &
                                                         dose_finish=(image_counter*exposure_per_frame%value) &
                                                                    + pre_exposure_amount%value, &
                                                         pixel_size=pixel_size%value)
        enddo
    endif

    */


	sum_image.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
	sum_image.SetToConstant(0.0);

	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		sum_image.AddImage(&image_stack[image_counter]);
	}

	// now we just need to write out the final sum..

	sum_image.WriteSlice(&output_file, 1);

	delete [] x_shifts;
	delete [] y_shifts;
	delete [] image_stack;

	return true;
}

void unblur_refine_alignment(Image *input_stack, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float pixel_size, float *x_shifts, float *y_shifts)
{
	long pixel_counter;
	long image_counter;
	long iteration_counter;

	int number_of_middle_image;

	float *current_x_shifts = new float(number_of_images);
	float *current_y_shifts = new float(number_of_images);

	float max_shift;

	Image sum_of_images;
	Image sum_of_images_minus_current;

	Peak my_peak;

	sum_of_images.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);
	sum_of_images.SetToConstant(0.0);

	sum_of_images_minus_current.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);

	// get the middle image

	if (IsOdd(number_of_images))
	{
		number_of_middle_image = (number_of_images +1) / 2;
	}
	else
	{
		number_of_middle_image = number_of_images / 2 + 1;
	}

	// prepare the initial sum

	for (image_counter = 0; image_counter < number_of_images; image_counter++)
	{
		sum_of_images.AddImage(&input_stack[image_counter]);
	}

	/* prepare smoothing curves

    ! Prepare the smoothing curve
    call x_shifts%Init(size(stack_of_images))
    call y_shifts%Init(size(stack_of_images))

    */

	// perform the main alignment loop until we reach a max shift less than wanted, or max iterations

	for (iteration_counter = 0; iteration_counter < max_iterations; iteration_counter++)
	{
		max_shift = -std::numeric_limits<float>::max();

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			// prepare the sum reference by subtracting out the current image, applying a bfactor and masking central cross

			sum_of_images_minus_current = sum_of_images;
			sum_of_images_minus_current.SubtractImage(&input_stack[image_counter]);
			sum_of_images_minus_current.ApplyBFactor(unitless_bfactor);

			if (mask_central_cross == true)
			{
				sum_of_images_minus_current.MaskCentralCross(width_of_vertical_line, width_of_horizontal_line);
			}

			// compute the cross correaltion function and find the peak

			sum_of_images_minus_current.CalculateCrossCorrelationImageWith(&input_stack[image_counter]);
			my_peak = sum_of_images_minus_current.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);

			// update the shifts..

			current_x_shifts[image_counter] += my_peak.x;
			current_y_shifts[image_counter] += my_peak.y;

			if (my_peak.x > max_shift) max_shift = my_peak.x;
			if (my_peak.y > max_shift) max_shift = my_peak.y;

			/*
			 ! Apply [spline|polynomial|no] smoothing to the shifts
        	 call x_shifts%ClearData()
             call y_shifts%ClearData()
             do image_counter=1, size(stack_of_images)
               call x_shifts%AddPoint(real(image_counter), additional_shifts_x(image_counter))
               call y_shifts%AddPoint(real(image_counter), additional_shifts_y(image_counter))
             enddo


        select case (smoothing_type)
            case (spline)
                call x_shifts%FitSplineToData()
                call x_shifts%CopySplineModel(additional_shifts_x)
                call y_shifts%FitSplineToData()
                call y_shifts%CopySplineModel(additional_shifts_y)
            case (no_smoothing)
                call x_shifts%CopyYData(additional_shifts_x)
                call y_shifts%CopyYData(additional_shifts_y)
            case (polynomial)
                order_of_polynomial = min(6,size(stack_of_images)-1)
                call x_shifts%FitPolynomialToData(order_of_polynomial)
                call x_shifts%CopyPolynomialModel(additional_shifts_x)
                call y_shifts%FitPolynomialToData(order_of_polynomial)
                call y_shifts%CopyPolynomialModel(additional_shifts_y)
        end select

			 */

			// work out the maximum_x_shif


		}

		// subtract shift of the middle image from all images to keep things centred around it

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			current_x_shifts[image_counter] -= current_x_shifts[number_of_middle_image];
			current_y_shifts[image_counter] -= current_y_shifts[number_of_middle_image];
		}

		// actually shift the images, also add the subtracted shifts to the overall shifts

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			input_stack[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 1.0);

			x_shifts[image_counter] += current_x_shifts[image_counter];
			y_shifts[image_counter] += current_y_shifts[image_counter];
		}

		// check to see if the convergence criteria have been reached and return if so

		if (iteration_counter >= max_iterations || max_shift <= max_shift_convergence_threshold)
		{
			delete [] current_x_shifts;
			delete [] current_y_shifts;
			return;
		}

		// going to be doing another round so we need to make the new sum..

		sum_of_images.SetToConstant(0.0);

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			sum_of_images.AddImage(&input_stack[image_counter]);
		}

	}

	delete [] current_x_shifts;
	delete [] current_y_shifts;

}





