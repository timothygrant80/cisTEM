#include "../../core/core_headers.h"

class
UnBlurApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

void unblur_refine_alignment(Image *input_stack, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float pixel_size, float *x_shifts, float *y_shifts);

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
	bool movie_is_gain_corrected = true;
	wxString gain_filename = "";
	float output_binning_factor = 1;

	bool set_expert_options;
	bool correct_mag_distortion;
	float mag_distortion_angle;
	float mag_distortion_major_scale;
	float mag_distortion_minor_scale;
	int first_frame;
	int last_frame;



	 UserInput *my_input = new UserInput("Unblur", 1.0);

	 input_filename = my_input->GetFilenameFromUser("Input stack filename", "The input file, containing your raw movie frames", "my_movie.mrc", true );
	 output_filename = my_input->GetFilenameFromUser("Output aligned sum", "The output file, containing a weighted sum of the aligned input frames", "my_aligned_sum.mrc", false);
	 original_pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	 output_binning_factor = my_input->GetFloatFromUser("Output binning factor", "Output images will be binned (downsampled) by this factor relative to the input images", "1", 1);
	 should_dose_filter = my_input->GetYesNoFromUser("Apply Exposure filter?", "Apply an exposure-dependent filter to frames before summing them", "yes");

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

	 set_expert_options = my_input->GetYesNoFromUser("Set Expert Options?", "Set these for more control, hopefully not needed", "no");

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
	 		 should_restore_power = my_input->GetYesNoFromUser("Restore Noise Power?", "Restore the power of the noise to the level it would be without exposure filtering", "yes");
	 	 }

	 	 movie_is_gain_corrected = my_input->GetYesNoFromUser("Input stack is gain-corrected?", "The input frames are already gain-corrected", "yes");

	 	 if (!movie_is_gain_corrected)
	 	 {
	 		 gain_filename = my_input->GetFilenameFromUser("Gain image filename", "The filename of the camera's gain reference image", "my_gain_reference.dm4", true);
	 	 }

	 	 first_frame = my_input->GetIntFromUser("First frame to use for sum", "You can use this to ignore the first n frames", "1", 1);
	 	 last_frame = my_input->GetIntFromUser("Last frame to use for sum (0 for last frame)", "You can use this to ignore the last n frames", "0", 0);
 	 }
 	 else
 	 {
 		 minimum_shift_in_angstroms = original_pixel_size * output_binning_factor + 0.001;
 		 maximum_shift_in_angstroms = 100.0;
 		 bfactor_in_angstroms = 1500.0;
 		 vertical_mask_size = 1;
 		 horizontal_mask_size = 1;
 		 termination_threshold_in_angstroms = original_pixel_size * output_binning_factor / 2;
 		 max_iterations = 20;
 		 should_restore_power = true;
 		 movie_is_gain_corrected = true;
 		 gain_filename = "";
 		 first_frame = 1;
 		 last_frame = 0;

 	 }

	correct_mag_distortion = my_input->GetYesNoFromUser("Correct Magnification Distortion?", "If yes, a magnification distortion can be corrected", "no");

	if (correct_mag_distortion == true)
	{
		mag_distortion_angle = my_input->GetFloatFromUser("Distortion Angle (Degrees)", "The distortion angle in degrees", "0.0");
		mag_distortion_major_scale = my_input->GetFloatFromUser("Major Scale", "The major axis scale factor", "1.0", 0.0);
		mag_distortion_minor_scale = my_input->GetFloatFromUser("Minor Scale", "The minor axis scale factor", "1.0", 0.0);;
	}
	else
	{
		mag_distortion_angle = 0.0;
		mag_distortion_major_scale = 1.0;
		mag_distortion_minor_scale = 1.0;
	}

	 delete my_input;

	// this are defaulted to off in the interactive version for now
	bool write_out_amplitude_spectrum = false;
	std::string amplitude_spectrum_filename = "/dev/null";
	bool write_out_small_sum_image = false;
	std::string small_sum_image_filename = "/dev/null";

	my_current_job.Reset(25);
	my_current_job.ManualSetArguments("ttfffbbfifbiifffbsfbfffbtbtii",input_filename.c_str(),
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
																 pre_exposure_amount,
																 movie_is_gain_corrected,
																 gain_filename.ToStdString().c_str(),
																 output_binning_factor,
																 correct_mag_distortion,
																 mag_distortion_angle,
																 mag_distortion_major_scale,
																 mag_distortion_minor_scale,
																 write_out_amplitude_spectrum,
																 amplitude_spectrum_filename.c_str(),
																 write_out_small_sum_image,
																 small_sum_image_filename.c_str(),
																 first_frame,
																 last_frame);


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
	Image sum_image_no_dose_filter;

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
	bool		movie_is_gain_corrected				= my_current_job.arguments[16].ReturnBoolArgument();
	wxString	gain_filename						= my_current_job.arguments[17].ReturnStringArgument();
	float		output_binning_factor				= my_current_job.arguments[18].ReturnFloatArgument();
	bool        correct_mag_distortion				= my_current_job.arguments[19].ReturnBoolArgument();
    float       mag_distortion_angle				= my_current_job.arguments[20].ReturnFloatArgument();
    float       mag_distortion_major_scale          = my_current_job.arguments[21].ReturnFloatArgument();
	float       mag_distortion_minor_scale          = my_current_job.arguments[22].ReturnFloatArgument();
	bool 		write_out_amplitude_spectrum 		= my_current_job.arguments[23].ReturnBoolArgument();
	std::string amplitude_spectrum_filename 		= my_current_job.arguments[24].ReturnStringArgument();
	bool 		write_out_small_sum_image 			= my_current_job.arguments[25].ReturnBoolArgument();
	std::string small_sum_image_filename 			= my_current_job.arguments[26].ReturnStringArgument();
	int         first_frame							= my_current_job.arguments[27].ReturnIntegerArgument();
	int         last_frame							= my_current_job.arguments[28].ReturnIntegerArgument();


	//my_current_job.PrintAllArguments();

	// Profiling
	wxDateTime	overall_start = wxDateTime::Now();
	wxDateTime 	overall_finish;
	wxDateTime 	read_frames_start;
	wxDateTime	read_frames_finish;
	wxDateTime	first_alignment_start;
	wxDateTime	first_alignment_finish;
	wxDateTime	main_alignment_start;
	wxDateTime	main_alignment_finish;
	wxDateTime	final_alignment_start;
	wxDateTime	final_alignment_finish;


	// The Files

	if (! DoesFileExist(input_filename))
	{
		SendError(wxString::Format("Error: Input movie %s not found\n", input_filename));
		exit(-1);
	}
	ImageFile input_file(input_filename, false);
	//MRCFile output_file(output_filename, true); changed to quick and dirty write as the file is only used once, and this way it is not created until it is actually written, which is cleaner for cancelled / crashed jobs
	ImageFile gain_file;

	if (! movie_is_gain_corrected)
	{
		gain_file.OpenFile(gain_filename.ToStdString(), false);
	}

	long number_of_input_images = input_file.ReturnNumberOfSlices();

	if (last_frame == 0) last_frame = number_of_input_images;

	if (first_frame > number_of_input_images)
	{
		SendError(wxString::Format("(%s) First frame is greater than total number of frames, using frame 1 instead.", input_filename));
		first_frame = 1;
	}

	if (last_frame > number_of_input_images)
	{
		SendError(wxString::Format("(%s) Specified last frame is greater than total number of frames.. using last frame instead."));
		last_frame = number_of_input_images;
	}

	long slice_byte_size;

	Image *unbinned_image_stack; // We will allocate this later depending on if we are binning or not.
	Image *image_stack = new Image[number_of_input_images];
	Image gain_image;

	// output sizes..

	int output_x_size;
	int output_y_size;

	if (output_binning_factor > 1.0001)
	{
		output_x_size = myroundint(float(input_file.ReturnXSize()) / output_binning_factor);
		output_y_size = myroundint(float(input_file.ReturnYSize()) / output_binning_factor);
	}
	else
	{
		output_x_size = input_file.ReturnXSize();
		output_y_size = input_file.ReturnYSize();
	}


	// work out the output pixel size..

	float x_bin_factor = float(input_file.ReturnXSize()) / float(output_x_size);
	float y_bin_factor = float(input_file.ReturnYSize()) / float(output_y_size);
	float average_bin_factor = (x_bin_factor + y_bin_factor) / 2.0;

	float output_pixel_size = original_pixel_size * float(average_bin_factor);

	// change if we need to correct for the distortion..

	if (correct_mag_distortion == true)
	{
		output_pixel_size = ReturnMagDistortionCorrectedPixelSize(output_pixel_size, mag_distortion_major_scale, mag_distortion_minor_scale);
	}


	// Arrays to hold the shifts..

	float *x_shifts = new float[number_of_input_images];
	float *y_shifts = new float[number_of_input_images];

	// Arrays to hold the 1D dose filter, and 1D restoration filter..

	float *dose_filter;
	float *dose_filter_sum_of_squares;



	// Electron dose object for if dose filtering..

	ElectronDose *my_electron_dose;

	if (should_dose_filter == true) my_electron_dose = new ElectronDose(acceleration_voltage, output_pixel_size);

	// some quick checks..

	if (number_of_input_images <= 2)
	{
		SendError(wxString::Format("Error: Movie (%s) contains less than 3 frames.. Terminating.", input_filename));
		wxSleep(10);
		exit(-1);
	}

	// Read in gain reference
	if (!movie_is_gain_corrected) { gain_image.ReadSlice(&gain_file,1);	}

	// Read in, gain-correct, FFT and resample all the images..

	read_frames_start = wxDateTime::Now();
	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		// Read from disk
		image_stack[image_counter].ReadSlice(&input_file,image_counter+1);

		// Gain correction
		if (! movie_is_gain_corrected)
		{
			if (! image_stack[image_counter].HasSameDimensionsAs(&gain_image))
			{
				SendError(wxString::Format("Error: location %li of input file (%s) does not have same dimensions as the gain image (%s)", image_counter+1, input_filename, gain_filename));
				wxSleep(10);
				exit(-1);
			}
			//if (image_counter == 0) SendInfo(wxString::Format("Info: multiplying %s by gain %s\n",input_filename,gain_filename.ToStdString()));
			image_stack[image_counter].MultiplyPixelWise(gain_image);
		}

		image_stack[image_counter].ReplaceOutliersWithMean(6);

		if (correct_mag_distortion == true)
		{
			image_stack[image_counter].CorrectMagnificationDistortion(mag_distortion_angle, mag_distortion_major_scale, mag_distortion_minor_scale);
		}

		// FT
		image_stack[image_counter].ForwardFFT(true);
		image_stack[image_counter].ZeroCentralPixel();

		// Resize the FT (binning)
		if (output_binning_factor > 1.0001)
		{
			image_stack[image_counter].Resize(myroundint(image_stack[image_counter].logical_x_dimension/output_binning_factor),myroundint(image_stack[image_counter].logical_y_dimension/output_binning_factor),1);
		}

		// Init shifts
		x_shifts[image_counter] = 0.0;
		y_shifts[image_counter] = 0.0;
	}

	input_file.CloseFile();

	read_frames_finish = wxDateTime::Now();


	// if we are binning - choose a binning factor..

	pre_binning_factor = int(myround(5. / output_pixel_size));
	if (pre_binning_factor < 1) pre_binning_factor = 1;

//	wxPrintf("Prebinning factor = %i\n", pre_binning_factor);

	// if we are going to be binning, we need to allocate the unbinned array..

	if (pre_binning_factor > 1)
	{
		unbinned_image_stack = image_stack;
		image_stack = new Image[number_of_input_images];
		pixel_size = output_pixel_size * pre_binning_factor;
	}
	else
	{
		pixel_size = output_pixel_size;
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
			image_stack[image_counter].Allocate(unbinned_image_stack[image_counter].logical_x_dimension / pre_binning_factor, unbinned_image_stack[image_counter].logical_y_dimension / pre_binning_factor, 1, false);
			unbinned_image_stack[image_counter].ClipInto(&image_stack[image_counter]);
			//image_stack[image_counter].QuickAndDirtyWriteSlice("binned.mrc", image_counter + 1);
		}

		// for the binned images, we don't want to insist on a super low termination factor.

		if (termination_threshold_in_pixels < 1 && pre_binning_factor > 1) termination_threshold_in_pixels = 1;

	}

	// do the initial refinement (only 1 round - with the min shift)
	first_alignment_start = wxDateTime::Now();
	//SendInfo(wxString::Format("Doing first alignment on %s\n",input_filename));
	unblur_refine_alignment(image_stack, number_of_input_images, 1, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, min_shift_in_pixels, max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, x_shifts, y_shifts);
	first_alignment_finish = wxDateTime::Now();

	// now do the actual refinement..
	main_alignment_start = wxDateTime::Now();
	//SendInfo(wxString::Format("Doing main alignment on %s\n",input_filename));
	unblur_refine_alignment(image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, pixel_size, x_shifts, y_shifts);
	main_alignment_finish = wxDateTime::Now();


	// if we have been using pre-binning, we need to do a refinment on the unbinned data..
	final_alignment_start = wxDateTime::Now();
	if (pre_binning_factor > 1)
	{
		// we don't need the binned images anymore..

		delete [] image_stack;
		image_stack = unbinned_image_stack;
		pixel_size = output_pixel_size;

		// Adjust the shifts, then phase shift the original images

		for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
		{
			x_shifts[image_counter] *= pre_binning_factor;
			y_shifts[image_counter] *= pre_binning_factor;

			image_stack[image_counter].PhaseShift(x_shifts[image_counter], y_shifts[image_counter], 0.0);
		}

		// convert parameters to pixels with new pixel size..

		min_shift_in_pixels = minumum_shift_in_angstroms / output_pixel_size;
		max_shift_in_pixels = maximum_shift_in_angstroms / output_pixel_size;
		termination_threshold_in_pixels = termination_threshold_in_angstoms / output_pixel_size;

		// recalculate the bfactor

		unitless_bfactor = bfactor_in_angstoms / pow(output_pixel_size, 2);

		// do the refinement..
		//SendInfo(wxString::Format("Doing final unbinned alignment on %s\n",input_filename));
		unblur_refine_alignment(image_stack, number_of_input_images, max_iterations, unitless_bfactor, should_mask_central_cross, vertical_mask_size, horizontal_mask_size, 0., max_shift_in_pixels, termination_threshold_in_pixels, output_pixel_size, x_shifts, y_shifts);

		// if allocated delete the binned stack, and swap the unbinned to image_stack - so that no matter what is happening we can just use image_stack



	}
	final_alignment_finish = wxDateTime::Now();

	// we should be finished with alignment, now we just need to make the final sum..

	sum_image.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
	sum_image.SetToConstant(0.0);

	if (should_dose_filter == true)
	{
		if (write_out_amplitude_spectrum == true)
		{
			sum_image_no_dose_filter.Allocate(image_stack[0].logical_x_dimension, image_stack[0].logical_y_dimension, false);
			sum_image_no_dose_filter.SetToConstant(0.0);
		}

		// allocate arrays for the filter, and the sum of squares..

		dose_filter = new float[image_stack[0].real_memory_allocated / 2];
		dose_filter_sum_of_squares = new float[image_stack[0].real_memory_allocated / 2];

		for (pixel_counter = 0; pixel_counter < image_stack[0].real_memory_allocated / 2; pixel_counter++)
		{
			dose_filter[pixel_counter] = 0.0;
			dose_filter_sum_of_squares[pixel_counter] = 0.0;
		}

		for (image_counter = first_frame - 1; image_counter < last_frame; image_counter++)
		{
			my_electron_dose->CalculateDoseFilterAs1DArray(&image_stack[image_counter], dose_filter, (image_counter * exposure_per_frame) + pre_exposure_amount, ((image_counter + 1) * exposure_per_frame) + pre_exposure_amount);

			// filter the image, and also calculate the sum of squares..

			if (write_out_amplitude_spectrum == true)
			{
				sum_image_no_dose_filter.AddImage(&image_stack[image_counter]);
			}

			for (pixel_counter = 0; pixel_counter < image_stack[image_counter].real_memory_allocated / 2; pixel_counter++)
			{
				image_stack[image_counter].complex_values[pixel_counter] *= dose_filter[pixel_counter];
				dose_filter_sum_of_squares[pixel_counter] += pow(dose_filter[pixel_counter], 2);
				//if (image_counter == 65) wxPrintf("%f\n", dose_filter[pixel_counter]);
			}

			sum_image.AddImage(&image_stack[image_counter]);
		}
	}
	else // just add them
	{
		for (image_counter = first_frame - 1; image_counter < last_frame; image_counter++)
		{
			sum_image.AddImage(&image_stack[image_counter]);
		}
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

	// do we need to write out the amplitude spectra

	if (write_out_amplitude_spectrum == true)
	{
		Image current_power_spectrum;
		current_power_spectrum.Allocate(sum_image.logical_x_dimension,sum_image.logical_y_dimension,true);

//		if (should_dose_filter == true) sum_image_no_dose_filter.ComputeAmplitudeSpectrumFull2D(&current_power_spectrum);
	//	else sum_image.ComputeAmplitudeSpectrumFull2D(&current_power_spectrum);

		sum_image.ComputeAmplitudeSpectrumFull2D(&current_power_spectrum);

		// Set origin of amplitude spectrum to 0.0
		current_power_spectrum.real_values[current_power_spectrum.ReturnReal1DAddressFromPhysicalCoord(current_power_spectrum.physical_address_of_box_center_x,current_power_spectrum.physical_address_of_box_center_y,current_power_spectrum.physical_address_of_box_center_z)] = 0.0;

		// Forward Transform
		current_power_spectrum.ForwardFFT();

		// make it square

		int micrograph_square_dimension = std::max(sum_image.logical_x_dimension, sum_image.logical_y_dimension);
		if (IsOdd((micrograph_square_dimension))) micrograph_square_dimension++;

		if (sum_image.logical_x_dimension != micrograph_square_dimension || sum_image.logical_y_dimension != micrograph_square_dimension)
		{
			Image current_input_image_square;
			current_input_image_square.Allocate(micrograph_square_dimension,micrograph_square_dimension,false);
			current_power_spectrum.ClipInto(&current_input_image_square, 0);
			current_power_spectrum.Consume(&current_input_image_square);
		}

		// how big will the amplitude spectra have to be in total to have the central 512x512 be a 2.8 angstrom Nyquist?

		// this is the (in the amplitudes real space) scale factor to make the nyquist 2.8 (inverse as real space)

		float pixel_size_scale_factor;
		if (output_pixel_size < 1.4) pixel_size_scale_factor = 1.4 /  output_pixel_size;
		else pixel_size_scale_factor = 1.0;

		// this is the scale factor to make the box 512
		float box_size_scale_factor = 512.0  / float(micrograph_square_dimension);

		// overall scale factor

		float overall_scale_factor = pixel_size_scale_factor * box_size_scale_factor;

		{
			Image scaled_spectrum;
			scaled_spectrum.Allocate(myroundint(micrograph_square_dimension * overall_scale_factor),myroundint(micrograph_square_dimension * overall_scale_factor),false);
			current_power_spectrum.ClipInto(&scaled_spectrum, 0);
			scaled_spectrum.BackwardFFT();
			current_power_spectrum.Allocate(512, 512, 1, true);
			scaled_spectrum.ClipInto(&current_power_spectrum, scaled_spectrum.ReturnAverageOfRealValuesOnEdges());
		}

		// now we need to filter it

		float average;
		float sigma;

		current_power_spectrum.ComputeAverageAndSigmaOfValuesInSpectrum(float(current_power_spectrum.logical_x_dimension)*0.5,float(current_power_spectrum.logical_x_dimension),average,sigma,12);
		current_power_spectrum.DivideByConstant(sigma);
		current_power_spectrum.SetMaximumValueOnCentralCross(average/sigma+10.0);
		current_power_spectrum.ForwardFFT();
		current_power_spectrum.CosineMask(0, 0.05, true);
		current_power_spectrum.BackwardFFT();
		current_power_spectrum.SetMinimumAndMaximumValues(average - 1.0, average + 3.0);
		//current_power_spectrum.CosineRingMask(0.05,0.45, 0.05);
		//average_spectrum->QuickAndDirtyWriteSlice("dbg_average_spectrum_before_conv.mrc",1);
		current_power_spectrum.QuickAndDirtyWriteSlice(amplitude_spectrum_filename, 1);
	}

	//  Shall we write out a scaled image?

	if (write_out_small_sum_image == true)
	{
		// work out a good size..
		int largest_dimension =  std::max(sum_image.logical_x_dimension, sum_image.logical_y_dimension);
		float scale_factor = float(SCALED_IMAGE_SIZE) / float(largest_dimension);

		if (scale_factor < 1.0)
		{
			Image buffer_image;
			buffer_image.Allocate(myroundint(sum_image.logical_x_dimension * scale_factor), myroundint(sum_image.logical_y_dimension * scale_factor), 1, false);
			sum_image.ClipInto(&buffer_image);
			buffer_image.QuickAndDirtyWriteSlice(small_sum_image_filename, 1);
		}
	}

	// now we just need to write out the final sum..

	MRCFile output_file(output_filename, true);
	sum_image.BackwardFFT();
	sum_image.WriteSlice(&output_file, 1); // I made this change as the file is only used once, and this way it is not created until it is actually written, which is cleaner for cancelled / crashed jobs
	output_file.SetPixelSize(output_pixel_size);
	EmpiricalDistribution density_distribution;
	sum_image.UpdateDistributionOfRealValues(&density_distribution);
	output_file.SetDensityStatistics(density_distribution.GetMinimum(), density_distribution.GetMaximum(), density_distribution.GetSampleMean(), sqrtf(density_distribution.GetSampleVariance()));
	output_file.CloseFile();


	// fill the result..


	float *result_array = new float[number_of_input_images * 2];

	for (image_counter = 0; image_counter < number_of_input_images; image_counter++)
	{
		result_array[image_counter] = x_shifts[image_counter] * output_pixel_size;
		result_array[image_counter + number_of_input_images] = y_shifts[image_counter] * output_pixel_size;

		wxPrintf("image #%li = %f, %f\n", image_counter, result_array[image_counter], result_array[image_counter + number_of_input_images]);
	}

	my_result.SetResult(number_of_input_images * 2, result_array);

	delete [] result_array;
	delete [] x_shifts;
	delete [] y_shifts;
	delete [] image_stack;

	if (should_dose_filter == true)
	{
		delete my_electron_dose;
		delete [] dose_filter;
		delete [] dose_filter_sum_of_squares;
	}

	overall_finish = wxDateTime::Now();

	//SendInfo(wxString::Format("Timings for %s: Overall: %s; reading, gain, FT, resampling of stack: %s; initial ali: %s; main ali: %s; unbinned ali: %s\n",input_filename,(overall_finish-overall_start).Format(),(read_frames_finish-read_frames_start).Format(),(first_alignment_finish-first_alignment_start).Format(),(main_alignment_finish-main_alignment_start).Format(),(final_alignment_finish-final_alignment_start).Format()));

	return true;
}

void unblur_refine_alignment(Image *input_stack, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float pixel_size, float *x_shifts, float *y_shifts)
{
	long pixel_counter;
	long image_counter;
	long iteration_counter;

	int number_of_middle_image = number_of_images / 2;

	float *current_x_shifts = new float[number_of_images];
	float *current_y_shifts = new float[number_of_images];

	float middle_image_x_shift;
	float middle_image_y_shift;

	float max_shift;
	float total_shift;

	Image sum_of_images;
	Image sum_of_images_minus_current;

	Peak my_peak;

	Curve x_shifts_curve;
	Curve y_shifts_curve;

	sum_of_images.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);
	sum_of_images.SetToConstant(0.0);

	sum_of_images_minus_current.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);



	// prepare the initial sum

	for (image_counter = 0; image_counter < number_of_images; image_counter++)
	{
		sum_of_images.AddImage(&input_stack[image_counter]);
		current_x_shifts[image_counter] = 0;
		current_y_shifts[image_counter] = 0;
	}

	// perform the main alignment loop until we reach a max shift less than wanted, or max iterations

	for (iteration_counter = 1; iteration_counter <= max_iterations; iteration_counter++)
	{
	//	wxPrintf("Starting iteration number %li\n\n", iteration_counter);
		max_shift = -FLT_MAX;

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

			// compute the cross correlation function and find the peak

		    sum_of_images_minus_current.CalculateCrossCorrelationImageWith(&input_stack[image_counter]);
		    my_peak = sum_of_images_minus_current.FindPeakWithParabolaFit(inner_radius_for_peak_search, outer_radius_for_peak_search);

			// update the shifts..

			current_x_shifts[image_counter] = my_peak.x;
			current_y_shifts[image_counter] = my_peak.y;
		}

		// smooth the shifts

		x_shifts_curve.ClearData();
		y_shifts_curve.ClearData();

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			x_shifts_curve.AddPoint(image_counter, x_shifts[image_counter] + current_x_shifts[image_counter]);
			y_shifts_curve.AddPoint(image_counter, y_shifts[image_counter] + current_y_shifts[image_counter]);

			//wxPrintf("Before = %li : %f, %f\n", image_counter, x_shifts[image_counter] + current_x_shifts[image_counter], y_shifts[image_counter] + current_y_shifts[image_counter]);
		}


		x_shifts_curve.FitSavitzkyGolayToData(5, 3);
		y_shifts_curve.FitSavitzkyGolayToData(5, 3);

		// copy them back..

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			current_x_shifts[image_counter] = x_shifts_curve.savitzky_golay_fit[image_counter] - x_shifts[image_counter];
			current_y_shifts[image_counter] = y_shifts_curve.savitzky_golay_fit[image_counter] - y_shifts[image_counter];
		//	wxPrintf("After = %li : %f, %f\n", image_counter, x_shifts_curve.savitzky_golay_fit[image_counter], y_shifts_curve.savitzky_golay_fit[image_counter]);
		}



		// subtract shift of the middle image from all images to keep things centred around it

		middle_image_x_shift = current_x_shifts[number_of_middle_image];
		middle_image_y_shift = current_y_shifts[number_of_middle_image];

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			current_x_shifts[image_counter] -= middle_image_x_shift;
			current_y_shifts[image_counter] -= middle_image_y_shift;

			total_shift = sqrt(pow(current_x_shifts[image_counter], 2) + pow(current_y_shifts[image_counter], 2));
			if (total_shift > max_shift) max_shift = total_shift;

		}

		// actually shift the images, also add the subtracted shifts to the overall shifts

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			input_stack[image_counter].PhaseShift(current_x_shifts[image_counter], current_y_shifts[image_counter], 0.0);

			x_shifts[image_counter] += current_x_shifts[image_counter];
			y_shifts[image_counter] += current_y_shifts[image_counter];
		}

		// check to see if the convergence criteria have been reached and return if so

		if (iteration_counter >= max_iterations || max_shift <= max_shift_convergence_threshold)
		{
		//	wxPrintf("returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);
			delete [] current_x_shifts;
			delete [] current_y_shifts;
			return;
		}
		else
		{
		//	wxPrintf("Not. returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);

		}

		// going to be doing another round so we need to make the new sum..

		sum_of_images.SetToConstant(0.0);

		for (image_counter = 0; image_counter < number_of_images; image_counter++)
		{
			sum_of_images.AddImage(&input_stack[image_counter]);
		}

	}
}





