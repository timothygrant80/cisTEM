#include "../../core/core_headers.h"

class
CtffindApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

class ImageCTFComparison
{
public:
	Image 	img;		// Usually an amplitude spectrum
	CTF		ctf;
	float	pixel_size;
	bool	find_phase_shift;
};

// This is the function which will be minimised
float CtffindObjectiveFunction(void *scoring_parameters, float array_of_values[] )
{
	ImageCTFComparison *comparison_object = reinterpret_cast < ImageCTFComparison *> (scoring_parameters);

	CTF my_ctf = comparison_object->ctf;
	my_ctf.SetDefocus(array_of_values[0],array_of_values[1],array_of_values[2]);
	if (comparison_object->find_phase_shift)
	{
		my_ctf.SetAdditionalPhaseShift(array_of_values[3]);
	}

	//MyDebugPrint("(CtffindObjectiveFunction) D1 = %6.2f D2 = %6.2f, Ast = %5.2f, Score = %g",array_of_values[0],array_of_values[1],array_of_values[2],- comparison_object->img.GetCorrelationWithCTF(my_ctf));

	// Evaluate the function
	return - comparison_object->img.GetCorrelationWithCTF(my_ctf);
}

float FindRotationalAlignmentBetweenTwoStacksOfImages(Image *self, Image *other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius);

IMPLEMENT_APP(CtffindApp)

// override the DoInteractiveUserInput

void CtffindApp::DoInteractiveUserInput()
{

	float lowest_allowest_minimum_resolution = 50.0;

	std::string input_filename;
	bool input_is_a_movie;
	int number_of_frames_to_average;
	std::string output_diagnostic_filename;
	float pixel_size;
	float acceleration_voltage;
	float spherical_aberration;
	float amplitude_contrast;
	int box_size;
	float minimum_resolution;
	float maximum_resolution;
	float minimum_defocus;
	float maximum_defocus;
	float defocus_search_step;
	float astigmatism_tolerance;
	bool find_additional_phase_shift;
	float minimum_additional_phase_shift;
	float maximum_additional_phase_shift;
	float additional_phase_shift_search_step;

	UserInput *my_input = new UserInput("Ctffind", 0.0);

	input_filename  			= my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true );

	MRCFile input_file(input_filename,false);
	if (input_file.ReturnZSize() > 1)
	{
		input_is_a_movie 		= my_input->GetYesNoFromUser("Input is a movie (stack of frames)","Answer yes if the input file is a stack of frames from a dose-fractionated movie. If not, each image will be processed separately","no");
	}
	else
	{
		input_is_a_movie = false;
	}

	if (input_is_a_movie)
	{
		number_of_frames_to_average = my_input->GetIntFromUser("Number of frames to average together","If the number of electrons per frame is too low, there may be strong artefacts in the estimated power spectrum. This can be alleviated by averaging frames with each other in real space before computing their Fourier transforms","1");
	}
	else
	{
		number_of_frames_to_average = 1;
	}

	output_diagnostic_filename	= my_input->GetFilenameFromUser("Output diagnostic image file name","Will contain the experimental power spectrum and the best CTF fit","diagnostic_output.mrc",false);
	pixel_size 					= my_input->GetFloatFromUser("Pixel size","In Angstroms","1.0",0.0,9999999.99);
	acceleration_voltage 		= my_input->GetFloatFromUser("Acceleration voltage","in kV","300.0",0.0,99999999.99);
	spherical_aberration 		= my_input->GetFloatFromUser("Spherical aberration","in mm","2.7",0.0,999999999.99);
	amplitude_contrast 			= my_input->GetFloatFromUser("Amplitude contrast","Fraction of amplitude contrast","0.07",0.0,1.0);
	box_size 					= my_input->GetIntFromUser("Size of amplitude spectrum to compute","in pixels","512",128,9999999);
	minimum_resolution 			= my_input->GetFloatFromUser("Minimum resolution","Lowest resolution used for fitting CTF (Angstroms)","30.0",0.0,lowest_allowest_minimum_resolution);
	maximum_resolution 			= my_input->GetFloatFromUser("Maximum resolution","Highest resolution used for fitting CTF (Angstroms)","5.0",0.0,minimum_resolution);
	minimum_defocus 			= my_input->GetFloatFromUser("Minimum defocus","Positive values for underfocus. Lowest value to search over (Angstroms)","5000.0",-999999.99,999999999.99);
	maximum_defocus 			= my_input->GetFloatFromUser("Maximum defocus","Positive values for underfocus. Highest value to search over (Angstroms)","50000.0",minimum_defocus,9999999999.99);
	defocus_search_step 		= my_input->GetFloatFromUser("Defocus search step","Step size for defocus search (Angstroms)","500.0",1.0,99999999999.99);
	astigmatism_tolerance 		= my_input->GetFloatFromUser("Expected (tolerated) astigmatism","Astigmatism values much larger than this will be penalised (Angstroms; set to negative to remove this restraint","100.0");
	find_additional_phase_shift = my_input->GetYesNoFromUser("Find additional phase shift?","Input micrograph was recorded using a phase plate with variable phase shift, which you want to find","no");

	if (find_additional_phase_shift)
	{
		minimum_additional_phase_shift 		= my_input->GetFloatFromUser("Minimum phase shift","Lower bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians","0.0",-3.15,3.15);
		maximum_additional_phase_shift 		= my_input->GetFloatFromUser("Maximum phase shift","Upper bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians","0.0",minimum_additional_phase_shift,3.15);
		additional_phase_shift_search_step 	= my_input->GetFloatFromUser("Phase shift search step","Step size for phase shift search (radians)","0.2",0.001,maximum_additional_phase_shift-minimum_additional_phase_shift);
	}
	else
	{
		minimum_additional_phase_shift = 0.0;
		maximum_additional_phase_shift = 0.0;
		additional_phase_shift_search_step = 0.0;
	}


	delete my_input;

	my_current_job.Reset(19);
	my_current_job.ManualSetArguments("tbitffffiffffffbfff",	input_filename.c_str(),
													 	 	 	input_is_a_movie,
																number_of_frames_to_average,
																output_diagnostic_filename.c_str(),
																pixel_size,
																acceleration_voltage,
																spherical_aberration,
																amplitude_contrast,
																box_size,
																minimum_resolution,
																maximum_resolution,
																minimum_defocus,
																maximum_defocus,
																defocus_search_step,
																astigmatism_tolerance,
																find_additional_phase_shift,
																minimum_additional_phase_shift,
																maximum_additional_phase_shift,
																additional_phase_shift_search_step);


}



// override the do calculation method which will be what is actually run..

bool CtffindApp::DoCalculation()
{

	// Arguments for this job

	std::string input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	bool        input_is_a_movie 					= my_current_job.arguments[1].ReturnBoolArgument();
	int         number_of_frames_to_average			= my_current_job.arguments[2].ReturnIntegerArgument();
	std::string output_diagnostic_filename			= my_current_job.arguments[3].ReturnStringArgument();
	float 		pixel_size							= my_current_job.arguments[4].ReturnFloatArgument();
	float 		acceleration_voltage				= my_current_job.arguments[5].ReturnFloatArgument();
	float       spherical_aberration				= my_current_job.arguments[6].ReturnFloatArgument();
	float 		amplitude_contrast					= my_current_job.arguments[7].ReturnFloatArgument();
	int         box_size							= my_current_job.arguments[8].ReturnIntegerArgument();
	float 		minimum_resolution					= my_current_job.arguments[9].ReturnFloatArgument();
	float       maximum_resolution					= my_current_job.arguments[10].ReturnFloatArgument();
	float       minimum_defocus						= my_current_job.arguments[11].ReturnFloatArgument();
	float       maximum_defocus						= my_current_job.arguments[12].ReturnFloatArgument();
	float       defocus_search_step					= my_current_job.arguments[13].ReturnFloatArgument();
	float       astigmatism_tolerance               = my_current_job.arguments[14].ReturnFloatArgument();
	bool       	find_additional_phase_shift         = my_current_job.arguments[15].ReturnBoolArgument();
	float  		minimum_additional_phase_shift		= my_current_job.arguments[16].ReturnFloatArgument();
	float		maximum_additional_phase_shift		= my_current_job.arguments[17].ReturnFloatArgument();
	float		additional_phase_shift_search_step	= my_current_job.arguments[18].ReturnFloatArgument();

	// These variables will be set by command-line options
	const bool		old_school_input = false;
	const bool		amplitude_spectrum_input = false;
	const bool		filtered_amplitude_spectrum_input = false;

	/*
	 *  Scoring function
	 */
	float MyFunction(float []);

	// Other variables
	int					number_of_movie_frames;
	int         		number_of_micrographs;
	MRCFile				input_file(input_filename,false);
	Image				average_spectrum;
	wxString			output_text_fn;
	ProgressBar			*my_progress_bar;
	NumericTextFile		*output_text;
	int					current_micrograph_number;
	int					number_of_tiles_used;
	Image 				current_power_spectrum;
	int					current_first_frame_within_average;
	int					current_frame_within_average;
	int					current_input_location;
	Image				current_input_image;
	Image				current_input_image_square;
	int					micrograph_square_dimension;
	Image				temp_image;
	Image				resampled_power_spectrum;
	bool				resampling_is_necessary;
	CTF					current_ctf;
	float				average, sigma;
	int					convolution_box_size;
	ImageCTFComparison	comparison_object;
	float 				estimated_astigmatism_angle;
	float				bf_halfrange[4];
	float				bf_midpoint[4];
	float				bf_stepsize[4];
	float				cg_starting_point[4];
	float				cg_accuracy[4];
	int 				number_of_search_dimensions;
	BruteForceSearch   	*brute_force_search;
	int					counter;
	ConjugateGradient   *conjugate_gradient_minimizer;
	int 				current_output_location;

	// Some argument checking
	if (minimum_resolution < maximum_resolution)
	{
		SendError(wxString::Format("Error: Minimum resolution (%f) higher than maximum resolution (%f). Terminating.", minimum_resolution,maximum_resolution));
		ExitMainLoop();
	}
	if (minimum_defocus > maximum_defocus)
	{
		SendError(wxString::Format("Minimum defocus must be less than maximum defocus. Terminating."));
		ExitMainLoop();
	}

	// How many micrographs are we dealing with
	if (input_is_a_movie)
	{
		// We only support 1 movie per file
		number_of_movie_frames = input_file.ReturnZSize();
		number_of_micrographs = 1;
	}
	else
	{
		number_of_movie_frames = 1;
		number_of_micrographs = input_file.ReturnZSize();
	}

	if (is_running_locally)
	{
		// Print out information about input file
		// (Not implemented yet)
		MyPrintWithDetails("TODO: print information about input file\n");

		// Prepare the output text file
		output_text_fn = FilenameReplaceExtension(output_diagnostic_filename,"txt");
		output_text = new NumericTextFile(output_text_fn,OPEN_TO_WRITE,7);

		// Print header to the output text file
		output_text->WriteCommentLine("# Output from CTFFind version %s run on %s\n","0.0.0",wxDateTime::Now().FormatISOCombined().ToStdString());
		output_text->WriteCommentLine("# Input file: %s ; Number of micrographs: %i\n",input_filename,number_of_micrographs);
		output_text->WriteCommentLine("# Pixel size: %f0.3 Angstroms ; acceleration voltage: %f0.1 keV ; spherical aberration: %0.1 mm ; amplitude contrast: %f0.2\n",pixel_size,acceleration_voltage,spherical_aberration,amplitude_contrast);
		output_text->WriteCommentLine("# Box size: %i pixels ; min. res.: %f0.1 Angstroms ; max. res.: %f0.1 Angstroms ; min. def.: %f0.1 um; max. def. %f0.1 um\n",box_size,minimum_resolution,maximum_resolution,minimum_defocus,maximum_defocus);
		output_text->WriteCommentLine("# Columns: #1 - micrograph number; #2 - defocus 1 [Angstroms]; #3 - defocus 2; #4 - azimuth of astigmatism; #5 - additional phase shift [radians]; #6 - cross correlation; #7 - spacing (in Angstroms) up to which CTF rings were fit successfully\n");

		// Prepare a text file with 1D rotational average spectra
		output_text_fn = FilenameAddSuffix(output_text_fn.ToStdString(),"_avrot");

		if (! old_school_input && number_of_micrographs > 1)
		{
			wxPrintf("Will estimate the CTF parmaeters for %i micrographs.\n",number_of_micrographs);
			wxPrintf("Results will be written to this file: %s\n",output_text->ReturnFilename());
			my_progress_bar = new ProgressBar(number_of_micrographs);
		}
	}


	// Prepare the average spectrum image
	average_spectrum.Allocate(box_size,box_size,true);

	// Loop over micrographs
	for (current_micrograph_number=1; current_micrograph_number <= number_of_micrographs; current_micrograph_number++)
	{
		if (old_school_input || number_of_micrographs == 1) wxPrintf("Working on micrograph %i of %i\n", current_micrograph_number, number_of_micrographs);

		number_of_tiles_used = 0;
		average_spectrum.SetToConstant(0.0);
		average_spectrum.is_in_real_space = true;

		if (amplitude_spectrum_input || filtered_amplitude_spectrum_input)
		{
			current_power_spectrum.ReadSlice(&input_file,current_micrograph_number);
			current_power_spectrum.ForwardFFT();
			average_spectrum.Allocate(box_size,box_size,1,false);
			current_power_spectrum.ClipInto(&average_spectrum);
			average_spectrum.BackwardFFT();
		}
		else
		{
			for (current_first_frame_within_average = 1; current_first_frame_within_average <= number_of_movie_frames; current_first_frame_within_average += number_of_frames_to_average)
			{
				for (current_frame_within_average = 1; current_frame_within_average <= number_of_movie_frames; current_frame_within_average++)
				{
					current_input_location = current_first_frame_within_average + number_of_movie_frames * (current_micrograph_number-1) + (current_frame_within_average-1);
					if (current_input_location > number_of_movie_frames * current_micrograph_number) continue;
					current_input_image.ReadSlice(&input_file,current_input_location);
					if (current_input_image.IsConstant())
					{
						SendError(wxString::Format("Error: location %i of input file %s is blank",current_input_location, input_filename));
						ExitMainLoop();
					}
					// Make the image square
					micrograph_square_dimension = std::max(current_input_image.logical_x_dimension,current_input_image.logical_y_dimension);
					if (IsOdd((micrograph_square_dimension))) micrograph_square_dimension++;
					if (current_input_image.logical_x_dimension != micrograph_square_dimension || current_input_image.logical_y_dimension != micrograph_square_dimension)
					{
						current_input_image_square.Allocate(micrograph_square_dimension,micrograph_square_dimension,true);
						current_input_image.ClipInto(&current_input_image_square,current_input_image.ReturnAverageOfRealValues());
						current_input_image.Consume(&current_input_image_square);
					}
					//
					if (current_frame_within_average == 1)
					{
						temp_image.Allocate(current_input_image.logical_x_dimension,current_input_image.logical_y_dimension,true);
						temp_image.SetToConstant(0.0);
					}
					temp_image.AddImage(&current_input_image);
				} // end of loop over frames to average together
				current_input_image.Consume(&temp_image);

				// Taper the edges of the micrograph in real space, to lessen Gibbs artefacts
				current_input_image.TaperEdges();

				number_of_tiles_used++;

				// Compute the amplitude spectrum
				current_power_spectrum.Allocate(current_input_image.logical_x_dimension,current_input_image.logical_y_dimension,true);
				current_input_image.ForwardFFT(false);
				current_input_image.ComputeAmplitudeSpectrumFull2D(&current_power_spectrum);

				current_power_spectrum.QuickAndDirtyWriteSlice("dbg_spec_before_resampling.mrc",1);

				// Set origin of amplitude spectrum to 0.0
				current_power_spectrum.real_values[current_power_spectrum.ReturnReal1DAddressFromPhysicalCoord(current_power_spectrum.physical_address_of_box_center_x,current_power_spectrum.physical_address_of_box_center_y,current_power_spectrum.physical_address_of_box_center_z)] = 0.0;

				// Resample the amplitude spectrum
				resampling_is_necessary = current_power_spectrum.logical_x_dimension != box_size || current_power_spectrum.logical_y_dimension != box_size;
				if (resampling_is_necessary)
				{
					current_power_spectrum.ForwardFFT(false);
					resampled_power_spectrum.Allocate(box_size,box_size,1,false);
					current_power_spectrum.ClipInto(&resampled_power_spectrum);
					resampled_power_spectrum.BackwardFFT();
				}
				else
				{
					resampled_power_spectrum = current_power_spectrum;
				}

				average_spectrum.AddImage(&resampled_power_spectrum);
			} // end of loop over movie frames

			// We need to take care of the scaling of the FFTs, as well as the averaging of tiles
			if (resampling_is_necessary)
			{
				average_spectrum.MultiplyByConstant(1.0 / ( float(number_of_tiles_used) * current_input_image.logical_x_dimension * current_input_image.logical_y_dimension * current_power_spectrum.logical_x_dimension * current_power_spectrum.logical_y_dimension ) );
			}
			else
			{
				average_spectrum.MultiplyByConstant(1.0 / ( float(number_of_tiles_used) * current_input_image.logical_x_dimension * current_input_image.logical_y_dimension ) );
			}

		} // end of test of whether we were given amplitude spectra on input


		average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_before_bg_sub.mrc",1);


		// Filter the amplitude spectrum, remove background
		if (! filtered_amplitude_spectrum_input)
		{
			// Try to weaken cross artefacts
			average_spectrum.ComputeAverageAndSigmaOfValuesInSpectrum(float(average_spectrum.logical_x_dimension)*pixel_size/minimum_resolution,float(average_spectrum.logical_x_dimension),average,sigma,12);
			average_spectrum.SetMaximumValueOnCentralCross(average+10.0*sigma);

			average_spectrum.QuickAndDirtyWriteSlice("dbg_average_spectrum_before_conv.mrc",1);

			// Compute low-pass filtered version of the spectrum
			convolution_box_size = int( float(average_spectrum.logical_x_dimension) * pixel_size / minimum_resolution * sqrt(2.0) );
			if (IsEven(convolution_box_size)) convolution_box_size++;
			current_power_spectrum.Allocate(average_spectrum.logical_x_dimension,average_spectrum.logical_y_dimension,true);
			average_spectrum.SpectrumBoxConvolution(&current_power_spectrum,convolution_box_size,float(average_spectrum.logical_x_dimension)*pixel_size/minimum_resolution);

			current_power_spectrum.QuickAndDirtyWriteSlice("dbg_spec_convoluted.mrc",1);

			// POTENTIAL OPTIMIZATION: do not store the convoluted spectrum as a separate image - just subtract one convoluted pixel at a time from the image

			// Subtract low-pass-filtered spectrum from the spectrum. This should remove the background slope.
			average_spectrum.SubtractImage(&current_power_spectrum);

			average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_before_thresh.mrc",1);

			// Threshold high values
			average_spectrum.SetMaximumValue(average_spectrum.ReturnMaximumValue(3,3));
		}

		// We now have a spectrum which we can use to fit CTFs
		average_spectrum.QuickAndDirtyWriteSlice("dbg_spec.mrc",1);


		// Set up the CTF object
		current_ctf.Init(acceleration_voltage,spherical_aberration,amplitude_contrast,minimum_defocus,minimum_defocus,0.0,1.0/minimum_resolution,1.0/maximum_resolution,astigmatism_tolerance,pixel_size,minimum_additional_phase_shift);
		current_ctf.SetDefocus(minimum_defocus/pixel_size,minimum_defocus/pixel_size,0.0);
		current_ctf.SetAdditionalPhaseShift(minimum_additional_phase_shift);


		// Set up the comparison object
		comparison_object.ctf = current_ctf;
		comparison_object.img = average_spectrum;
		comparison_object.pixel_size = pixel_size;
		comparison_object.find_phase_shift = find_additional_phase_shift;

		if (old_school_input)
		{
			wxPrintf("\nSEARCHING CTF PARAMETERS...\n");
		}


		// Let's look for the astigmatism angle first
		temp_image = average_spectrum;
		temp_image.ApplyMirrorAlongY();
		temp_image.QuickAndDirtyWriteSlice("dbg_spec_y.mrc",1);
		estimated_astigmatism_angle = 0.5 * FindRotationalAlignmentBetweenTwoStacksOfImages(&average_spectrum,&temp_image,1,90.0,5.0,pixel_size/minimum_resolution,pixel_size/maximum_resolution);

		MyDebugPrint ("Estimated astigmatism angle = %f\n", estimated_astigmatism_angle);


		// We can now look for the defocus value
		bf_halfrange[0] = 0.5 * (maximum_defocus-minimum_defocus)/pixel_size;
		bf_halfrange[1] = bf_halfrange[0];
		bf_halfrange[2] = 0.0;
		bf_halfrange[3] = 0.5 * (maximum_additional_phase_shift-minimum_additional_phase_shift);

		bf_midpoint[0] = minimum_defocus/pixel_size + bf_halfrange[0];
		bf_midpoint[1] = bf_midpoint[0];
		bf_midpoint[2] = estimated_astigmatism_angle / 180.0 * PI;
		bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

		bf_stepsize[0] = defocus_search_step/pixel_size;
		bf_stepsize[1] = bf_stepsize[0];
		bf_stepsize[2] = 0.0;
		bf_stepsize[3] = additional_phase_shift_search_step;

		if (find_additional_phase_shift)
		{
			number_of_search_dimensions = 4;
		}
		else
		{
			number_of_search_dimensions = 3;
		}

		// Actually run the BF search
		brute_force_search = new BruteForceSearch();
		brute_force_search->Init(&CtffindObjectiveFunction,&comparison_object,number_of_search_dimensions,bf_midpoint,bf_halfrange,bf_stepsize,false,false);
		brute_force_search->Run();

		// The end point of the BF search is the beginning of the CG search
		for (counter=0;counter<number_of_search_dimensions;counter++)
		{
			cg_starting_point[counter] = brute_force_search->GetBestValue(counter);
		}

		//
		current_ctf.SetDefocus(cg_starting_point[0]*pixel_size,cg_starting_point[1]*pixel_size,cg_starting_point[2]);
		if (find_additional_phase_shift)
		{
			current_ctf.SetAdditionalPhaseShift(cg_starting_point[3]);
		}
		current_ctf.EnforceConvention();

		// Print out the results of brute force search
		if (old_school_input || DEBUG)
		{
			wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
			wxPrintf("%12.2f%12.2f%12.2f%12.5f",current_ctf.GetDefocus1()*pixel_size,current_ctf.GetDefocus2()*pixel_size,current_ctf.GetAstigmatismAzimuth()*180.0/PI,-brute_force_search->GetBestScore());
			if (DEBUG)
			{
				MyDebugPrint("Found the following phase shift: %g\n", current_ctf.GetAdditionalPhaseShift());
			}
		}

		// Now we refine in the neighbourhood by using Powell's conjugate gradient algorithm
		if (old_school_input || DEBUG)
		{
			wxPrintf("\nREFINING CTF PARAMETERS...\n");
			wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
		}
		cg_accuracy[0] = 100.0;
		cg_accuracy[1] = 100.0;
		cg_accuracy[2] = 0.5;
		cg_accuracy[3] = 0.05;
		conjugate_gradient_minimizer = new ConjugateGradient();
		conjugate_gradient_minimizer->Init(&CtffindObjectiveFunction,&comparison_object,number_of_search_dimensions,cg_starting_point,cg_accuracy);
		conjugate_gradient_minimizer->Run();

		// Remember the results of the refinement
		for (counter=0;counter<number_of_search_dimensions;counter++)
		{
			cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
		}
		current_ctf.SetDefocus(cg_starting_point[0]/pixel_size,cg_starting_point[1]/pixel_size,cg_starting_point[2]/180.0*PI);
		if (find_additional_phase_shift)
		{
			current_ctf.SetAdditionalPhaseShift(cg_starting_point[3]);
		}
		current_ctf.EnforceConvention();

		// Print results to the terminal
		if (old_school_input || DEBUG)
		{
			wxPrintf("%12.2f%12.2f%12.2f%12.5f   Final Values",current_ctf.GetDefocus1()*pixel_size,current_ctf.GetDefocus2()*pixel_size,current_ctf.GetAstigmatismAzimuth()*180.0/PI,-conjugate_gradient_minimizer->GetBestScore());
			if (DEBUG)
			{
				MyDebugPrint("Found the following phase shift: %g\n", current_ctf.GetAdditionalPhaseShift());
			}
		}

		// Generate diagnostic image
		current_output_location = current_micrograph_number;
		average_spectrum.AddConstant(-1.0 * average_spectrum.ReturnAverageOfRealValuesOnEdges());
		average_spectrum.ComputeAverageAndSigmaOfValuesInSpectrum(	sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(2,0.0))*average_spectrum.logical_x_dimension,
															      	std::max(current_ctf.GetHighestFrequencyForFitting(),sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(3,0.0)))*average_spectrum.logical_x_dimension,
																	average,sigma);


	} // End of loop over micrographs




/*

        ! Generate diagnostic image
        current_output_location = current_micrograph_number
        call average_spectrum%AddConstant((-1.0)*average_spectrum%GetAverageOfValuesOnEdges())
        call average_spectrum%ComputeAverageAndSigmaOfValuesInSpectrum(                                         &
                                                            current_ctf%ComputeFrequencyOfAZero(0.0,2)          &
                                                            *average_spectrum%GetLogicalDimension(1),           &
                                                            max(current_ctf%GetHighestFrequencyForFitting(),    &
                                                            current_ctf%ComputeFrequencyOfAZero(0.0,3))         &
                                                            *average_spectrum%GetLogicalDimension(1),           &
                                                            average,sigma)
        call average_spectrum%ApplyCircularMask(5.0,inverse=.true.)
        call average_spectrum%SetMaximumValueOnCentralCross(average)
        call average_spectrum%SetMinimumAndMaximumValue(average-4.0*sigma,average+4.0*sigma)
        call average_spectrum%ComputeAverageAndSigmaOfValuesInSpectrum(                                         &
                                                            current_ctf%ComputeFrequencyOfAZero(0.0,2)          &
                                                            *average_spectrum%GetLogicalDimension(1),           &
                                                            max(current_ctf%GetHighestFrequencyForFitting(),    &
                                                            current_ctf%ComputeFrequencyOfAZero(0.0,3))         &
                                                            *average_spectrum%GetLogicalDimension(1),           &
                                                            average,sigma)
        call average_spectrum%AddConstant(-1.0*average)
        call average_spectrum%MultiplyByConstant(1.0e0/sigma)
        call average_spectrum%AddConstant(average)
        call average_spectrum%Compute1DRotationalAverage(rotational_average)
        if (debug) then
            call average_spectrum%WriteToDisk('dbg_average_spectrum_before_rescaling.mrc')
        endif
        if (compute_extra_stats .or. boost_ring_contrast) then
            call average_spectrum%ComputeRotationalAverageOfPowerSpectrum( &
                                                                          current_ctf,spatial_frequency,            &
                                                                          rotational_average_astig,                 &
                                                                          rotational_average_astig_fit,             &
                                                                          frc_of_fit,frc_of_fit_sigma,              &
                                                                          rescale_input=boost_ring_contrast,        &
                                                                          squared_ctf_was_fit=fit_squared_ctf)
        endif

        call average_spectrum%ComputeAverageAndSigmaOfValuesInSpectrum(                                         &
                                                            current_ctf%ComputeFrequencyOfAZero(0.0,2)          &
                                                            *average_spectrum%GetLogicalDimension(1),           &
                                                            max(current_ctf%GetHighestFrequencyForFitting(),    &
                                                            current_ctf%ComputeFrequencyOfAZero(0.0,3))         &
                                                            *average_spectrum%GetLogicalDimension(1),           &
                                                            average,sigma)

        call average_spectrum%SetMinimumAndMaximumValue(average-1.0*sigma,average+2.0*sigma)
        if (debug) call average_spectrum%WriteToDisk('dbg_average_spectrum_before_overlay.mrc')
        call average_spectrum%OverlayCTF(current_ctf,squared_ctf=fit_squared_ctf)
        call average_spectrum%WriteToDisk(output_diagnostic_filename%value,current_output_location)

        ! Until what frequency were CTF rings detected?
        if (compute_extra_stats) then
            do last_bin_with_good_fit=2,size(frc_of_fit)
                if ( (count(frc_of_fit(1:last_bin_with_good_fit-1) .gt. 0.20d0) .gt. 3 .and. &
                           (frc_of_fit(last_bin_with_good_fit) .le. 0.2d0)) .or. &
                     (count(frc_of_fit(1:last_bin_with_good_fit-1) .gt. frc_significance_threshold) .gt. 3 .and. &
                                    ((frc_of_fit(last_bin_with_good_fit) .lt. frc_significance_threshold &
                                    .and. rotational_average_astig(last_bin_with_good_fit) .gt. 2.0d0) &
                                .or. (frc_of_fit(last_bin_with_good_fit) .lt. frc_significance_threshold &
                                    .and. rotational_average_astig(last_bin_with_good_fit) .lt. -2.0d0))&
                                                                                    )) then
                    ! last_bin_with_good_fit will now be set to point to the last frequencies at which Thon rings were still well fit
                    exit
                endif
            enddo
            last_bin_with_good_fit = min(last_bin_with_good_fit,size(frc_of_fit))
        else
            last_bin_with_good_fit = 1
        endif

        ! Print more detailled results to terminal
        if (number_of_micrographs .eq. 1) then
            write(*,'(a,f0.2,1x,a,1x,f0.2,a)')  'Estimated defocus values        : ', &
                                            current_ctf%GetDefocus1InAngstroms(pixel_size%value), ',', &
                                            current_ctf%GetDefocus2InAngstroms(pixel_size%value), ' Angstroms'
            write(*,'(a,f0.2,a)')               'Estimated azimuth of astigmatism: ', &
                                            current_ctf%GetAstigmatismAzimuthInDegrees(), ' degrees'
            if (find_additional_phase_shift%value) then
                write(*,'(a,f0.3,a,f0.2,a)')    'Additional phase shift          : ', &
                                                    current_ctf%GetAdditionalPhaseShift(), &
                                                    ' (', current_ctf%GetAdditionalPhaseShift()/3.1415,' pi)'
            endif
            write(*,'(a,f0.5)')                 'Score                           : ', &
                                            -conjugate_gradient%GetBestScore()
            if (compute_extra_stats) then
                write(*,'(a,f0.1,a)')               'Thon rings with good fit up to  : ', &
                                                pixel_size%value/spatial_frequency(last_bin_with_good_fit), ' Angstroms'
            endif
        endif


        ! Write out results to summary file
        values_to_write_out(1) = real(current_micrograph_number)
        values_to_write_out(2:4) = current_ctf%GetDefocusParametersInAngstromsAndDegrees(pixel_size%value)
        values_to_write_out(5) = current_ctf%GetAdditionalPhaseShift()
        values_to_write_out(6) = -conjugate_gradient%GetBestScore()
        if (compute_extra_stats) then
            values_to_write_out(7) = pixel_size%value/spatial_frequency(last_bin_with_good_fit)
        else
            values_to_write_out(7) = 0.0
        endif
        call output_text%WriteDataLine(values_to_write_out)



        ! Write avrot
        ! \todo Add to the output a line with non-normalized avrot, so that users can check for things like ice crystal reflections - see
        !!
        if (compute_extra_stats) then
            if (output_text_avrot%number_of_data_lines .eq. 0) then
                call output_text_avrot%Init(output_text_fn,OPEN_TO_WRITE,size(rotational_average))
                call output_text_avrot%WriteCommentLine(comment_lines(1))
                call output_text_avrot%WriteCommentLine(comment_lines(2))
                call output_text_avrot%WriteCommentLine(comment_lines(3))
                call output_text_avrot%WriteCommentLine(comment_lines(4))
                call output_text_avrot%WriteCommentLine('6 lines per micrograph: #1 - spatial frequency (1/pixels); '//&
                                                 '#2 - 1D rotational average of spectrum (assuming no astigmatism); '//&
                                                 '#3 - 1D rotational average of spectrum; #4 - CTF fit; '//&
                                                 '#5 - cross-correlation between spectrum and CTF fit; '//&
                                                 '#6 - 2sigma of expected cross correlation of noise')
            endif
            call output_text_avrot%WriteDataLine(real(spatial_frequency))
            call output_text_avrot%WriteDataLine(real(rotational_average))
            call output_text_avrot%WriteDataLine(real(rotational_average_astig))
            call output_text_avrot%WriteDataLine(real(rotational_average_astig_fit))
            call output_text_avrot%WriteDataLine(real(frc_of_fit))
            call output_text_avrot%WriteDataLine(real(frc_of_fit_sigma))
        endif

        ! Mark progress
        if (.not. old_school_input .and. number_of_micrographs .gt. 1) then
            call my_progress_bar%Update(current_micrograph_number)
        endif

    enddo ! end of loop over micrographs

    if (.not. old_school_input .and. number_of_micrographs .gt. 1) then
        call my_progress_bar%Finish()
    endif

    ! Tell the user where the outputs are
    write(*,'(/2a)')'Summary of results                          : ', trim(adjustl(output_text%filename))
    write(*,'(2a)') 'Diagnostic images                           : ', trim(adjustl(output_diagnostic_filename%value))
    if (compute_extra_stats) then
        write(*,'(2a)') 'Detailled results, including 1D fit profiles: ', trim(adjustl(output_text_avrot%filename))
        write(*,'(2a)') 'Use this command to plot 1D fit profiles    : ctffind_plot_results.sh ', &
                                                                      trim(adjustl(output_text_avrot%filename))
    endif



 */


	return true;
}

// Align rotationally a (stack) of image(s) against another image. Return the rotation angle that gives the best normalised cross-correlation.
float FindRotationalAlignmentBetweenTwoStacksOfImages(Image *self, Image *other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius)
{
	MyDebugAssertTrue(self[0].is_in_memory, "Memory not allocated");
	MyDebugAssertTrue(self[0].is_in_real_space, "Not in real space");
	MyDebugAssertTrue(self[0].logical_z_dimension == 1, "Meant for images, not volumes");
	MyDebugAssertTrue(other_image[0].is_in_memory, "Memory not allocated - other_image");
	MyDebugAssertTrue(other_image[0].is_in_real_space, "Not in real space - other_image");
	MyDebugAssertTrue(other_image[0].logical_z_dimension == 1, "Meant for images, not volumes - other_image");
	MyDebugAssertTrue(self[0].HasSameDimensionsAs(&other_image[0]),"Images and reference images do not have same dimensions.");

	// Local variables
	const float minimum_radius_sq = pow(minimum_radius,2);
	const float maximum_radius_sq = pow(maximum_radius,2);
	const float inverse_logical_x_dimension = 1.0 / float(self[0].logical_x_dimension);
	const float inverse_logical_y_dimension = 1.0 / float(self[0].logical_y_dimension);
	float best_cc = - std::numeric_limits<float>::max();
	float best_rotation = - std::numeric_limits<float>::max();
	float current_rotation = - search_half_range;
	float current_rotation_rad;
	EmpiricalDistribution cc_numerator_dist(false);
	EmpiricalDistribution cc_denom_self_dist(false);
	EmpiricalDistribution cc_denom_other_dist(false);
	int current_image;
	int i, i_logi;
	float i_logi_frac, ii_phys;
	int j, j_logi;
	float j_logi_frac, jj_phys;
	float current_interpolated_value;
	long address_in_other_image;
	float current_cc;



	// Loop over possible rotations
	while ( current_rotation < search_half_range + search_step_size )
	{

		current_rotation_rad = current_rotation / 180.0 * PI;
		cc_numerator_dist.Reset();
		cc_denom_self_dist.Reset();
		cc_denom_other_dist.Reset();
		// Loop over the array of images
		for (current_image=0; current_image < number_of_images; current_image++)
		{
			// Loop over the other (reference) image
			address_in_other_image = 0;
			for (j=0; j < other_image[0].logical_y_dimension; j++)
			{
				j_logi = j - other_image[0].physical_address_of_box_center_y;
				j_logi_frac = pow(j_logi * inverse_logical_y_dimension,2);
				for (i=0; i < other_image[0].logical_x_dimension; i++)
				{
					i_logi = i - other_image[0].physical_address_of_box_center_x;
					i_logi_frac = pow(i_logi * inverse_logical_x_dimension,2) + j_logi_frac;

					if (i_logi_frac >= minimum_radius_sq && i_logi_frac <= maximum_radius_sq)
					{
						// We do ccw rotation to go from other_image (reference) to self (input image)
						ii_phys = i_logi * cos(current_rotation_rad) - j_logi * sin(current_rotation_rad) + self[0].physical_address_of_box_center_x ;
						jj_phys = i_logi * sin(current_rotation_rad) + j_logi * cos(current_rotation_rad) + self[0].physical_address_of_box_center_y ;
						//
						if (int(ii_phys) > 0 && int(ii_phys)+1 < self[0].logical_x_dimension && int(jj_phys) > 0 && int(jj_phys)+1 < self[0].logical_y_dimension ) // potential optimization: we have to compute the floor and ceiling in the interpolation routine. Is it not worth doing the bounds checking in the interpolation routine somehow?
						{
							self[0].GetRealValueByLinearInterpolationNoBoundsCheckImage(ii_phys,jj_phys,current_interpolated_value);
							//MyDebugPrint("%g %g\n",current_interpolated_value,other_image[0].real_values[address_in_other_image]);
							cc_numerator_dist.AddSampleValue(current_interpolated_value * other_image[current_image].real_values[address_in_other_image]);
							cc_denom_other_dist.AddSampleValue(pow(other_image[0].real_values[address_in_other_image],2)); // potential optimization: since other_image is not being rotated, we should only need to compute this quantity once, not for every potential rotation
							cc_denom_self_dist.AddSampleValue(pow(current_interpolated_value,2));
						}
					}
					address_in_other_image++;
				} // i
				address_in_other_image += other_image[0].padding_jump_value;
			} // end of loop over other (reference) image
		} // end of loop over array of images

		current_cc = cc_numerator_dist.GetSampleSum() / sqrt(cc_denom_other_dist.GetSampleSum()*cc_denom_self_dist.GetSampleSum());


		if (current_cc > best_cc)
		{
			best_cc = current_cc;
			best_rotation = current_rotation;
		}

		// Increment the rotation
		current_rotation += search_step_size;

	} // end of loop over rotations

	return best_rotation;
}


