#include "../../core/core_headers.h"

const std::string ctffind_version = "4.1.3";

class
CtffindApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();
	void AddCommandLineOptions();


	private:

};

class ImageCTFComparison
{
public:
	ImageCTFComparison(int wanted_number_of_images, CTF wanted_ctf, float wanted_pixel_size, bool should_find_phase_shift, bool wanted_astigmatism_is_known, float wanted_known_astigmatism, float wanted_known_astigmatism_angle, bool should_fit_defocus_sweep);
	~ImageCTFComparison();
	void SetImage(int wanted_image_number, Image *new_image);
	void SetCTF(CTF new_ctf);
	CTF ReturnCTF();
	bool AstigmatismIsKnown();
	float ReturnKnownAstigmatism();
	float ReturnKnownAstigmatismAngle();
	bool FindPhaseShift();

	int 	number_of_images;
	Image 	*img;		// Usually an amplitude spectrum, or an array of amplitude spectra

private:
	CTF		ctf;
	float	pixel_size;
	bool	find_phase_shift;
	bool	astigmatism_is_known;
	float	known_astigmatism;
	float 	known_astigmatism_angle;
	bool 	fit_defocus_sweep;
};

class CurveCTFComparison
{
public:
	float	*curve;	// Usually the 1D rotational average of the amplitude spectrum of an image
	int		number_of_bins;
	float	reciprocal_pixel_size; // In reciprocal pixels
	CTF		ctf;
	bool 	find_phase_shift;
};

ImageCTFComparison::ImageCTFComparison(int wanted_number_of_images, CTF wanted_ctf, float wanted_pixel_size, bool should_find_phase_shift, bool wanted_astigmatism_is_known, float wanted_known_astigmatism, float wanted_known_astigmatism_angle, bool should_fit_defocus_sweep)
{
	MyDebugAssertTrue(wanted_number_of_images >= 0, "Bad wanted number of images: %i\n",wanted_number_of_images);
	number_of_images = wanted_number_of_images;
	img = new Image [wanted_number_of_images];

	ctf = wanted_ctf;
	pixel_size = wanted_pixel_size;
	find_phase_shift = should_find_phase_shift;
	astigmatism_is_known = wanted_astigmatism_is_known;
	known_astigmatism = wanted_known_astigmatism;
	known_astigmatism_angle = wanted_known_astigmatism_angle;
	fit_defocus_sweep = should_fit_defocus_sweep;
}

ImageCTFComparison::~ImageCTFComparison()
{
	for (int image_counter = 0; image_counter < number_of_images; image_counter++)
	{
		img[image_counter].Deallocate();
	}
	delete [] img;
}

void ImageCTFComparison::SetImage(int wanted_image_number, Image *new_image)
{
	MyDebugAssertTrue(wanted_image_number >= 0 && wanted_image_number < number_of_images, "Wanted image number (%i) is out of bounds", wanted_image_number);
	img[wanted_image_number].CopyFrom(new_image);
}

void ImageCTFComparison::SetCTF(CTF new_ctf)
{
	ctf = new_ctf;
}

CTF ImageCTFComparison::ReturnCTF() { return ctf; }
bool ImageCTFComparison::AstigmatismIsKnown() { return astigmatism_is_known; }
float ImageCTFComparison::ReturnKnownAstigmatism() { return known_astigmatism; }
float ImageCTFComparison::ReturnKnownAstigmatismAngle() { return known_astigmatism_angle; }
bool ImageCTFComparison::FindPhaseShift() { return find_phase_shift; }


// This is the function which will be minimised
float CtffindObjectiveFunction(void *scoring_parameters, float array_of_values[] )
{
	ImageCTFComparison *comparison_object = reinterpret_cast < ImageCTFComparison *> (scoring_parameters);

	CTF my_ctf = comparison_object->ReturnCTF();
	if (comparison_object->AstigmatismIsKnown())
	{
		MyDebugAssertTrue(comparison_object->ReturnKnownAstigmatism() >= 0.0,"Known asitgmatism must be >= 0.0");
		my_ctf.SetDefocus(array_of_values[0],array_of_values[0] - comparison_object->ReturnKnownAstigmatism(), comparison_object->ReturnKnownAstigmatismAngle());
	}
	else
	{
		my_ctf.SetDefocus(array_of_values[0],array_of_values[1],array_of_values[2]);
	}
	if (comparison_object->FindPhaseShift())
	{
		if (comparison_object->AstigmatismIsKnown())
		{
			my_ctf.SetAdditionalPhaseShift(array_of_values[1]);
		}
		else
		{
			my_ctf.SetAdditionalPhaseShift(array_of_values[3]);
		}
	}

	//MyDebugPrint("(CtffindObjectiveFunction) D1 = %6.2f D2 = %6.2f, Ast = %5.2f, Score = %g",my_ctf.GetDefocus1(),my_ctf.GetDefocus2(),my_ctf.GetAstigmatismAzimuth(),- comparison_object->img.GetCorrelationWithCTF(my_ctf));

	// Evaluate the function
	return - comparison_object->img[0].GetCorrelationWithCTF(my_ctf);
}

//#pragma GCC push_options
//#pragma GCC optimize ("O0")

// This is the function which will be minimised when dealing with 1D fitting
float CtffindCurveObjectiveFunction(void *scoring_parameters, float array_of_values[] )
{
	CurveCTFComparison *comparison_object = reinterpret_cast < CurveCTFComparison *> (scoring_parameters);

	CTF my_ctf = comparison_object->ctf;
	my_ctf.SetDefocus(array_of_values[0],array_of_values[0],0.0);
	if (comparison_object->find_phase_shift)
	{
		my_ctf.SetAdditionalPhaseShift(array_of_values[1]);
	}

	// Compute the cross-correlation
	double cross_product = 0.0;
	double norm_curve = 0.0;
	double norm_ctf = 0.0;
	int number_of_values = 0;
	int bin_counter;
	float current_spatial_frequency_squared;
	const float lowest_freq = pow(my_ctf.GetLowestFrequencyForFitting(),2);
	const float highest_freq = pow(my_ctf.GetHighestFrequencyForFitting(),2);
	float current_ctf_value;

	for ( bin_counter = 0 ; bin_counter < comparison_object->number_of_bins; bin_counter ++ )
	{
		current_spatial_frequency_squared = pow(float(bin_counter)*comparison_object->reciprocal_pixel_size,2);
		if (current_spatial_frequency_squared > lowest_freq && current_spatial_frequency_squared < highest_freq)
		{
			current_ctf_value = fabsf(my_ctf.Evaluate(current_spatial_frequency_squared,0.0));
			MyDebugAssertTrue(current_ctf_value >= -1.0 && current_ctf_value <= 1.0,"Bad ctf value: %f",current_ctf_value);
			number_of_values++;
			cross_product += comparison_object->curve[bin_counter] * current_ctf_value;
			norm_curve += pow(comparison_object->curve[bin_counter],2);
			norm_ctf += pow(current_ctf_value,2);
		}
	}

	MyDebugAssertTrue(norm_ctf > 0.0,"Bad norm_ctf: %f\n", norm_ctf);
	MyDebugAssertTrue(norm_curve > 0.0,"Bad norm_curve: %f\n", norm_curve);

	//MyDebugPrint("(CtffindCurveObjectiveFunction) D1 = %6.2f, Score = %g",array_of_values[0], - cross_product / sqrtf(norm_ctf * norm_curve));

	// Note, we are not properly normalizing the cross correlation coefficient. For our
	// purposes this should be OK, since the average power of the theoretical CTF should not
	// change much with defocus. At least I hope so.
	return - cross_product / sqrtf(norm_ctf * norm_curve);


}

//#pragma GCC pop_options

float FindRotationalAlignmentBetweenTwoStacksOfImages(Image *self, Image *other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius);
void ComputeImagesWithNumberOfExtremaAndCTFValues(CTF *ctf, Image *number_of_extrema, Image *ctf_values);
int ReturnSpectrumBinNumber(int number_of_bins, float number_of_extrema_profile[], Image *number_of_extrema, long address, Image *ctf_values, float ctf_values_profile[]);
void ComputeRotationalAverageOfPowerSpectrum( Image *spectrum, CTF *ctf, Image *number_of_extrema, Image *ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[]);
void OverlayCTF( Image *spectrum, CTF *ctf);
void ComputeFRCBetween1DSpectrumAndFit( int number_of_bins, double average[], double fit[], float number_of_extrema_profile[], double frc[], double frc_sigma[]);
void RescaleSpectrumAndRotationalAverage( Image *spectrum, Image *number_of_extrema, Image *ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[], int last_bin_without_aliasing, int last_bin_with_good_fit );


IMPLEMENT_APP(CtffindApp)

// override the DoInteractiveUserInput

void CtffindApp::DoInteractiveUserInput()
{

	float lowest_allowed_minimum_resolution = 50.0;

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
	bool astigmatism_is_known;
	float known_astigmatism;
	float known_astigmatism_angle;
	bool large_astigmatism_expected;
	bool should_restrain_astigmatism;
	float astigmatism_tolerance;
	bool find_additional_phase_shift;
	float minimum_additional_phase_shift;
	float maximum_additional_phase_shift;
	float additional_phase_shift_search_step;
	bool give_expert_options;
	bool resample_if_pixel_too_small;

	// Things we need for old school input
	double temp_double;
	long temp_long;
	float xmag, dstep;
	const bool		old_school_input          = command_line_parser.FoundSwitch("old-school-input");
	const bool 		old_school_input_ctffind4 = command_line_parser.FoundSwitch("old-school-input-ctffind4");


	if (old_school_input || old_school_input_ctffind4)
	{

		astigmatism_is_known = false;
		known_astigmatism = 0.0;
		known_astigmatism_angle = 0.0;
		resample_if_pixel_too_small = true;

		char buf[4096];
		wxString my_string;

		// Line 1
		std::cin.getline(buf,4096);
		input_filename = buf;

		// Line 2
		std::cin.getline(buf,4096);
		output_diagnostic_filename = buf;

		// Line 3
		std::cin.getline(buf,4096);
		my_string = buf;
		wxStringTokenizer tokenizer(my_string,",");
		if (tokenizer.CountTokens() != 5)
		{
			MyPrintfRed("Bad number of arguments (%i, expected %i) in line 3 of input\n",tokenizer.CountTokens(),5);
			abort();
		}
		while (tokenizer.HasMoreTokens())
		{
			switch (tokenizer.GetPosition())
			{
				case 0: tokenizer.GetNextToken().ToDouble(&temp_double);
						spherical_aberration = float(temp_double);
						break;
				case 1: tokenizer.GetNextToken().ToDouble(&temp_double);
						acceleration_voltage = float(temp_double);
						break;
				case 2: tokenizer.GetNextToken().ToDouble(&temp_double);
						amplitude_contrast = float(temp_double);
						break;
				case 3: tokenizer.GetNextToken().ToDouble(&temp_double);
						xmag = float(temp_double);
						break;
				case 4: tokenizer.GetNextToken().ToDouble(&temp_double);
						dstep = float(temp_double);
						break;
			}
		}
		pixel_size = dstep * 10000.0 / xmag;

		// Line 4
		std::cin.getline(buf,4096);
		my_string = buf;
		tokenizer.SetString(my_string,",");
		if (tokenizer.CountTokens() != 7)
		{
			MyPrintfRed("Bad number of arguments (%i, expected %i) in line 4 of input\n",tokenizer.CountTokens(),7);
			abort();
		}
		while (tokenizer.HasMoreTokens())
		{
			switch (tokenizer.GetPosition())
			{
				case 0: tokenizer.GetNextToken().ToLong(&temp_long);
						box_size = int(temp_long);
						break;
				case 1: tokenizer.GetNextToken().ToDouble(&temp_double);
						minimum_resolution = float(temp_double);
						break;
				case 2: tokenizer.GetNextToken().ToDouble(&temp_double);
						maximum_resolution = float(temp_double);
						break;
				case 3: tokenizer.GetNextToken().ToDouble(&temp_double);
						minimum_defocus= float(temp_double);
						break;
				case 4: tokenizer.GetNextToken().ToDouble(&temp_double);
						maximum_defocus = float(temp_double);
						break;
				case 5: tokenizer.GetNextToken().ToDouble(&temp_double);
						defocus_search_step = float(temp_double);
						break;
				case 6: tokenizer.GetNextToken().ToDouble(&temp_double);
						astigmatism_tolerance = float(temp_double);
						break;
			}
		}
		// If we are getting dAst = 0.0, which is the default in Relion, the user probably
		// expects the ctffind3 behaviour, which is no restraint on astigmatism
		if (astigmatism_tolerance == 0.0) astigmatism_tolerance = -100.0;

		// Output for old-school users
		if (is_running_locally)
		{
			wxPrintf("\n CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]");
			wxPrintf("%5.1f%9.1f%8.2f%10.1f%9.3f\n\n",spherical_aberration,acceleration_voltage,amplitude_contrast,xmag,dstep);
		}

		// Extra lines of input
		if (old_school_input_ctffind4)
		{
			// Line 5
			std::cin.getline(buf,4096);
			my_string = buf;
			tokenizer.SetString(my_string,",");
			if (tokenizer.CountTokens() != 2)
			{
				MyPrintfRed("Bad number of arguments (%i, expected %i) in line 5 of input\n",tokenizer.CountTokens(),2);
				abort();
			}
			while (tokenizer.HasMoreTokens())
			{
				switch (tokenizer.GetPosition())
				{
					case 0: tokenizer.GetNextToken().ToDouble(&temp_double);
							if (int(temp_double) != 0) {
								input_is_a_movie = true;
							}
							else
							{
								input_is_a_movie = false;
							}
							break;
					case 1: tokenizer.GetNextToken().ToDouble(&temp_double);
							number_of_frames_to_average = 1;
							if (input_is_a_movie) { number_of_frames_to_average = int(temp_double); }
							break;
				}
			}

			// Line 6
			std::cin.getline(buf,4096);
			my_string = buf;
			tokenizer.SetString(my_string,",");
			if (tokenizer.CountTokens() != 4)
			{
				MyPrintfRed("Bad number of arguments (%i, expected %i) in line 6 of input\n",tokenizer.CountTokens(),4);
				abort();
			}
			while (tokenizer.HasMoreTokens())
			{
				switch (tokenizer.GetPosition())
				{
					case 0: tokenizer.GetNextToken().ToDouble(&temp_double);
							if (int(temp_double) != 0) {
								find_additional_phase_shift = true;
							}
							else
							{
								find_additional_phase_shift = false;
							}
							break;
					case 1: tokenizer.GetNextToken().ToDouble(&temp_double);
							minimum_additional_phase_shift = 0.0;
							if (find_additional_phase_shift) { minimum_additional_phase_shift = float(temp_double); }
							break;
					case 2: tokenizer.GetNextToken().ToDouble(&temp_double);
							maximum_additional_phase_shift = 0.0;
							if (find_additional_phase_shift) { maximum_additional_phase_shift = float(temp_double); }
							break;
					case 3: tokenizer.GetNextToken().ToDouble(&temp_double);
							additional_phase_shift_search_step = 0.0;
							if (find_additional_phase_shift) { additional_phase_shift_search_step = float(temp_double); }
							break;
				}
			}
		} // end of old school ctffind4 input
		else
		{
			input_is_a_movie = false;
			find_additional_phase_shift = false;
			minimum_additional_phase_shift = 0.0;
			maximum_additional_phase_shift = 0.0;
			additional_phase_shift_search_step = 0.0;
		}

		// Do some argument checking on movie processing option
		MRCFile input_file(input_filename,false);
		if (input_is_a_movie)
		{
			if (input_file.ReturnZSize() < number_of_frames_to_average)
			{
				SendError(wxString::Format("Input stack has %i images, so you cannot average %i frames together\n",input_file.ReturnZSize(),number_of_frames_to_average));
				ExitMainLoop();
			}
		}
		else
		{
			// We're not doing movie processing
			if (input_file.ReturnZSize() > 1)
			{
				SendError("Input stacks are only supported --old-school-input-ctffind4 if doing movie processing\n");
				ExitMainLoop();
			}

		}

		if (find_additional_phase_shift)
		{
			if (minimum_additional_phase_shift > maximum_additional_phase_shift)
			{
				SendError(wxString::Format("Minimum phase shift (%f) cannot be greater than maximum phase shift (%f)\n",minimum_additional_phase_shift,maximum_additional_phase_shift));
				ExitMainLoop();
			}
		}



	} // end of test for old-school-input or old-school-input-ctffind4
	else
	{

		UserInput *my_input = new UserInput("Ctffind", ctffind_version);

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
		pixel_size 					= my_input->GetFloatFromUser("Pixel size","In Angstroms","1.0",0.0);
		acceleration_voltage 		= my_input->GetFloatFromUser("Acceleration voltage","in kV","300.0",0.0);
		spherical_aberration 		= my_input->GetFloatFromUser("Spherical aberration","in mm","2.7",0.0);
		amplitude_contrast 			= my_input->GetFloatFromUser("Amplitude contrast","Fraction of amplitude contrast","0.07",0.0,1.0);
		box_size 					= my_input->GetIntFromUser("Size of amplitude spectrum to compute","in pixels","512",128);
		minimum_resolution 			= my_input->GetFloatFromUser("Minimum resolution","Lowest resolution used for fitting CTF (Angstroms)","30.0",0.0,lowest_allowed_minimum_resolution);
		maximum_resolution 			= my_input->GetFloatFromUser("Maximum resolution","Highest resolution used for fitting CTF (Angstroms)","5.0",0.0,minimum_resolution);
		minimum_defocus 			= my_input->GetFloatFromUser("Minimum defocus","Positive values for underfocus. Lowest value to search over (Angstroms)","5000.0");
		maximum_defocus 			= my_input->GetFloatFromUser("Maximum defocus","Positive values for underfocus. Highest value to search over (Angstroms)","50000.0",minimum_defocus);
		defocus_search_step 		= my_input->GetFloatFromUser("Defocus search step","Step size for defocus search (Angstroms)","500.0",1.0);
		astigmatism_is_known		= my_input->GetYesNoFromUser("Do you know what astigmatism is present?","Answer yes if you already know how much astigmatism was present. If you answer no, the program will search for the astigmatism and astigmatism angle","no");
		if (astigmatism_is_known)
		{
			large_astigmatism_expected = false;
			should_restrain_astigmatism = false;
			astigmatism_tolerance = -100.0;
			known_astigmatism		= my_input->GetFloatFromUser("Known astigmatism", "In Angstroms, the amount of astigmatism, defined as the difference between the defocus along the major and minor axes","0.0",0.0);
			known_astigmatism_angle = my_input->GetFloatFromUser("Known astigmatism angle", "In degrees, the angle of astigmatism","0.0");
		}
		else
		{
			large_astigmatism_expected	= my_input->GetYesNoFromUser("Do you expect very large astigmatism?","Answer yes if you expect very high astigmatism (say, much greater than 1000A). In that case, a slower search over 2D spectra will be used for the initial search","no");
			should_restrain_astigmatism = my_input->GetYesNoFromUser("Use a restraint on astigmatism?","If you answer yes, the CTF parameter search and refinement will penalise large astigmatism. You will specify the astigmatism tolerance in the next question. If you answer no, no such restraint will apply","yes");
			if (should_restrain_astigmatism)
			{
				astigmatism_tolerance 	= my_input->GetFloatFromUser("Expected (tolerated) astigmatism","Astigmatism values much larger than this will be penalised (Angstroms). Give a negative value to turn off this restraint.","200.0");
			}
			else
			{
				astigmatism_tolerance 	= -100.0; // a negative value here signals that we don't want any restraint on astigmatism
			}
		}

		find_additional_phase_shift = my_input->GetYesNoFromUser("Find additional phase shift?","Input micrograph was recorded using a phase plate with variable phase shift, which you want to find","no");

		if (find_additional_phase_shift)
		{
			minimum_additional_phase_shift 		= my_input->GetFloatFromUser("Minimum phase shift","Lower bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians","0.0",-3.15,3.15);
			maximum_additional_phase_shift 		= my_input->GetFloatFromUser("Maximum phase shift","Upper bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians","3.15",minimum_additional_phase_shift,3.15);
			additional_phase_shift_search_step 	= my_input->GetFloatFromUser("Phase shift search step","Step size for phase shift search (radians)","0.2",0.001,maximum_additional_phase_shift-minimum_additional_phase_shift);
		}
		else
		{
			minimum_additional_phase_shift = 0.0;
			maximum_additional_phase_shift = 0.0;
			additional_phase_shift_search_step = 0.0;
		}

		give_expert_options						= my_input->GetYesNoFromUser("Do you want to set expert options?","There are options which normally not changed, but can be accessed by answering yes here","no");
		if (give_expert_options)
		{
			resample_if_pixel_too_small 		= my_input->GetYesNoFromUser("Resample micrograph if pixel size too small?","When the pixel is too small, Thon rings appear very thin and near the origin of the spectrum, which can lead to suboptimal fitting. This options resamples micrographs to a more reasonable pixel size if needed","yes");
		}
		else
		{
			resample_if_pixel_too_small			 = true;
		}

		delete my_input;

	}

	my_current_job.Reset(23);
	my_current_job.ManualSetArguments("tbitffffiffffffbfffbffb",input_filename.c_str(),
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
																additional_phase_shift_search_step,
																astigmatism_is_known,
																known_astigmatism,
																known_astigmatism_angle,
																resample_if_pixel_too_small);


}


// Optional command-line stuff
void CtffindApp::AddCommandLineOptions()
{
	command_line_parser.AddLongSwitch("old-school-input","Pretend this is ctffind3 (for compatibility with old scripts and programs)");
	command_line_parser.AddLongSwitch("old-school-input-ctffind4","Accept parameters from stdin, like ctffind3, but with extra lines for ctffind4-specific options (movie processing and phase shift estimation");
	command_line_parser.AddLongSwitch("amplitude-spectrum-input","The input image is an amplitude spectrum, not a real-space image");
	command_line_parser.AddLongSwitch("filtered-amplitude-spectrum-input","The input image is filtered (background-subtracted) amplitude spectrum");
	command_line_parser.AddLongSwitch("fast","Skip computation of fit statistics as well as spectrum contrast enhancement");
}



// override the do calculation method which will be what is actually run..

bool CtffindApp::DoCalculation()
{

	// Arguments for this job

	const std::string 	input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	const bool			input_is_a_movie 					= my_current_job.arguments[1].ReturnBoolArgument();
	const int         	number_of_frames_to_average			= my_current_job.arguments[2].ReturnIntegerArgument();
	const std::string 	output_diagnostic_filename			= my_current_job.arguments[3].ReturnStringArgument();
	const float 		pixel_size_of_input_image			= my_current_job.arguments[4].ReturnFloatArgument();
	const float 		acceleration_voltage				= my_current_job.arguments[5].ReturnFloatArgument();
	const float       	spherical_aberration				= my_current_job.arguments[6].ReturnFloatArgument();
	const float 		amplitude_contrast					= my_current_job.arguments[7].ReturnFloatArgument();
	const int         	box_size							= my_current_job.arguments[8].ReturnIntegerArgument();
	const float 		minimum_resolution					= my_current_job.arguments[9].ReturnFloatArgument();
	const float       	maximum_resolution					= my_current_job.arguments[10].ReturnFloatArgument();
	const float       	minimum_defocus						= my_current_job.arguments[11].ReturnFloatArgument();
	const float       	maximum_defocus						= my_current_job.arguments[12].ReturnFloatArgument();
	const float       	defocus_search_step					= my_current_job.arguments[13].ReturnFloatArgument();
	const float       	astigmatism_tolerance               = my_current_job.arguments[14].ReturnFloatArgument();
	const bool       	find_additional_phase_shift         = my_current_job.arguments[15].ReturnBoolArgument();
	const float  		minimum_additional_phase_shift		= my_current_job.arguments[16].ReturnFloatArgument();
	const float			maximum_additional_phase_shift		= my_current_job.arguments[17].ReturnFloatArgument();
	const float			additional_phase_shift_search_step	= my_current_job.arguments[18].ReturnFloatArgument();
	const bool  		astigmatism_is_known				= my_current_job.arguments[19].ReturnBoolArgument();
	const float 		known_astigmatism					= my_current_job.arguments[20].ReturnFloatArgument();
	const float 		known_astigmatism_angle				= my_current_job.arguments[21].ReturnFloatArgument();
	const bool			resample_if_pixel_too_small			= my_current_job.arguments[22].ReturnBoolArgument();

	// These variables will be set by command-line options
	const bool		old_school_input = command_line_parser.FoundSwitch("old-school-input") || command_line_parser.FoundSwitch("old-school-input-ctffind4");
	const bool		amplitude_spectrum_input = command_line_parser.FoundSwitch("amplitude-spectrum-input");
	const bool		filtered_amplitude_spectrum_input = command_line_parser.FoundSwitch("filtered-amplitude-spectrum-input");
	const bool 		compute_extra_stats = ! command_line_parser.FoundSwitch("fast");
	const bool		boost_ring_contrast = ! command_line_parser.FoundSwitch("fast");

	// Resampling of input images to ensure that the pixel size isn't too small
	const float		target_nyquist_after_resampling = 2.8; // Angstroms
	const float 	target_pixel_size_after_resampling = 0.5 * target_nyquist_after_resampling;
	float 			pixel_size_for_fitting = pixel_size_of_input_image;
	int				temporary_box_size;

	// If the expected astigmatism is less than this value, we will do the initial search in 1D
	const float		maximum_expected_astigmatism_for_1D_search = 1000.0;


	/*
	 *  Scoring function
	 */
	float MyFunction(float []);

	// Other variables
	int					number_of_movie_frames;
	int         		number_of_micrographs;
	MRCFile				input_file(input_filename,false);
	Image				*average_spectrum = new Image();
	wxString			output_text_fn;
	ProgressBar			*my_progress_bar;
	NumericTextFile		*output_text;
	NumericTextFile		*output_text_avrot;
	int					current_micrograph_number;
	int					number_of_tiles_used;
	Image 				*current_power_spectrum = new Image();
	int					current_first_frame_within_average;
	int					current_frame_within_average;
	int					current_input_location;
	Image				*current_input_image = new Image();
	Image				*current_input_image_square = new Image();
	int					micrograph_square_dimension;
	Image				*temp_image = new Image();
	Image				*sum_image = new Image();
	Image				*resampled_power_spectrum = new Image();
	bool				resampling_is_necessary;
	CTF					current_ctf;
	float				average, sigma;
	int					convolution_box_size;
	ImageCTFComparison	*comparison_object_2D;
	CurveCTFComparison	comparison_object_1D;
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
	int					number_of_bins_in_1d_spectra;
	Curve				number_of_averaged_pixels;
	Curve				rotational_average;
	Image				*number_of_extrema_image = new Image();
	Image				*ctf_values_image = new Image();
	double				*rotational_average_astig = NULL;
	double				*spatial_frequency = NULL;
	double				*spatial_frequency_in_reciprocal_angstroms = NULL;
	double				*rotational_average_astig_fit = NULL;
	float				*number_of_extrema_profile = NULL;
	float				*ctf_values_profile = NULL;
	double				*fit_frc = NULL;
	double				*fit_frc_sigma = NULL;
	MRCFile				output_diagnostic_file(output_diagnostic_filename,true);
	int					last_bin_with_good_fit;
	double 				*values_to_write_out = new double[7];
	float				best_score_after_initial_phase;
	int					last_bin_without_aliasing;



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
		input_file.PrintInfo();
	}

	// Prepare the output text file
	output_text_fn = FilenameReplaceExtension(output_diagnostic_filename,"txt");

	if (is_running_locally)
	{
		output_text = new NumericTextFile(output_text_fn,OPEN_TO_WRITE,7);

		// Print header to the output text file
		output_text->WriteCommentLine("# Output from CTFFind version %s, run on %s\n",ctffind_version,wxDateTime::Now().FormatISOCombined(' ').ToUTF8().data());
		output_text->WriteCommentLine("# Input file: %s ; Number of micrographs: %i\n",input_filename.c_str(),number_of_micrographs);
		output_text->WriteCommentLine("# Pixel size: %0.3f Angstroms ; acceleration voltage: %0.1f keV ; spherical aberration: %0.1f mm ; amplitude contrast: %0.2f\n",pixel_size_of_input_image,acceleration_voltage,spherical_aberration,amplitude_contrast);
		output_text->WriteCommentLine("# Box size: %i pixels ; min. res.: %0.1f Angstroms ; max. res.: %0.1f Angstroms ; min. def.: %0.1f um; max. def. %0.1f um\n",box_size,minimum_resolution,maximum_resolution,minimum_defocus,maximum_defocus);
		output_text->WriteCommentLine("# Columns: #1 - micrograph number; #2 - defocus 1 [Angstroms]; #3 - defocus 2; #4 - azimuth of astigmatism; #5 - additional phase shift [radians]; #6 - cross correlation; #7 - spacing (in Angstroms) up to which CTF rings were fit successfully\n");
	}

	// Prepare a text file with 1D rotational average spectra
	output_text_fn = FilenameAddSuffix(output_text_fn.ToStdString(),"_avrot");

	if (! old_school_input && number_of_micrographs > 1 && is_running_locally)
	{
		wxPrintf("Will estimate the CTF parmaeters for %i micrographs.\n",number_of_micrographs);
		wxPrintf("Results will be written to this file: %s\n",output_text->ReturnFilename());
		wxPrintf("\nEstimating CTF parameters...\n\n");
		my_progress_bar = new ProgressBar(number_of_micrographs);
	}




	// Prepare the average spectrum image
	average_spectrum->Allocate(box_size,box_size,true);

	// Loop over micrographs
	for (current_micrograph_number=1; current_micrograph_number <= number_of_micrographs; current_micrograph_number++)
	{
		if (is_running_locally && (old_school_input || number_of_micrographs == 1)) wxPrintf("Working on micrograph %i of %i\n", current_micrograph_number, number_of_micrographs);

		number_of_tiles_used = 0;
		average_spectrum->SetToConstant(0.0);
		average_spectrum->is_in_real_space = true;

		if (amplitude_spectrum_input || filtered_amplitude_spectrum_input)
		{
			current_power_spectrum->ReadSlice(&input_file,current_micrograph_number);
			current_power_spectrum->ForwardFFT();
			average_spectrum->Allocate(box_size,box_size,1,false);
			current_power_spectrum->ClipInto(average_spectrum);
			average_spectrum->BackwardFFT();
		}
		else
		{
			for (current_first_frame_within_average = 1; current_first_frame_within_average <= number_of_movie_frames; current_first_frame_within_average += number_of_frames_to_average)
			{
				for (current_frame_within_average = 1; current_frame_within_average <= number_of_frames_to_average; current_frame_within_average++)
				{
					current_input_location = current_first_frame_within_average + number_of_movie_frames * (current_micrograph_number-1) + (current_frame_within_average-1);
					if (current_input_location > number_of_movie_frames * current_micrograph_number) continue;
					current_input_image->ReadSlice(&input_file,current_input_location);
					if (current_input_image->IsConstant())
					{
						SendError(wxString::Format("Error: location %i of input file %s is blank",current_input_location, input_filename));
						ExitMainLoop();
					}
					// Make the image square
					micrograph_square_dimension = std::max(current_input_image->logical_x_dimension,current_input_image->logical_y_dimension);
					if (IsOdd((micrograph_square_dimension))) micrograph_square_dimension++;
					if (current_input_image->logical_x_dimension != micrograph_square_dimension || current_input_image->logical_y_dimension != micrograph_square_dimension)
					{
						current_input_image_square->Allocate(micrograph_square_dimension,micrograph_square_dimension,true);
						//current_input_image->ClipInto(current_input_image_square,current_input_image->ReturnAverageOfRealValues());
						current_input_image->ClipIntoLargerRealSpace2D(current_input_image_square,current_input_image->ReturnAverageOfRealValues());
						current_input_image->Consume(current_input_image_square);
					}
					//
					if (current_frame_within_average == 1)
					{
						sum_image->Allocate(current_input_image->logical_x_dimension,current_input_image->logical_y_dimension,true);
						sum_image->SetToConstant(0.0);
					}
					sum_image->AddImage(current_input_image);
				} // end of loop over frames to average together
				current_input_image->Consume(sum_image);

				// Taper the edges of the micrograph in real space, to lessen Gibbs artefacts
				// Introduces an artefact of its own, so it's not clear on balance whether tapering helps, especially with modern micrographs from good detectors
				//current_input_image->TaperEdges();

				number_of_tiles_used++;

				// Compute the amplitude spectrum
				current_power_spectrum->Allocate(current_input_image->logical_x_dimension,current_input_image->logical_y_dimension,true);
				current_input_image->ForwardFFT(false);
				current_input_image->ComputeAmplitudeSpectrumFull2D(current_power_spectrum);

				//current_power_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_resampling.mrc",1);

				// Set origin of amplitude spectrum to 0.0
				current_power_spectrum->real_values[current_power_spectrum->ReturnReal1DAddressFromPhysicalCoord(current_power_spectrum->physical_address_of_box_center_x,current_power_spectrum->physical_address_of_box_center_y,current_power_spectrum->physical_address_of_box_center_z)] = 0.0;

				// Resample the amplitude spectrum
				if (resample_if_pixel_too_small && pixel_size_of_input_image < target_pixel_size_after_resampling)
				{
					// The input pixel was too small, so let's resample the amplitude spectrum into a large temporary box, before clipping the center out for fitting
					temporary_box_size = round(float(box_size) / pixel_size_of_input_image * target_pixel_size_after_resampling);
					if (IsOdd(temporary_box_size)) temporary_box_size++;
					resampling_is_necessary = current_power_spectrum->logical_x_dimension != box_size || current_power_spectrum->logical_y_dimension != box_size;
					if (resampling_is_necessary)
					{
						current_power_spectrum->ForwardFFT(false);
						resampled_power_spectrum->Allocate(temporary_box_size,temporary_box_size,1,false);
						current_power_spectrum->ClipInto(resampled_power_spectrum);
						resampled_power_spectrum->BackwardFFT();
						temp_image->Allocate(box_size,box_size,1,true);
						temp_image->SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
						resampled_power_spectrum->ClipInto(temp_image);
						resampled_power_spectrum->Consume(temp_image);
					}
					else
					{
						resampled_power_spectrum->CopyFrom(current_power_spectrum);
					}
					pixel_size_for_fitting = pixel_size_of_input_image * float(temporary_box_size) / float(box_size);
				}
				else
				{
					// The regular way (the input pixel size was large enough)
					resampling_is_necessary = current_power_spectrum->logical_x_dimension != box_size || current_power_spectrum->logical_y_dimension != box_size;
					if (resampling_is_necessary)
					{
						current_power_spectrum->ForwardFFT(false);
						resampled_power_spectrum->Allocate(box_size,box_size,1,false);
						current_power_spectrum->ClipInto(resampled_power_spectrum);
						resampled_power_spectrum->BackwardFFT();
					}
					else
					{
						resampled_power_spectrum->CopyFrom(current_power_spectrum);
					}
				}

				average_spectrum->AddImage(resampled_power_spectrum);
			} // end of loop over movie frames

			// We need to take care of the scaling of the FFTs, as well as the averaging of tiles
			if (resampling_is_necessary)
			{
				average_spectrum->MultiplyByConstant(1.0 / ( float(number_of_tiles_used) * current_input_image->logical_x_dimension * current_input_image->logical_y_dimension * current_power_spectrum->logical_x_dimension * current_power_spectrum->logical_y_dimension ) );
			}
			else
			{
				average_spectrum->MultiplyByConstant(1.0 / ( float(number_of_tiles_used) * current_input_image->logical_x_dimension * current_input_image->logical_y_dimension ) );
			}

		} // end of test of whether we were given amplitude spectra on input


		//average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_bg_sub.mrc",1);


		// Filter the amplitude spectrum, remove background
		if (! filtered_amplitude_spectrum_input)
		{
			// Try to weaken cross artefacts
			average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(float(average_spectrum->logical_x_dimension)*pixel_size_for_fitting/minimum_resolution,float(average_spectrum->logical_x_dimension),average,sigma,12);
			average_spectrum->DivideByConstant(sigma);
			average_spectrum->SetMaximumValueOnCentralCross(average+ sigma*10.0); //TODO: check whether average/sigma+10.0 is really what I meant to write. I think it's supposed to be average + 10*sigma

			//average_spectrum->QuickAndDirtyWriteSlice("dbg_average_spectrum_before_conv.mrc",1);

			// Compute low-pass filtered version of the spectrum
			convolution_box_size = int( float(average_spectrum->logical_x_dimension) * pixel_size_for_fitting / minimum_resolution * sqrt(2.0) );
			if (IsEven(convolution_box_size)) convolution_box_size++;
			current_power_spectrum->Allocate(average_spectrum->logical_x_dimension,average_spectrum->logical_y_dimension,true);
			current_power_spectrum->SetToConstant(0.0); // According to valgrind, this avoid potential problems later on.
			average_spectrum->SpectrumBoxConvolution(current_power_spectrum,convolution_box_size,float(average_spectrum->logical_x_dimension)*pixel_size_for_fitting/minimum_resolution);

			//current_power_spectrum->QuickAndDirtyWriteSlice("dbg_spec_convoluted.mrc",1);

			// Subtract low-pass-filtered spectrum from the spectrum. This should remove the background slope.
			average_spectrum->SubtractImage(current_power_spectrum);

			//average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_thresh.mrc",1);

			// Threshold high values
			average_spectrum->SetMaximumValue(average_spectrum->ReturnMaximumValue(3,3));
		}

		// We now have a spectrum which we can use to fit CTFs
		//average_spectrum->QuickAndDirtyWriteSlice("dbg_spec.mrc",1);


		// Set up the CTF object
		current_ctf.Init(acceleration_voltage,spherical_aberration,amplitude_contrast,minimum_defocus,minimum_defocus,0.0,1.0/minimum_resolution,1.0/maximum_resolution,astigmatism_tolerance,pixel_size_for_fitting,minimum_additional_phase_shift);
		current_ctf.SetDefocus(minimum_defocus/pixel_size_for_fitting,minimum_defocus/pixel_size_for_fitting,0.0);
		current_ctf.SetAdditionalPhaseShift(minimum_additional_phase_shift);


		// Set up the comparison object
		comparison_object_2D = new ImageCTFComparison(1,current_ctf,pixel_size_for_fitting,find_additional_phase_shift, astigmatism_is_known, known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PI, false);
		comparison_object_2D->SetImage(0,average_spectrum);
		/*
		comparison_object_2D.ctf = current_ctf;
		comparison_object_2D.img = average_spectrum;
		comparison_object_2D.pixel_size = pixel_size_for_fitting;
		comparison_object_2D.find_phase_shift = find_additional_phase_shift;
		comparison_object_2D.astigmatism_is_known = astigmatism_is_known;
		comparison_object_2D.known_astigmatism = known_astigmatism / pixel_size_for_fitting;
		comparison_object_2D.known_astigmatism_angle = known_astigmatism_angle / 180.0 * PI;
		*/

		if (is_running_locally && old_school_input)
		{
			wxPrintf("\nSEARCHING CTF PARAMETERS...\n");
		}


		// Let's look for the astigmatism angle first
		if (astigmatism_is_known)
		{
			estimated_astigmatism_angle = known_astigmatism_angle;
		}
		else
		{
			temp_image->CopyFrom(average_spectrum);
			temp_image->ApplyMirrorAlongY();
			//temp_image.QuickAndDirtyWriteSlice("dbg_spec_y.mrc",1);
			estimated_astigmatism_angle = 0.5 * FindRotationalAlignmentBetweenTwoStacksOfImages(average_spectrum,temp_image,1,90.0,5.0,pixel_size_for_fitting/minimum_resolution,pixel_size_for_fitting/maximum_resolution);
		}

		//MyDebugPrint ("Estimated astigmatism angle = %f\n", estimated_astigmatism_angle);



		/*
		 * Initial brute-force search, either 2D (if large astigmatism) or 1D
		 */
		if ((astigmatism_tolerance <= maximum_expected_astigmatism_for_1D_search && astigmatism_tolerance > 0.0) || ( astigmatism_is_known && known_astigmatism <= maximum_expected_astigmatism_for_1D_search) )
		{

			// 1D rotational average
			number_of_bins_in_1d_spectra = int(ceil(average_spectrum->ReturnMaximumDiagonalRadius()) + 2);
			rotational_average.SetupXAxis(0.0,sqrt(2.0)*0.5,number_of_bins_in_1d_spectra);
			number_of_averaged_pixels = rotational_average;
			average_spectrum->Compute1DRotationalAverage(rotational_average,number_of_averaged_pixels,true);



			comparison_object_1D.ctf = current_ctf;
			comparison_object_1D.curve = new float[number_of_bins_in_1d_spectra];
			for (counter=0; counter < number_of_bins_in_1d_spectra; counter++)
			{
				comparison_object_1D.curve[counter] = rotational_average.data_y[counter];
			}
			comparison_object_1D.find_phase_shift = find_additional_phase_shift;
			comparison_object_1D.number_of_bins = number_of_bins_in_1d_spectra;
			comparison_object_1D.reciprocal_pixel_size = average_spectrum->fourier_voxel_size_x;

			// We can now look for the defocus value
			bf_halfrange[0] = 0.5 * (maximum_defocus - minimum_defocus) / pixel_size_for_fitting;
			bf_halfrange[1] = 0.5 * (maximum_additional_phase_shift - minimum_additional_phase_shift);

			bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
			bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[1];

			bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
			bf_stepsize[1] = additional_phase_shift_search_step;

			if (find_additional_phase_shift)
			{
				number_of_search_dimensions = 2;
			}
			else
			{
				number_of_search_dimensions = 1;
			}

			// Actually run the BF search
			brute_force_search = new BruteForceSearch();
			brute_force_search->Init(&CtffindCurveObjectiveFunction,&comparison_object_1D,number_of_search_dimensions,bf_midpoint,bf_halfrange,bf_stepsize,false,false);
			brute_force_search->Run();

			// We can now do a local optimization
			// The end point of the BF search is the beginning of the CG search
			for (counter=0;counter<number_of_search_dimensions;counter++)
			{
				cg_starting_point[counter] = brute_force_search->GetBestValue(counter);
			}
			cg_accuracy[0] = 100.0;
			cg_accuracy[1] = 0.05;
			conjugate_gradient_minimizer = new ConjugateGradient();
			conjugate_gradient_minimizer->Init(&CtffindCurveObjectiveFunction,&comparison_object_1D,number_of_search_dimensions,cg_starting_point,cg_accuracy);
			conjugate_gradient_minimizer->Run();
			for (counter=0;counter<number_of_search_dimensions;counter++)
			{
				cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
			}
			current_ctf.SetDefocus(cg_starting_point[0],cg_starting_point[0],estimated_astigmatism_angle);
			if (find_additional_phase_shift)
			{
				current_ctf.SetAdditionalPhaseShift(cg_starting_point[1]);
			}

			// Remember the best score so far
			best_score_after_initial_phase = - conjugate_gradient_minimizer->GetBestScore();

			// Set up the 2D comparison object, which we will soon need
			comparison_object_2D->SetCTF(current_ctf);
			//comparison_object_2D.img = average_spectrum;
			//comparison_object_2D.pixel_size_of_input_image = pixel_size_of_input_image;
			//comparison_object_2D.find_phase_shift = find_additional_phase_shift;

			// Set up the starting point for the 2D conjugate gradient minimization
			if (astigmatism_is_known)
			{
				cg_starting_point[0] = current_ctf.GetDefocus1();
				if (find_additional_phase_shift) cg_starting_point[1] = current_ctf.GetAdditionalPhaseShift();
				if (find_additional_phase_shift)
				{
					number_of_search_dimensions = 2;
				}
				else
				{
					number_of_search_dimensions = 1;
				}
			}
			else
			{
				cg_starting_point[0] = current_ctf.GetDefocus1();
				cg_starting_point[1] = current_ctf.GetDefocus2();
				cg_starting_point[2] = estimated_astigmatism_angle; // We could run the mirror trick to get a better starting point - just not sure whether worth it
				if (find_additional_phase_shift) cg_starting_point[3] = current_ctf.GetAdditionalPhaseShift();
				if (find_additional_phase_shift)
				{
					number_of_search_dimensions = 4;
				}
				else
				{
					number_of_search_dimensions = 3;
				}
			}



			// Cleanup
			delete conjugate_gradient_minimizer;
			delete brute_force_search;
			delete [] comparison_object_1D.curve;
		}
		else
		{
			// 2D fitting


			// We can now look for the defocus value
			if (astigmatism_is_known)
			{
				bf_halfrange[0] = 0.5 * (maximum_defocus-minimum_defocus)/pixel_size_for_fitting;
				bf_halfrange[1] = 0.5 * (maximum_additional_phase_shift-minimum_additional_phase_shift);

				bf_midpoint[0] = minimum_defocus/pixel_size_for_fitting + bf_halfrange[0];
				bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[3];

				bf_stepsize[0] = defocus_search_step/pixel_size_for_fitting;
				bf_stepsize[1] = additional_phase_shift_search_step;

				if (find_additional_phase_shift)
				{
					number_of_search_dimensions = 2;
				}
				else
				{
					number_of_search_dimensions = 1;
				}
			}
			else
			{
				bf_halfrange[0] = 0.5 * (maximum_defocus-minimum_defocus)/pixel_size_for_fitting;
				bf_halfrange[1] = bf_halfrange[0];
				bf_halfrange[2] = 0.0;
				bf_halfrange[3] = 0.5 * (maximum_additional_phase_shift-minimum_additional_phase_shift);

				bf_midpoint[0] = minimum_defocus/pixel_size_for_fitting + bf_halfrange[0];
				bf_midpoint[1] = bf_midpoint[0];
				bf_midpoint[2] = estimated_astigmatism_angle / 180.0 * PI;
				bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

				bf_stepsize[0] = defocus_search_step/pixel_size_for_fitting;
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
			}

			// Actually run the BF search
			brute_force_search = new BruteForceSearch();
			brute_force_search->Init(&CtffindObjectiveFunction,comparison_object_2D,number_of_search_dimensions,bf_midpoint,bf_halfrange,bf_stepsize,false,is_running_locally);
			brute_force_search->Run();

			// The end point of the BF search is the beginning of the CG search
			for (counter=0;counter<number_of_search_dimensions;counter++)
			{
				cg_starting_point[counter] = brute_force_search->GetBestValue(counter);
			}

			//
			if (astigmatism_is_known)
			{
				current_ctf.SetDefocus(cg_starting_point[0],cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PI);
				if (find_additional_phase_shift)
				{
					current_ctf.SetAdditionalPhaseShift(cg_starting_point[1]);
				}
			}
			else
			{
				current_ctf.SetDefocus(cg_starting_point[0],cg_starting_point[1],cg_starting_point[2]);
				if (find_additional_phase_shift)
				{
					current_ctf.SetAdditionalPhaseShift(cg_starting_point[3]);
				}
			}
			current_ctf.EnforceConvention();

			// Remember the best score so far
			best_score_after_initial_phase = - brute_force_search->GetBestScore();

			delete brute_force_search;

		} // end of test for whether expected astigmatism is low enough to do 1D search


		// Print out the results of brute force search
		if (is_running_locally && old_school_input)
		{
			wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
			wxPrintf("%12.2f%12.2f%12.2f%12.5f\n",current_ctf.GetDefocus1()*pixel_size_for_fitting,current_ctf.GetDefocus2()*pixel_size_for_fitting,current_ctf.GetAstigmatismAzimuth()*180.0/PI,best_score_after_initial_phase);
		}

		// Now we refine in the neighbourhood by using Powell's conjugate gradient algorithm
		if (is_running_locally && old_school_input)
		{
			wxPrintf("\nREFINING CTF PARAMETERS...\n");
			wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
		}
		if (astigmatism_is_known)
		{
			cg_accuracy[0] = 100.0;
			cg_accuracy[1] = 0.05;
		}
		else
		{
			cg_accuracy[0] = 100.0;
			cg_accuracy[1] = 100.0;
			cg_accuracy[2] = 0.5;
			cg_accuracy[3] = 0.05;
		}
		conjugate_gradient_minimizer = new ConjugateGradient();
		conjugate_gradient_minimizer->Init(&CtffindObjectiveFunction,comparison_object_2D,number_of_search_dimensions,cg_starting_point,cg_accuracy);
		conjugate_gradient_minimizer->Run();

		// Remember the results of the refinement
		for (counter=0;counter<number_of_search_dimensions;counter++)
		{
			cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
		}
		if (astigmatism_is_known)
		{
			current_ctf.SetDefocus(cg_starting_point[0],cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PI);
			if (find_additional_phase_shift)
			{
				current_ctf.SetAdditionalPhaseShift(cg_starting_point[1]);
			}
		}
		else
		{
			current_ctf.SetDefocus(cg_starting_point[0],cg_starting_point[1],cg_starting_point[2]);
			if (find_additional_phase_shift)
			{
				current_ctf.SetAdditionalPhaseShift(cg_starting_point[3]);
			}
		}
		current_ctf.EnforceConvention();

		// Print results to the terminal
		if (is_running_locally && old_school_input)
		{
			wxPrintf("%12.2f%12.2f%12.2f%12.5f   Final Values\n",current_ctf.GetDefocus1()*pixel_size_for_fitting,current_ctf.GetDefocus2()*pixel_size_for_fitting,current_ctf.GetAstigmatismAzimuth()*180.0/PI,-conjugate_gradient_minimizer->GetBestScore());
			if (find_additional_phase_shift)
			{
				wxPrintf("Final phase shift = %0.3f radians\n",current_ctf.GetAdditionalPhaseShift());
			}
		}

		// Generate diagnostic image
		//average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_diag_start.mrc",1);
		current_output_location = current_micrograph_number;
		average_spectrum->AddConstant(-1.0 * average_spectrum->ReturnAverageOfRealValuesOnEdges());
		average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(	sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(2,0.0)) * average_spectrum->logical_x_dimension,
															      	std::max(current_ctf.GetHighestFrequencyForFitting(),sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(3,0.0)))*average_spectrum->logical_x_dimension,
																	average,sigma);
		average_spectrum->CircleMask(5.0,true);
		average_spectrum->SetMaximumValueOnCentralCross(average);
		average_spectrum->SetMinimumAndMaximumValues(average - 4.0 * sigma, average + 4.0 * sigma);
		average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(	sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(2,0.0)) * average_spectrum->logical_x_dimension,
																	std::max(current_ctf.GetHighestFrequencyForFitting(),sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(3,0.0)))*average_spectrum->logical_x_dimension,
																	average,sigma);
		average_spectrum->AddConstant(-1.0 * average);
		average_spectrum->MultiplyByConstant(1.0 / sigma);
		average_spectrum->AddConstant(average);

		//average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_diag_1.mrc",1);

		// 1D rotational average
		number_of_bins_in_1d_spectra = int(ceil(average_spectrum->ReturnMaximumDiagonalRadius()) + 2);
		rotational_average.SetupXAxis(0.0,sqrt(2.0)*0.5,number_of_bins_in_1d_spectra);
		rotational_average.ZeroYData();
		number_of_averaged_pixels.ZeroYData();
		average_spectrum->Compute1DRotationalAverage(rotational_average,number_of_averaged_pixels,true);

		// Rotational average, taking astigmatism into account
		if (compute_extra_stats)
		{
			number_of_extrema_image->Allocate(average_spectrum->logical_x_dimension,average_spectrum->logical_y_dimension,true);
			ctf_values_image->Allocate(average_spectrum->logical_x_dimension,average_spectrum->logical_y_dimension,true);
			spatial_frequency 				= new double[number_of_bins_in_1d_spectra];
			rotational_average_astig 		= new double[number_of_bins_in_1d_spectra];
			rotational_average_astig_fit	= new double[number_of_bins_in_1d_spectra];
			number_of_extrema_profile 		= new float[number_of_bins_in_1d_spectra];
			ctf_values_profile 				= new float[number_of_bins_in_1d_spectra];
			fit_frc							= new double[number_of_bins_in_1d_spectra];
			fit_frc_sigma					= new double[number_of_bins_in_1d_spectra];
			ComputeImagesWithNumberOfExtremaAndCTFValues(&current_ctf, number_of_extrema_image, ctf_values_image);
			//number_of_extrema_image.QuickAndDirtyWriteSlice("dbg_num_extrema.mrc",1);
			//ctf_values_image.QuickAndDirtyWriteSlice("dbg_ctf_values.mrc",1);
			ComputeRotationalAverageOfPowerSpectrum(average_spectrum, &current_ctf, number_of_extrema_image, ctf_values_image, number_of_bins_in_1d_spectra, spatial_frequency, rotational_average_astig, rotational_average_astig_fit, number_of_extrema_profile, ctf_values_profile);

			// Here, do FRC
			ComputeFRCBetween1DSpectrumAndFit(number_of_bins_in_1d_spectra,rotational_average_astig,rotational_average_astig_fit,number_of_extrema_profile,fit_frc,fit_frc_sigma);

			// At what bin does CTF aliasing become problematic?
			last_bin_without_aliasing = 0;
			int location_of_previous_extremum = 0;
			for (counter=1;counter<number_of_bins_in_1d_spectra;counter++)
			{
				if (number_of_extrema_profile[counter]-number_of_extrema_profile[counter-1] >= 0.9)
				{
					// We just reached a new extremum
					if (counter-location_of_previous_extremum < 4)
					{
						last_bin_without_aliasing = location_of_previous_extremum;
						break;
					}
					location_of_previous_extremum = counter;
				}
			}
			if (is_running_locally && old_school_input && last_bin_without_aliasing != 0)
			{
				wxPrintf("CTF aliasing apparent from %0.1f Angstroms",pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]);
			}
		}

		//average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_diag_2.mrc",1);

		// Until what frequency were CTF rings detected?
		if (compute_extra_stats)
		{
			static float low_threshold = 0.2;
			static float frc_significance_threshold = 0.5; // In analogy to the usual criterion when comparing experimental results to the atomic model
			bool at_last_bin_with_good_fit;
			int number_of_bins_above_low_threshold = 0;
			int number_of_bins_above_significance_threshold = 0;
			int first_bin_to_check = int(sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(1,0.0))*average_spectrum->logical_x_dimension);
			//wxPrintf("Will only check from bin %i of %i onwards\n", first_bin_to_check, number_of_bins_in_1d_spectra);
			last_bin_with_good_fit = -1;
			for (counter=first_bin_to_check;counter<number_of_bins_in_1d_spectra;counter++)
			{
				//wxPrintf("On bin %i, fit_frc = %f, rot averate astig = %f\n", counter, fit_frc[counter], rotational_average_astig[counter]);
				at_last_bin_with_good_fit = ((number_of_bins_above_low_threshold > 3) && (fit_frc[counter] < low_threshold))
											||
											((number_of_bins_above_significance_threshold > 3) &&   ( fit_frc[counter] < frc_significance_threshold) ); // && (rotational_average_astig[counter] > 2.0) )
																									//||( fit_frc[counter] < frc_significance_threshold) // && (rotational_average_astig[counter] < -2.0))
																								//	)
																								//	);
				if (at_last_bin_with_good_fit)
				{
					last_bin_with_good_fit = counter;
					break;
				}
				// Count number of bins above given thresholds
				if (fit_frc[counter] > low_threshold) number_of_bins_above_low_threshold++;
				if (fit_frc[counter] > frc_significance_threshold) number_of_bins_above_significance_threshold++;
			}
			//wxPrintf("%i bins out of %i checked were above significance threshold\n",number_of_bins_above_significance_threshold,number_of_bins_in_1d_spectra-first_bin_to_check);
			if ( number_of_bins_above_significance_threshold == number_of_bins_in_1d_spectra-first_bin_to_check) last_bin_with_good_fit = number_of_bins_in_1d_spectra - 1;
			last_bin_with_good_fit = std::min(last_bin_with_good_fit,number_of_bins_in_1d_spectra);
		}
		else
		{
			last_bin_with_good_fit = 0;
		}

		#ifdef DEBUG
		MyDebugAssertTrue(last_bin_with_good_fit >= 0 && last_bin_with_good_fit < number_of_bins_in_1d_spectra,"Did not find last bin with good fit: %i", last_bin_with_good_fit);
		#else
		if (last_bin_with_good_fit < 0 && last_bin_with_good_fit >= number_of_bins_in_1d_spectra)
		{
			last_bin_with_good_fit = 0;
		}
		#endif

		// Prepare output diagnostic image
		//average_spectrum->AddConstant(- average_spectrum->ReturnAverageOfRealValuesOnEdges()); // this used to be done in OverlayCTF / CTFOperation in the Fortran code
		//average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_diag_3.mrc",1);
		//average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_rescaling.mrc",1);
		if (compute_extra_stats) {
			RescaleSpectrumAndRotationalAverage(average_spectrum,number_of_extrema_image,ctf_values_image,number_of_bins_in_1d_spectra,spatial_frequency,rotational_average_astig,rotational_average_astig_fit,number_of_extrema_profile,ctf_values_profile,last_bin_without_aliasing,last_bin_with_good_fit);
		}
		//average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_thresholding.mrc",1);
		average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(	sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(2,0.0))*average_spectrum->logical_x_dimension,
																	std::max(current_ctf.GetHighestFrequencyForFitting(),sqrtf(current_ctf.ReturnSquaredSpatialFrequencyOfAZero(3,0.0)))*average_spectrum->logical_x_dimension,
																	average,sigma);
		average_spectrum->SetMinimumAndMaximumValues(average - sigma, average + 2.0 * sigma );

		//average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_overlay.mrc",1);
		OverlayCTF(average_spectrum, &current_ctf);
		average_spectrum->WriteSlice(&output_diagnostic_file,current_output_location);


		// Print more detailed results to terminal
		if (is_running_locally && number_of_micrographs == 1)
		{
			wxPrintf("Estimated defocus values        : %0.2f , %0.2f Angstroms\nEstimated azimuth of astigmatism: %0.2f degrees\n",current_ctf.GetDefocus1()*pixel_size_for_fitting,current_ctf.GetDefocus2()*pixel_size_for_fitting,current_ctf.GetAstigmatismAzimuth() / PI * 180.0);
			if (find_additional_phase_shift)
			{
				wxPrintf("Additional phase shift          : %0.3f degrees (%0.3f radians) (%0.3f pi)\n",current_ctf.GetAdditionalPhaseShift() / PI * 180.0, current_ctf.GetAdditionalPhaseShift(),current_ctf.GetAdditionalPhaseShift() / PI);
			}
			wxPrintf("Score                           : %0.5f\n", - conjugate_gradient_minimizer->GetBestScore());
			wxPrintf("Pixel size for fitting          : %0.3f Angstroms\n",pixel_size_for_fitting);
			if (compute_extra_stats)
			{
				wxPrintf("Thon rings with good fit up to  : %0.1f Angstroms\n",pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit]);
				if (last_bin_without_aliasing != 0)
				{
					wxPrintf("CTF aliasing apparent from      : %0.1f Angstroms\n", pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]);
				}
				else
				{
					wxPrintf("Did not detect CTF aliasing\n");
				}
			}
		}

		// Warn the user if significant aliasing occured within the fit range
		if (compute_extra_stats && last_bin_without_aliasing != 0 && spatial_frequency[last_bin_without_aliasing] < current_ctf.GetHighestFrequencyForFitting())
		{
			if (is_running_locally && number_of_micrographs == 1)
			{
				MyPrintfRed("Warning: CTF aliasing occurred within your CTF fitting range. Consider computing a larger spectrum (current size = %i).\n",box_size);
			}
			else
			{
				SendInfo(wxString::Format("Warning: for image %s (location %i of %i), CTF aliasing occurred within the CTF fitting range. Consider computing a larger spectrum (current size = %i)\n",input_filename,box_size, current_micrograph_number, number_of_micrographs));
			}
		}


		if (is_running_locally)
		{
			// Write out results to a summary file
			values_to_write_out[0] = current_micrograph_number;
			values_to_write_out[1] = current_ctf.GetDefocus1() * pixel_size_for_fitting;
			values_to_write_out[2] = current_ctf.GetDefocus2() * pixel_size_for_fitting;
			values_to_write_out[3] = current_ctf.GetAstigmatismAzimuth() * 180.0 / PI;
			values_to_write_out[4] = current_ctf.GetAdditionalPhaseShift();
			values_to_write_out[5] = - conjugate_gradient_minimizer->GetBestScore();
			if (compute_extra_stats)
			{
				values_to_write_out[6] = pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit];
			}
			else
			{
				values_to_write_out[6] = 0.0;
			}
			output_text->WriteLine(values_to_write_out);


			if ( (!old_school_input) && number_of_micrographs > 1 && is_running_locally) my_progress_bar->Update(current_micrograph_number);
		}

		// Write out avrot
		// TODO: add to the output a line with non-normalized avrot, so that users can check for things like ice crystal reflections
		if (compute_extra_stats)
		{
			if (current_micrograph_number == 1)
			{
				output_text_avrot = new NumericTextFile(output_text_fn,OPEN_TO_WRITE,number_of_bins_in_1d_spectra);
				output_text_avrot->WriteCommentLine("# Output from CTFFind version %s, run on %s\n",ctffind_version.c_str(),wxDateTime::Now().FormatISOCombined(' ').ToUTF8().data());
				output_text_avrot->WriteCommentLine("# Input file: %s ; Number of micrographs: %i\n",input_filename.c_str(),number_of_micrographs);
				output_text_avrot->WriteCommentLine("# Pixel size: %0.3f Angstroms ; acceleration voltage: %0.1f keV ; spherical aberration: %0.1f mm ; amplitude contrast: %0.2f\n",pixel_size_of_input_image,acceleration_voltage,spherical_aberration,amplitude_contrast);
				output_text_avrot->WriteCommentLine("# Box size: %i pixels ; min. res.: %0.1f Angstroms ; max. res.: %0.1f Angstroms ; min. def.: %0.1f um; max. def. %0.1f um; num. frames averaged: %i\n",box_size,minimum_resolution,maximum_resolution,minimum_defocus,maximum_defocus,number_of_frames_to_average);
				output_text_avrot->WriteCommentLine("# 6 lines per micrograph: #1 - spatial frequency (1/Angstroms); #2 - 1D rotational average of spectrum (assuming no astigmatism); #3 - 1D rotational average of spectrum; #4 - CTF fit; #5 - cross-correlation between spectrum and CTF fit; #6 - 2sigma of expected cross correlation of noise\n");
			}
			spatial_frequency_in_reciprocal_angstroms = new double[number_of_bins_in_1d_spectra];
			for (counter=0; counter<number_of_bins_in_1d_spectra;counter++)
			{
				spatial_frequency_in_reciprocal_angstroms[counter] = spatial_frequency[counter] / pixel_size_for_fitting;
			}
			output_text_avrot->WriteLine(spatial_frequency_in_reciprocal_angstroms);
			output_text_avrot->WriteLine(rotational_average.data_y);
			output_text_avrot->WriteLine(rotational_average_astig);
			output_text_avrot->WriteLine(rotational_average_astig_fit);
			output_text_avrot->WriteLine(fit_frc);
			output_text_avrot->WriteLine(fit_frc_sigma);
			delete [] spatial_frequency_in_reciprocal_angstroms;
		}

		delete comparison_object_2D;

	} // End of loop over micrographs

	if (is_running_locally && (! old_school_input) && number_of_micrographs > 1) {
		delete my_progress_bar;
		wxPrintf("\n");
	}

	// Tell the user where the outputs are
	if (is_running_locally)
	{
		wxPrintf("\nSummary of results                          : %s\n", output_text->ReturnFilename());
		wxPrintf("Diagnostic images                           : %s\n", output_diagnostic_filename);
		if (compute_extra_stats)
		{
			wxPrintf("Detailed results, including 1D fit profiles : %s\n",output_text_avrot->ReturnFilename());
			wxPrintf("Use this command to plot 1D fit profiles    : ctffind_plot_results.sh %s\n",output_text_avrot->ReturnFilename());
		}

		wxPrintf("\n\n");
	}


	// Send results back
	float results_array[7];
	results_array[0] = current_ctf.GetDefocus1() * pixel_size_for_fitting;				// Defocus 1 (Angstroms)
	results_array[1] = current_ctf.GetDefocus2() * pixel_size_for_fitting;				// Defocus 2 (Angstroms)
	results_array[2] = current_ctf.GetAstigmatismAzimuth() * 180.0 / PI;	// Astigmatism angle (degrees)
	results_array[3] = current_ctf.GetAdditionalPhaseShift();				// Additional phase shift (e.g. from phase plate) (radians)
	results_array[4] = - conjugate_gradient_minimizer->GetBestScore();		// CTFFIND score
	if (last_bin_with_good_fit == 0)
	{
		results_array[5] = 0.0;															//	A value of 0.0 indicates that the calculation to determine the goodness of fit failed for some reason
	}
	else
	{
		results_array[5] = pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit];		//	The resolution (Angstroms) up to which Thon rings are well fit by the CTF
	}
	if (last_bin_without_aliasing == 0)
	{
		results_array[6] = 0.0;															// 	A value of 0.0 indicates that no aliasing was detected
	}
	else
	{
		results_array[6] = pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]; 	//	The resolution (Angstroms) at which aliasing was just detected
	}

	my_result.SetResult(7,results_array);


	// Cleanup
	delete average_spectrum;
	delete current_power_spectrum;
	delete current_input_image;
	delete current_input_image_square;
	delete temp_image;
	delete sum_image;
	delete resampled_power_spectrum;
	delete number_of_extrema_image;
	delete ctf_values_image;
	delete [] values_to_write_out;
	if (is_running_locally) delete output_text;
	if (compute_extra_stats)
	{
		delete [] spatial_frequency;
		delete [] rotational_average_astig;
		delete [] rotational_average_astig_fit;
		delete [] number_of_extrema_profile;
		delete [] ctf_values_profile;
		delete [] fit_frc;
		delete [] fit_frc_sigma;
		delete output_text_avrot;
	}
	delete conjugate_gradient_minimizer;



	// Return
	return true;
}

//
void ComputeFRCBetween1DSpectrumAndFit( int number_of_bins, double average[], double fit[], float number_of_extrema_profile[], double frc[], double frc_sigma[])
{
	int bin_counter;
	int half_window_width[number_of_bins];
	int bin_of_previous_extremum;
	int i;
	int first_bin, last_bin;
	double spectrum_mean, fit_mean;
	double spectrum_sigma, fit_sigma;
	double cross_product;
	float number_of_bins_in_window;

	const int minimum_window_half_width = number_of_bins / 40;

	// First, work out the size of the window over which we'll compute the FRC value
	bin_of_previous_extremum = 0;
	for (bin_counter=1; bin_counter < number_of_bins; bin_counter++)
	{
		if (number_of_extrema_profile[bin_counter] != number_of_extrema_profile[bin_counter-1])
		{
			for (i=bin_of_previous_extremum;i<bin_counter;i++)
			{
				half_window_width[i] = std::max(minimum_window_half_width,int(1.5 * float(bin_counter - bin_of_previous_extremum + 1)));
				half_window_width[i] = std::min(half_window_width[i],number_of_bins/2 - 1);
				MyDebugAssertTrue(half_window_width[i] < number_of_bins/2,"Bad half window width: %i. Number of bins: %i\n",half_window_width[i],number_of_bins);
			}
			bin_of_previous_extremum = bin_counter;
		}
	}
	half_window_width[0] = half_window_width[1];
	for (bin_counter=bin_of_previous_extremum; bin_counter < number_of_bins; bin_counter++)
	{
		half_window_width[bin_counter] = half_window_width[bin_of_previous_extremum-1];
	}

	// Now compute the FRC for each bin
	for (bin_counter=0; bin_counter < number_of_bins; bin_counter++)
	{
		spectrum_mean = 0.0;
		fit_mean = 0.0;
		spectrum_sigma = 0.0;
		fit_sigma = 0.0;
		cross_product = 0.0;
		// Work out the boundaries
		first_bin = bin_counter - half_window_width[bin_counter];
		last_bin = bin_counter + half_window_width[bin_counter];
		if (first_bin < 0)
		{
			first_bin = 0;
			last_bin = 2 * half_window_width[bin_counter] + 1;
		}
		if (last_bin >= number_of_bins)
		{
			last_bin = number_of_bins - 1;
			first_bin = last_bin - 2 * half_window_width[bin_counter] - 1;
		}
		MyDebugAssertTrue(first_bin >=0 && first_bin < number_of_bins,"Bad first_bin: %i",first_bin);
		MyDebugAssertTrue(last_bin >=0 && last_bin < number_of_bins,"Bad last_bin: %i",last_bin);
		// First pass
		for (i=first_bin;i<=last_bin;i++)
		{
			spectrum_mean += average[i];
			fit_mean += fit[i];
		}
		number_of_bins_in_window = float(2 * half_window_width[bin_counter] + 1);
		spectrum_mean /= number_of_bins_in_window;
		fit_mean      /= number_of_bins_in_window;
		// Second pass
		for (i=first_bin;i<=last_bin;i++)
		{
			cross_product += (average[i] - spectrum_mean) * (fit[i] - fit_mean);
			spectrum_sigma += pow(average[i] - spectrum_mean,2);
			fit_sigma += pow(fit[i] - fit_mean,2);
		}
		MyDebugAssertTrue(spectrum_sigma > 0.0 && spectrum_sigma < 10000.0,"Bad spectrum_sigma: %f\n",spectrum_sigma);
		MyDebugAssertTrue(fit_sigma > 0.0 && fit_sigma < 10000.0,"Bad fit sigma: %f\n",fit_sigma);
		if (spectrum_sigma > 0.0 && fit_sigma > 0.0)
		{
			frc[bin_counter] = cross_product / (sqrtf(spectrum_sigma/number_of_bins_in_window) * sqrtf(fit_sigma/number_of_bins_in_window)) / number_of_bins_in_window;
		}
		else
		{
			frc[bin_counter] = 0.0;
		}
		frc_sigma[bin_counter] = 2.0 / sqrtf(number_of_bins_in_window);
		MyDebugAssertTrue(frc[bin_counter] >= -1.0 && frc[bin_counter] <= 1.0, "Bad FRC value: %f", frc[bin_counter]);
	}

}



//
void OverlayCTF( Image *spectrum, CTF *ctf)
{
	MyDebugAssertTrue(spectrum->is_in_memory, "Spectrum memory not allocated");

	//
	EmpiricalDistribution values_in_rings;
	EmpiricalDistribution values_in_fitting_range;
	int i;
	int j;
	long address;
	float i_logi, i_logi_sq;
	float j_logi, j_logi_sq;
	float current_spatial_frequency_squared;
	float current_azimuth;
	const float lowest_freq  = pow(ctf->GetLowestFrequencyForFitting(),2);
	const float highest_freq = pow(ctf->GetHighestFrequencyForFitting(),2);
	float current_ctf_value;
	float target_sigma;

	//spectrum->QuickAndDirtyWriteSlice("dbg_spec_overlay_entry.mrc",1);

	//
	address = 0;
	for (j=0;j < spectrum->logical_y_dimension;j++)
	{
		j_logi = float(j-spectrum->physical_address_of_box_center_y) * spectrum->fourier_voxel_size_y;
		j_logi_sq = powf(j_logi,2);
		for (i=0 ;i < spectrum->logical_x_dimension; i++)
		{
			i_logi = float(i-spectrum->physical_address_of_box_center_x) * spectrum->fourier_voxel_size_x;
			i_logi_sq = powf(i_logi,2);
			//
			current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
			//
			if (current_spatial_frequency_squared > lowest_freq && current_spatial_frequency_squared <= highest_freq)
			{
				current_azimuth = atan2(j_logi,i_logi);
				current_ctf_value = fabs(ctf->Evaluate(current_spatial_frequency_squared,current_azimuth));
				if (current_ctf_value > 0.5) values_in_rings.AddSampleValue(spectrum->real_values[address]);
				values_in_fitting_range.AddSampleValue(spectrum->real_values[address]);
				if (j < spectrum->physical_address_of_box_center_y && i < spectrum->physical_address_of_box_center_x) spectrum->real_values[address] = current_ctf_value;
			}
			if (current_spatial_frequency_squared <= lowest_freq)
			{
				spectrum->real_values[address] = 0.0;
			}
			//
			address++;
		}
		address += spectrum->padding_jump_value;
	}

	//spectrum->QuickAndDirtyWriteSlice("dbg_spec_overlay_1.mrc",1);

	/*

	// We will renormalize the experimental part of the diagnostic image
	target_sigma = sqrtf(values_in_rings.GetSampleVariance()) ;


	if (target_sigma > 0.0)
	{
		address = 0;
		for (j=0;j < spectrum->logical_y_dimension;j++)
		{
			j_logi = float(j-spectrum->physical_address_of_box_center_y) * spectrum->fourier_voxel_size_y;
			j_logi_sq = powf(j_logi,2);
			for (i=0 ;i < spectrum->logical_x_dimension; i++)
			{
				i_logi = float(i-spectrum->physical_address_of_box_center_x) * spectrum->fourier_voxel_size_x;
				i_logi_sq = powf(i_logi,2);
				//
				current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
				// Normalize the experimental part of the diagnostic image
				if (i > spectrum->physical_address_of_box_center_x || j > spectrum->physical_address_of_box_center_y)
				{
					spectrum->real_values[address] /= target_sigma;
				}
				else
				{
					// Normalize the outside of the theoretical part of the diagnostic image
					if (current_spatial_frequency_squared > highest_freq) spectrum->real_values[address] /= target_sigma;
				}

				address++;
			}
			address += spectrum->padding_jump_value;
		}
	}
	*/

	//spectrum->QuickAndDirtyWriteSlice("dbg_spec_overlay_final.mrc",1);
}


// Rescale the spectrum and its 1D rotational avereage so that the peaks and troughs are at 0.0 and 1.0. The location of peaks and troughs are worked out
// by parsing the suppilied 1D average_fit array
void RescaleSpectrumAndRotationalAverage( Image *spectrum, Image *number_of_extrema, Image *ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[], int last_bin_without_aliasing, int last_bin_with_good_fit )
{
	MyDebugAssertTrue(spectrum->is_in_memory, "Spectrum memory not allocated");
	MyDebugAssertTrue(number_of_bins > 1,"Bad number of bins: %i\n",number_of_bins);

	//
	const bool spectrum_is_blank = spectrum->IsConstant();
	const int rescale_based_on_maximum_number = 2; // This peak will be used as a renormalization.
	const int sg_width = 7;
	const int sg_order = 2;
	const bool rescale_peaks = false; // if this is false, only the background will be subtracted, the Thon rings "heights" will be unaffected
	float background[number_of_bins];
	float peak[number_of_bins];
	int bin_counter;
	bool at_a_maximum, at_a_minimum, maximum_at_previous_bin, minimum_at_previous_bin;
	int location_of_previous_maximum, location_of_previous_minimum;
	int current_maximum_number = 0;
	int normalisation_bin_number;
	int i;
	int j;
	bool actually_do_rescaling;
	int chosen_bin;
	long address;
	int last_bin_to_rescale;
	float min_scale_factor;
	float scale_factor;
	float rescale_peaks_to;

	Curve *minima_curve = new Curve;
	Curve *maxima_curve = new Curve;

	// Initialise arrays and variables
	for (bin_counter=0; bin_counter < number_of_bins; bin_counter++)
	{
		background[bin_counter] = 0.0;
		peak[bin_counter] = 0.0;
	}
	location_of_previous_maximum = 0;
	location_of_previous_minimum = 0;
	current_maximum_number = 0;
	at_a_maximum = false;
	at_a_minimum = true; // Note, this may not be true if we have the perfect phase plate

	//
	if ( ! spectrum_is_blank )
	{
		for (bin_counter=1; bin_counter < number_of_bins - 1; bin_counter ++)
		{
			// Remember where we were before - minimum, maximum or neither
			maximum_at_previous_bin = at_a_maximum;
			minimum_at_previous_bin = at_a_minimum;
			// Are we at a CTF min or max?
			at_a_minimum = (average_fit[bin_counter] <= average_fit[bin_counter-1]) && (average_fit[bin_counter] <= average_fit[bin_counter+1]);
			at_a_maximum = (average_fit[bin_counter] >= average_fit[bin_counter-1]) && (average_fit[bin_counter] >= average_fit[bin_counter+1]);
			// It could be that the CTF is constant in this region, in which case we stay at a minimum if we were there
			if (at_a_maximum && at_a_minimum)
			{
				at_a_minimum = minimum_at_previous_bin;
				at_a_maximum = maximum_at_previous_bin;
			}
			// Fill in values for the background or peak by linear interpolation
			if (at_a_minimum)
			{
				for (i=location_of_previous_minimum+1;i<=bin_counter;i++)
				{
					// Linear interpolation of average values at the peaks and troughs of the CTF
					background[i] = average[location_of_previous_minimum] * float(bin_counter-i) / float(bin_counter-location_of_previous_minimum) + average[bin_counter] * float(i-location_of_previous_minimum) / float(bin_counter-location_of_previous_minimum);
				}
				location_of_previous_minimum = bin_counter;
				minima_curve->AddPoint(spatial_frequency[bin_counter],average[bin_counter]);
			}
			if (at_a_maximum)
			{
				if ((! maximum_at_previous_bin) && (average_fit[bin_counter] > 0.7)) current_maximum_number = current_maximum_number + 1;
				for (i=location_of_previous_maximum+1;i<=bin_counter;i++)
				{
					// Linear interpolation of average values at the peaks and troughs of the CTF
					peak[i]       = average[location_of_previous_maximum] * float(bin_counter-i) / float(bin_counter-location_of_previous_maximum) + average[bin_counter] * float(i-location_of_previous_maximum) / float(bin_counter-location_of_previous_maximum);
					//
					if (current_maximum_number == rescale_based_on_maximum_number) normalisation_bin_number = bin_counter;
				}
				location_of_previous_maximum = bin_counter;
				maxima_curve->AddPoint(spatial_frequency[bin_counter],average[bin_counter]);
			}
			if (at_a_maximum && at_a_minimum)
			{
				MyPrintfRed("Rescale spectrum: Error. At a minimum and a maximum simultaneously.");
				abort();
			}
		}

		// Fit the minima and maximum curves using Savitzky-Golay smoothing
		if (maxima_curve->number_of_points >= sg_width) maxima_curve->FitSavitzkyGolayToData(sg_width, sg_order);
		if (minima_curve->number_of_points >= sg_width) minima_curve->FitSavitzkyGolayToData(sg_width, sg_order);

		// Replace the background and peak envelopes with the smooth min/max curves
		for (bin_counter=0;bin_counter<number_of_bins;bin_counter++)
		{
			if (minima_curve->number_of_points >= sg_width) background[bin_counter] =  minima_curve->ReturnSavitzkyGolayInterpolationFromX(spatial_frequency[bin_counter]);
			if (maxima_curve->number_of_points >= sg_width) peak[bin_counter]       =  maxima_curve->ReturnSavitzkyGolayInterpolationFromX(spatial_frequency[bin_counter]);
		}

		// Now that we have worked out a background and a peak envelope, let's do the actual rescaling
		actually_do_rescaling = (peak[normalisation_bin_number] - background[normalisation_bin_number]) > 0.0;
		if (last_bin_without_aliasing != 0)
		{
			last_bin_to_rescale = std::min(last_bin_with_good_fit,last_bin_without_aliasing);
		}
		else
		{
			last_bin_to_rescale = last_bin_with_good_fit;
		}
		if (actually_do_rescaling)
		{
			min_scale_factor = 0.2;
			rescale_peaks_to = 0.75;
			address = 0;
			for (j=0;j<spectrum->logical_y_dimension;j++)
			{
				for (i=0;i<spectrum->logical_x_dimension;i++)
				{
					chosen_bin = ReturnSpectrumBinNumber(number_of_bins,number_of_extrema_profile,number_of_extrema, address, ctf_values, ctf_values_profile);
					if (chosen_bin <= last_bin_to_rescale)
					{
						spectrum->real_values[address] -= background[chosen_bin]; // This alone makes the spectrum look very nice already
						if (rescale_peaks) spectrum->real_values[address] /= std::min(1.0f,std::max(min_scale_factor,peak[chosen_bin]-background[chosen_bin])) / rescale_peaks_to; // This is supposed to help "boost" weak Thon rings
					}
					else
					{
						spectrum->real_values[address] -= background[last_bin_to_rescale];
						if (rescale_peaks) spectrum->real_values[address] /= std::min(1.0f,std::max(min_scale_factor,peak[last_bin_to_rescale]-background[last_bin_to_rescale])) / rescale_peaks_to;
					}
					//
					address++;
				}
				address += spectrum->padding_jump_value;
			}
		}
		else
		{
			MyDebugPrint("(RescaleSpectrumAndRotationalAverage) Warning: bad peak/background detection");
		}

		// Rescale the 1D average
		if (peak[normalisation_bin_number] > background[normalisation_bin_number])
		{
			for (bin_counter=0;bin_counter<number_of_bins;bin_counter++)
			{

				average[bin_counter] = (average[bin_counter] - background[bin_counter]) / (peak[normalisation_bin_number] - background[normalisation_bin_number]) * 0.95;
				// We want peaks to reach at least 0.1
				if ( ((peak[bin_counter] - background[bin_counter]) < 0.1) && (fabs(peak[bin_counter]-background[bin_counter]) > 0.000001) && bin_counter <= last_bin_without_aliasing)
				{
					average[bin_counter] = average[bin_counter] / (peak[bin_counter]-background[bin_counter]) * ( peak[normalisation_bin_number] - background[normalisation_bin_number] ) * 0.1;
				}
			}
		}
		else
		{
			MyDebugPrint("(RescaleSpectrumAndRotationalAverage): unable to rescale 1D average experimental spectrum\n");
		}


	} // end of test of spectrum_is_blank

	// Cleanup
	delete minima_curve;
	delete maxima_curve;

}

//
void ComputeRotationalAverageOfPowerSpectrum( Image *spectrum, CTF *ctf, Image *number_of_extrema, Image *ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[])
{
	MyDebugAssertTrue(spectrum->is_in_memory, "Spectrum memory not allocated");
	MyDebugAssertTrue(number_of_extrema->is_in_memory,"Number of extrema image not allocated");
	MyDebugAssertTrue(ctf_values->is_in_memory,"CTF values image not allocated");
	MyDebugAssertTrue(spectrum->HasSameDimensionsAs(number_of_extrema),"Spectrum and number of extrema images do not have same dimensions");
	MyDebugAssertTrue(spectrum->HasSameDimensionsAs(ctf_values),"Spectrum and CTF values images do not have same dimensions");
	//
	const bool spectrum_is_blank = spectrum->IsConstant();
	const float min_angular_distances_from_axes_radians = 10.0 / 180.0 * PI;
	int counter;
	float azimuth_of_mid_defocus;
	float angular_distance_from_axes;
	float current_spatial_frequency_squared;
	int number_of_values[number_of_bins];
	int i, j;
	long address;
	float ctf_diff_from_current_bin;
	int chosen_bin;

	// Initialise the output arrays
	for (counter=0; counter<number_of_bins; counter++)
	{
		average[counter] = 0.0;
		average_fit[counter] = 0.0;
		ctf_values_profile[counter] = 0.0;
		number_of_values[counter] = 0;
	}

	//
	if (! spectrum_is_blank)
	{
		// For each bin of our 1D profile we compute the CTF. We choose the azimuth to be mid way between the two defoci of the astigmatic CTF
		azimuth_of_mid_defocus = ctf->GetAstigmatismAzimuth() + PI * 0.25;
		// We don't want the azimuth too close to the axes, which may have been blanked by the central-cross-artefact-suppression-system (tm)
		angular_distance_from_axes = fmod(azimuth_of_mid_defocus,PI * 0.5);
		if(fabs(angular_distance_from_axes) < min_angular_distances_from_axes_radians)
		{
			if (angular_distance_from_axes > 0.0)
			{
				azimuth_of_mid_defocus = min_angular_distances_from_axes_radians;
			}
			else
			{
				azimuth_of_mid_defocus = - min_angular_distances_from_axes_radians;
			}
		}
		if (fabs(angular_distance_from_axes) > 0.5 * PI - min_angular_distances_from_axes_radians)
		{
			if (angular_distance_from_axes > 0.0)
			{
				azimuth_of_mid_defocus = PI * 0.5 - min_angular_distances_from_axes_radians;
			}
			else
			{
				azimuth_of_mid_defocus = - PI * 0.5 + min_angular_distances_from_axes_radians;
			}
		}
		// Now that we've chosen an azimuth, we can compute the CTF for each bin of our 1D profile
		for (counter=0;counter < number_of_bins; counter++)
		{
			current_spatial_frequency_squared = powf(float(counter) * spectrum->fourier_voxel_size_y, 2);
			spatial_frequency[counter] = sqrt(current_spatial_frequency_squared);
			ctf_values_profile[counter] = ctf->Evaluate(current_spatial_frequency_squared,azimuth_of_mid_defocus);
			number_of_extrema_profile[counter] = ctf->ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(current_spatial_frequency_squared,azimuth_of_mid_defocus);
		}

		// Now we can loop over the spectrum again and decide to which bin to add each component
		address = 0;
		for (j=0; j<spectrum->logical_y_dimension; j++)
		{
			for (i=0; i < spectrum->logical_x_dimension; i++)
			{
				ctf_diff_from_current_bin = std::numeric_limits<float>::max();
				chosen_bin = ReturnSpectrumBinNumber(number_of_bins,number_of_extrema_profile,number_of_extrema, address, ctf_values, ctf_values_profile);
				average[chosen_bin] += spectrum->real_values[address];
				number_of_values[chosen_bin]++;
				//
				address++;
			}
			address += spectrum->padding_jump_value;
		}

		// Do the actual averaging
		for (counter = 0; counter < number_of_bins; counter++)
		{
			if (number_of_values[counter] > 0)
			{
				average[counter] = average[counter] / float(number_of_values[counter]);
			}
			else
			{
				average[counter] = 0.0;
			}
			average_fit[counter] = fabs(ctf_values_profile[counter]);
		}

	}
}


int ReturnSpectrumBinNumber(int number_of_bins, float number_of_extrema_profile[], Image *number_of_extrema, long address, Image *ctf_values, float ctf_values_profile[])
{
	int current_bin;
	float diff_number_of_extrema;
	float diff_number_of_extrema_previous;
	float diff_number_of_extrema_next;
	float ctf_diff_from_current_bin;
	float ctf_diff_from_current_bin_old;
	int chosen_bin;
	//
	//MyDebugPrint("address: %li - number of extrema: %f - ctf_value: %f\n", address, number_of_extrema->real_values[address], ctf_values->real_values[address]);
	MyDebugAssertTrue(address < number_of_extrema->real_memory_allocated,"Oops, bad address: %li\n",address);
	// Let's find the bin which has the same number of preceding extrema and the most similar ctf value
	ctf_diff_from_current_bin = std::numeric_limits<float>::max();
	chosen_bin = -1;
	for (current_bin=0; current_bin < number_of_bins; current_bin++)
	{
		diff_number_of_extrema = fabs(number_of_extrema->real_values[address] - number_of_extrema_profile[current_bin]);
		if (current_bin > 0)
		{
			diff_number_of_extrema_previous = fabs(number_of_extrema->real_values[address]- number_of_extrema_profile[current_bin-1]);
		}
		else
		{
			diff_number_of_extrema_previous = std::numeric_limits<float>::max();
		}
		if (current_bin < number_of_bins - 1)
		{
			diff_number_of_extrema_next = fabs(number_of_extrema->real_values[address] - number_of_extrema_profile[current_bin+1]);
		}
		else
		{
			diff_number_of_extrema_next = std::numeric_limits<float>::max();
		}
		//
		if (number_of_extrema->real_values[address] > number_of_extrema_profile[number_of_bins-1])
		{
			chosen_bin = number_of_bins - 1;
		}
		else
		{
			if ( diff_number_of_extrema <= 0.01 || (  diff_number_of_extrema <  diff_number_of_extrema_previous &&
					                                  diff_number_of_extrema <= diff_number_of_extrema_next &&
													  number_of_extrema_profile[std::max(current_bin-1,0)] != number_of_extrema_profile[std::min(current_bin+1,number_of_bins-1)]  ) )
			{
				// We're nearly there
				// Let's look for the position for the nearest CTF value
				ctf_diff_from_current_bin_old = ctf_diff_from_current_bin;
				ctf_diff_from_current_bin = fabs(ctf_values->real_values[address] - ctf_values_profile[current_bin]);
				if (ctf_diff_from_current_bin < ctf_diff_from_current_bin_old)
				{
					//MyDebugPrint("new chosen bin: %i\n",current_bin);
					chosen_bin = current_bin;
				}
			}
		}
	}
	if (chosen_bin == -1)
	{
		MyPrintfRed("Could not find bin\n");
		abort();
	}
	else
	{
		//MyDebugAssertTrue(chosen_bin > 0 && chosen_bin < number_of_bins,"Oops, bad chosen bin number: %i (number of bins = %i)\n",chosen_bin,number_of_bins);
		//MyDebugPrint("final chosen bin = %i\n", chosen_bin);
		return chosen_bin;
	}
}
/*
integer function ComputePowerSpectrumBinNumber(number_of_bins,number_of_extrema_profile,number_of_extrema, &
                                                i,j,ctf_values,ctf_values_profile) result(chosen_bin)
    integer,        intent(in)  ::  number_of_bins
    real,           intent(in)  ::  number_of_extrema_profile(:)
    type(Image),    intent(in)  ::  number_of_extrema
    integer,        intent(in)  ::  i,j                         !<  Physical memory address
    type(Image),    intent(in)  ::  ctf_values
    real,           intent(in)  ::  ctf_values_profile(:)
    ! private variables
    integer     ::  current_bin
    real        ::  diff_number_of_extrema, diff_number_of_extrema_previous, diff_number_of_extrema_next
    real        ::  ctf_diff_from_current_bin
    real        ::  ctf_diff_from_current_bin_old
    ! Let's find the bin which has the same number of preceding extrema and the most similar ctf value
    ctf_diff_from_current_bin = huge(1.0e0)
    chosen_bin = 0
    do current_bin=1,number_of_bins
        diff_number_of_extrema  = abs(number_of_extrema%real_values(i,j,1) - number_of_extrema_profile(current_bin))
        if (current_bin .gt. 1) then
            diff_number_of_extrema_previous = abs(number_of_extrema%real_values(i,j,1) &
                                                - number_of_extrema_profile(current_bin-1))
        else
            diff_number_of_extrema_previous = huge(1.0e0)
        endif
        if (current_bin .lt. number_of_bins) then
            diff_number_of_extrema_next     = abs(number_of_extrema%real_values(i,j,1) &
                                                - number_of_extrema_profile(current_bin+1))
        else
            diff_number_of_extrema_next = huge(1.0e0)
        endif
        if (number_of_extrema%real_values(i,j,1) .gt. number_of_extrema_profile(number_of_bins)) then
            chosen_bin = number_of_bins
        else
            if (        diff_number_of_extrema .le. 0.01 &
                .or.    (     diff_number_of_extrema .lt. diff_number_of_extrema_previous &
                        .and. diff_number_of_extrema .le. diff_number_of_extrema_next &
                        .and. number_of_extrema_profile(max(current_bin-1,1)) &
                            .ne. number_of_extrema_profile(min(current_bin+1,number_of_bins))) &
                ) then
                ! We're nearly there
                ! Let's look for the position of the nearest CTF value
                ctf_diff_from_current_bin_old = ctf_diff_from_current_bin
                ctf_diff_from_current_bin = abs(ctf_values%real_values(i,j,1) - ctf_values_profile(current_bin))
                if (ctf_diff_from_current_bin .lt. ctf_diff_from_current_bin_old) then
                    chosen_bin = current_bin
                endif
            endif
        endif
    enddo
    if (chosen_bin .eq. 0) then
        print *, number_of_extrema_profile
        print *, i, j, number_of_extrema%real_values(i,j,1), ctf_values%real_values(i,j,1)
        print *, diff_number_of_extrema, diff_number_of_extrema_previous, diff_number_of_extrema_next
        call this_program%TerminateWithFatalError('ComputeRotationalAverageOfPowerSpectrum','Could not find bin')
    endif
end function ComputePowerSpectrumBinNumber
*/


// Compute an image where each pixel stores the number of preceding CTF extrema. This is described as image "E" in Rohou & Grigorieff 2015 (see Fig 3)
void ComputeImagesWithNumberOfExtremaAndCTFValues(CTF *ctf, Image *number_of_extrema, Image *ctf_values)
{
	MyDebugAssertTrue(number_of_extrema->is_in_memory,"Memory not allocated");
	MyDebugAssertTrue(ctf_values->is_in_memory,"Memory not allocated");
	MyDebugAssertTrue(ctf_values->HasSameDimensionsAs(number_of_extrema),"Images do not have same dimensions");

	int i, j;
	float i_logi, i_logi_sq;
	float j_logi, j_logi_sq;
	float current_spatial_frequency_squared;
	float current_azimuth;
	long address;

	address = 0;
	for (j=0;j<number_of_extrema->logical_y_dimension;j++)
	{
		j_logi = float(j - number_of_extrema->physical_address_of_box_center_y) * number_of_extrema->fourier_voxel_size_y;
		j_logi_sq = pow(j_logi,2);
		for (i=0;i<number_of_extrema->logical_x_dimension;i++)
		{
			i_logi = float(i - number_of_extrema->physical_address_of_box_center_x) * number_of_extrema->fourier_voxel_size_x;
			i_logi_sq = pow(i_logi,2);
			// Where are we?
			current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
			if (current_spatial_frequency_squared > 0.0)
			{
				current_azimuth = atan2(j_logi,i_logi);
			}
			else
			{
				current_azimuth = 0.0;
			}
			//
			ctf_values->real_values[address] = ctf->Evaluate(current_spatial_frequency_squared,current_azimuth);
			number_of_extrema->real_values[address] = ctf->ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(current_spatial_frequency_squared,current_azimuth);
			//
			address++;
		}
		address += number_of_extrema->padding_jump_value;
	}

	number_of_extrema->is_in_real_space = true;
	ctf_values->is_in_real_space = true;
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
	EmpiricalDistribution cc_numerator_dist;
	EmpiricalDistribution cc_denom_self_dist;
	EmpiricalDistribution cc_denom_other_dist;
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


