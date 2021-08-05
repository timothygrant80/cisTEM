#include "../../core/core_headers.h"

class
ApplyCTFApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

IMPLEMENT_APP(ApplyCTFApp)

// override the DoInteractiveUserInput

void ApplyCTFApp::DoInteractiveUserInput()
{
	std::string input_filename;
	std::string output_filename;
	std::string text_filename;
	float pixel_size;
	float acceleration_voltage;
	float spherical_aberration;
	float amplitude_contrast;
	float defocus_1 = 0;
	float defocus_2 = 0;
	float astigmatism_angle = 0;
	float additional_phase_shift = 0;

	bool input_ctf_values_from_text_file;
	bool phase_flip_only;
	bool apply_wiener_filter = false;

	float wiener_filter_falloff_frequency = 100.0;
	float wiener_filter_falloff_fudge_factor = 1.0;
	float wiener_filter_scale_fudge_factor = 1.0;
	float wiener_filter_high_pass_radius = 200.0;

	bool set_expert_options;

	UserInput *my_input = new UserInput("ApplyCTF", 1.0);

	input_filename = my_input->GetFilenameFromUser("Input image filename", "The input file, containing one or more images in a stack", "input.mrc", true );
	output_filename = my_input->GetFilenameFromUser("Output filename", "The output file", "output.mrc", false);
	pixel_size = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	acceleration_voltage = my_input->GetFloatFromUser("Acceleration voltage (keV)", "Acceleration voltage, in keV", "300.0", 0.0,500.0);
	spherical_aberration = my_input->GetFloatFromUser("Spherical aberration (mm)","Objective lens spherical aberration","2.7",0.0);
	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast","Fraction of total contrast attributed to amplitude contrast","0.07",0.0);

	input_ctf_values_from_text_file = my_input->GetYesNoFromUser("Use a text file to input defocus values?", "If yes, a text file with one line per image is required", "NO");

	if (input_ctf_values_from_text_file == true)
	{
		text_filename = my_input->GetFilenameFromUser("File containing defocus values", "should have 3 or 4 values per line", "my_defocus.txt", true);
	}
	else
	{
		defocus_1 = my_input->GetFloatFromUser("Underfocus 1 (A)","In Angstroms, the objective lens underfocus along the first axis","1.2");
		defocus_2 = my_input->GetFloatFromUser("Underfocus 2 (A)","In Angstroms, the objective lens underfocus along the second axis","1.2");
		astigmatism_angle = my_input->GetFloatFromUser("Astigmatism angle","Angle between the first axis and the x axis of the image","0.0");
		additional_phase_shift = my_input->GetFloatFromUser("Additional phase shift (rad)","Additional phase shift relative to undiffracted beam, as introduced for example by a phase plate","0.0");
	}

	phase_flip_only = my_input->GetYesNoFromUser("Phase Flip Only", "If Yes, only phase flipping is performed", "NO");

	if (phase_flip_only == false)
	{
		apply_wiener_filter = my_input->GetYesNoFromUser("Apply Wiener Filter", "If Yes, apply Wiener filter as suggested by Tegunov, et.al.","NO");
	}
	if (apply_wiener_filter == true)
	{
		wiener_filter_falloff_frequency = my_input->GetFloatFromUser("SSNR Falloff frequency","In Angstromsm, the frequency at which SSNR falls off","100.0");
		wiener_filter_falloff_fudge_factor = my_input->GetFloatFromUser("SSNR Falloff speed","How fast does SSNR fall off","1.0");
		wiener_filter_scale_fudge_factor = my_input->GetFloatFromUser("Deconvolution strength","Strength of the deconvolution","1.0");
		wiener_filter_high_pass_radius = my_input->GetFloatFromUser("Highpass filter frequency","In Angstromsm, the frequency at which to cutoff low freq signal","200.0");

	}


	delete my_input;

	my_current_job.Reset(10);
	my_current_job.ManualSetArguments("ttffffffffbtbbffff",     	 input_filename.c_str(),
															 output_filename.c_str(),
															 pixel_size,
															 acceleration_voltage,
															 spherical_aberration,
															 amplitude_contrast,
															 defocus_1,
															 defocus_2,
															 astigmatism_angle,
															 additional_phase_shift,
															 input_ctf_values_from_text_file,
															 text_filename.c_str(),
															 phase_flip_only,
															 apply_wiener_filter,
															 wiener_filter_falloff_frequency,
															 wiener_filter_falloff_fudge_factor,
															 wiener_filter_scale_fudge_factor,
															 wiener_filter_high_pass_radius
															 );


}

// override the do calculation method which will be what is actually run..

bool ApplyCTFApp::DoCalculation()
{
	long image_counter;

	Image current_image;
	CTF current_ctf;

	// get the arguments for this job..

	std::string input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	std::string output_filename 					= my_current_job.arguments[1].ReturnStringArgument();
	float       pixel_size					        = my_current_job.arguments[2].ReturnFloatArgument();
	float 		acceleration_voltage				= my_current_job.arguments[3].ReturnFloatArgument();
	float 		spherical_aberration				= my_current_job.arguments[4].ReturnFloatArgument();
	float 		amplitude_contrast   				= my_current_job.arguments[5].ReturnFloatArgument();
	float		defocus_1							= my_current_job.arguments[6].ReturnFloatArgument();
	float		defocus_2							= my_current_job.arguments[7].ReturnFloatArgument();
	float		astigmatism_angle					= my_current_job.arguments[8].ReturnFloatArgument();
	float		additional_phase_shift				= my_current_job.arguments[9].ReturnFloatArgument();
	bool        input_ctf_values_from_text_file     = my_current_job.arguments[10].ReturnBoolArgument();
	std::string text_filename                       = my_current_job.arguments[11].ReturnStringArgument();
	bool        phase_flip_only                     = my_current_job.arguments[12].ReturnBoolArgument();
	bool        apply_wiener_filter                 = my_current_job.arguments[13].ReturnBoolArgument();

	float 		wiener_filter_falloff_frequency 	= my_current_job.arguments[14].ReturnFloatArgument();
	float 		wiener_filter_falloff_fudge_factor 	= my_current_job.arguments[15].ReturnFloatArgument();
	float 		wiener_filter_scale_fudge_factor 	= my_current_job.arguments[16].ReturnFloatArgument();
	float 		wiener_filter_high_pass_radius 			= my_current_job.arguments[17].ReturnFloatArgument();	
	float temp_float[5];

	ProgressBar			*my_progress_bar;

	//my_current_job.PrintAllArguments();

	// The Files
	MRCFile input_file(input_filename, false);
	MRCFile output_file(output_filename, true);
	NumericTextFile *input_text;
	long number_of_input_images = input_file.ReturnNumberOfSlices();

	if (input_ctf_values_from_text_file == true)
	{
		input_text = new NumericTextFile(text_filename, OPEN_TO_READ);
		if (input_text->number_of_lines != number_of_input_images)
		{
			SendError("Error: Number of lines in defocus text file != number of images!");
			DEBUG_ABORT;
		}

		if (input_text->records_per_line != 3 && input_text->records_per_line != 4)
		{
			SendError("Error: Expect 3 or 4 records per line in defocus text file");
			DEBUG_ABORT;
		}
	}

	// CTF object
	current_ctf.Init(acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,0.0,0.5,0.0,pixel_size,additional_phase_shift);

	// Loop over input images

	wxPrintf("\nApplying CTF...\n\n");
	my_progress_bar = new ProgressBar(number_of_input_images);

	for ( image_counter = 0 ; image_counter < number_of_input_images; image_counter++)
	{
		current_image.ReadSlice(&input_file,image_counter + 1 );
		current_image.ForwardFFT();

		if (input_ctf_values_from_text_file == true)
		{
			input_text->ReadLine(temp_float);

			if (input_text->records_per_line == 3)
			{
				current_ctf.Init(acceleration_voltage,spherical_aberration,amplitude_contrast,temp_float[0],temp_float[1],temp_float[2],0.0,0.5,0.0,pixel_size,0.0);
			}
			else
			{
				current_ctf.Init(acceleration_voltage,spherical_aberration,amplitude_contrast,temp_float[0],temp_float[1],temp_float[2],0.0,0.5,0.0,pixel_size,temp_float[3]);
			}
		}

		if (phase_flip_only == true) {current_image.ApplyCTFPhaseFlip(current_ctf);}
		else if (apply_wiener_filter == true) {current_image.OptimalFilterWarp(current_ctf,
											   pixel_size,
											   wiener_filter_falloff_fudge_factor,
											   wiener_filter_scale_fudge_factor,
											   wiener_filter_falloff_frequency,
											   wiener_filter_high_pass_radius);}
		else {current_image.ApplyCTF(current_ctf);}
		//current_image.ZeroCentralPixel();
		current_image.BackwardFFT();
		if (apply_wiener_filter == false) {
			current_image.InvertRealValues();
		}
		current_image.WriteSlice(&output_file,image_counter + 1 );

		my_progress_bar->Update(image_counter + 1);
	}

	if (input_ctf_values_from_text_file == true) delete input_text;

	delete my_progress_bar;

	return true;
}




