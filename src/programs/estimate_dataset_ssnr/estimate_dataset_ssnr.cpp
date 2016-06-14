#include "../../core/core_headers.h"

class
EstimateDataSetSSNR : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(EstimateDataSetSSNR)

// override the DoInteractiveUserInput

void EstimateDataSetSSNR::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("Estimate Dataset SSNR", 1.0);

	std::string input_filename		=		my_input->GetFilenameFromUser("Input particle stack", "Filename of input particle stack to estimate", "input.mrc", true );
	std::string output_filename		=		my_input->GetFilenameFromUser("Output SSNR curve", "Filename of output SSNR curve", "output.txt", false );
	std::string defocus_filename    =       my_input->GetFilenameFromUser("Defocus Text File", "Text file with defocus values for each particle", "defocus.txt", true );
	float pixel_size                =       my_input->GetFloatFromUser("Pixel Size (A)", "The pixel size in Angstroms", "1.0", 0);
	float acceleration_voltage      =       my_input->GetFloatFromUser("Acceleration voltage (keV)", "Acceleration voltage, in keV", "300.0", 0.0,500.0);
	float spherical_aberration      =       my_input->GetFloatFromUser("Spherical aberration (mm)","Objective lens spherical aberration","2.7",0.0);
	float amplitude_contrast        =       my_input->GetFloatFromUser("Amplitude contrast","Fraction of total contrast attributed to amplitude contrast","0.07",0.0);
	int number_of_ctf_rotations     =       my_input->GetFloatFromUser("Number of samples for astigmatism", "number of directions to sample to take astigmatism into account", "18", 1);
	float molecular_mass_in_kda   =       my_input->GetFloatFromUser("Molecular Mass in kDa", "The molecular weight of the sample", "350", 1);



	delete my_input;

	my_current_job.Reset(9);
	my_current_job.ManualSetArguments("tttffffif", input_filename.c_str(), output_filename.c_str(), defocus_filename.c_str(), pixel_size, acceleration_voltage, spherical_aberration, amplitude_contrast, number_of_ctf_rotations, molecular_mass_in_kda);
}

// override the do calculation method which will be what is actually run..

bool EstimateDataSetSSNR::DoCalculation()
{

	std::string	input_filename 						= my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename 					= my_current_job.arguments[1].ReturnStringArgument();
	std::string	defocus_filename 					= my_current_job.arguments[2].ReturnStringArgument();
	float pixel_size                                = my_current_job.arguments[3].ReturnFloatArgument();
	float acceleration_voltage                      = my_current_job.arguments[4].ReturnFloatArgument();
	float spherical_aberration                      = my_current_job.arguments[5].ReturnFloatArgument();
	float amplitude_contrast                        = my_current_job.arguments[6].ReturnFloatArgument();
	int number_of_ctf_rotations                     = my_current_job.arguments[7].ReturnIntegerArgument();
	float molecular_mass_in_kda                     = my_current_job.arguments[8].ReturnFloatArgument();

	int counter;
	int rotation_counter;
	float temp_float[10];
	long number_of_ctfs=0;

	float fourier_voxel_size;
	float rotation_step = (PI*0.5) / float(number_of_ctf_rotations);
	float current_azimuth = 0;

	double *sum_of_ctf_squares;
	float *spatial_frequency_squared;
	float current_sigma;

	float mask_radius;

	MRCFile my_input_file(input_filename,false);
	long number_of_input_images = my_input_file.ReturnNumberOfSlices();

	NumericTextFile defocus_text(defocus_filename, OPEN_TO_READ);

	if (defocus_text.number_of_lines != number_of_input_images)
	{
		SendError("Error: Number of lines in defocus text file != number of images!");
		abort();
	}

	if (defocus_text.records_per_line != 3 && defocus_text.records_per_line != 4)
	{
		SendError("Error: Expect 3 or 4 records per line in defocus text file");
		abort();
	}

	Image my_image;
	Image scaled_image;
	Image first_sampled_image;
	Image second_sampled_image;
	Image first_sampled_image_scaled;
	Image second_sampled_image_scaled;

	ResolutionStatistics my_statistics;
	Curve average_frc;

	// CTF object
    CTF	current_ctf;
	current_ctf.Init(acceleration_voltage,spherical_aberration,amplitude_contrast,0,0,0,0.0,0.5,0.0,pixel_size,0.0);

	my_statistics.Init(pixel_size);

	first_sampled_image.Allocate(my_input_file.ReturnXSize(), my_input_file.ReturnYSize(), true);
	second_sampled_image.Allocate(my_input_file.ReturnXSize(), my_input_file.ReturnYSize(), true);


	// mask radius

	mask_radius = powf(3.0 * kDa_to_Angstrom3(molecular_mass_in_kda) / 4.0 / PI / powf(pixel_size,3), 1.0 / 3.0);
	wxPrintf("mass = %f, mask_radius = %f\n", molecular_mass_in_kda, mask_radius);

	wxPrintf("\nEstimating SSNR...\n\n");
	ProgressBar *my_progress = new ProgressBar(my_input_file.ReturnNumberOfSlices());

	// read in first image and setup curve..

	my_image.ReadSlice(&my_input_file, 1);


//	my_image.CosineMask(mask_radius, 5);
	my_image.ForwardFFT();
	my_image.SubSampleWithNoisyResampling(&first_sampled_image, &second_sampled_image);

	//first_sampled_image.QuickAndDirtyWriteSlice("first.mrc", 1);
	//second_sampled_image.QuickAndDirtyWriteSlice("second.mrc", 1);

	my_statistics.CalculateFSC(first_sampled_image, second_sampled_image);

	average_frc.CopyFrom(&my_statistics.FSC);

	fourier_voxel_size = 0.5 / float(my_statistics.number_of_bins - 1);
	//fourier_voxel_size = first_sampled_image.fourier_voxel_size_x;

	sum_of_ctf_squares = new double[average_frc.number_of_points];
	spatial_frequency_squared = new float[average_frc.number_of_points];

	defocus_text.ReadLine(temp_float);
	current_ctf.SetDefocus(temp_float[0] / pixel_size, temp_float[1] / pixel_size, deg_2_rad(temp_float[2]));
	//wxPrintf("Defocus = %f, %f, %f\n", temp_float[0], temp_float[1], temp_float[2]);

	for (counter = 0; counter < average_frc.number_of_points; counter++)
	{
		spatial_frequency_squared[counter] = powf(float(counter) * fourier_voxel_size, 2);
		sum_of_ctf_squares[counter] = 0;
	}


	current_azimuth = 0;

	for (rotation_counter = 0; rotation_counter < number_of_ctf_rotations; rotation_counter++)
	{
		for (counter = 0; counter < average_frc.number_of_points; counter++)
		{
			sum_of_ctf_squares[counter] += powf(current_ctf.Evaluate(spatial_frequency_squared[counter], current_azimuth), 2);
		}

		current_azimuth += rotation_step;
		number_of_ctfs++;
	}




	// loop over remaining images..

	for ( long image_counter = 1; image_counter < my_input_file.ReturnNumberOfSlices(); image_counter++ )
	//for ( long image_counter = 1; image_counter <1; image_counter++ )
	{
		my_image.ReadSlice(&my_input_file,image_counter+1);
		my_image.ForwardFFT();
		my_image.SubSampleWithNoisyResampling(&first_sampled_image, &second_sampled_image);

//		first_sampled_image.QuickAndDirtyWriteSlice("first.mrc", image_counter + 1);
	//	second_sampled_image.QuickAndDirtyWriteSlice("second.mrc", image_counter + 1);

		my_statistics.CalculateFSC(first_sampled_image, second_sampled_image);

		//first_sampled_image.CosineMask(mask_radius, 5);
		//second_sampled_image.CosineMask(mask_radius, 5);

		defocus_text.ReadLine(temp_float);
		current_ctf.SetDefocus(temp_float[0] / pixel_size, temp_float[1] / pixel_size, deg_2_rad(temp_float[2]));

		current_azimuth = 0;

		for (rotation_counter = 0; rotation_counter < number_of_ctf_rotations; rotation_counter++)
		{
			for (counter = 0; counter < average_frc.number_of_points; counter++)
			{
				sum_of_ctf_squares[counter] += powf(current_ctf.Evaluate(spatial_frequency_squared[counter], current_azimuth), 2);
			}

			current_azimuth += rotation_step;
			number_of_ctfs++;
		}

		average_frc.AddWith(&my_statistics.FSC);
		my_progress->Update(image_counter + 1);

	}

	for (counter = 0; counter < average_frc.number_of_points; counter++)
	{
		average_frc.data_y[counter] /= float(my_input_file.ReturnNumberOfSlices());
		sum_of_ctf_squares[counter] /= double(number_of_ctfs);

		if (average_frc.data_y[counter] < 0.999999)
		{
			average_frc.data_y[counter] = (2.0 * average_frc.data_y[counter]) / (1.0 - average_frc.data_y[counter]);
		}
		else average_frc.data_y[counter] = 2000000;

		average_frc.data_y[counter] /= sum_of_ctf_squares[counter];
		//average_frc.data_y[counter] = sum_of_ctf_squares[counter];


	}

	average_frc.WriteToFile(output_filename);
	delete my_progress;

	delete [] sum_of_ctf_squares;
	delete [] spatial_frequency_squared;
	wxPrintf("\n\n");

	return true;
}
