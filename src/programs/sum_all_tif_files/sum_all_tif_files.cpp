#include "../../core/core_headers.h"
#include <wx/dir.h>

class
SumAllTIF : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(SumAllTIF)

void SumAllTIF::DoInteractiveUserInput()
{

	int new_z_size = 1;
	int max_threads;
	std::string output_dark_filename;
	std::string output_gain_filename;

	UserInput *my_input = new UserInput("SumAllTIFFiles", 1.0);

	std::string output_filename		=	my_input->GetFilenameFromUser("Output sum file name", "Filename of output image", "output.mrc", false );
	//bool invert_and_scale         =       my_input->GetYesNoFromUser("Take Reciprocal and Scale?", "If yes, the image will be 1/image and scaled to max density 1.", "YES");
	bool make_dark_and_gain			=   my_input->GetYesNoFromUser("Estimate Dark and Gain images", "If yes, a dark and gain image will be estimated", "YES");

	if (make_dark_and_gain == true)
	{
		output_dark_filename		=	my_input->GetFilenameFromUser("Output dark file name", "Filename of output dark image", "dark_image.mrc", false );
		output_gain_filename		=	my_input->GetFilenameFromUser("Output gain file name", "Filename of output gain image", "gain_image.mrc", false );
	}

	max_threads = my_input->GetIntFromUser("Max number of threads to use", "maximum number of threads to use for processing.", "1", 1);
	delete my_input;

	my_current_job.Reset(5);
	my_current_job.ManualSetArguments("tbtti", output_filename.c_str(), make_dark_and_gain, output_dark_filename.c_str(), output_gain_filename.c_str(), max_threads);
}

// override the do calculation method which will be what is actually run..

bool SumAllTIF::DoCalculation()
{
	long frame_counter;
	long file_counter;
	long pixel_counter;
	long total_summed = 0;

	int file_x_size;
	int file_y_size;

	std::string	output_filename 					= my_current_job.arguments[0].ReturnStringArgument();
	bool make_dark_and_gain                         = my_current_job.arguments[1].ReturnBoolArgument();
	std::string	output_dark_filename 				= my_current_job.arguments[2].ReturnStringArgument();
	std::string	output_gain_filename 				= my_current_job.arguments[3].ReturnStringArgument();
	int max_threads									= my_current_job.arguments[4].ReturnIntegerArgument();


	wxArrayString all_files;
	wxDir::GetAllFiles 	( ".", &all_files, "*.tif", wxDIR_FILES);
	all_files.Sort();

	ImageFile *current_input_file;

	Image output_sum_image;
	Image output_dark_image;
	Image output_gain_image;

	// global and local for threading...

	double *global_sum_image;
	double *global_sum_squares_image;

	double *sum_image;
	double *sum_squares_image;



	// find all the mrc files in the current directory..


	wxPrintf("\nThere are %li TIF files in this directory.\n", all_files.GetCount());

	current_input_file = new ImageFile(all_files.Item(0).ToStdString(), false);

	file_x_size = current_input_file->ReturnXSize();
	file_y_size = current_input_file->ReturnYSize();

	wxPrintf("\nFirst file is %s\nIt is %ix%i sized - all images had better be this size!\n\n", all_files.Item(0), current_input_file->ReturnXSize(), current_input_file->ReturnYSize());

	delete current_input_file;

	output_sum_image.Allocate(file_x_size, file_y_size, 1);
	output_sum_image.SetToConstant(0.0);

	if (make_dark_and_gain == true)
	{
		output_dark_image.Allocate(file_x_size, file_y_size, 1);
		output_gain_image.Allocate(file_x_size, file_y_size, 1);
	}

	global_sum_image = new double[output_sum_image.real_memory_allocated];
	ZeroDoubleArray(global_sum_image, output_sum_image.real_memory_allocated);

	if (make_dark_and_gain == true)
	{
		global_sum_squares_image = new double[output_sum_image.real_memory_allocated];
		ZeroDoubleArray(global_sum_squares_image, output_sum_image.real_memory_allocated);
	}

	// loop over all files, and do summing..

	wxPrintf("Summing All Files...\n\n");
	ProgressBar *my_progress = new ProgressBar(all_files.GetCount());

	// thread if available
#pragma omp parallel default(none) num_threads(max_threads) shared(make_dark_and_gain, global_sum_image, global_sum_squares_image, output_sum_image, all_files, total_summed, my_progress) private(file_counter, current_input_file, frame_counter, sum_image, sum_squares_image, pixel_counter)
	{ // bracket for omp

	Image buffer_image;
	int my_total_summed = 0;
	sum_image = new double[output_sum_image.real_memory_allocated];
	ZeroDoubleArray(sum_image, output_sum_image.real_memory_allocated);

	if (make_dark_and_gain == true)
	{
		sum_squares_image = new double[output_sum_image.real_memory_allocated];
		ZeroDoubleArray(sum_squares_image, output_sum_image.real_memory_allocated);
	}

#pragma omp for
	for (file_counter = 0; file_counter < all_files.GetCount(); file_counter++)
	{
		//wxPrintf("Summing file %s...\n", all_files.Item(file_counter));

		current_input_file = new ImageFile(all_files.Item(file_counter).ToStdString(), false);

		for (frame_counter = 0; frame_counter < current_input_file->ReturnNumberOfSlices(); frame_counter++)
		{
			buffer_image.ReadSlice(current_input_file, frame_counter + 1);

			my_total_summed++;

			for (pixel_counter = 0; pixel_counter < output_sum_image.real_memory_allocated; pixel_counter++)
			{
				sum_image[pixel_counter] += buffer_image.real_values[pixel_counter];

				if (make_dark_and_gain == true)
				{
					sum_squares_image[pixel_counter] += pow(buffer_image.real_values[pixel_counter], 2);
				}
			}
			//sum_image.AddImage(&buffer_image);
		}


		current_input_file->CloseFile();
		delete current_input_file;

		my_progress->Update(file_counter + 1);

	}

	// combine all the results..

#pragma omp critical
	{
		total_summed += my_total_summed;

		for (pixel_counter = 0; pixel_counter < output_sum_image.real_memory_allocated; pixel_counter++)
		{
			global_sum_image[pixel_counter] += sum_image[pixel_counter];

			if (make_dark_and_gain == true)
			{
				global_sum_squares_image[pixel_counter] += sum_squares_image[pixel_counter];
			}


		}
	}

	if (make_dark_and_gain == true)	delete [] sum_squares_image;
	delete [] sum_image;


	} // close bracket for omp


	for (pixel_counter = 0; pixel_counter < output_sum_image.real_memory_allocated; pixel_counter++)
	{
		output_sum_image.real_values[pixel_counter] = global_sum_image[pixel_counter];

		if (make_dark_and_gain == true)
		{
			output_dark_image.real_values[pixel_counter] = global_sum_image[pixel_counter] / double(total_summed);
			output_gain_image.real_values[pixel_counter] = sqrt(global_sum_squares_image[pixel_counter] / double(total_summed) - pow(global_sum_image[pixel_counter] / double(total_summed), 2));

			if (output_gain_image.real_values[pixel_counter] != 0.0f) output_gain_image.real_values[pixel_counter] = 1.0f / output_gain_image.real_values[pixel_counter];
		}
	}

	output_sum_image.QuickAndDirtyWriteSlice(output_filename, 1);

	if (make_dark_and_gain == true)
	{
		output_gain_image.QuickAndDirtyWriteSlice(output_gain_filename, 1);
		output_dark_image.QuickAndDirtyWriteSlice(output_dark_filename, 1);
	}


	delete [] global_sum_image;
	if (make_dark_and_gain == true)	delete [] global_sum_squares_image;

	delete my_progress;

	return true;
}


/*
// override the DoInteractiveUserInput

void SumAllTIF::DoInteractiveUserInput()
{

	int new_z_size = 1;

	UserInput *my_input = new UserInput("SumAllTIFFiles", 1.0);

	std::string output_filename		=		my_input->GetFilenameFromUser("Output sum file name", "Filename of output image", "output.mrc", false );
	bool invert_and_scale           =       my_input->GetYesNoFromUser("Take Reciprocal and Scale?", "If yes, the image will be 1/image and scaled to max density 1.", "YES");

	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("tb", output_filename.c_str(), invert_and_scale);
}

// override the do calculation method which will be what is actually run..

bool SumAllTIF::DoCalculation()
{

	long frame_counter;
	long file_counter;

	std::string	output_filename 					= my_current_job.arguments[0].ReturnStringArgument();
	bool invert_and_scale                           = my_current_job.arguments[1].ReturnBoolArgument();

	wxArrayString all_files;
	wxDir::GetAllFiles 	( ".", &all_files, "*.tif", wxDIR_FILES);
	all_files.Sort();

	ImageFile *current_input_file;

	Image buffer_image;
	Image sum_image;

	// find all the mrc files in the current directory..


	wxPrintf("\nThere are %li TIF files in this directory.\n", all_files.GetCount());

	current_input_file = new ImageFile(all_files.Item(0).ToStdString(), false);
	sum_image.Allocate(current_input_file->ReturnXSize(), current_input_file->ReturnYSize(), 1);
	sum_image.SetToConstant(0.0);

	wxPrintf("\nFirst file is %s\nIt is %ix%i sized - all images had better be this size!\n\n", all_files.Item(0), current_input_file->ReturnXSize(), current_input_file->ReturnYSize());

	delete current_input_file;

	// loop over all files, and do summing..

	wxPrintf("Summing All Files...\n\n");
	ProgressBar *my_progress = new ProgressBar(all_files.GetCount());

	for (file_counter = 0; file_counter < all_files.GetCount(); file_counter++)
	{
		//wxPrintf("Summing file %s...\n", all_files.Item(file_counter));

		current_input_file = new ImageFile(all_files.Item(file_counter).ToStdString(), false);

		for (frame_counter = 0; frame_counter < current_input_file->ReturnNumberOfSlices(); frame_counter++)
		{
			buffer_image.ReadSlice(current_input_file, frame_counter + 1);
			sum_image.AddImage(&buffer_image);
		}


		current_input_file->CloseFile();
		delete current_input_file;

		my_progress->Update(file_counter + 1);
	}

	delete my_progress;


	if (invert_and_scale == true)
	{
		//sum_image.QuickAndDirtyWriteSlice("ori.mrc", 1);
		sum_image.TakeReciprocalRealValues();
		float max_value = sum_image.ReturnMaximumValue();
		//wxPrintf("max value = %f", max_value);
		sum_image.QuickAndDirtyWriteSlice("reciprocal.mrc", 1);
		sum_image.DivideByConstant(max_value);

	}

	sum_image.QuickAndDirtyWriteSlice(output_filename, 1);

	return true;
}

*/
