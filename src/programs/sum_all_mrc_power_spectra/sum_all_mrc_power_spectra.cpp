/*
 * sum_all_mrc_power_spectra.cpp
 *
 *  Created on: Oct 25, 2019
 *      Author: himesb
 */




#include "../../core/core_headers.h"
#include <wx/dir.h>

class
SumAllMRCPowerSpectra : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(SumAllMRCPowerSpectra)

// override the DoInteractiveUserInput

void SumAllMRCPowerSpectra::DoInteractiveUserInput()
{

	int new_z_size = 1;
	int max_threads;


	UserInput *my_input = new UserInput("SumAllMRCPowerSpectra", 1.0);

	std::string output_filename		=	my_input->GetFilenameFromUser("Output sum file name", "Filename of output image", "output.mrc", false );
	//bool invert_and_scale         =       my_input->GetYesNoFromUser("Take Reciprocal and Scale?", "If yes, the image will be 1/image and scaled to max density 1.", "YES");


	max_threads = my_input->GetIntFromUser("Max number of threads to use", "maximum number of threads to use for processing.", "1",1,100);

	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("ti", output_filename.c_str(), max_threads);
}

// override the do calculation method which will be what is actually run..

bool SumAllMRCPowerSpectra::DoCalculation()
{

	long frame_counter;
	long file_counter;
	long pixel_counter;
	long total_summed = 0;

	int file_x_size;
	int file_y_size;
	int file_z_size;

	std::string	output_filename 					= my_current_job.arguments[0].ReturnStringArgument();
	int max_threads									= my_current_job.arguments[1].ReturnIntegerArgument();
	bool is_mrc = false;

	wxArrayString all_files;
	wxPrintf("first checking for .tif\n");

	wxDir::GetAllFiles 	( ".", &all_files, "*.tif", wxDIR_FILES);
	if (all_files.GetCount() > 0)
	{
		wxPrintf("\nThere are %li TIF files in this directory.\n", all_files.GetCount());

	}
	else
	{
		wxPrintf("no tif, checking for .tiff\n");
		wxDir::GetAllFiles 	( ".", &all_files, "*.tiff", wxDIR_FILES);
		if (all_files.GetCount() > 0)
		{
			wxPrintf("\nThere are %li TIF files in this directory.\n", all_files.GetCount());

		}
		else
		{
			wxPrintf("no tif or tiff, checking for .mrc\n");
			wxDir::GetAllFiles 	( ".", &all_files, "*.mrc", wxDIR_FILES);
			if (all_files.GetCount() > 0)
			{
				is_mrc = true;
				wxPrintf("\nThere are %li MRC files in this directory.\n", all_files.GetCount());

			}
			else
			{
				wxPrintf("\n\tError : did not find any files of tif, tiff, or mrc extension.\n\n");
				exit(-1);
			}
		}
	}
	all_files.Sort();

	MRCFile *current_mrc_file;
	ImageFile *current_tif_file;


	Image output_PS;

	double *output_PS_double;


	// find all the mrc files in the current directory..


	if (is_mrc)
	{
		current_mrc_file = new MRCFile(all_files.Item(0).ToStdString(), false);
		file_x_size = current_mrc_file->ReturnXSize();
		file_y_size = current_mrc_file->ReturnYSize();
		delete current_mrc_file;
	}
	else
	{
		current_tif_file = new ImageFile(all_files.Item(0).ToStdString(), false);
		file_x_size = current_tif_file->ReturnXSize();
		file_y_size = current_tif_file->ReturnYSize();
		delete current_tif_file;

	}




	int wanted_sq_size = 3456;

	wxPrintf("\nFirst file is %s\nIt is %ix%i sized - all images had better be this size!\n\n", all_files.Item(0), file_x_size, file_y_size);



//	output_PS.Allocate(file_x_size, file_y_size, 1);
	output_PS.Allocate(wanted_sq_size, wanted_sq_size, 1);

	output_PS.SetToConstant(0.0);
	const long N_mem =  output_PS.real_memory_allocated/2;

	output_PS_double = new double[max_threads*N_mem];

	ZeroDoubleArray(output_PS_double,max_threads*N_mem);





	// loop over all files, and do summing..

	wxPrintf("Summing All Files...\n\n");
	ProgressBar *my_progress = new ProgressBar(all_files.GetCount());

	int number_processed = 0;
	// thread if available
	#pragma omp parallel default(none) num_threads(max_threads) shared(output_PS, output_PS_double, max_threads, all_files, total_summed, number_processed, my_progress, is_mrc) private(file_counter, file_z_size, current_mrc_file, current_tif_file, frame_counter, pixel_counter)
	{ // bracket for omp

	int my_total_summed = 0;


	#pragma omp for
	for (file_counter = 0; file_counter < all_files.GetCount(); file_counter++)
	{
		//wxPrintf("Summing file %s...\n", all_files.Item(file_counter));
		Image buffer_image;
		int threadIDX = omp_get_thread_num();

		if (is_mrc)
		{
			current_mrc_file = new MRCFile(all_files.Item(file_counter).ToStdString(), false);
			file_z_size = current_mrc_file->ReturnNumberOfSlices();
		}
		else
		{
			current_tif_file = new ImageFile(all_files.Item(file_counter).ToStdString(), false);
			file_z_size = current_tif_file->ReturnNumberOfSlices();
		}

		for (frame_counter = 0; frame_counter < file_z_size; frame_counter++)
		{

			if (is_mrc)  buffer_image.ReadSlice(current_mrc_file, frame_counter + 1);
			else 		 buffer_image.ReadSlice(current_tif_file, frame_counter + 1);


			my_total_summed++;


				double absVal;
				buffer_image.Resize(3456,3456,1);
				buffer_image.ReplaceOutliersWithMean(5.0f);


				buffer_image.ForwardFFT();
				for (pixel_counter = 0; pixel_counter < N_mem; pixel_counter++)
				{
					absVal = std::abs(buffer_image.complex_values[pixel_counter]);
					output_PS_double[ threadIDX*N_mem + pixel_counter] += absVal*absVal;
				}
				buffer_image.is_in_real_space = true;
				buffer_image.object_is_centred_in_box = true;


			//sum_image.AddImage(&buffer_image);
		}


		if (is_mrc)
		{
			current_mrc_file->CloseFile();
			delete current_mrc_file;
		}
		{
			current_tif_file->CloseFile();
			delete current_tif_file;
		}


		#pragma omp atomic
		number_processed++;
		my_progress->Update(number_processed);

	}

	// combine all the results..

	#pragma omp critical
	{
		total_summed += my_total_summed;
		int threadIDX = omp_get_thread_num();

		if (threadIDX > 0)
		{
			for (pixel_counter = 0; pixel_counter < N_mem; pixel_counter++)
			{
				output_PS_double[pixel_counter] += output_PS_double[threadIDX*N_mem + pixel_counter];
			}
		}



	}

	long inner_counter = 0;
	for (pixel_counter = 0; pixel_counter < N_mem; pixel_counter++)
	{
		output_PS.real_values[inner_counter] += (float)output_PS_double[pixel_counter];
		inner_counter++;
		output_PS.real_values[inner_counter] = 0;// += (float)output_PS_double[pixel_counter];
		inner_counter++;
	}





	} // close bracket for omp

	delete [] output_PS_double;

	output_PS.is_in_real_space = false;
	Image buffer_image;
	buffer_image.CopyFrom(&output_PS);
	output_PS.ComputeAmplitudeSpectrumFull2D(&buffer_image,false,1.0f);
	output_PS.is_in_real_space = false;
	output_PS.SwapRealSpaceQuadrants();
	output_PS.QuickAndDirtyWriteSlice("./testPS.mrc",1,true,1.0);


	delete my_progress;

	wxPrintf("\n\nSum All MRC File finished Cleanly!\n\n");




	return true;
}
