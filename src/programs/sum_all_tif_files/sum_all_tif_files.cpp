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
