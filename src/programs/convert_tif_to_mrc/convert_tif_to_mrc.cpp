#include "../../core/core_headers.h"
#include <wx/dir.h>

class
ConvertTIF2MRC : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(ConvertTIF2MRC)

// override the DoInteractiveUserInput

void ConvertTIF2MRC::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("ConvertTIF2MRC", 1.0);

	std::string input_filename		=		my_input->GetFilenameFromUser("Input TIF file name", "Filename of input TIF image", "input.tif", true );
	std::string output_filename		=		my_input->GetFilenameFromUser("Output MRC file name", "Filename of output MRC image", "output.mrc", false );


	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("tt", input_filename.c_str(), output_filename.c_str());
}

// override the do calculation method which will be what is actually run..

bool ConvertTIF2MRC::DoCalculation()
{


	std::string	input_filename 					= my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename 					= my_current_job.arguments[1].ReturnStringArgument();

	ImageFile input_file;
	MRCFile output_file;

	Image buffer_image;

	input_file.OpenFile(input_filename, false);
	output_file.OpenFile(output_filename, true);

//	wxPrintf("Tif file = %ix%ix%i\n", input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize());

	wxPrintf("Converting File...\n\n");

	ProgressBar *my_progress = new ProgressBar(input_file.ReturnNumberOfSlices());

	for (int counter = 1; counter <= input_file.ReturnNumberOfSlices(); counter++ )
	{
		buffer_image.ReadSlice(&input_file, counter);
		buffer_image.WriteSlice(&output_file, counter);
		my_progress->Update(counter);
	}

	delete my_progress;


	return true;
}
