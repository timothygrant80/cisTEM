#include "../../core/core_headers.h"


class
EerRender : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(EerRender)

// override the DoInteractiveUserInput

void EerRender::DoInteractiveUserInput()
{
	UserInput *my_input = new UserInput("EerRender", 1.0);
	wxString input_eer	= my_input->GetFilenameFromUser("Input eer file name", "Name of input eer file", "input.eer", true );
	wxString output_coordinate	= my_input->GetFilenameFromUser("Output coordinate file name", "Name of output coordinate file", "output.txt", false );

	delete my_input;
//	my_current_job.Reset(9);
	my_current_job.ManualSetArguments("tt", input_eer.ToUTF8().data(), output_coordinate.ToUTF8().data());
}

// override the do calculation method which will be what is actually run..

bool EerRender::DoCalculation()
{

	wxString input_eer	= my_current_job.arguments[0].ReturnStringArgument();
	wxString output_coordinate	= my_current_job.arguments[1].ReturnStringArgument();
	EerFile input_file;
	int x_size;
	int y_size;
	int number_of_slices;
	clock_t startTime,endTime;
	startTime = clock();
	
	input_file.OpenFile(input_eer.ToStdString(), false, false, false);

	input_file.rleFrames();
	endTime = clock();
	MyDebugPrint("\nThe run time is: %fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
	return true;
}
