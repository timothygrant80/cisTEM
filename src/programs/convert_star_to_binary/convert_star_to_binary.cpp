#include "../../core/core_headers.h"

class
        ConvertStarToBin : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ConvertStarToBin)

// override the DoInteractiveUserInput

void ConvertStarToBin::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("ConvertStarToBin", 1.0);

    std::string input_filename  = my_input->GetFilenameFromUser("Input cisTEM STAR file", "Filename of cisTEM STAR file to convert", "input.star", true);
    std::string output_filename = my_input->GetFilenameFromUser("Output cisTEM binary file", "converted output in the cisTEM binary format", "output.cistem", false);

    delete my_input;

    my_current_job.Reset(2);
    my_current_job.ManualSetArguments("tt", input_filename.c_str( ), output_filename.c_str( ));
}

// override the do calculation method which will be what is actually run..

bool ConvertStarToBin::DoCalculation( ) {
    wxString input_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_filename = my_current_job.arguments[1].ReturnStringArgument( );

    wxPrintf("\nConverting...\n\n");

    cisTEMParameters converted_params;
    converted_params.ReadFromcisTEMStarFile(input_filename);
    converted_params.parameters_to_write = converted_params.parameters_that_were_read;
    converted_params.WriteTocisTEMBinaryFile(output_filename);
    wxPrintf("\n\n");

    return true;
}
