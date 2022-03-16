#include "../../core/core_headers.h"

class
        ConvertBinToStar : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ConvertBinToStar)

// override the DoInteractiveUserInput

void ConvertBinToStar::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("ConvertBinToStar", 1.0);

    std::string input_filename  = my_input->GetFilenameFromUser("Input cisTEM binary file", "Filename of cisTEM binary file to convert", "input.cistem", true);
    std::string output_filename = my_input->GetFilenameFromUser("Output cisTEM Star file", "converted output in the cisTEM Star format", "output.star", false);

    delete my_input;

    my_current_job.Reset(2);
    my_current_job.ManualSetArguments("tt", input_filename.c_str( ), output_filename.c_str( ));
}

// override the do calculation method which will be what is actually run..

bool ConvertBinToStar::DoCalculation( ) {
    wxString input_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_filename = my_current_job.arguments[1].ReturnStringArgument( );

    wxPrintf("\nConverting...\n\n");

    cisTEMParameters converted_params;
    converted_params.ReadFromcisTEMBinaryFile(input_filename);
    converted_params.parameters_to_write = converted_params.parameters_that_were_read;
    converted_params.WriteTocisTEMStarFile(output_filename);
    wxPrintf("\n\n");

    return true;
}
