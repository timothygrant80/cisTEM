#include "../../core/core_headers.h"

class
        ConvertParToStar : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ConvertParToStar)

// override the DoInteractiveUserInput

void ConvertParToStar::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("ConvertParToStar", 1.0);

    std::string input_filename_one = my_input->GetFilenameFromUser("Input PAR file", "Filename of frealign par file to convert", "input.par", true);
    std::string output_filename    = my_input->GetFilenameFromUser("Output cisTEM STAR file", "converted output in the new cisTEM format", "output.star", false);
    float       microscope_voltage = my_input->GetFloatFromUser("Wanted Microscope Voltage (kV)", "The microscope voltage in kV to be added to the output file", "300");
    float       microscope_cs      = my_input->GetFloatFromUser("Wanted Microscope Cs (mm)", "The microscope Cs in mm to be added to the output file", "2.7");
    float       amplitude_contrast = my_input->GetFloatFromUser("Wanted Amplitude Contrast", "The image amplitude contrast to be added to the output file", "0.07");
    float       pixel_size         = my_input->GetFloatFromUser("Pixel Size (A)", "The pixel size to be added to the output file", "1.0");
    float       beam_tilt_x        = my_input->GetFloatFromUser("Beam Tilt X (mrad)", "The horizontal beam tilt to be added to the output file", "0.0");
    float       beam_tilt_y        = my_input->GetFloatFromUser("Beam Tilt Y (mrad)", "The vertical beam tilt to be added to the output file", "0.0");
    float       image_shift_x      = my_input->GetFloatFromUser("Image Shift X", "The horizontal image shift to be added to the output file", "0.0");
    float       image_shift_y      = my_input->GetFloatFromUser("Image Shift Y", "The vertical image shift be added to the output file", "0.0");

    delete my_input;

    my_current_job.Reset(10);
    my_current_job.ManualSetArguments("ttffffffff", input_filename_one.c_str( ), output_filename.c_str( ), microscope_voltage, microscope_cs, amplitude_contrast, pixel_size, beam_tilt_x, beam_tilt_y, image_shift_x, image_shift_y);
}

// override the do calculation method which will be what is actually run..

bool ConvertParToStar::DoCalculation( ) {
    wxString input_filename_one = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_filename    = my_current_job.arguments[1].ReturnStringArgument( );
    float    microscope_voltage = my_current_job.arguments[2].ReturnFloatArgument( );
    float    microscope_cs      = my_current_job.arguments[3].ReturnFloatArgument( );
    float    amplitude_contrast = my_current_job.arguments[4].ReturnFloatArgument( );
    float    pixel_size         = my_current_job.arguments[5].ReturnFloatArgument( );
    float    beam_tilt_x        = my_current_job.arguments[6].ReturnFloatArgument( );
    float    beam_tilt_y        = my_current_job.arguments[7].ReturnFloatArgument( );
    float    image_shift_x      = my_current_job.arguments[8].ReturnFloatArgument( );
    float    image_shift_y      = my_current_job.arguments[9].ReturnFloatArgument( );

    wxPrintf("\nConverting...\n\n");
    cisTEMParameters converted_params;
    converted_params.ReadFromFrealignParFile(input_filename_one, pixel_size, microscope_voltage, microscope_cs, amplitude_contrast, beam_tilt_x, beam_tilt_y, image_shift_x, image_shift_y);

    converted_params.parameters_to_write.SetActiveParameters(POSITION_IN_STACK | IMAGE_IS_ACTIVE | PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | PHASE_SHIFT | OCCUPANCY | LOGP | SIGMA | SCORE | PIXEL_SIZE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y);
    converted_params.WriteTocisTEMStarFile(output_filename);
    wxPrintf("\n\n");

    return true;
}
