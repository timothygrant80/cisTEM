#include "../../core/core_headers.h"

class
        ConvertEERToMRC : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ConvertEERToMRC)

// override the DoInteractiveUserInput

void ConvertEERToMRC::DoInteractiveUserInput( ) {
    UserInput*  my_input              = new UserInput("ConvertEERToMRC", 1.0);
    std::string input_eer_filename    = my_input->GetFilenameFromUser("Input eer file name", "Name of input eer file", "input.eer", true);
    std::string output_image_filename = my_input->GetFilenameFromUser("Output movie mrc file", "Name of output converted mrc file", "output.mrc", false);
    int         super_res_level       = my_input->GetIntFromUser("Super Resolution Level (1, 2 or 4)", "do you want images 1, 2, or 4 times physical size", "1", 1, 4);

    if ( super_res_level == 3 ) {
        wxPrintf("You just had to put 3 didn't you!  Well, now I am sulking\n");
        exit(-1);
    }

    int temporal_frame_bin_factor = my_input->GetIntFromUser("Temporal Frame Bin Factor", "How many 'raw' frames should be added together for the output", "1", 1);

    bool        write_unaligned_sum = my_input->GetYesNoFromUser("Also write an unaligned sum?", "if yes the unaligned sum will be written", "NO");
    std::string unaligned_sum_filename;
    if ( write_unaligned_sum == true )
        unaligned_sum_filename = my_input->GetFilenameFromUser("Output summed movie files", "Filename to save unaligned sum", "output_sum.mrc", false);

    delete my_input;

    my_current_job.ManualSetArguments("ttiibt", input_eer_filename.c_str( ), output_image_filename.c_str( ), super_res_level, temporal_frame_bin_factor, write_unaligned_sum, unaligned_sum_filename.c_str( ));
}

// override the do calculation method which will be what is actually run..

bool ConvertEERToMRC::DoCalculation( ) {

    std::string input_eer_filename        = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_image_filename     = my_current_job.arguments[1].ReturnStringArgument( );
    int         super_res_level           = my_current_job.arguments[2].ReturnIntegerArgument( );
    int         temporal_frame_bin_factor = my_current_job.arguments[3].ReturnIntegerArgument( );
    bool        write_unaligned_sum       = my_current_job.arguments[4].ReturnBoolArgument( );
    std::string unaligned_sum_filename    = my_current_job.arguments[5].ReturnStringArgument( );

    EerFile input_file;

    input_file.OpenFile(input_eer_filename, false, false, false, super_res_level, temporal_frame_bin_factor);
    input_file.PrintInfo( );
    Image frame_stack;
    frame_stack.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices( ));
    MRCFile output_stack;
    output_stack.OpenFile(output_image_filename, true);
    frame_stack.WriteSlices(&output_stack, 1, frame_stack.logical_z_dimension);
    output_stack.PrintInfo( );

    if ( write_unaligned_sum ) {
        Image unaligned_sum;
        unaligned_sum.Allocate(frame_stack.logical_x_dimension, frame_stack.logical_y_dimension, 1);
        unaligned_sum.AddSlices(frame_stack);
        unaligned_sum.QuickAndDirtyWriteSlice(unaligned_sum_filename, 1);
    }

    return true;
}
