#include "../../core/core_headers.h"

class
        CreateMask : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(CreateMask)

// override the DoInteractiveUserInput

void CreateMask::DoInteractiveUserInput( ) {
    UserInput* my_input                    = new UserInput("CreateMask", 1.0);
    wxString   input_volume                = my_input->GetFilenameFromUser("Input image/volume file name", "Name of input image file", "input.mrc", true);
    wxString   output_volume               = my_input->GetFilenameFromUser("Output masked image/volume file name", "Name of output image with mask applied", "output.mrc", false);
    float      pixel_size                  = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.000001);
    float      outer_mask_radius           = my_input->GetFloatFromUser("Outer radius of mask (A)", "The mask radius in Angstroms", "100", 0.0);
    bool       auto_estimate_bin_threshold = my_input->GetYesNoFromUser("Auto Estimate Binarization threshold?", "If Yes, the initial binarization threshold will be estimated automatically, answer NO to provide your own (e.g. from Chimera thresholding)", "YES");
    float      initial_bin_threshold       = 0.0f;
    if ( auto_estimate_bin_threshold == false )
        initial_bin_threshold = my_input->GetFloatFromUser("Wanted initial binarization threshold", "Density will be thresholded at this value initially", "1", 0.0);

    float filter_resolution = my_input->GetFloatFromUser("Low-pass filter resolution (A)", "Low-pass filter resolution to be applied to the density", "10.0", 0.0);
    float rebin_value       = my_input->GetFloatFromUser("Re-Bin Value (0-1)", "Higher values will lead to smaller masks, lower values to larger masks", "0.35", 0.0, 1);

    delete my_input;
    //	my_current_job.Reset(9);
    my_current_job.ManualSetArguments("ttffffbf", input_volume.ToUTF8( ).data( ), output_volume.ToUTF8( ).data( ), pixel_size, outer_mask_radius, filter_resolution, rebin_value, auto_estimate_bin_threshold, initial_bin_threshold);
}

// override the do calculation method which will be what is actually run..

bool CreateMask::DoCalculation( ) {

    wxString input_volume                = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_volume               = my_current_job.arguments[1].ReturnStringArgument( );
    float    pixel_size                  = my_current_job.arguments[2].ReturnFloatArgument( );
    float    outer_mask_radius           = my_current_job.arguments[3].ReturnFloatArgument( );
    float    filter_resolution           = my_current_job.arguments[4].ReturnFloatArgument( );
    float    rebin_value                 = my_current_job.arguments[5].ReturnFloatArgument( );
    bool     auto_estimate_bin_threshold = my_current_job.arguments[6].ReturnBoolArgument( );
    float    initial_bin_threshold       = my_current_job.arguments[7].ReturnFloatArgument( );

    Image     input_image;
    ImageFile input_file;
    MRCFile   output_file;

    clock_t startTime, endTime;
    startTime = clock( );

    input_file.OpenFile(input_volume.ToStdString( ), false);
    input_image.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices( ));
    input_file.CloseFile( );

    input_image.ConvertToAutoMask(pixel_size, outer_mask_radius, filter_resolution, rebin_value, auto_estimate_bin_threshold, initial_bin_threshold);

    output_file.OpenFile(output_volume.ToStdString( ), true);
    input_image.WriteSlices(&output_file, 1, input_image.logical_z_dimension);
    output_file.CloseFile( );

    endTime = clock( );
    MyDebugPrint("\nThe run time is: %fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    return true;
}
