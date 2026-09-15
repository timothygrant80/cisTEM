#include "../../core/core_headers.h"

class
        ApplyMask : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ApplyMask)

// override the DoInteractiveUserInput

void ApplyMask::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("ApplyMask", 1.0);

    std::string input_volume      = my_input->GetFilenameFromUser("Input image/volume file name", "Name of input image file", "input.mrc", true);
    std::string input_mask        = my_input->GetFilenameFromUser("Input mask file name", "Name of input image file", "mask.mrc", true);
    std::string output_volume     = my_input->GetFilenameFromUser("Output masked image/volume file name", "Name of output image with mask applied", "output.mrc", false);
    float    pixel_size        = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.000001);
    float    cosine_edge       = my_input->GetFloatFromUser("Width of cosine edge (A)", "Width of the smooth edge to add to the mask in Angstroms", "10.0", 0.0);
    float    outside_weight    = my_input->GetFloatFromUser("Weight of density outside mask", "Factor to multiply density outside of the mask", "0.0", 0.0, 1.0);
    float    filter_radius     = my_input->GetFloatFromUser("Low-pass filter outside mask (A)", "Low-pass filter to be applied to the density outside the mask", "0.0", 0.0);
    float    outside_value     = my_input->GetFloatFromUser("Outside mask value", "Value used to set density outside the mask", "0.0", 0.0);
    bool     use_outside_value = my_input->GetYesNoFromUser("Use outside mask value", "Should the density outside the mask be set to the user-provided value", "No");

    delete my_input;

    //	my_current_job.Reset(9);
    my_current_job.ManualSetArguments("tttfffffb", input_volume.c_str(), input_mask.c_str(), output_volume.c_str(), pixel_size, cosine_edge, outside_weight, filter_radius, outside_value, use_outside_value);
}

// override the do calculation method which will be what is actually run..

bool ApplyMask::DoCalculation( ) {

    std::string input_volume      = my_current_job.arguments[0].ReturnStringArgument( );
    std::string input_mask        = my_current_job.arguments[1].ReturnStringArgument( );
    std::string output_volume     = my_current_job.arguments[2].ReturnStringArgument( );
    float    pixel_size        = my_current_job.arguments[3].ReturnFloatArgument( );
    float    cosine_edge       = my_current_job.arguments[4].ReturnFloatArgument( );
    float    outside_weight    = my_current_job.arguments[5].ReturnFloatArgument( );
    float    filter_radius     = my_current_job.arguments[6].ReturnFloatArgument( );
    float    outside_value     = my_current_job.arguments[7].ReturnFloatArgument( );
    bool     use_outside_value = my_current_job.arguments[8].ReturnBoolArgument( );

    MRCFile input3d_file(input_volume, false);
    MRCFile input_mask_file(input_mask, false);
    MRCFile output_file(output_volume, true);

    float filter_edge = 40.0;
    float mask_volume;
    Image my_image;
    Image my_mask;

    if ( input3d_file.ReturnZSize( ) > 1 )
        Printf("\nMasking Volume...\n");
    else
        Printf("\nMasking Image...\n");

    my_image.ReadSlices(&input3d_file, 1, input3d_file.ReturnNumberOfSlices( ));
    my_mask.ReadSlices(&input_mask_file, 1, input_mask_file.ReturnNumberOfSlices( ));
    if ( ! my_image.HasSameDimensionsAs(&my_mask) ) {
        if ( input3d_file.ReturnZSize( ) > 1 )
            Printf("\nVolume and mask file have different dimensions\n");
        else
            Printf("\nImage and mask file have different dimensions\n");
        DEBUG_ABORT;
    }
    if ( filter_radius == 0.0 )
        filter_radius = pixel_size;
    mask_volume = my_image.ApplyMask(my_mask, cosine_edge / pixel_size, outside_weight, pixel_size / filter_radius, pixel_size / filter_edge, outside_value, use_outside_value);
    my_image.WriteSlices(&output_file, 1, input3d_file.ReturnNumberOfSlices( ));

    Printf("\nMask volume = %g voxels\n\n", mask_volume);

    return true;
}
