#include "../../core/core_headers.h"

class
        ScaleWithMask : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ScaleWithMask)

// override the DoInteractiveUserInput

void ScaleWithMask::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("ScaleWithMask", 1.0);

    std::string input_ref_filename         = my_input->GetFilenameFromUser("Input reference volume", "Filename of volume to scale to", "ref.mrc", true);
    std::string input_volume_filename      = my_input->GetFilenameFromUser("Input volume to scale", "Filename of volume to scale", "input.mrc", true);
    std::string input_first_mask_filename  = my_input->GetFilenameFromUser("Input mask for volume #1", "Filename of volume to use as a mask prior to calculating scale filter", "mask1.mrc", true);
    std::string input_second_mask_filename = my_input->GetFilenameFromUser("Input mask for volume #2 (can be same)", "Filename of volume to use as a mask prior to calculating scale filter", "mask2.mrc", true);
    std::string output_filename            = my_input->GetFilenameFromUser("Output scaled volume", "Filename of output scaled volume", "scaled.mrc", false);
    bool        apply_resolution_cut_off   = my_input->GetYesNoFromUser("Cut-Off Resolution?", "If yes, the resolution will be cut off by a cosine at the specified resolution", "NO");

    float pixel_size         = 0.0;
    float resolution_cut_off = 0.0;

    pixel_size         = my_input->GetFloatFromUser("Pixel Size (A)", "The pixel size in angstroms", "1.0");
    resolution_cut_off = my_input->GetFloatFromUser("Wanted resolution cut-off (A)", "The cut-off resolution", "3.0");

    delete my_input;

    my_current_job.ManualSetArguments("tttttbff", input_ref_filename.c_str( ),
                                      input_volume_filename.c_str( ),
                                      input_first_mask_filename.c_str( ),
                                      input_second_mask_filename.c_str( ),
                                      output_filename.c_str( ),
                                      apply_resolution_cut_off,
                                      pixel_size,
                                      resolution_cut_off);
}

// override the do calculation method which will be what is actually run..

bool ScaleWithMask::DoCalculation( ) {

    std::string input_ref_filename         = my_current_job.arguments[0].ReturnStringArgument( );
    std::string input_volume_filename      = my_current_job.arguments[1].ReturnStringArgument( );
    std::string input_first_mask_filename  = my_current_job.arguments[2].ReturnStringArgument( );
    std::string input_second_mask_filename = my_current_job.arguments[3].ReturnStringArgument( );
    std::string output_filename            = my_current_job.arguments[4].ReturnStringArgument( );
    bool        apply_resolution_cut_off   = my_current_job.arguments[5].ReturnBoolArgument( );
    float       pixel_size                 = my_current_job.arguments[6].ReturnFloatArgument( );
    float       resolution_cut_off         = my_current_job.arguments[7].ReturnFloatArgument( );

    MRCFile my_input_ref_file(input_ref_filename, false);
    MRCFile my_input_volume_file(input_volume_filename, false);
    MRCFile my_first_input_mask_file(input_first_mask_filename, false);
    MRCFile my_second_input_mask_file(input_second_mask_filename, false);
    MRCFile my_output_file(output_filename, true);

    Image ref_volume;
    Image volume_to_scale;
    Image first_mask_volume;
    Image second_mask_volume;
    Image buffer_volume;

    Curve original_amp;
    Curve ref_amp;
    Curve number_of_measurements;

    ref_volume.ReadSlices(&my_input_ref_file, 1, my_input_ref_file.ReturnNumberOfSlices( ));
    volume_to_scale.ReadSlices(&my_input_volume_file, 1, my_input_volume_file.ReturnNumberOfSlices( ));
    first_mask_volume.ReadSlices(&my_first_input_mask_file, 1, my_first_input_mask_file.ReturnNumberOfSlices( ));
    second_mask_volume.ReadSlices(&my_second_input_mask_file, 1, my_second_input_mask_file.ReturnNumberOfSlices( ));

    buffer_volume.CopyFrom(&volume_to_scale);

    ref_volume.MultiplyPixelWise(first_mask_volume);
    volume_to_scale.MultiplyPixelWise(second_mask_volume);

    int number_of_points = myroundint(ref_volume.ReturnMaximumDiagonalRadius( ));

    ref_volume.ForwardFFT( );
    volume_to_scale.ForwardFFT( );
    buffer_volume.ForwardFFT( );

    if ( apply_resolution_cut_off == true ) {
        ref_volume.CosineMask(pixel_size / resolution_cut_off, ref_volume.fourier_voxel_size_x * 5);
    }

    // setup curves

    //wxPrintf("number of points = %i\n", number_of_points);

    original_amp.SetupXAxis(0, 0.5 * sqrtf(3.0), number_of_points);
    ref_amp.SetupXAxis(0, 0.5 * sqrtf(3.0), number_of_points);
    number_of_measurements.SetupXAxis(0, 0.5 * sqrtf(3.0), number_of_points);

    //original_amp.PrintToStandardOut();

    ref_volume.Compute1DPowerSpectrumCurve(&ref_amp, &number_of_measurements);
    volume_to_scale.Compute1DPowerSpectrumCurve(&original_amp, &number_of_measurements);

    ref_amp.SquareRoot( );
    original_amp.SquareRoot( );

    for ( long counter = 0; counter < ref_amp.number_of_points; counter++ ) {
        if ( original_amp.data_y[counter] != 0.0 && ref_amp.data_y[counter] != 0.0 ) {
            original_amp.data_y[counter] = ref_amp.data_y[counter] / original_amp.data_y[counter];
        }
        else
            original_amp.data_y[counter] = 1.0;
    }

    //original_amp.PrintToStandardOut();

    buffer_volume.ApplyCurveFilter(&original_amp);
    buffer_volume.BackwardFFT( );

    buffer_volume.WriteSlices(&my_output_file, 1, buffer_volume.logical_z_dimension);
    if ( apply_resolution_cut_off == true )
        my_output_file.SetPixelSize(pixel_size);
    else
        my_output_file.SetPixelSize(my_input_volume_file.my_header.ReturnPixelSize( ));

    wxPrintf("\n\nScale with mask finished cleanly!\n\n");

    return true;
}
