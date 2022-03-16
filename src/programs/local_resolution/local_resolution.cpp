#include "../../core/core_headers.h"

class
        LocalResolution : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(LocalResolution)

// override the DoInteractiveUserInput

void LocalResolution::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("LocalResolution", 0.1);

    ImageFile input_image_file;

    wxString input_volume_one  = my_input->GetFilenameFromUser("Input reconstruction 1", "The first input 3D reconstruction used for FSC calculation", "my_reconstruction_1.mrc", true);
    wxString input_volume_two  = my_input->GetFilenameFromUser("Input reconstruction 2", "The second input 3D reconstruction used for FSC calculation", "my_reconstruction_2.mrc", true);
    wxString input_volume_mask = my_input->GetFilenameFromUser("Input mask", "Positions where this mask volume has value 0.00 will be skipped during local resolution estimation", "my_mask.mrc", true);
    wxString output_volume     = my_input->GetFilenameFromUser("Output local resolution volume", "This volume will be a local resolution estimate map", "local_resolution.mrc", false);
    float    pixel_size        = my_input->GetFloatFromUser("Pixel size (A)", "Pixel size of the map in Angstroms", "1.0", 0.000001);
    wxString my_symmetry       = my_input->GetSymmetryFromUser("Particle symmetry", "The symmetry imposed on the input reconstructions", "C1");
    input_image_file.OpenFile(input_volume_one.ToStdString( ), false, false);
    int   first_slice         = my_input->GetIntFromUser("First slice", "First slice within the volume to estimate local resolution", "1", 1, input_image_file.ReturnNumberOfSlices( ));
    int   last_slice          = my_input->GetIntFromUser("Last slice", "Last slice within the volume to estimate local resolution (0 = last slice of volume)", "0", 0, input_image_file.ReturnNumberOfSlices( ));
    int   sampling_step       = my_input->GetIntFromUser("Sampling step", "Estimate the local resolution every STEP pixels in each direction. Set to 1 to estimate the resolution at every voxel", "2", 1, 16);
    int   box_size            = my_input->GetIntFromUser("Box size", "In pixels, the size of the small cubic box used to compute the local FSC", "20", 8, input_image_file.ReturnNumberOfSlices( ));
    bool  use_fixed_threshold = my_input->GetYesNoFromUser("Use a fixed FSC threshold?", "You can specify the value of the FSC threshold in the next question", "no");
    float fixed_threshold;
    float threshold_snr;
    float confidence_level;
    if ( use_fixed_threshold ) {
        fixed_threshold  = my_input->GetFloatFromUser("Fixed FSC threshold", "Commonly used values: 0.143", "0.5", -1.0, 1.0);
        threshold_snr    = -1.0;
        confidence_level = -1.0;
    }
    else {
        fixed_threshold  = -1.0;
        threshold_snr    = my_input->GetFloatFromUser("Threshold half-map SNR", "FSC value must correspond to at least this SNR. FSC=0.143 corresponds to SNR=0.167.", "0.167", 0.0);
        confidence_level = my_input->GetFloatFromUser("Confidence level", "In numbers of standard deviations of the SNR estimator", "3.0", 0.0);
    }

    input_image_file.CloseFile( );

    bool  experimental_options = my_input->GetYesNoFromUser("Set experimental options?", "Several experimental options are available", "no");
    bool  randomize_phases;
    float resolution_for_phase_randomization;
    bool  whiten_half_maps;
    int   padding_factor;
    randomize_phases                   = false;
    resolution_for_phase_randomization = -1.0;
    whiten_half_maps                   = true;
    padding_factor                     = 2;
    if ( experimental_options ) {
        randomize_phases                   = my_input->GetYesNoFromUser("Randomize input phases?", "The phases of the input halfmaps can be randomized beyond a chosen resolution before local volumes are extracted for FSC calculation", "no");
        resolution_for_phase_randomization = my_input->GetFloatFromUser("Resolution for phase randomization", "Phases of the input half maps will be randomized at resolutions higher than the given resolution", "20.0", 0.0);
        whiten_half_maps                   = my_input->GetYesNoFromUser("Whiten half-maps on input", "Whitening the half-maps helps obtain more localized resolution estimates", "yes");
        padding_factor                     = my_input->GetIntFromUser("Padding factor", "Give 1 if you don't want any padding", "2", 1);
    }

    delete my_input;

    my_current_job.Reset(18);
    my_current_job.ManualSetArguments("ttttftiiiibfffbfbi", input_volume_one.ToUTF8( ).data( ), input_volume_two.ToUTF8( ).data( ), input_volume_mask.ToUTF8( ).data( ), output_volume.ToUTF8( ).data( ), pixel_size, my_symmetry.ToUTF8( ).data( ), first_slice, last_slice, sampling_step, box_size, use_fixed_threshold, fixed_threshold, threshold_snr, confidence_level, randomize_phases, resolution_for_phase_randomization, whiten_half_maps, padding_factor);
}

bool LocalResolution::DoCalculation( ) {
    wxString input_volume_one_fn                = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_volume_two_fn                = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_volume_mask_fn               = my_current_job.arguments[2].ReturnStringArgument( );
    wxString output_volume_fn                   = my_current_job.arguments[3].ReturnStringArgument( );
    float    pixel_size                         = my_current_job.arguments[4].ReturnFloatArgument( );
    wxString my_symmetry                        = my_current_job.arguments[5].ReturnStringArgument( );
    int      first_slice                        = my_current_job.arguments[6].ReturnIntegerArgument( );
    int      last_slice                         = my_current_job.arguments[7].ReturnIntegerArgument( );
    int      sampling_step                      = my_current_job.arguments[8].ReturnIntegerArgument( );
    int      box_size                           = my_current_job.arguments[9].ReturnIntegerArgument( );
    bool     use_fixed_threshold                = my_current_job.arguments[10].ReturnBoolArgument( );
    float    fixed_threshold                    = my_current_job.arguments[11].ReturnFloatArgument( );
    float    threshold_snr                      = my_current_job.arguments[12].ReturnFloatArgument( );
    float    confidence_level                   = my_current_job.arguments[13].ReturnFloatArgument( );
    bool     randomize_phases                   = my_current_job.arguments[14].ReturnBoolArgument( );
    float    resolution_for_phase_randomization = my_current_job.arguments[15].ReturnFloatArgument( );
    bool     whiten_half_maps                   = my_current_job.arguments[16].ReturnBoolArgument( );
    int      padding_factor                     = my_current_job.arguments[17].ReturnIntegerArgument( );

    // Read volumes from disk
    ImageFile input_file_one(input_volume_one_fn.ToStdString( ), false);
    ImageFile input_file_two(input_volume_two_fn.ToStdString( ), false);
    ImageFile input_file_mask(input_volume_mask_fn.ToStdString( ), false);
    Image     input_volume_one;
    Image     input_volume_two;
    Image     input_volume_mask;
    input_volume_one.ReadSlices(&input_file_one, 1, input_file_one.ReturnNumberOfSlices( ));
    input_volume_two.ReadSlices(&input_file_two, 1, input_file_two.ReturnNumberOfSlices( ));
    input_volume_mask.ReadSlices(&input_file_mask, 1, input_file_mask.ReturnNumberOfSlices( ));
    MyDebugAssertTrue(input_volume_one.HasSameDimensionsAs(&input_volume_two), "The two input volumes do not have the same dimensions");
    MyDebugAssertTrue(input_volume_one.HasSameDimensionsAs(&input_volume_mask), "The mask does not have same dimensions as input volumes");

    /*
	 * Randomize phases beyond a given resolution. This can be useful
	 * when testing resolution criteria
	 */
    if ( randomize_phases && resolution_for_phase_randomization > 0.0 ) {
        wxPrintf("Randomizing phases beyond %0.2f A\n", resolution_for_phase_randomization);
        input_volume_one.ForwardFFT( );
        input_volume_two.ForwardFFT( );
        input_volume_one.RandomisePhases(pixel_size / resolution_for_phase_randomization);
        input_volume_two.RandomisePhases(pixel_size / resolution_for_phase_randomization);
        input_volume_one.BackwardFFT( );
        input_volume_two.BackwardFFT( );
    }

    // Prepare output volume
    Image local_resolution_volume(input_volume_one);
    local_resolution_volume.SetToConstant(-1.0);

    //
    LocalResolutionEstimator* estimator = new LocalResolutionEstimator( );
    estimator->SetAllUserParameters(&input_volume_one, &input_volume_two, &input_volume_mask, first_slice, last_slice, sampling_step, pixel_size, box_size, threshold_snr, confidence_level, use_fixed_threshold, fixed_threshold, my_symmetry, whiten_half_maps, padding_factor);
    estimator->EstimateLocalResolution(&local_resolution_volume);

    // Write output volume to disk
    local_resolution_volume.WriteSlicesAndFillHeader(output_volume_fn.ToStdString( ), pixel_size);

    // Cleanup
    delete estimator;

    return true;
}
