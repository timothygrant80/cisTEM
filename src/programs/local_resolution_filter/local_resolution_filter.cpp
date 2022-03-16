#include "../../core/core_headers.h"

class
        LocalResolutionFilter : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(LocalResolutionFilter)

// override the DoInteractiveUserInput

void LocalResolutionFilter::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("LocalResolutionFilter", 0.1);

    wxString input_volume         = my_input->GetFilenameFromUser("Input volume", "The volume to be filtered", "my_reconstruction.mrc", true);
    wxString local_resolution_map = my_input->GetFilenameFromUser("Local resolution map", "A volume with a local resolution estimate at each voxel", "local_resolution.mrc", true);
    wxString output_volume        = my_input->GetFilenameFromUser("Output volume", "This volume will be filtered", "my_filtered_reconstruction.mrc", false);
    float    pixel_size           = my_input->GetFloatFromUser("Pixel size (A)", "Pixel size of the map in Angstroms", "1.0", 0.000001);

    delete my_input;

    my_current_job.Reset(4);
    my_current_job.ManualSetArguments("tttf", input_volume.ToUTF8( ).data( ), local_resolution_map.ToUTF8( ).data( ), output_volume.ToUTF8( ).data( ), pixel_size);
}

bool LocalResolutionFilter::DoCalculation( ) {
    wxString input_volume_fn         = my_current_job.arguments[0].ReturnStringArgument( );
    wxString local_resolution_map_fn = my_current_job.arguments[1].ReturnStringArgument( );
    wxString output_volume_fn        = my_current_job.arguments[2].ReturnStringArgument( );
    float    pixel_size              = my_current_job.arguments[3].ReturnFloatArgument( );

    // Read volumes from disk
    ImageFile input_volume_file(input_volume_fn.ToStdString( ), false);
    ImageFile local_resolution_map_file(local_resolution_map_fn.ToStdString( ), false);
    Image     input_volume;
    Image     local_resolution_map;
    input_volume.ReadSlices(&input_volume_file, 1, input_volume_file.ReturnNumberOfSlices( ));
    local_resolution_map.ReadSlices(&local_resolution_map_file, 1, local_resolution_map_file.ReturnNumberOfSlices( ));
    MyDebugAssertTrue(input_volume.HasSameDimensionsAs(&local_resolution_map), "The two input volumes do not have the same dimensions");

    // Filter it
    const int number_of_levels = 20;
    input_volume.ApplyLocalResolutionFilter(local_resolution_map, pixel_size, number_of_levels);

    // Write the result to disk
    input_volume.WriteSlicesAndFillHeader(output_volume_fn.ToStdString( ), pixel_size);

    return true;
}
