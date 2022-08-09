#include "../../core/core_headers.h"

class
        MoveVolumeXYZApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MoveVolumeXYZApp)

// override the DoInteractiveUserInput

void MoveVolumeXYZApp::DoInteractiveUserInput( ) {
    wxString input_volume_file;
    wxString output_volume_file;

    float x_rot;
    float y_rot;
    float z_rot;

    float x_shift;
    float y_shift;
    float z_shift;

    UserInput* my_input = new UserInput("MoveVolumeXYZ", 1.00);

    input_volume_file  = my_input->GetFilenameFromUser("Input volume", "The volume you want to align", "my_volume.mrc", true);
    output_volume_file = my_input->GetFilenameFromUser("Output moved volume", "The volume that has been moved", "my_volume_move.mrc", false);
    x_rot              = my_input->GetFloatFromUser("X-Rotation (degrees)", "wanted X rotation in degrees", "0.0");
    y_rot              = my_input->GetFloatFromUser("Y-Rotation (degrees)", "wanted Y rotation in degrees", "0.0");
    z_rot              = my_input->GetFloatFromUser("Z-Rotation (degrees)", "wanted Z rotation in degrees", "0.0");
    x_shift            = my_input->GetFloatFromUser("X-Shift (pixels)", "wanted X shift in pixels", "0.0");
    y_shift            = my_input->GetFloatFromUser("Y-Shift (pixels)", "wanted Y shift in pixels", "0.0");
    z_shift            = my_input->GetFloatFromUser("Z-Shift (pixels)", "wanted Z shift in pixels", "0.0");

    delete my_input;

    my_current_job.Reset(8);
    my_current_job.ManualSetArguments("ttffffff", input_volume_file.ToUTF8( ).data( ),
                                      output_volume_file.ToUTF8( ).data( ),
                                      x_rot,
                                      y_rot,
                                      z_rot,
                                      x_shift,
                                      y_shift,
                                      z_shift);
}

// override the do calculation method which will be what is actually run..

bool MoveVolumeXYZApp::DoCalculation( ) {

    wxString input_volume_file  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString output_volume_file = my_current_job.arguments[1].ReturnStringArgument( );
    float    x_rot              = my_current_job.arguments[2].ReturnFloatArgument( );
    float    y_rot              = my_current_job.arguments[3].ReturnFloatArgument( );
    float    z_rot              = my_current_job.arguments[4].ReturnFloatArgument( );
    float    x_shift            = my_current_job.arguments[5].ReturnFloatArgument( );
    float    y_shift            = my_current_job.arguments[6].ReturnFloatArgument( );
    float    z_shift            = my_current_job.arguments[7].ReturnFloatArgument( );

    float average_value_at_edge;

    Image input_volume;
    Image output_volume;

    RotationMatrix current_matrix;
    RotationMatrix inverse_matrix;

    MRCFile* input_file       = new MRCFile(input_volume_file.ToStdString( ));
    float    input_pixel_size = input_file->ReturnPixelSize( );

    input_volume.ReadSlices(input_file, 1, input_file->ReturnNumberOfSlices( ));
    output_volume.Allocate(input_volume.logical_x_dimension, input_volume.logical_y_dimension, input_volume.logical_z_dimension);
    output_volume.SetToConstant(0.0f);

    delete input_file;

    average_value_at_edge = input_volume.ReturnAverageOfRealValuesOnEdges( );
    input_volume.AddConstant(-average_value_at_edge);
    current_matrix.SetToRotation(x_rot, y_rot, z_rot);
    inverse_matrix = current_matrix.ReturnTransposed( );
    input_volume.Rotate3DThenShiftThenApplySymmetry(inverse_matrix, x_shift, y_shift, z_shift, FLT_MAX);
    input_volume.AddConstant(average_value_at_edge);

    MRCFile* output_file;
    output_file = new MRCFile(output_volume_file.ToStdString( ), true);
    input_volume.WriteSlices(output_file, 1, input_volume.logical_z_dimension);
    output_file->SetPixelSize(input_pixel_size);
    delete output_file;
    return true;
}
