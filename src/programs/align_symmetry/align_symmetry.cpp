#include "../../core/core_headers.h"

class
        AlignSymmetryApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(AlignSymmetryApp)

// override the DoInteractiveUserInput

void AlignSymmetryApp::DoInteractiveUserInput( ) {
    wxString input_volume_file;
    wxString output_volume_file_no_sym;
    wxString output_volume_file_with_sym;
    wxString wanted_symmetry;
    float    start_angle_for_search;
    float    end_angle_for_search;
    float    initial_angular_step;

    UserInput* my_input = new UserInput("AlignSymmetry", 1.00);

    input_volume_file           = my_input->GetFilenameFromUser("Input volume", "The volume you want to align", "my_volume.mrc", true);
    wanted_symmetry             = my_input->GetSymmetryFromUser("Symmetry to align to", "The symmetry you want to align to", "C2");
    output_volume_file_no_sym   = my_input->GetFilenameFromUser("Output aligned volume", "The volume that has been aligned, but not symmetrised", "my_volume_ali.mrc", false);
    output_volume_file_with_sym = my_input->GetFilenameFromUser("Output symmetrised volume", "The volume that has been aligned and symmetrised", "my_volume_ali_sym.mrc", false);
    start_angle_for_search      = my_input->GetFloatFromUser("Start angle for search (degrees)", "Angle at which to begin the search on each axis", "-90.0");
    end_angle_for_search        = my_input->GetFloatFromUser("End angle for search (degrees)", "Angle at which to end the search on each axis", "90.0");
    initial_angular_step        = my_input->GetFloatFromUser("Initial angular search step (degrees)", "angular step for the initial search", "5.0");

    // start_angle_for_search = -90;
    // end_angle_for_search   = 90;
    delete my_input;

    int current_class = 0;
    my_current_job.Reset(8);
    my_current_job.ManualSetArguments("ttttfffi", input_volume_file.ToUTF8( ).data( ),
                                      wanted_symmetry.ToUTF8( ).data( ),
                                      output_volume_file_no_sym.ToUTF8( ).data( ),
                                      output_volume_file_with_sym.ToUTF8( ).data( ),
                                      start_angle_for_search,
                                      end_angle_for_search,
                                      initial_angular_step,
                                      current_class);
}

// override the do calculation method which will be what is actually run..

bool AlignSymmetryApp::DoCalculation( ) {

    wxString input_volume_file           = my_current_job.arguments[0].ReturnStringArgument( );
    wxString wanted_symmetry             = my_current_job.arguments[1].ReturnStringArgument( );
    wxString output_volume_file_no_sym   = my_current_job.arguments[2].ReturnStringArgument( );
    wxString output_volume_file_with_sym = my_current_job.arguments[3].ReturnStringArgument( );
    float    start_angle_for_search      = my_current_job.arguments[4].ReturnFloatArgument( );
    float    end_angle_for_search        = my_current_job.arguments[5].ReturnFloatArgument( );
    float    initial_angular_step        = my_current_job.arguments[6].ReturnFloatArgument( );
    int      current_class               = my_current_job.arguments[7].ReturnIntegerArgument( );

    float input_pixel_size;

    float current_x_angle;
    float current_y_angle;
    float current_z_angle;

    float average_x_shift = 0.0f;
    float average_y_shift = 0.0f;
    float average_z_shift = 0.0f;

    float total_shift_x = 0.0f;
    float total_shift_y = 0.0f;
    float total_shift_z = 0.0f;

    float rotated_x_shift;
    float rotated_y_shift;
    float rotated_z_shift;

    ProgressBar* progress;

    float best_x;
    float best_y;
    float best_z;

    float best_x_this_round;
    float best_y_this_round;
    float best_z_this_round;

    long pixel_counter;
    long address;
    int  symmetry_counter;

    float low_search_limit_x;
    float high_search_limit_x;

    float low_search_limit_y;
    float high_search_limit_y;

    float low_search_limit_z;
    float high_search_limit_z;

    float current_correlation;
    float best_correlation = -FLT_MAX;

    float end_angular_step = 0.1f;
    float current_angular_step;
    float start_angular_step = initial_angular_step;

    Image input_volume;
    Image output_volume;

    Image original_projection_image;
    Image check1_projection_image;
    Image check2_projection_image;

    Image current_projection_image;
    Image current_check1_projection_image;
    Image current_check2_projection_image;

    Image buffer_image;

    Peak  alignment_peak;
    float current_sum_score;

    ReconstructedVolume input_3d;
    MRCFile*            input_file = new MRCFile(input_volume_file.ToStdString( ));
    input_pixel_size               = input_file->ReturnPixelSize( );

    input_3d.InitWithDimensions(input_file->ReturnXSize( ), input_file->ReturnYSize( ), input_file->ReturnZSize( ), 1, wanted_symmetry);
    input_3d.density_map->ReadSlices(input_file, 1, input_3d.density_map->logical_z_dimension);
    input_3d.density_map->ZeroFloatAndNormalize(1, input_3d.density_map->logical_x_dimension);
    input_3d.PrepareForProjections(0.0, 0.5, false, false);

    original_projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    current_projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    check1_projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    current_check1_projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    check2_projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    current_check2_projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);

    input_volume.ReadSlices(input_file, 1, input_file->ReturnNumberOfSlices( ));
    input_volume.ZeroFloatAndNormalize(1, input_volume.logical_x_dimension);
    output_volume.Allocate(input_volume.logical_x_dimension, input_volume.logical_y_dimension, input_volume.logical_z_dimension);

    // input_3d.density_map->CopyFrom(&input_volume);
    // input_3d.density_map->ForwardFFT( );
    // input_3d.density_map->SwapRealSpaceQuadrants( ); // revert

    delete input_file;

    RotationMatrix current_matrix;
    RotationMatrix check_matrix1;
    RotationMatrix check_matrix2;
    RotationMatrix temp_matrix;
    RotationMatrix inverse_matrix;
    RotationMatrix identity_matrix;
    RotationMatrix combined_matrix;

    SymmetryMatrix symmetry_matrices;

    identity_matrix.SetToIdentity( );

    int total_number_to_search = 0;
    int number_searched        = 0;

    if ( is_running_locally == true ) {
        current_angular_step = start_angular_step * 5.0f;
        best_x_this_round    = 0.0f;
        best_y_this_round    = 0.0f;
        best_z_this_round    = 0.0f;

        low_search_limit_x  = start_angle_for_search;
        high_search_limit_x = end_angle_for_search;

        low_search_limit_y  = -90.0f;
        high_search_limit_y = 90.0f;

        low_search_limit_z  = -90.f;
        high_search_limit_z = 90.f;

        while ( current_angular_step > end_angular_step ) {
            current_angular_step /= 5.0f;
            if ( current_angular_step < end_angular_step )
                current_angular_step = end_angular_step;

            for ( current_z_angle = low_search_limit_z; current_z_angle <= high_search_limit_z; current_z_angle += current_angular_step ) {
                for ( current_y_angle = low_search_limit_y; current_y_angle <= high_search_limit_y; current_y_angle += current_angular_step ) {
                    for ( current_x_angle = low_search_limit_x; current_x_angle <= high_search_limit_x; current_x_angle += current_angular_step ) {
                        total_number_to_search++;
                    }
                }
            }

            low_search_limit_x  = best_x_this_round - current_angular_step;
            high_search_limit_x = best_x_this_round + current_angular_step;

            low_search_limit_y  = best_y_this_round - current_angular_step;
            high_search_limit_y = best_y_this_round + current_angular_step;

            low_search_limit_z  = best_z_this_round - current_angular_step;
            high_search_limit_z = best_z_this_round + current_angular_step;
        }
    }

    symmetry_matrices.Init(wanted_symmetry);

    if ( is_running_locally == true ) {
        wxPrintf("\nSearching For Rotation...\n\n");
        progress = new ProgressBar(total_number_to_search);
    }

    check_matrix1.SetToRotation(-45, -45, -45); // I kind of chose these randomly
    check_matrix2.SetToRotation(15, 70, -15);

    current_angular_step = start_angular_step * 5.0f;

    best_x_this_round = 0.0f;
    best_y_this_round = 0.0f;
    best_z_this_round = 0.0f;

    low_search_limit_x  = start_angle_for_search;
    high_search_limit_x = end_angle_for_search;

    low_search_limit_y  = -90.0f;
    high_search_limit_y = 90.0f;

    low_search_limit_z  = -90.f;
    high_search_limit_z = 90.f;

    while ( current_angular_step > end_angular_step ) {
        current_angular_step /= 5.0f;
        if ( current_angular_step < end_angular_step )
            current_angular_step = end_angular_step;

        for ( current_z_angle = low_search_limit_z; current_z_angle <= high_search_limit_z; current_z_angle += current_angular_step ) {
            for ( current_y_angle = low_search_limit_y; current_y_angle <= high_search_limit_y; current_y_angle += current_angular_step ) {
                for ( current_x_angle = low_search_limit_x; current_x_angle <= high_search_limit_x; current_x_angle += current_angular_step ) {
                    current_sum_score = 0.0f;
                    current_matrix.SetToRotation(current_x_angle, current_y_angle, current_z_angle);

                    input_3d.density_map->ExtractSliceByRotMatrix(original_projection_image, current_matrix);

                    temp_matrix = current_matrix * check_matrix1;
                    input_3d.density_map->ExtractSliceByRotMatrix(check1_projection_image, temp_matrix);

                    temp_matrix = current_matrix * check_matrix2;
                    input_3d.density_map->ExtractSliceByRotMatrix(check2_projection_image, temp_matrix);

                    original_projection_image.ZeroCentralPixel( );
                    original_projection_image.DivideByConstant(sqrt(original_projection_image.ReturnSumOfSquares( )));

                    check1_projection_image.ZeroCentralPixel( );
                    check1_projection_image.DivideByConstant(sqrt(check1_projection_image.ReturnSumOfSquares( )));

                    check2_projection_image.ZeroCentralPixel( );
                    check2_projection_image.DivideByConstant(sqrt(check2_projection_image.ReturnSumOfSquares( )));

                    for ( symmetry_counter = 1; symmetry_counter < symmetry_matrices.number_of_matrices; symmetry_counter++ ) {
                        temp_matrix = current_matrix * symmetry_matrices.rot_mat[symmetry_counter];
                        input_3d.density_map->ExtractSliceByRotMatrix(current_projection_image, temp_matrix);

                        current_projection_image.ZeroCentralPixel( );
                        current_projection_image.DivideByConstant(sqrt(current_projection_image.ReturnSumOfSquares( )));

                        current_projection_image.CalculateCrossCorrelationImageWith(&original_projection_image);
                        alignment_peak = current_projection_image.FindPeakWithIntegerCoordinates( );
                        current_sum_score += alignment_peak.value;

                        temp_matrix = current_matrix * symmetry_matrices.rot_mat[symmetry_counter];
                        temp_matrix = temp_matrix * check_matrix1;
                        input_3d.density_map->ExtractSliceByRotMatrix(current_check1_projection_image, temp_matrix);

                        current_check1_projection_image.ZeroCentralPixel( );
                        current_check1_projection_image.DivideByConstant(sqrt(current_check1_projection_image.ReturnSumOfSquares( )));

                        current_check1_projection_image.CalculateCrossCorrelationImageWith(&check1_projection_image);
                        alignment_peak = current_check1_projection_image.FindPeakWithIntegerCoordinates( );
                        current_sum_score += alignment_peak.value;

                        temp_matrix = current_matrix * symmetry_matrices.rot_mat[symmetry_counter];
                        temp_matrix = temp_matrix * check_matrix2;
                        input_3d.density_map->ExtractSliceByRotMatrix(current_check2_projection_image, temp_matrix);

                        current_check2_projection_image.ZeroCentralPixel( );
                        current_check2_projection_image.DivideByConstant(sqrt(current_check2_projection_image.ReturnSumOfSquares( )));

                        current_check2_projection_image.CalculateCrossCorrelationImageWith(&check2_projection_image);
                        alignment_peak = current_check2_projection_image.FindPeakWithIntegerCoordinates( );
                        current_sum_score += alignment_peak.value;
                    }

                    if ( current_sum_score > best_correlation ) {
                        best_correlation = current_sum_score;
                        best_x           = current_x_angle;
                        best_y           = current_y_angle;
                        best_z           = current_z_angle;
                        //wxPrintf("Results = %f, %f, %f (%i) = %f\n", best_x, best_y, best_z, number_searched + 1, best_correlation);
                    }

                    number_searched++;

                    if ( is_running_locally == true ) {
                        progress->Update(number_searched);
                    }
                    else {
                        float      temp_float  = number_searched;
                        JobResult* temp_result = new JobResult;
                        temp_result->SetResult(1, &temp_float);
                        AddJobToResultQueue(temp_result);
                    }
                }
            }
        }

        best_x_this_round = best_x;
        best_y_this_round = best_y;
        best_z_this_round = best_z;

        low_search_limit_x  = best_x_this_round - current_angular_step;
        high_search_limit_x = best_x_this_round + current_angular_step;

        low_search_limit_y  = best_y_this_round - current_angular_step;
        high_search_limit_y = best_y_this_round + current_angular_step;

        low_search_limit_z  = best_z_this_round - current_angular_step;
        high_search_limit_z = best_z_this_round + current_angular_step;
    }

    if ( is_running_locally == true ) {
        delete progress;
    }

    // work out the shifts..

    if ( is_running_locally == true ) {
        wxPrintf("\nSearching For Shifts...\n");
    }

    // Top down projection - gives an estimate of X/Y shifts

    //best_x = 0.0f;
    //best_y = 0.0f;
    //best_z = 0.0f;

    // moved_shift.mrc

    //best_x = -30.7f;
    //best_y = -14.0f;
    //best_z = 25.6f;

    // betagal.mrc

    //	best_x = 60.8f;
    //	best_y = -49.7f;
    //	best_z = 41.7f;

    current_matrix.SetToRotation(best_x, best_y, best_z);
    inverse_matrix = current_matrix.ReturnTransposed( );
    input_3d.density_map->ExtractSliceByRotMatrix(original_projection_image, current_matrix);

    average_x_shift = 0.0f;
    average_y_shift = 0.0f;
    average_z_shift = 0.0f;

    for ( symmetry_counter = 1; symmetry_counter < symmetry_matrices.number_of_matrices; symmetry_counter++ ) {
        temp_matrix = current_matrix * symmetry_matrices.rot_mat[symmetry_counter];
        input_3d.density_map->ExtractSliceByRotMatrix(current_projection_image, temp_matrix);

        current_projection_image.ZeroCentralPixel( );
        current_projection_image.DivideByConstant(sqrt(current_projection_image.ReturnSumOfSquares( )));

        current_projection_image.CalculateCrossCorrelationImageWith(&original_projection_image);
        alignment_peak = current_projection_image.FindPeakWithParabolaFit( );
        inverse_matrix.RotateCoords(alignment_peak.x, alignment_peak.y, alignment_peak.z, rotated_x_shift, rotated_y_shift, rotated_z_shift);
        average_x_shift += rotated_x_shift;
        average_y_shift += rotated_y_shift;
        average_z_shift += rotated_z_shift;
    }

    average_x_shift /= float(symmetry_matrices.number_of_matrices);
    average_y_shift /= float(symmetry_matrices.number_of_matrices);
    average_z_shift /= float(symmetry_matrices.number_of_matrices);

    current_matrix.RotateCoords(average_x_shift, average_y_shift, average_z_shift, rotated_x_shift, rotated_y_shift, rotated_z_shift);
    //	wxPrintf("shifts1 = %f, %f, %f\n", rotated_x_shift, rotated_y_shift, rotated_z_shift);
    total_shift_x += rotated_x_shift;
    total_shift_y += rotated_y_shift;
    // z_shift should be zero

    // do a z alignment, if the symmetry is not C

    //if (wanted_symmetry[0] != 'C' && wanted_symmetry[0] != 'c')
    {

        check_matrix1.SetToRotation(0, 90, 0);
        combined_matrix = current_matrix * check_matrix1;
        inverse_matrix  = combined_matrix.ReturnTransposed( );

        input_3d.density_map->ExtractSliceByRotMatrix(check1_projection_image, combined_matrix);
        check1_projection_image.ZeroCentralPixel( );
        check1_projection_image.DivideByConstant(sqrt(check1_projection_image.ReturnSumOfSquares( )));

        average_x_shift = 0;
        average_y_shift = 0;
        average_z_shift = 0;

        for ( symmetry_counter = 1; symmetry_counter < symmetry_matrices.number_of_matrices; symmetry_counter++ ) {
            temp_matrix = current_matrix * symmetry_matrices.rot_mat[symmetry_counter];
            temp_matrix = temp_matrix * check_matrix1;
            input_3d.density_map->ExtractSliceByRotMatrix(current_check1_projection_image, temp_matrix);

            current_check1_projection_image.ZeroCentralPixel( );
            current_check1_projection_image.DivideByConstant(sqrt(current_check1_projection_image.ReturnSumOfSquares( )));

            current_check1_projection_image.CalculateCrossCorrelationImageWith(&check1_projection_image);
            alignment_peak = current_check1_projection_image.FindPeakWithParabolaFit( );

            inverse_matrix.RotateCoords(alignment_peak.x, alignment_peak.y, alignment_peak.z, rotated_x_shift, rotated_y_shift, rotated_z_shift);
            average_x_shift += rotated_x_shift;
            average_y_shift += rotated_y_shift;
            average_z_shift += rotated_z_shift;
        }

        average_x_shift /= float(symmetry_matrices.number_of_matrices);
        average_y_shift /= float(symmetry_matrices.number_of_matrices);
        average_z_shift /= float(symmetry_matrices.number_of_matrices);

        combined_matrix.RotateCoords(average_x_shift, average_y_shift, average_z_shift, rotated_x_shift, rotated_y_shift, rotated_z_shift);
        //wxPrintf("shifts2 = %f, %f, %f\n", rotated_x_shift, rotated_y_shift, rotated_z_shift);
        total_shift_y += rotated_y_shift;
        total_shift_z += -rotated_x_shift;

        // Other 90 angle..

        check_matrix1.SetToRotation(90, 0, 0);
        combined_matrix = current_matrix * check_matrix1;
        inverse_matrix  = combined_matrix.ReturnTransposed( );

        input_3d.density_map->ExtractSliceByRotMatrix(check1_projection_image, combined_matrix);
        check1_projection_image.ZeroCentralPixel( );
        check1_projection_image.DivideByConstant(sqrt(check1_projection_image.ReturnSumOfSquares( )));

        average_x_shift = 0;
        average_y_shift = 0;
        average_z_shift = 0;

        for ( symmetry_counter = 1; symmetry_counter < symmetry_matrices.number_of_matrices; symmetry_counter++ ) {
            temp_matrix = current_matrix * symmetry_matrices.rot_mat[symmetry_counter];
            temp_matrix = temp_matrix * check_matrix1;
            input_3d.density_map->ExtractSliceByRotMatrix(current_check1_projection_image, temp_matrix);

            current_check1_projection_image.ZeroCentralPixel( );
            current_check1_projection_image.DivideByConstant(sqrt(current_check1_projection_image.ReturnSumOfSquares( )));

            current_check1_projection_image.CalculateCrossCorrelationImageWith(&check1_projection_image);
            alignment_peak = current_check1_projection_image.FindPeakWithParabolaFit( );

            inverse_matrix.RotateCoords(alignment_peak.x, alignment_peak.y, alignment_peak.z, rotated_x_shift, rotated_y_shift, rotated_z_shift);
            average_x_shift += rotated_x_shift;
            average_y_shift += rotated_y_shift;
            average_z_shift += rotated_z_shift;
        }

        average_x_shift /= float(symmetry_matrices.number_of_matrices);
        average_y_shift /= float(symmetry_matrices.number_of_matrices);
        average_z_shift /= float(symmetry_matrices.number_of_matrices);

        combined_matrix.RotateCoords(average_x_shift, average_y_shift, average_z_shift, rotated_x_shift, rotated_y_shift, rotated_z_shift);
        //	wxPrintf("shifts3 = %f, %f, %f\n", rotated_x_shift, rotated_y_shift, rotated_z_shift);

        total_shift_x += rotated_x_shift;
        total_shift_z += rotated_y_shift;

        total_shift_x /= 2.0f;
        total_shift_y /= 2.0f;
        total_shift_z /= 2.0f;
    }

    if ( is_running_locally == true ) {
        inverse_matrix = current_matrix.ReturnTransposed( );

        buffer_image.CopyFrom(&input_volume);
        buffer_image.Rotate3DThenShiftThenApplySymmetry(inverse_matrix, total_shift_x, total_shift_y, total_shift_z, buffer_image.logical_x_dimension / 2.0f);

        MRCFile* output_file;
        output_file = new MRCFile(output_volume_file_no_sym.ToStdString( ), true);
        buffer_image.WriteSlices(output_file, 1, input_volume.logical_z_dimension);
        output_file->SetPixelSize(input_pixel_size);
        delete output_file;

        input_volume.Rotate3DThenShiftThenApplySymmetry(inverse_matrix, total_shift_x, total_shift_y, total_shift_z, buffer_image.logical_x_dimension / 2.0f, wanted_symmetry);
        output_file = new MRCFile(output_volume_file_with_sym.ToStdString( ), true);
        input_volume.WriteSlices(output_file, 1, input_volume.logical_z_dimension);
        output_file->SetPixelSize(input_pixel_size);
        delete output_file;

        wxPrintf("\nResults :-\n\nX-Rot = %.2f degrees\nY-Rot =  %.2f degrees\nZ-Rot = %.2f degrees\n\nX-Shift = %.2f pix.\nY-Shift = %.2f pix.\nZ-Shift = %.2f pix. Best Score: %3.3f\n", best_x, best_y, best_z, total_shift_x, total_shift_y, total_shift_z, best_correlation);
        wxPrintf("\nAlignSymmetry: Normal termination\n\n");
    }
    else {
        float result[8];

        result[0] = best_x;
        result[1] = best_y;
        result[2] = best_z;
        result[3] = total_shift_x;
        result[4] = total_shift_y;
        result[5] = total_shift_z;
        result[6] = best_correlation;
        result[7] = current_class;

        my_result.SetResult(8, result);
    }

    return true;
}
