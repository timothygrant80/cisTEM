#include "../../core/core_headers.h"

class
        EstimateBeamTiltApp : public MyApp {
  public:
    bool DoCalculation( );

  private:
};
IMPLEMENT_APP(EstimateBeamTiltApp)

// override the do calculation method which will be what is actually run..

bool EstimateBeamTiltApp::DoCalculation( ) {

    wxString input_phase_difference_image = my_current_job.arguments[0].ReturnStringArgument( );
    float    pixel_size                   = my_current_job.arguments[1].ReturnFloatArgument( );
    float    voltage_kV                   = my_current_job.arguments[2].ReturnFloatArgument( );
    float    spherical_aberration_mm      = my_current_job.arguments[3].ReturnFloatArgument( );
    int      first_position_to_search     = my_current_job.arguments[4].ReturnIntegerArgument( );
    int      last_position_to_search      = my_current_job.arguments[5].ReturnIntegerArgument( );

    float score;

    CTF   input_ctf;
    Image phase_difference_sum;
    Image temp_image;
    Image beamtilt_image;

    float beamtilt_x;
    float beamtilt_y;
    float particle_shift_x;
    float particle_shift_y;
    float phase_multiplier = 1.0f;

    input_ctf.Init(voltage_kV, spherical_aberration_mm, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, pixel_size, 0.0f);
    phase_difference_sum.QuickAndDirtyReadSlice(input_phase_difference_image.ToStdString( ), 1);
    phase_difference_sum.ForwardFFT( );

    temp_image.Allocate(phase_difference_sum.logical_x_dimension, phase_difference_sum.logical_y_dimension, 1);
    beamtilt_image.Allocate(phase_difference_sum.logical_x_dimension, phase_difference_sum.logical_y_dimension, 1);

    score = phase_difference_sum.FindBeamTilt(input_ctf, pixel_size, temp_image, beamtilt_image, phase_difference_sum, beamtilt_x, beamtilt_y, particle_shift_x, particle_shift_y, phase_multiplier, is_running_locally, first_position_to_search, last_position_to_search, this);

    float result_data[5];
    result_data[0] = score;
    result_data[1] = beamtilt_x;
    result_data[2] = beamtilt_y;
    result_data[3] = particle_shift_x;
    result_data[4] = particle_shift_y;

    my_result.SetResult(5, result_data);

    return true;
}
