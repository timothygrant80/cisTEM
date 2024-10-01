#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#include "../../../core/core_headers.h"
#endif

#include "../common/common.h"
#include "../../../core/scattering_potential.h"
#include "simple_3d.h"

void Simple3dSimulationRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting a test to be a runner", false);

    wxString cistem_ref_dir = CheckForReferenceImages( );

    TEST(MyTest(cistem_ref_dir, temp_directory));

    SamplesPrintEndMessage( );

    return;
}

bool MyTest(const wxString& cistem_ref_dir, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    float acceptable_ccc = 0.99;

    wxString atomic_coordinates = cistem_ref_dir + "/6pch_updated.cif";

    // For a simple 3d conversion from atomic coordinates, via pdb mmcif or similar, we make a cubic volume
    // where the size is defined by the image volume we pass in to the methd.
    // FIXME: I get zero output for size 384 - add a size scan later

    constexpr int image_size = 320;
    Image         test_sim;
    Image         clean_copy;
    test_sim.Allocate(image_size, image_size, image_size, true, true);

    // Now we intiialize a scattering potential object which will parse the coordinates and setup a bunch of metadata
    ScatteringPotential sp = ScatteringPotential(atomic_coordinates, test_sim.logical_x_dimension);

    // The scattering potential needs to know a few things about our "imaging conditions"
    // namly pixel size and acceelration voltage.
    float pixel_size = 1.0;
    float voltage    = 300.0;
    sp.SetImagingParameters(pixel_size, voltage);

    // If we are using an alpha-fold prediction, the bfactors stored are confidence scores and need
    // to be convereted to bfactors. In this case we have "real" bfactors, so we don't need to do anything.
    bool is_from_alpha_fold = false;
    sp.InitPdbObject(is_from_alpha_fold);

    // Finally, we need to know what orientation the molecule should be placed in.
    // For translation, the default is to center atoms on their COM prior to rotation. This
    // may be overridden by passing a wanted COM as a double[3] array following is_from_alpha_fold.
    // Here we'll use the default and the identity matrix.
    RotationMatrix molecular_orientation;
    molecular_orientation.SetToIdentity( );

    // For our tests, we don't bother with parallelism.
    int wanted_number_of_threads = 1;
    sp.calc_scattering_potential(test_sim, molecular_orientation, wanted_number_of_threads);

    test_sim.QuickAndDirtyWriteSlices(temp_directory.ToStdString( ) + "/test_sim.mrc", 1, image_size, true, pixel_size);
    // we'll keep a clean copy of the original, unrotated image for re-use later.
    clean_copy.CopyFrom(&test_sim);

    // We can compare transformations to those imposed by the image processing functions in cisTEM.
    Image xformed_sim;
    Image xformed_map;
    xformed_map.CopyFrom(&clean_copy);
    xformed_map.ForwardFFT( );
    xformed_map.SwapRealSpaceQuadrants( );
    Image xformed_projection, xformed_sim_projection;

    xformed_projection.Allocate(xformed_map.logical_x_dimension, xformed_map.logical_y_dimension, 1, false, true);
    xformed_sim_projection.Allocate(xformed_map.logical_x_dimension, xformed_map.logical_y_dimension, 1, false, true);

    // Rotations are specified by Z(phi) Y(*theta) Z(psi) euler angles, which are proper Euler angles.
    // These are combined into a matrix and used as follows:
    // R(phi)*R(theata)*R(psi) * xyz' :: where xyz' is a column vector, and R() is a matrix defining a + = clockwise looking down the axis toward the origin

    SamplesBeginTest("Rotate (90,0,0)", passed);

    // We'll start with a simple rotation around Z
    AnglesAndShifts angles_and_shifts(90.f, 0.f, 0.f);
    AnglesAndShifts dummy_(0.f, 0.f, 0.f); // we need a dummy to extract the already rotated image from simulationin projection

    xformed_map.ExtractSlice(xformed_projection, angles_and_shifts);
    xformed_projection.SwapRealSpaceQuadrants( );
    xformed_projection.BackwardFFT( );
    // Now here is a confusing bit. The images in cistem are transformed by transforming their coordinate system and then interpolating the values.
    // This means we have a "passive" transformation, where R(phi)*R(theata)*R(psi) rotates the volume by R(-psi)*R(-theta)*R(-phi)
    // In the case of the simulator, the atomic coordinates are themselves transformed (no interpolation).
    // This means we ahve an "active" transformation, so we use the transfomation that produces such
    molecular_orientation.SetToEulerRotation(-angles_and_shifts.ReturnPsiAngle( ), -angles_and_shifts.ReturnThetaAngle( ), -angles_and_shifts.ReturnPhiAngle( ));
    sp.calc_scattering_potential(test_sim, molecular_orientation, wanted_number_of_threads);

    test_sim.ForwardFFT( );
    test_sim.SwapRealSpaceQuadrants( );
    test_sim.ExtractSlice(xformed_sim_projection, dummy_);
    xformed_sim_projection.SwapRealSpaceQuadrants( );
    xformed_sim_projection.BackwardFFT( );
    // Now let's compare the resulting projections
    // xformed_projection.QuickAndDirtyWriteSlice(temp_directory.ToStdString( ) + "/xformed_projection1.mrc", 1, false, pixel_size);
    // xformed_sim_projection.QuickAndDirtyWriteSlice(temp_directory.ToStdString( ) + "/xformed_sim_projection1.mrc", 1, false, pixel_size);

    passed = CompareRealValues(xformed_projection, xformed_sim_projection, acceptable_ccc);

    all_passed = passed ? all_passed : false;

    SamplesTestResult(passed);

    SamplesBeginTest("Rotate (117.,35.8,-20.f)", passed);

    // We'll start with a simple rotation around Z
    angles_and_shifts.GenerateEulerMatrices(117., 35.8, -20.f);

    xformed_map.ExtractSlice(xformed_projection, angles_and_shifts);
    xformed_projection.SwapRealSpaceQuadrants( );
    xformed_projection.BackwardFFT( );
    // Now here is a confusing bit. The images in cistem are transformed by transforming their coordinate system and then interpolating the values.
    // This means we have a "passive" transformation, where R(phi)*R(theata)*R(psi) rotates the volume by R(-psi)*R(-theta)*R(-phi)
    // In the case of the simulator, the atomic coordinates are themselves transformed (no interpolation).
    // This means we ahve an "active" transformation, so we use the transfomation that produces such
    molecular_orientation.SetToEulerRotation(-angles_and_shifts.ReturnPsiAngle( ), -angles_and_shifts.ReturnThetaAngle( ), -angles_and_shifts.ReturnPhiAngle( ));
    test_sim.is_in_real_space         = true;
    test_sim.object_is_centred_in_box = true;
    sp.calc_scattering_potential(test_sim, molecular_orientation, wanted_number_of_threads);

    test_sim.ForwardFFT( );
    test_sim.SwapRealSpaceQuadrants( );
    test_sim.ExtractSlice(xformed_sim_projection, dummy_);
    xformed_sim_projection.SwapRealSpaceQuadrants( );
    xformed_sim_projection.BackwardFFT( );
    // Now let's compare the resulting projections
    xformed_projection.QuickAndDirtyWriteSlice(temp_directory.ToStdString( ) + "/xformed_projection2.mrc", 1, false, pixel_size);
    xformed_sim_projection.QuickAndDirtyWriteSlice(temp_directory.ToStdString( ) + "/xformed_sim_projection2.mrc", 1, false, pixel_size);

    passed = CompareRealValues(xformed_projection, xformed_sim_projection, acceptable_ccc);

    all_passed = passed ? all_passed : false;

    SamplesTestResult(passed);

    SamplesBeginTest("Rotate and check for shift FUZZ(10)", passed);
    RandomNumberGenerator my_rand(pi_v<float>);
    for ( int i = 0; i < 10; i++ ) {

        // FIXME add a get random orientation with even sampling to functions and also use that elsewher
        angles_and_shifts.GenerateEulerMatrices(my_rand.GetUniformRandomSTD(0.f, 360.f), my_rand.GetUniformRandomSTD(0.f, 180.f), my_rand.GetUniformRandomSTD(0.f, 360.f));

        test_sim.is_in_real_space         = true;
        test_sim.object_is_centred_in_box = true;

        sp.calc_scattering_potential(test_sim, dummy_.euler_matrix, wanted_number_of_threads);
        xformed_map.CopyFrom(&test_sim);
        xformed_map.ForwardFFT( );
        xformed_map.SwapRealSpaceQuadrants( );

        xformed_map.ExtractSlice(xformed_projection, angles_and_shifts);
        xformed_projection.BackwardFFT( );
        xformed_projection.ZeroFloatAndNormalize( );

        molecular_orientation.SetToEulerRotation(-angles_and_shifts.ReturnPsiAngle( ), -angles_and_shifts.ReturnThetaAngle( ), -angles_and_shifts.ReturnPhiAngle( ));
        test_sim.is_in_real_space         = true;
        test_sim.object_is_centred_in_box = true;
        sp.calc_scattering_potential(test_sim, molecular_orientation, wanted_number_of_threads);

        test_sim.ForwardFFT( );
        test_sim.SwapRealSpaceQuadrants( );
        test_sim.ExtractSlice(xformed_sim_projection, dummy_);

        xformed_sim_projection.BackwardFFT( );
        xformed_sim_projection.ZeroFloatAndNormalize( );

        // float peak_scalar = 20.0f;
        // xformed_sim_projection.CalculatePhaseCrossCorrelationImageWith(xformed_projection, found_peak, peak_scalar, false);
        // // found_peak = xformed_sim_projection.FindPeakWithParabolaFit( );

        // found_peak.x = found_peak.x / (1 + peak_scalar);
        // found_peak.y = found_peak.y / (1 + peak_scalar);

        // current_shift = sqrtf(found_peak.x * found_peak.x + found_peak.y * found_peak.y);
        // avg_shift += current_shift;
        // counter++;

        // wxPrintf("\nFound peak %f at %f, %f for angle %i, %f", found_peak.value, found_peak.x, found_peak.y, rot_y, found_peak.x / sinf(deg_2_rad(rot_y)));

        // The image methods blur around the edges so we drop the CCC requirment a bit
        passed = CompareRealValues(xformed_projection, xformed_sim_projection, acceptable_ccc);

        all_passed = passed ? all_passed : false;
    }

    // wxPrintf("Best shift is %f at %f, %f, %f", min_shifts, bestx, besty, bestz);
    // wxPrintf("\n");
    // exit(1);

    SamplesTestResult(passed);

    return all_passed;
}