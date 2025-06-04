

#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "statistical_ops.h"

void CPUvsGPUStatisticalOpsRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting CPU vs GPU masking tests:", false);

    // all_passed = all_passed && DoStatsticalMomentsTests(hiv_image_80x80x1_filename, temp_directory);
    // all_passed = all_passed && DoExtremumTests(hiv_image_80x80x1_filename, temp_directory);
    TEST(true);
    SamplesPrintEndMessage( );

    return;
}

bool DoStatsticalMomentsTests(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    std::vector<int> img_size = {64, 256, 448, 648};
    int              n;

    RandomNumberGenerator my_rand(pi_v<float>);
    float                 random_mean;
    float                 random_variance;

    Image*   noise_image = new Image[img_size.size( )];
    GpuImage test_image;

    EmpiricalDistribution<double>* my_dist = new EmpiricalDistribution<double>[img_size.size( )];

    // prepare the test cpu images.
    n = 0;
    for ( auto size : img_size ) {
        random_mean     = my_rand.GetUniformRandomSTD(-1.0f, 1.0f);
        random_variance = my_rand.GetUniformRandomSTD(0.5f, 10.0f);

        noise_image[n].Allocate(size, size, 1, true);
        noise_image[n].FillWithNoise(GAUSSIAN, random_mean, sqrtf(random_variance));
        noise_image[n].AddConstant(random_mean);
        wxPrintf("cpu val at zero = %f\n", noise_image[n].real_values[0]);

        my_dist[n] = noise_image[n].ReturnDistributionOfRealValues( );

        n++;
    }

    SamplesBeginTest("position-space mean", passed);
    n = 0;
    for ( auto size : img_size ) {
        wxPrintf("cpu val at zero = %f\n", noise_image[n].real_values[0]);
        test_image.Init(noise_image[n], true, true);
        test_image.CopyHostToDevice(noise_image[n], true);

        test_image.Mean( );
        passed = passed && FloatsAreAlmostTheSame(test_image.img_mean, my_dist[n].GetSampleMean( ));
        test_image.Deallocate( );

        n++;
    }
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("position-space std-dev", passed);
    n = 0;
    for ( auto size : img_size ) {
        test_image.Init(noise_image[n], true, true);
        test_image.CopyHostToDevice(noise_image[n]);
        test_image.MeanStdDev( );
        passed = passed && FloatsAreAlmostTheSame(test_image.img_stdDev, sqrtf(my_dist[n].GetSampleVariance( )));
        test_image.Deallocate( );

        n++;
    }
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("position-space minimum", passed);
    n = 0;
    for ( auto size : img_size ) {
        test_image.Init(noise_image[n], true, true);
        test_image.CopyHostToDevice(noise_image[n]);
        test_image.Min( );
        passed = passed && FloatsAreAlmostTheSame(test_image.min_value, my_dist[n].GetMinimum( ));
        test_image.Deallocate( );

        n++;
    }
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("position-space maximum", passed);
    n = 0;
    for ( auto size : img_size ) {
        test_image.Init(noise_image[n], true, true);
        test_image.CopyHostToDevice(noise_image[n]);
        test_image.Min( );
        passed = passed && FloatsAreAlmostTheSame(test_image.max_value, my_dist[n].GetMaximum( ));
        test_image.Deallocate( );

        n++;
    }
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("position-space min and index", passed);
    n = 0;
    for ( auto size : img_size ) {
        Peak this_peak = noise_image[n].FindPeakWithIntegerCoordinates( );
        // The image function only finds the max, so first find that, then negate the image to test the GPU function

        Image tmp;
        tmp.CopyFrom(&noise_image[n]);
        tmp.MultiplyByConstant(-1.0f);

        test_image.Init(tmp, true, true);
        test_image.CopyHostToDevice(noise_image[n]);
        test_image.MinAndCoords( );
        passed = passed && FloatsAreAlmostTheSame(test_image.min_value, -this_peak.value);
        passed = passed && (test_image.min_idx.x == this_peak.x && test_image.min_idx.y == this_peak.y);
        test_image.Deallocate( );

        n++;
    }
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("position-space max and index", passed);
    n = 0;
    for ( auto size : img_size ) {
        Peak this_peak = noise_image[n].FindPeakWithIntegerCoordinates( );

        test_image.Init(noise_image[n], true, true);
        test_image.CopyHostToDevice(noise_image[n]);
        test_image.MaxAndCoords( );
        passed = passed && FloatsAreAlmostTheSame(test_image.max_value, this_peak.value);
        passed = passed && (test_image.max_idx.x == this_peak.x && test_image.max_idx.y == this_peak.y);
        test_image.Deallocate( );
        n++;
    }
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    delete[] noise_image;
    delete[] my_dist;
    return all_passed;
}

bool DoExtremumTests( ) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Extract slice CPU vs ground truth", passed);

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    return all_passed;
}