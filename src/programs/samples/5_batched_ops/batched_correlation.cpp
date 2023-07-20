#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"
#include "../../refine3d/batched_search.h"

#include "../common/common.h"
#include "batched_correlation.h"

// #define DO_EXPLICIT_BROADCAST

void BatchedCorrelationRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting batched_ops tests:", false);

    TEST(DoBatchedCorrelationTest(hiv_image_80x80x1_filename, temp_directory));

    SamplesPrintEndMessage( );

    return;
}

bool DoBatchedCorrelationTest(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    MRCFile input_file(hiv_images_80x80x10_filename.ToStdString( ), false);

    const bool test_mirror = false;

    Image    ref_img, test_imgs;
    GpuImage d_ref_img, d_test_imgs;

    Peak* peak_buffer;
    Peak* d_peak_buffer;

    // Create a stack of noisy images from the first image in the hiv_images_80x80x10 stack.
    // Each will have a different SNR and hence different CC.
    // The variable noise is to mimic the in-plane rotation cache in euler_search[_gpu].cpp, i.e. each noisy image is a standin
    // for the rotated search image.
    constexpr int n_search_images = 20;
    Image         seq_rotation_cache[n_search_images];
    GpuImage      d_seq_rotation_cache[n_search_images];

    for ( int i = 0; i < n_search_images; i++ ) {
        seq_rotation_cache[i].ReadSlice(&input_file, 1);
        // First norm is to make sure the noise and image values are on the same scale
        seq_rotation_cache[i].ZeroFloatAndNormalize( );
        seq_rotation_cache[i].AddNoiseFromNormalDistribution(0, n_search_images / (1.f + float(i)));
        // Second norm is to make sure the image and reference values are on the same scale for correlation.
        seq_rotation_cache[i].ZeroFloatAndNormalize( );
        seq_rotation_cache[i].ForwardFFT( );
        seq_rotation_cache[i].complex_values[0] = 0.f + I * 0.f;
        // seq_rotation_cache[i].QuickAndDirtyWriteSlice("/tmp/seq_rotation_cache_" + std::to_string(i) + ".mrc", 1);
        d_seq_rotation_cache[i].Init(seq_rotation_cache[i]);
        d_seq_rotation_cache[i].CopyHostToDeviceAndSynchronize( );
    }

    // A single reference image is used for all tests. This is to mimimc one iteration of the
    // outer loop through the stack of reference projections ([Gpu]Image* projections in euler_search[_gpu].cpp).

    std::array<float, n_search_images> correlation_results_ground_truth;
    std::array<float, n_search_images> correlation_results;

    // Setup ground truth, we'll use non-batched ops
    ref_img.ReadSlice(&input_file, 1);
    ref_img.ZeroFloatAndNormalize( );
    ref_img.ForwardFFT( );
    ref_img.complex_values[0] = 0.f + I * 0.f;
    d_ref_img.Init(ref_img);
    d_ref_img.CopyHostToDeviceAndSynchronize( );

    bool is_for_ground_truth = true;
    // First we'll get the ground truth results
    RunBatchedCorrelation(d_ref_img, d_seq_rotation_cache, n_search_images, 1, test_mirror, correlation_results_ground_truth.data( ), is_for_ground_truth);

    for ( auto cc : correlation_results_ground_truth ) {
        wxPrintf("%f\n", cc);
    }

    is_for_ground_truth = false;
    std::string test_name;
    std::string test_base = "Batch Size ";
    for ( int wanted_batch_size = 1; wanted_batch_size < 12; wanted_batch_size++ ) {
        test_name = test_base + std::to_string(wanted_batch_size) + " ";

        SamplesBeginTest(test_name.c_str( ), passed);

        correlation_results.fill(0.f);

        RunBatchedCorrelation(d_ref_img, d_seq_rotation_cache, n_search_images, wanted_batch_size, test_mirror, correlation_results.data( ), is_for_ground_truth);

        // Because we added less and less noise, the cc max will be at the largest index in each batch
        int counter = 0;
        wxPrintf("\n");
        for ( int cc = wanted_batch_size - 1; cc < correlation_results.size( ); cc += wanted_batch_size ) {
            wxPrintf("%i %i %f\n", wanted_batch_size, cc, correlation_results_ground_truth[cc] - correlation_results[counter]);
            passed = passed && (RelativeErrorIsLessThanEpsilon(correlation_results_ground_truth[cc], correlation_results[counter]));
            counter++;
        }

        all_passed = passed ? all_passed : false;
        SamplesTestResult(true);
    }

    return all_passed;
}

void RunBatchedCorrelation(GpuImage& d_ref_img, GpuImage* d_seq_rotation_cache, int n_search_images, int batch_size, bool test_mirror, float* results, bool is_ground_truth) {

    BatchedSearch batch;

    batch.Init(d_ref_img, n_search_images, batch_size, test_mirror);

    GpuImage d_correlation_img;
    d_correlation_img.Allocate(d_ref_img.dims.x, d_ref_img.dims.y, batch_size, false);

#ifdef DO_EXPLICIT_BROADCAST
    GpuImage d_reference_img;
    d_reference_img.Allocate(d_ref_img.dims.x, d_ref_img.dims.y, batch_size, false);

    for ( int iTest = 0; iTest < batch_size; iTest++ ) {
        cudaErr(cudaMemcpy(&d_reference_img.real_values_gpu[batch.stride( ) * iTest], d_ref_img.real_values, sizeof(float) * batch.stride( ), cudaMemcpyDeviceToDevice));
    }
#endif
    // First loop is to setup the rotation cache with n_batches of <= batchsize 3D images
    // This is so each batch is contiguous in the memory.
    GpuImage* rotation_cache = new GpuImage[batch.n_batches( )];
    int       counter        = 0;
    for ( batch.index = 0; batch.index < batch.n_batches( ); batch.index++ ) {
        // std::cerr << "Batch " << batch.index << std::endl;
        // std::cerr << "size n last " << batch.batch_size( ) << " : " << batch.n_in_last_batch( ) << std::endl;
        // std::cerr << "n_images " << batch.n_images_in_this_batch( ) << std::endl;
        rotation_cache[batch.index].Allocate(d_seq_rotation_cache[0].dims.x, d_seq_rotation_cache[0].dims.y, batch.n_images_in_this_batch( ), false);
        for ( int intra_batch_idx = 0; intra_batch_idx < batch.n_images_in_this_batch( ); intra_batch_idx++ ) {
            cudaErr(cudaMemcpy(&rotation_cache[batch.index].real_values_gpu[batch.stride( ) * intra_batch_idx], d_seq_rotation_cache[counter].real_values, sizeof(float) * batch.stride( ), cudaMemcpyDeviceToDevice));
            counter++;
        }

        // d_seq_rotation_cache[counter].QuickAndDirtyWriteSlices("/tmp/d_seq_rotation_cache_" + std::to_string(counter) + ".mrc", 1, 1);
    }

#ifndef DO_EXPLICIT_BROADCAST
    GpuImage d_reference_img(d_ref_img);
#endif

    // Second loop is to actually do the correlations
    bool repeat = true;
    for ( batch.index = 0; batch.index < batch.n_batches( ); batch.index++ ) {
        d_correlation_img.is_in_real_space = false;

        // dims.z of calling image (roation cache) determines what extent of the correlation map to use
        rotation_cache[batch.index].MultiplyPixelWiseComplexConjugate(d_reference_img, d_correlation_img);

        // d_correlation_img.QuickAndDirtyWriteSlices("/tmp/d_correlation_img_" + std::to_string(batch.index) + ".mrc", 1, 1);
        // exit(1);
        if ( is_ground_truth ) {
            d_correlation_img.BackwardFFT( );
        }
        else {
            d_correlation_img.BackwardFFTBatched(batch.batch_size( ));
        }

        d_correlation_img.is_in_real_space         = true;
        d_correlation_img.object_is_centred_in_box = false;
        // TODO overload to take just the batch object directly
        Peak found_peak      = d_correlation_img.FindPeakAtOriginFast2D(&batch);
        results[batch.index] = found_peak.value;
    }
    delete[] rotation_cache;

    return;
}