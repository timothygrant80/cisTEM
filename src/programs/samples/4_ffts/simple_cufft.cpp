#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "simple_cufft.h"

#define RUN_TIMING_TESTS
#ifdef RUN_TIMING_TESTS
using namespace cistem_timer;
#else
using namespace cistem_timer_noop;
#endif

void SimpleCuFFTRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting GPU FFT tests:", false);

    TEST(DoInPlaceR2CandC2R(hiv_image_80x80x1_filename, temp_directory));

    TEST(DoInPlaceR2CandC2RBatched(hiv_image_80x80x1_filename, temp_directory));

    SamplesPrintEndMessage( );

    return;
}

bool DoInPlaceR2CandC2R(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Square 2D R2C/C2R inplace ffts", passed);

    // Compare the default ops to the cpu (FFTW/MKL version)
    // These are single precision, inplace ffts, with FFTW layout.

    Image test_image;

    std::vector<size_t> fft_sizes   = {64, 256, 384, 512, 648, 1024, 3456, 4096};
    constexpr size_t    max_3d_size = 648;

    constexpr bool should_allocate_in_real_space = true;
    constexpr bool should_make_fftw_plan         = true;
    // First we will test square images.
    // TODO: add fuzzing over noise distributions, maybe that fits better somewhere else? Or better yet, this method
    // could be called for various noise distributions.
    for ( auto iSize : fft_sizes ) {

        test_image.Allocate(iSize, iSize, 1, should_allocate_in_real_space, should_make_fftw_plan);
        test_image.FillWithNoiseFromNormalDistribution(0.f, 1.f);
        GpuImage gpu_test_image;
        gpu_test_image.Init(test_image);
        gpu_test_image.CopyHostToDeviceAndSynchronize(test_image);

        test_image.ForwardFFT( );
        gpu_test_image.ForwardFFT( );

        Image gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, false, false);

        passed = passed && CompareComplexValues(test_image, gpu_cpu_buffer);

        test_image.BackwardFFT( );
        gpu_test_image.BackwardFFT( );

        // This call also frees the GPU memory and unpins the hsot memory
        gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, true, true);

        passed = passed && CompareRealValues(test_image, gpu_cpu_buffer);
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("Cubic  3D R2C/C2R inplace ffts", passed);
    for ( auto iSize : fft_sizes ) {
        if ( iSize > max_3d_size ) {
            continue;
        }
        test_image.Allocate(iSize, iSize, 1, should_allocate_in_real_space, should_make_fftw_plan);
        test_image.FillWithNoiseFromNormalDistribution(0.f, 1.f);
        GpuImage gpu_test_image;
        gpu_test_image.Init(test_image);
        gpu_test_image.CopyHostToDeviceAndSynchronize(test_image);

        test_image.ForwardFFT( );
        gpu_test_image.ForwardFFT( );

        Image gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, false, false);

        passed = passed && CompareComplexValues(test_image, gpu_cpu_buffer);

        test_image.BackwardFFT( );
        gpu_test_image.BackwardFFT( );

        // This call also frees the GPU memory and unpins the hsot memory
        gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, true, true);

        passed = passed && CompareRealValues(test_image, gpu_cpu_buffer);
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("Non-Sq 2D R2C/C2R inplace ffts", passed);
    for ( int i = 1; i < fft_sizes.size( ); i++ ) {

        test_image.Allocate(fft_sizes[i], fft_sizes[i - 1], 1, should_allocate_in_real_space, should_make_fftw_plan);
        test_image.FillWithNoiseFromNormalDistribution(0.f, 1.f);
        GpuImage gpu_test_image;
        gpu_test_image.Init(test_image);
        gpu_test_image.CopyHostToDeviceAndSynchronize(test_image);

        test_image.ForwardFFT( );
        gpu_test_image.ForwardFFT( );

        Image gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, false, false);

        passed = passed && CompareComplexValues(test_image, gpu_cpu_buffer);

        test_image.BackwardFFT( );
        gpu_test_image.BackwardFFT( );

        // This call also frees the GPU memory and unpins the hsot memory
        gpu_cpu_buffer = gpu_test_image.CopyDeviceToNewHost(true, true, true);

        passed = passed && CompareRealValues(test_image, gpu_cpu_buffer);
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);
    return all_passed;
}

bool DoInPlaceR2CandC2RBatched(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Batched 2d ffts (accuracy)", passed);

    constexpr size_t image_size = 256;
    constexpr size_t batch_size = 400;

    std::array<Image, batch_size>    cpu_individual;
    std::array<GpuImage, batch_size> gpu_individual;

    // Create two 3d images to mirror the stack of 2ds, but with contiguous image datas memory.
    Image    cpu_batch;
    GpuImage gpu_batch;

    constexpr bool should_allocate_in_real_space = true;
    constexpr bool should_make_fftw_plan         = true;

    // Allocate and fill the 3d with random data, and then we will point each of the individual 2ds at the
    // correct place in the 3d's image memory.

    cpu_batch.Allocate(image_size, image_size, batch_size, should_allocate_in_real_space, should_make_fftw_plan);
    cpu_batch.FillWithNoiseFromNormalDistribution(0.f, 1.f);

    // The address range is tracked for each pointer accessing subsets of the full block of memory, so we DO NOT want to pin it here.
    constexpr bool pin_host_memory = false;
    gpu_batch.Init(cpu_batch, pin_host_memory, true);
    gpu_batch.CopyHostToDeviceAndSynchronize(cpu_batch, pin_host_memory);

    int stride = (cpu_batch.padding_jump_value + cpu_batch.logical_x_dimension) * cpu_batch.logical_y_dimension;
    for ( int i = 0; i < batch_size; i++ ) {
        cpu_individual[i].Allocate(image_size, image_size, 1, should_allocate_in_real_space, should_make_fftw_plan);

        cpu_individual[i].real_values    = &cpu_batch.real_values[i * stride];
        cpu_individual[i].complex_values = (std::complex<float>*)cpu_individual[i].real_values;
        // Since we are re-using the

        gpu_individual[i].Init(cpu_individual[i], true, true);
        gpu_individual[i].CopyHostToDeviceAndSynchronize(cpu_individual[i]);
    }

    StopWatch timer;

    timer.start("GPU 2d");
    for ( int i = 0; i < batch_size; i++ ) {
        gpu_individual[i].ForwardFFT( );
        gpu_individual[i].BackwardFFT( );
    }
    timer.lap_sync("GPU 2d");

    timer.start("Gpu 2d batched");
    gpu_batch.ForwardFFTBatched( );
    gpu_batch.BackwardFFTBatched( );
    timer.lap_sync("Gpu 2d batched");

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    // Create some clean images for our equality test.
    Image    tmp_ind, tmp_batch;
    GpuImage tmp_gpu_batch;

    // Again, this memory is already pinned on the host, so don't attempt to re-pin it here.
    tmp_gpu_batch.Init(cpu_individual[0], false, true);
    for ( int i = 0; i < batch_size; i++ ) {
        // if we add the BackwardFFT to the test, we may not want to clear the memory here
        tmp_ind                   = gpu_individual[i].CopyDeviceToNewHost(true, true, true);
        tmp_gpu_batch.real_values = &gpu_batch.real_values[stride * i];
        tmp_batch                 = tmp_gpu_batch.CopyDeviceToNewHost(true, false, false);

        passed = passed && CompareRealValues(tmp_ind, tmp_batch);
    }

    // We can't safely leave these pointers at memory the don't own.
    // I guess this also means the memory allocated originally is not cleared? I suppose some type of "borrowmemory"
    // Function that upone destruction "returns" the memory (just doesn't free it) and then *does* free it's own memory
    // would be useful.
    tmp_gpu_batch.real_values = nullptr;
    for ( int i = 0; i < batch_size; i++ ) {
        cpu_individual[i].real_values = nullptr;
    }

    float ratio_seq_to_batched = timer.get_ratio_of_times("GPU 2d", "Gpu 2d batched");
    // wxPrintf("Ratio of tims is %f\n", ratio_seq_to_batched);
    // timer.print_times( );

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    SamplesBeginTest("Batched 2d ffts (performance)", passed);

    passed = passed && ratio_seq_to_batched > 2.5f;
    if ( ! passed )
        wxPrintf("\n Ratio seq to batched %f\n", ratio_seq_to_batched);

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);
    // for ( int i = 0; i < batch_size; i++ ) {
    //     gpu_individual[i].Deallocate( );
    // }
    // delete[] cpu_individual;
    // delete[] gpu_individual;

    return all_passed;
}