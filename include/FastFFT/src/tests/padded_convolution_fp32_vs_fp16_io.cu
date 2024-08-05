#include "tests.h"
#include <cufft.h>
#include <cufftXt.h>

template <int Rank>
void compare_libraries(std::vector<int> size, FastFFT::SizeChangeType::Enum size_change_type, bool do_rectangle) {

    using SCT = FastFFT::SizeChangeType::Enum;

    constexpr bool print_out_time = true;
    // bool set_padding_callback = false; // the padding callback is slower than pasting in b/c the read size of the pointers is larger than the actual data. do not use.
    bool set_conjMult_callback   = true;
    bool is_size_change_decrease = false;

    if ( size_change_type == SCT::decrease ) {
        is_size_change_decrease = true;
    }

    // For an increase or decrease in size, we have to shrink the loop by one,
    // for a no_change, we don't because every size is compared to itself.
    int loop_limit = 1;
    if ( size_change_type == SCT::no_change )
        loop_limit = 0;

    // Currently, to test a non-square input, the fixed input sizes are used
    // and the input x size is reduced by input_x / make_rect_x
    int make_rect_x;
    int make_rect_y = 1;
    if ( do_rectangle )
        make_rect_x = 2;
    else
        make_rect_x = 1;

    if ( Rank == 3 && do_rectangle ) {
        std::cout << "ERROR: cannot do 3d and rectangle at the same time" << std::endl;
        return;
    }

    short4 input_size;
    short4 output_size;
    for ( int iSize = 0; iSize < size.size( ) - loop_limit; iSize++ ) {
        int oSize;
        int loop_size;
        // TODO: the logic here is confusing, clean it up
        if ( size_change_type != SCT::no_change ) {
            oSize     = iSize + 1;
            loop_size = size.size( );
        }
        else {
            oSize     = iSize;
            loop_size = oSize + 1;
        }

        while ( oSize < loop_size ) {

            if ( is_size_change_decrease ) {
                output_size = make_short4(size[iSize] / make_rect_x, size[iSize] / make_rect_y, 1, 0);
                input_size  = make_short4(size[oSize] / make_rect_x, size[oSize] / make_rect_y, 1, 0);
                if ( Rank == 3 ) {
                    output_size.z = size[iSize];
                    input_size.z  = size[oSize];
                }
            }
            else {
                input_size  = make_short4(size[iSize] / make_rect_x, size[iSize] / make_rect_y, 1, 0);
                output_size = make_short4(size[oSize] / make_rect_x, size[oSize] / make_rect_y, 1, 0);
                if ( Rank == 3 ) {
                    input_size.z  = size[iSize];
                    output_size.z = size[oSize];
                }
            }
            if ( print_out_time ) {
                printf("Testing padding from %i,%i,%i to %i,%i,%i\n", input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
            }

            if ( (input_size.x == output_size.x && input_size.y == output_size.y && input_size.z == output_size.z) ) {
                // Also will change the path called in FastFFT to just be fwd/inv xform.
                set_conjMult_callback = false;
            }

            // bool test_passed = true;

            Image<float, float2> FT_input(input_size);
            Image<float, float2> FT_output(output_size);
            Image<float, float2> FT_fp16_input(input_size);
            Image<float, float2> FT_fp16_output(output_size);

            short4 target_size;

            if ( is_size_change_decrease )
                target_size = input_size; // assuming xcorr_fwd_NOOP_inv_DECREASE
            else
                target_size = output_size;

            Image<float, float2> target_search_image(target_size);
            Image<float, float2> target_search_image_fp16(input_size);
            Image<float, float2> positive_control(target_size);

            // We just make one instance of the FourierTransformer class, with calc type float.
            // For the time being input and output are also float. TODO caFlc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
            FastFFT::FourierTransformer<float, float, float2, Rank> FT;
            FastFFT::FourierTransformer<float, float, float2, Rank> targetFT;

            // Create an instance to copy memory also for the cufft tests.
            FastFFT::FourierTransformer<float, __half, __half2, Rank> FT_fp16;
            FastFFT::FourierTransformer<float, __half, __half2, Rank> targetFT_fp16;

            float*  FT_buffer;
            float*  targetFT_buffer;
            __half* FT_fp16_buffer;
            __half* targetFT_fp16_buffer;

            if ( is_size_change_decrease ) {
                FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, input_size.x, input_size.y, input_size.z);
                FT.SetInverseFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
                targetFT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, input_size.x, input_size.y, input_size.z);
                targetFT.SetInverseFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);

                FT_fp16.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, input_size.x, input_size.y, input_size.z);
                FT_fp16.SetInverseFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
                targetFT_fp16.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, input_size.x, input_size.y, input_size.z);
                targetFT_fp16.SetInverseFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
            }
            else {
                FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
                FT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);
                targetFT.SetForwardFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);
                targetFT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);

                FT_fp16.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
                FT_fp16.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);
                targetFT_fp16.SetForwardFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);
                targetFT_fp16.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);
            }

            short4 fwd_dims_in  = FT.ReturnFwdInputDimensions( );
            short4 fwd_dims_out = FT.ReturnFwdOutputDimensions( );
            short4 inv_dims_in  = FT.ReturnInvInputDimensions( );
            short4 inv_dims_out = FT.ReturnInvOutputDimensions( );

            FT_input.real_memory_allocated  = FT.ReturnInputMemorySize( );
            FT_output.real_memory_allocated = FT.ReturnInvOutputMemorySize( );

            size_t device_memory = std::max(FT_input.real_memory_allocated, FT_output.real_memory_allocated);
            cudaErr(cudaMallocAsync((void**)&FT_buffer, device_memory * sizeof(float), cudaStreamPerThread));
            cudaErr(cudaMallocAsync((void**)&targetFT_buffer, device_memory * sizeof(float), cudaStreamPerThread));
            // Set to zero
            cudaErr(cudaMemsetAsync(FT_buffer, 0, device_memory * sizeof(float), cudaStreamPerThread));
            cudaErr(cudaMemsetAsync(targetFT_buffer, 0, device_memory * sizeof(float), cudaStreamPerThread));

            cudaErr(cudaMallocAsync((void**)&FT_fp16_buffer, device_memory * sizeof(__half), cudaStreamPerThread));
            cudaErr(cudaMallocAsync((void**)&targetFT_fp16_buffer, device_memory * sizeof(__half), cudaStreamPerThread));
            // Set to zero
            cudaErr(cudaMemsetAsync(FT_fp16_buffer, 0, device_memory * sizeof(__half), cudaStreamPerThread));
            cudaErr(cudaMemsetAsync(targetFT_fp16_buffer, 0, device_memory * sizeof(__half), cudaStreamPerThread));
            if ( is_size_change_decrease )
                target_search_image.real_memory_allocated = targetFT.ReturnInputMemorySize( );
            else
                target_search_image.real_memory_allocated = targetFT.ReturnInvOutputMemorySize( ); // the larger of the two.

            positive_control.real_memory_allocated = target_search_image.real_memory_allocated; // this won't change size

            bool set_fftw_plan = false;
            FT_input.Allocate(set_fftw_plan);
            FT_output.Allocate(set_fftw_plan);
            FT_fp16_input.Allocate(set_fftw_plan);
            FT_fp16_output.Allocate(set_fftw_plan);

            target_search_image.Allocate(true);
            target_search_image_fp16.Allocate(true);
            positive_control.Allocate(true);

            // Set a unit impulse at the center of the input array.
            // For now just considering the real space image to have been implicitly quadrant swapped so the center is at the origin.
            FT.SetToConstant(FT_input.real_values, FT_input.real_memory_allocated, 0.0f);
            FT.SetToConstant(FT_output.real_values, FT_output.real_memory_allocated, 0.0f);
            FT.SetToConstant(FT_fp16_input.real_values, FT_fp16_input.real_memory_allocated, 0.0f);
            FT.SetToConstant(FT_fp16_output.real_values, FT_fp16_output.real_memory_allocated, 0.0f);
            FT.SetToConstant(target_search_image.real_values, target_search_image.real_memory_allocated, 0.0f);
            FT.SetToConstant(target_search_image_fp16.real_values, target_search_image_fp16.real_memory_allocated, 0.0f);
            FT.SetToConstant(positive_control.real_values, target_search_image.real_memory_allocated, 0.0f);

            // Place these values at the origin of the image and after convolution, should be at 0,0,0.
            float testVal_1                         = 2.0f;
            float testVal_2                         = set_conjMult_callback ? 3.0f : 1.0; // This way the test conditions are the same, the 1. indicating no conj
            FT_input.real_values[0]                 = testVal_1;
            FT_fp16_input.real_values[0]            = testVal_1;
            target_search_image.real_values[0]      = testVal_2;
            target_search_image_fp16.real_values[0] = testVal_2;
            positive_control.real_values[0]         = testVal_1;

            // Transform the target on the host prior to transfer.
            target_search_image.FwdFFT( );
            target_search_image_fp16.FwdFFT( );

            cudaErr(cudaMemcpyAsync(FT_buffer, FT_input.real_values, FT_input.real_memory_allocated * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
            cudaErr(cudaMemcpyAsync(targetFT_buffer, target_search_image.real_values, target_search_image.real_memory_allocated * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
            FT_fp16_input.ConvertFP32ToFP16( );
            target_search_image_fp16.ConvertFP32ToFP16( );
            cudaErr(cudaMemcpyAsync(FT_fp16_buffer, FT_fp16_input.real_values, FT_fp16_input.real_memory_allocated * sizeof(__half), cudaMemcpyHostToDevice, cudaStreamPerThread));
            cudaErr(cudaMemcpyAsync(targetFT_fp16_buffer, target_search_image_fp16.real_values, target_search_image_fp16.real_memory_allocated * sizeof(__half), cudaMemcpyHostToDevice, cudaStreamPerThread));
            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

            // Positive control on the host.
            // After both forward FFT's we should constant values in each pixel = testVal_1 and testVal_2.
            // After the Conjugate multiplication, we should have a constant value of testVal_1*testVal_2.
            // After the inverse FFT, we should have a constant value of testVal_1*testVal_2 in the center pixel and 0 everywhere else.
            positive_control.FwdFFT( );
            if ( set_conjMult_callback )
                positive_control.MultiplyConjugateImage(target_search_image.complex_values);
            positive_control.InvFFT( );

            CheckUnitImpulseRealImage(positive_control, __LINE__);

            if ( positive_control.real_values[0] == positive_control.size.x * positive_control.size.y * positive_control.size.z * testVal_1 * testVal_2 ) {
                if ( print_out_time ) {
                    std::cout << "Test passed for FFTW positive control." << std::endl;
                }
            }
            else {
                MyTestPrintAndExit(false, "Test failed for FFTW positive control. Value at zero is  " + std::to_string(positive_control.real_values[0]));
            }

            FT_output.create_timing_events( );

            FastFFT::KernelFunction::my_functor<float, 0, FastFFT::KernelFunction::NOOP>     noop;
            FastFFT::KernelFunction::my_functor<float, 4, FastFFT::KernelFunction::CONJ_MUL> conj_mul;

            //////////////////////////////////////////
            //////////////////////////////////////////
            // Warm up and check for accuracy
            // we set set_conjMult_callback = false
            if ( set_conjMult_callback || is_size_change_decrease ) {
                // FT.CrossCorrelate(targetFT.d_ptr.momentum_space, false);
                // Will type deduction work here?
                FT.FwdImageInvFFT(FT_buffer, reinterpret_cast<float2*>(targetFT_buffer), FT_buffer, noop, conj_mul, noop);
                FT_fp16.FwdImageInvFFT(FT_fp16_buffer, reinterpret_cast<__half2*>(targetFT_fp16_buffer), FT_fp16_buffer, noop, conj_mul, noop);
            }
            else {
                FT.FwdFFT(FT_buffer);
                FT.InvFFT(FT_buffer);
                FT_fp16.FwdFFT(FT_fp16_buffer);
                FT_fp16.InvFFT(FT_fp16_buffer);
            }

            int n_loops;
            if ( Rank == 3 ) {
                int max_size = std::max(fwd_dims_in.x, fwd_dims_out.x);
                if ( max_size < 128 ) {
                    n_loops = 1000;
                }
                else if ( max_size <= 256 ) {
                    n_loops = 400;
                }
                else if ( max_size <= 512 ) {
                    n_loops = 150;
                }
                else {
                    n_loops = 50;
                }
            }
            else {
                int max_size = std::max(fwd_dims_in.x, fwd_dims_out.x);
                if ( max_size < 256 ) {
                    n_loops = 10000;
                }
                else if ( max_size <= 512 ) {
                    n_loops = 5000;
                }
                else if ( max_size <= 2048 ) {
                    n_loops = 2500;
                }
                else {
                    n_loops = 1000;
                }
            }

            FT_output.record_start( );
            for ( int i = 0; i < n_loops; ++i ) {
                if ( set_conjMult_callback || is_size_change_decrease ) {
                    //   FT.CrossCorrelate(targetFT.d_ptr.momentum_space_buffer, false);
                    // Will type deduction work here?
                    FT.FwdImageInvFFT(FT_buffer, reinterpret_cast<float2*>(targetFT_buffer), FT_buffer, noop, conj_mul, noop);
                }
                else {
                    FT.FwdFFT(FT_buffer);
                    FT.InvFFT(FT_buffer);
                }
            }
            FT_output.record_stop( );
            FT_output.synchronize( );
            FT_output.print_time("FastFFT", print_out_time);
            float FastFFT_time = FT_output.elapsed_gpu_ms;

            FT_output.record_start( );
            for ( int i = 0; i < n_loops; ++i ) {
                if ( set_conjMult_callback || is_size_change_decrease ) {
                    //   FT.CrossCorrelate(targetFT.d_ptr.momentum_space_buffer, false);
                    // Will type deduction work here?
                    FT_fp16.FwdImageInvFFT(FT_fp16_buffer, reinterpret_cast<__half2*>(targetFT_fp16_buffer), FT_fp16_buffer, noop, conj_mul, noop);
                }
                else {
                    FT_fp16.FwdFFT(FT_fp16_buffer);
                    FT_fp16.InvFFT(FT_fp16_buffer);
                }
            }
            FT_output.record_stop( );
            FT_output.synchronize( );
            FT_output.print_time("FastFFT_fp16", print_out_time);
            float FastFFT_time_fp16 = FT_output.elapsed_gpu_ms;

            std::cout << "For size " << input_size.x << " to " << output_size.x << ": ";
            std::cout << "Ratio FP32/FP16 : " << FastFFT_time / FastFFT_time_fp16 << "\n\n"
                      << std::endl;

            oSize++;
            // We don't want to loop if the size is not actually changing.
            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
            cudaErr(cudaFree(FT_buffer));
            cudaErr(cudaFree(targetFT_buffer));
            cudaErr(cudaFree(FT_fp16_buffer));
            cudaErr(cudaFree(targetFT_fp16_buffer));
        } // while loop over pad to size

    } // for loop over pad from size
}

int main(int argc, char** argv) {

    using SCT = FastFFT::SizeChangeType::Enum;

    std::string test_name;
    // Default to running all tests
    bool run_2d_performance_tests = false;
    bool run_3d_performance_tests = false;

    const std::string_view text_line = "simple convolution";
    FastFFT::CheckInputArgs(argc, argv, text_line, run_2d_performance_tests, run_3d_performance_tests);

    // TODO: size decrease
    if ( run_2d_performance_tests ) {
#ifdef HEAVYERRORCHECKING_FFT
        std::cout << "Running performance tests with heavy error checking.\n";
        std::cout << "This doesn't make sense as the synchronizations are invalidating.\n";
// exit(1);
#endif
        SCT size_change_type;
        // Set the SCT to no_change, increase, or decrease
        size_change_type = SCT::no_change;
        compare_libraries<2>(FastFFT::test_size, size_change_type, false);
        // compare_libraries<2>(test_size_rectangle, do_3d, size_change_type, true);

        size_change_type = SCT::increase;
        compare_libraries<2>(FastFFT::test_size, size_change_type, false);
        // compare_libraries<2>(test_size_rectangle, do_3d, size_change_type, true);

        size_change_type = SCT::decrease;
        compare_libraries<2>(FastFFT::test_size, size_change_type, false);
    }

    if ( run_3d_performance_tests ) {
#ifdef HEAVYERRORCHECKING_FFT
        std::cout << "Running performance tests with heavy error checking.\n";
        std::cout << "This doesn't make sense as the synchronizations are invalidating.\n";
#endif

        SCT size_change_type;

        size_change_type = SCT::no_change;
        compare_libraries<3>(FastFFT::test_size_3d, size_change_type, false);

        // TODO: These are not yet completed.
        // size_change_type = SCT::increase;
        // compare_libraries<3>(FastFFT::test_size, do_3d, size_change_type, false);

        // size_change_type = SCT::decrease;
        // compare_libraries(FastFFT::test_size, do_3d, size_change_type, false);
    }

    return 0;
};