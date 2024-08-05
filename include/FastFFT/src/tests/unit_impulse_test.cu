
#include "tests.h"

template <int Rank>
bool unit_impulse_test(std::vector<int> size, bool do_increase_size, bool inplace_transform = true) {

    bool              all_passed = true;
    std::vector<bool> init_passed(size.size( ), true);
    std::vector<bool> FFTW_passed(size.size( ), true);
    std::vector<bool> FastFFT_forward_passed(size.size( ), true);
    std::vector<bool> FastFFT_roundTrip_passed(size.size( ), true);

    short4 input_size;
    short4 output_size;
    for ( int iSize = 0; iSize < size.size( ) - 1; iSize++ ) {
        int oSize = iSize + 1;
        while ( oSize < size.size( ) ) {

            // std::cout << std::endl << "Testing padding from  " << size[iSize] << " to " << size[oSize] << std::endl;
            if ( do_increase_size ) {
                if ( Rank == 3 ) {
                    input_size  = make_short4(size[iSize], size[iSize], size[iSize], 0);
                    output_size = make_short4(size[oSize], size[oSize], size[oSize], 0);
                }
                else {
                    input_size  = make_short4(size[iSize], size[iSize], 1, 0);
                    output_size = make_short4(size[oSize], size[oSize], 1, 0);
                }
            }
            else {
                if ( Rank == 3 ) {
                    output_size = make_short4(size[iSize], size[iSize], size[iSize], 0);
                    input_size  = make_short4(size[oSize], size[oSize], size[oSize], 0);
                }
                else {
                    output_size = make_short4(size[iSize], size[iSize], 1, 0);
                    input_size  = make_short4(size[oSize], size[oSize], 1, 0);
                }
            }

            float sum;

            Image<float, float2> host_input(input_size);
            Image<float, float2> host_output(output_size);
            Image<float, float2> device_output(output_size);

            // We just make one instance of the FourierTransformer class, with calc type float.
            // For the time being input and output are also float. TODO calc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
            FastFFT::FourierTransformer<float, float, float2, Rank> FT;

            float* FT_buffer;
            // This is only used for the out of place test.
            float* FT_buffer_output;
            // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
            FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
            FT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);

            // The padding (dims.w) is calculated based on the setup
            short4 dims_fwd_in  = FT.ReturnFwdInputDimensions( );
            short4 dims_fwd_out = FT.ReturnFwdOutputDimensions( );
            short4 dims_inv_in  = FT.ReturnInvInputDimensions( );
            short4 dims_inv_out = FT.ReturnInvOutputDimensions( );
            // Determine how much memory we need, working with FFTW/CUDA style in place transform padding.
            // Note: there is no reason we really need this, because the xforms will always be out of place.
            //       For now, this is just in place because all memory in cisTEM is allocated accordingly.
            host_input.real_memory_allocated  = FT.ReturnInputMemorySize( );
            host_output.real_memory_allocated = FT.ReturnInvOutputMemorySize( );

            // On the device, we will always allocate enough memory for the larger of input/output including the buffer array.
            // Minmize the number of calls to malloc which are slow and can lead to fragmentation.
            device_output.real_memory_allocated = std::max(host_input.real_memory_allocated, host_output.real_memory_allocated);
            cudaErr(cudaMallocAsync((void**)&FT_buffer, device_output.real_memory_allocated * sizeof(float), cudaStreamPerThread));
            cudaErr(cudaMemsetAsync(FT_buffer, 0, device_output.real_memory_allocated * sizeof(float), cudaStreamPerThread));
            if ( ! inplace_transform ) {
                cudaErr(cudaMallocAsync((void**)&FT_buffer_output, device_output.real_memory_allocated * sizeof(float), cudaStreamPerThread));
                cudaErr(cudaMemsetAsync(FT_buffer_output, 0, device_output.real_memory_allocated * sizeof(float), cudaStreamPerThread));
            }
            // In your own programs, you will be handling this memory allocation yourself. We'll just make something here.
            // I think fftwf_malloc may potentially create a different alignment than new/delete, but kinda doubt it. For cisTEM consistency...
            bool set_fftw_plan = true;
            host_input.Allocate(set_fftw_plan);
            host_output.Allocate(set_fftw_plan);

            // Now we want to associate the host memory with the device memory. The method here asks if the host pointer is pinned (in page locked memory) which
            // ensures faster transfer. If false, it will be pinned for you.
            // FIXME:
            // FT.SetInputPointer(host_input.real_values);

            // Set a unit impulse at the center of the input array.
            FT.SetToConstant(host_input.real_values, host_input.real_memory_allocated, 0.0f);
            FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 0.0f);

            host_input.real_values[0]  = 1.0f;
            host_output.real_values[0] = 1.0f;

            // This will exit if fail, so the following bools are not really needed any more.
            CheckUnitImpulseRealImage(host_output, __LINE__);

            // It doesn't really matter which one we copy here, it would make sense to do the smaller one though.
            cudaErr(cudaMemcpyAsync(FT_buffer, host_output.real_values, host_output.real_memory_allocated * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));

            // We need to wait for the copy to finish before we can do the FFT on the host.
            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
            host_output.FwdFFT( );

            host_output.fftw_epsilon = host_output.ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated / 2);
            // std::cout << "host " << host_output.fftw_epsilon << " " << host_output.real_memory_allocated<< std::endl;

            host_output.fftw_epsilon -= (host_output.real_memory_allocated / 2);
            if ( std::abs(host_output.fftw_epsilon) > 1e-8 ) {
                all_passed         = false;
                FFTW_passed[iSize] = false;
            }

            // MyFFTDebugAssertTestTrue( std::abs(host_output.fftw_epsilon) < 1e-8 , "FFTW unit impulse forward FFT");

            // Just to make sure we don't get a false positive, set the host memory to some undesired value.
            // FIXME: This wouldn't work for size decrease
            FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 2.0f);

            // This method will call the regular FFT kernels given the input/output dimensions are equal when the class is instantiated.
            // bool swap_real_space_quadrants = true;
            if ( inplace_transform ) {
                FT.FwdFFT(FT_buffer);
            }
            else {
                FT.FwdFFT(FT_buffer, FT_buffer_output);
                // To make sure we are not getting a false positive, set the input to some undesired value.
                cudaErr(cudaMemsetAsync(FT_buffer, 0, device_output.real_memory_allocated * sizeof(float), cudaStreamPerThread));
            }

            bool continue_debugging = true;
            // We don't want this to break compilation of other tests, so only check at runtime.
            if constexpr ( FFT_DEBUG_STAGE < 5 ) {
                if ( do_increase_size ) {

                    FT.CopyDeviceToHostAndSynchronize(host_output.real_values);
                    // Right now, only testing a size change on the forward transform,
                    continue_debugging = debug_partial_fft<FFT_DEBUG_STAGE, Rank>(host_output, dims_fwd_in, dims_fwd_out, dims_inv_in, dims_inv_out, __LINE__);
                    sum                = host_output.ReturnSumOfComplexAmplitudes(host_output.complex_values, host_output.real_memory_allocated / 2);
                }
                else {
                    FT.CopyDeviceToHostAndSynchronize(host_input.real_values, FT.ReturnInputMemorySize( ));
                    continue_debugging = debug_partial_fft<FFT_DEBUG_STAGE, Rank>(host_input, dims_fwd_in, dims_fwd_out, dims_inv_in, dims_inv_out, __LINE__);
                    sum                = host_input.ReturnSumOfComplexAmplitudes(host_input.complex_values, host_input.real_memory_allocated / 2);
                }

                sum -= (host_output.real_memory_allocated / 2);

                // FIXME : shared comparison functions
                if ( abs(sum) > 1e-8 ) {
                    all_passed                    = false;
                    FastFFT_forward_passed[iSize] = false;
                }
            }

            MyTestPrintAndExit(continue_debugging, "Partial FFT debug stage " + std::to_string(FFT_DEBUG_STAGE));
            // MyFFTDebugAssertTestTrue( abs(sum - host_output.fftw_epsilon) < 1e-8, "FastFFT unit impulse forward FFT");
            FT.SetToConstant(host_output.real_values, host_output.real_memory_allocated, 2.0f);

            if ( inplace_transform ) {
                FT.InvFFT(FT_buffer);
            }
            else {
                // Switch the role of the buffers
                FT.InvFFT(FT_buffer_output, FT_buffer);
                cudaErr(cudaMemsetAsync(FT_buffer_output, 0, device_output.real_memory_allocated * sizeof(float), cudaStreamPerThread));
            }
            FT.CopyDeviceToHostAndSynchronize(host_output.real_values);

            if constexpr ( FFT_DEBUG_STAGE > 4 ) {
                // Right now, only testing a size change on the forward transform,
                // continue_debugging = debug_partial_fft<FFT_DEBUG_STAGE, Rank>(host_output, dims_fwd_in, dims_fwd_out, dims_inv_in, dims_inv_out, __LINE__);

                sum = host_output.ReturnSumOfReal(host_output.real_values, dims_fwd_out);
                if ( sum != dims_fwd_out.x * dims_fwd_out.y * dims_fwd_out.z ) {
                    all_passed                      = false;
                    FastFFT_roundTrip_passed[iSize] = false;
                }
            }
            // MyTestPrintAndExit(continue_debugging, "Partial FFT debug stage " + std::to_string(FFT_DEBUG_STAGE));

            // std::cout << "size in/out " << dims_fwd_in.x << ", " << dims_fwd_out.x << std::endl;
            // MyFFTDebugAssertTestTrue( sum == dims_fwd_out.x*dims_fwd_out.y*dims_fwd_out.z,"FastFFT unit impulse round trip FFT");

            oSize++;
            cudaErr(cudaFreeAsync(FT_buffer, cudaStreamPerThread));
            if ( ! inplace_transform )
                cudaErr(cudaFreeAsync(FT_buffer_output, cudaStreamPerThread));
        } // while loop over pad to size
    } // for loop over pad from size

    std::string is_in_place;
    if ( inplace_transform )
        is_in_place = "in place";
    else
        is_in_place = "out of place";

    if ( all_passed ) {
        if ( ! do_increase_size )
            std::cout << "    All rank " << Rank << " size_decrease unit impulse, " << is_in_place << " tests passed!" << std::endl;
        else
            std::cout << "    All rank " << Rank << " size_increase unit impulse, " << is_in_place << "  tests passed!" << std::endl;
    }
    else {
        for ( int n = 0; n < size.size( ); n++ ) {
            if ( ! init_passed[n] )
                std::cout << "    Initialization failed for size " << size[n] << " rank " << Rank << " " << is_in_place << std::endl;
            if ( ! FFTW_passed[n] )
                std::cout << "    FFTW failed for size " << size[n] << " rank " << Rank << " " << is_in_place << std::endl;
            if ( ! FastFFT_forward_passed[n] )
                std::cout << "    FastFFT failed for forward transform size " << size[n] << " rank " << Rank << " " << is_in_place << " increase " << do_increase_size << std::endl;
            if ( ! FastFFT_roundTrip_passed[n] )
                std::cout << "    FastFFT failed for roundtrip transform size " << size[n] << " rank " << Rank << " " << is_in_place << " increase " << do_increase_size << std::endl;
        }
    }
    return all_passed;
}

int main(int argc, char** argv) {

    std::string test_name;
    // Default to running all tests
    bool run_2d_unit_tests = false;
    bool run_3d_unit_tests = false;

    const std::string_view text_line = "unit impulse";
    FastFFT::CheckInputArgs(argc, argv, text_line, run_2d_unit_tests, run_3d_unit_tests);

    // TODO: size decrease
    if ( run_2d_unit_tests ) {
        constexpr bool do_increase_size_first = true;
        constexpr bool second_round           = ! do_increase_size_first;
        if ( ! unit_impulse_test<2>(FastFFT::test_size, do_increase_size_first) )
            return 1;
        if ( ! unit_impulse_test<2>(FastFFT::test_size, second_round) )
            return 1;
        // Also test case where the external pointer is different on output
        if ( ! unit_impulse_test<2>(FastFFT::test_size, true, false) )
            return 1;
    }

    if ( run_3d_unit_tests ) {
        // FIXME: tests are failing for 3d
        // constexpr bool do_increase_size_first = true;
        // constexpr bool second_round           = ! do_increase_size_first;
        // if ( ! unit_impulse_test<3>(FastFFT::test_size_3d, do_increase_size_first) )
        //     return 1;
        // if (! unit_impulse_test(test_size_3d, true, true)) return 1;
    }

    return 0;
};