// Named .cu for convenience with building

// The purpose of this test is to ensure that we can build a "pure" cpp file and only link against the CUDA business at the end

// Note FFT_DEBUG_STAGE is not handled here.

#include "../../include/FastFFT.h"

int main(int argc, char** argv) {

#if ( FFT_DEBUG_STAGE != 8 )
    std::cout << "Error: FFT_DEBUG_STAGE must be set to 8 when running this test." << std::endl;
    std::exit(1);
#endif

    const int input_size = 64;

    FastFFT::FourierTransformer<float, float, float2, 2> FT;

    float* d_input = nullptr;

    // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
    FT.SetForwardFFTPlan(input_size, input_size, 1, input_size, input_size, 1);
    FT.SetInverseFFTPlan(input_size, input_size, 1, input_size, input_size, 1);

    // The padding (dims.w) is calculated based on the setup
    short4 dims_in  = FT.ReturnFwdInputDimensions( );
    short4 dims_out = FT.ReturnFwdOutputDimensions( );

    std::array<float, (input_size + 2) * input_size> host_input;
    std::array<float, (input_size + 2) * input_size> host_output;

    int host_input_real_memory_allocated  = FT.ReturnInputMemorySize( );
    int host_output_real_memory_allocated = FT.ReturnInvOutputMemorySize( );

    cudaErr(cudaMallocAsync(&d_input, sizeof(float) * host_input_real_memory_allocated, cudaStreamPerThread));

    if ( host_input_real_memory_allocated != host_output_real_memory_allocated ) {
        std::cout << "Error: input and output memory sizes do not match" << std::endl;
        std::cout << "Input: " << host_input_real_memory_allocated << " Output: " << host_output_real_memory_allocated << std::endl;
        return 1;
    }

    if ( host_input_real_memory_allocated != host_input.size( ) ) {
        std::cout << "Error: input memory size does not match expected" << std::endl;
        std::cout << "Input: " << host_input_real_memory_allocated << " Expected: " << host_input.size( ) << std::endl;
        return 1;
    }

    // fill with negative ones so we can make sure the copy and set function works
    host_input.fill(-1.0f);
    host_output.fill(-1.0f);

    // Check basic initialization function
    FT.SetToConstant(host_input.data( ), host_output_real_memory_allocated, 3.14f);
    for ( auto& val : host_input ) {
        if ( val != 3.14f ) {
            std::cout << "Error: input memory not set to constant" << std::endl;
            return 1;
        }
    }

    // Now set to a unit impulse
    host_input.fill(0.0f);
    host_input.at(0) = 1.0f;

    // Copy to the device
    cudaErr(cudaMemcpyAsync(d_input, host_input.data( ), sizeof(float) * host_input_real_memory_allocated, cudaMemcpyHostToDevice, cudaStreamPerThread));
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    // Do a round trip FFT
    FT.FwdFFT(d_input);
    FT.InvFFT(d_input);

    // Now copy back to the output array (still set to -1)
    cudaErr(cudaMemcpyAsync(host_output.data( ), d_input, sizeof(float) * host_output_real_memory_allocated, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    if ( host_output.at(0) == input_size * input_size ) {
        std::cout << "Success: output memory copied back correctly after fft/ifft pair" << std::endl;
    }
    else {
        std::cout << "Error: output memory not copied back correctly" << std::endl;
        std::cout << "Output: " << host_output.at(0) << " Expected: " << input_size * input_size << std::endl;
        return 1;
    }

    return 0;
}