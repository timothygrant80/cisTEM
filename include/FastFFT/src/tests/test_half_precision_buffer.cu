
#include "tests.h"

template <int Rank>
bool half_precision_buffer_test(int size) {

    bool all_passed = true;

    short4 input_size;

    input_size = make_short4(size, size, 1, 0);

    Image<float, float2> host_input_fp32(input_size);
    Image<float, float2> host_output_fp16(input_size);

    FastFFT::FourierTransformer<float, float, float2> FT;

    host_input_fp32.Allocate(false);
    host_output_fp16.Allocate(false);

    FT.SetToRandom(host_input_fp32.real_values, host_input_fp32.real_memory_allocated, 0.f, 1.0f);
    for ( int i = 0; i < host_input_fp32.real_memory_allocated; i++ ) {
        host_output_fp16.real_values[i] = host_input_fp32.real_values[i];
    }

    float fp32sum = host_input_fp32.ReturnSumOfReal(host_input_fp32.real_values, input_size, false);
    float fp16sum = host_output_fp16.ReturnSumOfReal(host_output_fp16.real_values, input_size, false);
    std::cout << "Sum of real values in fp16 buffer: " << fp16sum << std::endl;
    std::cout << "Sum of real values in fp32 buffer: " << fp32sum << std::endl;
    if ( fp32sum != fp16sum ) {
        std::cerr << "Error: Sum of real values in input and output buffers do not match." << std::endl;
        std::cout << "Sum of real values in fp16 buffer: " << fp16sum << std::endl;
        std::cout << "Sum of real values in fp32 buffer: " << fp32sum << std::endl;
        all_passed = false;
    }

    // Now convert one buffer to fp 16
    host_output_fp16.ConvertFP32ToFP16( );
    // And convert back, results should almost be the same
    host_output_fp16.ConvertFP16ToFP32( );
    float diff_value;
    for ( int i = 0; i < host_input_fp32.real_memory_allocated; i++ ) {
        // Check that the floats are the same up to the third decimal point but different past it
        diff_value = std::abs(host_input_fp32.real_values[i] - host_output_fp16.real_values[i]);
        if ( diff_value > 0.001f && diff_value < 0.0001 ) {
            std::cerr << "fp32 " << host_input_fp32.real_values[i] << std::endl;
            std::cerr << " fp16 " << host_output_fp16.real_values[i] << std::endl;
            all_passed = false;
        }
    }

    return all_passed;
}

int main(int argc, char** argv) {

    std::string test_name;
    // Default to running all tests
    bool run_2d_unit_tests = false;
    bool run_3d_unit_tests = false;

    const std::string_view text_line = "half precision buffers";
    FastFFT::CheckInputArgs(argc, argv, text_line, run_2d_unit_tests, run_3d_unit_tests);

    if ( run_2d_unit_tests ) {
        if ( ! half_precision_buffer_test<2>(64) )
            return 1;
    }

    return 0;
};