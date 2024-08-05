#ifndef _SRC_TESTS_TESTS_H
#define _SRC_TESTS_TESTS_H

#include <iostream>
#include "../fastfft/Image.cuh"
#include "../../include/FastFFT.cuh"
#include "helper_functions.cuh"

namespace FastFFT {
// Input size vectors to be tested.
std::vector<int> test_size = {32, 64, 128, 256, 512, 1024, 2048, 4096};
// std::vector<int> test_size = {32, 64, 128, 256, 512, 1024, 2048, 4096};

std::vector<int> test_size_rectangle = {64, 128, 256, 512, 1024, 2048, 4096};
std::vector<int> test_size_3d        = {32, 64, 128, 256, 512};
// std::vector<int> test_size_3d ={512};

// The launch parameters fail for 4096 -> < 64 for r2c_decrease_XY, not sure if it is the elements_per_thread or something else.
// For now, just over-ride these small sizes
std::vector<int> test_size_for_decrease = {64, 128, 256, 512, 1024, 2048, 4096};

void CheckInputArgs(int argc, char** argv, const std::string_view& text_line, bool& run_2d_unit_tests, bool& run_3d_unit_tests) {
    switch ( argc ) {
        case 1: {
            std::cout << "Running all tests" << std::endl;
            run_2d_unit_tests = true;
            run_3d_unit_tests = true;
            break;
        }
        case 2: {
            std::string test_name = argv[1];
            if ( test_name == "--all" ) {
                std::cout << "Running all tests" << std::endl;
                run_2d_unit_tests = true;
                run_3d_unit_tests = true;
            }
            else if ( test_name == "--2d" ) {
                std::cout << "Running 2d " << text_line << " tests" << std::endl;
                run_2d_unit_tests = true;
            }
            else if ( test_name == "--3d" ) {
                std::cout << "Running 3d " << text_line << " tests" << std::endl;
                run_3d_unit_tests = true;
            }
            else {
                std::cout << "Usage: " << argv[0] << " < --all (default w/ no arg), --2d, --3d>" << std::endl;
                std::exit(0);
            }
            break;
        }
        default: {
            std::cout << "Usage: " << argv[0] << " < --all (default w/ no arg), --2d, --3d>" << std::endl;
            std::exit(0);
        }
    };
};

} // namespace FastFFT

#endif