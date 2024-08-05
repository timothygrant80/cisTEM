#ifndef SRC_CPP_HELPER_FUNCTIONS_CUH_
#define SRC_CPP_HELPER_FUNCTIONS_CUH_

#ifndef FFT_DEBUG_LEVEL
#error "FFT_DEBUG_LEVEL not defined"
#endif

#ifndef FFT_DEBUG_STAGE
#error "FFT_DEBUG_STAGE not defined"
#endif

#include <iostream>

#include "../fastfft/Image.cuh"
#include "../../include/FastFFT.cuh"

// clang-format off
#define MyTestPrintAndExit(cond, ...) { if(! cond) {std::cerr << __VA_ARGS__ << " From: " << __FILE__ << ":" << __LINE__ << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1); }}

// clang-format on

void PrintArray(float2* array, short NX, short NY, short NZ, int line_wrapping = 34) {
    // COMPLEX TODO make these functions.
    int n = 0;
    for ( int z = 0; z < NZ; z++ ) {
        for ( int x = 0; x < NX; x++ ) {
            std::cout << x << "[ ";
            for ( int y = 0; y < NY; y++ ) {
                // TODO: could these use the indexing functions?
                std::cout << array[x + NX * (y + z * NY)].x << "," << array[x + NX * (y + z * NY)].y << " ";
                n++;
                if ( n == line_wrapping ) {
                    n = 0;
                    std::cout << std::endl;
                } // line wrapping
            }
            std::cout << "] " << NY << std::endl;
            n = 0;
        }
        if ( NZ > 0 )
            std::cout << " ... ... ... " << z << " ... ... ..." << std::endl;
    }
};

void PrintArray(float* array, short NX, short NY, short NZ, short NW, int line_wrapping = 34) {
    int n = 0;
    for ( int z = 0; z < NZ; z++ ) {
        for ( int x = 0; x < NX; x++ ) {

            std::cout << x << "[ ";
            for ( int y = 0; y < NY; y++ ) {
                std::cout << array[x + (2 * NW) * (y + z * NY)] << " ";
                n++;
                if ( n == line_wrapping ) {
                    n = 0;
                    std::cout << std::endl;
                } // line wrapping
            }
            std::cout << "] " << NY << std::endl;
            n = 0;
        }
        if ( NZ > 0 )
            std::cout << " ... ... ... " << z << " ... ... ..." << std::endl;
    }
};

void PrintArray_XZ(float2* array, short NX, short NY, short NZ, int line_wrapping = 34) {
    // COMPLEX TODO make these functions.
    int n = 0;
    for ( int x = 0; x < NX; x++ ) {
        for ( int z = 0; z < NZ; z++ ) {

            std::cout << z << "[ ";
            for ( int y = 0; y < NY; y++ ) {
                std::cout << array[z + NZ * (y + x * NY)].x << "," << array[z + NZ * (y + x * NY)].y << " ";
                n++;
                if ( n == line_wrapping ) {
                    n = 0;
                    std::cout << std::endl;
                } // line wrapping
            }
            std::cout << "] " << NY << std::endl;
            n = 0;
        }
        if ( NZ > 0 )
            std::cout << " ... ... ... " << x << " ... ... ..." << std::endl;
    }
};

template <typename realType, typename complexType>
void CheckUnitImpulseRealImage(Image<realType, complexType>& positive_control, int input_line) {

    long address = 0;
    // Loop over the real values z,y,x skipping the fft padding
    for ( int k = 0; k < positive_control.size.z; k++ ) {
        for ( int j = 0; j < positive_control.size.y; j++ ) {
            for ( int i = 0; i < positive_control.size.x; i++ ) {
                // Only check the address if we have too.
                if ( positive_control.real_values[address] != 0.0f && address != 0 ) {
                    PrintArray(positive_control.real_values, positive_control.size.x, positive_control.size.y, positive_control.size.z, positive_control.size.w);
                    MyTestPrintAndExit(false, "Check Unit Impulse Real control val " + std::to_string(positive_control.real_values[address]) + " different from zero at line " + std::to_string(input_line));
                }
                address++;
            }
            address += positive_control.padding_jump_value;
        }
    }
    return;
}

/**
 * @brief  For debugging the individual stages of the xforms
 *  Note: for some reason, passing by value altered the values while passing by reference did not. (Opposite?)
// eg.
 
 * @param test_image 
 * @param fwd_dims_in 
 * @param fwd_dims_out 
 * @param inv_dims_in 
 * @param inv_dims_out 
 * @param input_line 
 * @return true 
 * @return false 
 */
template <int fft_debug_stage, int Rank, typename realType, typename complexType>
bool debug_partial_fft(Image<realType, complexType>& test_image,
                       short4                        fwd_dims_in,
                       short4                        fwd_dims_out,
                       short4                        inv_dims_in,
                       short4                        inv_dims_out,
                       int                           input_line) {

    bool debug_stage_is_8 = false;
    if constexpr ( fft_debug_stage == 0 ) {
        PrintArray(test_image.real_values, fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.z, fwd_dims_in.w);
    }
    else if constexpr ( fft_debug_stage == 1 ) {
        if ( Rank == 2 )
            // Transformed X transposed XY
            PrintArray(test_image.complex_values, fwd_dims_in.y, fwd_dims_out.w, fwd_dims_in.z);
        else
            // Transformed X transposed XZ
            PrintArray(test_image.complex_values, fwd_dims_in.z, fwd_dims_in.y, fwd_dims_out.w);
    }
    else if constexpr ( fft_debug_stage == 2 ) {
        if ( Rank == 2 )
            // Noop, Transformed X transposed XY
            PrintArray(test_image.complex_values, fwd_dims_in.y, fwd_dims_out.w, fwd_dims_in.z);
        else
            // Transformed Z, permute XYZ
            PrintArray(test_image.complex_values, fwd_dims_in.y, fwd_dims_out.w, fwd_dims_out.z);
    }
    else if constexpr ( fft_debug_stage == 3 ) {
        if ( Rank == 2 )
            // Transormed Y, no reordering
            PrintArray(test_image.complex_values, fwd_dims_out.y, fwd_dims_out.w, fwd_dims_out.z);
        else
            // Transormed Y, no reordering
            PrintArray(test_image.complex_values, fwd_dims_out.y, fwd_dims_out.w, fwd_dims_out.z);
    }
    else if constexpr ( fft_debug_stage == 4 ) {
        // Same for 2d/3d intra-transorm op (if specified)
        PrintArray(test_image.complex_values, fwd_dims_out.y, fwd_dims_out.w, fwd_dims_out.z);
    }
    else if constexpr ( fft_debug_stage == 5 ) {
        if ( Rank == 2 )
            // Inv Transformed Y, no transpose
            PrintArray(test_image.complex_values, inv_dims_out.y, inv_dims_in.w, inv_dims_out.z);
        else
            // Inv Transformed Y, swap YZ
            PrintArray(test_image.complex_values, inv_dims_in.z, inv_dims_in.w, inv_dims_out.y);
    }
    else if constexpr ( fft_debug_stage == 6 ) {
        if ( Rank == 2 )
            // Nothing different from debug 5 for 2d
            PrintArray(test_image.complex_values, inv_dims_out.y, inv_dims_in.w, inv_dims_out.z);
        else
            // Inv Transformed Z, permute XYZ
            PrintArray(test_image.complex_values, inv_dims_in.w, inv_dims_out.y, inv_dims_out.z);
    }
    else if constexpr ( fft_debug_stage == 7 ) {
        if ( Rank == 2 )
            // Inv transformed X, no transpose
            PrintArray(test_image.real_values, inv_dims_out.x, inv_dims_out.y, inv_dims_out.z, inv_dims_out.w);
        else
            // Inv transformed X, no transpose
            PrintArray(test_image.real_values, inv_dims_out.x, inv_dims_out.y, inv_dims_out.z, inv_dims_out.w);
    }
    else if constexpr ( fft_debug_stage == 8 ) {
        debug_stage_is_8 = true;
    }
    else
        MyTestPrintAndExit(false, "FFT_DEBUG_STAGE not recognized " + std::to_string(FFT_DEBUG_STAGE));

    // std::cerr << "Debug stage " << fft_debug_stage << " passed." << std::endl;

    return debug_stage_is_8;
}

#endif