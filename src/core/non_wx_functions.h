#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include "../constants/constants.h"
#include "../../include/ieee-754-half/half.hpp"

// TODO [Deprecated]: If the forwarded type specific ZeroArray functions do not produce any known errors a year from now, remove the deprecated functions.
// Now being 2023-Nov-6

/**
 * @brief Zero out an array of any type, with bool(0) being false.
 * 
 * @tparam T_ (Type deduction should always work)
 * @param array_to_zero 
 * @param size_of_array 
 */
template <typename T_>
inline void ZeroArray(T_* array_to_zero, int size_of_array) {
    // Note: the SIMD directives only seem to impact float & bool when tested, but it was a 4x speedup.
    if constexpr ( std::is_same_v<T_, bool> ) {
#pragma omp simd
        for ( int counter = 0; counter < size_of_array; counter++ )
            array_to_zero[counter] = false;
    }
    else {
        constexpr T_ zero_value = T_(0);
#pragma omp simd
        for ( int counter = 0; counter < size_of_array; counter++ )
            array_to_zero[counter] = zero_value;
    }
}

/**
 * @brief [Deprecated] please use ZeroArray for new work.
 * 
 * @param array_to_zero bool specific
 * @param size_of_array 
 */
inline void ZeroBoolArray(bool* array_to_zero, int size_of_array) {
    ZeroArray(array_to_zero, size_of_array);
}

/**
 * @brief [Deprecated] please use ZeroArray for new work.
 * 
 * @param array_to_zero int specific
 * @param size_of_array 
 */
inline void ZeroIntArray(int* array_to_zero, int size_of_array) {
    ZeroArray(array_to_zero, size_of_array);
}

/**
 * @brief [Deprecated] please use ZeroArray for new work.
 * 
 * @param array_to_zero long specific
 * @param size_of_array 
 */
inline void ZeroLongArray(long* array_to_zero, int size_of_array) {
    ZeroArray(array_to_zero, size_of_array);
}

/**
 * @brief [Deprecated] please use ZeroArray for new work.
 * 
 * @param array_to_zero float specific
 * @param size_of_array 
 */
inline void ZeroFloatArray(float* array_to_zero, int size_of_array) {
    ZeroArray(array_to_zero, size_of_array);
}

/**
 * @brief [Deprecated] please use ZeroArray for new work.
 * 
 * @param array_to_zero double specific
 * @param size_of_array 
 */
inline void ZeroDoubleArray(double* array_to_zero, int size_of_array) {
    ZeroArray(array_to_zero, size_of_array);
}

inline bool IsOdd(const int number) {
    if ( (number & 1) == 0 )
        return false;
    else
        return true;
}

inline bool IsEven(const int number_to_check) {
    return ! IsOdd(number_to_check);
}

/**
 * @brief Returns an even value, always larger than the input value.
 * WARNING: This is intended for resizing/padding where number to make even is > 0.
 * Negative ints will be made closer to zero, which may not be what you want.
 * 
 * @param number_to_make_even 
 * @return int 
 */
inline int MakeEven(int number_to_make_even) {
    if ( IsEven(number_to_make_even) )
        return number_to_make_even;
    else
        return number_to_make_even + 1;
}

inline int get_next_power_of_two(const int input_value) {
    int tmp_val = 1;
    while ( tmp_val < input_value )
        tmp_val = tmp_val << 1;

    return tmp_val;
}

inline bool is_power_of_two(const int input_value) {
    int tmp_val = get_next_power_of_two(input_value);

    if ( tmp_val > input_value )
        return false;
    else
        return true;
}

inline float rad_2_deg(float radians) {
    constexpr float scale_factor = (180.f / pi_v<float>);
    return radians * scale_factor;
}

inline float deg_2_rad(float degrees) {
    constexpr float scale_factor = (pi_v<float> / 180.f);
    return degrees * scale_factor;
}

inline float clamp_angular_range_0_to_2pi(float angle, bool units_are_degrees = false) {
    // Clamps the angle to be in the range ( 0,+360 ] { exclusive, inclusive }
    if ( units_are_degrees ) {
        angle = fmodf(angle, 360.0f);
    }
    else {
        angle = fmodf(angle, 2.0f * pi_v<float>);
    }
    return angle;
}

inline float clamp_angular_range_negative_pi_to_pi(float angle, bool units_are_degrees = false) {
    // Clamps the angle to be in the range ( -180,+180 ] { exclusive, inclusive }
    if ( units_are_degrees ) {
        angle = fmodf(angle, 360.0f);
        if ( angle > 180.0f )
            angle -= 360.0f;
        if ( angle <= -180.0f )
            angle += 360.0f;
        ;
    }
    else {
        angle = fmodf(angle, 2.0f * pi_v<float>);
        if ( angle > pi_v<float> )
            angle -= 2.0f * pi_v<float>;
        if ( angle <= -pi_v<float> )
            angle += 2.0f * pi_v<float>;
    }
    return angle;
}

inline float sinc(float radians) {
    if ( radians == 0.0 )
        return 1.0;
    if ( radians >= 0.01 )
        return sinf(radians) / radians;
    float temp_float = radians * radians;
    return 1.0 - temp_float / 6.0 + temp_float * temp_float / 120.0;
}

inline double myround(double a) {
    if ( a > 0 )
        return double(long(a + 0.5));
    else
        return double(long(a - 0.5));
}

inline float myround(float a) {
    if ( a > 0 )
        return float(int(a + 0.5));
    else
        return float(int(a - 0.5));
}

inline int myroundint(double a) {
    if ( a > 0 )
        return int(a + 0.5);
    else
        return int(a - 0.5);
}

inline int myroundint(float a) {
    if ( a > 0 )
        return int(a + 0.5);
    else
        return int(a - 0.5);
}

inline int RoundAndMakeEven(float a) {
    return MakeEven(myroundint(a));
}

/**
 * @brief This function reproduces the modulo operation in Python, which is different than C++ for negative numbers.
 * 
 * @param a 
 * @param b 
 * @return int 
 */
inline int PythonLikeModulo(int a, int b) {
    if ( a < 0 || b < 0 )
        return ((a % b) + b) % b;
    else
        return a % b;
}

/**
 * @brief Some functions will fail if their output is directed at /dev/null, so this function is here to prevent that.
 * Examples may be found in numeric_text_file and mrc_file.cpp
 * 
 * @param wanted_filename 
 * @return true 
 * @return false 
 */
inline bool StartsWithDevNull(const std::string& wanted_filename) {
    switch ( wanted_filename.length( ) ) {
        case 9:
            return (wanted_filename == "/dev/null");
        case 10:
            return (wanted_filename == "/dev/null/");
        default:
            if ( wanted_filename.length( ) > 10 ) {
                return (wanted_filename.substr(0, 10) == "/dev/null/");
            }
            else {
                return false;
            }
    }
}

/**
 * @brief Given a real-space shift, calculate the phase factor needed to apply the shift in Fourier space.
 * Note: The same method could be used to shift the complex values in a Fourier transform by applying them to 
 * a complex-valued "real space" image.
 * 
 * @param real_space_shift 
 * @param distance_from_origin 
 * @param dimension_size 
 * @return float 
 */
inline float ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size) {
    return real_space_shift * distance_from_origin * 2.0f * pi_v<float> / dimension_size;
}

/**
 * @brief Returns the complex exponential from converting the cartesian phase shifts into a single 3d.
 * 
 * @param phase_x 
 * @param phase_y 
 * @param phase_z 
 * @return std::complex<float> 
 */
inline std::complex<float> Return3DPhaseFromIndividualDimensions(float phase_x, float phase_y, float phase_z) {
    float temp_phase = -phase_x - phase_y - phase_z;

    return cosf(temp_phase) + sinf(temp_phase) * I;
}

inline std::complex<float> MakeComplex(const float& real, const float& imag) {
    std::complex<float> ret_val(real, imag);
    return ret_val;
}

inline bool DoublesAreAlmostTheSame(double a, double b) {
    return (fabs(a - b) < cistem::double_epsilon);
}

inline bool FloatsAreAlmostTheSame(float a, float b) {
    return (fabs(a - b) < cistem::float_epsilon);
}

inline bool HalfFloatsAreAlmostTheSame(const half_float::half a, const half_float::half b) {
    return (fabs(float(a) - float(b)) < cistem::half_float_epsilon);
}

template <typename T>
bool RelativeErrorIsLessThanEpsilon(T reference, T test_value, bool print_if_failed = true, T epsilon = 0.0001) {

    bool ret_val;
    // I'm not sure if this is the best way to guard against very small division
    if ( abs(reference) < epsilon || abs(test_value) < epsilon )
        ret_val = (std::abs((reference - test_value)) < epsilon);
    else
        ret_val = (std::abs((reference - test_value) / reference) < epsilon);

    if ( print_if_failed && ! ret_val ) {
        std::cerr << "RelativeErrorIsLessThanEpsilon failed: " << reference << " " << test_value << " " << epsilon << " " << std::abs((reference - test_value) / reference) << std::endl;
    }
    return ret_val;
};

inline float kDa_to_Angstrom3(float kilo_daltons) {
    return kilo_daltons * 1000.0 / 0.81;
}

// assumes that max x,y is 1
inline float ConvertProjectionXYToThetaInDegrees(float x, float y) {
    return rad_2_deg(asin(sqrtf(powf(x, 2) + powf(y, 2))));
}

// FIXME: This must assume a given period, -pi pi probably? Should it be clamped?
inline float ConvertXYToPhiInDegrees(float x, float y) {
    if ( x == 0 && y == 0 )
        return 0;
    else
        return rad_2_deg(atan2f(y, x));
}

inline float ReturnWavelenthInAngstroms(float acceleration_voltage_kV) {
    return 12.2639f / sqrtf(1000.0f * acceleration_voltage_kV + 0.97845e-6 * powf(1000.0f * acceleration_voltage_kV, 2));
}
