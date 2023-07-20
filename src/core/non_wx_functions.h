#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include "../constants/constants.h"

inline void ZeroBoolArray(bool* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = false;
    }
}

inline void ZeroIntArray(int* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0;
    }
}

inline void ZeroLongArray(long* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0;
    }
}

inline void ZeroFloatArray(float* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0.0;
    }
}

inline void ZeroDoubleArray(double* array_to_zero, int size_of_array) {
    for ( int counter = 0; counter < size_of_array; counter++ ) {
        array_to_zero[counter] = 0.0;
    }
}

inline bool IsEven(int number_to_check) {
    if ( number_to_check % 2 == 0 )
        return true;
    else
        return false;
}

inline int MakeEven(int number_to_make_even) {
    if ( IsEven(number_to_make_even) )
        return number_to_make_even;
    else
        return number_to_make_even + 1;
}

// Function to check if x is power of 2
inline bool is_power_of_two(int n) {
    if ( n == 0 )
        return false;
    return (ceil(log2((float)n)) == floor(log2((float)n)));
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

inline bool IsOdd(int number) {
    if ( (number & 1) == 0 )
        return false;
    else
        return true;
}

inline bool StartsWithDevNull(const std::string& wanted_filename) {
    if ( wanted_filename.size( ) > 8 ) {
        // It is long enough to point to /dev/null/...
        constexpr const char* dev_null{"/dev/null"};
        for ( int iChar = 0; iChar < 9; iChar++ ) {
            if ( wanted_filename[iChar] != dev_null[iChar] ) {
                return false;
            }
        }
    }
    else {
        return false;
    }
    return true;
}

inline float ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size) {
    return real_space_shift * distance_from_origin * 2.0 * pi_v<float> / dimension_size;
}

inline std::complex<float> Return3DPhaseFromIndividualDimensions(float phase_x, float phase_y, float phase_z) {
    float temp_phase = -phase_x - phase_y - phase_z;

    return cosf(temp_phase) + sinf(temp_phase) * I;
}

inline std::complex<float> MakeComplex(const float& real, const float& imag) {
    std::complex<float> ret_val(real, imag);
    return ret_val;
}

inline bool DoublesAreAlmostTheSame(double a, double b) {
    return (fabs(a - b) < 0.000001);
}

inline bool FloatsAreAlmostTheSame(float a, float b) {
    return (fabs(a - b) < 0.0001);
}

inline bool HalfFloatsAreAlmostTheSame(float a, float b) {
    return (fabs(a - b) < 0.001);
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

inline float ReturnWavelenthInAngstroms(float kV) {
    return 12.262643247 / sqrtf(kV * 1000 + 0.97845e-6 * powf(kV * 1000, 2));
}
