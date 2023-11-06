#include "../../core/non_wx_functions.h"
#include "../../../include/ieee-754-half/half.hpp"
#include "../../../include/catch2/catch.hpp"
#include <iostream>

// TODO [Deprecated]: If the forwarded type specific ZeroArray functions do not produce any known errors a year from now, remove the deprecated functions.
// Now being 2023-Nov-6

TEST_CASE("ZeroBoolArray sets all elements to false", "[ZeroBoolArray]") {
    bool arr[5] = {true, true, true, true, true};
    ZeroBoolArray(arr, 5);
    for ( int i = 0; i < 5; i++ ) {
        REQUIRE(arr[i] == false);
    }
}

TEST_CASE("ZeroIntArray sets all elements to 0", "[ZeroIntArray]") {
    int arr[5] = {1, 2, 3, 4, 5};
    ZeroIntArray(arr, 5);
    for ( int i = 0; i < 5; i++ ) {
        REQUIRE(arr[i] == 0);
    }
}

TEST_CASE("ZeroLongArray sets all elements to 0", "[ZeroLongArray]") {
    long arr[5] = {1, 2, 3, 4, 5};
    ZeroLongArray(arr, 5);
    for ( int i = 0; i < 5; i++ ) {
        REQUIRE(arr[i] == 0);
    }
}

TEST_CASE("ZeroFloatArray sets all elements to 0.0", "[ZeroFloatArray]") {
    float arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    ZeroFloatArray(arr, 5);
    for ( int i = 0; i < 5; i++ ) {
        REQUIRE(arr[i] == 0.0);
    }
}

TEST_CASE("ZeroDoubleArray sets all elements to 0.0", "[ZeroDoubleArray]") {
    double arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    ZeroDoubleArray(arr, 5);
    for ( int i = 0; i < 5; i++ ) {
        REQUIRE(arr[i] == 0.0);
    }
}

TEST_CASE("IsEven returns true for even numbers and false for odd numbers", "[IsEven]") {
    REQUIRE(IsEven(0) == true);
    REQUIRE(IsEven(1) == false);
    REQUIRE(IsEven(2) == true);
    REQUIRE(IsEven(3) == false);
    REQUIRE(IsEven(4) == true);
}

TEST_CASE("MakeEven returns the input number if it's even, or the next even number if it's odd", "[MakeEven]") {
    REQUIRE(MakeEven(0) == 0);
    REQUIRE(MakeEven(1) == 2);
    REQUIRE(MakeEven(2) == 2);
    REQUIRE(MakeEven(3) == 4);
    REQUIRE(MakeEven(4) == 4);
}

TEST_CASE("is_power_of_two returns true for powers of two and false otherwise", "[is_power_of_two]") {
    REQUIRE(is_power_of_two(0) == false);
    REQUIRE(is_power_of_two(1) == true);
    REQUIRE(is_power_of_two(2) == true);
    REQUIRE(is_power_of_two(3) == false);
    REQUIRE(is_power_of_two(4) == true);
    REQUIRE(is_power_of_two(5) == false);
    REQUIRE(is_power_of_two(8) == true);
}

TEST_CASE("rad_2_deg converts radians to degrees", "[rad_2_deg]") {
    REQUIRE(rad_2_deg(0.0) == 0.0);
    REQUIRE(rad_2_deg(pi_v<float> / 2) == 90.0);
    REQUIRE(rad_2_deg(pi_v<float>) == 180.0);
    REQUIRE(rad_2_deg(2 * pi_v<float>) == 360.0);
}

TEST_CASE("deg_2_rad converts degrees to radians", "[deg_2_rad]") {
    REQUIRE(deg_2_rad(0.0) == 0.0);
    REQUIRE(deg_2_rad(90.0) == pi_v<float> / 2);
    REQUIRE(deg_2_rad(180.0) == pi_v<float>);
    REQUIRE(deg_2_rad(360.0) == 2 * pi_v<float>);
}

TEST_CASE("clamp_angular_range_0_to_2pi clamps angles to the range (0, 2*pi]", "[clamp_angular_range_0_to_2pi]") {
    REQUIRE(clamp_angular_range_0_to_2pi(0.0) == Approx(0.0f));
    REQUIRE(clamp_angular_range_0_to_2pi(pi_v<float>) == Approx(pi_v<float>));
    REQUIRE(clamp_angular_range_0_to_2pi(2 * pi_v<float>) == Approx(0.0f));
    REQUIRE(clamp_angular_range_0_to_2pi(3 * pi_v<float>) == Approx(pi_v<float>));
}

TEST_CASE("clamp_angular_range_negative_pi_to_pi clamps angles to the range (-pi, pi]", "[clamp_angular_range_negative_pi_to_pi]") {
    REQUIRE(clamp_angular_range_negative_pi_to_pi(0.0) == Approx(0.0f));
    REQUIRE(clamp_angular_range_negative_pi_to_pi(-pi_v<float>) == Approx(pi_v<float>));
    REQUIRE(clamp_angular_range_negative_pi_to_pi(2 * pi_v<float>) == Approx(0.0f));
    REQUIRE(clamp_angular_range_negative_pi_to_pi(3 * pi_v<float>) == Approx(pi_v<float>));
}

TEST_CASE("sinc returns the sinc function of the input", "[sinc]") {
    REQUIRE(sinc(0.0) == 1.0);
    REQUIRE(sinc(0.01) == Approx(0.999983));
    REQUIRE(sinc(0.1) == Approx(0.998334));
    REQUIRE(sinc(1.0) == Approx(0.841471));
}

TEST_CASE("myround rounds the input to the nearest integer", "[myround]") {
    REQUIRE(myround(0.0) == 0.0);
    REQUIRE(myround(0.4) == 0.0);
    REQUIRE(myround(0.5) == 1.0);
    REQUIRE(myround(1.5) == 2.0);
    REQUIRE(myround(-0.4) == 0.0);
    REQUIRE(myround(-0.5) == -1.0);
    REQUIRE(myround(-1.5) == -2.0);
}

TEST_CASE("myroundint rounds the input to the nearest integer and returns an integer", "[myroundint]") {
    REQUIRE(myroundint(0.0) == 0);
    REQUIRE(myroundint(0.4) == 0);
    REQUIRE(myroundint(0.5) == 1);
    REQUIRE(myroundint(1.5) == 2);
    REQUIRE(myroundint(-0.4) == 0);
    REQUIRE(myroundint(-0.5) == -1);
    REQUIRE(myroundint(-1.5) == -2);
}

TEST_CASE("RoundAndMakeEven rounds the input to the nearest integer and makes it even", "[RoundAndMakeEven]") {
    REQUIRE(RoundAndMakeEven(0.0) == 0);
    REQUIRE(RoundAndMakeEven(0.4) == 0);
    REQUIRE(RoundAndMakeEven(0.5) == 2);
    REQUIRE(RoundAndMakeEven(1.5) == 2);
    REQUIRE(RoundAndMakeEven(-0.4) == 0);
    REQUIRE(RoundAndMakeEven(-0.5) == 0);
    REQUIRE(RoundAndMakeEven(-2.5) == -2);
}

TEST_CASE("IsOdd returns true for odd numbers and false for even numbers", "[IsOdd]") {
    REQUIRE(IsOdd(0) == false);
    REQUIRE(IsOdd(1) == true);
    REQUIRE(IsOdd(2) == false);
    REQUIRE(IsOdd(3) == true);
    REQUIRE(IsOdd(4) == false);
}

TEST_CASE("ReturnPhaseFromShift returns the phase from a real space shift", "[ReturnPhaseFromShift]") {
    REQUIRE(ReturnPhaseFromShift(0.0, 0.0, 1.0) == 0.0);
    REQUIRE(ReturnPhaseFromShift(1.0, 1.0, 2.0) == Approx(pi_v<float>));
}

TEST_CASE("Return3DPhaseFromIndividualDimensions returns the 3D phase from individual x, y, and z phases", "[Return3DPhaseFromIndividualDimensions]") {
    constexpr std::complex<float> expected(0.9601702f, 0.279415f);
    std::complex<float>           test_val = Return3DPhaseFromIndividualDimensions(1, 2, 3);

    REQUIRE(real(test_val) == Approx(real(expected)));
    REQUIRE(imag(test_val) == Approx(imag(expected)));
}

TEST_CASE("MakeComplex returns a complex number with the given real and imaginary parts", "[MakeComplex]") {
    std::complex<float> expected(1.0, 2.0);
    REQUIRE(MakeComplex(1.0, 2.0) == expected);
}

TEST_CASE("DoublesAreAlmostTheSame returns true if the difference between the two doubles is less than 1e-5", "[DoublesAreAlmostTheSame]") {
    REQUIRE(DoublesAreAlmostTheSame(1.0, 1.0) == true);
    REQUIRE(DoublesAreAlmostTheSame(1.0, 1.0000001) == true);
    REQUIRE(DoublesAreAlmostTheSame(1.0, 1.00001) == false);
}

TEST_CASE("FloatsAreAlmostTheSame returns true if the difference between the two floats is less than 1e-4", "[FloatsAreAlmostTheSame]") {
    REQUIRE(FloatsAreAlmostTheSame(1.0f, 1.0f) == true);
    REQUIRE(FloatsAreAlmostTheSame(1.0f, 1.00001f) == true);
    REQUIRE(FloatsAreAlmostTheSame(1.0f, 1.001f) == false);
}

TEST_CASE("HalfFloatsAreAlmostTheSame returns true if the difference between the two floats is less than 1e-3", "[HalfFloatsAreAlmostTheSame]") {
    REQUIRE(HalfFloatsAreAlmostTheSame(half_float::half(1.0f), half_float::half(1.0f)) == true);
    REQUIRE(HalfFloatsAreAlmostTheSame(half_float::half(1.0f), half_float::half(1.001f)) == true);
    REQUIRE(HalfFloatsAreAlmostTheSame(half_float::half(1.0f), half_float::half(1.01f)) == false);
}

TEST_CASE("RelativeErrorIsLessThanEpsilon returns true if the relative error between the reference and test values is less than epsilon", "[RelativeErrorIsLessThanEpsilon]") {
    constexpr bool print_message_if_failed = false;
    REQUIRE(RelativeErrorIsLessThanEpsilon(1.0, 1.0, print_message_if_failed) == true);
    REQUIRE(RelativeErrorIsLessThanEpsilon(1.0, 1.0001, print_message_if_failed) == true);
    REQUIRE(RelativeErrorIsLessThanEpsilon(1.0, 1.001, print_message_if_failed) == false);
    REQUIRE(RelativeErrorIsLessThanEpsilon(0.0, 0.0, print_message_if_failed) == true);
    REQUIRE(RelativeErrorIsLessThanEpsilon(0.0, 0.0001, print_message_if_failed) == false);
}

TEST_CASE("kDa_to_Angstrom3 converts kilo daltons to cubic angstroms", "[kDa_to_Angstrom3]") {
    // As much as this number looks like a joke : )
    REQUIRE(kDa_to_Angstrom3(1.0) == Approx(1234.5679));
}

TEST_CASE("ConvertProjectionXYToThetaInDegrees converts projection x and y to theta in degrees", "[ConvertProjectionXYToThetaInDegrees]") {
    // I Don't undertand the logic behind this function, so can't write a test.
    // FIXME:
}

TEST_CASE("ConvertXYToPhiInDegrees converts x and y to phi in degrees", "[ConvertXYToPhiInDegrees]") {
    // I Don't undertand the logic behind this function, so can't write a test.
    // FIXME:
}

TEST_CASE("ReturnWavelengthInAngstroms returns the wavelength in angstroms for a given kV", "[ReturnWavelengthInAngstroms]") {
    REQUIRE(FloatsAreAlmostTheSame(ReturnWavelenthInAngstroms(300.0f), 0.01969f));
    REQUIRE(FloatsAreAlmostTheSame(ReturnWavelenthInAngstroms(200.0f), 0.02508f));
    REQUIRE(FloatsAreAlmostTheSame(ReturnWavelenthInAngstroms(120.0f), 0.03349f));
}

// test the check on filenames starting with /dev/null
TEST_CASE("Check for /dev/null", "[non_wx_functions]") {

    std::array<std::string, 4> dev_null_array     = {"/dev/null", "/dev/null/", "/dev/null/blah", "/dev/null/blah/blah"};
    std::array<std::string, 4> not_dev_null_array = {"dev/null", "/dev/nullish", "/tmp/null", "/dev/nullish/blah/blah"};

    for ( int i = 0; i < dev_null_array.size( ); i++ ) {
        REQUIRE(StartsWithDevNull(dev_null_array[i]));
    }

    for ( int i = 0; i < not_dev_null_array.size( ); i++ ) {
        REQUIRE(StartsWithDevNull(not_dev_null_array[i]) == false);
    }
}
