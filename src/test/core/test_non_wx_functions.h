#include "../../core/defines.h"
#include "../../core/non_wx_functions.h"
#include "../../../include/catch2/catch.hpp"

TEST_CASE("Array functions", "[non_wx_functions]") {

    // TODO: These functions should just be replaced by the std::array functions used
    // to test them.

    // The setup, doesn't really matter what they are so long as they are non-zero
    std::array<bool, 3>   bool_array   = {true, false, true};
    std::array<int, 3>    int_array    = {1, 2, 3};
    std::array<float, 3>  float_array  = {1.0f, 2.0f, 3.0f};
    std::array<long, 4>   long_array   = {1, 2, 3, 4};
    std::array<double, 3> double_array = {1.0, 2.0, 3.0};

    ZeroBoolArray(bool* bool_array.data( ), bool_array.size( ));
    for ( auto& b : bool_array ) {
        REQUIRE(b == false);
    }

    ZeroIntArray(int* int_array.data( ), int_array.size( ));
    for ( auto& i : int_array ) {
        REQUIRE(i == 0);
    }

    ZeroLongArray(long* long_array.data( ), long_array.size( ));
    for ( auto& l : long_array ) {
        REQUIRE(l == 0);
    }

    ZeroFloatArray(float* float_array.data( ), float_array.size( ));
    for ( auto& f : float_array ) {
        REQUIRE(f == 0.0f);
    }

    ZeroDoubleArray(double* double_array.data( ), double_array.size( ));
    for ( auto& d : double_array ) {
        REQUIRE(d == 0.0);
    }
}

TEST_CASE("Even Odd funcitons", "[non_wx_functions]") {

    std::array<int, 4> even_array = {2, 4, 6, 8};
    std::array<int, 4> odd_array  = {1, 3, 5, 7};

    for ( auto& e : even_array ) {
        REQUIRE(IsEven(e) == true);
    }

    for ( auto& o : odd_array ) {
        REQUIRE(IsEven(o) == false);
    }

    // TODO: how to assert dependency of these two test on the first two?
    for ( auto& e : even_array ) {
        REQUIRE(MakeEven(e) == IsEven(e));
    }

    for ( auto& o : odd_array ) {
        REQUIRE(MakeEven(o) != IsEven(o));
    }
}

// Test template for the is_power_of_two function
TEST_CASE("is_power_of_two", "[non_wx_functions]") {

    std::array<int, 4> power_of_two_array     = {2, 4, 8, 16};
    std::array<int, 4> not_power_of_two_array = {3, 5, 9, 15};

    for ( auto& p : power_of_two_array ) {
        REQUIRE(is_power_of_two(p) == true);
    }

    for ( auto& n : not_power_of_two_array ) {
        REQUIRE(is_power_of_two(n) == false);
    }
}

// Test the angle conversion functions
TEST_CASE("Angle conversion functions", "[non_wx_functions]") {

    std::array<float, 4> degree_array = {0.0, 90.0, 180.0, 270.0, 540.0, -180.0};
    std::array<float, 4> radian_array = {0.0, half_pi_v<float>, pi_v<float>, 3.0 * half_pi_v<float>, 3.0 * pi_v<float>, -pi_v<float>};

    for ( int i = 0; i < degree_array.size( ); i++ ) {
        REQUIRE(FloatsAreAlmostTheSame(deg_2_rad(degree_array[i]), radian_array[i]));
    }

    for ( int i = 0; i < radian_array.size( ); i++ ) {
        REQUIRE(FloatsAreAlmostTheSame(rad_2_deg(radian_array[i]), degree_array[i]));
    }

    for ( int i = 0; i < degree_array.size( ); i++ ) {
        REQUIRE(clamp_angular_range_0_to_2pi(degree_array[i] > 0.0f && degree_array[i] < 360.0f));
    }

    for ( int i = 0; i < radian_array.size( ); i++ ) {
        REQUIRE(clamp_angular_range_0_to_2pi(radian_array[i] > 0.0f && radian_array[i] < 2.0f * pi_v<float>));
    }

    for ( int i = 0; i < degree_array.size( ); i++ ) {
        REQUIRE(clamp_angular_range_minus_pi_to_pi(degree_array[i] > -180.0f && degree_array[i] < 180.0f));
    }

    for ( int i = 0; i < radian_array.size( ); i++ ) {
        REQUIRE(clamp_angular_range_minus_pi_to_pi(radian_array[i] > -pi_v<float> && radian_array[i] < pi_v<float>));
    }
}

// test the sinc function
TEST_CASE("Sinc function", "[non_wx_functions]") {

    std::array<float, 4> sinc_array = {1.0, 2.0, 3.0};

    REQUIRE(sinc(0.0) == 1.0);

    for ( int i = 0; i < sinc_array.size( ); i++ ) {
        REQUIRE(FloatsAreAlmostTheSame(sinc(sinc_array[i]), sin(sinc_array[i]) / sinc_array[i]));
    }
}

// test the check on filenames starting with /dev/null
TEST_CASE("Check for /dev/null", "[non_wx_functions]") {

    std::array<std::string, 4> dev_null_array     = {"/dev/null", "/dev/null/", "/dev/null/blah", "/dev/null/blah/blah"};
    std::array<std::string, 4> not_dev_null_array = {"dev/null", "/dev/nullish", "/tmp/null", "/dev/nullish/blah/blah"};

    for ( int i = 0; i < dev_null_array.size( ); i++ ) {
        REQUIRE(IsDevNull(dev_null_array[i]));
    }

    for ( int i = 0; i < not_dev_null_array.size( ); i++ ) {
        REQUIRE(! IsDevNull(not_dev_null_array[i]));
    }
}
