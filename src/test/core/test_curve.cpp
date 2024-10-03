#include "../../core/core_headers.h"
#include "../../../include/catch2/catch.hpp"

/*
Coverate is okay, but does not cover any of the fitting routines.
*/

constexpr int number_of_points = 8;

void set_default_y_data(Curve& curve) {
    curve.data_y = {0.0, 1.0, 2.0, 3.0, 4.0, 0.0, -1.0, -2.0};
}

Curve initialize_curve( ) {
    Curve curve;
    curve.SetupXAxis(0.0, 10.0, number_of_points);
    set_default_y_data(curve);
    return curve;
}

TEST_CASE("Curve basic setup and intialization", "[Curve]") {

    Curve curve = initialize_curve( );

    REQUIRE(curve.data_x.size( ) == number_of_points);

    set_default_y_data(curve);

    REQUIRE(curve.data_x.size( ) == curve.data_y.size( ));
    REQUIRE(curve.NumberOfPoints( ) == number_of_points);

    Curve other_curve(curve);

    curve.ClearData( );

    REQUIRE(curve.data_x.size( ) == 0);
    REQUIRE(curve.data_y.size( ) == 0);

    REQUIRE(other_curve.data_x.size( ) == number_of_points);
    REQUIRE(other_curve.data_y.size( ) == number_of_points);

    other_curve.AddPoint(10.0, 10.0);
    REQUIRE(other_curve.data_x.size( ) == number_of_points + 1);
    REQUIRE(other_curve.data_y.size( ) == number_of_points + 1);
}

TEST_CASE("Curve basic arithmetic", "[Curve arithmetic]") {

    Curve curve = initialize_curve( );

    Curve other_curve;
    other_curve = curve;

    curve.MultiplyByConstant(2.0);
    for ( int i = 0; i < number_of_points; i++ ) {
        REQUIRE(curve.data_y[i] == 2.0 * other_curve.data_y[i]);
    }

    set_default_y_data(curve);
    curve.AddWith(&other_curve);
    curve.AddWith(&other_curve);
    for ( int i = 0; i < number_of_points; i++ ) {
        REQUIRE(curve.data_y[i] == 3.0 * other_curve.data_y[i]);
    }

    set_default_y_data(curve);
    curve.DivideBy(&other_curve);
    for ( int i = 0; i < number_of_points; i++ ) {
        REQUIRE((curve.data_y[i] == 1.0 || curve.data_y[i] == 0.0));
    }

    set_default_y_data(curve);
    curve.AddConstant(10.0);
    for ( int i = 0; i < number_of_points; i++ ) {
        REQUIRE(curve.data_y[i] == other_curve.data_y[i] + 10.0);
    }

    set_default_y_data(curve);
    curve.MultiplyByConstant(2.0);
    for ( int i = 0; i < number_of_points; i++ ) {
        REQUIRE(curve.data_y[i] == other_curve.data_y[i] * 2.0);
    }
}

TEST_CASE("Curve truncation/flattening", "[Curve clipping]") {
    Curve curve = initialize_curve( );
    Curve other_curve(curve);

    set_default_y_data(curve);
    curve.ZeroAfterIndex(3);
    for ( int i = 0; i < number_of_points; i++ ) {
        if ( i > 3 ) {
            REQUIRE(curve.data_y[i] == 0.0);
        }
        else {
            REQUIRE(curve.data_y[i] == other_curve.data_y[i]);
        }
    }

    set_default_y_data(curve);
    curve.FlattenBeforeIndex(3);
    for ( int i = 0; i < number_of_points; i++ ) {
        if ( i < 3 ) {
            REQUIRE(curve.data_y[i] == other_curve.data_y[3]);
        }
        else {
            REQUIRE(curve.data_y[i] == other_curve.data_y[i]);
        }
    }
}

TEST_CASE("Curve assignment ops", "[Curve value assignment]") {
    Curve curve = initialize_curve( );

    set_default_y_data(curve);
    curve.ZeroYData( );
    for ( int i = 0; i < number_of_points; i++ ) {
        REQUIRE(curve.data_y[i] == 0.0);
    }

    set_default_y_data(curve);
    curve.SetYToConstant(10.0);
    for ( int i = 0; i < number_of_points; i++ ) {
        REQUIRE(curve.data_y[i] == 10.0);
    }
}

TEST_CASE("Curve statistics and comparisons", "[Curve stats]") {
    Curve curve = initialize_curve( );
    Curve other_curve(curve);

    float max_val, mode;
    set_default_y_data(curve);
    curve.ComputeMaximumValueAndMode(max_val, mode);
    REQUIRE(max_val == 4.0);
    REQUIRE(FloatsAreAlmostTheSame(mode, 5.71429f));

    float returned_mode = curve.ReturnMode( );
    REQUIRE(returned_mode == mode);

    float returned_max_val = curve.ReturnMaximumValue( );
    REQUIRE(returned_max_val == max_val);

    set_default_y_data(curve);
    set_default_y_data(other_curve);

    REQUIRE(curve.YIsAlmostEqual(other_curve) == true);
    curve.AddConstant(0.0000001f);
    REQUIRE(curve.YIsAlmostEqual(other_curve) == true);
    curve.AddConstant(0.0001f);
    REQUIRE(curve.YIsAlmostEqual(other_curve) == false);

    set_default_y_data(curve);
    curve.Absolute( );
    REQUIRE(curve.data_y[2] == curve.data_y.back( ));

    // The next few require all positive, so we run after Absolute
    curve.NormalizeMaximumValue( );
    REQUIRE(FloatsAreAlmostTheSame(curve.data_y[3], 0.75f));

    float average = curve.ReturnAverageValue( );
    REQUIRE(FloatsAreAlmostTheSame(average, 3.25f / number_of_points));
}