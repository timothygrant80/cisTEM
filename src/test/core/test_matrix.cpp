#include "../../core/matrix.h"
#include "../../core/non_wx_functions.h"
#include "../../../include/catch2/catch.hpp"

TEST_CASE("RotationMatrix class", "[RotationMatrix]") {

    using namespace Catch::literals;
    RotationMatrix rm;
    rm.SetToIdentity( );
    REQUIRE(rm.m[0][0] == 1.0f);
    REQUIRE(rm.m[0][1] == 0.0f);
    REQUIRE(rm.m[0][2] == 0.0f);
    REQUIRE(rm.m[1][0] == 0.0f);
    REQUIRE(rm.m[1][1] == 1.0f);
    REQUIRE(rm.m[1][2] == 0.0f);
    REQUIRE(rm.m[2][0] == 0.0f);
    REQUIRE(rm.m[2][1] == 0.0f);
    REQUIRE(rm.m[2][2] == 1.0f);

    rm.SetToConstant(2.0f);
    REQUIRE(rm.m[0][0] == 2.0f);
    REQUIRE(rm.m[0][1] == 2.0f);
    REQUIRE(rm.m[0][2] == 2.0f);
    REQUIRE(rm.m[1][0] == 2.0f);
    REQUIRE(rm.m[1][1] == 2.0f);
    REQUIRE(rm.m[1][2] == 2.0f);
    REQUIRE(rm.m[2][0] == 2.0f);
    REQUIRE(rm.m[2][1] == 2.0f);
    REQUIRE(rm.m[2][2] == 2.0f);

    // The rotation matrix can be set with Euler angles phi theta psi, which by themselves are used to rotate right handed coordinate systems
    // which results in a passive transformation of image coordinates. e.g. a vector identifying a 2d coordinate is rotate into 3d and that
    // value is interpolated and added to the 2d.

    // Positive rotations clockwise about the specified axis looking down the axis towards the origin

    // Z(phi) Y(theta) Z(psi)
    rm.SetToEulerRotation(90.0f, 0.0f, 0.0f);
    float x_in = 0.0f;
    float y_in = 1.0f;
    float z_in = 0.0f;

    float x, y, z;

    // Rotate the y axis onto the negative x axis
    rm.RotateCoords(x_in, y_in, z_in, x, y, z);
    REQUIRE(RelativeErrorIsLessThanEpsilon(x, -1.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));

    rm.SetToEulerRotation(0, 0, -90);
    // rotate the y axis onto the positive x axis. Note This should have the same effect as if the angle were phi instead of psi
    rm.RotateCoords(x_in, y_in, z_in, x, y, z);
    REQUIRE(RelativeErrorIsLessThanEpsilon(x, 1.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));

    // obviously the z rotations should annihilate if negative of each other
    rm.SetToEulerRotation(-90, 0, 90);
    rm.RotateCoords(x_in, y_in, z_in, x, y, z);
    REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(y, 1.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));

    // combining rotations gets a little more complicated
    // R( Z(phi) Y(theta) Z(psi) ) * vector can be interpreted as:
    // extrinsic rotations, the rotation matrix would result in rotations about a fixed coordinate system about Z(psi) then Y(theta) then Z(phi)
    // intrinsic rotations, the rotation matrix would result in rotations about the moving coordinate system // Z(phi) Y'(theta) Z''(psi)
    rm.SetToEulerRotation(0, 90, 90);
    rm.RotateCoords(x_in, y_in, z_in, x, y, z);
    REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(z, 1.0f));

    rm.SetToEulerRotation(90, 90, 0);
    rm.RotateCoords(x_in, y_in, z_in, x, y, z);
    REQUIRE(RelativeErrorIsLessThanEpsilon(x, -1.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));

    rm.SetToEulerRotation(90, 90, -90);
    rm.RotateCoords(x_in, y_in, z_in, x, y, z);
    REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
    REQUIRE(RelativeErrorIsLessThanEpsilon(z, -1.0f));
}