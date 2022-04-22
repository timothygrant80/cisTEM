#include "../../core/matrix.h"
#include "../../../include/catch2/catch.hpp"

TEST_CASE("RotationMatrix class", "[RotationMatrix]") {
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

    REQUIRE(1 == 0)
}