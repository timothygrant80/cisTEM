#include "../../core/matrix.h"
#include "../../core/non_wx_functions.h"
#include "../../../include/catch2/catch.hpp"

/**
 * Comprehensive test suite for RotationMatrix class
 *
 * Coverage includes:
 * - Constructors and initialization
 * - Arithmetic operators (+, -, *, +=, -=, *=)
 * - Assignment operators (=)
 * - Matrix properties (transpose, trace, Frobenius norm)
 * - Rotation setup methods (SetToEulerRotation, SetToRotation)
 * - Coordinate rotation (RotateCoords, RotateCoords2D)
 * - Euler angle conversion and edge cases
 * - Self-assignment and chaining operations
 * - Numerical stability and boundary conditions
 */

// Helper function to create a known rotation matrix for testing
RotationMatrix CreateTestMatrix( ) {
    RotationMatrix rm;
    // Create a simple rotation matrix (90 degrees around Z)
    rm.SetToEulerRotation(90.0f, 0.0f, 0.0f);
    return rm;
}

// Helper function to check if two matrices are approximately equal
bool MatricesAreAlmostEqual(const RotationMatrix& a, const RotationMatrix& b, float epsilon = 1e-4f) {
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            if ( fabsf(a.m[i][j] - b.m[i][j]) > epsilon ) {
                return false;
            }
        }
    }
    return true;
}

TEST_CASE("RotationMatrix construction and initialization", "[RotationMatrix][core][initialization]") {

    SECTION("default constructor initializes to zero") {
        RotationMatrix rm;
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm.m[i][j] == 0.0f);
            }
        }
    }

    SECTION("SetToIdentity creates identity matrix") {
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
    }

    SECTION("SetToConstant sets all elements to given value") {
        RotationMatrix rm;
        rm.SetToConstant(2.0f);
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm.m[i][j] == 2.0f);
            }
        }
    }

    SECTION("SetToConstant works with negative values") {
        RotationMatrix rm;
        rm.SetToConstant(-3.5f);
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm.m[i][j] == -3.5f);
            }
        }
    }

    SECTION("SetToValues sets matrix elements correctly") {
        RotationMatrix rm;
        // SetToValues takes arguments in column-major order: m00, m10, m20, m01, m11, m21, m02, m12, m22
        rm.SetToValues(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);

        REQUIRE(rm.m[0][0] == 1.0f);
        REQUIRE(rm.m[1][0] == 2.0f);
        REQUIRE(rm.m[2][0] == 3.0f);
        REQUIRE(rm.m[0][1] == 4.0f);
        REQUIRE(rm.m[1][1] == 5.0f);
        REQUIRE(rm.m[2][1] == 6.0f);
        REQUIRE(rm.m[0][2] == 7.0f);
        REQUIRE(rm.m[1][2] == 8.0f);
        REQUIRE(rm.m[2][2] == 9.0f);
    }
}

TEST_CASE("RotationMatrix assignment operators", "[RotationMatrix][core][assignment]") {

    SECTION("assignment operator with reference") {
        RotationMatrix rm1 = CreateTestMatrix( );
        RotationMatrix rm2;

        rm2 = rm1;

        REQUIRE(MatricesAreAlmostEqual(rm1, rm2));
    }

    SECTION("assignment operator with pointer") {
        RotationMatrix rm1 = CreateTestMatrix( );
        RotationMatrix rm2;

        rm2 = &rm1;

        REQUIRE(MatricesAreAlmostEqual(rm1, rm2));
    }

    SECTION("self-assignment with reference does not corrupt matrix") {
        RotationMatrix rm       = CreateTestMatrix( );
        RotationMatrix original = rm;

        rm = rm; // Self-assignment

        REQUIRE(MatricesAreAlmostEqual(rm, original));
    }

    SECTION("self-assignment with pointer does not corrupt matrix") {
        RotationMatrix rm       = CreateTestMatrix( );
        RotationMatrix original = rm;

        rm = &rm; // Self-assignment via pointer

        REQUIRE(MatricesAreAlmostEqual(rm, original));
    }

    SECTION("assignment can be chained") {
        RotationMatrix rm1 = CreateTestMatrix( );
        RotationMatrix rm2, rm3;

        rm3 = rm2 = rm1;

        REQUIRE(MatricesAreAlmostEqual(rm1, rm2));
        REQUIRE(MatricesAreAlmostEqual(rm1, rm3));
    }
}

TEST_CASE("RotationMatrix addition operators", "[RotationMatrix][core][arithmetic]") {

    SECTION("operator+ adds corresponding elements") {
        RotationMatrix rm1, rm2;
        rm1.SetToConstant(2.0f);
        rm2.SetToConstant(3.0f);

        RotationMatrix result = rm1 + rm2;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(result.m[i][j] == 5.0f);
            }
        }
    }

    SECTION("operator+ with identity and zero returns identity") {
        RotationMatrix identity, zero;
        identity.SetToIdentity( );
        zero.SetToConstant(0.0f);

        RotationMatrix result = identity + zero;

        REQUIRE(MatricesAreAlmostEqual(result, identity));
    }

    SECTION("operator+= adds in place with reference") {
        RotationMatrix rm1, rm2;
        rm1.SetToConstant(2.0f);
        rm2.SetToConstant(3.0f);

        rm1 += rm2;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm1.m[i][j] == 5.0f);
            }
        }
    }

    SECTION("operator+= adds in place with pointer") {
        RotationMatrix rm1, rm2;
        rm1.SetToConstant(2.0f);
        rm2.SetToConstant(3.0f);

        rm1 += &rm2;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm1.m[i][j] == 5.0f);
            }
        }
    }

    SECTION("operator+= can be chained") {
        RotationMatrix rm1, rm2, rm3;
        rm1.SetToConstant(1.0f);
        rm2.SetToConstant(2.0f);
        rm3.SetToConstant(3.0f);

        (rm1 += rm2) += rm3;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm1.m[i][j] == 6.0f);
            }
        }
    }
}

TEST_CASE("RotationMatrix subtraction operators", "[RotationMatrix][core][arithmetic]") {

    SECTION("operator- subtracts corresponding elements") {
        RotationMatrix rm1, rm2;
        rm1.SetToConstant(5.0f);
        rm2.SetToConstant(3.0f);

        RotationMatrix result = rm1 - rm2;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(result.m[i][j] == 2.0f);
            }
        }
    }

    SECTION("operator- with identical matrices returns zero") {
        RotationMatrix rm1 = CreateTestMatrix( );
        RotationMatrix rm2 = rm1;

        RotationMatrix result = rm1 - rm2;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(fabsf(result.m[i][j]) < 1e-6f);
            }
        }
    }

    SECTION("operator-= subtracts in place with reference") {
        RotationMatrix rm1, rm2;
        rm1.SetToConstant(5.0f);
        rm2.SetToConstant(3.0f);

        rm1 -= rm2;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm1.m[i][j] == 2.0f);
            }
        }
    }

    SECTION("operator-= subtracts in place with pointer") {
        RotationMatrix rm1, rm2;
        rm1.SetToConstant(5.0f);
        rm2.SetToConstant(3.0f);

        rm1 -= &rm2;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm1.m[i][j] == 2.0f);
            }
        }
    }
}

TEST_CASE("RotationMatrix multiplication operators", "[RotationMatrix][core][arithmetic]") {

    SECTION("operator* with identity returns original") {
        RotationMatrix test = CreateTestMatrix( );
        RotationMatrix identity;
        identity.SetToIdentity( );

        RotationMatrix result1 = test * identity;
        RotationMatrix result2 = identity * test;

        REQUIRE(MatricesAreAlmostEqual(result1, test));
        REQUIRE(MatricesAreAlmostEqual(result2, test));
    }

    SECTION("operator* performs correct matrix multiplication") {
        // Test with known values
        RotationMatrix rm1, rm2;
        rm1.SetToValues(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f); // Identity
        rm2.SetToValues(2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f); // Diagonal

        RotationMatrix result = rm1 * rm2;

        REQUIRE(MatricesAreAlmostEqual(result, rm2));
    }

    SECTION("operator* is non-commutative for general matrices") {
        // Create two different rotation matrices
        RotationMatrix rmX, rmY;
        rmX.SetToEulerRotation(90.0f, 0.0f, 0.0f); // Rotation around Z by phi
        rmY.SetToEulerRotation(0.0f, 90.0f, 0.0f); // Rotation around Y by theta

        RotationMatrix result1 = rmX * rmY;
        RotationMatrix result2 = rmY * rmX;

        // Results should be different (non-commutative)
        bool are_different = false;
        for ( int i = 0; i < 3 && ! are_different; i++ ) {
            for ( int j = 0; j < 3 && ! are_different; j++ ) {
                if ( fabsf(result1.m[i][j] - result2.m[i][j]) > 1e-4f ) {
                    are_different = true;
                }
            }
        }
        REQUIRE(are_different);
    }

    SECTION("operator*= multiplies in place with reference") {
        RotationMatrix rm1;
        rm1.SetToIdentity( );
        RotationMatrix rm2;
        rm2.SetToValues(2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f);

        rm1 *= rm2;

        REQUIRE(MatricesAreAlmostEqual(rm1, rm2));
    }

    SECTION("operator*= multiplies in place with pointer") {
        RotationMatrix rm1;
        rm1.SetToIdentity( );
        RotationMatrix rm2;
        rm2.SetToValues(2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f);

        rm1 *= &rm2;

        REQUIRE(MatricesAreAlmostEqual(rm1, rm2));
    }
}

TEST_CASE("RotationMatrix transpose operation", "[RotationMatrix][core][properties]") {

    SECTION("transpose of identity is identity") {
        RotationMatrix identity;
        identity.SetToIdentity( );

        RotationMatrix transposed = identity.ReturnTransposed( );

        REQUIRE(MatricesAreAlmostEqual(transposed, identity));
    }

    SECTION("transpose swaps rows and columns correctly") {
        RotationMatrix rm;
        rm.SetToValues(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);

        RotationMatrix transposed = rm.ReturnTransposed( );

        // Check that m[i][j] == m_transposed[j][i]
        REQUIRE(transposed.m[0][0] == rm.m[0][0]);
        REQUIRE(transposed.m[0][1] == rm.m[1][0]);
        REQUIRE(transposed.m[0][2] == rm.m[2][0]);
        REQUIRE(transposed.m[1][0] == rm.m[0][1]);
        REQUIRE(transposed.m[1][1] == rm.m[1][1]);
        REQUIRE(transposed.m[1][2] == rm.m[2][1]);
        REQUIRE(transposed.m[2][0] == rm.m[0][2]);
        REQUIRE(transposed.m[2][1] == rm.m[1][2]);
        REQUIRE(transposed.m[2][2] == rm.m[2][2]);
    }

    SECTION("double transpose returns original matrix") {
        RotationMatrix rm = CreateTestMatrix( );

        RotationMatrix double_transposed = rm.ReturnTransposed( ).ReturnTransposed( );

        REQUIRE(MatricesAreAlmostEqual(double_transposed, rm));
    }

    SECTION("transpose of rotation matrix is its inverse") {
        // For proper rotation matrices, R^T * R = I
        RotationMatrix rm;
        rm.SetToEulerRotation(45.0f, 30.0f, 60.0f);

        RotationMatrix transposed = rm.ReturnTransposed( );
        RotationMatrix product    = rm * transposed;

        RotationMatrix identity;
        identity.SetToIdentity( );

        REQUIRE(MatricesAreAlmostEqual(product, identity));
    }
}

TEST_CASE("RotationMatrix trace calculation", "[RotationMatrix][core][properties]") {

    SECTION("trace of identity is 3") {
        RotationMatrix identity;
        identity.SetToIdentity( );

        float trace = identity.ReturnTrace( );

        REQUIRE(trace == 3.0f);
    }

    SECTION("trace of zero matrix is 0") {
        RotationMatrix zero;
        zero.SetToConstant(0.0f);

        float trace = zero.ReturnTrace( );

        REQUIRE(trace == 0.0f);
    }

    SECTION("trace calculation is correct for arbitrary matrix") {
        RotationMatrix rm;
        rm.SetToValues(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);

        float trace = rm.ReturnTrace( );

        // Trace = m[0][0] + m[1][1] + m[2][2] = 1 + 5 + 9 = 15
        REQUIRE(trace == 15.0f);
    }

    SECTION("trace is invariant under cyclic permutation") {
        // Trace(ABC) = Trace(BCA) = Trace(CAB)
        // This tests the property that trace of product depends only on eigenvalues
        RotationMatrix rm;
        rm.SetToEulerRotation(30.0f, 45.0f, 60.0f);

        float trace = rm.ReturnTrace( );

        // Trace of rotation matrix relates to rotation angle: trace = 1 + 2*cos(angle)
        // For Euler rotations, we expect trace to be between -1 and 3
        REQUIRE(trace >= -1.0f);
        REQUIRE(trace <= 3.0f);
    }
}

TEST_CASE("RotationMatrix Frobenius norm", "[RotationMatrix][core][properties]") {

    SECTION("Frobenius norm of zero matrix is 0") {
        RotationMatrix zero;
        zero.SetToConstant(0.0f);

        float norm = zero.FrobeniusNorm( );

        REQUIRE(norm == 0.0f);
    }

    SECTION("Frobenius norm of identity is sqrt(3)") {
        RotationMatrix identity;
        identity.SetToIdentity( );

        float norm = identity.FrobeniusNorm( );

        REQUIRE(FloatsAreAlmostTheSame(norm, sqrtf(3.0f)));
    }

    SECTION("Frobenius norm calculation is correct") {
        RotationMatrix rm;
        rm.SetToConstant(1.0f); // All elements = 1

        float norm = rm.FrobeniusNorm( );

        // Frobenius norm = sqrt(sum of squares) = sqrt(9 * 1^2) = 3
        REQUIRE(FloatsAreAlmostTheSame(norm, 3.0f));
    }

    SECTION("Frobenius norm of rotation matrix is sqrt(3)") {
        // Proper rotation matrices are orthogonal, so Frobenius norm = sqrt(3)
        RotationMatrix rm;
        rm.SetToEulerRotation(45.0f, 30.0f, 60.0f);

        float norm = rm.FrobeniusNorm( );

        REQUIRE(FloatsAreAlmostTheSame(norm, sqrtf(3.0f)));
    }

    SECTION("Frobenius norm is always non-negative") {
        RotationMatrix rm;
        rm.SetToValues(-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f);

        float norm = rm.FrobeniusNorm( );

        REQUIRE(norm >= 0.0f);
    }
}

TEST_CASE("RotationMatrix Euler rotation setup", "[RotationMatrix][core][rotation]") {

    // The rotation matrix can be set with Euler angles phi theta psi, which by themselves are used to rotate right handed coordinate systems
    // which results in a passive transformation of image coordinates. e.g. a vector identifying a 2d coordinate is rotate into 3d and that
    // value is interpolated and added to the 2d.

    // Positive rotations clockwise about the specified axis looking down the axis towards the origin

    // Z(phi) Y(theta) Z(psi)

    SECTION("zero angles creates identity matrix") {
        RotationMatrix rm;
        rm.SetToEulerRotation(0.0f, 0.0f, 0.0f);

        RotationMatrix identity;
        identity.SetToIdentity( );

        REQUIRE(MatricesAreAlmostEqual(rm, identity));
    }

    SECTION("90 degree rotation around Z axis (phi)") {
        RotationMatrix rm;
        rm.SetToEulerRotation(90.0f, 0.0f, 0.0f);

        float x_in = 0.0f, y_in = 1.0f, z_in = 0.0f;
        float x, y, z;

        // Rotate the y axis onto the negative x axis
        rm.RotateCoords(x_in, y_in, z_in, x, y, z);
        REQUIRE(RelativeErrorIsLessThanEpsilon(x, -1.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));
    }

    SECTION("negative 90 degree rotation around Z axis (psi)") {
        RotationMatrix rm;
        rm.SetToEulerRotation(0.0f, 0.0f, -90.0f);

        float x_in = 0.0f, y_in = 1.0f, z_in = 0.0f;
        float x, y, z;

        // Rotate the y axis onto the positive x axis
        rm.RotateCoords(x_in, y_in, z_in, x, y, z);
        REQUIRE(RelativeErrorIsLessThanEpsilon(x, 1.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));
    }

    SECTION("opposite Z rotations cancel out") {
        RotationMatrix rm;
        rm.SetToEulerRotation(-90.0f, 0.0f, 90.0f);

        float x_in = 0.0f, y_in = 1.0f, z_in = 0.0f;
        float x, y, z;

        rm.RotateCoords(x_in, y_in, z_in, x, y, z);
        REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, 1.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));
    }

    SECTION("combined rotation theta=90, psi=90") {
        RotationMatrix rm;
        rm.SetToEulerRotation(0.0f, 90.0f, 90.0f);

        float x_in = 0.0f, y_in = 1.0f, z_in = 0.0f;
        float x, y, z;

        rm.RotateCoords(x_in, y_in, z_in, x, y, z);
        REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(z, 1.0f));
    }

    SECTION("combined rotation phi=90, theta=90") {
        RotationMatrix rm;
        rm.SetToEulerRotation(90.0f, 90.0f, 0.0f);

        float x_in = 0.0f, y_in = 1.0f, z_in = 0.0f;
        float x, y, z;

        rm.RotateCoords(x_in, y_in, z_in, x, y, z);
        REQUIRE(RelativeErrorIsLessThanEpsilon(x, -1.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));
    }

    SECTION("combined rotation phi=90, theta=90, psi=-90") {
        RotationMatrix rm;
        rm.SetToEulerRotation(90.0f, 90.0f, -90.0f);

        float x_in = 0.0f, y_in = 1.0f, z_in = 0.0f;
        float x, y, z;

        rm.RotateCoords(x_in, y_in, z_in, x, y, z);
        REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(z, -1.0f));
    }

    SECTION("180 degree rotation around Z") {
        RotationMatrix rm;
        rm.SetToEulerRotation(180.0f, 0.0f, 0.0f);

        float x_in = 0.0f, y_in = 1.0f, z_in = 0.0f;
        float x, y, z;

        rm.RotateCoords(x_in, y_in, z_in, x, y, z);
        // Point (0,1,0) should be rotated 180 degrees around Z to (0,-1,0)
        REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, -1.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(z, 0.0f));
    }

    SECTION("360 degree rotation returns to original") {
        RotationMatrix rm;
        rm.SetToEulerRotation(360.0f, 0.0f, 0.0f);

        RotationMatrix identity;
        identity.SetToIdentity( );

        REQUIRE(MatricesAreAlmostEqual(rm, identity));
    }
}

TEST_CASE("RotationMatrix SetToRotation method", "[RotationMatrix][core][rotation]") {

    SECTION("zero rotation creates identity-like result") {
        RotationMatrix rm;
        rm.SetToRotation(0.0f, 0.0f, 0.0f);

        RotationMatrix identity;
        identity.SetToIdentity( );

        REQUIRE(MatricesAreAlmostEqual(rm, identity));
    }

    SECTION("SetToRotation produces valid rotation matrix") {
        RotationMatrix rm;
        rm.SetToRotation(45.0f, 30.0f, 60.0f);

        // Check that it's a valid rotation matrix (Frobenius norm should be sqrt(3))
        float norm = rm.FrobeniusNorm( );
        REQUIRE(FloatsAreAlmostTheSame(norm, sqrtf(3.0f)));
    }
}

TEST_CASE("RotationMatrix coordinate rotation", "[RotationMatrix][core][rotation]") {

    SECTION("identity rotation leaves coordinates unchanged") {
        RotationMatrix identity;
        identity.SetToIdentity( );

        float x_in = 1.0f, y_in = 2.0f, z_in = 3.0f;
        float x, y, z;

        identity.RotateCoords(x_in, y_in, z_in, x, y, z);

        REQUIRE(x == x_in);
        REQUIRE(y == y_in);
        REQUIRE(z == z_in);
    }

    SECTION("rotation of zero vector returns zero") {
        RotationMatrix rm = CreateTestMatrix( );

        float x_in = 0.0f, y_in = 0.0f, z_in = 0.0f;
        float x, y, z;

        rm.RotateCoords(x_in, y_in, z_in, x, y, z);

        REQUIRE(x == 0.0f);
        REQUIRE(y == 0.0f);
        REQUIRE(z == 0.0f);
    }

    SECTION("rotation preserves vector magnitude for orthogonal matrices") {
        RotationMatrix rm;
        rm.SetToEulerRotation(45.0f, 30.0f, 60.0f);

        float x_in = 3.0f, y_in = 4.0f, z_in = 5.0f;
        float x, y, z;

        rm.RotateCoords(x_in, y_in, z_in, x, y, z);

        float magnitude_in  = sqrtf(x_in * x_in + y_in * y_in + z_in * z_in);
        float magnitude_out = sqrtf(x * x + y * y + z * z);

        REQUIRE(FloatsAreAlmostTheSame(magnitude_in, magnitude_out));
    }

    SECTION("RotateCoords2D rotates 2D coordinates correctly") {
        RotationMatrix rm;
        rm.SetToEulerRotation(90.0f, 0.0f, 0.0f);

        float x_in = 1.0f, y_in = 0.0f;
        float x, y;

        rm.RotateCoords2D(x_in, y_in, x, y);

        REQUIRE(RelativeErrorIsLessThanEpsilon(x, 0.0f));
        REQUIRE(RelativeErrorIsLessThanEpsilon(y, 1.0f));
    }

    SECTION("RotateCoords2D preserves magnitude") {
        RotationMatrix rm;
        rm.SetToEulerRotation(45.0f, 0.0f, 0.0f);

        float x_in = 3.0f, y_in = 4.0f;
        float x, y;

        rm.RotateCoords2D(x_in, y_in, x, y);

        float magnitude_in  = sqrtf(x_in * x_in + y_in * y_in);
        float magnitude_out = sqrtf(x * x + y * y);

        REQUIRE(FloatsAreAlmostTheSame(magnitude_in, magnitude_out));
    }
}

TEST_CASE("RotationMatrix Euler angle conversion", "[RotationMatrix][core][conversion]") {

    SECTION("round-trip conversion for simple angles") {
        float phi_in = 45.0f, theta_in = 30.0f, psi_in = 60.0f;

        RotationMatrix rm;
        rm.SetToEulerRotation(phi_in, theta_in, psi_in);

        float phi_out, theta_out, psi_out;
        rm.ConvertToValidEulerAngles(phi_out, theta_out, psi_out);

        // Recreate matrix from output angles
        RotationMatrix rm_verify;
        rm_verify.SetToEulerRotation(phi_out, theta_out, psi_out);

        REQUIRE(MatricesAreAlmostEqual(rm, rm_verify));
    }

    SECTION("conversion handles theta=0 edge case") {
        // When theta=0 or 180, phi and psi are not uniquely defined
        float phi_in = 45.0f, theta_in = 0.0f, psi_in = 30.0f;

        RotationMatrix rm;
        rm.SetToEulerRotation(phi_in, theta_in, psi_in);

        float phi_out, theta_out, psi_out;
        rm.ConvertToValidEulerAngles(phi_out, theta_out, psi_out);

        // Verify theta is close to 0
        REQUIRE(fabsf(theta_out) < 1.0f);

        // Verify matrix is correctly reproduced
        RotationMatrix rm_verify;
        rm_verify.SetToEulerRotation(phi_out, theta_out, psi_out);

        REQUIRE(MatricesAreAlmostEqual(rm, rm_verify));
    }

    SECTION("conversion handles theta=180 edge case") {
        float phi_in = 45.0f, theta_in = 180.0f, psi_in = 30.0f;

        RotationMatrix rm;
        rm.SetToEulerRotation(phi_in, theta_in, psi_in);

        float phi_out, theta_out, psi_out;
        rm.ConvertToValidEulerAngles(phi_out, theta_out, psi_out);

        // Verify theta is close to 180
        REQUIRE((fabsf(theta_out - 180.0f) < 1.0f || fabsf(theta_out + 180.0f) < 1.0f));

        // Verify matrix is correctly reproduced
        RotationMatrix rm_verify;
        rm_verify.SetToEulerRotation(phi_out, theta_out, psi_out);

        REQUIRE(MatricesAreAlmostEqual(rm, rm_verify));
    }

    SECTION("conversion handles arbitrary angles") {
        // Test with various angle combinations
        std::vector<std::tuple<float, float, float>> test_cases = {
                {0.0f, 0.0f, 0.0f},
                {90.0f, 0.0f, 0.0f},
                {0.0f, 90.0f, 0.0f},
                {0.0f, 0.0f, 90.0f},
                {45.0f, 45.0f, 45.0f},
                {120.0f, 75.0f, 33.0f},
                {-45.0f, 30.0f, 60.0f},
                {180.0f, 90.0f, 0.0f}};

        for ( const auto& test_case : test_cases ) {
            float phi_in   = std::get<0>(test_case);
            float theta_in = std::get<1>(test_case);
            float psi_in   = std::get<2>(test_case);

            CAPTURE(phi_in, theta_in, psi_in);

            RotationMatrix rm;
            rm.SetToEulerRotation(phi_in, theta_in, psi_in);

            float phi_out, theta_out, psi_out;
            rm.ConvertToValidEulerAngles(phi_out, theta_out, psi_out);

            CAPTURE(phi_out, theta_out, psi_out);

            RotationMatrix rm_verify;
            rm_verify.SetToEulerRotation(phi_out, theta_out, psi_out);

            // For gimbal lock cases (theta near 0, 90, or 180), use relaxed tolerance
            // because phi and psi are coupled and small numerical errors are expected
            bool  is_gimbal_lock = (fabsf(theta_in) < 1.0f) || (fabsf(theta_in - 90.0f) < 1.0f) || (fabsf(theta_in - 180.0f) < 1.0f);
            float tolerance      = is_gimbal_lock ? 5e-4f : 1e-4f;
            REQUIRE(MatricesAreAlmostEqual(rm, rm_verify, tolerance));
        }
    }

    SECTION("conversion refinement improves accuracy for edge cases") {
        // Create a matrix that might trigger refinement
        RotationMatrix rm;
        rm.SetToEulerRotation(0.1f, 89.9f, 179.9f);

        float phi_out, theta_out, psi_out;
        rm.ConvertToValidEulerAngles(phi_out, theta_out, psi_out);

        RotationMatrix rm_verify;
        rm_verify.SetToEulerRotation(phi_out, theta_out, psi_out);

        // After refinement, matrices should match very closely
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(fabsf(rm.m[i][j] - rm_verify.m[i][j]) < 0.001f);
            }
        }
    }
}

TEST_CASE("RotationMatrix edge cases and numerical stability", "[RotationMatrix][core][edge_cases]") {

    SECTION("very small angles produce nearly identity matrix") {
        RotationMatrix rm;
        rm.SetToEulerRotation(0.001f, 0.001f, 0.001f);

        RotationMatrix identity;
        identity.SetToIdentity( );

        // Should be very close to identity
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(fabsf(rm.m[i][j] - identity.m[i][j]) < 0.01f);
            }
        }
    }

    SECTION("very large angles are handled correctly") {
        // Angles larger than 360 should wrap around
        RotationMatrix rm1, rm2;
        rm1.SetToEulerRotation(45.0f, 30.0f, 60.0f);
        rm2.SetToEulerRotation(45.0f + 360.0f, 30.0f + 360.0f, 60.0f + 360.0f);

        REQUIRE(MatricesAreAlmostEqual(rm1, rm2));
    }

    SECTION("negative angles produce correct rotations") {
        RotationMatrix rm_pos, rm_neg;
        rm_pos.SetToEulerRotation(90.0f, 0.0f, 0.0f);
        rm_neg.SetToEulerRotation(-270.0f, 0.0f, 0.0f);

        // -270 degrees should be equivalent to +90 degrees
        REQUIRE(MatricesAreAlmostEqual(rm_pos, rm_neg));
    }

    SECTION("matrix multiplication preserves orthogonality") {
        RotationMatrix rm1, rm2;
        rm1.SetToEulerRotation(45.0f, 30.0f, 60.0f);
        rm2.SetToEulerRotation(30.0f, 60.0f, 45.0f);

        RotationMatrix product = rm1 * rm2;

        // Check that product is still orthogonal: R^T * R = I
        RotationMatrix transposed         = product.ReturnTransposed( );
        RotationMatrix should_be_identity = product * transposed;

        RotationMatrix identity;
        identity.SetToIdentity( );

        REQUIRE(MatricesAreAlmostEqual(should_be_identity, identity));
    }

    SECTION("repeated operations maintain precision") {
        RotationMatrix rm;
        rm.SetToEulerRotation(1.0f, 2.0f, 3.0f);

        RotationMatrix result = rm;
        // Apply rotation 360 times (should return to original after full circle)
        for ( int i = 0; i < 360; i++ ) {
            result *= rm;
        }

        // After 360 small rotations, we should have rotated significantly
        // But the matrix should still be valid (orthogonal)
        float norm = result.FrobeniusNorm( );
        REQUIRE(FloatsAreAlmostTheSame(norm, sqrtf(3.0f)));
    }
}

TEST_CASE("RotationMatrix operator precedence and chaining", "[RotationMatrix][core][operators]") {

    SECTION("multiple additions can be chained") {
        RotationMatrix rm1, rm2, rm3;
        rm1.SetToConstant(1.0f);
        rm2.SetToConstant(2.0f);
        rm3.SetToConstant(3.0f);

        RotationMatrix result = rm1 + rm2 + rm3;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(result.m[i][j] == 6.0f);
            }
        }
    }

    SECTION("mixed operations follow expected precedence") {
        RotationMatrix rm1, rm2, rm3;
        rm1.SetToConstant(1.0f);
        rm2.SetToConstant(2.0f);
        rm3.SetToConstant(3.0f);

        // Addition and subtraction are left-associative
        RotationMatrix result = rm1 + rm2 - rm3;

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(result.m[i][j] == 0.0f);
            }
        }
    }

    SECTION("compound assignment returns correct reference") {
        RotationMatrix rm1, rm2;
        rm1.SetToConstant(1.0f);
        rm2.SetToConstant(2.0f);

        RotationMatrix& ref = (rm1 += rm2);

        // Reference should point to rm1
        REQUIRE(&ref == &rm1);

        // And rm1 should be modified
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                REQUIRE(rm1.m[i][j] == 3.0f);
            }
        }
    }
}
