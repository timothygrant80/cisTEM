#ifndef _SRC_CORE_ZERNIKE_H_
#define _SRC_CORE_ZERNIKE_H_

/*
 * Zernike polynomials for the parametrisation of higher-order optical aberrations, in the ordering and
 * normalisation of RELION 3.1 (Zivanov, Nakane & Scheres, IUCrJ 7:253-267, 2020; RELION's
 * src/jaz/math/Zernike.cpp), so that a coefficient list taken from a RELION STAR file (rlnOddZernike /
 * rlnEvenZernike) means the same thing here. As in RELION the polynomials are unnormalised (R_n^m carries its
 * factorial weights and no sqrt(2(n+1)) factor) and are evaluated at the spatial frequency in 1/A, without
 * any scaling to a unit disc.
 *
 * Odd polynomials (n odd) describe antisymmetric aberrations, which are pure phase shifts of the image:
 * index 0 and 1 are the order-1 terms (image shift along y and x), 2..5 the order-3 terms (trefoil Z3^-3,
 * beam tilt Z3^-1 and Z3^1, trefoil Z3^3) and 6..11 the order-5 terms.
 * Even polynomials (n even) describe symmetric aberrations, which add to the argument of the CTF: index 0 is
 * Z0^0 (a constant phase), 1..3 the order-2 terms (astigmatism, defocus, astigmatism) and 4..8 the order-4
 * terms (four-fold astigmatism Z4^-4, Z4^-2, spherical aberration Z4^0, Z4^2, Z4^4).
 */
namespace zernike {

// Number of coefficients in RELION's odd (n = 1, 3, ..., highest_order) and even (n = 0, 2, ..., highest_order) sets
int NumberOfOddCoefficients(int highest_order);
int NumberOfEvenCoefficients(int highest_order);

// Azimuthal order m and radial order n of the index'th polynomial of each set
void OddIndexToMN(int index, int& m, int& n);
void EvenIndexToMN(int index, int& m, int& n);

// The radial polynomial R_n^m(rho), m >= 0; zero when n - m is odd
double RadialPolynomial(int m, int n, double rho);

// Z_n^m(rho, phi) = R_n^|m|(rho) cos(m phi) for m >= 0 and R_n^|m|(rho) sin(|m| phi) for m < 0
double Evaluate(int m, int n, double rho, double phi);
double EvaluateCartesian(int m, int n, double x, double y);

} // namespace zernike

#endif
