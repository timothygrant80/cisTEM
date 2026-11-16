#include "core_headers.h"

namespace zernike {

int NumberOfOddCoefficients(int highest_order) {
    const int l = (highest_order - 1) / 2 + 1;
    return l * l + l;
}

int NumberOfEvenCoefficients(int highest_order) {
    const int l = highest_order / 2;
    return (l + 1) * (l + 1);
}

void OddIndexToMN(int index, int& m, int& n) {
    const int k  = int((sqrt(1.0 + 4.0 * index) - 1.0) / 2.0);
    const int i0 = k * k + k;
    n            = 2 * k + 1;
    m            = 2 * (index - i0) - n;
}

void EvenIndexToMN(int index, int& m, int& n) {
    const int k = int(sqrt(double(index)));
    n           = 2 * k;
    m           = 2 * (index - k * k - k);
}

static double Factorial(int k) {
    double out = 1.0;
    for ( int i = 2; i <= k; i++ )
        out *= double(i);
    return out;
}

double RadialPolynomial(int m, int n, double rho) {
    MyDebugAssertTrue(m >= 0 && m <= n, "Bad Zernike orders m = %i, n = %i", m, n);
    if ( (n - m) % 2 == 1 )
        return 0.0;

    double out = 0.0;
    for ( int k = 0; k <= (n - m) / 2; k++ ) {
        const double sign = (k % 2 == 0) ? 1.0 : -1.0;
        out += sign * Factorial(n - k) / (Factorial(k) * Factorial((n + m) / 2 - k) * Factorial((n - m) / 2 - k)) * pow(rho, n - 2 * k);
    }
    return out;
}

double Evaluate(int m, int n, double rho, double phi) {
    if ( m >= 0 )
        return RadialPolynomial(m, n, rho) * cos(m * phi);
    else
        return RadialPolynomial(-m, n, rho) * sin(-m * phi);
}

double EvaluateCartesian(int m, int n, double x, double y) {
    if ( x == 0.0 && y == 0.0 )
        return Evaluate(m, n, 0.0, 0.0);
    else
        return Evaluate(m, n, sqrt(x * x + y * y), atan2(y, x));
}

} // namespace zernike
