#ifndef _SRC_CONSTANTS_CISTEM_NUMBERS_H_
#define _SRC_CONSTANTS_CISTEM_NUMBERS_H_

#include <complex>

#if __cplusplus > 201703L
#include <numbers>
using namespace std::numbers;
#else
// For now we do not have c++20 in compiling gpu code so we need to define this for constants. Modified from /usr/include/c++/11/numbers
/// pi

template <typename _Tp>
inline constexpr _Tp e_v = static_cast<_Tp>(2.7182818284590452353602874713526624977572470936999595749669676277240766303535L);

template <typename _Tp>
inline constexpr _Tp log2e_v = static_cast<_Tp>(1.4426950408889634073599246810018921374266459541529859341354494069314718055994L);

template <typename _Tp>
inline constexpr _Tp log10e_v = static_cast<_Tp>(0.43429448190325182765112891891660508229439700580366656611445378316527120190914L);

template <typename _Tp>
// inline constexpr _Tp pi_v = _Enable_if_floating<_Tp>(3.141592653589793238462643383279502884L);
inline constexpr _Tp pi_v = static_cast<_Tp>(3.141592653589793238462643383279502884L);

template <typename _Tp>
inline constexpr _Tp two_pi_v = _Tp(2) * pi_v<_Tp>;

template <typename _Tp>
inline constexpr _Tp half_pi_v = _Tp(0.5) * pi_v<_Tp>;

template <typename _Tp>
inline constexpr _Tp inv_pi_v = _Tp(1) / pi_v<_Tp>;

template <typename _Tp>
inline constexpr _Tp inv_sqrtpi_v = _Tp(1) / sqrt(pi_v<_Tp>);

template <typename _Tp>
inline constexpr _Tp ln2_v = static_cast<_Tp>(0.693147180559945309417232121458176568L);

template <typename _Tp>
inline constexpr _Tp ln10_v = static_cast<_Tp>(2.302585092994045684017991454684364208L);

template <typename _Tp>
inline constexpr _Tp sqrt2_v = static_cast<_Tp>(1.414213562373095048801688724209698079L);

template <typename _Tp>
inline constexpr _Tp sqrt3_v = static_cast<_Tp>(1.732050807568877293527446341505872366L);

template <typename _Tp>
inline constexpr _Tp inv_sqrt3_v = _Tp(1) / sqrt3_v<_Tp>;

constexpr std::complex<float> I(0.0, 1.0);

#endif

#endif