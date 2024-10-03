#ifndef __INCLUDE_DETAILS_CONFIG_H__
#define __INCLUDE_DETAILS_CONFIG_H__

#include <chrono>
#include <random>
#include <iostream>

// Forward declaration so we can leave the inclusion of cuda_fp16.h to FastFFT.cu
struct __half;
struct __half2;

#ifndef cisTEM_USING_FastFFT // ifdef being used in cisTEM that defines these
#if __cplusplus >= 202002L
#include <numbers>
using namespace std::numbers;
#else
#if __cplusplus < 201703L
#message "C++ is " __cplusplus
#error "C++17 or later required"
#else
template <typename _Tp>
// inline constexpr _Tp pi_v = _Enable_if_floating<_Tp>(3.141592653589793238462643383279502884L);
inline constexpr _Tp pi_v = 3.141592653589793238462643383279502884L;
#endif // __cplusplus require > 17
#endif // __cplusplus 20 support
#endif // enable FastFFT

#endif