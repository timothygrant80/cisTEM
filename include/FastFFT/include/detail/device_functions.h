#ifndef __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__
#define __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__

#include "../cufftdx/include/cufftdx.hpp"

namespace FastFFT {

// Complex a * conj b multiplication
template <typename ComplexType, typename ScalarType>
static __device__ __host__ inline auto ComplexConjMulAndScale(const ComplexType a, const ComplexType b, ScalarType s) -> decltype(b) {
    ComplexType c;
    c.x = s * (a.x * b.x + a.y * b.y);
    c.y = s * (a.y * b.x - a.x * b.y);
    return c;
}

#define USEFASTSINCOS
// The __sincosf doesn't appear to be the problem with accuracy, likely just the extra additions, but it probably also is less flexible with other types. I don't see a half precision equivalent.
#ifdef USEFASTSINCOS
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c) {
    __sincosf(arg, s, c);
}
#else
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c) {
    sincos(arg, s, c);
}
#endif

} // namespace FastFFT

#endif // __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__