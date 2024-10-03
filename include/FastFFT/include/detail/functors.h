#ifndef __INCLUDE_DETAILS_FUNCTORS_H__
#define __INCLUDE_DETAILS_FUNCTORS_H__

#include "concepts.h"

// TODO: doc and namespace
// FIXME: the is_final is a bit of a hack to make sure we can tell if the functor is a NOOP or not

namespace FastFFT {

namespace KernelFunction {

// Maybe a better way to check , but using keyword final to statically check for non NONE types
// TODO: is marking the operatior()() inline sufficient or does the struct need to be marked inline as well?
template <typename T, int N_ARGS, IKF_t U, typename = void>
struct my_functor {};

template <typename T>
struct my_functor<T, 0, IKF_t::NOOP> {
    __device__ __forceinline__
            T
            operator( )( ) {
        printf("really specific NOOP\n");
        return 0;
    }
};

// EnableIf<IsAllowedComplexBaseType<T>>
//struct my_functor<T, 4, IKF_t::CONJ_MUL, EnableIf<IsAllowedComplexBaseType<T>>> final {
template <typename T>
struct my_functor<T, 4, IKF_t::CONJ_MUL, EnableIf<IsAllowedRealType<T>>> final {
    __device__ __forceinline__ void
    operator( )(T& template_fft_x, T& template_fft_y, const T& target_fft_x, const T& target_fft_y) {
        // Is there a better way than declaring this variable each time?
        // This is target * conj(template)
        T tmp          = (template_fft_x * target_fft_x + template_fft_y * target_fft_y);
        template_fft_y = (template_fft_x * target_fft_y - template_fft_y * target_fft_x);
        template_fft_x = tmp;
    }
};

template <typename T>
struct my_functor<T, 4, IKF_t::CONJ_MUL_THEN_SCALE, EnableIf<IsAllowedRealType<T>>> final {

    // Pass in the scale factor on construction
    my_functor(const T& scale_factor) : scale_factor(scale_factor) {}

    __device__ __forceinline__ void
    operator( )(T& template_fft_x, T& template_fft_y, const T& target_fft_x, const T& target_fft_y) {
        // Is there a better way than declaring this variable each time?
        // This is target * conj(template)
        T tmp          = (template_fft_x * target_fft_x + template_fft_y * target_fft_y) * scale_factor;
        template_fft_y = (template_fft_x * target_fft_y - template_fft_y * target_fft_x) * scale_factor;
        template_fft_x = tmp;
    }

  private:
    const T scale_factor;
};

template <typename T>
struct my_functor<T, 2, IKF_t::SCALE, EnableIf<IsAllowedInputType<T>>> final {

    const T scale_factor;

    // Pass in the scale factor on construction
    __device__ __forceinline__ my_functor(const T& scale_factor) : scale_factor(scale_factor) {}

    __device__ __forceinline__ float
    operator( )(float input_value) {
        return input_value *= scale_factor;
    }

    __device__ __forceinline__ float2
    operator( )(float2 input_value) {
        input_value.x *= scale_factor;
        input_value.y *= scale_factor;
        return input_value;
    }
};

} // namespace KernelFunction
} // namespace FastFFT

#endif