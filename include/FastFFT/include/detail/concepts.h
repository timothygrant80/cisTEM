#ifndef __INCLUDE_DETAIL_CONCEPTS_H__
#define __INCLUDE_DETAIL_CONCEPTS_H__

#include <type_traits>

template <typename K>
constexpr inline bool IS_IKF_t( ) {
    if constexpr ( std::is_final_v<K> ) {
        return true;
    }
    else {
        return false;
    }
};

namespace FastFFT {

namespace KernelFunction {

// Define an enum for different functors
// Intra Kernel Function Type
enum IKF_t { NOOP,
             SCALE,
             CONJ_MUL,
             CONJ_MUL_THEN_SCALE };
} // namespace KernelFunction

// To limit which kernels are instantiated, define a set of constants for the FFT method to be used at compile time.
constexpr int Generic_Fwd_FFT           = 1;
constexpr int Generic_Inv_FFT           = 2;
constexpr int Generic_Fwd_Image_Inv_FFT = 3;

template <bool, typename T = void>
struct EnableIfT {};

template <typename T>
struct EnableIfT<true, T> { using Type = T; };

template <bool cond, typename T = void>
using EnableIf = typename EnableIfT<cond, T>::Type;

template <typename IntraOpType>
constexpr bool HasIntraOpFunctor = IS_IKF_t<IntraOpType>( );

// 3d is always odd (3 for fwd/inv or 5 for round trip)
// 2d is odd if it is round trip (3) and even if fwd/inv (2 )
template <int FFT_ALGO_t, int Rank>
constexpr bool IsAlgoRoundTrip = (FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT);

template <typename IntraOpType, int FFT_ALGO_t>
constexpr bool IfAppliesIntraOpFunctor_HasIntraOpFunctor = (FFT_ALGO_t != Generic_Fwd_Image_Inv_FFT || (FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT && HasIntraOpFunctor<IntraOpType>));

template <typename T>
constexpr bool IsComplexType = (std::is_same_v<T, float2> || std::is_same_v<T, __half2>);

template <typename... Args>
constexpr bool IsPointerOrNullPtrType = (... && (std::is_same<Args, std::nullptr_t>::value || std::is_pointer_v<std::decay_t<Args>>));

template <typename... Args>
constexpr bool IsAllowedRealType = (... && (std::is_same_v<Args, __half> || std::is_same_v<Args, float>));

template <typename... Args>
constexpr bool IsAllowedComplexBaseType = IsAllowedRealType<Args...>;

template <typename... Args>
constexpr bool IsAllowedComplexType = (... && (std::is_same_v<Args, __half2> || std::is_same_v<Args, float2>));

template <typename... Args>
constexpr bool IsAllowedInputType = (... && (std::is_same_v<Args, __half> || std::is_same_v<Args, float> || std::is_same_v<Args, __half2> || std::is_same_v<Args, float2>));

template <typename T1_wanted, typename T2_wanted, typename T1, typename T2>
constexpr bool CheckPointerTypesForMatch = (std::is_same_v<T1_wanted, T1> && std::is_same_v<T2_wanted, T2>);

} // namespace FastFFT
#endif // __INCLUDE_DETAIL_CONCEPTS_H__