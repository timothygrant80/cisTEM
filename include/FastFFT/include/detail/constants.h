#ifndef __INCLUDE_DETAIL_CONSTANTS_H__
#define __INCLUDE_DETAIL_CONSTANTS_H__

namespace FastFFT {
// TODO this probably needs to depend on the size of the xform, at least small vs large.
constexpr const int elements_per_thread_16   = 4;
constexpr const int elements_per_thread_32   = 8;
constexpr const int elements_per_thread_64   = 8;
constexpr const int elements_per_thread_128  = 8;
constexpr const int elements_per_thread_256  = 8;
constexpr const int elements_per_thread_512  = 8;
constexpr const int elements_per_thread_1024 = 8;
constexpr const int elements_per_thread_2048 = 8;
constexpr const int elements_per_thread_4096 = 8;
constexpr const int elements_per_thread_8192 = 16;

} // namespace FastFFT

#endif