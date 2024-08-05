#ifndef __INCLUDE_DETAIL_FASTFFT_TYPES_H__
#define __INCLUDE_DETAIL_FASTFFT_TYPES_H__

#include <array>
#include <string_view>

namespace FastFFT {

namespace DataType {
// Used to specify input/calc/output data types
enum Enum { int4_2,
            uint8,
            int8,
            uint16,
            int16,
            fp16,
            bf16,
            tf32,
            uint32,
            int32,
            fp32 };

constexpr std::array<std::string_view, 11> name = {"int4_2", "uint8", "int8", "uint16", "int16", "fp16", "bf16", "tf32", "uint32", "int32", "fp32"};

} // namespace DataType

namespace SizeChangeType {
// FIXME this seems like a bad idea. Added due to conflicing labels in switch statements, even with explicitly scope.
enum Enum : uint8_t { increase,
                      decrease,
                      no_change };
} // namespace SizeChangeType

namespace OriginType {
// Used to specify the origin of the data
enum Enum : int { natural,
                  centered,
                  quadrant_swapped };

constexpr std::array<std::string_view, 3> name = {"natural", "centered", "quadrant_swapped"};

} // namespace OriginType

namespace DimensionCheckType {
enum Enum : uint8_t { CopyFromHost,
                      CopyToHost,
                      FwdTransform,
                      InvTransform };

} // namespace DimensionCheckType

} // namespace FastFFT

#endif // __INCLUDE_DETAIL_FASTFFT_TYPES_H__