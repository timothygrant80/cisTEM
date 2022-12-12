#ifndef _src_constants_constants_h_
#define _src_constants_constants_h_

#include <array>
#include <string_view>

// Numerical constants for frequently used values like pi, sqrt(2), etc.
#include "cistem_numbers.h"

// Constants for the electron scattering potential
#include "electron_scattering.h"

// Place system wide constants and enums here. Gradually, we would like to replace the many defines.
namespace cistem {

// The default border to exclude when choosing peaks, e.g. in match_template, refine_template, prepare_stack_matchtemplate, make_template_result.
constexpr const int fraction_of_box_size_to_exclude_for_border = 4;
constexpr const int maximum_number_of_detections               = 1000;

namespace physical_constants {

// From Shang and Sigworth, average of polar and non-polar from table 1 (with correction to polar radius 3, 1.7-->3.0);
// terms 3-5 have the average atomic radius of C,N,O,H added to shift the curve to be relative to atomic center.
constexpr float                hydration_radius_xtra_shift = -0.5f;
constexpr std::array<float, 8> hydration_radius_vals       = {0.1750f, -0.1350f, 2.23f, 3.43f, 4.78f, 1.0000f, 1.7700f, 0.9550f};

} // namespace physical_constants

/*
    SCOPED ENUMS:
        Rather than specifying a scoped enum as enum class, we use the following technique to define scoped enums while
        avoiding the need for static_cast<type>(enum) anywhere we want to do an assignment or comparison.      
*/

namespace electron_dose {
constexpr float critical_dose_a         = 0.24499f;
constexpr float critical_dose_b         = -1.6649f;
constexpr float critical_dose_c         = 2.8141f;
constexpr float reduced_critical_dose_b = critical_dose_b / 2.f;
} // namespace electron_dose

// To ensure data base type parity, force int type (even though this should be the default).
namespace job_type {
enum Enum : int {
    // TODO: extend this to remove other existing job_type defines.
    template_match_full_search,
    template_match_refinement
};

} // namespace job_type

namespace workflow {
enum Enum : int { single_particle,
                  template_matching,
                  pharma };
} // namespace workflow

namespace PCOS_image_type {
enum Enum : int { reference_volume_t,
                  particle_image_t,
                  ctf_image_t,
                  beamtilt_image_t };
}

namespace fft_type {
// inplace/outofplace
// input type
// compute_type
// output_type

enum Enum : int { unset,
                  inplace_32f_32f_32f,
                  outofplace_32f_32f_32f,
                  inplace_32f_32f_32f_batched,
                  outofplace_32f_32f_32f_batched };

constexpr std::array<std::string_view, 5> names = {"unset",
                                                   "inplace_32f_32f_32f",
                                                   "outofplace_32f_32f_32f",
                                                   "inplace_32f_32f_32f_batched",
                                                   "outofplace_32f_32f_32f_batched"};
} // namespace fft_type

namespace gpu {

constexpr int warp_size             = 32;
constexpr int min_threads_per_block = warp_size;
constexpr int max_threads_per_block = 1024;

// Currently we just support up to 3d tensors to match the Image class
constexpr int max_tensor_manager_dimensions = 3;
constexpr int max_tensor_manager_tensors    = 4;

namespace tensor_op {
enum Enum : int {
    reduction,
    contraction,
    binary,
    ternary,
};
} // namespace tensor_op

namespace tensor_id {
enum Enum : int {
    A,
    B,
    C,
    D,
};

// must match the above enum tensor_id
constexpr std::array<char, 4> tensor_names = {'A', 'B', 'C', 'D'};
} // namespace tensor_id

// valid as of cuda 11.7 from cufft.h definition of cufftResult_t
constexpr std::array<std::string_view, 17> cufft_error_types = {
        "CUFFT_SUCCESS",
        "CUFFT_INVALID_PLAN",
        "CUFFT_ALLOC_FAILED",
        "CUFFT_INVALID_TYPE",
        "CUFFT_INVALID_VALUE",
        "CUFFT_INTERNAL_ERROR",
        "CUFFT_EXEC_FAILED",
        "CUFFT_SETUP_FAILED",
        "CUFFT_INVALID_SIZE ",
        "CUFFT_UNALIGNED_DATA",
        "CUFFT_INCOMPLETE_PARAMETER_LIST",
        "CUFFT_INVALID_DEVICE",
        "CUFFT_PARSE_ERROR",
        "CUFFT_NO_WORKSPACE",
        "CUFFT_NOT_IMPLEMENTED",
        "CUFFT_LICENSE_ERROR",
        "CUFFT_NOT_SUPPORTED"};
} // namespace gpu

} // namespace cistem

#endif
