#ifndef _src_core_cistem_constants_h_
#define _src_core_cistem_constants_h_

// Place system wide constants and enums here. Gradually, we would like to replace the many defines.
namespace cistem {

// The default border to exclude when choosing peaks, e.g. in match_template, refine_template, prepare_stack_matchtemplate, make_template_result.
constexpr const int fraction_of_box_size_to_exclude_for_border = 4;
constexpr const int maximum_number_of_detections               = 1000;

/*
    SCOPED ENUMS:
        Rather than specifying a scoped enum as enum class, we use the following technique to define scoped enums while
        avoiding the need for static_cast<type>(enum) anywhere we want to do an assignment or comparison.      
*/

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
                  template_matching };
} // namespace workflow

} // namespace cistem

#endif
