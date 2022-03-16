#ifndef _src_core_cistem_constants_h_
#define _src_core_cistem_constants_h_

// Place system wide constants and enums here. Gradually, we would like to replace the many defines.
namespace cistem {

// The default border to exclude when choosing peaks, e.g. in match_template, refine_template, prepare_stack_matchtemplate, make_template_result.
constexpr const int fraction_of_box_size_to_exclude_for_border = 4;
constexpr const int maximum_number_of_detections               = 1000;

} // namespace cistem

#endif
