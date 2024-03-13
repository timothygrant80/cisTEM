#include "../../constants/constants.h"

// Shift the curves to the right as the values from Shang/Sigworth are distance to VDW radius (avg C/O/N/H = 1.48 A)
// FIXME now that you are saving distances, you can also consider polar/non-polar residues separately for an "effective" distance since the curves have the same shape with a linear offset.
constexpr float PUSH_BACK_BY = -1.48;

inline float return_hydration_weight(float& radius, const float pixel_size) {
    using namespace cistem::physical_constants;

    return 0.5f + 0.5f * std::erff((radius + PUSH_BACK_BY) - (hydration_radius_vals[2] + hydration_radius_xtra_shift * pixel_size) / (sqrt2_v<float> * hydration_radius_vals[5])) +
           hydration_radius_vals[0] * expf(-powf((radius + PUSH_BACK_BY) - (hydration_radius_vals[3] + hydration_radius_xtra_shift * pixel_size), 2) / (2 * powf(hydration_radius_vals[6], 2))) +
           hydration_radius_vals[1] * expf(-powf((radius + PUSH_BACK_BY) - (hydration_radius_vals[4] + hydration_radius_xtra_shift * pixel_size), 2) / (2 * powf(hydration_radius_vals[7], 2)));
}

// Same as above but taper to zero from 3 - 7 Ang
inline float return_hydration_weight_tapered(float taper_from, float& radius, const float pixel_size) {
    using namespace cistem::physical_constants;

    return (0.5f + 0.5f * std::erff((radius + PUSH_BACK_BY) - (hydration_radius_vals[2] + hydration_radius_xtra_shift * pixel_size) / (sqrt2_v<float> * hydration_radius_vals[5])) +
            hydration_radius_vals[0] * expf(-powf((radius + PUSH_BACK_BY) - (hydration_radius_vals[3] + hydration_radius_xtra_shift * pixel_size), 2) / (2 * powf(hydration_radius_vals[6], 2))) +
            hydration_radius_vals[1] * expf(-powf((radius + PUSH_BACK_BY) - (hydration_radius_vals[4] + hydration_radius_xtra_shift * pixel_size), 2) / (2 * powf(hydration_radius_vals[7], 2)))) *
           (0.5f + 0.5f * cosf(radius - taper_from));
}
