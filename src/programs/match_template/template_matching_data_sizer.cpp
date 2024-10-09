#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#if defined(cisTEM_USING_FastFFT) && defined(ENABLEGPU)
#include "../../../include/FastFFT/include/FastFFT.h"
#endif

#include "template_matching_data_sizer.h"

// #define DEBUG_IMG_OUTPUT "/tmp"
// #define DEBUG_TM_SIZER_PRINT

TemplateMatchingDataSizer::TemplateMatchingDataSizer(MyApp* wanted_parent_ptr,
                                                     Image& input_image,
                                                     Image& wanted_template,
                                                     float  wanted_pixel_size,
                                                     float  wanted_template_padding)
    : parent_match_template_app_ptr{wanted_parent_ptr},
      pixel_size{wanted_pixel_size},
      template_padding{wanted_template_padding} {

    MyDebugAssertTrue(pixel_size > 0.0f, "Pixel size must be greater than zero");
    // TODO: remove this constraint
    MyAssertTrue(template_padding == 1.0f, "Padding must be  equal to 1.0");
    image_size.x = input_image.logical_x_dimension;
    image_size.y = input_image.logical_y_dimension;
    image_size.z = input_image.logical_z_dimension;
    image_size.w = (input_image.logical_x_dimension + input_image.padding_jump_value) / 2;

    template_size.x = wanted_template.logical_x_dimension;
    template_size.y = wanted_template.logical_y_dimension;
    template_size.z = wanted_template.logical_z_dimension;
    template_size.w = (wanted_template.logical_x_dimension + wanted_template.padding_jump_value) / 2;
};

TemplateMatchingDataSizer::~TemplateMatchingDataSizer( ){
        // Nothing to do here
};

/**
 * @brief Peform checks on the wanted high resolution limit, set range of prime factors that are acceptable based on whether we are using FastFFT or not.
 * 
 * @param wanted_high_resolution_limit 
 * @param use_fast_fft 
 */
void TemplateMatchingDataSizer::SetImageAndTemplateSizing(const float wanted_high_resolution_limit, const bool use_fast_fft) {
    MyDebugAssertFalse(sizing_is_set, "Sizing has already been set");
    // Make sure we aren't trying to limit beyond Nyquist, and if < Nyquist set resampling needed to true.
    SetHighResolutionLimit(wanted_high_resolution_limit);
    this->use_fast_fft = use_fast_fft;

    // Setup some limits. These could probably just go directly into their specific methods in this class
    if ( use_fast_fft ) {
        primes.assign({2});
        max_increase_by_fraction_of_image = 2.f;
    }
    else {
        primes.assign({2, 3, 5, 7, 9, 13});
        max_increase_by_fraction_of_image = 0.1f;
    }

    GetFFTSize( );
};

/**
 * @brief Always remove outliers, center and whiten prior to any transormations, resampling or chunking of the input image.
 * 
 * We ALWAYS want the starting image statistics to be the same, regardless of the final size.
 * 
 * @param input_image 
 */
void TemplateMatchingDataSizer::PreProcessInputImage(Image& input_image, int mask_central_cross_width, bool swap_real_space_quadrants, bool normalize_to_variance_one) {

    // We whiten the image prior to any padding etc in particular to remove any low-frequency gradients that would add to boundary dislocations.
    // We may also whiten following any further resampling and resizing or other ops that are done to the image. We need to keep track of the total filtering applied.
    Curve number_of_terms;
    Curve local_whitening_filter;
    local_whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    if ( ! whitening_filter_ptr ) {
        // We'll accumulate the local whitening filter at the end of the method
        whitening_filter_ptr = std::make_unique<Curve>(local_whitening_filter);
        whitening_filter_ptr->SetYToConstant(1.0f);
    }
    // We could also check and FFT if necessary similar to Resize() but we are assuming the input image is in real space.
    MyDebugAssertTrue(input_image.is_in_real_space, "Input image must be in real space");
    // This won't work for movie frames (13.0 is used in unblur) TODO use poisson stats
    input_image.ReplaceOutliersWithMean(5.0f);

#ifdef DEBUG_IMG_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
        input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image.mrc", 1);
#endif
    input_image.ForwardFFT( );

    if ( swap_real_space_quadrants )
        input_image.SwapRealSpaceQuadrants( );

    input_image.ZeroCentralPixel( );

    input_image.Compute1DPowerSpectrumCurve(&local_whitening_filter, &number_of_terms);
    local_whitening_filter.SquareRoot( );
    local_whitening_filter.Reciprocal( );
    local_whitening_filter.MultiplyByConstant(1.0f / local_whitening_filter.ReturnMaximumValue( ));

    input_image.ApplyCurveFilter(&local_whitening_filter);

    if ( mask_central_cross_width > 0 )
        input_image.MaskCentralCross(mask_central_cross_width, mask_central_cross_width);

    if ( whitening_filter_ptr ) {
        whitening_filter_ptr->ResampleCurve(whitening_filter_ptr.get( ), local_whitening_filter.NumberOfPoints( ));
        local_whitening_filter.ResampleCurve(&local_whitening_filter, whitening_filter_ptr->NumberOfPoints( ));
    }
    // Record this filtering for later use
    whitening_filter_ptr->MultiplyBy(local_whitening_filter);

    input_image.ZeroCentralPixel( );
    if ( normalize_to_variance_one ) {
        // Presumably for Pre-processing (where we need the realspace variance = 1 so noise in padding reginos matchs)
        // TODO: rename so the real space vs FFT is clear
        input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));
        input_image.BackwardFFT( );
    }
    else {
        // When used in cross-correlation, we need the extra division.
        input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( ) / float(input_image.number_of_real_space_pixels)));
    }

#ifdef DEBUG_IMG_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
        input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image_whitened.mrc", 1);
#endif
    // Saving a copy of the pre-processed image for later use to determine peak heights not damped by resampling.
    // TODO: we can also use this (with z > )
    pre_processed_image.at(0) = input_image;
};

void TemplateMatchingDataSizer::CheckSizing( ) {
    // TODO: remove thiss
    if ( use_fast_fft ) {
        // We currently can only use FastFFT for power of 2 size images and template
        // TODO: the limit on templates should be trivial to remove since they are padded.
        // TOOD: Should probably consider this in the code above working out next good size, rather than only allowing power of 2,
        //       which will take a k3 to 8k x 8k it would be better to pad either
        //       a) crop to 4k x 4k (lossy but immediately supported)
        //       b) split into two images and pad each to either 4k x 4k (possibly a bit slower and not yet supported)
        // FIXME:
        if ( image_search_size.x != image_search_size.y ) {
            parent_match_template_app_ptr->SendInfo("FastFFT currently only supports square images, padding smaller up\n");
        }
        if ( image_search_size.x > image_search_size.y ) {
            image_search_size.y = image_search_size.x;
        }
        else {
            image_search_size.x = image_search_size.y;
        }
        if ( image_search_size.x > 4096 || image_search_size.y > 4096 ) {
            parent_match_template_app_ptr->SendError("FastFFT only supports images up to 4096x4096\n");
        }
    }
    else {
        // TODO: what else should be be verifying - rotation if warranted and power of two / nice siziing?
    }
};

void TemplateMatchingDataSizer::GetFFTSize( ) {

    // TODO: this should consider how close we are to the next power of two, which for the time being,
    // we are explicitly padding to.
    int bin_offset_3d          = 0;
    int bin_offset_2d          = 0;
    int closest_3d_binned_size = template_size.x;
    int best_3d_bin_offset     = 0;

    // We'll record the average of these two
    float closest_3d_binning_factor = 1.f;
    float closest_2d_binning_factor = 1.f;

    // We want the binning to be isotropic, and the easiest way to ensure that is to first pad any non-square input_image to a square size in realspace.
    const int   max_square_size       = std::max(image_size.x, image_size.y);
    const float target_binning_factor = high_resolution_limit / pixel_size / 2.0f;

    float current_binning_factor;
    int   current_binned_size;

    if ( resampling_is_needed ) {

        // Presumably we'll be using a power of 2 square size anyway for FastFFT (though rectangular images should be supported at some point.)
        // The other requirement is to ensure the resulting pixel size is the same for the reference and the search images.
        // Ideally, we would just calculate a scattering potential at the correct size, but even when that capability is added, we still should
        // allow the user to supply a template that is generated from a map that may not have a good model (e.g. something at 5-6 A resolution may still be usefule for TM in situ,
        // but may be to low res to build a decent atomic model into.)

        // The most challenging part is matching the pixel size of the input 3d and the input images. Presumably, the smaller 3d will be the limiting factor b/c of the larger
        // Fourier voxel step.

        // Start by taking the (possibly) largest deviation from the wanted pixel size, but the (possibly) smallest final image size to determine subsequent penalties for padding.
        current_binning_factor = GetRealizedBinningFactor(target_binning_factor, template_size.x);
        current_binned_size    = GetBinnedSize(template_size.x, current_binning_factor);

        closest_3d_binned_size         = current_binned_size;
        closest_3d_binning_factor      = current_binning_factor;
        const int best_power_of_2_size = get_next_power_of_two(closest_3d_binned_size);

        // Now see how close we can get to the wanted pixel size without going over the best power of 2 size
        // float smallest_error = fabsf(pixel_size * target_binning_factor - pixel_size * current_binning_factor) / pixel_size * 100.f;
        float smallest_error = fabsf(target_binning_factor - current_binning_factor) * 100.f;

        // TODO: it may make sense to save a few of the best values as they could produce a smaller total error in mismatch between the
        // Image and template, which is likely more important than any small error in wanted vs realized pixel size.

        while ( current_binned_size <= best_power_of_2_size ) {
            current_binning_factor = GetRealizedBinningFactor(target_binning_factor, template_size.x + bin_offset_3d);
            current_binned_size    = GetBinnedSize(template_size.x + bin_offset_3d, current_binning_factor);
            float current_error    = fabsf(target_binning_factor - current_binning_factor) * 100.f;
            if ( current_error < smallest_error ) {
                closest_3d_binned_size    = current_binned_size;
                closest_3d_binning_factor = current_binning_factor;
                smallest_error            = current_error;
                best_3d_bin_offset        = bin_offset_3d;
            }
            bin_offset_3d++;
        }

        // FIXME: We should enforce the 4k restriction somwhere else.
        constexpr int   max_2d_power_of_2_size      = 4096;
        constexpr float acceptable_pixel_size_error = 0.00005f;

        // How close can we get the image pixel size to match that of the resampled 3d? The target is now shifted from the users input, to the best 3d binning we found
        current_binning_factor     = GetRealizedBinningFactor(closest_3d_binning_factor, max_square_size);
        current_binned_size        = GetBinnedSize(max_square_size, current_binning_factor);
        int closest_2d_binned_size = current_binned_size;
        closest_2d_binning_factor  = current_binning_factor;

        smallest_error = fabsf(closest_2d_binning_factor - closest_3d_binning_factor);

        int bin_offset_2d = 1;
        while ( current_binned_size <= max_2d_power_of_2_size ) {
            if ( smallest_error * pixel_size < acceptable_pixel_size_error ) {
                break;
            }
            current_binning_factor = GetRealizedBinningFactor(closest_3d_binning_factor, max_square_size + bin_offset_2d);
            current_binned_size    = GetBinnedSize(max_square_size + bin_offset_2d, current_binning_factor);
            float current_error    = fabsf(current_binning_factor - closest_3d_binning_factor);
            if ( current_error < smallest_error ) {
                closest_2d_binned_size    = current_binned_size;
                closest_2d_binning_factor = current_binning_factor;
                smallest_error            = current_error;
            }

            bin_offset_2d++;
        }

        image_pre_scaling_size.x = max_square_size + bin_offset_2d;
        image_pre_scaling_size.y = max_square_size + bin_offset_2d;
        image_pre_scaling_size.z = 1; // FIXME: once we add chunking ...

        image_cropped_size.x = closest_2d_binned_size;
        image_cropped_size.y = closest_2d_binned_size;
        image_cropped_size.z = 1; // FIXME: once we add chunking ...
    }
    else {
        // FIXME: get rid of the hardcoded ones
        image_pre_scaling_size.x = image_size.x;
        image_pre_scaling_size.y = image_size.y;
        image_pre_scaling_size.z = 1; // FIXME: once we add chunking ...

        image_cropped_size.x = image_size.x;
        image_cropped_size.y = image_size.y;
        image_cropped_size.z = 1; // FIXME: once we add chunking ...
    }

    // When FastFFT is enabled, primes are limited to 2 by the calling method.
    int factor_result_pos;
    for ( auto& prime_value : primes ) {
        factor_result_pos = ReturnClosestFactorizedUpper(image_cropped_size.x, prime_value, true, MUST_BE_FACTOR_OF);
        if ( (float)(-image_cropped_size.x + factor_result_pos) < float(image_cropped_size.x) * max_increase_by_fraction_of_image ) {
            image_search_size.x = factor_result_pos;
            break;
        }
    }

    for ( auto& prime_value : primes ) {
        factor_result_pos = ReturnClosestFactorizedUpper(image_cropped_size.y, prime_value, true, MUST_BE_FACTOR_OF);
        if ( (float)(-image_cropped_size.y + factor_result_pos) < float(image_cropped_size.y) * max_increase_by_fraction_of_image ) {
            image_search_size.y = factor_result_pos;
            break;
        }
    }

    image_search_size.z = 1; // FIXME: once we add chunking ...

    template_pre_scaling_size.x = best_3d_bin_offset + template_size.x;
    template_pre_scaling_size.y = best_3d_bin_offset + template_size.y;
    template_pre_scaling_size.z = best_3d_bin_offset + template_size.z;

    template_cropped_size.x = closest_3d_binned_size;
    template_cropped_size.y = closest_3d_binned_size;
    template_cropped_size.z = closest_3d_binned_size;

    // In the general case, there are no restrictions on the template being a power of two, but we should want a decent size
    int prime_factor_3d    = use_fast_fft ? 2 : 5;
    template_search_size.x = ReturnClosestFactorizedUpper(template_cropped_size.x, prime_factor_3d, true, MUST_BE_POWER_OF_TWO);
    template_search_size.y = template_search_size.x;
    template_search_size.z = template_search_size.x;
    // We know this is an even dimension so adding 2
    template_search_size.w = (template_search_size.x + 2) / 2;

    search_pixel_size = pixel_size * 0.5f * (closest_2d_binning_factor + closest_3d_binning_factor);

#ifdef DEBUG_TM_SIZER_PRINT
    wxPrintf("The input image will be padded by %d,%d, cropped to %d,%d and then padded again to %d,%d\n",
             image_pre_scaling_size.x - image_size.x, image_pre_scaling_size.y - image_size.y,
             image_cropped_size.x, image_cropped_size.y,
             image_search_size.x, image_search_size.y);
    wxPrintf("template_size = %i\n", template_size.x);
    wxPrintf("closest_3d_binning_factor = %f\n", closest_3d_binning_factor);
    wxPrintf("closest_3d_binning_factor * pixel_size = %f\n", closest_3d_binning_factor * pixel_size);
    wxPrintf("closest 2d binning factor = %f\n", closest_2d_binning_factor);
    wxPrintf("closest 2d binning factor * pixel_size = %f\n", closest_2d_binning_factor * pixel_size);
    wxPrintf("original image size = %i\n", int(image_size.x));
    wxPrintf("wanted_binned_size = %i,%i\n", image_cropped_size.x, image_cropped_size.y);
    wxPrintf("input  pixel size: %3.6f\n", pixel_size);
    wxPrintf("target pixel size: %3.6f\n", target_binning_factor * pixel_size);
    wxPrintf("search pixel size: %3.6f\n", search_pixel_size);
#endif
    // Now try to increase the padding of the input image to match the 3d
    CheckSizing( );
    sizing_is_set = true;

    int pre_binning_padding_x;
    int post_binning_padding_x;
    int pre_binning_padding_y;
    int post_binning_padding_y;

    // Things are simplified because the padding is always resulting in an even dimensions
    // NOTE: assuming integer division.
    // This is the first padding, if resampling it will be to an even and square size, otherwise it will be to a nice fourier size and the final step.
    GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);
#ifdef DEBUG_TM_SIZER_PRINT
    wxPrintf("pre_binning_padding_x = %i\n", pre_binning_padding_x);
    wxPrintf("pre_binning_padding_y = %i\n", pre_binning_padding_y);
    wxPrintf("post_binning_padding_x = %i\n", post_binning_padding_x);
    wxPrintf("post_binning_padding_y = %i\n", post_binning_padding_y);
#endif

    if ( resampling_is_needed ) {
        // Here we need to scale the padding to account for resampling.
        // I think the easiest way to handle fractional reduction, which could result in an odd number of invalid rows/columns is to round up
        pre_binning_padding_x  = myroundint(ceilf(float(pre_binning_padding_x) / GetFullBinningFactor( )));
        pre_binning_padding_y  = myroundint(ceilf(float(pre_binning_padding_y) / GetFullBinningFactor( )));
        post_binning_padding_x = myroundint(ceilf(float(post_binning_padding_x) / GetFullBinningFactor( )));
        post_binning_padding_y = myroundint(ceilf(float(post_binning_padding_y) / GetFullBinningFactor( )));

#ifdef DEBUG_TM_SIZER_PRINT
        wxPrintf("binning factor = %f\n", GetFullBinningFactor( ));
        wxPrintf("pre_binning_padding_x = %i\n", pre_binning_padding_x);
        wxPrintf("pre_binning_padding_y = %i\n", pre_binning_padding_y);
        wxPrintf("post_binning_padding_x = %i\n", post_binning_padding_x);
        wxPrintf("post_binning_padding_y = %i\n", post_binning_padding_y);
#endif
        // Now add on any padding needed to make the image a power of two
        // These are both even dimensions, so we can just use the symmetric padding.
        pre_binning_padding_x += (image_search_size.x - image_cropped_size.x) / 2;
        pre_binning_padding_y += (image_search_size.y - image_cropped_size.y) / 2;
        post_binning_padding_x += (image_search_size.x - image_cropped_size.x) / 2;
        post_binning_padding_y += (image_search_size.y - image_cropped_size.y) / 2;
#ifdef DEBUG_TM_SIZER_PRINT

        wxPrintf("+= pre_binning_padding_x = %i\n", pre_binning_padding_x);
        wxPrintf("+= pre_binning_padding_y = %i\n", pre_binning_padding_y);
        wxPrintf("+= post_binning_padding_x = %i\n", post_binning_padding_x);
        wxPrintf("+= post_binning_padding_y = %i\n", post_binning_padding_y);
#endif
    }
    SetValidSearchImageIndiciesFromPadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);
};

void TemplateMatchingDataSizer::SetValidSearchImageIndiciesFromPadding(const int pre_padding_x, const int pre_padding_y, const int post_padding_x, const int post_padding_y) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(valid_bounds_are_set, "Valid bounds have already been set");

    // This could be too large if the user created a reference with a very large box size
    // This could be too small if the search is far from focus and has a tightly cropped box size
    // TODO: Consider these cases, but for now we are only considering what is valid in the sense of the image processing.
    // NOTE: On whether or not to remove these results:
    //       While it is true that a peak halfway into the template width from the edge may still be strong enough to be detectible,
    //       and thereby we create "false negatives" by enforcing this exclusion border the more important issue is to return only peak heights that are not due to the location
    //       in the image. We cannot determine how the downstream processing may compare peak to peak within an image, and so we must be conservative,
    //       in the goal of not leading these downstream analysis to errant conclusions.
    int template_padding = template_cropped_size.x / cistem::fraction_of_box_size_to_exclude_for_border;

    search_image_valid_area_lower_bound_x = pre_padding_x + template_padding;
    search_image_valid_area_lower_bound_y = pre_padding_y + template_padding;
    search_image_valid_area_upper_bound_x = image_search_size.x - 1 - post_padding_x - template_padding;
    search_image_valid_area_upper_bound_y = image_search_size.y - 1 - post_padding_y - template_padding;

    pre_padding.x = search_image_valid_area_lower_bound_x;
    pre_padding.y = search_image_valid_area_lower_bound_y;

    roi.x = search_image_valid_area_upper_bound_x - search_image_valid_area_lower_bound_x + 1;
    roi.y = search_image_valid_area_upper_bound_y - search_image_valid_area_lower_bound_y + 1;

    number_of_valid_search_pixels = long(roi.x) * long(roi.y);
    MyDebugAssertTrue(number_of_valid_search_pixels > 0, "The number of valid search pixels is less than 1");

#ifdef DEBUG_TM_SIZER_PRINT
    wxPrintf("The valid search area is %i %i %i %i\n", search_image_valid_area_lower_bound_x, search_image_valid_area_lower_bound_y, search_image_valid_area_upper_bound_x, search_image_valid_area_upper_bound_y);
    wxPrintf("The number of valid search pixels is %li\n", number_of_valid_search_pixels);
#endif
    valid_bounds_are_set = true;
};

void TemplateMatchingDataSizer::GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(int& pre_padding_x, int& pre_padding_y, int& post_padding_x, int& post_padding_y) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(padding_is_set, "Padding has already been set");
    // There are no restrictions on the input image for this function, it may be sq or rect, even or odd,
    // but presumably there is only one layer of padding and it is >= 0
    // TODO: we need to consider the template
    int padding_x_TOTAL;
    int padding_y_TOTAL;

    pre_padding.x        = image_pre_scaling_size.x - image_size.x;
    pre_padding.y        = image_pre_scaling_size.y - image_size.y;
    pre_padding_search.x = image_search_size.x - image_size.x;
    if ( resampling_is_needed ) {
        padding_x_TOTAL = image_pre_scaling_size.x - image_size.x;
        padding_y_TOTAL = image_pre_scaling_size.y - image_size.y;
    }
    else {
        // When not useing fast FFT there is at most one padding step from input size to a nice fourier size.
        padding_x_TOTAL = image_search_size.x - image_cropped_size.x;
        padding_y_TOTAL = image_search_size.y - image_cropped_size.y;
    }

#ifdef DEBUG_TM_SIZER_PRINT
    wxPrintf("in get input image to even and square or prime factored size padding\n");
    wxPrintf("image_size = %i %i %i\n", image_size.x, image_size.y, image_size.z);
    wxPrintf("image_pre_scaling_size = %i %i %i\n", image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z);
    wxPrintf("image_cropped_size = %i %i %i\n", image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
    wxPrintf("image_search_size = %i %i %i\n", image_search_size.x, image_search_size.y, image_search_size.z);
#endif

    if ( IsEven(image_size.x) ) {
        post_padding_x = padding_x_TOTAL / 2;
        pre_padding_x  = padding_x_TOTAL / 2;
    }
    else {
        post_padding_x = padding_x_TOTAL / 2;
        pre_padding_x  = padding_x_TOTAL / 2 + 1;
    }
    if ( IsEven(image_size.y) ) {
        post_padding_y = padding_y_TOTAL / 2;
        pre_padding_y  = padding_y_TOTAL / 2;
    }
    else {
        post_padding_y = padding_y_TOTAL / 2;
        pre_padding_y  = padding_y_TOTAL / 2 + 1;
    }

    padding_is_set = true;
    return;
}

/**
 * @brief Check to see if the resolution limit is within the Nyquist limit. Also set a flag that indicates whether resampling is needed.
 * 
 * @param wanted_high_resolution_limit 
 */
void TemplateMatchingDataSizer::SetHighResolutionLimit(const float wanted_high_resolution_limit) {
    if ( wanted_high_resolution_limit < 2.0f * pixel_size )
        high_resolution_limit = 2.0f * pixel_size;
    else
        high_resolution_limit = wanted_high_resolution_limit;

    if ( FloatsAreAlmostTheSame(high_resolution_limit, 2.0f * pixel_size) )
        resampling_is_needed = false;
    else
        resampling_is_needed = true;
};

void TemplateMatchingDataSizer::ResizeTemplate_preSearch(Image& template_image, const bool use_lerp_not_fourier_resampling) {

    if ( use_lerp_not_fourier_resampling ) {
        // We only need to set the 3d to be the padded power of two size and have the resampling be handled during the projection step.
        // search size always >= cropped size
        template_image.Resize(std::max(template_size.x, template_search_size.x),
                              std::max(template_size.y, template_search_size.y),
                              std::max(template_size.z, template_search_size.z),
                              template_image.ReturnAverageOfRealValuesOnEdges( ));
    }
    else {
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            // Print out the size of each step
            wxPrintf("template_size = %i %i %i\n", template_size.x, template_size.y, template_size.z);
            wxPrintf("template_pre_scaling_size = %i %i %i\n", template_pre_scaling_size.x, template_pre_scaling_size.y, template_pre_scaling_size.z);
            wxPrintf("template_cropped_size = %i %i %i\n", template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
            wxPrintf("template_search_size = %i %i %i\n", template_search_size.x, template_search_size.y, template_search_size.z);
            template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image.mrc", 1, template_size.z / 2);
        }
#endif
        template_image.Resize(template_pre_scaling_size.x, template_pre_scaling_size.y, template_pre_scaling_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
#ifdef DEBUG_IMG_OUTPUT
        template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image_resized_pre_scale.mrc", 1, template_pre_scaling_size.z / 2);
#endif
        template_image.ForwardFFT( );
        template_image.Resize(template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
        template_image.BackwardFFT( );
#ifdef DEBUG_IMG_OUTPUT
        template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image_resized_cropped.mrc", 1, template_cropped_size.z / 2);
#endif
        template_image.Resize(template_search_size.x, template_search_size.y, template_search_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image_resized.mrc", 1, template_search_size.z / 2);
        }
#endif
    }
};

void TemplateMatchingDataSizer::ResizeTemplate_postSearch(Image& template_image) {
    MyAssertTrue(false, "Not yet implemented");
};

void TemplateMatchingDataSizer::ResizeImage_preSearch(Image& input_image) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");

    if ( resampling_is_needed ) {
        wxPrintf("Resampling the input image\n");
        Image tmp_sq;

        tmp_sq.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, true);
#ifdef USE_ZERO_PADDING_NOT_NOISE
        bool skip_padding_in_clipinto = false;
#else
        bool skip_padding_in_clipinto = true;
        tmp_sq.FillWithNoiseFromNormalDistribution(0.f, 1.0f);
#endif

        input_image.ClipInto(&tmp_sq, 0.0f, false, 1.0f, 0, 0, 0, skip_padding_in_clipinto);
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            tmp_sq.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/tmp_sq.mrc", 1);
#endif
        tmp_sq.ForwardFFT( );
        tmp_sq.Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
        tmp_sq.ZeroCentralPixel( );
        tmp_sq.DivideByConstant(sqrtf(tmp_sq.ReturnSumOfSquares( )));
        tmp_sq.BackwardFFT( );
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            tmp_sq.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/tmp_sq_resized.mrc", 1);
#endif

        input_image.Allocate(image_search_size.x, image_search_size.y, image_search_size.z, true);
#ifndef USE_ZERO_PADDING_NOT_NOISE
        input_image.FillWithNoiseFromNormalDistribution(0.f, 1.0f);
#endif
        tmp_sq.ClipInto(&input_image, 0.0f, false, 1.0f, 0, 0, 0, skip_padding_in_clipinto);

#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image_resized.mrc", 1);
        DEBUG_ABORT;
#endif
    }
    else {
        wxPrintf("not resampling\n");
    }

// NOTE: rotation must always be the FINAL step in pre-processing / resizing and it is always the first to be inverted at the end.
#ifdef ROTATEFORSPEED
    if ( ! is_power_of_two(image_search_size.x) && is_power_of_two(image_search_size.y) ) {
        // The speedup in the FFT for better factorization is also dependent on the dimension. The full transform (in cufft anyway) is faster if the best dimension is on X.
        // TODO figure out how to check the case where there is no factor of two, but one dimension is still faster. Probably getting around to writing an explicit planning tool would be useful.
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("Rotating the search image for speed\n");
        }
        input_image.RotateInPlaceAboutZBy90Degrees(true);
        // bool preserve_origin = true;
        // input_reconstruction.RotateInPlaceAboutZBy90Degrees(true, preserve_origin);
        // The amplitude spectrum is also rotated
        is_rotated_by_90 = true;
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image_rotated.mrc", 1);
#endif
    }
    else {
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("Not rotating the search image for speed even though it is enabled\n");
        }
        is_rotated_by_90 = false;
    }
#endif
};

void TemplateMatchingDataSizer::ResizeImage_postSearch(Image& input_image,
                                                       Image& max_intensity_projection,
                                                       Image& best_psi,
                                                       Image& best_phi,
                                                       Image& best_theta,
                                                       Image& best_defocus,
                                                       Image& best_pixel_size,
                                                       Image& correlation_pixel_sum_image,
                                                       Image& correlation_pixel_sum_of_squares_image) {

    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(use_fast_fft ? is_rotated_by_90 : false, "Rotating the search image when using fastfft does  not make sense given the current square size restriction of FastFFT");
    MyDebugAssertTrue(max_intensity_projection.logical_x_dimension <= (is_rotated_by_90 ? image_size.y : image_size.x), "The max intensity projection is larger than the original image size");
    MyDebugAssertTrue(max_intensity_projection.logical_y_dimension <= (is_rotated_by_90 ? image_size.x : image_size.y), "The max intensity projection is larger than the original image size");
    MyDebugAssertTrue(pre_processed_image.at(0).is_in_memory, "The pre-processed image is not in memory");
    MyDebugAssertFalse(pre_processed_image.at(1).is_in_memory, "Chunking the search image is not supported, but the pre-processed image has allocated mem in the second chunk");

    // We want the unsampled regions to have the same mean AND variance as the sampled regions,

    RandomNumberGenerator my_rand(pi_v<float>);

    // now loop over and update the mean values int rand_y_idx;
    int  rand_x_idx, rand_y_idx;
    long rand_address;

    // To avoid dislocations around the edge, the random padding is not enough for these images with their much larger dynamic range.
    // We might devise a mirrored or replicative padding, but a simpler solution is just to handle dividing by N here rather than latter in the processing.

    int x_lower_bound = pre_padding.x;
    int y_lower_bound = pre_padding.y;
    int x_upper_bound = pre_padding.x + roi.x;
    int y_upper_bound = pre_padding.y + roi.y;
    int i_x, i_y;

    float x_radius = float(max_intensity_projection.physical_address_of_box_center_x - x_lower_bound);
    float y_radius = float(max_intensity_projection.physical_address_of_box_center_y - y_lower_bound);

    long pixel_counter = 0;
    for ( int y = 0; y < max_intensity_projection.logical_y_dimension; y++ ) {
        for ( int x = 0; x < max_intensity_projection.logical_x_dimension; x++ ) {
            // Mirror the edges up to the upper bound then work our way back. This will not be perfect when the image is padded more than 2x the ROI
            if ( x < x_lower_bound ) {
                i_x = 2 * x_lower_bound - x - 1;
                if ( i_x >= x_upper_bound ) {
                    i_x = 2 * x_upper_bound - i_x - 1;
                }
            }
            else if ( x >= x_upper_bound ) {
                i_x = 2 * x_upper_bound - x - 1;
                if ( i_x < x_lower_bound ) {
                    i_x = 2 * x_lower_bound - i_x - 1;
                }
                MyDebugAssertFalse(i_x >= x_upper_bound, "i_x is out of bounds");
            }
            else {
                i_x = x;
            }
            if ( y < y_lower_bound ) {
                i_y = 2 * y_lower_bound - y - 1;
                if ( i_y >= y_upper_bound ) {
                    i_y = 2 * y_upper_bound - i_y - 1;
                }
            }
            else if ( y >= y_upper_bound ) {
                i_y = 2 * y_upper_bound - y - 1;
                if ( i_y < y_lower_bound ) {
                    i_y = 2 * y_lower_bound - i_y - 1;
                }
                MyDebugAssertFalse(i_y >= y_upper_bound, "i_y is out of bounds");
            }
            else {
                i_y = y;
            }
            if ( x < pre_padding.x || x >= pre_padding.x + roi.x || y < pre_padding.y || y >= pre_padding.y + roi.y ) {

                rand_address = max_intensity_projection.ReturnReal1DAddressFromPhysicalCoord(i_x, i_y, 0);

                max_intensity_projection.real_values[pixel_counter]               = max_intensity_projection.real_values[rand_address];
                best_psi.real_values[pixel_counter]                               = best_psi.real_values[rand_address];
                best_phi.real_values[pixel_counter]                               = best_phi.real_values[rand_address];
                best_theta.real_values[pixel_counter]                             = best_theta.real_values[rand_address];
                best_defocus.real_values[pixel_counter]                           = best_defocus.real_values[rand_address];
                best_pixel_size.real_values[pixel_counter]                        = best_pixel_size.real_values[rand_address];
                correlation_pixel_sum_image.real_values[pixel_counter]            = correlation_pixel_sum_image.real_values[rand_address];
                correlation_pixel_sum_of_squares_image.real_values[pixel_counter] = correlation_pixel_sum_of_squares_image.real_values[rand_address];
            }
            pixel_counter++;
        }
        pixel_counter += max_intensity_projection.padding_jump_value;
    }

    // Work through the transformations backward to get to the original image size
    if ( is_rotated_by_90 ) {

        // swap the bounds
        float tmp_x = x_radius;
        x_radius    = y_radius;
        y_radius    = tmp_x;

        // swap back all the images prior to re-sizing
        input_image.BackwardFFT( );
        input_image.RotateInPlaceAboutZBy90Degrees(false);
        max_intensity_projection.RotateInPlaceAboutZBy90Degrees(false);

        best_psi.RotateInPlaceAboutZBy90Degrees(false);
        // If the template is also rotated, then this additional accounting is not needed.
        // To account for the pre-rotation, psi needs to have 90 added to it.
        best_psi.AddConstant(90.0f);
        // We also want the angles to remain in (0,360] so loop over and clamp
        for ( int idx = 0; idx < best_psi.real_memory_allocated; idx++ ) {
            best_psi.real_values[idx] = clamp_angular_range_0_to_2pi(best_psi.real_values[idx], true);
        }
        best_theta.RotateInPlaceAboutZBy90Degrees(false);
        best_phi.RotateInPlaceAboutZBy90Degrees(false);
        best_defocus.RotateInPlaceAboutZBy90Degrees(false);
        best_pixel_size.RotateInPlaceAboutZBy90Degrees(false);

        correlation_pixel_sum_image.RotateInPlaceAboutZBy90Degrees(false);
        correlation_pixel_sum_of_squares_image.RotateInPlaceAboutZBy90Degrees(false);
    }

    // We need to use nearest neighbor interpolation to cast all existing values back to the original size.
    Image           tmp_mip, tmp_psi, tmp_phi, tmp_theta, tmp_defocus, tmp_pixel_size, tmp_sum, tmp_sum_sq;
    constexpr float NN_no_value = -std::numeric_limits<float>::max( );
    constexpr float no_value    = 0.f;

    if ( resampling_is_needed ) {

        // original size -> pad to square -> crop to binned -> pad to fourier
        // The new images at the square binned size (remove the padding to power of two)

        // We'll fill all the images with -FLT_MAX to indicate to downstream code that the values are not valid measurements from an experiment.
        tmp_phi.Allocate(image_size.x, image_size.y, image_size.z, true);
        tmp_theta.Allocate(image_size.x, image_size.y, image_size.z, true);
        tmp_psi.Allocate(image_size.x, image_size.y, image_size.z, true);
        tmp_defocus.Allocate(image_size.x, image_size.y, image_size.z, true);
        tmp_pixel_size.Allocate(image_size.x, image_size.y, image_size.z, true);

        tmp_phi.SetToConstant(NN_no_value);
        tmp_theta.SetToConstant(NN_no_value);
        tmp_psi.SetToConstant(NN_no_value);
        tmp_defocus.SetToConstant(NN_no_value);
        tmp_pixel_size.SetToConstant(NN_no_value);

        long        searched_image_address = 0;
        long        out_of_bounds_value    = 0;
        long        address                = 0;
        const float actual_image_binning   = GetFullBinningFactor( );

        // Loop over the (possibly) binned image coordinates
        for ( int j = search_image_valid_area_lower_bound_y; j <= search_image_valid_area_upper_bound_y; j++ ) {
            int y_offset_from_origin = j - max_intensity_projection.physical_address_of_box_center_y;
            for ( int i = search_image_valid_area_lower_bound_x; i <= search_image_valid_area_upper_bound_x; i++ ) {
                // Get this pixels offset from the center of the box
                int x_offset_from_origin = i - max_intensity_projection.physical_address_of_box_center_x;

                // Scale by the binning
                // TODO: not sure if round or truncation (floor) makes more sense here
                int x_non_binned = tmp_phi.physical_address_of_box_center_x + myroundint(float(x_offset_from_origin) * actual_image_binning);
                int y_non_binned = tmp_phi.physical_address_of_box_center_y + myroundint(float(y_offset_from_origin) * actual_image_binning);

                if ( x_non_binned >= 0 && x_non_binned < tmp_phi.logical_x_dimension && y_non_binned >= 0 && y_non_binned < tmp_phi.logical_y_dimension ) {
                    address                = tmp_phi.ReturnReal1DAddressFromPhysicalCoord(x_non_binned, y_non_binned, 0);
                    searched_image_address = max_intensity_projection.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
                }
                else {
                    // FIXME: This print block needs to be removed after initial debugging.
                    wxPrintf("x_non_binned = %d, y_non_binned = %d\n", x_non_binned, y_non_binned);
                    wxPrintf("%f actual_image_binning = %f\n", search_pixel_size, actual_image_binning);
                    wxPrintf("tmp mip size = %d %d\n", tmp_phi.logical_x_dimension, tmp_phi.logical_y_dimension);
                    wxPrintf("max_intensity_projection size = %d %d\n", max_intensity_projection.logical_x_dimension, max_intensity_projection.logical_y_dimension);
                    address = -1;
                }

                // There really shouldn't be any peaks out of bounds
                // I think we should only every update an address once, so let's check it here for now.
                if ( address < 0 || address > tmp_phi.real_memory_allocated ) {
                    out_of_bounds_value++;
                }
                else {
                    MyDebugAssertFalse(tmp_phi.real_values[address] == no_value, "Address already updated");
                    tmp_phi.real_values[address]        = best_phi.real_values[searched_image_address];
                    tmp_theta.real_values[address]      = best_theta.real_values[searched_image_address];
                    tmp_psi.real_values[address]        = best_psi.real_values[searched_image_address];
                    tmp_defocus.real_values[address]    = best_defocus.real_values[searched_image_address];
                    tmp_pixel_size.real_values[address] = best_pixel_size.real_values[searched_image_address];
                }
            }
        }

        MyDebugAssertTrue(out_of_bounds_value == 0, "There are out of bounds values in calculating the NN interpolation of the max intensity projection");

        tmp_mip.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, false);
        tmp_sum.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, false);
        tmp_sum_sq.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, false);

        // Resize from any fourier padding to the cropped size
        max_intensity_projection.Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
        correlation_pixel_sum_image.Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
        correlation_pixel_sum_of_squares_image.Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);

        // Now undo the fourier binning
        max_intensity_projection.ForwardFFT( );
        max_intensity_projection.ClipInto(&tmp_mip, 0.0f, false, 1.0f, 0, 0, 0, true);
        tmp_mip.BackwardFFT( );
        max_intensity_projection.Allocate(image_size.x, image_size.y, image_size.z, true);
        tmp_mip.ClipInto(&max_intensity_projection, 0.0f, false, 1.0f, 0, 0, 0, true);

        // Dilate the radius of the valid area mask
        x_radius *= GetFullBinningFactor( );
        y_radius *= GetFullBinningFactor( );

        correlation_pixel_sum_image.ForwardFFT( );
        correlation_pixel_sum_image.ClipInto(&tmp_sum, 0.0f, false, 1.0f, 0, 0, 0, true);
        tmp_sum.BackwardFFT( );
        correlation_pixel_sum_image.Allocate(image_size.x, image_size.y, image_size.z, true);
        tmp_sum.ClipInto(&correlation_pixel_sum_image, 0.0f, false, 1.0f, 0, 0, 0, true);

        correlation_pixel_sum_of_squares_image.ForwardFFT( );
        correlation_pixel_sum_of_squares_image.ClipInto(&tmp_sum_sq, 0.0f, false, 1.0f, 0, 0, 0, true);
        tmp_sum_sq.BackwardFFT( );
        correlation_pixel_sum_of_squares_image.Allocate(image_size.x, image_size.y, image_size.z, true);
        tmp_sum_sq.ClipInto(&correlation_pixel_sum_of_squares_image, 0.0f, false, 1.0f, 0, 0, 0, true);
    } // end resampling_is_needed

    // Create a mask that will be filled based on the possibly rotated and resized search image, and then rescaled in the same manner, so that we can use this for adjusting the
    // stats images/ histogram elsewhere post resizing.
    valid_area_mask.Allocate(max_intensity_projection.logical_x_dimension, max_intensity_projection.logical_y_dimension, 1, true);
    valid_area_mask.SetToConstant(1.0f);
    constexpr float mask_radius = 7.f;

    valid_area_mask.SetToConstant(1.0f);
    valid_area_mask.CosineRectangularMask(x_radius, y_radius, 0, mask_radius, false, true, 0.f);

    valid_area_mask.Binarise(0.9f);
    valid_area_mask.ZeroFFTWPadding( );

    if ( resampling_is_needed ) {

        FillInNearestNeighbors(best_psi, tmp_psi, valid_area_mask, NN_no_value);
        FillInNearestNeighbors(best_phi, tmp_phi, valid_area_mask, NN_no_value);
        FillInNearestNeighbors(best_theta, tmp_theta, valid_area_mask, NN_no_value);
        FillInNearestNeighbors(best_defocus, tmp_defocus, valid_area_mask, NN_no_value);
        FillInNearestNeighbors(best_pixel_size, tmp_pixel_size, valid_area_mask, NN_no_value);
    }

    // For the other images, calculate the mean under the mask and change the padding to this so the display contrast is okay
    double mip_mean        = 0.0;
    double phi_mean        = 0.0;
    double theta_mean      = 0.0;
    double psi_mean        = 0.0;
    double defocus_mean    = 0.0;
    double pixel_size_mean = 0.0;
    double n_counted       = 0.0;

    for ( long address = 0; address < max_intensity_projection.real_memory_allocated; address++ ) {
        n_counted += valid_area_mask.real_values[address];
        if ( valid_area_mask.real_values[address] > 0.0f ) {
            mip_mean += max_intensity_projection.real_values[address] * valid_area_mask.real_values[address];
            phi_mean += best_phi.real_values[address] * valid_area_mask.real_values[address];
            theta_mean += best_theta.real_values[address] * valid_area_mask.real_values[address];
            psi_mean += best_psi.real_values[address] * valid_area_mask.real_values[address];
            defocus_mean += best_defocus.real_values[address] * valid_area_mask.real_values[address];
            pixel_size_mean += best_pixel_size.real_values[address] * valid_area_mask.real_values[address];
        }
    }

    for ( long address = 0; address < max_intensity_projection.real_memory_allocated; address++ ) {
        if ( valid_area_mask.real_values[address] == 0.0f ) {
            max_intensity_projection.real_values[address]               = mip_mean / n_counted;
            best_phi.real_values[address]                               = phi_mean / n_counted;
            best_theta.real_values[address]                             = theta_mean / n_counted;
            best_psi.real_values[address]                               = psi_mean / n_counted;
            best_defocus.real_values[address]                           = defocus_mean / n_counted;
            best_pixel_size.real_values[address]                        = pixel_size_mean / n_counted;
            correlation_pixel_sum_of_squares_image.real_values[address] = 0.0f;
            correlation_pixel_sum_image.real_values[address]            = 0.0f;
        }
    }
};

void TemplateMatchingDataSizer::FillInNearestNeighbors(Image& output_image, Image& nn_upsampled_image, Image& valid_area_mask, const float no_value) {

    // Set the non-valid area to zero (not no_value) so that we can use the no_value to check if the pixel has been filled in.
    nn_upsampled_image.MultiplyPixelWise(valid_area_mask);
    output_image.CopyFrom(&nn_upsampled_image);
    int size_neighborhood = 3;
    while ( float(size_neighborhood) < cistem::match_template::MAX_BINNING_FACTOR ) {
        if ( GetFullBinningFactor( ) <= float(size_neighborhood) ) {
            break;
        }
        else
            size_neighborhood += 2;
    }
    // We could try to dilate out each neighborhood, but this will be slower given the bad memory access. Better to do a little extra.
    int offset_max = size_neighborhood / 2;

    // Loop over the image
    for ( int j = 0; j < nn_upsampled_image.logical_y_dimension; j++ ) {
        for ( int i = 0; i < nn_upsampled_image.logical_x_dimension; i++ ) {
            float current_value = nn_upsampled_image.ReturnRealPixelFromPhysicalCoord(i, j, 0);
            if ( current_value == no_value ) {
                int   min_distance_squared = std::numeric_limits<int>::max( );
                float closest_value        = no_value;

                // First check the line in memory that includes the current pixel, setting boundaries in the for loop
                for ( int x = std::max(i - offset_max, 0); x <= std::min(i + offset_max, nn_upsampled_image.logical_x_dimension - 1); x++ ) {
                    // We don't need to check the current pixel
                    if ( x != i ) {
                        // No need to load the value if the distance is already too large
                        if ( x * x < min_distance_squared ) {
                            current_value = nn_upsampled_image.ReturnRealPixelFromPhysicalCoord(x, j, 0);
                            if ( current_value != no_value ) {
                                min_distance_squared = x * x;
                                closest_value        = current_value;
                            }
                        }
                    }
                }
                // If we still haven't found it, we'll check each row left and right,
                int y_offset = 1;

                // We can't get any closer than 1, so if we've already found a value, we can stop
                if ( min_distance_squared == 1 ) {
                    goto endOfElse;
                }

                while ( y_offset <= offset_max ) {
                    for ( int y = j - y_offset; y <= j + y_offset; y += 2 * y_offset ) {
                        // We can't set the limits in the for loop initializer and just bracket the end, so use a conditional here
                        if ( y < 0 || y >= nn_upsampled_image.logical_y_dimension ) {
                            continue;
                        }
                        for ( int x = std::max(i - offset_max, 0); x <= std::min(i + offset_max, nn_upsampled_image.logical_x_dimension - 1); x++ ) {
                            // This time we hit all pixels in the line

                            // No need to load the value if the distance is already too large
                            if ( y * y + x * x < min_distance_squared ) {
                                current_value = nn_upsampled_image.ReturnRealPixelFromPhysicalCoord(x, y, 0);
                                if ( current_value != no_value ) {
                                    min_distance_squared = y * y + x * x;
                                    closest_value        = current_value;
                                }
                            }
                            // The smallest distance we can get now is == to the x offset
                            if ( min_distance_squared == y_offset * y_offset ) {
                                goto endOfElse;
                            }
                        }
                    }
                    // If we get here, we've checked the full neighborhood size y_offset * 2, which meanse the smallest distance > y_offset is the corner, = sqrt(y_offset^2 + y_offset^2)
                    if ( min_distance_squared == 2 * y_offset * y_offset ) {
                        goto endOfElse;
                    }
                    y_offset++;
                }

            endOfElse:
                MyDebugAssertFalse(closest_value == no_value, "No value found");
                output_image.real_values[output_image.ReturnReal1DAddressFromPhysicalCoord(i, j, 0)] = closest_value;
            }
        }
    }
}
