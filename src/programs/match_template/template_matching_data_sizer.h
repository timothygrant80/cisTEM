#ifndef __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_DATA_SIZER_H_
#define __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_DATA_SIZER_H_

#include <memory>

#include "../../constants/constants.h"

// #define USE_NEAREST_NEIGHBOR_INTERPOLATION
// The USE_ZERO_PADDING_NOT_NOISE should be defined as padding with noise in real-space adds power to the noise
// in every Fourier voxel reducing the SSNR. I'm leaving the option here to test noise padding for pathological images in the future.
//#define USE_ZERO_PADDING_NOT_NOISE
// #define USE_REPLICATIVE_PADDING

constexpr bool  MUST_BE_POWER_OF_TWO                   = false; // Required for half-precision xforms
constexpr int   MUST_BE_FACTOR_OF                      = 0; // May be faster
constexpr float max_reduction_by_fraction_of_reference = 0.000001f; // FIXME the cpu version is crashing when the image is reduced, but not the GPU
constexpr int   MAX_3D_PADDING                         = 196;

/**
 * @brief This class is used to optionally resample, pad or cut into chunks the search image.
 * 
 * It relies on knowing the template size, particularly in the case of chunking.
 * This is a draft object and will be moved to a more appropriate location. TODO
 * 
 */
class TemplateMatchingDataSizer {

    // Keep a copy of the original image following pre-processing but no resizing. We'll use this
    // to deterimine what the peak would be if the image were not resized.
    // We are currently supporting at most 2 chunks (to make a k3 without super res into 2 4k images)
    std::array<Image, 2> pre_processed_image;

    // This is a non-data owning class, but we want references to the underlying image/template data
    int4 image_size;
    int4 image_pre_scaling_size;
    int4 image_cropped_size;
    int4 image_search_size;

    // Logical coordinates in the search image frame corresponding to non-padding regions
    // These are the only values mapped by NN interp to the result image and are
    // needed ahead of the search to exclude padding from the histogram and ideally also from FastFFT calculations.
    // NOTE: these values do NOT accound for any potential 90 degree rotation about Z, however, when padding to a square dimension, this is not relevant
    int search_image_valid_area_lower_bound_x;
    int search_image_valid_area_lower_bound_y;
    int search_image_valid_area_upper_bound_x;
    int search_image_valid_area_upper_bound_y;

    // n elements that are not valid search results,  pre_padding.x then is also the first valid physical x-index in zero based coordinates
    int2 pre_padding;
    int2 pre_padding_search;
    // logical x/y size of the region of interest.
    int2 roi;
    int2 roi_search;

    Image valid_area_mask;

    long number_of_valid_search_pixels;
    long number_of_pixels_for_normalization;

    int4 template_size;
    int4 template_pre_scaling_size;
    int4 template_cropped_size;
    int4 template_search_size;

    float pixel_size{0.f};
    float search_pixel_size{0.f};
    float template_padding{ };
    float high_resolution_limit{-1.f};
    bool  resampling_is_needed{false};
    bool  is_rotated_by_90{false};
    bool  use_fast_fft{false};
    bool  sizing_is_set{false};
    bool  padding_is_set{false};
    bool  valid_bounds_are_set{false};
    bool  image_is_split_into_chunks{false};

    std::vector<int> primes;

    float max_increase_by_fraction_of_image;

    void SetHighResolutionLimit(const float wanted_high_resolution_limit);
    void GetFFTSize( );
    void CheckSizing( );

    void GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(int& pre_padding_x, int& pre_padding_y, int& post_padding_x, int& post_padding_y);
    void SetValidSearchImageIndiciesFromPadding(const int pre_padding_x, const int pre_padding_y, const int post_padding_x, const int post_padding_y);

    void FillInNearestNeighbors(Image& output_image, Image& nn_upsampled_image, Image& valid_area_mask, const float no_value);

    MyApp* parent_match_template_app_ptr;

  public:
    TemplateMatchingDataSizer(MyApp* parent_match_template_app_ptr, Image& input_image, Image& wanted_template, float wanted_pixel_size, float wanted_template_padding);
    ~TemplateMatchingDataSizer( );

    std::unique_ptr<Curve> whitening_filter_ptr;

    // Don't allow copy or move. FIXME: if we don't add any dynamically allocated data, we can remove this.
    // TemplateMatchingDataSizer(const TemplateMatchingDataSizer&)            = delete;
    // TemplateMatchingDataSizer& operator=(const TemplateMatchingDataSizer&) = delete;
    // TemplateMatchingDataSizer(TemplateMatchingDataSizer&&)                 = delete;
    // TemplateMatchingDataSizer& operator=(TemplateMatchingDataSizer&&)      = delete;

    void SetImageAndTemplateSizing(const float wanted_high_resolution_limit, const bool use_fast_fft);
    void PreProcessInputImage(Image& input_image, bool swap_real_space_quadrants, bool normalize_to_variance_one);

    void PreProcessResizedInputImage(Image& input_image) { PreProcessInputImage(input_image, true, false); }

    void ResizeTemplate_preSearch(Image& template_image, const bool use_lerp_not_fourier_resampling = false, const bool allow_upsampling = false);
    void ResizeTemplate_postSearch(Image& template_image);

    // All statistical images (mip, psi etc.) are originally allocated based on the pre-processed input_image size,
    // and so only the input image needs attention at the outset. Following the search, all statistical images
    // will also need to be resized.
    void ResizeImage_preSearch(Image& input_image, const int central_cross_half_width);
    void ResizeImage_postSearch(Image& max_intensity_projection,
                                Image& best_psi,
                                Image& best_phi,
                                Image& best_theta,
                                Image& best_defocus,
                                Image& best_pixel_size,
                                Image& correlation_pixel_sum_image,
                                Image& correlation_pixel_sum_of_squares_image);

    inline void PrintImageSizes( ) {
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("old x, y = %i %i\n  new x, y = %i %i\n", image_size.x, image_size.y, image_search_size.x, image_search_size.y);
        }
    }

    inline bool IsResamplingNeeded( ) const {
        return resampling_is_needed;
    }

    inline bool IsRotatedBy90( ) const {
        return is_rotated_by_90;
    }

    inline int GetImageSizeX( ) const {
        return image_search_size.x;
    }

    inline int GetImageSizeY( ) const {
        return image_search_size.y;
    }

    inline int GetImageSearchSizeX( ) const {
        return image_search_size.x;
    }

    inline int GetImageSearchSizeY( ) const {
        return image_search_size.y;
    }

    inline int GetTemplateSearchSizeX( ) const {
        return template_search_size.x;
    }

    inline int GetTemplateSizeX( ) const {
        return template_size.x;
    }

    inline long GetNumberOfValidSearchPixels( ) const {
        MyDebugAssertTrue(valid_bounds_are_set, "Valid bounds not set");
        return number_of_valid_search_pixels;
    }

    inline long GetNumberOfPixelsForNormalization( ) const {
        MyDebugAssertTrue(valid_bounds_are_set, "Valid bounds not set");
#ifdef USE_ZERO_PADDING_NOT_NOISE
        return number_of_pixels_for_normalization;
#else
        return long(image_search_size.x * image_search_size.y);
#endif
    }

    inline float GetPixelSize( ) const {
        MyDebugAssertFalse(pixel_size == 0.0f, "Pixel size not set");
        return pixel_size;
    }

    inline float GetHighResolutionLimit( ) const {
        MyDebugAssertFalse(high_resolution_limit == -1.0f, "High resolution limit not set");
        return high_resolution_limit;
    }

    inline float GetSearchPixelSize( ) const {
        MyDebugAssertFalse(search_pixel_size == 0.0f, "Search pixel size not set");
        return search_pixel_size;
    }

    inline float GetFullBinningFactor( ) const {
        return GetSearchPixelSize( ) / GetPixelSize( );
    }

    inline void GetValidXYPhysicalIdicies(int& lower_x, int& lower_y, int& upper_x, int& upper_y) {
        lower_x = search_image_valid_area_lower_bound_x;
        lower_y = search_image_valid_area_lower_bound_y;
        upper_x = search_image_valid_area_upper_bound_x;
        upper_y = search_image_valid_area_upper_bound_y;
    }

    inline int GetBinnedSize(float input_size, float wanted_binning_factor) {
        return int(input_size / wanted_binning_factor + 0.5f);
    }

    // Note that the input_size will be cast from int -> float on the function call.
    inline float GetRealizedBinningFactor(float wanted_binning_factor, float input_size) {
        int wanted_binned_size = GetBinnedSize(input_size, wanted_binning_factor);
        if ( IsOdd(wanted_binned_size) )
            wanted_binned_size++;
        return input_size / float(wanted_binned_size);
    }

    inline int2 GetPrePadding( ) const {
        return pre_padding;
    }

    inline int2 GetPrePaddingSearch( ) const {
        return pre_padding_search;
    }

    inline int2 GetRoi( ) const {
        return roi;
    }

    inline int2 GetRoiSearch( ) const {
        return roi_search;
    }

    inline int GetPrePaddingX( ) const {
        return pre_padding.x;
    }

    inline int GetPrePaddingY( ) const {
        return pre_padding.y;
    }

    inline int GetPrePaddingSearchY( ) const {
        return pre_padding_search.y;
    }

    inline int GetRoiX( ) const {
        return roi.x;
    }

    inline int GetRoiSearchX( ) const {
        return roi_search.x;
    }

    inline int GetRoiY( ) const {
        return roi.y;
    }

    inline int GetRoiSearchY( ) const {
        return roi_search.y;
    }

    inline float* GetValidAreaMask( ) {
        return valid_area_mask.real_values;
    }
};

#endif