#ifndef __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_PEAK_EXTRACTOR_H_
#define __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_PEAK_EXTRACTOR_H_

#include "../../core/core_headers.h"

#include <memory>

class TemplateMatchingPeakExtractor {
  public:
    TemplateMatchingPeakExtractor(
            Image& mip_image,
            Image& phi_image,
            Image& theta_image,
            Image& psi_image,
            Image& defocus_image,
            Image* pixel_size_image,
            float  input_pixel_size,
            float  search_pixel_size);

    bool NeedsDownsampling( ) const;

    void TransferAndSortPeakInfo(std::vector<Peak>&                  peak_list,
                                 std::vector<Peak>&                  upsampled_peak_list,
                                 bool                                use_corrected_peak,
                                 ArrayOfTemplateMatchFoundPeakInfos& output) const;

    void ReadPeaksFromCoordinateFile(NumericTextFile&                    coordinate_file,
                                     std::vector<Peak>&                  peak_list,
                                     ArrayOfTemplateMatchFoundPeakInfos& peak_infos) const;

    void CreateResultImages(
            const std::vector<Peak>&                  peak_list,
            const ArrayOfTemplateMatchFoundPeakInfos& peak_infos,
            Image&                                    input_reconstruction,
            Image&                                    current_projection,
            Image&                                    result_image,
            bool                                      create_slab,
            Image*                                    padded_projection     = nullptr,
            Image*                                    slab                  = nullptr,
            Image*                                    binned_reconstruction = nullptr,
            float                                     binned_pixel_size     = 0.0f);

    void CreateParticleStack(
            const std::vector<Peak>&                  peak_list,
            const ArrayOfTemplateMatchFoundPeakInfos& peak_infos,
            Image&                                    micrograph,
            const wxString&                           output_stack_filename,
            const wxString&                           output_star_filename,
            int                                       box_size,
            float                                     mip_to_micrograph_scale,
            float                                     voltage_kV,
            float                                     spherical_aberration_mm,
            float                                     amplitude_contrast,
            float                                     average_defocus_1,
            float                                     average_defocus_2,
            float                                     average_defocus_angle,
            const wxString&                           input_image_filename) const;

  private:
    void PrepareReconstruction(Image& reconstruction);

    // Image references for parameter lookup
    Image& mip_image_;
    Image& phi_image_;
    Image& theta_image_;
    Image& psi_image_;
    Image& defocus_image_;
    Image* pixel_size_image_; // nullable

    // Pixel sizes
    float input_pixel_size_;
    float search_pixel_size_;

    // Downsampling state
    bool                   needs_downsampling_;
    bool                   has_downsampled_;
    std::unique_ptr<Image> downsampled_reconstruction_;
};

#endif
