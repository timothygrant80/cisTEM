#include "template_matching_peak_extractor.h"

#include <algorithm>
#include <numeric>

/**
 * @brief Construct a peak extractor that references MIP and parameter images for metadata lookup.
 *
 * The constructor is intentionally lightweight - it does not modify any images. Reconstruction
 * preparation (downsampling, FFT) is deferred to `PrepareReconstruction` which is called lazily
 * by `CreateResultImages`. This allows the caller to modify the reconstruction (e.g. apply padding
 * in make_template_result) between constructing the extractor and creating result images.
 *
 * `pixel_size_image` is a pointer rather than a reference because prepare_stack_matchtemplate
 * does not have a pixel size image - it stores pixel size per-micrograph, not per-peak.
 * When null, `TransferAndSortPeakInfo` uses `search_pixel_size` for all peaks instead.
 *
 * Dimension assertions catch mismatched parameter images early rather than producing
 * silent corruption when looking up metadata at peak addresses.
 */
TemplateMatchingPeakExtractor::TemplateMatchingPeakExtractor(
        Image& mip_image,
        Image& phi_image,
        Image& theta_image,
        Image& psi_image,
        Image& defocus_image,
        Image* pixel_size_image,
        float  input_pixel_size,
        float  search_pixel_size)
    : mip_image_(mip_image),
      phi_image_(phi_image),
      theta_image_(theta_image),
      psi_image_(psi_image),
      defocus_image_(defocus_image),
      pixel_size_image_(pixel_size_image),
      input_pixel_size_(input_pixel_size),
      search_pixel_size_(search_pixel_size),
      needs_downsampling_(! FloatsAreAlmostTheSame(search_pixel_size, input_pixel_size) && search_pixel_size > input_pixel_size),
      has_downsampled_(false),
      downsampled_reconstruction_(nullptr) {

    // Only assert dimensions when parameter images are allocated. When reading peaks from a
    // coordinate file (e.g. prepare_stack_matchtemplate), these images are not needed.
    if ( phi_image_.is_in_memory )
        MyDebugAssertTrue(phi_image_.HasSameDimensionsAs(&mip_image_), "Phi image must have same dimensions as MIP");
    if ( theta_image_.is_in_memory )
        MyDebugAssertTrue(theta_image_.HasSameDimensionsAs(&mip_image_), "Theta image must have same dimensions as MIP");
    if ( psi_image_.is_in_memory )
        MyDebugAssertTrue(psi_image_.HasSameDimensionsAs(&mip_image_), "Psi image must have same dimensions as MIP");
    if ( defocus_image_.is_in_memory )
        MyDebugAssertTrue(defocus_image_.HasSameDimensionsAs(&mip_image_), "Defocus image must have same dimensions as MIP");
    if ( pixel_size_image_ != nullptr ) {
        MyDebugAssertTrue(pixel_size_image_->HasSameDimensionsAs(&mip_image_), "Pixel size image must have same dimensions as MIP");
    }
}

bool TemplateMatchingPeakExtractor::NeedsDownsampling( ) const {
    return needs_downsampling_;
}

/**
 * @brief Look up angles, defocus, and pixel size at each peak's physical address in the
 *        parameter images and populate an output array of TemplateMatchFoundPeakInfo.
 *
 * Peaks from `FindPeakWithIntegerCoordinatesForManyPeaks` carry a `physical_address_within_image`
 * that directly indexes into the parameter images' `real_values` arrays since all images share the
 * same dimensions and memory layout. This avoids recomputing 2D->1D address mappings.
 *
 * Bounds and NaN checks are included because edge peaks can have addresses near or beyond
 * the FFTW padding boundary, and corrupted MIP values (e.g. from numerical issues in the
 * stats images) could produce NaN peak values that would propagate through downstream code.
 * Invalid peaks are skipped with a warning rather than aborting, since losing one peak is
 * preferable to losing the entire result set.
 */
void TemplateMatchingPeakExtractor::TransferAndSortPeakInfo(std::vector<Peak>&                  peak_list,
                                                            std::vector<Peak>&                  upsampled_peak_list,
                                                            bool                                use_corrected_peak,
                                                            ArrayOfTemplateMatchFoundPeakInfos& output) const {

    // Sort both lists in descending order by the appropriate peak value
    std::vector<size_t> indices(peak_list.size( ));
    std::iota(indices.begin( ), indices.end( ), 0);
    if ( use_corrected_peak ) {
        std::sort(indices.begin( ), indices.end( ),
                  [&upsampled_peak_list](size_t a, size_t b) {
                      return upsampled_peak_list[a].value > upsampled_peak_list[b].value;
                  });
    }
    else {
        std::sort(indices.begin( ), indices.end( ),
                  [&peak_list](size_t a, size_t b) {
                      return peak_list[a].value > peak_list[b].value;
                  });
    }

    std::vector<Peak> sorted_peak_list, sorted_upsampled_peak_list;
    sorted_peak_list.reserve(peak_list.size( ));
    sorted_upsampled_peak_list.reserve(upsampled_peak_list.size( ));
    for ( size_t idx : indices ) {
        sorted_peak_list.push_back(peak_list[idx]);
        sorted_upsampled_peak_list.push_back(upsampled_peak_list[idx]);
    }
    peak_list           = std::move(sorted_peak_list);
    upsampled_peak_list = std::move(sorted_upsampled_peak_list);

    TemplateMatchFoundPeakInfo peak_info;

    for ( size_t i = 0; i < peak_list.size( ); i++ ) {
        const Peak& peak      = peak_list[i];
        const Peak& upsampled = upsampled_peak_list[i];
        int         px        = myroundint(peak.x);
        int         py        = myroundint(peak.y);
        if ( px < 0 || px >= mip_image_.logical_x_dimension || py < 0 || py >= mip_image_.logical_y_dimension ) {
            wxPrintf("WARNING: Peak at (%f, %f) is out of bounds, skipping.\n", peak.x, peak.y);
            continue;
        }
        if ( peak.physical_address_within_image < 0 || peak.physical_address_within_image >= mip_image_.real_memory_allocated ) {
            wxPrintf("WARNING: Peak physical address %ld is out of bounds, skipping.\n", peak.physical_address_within_image);
            continue;
        }

        if ( std::isnan(peak.value) || peak.value <= std::numeric_limits<float>::lowest( ) ||
             (use_corrected_peak && std::isnan(upsampled.value)) ) {
            continue;
        }

        long address = peak.physical_address_within_image;

        peak_info.x_pos       = peak.x * search_pixel_size_;
        peak_info.y_pos       = peak.y * search_pixel_size_;
        peak_info.phi         = phi_image_.real_values[address];
        peak_info.theta       = theta_image_.real_values[address];
        peak_info.psi         = psi_image_.real_values[address];
        peak_info.defocus     = defocus_image_.real_values[address];
        peak_info.pixel_size  = (pixel_size_image_ != nullptr) ? pixel_size_image_->real_values[address] : search_pixel_size_;
        peak_info.peak_height = use_corrected_peak ? upsampled.value : peak.value;

        output.Add(peak_info);
    }
}

/**
 * @brief Read peaks from a coordinate file and populate both a Peak vector and a peak_infos array.
 *
 * The coordinate file format is 8 columns: psi, theta, phi, x_ang, y_ang, defocus, pixel_size, peak_height.
 * Coordinates are stored in Angstroms in the file and converted to MIP pixel coordinates here
 * by dividing by `search_pixel_size_`. This matches the convention used when writing the file
 * in make_template_result and prepare_stack_matchtemplate.
 *
 * Both `peak_list` and `peak_infos` are populated so that downstream code (CreateResultImages,
 * CreateParticleStack) receives the same data structures regardless of whether peaks came from
 * a search or a file. The Peak struct needs `physical_address_within_image` set correctly
 * because `CreateResultImages` uses the x/y pixel coordinates for projection insertion, and
 * the address is needed if the caller wants to do further lookups.
 *
 * The same bounds/NaN checks as TransferAndSortPeakInfo are applied - coordinate files can contain
 * stale entries from previous runs at different binning levels.
 */
void TemplateMatchingPeakExtractor::ReadPeaksFromCoordinateFile(NumericTextFile&                    coordinate_file,
                                                                std::vector<Peak>&                  peak_list,
                                                                ArrayOfTemplateMatchFoundPeakInfos& peak_infos) const {

    float                      coordinates[8];
    TemplateMatchFoundPeakInfo peak_info;

    for ( int line = 0; line < coordinate_file.number_of_lines; line++ ) {
        coordinate_file.ReadLine(coordinates);

        float x_px = coordinates[3] / search_pixel_size_;
        float y_px = coordinates[4] / search_pixel_size_;

        int px = myroundint(x_px);
        int py = myroundint(y_px);
        if ( px < 0 || px >= mip_image_.logical_x_dimension || py < 0 || py >= mip_image_.logical_y_dimension ) {
            wxPrintf("WARNING: Coordinate file peak at (%f, %f) px is out of bounds, skipping.\n", x_px, y_px);
            continue;
        }

        long address = mip_image_.ReturnReal1DAddressFromPhysicalCoord(px, py, 0);
        if ( address < 0 || address >= mip_image_.real_memory_allocated ) {
            wxPrintf("WARNING: Coordinate file peak address %ld is out of bounds, skipping.\n", address);
            continue;
        }

        if ( std::isnan(coordinates[7]) || coordinates[7] <= std::numeric_limits<float>::lowest( ) ) {
            continue;
        }

        peak_info.psi         = coordinates[0];
        peak_info.theta       = coordinates[1];
        peak_info.phi         = coordinates[2];
        peak_info.x_pos       = coordinates[3];
        peak_info.y_pos       = coordinates[4];
        peak_info.defocus     = coordinates[5];
        peak_info.pixel_size  = coordinates[6];
        peak_info.peak_height = coordinates[7];

        peak_infos.Add(peak_info);

        peak_list.emplace_back(x_px, y_px, 1.f, coordinates[7], address);
    }
}

/**
 * @brief Downsample (if needed), normalize, and FFT-prepare a reconstruction for projection extraction.
 *
 * When the search was run at a binned pixel size, we need to downsample the reconstruction to
 * match. This is done out-of-place into `downsampled_reconstruction_` so the caller's original
 * reconstruction is not modified - important because make_template_result may have already
 * applied padding to it and we don't want to interfere with that.
 *
 * The downsampled copy is cached via `has_downsampled_` so that if CreateResultImages were
 * called multiple times (not current usage but defensive), we don't redundantly downsample.
 *
 * Normalization uses `ReturnAverageOfMaxN()` rather than the global max to be more robust
 * against single-voxel outliers - this has been the historical approach for result image
 * visualization in cisTEM.
 *
 * The sqrt(Nx * Ny * sqrt(Nz)) scaling factor after FFT compensates for the FFTW normalization
 * convention so that extracted 2D projections have correct relative intensities. ZeroCentralPixel
 * removes the DC component, and SwapRealSpaceQuadrants prepares for ExtractSlice which expects
 * the quadrants in this arrangement.
 *
 * Important: This method modifies the working reconstruction in-place (the downsampled copy if
 * downsampling was needed, otherwise the passed-in reconstruction). After calling this, the
 * reconstruction is in Fourier space and should only be used via ExtractSlice.
 */
void TemplateMatchingPeakExtractor::PrepareReconstruction(Image& reconstruction) {

    Image* working_reconstruction = &reconstruction;

    if ( needs_downsampling_ && ! has_downsampled_ ) {
        downsampled_reconstruction_ = std::make_unique<Image>( );
        downsampled_reconstruction_->CopyFrom(&reconstruction);

        float binning_factor = search_pixel_size_ / input_pixel_size_;
        int   new_size       = int(reconstruction.logical_x_dimension / binning_factor + 0.5f);
        if ( IsOdd(new_size) )
            new_size++;

        downsampled_reconstruction_->ForwardFFT( );
        downsampled_reconstruction_->Resize(new_size, new_size, new_size);
        downsampled_reconstruction_->BackwardFFT( );
        has_downsampled_ = true;
    }

    if ( downsampled_reconstruction_ != nullptr ) {
        working_reconstruction = downsampled_reconstruction_.get( );
    }

    float max_density = working_reconstruction->ReturnAverageOfMaxN( );
    working_reconstruction->DivideByConstant(max_density);

    working_reconstruction->ForwardFFT( );
    working_reconstruction->MultiplyByConstant(sqrtf(working_reconstruction->logical_x_dimension * working_reconstruction->logical_y_dimension * sqrtf(working_reconstruction->logical_z_dimension)));
    working_reconstruction->ZeroCentralPixel( );
    working_reconstruction->SwapRealSpaceQuadrants( );
}

/**
 * @brief Extract projections from a 3D reconstruction at each peak's orientation and insert
 *        them into a 2D result montage image. Optionally insert rotated reconstructions into
 *        a 3D slab volume.
 *
 * This consolidates the projection extraction loop that was previously duplicated across
 * match_template.cpp and make_template_result.cpp (via the old ProcessNextPeak method).
 *
 * The method first calls PrepareReconstruction, which handles downsampling and FFT setup.
 * When `padded_projection` is non-null (make_template_result with padding > 1), the extraction
 * goes through a larger padded image first: extract into padded -> BFFT -> clip into projection
 * size -> FFFT. This produces higher-quality projections at the cost of the larger FFT.
 * When null (match_template), we extract directly at the projection size.
 *
 * The edge-average subtraction after BFFT removes the mean background from each projection
 * so that when inserted into the result image, projections don't create visible rectangular
 * boundaries at their edges.
 *
 * Slab insertion (make_template_result only) rotates the binned reconstruction by the inverse
 * angles to place the template in the orientation it was found at, then inserts it at the
 * peak position scaled to the slab's coarser pixel size. The z-offset uses the defocus value
 * to position the particle at the correct depth in the slab.
 *
 * The `binned_reconstruction` for the slab must be pre-prepared by the caller (copied, resized,
 * normalized) before passing it here. This is because the slab binning is independent of the
 * search binning handled by PrepareReconstruction.
 */
void TemplateMatchingPeakExtractor::CreateResultImages(
        const std::vector<Peak>&                  peak_list,
        const ArrayOfTemplateMatchFoundPeakInfos& peak_infos,
        Image&                                    input_reconstruction,
        Image&                                    current_projection,
        Image&                                    result_image,
        bool                                      create_slab,
        Image*                                    padded_projection,
        Image*                                    slab,
        Image*                                    binned_reconstruction,
        float                                     binned_pixel_size) {

    PrepareReconstruction(input_reconstruction);

    Image* working_reconstruction = (downsampled_reconstruction_ != nullptr)
                                            ? downsampled_reconstruction_.get( )
                                            : &input_reconstruction;

    AnglesAndShifts angles;
    size_t          num_peaks = std::min(peak_list.size( ), static_cast<size_t>(peak_infos.GetCount( )));
    num_peaks                 = std::min(num_peaks, static_cast<size_t>(cistem::match_template::MAX_ALLOWED_NUMBER_OF_PEAKS));

    for ( int i = 0; i < num_peaks; i++ ) {
        const Peak&                       peak = peak_list[i];
        const TemplateMatchFoundPeakInfo& info = peak_infos[i];

        angles.Init(info.phi, info.theta, info.psi, 0.0, 0.0);

        if ( padded_projection != nullptr ) {
            working_reconstruction->ExtractSlice(*padded_projection, angles, 1.0f, false);
            padded_projection->SwapRealSpaceQuadrants( );
            padded_projection->BackwardFFT( );
            padded_projection->ClipInto(&current_projection);
            current_projection.ForwardFFT( );
        }
        else {
            working_reconstruction->ExtractSlice(current_projection, angles, 1.0f, false);
            current_projection.SwapRealSpaceQuadrants( );
        }

        current_projection.MultiplyByConstant(sqrtf(current_projection.logical_x_dimension * current_projection.logical_y_dimension));
        current_projection.BackwardFFT( );
        current_projection.AddConstant(-current_projection.ReturnAverageOfRealValuesOnEdges( ));

        result_image.InsertOtherImageAtSpecifiedPosition(&current_projection,
                                                         peak.x - result_image.physical_address_of_box_center_x,
                                                         peak.y - result_image.physical_address_of_box_center_y,
                                                         0, 0.0f);

        if ( create_slab && slab != nullptr && binned_reconstruction != nullptr ) {
            Image rotated_reconstruction;
            angles.Init(-info.psi, -info.theta, -info.phi, 0.0, 0.0);
            rotated_reconstruction.CopyFrom(binned_reconstruction);
            rotated_reconstruction.Rotate3DByRotationMatrixAndOrApplySymmetry(angles.euler_matrix);

            slab->InsertOtherImageAtSpecifiedPosition(&rotated_reconstruction,
                                                      myroundint((peak.x - result_image.physical_address_of_box_center_x) * (search_pixel_size_ / binned_pixel_size)),
                                                      myroundint((peak.y - result_image.physical_address_of_box_center_y) * (search_pixel_size_ / binned_pixel_size)),
                                                      -myroundint(info.defocus / binned_pixel_size),
                                                      0.0f);
        }
    }
}

/**
 * @brief Cut particles from a micrograph and write a particle image stack and cisTEM star file.
 *
 * This consolidates the particle extraction loop from prepare_stack_matchtemplate. Peak
 * coordinates are in MIP pixel space and must be scaled to micrograph pixels via
 * `mip_to_micrograph_scale` (= search_pixel_size / micrograph_pixel_size) before clipping.
 *
 * Each particle is normalized by subtracting the edge mean and dividing by sqrt(variance).
 * The edge mean (not the global mean) is used because it better represents the background
 * level at the particle boundary, producing cleaner particles for downstream processing.
 * Zero variance is guarded against to avoid division by zero for blank regions.
 *
 * The star file stores `search_pixel_size_` as the pixel size rather than the micrograph
 * pixel size because the downstream refinement programs need to know the pixel size at which
 * the angles were determined. The defocus values stored are the sum of the micrograph average
 * defocus and the per-peak defocus offset from template matching.
 *
 * The first slice is written with `overwrite=true` to create a new file, and subsequent
 * slices append. This matches MRC stack conventions.
 */
void TemplateMatchingPeakExtractor::CreateParticleStack(
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
        const wxString&                           input_image_filename) const {

    Image current_particle;
    current_particle.Allocate(box_size, box_size, true);

    float micrograph_mean = micrograph.ReturnAverageOfRealValues( );

    cisTEMParameterLine output_parameters;
    cisTEMParameters    output_star_file;
    output_star_file.PreallocateMemoryAndBlank(peak_infos.GetCount( ) + 1);

    size_t num_peaks = std::min(peak_list.size( ), static_cast<size_t>(peak_infos.GetCount( )));

    for ( int i = 0; i < num_peaks; i++ ) {
        const Peak&                       peak = peak_list[i];
        const TemplateMatchFoundPeakInfo& info = peak_infos[i];

        float scaled_x = peak.x * mip_to_micrograph_scale;
        float scaled_y = peak.y * mip_to_micrograph_scale;

        micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0,
                            int(scaled_x - micrograph.physical_address_of_box_center_x),
                            int(scaled_y - micrograph.physical_address_of_box_center_y), 0);

        float variance = current_particle.ReturnVarianceOfRealValues( );
        if ( variance == 0.0f )
            variance = 1.0f;
        current_particle.AddMultiplyConstant(-current_particle.ReturnAverageOfRealValuesOnEdges( ), 1.0f / sqrtf(variance));

        int position = i + 1;
        if ( position == 1 )
            current_particle.QuickAndDirtyWriteSlice(output_stack_filename.ToStdString( ), position, true, search_pixel_size_);
        else
            current_particle.QuickAndDirtyWriteSlice(output_stack_filename.ToStdString( ), position);

        output_parameters.SetAllToZero( );
        output_parameters.position_in_stack                  = position;
        output_parameters.psi                                = info.psi;
        output_parameters.theta                              = info.theta;
        output_parameters.phi                                = info.phi;
        output_parameters.defocus_1                          = average_defocus_1 + info.defocus;
        output_parameters.defocus_2                          = average_defocus_2 + info.defocus;
        output_parameters.defocus_angle                      = average_defocus_angle;
        output_parameters.pixel_size                         = search_pixel_size_;
        output_parameters.microscope_voltage_kv              = voltage_kV;
        output_parameters.microscope_spherical_aberration_mm = spherical_aberration_mm;
        output_parameters.amplitude_contrast                 = amplitude_contrast;
        output_parameters.occupancy                          = 1.0f;
        output_parameters.sigma                              = 10.0f;
        output_parameters.logp                               = 5000.0f;
        output_parameters.score                              = 50.0f;
        output_parameters.image_is_active                    = 1;
        output_parameters.stack_filename                     = output_stack_filename;
        output_parameters.original_image_filename            = input_image_filename;

        output_star_file.all_parameters[position] = output_parameters;
    }

    output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, 1, num_peaks);
}
