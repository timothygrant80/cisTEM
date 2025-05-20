#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "resample.h"

constexpr int logical_input_size = 384;

void ResampleRunner(const wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting downsampling tests:", false);

    wxString cistem_ref_dir = CheckForReferenceImages( );

    constexpr bool test_is_to_be_run = true;
    if constexpr ( test_is_to_be_run ) {
        // If we are in the dev container the CISTEM_REF_IMAGES variable should be defined, pointing to images we need.
        TEST(DoFourierExpandVsLerpResize(cistem_ref_dir, temp_directory));
        TEST(DoCTFImageVsTexture(cistem_ref_dir, temp_directory));
        TEST(DoFourierCropVsLerpResize(cistem_ref_dir, temp_directory));
        TEST(DoLerpWithCTF(cistem_ref_dir, temp_directory));
    }
    else
        SamplesTestResultCanFail(false);

    SamplesPrintEndMessage( );

    return;
}

struct ResampleRunnerObjects {

    std::string volume_filename;

    Image     cpu_volume;
    ImageFile cpu_volume_file;
    Image     ctf_image;
    Image     swapped_ctf_image;

    GpuImage gpu_volume;
    GpuImage gpu_prj;
    GpuImage gpu_prj_tex;
    GpuImage d_ctf_image;

    CTF             ctf;
    AnglesAndShifts prj_angles;

    const float pixel_size{1.f};
    const float resolution_limit{1.f};
    const bool  apply_resolution_limit{ };
    const bool  apply_shifts{ };
    const bool  swap_real_space_quadrants{true};
    const bool  zero_central_pixel{true};

    float real_space_binning_factor{1.f};

    ResampleRunnerObjects(const wxString& cistem_ref_dir, const wxString& temp_directory, const int wanted_size = logical_input_size) {

        volume_filename = cistem_ref_dir.ToStdString( ) + "/ribo_ref.mrc";
        // Read in and normalize the 3d to use for projection

        const bool over_write_input = false;
        cpu_volume_file.OpenFile(volume_filename, over_write_input);
        cpu_volume.ReadSlices(&cpu_volume_file, 1, cpu_volume_file.ReturnNumberOfSlices( ));
        cpu_volume.ZeroFloatAndNormalize( );

        // Make sure the volume has the expected size
        MyAssertTrue(cpu_volume.logical_x_dimension == logical_input_size && cpu_volume.IsCubic( ), "The volume should be 384x384x384");

        if ( wanted_size != logical_input_size ) {
            cpu_volume.Resize(wanted_size, wanted_size, wanted_size);
        }
        // Prepare for GPU projection
        constexpr bool also_swap_real_space_quadrants = true;
        cpu_volume.SwapFourierSpaceQuadrants(also_swap_real_space_quadrants);
        // Associate the gpu volume with the cpu volume, getting meta data and pinning the host pointer.
        gpu_volume.Init(cpu_volume, false, true);
        gpu_volume.CopyHostToDeviceTextureComplex<3>(cpu_volume);

        // Generate a CTF image
        GetCTFGivenPixelSizeInAngstroms(ctf, pixel_size);
        ctf_image.Allocate(wanted_size, wanted_size, 1, false);
        ctf_image.CalculateCTFImage(ctf);

        swapped_ctf_image = ctf_image;
        swapped_ctf_image.SwapFourierSpaceQuadrants(false, true);

        // Now we'll grab a projection and apply the CTF to it in the same kernel.
        prj_angles.Init(10.f, -20.f, 130.f, 0.f, 0.f);
        gpu_prj.InitializeBasedOnCpuImage(ctf_image, true, true);
        gpu_prj_tex.InitializeBasedOnCpuImage(ctf_image, false, true);
        d_ctf_image.InitializeBasedOnCpuImage(ctf_image, false, true);
        d_ctf_image.CopyHostToDevice(ctf_image);
        d_ctf_image.CopyFP32toFP16buffer(false);
    }

    inline void GetCTFGivenPixelSizeInAngstroms(CTF& input_ctf, float pixel_size_in_angstroms) {
        input_ctf.Init(300.f, 2.7f, 0.07f, 12000.f, 12000.f, 40.f, pixel_size_in_angstroms, 0.f);
    }
};

bool DoCTFImageVsTexture(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("CTF image vs texture", passed);

    ResampleRunnerObjects o_(cistem_ref_dir, temp_directory);

    constexpr bool do_not_use_ctf_texture = false;
    constexpr bool apply_ctf              = true;
    o_.gpu_prj.ExtractSliceShiftAndCtf<apply_ctf, do_not_use_ctf_texture>(&o_.gpu_volume,
                                                                          &o_.d_ctf_image,
                                                                          o_.prj_angles,
                                                                          o_.pixel_size,
                                                                          o_.real_space_binning_factor,
                                                                          o_.resolution_limit,
                                                                          o_.apply_resolution_limit,
                                                                          o_.swap_real_space_quadrants,
                                                                          o_.apply_shifts,
                                                                          o_.zero_central_pixel);

    o_.gpu_prj_tex.CopyHostToDeviceTextureComplex<2>(o_.swapped_ctf_image);

    constexpr bool use_ctf_texture = true;
    o_.gpu_prj_tex.ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(&o_.gpu_volume,
                                                                       &o_.gpu_prj_tex,
                                                                       o_.prj_angles,
                                                                       o_.pixel_size,
                                                                       o_.real_space_binning_factor,
                                                                       o_.resolution_limit,
                                                                       o_.apply_resolution_limit,
                                                                       o_.swap_real_space_quadrants,
                                                                       o_.apply_shifts,
                                                                       o_.zero_central_pixel);

    o_.gpu_prj.BackwardFFT( );
    o_.gpu_prj_tex.BackwardFFT( );

    o_.gpu_prj.NormalizeRealSpaceStdDeviation(1.f, 0.f, 0.f);
    o_.gpu_prj_tex.NormalizeRealSpaceStdDeviation(1.f, 0.f, 0.f);

    o_.gpu_prj_tex.SubtractImage(o_.gpu_prj);
    float SS = o_.gpu_prj_tex.ReturnSumOfSquares( );

    passed = FloatsAreAlmostTheSame(SS, 0.f);

    SamplesTestResult(passed);

    return all_passed;
}

bool DoFourierCropVsLerpResize(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Extract slice and downsample", passed);

    ResampleRunnerObjects o_(cistem_ref_dir, temp_directory);

    // For direct comparison to gpu_prj_cropped, incorporate the lerp into the projection obviating the need for a separate crop.
    // For the case where we would first bin but then zero-pad to some other larger size, for example, to have a nice power of 2 image.
    // Make sure there are no non-zero vals
    o_.gpu_prj.SetToConstant(0.0f);

    // The gpu projection method expects quadrants to be swapped.
    o_.gpu_prj.object_is_centred_in_box = false;

    // Dummy ctf image
    GpuImage dummy_ctf_image;

    constexpr bool apply_ctf       = false;
    constexpr bool use_ctf_texture = false;

    float real_space_binning_factor = 1.0f;
    // Project the full size image to be fourier cropped and compared
    o_.gpu_prj.ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(&o_.gpu_volume,
                                                                   &dummy_ctf_image,
                                                                   o_.prj_angles,
                                                                   1.0f,
                                                                   o_.real_space_binning_factor,
                                                                   o_.resolution_limit,
                                                                   o_.apply_resolution_limit,
                                                                   o_.swap_real_space_quadrants,
                                                                   o_.zero_central_pixel);

    std::array<int, 5> cropped_sizes{382, 192, 96, 48, 24};
    for ( auto& cropped_size : cropped_sizes ) {
        GpuImage binned_img, cropped_img;
        binned_img.Allocate(cropped_size, cropped_size, 1, false, false);
        cropped_img.Allocate(cropped_size, cropped_size, 1, false, false);
        real_space_binning_factor = float(o_.gpu_volume.logical_x_dimension) / float(cropped_size);
        binned_img.ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(&o_.gpu_volume,
                                                                       &dummy_ctf_image,
                                                                       o_.prj_angles,
                                                                       1.0f,
                                                                       o_.real_space_binning_factor,
                                                                       o_.resolution_limit,
                                                                       o_.apply_resolution_limit,
                                                                       o_.swap_real_space_quadrants,
                                                                       o_.apply_shifts,
                                                                       o_.zero_central_pixel);
        o_.gpu_prj.ClipIntoFourierSpace(&cropped_img, 0.f);

        binned_img.BackwardFFT( );
        cropped_img.BackwardFFT( );
        // Calculate the mean square error between the two images
        cropped_img.SubtractImage(binned_img);
        float SS = cropped_img.ReturnSumOfSquares( );
        passed   = passed && (FloatsAreAlmostTheSame(SS, 0.0f));
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    return true;
}

bool DoFourierExpandVsLerpResize(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Extract slice and upsample", passed);

    ResampleRunnerObjects o_(cistem_ref_dir, temp_directory);

    // For direct comparison to gpu_prj_cropped, incorporate the lerp into the projection obviating the need for a separate crop.
    // For the case where we would first bin but then zero-pad to some other larger size, for example, to have a nice power of 2 image.
    // Make sure there are no non-zero vals
    o_.gpu_prj.SetToConstant(0.0f);

    // The gpu projection method expects quadrants to be swapped.
    o_.gpu_prj.object_is_centred_in_box = false;

    // Dummy ctf image
    GpuImage dummy_ctf_image;

    constexpr bool apply_ctf       = false;
    constexpr bool use_ctf_texture = false;

    float real_space_binning_factor = 1.0f;

    std::array<int, 1> upsampled_sizes{512};
    for ( auto& upsampled_size : upsampled_sizes ) {
        ResampleRunnerObjects up_sampled_o_(cistem_ref_dir, temp_directory, upsampled_size);

        GpuImage binned_img, cropped_img;
        // The binned image will be interpolated from the regular size image
        binned_img.Allocate(upsampled_size, upsampled_size, 1, false, false);
        // The cropped image will be extracted from the upsampled 3d volume
        cropped_img.Allocate(upsampled_size, upsampled_size, 1, false, false);
        real_space_binning_factor = float(o_.gpu_volume.logical_x_dimension) / float(upsampled_size);
        std::cerr << "real_space_binning_factor: " << real_space_binning_factor << std::endl;
        cropped_img.ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(&up_sampled_o_.gpu_volume,
                                                                        &dummy_ctf_image,
                                                                        up_sampled_o_.prj_angles,
                                                                        1.0f,
                                                                        1.0f, //real_space_binning_factor
                                                                        up_sampled_o_.resolution_limit,
                                                                        up_sampled_o_.apply_resolution_limit,
                                                                        up_sampled_o_.swap_real_space_quadrants,
                                                                        up_sampled_o_.apply_shifts,
                                                                        up_sampled_o_.zero_central_pixel);

        binned_img.ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(&o_.gpu_volume,
                                                                       &dummy_ctf_image,
                                                                       o_.prj_angles,
                                                                       1.0f,
                                                                       1.0f,
                                                                       o_.resolution_limit,
                                                                       o_.apply_resolution_limit,
                                                                       o_.swap_real_space_quadrants,
                                                                       o_.apply_shifts,
                                                                       o_.zero_central_pixel);

        binned_img.BackwardFFT( );
        cropped_img.BackwardFFT( );

        // We expect there to be some difference outside the original box size, so lets mask that before comparison.
        Image host_binned_img  = binned_img.CopyDeviceToNewHost(true, true);
        Image host_cropped_img = cropped_img.CopyDeviceToNewHost(true, true);

        host_binned_img.CorrectSinc( );
        host_binned_img.CorrectSinc( );
        host_binned_img.CosineMask(logical_input_size * 0.35, 7.f);
        host_cropped_img.CosineMask(logical_input_size * 0.35, 7.f);

        host_binned_img.ZeroFloatAndNormalize( );
        host_cropped_img.ZeroFloatAndNormalize( );

        // lets take a peak at the images
        host_binned_img.QuickAndDirtyWriteSlices("binned_img_" + std::to_string(upsampled_size) + ".mrc", 1, 1);
        host_cropped_img.QuickAndDirtyWriteSlices("cropped_img_" + std::to_string(upsampled_size) + ".mrc", 1, 1);
        // Calculate the mean square error between the two images
        cropped_img.SubtractImage(binned_img);
        float SS = cropped_img.ReturnSumOfSquares( );
        std::cerr << "SS: " << SS << std::endl;
        passed = passed && (FloatsAreAlmostTheSame(SS, 0.0f));
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    exit(0);

    return true;
}

bool DoLerpWithCTF(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Extract slice and downsample lerp+ctf", passed);

    ResampleRunnerObjects o_(cistem_ref_dir, temp_directory);

    constexpr bool apply_ctf       = true;
    constexpr bool use_ctf_texture = true;
    constexpr bool skip_ctf        = false;

    o_.gpu_prj_tex.CopyHostToDeviceTextureComplex<2>(o_.swapped_ctf_image);

    // Project the full size image to be fourier cropped and compared
    o_.gpu_prj.ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(&o_.gpu_volume,
                                                                   &o_.gpu_prj_tex,
                                                                   o_.prj_angles,
                                                                   1.0f,
                                                                   o_.real_space_binning_factor,
                                                                   o_.resolution_limit,
                                                                   o_.apply_resolution_limit,
                                                                   o_.swap_real_space_quadrants,
                                                                   o_.zero_central_pixel);

    std::array<int, 5> cropped_sizes{382, 192, 96, 48, 24};

    for ( auto& cropped_size : cropped_sizes ) {
        GpuImage binned_img, cropped_img, binned_ctf_applied_after;
        binned_img.Allocate(cropped_size, cropped_size, 1, false, false);
        cropped_img.Allocate(cropped_size, cropped_size, 1, false, false);
        binned_ctf_applied_after.Allocate(cropped_size, cropped_size, 1, false, false);

        o_.real_space_binning_factor = float(o_.gpu_volume.logical_x_dimension) / float(cropped_size);
        binned_img.ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(&o_.gpu_volume,
                                                                       &o_.gpu_prj_tex,
                                                                       o_.prj_angles,
                                                                       1.0f,
                                                                       o_.real_space_binning_factor,
                                                                       o_.resolution_limit,
                                                                       o_.apply_resolution_limit,
                                                                       o_.swap_real_space_quadrants,
                                                                       o_.apply_shifts,
                                                                       o_.zero_central_pixel);

        binned_ctf_applied_after.ExtractSliceShiftAndCtf<skip_ctf, use_ctf_texture>(&o_.gpu_volume,
                                                                                    &o_.gpu_prj_tex,
                                                                                    o_.prj_angles,
                                                                                    1.0f,
                                                                                    o_.real_space_binning_factor,
                                                                                    o_.resolution_limit,
                                                                                    o_.apply_resolution_limit,
                                                                                    o_.swap_real_space_quadrants,
                                                                                    o_.apply_shifts,
                                                                                    o_.zero_central_pixel);

        // Our positive control, ctf applied to full size projection and fourier cropped
        o_.gpu_prj.ClipIntoFourierSpace(&cropped_img, 0.f);

        // Make a ctf image to apply to the skipped images
        CTF binned_ctf;
        o_.GetCTFGivenPixelSizeInAngstroms(binned_ctf, 1.0f * o_.real_space_binning_factor);
        Image cropped_ctf(cropped_size, cropped_size, 1, false);
        cropped_ctf.CalculateCTFImage(binned_ctf);
        GpuImage d_cropped_ctf(cropped_ctf);
        d_cropped_ctf.CopyHostToDevice(cropped_ctf);
        binned_ctf_applied_after.MultiplyPixelWise(d_cropped_ctf);

        binned_img.BackwardFFT( );
        cropped_img.BackwardFFT( );
        binned_ctf_applied_after.BackwardFFT( );

        // binned_img.QuickAndDirtyWriteSlices("binned_img_" + std::to_string(cropped_size) + ".mrc", 1, 1);
        // cropped_img.QuickAndDirtyWriteSlices("cropped_img_" + std::to_string(cropped_size) + ".mrc", 1, 1);
        // binned_ctf_applied_after.QuickAndDirtyWriteSlices("binned_ctf_applied_after_" + std::to_string(cropped_size) + ".mrc", 1, 1);

        binned_img.NormalizeRealSpaceStdDeviation(1.f, 0.f, 0.f);
        cropped_img.NormalizeRealSpaceStdDeviation(1.f, 0.f, 0.f);
        binned_ctf_applied_after.NormalizeRealSpaceStdDeviation(1.f, 0.f, 0.f);

        cropped_img.SubtractImage(binned_img);
        binned_ctf_applied_after.SubtractImage(binned_img);

        float SS_ctf_intra = cropped_img.ReturnSumOfSquares( );
        float SS_ctf_post  = binned_ctf_applied_after.ReturnSumOfSquares( );

        // We expect the CTF to be applied after the lerp to be the same as the CTF applied to the full size image and then cropped.
        passed = passed && FloatsAreAlmostTheSame(SS_ctf_intra, 0.f) && FloatsAreAlmostTheSame(SS_ctf_post, 0.f);
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(all_passed);

    return true;
}