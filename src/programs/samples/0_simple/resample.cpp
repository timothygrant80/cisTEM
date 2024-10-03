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

void ResampleRunner(const wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting downsampling tests:", false);

    wxString cistem_ref_dir = CheckForReferenceImages( );

    constexpr bool test_is_to_be_run = false;
    if constexpr ( test_is_to_be_run ) {
        // If we are in the dev container the CISTEM_REF_IMAGES variable should be defined, pointing to images we need.
        TEST(DoFourierCropVsLerpResize(cistem_ref_dir, temp_directory));
    }
    else
        SamplesTestResultCanFail(false);

    SamplesPrintEndMessage( );

    return;
}

bool DoFourierCropVsLerpResize(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool passed     = true;
    bool all_passed = true;

    const int logical_input_size = 384;

    AnglesAndShifts prj_angles(10.f, -20.f, 130.f, 0.f, 0.f);

    SamplesBeginTest("Extract slice and downsample", passed);

    std::string volume_filename          = cistem_ref_dir.ToStdString( ) + "/ribo_ref.mrc";
    std::string prj_input_filename_base  = cistem_ref_dir.ToStdString( ) + "/ribo_ref_prj_";
    std::string prj_output_filename_base = temp_directory.ToStdString( ) + "/ribo_ref_prj_";

    bool      over_write_input = false;
    Image     cpu_volume;
    ImageFile cpu_volume_file;

    GpuImage gpu_volume;
    GpuImage gpu_prj_full; // project 384 then crop to 192 (downsample by 2)
    GpuImage gpu_prj_cropped;
    GpuImage gpu_prj_lerp; // project and resample in the same step
    GpuImage gpu_prj_lerp_non_binned_size;

    // Read in and normalize the 3d to use for projection
    cpu_volume_file.OpenFile(volume_filename, over_write_input);
    cpu_volume.ReadSlices(&cpu_volume_file, 1, cpu_volume_file.ReturnNumberOfSlices( ));
    cpu_volume.ZeroFloatAndNormalize( );

    // Make sure the volume has the expected size
    MyAssertTrue(cpu_volume.logical_x_dimension == logical_input_size && cpu_volume.IsCubic( ), "The volume should be 384x384x384");

    // Prepare for GPU projection
    constexpr bool also_swap_real_space_quadrants = true;
    cpu_volume.SwapFourierSpaceQuadrants(also_swap_real_space_quadrants);
    // Associate the gpu volume with the cpu volume, getting meta data and pinning the host pointer.
    gpu_volume.Init(cpu_volume, false, true);
    gpu_volume.CopyHostToDeviceTextureComplex3d(cpu_volume);

    // For the positive control, project at the full size, and fourier crop to the binned size
    gpu_prj_full.Allocate(logical_input_size, logical_input_size, 1, false, false);
    gpu_prj_cropped.Allocate(logical_input_size / 2, logical_input_size / 2, 1, false, false);

    // For direct comparison to gpu_prj_cropped, incorporate the lerp into the projection obviating the need for a separate crop.
    gpu_prj_lerp.Allocate(logical_input_size / 2, logical_input_size / 2, 1, false, false);
    // For the case where we would first bin but then zero-pad to some other larger size, for example, to have a nice power of 2 image.
    gpu_prj_lerp_non_binned_size.Allocate(logical_input_size + 128, logical_input_size + 128, 1, false, false);
    // Make sure there are no non-zero vals
    gpu_prj_full.SetToConstant(0.0f);
    gpu_prj_cropped.SetToConstant(0.0f);
    gpu_prj_lerp.SetToConstant(0.0f);
    gpu_prj_lerp_non_binned_size.SetToConstant(0.0f);

    // The gpu projection method expects quadrants to be swapped.
    gpu_prj_full.object_is_centred_in_box                 = false;
    gpu_prj_cropped.object_is_centred_in_box              = false;
    gpu_prj_lerp.object_is_centred_in_box                 = false;
    gpu_prj_lerp_non_binned_size.object_is_centred_in_box = false;

    // Dummy ctf image
    GpuImage dummy_ctf_image;

    constexpr float resolution_limit          = 1.0f;
    constexpr bool  apply_resolution_limit    = false;
    constexpr bool  apply_shifts              = false;
    constexpr bool  swap_real_space_quadrants = true;
    constexpr bool  apply_ctf                 = false;
    constexpr bool  absolute_ctf              = false;
    constexpr bool  zero_central_pixel        = true;

    // Project the full size image
    gpu_prj_full.ExtractSliceShiftAndCtf(&gpu_volume, &dummy_ctf_image, prj_angles, 1.0f, 1.0f, resolution_limit, apply_resolution_limit, swap_real_space_quadrants, apply_ctf, absolute_ctf, zero_central_pixel);

    // gpu_prj_full.SwapRealSpaceQuadrants( );
    // Crop the full size image
    gpu_prj_full.ClipIntoFourierSpace(&gpu_prj_cropped, 0.f, true, false);

    gpu_prj_lerp.ExtractSliceShiftAndCtf(&gpu_volume, &dummy_ctf_image, prj_angles, 1.0f, 2.0f, resolution_limit, apply_resolution_limit, swap_real_space_quadrants, apply_shifts, apply_ctf, absolute_ctf, zero_central_pixel);
    // gpu_prj_lerp.SwapRealSpaceQuadrants( );

    gpu_prj_lerp_non_binned_size.ExtractSliceShiftAndCtf(&gpu_volume, &dummy_ctf_image, prj_angles, 1.0f, 2.0f, resolution_limit, apply_resolution_limit, swap_real_space_quadrants, apply_shifts, apply_ctf, absolute_ctf, zero_central_pixel);
    // gpu_prj_lerp_non_binned_size.SwapRealSpaceQuadrants( );

    // Save both for inspection (temporarily)
    gpu_prj_full.QuickAndDirtyWriteSlice(prj_output_filename_base + "full.mrc", 1);
    gpu_prj_cropped.QuickAndDirtyWriteSlice(prj_output_filename_base + "cropped.mrc", 1);
    gpu_prj_lerp.QuickAndDirtyWriteSlice(prj_output_filename_base + "lerp.mrc", 1);
    gpu_prj_lerp_non_binned_size.QuickAndDirtyWriteSlice(prj_output_filename_base + "lerp_non_binned_size.mrc", 1);

    gpu_prj_lerp.SubtractImage(&gpu_prj_cropped);
    gpu_prj_lerp.BackwardFFT( );
    float sum = gpu_prj_lerp.ReturnSumOfSquares( );
    std::cerr << "Sum is " << sum << std::endl;

    wxPrintf("\n\nI am HERE in DoFourierCropVsLerpResize\n\n");
    exit(0);
    return true;
}