/*
 * cpu_vs_gpu.hpp
 *
 *  Created on: Aug 10, 2021
 *      Author: B.A. Himes, Shiran Dror
 *
 *      Goal:
 *      	Compare resize functions on CPU and GPU
 *
 *
 */
#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#include "../../../gpu/GpuImage.h"
#else
#include "../../../core/core_headers.h"
#endif

#include "../common/common.h"
#include "resize_comparison.h"

void CPUvsGPUResizeRunner(wxString hiv_image_80x80x1_filename, wxString temp_directory) {

    SamplesPrintTestStartMessage("Starting CPU vs GPU resize tests:", false);

    TEST(DoCPUvsGPURealSpaceResize(hiv_image_80x80x1_filename, temp_directory));
    TEST(DoCPUvsGPUFourierResize(hiv_image_80x80x1_filename, temp_directory));

    SamplesPrintEndMessage( );

    return;
}

bool DoCPUvsGPURealSpaceResize(wxString hiv_image_80x80x1_filename, wxString temp_directory) {

    bool passed     = true;
    bool all_passed = true;
    int  logical_x_start;
    int  logical_y_start;

    wxString tmp_img_filename = temp_directory + "/tmp1.mrc";

    MRCFile input_file(hiv_image_80x80x1_filename.ToStdString( ), false);
    MRCFile output_file(tmp_img_filename.ToStdString( ), true);

    all_passed = passed ? all_passed : false;

    // This really is more like basic i/o, but since we compare to cpu values, i guess this is okay here.
    SamplesBeginTest("Read onto GPU and copy to new CPU image", passed);

    Image    cpu_image;
    Image    gpu_host_image;
    GpuImage gpu_image;

    cpu_image.ReadSlice(&input_file, 1);
    gpu_host_image.ReadSlice(&input_file, 1);

    // Record the starting sizes
    logical_x_start = cpu_image.logical_x_dimension;
    logical_y_start = cpu_image.logical_y_dimension;

    gpu_image.Init(gpu_host_image);
    gpu_image.CopyHostToDevice(gpu_host_image);

    // Copy back to a new image, blocking until complete and freeing the gpu memory
    Image new_cpu_image_from_gpu = gpu_image.CopyDeviceToNewHost(true, true);

    passed = CompareRealValues(cpu_image, new_cpu_image_from_gpu);

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    // Real space size reduction
    SamplesBeginTest("Real Space size reduce", passed);

    // Get a clean copy
    cpu_image.ReadSlice(&input_file, 1);
    cpu_image.Resize(logical_x_start / 2, logical_y_start / 2, 1, 0);

    gpu_host_image.ReadSlice(&input_file, 1);
    // Deallocated in CopyDeviceToNewHost call
    gpu_image.Init(gpu_host_image);

    gpu_image.CopyHostToDevice(gpu_host_image);

    gpu_image.Resize(logical_x_start / 2, logical_y_start / 2, 1, 0.f);

    Image decreased_host_image = gpu_image.CopyDeviceToNewHost(true, true);

    passed     = CompareRealValues(cpu_image, decreased_host_image);
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    // Real space size reduction
    SamplesBeginTest("Real Space size increase", passed);

    // Get a clean copy
    cpu_image.ReadSlice(&input_file, 1);
    cpu_image.Resize(logical_x_start * 2, logical_y_start * 2, 1, 0);

    gpu_host_image.ReadSlice(&input_file, 1);
    // Deallocated in CopyDeviceToNewHost call
    gpu_image.Init(gpu_host_image);

    gpu_image.CopyHostToDevice(gpu_host_image);

    gpu_image.Resize(logical_x_start * 2, logical_y_start * 2, 1, 0.f);

    Image increased_host_image = gpu_image.CopyDeviceToNewHost(true, true);

    passed     = CompareRealValues(cpu_image, increased_host_image);
    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    return all_passed;
}

bool DoCPUvsGPUFourierResize(wxString hiv_image_80x80x1_filename, wxString temp_directory) {
    bool passed     = true;
    bool all_passed = true;

    MRCFile input_file(hiv_image_80x80x1_filename.ToStdString( ), false);

    wxString tmp_img_filename = temp_directory + "/tmp1.mrc";

    MRCFile output_file(tmp_img_filename.ToStdString( ), true);

    SamplesBeginTest("Fourier Crop CPU and GPU images", passed);
    Image cpu_image;
    cpu_image.ReadSlice(&input_file, 1);

    // resize test

    cpu_image.ForwardFFT( );
    cpu_image.Resize(40, 40, 1, 0);
    cpu_image.BackwardFFT( );

    Image gpu_host_image;
    gpu_host_image.ReadSlice(&input_file, 1);

    GpuImage device_image(gpu_host_image);
    device_image.CopyHostToDevice(gpu_host_image);

    device_image.ForwardFFT( );
    device_image.Resize(40, 40, 1, 0);

    device_image.BackwardFFT( );
    Image resized_host_image = device_image.CopyDeviceToNewHost(true, true);

    passed = CompareRealValues(cpu_image, resized_host_image);

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    return all_passed;
}
