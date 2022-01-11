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


#ifdef ENABLEGPU
  #include "../../../gpu/gpu_core_headers.h"
#else
  #include "../../../core/core_headers.h"
#endif

#include "../common/common.h"
#include "resize_comparison.h"


bool DoCPUvsGPUResize(wxString hiv_image_80x80x1_filename, wxString temp_directory)
{
  bool passed;
  bool all_passed = true;

  SamplesPrintTestStartMessage("Starting CPU vs GPU resize tests:", false);

  all_passed = all_passed && DoCPUvsGPURealSpaceResize(hiv_image_80x80x1_filename, temp_directory);
  all_passed = all_passed && DoCPUvsGPUFourierResize(hiv_image_80x80x1_filename, temp_directory);

  SamplesBeginTest("CPU vs GPU overall", passed);
  SamplesPrintResult(all_passed, __LINE__);
  wxPrintf("\n\n");
  return all_passed;
}

bool DoCPUvsGPURealSpaceResize(wxString hiv_image_80x80x1_filename, wxString temp_directory)
{

  bool passed = true;
  bool all_passed = true;

  MRCFile input_file(hiv_image_80x80x1_filename.ToStdString(), false);

  wxString temp_filename = temp_directory + "/tmp1.mrc";

  MRCFile output_file(temp_filename.ToStdString(), false);

  all_passed = all_passed && passed;
  SamplesBeginTest("Read onto GPU and copy to new CPU image", passed);
  Image cpu_image;
  cpu_image.ReadSlice(&input_file, 1);

  Image from_gpu_image;
  from_gpu_image.ReadSlice(&input_file, 1);
  GpuImage gpu_image(from_gpu_image);
  gpu_image.CopyHostToDevice();
  Image new_cpu_image_from_gpu = gpu_image.CopyDeviceToNewHost(true, true);
  
  passed = ProperCompareRealValues(cpu_image, new_cpu_image_from_gpu);
  
  all_passed = all_passed && passed;
  SamplesTestResult(passed);


  SamplesBeginTest("Real Space resize CPU and GPU images", passed);
  cpu_image.Resize(40, 40, 1, 0);

  Image host_image;

  host_image.ReadSlice(&input_file, 1);

  GpuImage device_image(host_image);

  device_image.CopyHostToDevice();

  device_image.Resize(40, 40, 1, 0);

  Image resized_host_image = device_image.CopyDeviceToNewHost(true, true);

  passed = ProperCompareRealValues(cpu_image, resized_host_image);
  all_passed = all_passed && passed;
  SamplesTestResult(passed);

  return all_passed;
}

bool DoCPUvsGPUFourierResize(wxString hiv_image_80x80x1_filename, wxString temp_directory) 
{
  bool passed = true;
  bool all_passed = true;


  MRCFile input_file(hiv_image_80x80x1_filename.ToStdString(), false);

  wxString temp_filename = temp_directory + "/tmp1.mrc";

  MRCFile output_file(temp_filename.ToStdString(), false);

  SamplesBeginTest("Fourier Crop CPU and GPU images", passed);
  Image cpu_image;
  cpu_image.ReadSlice(&input_file, 1);

  // resize test

  cpu_image.ForwardFFT();
  cpu_image.Resize(40, 40, 1, 0);
  cpu_image.BackwardFFT();

  Image host_image;
  host_image.ReadSlice(&input_file, 1);

  GpuImage device_image(host_image);
  device_image.CopyHostToDevice();
  
  device_image.ForwardFFT();
  device_image.Resize(40, 40, 1, 0);

  device_image.BackwardFFT();
  Image resized_host_image = device_image.CopyDeviceToNewHost(true, true);
  
  passed = ProperCompareRealValues(cpu_image, resized_host_image);

  all_passed = all_passed && passed;
  SamplesTestResult(passed);

  return all_passed;

}
