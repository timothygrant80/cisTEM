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

#include <iostream>
#include <string>
#include <wx/string.h>
#include <wx/wxcrtvararg.h>

void print2DArray(Image &image) {
  int i = 0;
  wxPrintf("Image real space data:\n");
	for (int z = 0; z < image.logical_z_dimension; z++)
	{
	  for (int y = 0; y < image.logical_y_dimension; y++)
	  {
		for (int x = 0; x < image.logical_x_dimension; x++)
		{
      	wxPrintf("%f\t", image.real_values[i]);
		 
		  i++;
		}
    wxPrintf("\n");
		i += image.padding_jump_value;
	  }
    wxPrintf("\n");
	}
}

void PrintArray(float *p, int maxLoops = 10)
{
  wxPrintf("Starting loop through array.\n");

  if (p == nullptr)
  {
	wxPrintf("pointer is null, aborting.\n");
	return;
  }
  for (int i = 0; i < maxLoops; i++)
  {
	wxPrintf("%s \n", std::to_string(i));
	// wxPrintf(" %s\n", *arr);
	// std::cout<< *arr <<" ";
	std::cout << *(p + i) << std::endl;

	p++;
  }
  wxPrintf("Loop done.\n");
}

// bool IsPointerNull(float *p) {
//     if (p == nullptr) {
//         wxPrintf("pointer is null!\n");
//         return true;
//     }
//     wxPrintf("pointer is valid!\n");
//     return false;
// }

bool ProperCompareRealValues(Image &first_image, Image &second_image,  float epsilon = 1e-5)
{
  bool passed;
  if (first_image.real_memory_allocated != second_image.real_memory_allocated)
  {

    // wxPrintf(" real_memory_allocated values are not the same. [Failed]\n");
    // wxPrintf(" cpu_image.real_memory_allocated ==  %s\n",
    //         std::to_string(first_image.real_memory_allocated));
    // wxPrintf(" resized_host_image.real_memory_allocated ==  %s\n",
    //         std::to_string(second_image.real_memory_allocated));

    passed = false;
  }
  else
  {

  // print2DArray(first_image);
  // print2DArray(second_image);

	int total_pixels = 0;
	int unequal_pixels = 0;
	// wxPrintf(" real_memory_allocated values are the same. (%s) Starting loop\n", std::to_string(first_image.real_memory_allocated));
	// wxPrintf(" cpu_image.real_values[0] == (%s)\n", std::to_string(first_image.real_values[0]));
	// wxPrintf(" resized_host_image.real_values[0] == (%s)\n", std::to_string(second_image.real_values[0]));

	int i = 0;
	for (int z = 0; z < first_image.logical_z_dimension; z++)
	{
	  for (int y = 0; y < first_image.logical_y_dimension; y++)
	  {
		for (int x = 0; x < first_image.logical_x_dimension; x++)
		{
		  if (std::fabs(first_image.real_values[i] - second_image.real_values[i]) > epsilon) {
            unequal_pixels++;
            if (unequal_pixels < 5) {
              wxPrintf(" Unequal pixels at position: %s, value 1: %s, value 2: %s.\n", std::to_string(i),
                                                                                        std::to_string(first_image.real_values[i]),
                                                                                        std::to_string(second_image.real_values[i]));
            }
              //wxPrintf(" Diff: %f\n", first_image.real_values[i]-second_image.real_values[i]);
        }
		  total_pixels++;
		  i++;
		}
		i += first_image.padding_jump_value;
	  }
	}

	passed = true;
	if (unequal_pixels > 0)
	{
	  int unequal_percent = 100 * (unequal_pixels / total_pixels);
	  wxString err_message = std::to_string(unequal_pixels) + " out of " +
	                         std::to_string(total_pixels) + "(" +
	                         std::to_string(unequal_percent) +
	                         "%) of pixels are not equal between CPU and GPU "
	                         "images after resizing. [Failed]\n";
	  wxPrintf(err_message);

	  wxPrintf("Padding values 1: %s, and 2: %s\n",
	           std::to_string(first_image.padding_jump_value),
	           std::to_string(second_image.padding_jump_value));
	  passed = false;
	}
  }

// TODO make this match what is done in disk_io.cpp, or better move the print test to classes.
  if (passed)
  {
	  wxPrintf("\n\tCpu/Gpu images are ~equal after resizing (epsilon= %f).  [Success]\n", epsilon);
  }
  else
  {
	  wxPrintf("\tCpu/Gpu images are not equal after resizing (epsilon= %f). [Failed]\n", epsilon);
  }
  return passed;
}


bool DoCPUvsGPUResize(wxString hiv_image_80x80x1_filename,
                      wxString temp_directory)
{

  bool passed = true;
  bool print_verbose = false;

  MRCFile input_file(hiv_image_80x80x1_filename.ToStdString(), false);

  wxString temp_filename = temp_directory + "/tmp1.mrc";

  MRCFile output_file(temp_filename.ToStdString(), false);

  if (print_verbose) wxPrintf(" Starting CPU vs GPU compare (not resizing).\n");
  Image cpu_image;
  cpu_image.ReadSlice(&input_file, 1);

  Image from_gpu_image;
  from_gpu_image.ReadSlice(&input_file, 1);
  GpuImage gpu_image(from_gpu_image);
  gpu_image.CopyHostToDevice();
  Image new_cpu_image_from_gpu = gpu_image.CopyDeviceToNewHost(true, true);
  passed = ProperCompareRealValues(cpu_image, new_cpu_image_from_gpu);
  if (print_verbose) wxPrintf(" CPU vs GPU compare (not resizing) ended.\n");


    // resize test
  if (print_verbose) wxPrintf(" Starting CPU resize.\n");
  cpu_image.Resize(40, 40, 1, 0);

  Image host_image;

  host_image.ReadSlice(&input_file, 1);

  GpuImage device_image(host_image);

  if (print_verbose) wxPrintf(" GPU image initiated from host image.\n");
  device_image.CopyHostToDevice();

  if (print_verbose) wxPrintf(" Image copied from host to device.\n");

  device_image.Resize(40, 40, 1, 0);

  if (print_verbose) wxPrintf(" Image resized.\n");

  Image resized_host_image = device_image.CopyDeviceToNewHost(true, true);

  if (print_verbose) wxPrintf(" Image copied from device to host.\n");

  passed = ProperCompareRealValues(cpu_image, resized_host_image);

  if (print_verbose) wxPrintf(" End.\n");
  return passed;
}

bool DoGPUComplexResize(wxString hiv_image_80x80x1_filename, wxString temp_directory) 
{
  bool passed = true;
  int z_cpu, z_gpu;
  bool print_verbose = false;

  MRCFile input_file(hiv_image_80x80x1_filename.ToStdString(), false);


  wxString temp_filename = temp_directory + "/tmp1.mrc";

  MRCFile output_file(temp_filename.ToStdString(), false);

  if (print_verbose) wxPrintf("\tStarting CPU vs GPU compare (not resizing).\n");
  Image cpu_image;
  cpu_image.ReadSlice(&input_file, 1);
  // Image from_gpu_image;
  // from_gpu_image.ReadSlice(&input_file, 1);
  // GpuImage gpu_image(from_gpu_image);
  // gpu_image.CopyHostToDevice();
  // Image new_cpu_image_from_gpu = gpu_image.CopyDeviceToNewHost(true, true);
  // passed = ProperCompareRealValues(cpu_image, new_cpu_image_from_gpu);
  // wxPrintf(" CPU vs GPU compare (not resizing) ended.\n");



    // resize test

  if (print_verbose) wxPrintf(" Transforming CPU image to Fourier space.\n");
  cpu_image.ForwardFFT();

  if (print_verbose) wxPrintf(" Resizing CPU Image.\n");
  cpu_image.Resize(40, 40, 1, 0);
  cpu_image.BackwardFFT();
  // z_cpu = cpu_image.logical_z_dimension;
  // wxPrintf("%i\n", z_cpu);
  Image host_image;

  host_image.ReadSlice(&input_file, 1);
//   z_gpu = host_image.logical_z_dimension;
// wxPrintf("%i\n", z_gpu);
  GpuImage device_image(host_image);

  if (print_verbose) wxPrintf(" GPU image initiated from host image.\n");
  device_image.CopyHostToDevice();
  
  if (print_verbose)wxPrintf(" Transforming to Fourier space.\n");
  device_image.ForwardFFT();


  if (print_verbose) wxPrintf(" Resizing image.\n");
  device_image.Resize(40, 40, 1, 0);


  if (print_verbose) wxPrintf(" Transforming GPU image to real space.\n");
  device_image.BackwardFFT();

  if (print_verbose) wxPrintf(" Copying image from device to host.\n");
  Image resized_host_image = device_image.CopyDeviceToNewHost(true, true);
// z_gpu = resized_host_image.logical_z_dimension;
// wxPrintf("%i\n", z_gpu);
  
  passed = ProperCompareRealValues(cpu_image, resized_host_image);

  if (print_verbose) wxPrintf(" Exporting images.\n");
  // wxPrintf("resized_host_image.is_in.real_space: %d\n", resized_host_image.is_in_real_space);
  // wxPrintf("resized_host_image.is_in.is_in_memory: %d\n", resized_host_image.is_in_memory);

  // std::string name1 = std::tmpnam(nullptr);
  // wxPrintf("Tmp outputs are at %s\n", name1.c_str());
  
  // cpu_image.QuickAndDirtyWriteSlice( name1 + "_cpu.mrc", 1, true, 1.0);
  // resized_host_image.QuickAndDirtyWriteSlice( name1 + "_gpu.mrc", 1, true, 1.0);


  if (print_verbose) wxPrintf(" End.\n");
  return passed;

}