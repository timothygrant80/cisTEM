//#include <wx/wx.h>

#include "../../core/core_headers.h"
#include "0_Simple/disk_io_image.cpp"
#ifdef ENABLEGPU
#include "1_GPU_comparison/cpu_vs_gpu.cpp"
#endif

#include "classes/TestFile.cpp"
#include "classes/EmbeddedTestFile.cpp"
#include "classes/NumericTestFile.cpp"
#include "samples_functional_testing.h"
// #define PrintResult(result)	PrintResultWorker(result, __LINE__);
// #define FailTest {if (test_has_passed == true) PrintResultWorker(false,
// __LINE__); test_has_passed = false;}//#include
// "samples_functional_testing.hpp"

// TODO //
// TEST 3D's
// ODD Sized IMAGES
// NON SQUARE IMAGES

#include "../console_test/hiv_image_80x80x1.cpp"
#include "../console_test/hiv_images_shift_noise_80x80x10.cpp"
#include "../console_test/sine_128x128x1.cpp"

IMPLEMENT_APP(SamplesTestingApp);

bool SamplesTestingApp::OnInit() {

  wxPrintf("Starting samples testing: \n\n");
  
  DoDiskIOImageTests(hiv_images_80x80x10_filename, temp_directory);

  #ifdef ENABLEGPU
  DoGPUComplexResize(hiv_image_80x80x1_filename, temp_directory);
    //DoCPUvsGPUResize(hiv_image_80x80x1_filename, temp_directory);
  #else
    wxPrintf("GPU support disabled. skipping GPU tests.\n");
  #endif

  wxPrintf("\nSamples testing done!\n\n");
  return false;
}


void SamplesTestingApp::WriteFiles() {

  wxPrintf("\nWriting out embedded test files to '%s'...\n", temp_directory);
  fflush(stdout);

  testFiles.push_back(new EmbeddedTestFile(hiv_image_80x80x1_filename, hiv_image_80x80x1_array, sizeof(hiv_image_80x80x1_array)));
  testFiles.push_back(new EmbeddedTestFile(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array)));
  testFiles.push_back(new EmbeddedTestFile(sine_wave_128x128x1_filename, sine_128x128x1_array, sizeof(sine_128x128x1_array)));
  testFiles.push_back(new NumericTestFile(numeric_text_filename));

  wxPrintf("done writing files!\n\n");
}


SamplesTestingApp::SamplesTestingApp() {
	// constructor: set file names and temp folder, write embbeded files to harddrive.
  temp_directory = wxFileName::GetHomeDir();

  hiv_image_80x80x1_filename    = temp_directory + "/hiv_image_80x80x1.mrc";
  hiv_images_80x80x10_filename  = temp_directory + "/hiv_images_shift_noise_80x80x10.mrc";
  sine_wave_128x128x1_filename  = temp_directory + "/sine_wave_128x128x1.mrc";
  numeric_text_filename         = temp_directory + "/numbers.num";

  WriteFiles();
}

SamplesTestingApp::~SamplesTestingApp() {
	// destructor: remove all files written to harddrive.
	wxPrintf("\nRemoving test files from '%s'... \n", temp_directory);

  for(auto &it:testFiles) delete it;
  testFiles.clear();
  // removeFile(hiv_image_80x80x1_filename.mb_str());
  // removeFile(hiv_images_80x80x10_filename.mb_str());
  // removeFile(sine_wave_128x128x1_filename.mb_str());
  // removeFile(numeric_text_filename.mb_str());
  wxPrintf("done!\n");
}
