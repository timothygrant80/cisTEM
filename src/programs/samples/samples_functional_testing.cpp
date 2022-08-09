//#include <wx/wx.h>

// #include "common/samples_headers.h"
#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#else
#include "../../core/core_headers.h"
#endif

// Helper classes for interacting with the test data.
#include "common/common.h"

// Sample functions
#include "0_simple/simple.h"
#include "1_cpu_gpu_comparison/cpu_gpu_comparison.h"

// Test data
#include "../console_test/hiv_image_80x80x1.cpp"
#include "../console_test/hiv_images_shift_noise_80x80x10.cpp"
#include "../console_test/sine_128x128x1.cpp"

#include "samples_functional_testing.h"

IMPLEMENT_APP(SamplesTestingApp)

bool SamplesTestingApp::DoCalculation( ) {

    // This is returned and for automated testing allows you to only visualize the overall results,
    // and if false, using github actions you can expand to see the full results.
    bool all_tests_passed = true;

    SamplesPrintTestStartMessage("Starting samples testing", true);

    all_tests_passed = all_tests_passed && DoDiskIOImageTests(hiv_images_80x80x10_filename, temp_directory);

#ifdef ENABLEGPU
    all_tests_passed = all_tests_passed && DoCPUvsGPUResize(hiv_image_80x80x1_filename, temp_directory);
#else
    wxPrintf("GPU support disabled. skipping GPU tests.\n");
#endif

    SamplesPrintTestStartMessage("Samples testing done!", true);
    ProgramSpecificCleanUp( );

    return true;
}

void SamplesTestingApp::DoInteractiveUserInput( ) {
    // noop (otherwise this triggers an error in MyApp.)
}

void SamplesTestingApp::ProgramSpecificInit( ) {
    // constructor: set file names and temp folder, write embbeded files to harddrive.
    temp_directory = wxFileName::GetHomeDir( );

    hiv_image_80x80x1_filename   = temp_directory + "/hiv_image_80x80x1.mrc";
    hiv_images_80x80x10_filename = temp_directory + "/hiv_images_shift_noise_80x80x10.mrc";
    sine_wave_128x128x1_filename = temp_directory + "/sine_wave_128x128x1.mrc";
    numeric_text_filename        = temp_directory + "/numbers.num";

    WriteFiles( );
}

// void SamplesTestingApp::ProgramSpecificCleanup()
// {
// 	// destructor: remove all files written to harddrive.
// 	wxPrintf("\nRemoving test files from '%s'... \n", temp_directory);

//   for(auto &it:testFiles) delete it;
//   testFiles.clear();
//   // removeFile(hiv_image_80x80x1_filename.mb_str());
//   // removeFile(hiv_images_80x80x10_filename.mb_str());
//   // removeFile(sine_wave_128x128x1_filename.mb_str());
//   // removeFile(numeric_text_filename.mb_str());
//   wxPrintf("done!\n");
// }

void SamplesTestingApp::WriteFiles( ) {

    /* Write out the test files in mrc (images) or txt (numeric txt) */
    wxPrintf("\nWriting out embedded test files to '%s'...\n\n", temp_directory);
    fflush(stdout);

    file_tracker.testFiles.push_back(new EmbeddedTestFile(hiv_image_80x80x1_filename, hiv_image_80x80x1_array, sizeof(hiv_image_80x80x1_array)));
    file_tracker.testFiles.push_back(new EmbeddedTestFile(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array)));
    file_tracker.testFiles.push_back(new EmbeddedTestFile(sine_wave_128x128x1_filename, sine_128x128x1_array, sizeof(sine_128x128x1_array)));
    file_tracker.testFiles.push_back(new NumericTestFile(numeric_text_filename));

    wxPrintf("\ndone writing files!\n\n\n");
}
