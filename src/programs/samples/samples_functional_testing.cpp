#include "../../core/core_headers.h"
#include <cstdio>
#include <list>



#include "classes/TestFile.cpp"
#include "classes/EmbeddedTestFile.cpp"
#include "classes/NumericTestFile.cpp"
#include "classes/TestResult.hpp"

#include "0_Simple/disk_io_image.cpp"


// #define PrintResult(result)	PrintResultSlave(result, __LINE__);
// #define FailTest {if (test_has_passed == true) PrintResultSlave(false,
// __LINE__); test_has_passed = false;}//#include
// "samples_functional_testing.hpp"

// TODO //
// TEST 3D's
// ODD Sized IMAGES
// NON SQUARE IMAGES

#include "../console_test/hiv_image_80x80x1.cpp"
#include "../console_test/hiv_images_shift_noise_80x80x10.cpp"
#include "../console_test/sine_128x128x1.cpp"

class SamplesTestingApp : public wxAppConsole
{

	// constructor: set file names and temp folder, write embbeded files to harddrive.
  wxString temp_directory = wxFileName::GetHomeDir();
  wxString hiv_image_80x80x1_filename = 	temp_directory + "/hiv_image_80x80x1.mrc";
  wxString hiv_images_80x80x10_filename = 	temp_directory + "/hiv_images_shift_noise_80x80x10.mrc";  
  wxString sine_wave_128x128x1_filename = 	temp_directory + "/sine_wave_128x128x1.mrc";  
  wxString numeric_text_filename = 			temp_directory + "/numbers.num";

	std::list<TestFile*> testFiles;
	
	public:
		SamplesTestingApp();
		~SamplesTestingApp();
    	bool OnInit();
		void WriteFiles();
		//void WriteEmbeddedArray(const char *filename, const unsigned char *array, long length);
		//void WriteNumericTextFile(const char *filename);
   
		bool DoCalculation();
		void DoInteractiveUserInput();

};


IMPLEMENT_APP(SamplesTestingApp);

bool SamplesTestingApp::OnInit() {

  wxPrintf("Starting samples testing.\n\n");

  DoDiskIOImageTests(hiv_images_80x80x10_filename, temp_directory);

  wxPrintf("\n\nSamples testing done.\n");
  return false;
}



void SamplesTestingApp::WriteFiles() {

  wxPrintf("\nWriting out embedded test files to '%s'...\n", temp_directory);
  fflush(stdout);

  testFiles.push_back(new EmbeddedTestFile(hiv_image_80x80x1_filename, hiv_image_80x80x1_array, sizeof(hiv_image_80x80x1_array)));
  testFiles.push_back(new EmbeddedTestFile(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array)));
  testFiles.push_back(new EmbeddedTestFile(sine_wave_128x128x1_filename, sine_128x128x1_array, sizeof(sine_128x128x1_array)));
  testFiles.push_back(new NumericTestFile(numeric_text_filename));

  wxPrintf("done writing files!\n");
}


SamplesTestingApp::SamplesTestingApp() {

  WriteFiles();
}

SamplesTestingApp::~SamplesTestingApp() {
	// destructor: remove all files written to harddrive.
	wxPrintf("Removing test files from '%s'... \n", temp_directory);

  for(auto &it:testFiles) delete it;
  testFiles.clear();
  // removeFile(hiv_image_80x80x1_filename.mb_str());
  // removeFile(hiv_images_80x80x10_filename.mb_str());
  // removeFile(sine_wave_128x128x1_filename.mb_str());
  // removeFile(numeric_text_filename.mb_str());
  wxPrintf("done!\n");
}