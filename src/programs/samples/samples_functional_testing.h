
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This block is just from console_test.cpp - Evaluate what is actually needed after image selections have been made. TODO
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//#include <wx/wx.h>
//#include <wx/app.h>
//#include <wx/cmdline.h>
//#include <cstdio>
//#include "wx/socket.h"
//
//#include "../../core/core_headers.h"
//
//// embedded images..
//
//#include "hiv_image_80x80x1.cpp"
//#include "hiv_images_shift_noise_80x80x10.cpp"
//#include "sine_128x128x1.cpp"
//
//#define PrintResult(result)	PrintResultWorker(result, __LINE__);
//#define FailTest {if (test_has_passed == true) PrintResultWorker(false, __LINE__); test_has_passed = false;}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <list>




class SamplesTestingApp : public wxAppConsole
{
	wxString temp_directory;
	wxString hiv_image_80x80x1_filename;
	wxString hiv_images_80x80x10_filename;
	wxString sine_wave_128x128x1_filename;
	wxString numeric_text_filename;
	std::list<TestFile*> testFiles;
	
	public:
		SamplesTestingApp();
		~SamplesTestingApp();
        virtual bool OnInit();
		void WriteFiles();
		//void WriteEmbeddedArray(const char *filename, const unsigned char *array, long length);
		//void WriteNumericTextFile(const char *filename);
   
		bool DoCalculation();
		void DoInteractiveUserInput();

};

