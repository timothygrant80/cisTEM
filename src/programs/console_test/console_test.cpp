#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include "wx/socket.h"

#include "../../core/core_headers.h"

// embedded images..

#include "hiv_image_80x80x1.cpp"
#include "hiv_images_shift_noise_80x80x10.cpp"
#include "sine_128x128x1.cpp"

#define PrintResult(result)	PrintResultSlave(result, __LINE__);


#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_UNDERLINE     "\e[4m"
#define ANSI_UNDERLINE_OFF "\e[24m"
#define ANSI_BLINK_SLOW "\x1b[5m"
#define ANSI_BLINK_OFF "\x1b[25m"







class
MyTestApp : public wxAppConsole
{
	wxString hiv_image_80x80x1_filename;
	wxString hiv_images_80x80x10_filename;
	wxString sine_wave_128x128x1_filename;

	public:
		virtual bool OnInit();

		void TestMRCFunctions();
		void TestFFTFunctions();
		void TestScalingAndSizingFunctions();

		void BeginTest(const char *test_name);
		void PrintTitle(const char *title);
		void PrintResultSlave( bool passed, int line);
		void WriteEmbeddedFiles();
		void WriteEmbeddedArray(const char *filename, const unsigned char *array, long length);
};


IMPLEMENT_APP(MyTestApp)

bool MyTestApp::OnInit()
{
	wxPrintf("\n\n\n     **   ");
	wxPrintf(ANSI_UNDERLINE "ProjectX Library Tester" ANSI_UNDERLINE_OFF);
	wxPrintf("   **\n");

	//wxPrintf("")

	WriteEmbeddedFiles();
	wxPrintf("\n");

	// Do tests..

	//PrintTitle("Basic I/O Functions");

	TestMRCFunctions();
	TestFFTFunctions();
	TestScalingAndSizingFunctions();


	wxPrintf("\n\n\n");
	return false;
}

void MyTestApp::TestScalingAndSizingFunctions()
{
	bool test_passed = true;

	BeginTest("Image::ClipInto");

	MRCFile input_file(hiv_images_80x80x10_filename.ToStdString(), false);
	Image test_image;
	Image clipped_image;

	// test real space clipping..

	clipped_image.Allocate(160,160,1);

	test_image.ReadSlice(&input_file, 1);
	test_image.ClipInto(&clipped_image, 0);

	//wxPrintf("value = %f\n", clipped_image.ReturnRealPixelValue(119,119));
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), -0.340068) == false) test_passed = false;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(80,80, 0), 1.819805) == false) test_passed = false;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(119,119, 0), 0.637069) == false) test_passed = false;


	// test Fourier space clipping

	test_image.ForwardFFT();
	test_image.ClipInto(&clipped_image, 0);
	clipped_image.BackwardFFT();
	MRCFile output("test.mrc", true);
	clipped_image.WriteSlice(&output, 1);


	PrintResult(test_passed);


}

void MyTestApp::TestMRCFunctions()
{
	bool test_passed = true;

	BeginTest("MRCFile::OpenFile");

	MRCFile input_file(hiv_images_80x80x10_filename.ToStdString(), false);

	// check dimensions..

	if (input_file.ReturnNumberOfSlices() != 10) test_passed = false;
	if (input_file.ReturnXSize() != 80) test_passed = false;
	if (input_file.ReturnYSize() != 80) test_passed = false;
	if (input_file.ReturnZSize() != 10) test_passed = false;

	PrintResult(test_passed);

	test_passed = true;
	BeginTest("Image::ReadSlice");

	Image test_image;
	test_image.ReadSlice(&input_file, 1);

	// check dimensions and type

	if (test_image.is_in_real_space == false) test_passed = false;
	if (test_image.logical_x_dimension != 80) test_passed = false;
	if (test_image.logical_y_dimension != 80) test_passed = false;
	if (test_image.logical_z_dimension != 1) test_passed = false;

	// check first and last pixel...

	test_passed = DoublesAreAlmostTheSame(test_image.real_values[0], -0.340068);
	test_passed = DoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069);


	PrintResult(true);
}

void MyTestApp::TestFFTFunctions()
{
	bool test_passed = true;
	long counter;

	BeginTest("Image::ForwardFFT");

	MRCFile input_file(sine_wave_128x128x1_filename.ToStdString(), false);

	// make an image that is all 1..

	Image test_image;
	test_image.Allocate(64,64,1);
	test_image.SetToConstant(1);

	// ForwardFFT

	test_image.ForwardFFT();

	// first pixel should be 1,0

	if (DoublesAreAlmostTheSame(creal(test_image.complex_values[0]), 1) == false || DoublesAreAlmostTheSame(cimag(test_image.complex_values[0]), 0) == false) test_passed = false;

	// if we set this to 0,0 - all remaining pixels should now be 0

	test_image.complex_values[0] = 0 + 0 * I;

	for (counter = 0; counter < test_image.real_memory_allocated / 2; counter++)
	{
		if (DoublesAreAlmostTheSame(creal(test_image.complex_values[counter]), 0) == false || DoublesAreAlmostTheSame(cimag(test_image.complex_values[counter]), 0) == false) test_passed = false;
	}

	// sine wave

	test_image.ReadSlice(&input_file, 1);
	test_image.ForwardFFT();

	// now one pixel should be set, and the rest should be 0..

	if (DoublesAreAlmostTheSame(creal(test_image.complex_values[20]), 0) == false || DoublesAreAlmostTheSame(cimag(test_image.complex_values[20]), -5) == false)
	{
		test_passed = false;
	}

	// set it to 0, then everything should be zero..

	test_image.complex_values[20] = 0 + 0 * I;

	for (counter = 0; counter < test_image.real_memory_allocated / 2; counter++)
	{
		if (creal(test_image.complex_values[counter]) >  0.000001 || cimag(test_image.complex_values[counter]) > 0.000001)
		{
			test_passed = false;
		}
	}

	PrintResult(test_passed);

	// Backward FFT

	test_passed = true;
	BeginTest("Image::BackwardFFT");

	test_image.Allocate(64,64,1, false);
	test_image.SetToConstant(0.0);
	test_image.complex_values[0] = 1 + 0 * I;
	test_image.BackwardFFT();
	//test_image.RemoveFFTWPadding();

	/*
	for (counter = 0; counter < test_image.logical_x_dimension * test_image.logical_y_dimension; counter++)
	{
		if (DoublesAreAlmostTheSame(test_image.real_values[counter], 1.0) == false)
		{
			test_passed = false;
			wxPrintf("pixel %li = %f\n", counter, test_image.real_values[counter]);
		}

	}*/

	//test_image.AddFFTWPadding();

	PrintResult(test_passed);

}



void MyTestApp::BeginTest(const char *test_name)
{
	int length = strlen(test_name);
	int blank_space = 40 - length;
	wxPrintf("Testing %s ", test_name);

	for (int counter = 0; counter < blank_space; counter++)
	{
		wxPrintf(" ");
	}

	wxPrintf(": ");
}

void MyTestApp::PrintResultSlave(bool passed, int line)
{

	if (passed == true)
	{
		wxPrintf(ANSI_COLOR_GREEN "PASSED!" ANSI_COLOR_RESET);
	}
	else
	{
		wxPrintf(ANSI_COLOR_RED ANSI_BLINK_SLOW "FAILED! (Line : %i)" ANSI_BLINK_OFF ANSI_COLOR_RESET, line);
	}

	wxPrintf("\n");
}

void MyTestApp::PrintTitle(const char *title)
{
	wxPrintf("\n");
	wxPrintf(ANSI_UNDERLINE "%s" ANSI_UNDERLINE_OFF, title);
	wxPrintf("\n\n");
}

void MyTestApp::WriteEmbeddedFiles()
{
	wxString temp_directory = wxFileName::GetTempDir();
	wxPrintf("\nWriting out embedded test files to '%s'...", temp_directory);
	fflush(stdout);

	hiv_image_80x80x1_filename = temp_directory + "/hiv_image_80x80x1.mrc";
	hiv_images_80x80x10_filename = temp_directory + "/hiv_images_shift_noise_80x80x10.mrc";
	sine_wave_128x128x1_filename = temp_directory + "/sine_wave_128x128x1.mrc";

	WriteEmbeddedArray(hiv_image_80x80x1_filename, hiv_image_80x80x1_array, sizeof(hiv_image_80x80x1_array));
	WriteEmbeddedArray(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
	WriteEmbeddedArray(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
	WriteEmbeddedArray(sine_wave_128x128x1_filename, sine_128x128x1_array, sizeof(sine_128x128x1_array));

	wxPrintf("done!\n");


}

void MyTestApp::WriteEmbeddedArray(const char *filename, const unsigned char *array, long length)
{

	FILE *output_file = NULL;
	output_file = fopen(filename, "wb+");

	if (output_file == NULL)
	{
		wxPrintf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n", filename);
		abort();

	}

	 fwrite (array , sizeof(unsigned char), length, output_file);

	 fclose(output_file);


}


