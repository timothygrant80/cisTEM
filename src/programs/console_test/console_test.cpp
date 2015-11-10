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
#define FailTest {if (test_has_passed == true) PrintResultSlave(false, __LINE__); test_has_passed = false;}





// TODO //
// TEST 3D's
// ODD Sized IMAGES
// NON SQUARE IMAGES




class
MyTestApp : public wxAppConsole
{
	wxString hiv_image_80x80x1_filename;
	wxString hiv_images_80x80x10_filename;
	wxString sine_wave_128x128x1_filename;

	public:
		virtual bool OnInit();

		bool test_has_passed;

		void TestMRCFunctions();
		void TestFFTFunctions();
		void TestScalingAndSizingFunctions();
		void TestFilterFunctions();
		void TestAlignmentFunctions();
		void TestImageArithmeticFunctions();

		void BeginTest(const char *test_name);
		void EndTest();
		void PrintTitle(const char *title);
		void PrintResultSlave( bool passed, int line);
		void WriteEmbeddedFiles();
		void WriteEmbeddedArray(const char *filename, const unsigned char *array, long length);
};


IMPLEMENT_APP(MyTestApp)

bool MyTestApp::OnInit()
{
	wxPrintf("\n\n\n     **   ");
	if (OutputIsAtTerminal() == true) wxPrintf(ANSI_UNDERLINE "ProjectX Library Tester" ANSI_UNDERLINE_OFF);
	else wxPrintf("ProjectX Library Tester");
	wxPrintf("   **\n");

	//wxPrintf("")

	WriteEmbeddedFiles();
	wxPrintf("\n");

	// Do tests..

	//PrintTitle("Basic I/O Functions");

	TestMRCFunctions();
	TestImageArithmeticFunctions();
	TestFFTFunctions();
	TestScalingAndSizingFunctions();
	TestFilterFunctions();
	TestAlignmentFunctions();


	wxPrintf("\n\n\n");
	return false;
}

void MyTestApp::TestImageArithmeticFunctions()
{
	// AddImage
	BeginTest("Image::AddImage");

	Image test_image;
	Image ref_image;
	Peak my_peak;

	test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString(), 1);
	ref_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString(), 2);
	test_image.AddImage(&ref_image);

	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -1.313164) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), 3.457573) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79,79, 0), 0.318875) == false) FailTest;

	EndTest();


}


void MyTestApp::TestAlignmentFunctions()
{
	// Phaseshift
	BeginTest("Image::PhaseShift");

	Image test_image;
	Image ref_image;
	Peak my_peak;

	test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString(), 1);
	test_image.PhaseShift(20,20,0);

	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -1.010296) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), -2.280109) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79,79, 0), 0.239702) == false) FailTest;

	EndTest();

	// CalculateCrossCorrelationImageWith
	BeginTest("Image::CalculateCrossCorrelationImageWith");

	test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString(), 1);
	ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(), 1);
	test_image.CalculateCrossCorrelationImageWith(&ref_image);

	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 0.004323) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), 0.543692) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79,79, 0), 0.006927) == false) FailTest;

	EndTest();

	//FindPeakWithIntegerCoordinates

	BeginTest("Image::FindPeakWithIntegerCoordinates");

	test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(), 1);
	ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(), 1);
	test_image.PhaseShift(7, 10, 0);

	test_image.CalculateCrossCorrelationImageWith(&ref_image);
	my_peak = test_image.FindPeakWithIntegerCoordinates();

	if (DoublesAreAlmostTheSame(my_peak.x, 7.0) == false) FailTest;
	if (DoublesAreAlmostTheSame(my_peak.y, 10.0) == false) FailTest;
	if (DoublesAreAlmostTheSame(my_peak.z, 0) == false) FailTest;
	if (DoublesAreAlmostTheSame(my_peak.value, 1) == false) FailTest;

	EndTest();

	//FindPeakWithParabolaFit

	BeginTest("Image::FindPeakWithParabolaFit");

	test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(), 1);
	ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(), 1);
	test_image.PhaseShift(7.3, 10.7, 0);

	test_image.CalculateCrossCorrelationImageWith(&ref_image);
	my_peak = test_image.FindPeakWithParabolaFit();

	if (DoublesAreAlmostTheSame(my_peak.x, 7.295232) == false) FailTest;
	if (DoublesAreAlmostTheSame(my_peak.y, 10.704828) == false) FailTest;
	if (DoublesAreAlmostTheSame(my_peak.z, 0) == false) FailTest;
	if (DoublesAreAlmostTheSame(my_peak.value, 0.993427) == false) FailTest;

	EndTest();
}

void MyTestApp::TestFilterFunctions()
{
	// BFACTOR
	BeginTest("Image::ApplyBFactor");

	Image test_image;

	test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString(), 1);
	test_image.ForwardFFT();
	test_image.ApplyBFactor(1500);
	test_image.BackwardFFT();

	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 0.027244) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), 1.320998) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79,79, 0), 0.012282) == false) FailTest;

	EndTest();

	// Mask central cross

	BeginTest("Image::MaskCentralCross");

	test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString(), 1);
	test_image.MaskCentralCross(2,2);
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.256103) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), 0.158577) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79,79, 0), 0.682073) == false) FailTest;

	EndTest();


}

void MyTestApp::TestScalingAndSizingFunctions()
{

	BeginTest("Image::ClipInto");

	MRCFile input_file(hiv_images_80x80x10_filename.ToStdString(), false);
	Image test_image;
	Image clipped_image;
	fftw_complex test_pixel;

	// test real space clipping bigger..

	clipped_image.Allocate(160,160,1);

	test_image.ReadSlice(&input_file, 1);
	test_image.ClipInto(&clipped_image, 0);


	//wxPrintf("value = %f\n", clipped_image.ReturnRealPixelValue(119,119));
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), -0.340068) == false) FailTest;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(80,80, 0), 1.819805) == false) FailTest;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(119,119, 0), 0.637069) == false) FailTest;

	// test real space clipping smaller..

	clipped_image.Allocate(50,50,1);
	test_image.ClipInto(&clipped_image, 0);


	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -2.287762) == false) FailTest;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.819805) == false) FailTest;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(49, 49, 0), -1.773780) == false) FailTest;


	// test Fourier space clipping bigger

	clipped_image.Allocate(160,160,1);
	test_image.ForwardFFT();
	test_image.ClipInto(&clipped_image, 0);

	// check some values

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), -100.0) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.0) == false) FailTest;

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.0) == false) FailTest;

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.045677) == false) FailTest;

	// test Fourier space clipping smaller

	clipped_image.Allocate(50,50,1);
	test_image.ClipInto(&clipped_image, 0);

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.0) == false) FailTest;

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100);
	//wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
	if (DoublesAreAlmostTheSame(creal(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.045677) == false) FailTest;


	EndTest();

	// Check Resize..


	BeginTest("Image::Resize");

	test_image.ReadSlice(&input_file, 1);
	test_image.Resize(160, 160, 1);

	//Real space big

	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), -0.340068) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(80,80, 0), 1.819805) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(119,119, 0), 0.637069) == false) FailTest;

	// Real space small

	test_image.ReadSlice(&input_file, 1);
	test_image.Resize(50, 50, 1);

	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -2.287762) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.819805) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(49, 49, 0), -1.773780) == false) FailTest;

	// Fourier space big

	test_image.ReadSlice(&input_file, 1);
	test_image.ForwardFFT();
	test_image.Resize(160, 160, 1);

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), -100.0) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.0) == false) FailTest;

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.0) == false) FailTest;

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.045677) == false) FailTest;

	// Fourier space small


	test_image.ReadSlice(&input_file, 1);
	test_image.ForwardFFT();
	test_image.Resize(50, 50, 1);

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100);
	if (DoublesAreAlmostTheSame(creal(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.0) == false) FailTest;

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100);
	//wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
	if (DoublesAreAlmostTheSame(creal(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(cimag(test_pixel), 0.045677) == false) FailTest;




	EndTest();

	//wxPrintf ("real = %f, imag = %f", creal(test_pixel), cimag(test_pixel), 0.0);
}

void MyTestApp::TestMRCFunctions()
{
	BeginTest("MRCFile::OpenFile");

	MRCFile input_file(hiv_images_80x80x10_filename.ToStdString(), false);

	// check dimensions..

	if (input_file.ReturnNumberOfSlices() != 10) FailTest;
	if (input_file.ReturnXSize() != 80) FailTest;
	if (input_file.ReturnYSize() != 80) FailTest;
	if (input_file.ReturnZSize() != 10) FailTest;

	EndTest();

	BeginTest("Image::ReadSlice");

	Image test_image;
	test_image.ReadSlice(&input_file, 1);

	// check dimensions and type

	if (test_image.is_in_real_space == false) FailTest;
	if (test_image.logical_x_dimension != 80) FailTest;
	if (test_image.logical_y_dimension != 80) FailTest;
	if (test_image.logical_z_dimension != 1) FailTest;

	// check first and last pixel...

	if (DoublesAreAlmostTheSame(test_image.real_values[0], -0.340068) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069) == false) FailTest;

	EndTest();
}

void MyTestApp::TestFFTFunctions()
{

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

	if (DoublesAreAlmostTheSame(creal(test_image.complex_values[0]), 1) == false || DoublesAreAlmostTheSame(cimag(test_image.complex_values[0]), 0) == false) FailTest;

	// if we set this to 0,0 - all remaining pixels should now be 0

	test_image.complex_values[0] = 0 + 0 * I;

	for (counter = 0; counter < test_image.real_memory_allocated / 2; counter++)
	{
		if (DoublesAreAlmostTheSame(creal(test_image.complex_values[counter]), 0) == false || DoublesAreAlmostTheSame(cimag(test_image.complex_values[counter]), 0) == false) FailTest;
	}

	// sine wave

	test_image.ReadSlice(&input_file, 1);
	test_image.ForwardFFT();

	// now one pixel should be set, and the rest should be 0..

	if (DoublesAreAlmostTheSame(creal(test_image.complex_values[20]), 0) == false || DoublesAreAlmostTheSame(cimag(test_image.complex_values[20]), -5) == false) FailTest;
	// set it to 0, then everything should be zero..

	test_image.complex_values[20] = 0 + 0 * I;

	for (counter = 0; counter < test_image.real_memory_allocated / 2; counter++)
	{
		if (creal(test_image.complex_values[counter]) >  0.000001 || cimag(test_image.complex_values[counter]) > 0.000001) FailTest;
	}

	EndTest();

	// Backward FFT


	BeginTest("Image::BackwardFFT");

	test_image.Allocate(64,64,1, false);
	test_image.SetToConstant(0.0);
	test_image.complex_values[0] = 1 + 0 * I;
	test_image.BackwardFFT();
	test_image.RemoveFFTWPadding();


	for (counter = 0; counter < test_image.logical_x_dimension * test_image.logical_y_dimension; counter++)
	{
		if (DoublesAreAlmostTheSame(test_image.real_values[counter], 1.0) == false) FailTest;
	}

	EndTest();

}



void MyTestApp::BeginTest(const char *test_name)
{
	int length = strlen(test_name);
	int blank_space = 45 - length;
	wxPrintf("Testing %s ", test_name);
	test_has_passed = true;

	for (int counter = 0; counter < blank_space; counter++)
	{
		wxPrintf(" ");
	}

	wxPrintf(": ");
}

void MyTestApp::EndTest()
{
	if (test_has_passed == true) PrintResult(true);
}

void MyTestApp::PrintResultSlave(bool passed, int line)
{

	if (passed == true)
	{
		if (OutputIsAtTerminal() == true) wxPrintf(ANSI_COLOR_GREEN "PASSED!" ANSI_COLOR_RESET);
		else wxPrintf("PASSED!");
	}
	else
	{
		if (OutputIsAtTerminal() == true) wxPrintf(ANSI_COLOR_RED "FAILED! (Line : %i)" ANSI_COLOR_RESET, line);
		else wxPrintf("FAILED! (Line : %i)", line);
	}

	wxPrintf("\n");
}

void MyTestApp::PrintTitle(const char *title)
{
	wxPrintf("\n");
	if (OutputIsAtTerminal() == true) wxPrintf(ANSI_UNDERLINE "%s" ANSI_UNDERLINE_OFF, title);
	else  wxPrintf("%s", title);
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


