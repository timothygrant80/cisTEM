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

#define PrintResult(result)	PrintResultWorker(result, __LINE__);
#define FailTest {if (test_has_passed == true) PrintResultWorker(false, __LINE__); test_has_passed = false;}





// TODO //
// TEST 3D's
// ODD Sized IMAGES
// NON SQUARE IMAGES




class
MyTestApp : public MyApp //public wxAppConsole
{
	wxString hiv_image_80x80x1_filename;
	wxString hiv_images_80x80x10_filename;
	wxString sine_wave_128x128x1_filename;
	wxString numeric_text_filename;
	wxString temp_directory;

	public:
    // We need DoCalculation so we can have a bool return type for automated testing and a noop DoInteractiveUserInput to allow it to run from the console.
		bool DoCalculation();
    void DoInteractiveUserInput();

		bool test_has_passed;
    bool all_tests_have_passed;

		void TestMRCFunctions();
		void TestFFTFunctions();
		void TestScalingAndSizingFunctions();
		void TestFilterFunctions();
		void TestAlignmentFunctions();
		void TestImageArithmeticFunctions();
		void TestSpectrumBoxConvolution();
		void TestImageLoopingAndAddressing();
		void TestNumericTextFiles();
		void TestClipIntoFourier();
		void TestMaskCentralCross();
		void TestStarToBinaryFileConversion();
    void TestElectronExposureFilter();
    void TestEmpiricalDistribution();
    void TestSumOfSquaresFourierAndFFTNormalization();

		void BeginTest(const char *test_name);
		void EndTest();
		void PrintTitle(const char *title);
		void PrintResultWorker( bool passed, int line);
		void WriteEmbeddedFiles();
		void WriteEmbeddedArray(const char *filename, const unsigned char *array, long length);
		void WriteNumericTextFile(const char *filename);
};


IMPLEMENT_APP(MyTestApp)

void MyTestApp::DoInteractiveUserInput()
{
  // noop
}

bool MyTestApp::DoCalculation()
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

  all_tests_have_passed = true;

	TestMRCFunctions();
	TestImageArithmeticFunctions();
	TestFFTFunctions();
	TestScalingAndSizingFunctions();
	TestFilterFunctions();
	TestAlignmentFunctions();
	TestSpectrumBoxConvolution();
	TestImageLoopingAndAddressing();
	TestNumericTextFiles();
	TestClipIntoFourier();
	TestMaskCentralCross();
	TestStarToBinaryFileConversion();
  TestElectronExposureFilter();
  TestEmpiricalDistribution();
  TestSumOfSquaresFourierAndFFTNormalization();

	wxPrintf("\n\n\n");

  if ( ! all_tests_have_passed ) std::exit(-1);
  else return 0;
}


// A partial test for the ClipInto method, specifically when clipping a Fourier transform into
// a larger volume
void MyTestApp::TestClipIntoFourier()
{
	BeginTest("Image::ClipIntoFourier");

	Image test_image_original;
	Image test_image;
	Image test_image_ampl;
	Image big_image;
	Image big_image_ampl;
	int address;
	int i,j,k;
	int i_logi, j_logi, k_logi;
	const float error_tolerance = 0.0001;
	const bool write_files_out = false;


	// Create a test image
	test_image.Allocate(4,4,4,true);
	address = 0;
	for (k=0;k<4;k++)
	{
		for (j=0;j<4;j++)
		{
			for (i=0;i<4;i++)
			{
				test_image.real_values[address] = float(address);
				address++;
			}
			address += test_image.padding_jump_value;
		}
	}
	global_random_number_generator.SetSeed(0);
	test_image.AddGaussianNoise(100.0);

	// Keep a copy
	test_image_original.CopyFrom(&test_image);

	// Write test image to disk
	if (write_files_out) test_image.QuickAndDirtyWriteSlices("dbg_start.mrc",1,4);

	// Clip into a larger image when in Fourier space
	big_image.Allocate(8,8,8,false);
	test_image.ForwardFFT();
	test_image.ComputeAmplitudeSpectrum(&test_image_ampl);
	if (write_files_out) test_image_ampl.QuickAndDirtyWriteSlices("dbg_start_ampl.mrc",1,4);
	test_image.ClipInto(&big_image);
	big_image.ComputeAmplitudeSpectrum(&big_image_ampl);
	if (write_files_out) big_image_ampl.QuickAndDirtyWriteSlices("dbg_big_ampl.mrc",1,8);

	//wxPrintf("%f %f %f %f\n",cabsf(test_image.complex_values[2]),cabsf(big_image.complex_values[2]),test_image_ampl.real_values[2], big_image_ampl.real_values[2]);

	// Do a few checks of the pixel amplitudes
	if (abs(abs(test_image.complex_values[0]) - abs(big_image.complex_values[0])) > error_tolerance) FailTest;
	if (abs(abs(test_image.complex_values[1]) - abs(big_image.complex_values[1])) > error_tolerance) FailTest;
	if (abs(abs(test_image.complex_values[2]) - abs(big_image.complex_values[2])) > error_tolerance) FailTest;

	for (k_logi = -2; k_logi <= 2; k_logi ++)
	{
		for (j_logi = -2; j_logi <= 2; j_logi ++)
		{
			for (i_logi = 0; i_logi <= 2; i_logi ++)
			{
				if (big_image.complex_values[big_image.ReturnFourier1DAddressFromLogicalCoord(i_logi,j_logi,k_logi)] == 0.0f )
				{
					wxPrintf("\nComplex pixel with logical coords %i %i %i was not set!\n",i_logi,j_logi,k_logi);
					FailTest;
				}
			}
		}
	}

	// Clip back into smaller image - should get back to where we started
	big_image.BackwardFFT();
	if (write_files_out) big_image.QuickAndDirtyWriteSlices("dbg_big_real_space.mrc",1,8);
	big_image.ForwardFFT();
	big_image.ClipInto(&test_image);
	test_image.BackwardFFT();

	// Write output image to disk
	if (write_files_out) test_image.QuickAndDirtyWriteSlices("dbg_finish.mrc",1,4);

	// Check we still have the same values
	address = 0;
	for (k=0;k<4;k++)
	{
		for (j=0;j<4;j++)
		{
			for (i=0;i<4;i++)
			{
				if (abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001)
				{
					wxPrintf("Voxel at address %i use to have value %f, now is %f\n",address, test_image_original.real_values[address],test_image.real_values[address]);
					FailTest;
				}
				address++;
			}
			address += test_image.padding_jump_value;
		}
	}


	/*
	 *  Now test with odd dimensions into odd dimensions
	 */
	test_image.Allocate(5,5,5,true);
	test_image.SetToConstant(0.0);
	test_image.AddGaussianNoise(100.0);
	test_image_original.CopyFrom(&test_image);
	test_image.ForwardFFT();
	big_image.Allocate(9,9,9,false);
	test_image.ClipInto(&big_image);
	big_image.BackwardFFT();
	big_image.ForwardFFT();
	big_image.ClipInto(&test_image);
	test_image.BackwardFFT();
	// Check we still have the same values
	address = 0;
	for (k=0;k<5;k++)
	{
		for (j=0;j<5;j++)
		{
			for (i=0;i<5;i++)
			{
				if (abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001)
				{
					wxPrintf("Voxel at address %i use to have value %f, now is %f\n",address, test_image_original.real_values[address],test_image.real_values[address]);
					FailTest;
				}
				address++;
			}
			address += test_image.padding_jump_value;
		}
	}

	/*
	 *  Now test with odd dimensions into even dimensions
	 */
	test_image.Allocate(5,5,5,true);
	test_image.SetToConstant(0.0);
	test_image.AddGaussianNoise(100.0);
	test_image_original.CopyFrom(&test_image);
	test_image.ForwardFFT();
	big_image.Allocate(8,8,8,false);
	test_image.ClipInto(&big_image);
	big_image.BackwardFFT();
	big_image.ForwardFFT();
	big_image.ClipInto(&test_image);
	test_image.BackwardFFT();
	// Check we still have the same values
	address = 0;
	for (k=0;k<5;k++)
	{
		for (j=0;j<5;j++)
		{
			for (i=0;i<5;i++)
			{
				if (abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001)
				{
					wxPrintf("Voxel at address %i use to have value %f, now is %f\n",address, test_image_original.real_values[address],test_image.real_values[address]);
					FailTest;
				}
				address++;
			}
			address += test_image.padding_jump_value;
		}
	}

	/*
	 *  Now test with even dimensions into odd dimensions
	 */
	test_image.Allocate(4,4,4,true);
	test_image.SetToConstant(0.0);
	test_image.AddGaussianNoise(100.0);
	test_image_original.CopyFrom(&test_image);
	test_image.ForwardFFT();
	big_image.Allocate(9,9,9,false);
	test_image.ClipInto(&big_image);
	big_image.BackwardFFT();
	big_image.ForwardFFT();
	big_image.ClipInto(&test_image);
	test_image.BackwardFFT();
	// Check we still have the same values
	address = 0;
	for (k=0;k<4;k++)
	{
		for (j=0;j<4;j++)
		{
			for (i=0;i<4;i++)
			{
				if (abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001)
				{
					wxPrintf("Voxel at address %i use to have value %f, now is %f\n",address, test_image_original.real_values[address],test_image.real_values[address]);
					FailTest;
				}
				address++;
			}
			address += test_image.padding_jump_value;
		}
	}





	EndTest();
}

void MyTestApp::TestStarToBinaryFileConversion()
{
	BeginTest("Star File To Binary Conversion");
	// generate set of 10k random parameters

	cisTEMParameters test_parameters;
	cisTEMParameterLine temp_line;

	for (int counter = 0; counter <1000; counter++)
	{
		temp_line.amplitude_contrast = global_random_number_generator.GetUniformRandom() * 1;
		temp_line.assigned_subset  = myroundint(global_random_number_generator.GetUniformRandom() * 10);
		temp_line.beam_tilt_group = myroundint(global_random_number_generator.GetUniformRandom() * 10);
		temp_line.beam_tilt_x = global_random_number_generator.GetUniformRandom() * 10;
		temp_line.beam_tilt_y = global_random_number_generator.GetUniformRandom() * 10;
		temp_line.best_2d_class = myroundint(global_random_number_generator.GetUniformRandom() * 100);
		temp_line.defocus_1 = global_random_number_generator.GetUniformRandom() * 30000;
		temp_line.defocus_2 = global_random_number_generator.GetUniformRandom() * 30000;
		temp_line.defocus_angle = global_random_number_generator.GetUniformRandom() * 180;
		temp_line.image_is_active = myroundint(global_random_number_generator.GetUniformRandom() * 1);
		temp_line.image_shift_x = global_random_number_generator.GetUniformRandom() * 10;
		temp_line.image_shift_y = global_random_number_generator.GetUniformRandom() * 10;
		temp_line.logp = global_random_number_generator.GetUniformRandom() * 10000;
		temp_line.microscope_spherical_aberration_mm = global_random_number_generator.GetUniformRandom() * 2.7;
		temp_line.microscope_voltage_kv = global_random_number_generator.GetUniformRandom() * 300;
		temp_line.occupancy = global_random_number_generator.GetUniformRandom() * 100;
		temp_line.original_image_filename = wxString::Format("This_is_an_original_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom() * 100000);
		temp_line.particle_group =  myroundint(global_random_number_generator.GetUniformRandom() * 10);
		temp_line.phase_shift = global_random_number_generator.GetUniformRandom() * 3.14;
		temp_line.phi = global_random_number_generator.GetUniformRandom() * 180;
		temp_line.pixel_size = global_random_number_generator.GetUniformRandom() * 2;
		temp_line.position_in_stack = counter + 1;
		temp_line.pre_exposure = global_random_number_generator.GetUniformRandom() * 10;
		temp_line.psi = global_random_number_generator.GetUniformRandom() * 180;
		temp_line.reference_3d_filename = wxString::Format("This_is_a_reference_3d_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom() * 100000);
		temp_line.score = global_random_number_generator.GetUniformRandom() * 100;
		temp_line.sigma = global_random_number_generator.GetUniformRandom() * 180;
		temp_line.stack_filename = wxString::Format("This_is_a_stack_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom() * 100000);
		temp_line.theta = global_random_number_generator.GetUniformRandom() * 180;
		temp_line.total_exposure = global_random_number_generator.GetUniformRandom() * 100;
		temp_line.x_shift = global_random_number_generator.GetUniformRandom() * 50;
		temp_line.y_shift = global_random_number_generator.GetUniformRandom() * 50;

		test_parameters.all_parameters.Add(temp_line);

	}

	test_parameters.parameters_to_write.SetAllToTrue();

	temp_directory = wxFileName::GetTempDir();

	// write star and binary file..

	wxString original_star_filename = temp_directory + "/star_file.star";
	wxString original_binary_filename = temp_directory + "/binary_file.cistem";
	wxString star_from_binary_filename = temp_directory + "/star_file_converted_from_binary.star";
	wxString binary_from_star_filename = temp_directory + "/binary_file_converted_from_star.cistem";


	test_parameters.WriteTocisTEMStarFile(original_star_filename.ToStdString().c_str());
	test_parameters.WriteTocisTEMBinaryFile(original_binary_filename.ToStdString().c_str());


	// read in binary file and write out star file..

	test_parameters.ClearAll();
	test_parameters.ReadFromcisTEMBinaryFile(original_binary_filename.ToStdString().c_str());
	test_parameters.WriteTocisTEMStarFile(star_from_binary_filename.ToStdString().c_str());

	// read in star file and write to binary..

	test_parameters.ClearAll();
	test_parameters.ReadFromcisTEMStarFile(original_star_filename.ToStdString().c_str());
	test_parameters.WriteTocisTEMBinaryFile(binary_from_star_filename.ToStdString().c_str());

	// Check the sizes are the same - this isn't a very thorough test, but better than nothing.
	// It is at least an easy way to get files with all the parameters written out to quickly check by eye

	long original_star_size_in_bytes = ReturnFileSizeInBytes(original_star_filename.ToStdString().c_str());
	long original_binary_size_in_bytes = ReturnFileSizeInBytes(original_binary_filename.ToStdString().c_str());
	long star_from_binary_size_in_bytes = ReturnFileSizeInBytes(star_from_binary_filename.ToStdString().c_str());
	long binary_from_star_size_in_bytes = ReturnFileSizeInBytes(binary_from_star_filename.ToStdString().c_str());

	if (original_star_size_in_bytes != star_from_binary_size_in_bytes) FailTest;
	if (original_binary_size_in_bytes != binary_from_star_size_in_bytes) FailTest;

	// Check the star files are byte identical (apart from bytes in the text file which represent the write time)..
	// the binary files are not expected to be byte identical as there is a change in precision after being written out to star file.

	char original_star_file[original_star_size_in_bytes];
	char star_file_from_binary_file[star_from_binary_size_in_bytes];

	FILE *current_file;

	current_file = fopen(original_star_filename.ToStdString().c_str(), "rb");
	fread(original_star_file, 1, original_star_size_in_bytes, current_file);
	fclose(current_file);

	current_file = fopen(star_from_binary_filename.ToStdString().c_str(), "rb");
	fread(star_file_from_binary_file, 1, star_from_binary_size_in_bytes, current_file);
	fclose(current_file);

	for (long byte_counter = 63; byte_counter < original_star_size_in_bytes; byte_counter++) // start at byte 63, the before that is a comment which includes the time written
	{
		if (original_star_file[byte_counter] != star_file_from_binary_file[byte_counter]) FailTest;
	}

	EndTest();
}


void MyTestApp::TestElectronExposureFilter()
{
	BeginTest("Test Electron Exposure Filter");

/*
  Test exposure filter for a simple square image size, over all voltages and three pixel sizes.
  The "ground truth" values are taken from a print out using the code for electron_dose.* from commit 
  cdd0c04e984412661c983ab7176954093b502ad5 Dec 9, 2021
  using this line in the inner loop (once for odd once for even)
    for (auto & indx : indx_even)
    {
        wxPrintf("%3.9f, ", dose_filter_odd[indx]);
    }
*/
  const int size_small = 1024;
  
  std::vector<float> accelerating_voltage_vector = { 300.f, 200.f, 100.f}; // only three supported values.
  std::vector<float> pixel_size_vector = { 0.72, 1.0, 2.1 }; // values not chosen for any  good reason.
  std::vector<int>   indx_even = { 0, 13, size_small / 3, size_small - 1};
  std::vector<int>   indx_odd  = { 0, 13, (size_small+1) / 3, size_small};

  std::vector<float> ground_truth_even= {1.000000000, 0.929921567, 0.017324856, 0.010133515, 1.000000000, 0.958591580, 0.031609546, 0.015432503, 1.000000000, 0.987710178, 0.155882418, 0.065504745, 1.000000000, 0.913183212, 0.006285460, 0.003215143, 1.000000000, 0.948510230, 0.013328240, 0.005439333, 1.000000000, 0.984661400, 0.097948194, 0.033139121, 1.000000000, 0.872345626, 0.000488910, 0.000178414, 1.000000000, 0.923584640, 0.001513943, 0.000393374, 1.000000000, 0.977023780, 0.030387951, 0.005955920};
  std::vector<float> ground_truth_odd = {1.000000000, 0.930029809, 0.017352197, 0.010144006, 1.000000000, 0.958656907, 0.031672075, 0.015455280, 1.000000000, 0.987729967, 0.156189293, 0.065646693, 1.000000000, 0.913316071, 0.006297863, 0.003219303, 1.000000000, 0.948590994, 0.013361209, 0.005449366, 1.000000000, 0.984686077, 0.098189279, 0.033228900, 1.000000000, 0.872536480, 0.000490361, 0.000178761, 1.000000000, 0.923702955, 0.001519577, 0.000394466, 1.000000000, 0.977060616, 0.030500494, 0.005980202};
  int ground_truth_counter = 0;
  ElectronDose* my_electron_dose;


  Image test_image_small_even,test_image_small_odd;
  test_image_small_even.Allocate(size_small, size_small, true);
  test_image_small_odd.Allocate(size_small+1, size_small+1, true);

	float* dose_filter_even = new float[test_image_small_even.real_memory_allocated / 2];
	float* dose_filter_odd  = new float[test_image_small_odd.real_memory_allocated / 2];

  for (auto & acceleration_voltage : accelerating_voltage_vector)
  {
    for (auto & pixel_size : pixel_size_vector)
    {
      my_electron_dose = new ElectronDose(acceleration_voltage, pixel_size);

      ZeroFloatArray(dose_filter_even, test_image_small_even.real_memory_allocated/2);
      ZeroFloatArray(dose_filter_odd, test_image_small_odd.real_memory_allocated/2);

			my_electron_dose->CalculateDoseFilterAs1DArray(&test_image_small_even, dose_filter_even, 15.f, 30.f);
      my_electron_dose->CalculateDoseFilterAs1DArray(&test_image_small_odd, dose_filter_odd, 15.f, 30.f);

      for (auto & indx : indx_even)
      {
        if ( ! FloatsAreAlmostTheSame(dose_filter_even[indx], ground_truth_even[ground_truth_counter]) ) {wxPrintf("Failed for kv,pix,ev: %3.f %3.3f, values %f %f\n",acceleration_voltage,pixel_size,dose_filter_even[indx], ground_truth_even[ground_truth_counter]);FailTest;}
        if ( ! FloatsAreAlmostTheSame(dose_filter_odd[indx], ground_truth_odd[ground_truth_counter]) ) {wxPrintf("Failed for kv,pix,od: %3.f %3.3f, values %f %f\n",acceleration_voltage,pixel_size,dose_filter_odd[indx], ground_truth_odd[ground_truth_counter]);FailTest;}
        ground_truth_counter++;
      }

      delete my_electron_dose;
    }
  }
  
  delete [] dose_filter_even;
  delete [] dose_filter_odd;

	EndTest();
}

void MyTestApp::TestEmpiricalDistribution()
{
  BeginTest("Test Empirical Distribution");

  Image test_image;
  test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(),1);

  EmpiricalDistribution my_dist = test_image.ReturnDistributionOfRealValues();

  if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean() + 1.f, 1.0f) ) FailTest;
  if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance(), 1.0f) ) FailTest;
  if (   my_dist.GetNumberOfSamples() != 6400 ) FailTest;
  if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetMinimum(), -3.1520f) ) FailTest;
  if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetMaximum(),  7.0222f) ) FailTest;
  if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleSumOfSquares(), 6400.f) ) FailTest;

  EndTest();
}

void MyTestApp::TestSumOfSquaresFourierAndFFTNormalization()
{
  // Test depends on TestFFTFunctions passing.
  
  BeginTest("Test Sum Of Squares Fourier");

  Image test_image;
  test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(),1);
  test_image.ForwardFFT(false);

  // By Parsevals theorem, the sum of squares of the Fourier transform of an image is equal to the sum of squares of the real image.
  // On the foward FFT the variance is scaled by N and on the inverse by N again. (This makes it as though the original values were scaled by N round-trip.)
  // So without normalization, the sum of squares should be N * realspace sumof squares.
  float sum_of_squares = test_image.ReturnSumOfSquares();
  if ( ! RelativeErrorIsLessThanEpsilon(sum_of_squares, 6400.f * 6400.f) ) FailTest;

  // We normalize for the full round trip on the forward FFT, so in this case the sum of squares should be
  // realspace sumof squares / N.
  test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(),1);
  test_image.ForwardFFT(true);
  sum_of_squares = test_image.ReturnSumOfSquares();
  if ( ! RelativeErrorIsLessThanEpsilon(sum_of_squares,1.f) ) FailTest;

  EndTest();
}

void MyTestApp::TestImageLoopingAndAddressing()
{
	BeginTest("Image::LoopingAndAddressing");

	Image test_image;

	//
	// Even
	//
	test_image.Allocate(4,4,4,true);

	if (test_image.physical_upper_bound_complex_x != 2) FailTest;
	if (test_image.physical_upper_bound_complex_y != 3) FailTest;
	if (test_image.physical_upper_bound_complex_z != 3) FailTest;

	if (test_image.physical_address_of_box_center_x != 2) FailTest;
	if (test_image.physical_address_of_box_center_y != 2) FailTest;
	if (test_image.physical_address_of_box_center_z != 2) FailTest;

	if (test_image.physical_index_of_first_negative_frequency_y != 2) FailTest;
	if (test_image.physical_index_of_first_negative_frequency_z != 2) FailTest;

	if (test_image.logical_upper_bound_complex_x != 2) FailTest;
	if (test_image.logical_upper_bound_complex_y != 1) FailTest;
	if (test_image.logical_upper_bound_complex_z != 1) FailTest;

	if (test_image.logical_lower_bound_complex_x != -2) FailTest;
	if (test_image.logical_lower_bound_complex_y != -2) FailTest;
	if (test_image.logical_lower_bound_complex_z != -2) FailTest;

	if (test_image.logical_upper_bound_real_x != 1) FailTest;
	if (test_image.logical_upper_bound_real_y != 1) FailTest;
	if (test_image.logical_upper_bound_real_z != 1) FailTest;

	if (test_image.logical_lower_bound_real_x != -2) FailTest;
	if (test_image.logical_lower_bound_real_y != -2) FailTest;
	if (test_image.logical_lower_bound_real_z != -2) FailTest;

	if (test_image.padding_jump_value != 2) FailTest;

	//
	// Odd
	//
	test_image.Allocate(5,5,5,true);

	if (test_image.physical_upper_bound_complex_x != 2) FailTest;
	if (test_image.physical_upper_bound_complex_y != 4) FailTest;
	if (test_image.physical_upper_bound_complex_z != 4) FailTest;

	if (test_image.physical_address_of_box_center_x != 2) FailTest;
	if (test_image.physical_address_of_box_center_y != 2) FailTest;
	if (test_image.physical_address_of_box_center_z != 2) FailTest;

	if (test_image.physical_index_of_first_negative_frequency_y != 3) FailTest;
	if (test_image.physical_index_of_first_negative_frequency_z != 3) FailTest;

	if (test_image.logical_upper_bound_complex_x != 2) FailTest;
	if (test_image.logical_upper_bound_complex_y != 2) FailTest;
	if (test_image.logical_upper_bound_complex_z != 2) FailTest;

	if (test_image.logical_lower_bound_complex_x != -2) FailTest;
	if (test_image.logical_lower_bound_complex_y != -2) FailTest;
	if (test_image.logical_lower_bound_complex_z != -2) FailTest;

	if (test_image.logical_upper_bound_real_x != 2) FailTest;
	if (test_image.logical_upper_bound_real_y != 2) FailTest;
	if (test_image.logical_upper_bound_real_z != 2) FailTest;

	if (test_image.logical_lower_bound_real_x != -2) FailTest;
	if (test_image.logical_lower_bound_real_y != -2) FailTest;
	if (test_image.logical_lower_bound_real_z != -2) FailTest;


	if (test_image.padding_jump_value != 1) FailTest;

	EndTest();

}

void MyTestApp::TestSpectrumBoxConvolution()
{
	BeginTest("Image::SpectrumBoxConvolution");

	Image test_image;
	Image output_image;

	test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString(),1);

	output_image.Allocate(test_image.logical_x_dimension,test_image.logical_y_dimension,test_image.logical_z_dimension);
	test_image.SpectrumBoxConvolution(&output_image,7,3);

	if (DoublesAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.049189) == false) FailTest;
	if (DoublesAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), 1.634473) == false) FailTest;
	if (DoublesAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(79,79, 0),-0.049189) == false) FailTest;

	EndTest();
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

void MyTestApp::TestNumericTextFiles()
{
	// AddImage
	BeginTest("NumericTextFile::Init");

	NumericTextFile test_file(numeric_text_filename, OPEN_TO_READ);

	if (test_file.number_of_lines != 4) FailTest;
	if (test_file.records_per_line != 5) FailTest;

	EndTest();

	BeginTest("NumericTextFile::ReadLine");
	float temp_float[5];

	test_file.ReadLine(temp_float);

	if (int(temp_float[0]) != 1) FailTest;
	if (int(temp_float[1]) != 2) FailTest;
	if (int(temp_float[2]) != 3) FailTest;
	if (int(temp_float[3]) != 4) FailTest;
	if (int(temp_float[4]) != 5) FailTest;

	test_file.ReadLine(temp_float);

	if (DoublesAreAlmostTheSame(temp_float[0], 6.0) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[1], 7.1) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[2], 8.3) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[3], 9.4) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[4], 10.5) == false) FailTest;

	test_file.ReadLine(temp_float);

	if (DoublesAreAlmostTheSame(temp_float[0], 11.2) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[1], 12.7) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[2], 13.2) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[3], 14.1) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[4], 15.8) == false) FailTest;

	test_file.ReadLine(temp_float);

	if (DoublesAreAlmostTheSame(temp_float[0], 16.1245) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[1], 17.81003) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[2], 18.5467) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[3], 19.7621) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[4], 20.11111) == false) FailTest;

	EndTest();

	BeginTest("NumericTextFile::WriteLine");

	wxString output_filename = temp_directory + "/number_out.num";
	NumericTextFile output_test_file(output_filename, OPEN_TO_WRITE, 5);

	temp_float[0] = 0.1;
	temp_float[1] = 0.2;
	temp_float[2] = 0.3;
	temp_float[3] = 0.4;
	temp_float[4] = 0.5;

	output_test_file.WriteCommentLine("This is a comment line %i", 5);
	output_test_file.WriteLine(temp_float);
	output_test_file.WriteCommentLine("Another comment = %s", "booooo!");
	temp_float[0] = 0.67;
	temp_float[1] = 0.78;
	temp_float[2] = 0.89;
	temp_float[3] = 0.91;
	temp_float[4] = 1.02;

	output_test_file.WriteLine(temp_float);
	output_test_file.Flush();

	test_file.Close();
	test_file.Open(output_filename, OPEN_TO_READ);

	if (test_file.number_of_lines != 2) FailTest;
	if (test_file.records_per_line != 5) FailTest;

	test_file.ReadLine(temp_float);

	if (DoublesAreAlmostTheSame(temp_float[0], 0.1) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[1], 0.2) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[2], 0.3) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[3], 0.4) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[4], 0.5) == false) FailTest;

	test_file.ReadLine(temp_float);

	if (DoublesAreAlmostTheSame(temp_float[0], 0.67) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[1], 0.78) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[2], 0.89) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[3], 0.91) == false) FailTest;
	if (DoublesAreAlmostTheSame(temp_float[4], 1.02) == false) FailTest;

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

	if (my_peak.x > 7.3 || my_peak. x < 7.29) FailTest;
	if (my_peak.y > 10.70484 || my_peak.y < 10.70481) FailTest;
	if (DoublesAreAlmostTheSame(my_peak.z, 0) == false) FailTest;
	if (my_peak.value > 0.99343 || my_peak.value < 0.99342) FailTest;

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
	test_image.ForwardFFT();
	test_image.MaskCentralCross(2,2);
	test_image.BackwardFFT();
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.256103) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40,40, 0), 0.158577) == false) FailTest;
	if (DoublesAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79,79, 0), 0.682073) == false) FailTest;

	EndTest();


}

void MyTestApp::TestMaskCentralCross()
{
	BeginTest("Image::MaskCentralCross");

	Image my_image;

	my_image.Allocate(128,128,1);
	my_image.SetToConstant(1.0);
	my_image.MaskCentralCross(3,3);

	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0,0,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(127,127,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0,127,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(127,0,0),1.0)) FailTest;

	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67,67,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67,61,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61,61,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61,67,0),1.0)) FailTest;

	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0,61,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0,67,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61,0,0),1.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67,0,0),1.0)) FailTest;

	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66,66,0),0.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66,62,0),0.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62,62,0),0.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62,66,0),0.0)) FailTest;

	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0,62,0),0.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0,66,0),0.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62,0,0),0.0)) FailTest;
	if (! DoublesAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66,0,0),0.0)) FailTest;



	EndTest();

}

void MyTestApp::TestScalingAndSizingFunctions()
{

	BeginTest("Image::ClipInto");

	MRCFile input_file(hiv_images_80x80x10_filename.ToStdString(), false);
	Image test_image;
	Image clipped_image;
	std::complex<float> test_pixel;

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

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), -100.0) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.0) == false) FailTest;

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.0) == false) FailTest;

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.045677) == false) FailTest;

	// test Fourier space clipping smaller

	clipped_image.Allocate(50,50,1);
	test_image.ClipInto(&clipped_image, 0);

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.0) == false) FailTest;

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
	//wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
	if (DoublesAreAlmostTheSame(real(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.045677) == false) FailTest;

	// test real space clipping smaller to odd..

	test_image.ReadSlice(&input_file, 1);
	clipped_image.Allocate(49,49,1);
	test_image.ClipInto(&clipped_image, 0);

	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.391899) == false) FailTest;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.689942) == false) FailTest;
	if (DoublesAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(48, 48, 0), -1.773780) == false) FailTest;

	// test fourier space flipping smaller to odd..

	test_image.ForwardFFT();
	test_image.ClipInto(&clipped_image, 0);

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.0) == false) FailTest;

	test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
	//wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
	if (DoublesAreAlmostTheSame(real(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.045677) == false) FailTest;

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

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), -100.0) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.0) == false) FailTest;

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.0) == false) FailTest;

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.045677) == false) FailTest;

	// Fourier space small


	test_image.ReadSlice(&input_file, 1);
	test_image.ForwardFFT();
	test_image.Resize(50, 50, 1);

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
	if (DoublesAreAlmostTheSame(real(test_pixel), -0.010919) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.0) == false) FailTest;

	test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
	//wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
	if (DoublesAreAlmostTheSame(real(test_pixel), 0.075896) == false || DoublesAreAlmostTheSame(imag(test_pixel), 0.045677) == false) FailTest;




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

	if (DoublesAreAlmostTheSame(real(test_image.complex_values[0]), 1) == false || DoublesAreAlmostTheSame(imag(test_image.complex_values[0]), 0) == false) FailTest;

	// if we set this to 0,0 - all remaining pixels should now be 0

	test_image.complex_values[0] = 0.0f + 0.0f * I;

	for (counter = 0; counter < test_image.real_memory_allocated / 2; counter++)
	{
		if (DoublesAreAlmostTheSame(real(test_image.complex_values[counter]), 0) == false || DoublesAreAlmostTheSame(imag(test_image.complex_values[counter]), 0) == false) FailTest;
	}

	// sine wave

	test_image.ReadSlice(&input_file, 1);
	test_image.ForwardFFT();

	// now one pixel should be set, and the rest should be 0..

	if (DoublesAreAlmostTheSame(real(test_image.complex_values[20]), 0) == false || DoublesAreAlmostTheSame(imag(test_image.complex_values[20]), -5) == false) FailTest;
	// set it to 0, then everything should be zero..

	test_image.complex_values[20] = 0.0f + 0.0f * I;

	for (counter = 0; counter < test_image.real_memory_allocated / 2; counter++)
	{
		if (real(test_image.complex_values[counter]) >  0.000001 || imag(test_image.complex_values[counter]) > 0.000001) FailTest;
	}

	EndTest();

	// Backward FFT


	BeginTest("Image::BackwardFFT");

	test_image.Allocate(64,64,1, false);
	test_image.SetToConstant(0.0);
	test_image.complex_values[0] = 1.0f + 0.0f * I;
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
	if (test_has_passed == true) 
  {
    PrintResult(true);
  }
  else
  {
    // Sets the final return value, used in auto build &
    all_tests_have_passed = false;
  }
}

void MyTestApp::PrintResultWorker(bool passed, int line)
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
	temp_directory = wxFileName::GetTempDir();
	wxPrintf("\nWriting out embedded test files to '%s'...", temp_directory);
	fflush(stdout);

	hiv_image_80x80x1_filename = temp_directory + "/hiv_image_80x80x1.mrc";
	hiv_images_80x80x10_filename = temp_directory + "/hiv_images_shift_noise_80x80x10.mrc";
	sine_wave_128x128x1_filename = temp_directory + "/sine_wave_128x128x1.mrc";

	WriteEmbeddedArray(hiv_image_80x80x1_filename, hiv_image_80x80x1_array, sizeof(hiv_image_80x80x1_array));
	WriteEmbeddedArray(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
	WriteEmbeddedArray(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
	WriteEmbeddedArray(sine_wave_128x128x1_filename, sine_128x128x1_array, sizeof(sine_128x128x1_array));

	numeric_text_filename = temp_directory + "/numbers.num";
	WriteNumericTextFile(numeric_text_filename);

	wxPrintf("done!\n");


}

void MyTestApp::WriteEmbeddedArray(const char *filename, const unsigned char *array, long length)
{

	FILE *output_file = NULL;
	output_file = fopen(filename, "wb+");

	if (output_file == NULL)
	{
		wxPrintf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n", filename);
		wxPrintf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n", filename);
		DEBUG_ABORT;

	}

	 fwrite (array , sizeof(unsigned char), length, output_file);

	 fclose(output_file);
}

void MyTestApp::WriteNumericTextFile(const char *filename)
{

	FILE *output_file = NULL;
	output_file = fopen(filename, "wb+");

	if (output_file == NULL)
	{
		wxPrintf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n", filename);
		wxPrintf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n", filename);
		DEBUG_ABORT;

	}

	fprintf(output_file, "# This is comment, starting with #\n");
	fprintf(output_file, "C This is comment, starting with C\n");
	fprintf(output_file, "%f %f %f %f %f\n%f %f %f %f %f\n", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.1, 8.3, 9.4, 10.5);
	fprintf(output_file, "# The next line will be blank, but contain 5 spaces\n     \n");
	fprintf(output_file, "%f %f %f %f %f\n", 11.2, 12.7, 13.2, 14.1, 15.8);
	fprintf(output_file, "   # This comment line starts with #, but not at the first character\n");
	fprintf(output_file, "   C This comment line starts with C, but not at the first character\n");
	fprintf(output_file, "C The next line will have varying spaces between the datapoints\n");
	fprintf(output_file, "   %f %f   %f       %f          %f\n", 16.1245, 17.81003, 18.5467, 19.7621, 20.11111);

	fclose(output_file);
}






