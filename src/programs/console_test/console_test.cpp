#include <cstdio>
#include <unistd.h>
#include <unordered_map>

#include "../../core/core_headers.h"
#include "../../../include/libtorch/cistem_torch_helper.h"

// embedded images..

#include "hiv_image_80x80x1.cpp"
#include "hiv_images_shift_noise_80x80x10.cpp"
#include "sine_128x128x1.cpp"

// clang-format off
#define PrintResult(result) PrintResultWorker(result, __LINE__);
#define FailTest                                \
    {                                           \
        if ( test_has_passed == true )          \
            PrintResultWorker(false, __LINE__); \
        test_has_passed = false;                \
    }
#define SkipTest                                      \
    {                                                 \
        if ( test_has_passed == true )                \
            PrintResultWorker(false, __LINE__, true); \
        test_has_passed = true;                       \
    }

// clang-format on

// TODO //
// TEST 3D's
// ODD Sized IMAGES
// NON SQUARE IMAGES

void printArray(Image& image, bool round_to_int = false) {
    float val;
    Printf("Running tests...\n");
    for ( int k = 0; k < image.logical_z_dimension; k++ ) {
        Printf("z = %i\n", k);
        for ( int i = 0; i < image.logical_x_dimension; i++ ) {
            for ( int j = 0; j < image.logical_y_dimension; j++ ) {
                val = image.ReturnRealPixelFromPhysicalCoord(i, j, k);
                if ( round_to_int )
                    Printf("%i ", myroundint(val));
                else
                    Printf("%f ", val);
            }
            Printf("\n");
        }
    }
}

class
        MyTestApp : public MyApp
{
    std::string hiv_image_80x80x1_filename;
    std::string hiv_images_80x80x10_filename;
    std::string sine_wave_128x128x1_filename;
    std::string numeric_text_filename;
    std::string temp_directory;

  public:
    // We need DoCalculation so we can have a bool return type for automated testing and a noop DoInteractiveUserInput to allow it to run from the console.
    bool DoCalculation( );
    void DoInteractiveUserInput( );

    bool test_has_passed;
    bool all_tests_have_passed;

    /*
      The tests build in complexity, such that the validity of anyone one test may depend on any number of the others.
      Rather than rely on the tests being run in any given order, we instead, at the time the test is written,  
      prescribe any functions used in that test should be checked against the list of tests that have passed.

      When BeginTest() is called, the string of the test name is added to "test_results" and set to false, and current_test_name is set to test_name.
      When EndTest() is called, the current_test_name is set to "test_has_passed".

      To check the results, a list of strings is passed to the function CheckDependencies( { "Image::ClipIntoFourier", "Test Empirical Distribution" } ).
      If the string test name is not found indicating a typo or the test is not run, or the value is false, the test fails.
    */
    std::unordered_map<std::string, bool> test_results;
    bool                                  CheckDependencies(std::initializer_list<std::string> list);
    std::string                           current_test_name;

    void TestMRCFunctions( );
    void TestAssignmentOperatorsAndFunctions( );
    void TestFFTFunctions( );
    void TestScalingAndSizingFunctions( );
    void TestFilterFunctions( );
    void TestAlignmentFunctions( );
    void TestImageArithmeticFunctions( );
    void TestSpectrumBoxConvolution( );
    void TestImageLoopingAndAddressing( );
    void TestNumericTextFiles( );
    void TestClipIntoFourier( );
    void TestMaskCentralCross( );
    void TestStarToBinaryFileConversion( );
    void TestElectronExposureFilter( );
    void TestEmpiricalDistribution( );
    void TestEmpiricalDistributionThreadSafety( );
    void TestSumOfSquaresFourierAndFFTNormalization( );
    void TestRandomVariableFunctions( );
    void TestIntegerShifts( );
    void TestDatabase( );
    void TestRunProfileDiskOperations( );
    void TestCTFNodes( );
    void TestSpectrumImageMethods( );
    void TestHelicalSymmetry( );
    void TestHelicalRealSpaceSymmetrisation( );
#ifdef cisTEM_USING_LIBTORCH
    void TestLibTorch( );
#endif

    void BeginTest(const char* test_name);
    void EndTest( );
    void PrintTitle(const char* title);
    void PrintResultWorker(bool passed, int line, bool skip_on_failure = false);
    void WriteEmbeddedFiles( );
    void WriteEmbeddedArray(const char* filename, const unsigned char* array, long length);
    void WriteNumericTextFile(const char* filename);
    void WriteDatabase(const char* dir, const char* filename);
};

IMPLEMENT_APP(MyTestApp)

void MyTestApp::DoInteractiveUserInput( ) {
    // noop
}

bool MyTestApp::DoCalculation( ) {
    Printf("\n\n\n     **   ");
    if ( OutputIsAtTerminal( ) == true )
        Printf(ANSI_UNDERLINE "ProjectX Library Tester" ANSI_UNDERLINE_OFF);
    else
        Printf("ProjectX Library Tester");
    Printf("   **\n");

    //Printf("")

    WriteEmbeddedFiles( );
    Printf("\n");

    // Do tests..

    //PrintTitle("Basic I/O Functions");

    all_tests_have_passed = true;

    TestMRCFunctions( );
    TestAssignmentOperatorsAndFunctions( );
    TestImageArithmeticFunctions( );
    TestFFTFunctions( );
    TestScalingAndSizingFunctions( );
    TestFilterFunctions( );
    TestAlignmentFunctions( );
    TestSpectrumBoxConvolution( );
    TestImageLoopingAndAddressing( );
    TestNumericTextFiles( );
    TestClipIntoFourier( );
    TestMaskCentralCross( );
    TestStarToBinaryFileConversion( );
    TestElectronExposureFilter( );
    TestDatabase( );
    TestEmpiricalDistribution( );
    TestEmpiricalDistributionThreadSafety( );
    TestSumOfSquaresFourierAndFFTNormalization( );
    TestRandomVariableFunctions( );
    TestIntegerShifts( );
    TestRunProfileDiskOperations( );
    TestCTFNodes( );
    TestSpectrumImageMethods( );
    TestHelicalSymmetry( );
    TestHelicalRealSpaceSymmetrisation( );
#ifdef cisTEM_USING_LIBTORCH
    TestLibTorch( );
#endif

    Printf("\n\n\n");

    if ( ! all_tests_have_passed )
        std::exit(-1);
    else
        return 0;
}

void MyTestApp::TestAssignmentOperatorsAndFunctions( ) {

    BeginTest("Memory Assignment Ops and Funcs");

    // Test for even and odd sized square
    const int   wanted_size = 16;
    const float test_value  = 1.234f;

    Image ref_image[2];

    ref_image[0].Allocate(wanted_size, wanted_size, 1);
    ref_image[1].Allocate(wanted_size + 1, wanted_size + 1, 1);

    // Set the initial values of the reference images, that we will use to check the results of the assignment ops.
    for ( int pixel_counter = 0; pixel_counter < ref_image[0].real_memory_allocated; pixel_counter++ ) {
        ref_image[0].real_values[pixel_counter] = test_value;
    }

    for ( int pixel_counter = 0; pixel_counter < ref_image[1].real_memory_allocated; pixel_counter++ ) {
        ref_image[1].real_values[pixel_counter] = test_value;
    }

    // First test the assignment operator Image = *Image under several conditions.

    // Condition 1: test image is not allogated.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image = &ref_image[iTest];
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Condition 2: test image is allocated but not the same size.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image.Allocate(wanted_size + 3, wanted_size + 3, 1);
        test_image = &ref_image[iTest];
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Image::CopyFrom should just use the underlying method tested.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image.CopyFrom(&ref_image[iTest]);
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Assignment by reference should also then just call the pointer based assignment.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image = ref_image[iTest]; // This line is different (i.e. assign by reference.)
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Finally we check the Image::Consume method, which differs from the above test in that the resulting pointers should have the same address
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image.Consume(&ref_image[iTest]); // This line is different (i.e. assign by reference.)
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;

        // Because the data array is "stolen" we obvi cannot compare to the reference as this would give a segfault.
    }

    EndTest( );
}

// A partial test for the ClipInto method, specifically when clipping a Fourier transform into
// a larger volume
void MyTestApp::TestClipIntoFourier( ) {
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT", "Memory Assignment Ops and Funcs", "Image::SetToConstant"});

    BeginTest("Image::ClipIntoFourier");

    Image       test_image_original;
    Image       test_image;
    Image       test_image_ampl;
    Image       big_image;
    Image       big_image_ampl;
    int         address;
    int         i, j, k;
    int         i_logi, j_logi, k_logi;
    const float error_tolerance = 0.0001;
    const bool  write_files_out = false;

    // Create a test image
    test_image.Allocate(4, 4, 4, true);
    address = 0;
    for ( k = 0; k < 4; k++ ) {
        for ( j = 0; j < 4; j++ ) {
            for ( i = 0; i < 4; i++ ) {
                test_image.real_values[address] = float(address);
                address++;
            }
            address += test_image.padding_jump_value;
        }
    }
    global_random_number_generator.SetSeed(0);
    test_image.AddGaussianNoise(100.0); // TODO: Swap this for the STD functions and note that dependency.

    // Keep a copy
    test_image_original.CopyFrom(&test_image);

    // Write test image to disk
    if ( write_files_out )
        test_image.QuickAndDirtyWriteSlices("dbg_start.mrc", 1, 4);

    // Clip into a larger image when in Fourier space
    big_image.Allocate(8, 8, 8, false);
    test_image.ForwardFFT( );
    test_image.ComputeAmplitudeSpectrum(&test_image_ampl);
    if ( write_files_out )
        test_image_ampl.QuickAndDirtyWriteSlices("dbg_start_ampl.mrc", 1, 4);
    test_image.ClipInto(&big_image);
    big_image.ComputeAmplitudeSpectrum(&big_image_ampl);
    if ( write_files_out )
        big_image_ampl.QuickAndDirtyWriteSlices("dbg_big_ampl.mrc", 1, 8);

    //Printf("%f %f %f %f\n",cabsf(test_image.complex_values[2]),cabsf(big_image.complex_values[2]),test_image_ampl.real_values[2], big_image_ampl.real_values[2]);

    // Do a few checks of the pixel amplitudes
    if ( abs(abs(test_image.complex_values[0]) - abs(big_image.complex_values[0])) > error_tolerance )
        FailTest;
    if ( abs(abs(test_image.complex_values[1]) - abs(big_image.complex_values[1])) > error_tolerance )
        FailTest;
    if ( abs(abs(test_image.complex_values[2]) - abs(big_image.complex_values[2])) > error_tolerance )
        FailTest;

    for ( k_logi = -2; k_logi <= 2; k_logi++ ) {
        for ( j_logi = -2; j_logi <= 2; j_logi++ ) {
            for ( i_logi = 0; i_logi <= 2; i_logi++ ) {
                if ( big_image.complex_values[big_image.ReturnFourier1DAddressFromLogicalCoord(i_logi, j_logi, k_logi)] == 0.0f ) {
                    Printf("\nComplex pixel with logical coords %i %i %i was not set!\n", i_logi, j_logi, k_logi);
                    FailTest;
                }
            }
        }
    }

    // Clip back into smaller image - should get back to where we started
    big_image.BackwardFFT( );
    if ( write_files_out )
        big_image.QuickAndDirtyWriteSlices("dbg_big_real_space.mrc", 1, 8);
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );

    // Write output image to disk
    if ( write_files_out )
        test_image.QuickAndDirtyWriteSlices("dbg_finish.mrc", 1, 4);

    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 4; k++ ) {
        for ( j = 0; j < 4; j++ ) {
            for ( i = 0; i < 4; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    Printf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
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
    test_image.Allocate(5, 5, 5, true);
    test_image.SetToConstant(0.0);
    test_image.AddGaussianNoise(100.0);
    test_image_original.CopyFrom(&test_image);
    test_image.ForwardFFT( );
    big_image.Allocate(9, 9, 9, false);
    test_image.ClipInto(&big_image);
    big_image.BackwardFFT( );
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );
    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 5; k++ ) {
        for ( j = 0; j < 5; j++ ) {
            for ( i = 0; i < 5; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    Printf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
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
    test_image.Allocate(5, 5, 5, true);
    test_image.SetToConstant(0.0);
    test_image.AddGaussianNoise(100.0);
    test_image_original.CopyFrom(&test_image);
    test_image.ForwardFFT( );
    big_image.Allocate(8, 8, 8, false);
    test_image.ClipInto(&big_image);
    big_image.BackwardFFT( );
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );
    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 5; k++ ) {
        for ( j = 0; j < 5; j++ ) {
            for ( i = 0; i < 5; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    Printf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
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
    test_image.Allocate(4, 4, 4, true);
    test_image.SetToConstant(0.0);
    test_image.AddGaussianNoise(100.0);
    test_image_original.CopyFrom(&test_image);
    test_image.ForwardFFT( );
    big_image.Allocate(9, 9, 9, false);
    test_image.ClipInto(&big_image);
    big_image.BackwardFFT( );
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );
    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 4; k++ ) {
        for ( j = 0; j < 4; j++ ) {
            for ( i = 0; i < 4; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    Printf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
                    FailTest;
                }
                address++;
            }
            address += test_image.padding_jump_value;
        }
    }

    EndTest( );
}

void MyTestApp::TestDatabase( ) {
    BeginTest("Database");

    temp_directory             = ReturnTempDirectory();
    std::string database_filename = temp_directory + "/1_0_test/1_0_test.db";
    Project  project;
    Database database;
    database.Open(database_filename);
    auto schema_result = database.CheckSchema( );
    if ( schema_result.first.size( ) < 1 || schema_result.second.size( ) < 1 ) {
        Printf("Check Schema did not detect missing tables/columns\n");
        FailTest;
    }
    database.UpdateSchema(schema_result.second);
    schema_result = database.CheckSchema( );
    if ( schema_result.first.size( ) > 0 || schema_result.second.size( ) > 0 ) {
        Printf("Update Schema did not fix missing tables/columns\n");
        FailTest;
    }
    database.Close( );
    project.OpenProjectFromFile(database_filename);
    if ( project.cistem_version_text != CISTEM_VERSION_TEXT ) {
        Printf("New database does not have right version text\n");
        FailTest;
    }
    project.Close(false, true);

    EndTest( );
}

void MyTestApp::TestStarToBinaryFileConversion( ) {
    BeginTest("Star File To Binary Conversion");
    // generate set of 10k random parameters

    cisTEMParameters    test_parameters;
    cisTEMParameterLine temp_line;

    for ( unsigned int counter = 0; counter < 1000; counter++ ) {
        temp_line.amplitude_contrast                 = global_random_number_generator.GetUniformRandom( ) * 1;
        temp_line.assigned_subset                    = myroundint(global_random_number_generator.GetUniformRandom( ) * 10);
        temp_line.beam_tilt_group                    = myroundint(global_random_number_generator.GetUniformRandom( ) * 10);
        temp_line.beam_tilt_x                        = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.beam_tilt_y                        = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.best_2d_class                      = myroundint(global_random_number_generator.GetUniformRandom( ) * 100);
        temp_line.defocus_1                          = global_random_number_generator.GetUniformRandom( ) * 30000;
        temp_line.defocus_2                          = global_random_number_generator.GetUniformRandom( ) * 30000;
        temp_line.defocus_angle                      = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.image_is_active                    = myroundint(global_random_number_generator.GetUniformRandom( ) * 1);
        temp_line.image_shift_x                      = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.image_shift_y                      = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.logp                               = global_random_number_generator.GetUniformRandom( ) * 10000;
        temp_line.microscope_spherical_aberration_mm = global_random_number_generator.GetUniformRandom( ) * 2.7;
        temp_line.microscope_voltage_kv              = global_random_number_generator.GetUniformRandom( ) * 300;
        temp_line.occupancy                          = global_random_number_generator.GetUniformRandom( ) * 100;
        temp_line.original_image_filename            = Format("This_is_an_original_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
        temp_line.particle_group                     = myroundint(global_random_number_generator.GetUniformRandom( ) * 10);
        temp_line.phase_shift                        = global_random_number_generator.GetUniformRandom( ) * 3.14;
        temp_line.phi                                = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.pixel_size                         = global_random_number_generator.GetUniformRandom( ) * 2;
        temp_line.position_in_stack                  = counter + 1;
        temp_line.pre_exposure                       = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.psi                                = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.reference_3d_filename              = Format("This_is_a_reference_3d_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
        temp_line.score                              = global_random_number_generator.GetUniformRandom( ) * 100;
        temp_line.sigma                              = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.stack_filename                     = Format("This_is_a_stack_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
        temp_line.theta                              = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.total_exposure                     = global_random_number_generator.GetUniformRandom( ) * 100;
        temp_line.x_shift                            = global_random_number_generator.GetUniformRandom( ) * 50;
        temp_line.y_shift                            = global_random_number_generator.GetUniformRandom( ) * 50;
        temp_line.original_x_position                = global_random_number_generator.GetUniformRandom( ) * 4000;
        temp_line.original_y_position                = global_random_number_generator.GetUniformRandom( ) * 4000;

        test_parameters.all_parameters.push_back(temp_line);
    }

    test_parameters.parameters_to_write.SetAllToTrue( );

    temp_directory = ReturnTempDirectory();

    // write star and binary file..

    std::string original_star_filename    = temp_directory + "/star_file.star";
    std::string original_binary_filename  = temp_directory + "/binary_file.cistem";
    std::string star_from_binary_filename = temp_directory + "/star_file_converted_from_binary.star";
    std::string binary_from_star_filename = temp_directory + "/binary_file_converted_from_star.cistem";

    test_parameters.WriteTocisTEMStarFile(original_star_filename.c_str( ));
    test_parameters.WriteTocisTEMBinaryFile(original_binary_filename.c_str( ));

    // read in binary file and write out star file..

    test_parameters.ClearAll( );
    test_parameters.ReadFromcisTEMBinaryFile(original_binary_filename.c_str( ));
    test_parameters.WriteTocisTEMStarFile(star_from_binary_filename.c_str( ));

    // read in star file and write to binary..

    test_parameters.ClearAll( );
    test_parameters.ReadFromcisTEMStarFile(original_star_filename.c_str( ));
    test_parameters.WriteTocisTEMBinaryFile(binary_from_star_filename.c_str( ));

    // Check the sizes are the same - this isn't a very thorough test, but better than nothing.
    // It is at least an easy way to get files with all the parameters written out to quickly check by eye

    long original_star_size_in_bytes    = ReturnFileSizeInBytes(original_star_filename.c_str( ));
    long original_binary_size_in_bytes  = ReturnFileSizeInBytes(original_binary_filename.c_str( ));
    long star_from_binary_size_in_bytes = ReturnFileSizeInBytes(star_from_binary_filename.c_str( ));
    long binary_from_star_size_in_bytes = ReturnFileSizeInBytes(binary_from_star_filename.c_str( ));

    if ( original_star_size_in_bytes != star_from_binary_size_in_bytes )
        FailTest;
    if ( original_binary_size_in_bytes != binary_from_star_size_in_bytes )
        FailTest;

    // Check the star files are byte identical (apart from bytes in the text file which represent the write time)..
    // the binary files are not expected to be byte identical as there is a change in precision after being written out to star file.

    char original_star_file[original_star_size_in_bytes];
    char star_file_from_binary_file[star_from_binary_size_in_bytes];

    FILE* current_file;

    current_file = fopen(original_star_filename.c_str( ), "rb");
    fread(original_star_file, 1, original_star_size_in_bytes, current_file);
    fclose(current_file);

    current_file = fopen(star_from_binary_filename.c_str( ), "rb");
    fread(star_file_from_binary_file, 1, star_from_binary_size_in_bytes, current_file);
    fclose(current_file);

    // To avoid the first line which contains the cistem version and a timestamp, we look for the first line return.
    bool reading_first_line = true;
    for ( long byte_counter = 0; byte_counter < original_star_size_in_bytes; byte_counter++ ) {
        if ( reading_first_line ) {
            if ( original_star_file[byte_counter] == '\n' ) {
                reading_first_line = false;
            }
        }
        else {
            if ( original_star_file[byte_counter] != star_file_from_binary_file[byte_counter] ) {
                std::cerr << "failed on byte" << byte_counter << std::endl;
                FailTest;
            }
        }
    }

    EndTest( );
}

void MyTestApp::TestElectronExposureFilter( ) {
    // TODO: Depends on ZeroFloatArray, but this has no test.
    BeginTest("Test Electron Exposure Filter");

    /*
  Test exposure filter for a simple square image size, over all voltages and three pixel sizes.
  The "ground truth" values are taken from a print out using the code for electron_dose.* from commit 
  cdd0c04e984412661c983ab7176954093b502ad5 Dec 9, 2021
  using this line in the inner loop (once for odd once for even)
    for (auto & indx : indx_even)
    {
        Printf("%3.9f, ", dose_filter_odd[indx]);
    }
  */
    const int size_small = 1024;

    std::vector<float> accelerating_voltage_vector = {300.f, 200.f, 100.f}; // only three supported values.
    std::vector<float> pixel_size_vector           = {0.72, 1.0, 2.1}; // values not chosen for any  good reason.
    std::vector<int>   indx_even                   = {0, 13, size_small / 3, size_small - 1};
    std::vector<int>   indx_odd                    = {0, 13, (size_small + 1) / 3, size_small};

    std::vector<float> ground_truth_even    = {1.000000000, 0.929921567, 0.017324856, 0.010133515, 1.000000000, 0.958591580, 0.031609546, 0.015432503, 1.000000000, 0.987710178, 0.155882418, 0.065504745, 1.000000000, 0.913183212, 0.006285460, 0.003215143, 1.000000000, 0.948510230, 0.013328240, 0.005439333, 1.000000000, 0.984661400, 0.097948194, 0.033139121, 1.000000000, 0.872345626, 0.000488910, 0.000178414, 1.000000000, 0.923584640, 0.001513943, 0.000393374, 1.000000000, 0.977023780, 0.030387951, 0.005955920};
    std::vector<float> ground_truth_odd     = {1.000000000, 0.930029809, 0.017352197, 0.010144006, 1.000000000, 0.958656907, 0.031672075, 0.015455280, 1.000000000, 0.987729967, 0.156189293, 0.065646693, 1.000000000, 0.913316071, 0.006297863, 0.003219303, 1.000000000, 0.948590994, 0.013361209, 0.005449366, 1.000000000, 0.984686077, 0.098189279, 0.033228900, 1.000000000, 0.872536480, 0.000490361, 0.000178761, 1.000000000, 0.923702955, 0.001519577, 0.000394466, 1.000000000, 0.977060616, 0.030500494, 0.005980202};
    int                ground_truth_counter = 0;
    ElectronDose*      my_electron_dose;

    Image test_image_small_even, test_image_small_odd;
    test_image_small_even.Allocate(size_small, size_small, true);
    test_image_small_odd.Allocate(size_small + 1, size_small + 1, true);

    float* dose_filter_even = new float[test_image_small_even.real_memory_allocated / 2];
    float* dose_filter_odd  = new float[test_image_small_odd.real_memory_allocated / 2];

    for ( auto& acceleration_voltage : accelerating_voltage_vector ) {
        for ( auto& pixel_size : pixel_size_vector ) {
            my_electron_dose = new ElectronDose(acceleration_voltage, pixel_size);

            ZeroFloatArray(dose_filter_even, test_image_small_even.real_memory_allocated / 2);
            ZeroFloatArray(dose_filter_odd, test_image_small_odd.real_memory_allocated / 2);

            my_electron_dose->CalculateDoseFilterAs1DArray(&test_image_small_even, dose_filter_even, 15.f, 30.f);
            my_electron_dose->CalculateDoseFilterAs1DArray(&test_image_small_odd, dose_filter_odd, 15.f, 30.f);

            for ( auto& indx : indx_even ) {
                if ( ! FloatsAreAlmostTheSame(dose_filter_even[indx], ground_truth_even[ground_truth_counter]) ) {
                    Printf("Failed for kv,pix,ev: %3.f %3.3f, values %f %f\n", acceleration_voltage, pixel_size, dose_filter_even[indx], ground_truth_even[ground_truth_counter]);
                    FailTest;
                }
                if ( ! FloatsAreAlmostTheSame(dose_filter_odd[indx], ground_truth_odd[ground_truth_counter]) ) {
                    Printf("Failed for kv,pix,od: %3.f %3.3f, values %f %f\n", acceleration_voltage, pixel_size, dose_filter_odd[indx], ground_truth_odd[ground_truth_counter]);
                    FailTest;
                }
                ground_truth_counter++;
            }

            delete my_electron_dose;
        }
    }

    delete[] dose_filter_even;
    delete[] dose_filter_odd;

    EndTest( );
}

void MyTestApp::TestEmpiricalDistribution( ) {
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice"});

    BeginTest("Empirical Distribution");

    Image test_image;
    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);

    EmpiricalDistribution<double> my_dist = test_image.ReturnDistributionOfRealValues( );

    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ) + 1.f, 1.0f) )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), 1.0f) )
        FailTest;
    if ( my_dist.GetNumberOfSamples( ) != 6400 )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetMinimum( ), -3.1520f) )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetMaximum( ), 7.0222f) )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleSumOfSquares( ), 6400.f) )
        FailTest;

    EndTest( );
}

/** 
 * RAII helper to capture logging output into a buffer safely. If we expand usage, could move this to defines.
 * MyDebugWarnThreadSafety writes the warning to stderr, so stderr is what is redirected here; the redirection
 * is undone by StderrRedirect's destructor, so it is put back however call() returns.
 */
struct LogCaptureRAII {
    std::string buffer;

    LogCaptureRAII( ) {
    }

    // Disable copy, allow move if needed in future
    LogCaptureRAII(const LogCaptureRAII&)            = delete;
    LogCaptureRAII& operator=(const LogCaptureRAII&) = delete;

    // Sends everything written to stderr while it is alive to `text` instead.
    struct StderrRedirect {
        std::string& text;
        int          saved_descriptor;
        FILE*        capture_file;

        StderrRedirect(std::string& wanted_text) : text(wanted_text) {
            fflush(stderr);
            saved_descriptor = dup(fileno(stderr));
            capture_file     = tmpfile( );
            if ( capture_file != NULL )
                dup2(fileno(capture_file), fileno(stderr));
        }

        ~StderrRedirect( ) {
            fflush(stderr);
            if ( saved_descriptor >= 0 ) {
                dup2(saved_descriptor, fileno(stderr));
                close(saved_descriptor);
            }
            if ( capture_file != NULL ) {
                rewind(capture_file);
                char   read_buffer[1024];
                size_t bytes_read;
                while ( (bytes_read = fread(read_buffer, 1, sizeof(read_buffer), capture_file)) > 0 )
                    text.append(read_buffer, bytes_read);
                fclose(capture_file);
            }
        }
    };

    template <class Obj, class MemFn, class... Args>
    decltype(auto) call(Obj&& obj, MemFn memfn, Args&&... args) {
        clear( );
        StderrRedirect redirect(buffer);

        // memfn is a pointer-to-member (e.g., &T::foo)
        if constexpr ( std::is_void_v<std::invoke_result_t<MemFn, Obj, Args...>> ) {

            (std::forward<Obj>(obj).*memfn)(std::forward<Args>(args)...);
            return;
        }
        else {
            auto&& r = (std::forward<Obj>(obj).*memfn)(std::forward<Args>(args)...);
            return std::forward<decltype(r)>(r);
        }
    }

    // Check whether the captured buffer contains the given substring
    bool contains(const std::string& string_to_check) {
        return ! buffer.empty( ) && buffer.find(string_to_check) != std::string::npos;
    }

    void clear( ) {
        buffer.clear( );
    }
};

/**
 * @brief This is the prototypical case for functions we know not to be thread safe, which should print a warning if declared shared and not otherwise.
 * 
 */
void MyTestApp::TestEmpiricalDistributionThreadSafety( ) {
    BeginTest("Empirical Distribution Thread Safety Warnings");

    // Outside any parallel region
    EmpiricalDistribution<double> dist_outside; // Ensure we have a fresh instance for each test
    EmpiricalDistribution<double> dist_shared; // Ensure we have a fresh instance for each test
    EmpiricalDistribution<double> dist_private; // Ensure we have a fresh instance for each test

    // The constructed flag should match the runtime query
    if ( dist_outside.GetConstructedInParallelRegion( ) != ReturnInParallelRegionBool( ) )
        FailTest;

        // If we are in debug mode with OpenMP, capture a warning from inside a parallel region
#if defined(DEBUG) && defined(_OPENMP)
#pragma omp parallel num_threads(2)
    {
        // Inside vs outside: the flags should differ
        if ( dist_outside.GetConstructedInParallelRegion( ) == ReturnInParallelRegionBool( ) )
            FailTest;

// Capture exactly once to avoid interleaving from multiple threads
// We just want to know that we are in a parallel region but one thread exectuing is enough for the test.
#pragma omp single
        {
            // Trigger the warning without mutating shared state
            LogCaptureRAII cap;
            cap.call(dist_outside, &EmpiricalDistribution<double>::IsConstant);
            if ( ! cap.contains("Warning: Potential thread safety") )
                FailTest;
        }
    }

    // We shouldnot have the warning here because we called outside a parallel region
    {
        LogCaptureRAII cap;
        cap.call(dist_outside, &EmpiricalDistribution<double>::IsConstant);
        if ( cap.contains("Warning: Potential thread safety") )
            FailTest;
    }

// Now we'll test shared and private variables which should Fail and not fail respectively
#pragma omp parallel num_threads(2) shared(dist_shared) private(dist_private)
    {
#pragma omp single
        {
            {
                LogCaptureRAII cap;
                cap.call(dist_shared, &EmpiricalDistribution<double>::IsConstant);
                if ( ! cap.contains("Warning: Potential thread safety") )
                    FailTest;
            }

            {
                LogCaptureRAII cap;
                cap.call(dist_private, &EmpiricalDistribution<double>::IsConstant);
                if ( cap.contains("Warning: Potential thread safety") )
                    FailTest;
            }
        }
    }

#endif

    EndTest( );
}

void MyTestApp::TestSumOfSquaresFourierAndFFTNormalization( ) {
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});

    BeginTest("Sum Of Squares Fourier");

    Image test_image;
    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);
    test_image.ForwardFFT(false);

    // By Parsevals theorem, the sum of squares of the Fourier transform of an image is equal to the sum of squares of the real image.
    // On the foward FFT the variance is scaled by N and on the inverse by N again. (This makes it as though the original values were scaled by N round-trip.)
    // So without normalization, the sum of squares should be N * realspace sumof squares.
    float sum_of_squares = test_image.ReturnSumOfSquares( );
    if ( ! RelativeErrorIsLessThanEpsilon(sum_of_squares, 6400.f * 6400.f) )
        FailTest;

    // We normalize for the full round trip on the forward FFT, so in this case the sum of squares should be
    // realspace sumof squares / N.
    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);
    test_image.ForwardFFT(true);
    sum_of_squares = test_image.ReturnSumOfSquares( );
    if ( ! RelativeErrorIsLessThanEpsilon(sum_of_squares, 1.f) )
        FailTest;

    // To check the 2d and 3d cases, we can also use some toy images.
    std::array<int, 2>  test_sizes = {6, 7};
    std::array<bool, 2> test_3d    = {false, true};
    int                 size_3d;
    constexpr float     img_values = 1.0f;
    for ( auto& size : test_sizes ) {
        for ( auto& do3d : test_3d ) {
            if ( do3d )
                size_3d = size;
            else
                size_3d = 1;
            test_image.Allocate(size, size, size_3d);
            // Make sure the meta data is correctly reset:
            test_image.object_is_centred_in_box = true;
            test_image.SetToConstant(0.0f);
            int ox, oy, oz;
            ox = test_image.physical_address_of_box_center_x;
            oy = test_image.physical_address_of_box_center_y;
            oz = test_image.physical_address_of_box_center_z;

            // Set a unit impulse at the centered in the box origin and a cross in the xy plane
            float sum = 0.f;

            test_image.real_values[test_image.ReturnReal1DAddressFromPhysicalCoord(ox, oy, oz)] = img_values;
            sum += img_values;
            for ( int i = -2; i < 3; i += 4 ) {
                for ( int j = -2; j < 3; j += 4 ) {
                    test_image.real_values[test_image.ReturnReal1DAddressFromPhysicalCoord(ox + i, oy + j, oz)] = img_values;
                    sum += img_values;
                }
            }
            // Don't do any scaling as it doesn't matter anyway since we'll use ReturnSumOfSquares to set the power to 1
            test_image.ForwardFFT(false);
            // We need to scale by 1/root(N) otherwise the we will get Sqrt(N) as the sum of squares
            test_image.DivideByConstant(sqrtf(test_image.ReturnSumOfSquares( ) * test_image.number_of_real_space_pixels));
            test_image.BackwardFFT( );
            // ReturnSumOfSquares in real space actually returns SumOfSquares/N.
            if ( ! FloatsAreAlmostTheSame(test_image.ReturnSumOfSquares( ) * test_image.number_of_real_space_pixels, 1.f) )
                FailTest;
        }
    }
    EndTest( );
}

void MyTestApp::TestRandomVariableFunctions( ) {
    BeginTest("Random Variable Functions");

    float acceptable_error = 0.025f; // this is a random sample so it won't be so exact.

    // We want a reasonably large image to ensure that we sample the distribution well
    Image test_image;
    test_image.Allocate(1024, 1024, 1, true, false);
    EmpiricalDistribution<double> my_dist = test_image.ReturnDistributionOfRealValues( );

    // Test multiple configurations of Gaussian noise, mean/sd 0,1 -2,1 3.1,4
    const std::vector<float> test_normal_vals{0.f, 1.f, -2.f, 1.f, 3.1f, 4.f};
    for ( int i = 0; i < test_normal_vals.size( ); i += 2 ) {
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoise(GAUSSIAN, test_normal_vals[i], test_normal_vals[i + 1]);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        // Avoid zero division by adding 1
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ) + 1.f, test_normal_vals[i] + 1.f, true, acceptable_error) )
            FailTest;
        if ( ! RelativeErrorIsLessThanEpsilon(sqrtf(my_dist.GetSampleVariance( )), test_normal_vals[i + 1], true, acceptable_error) )
            FailTest;
    }

    // Test multiple configurations of Poisson noise, mean/sd 0.1,1.2,4.0
    // mean and variance should be ~ equal
    const std::vector<float> test_poisson_vals{0.1, 1.2, 4.0};
    for ( auto& val : test_poisson_vals ) {
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoise(POISSON, val);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ), val, true, acceptable_error) )
            FailTest;
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), val, true, acceptable_error) )
            FailTest;
    }

    // Test uniform distribution, defined by its max, min and mean should be close to zero
    const float uniform_min = -1.f;
    const float uniform_max = 1.f;
    test_image.SetToConstant(0.f);
    my_dist.Reset( );
    test_image.AddNoise(UNIFORM, uniform_min, uniform_max);
    test_image.UpdateDistributionOfRealValues(&my_dist);
    // Avoid zero division by adding 1
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ) + 1.f, 1.f, true, acceptable_error) )
        FailTest;
    if ( my_dist.GetMinimum( ) < uniform_min )
        FailTest;
    if ( my_dist.GetMaximum( ) > uniform_max )
        FailTest;

    // Test the exponential distribution, mean = 1/lambda, sd = 1/lambda,
    // Re-use the poisson test values
    for ( auto& val : test_poisson_vals ) {
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoise(EXPONENTIAL, val);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ), 1.f / val, true, acceptable_error) )
            FailTest;
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), 1.f / (val * val), true, acceptable_error) )
            FailTest;
    }

    // Test the gamma distribution, mean = alpha*theta, variance = alpha*theta^2
    // Re-use the poisson test values
    float beta;
    for ( auto& alpha : test_poisson_vals ) {
        beta = alpha + 1.f;
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoise(GAMMA, alpha, beta);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ), alpha * beta, true, 2.0f * acceptable_error) ) {
            Printf("m,a/b %f %f\n", my_dist.GetSampleMean( ), alpha * beta);
            FailTest;
        }
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), alpha * beta * beta, true, 2.0f * acceptable_error) )
            FailTest;
    }

    EndTest( );
}

void MyTestApp::TestIntegerShifts( ) {
    // Dependencies?
    BeginTest("Integer Shifts and Rotations");
    CheckDependencies({"Memory Assignment Ops and Funcs"});

    // TODO Image::RotateQuadrants should be tested, but I'm not clear on what it is supposed to do.

    // Goal is to verify image transforms that work without interpolation.
    const int eve_size = 16;
    const int odd_size = 17;

    Image odd_image, eve_image;
    odd_image.Allocate(odd_size, odd_size, 1, true, false);
    eve_image.Allocate(eve_size, eve_size, 1, true, false);

    // Set values around the center, which is also the origin for rotations.
    const int ocx = odd_image.physical_address_of_box_center_x;
    const int ocy = odd_image.physical_address_of_box_center_y;
    const int ecx = eve_image.physical_address_of_box_center_x;
    const int ecy = eve_image.physical_address_of_box_center_y;

    // Locations and test values.
    std::vector<float> tv = {1.f, 2.f, 3.f, 4.f};
    std::vector<int>   sx = {2, 0, -2, 0}; // x shift
    std::vector<int>   sy = {0, 2, 0, -2}; // y shift

    std::vector<int> addr_odd = {0, 0, 0, 0};
    std::vector<int> addr_eve = {0, 0, 0, 0};

    const int origin_addr_odd = odd_image.ReturnReal1DAddressFromPhysicalCoord(ocx, ocy, 0);
    const int origin_addr_eve = eve_image.ReturnReal1DAddressFromPhysicalCoord(ecx, ecy, 0);
    float     tmp             = 0.f;
    for ( int i = 0; i < addr_odd.size( ); i++ ) {
        addr_odd[i] = odd_image.ReturnReal1DAddressFromPhysicalCoord(ocx + sx[i], ocy + sy[i], 0);
        addr_eve[i] = eve_image.ReturnReal1DAddressFromPhysicalCoord(ecx + sx[i], ecy + sy[i], 0);
        tmp += tv[i];
    }

    // make sure we don't change the value for the origin
    const float origin_val = tmp;

    /*
      y ->
   x 
   |  . . .0 0 3 0 0
   V  . . .0 0 0 0 0
      . . .4 0 6 0 2
      . . .0 0 0 0 0
      . . .0 0 1 0 0

      Note that for an even sized image, the origin is effectively shifted by -1 in x on a +90 rotation
      This is an "artifact" of defining the origin based on the pixel grid and not the underlying image data.
      0 0     0 6
      0 6  -> 0 0
  */

    odd_image.SetToConstant(0.f);
    eve_image.SetToConstant(0.f);
    odd_image.real_values[origin_addr_odd] = origin_val;
    eve_image.real_values[origin_addr_eve] = origin_val;
    int from_addr;
    for ( int i = 0; i < tv.size( ); i++ ) {
        odd_image.real_values[addr_odd[i]] = tv[i];
        eve_image.real_values[addr_eve[i]] = tv[i];
    }

    Image odd_test_image, eve_test_image;
    odd_test_image.CopyFrom(&odd_image);
    eve_test_image.CopyFrom(&eve_image);

    bool rotate_by_positive_90_degrees = true;
    bool preserve_origin               = false;
    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve - 1] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = (i + tv.size( ) + 1) % tv.size( );
        if ( odd_test_image.real_values[addr_odd[from_addr]] != tv[i] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[from_addr] - 1] != tv[i] )
            FailTest;
    }

    // Test rotating back by -90 degress, everything should be identical to the starting conditions.
    rotate_by_positive_90_degrees = false;
    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = i;
        if ( odd_test_image.real_values[addr_odd[from_addr]] != tv[i] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[from_addr]] != tv[i] )
            FailTest;
    }

    // Test forward rotation with a integer base shift (shouldn't need offsets in Y)
    rotate_by_positive_90_degrees = true;
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    eve_test_image.RealSpaceIntegerShift(-1, 0, 0);

    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = (i + tv.size( ) + 1) % tv.size( );
        if ( eve_test_image.real_values[addr_eve[from_addr]] != tv[i] )
            FailTest;
    }

    // Test combined shift and rotate, from a clean copy
    preserve_origin = true;
    odd_test_image.CopyFrom(&odd_image);
    eve_test_image.CopyFrom(&eve_image);

    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);

    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;

    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = (i + tv.size( ) + 1) % tv.size( );
        if ( odd_test_image.real_values[addr_odd[from_addr]] != tv[i] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[from_addr]] != tv[i] )
            FailTest;
    }

    // Test combined shift and inverse rotate
    rotate_by_positive_90_degrees = false;
    odd_test_image.CopyFrom(&odd_image);
    eve_test_image.CopyFrom(&eve_image);
    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);

    // origin should be the same, pixel values should be permuted one value clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        // Note the difference in address here.
        from_addr = (i + 1) % tv.size( );
        if ( odd_test_image.real_values[addr_odd[i]] != tv[from_addr] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[i]] != tv[from_addr] )
            FailTest;
    }

    EndTest( );
}

void MyTestApp::TestImageLoopingAndAddressing( ) {
    BeginTest("Image::LoopingAndAddressing");

    Image test_image;

    //
    // Even
    //
    test_image.Allocate(4, 4, 4, true);

    if ( test_image.physical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_y != 3 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_z != 3 )
        FailTest;

    if ( test_image.physical_address_of_box_center_x != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_y != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_z != 2 )
        FailTest;

    if ( test_image.physical_index_of_first_negative_frequency_y != 2 )
        FailTest;
    if ( test_image.physical_index_of_first_negative_frequency_z != 2 )
        FailTest;

    if ( test_image.logical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_y != 1 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_z != 1 )
        FailTest;

    if ( test_image.logical_lower_bound_complex_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_z != -2 )
        FailTest;

    if ( test_image.logical_upper_bound_real_x != 1 )
        FailTest;
    if ( test_image.logical_upper_bound_real_y != 1 )
        FailTest;
    if ( test_image.logical_upper_bound_real_z != 1 )
        FailTest;

    if ( test_image.logical_lower_bound_real_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_z != -2 )
        FailTest;

    if ( test_image.padding_jump_value != 2 )
        FailTest;

    //
    // Odd
    //
    test_image.Allocate(5, 5, 5, true);

    if ( test_image.physical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_y != 4 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_z != 4 )
        FailTest;

    if ( test_image.physical_address_of_box_center_x != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_y != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_z != 2 )
        FailTest;

    if ( test_image.physical_index_of_first_negative_frequency_y != 3 )
        FailTest;
    if ( test_image.physical_index_of_first_negative_frequency_z != 3 )
        FailTest;

    if ( test_image.logical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_y != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_z != 2 )
        FailTest;

    if ( test_image.logical_lower_bound_complex_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_z != -2 )
        FailTest;

    if ( test_image.logical_upper_bound_real_x != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_real_y != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_real_z != 2 )
        FailTest;

    if ( test_image.logical_lower_bound_real_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_z != -2 )
        FailTest;

    if ( test_image.padding_jump_value != 1 )
        FailTest;

    EndTest( );
}

void MyTestApp::TestSpectrumBoxConvolution( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});

    BeginTest("Image::SpectrumBoxConvolution");

    Image test_image;
    Image output_image;

    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);

    output_image.Allocate(test_image.logical_x_dimension, test_image.logical_y_dimension, test_image.logical_z_dimension);
    test_image.SpectrumBoxConvolution(&output_image, 7, 3);

    if ( FloatsAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.049189) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 1.634473) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), -0.049189) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestImageArithmeticFunctions( ) {
    CheckDependencies({"MRCFile::ReadSlice", "MRCFile::OpenFile"});
    // AddImage
    BeginTest("Image::AddImage");

    Image test_image;
    Image ref_image;
    Peak  my_peak;

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename, 1);
    ref_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename, 2);
    test_image.AddImage(&ref_image);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -1.313164) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 3.457573) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.318875) == false )
        FailTest;

    EndTest( );

    BeginTest("Image::SetToConstant");
    // SetToConstant, also sets FFTW padding addresses as well
    test_image.SetToConstant(3.14f);
    for ( long pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
        if ( test_image.real_values[pixel_counter] != 3.14f )
            FailTest;
    }
    EndTest( );
}

void MyTestApp::TestNumericTextFiles( ) {
    // AddImage
    BeginTest("NumericTextFile::Init");

    for ( int with_default_constructor = 0; with_default_constructor < 2; with_default_constructor++ ) {
        if ( with_default_constructor > 0 ) {
            NumericTextFile test_file;
            test_file.Open(numeric_text_filename, OPEN_TO_READ);
            if ( test_file.number_of_lines != 4 )
                FailTest;
            if ( test_file.records_per_line != 5 )
                FailTest;
        }
        else {
            NumericTextFile test_file(numeric_text_filename, OPEN_TO_READ);
            if ( test_file.number_of_lines != 4 )
                FailTest;
            if ( test_file.records_per_line != 5 )
                FailTest;
        }
    }
    EndTest( );

    NumericTextFile test_file(numeric_text_filename, OPEN_TO_READ);

    BeginTest("NumericTextFile::ReadLine");
    std::array<float, 5>                temp_float;
    std::array<double, 5>               temp_double;
    std::array<std::array<float, 5>, 4> line_values;
    line_values[0] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    line_values[1] = {6.0f, 7.1f, 8.3f, 9.4f, 10.5f};
    line_values[2] = {11.2f, 12.7f, 13.2f, 14.1f, 15.8f};
    line_values[3] = {16.1245f, 17.81003f, 18.5467f, 19.7621f, 20.11111f};

    for ( int line = 0; line < line_values.size( ); line++ ) {
        test_file.ReadLine(temp_float.data( ));
        for ( int i = 0; i < line_values[line].size( ); i++ ) {
            if ( FloatsAreAlmostTheSame(temp_float[i], line_values[line][i]) == false )
                FailTest;
        }
    }

    EndTest( );

    BeginTest("NumericTextFile::WriteLine float");

    std::string        output_filename = temp_directory + "/number_out.num";
    NumericTextFile output_test_file(output_filename, OPEN_TO_WRITE, 5);

    for ( int i = 0; i < line_values[0].size( ); i++ ) {
        temp_float[i] = line_values[0][i];
    }
    output_test_file.WriteLine(temp_float.data( ));
    output_test_file.WriteCommentLine("This is a comment line %i", 5);

    for ( int i = 0; i < line_values[1].size( ); i++ ) {
        temp_float[i] = line_values[1][i];
    }

    output_test_file.WriteLine(temp_float.data( ));
    output_test_file.WriteCommentLine("Another comment = %s", "booooo!");

    output_test_file.Flush( );

    output_test_file.Close( );
    output_test_file.Open(output_filename, OPEN_TO_READ);
    // We only expect non-comment lines to be counted.
    if ( output_test_file.number_of_lines != 2 )
        FailTest;
    if ( output_test_file.records_per_line != 5 )
        FailTest;

    for ( int line = 0; line < 2; line++ ) {
        output_test_file.ReadLine(temp_float.data( ));
        for ( int i = 0; i < line_values[line].size( ); i++ ) {
            if ( FloatsAreAlmostTheSame(temp_float[i], line_values[line][i]) == false )
                FailTest;
        }
    }
    output_test_file.Close( );
    EndTest( );

    BeginTest("NumericTextFile::WriteLine double");

    output_filename = temp_directory + "/number_out.num";
    output_test_file.Open(output_filename, OPEN_TO_WRITE, 5);

    for ( int i = 0; i < line_values[0].size( ); i++ ) {
        temp_double[i] = line_values[0][i];
    }
    output_test_file.WriteLine(temp_double.data( ));
    output_test_file.WriteCommentLine("This is a comment line %i", 5);

    for ( int i = 0; i < line_values[1].size( ); i++ ) {
        temp_double[i] = line_values[1][i];
    }

    output_test_file.WriteLine(temp_double.data( ));
    output_test_file.WriteCommentLine("Another comment = %s", "booooo!");

    output_test_file.Flush( );

    output_test_file.Close( );
    output_test_file.Open(output_filename, OPEN_TO_READ);

    if ( output_test_file.number_of_lines != 2 )
        FailTest;
    if ( output_test_file.records_per_line != 5 )
        FailTest;

    for ( int line = 0; line < 2; line++ ) {
        output_test_file.ReadLine(temp_float.data( ));
        for ( int i = 0; i < line_values[line].size( ); i++ ) {
            if ( FloatsAreAlmostTheSame(temp_float[i], line_values[line][i]) == false )
                FailTest;
        }
    }
    output_test_file.Close( );
    EndTest( );
}

void MyTestApp::TestAlignmentFunctions( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});
    // TODO: CalculateCrossCorrelationImageWith depends on SwapRealSpaceQuadrants, which depends on PhaseShift

    // Phaseshift
    BeginTest("Image::PhaseShift");

    Image test_image;
    Image ref_image;
    Peak  my_peak;

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename, 1);
    test_image.PhaseShift(20, 20, 0);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -1.010296) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), -2.280109) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.239702) == false )
        FailTest;

    EndTest( );

    // SwapRealSpaceQuadrants
    BeginTest("Image::SwapRealSpaceQuadrants");
    // Test for even/odd and 2 and 3D images
    std::array<int, 2>  test_sizes = {4, 5};
    std::array<bool, 2> test_3d    = {false, true};
    int                 size_z     = 0;
    for ( auto& size : test_sizes ) {
        for ( auto& is_3d : test_3d ) {
            if ( is_3d )
                size_z = size;
            else
                size_z = 1;
            test_image.Allocate(size, size, size_z);
            test_image.SetToConstant(0.0f);
            // Set a unit impulse at the centered in the box origin
            test_image.real_values[test_image.ReturnReal1DAddressFromPhysicalCoord(test_image.physical_address_of_box_center_x, test_image.physical_address_of_box_center_y, test_image.physical_address_of_box_center_z)] = 1.0f;
            test_image.SwapRealSpaceQuadrants( );
            if ( ! FloatsAreAlmostTheSame(test_image.real_values[0], 1.0f) )
                FailTest;
            // Swapping back should put the impulse back in the center
            test_image.SwapRealSpaceQuadrants( );
            if ( ! FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(test_image.physical_address_of_box_center_x, test_image.physical_address_of_box_center_y, test_image.physical_address_of_box_center_z), 1.0f) )
                FailTest;
        }
    }
    EndTest( );

    // A bare minimal test to make sure the origin of rotation is as expected.
    BeginTest("Image::ExtractSlice");
    Image test_vol;

    // Test for even/odd and 2 and 3D images
    AnglesAndShifts     test_extract_angles(90.f, 0.f, 0.f, 0.f, 0.f);
    std::complex<float> origin_value;
    for ( auto& size : test_sizes ) {
        size += 2;
        test_image.Allocate(size, size, 1);
        int size_3d = 8 * size;
        if ( IsOdd(size) )
            size_3d++;
        test_vol.Allocate(size_3d, size_3d, size_3d);
        // Make sure the meta data is correctly reset:
        test_vol.object_is_centred_in_box   = true;
        test_image.object_is_centred_in_box = true;
        test_image.SetToConstant(0.0f);
        test_vol.SetToConstant(0.0f);
        int ox, oy, oz;
        ox = test_vol.physical_address_of_box_center_x;
        oy = test_vol.physical_address_of_box_center_y;
        oz = test_vol.physical_address_of_box_center_z;
        // Set a unit impulse at the centered in the box origin and a cross in the xy plane

        float sum = 0.f;

        test_vol.real_values[test_vol.ReturnReal1DAddressFromPhysicalCoord(ox, oy, oz)] = 1.0f;
        sum += 1.f;
        for ( int i = -2; i < 3; i += 4 ) {
            for ( int j = -2; j < 3; j += 4 ) {
                test_vol.real_values[test_vol.ReturnReal1DAddressFromPhysicalCoord(ox + i, oy + j, oz)] = 1.0f;
                sum += 1.f;
            }
        }

        test_vol.ForwardFFT(false);

        test_vol.SwapRealSpaceQuadrants( );
        test_vol.ExtractSlice(test_image, test_extract_angles, 0., false);
        test_image.SwapRealSpaceQuadrants( );
        test_image.BackwardFFT( );

        // Leaving for notes:
        // An un-normalized padded 3d FT, projection, removal of zero pixel, padded back 2d fft,
        // crop, normalized forward 2d, un-normalized back 2d FFT results in a total change in power as below.
        // float p_   = test_vol.ReturnSumOfSquares( ) * test_vol.number_of_real_space_pixels;
        // float sum_ = test_vol.ReturnSumOfRealValues( );
        // float n2_  = float(size_3d * size_3d);
        // float n3_  = float(size_3d * size_3d * size_3d);
        // // I needed an extra sqrt(n2_) here ?
        // float p_out_calc_ = powf(n2_, 1.5f) * (n3_ * p_ - sum_ * sum_);

        EmpiricalDistribution<double> test_dist;
        test_image.UpdateDistributionOfRealValues(&test_dist);
        float scale = test_dist.GetMaximum( );
        test_image.MultiplyByConstant(1.f / scale);
        // there is a sinc and some power loss during the cropping so it won't be perfectly 1
        for ( int i = 0; i < test_image.real_memory_allocated; i++ )
            test_image.real_values[i] = roundf(test_image.real_values[i]);
        ox = test_image.physical_address_of_box_center_x;
        oy = test_image.physical_address_of_box_center_y;
        oz = test_image.physical_address_of_box_center_z;
        if ( ! FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(ox, oy, 0), 1.0f) ) {
            FailTest;
        }
        for ( int i = -2; i < 3; i += 4 ) {
            for ( int j = -2; j < 3; j += 4 ) {
                if ( ! FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(ox + i, oy + j, 0), 1.0f) ) {
                    FailTest;
                }
            }
        }
    }

    EndTest( );

    // CalculateCrossCorrelationImageWith
    BeginTest("Image::CalculateCrossCorrelationImageWith");

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename, 1);
    ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);
    test_image.CalculateCrossCorrelationImageWith(&ref_image);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 0.004323) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 0.543692) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.006927) == false )
        FailTest;

    EndTest( );

    //FindPeakWithIntegerCoordinates

    BeginTest("Image::FindPeakWithIntegerCoordinates");

    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);
    ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);
    test_image.PhaseShift(7, 10, 0);

    test_image.CalculateCrossCorrelationImageWith(&ref_image);
    my_peak = test_image.FindPeakWithIntegerCoordinates( );

    if ( FloatsAreAlmostTheSame(my_peak.x, 7.0) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.y, 10.0) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.z, 0) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.value, 1) == false )
        FailTest;

    EndTest( );

    //FindPeakWithParabolaFit

    BeginTest("Image::FindPeakWithParabolaFit");

    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);
    ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename, 1);
    test_image.PhaseShift(7.3, 10.7, 0);

    test_image.CalculateCrossCorrelationImageWith(&ref_image);
    my_peak = test_image.FindPeakWithParabolaFit( );

    if ( my_peak.x > 7.3 || my_peak.x < 7.29 )
        FailTest;
    if ( my_peak.y > 10.70484 || my_peak.y < 10.70481 )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.z, 0) == false )
        FailTest;
    if ( my_peak.value > 0.99343 || my_peak.value < 0.99342 )
        FailTest;

    EndTest( );
}

void MyTestApp::TestFilterFunctions( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});

    // BFACTOR
    BeginTest("Image::ApplyBFactor");

    Image test_image;

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename, 1);
    test_image.ForwardFFT( );
    test_image.ApplyBFactor(1500);
    test_image.BackwardFFT( );

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 0.027244) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 1.320998) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.012282) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestMaskCentralCross( ) {
    CheckDependencies({"Image::SetToConstant"});

    BeginTest("Image::MaskCentralCross");

    Image my_image;

    my_image.Allocate(128, 128, 1);
    my_image.SetToConstant(1.0);
    my_image.MaskCentralCross(3, 3);

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(127, 127, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 127, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(127, 0, 0), 1.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67, 67, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67, 61, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61, 61, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61, 67, 0), 1.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 61, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 67, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61, 0, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67, 0, 0), 1.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66, 66, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66, 62, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62, 62, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62, 66, 0), 0.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 62, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 66, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62, 0, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66, 0, 0), 0.0) )
        FailTest;

    EndTest( );
}

void MyTestApp::TestScalingAndSizingFunctions( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Memory Assignment Ops and Funcs"});

    BeginTest("Image::ClipInto");

    MRCFile             input_file(hiv_images_80x80x10_filename, false);
    Image               test_image;
    Image               clipped_image;
    std::complex<float> test_pixel;

    // test real space clipping bigger..

    clipped_image.Allocate(160, 160, 1);

    test_image.ReadSlice(&input_file, 1);
    test_image.ClipInto(&clipped_image, 0);

    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), -0.340068) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(80, 80, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(119, 119, 0), 0.637069) == false )
        FailTest;

    // test real space clipping smaller..

    clipped_image.Allocate(50, 50, 1);
    test_image.ClipInto(&clipped_image, 0);

    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -2.287762) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(49, 49, 0), -1.773780) == false )
        FailTest;

    // test Fourier space clipping bigger

    clipped_image.Allocate(160, 160, 1);
    test_image.ForwardFFT( );
    test_image.ClipInto(&clipped_image, 0);

    // check some values

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -100.0) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    // test Fourier space clipping smaller

    clipped_image.Allocate(50, 50, 1);
    test_image.ClipInto(&clipped_image, 0);

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    // test real space clipping smaller to odd..

    test_image.ReadSlice(&input_file, 1);
    clipped_image.Allocate(49, 49, 1);
    test_image.ClipInto(&clipped_image, 0);

    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.391899) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.689942) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(48, 48, 0), -1.773780) == false )
        FailTest;

    // test fourier space flipping smaller to odd..

    test_image.ForwardFFT( );
    test_image.ClipInto(&clipped_image, 0);

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    EndTest( );

    // Check Resize..

    BeginTest("Image::Resize");

    test_image.ReadSlice(&input_file, 1);
    test_image.Resize(160, 160, 1);

    //Real space big

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), -0.340068) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(80, 80, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(119, 119, 0), 0.637069) == false )
        FailTest;

    // Real space small

    test_image.ReadSlice(&input_file, 1);
    test_image.Resize(50, 50, 1);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -2.287762) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(49, 49, 0), -1.773780) == false )
        FailTest;

    // Fourier space big

    test_image.ReadSlice(&input_file, 1);
    test_image.ForwardFFT( );
    test_image.Resize(160, 160, 1);

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -100.0) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    // Fourier space small

    test_image.ReadSlice(&input_file, 1);
    test_image.ForwardFFT( );
    test_image.Resize(50, 50, 1);

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestMRCFunctions( ) {
    BeginTest("MRCFile::OpenFile");

    MRCFile input_file(hiv_images_80x80x10_filename, false);

    // check dimensions..

    if ( input_file.ReturnNumberOfSlices( ) != 10 )
        FailTest;
    if ( input_file.ReturnXSize( ) != 80 )
        FailTest;
    if ( input_file.ReturnYSize( ) != 80 )
        FailTest;
    if ( input_file.ReturnZSize( ) != 10 )
        FailTest;

    EndTest( );

    BeginTest("MRCFile::ReadSlice");

    Image test_image;
    test_image.ReadSlice(&input_file, 1);

    // check dimensions and type

    if ( test_image.is_in_real_space == false )
        FailTest;
    if ( test_image.logical_x_dimension != 80 )
        FailTest;
    if ( test_image.logical_y_dimension != 80 )
        FailTest;
    if ( test_image.logical_z_dimension != 1 )
        FailTest;

    // check first and last pixel...

    if ( FloatsAreAlmostTheSame(test_image.real_values[0], -0.340068) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestFFTFunctions( ) {

    long counter;

    BeginTest("Image::ForwardFFT");
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::SetToConstant"});

    MRCFile input_file(sine_wave_128x128x1_filename, false);

    // make an image that is all 1..

    Image test_image;
    test_image.Allocate(64, 64, 1);
    test_image.SetToConstant(1);

    // ForwardFFT

    test_image.ForwardFFT( );

    // first pixel should be 1,0

    if ( FloatsAreAlmostTheSame(real(test_image.complex_values[0]), 1) == false || FloatsAreAlmostTheSame(imag(test_image.complex_values[0]), 0) == false )
        FailTest;

    // if we set this to 0,0 - all remaining pixels should now be 0

    test_image.complex_values[0] = 0.0f + 0.0f * I;

    for ( counter = 0; counter < test_image.real_memory_allocated / 2; counter++ ) {
        if ( FloatsAreAlmostTheSame(real(test_image.complex_values[counter]), 0) == false || FloatsAreAlmostTheSame(imag(test_image.complex_values[counter]), 0) == false )
            FailTest;
    }

    // sine wave

    test_image.ReadSlice(&input_file, 1);
    test_image.ForwardFFT( );

    // now one pixel should be set, and the rest should be 0..

    if ( FloatsAreAlmostTheSame(real(test_image.complex_values[20]), 0) == false || FloatsAreAlmostTheSame(imag(test_image.complex_values[20]), -5) == false )
        FailTest;
    // set it to 0, then everything should be zero..

    test_image.complex_values[20] = 0.0f + 0.0f * I;

    for ( counter = 0; counter < test_image.real_memory_allocated / 2; counter++ ) {
        if ( real(test_image.complex_values[counter]) > 0.000001 || imag(test_image.complex_values[counter]) > 0.000001 )
            FailTest;
    }

    EndTest( );

    // Backward FFT

    BeginTest("Image::BackwardFFT");

    test_image.Allocate(64, 64, 1, false);
    test_image.SetToConstant(0.0);
    test_image.complex_values[0] = 1.0f + 0.0f * I;
    test_image.BackwardFFT( );
    test_image.RemoveFFTWPadding( );

    for ( counter = 0; counter < test_image.logical_x_dimension * test_image.logical_y_dimension; counter++ ) {
        if ( FloatsAreAlmostTheSame(test_image.real_values[counter], 1.0) == false )
            FailTest;
    }

    EndTest( );
}

void MyTestApp::TestRunProfileDiskOperations( ) {
    BeginTest("RunProfileManager Disk Operations");

    RunProfileManager run_profile_manager;

    // Add run profiles
    run_profile_manager.AddDefaultLocalProfile( );
    run_profile_manager.AddBlankProfile( );

    int num                                                   = run_profile_manager.number_of_run_profiles;
    run_profile_manager.run_profiles[num - 1].name            = Format("This_is_a_name_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
    run_profile_manager.run_profiles[num - 1].manager_command = Format("This_is_a $command string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
    run_profile_manager.run_profiles[num - 1].name            = Format("This_is_a_name_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
    run_profile_manager.run_profiles[num - 1].AddCommand("$command", 2, 1, false, 0, 10);

    // Write out to disk
    temp_directory = ReturnTempDirectory();

    std::vector<int> to_write;
    to_write.push_back(0);
    to_write.push_back(1);

    run_profile_manager.WriteRunProfilesToDisk(temp_directory + "/run_profiles.txt", to_write);

    // Ensure that run profiles are equal upon reading them back in
    RunProfileManager run_profile_manager2;
    run_profile_manager2.ImportRunProfilesFromDisk(temp_directory + "/run_profiles.txt");

    if ( run_profile_manager2.number_of_run_profiles != run_profile_manager.number_of_run_profiles ) {
        FailTest;
    }

    if ( run_profile_manager2.run_profiles[0] != run_profile_manager.run_profiles[0] ) {
        FailTest;
    }

    if ( run_profile_manager2.run_profiles[1] != run_profile_manager.run_profiles[1] ) {
        FailTest;
    }

    // Ensure that the equality operator works by changing the run profiles
    run_profile_manager2.run_profiles[0].name = "This is a new name";
    run_profile_manager2.run_profiles[1].AddCommand("$command", 2, 1, false, 0, 10);

    if ( run_profile_manager2.run_profiles[0] == run_profile_manager.run_profiles[0] ) {
        FailTest;
    }

    if ( run_profile_manager2.run_profiles[1] == run_profile_manager.run_profiles[1] ) {
        FailTest;
    }

    run_profile_manager2.AddBlankProfile( );
    run_profile_manager2.AddBlankProfile( );
    run_profile_manager2.AddBlankProfile( );
    run_profile_manager2.AddBlankProfile( );

    EndTest( );
}

void MyTestApp::TestCTFNodes( ) {
    BeginTest("CTF Nodes");

    Curve ctf_curve1;
    Curve ctf_curve2;
    ctf_curve1.SetupXAxis(0.0, 0.5, 500);
    ctf_curve2.SetupXAxis(0.0, 0.5, 500);

    CTF ctf1(300, 2.7, 0.07, 5000, 5000, 0, 1.0, 0.0);

    ctf_curve1.SetYToConstant(1.0);
    ctf_curve2.SetYToConstant(1.0);
    ctf_curve1.ApplyCTF(ctf1);
    // Generate Powerspectrum
    ctf_curve1.MultiplyBy(ctf_curve1);
    ctf_curve2.ApplyPowerspectrumWithThickness(ctf1);
    if ( ctf_curve1.YIsAlmostEqual(ctf_curve2) == false ) {
        // This is to override a failure, which occurs randomly when using gcc
        // There is probably some undefined behaviour in the code somewhere
        SkipTest;
    }

    CTF ctf2;
    // CTF with a sample thickness parameter of 100.0
    ctf2.Init(300, 2.7, 0.07, 5000, 5000, 0, 1.0, 0.0, 100.0);

    ctf_curve1.SetYToConstant(1.0);
    ctf_curve2.SetYToConstant(1.0);
    ctf_curve1.ApplyCTF(ctf2);
    // Generate powerspectrum
    ctf_curve1.MultiplyBy(ctf_curve1);

    ctf_curve2.ApplyPowerspectrumWithThickness(ctf2);

    // CTF is different when thickness is 100
    if ( ctf_curve1.YIsAlmostEqual(ctf_curve2) == true ) {
        // This is to override a failure, which occurs randomly when using gcc
        // There is probably some undefined behaviour in the code somewhere
        SkipTest;
    }

    // Test manually integrating ctf and compare with thickness formula
    ctf_curve1.SetYToConstant(0.0);
    Curve ctf_curve3;
    ctf_curve3.SetupXAxis(0.0, 0.5, 500);
    int counter = 0;

    for ( float z_level = -495.0; z_level < 500.0; z_level = z_level + 10.0f ) {
        ctf1.Init(300, 2.7, 0.07, 5000 + z_level, 5000 + z_level, 0, 1.0, 0.0, 0.0);
        counter++;
        ctf_curve3.SetYToConstant(1.0);
        ctf_curve3.ApplyCTF(ctf1);
        ctf_curve3.MultiplyBy(ctf_curve3);
        ctf_curve1.AddWith(&ctf_curve3);
    }

    // Now want to compare ctf_curve1 with ctf_curve2, but FloatIsAlmostEqual is to stringent.
    // The code below is a hack to get around this. Ideally, FloatIsAlmostEqual should be modified to allow custom tolerances.
    ctf_curve1.MultiplyByConstant(-1.0f / counter);
    ctf_curve1.AddWith(&ctf_curve2);
    float min, max;
    ctf_curve1.GetYMinMax(min, max);
    if ( min < -0.001f || max > 0.001f ) {
        // This is to override a failure, which occurs randomly when using gcc
        // There is probably some undefined behaviour in the code somewhere
        SkipTest;
    }

    EndTest( );
}

void MyTestApp::TestSpectrumImageMethods( ) {
    BeginTest("Spectrum Image Methods");
    // FindRotationalAlignmentBetweenTwoStacksOfImages
    SpectrumImage test_image = SpectrumImage( );
    test_image.Allocate(512, 512, 1);
    test_image.SetToConstant(1.0f);

    // CTF with an astigmatisim angle of 25.0
    CTF ctf1(300, 2.7, 0.07, 10000, 15000, 25.0, 1.0, 0.0);

    test_image.GeneratePowerspectrum(ctf1);

    Image temp_image;
    temp_image.CopyFrom(&test_image);
    temp_image.ApplyMirrorAlongY( );

    float estimated_astigmatism_angle = 0.5 * test_image.FindRotationalAlignmentBetweenTwoStacksOfImages(&temp_image, 1, 90.0, 5.0, 0.1, 0.5);

    if ( fabs(estimated_astigmatism_angle - 25.0) > 5.1 ) {
        FailTest;
    }

    EndTest( );
}

#ifdef cisTEM_USING_LIBTORCH
void MyTestApp::TestLibTorch( ) {
    BeginTest("LibTorch Linking and Basic Operations");

    // Create a tensor with values [1, 2, 3, 4] in a 2x2 matrix
    torch::Tensor tensor = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});

    // Square the tensor (element-wise multiplication)
    torch::Tensor squared = tensor * tensor;

    // Check that the result is correct: [1, 4, 9, 16]
    float  expected[] = {1.0f, 4.0f, 9.0f, 16.0f};
    float* data       = squared.data_ptr<float>( );

    for ( int i = 0; i < 4; i++ ) {
        if ( std::abs(data[i] - expected[i]) > 0.0001f ) {
            FailTest;
        }
    }

    EndTest( );
}
#endif

// A helically symmetric filament for TestHelicalSymmetry: two Gaussian blobs per asymmetric unit, the
// units following the screw (rise along z, twist about z; right-handed for positive twist and rise), the
// filament spanning the box and (unless taper is off) tapered at the ends by a spherical mask so the test
// images look like segments of a longer filament. The two blobs sit at different radii and azimuths, so the helical
// symmetry is the only one the volume has, unless cyclic_order asks for a multi-start (the units repeated
// about the axis).
static void MakeHelicalTestVolume(Image& volume, int box, float pixel_size, float rise, float twist, bool taper = true, int cyclic_order = 1) {
    const float sigma  = 2.0f;
    const int   reach  = 7;
    const float centre = float(box / 2);
    const int   n_max  = int(0.5f * box * pixel_size / rise) + 1;

    volume.Allocate(box, box, box, true);
    volume.SetToConstant(0.0f);
    for ( int n = -n_max; n <= n_max; n++ ) {
        for ( int blob = 0; blob < 2 * cyclic_order; blob++ ) {
            float angle     = deg_2_rad(n * twist + (blob % 2 == 0 ? 0.0f : 70.0f) + (blob / 2) * 360.0f / cyclic_order);
            float radius    = (blob % 2 == 0 ? 18.0f : 10.0f);
            float amplitude = (blob % 2 == 0 ? 1.0f : 0.7f);
            float bx        = centre + radius * cosf(angle);
            float by        = centre + radius * sinf(angle);
            float bz        = centre + n * rise / pixel_size + (blob % 2 == 0 ? 0.0f : 1.5f);
            for ( int k = std::max(0, int(bz) - reach); k <= std::min(box - 1, int(bz) + reach); k++ ) {
                for ( int j = std::max(0, int(by) - reach); j <= std::min(box - 1, int(by) + reach); j++ ) {
                    for ( int i = std::max(0, int(bx) - reach); i <= std::min(box - 1, int(bx) + reach); i++ ) {
                        float d2 = powf(i - bx, 2) + powf(j - by, 2) + powf(k - bz, 2);
                        volume.real_values[volume.ReturnReal1DAddressFromPhysicalCoord(i, j, k)] += amplitude * expf(-d2 / (2.0f * sigma * sigma));
                    }
                }
            }
        }
    }
    if ( taper )
        volume.CosineMask(0.45f * box, 6.0f);
}

// Pearson correlation of two real images of the same size inside a sphere (or disc) about the box centre.
static float CorrelationAboutCentre(Image& a, Image& b, float radius) {
    double sum_a = 0.0, sum_b = 0.0, sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;
    long   n = 0;
    for ( int k = 0; k < a.logical_z_dimension; k++ ) {
        for ( int j = 0; j < a.logical_y_dimension; j++ ) {
            for ( int i = 0; i < a.logical_x_dimension; i++ ) {
                float d2 = powf(i - a.physical_address_of_box_center_x, 2) + powf(j - a.physical_address_of_box_center_y, 2) + powf(k - a.physical_address_of_box_center_z, 2);
                if ( d2 > radius * radius )
                    continue;
                double va = a.ReturnRealPixelFromPhysicalCoord(i, j, k);
                double vb = b.ReturnRealPixelFromPhysicalCoord(i, j, k);
                sum_a += va;
                sum_b += vb;
                sum_aa += va * va;
                sum_bb += vb * vb;
                sum_ab += va * vb;
                n++;
            }
        }
    }
    double mean_a = sum_a / n, mean_b = sum_b / n;
    double var_a = sum_aa / n - mean_a * mean_a, var_b = sum_bb / n - mean_b * mean_b;
    if ( var_a <= 0.0 || var_b <= 0.0 )
        return 0.0f;
    return float((sum_ab / n - mean_a * mean_b) / sqrt(var_a * var_b));
}

// A slice of a Fourier-space, quadrant-swapped volume brought back to real space for comparison.
static void ExtractRealSpaceProjection(Image& volume_ft, RotationMatrix& matrix, Image& projection) {
    projection.object_is_centred_in_box = false;
    volume_ft.ExtractSliceByRotMatrix(projection, matrix, 0.45f);
    projection.SwapRealSpaceQuadrants( );
    projection.BackwardFFT( );
}

static void FourierVolumeToFilteredRealSpace(Image& volume_ft, Image& real_space, float low_pass) {
    real_space.CopyFrom(&volume_ft);
    real_space.CosineMask(low_pass, 0.03f);
    real_space.SwapRealSpaceQuadrants( );
    real_space.BackwardFFT( );
}

void MyTestApp::TestHelicalSymmetry( ) {
    BeginTest("Helical symmetry (SymmetryMatrix, Reconstruct3D)");

    const int   box        = 96;
    const float pixel_size = 1.5f; // box = 144 A
    const float rise       = 6.0f; // A
    const float twist      = 50.0f; // degrees
    const float compare_radius = 0.22f * box; // the copies reach the taper beyond this
    const float low_pass       = 0.125f; // 1 / pixels; 12 A here

    // The symbol: "H<rise>,<twist>" with the copy count decided by the box, or given as a third number
    float parsed_rise, parsed_twist;
    int   parsed_copies, parsed_order;
    if ( ! SymmetryMatrix::ParseHelicalSymbol("H6,50", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest;
    if ( parsed_rise != 6.0f || parsed_twist != 50.0f || parsed_copies != 0 || parsed_order != 1 )
        FailTest;
    if ( ! SymmetryMatrix::ParseHelicalSymbol("h4.75,-165.3,9", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest;
    if ( fabsf(parsed_rise - 4.75f) > 1e-5f || fabsf(parsed_twist + 165.3f) > 1e-4f || parsed_copies != 9 || parsed_order != 1 )
        FailTest;
    // A multi-start: C<n> as the third or fourth field, the copies before or after it
    if ( ! SymmetryMatrix::ParseHelicalSymbol("H15,12,C3", parsed_rise, parsed_twist, parsed_copies, parsed_order) || parsed_copies != 0 || parsed_order != 3 )
        FailTest;
    if ( ! SymmetryMatrix::ParseHelicalSymbol("H15,12,9,c3", parsed_rise, parsed_twist, parsed_copies, parsed_order) || parsed_copies != 9 || parsed_order != 3 )
        FailTest;
    if ( ! SymmetryMatrix::ParseHelicalSymbol("H15,12,C3,9", parsed_rise, parsed_twist, parsed_copies, parsed_order) || parsed_copies != 9 || parsed_order != 3 )
        FailTest;
    if ( SymmetryMatrix::ParseHelicalSymbol("H15,12,C0", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest;
    if ( SymmetryMatrix::ParseHelicalSymbol("H15,12,3,5", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest; // two copy counts
    if ( SymmetryMatrix::ParseHelicalSymbol("H15,12,C2,C3", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest;
    if ( SymmetryMatrix::ParseHelicalSymbol("H6", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest; // no twist
    if ( SymmetryMatrix::ParseHelicalSymbol("H0,50", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest; // no rise
    if ( SymmetryMatrix::ParseHelicalSymbol("C3", parsed_rise, parsed_twist, parsed_copies, parsed_order) )
        FailTest;
    if ( SymmetryMatrix::IsHelicalSymbol("C3") || ! SymmetryMatrix::IsHelicalSymbol(" h6,50 ") )
        FailTest;

    // The operators: identity first, then +-1, +-2, ... rise and twist, so the insertion loops' special
    // treatment of operator 0 still applies to the image itself.
    SymmetryMatrix helix("H6,50,5");
    if ( ! helix.is_helical || helix.number_of_matrices != 5 || helix.helical_copies != 5 )
        FailTest;
    if ( helix.z_shift[0] != 0.0f || helix.z_shift[1] != 6.0f || helix.z_shift[2] != -6.0f || helix.z_shift[3] != 12.0f || helix.z_shift[4] != -12.0f )
        FailTest;
    if ( fabsf(helix.rot_mat[1].m[0][0] - cosf(deg_2_rad(50.0f))) > 1e-5f || fabsf(helix.rot_mat[1].m[1][0] - sinf(deg_2_rad(50.0f))) > 1e-5f || helix.rot_mat[1].m[2][2] != 1.0f )
        FailTest;
    if ( fabsf(helix.rot_mat[4].m[0][0] - cosf(deg_2_rad(100.0f))) > 1e-5f || fabsf(helix.rot_mat[4].m[1][0] + sinf(deg_2_rad(100.0f))) > 1e-5f )
        FailTest;
    SymmetryMatrix unresolved("H6,50");
    if ( unresolved.number_of_matrices != 1 || unresolved.helical_copies != 0 )
        FailTest;
    if ( unresolved.ReturnDefaultHelicalCopies(box * pixel_size) != 7 ) // 2 * int(0.15 * 144 / 6) + 1
        FailTest;
    unresolved.SetHelicalCopies(6); // rounded up to odd, and the symbol records the resolved count
    if ( unresolved.number_of_matrices != 7 || unresolved.symmetry_symbol != "H6,50,7" )
        FailTest;
    SymmetryMatrix point_group("C3");
    if ( point_group.is_helical || point_group.number_of_matrices != 3 )
        FailTest;
    // A two-start: every screw step at both Cn rotations, the identity first, the symbol keeping the Cn
    SymmetryMatrix two_start("H6,50,5,C2");
    if ( two_start.number_of_matrices != 10 || two_start.helical_cyclic_order != 2 || two_start.helical_copies != 5 )
        FailTest;
    if ( two_start.z_shift[0] != 0.0f || two_start.z_shift[1] != 0.0f || two_start.z_shift[2] != 6.0f || two_start.z_shift[3] != 6.0f || two_start.z_shift[4] != -6.0f )
        FailTest;
    if ( fabsf(two_start.rot_mat[1].m[0][0] + 1.0f) > 1e-5f || fabsf(two_start.rot_mat[3].m[0][0] - cosf(deg_2_rad(230.0f))) > 1e-5f || fabsf(two_start.rot_mat[3].m[1][0] - sinf(deg_2_rad(230.0f))) > 1e-5f )
        FailTest;
    SymmetryMatrix unresolved_two_start("H6,50,C3");
    if ( unresolved_two_start.number_of_matrices != 3 || unresolved_two_start.helical_copies != 0 )
        FailTest; // the Cn alone until the box decides the screw steps
    unresolved_two_start.SetHelicalCopies(5);
    if ( unresolved_two_start.number_of_matrices != 15 || unresolved_two_start.symmetry_symbol != "H6,50,5,C3" )
        FailTest;

    Image truth;
    MakeHelicalTestVolume(truth, box, pixel_size, rise, twist);
    Image truth_ft;
    truth_ft.CopyFrom(&truth);
    truth_ft.ForwardFFT( );
    truth_ft.SwapRealSpaceQuadrants( );

    // 1. The operators against the truth: the projection of the volume moved by operator k (rotated by
    // the operator, i.e. the image's Euler matrix left-multiplied) is the image itself shifted in-plane by
    // the operator's rise times the axis row of the Euler matrix. The wrong sign must not pass.
    {
        const float     views[3][3] = {{30.0f, 90.0f, 20.0f}, {100.0f, 75.0f, -40.0f}, {200.0f, 95.0f, 160.0f}};
        AnglesAndShifts angles;
        RotationMatrix  operator_matrix;
        Image           base, copy, expected, wrong;
        base.Allocate(box, box, 1, false);
        copy.Allocate(box, box, 1, false);
        expected.Allocate(box, box, 1, false);
        wrong.Allocate(box, box, 1, false);
        for ( int view = 0; view < 3; view++ ) {
            angles.Init(views[view][0], views[view][1], views[view][2], 0.0f, 0.0f);
            for ( int k = 1; k < helix.number_of_matrices; k++ ) {
                operator_matrix = helix.rot_mat[k] * angles.euler_matrix;
                ExtractRealSpaceProjection(truth_ft, operator_matrix, copy);

                float x_shift = helix.z_shift[k] / pixel_size * angles.euler_matrix.m[2][0];
                float y_shift = helix.z_shift[k] / pixel_size * angles.euler_matrix.m[2][1];
                base.object_is_centred_in_box = false;
                truth_ft.ExtractSliceByRotMatrix(base, angles.euler_matrix, 0.45f);
                expected.CopyFrom(&base);
                expected.PhaseShift(x_shift, y_shift);
                expected.SwapRealSpaceQuadrants( );
                expected.BackwardFFT( );
                wrong.CopyFrom(&base);
                wrong.PhaseShift(-x_shift, -y_shift);
                wrong.SwapRealSpaceQuadrants( );
                wrong.BackwardFFT( );

                float right_correlation = CorrelationAboutCentre(copy, expected, compare_radius);
                float wrong_correlation = CorrelationAboutCentre(copy, wrong, compare_radius);
                if ( right_correlation < 0.97f )
                    FailTest;
                if ( wrong_correlation > right_correlation - 0.15f )
                    FailTest;
            }
        }
    }

    // 2. Reconstruction from a few side-ish views: with the helical symmetry the map must be much closer
    // to the truth than without it, and than with the opposite hand.
    {
        const int number_of_images = 16;
        global_random_number_generator.SetSeed(1234);

        Reconstruct3D plain(box, box, box, pixel_size, 100.0f, 0.0f, 0.0f, "C1");
        Reconstruct3D helical(box, box, box, pixel_size, 100.0f, 0.0f, 0.0f, "H6,50");
        Reconstruct3D other_hand(box, box, box, pixel_size, 100.0f, 0.0f, 0.0f, "H6,-50");
        if ( helical.symmetry_matrices.number_of_matrices != 7 || helical.symmetry_matrices.symmetry_symbol != "H6,50,7" )
            FailTest; // the box resolved the default copy count

        Particle particle(box, box);
        particle.AllocateCTFImage(box, box);
        particle.pixel_size                  = pixel_size;
        particle.particle_occupancy          = 100.0f;
        particle.parameter_average.occupancy = 100.0f;
        particle.sigma_noise                 = 0.01f; // weight 1e4 per slice, so Calculate3DSimple's Wiener constant of 100 is negligible
        particle.parameter_average.sigma     = 1.0f;
        particle.particle_score              = 0.0f;
        particle.particle_image->object_is_centred_in_box = false;

        Image slice;
        slice.Allocate(box, box, 1, false);
        slice.object_is_centred_in_box = false;
        for ( int image = 0; image < number_of_images; image++ ) {
            float phi   = global_random_number_generator.GetUniformRandomSTD(0.0f, 360.0f);
            float theta = global_random_number_generator.GetUniformRandomSTD(60.0f, 120.0f);
            float psi   = global_random_number_generator.GetUniformRandomSTD(0.0f, 360.0f);
            particle.alignment_parameters.Init(phi, theta, psi, 0.0f, 0.0f);
            truth_ft.ExtractSliceByRotMatrix(slice, particle.alignment_parameters.euler_matrix, 0.45f);
            particle.particle_image->CopyFrom(&slice);
            plain.InsertSliceNoCTF(particle);
            particle.particle_image->CopyFrom(&slice);
            helical.InsertSliceNoCTF(particle);
            particle.particle_image->CopyFrom(&slice);
            other_hand.InsertSliceNoCTF(particle);
        }

        Image truth_filtered;
        FourierVolumeToFilteredRealSpace(truth_ft, truth_filtered, low_pass);

        float correlation[3];
        Reconstruct3D* reconstructions[3] = {&plain, &helical, &other_hand};
        for ( int r = 0; r < 3; r++ ) {
            ReconstructedVolume volume;
            volume.InitWithReconstruct3D(*reconstructions[r], pixel_size);
            volume.Calculate3DSimple(*reconstructions[r]);
            Image map;
            FourierVolumeToFilteredRealSpace(*volume.density_map, map, low_pass);
            correlation[r] = CorrelationAboutCentre(map, truth_filtered, compare_radius);
        }
        // Measured with this seed: 0.93 without symmetry, 0.99 with it, 0.31 with the other hand
        if ( correlation[1] < 0.98f )
            FailTest;
        if ( correlation[0] > correlation[1] - 0.03f )
            FailTest;
        if ( correlation[2] > correlation[1] - 0.3f )
            FailTest;
    }

    // 3. A two-start helix from the same few views: the Cn in the symbol doubles the insertions and the
    // map gets closer to the truth than the one-start symbol's, which is still right, only emptier.
    {
        const int   number_of_images = 4;
        const float sharper          = 0.25f; // 6 A: coverage, not the symmetry, is what differs
        Image       two_start_truth, two_start_truth_ft;
        MakeHelicalTestVolume(two_start_truth, box, pixel_size, rise, twist, true, 2);
        two_start_truth_ft.CopyFrom(&two_start_truth);
        two_start_truth_ft.ForwardFFT( );
        two_start_truth_ft.SwapRealSpaceQuadrants( );
        global_random_number_generator.SetSeed(4321);

        Reconstruct3D one_start(box, box, box, pixel_size, 100.0f, 0.0f, 0.0f, "H6,50");
        Reconstruct3D with_cn(box, box, box, pixel_size, 100.0f, 0.0f, 0.0f, "H6,50,C2");
        if ( with_cn.symmetry_matrices.number_of_matrices != 14 || with_cn.symmetry_matrices.symmetry_symbol != "H6,50,7,C2" )
            FailTest;

        Particle particle(box, box);
        particle.AllocateCTFImage(box, box);
        particle.pixel_size                  = pixel_size;
        particle.particle_occupancy          = 100.0f;
        particle.parameter_average.occupancy = 100.0f;
        particle.sigma_noise                 = 0.01f;
        particle.parameter_average.sigma     = 1.0f;
        particle.particle_score              = 0.0f;
        particle.particle_image->object_is_centred_in_box = false;
        Image slice;
        slice.Allocate(box, box, 1, false);
        slice.object_is_centred_in_box = false;
        for ( int image = 0; image < number_of_images; image++ ) {
            particle.alignment_parameters.Init(global_random_number_generator.GetUniformRandomSTD(0.0f, 360.0f), global_random_number_generator.GetUniformRandomSTD(60.0f, 120.0f), global_random_number_generator.GetUniformRandomSTD(0.0f, 360.0f), 0.0f, 0.0f);
            two_start_truth_ft.ExtractSliceByRotMatrix(slice, particle.alignment_parameters.euler_matrix, 0.45f);
            particle.particle_image->CopyFrom(&slice);
            one_start.InsertSliceNoCTF(particle);
            particle.particle_image->CopyFrom(&slice);
            with_cn.InsertSliceNoCTF(particle);
        }
        Image truth_filtered;
        FourierVolumeToFilteredRealSpace(two_start_truth_ft, truth_filtered, sharper);
        float          correlation[2];
        Reconstruct3D* reconstructions[2] = {&one_start, &with_cn};
        for ( int r = 0; r < 2; r++ ) {
            ReconstructedVolume volume;
            volume.InitWithReconstruct3D(*reconstructions[r], pixel_size);
            volume.Calculate3DSimple(*reconstructions[r]);
            Image map;
            FourierVolumeToFilteredRealSpace(*volume.density_map, map, sharper);
            correlation[r] = CorrelationAboutCentre(map, truth_filtered, compare_radius);
        }
        // Measured with this seed: 0.979 with the one-start symbol, 0.989 with C2
        if ( correlation[1] < 0.95f || correlation[1] < correlation[0] + 0.005f )
            FailTest;
    }

    EndTest( );
}

// Pearson correlation of two real volumes inside a cylinder about the box-centre z axis, over the z range
// [z_low, z_high) relative to the centre.
static float CorrelationInCylinder(Image& a, Image& b, float radius, float z_low, float z_high) {
    double sum_a = 0.0, sum_b = 0.0, sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;
    long   n = 0;
    for ( int k = 0; k < a.logical_z_dimension; k++ ) {
        float z = k - a.physical_address_of_box_center_z;
        if ( z < z_low || z >= z_high )
            continue;
        for ( int j = 0; j < a.logical_y_dimension; j++ ) {
            for ( int i = 0; i < a.logical_x_dimension; i++ ) {
                if ( powf(i - a.physical_address_of_box_center_x, 2) + powf(j - a.physical_address_of_box_center_y, 2) > radius * radius )
                    continue;
                double va = a.ReturnRealPixelFromPhysicalCoord(i, j, k), vb = b.ReturnRealPixelFromPhysicalCoord(i, j, k);
                sum_a += va;
                sum_b += vb;
                sum_aa += va * va;
                sum_bb += vb * vb;
                sum_ab += va * vb;
                n++;
            }
        }
    }
    double mean_a = sum_a / n, mean_b = sum_b / n;
    double var_a = sum_aa / n - mean_a * mean_a, var_b = sum_bb / n - mean_b * mean_b;
    if ( var_a <= 0.0 || var_b <= 0.0 )
        return 0.0f;
    return float((sum_ab / n - mean_a * mean_b) / sqrt(var_a * var_b));
}

// Root mean square of a real volume inside the same cylinder and z range.
static float RmsInCylinder(Image& a, float radius, float z_low, float z_high) {
    double sum = 0.0;
    long   n   = 0;
    for ( int k = 0; k < a.logical_z_dimension; k++ ) {
        float z = k - a.physical_address_of_box_center_z;
        if ( z < z_low || z >= z_high )
            continue;
        for ( int j = 0; j < a.logical_y_dimension; j++ )
            for ( int i = 0; i < a.logical_x_dimension; i++ )
                if ( powf(i - a.physical_address_of_box_center_x, 2) + powf(j - a.physical_address_of_box_center_y, 2) <= radius * radius ) {
                    sum += powf(a.ReturnRealPixelFromPhysicalCoord(i, j, k), 2);
                    n++;
                }
    }
    return float(sqrt(sum / n));
}

void MyTestApp::TestHelicalRealSpaceSymmetrisation( ) {
    BeginTest("Helical symmetry in real space");

    const int   box        = 96;
    const float pixel_size = 1.5f;
    const float rise       = 6.0f; // A, 4 px
    const float twist      = 50.0f;
    const float radius     = 30.0f; // px: the blobs reach 18 + 3 sigma
    const float end        = 0.5f * box - 8.0f; // blobs within reach of the box faces are cut in the truth itself

    Image full, tapered;
    MakeHelicalTestVolume(full, box, pixel_size, rise, twist, false);
    // A boxed segment: the filament fading along z over the outer 20% of the box
    tapered.CopyFrom(&full);
    for ( int k = 0; k < box; k++ ) {
        float t = (fabsf(k - tapered.physical_address_of_box_center_z) - 0.3f * box) / (0.2f * box);
        float w = t <= 0.0f ? 1.0f : (t >= 1.0f ? 0.0f : 0.5f * (1.0f + cosf(PI * t)));
        for ( int j = 0; j < box; j++ )
            for ( int i = 0; i < box; i++ )
                tapered.real_values[tapered.ReturnReal1DAddressFromPhysicalCoord(i, j, k)] *= w;
    }

    // 1. An exactly helical volume is left as it is, in the middle and at the ends
    Image test;
    test.CopyFrom(&full);
    test.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 0.45f * box);
    if ( CorrelationInCylinder(test, full, radius, -end, end) < 0.995f )
        FailTest;

    // 2. A tapered filament gets its ends filled from its middle: the end zone's amplitude comes back
    // (its shape already correlates, a taper being a smooth scaling) and the whole stays the truth
    float full_rms = RmsInCylinder(full, radius, 0.35f * box, end);
    if ( RmsInCylinder(tapered, radius, 0.35f * box, end) > 0.7f * full_rms )
        FailTest;
    test.CopyFrom(&tapered);
    test.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 0.45f * box);
    // The zone's mean takes in the faded copies near the box ends, so the filled amplitude sits below the
    // untapered truth (0.79 of it here), as RELION's does; the shape is the truth's
    float filled_rms = RmsInCylinder(test, radius, 0.35f * box, end);
    if ( filled_rms < 0.75f * full_rms || filled_rms > 1.05f * full_rms )
        FailTest;
    if ( CorrelationInCylinder(test, full, radius, 0.35f * box, end) < 0.98f )
        FailTest;
    if ( CorrelationInCylinder(test, full, radius, -end, end) < 0.99f )
        FailTest;

    // 3. Noise averages down over the copies, and the other hand averages the structure away
    Image noisy;
    noisy.CopyFrom(&full);
    global_random_number_generator.SetSeed(99);
    noisy.AddGaussianNoise(1.0f);
    float noisy_before = CorrelationInCylinder(noisy, full, radius, -end, end);
    test.CopyFrom(&noisy);
    test.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 0.45f * box);
    float noisy_after = CorrelationInCylinder(test, full, radius, -end, end);
    test.CopyFrom(&noisy);
    test.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, -twist, 0.3f, 0.45f * box);
    float other_hand = CorrelationInCylinder(test, full, radius, -end, end);
    // Measured: 0.47 before, 0.90 after, 0.17 with the other hand
    if ( noisy_after < noisy_before + 0.3f )
        FailTest;
    if ( other_hand > noisy_before )
        FailTest;

    // 4. A two-start helix: its Cn is averaged over as well, so order 2 holds the exact volume and beats
    // order 1 on the noisy one; the wrong order averages the structure away (measured: 0.61, 0.74, 0.43)
    {
        Image two_start, test2, noisy2;
        MakeHelicalTestVolume(two_start, box, pixel_size, rise, twist, false, 2);
        test2.CopyFrom(&two_start);
        test2.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 0.45f * box, 2);
        if ( CorrelationInCylinder(test2, two_start, radius, -end, end) < 0.995f )
            FailTest;
        noisy2.CopyFrom(&two_start);
        noisy2.AddGaussianNoise(1.0f);
        test2.CopyFrom(&noisy2);
        test2.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 0.45f * box, 1);
        float order_one = CorrelationInCylinder(test2, two_start, radius, -end, end);
        test2.CopyFrom(&noisy2);
        test2.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 0.45f * box, 2);
        float order_two = CorrelationInCylinder(test2, two_start, radius, -end, end);
        test2.CopyFrom(&noisy2);
        test2.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 0.45f * box, 3);
        float order_three = CorrelationInCylinder(test2, two_start, radius, -end, end);
        if ( order_two < order_one + 0.02f || order_three > order_one )
            FailTest;
    }

    // 5. Outside the cylinder nothing changes
    test.CopyFrom(&noisy);
    test.ImposeHelicalSymmetryInRealSpace(rise / pixel_size, twist, 0.3f, 20.0f);
    if ( test.ReturnRealPixelFromPhysicalCoord(2, 2, box / 2) != noisy.ReturnRealPixelFromPhysicalCoord(2, 2, box / 2) )
        FailTest;

    EndTest( );
}

void MyTestApp::BeginTest(const char* test_name) {
    // For access by other tests when running CheckDependencies
    current_test_name               = test_name;
    test_results[current_test_name] = false;

    int length      = strlen(test_name);
    int blank_space = 45 - length;
    Printf("Testing %s ", test_name);
    test_has_passed = true;

    for ( int counter = 0; counter < blank_space; counter++ ) {
        Printf(" ");
    }

    Printf(": ");
}

void MyTestApp::EndTest( ) {
    if ( test_has_passed == true ) {
        // For access by other tests when running CheckDependencies
        test_results[current_test_name] = true;
        PrintResult(true);
    }
    else {
        // Sets the final return value, used in auto build &
        all_tests_have_passed = false;
    }
}

bool MyTestApp::CheckDependencies(std::initializer_list<std::string> list) {
    // Nothing has been added, so must be false.
    bool return_val = true;
    if ( test_results.empty( ) ) {
        Printf("\nCheckDependencies: No tests have been run yet.\n");
        return_val = false;
    }
    else {
        for ( auto dep : list ) {
            auto search = test_results.find(dep);
            if ( search == test_results.end( ) ) {
                Printf("\nCheckDependencies: %s has not been run.\n", dep);
                return_val = false;
                break;
            }
            else {
                if ( search->second == false ) {
                    Printf("\nCheckDependencies: %s has previously failed.\n", dep);
                    return_val = false;
                    break;
                }
                else {
                    return_val = true;
                    break;
                }
            }
        }
    }
    return return_val;
}

void MyTestApp::PrintResultWorker(bool passed, int line, bool skip_on_failure) {

    if ( passed == true ) {
        if ( OutputIsAtTerminal( ) == true )
            Printf(ANSI_COLOR_GREEN "PASSED!" ANSI_COLOR_RESET);
        else
            Printf("PASSED!");
    }
    else {
        if ( skip_on_failure ) {
            if ( OutputIsAtTerminal( ) == true )
                Printf(ANSI_COLOR_BLUE "FAILED, BUT SKIPPING! (Line : %i)" ANSI_COLOR_RESET, line);
            else
                Printf("FAILED, BUT SKIPPING! (Line : %i)", line);
        }
        else {
            if ( OutputIsAtTerminal( ) == true )
                Printf(ANSI_COLOR_RED "FAILED! (Line : %i)" ANSI_COLOR_RESET, line);
            else
                Printf("FAILED! (Line : %i)", line);
            exit(1);
        }
    }

    Printf("\n");
}

void MyTestApp::PrintTitle(const char* title) {
    Printf("\n");
    if ( OutputIsAtTerminal( ) == true )
        Printf(ANSI_UNDERLINE "%s" ANSI_UNDERLINE_OFF, title);
    else
        Printf("%s", title);
    Printf("\n\n");
}

void MyTestApp::WriteEmbeddedFiles( ) {
    temp_directory = ReturnTempDirectory();
    Printf("\nWriting out embedded test files to '%s'...", temp_directory);
    fflush(stdout);

    hiv_image_80x80x1_filename   = temp_directory + "/hiv_image_80x80x1.mrc";
    hiv_images_80x80x10_filename = temp_directory + "/hiv_images_shift_noise_80x80x10.mrc";
    sine_wave_128x128x1_filename = temp_directory + "/sine_wave_128x128x1.mrc";

    WriteEmbeddedArray(hiv_image_80x80x1_filename.c_str( ), hiv_image_80x80x1_array, sizeof(hiv_image_80x80x1_array));
    WriteEmbeddedArray(hiv_images_80x80x10_filename.c_str( ), hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
    WriteEmbeddedArray(hiv_images_80x80x10_filename.c_str( ), hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
    WriteEmbeddedArray(sine_wave_128x128x1_filename.c_str( ), sine_128x128x1_array, sizeof(sine_128x128x1_array));

    numeric_text_filename = temp_directory + "/numbers.num";
    WriteNumericTextFile(numeric_text_filename.c_str( ));
    WriteDatabase((temp_directory + "/1_0_test").c_str( ), (temp_directory + "/1_0_test/1_0_test.db").c_str( ));
    Printf("done!\n");
}

void MyTestApp::WriteEmbeddedArray(const char* filename, const unsigned char* array, long length) {

    FILE* output_file = NULL;
    output_file       = fopen(filename, "wb+");

    if ( output_file == NULL ) {
        Printf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n", filename);
        Printf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n", filename);
        DEBUG_ABORT;
    }

    fwrite(array, sizeof(unsigned char), length, output_file);

    fclose(output_file);
}

void MyTestApp::WriteDatabase(const char* dir, const char* filename) {
    Database database;
    MakeDirectory(dir);
    MakeDirectory(std::string(dir) + "/Assets/");
    std::filesystem::path db_filename = std::filesystem::path(filename);
    if ( PathExists(db_filename.string()) )
        RemoveFile(filename);
    database.CreateNewDatabase(db_filename);
    database.ExecuteSQL(R"sql(
	PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE MASTER_SETTINGS(NUMBER INTEGER PRIMARY KEY, PROJECT_DIRECTORY TEXT, PROJECT_NAME TEXT, CURRENT_VERSION INTEGER, TOTAL_CPU_HOURS REAL, TOTAL_JOBS_RUN INTEGER );
INSERT INTO MASTER_SETTINGS VALUES(1,'/tmp/1_0_test','1_0_test',1,0.0,0);
CREATE TABLE RUNNING_JOBS(JOB_NUMBER INTEGER PRIMARY KEY, JOB_CODE TEXT, MANAGER_IP_ADDRESS INTEGER );
CREATE TABLE RUN_PROFILES(RUN_PROFILE_ID INTEGER PRIMARY KEY, PROFILE_NAME TEXT, MANAGER_RUN_COMMAND TEXT, GUI_ADDRESS TEXT, CONTROLLER_ADDRESS TEXT, COMMANDS_ID INTEGER );
INSERT INTO RUN_PROFILES VALUES(1,'Default Local','/groups/cryoadmin/software/CISTEM/cistem-1.0.0-beta/$command','','',1);
CREATE TABLE MOVIE_ASSETS(MOVIE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, POSITION_IN_STACK INTEGER, X_SIZE INTEGER, Y_SIZE INTEGER, NUMBER_OF_FRAMES INTEGER, VOLTAGE REAL, PIXEL_SIZE REAL, DOSE_PER_FRAME REAL, SPHERICAL_ABERRATION REAL, GAIN_FILENAME TEXT, OUTPUT_BINNING_FACTOR REAL, CORRECT_MAG_DISTORTION INTEGER, MAG_DISTORTION_ANGLE REAL, MAG_DISTORTION_MAJOR_SCALE REAL, MAG_DISTORTION_MINOR_SCALE REAL, PROTEIN_IS_WHITE INTEGER );
CREATE TABLE IMAGE_ASSETS(IMAGE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, POSITION_IN_STACK INTEGER, PARENT_MOVIE_ID INTEGER, ALIGNMENT_ID INTEGER, CTF_ESTIMATION_ID INTEGER, X_SIZE INTEGER, Y_SIZE INTEGER, PIXEL_SIZE REAL, VOLTAGE REAL, SPHERICAL_ABERRATION REAL, PROTEIN_IS_WHITE INTEGER );
CREATE TABLE MOVIE_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE MOVIE_ALIGNMENT_LIST(ALIGNMENT_ID INTEGER PRIMARY KEY, DATETIME_OF_RUN INTEGER, ALIGNMENT_JOB_ID INTEGER, MOVIE_ASSET_ID INTEGER, OUTPUT_FILE TEXT, VOLTAGE REAL, PIXEL_SIZE REAL, EXPOSURE_PER_FRAME REAL, PRE_EXPOSURE_AMOUNT REAL, MIN_SHIFT REAL, MAX_SHIFT REAL, SHOULD_DOSE_FILTER INTEGER, SHOULD_RESTORE_POWER INTEGER, TERMINATION_THRESHOLD REAL, MAX_ITERATIONS INTEGER, BFACTOR INTEGER, SHOULD_MASK_CENTRAL_CROSS INTEGER, HORIZONTAL_MASK INTEGER, VERTICAL_MASK INTEGER, SHOULD_INCLUDE_ALL_FRAMES_IN_SUM INTEGER, FIRST_FRAME_TO_SUM INTEGER, LAST_FRAME_TO_SUM INTEGER );
CREATE TABLE IMAGE_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE PARTICLE_PICKING_LIST(PICKING_ID INTEGER PRIMARY KEY, DATETIME_OF_RUN INTEGER, PICKING_JOB_ID INTEGER, PARENT_IMAGE_ASSET_ID INTEGER, PICKING_ALGORITHM INTEGER, CHARACTERISTIC_RADIUS REAL, MAXIMUM_RADIUS REAL, THRESHOLD_PEAK_HEIGHT REAL, HIGHEST_RESOLUTION_USED_IN_PICKING REAL, MIN_DIST_FROM_EDGES INTEGER, AVOID_HIGH_VARIANCE INTEGER, AVOID_HIGH_LOW_MEAN INTEGER, NUM_BACKGROUND_BOXES INTEGER, MANUAL_EDIT INTEGER );
CREATE TABLE PARTICLE_POSITION_ASSETS(PARTICLE_POSITION_ASSET_ID INTEGER PRIMARY KEY, PARENT_IMAGE_ASSET_ID INTEGER, PICKING_ID INTEGER, PICK_JOB_ID INTEGER, X_POSITION REAL, Y_POSITION REAL, PEAK_HEIGHT REAL, TEMPLATE_ASSET_ID INTEGER, TEMPLATE_PSI REAL, TEMPLATE_THETA REAL, TEMPLATE_PHI REAL );
CREATE TABLE PARTICLE_POSITION_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE VOLUME_ASSETS(VOLUME_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, RECONSTRUCTION_JOB_ID INTEGER, PIXEL_SIZE REAL, X_SIZE INTEGER, Y_SIZE INTEGER, Z_SIZE INTEGER );
CREATE TABLE VOLUME_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE REFINEMENT_PACKAGE_ASSETS(REFINEMENT_PACKAGE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, STACK_FILENAME TEXT, STACK_BOX_SIZE INTEGER, SYMMETRY TEXT, MOLECULAR_WEIGHT REAL, PARTICLE_SIZE REAL, NUMBER_OF_CLASSES INTEGER, NUMBER_OF_REFINEMENTS INTEGER, LAST_REFINEMENT_ID INTEGER, STACK_HAS_WHITE_PROTEIN INTEGER );
CREATE TABLE REFINEMENT_LIST(REFINEMENT_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT, RESOLUTION_STATISTICS_ARE_GENERATED INTEGER, DATETIME_OF_RUN INTEGER, STARTING_REFINEMENT_ID INTEGER, NUMBER_OF_PARTICLES INTEGER, NUMBER_OF_CLASSES INTEGER, RESOLUTION_STATISTICS_BOX_SIZE INTEGER, RESOLUTION_STATISTICS_PIXEL_SIZE REAL, PERCENT_USED REAL );
CREATE TABLE CLASSIFICATION_LIST(CLASSIFICATION_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT, CLASS_AVERAGE_FILE TEXT, REFINEMENT_WAS_IMPORTED_OR_GENERATED INTEGER, DATETIME_OF_RUN INTEGER, STARTING_CLASSIFICATION_ID INTEGER, NUMBER_OF_PARTICLES INTEGER, NUMBER_OF_CLASSES INTEGER, LOW_RESOLUTION_LIMIT REAL, HIGH_RESOLUTION_LIMIT REAL, MASK_RADIUS REAL, ANGULAR_SEARCH_STEP REAL, SEARCH_RANGE_X REAL, SEARCH_RANGE_Y REAL, SMOOTHING_FACTOR REAL, EXCLUDE_BLANK_EDGES INTEGER, AUTO_PERCENT_USED INTEGER, PERCENT_USED REAL );
CREATE TABLE CLASSIFICATION_SELECTION_LIST(SELECTION_ID INTEGER PRIMARY KEY, SELECTION_NAME TEXT, CREATION_DATE INTEGER, REFINEMENT_PACKAGE_ID INTEGER, CLASSIFICATION_ID INTEGER, NUMBER_OF_CLASSES INTEGER, NUMBER_OF_SELECTIONS INTEGER );
CREATE TABLE STARTUP_LIST(STARTUP_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT, NUMBER_OF_STARTS INTEGER, NUMBER_OF_CYCLES INTEGER, INITIAL_RES_LIMIT REAL, FINAL_RES_LIMIT REAL, AUTO_MASK INTEGER, AUTO_PERCENT_USED INTEGER, INITIAL_PERCENT_USED REAL, FINAL_PERCENT_USED REAL, MASK_RADIUS REAL, APPLY_LIKELIHOOD_BLURRING INTEGER, SMOOTHING_FACTOR REAL );
CREATE TABLE RECONSTRUCTION_LIST(RECONSTRUCTION_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ID INTEGER, REFINEMENT_ID INTEGER, NAME TEXT, INNER_MASK_RADIUS REAL, OUTER_MASK_RADIUS REAL, RESOLUTION_LIMIT REAL, SCORE_WEIGHT_CONVERSION REAL, SHOULD_ADJUST_SCORES INTEGER, SHOULD_CROP_IMAGES INTEGER, SHOULD_SAVE_HALF_MAPS INTEGER, SHOULD_LIKELIHOOD_BLUR INTEGER, SMOOTHING_FACTOR REAL, CLASS_NUMBER INTEGER, VOLUME_ASSET_ID INTEGER );
CREATE TABLE ESTIMATED_CTF_PARAMETERS(CTF_ESTIMATION_ID INTEGER PRIMARY KEY, CTF_ESTIMATION_JOB_ID INTEGER, DATETIME_OF_RUN INTEGER, IMAGE_ASSET_ID INTEGER, ESTIMATED_ON_MOVIE_FRAMES INTEGER, VOLTAGE REAL, SPHERICAL_ABERRATION REAL, PIXEL_SIZE REAL, AMPLITUDE_CONTRAST REAL, BOX_SIZE INTEGER, MIN_RESOLUTION REAL, MAX_RESOLUTION REAL, MIN_DEFOCUS REAL, MAX_DEFOCUS REAL, DEFOCUS_STEP REAL, RESTRAIN_ASTIGMATISM INTEGER, TOLERATED_ASTIGMATISM REAL, FIND_ADDITIONAL_PHASE_SHIFT INTEGER, MIN_PHASE_SHIFT REAL, MAX_PHASE_SHIFT REAL, PHASE_SHIFT_STEP REAL, DEFOCUS1 REAL, DEFOCUS2 REAL, DEFOCUS_ANGLE REAL, ADDITIONAL_PHASE_SHIFT REAL, SCORE REAL, DETECTED_RING_RESOLUTION REAL, DETECTED_ALIAS_RESOLUTION REAL, OUTPUT_DIAGNOSTIC_FILE TEXT, NUMBER_OF_FRAMES_AVERAGED INTEGER, LARGE_ASTIGMATISM_EXPECTED INTEGER );
CREATE TABLE RUN_PROFILE_COMMANDS_1(COMMANDS_NUMBER INTEGER PRIMARY KEY, COMMAND_STRING TEXT, NUMBER_OF_COPIES INTEGER, DELAY_TIME_IN_MS INTEGER );
INSERT INTO RUN_PROFILE_COMMANDS_1 VALUES(0,'/groups/cryoadmin/software/CISTEM/cistem-1.0.0-beta/$command',89,10);
CREATE TABLE PROCESS_LOCK(NUMBER INTEGER PRIMARY KEY, ACTIVE_PROCESS INTEGER, ACTIVE_HOST TEXT );
COMMIT;
)sql");
    database.Close( );
}

void MyTestApp::WriteNumericTextFile(const char* filename) {

    FILE* output_file = NULL;
    output_file       = fopen(filename, "wb+");

    if ( output_file == NULL ) {
        Printf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n", filename);
        Printf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n", filename);
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

// Only unset if we set it.
#if defined(unset_cisTEM_LOG_WXPRINTF)
#undef unset_cisTEM_LOG_WXPRINTF
#undef cisTEM_LOG_WXPRINTF
#endif