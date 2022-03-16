/*
 * disk_io_image.hpp
 *
 *  Created on: Apr 1, 2021
 *      Author: B.A. Himes, Shiran Dror
 *
 *      Goal:
 *      	The purpose of this recipe, is to demonstrate the different
 * means of reading/writing images to disk using cisTEM.
 *
 *      Scope:
 *      	At the time of writing, all disk i/o must go through the cpu,
 * but hopefully we can do direct reads to the gpu in the future. A pointer to
 * this recipe will be <here>.
 *
 *      Background:
 *      	The Image class in cisTEM can be used to represent 1,2 or 3D
 * images, in memory, with extensions to higher dimensions only available via
 * arrays of Images (or pointers to them.) To read an image in from disk, we
 * need to know about its representation, the properties of which can be
 * accessed via the Image_file class. This provides an interface with the
 *      	primary types supported in cisTEM: MRC, TIF(F), DM, EER. We
 * predominantly use the MRC format, which is very similar to CCP4. There are
 * class specializations of mrc_file and mrc_header.
 *
 *
 *
 *
 *
 */

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#include "../../../core/core_headers.h"
#endif

#include "../common/helper_functions.h"
#include "disk_io_image.h"

bool DoDiskIOImageTests(wxString hiv_images_80x80x10_filename, wxString temp_directory) {

    bool passed = true, allPassed = true;

    SamplesPrintTestStartMessage("Starting disk I/O image tests", false);
    //MRCFile input_file(std::string(hiv_image_80x80x1_filename.mb_str()), false);
    MRCFile input_file(hiv_images_80x80x10_filename.ToStdString( ), false);

    wxString temp_filename = temp_directory + "/tmp1.mrc";

    MRCFile output_file(temp_filename.ToStdString( ), false);

    Image test_image;
    test_image.ReadSlice(&input_file, 1);

    // The easiest way to read or write and image is simply to use the "quick and
    // dirty methods" in Image class
    // TODO instantiate an image object, read it in using quick and dirt read
    // slice.

    SamplesBeginTest("Quick and dirty instantiation", passed);
    Image quick_image;

    try {

        quick_image.QuickAndDirtyReadSlice(input_file.filename.ToStdString( ), 1);
    } catch ( ... ) {

        passed = false;
    }

    allPassed = allPassed && passed;
    SamplesTestResult(passed);

    // At times, we may want to have information about the image, prior to reading
    // it into memory. The most general way to do this is to create an Image_file
    // object, which handles all supported types (see Background)
    // TODO create image file type, refer to console_test
    // MyTestApp::TestMRCFunctions()
    // TODO run size checks as in above
    SamplesBeginTest("Dimensions", passed);

    passed = input_file.ReturnNumberOfSlices( ) == 10;
    passed = (input_file.ReturnXSize( ) == 80) && passed;
    passed = (input_file.ReturnYSize( ) == 80) && passed;
    passed = (input_file.ReturnZSize( ) == 10) && passed;
    SamplesTestResult(passed);
    allPassed = allPassed && passed;

    // We can do a little bit more with MRC files, particularly modifying the
    // header information. Here, we'll set the pixel size to 2
    // TODO Same as image file, create, read, add checks as in
    // MyTestApp::TestMRCFunctions()
    SamplesBeginTest("MRC file", passed);

    // check dimensions and type

    passed = test_image.is_in_real_space != false;
    passed = (test_image.logical_x_dimension == 80) && passed;
    passed = (test_image.logical_y_dimension == 80) && passed;
    passed = (test_image.logical_z_dimension == 1) && passed;

    // check first and last pixel...

    passed = (DoublesAreAlmostTheSame(test_image.real_values[0], -0.340068) != false) && passed;
    passed = (DoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069) != false) && passed;

    allPassed = allPassed && passed;
    SamplesTestResult(passed);

    // if (passed == false) {
    //   SamplesTestResult("\ttest_image.is_in_real_space == false",  test_image.is_in_real_space != false);
    //   SamplesTestResult("\ttest_image.logical_x_dimension == 80",  test_image.logical_x_dimension == 80);
    //   SamplesTestResult("\ttest_image.logical_y_dimension == 80",  test_image.logical_y_dimension == 80);
    //   SamplesTestResult("\ttest_image.logical_z_dimension == 1);",  test_image.logical_z_dimension == 1);
    //   SamplesTestResult("\tDoublesAreAlmostTheSame(test_image.real_values[0], -0.340068) != false",  DoublesAreAlmostTheSame(test_image.real_values[0], -0.340068) != false);
    //    //wxPrintf(test_image.real_values[0], " DoublesAreAlmostTheSame(test_image.real_values[0], -0.340068)\n");
    //    wxString ffs = std::to_string(test_image.real_values[0]);
    //   wxPrintf(ffs + "\n");
    //   SamplesTestResult("\tDoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069) != false",  DoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069) != false);
    //   ffs = std::to_string(test_image.real_values[test_image.real_memory_allocated - 3]);
    //     wxPrintf(ffs + "\n");
    //   //wxPrintf(test_image.real_values[test_image.real_memory_allocated - 3], " DoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069)\n");
    // }

    // TODO Additionally add check on pixel size (should == 1) but use the
    // approximate check that is in  MyTestApp::TestMRCFunctions() for pixel
    // values
    // TODO Modify pixel size, set to 2
    SamplesBeginTest("Pixel", passed);

    input_file.SetPixelSize(1.0);

    passed    = DoublesAreAlmostTheSame(input_file.ReturnPixelSize( ), 1.0) == true;
    allPassed = allPassed && passed;
    SamplesTestResult(passed);

    // We'll skip the quick and dirty write slices, and now write a temporary file
    // with our modified pixel size.
    // TODO use mrc_file method write slice to disk, to write tmp1.mrc (in the
    // temp directory above, you'll need a new wxString too

    // Alternatively, you can pass a pointer to your mrc_file object to the image
    // object method WriteSlices
    // TODO write out tmp2.mrc using the Image class method
    SamplesBeginTest("Write temp file", passed);

    try {
        test_image.ReadSlice(&input_file, 1);
        test_image.WriteSlice(&output_file, 1);
    } catch ( ... ) {
        passed = false;
    }
    allPassed = allPassed && passed;
    SamplesTestResult(passed);

    // TODO read in both tmp images, and check that the pixel size has been
    // changed appropriately (with FAIL etc.)
    SamplesBeginTest("Change and compare pixel sizes", passed);
    output_file.SetPixelSize(2.0);
    passed = (input_file.ReturnPixelSize( ) == 1.0) &&
             (output_file.ReturnPixelSize( ) == 2.0);
    allPassed = allPassed && passed;
    SamplesTestResult(passed);

    // TODO remove the tmp images from disk
    SamplesBeginTest("Delete MRC file", passed);

    try {
        output_file.CloseFile( );
        passed = remove(output_file.filename.mb_str( )) == 0;
    } catch ( ... ) {
        passed = false;
    }
    allPassed = allPassed && passed;
    SamplesTestResult(passed);
    // TODO call end test and ensure the printout indicates this test
    // (disk_io_image) has pass/failed.
    SamplesBeginTest("disk I/O images overall", passed);
    SamplesPrintResult(allPassed, __LINE__);
    wxPrintf("\n\n");
    return allPassed;
}
