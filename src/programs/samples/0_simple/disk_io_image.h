/*
 * disk_io_image.hpp
 *
 *  Created on: Apr 1, 2021
 *      Author: B.A. Himes, Shiran Dror
 *
 *      Goal:
 *      	The purpose of this recipe, is to demonstrate the different means of reading/writing images to disk using cisTEM.
 *
 *      Scope:
 *      	At the time of writing, all disk i/o must go through the cpu, but hopefully we can do direct reads to the gpu in the future. A pointer to this recipe will be <here>.
 *
 *      Background:
 *      	The Image class in cisTEM can be used to represent 1,2 or 3D images, in memory, with extensions to higher dimensions only available via arrays of Images (or pointers to them.)
 *      	To read an image in from disk, we need to know about its representation, the properties of which can be accessed via the Image_file class. This provides an interface with the
 *      	primary types supported in cisTEM: MRC, TIF(F), DM, EER. We predominantly use the MRC format, which is very similar to CCP4. There are class specializations of mrc_file and mrc_header.
 *
 *
 *
 *
 *
 */

#ifndef SRC_PROGRAMS_SAMPLES_0_SIMPLE_DISK_IO_IMAGE_HPP_
#define SRC_PROGRAMS_SAMPLES_0_SIMPLE_DISK_IO_IMAGE_HPP_

// Assuming this is called from samples_functional_testing.cpp, you will have a file written to disk "${HOME}/hiv_image_80x80x1.mrc"
// wxString temp_directory = wxFileName::GetHomeDir();
// wxString hiv_image_80x80x1_filename = temp_directory + "/hiv_image_80x80x1.mrc";

// The easiest way to read or write and image is simply to use the "quick and dirty methods" in Image class
// TODO instantiate an image object, read it in using quick and dirt read slice.

// At times, we may want to have information about the image, prior to reading it into memory. The most general way to do this is to create an Image_file object, which handles all supported types (see Background)
// TODO create image file type, refer to console_test  MyTestApp::TestMRCFunctions()
// TODO run size checks as in above

// We can do a little bit more with MRC files, particularly modifying the header information. Here, we'll set the pixel size to 2
// TODO Same as image file, create, read, add checks as in MyTestApp::TestMRCFunctions()
// TODO Additionally add check on pixel size (should == 1) but use the approximate check that is in  MyTestApp::TestMRCFunctions() for pixel values
// TODO Modify pixel size, set to 2

// We'll skip the quick and dirty write slices, and now write a temporary file with our modified pixel size.
// TODO use mrc_file method write slice to disk, to write tmp1.mrc (in the temp directory above, you'll need a new wxString too

// Alternatively, you can pass a pointer to your mrc_file object to the image object method WriteSlices
// TODO write out tmp2.mrc using the Image class method

// TODO read in both tmp images, and check that the pixel size has been changed appropriately (with FAIL etc.)
// TODO remove the tmp images from disk

// TODO call end test and ensure the printout indicates this test (disk_io_image) has pass/failed.

void PrintResult(wxString testName, bool result);
void DiskIOImageRunner(wxString hiv_images_80x80x10_filename, wxString temp_directory);
bool DiskIOImageTests(wxString hiv_images_80x80x10_filename, wxString temp_directory);

void TestResult(wxString testName, bool result);

#endif /* SRC_PROGRAMS_SAMPLES_0_SIMPLE_DISK_IO_IMAGE_HPP_ */
