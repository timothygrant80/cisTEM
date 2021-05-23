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


#include "disk_io_image.hpp"


bool DoDiskIOImageTests(wxString hiv_image_80x80x10_filename, wxString temp_directory) {

  TestResult tr;
  bool passed = true, allPassed = true;
  wxPrintf("  Starting disk I/O image tests.\n\n");

  // The easiest way to read or write and image is simply to use the "quick and
  // dirty methods" in Image class. This avoids creating a seperate Image_file object, which 
  // is needed to modify properties in the image header.
  wxString testName = "Quick and dirty instantiation";
    Image quick_image;
    try { 
      // This image stack has 10 2D images, we read the first. See also QuickAndDirtyReadSlices
      quick_image.QuickAndDirtyReadSlice(hiv_image_80x80x10_filename.ToStdString(), 1);
      // Even though we don't have the image header information, we can still get most of that from the
      // image object properties, e.g. quick_image.logical_x_dimension
        // check first and last pixel...
        } 
    catch (...) {passed = false;} 
    // Also check that the values read in are the correct ones.
    // Note that all images are padded by one or two words in the x dimension for in-place FFT compatibility. 
    // If 1 float is allocated, we want index 0, hence the -1
    passed &= (FloatsAreAlmostTheSame(quick_image.real_values[0], -0.340068) != false);
    passed &= (FloatsAreAlmostTheSame(quick_image.real_values[quick_image.real_memory_allocated - quick_image.padding_jump_value - 1], 0.637069) != false);
    tr.PrintResults(testName, passed);


  testName = "ReadSlice";
    // If we want to read or modify image heade information, we need an Image_file object, the MRCFile is a specialized version
    // This also useful if we want to know something about the dimensions without loading the whole image into memory, for example, for pre-allocation of all or part of a large array.
    //  input_file.ReturnXSize()
    Image test_image;
    MRCFile input_file;
    try { 
      // Note, we could have passed the same arguments to the MRCFile constructor
      input_file.OpenFile(hiv_image_80x80x10_filename.ToStdString(), false); 
      test_image.ReadSlice(&input_file, 1);
    } 
    catch (...) {passed = false;} 
    passed &= (FloatsAreAlmostTheSame(test_image.real_values[0], -0.340068) != false);
    passed &= (FloatsAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - test_image.padding_jump_value - 1], 0.637069) != false);
    tr.PrintResults(testName, passed);

  testName = "WriteSlice"; // If read slice fails, obviously this fails
    MRCFile output_file;
    try {
      wxString temp_filename = temp_directory + "/tmp1.mrc";
      output_file.OpenFile(temp_filename.ToStdString(), false);
    }
    catch (...) {passed = false;} 
    tr.PrintResults(testName, passed);


  testName = "Dimensions from Image Object";
    try {
      passed = input_file.ReturnNumberOfSlices() == 10
      ;
      passed = (input_file.ReturnXSize() == 80) && passed;
      passed = (input_file.ReturnYSize() == 80) && passed;
      passed = (input_file.ReturnZSize() == 10) && passed;
    }
    catch (...) {passed = false;} 
    tr.PrintResults(testName, passed);

  // We can do a little bit more with MRC files, particularly modifying the
  // header information. Here, we'll set the pixel size to 2
  // TODO Same as image file, create, read, add checks as in
  // MyTestApp::TestMRCFunctions()
  testName = "Dimensions from MRC file (header) object";
    try {
      // check dimensions and type
      passed = test_image.is_in_real_space != false;
      passed = (test_image.logical_x_dimension == 80) && passed;
      passed = (test_image.logical_y_dimension == 80) && passed;
      passed = (test_image.logical_z_dimension == 1) && passed;
    }
    catch (...) {passed = false;} 


    if (passed == false) {
      TestResult tr_f;
      passed = test_image.is_in_real_space != false; 
      tr_f.PrintResults("\ttest_image.is_in_real_space == false",  passed);
      passed = test_image.logical_x_dimension == 80;
      tr_f.PrintResults("\ttest_image.logical_x_dimension == 80", passed);
      passed = test_image.logical_y_dimension == 80;
      tr_f.PrintResults("\ttest_image.logical_y_dimension == 80", passed);
      passed = test_image.logical_z_dimension == 1;
      tr_f.PrintResults("\ttest_image.logical_z_dimension == 1);", passed);
      passed = FloatsAreAlmostTheSame(test_image.real_values[0], -0.340068) != false;
      tr_f.PrintResults("\tDoublesAreAlmostTheSame(test_image.real_values[0], -0.340068) != false",  passed);
      //wxPrintf(test_image.real_values[0], " DoublesAreAlmostTheSame(test_image.real_values[0], -0.340068)\n");
      wxString ffs = std::to_string(test_image.real_values[0]);
      wxPrintf(ffs + "\n");
      passed = FloatsAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - test_image.padding_jump_value - 1], 0.637069) != false;
      tr_f.PrintResults("\tDoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069) != false", passed);
      ffs = std::to_string(test_image.real_values[test_image.real_memory_allocated - 3]);
        wxPrintf(ffs + "\n");
      //wxPrintf(test_image.real_values[test_image.real_memory_allocated - 3], " DoublesAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069)\n");
    }

    tr.PrintResults(testName, passed);

  

    // TODO Additionally add check on pixel size (should == 1) but use the
    // approximate check that is in  MyTestApp::TestMRCFunctions() for pixel
    // values
    // TODO Modify pixel size, set to 2
    testName = "Set Pixel Size";
      try{
        input_file.SetPixelSize(1.0);
        passed = DoublesAreAlmostTheSame(input_file.ReturnPixelSize(), 1.0) == true;
      } 
      catch (...) {passed = false;} 
      tr.PrintResults(testName, passed);


    testName = "Change and compare pixel sizes";
      // TODO read in both tmp images, and check that the pixel size has been
      // changed appropriately (with FAIL etc.)
      output_file.SetPixelSize(2.0);
      passed = (input_file.ReturnPixelSize() == 1.0) && (output_file.ReturnPixelSize() == 2.0);
      tr.PrintResults(testName, passed);

    testName = "Delete MRC file";
      try {
        output_file.CloseFile();
        passed = remove(output_file.filename.mb_str()) == 0;
      } catch (...) {
        passed = false;
      }
      tr.PrintResults(testName, passed);

  // TODO call end test and ensure the printout indicates this test
  // (disk_io_image) has pass/failed.
  testName = "Disk I/O image all tests";

  allPassed = tr.ReturnAllPassed();
  passed = tr.ReturnAllPassed();
  tr.PrintResults(testName, passed); // This will set tr.ReutrnAllPassed = true, so we need to record the value in this scope in allPassed for return
  return allPassed;
}
