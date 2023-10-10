#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#include "../../../core/core_headers.h"
#endif

#include "test_file.h"
#include "helper_functions.h"

void print2DArray(Image& image) {
    int i = 0;
    wxPrintf("Image real space data:\n");
    for ( int z = 0; z < image.logical_z_dimension; z++ ) {
        for ( int y = 0; y < image.logical_y_dimension; y++ ) {
            for ( int x = 0; x < image.logical_x_dimension; x++ ) {
                wxPrintf("%f\t", image.real_values[i]);

                i++;
            }
            wxPrintf("\n");
            i += image.padding_jump_value;
        }
        wxPrintf("\n");
    }
}

void PrintArray(float* p, int maxLoops) {
    wxPrintf("Starting loop through array.\n");

    if ( p == nullptr ) {
        wxPrintf("pointer is null, aborting.\n");
        return;
    }
    for ( int i = 0; i < maxLoops; i++ ) {
        wxPrintf("%s \n", std::to_string(i));
        // wxPrintf(" %s\n", *arr);
        // std::cout<< *arr <<" ";
        std::cout << *(p + i) << std::endl;

        p++;
    }
    wxPrintf("Loop done.\n");
}

// bool IsPointerNull(float *p) {
//     if (p == nullptr) {
//         wxPrintf("pointer is null!\n");
//         return true;
//     }
//     wxPrintf("pointer is valid!\n");
//     return false;
// }

bool CompareRealValues(Image& first_image, Image& second_image, float minimum_ccc, float mask_radius) {

    MyDebugAssertTrue(first_image.is_in_memory, "First image is not in memory");
    MyDebugAssertTrue(second_image.is_in_memory, "Second image is not in memory");
    MyDebugAssertTrue(first_image.is_in_real_space, "First image is not in real space");
    MyDebugAssertTrue(second_image.is_in_real_space, "Second image is not in real space");
    MyDebugAssertTrue(first_image.HasSameDimensionsAs(&second_image), "Images must have same dimensions");

    first_image.ZeroFloatAndNormalize(1.f, mask_radius);
    second_image.ZeroFloatAndNormalize(1.f, mask_radius);

    float score = first_image.ReturnCorrelationCoefficientUnnormalized(second_image, mask_radius);

    if ( score < minimum_ccc ) {
        // wxPrintf("\nFailed CCC is %g\n", score);
        // first_image.QuickAndDirtyWriteSlice("first_image.mrc", 1);
        // second_image.QuickAndDirtyWriteSlice("second_image.mrc", 1);
        return false;
    }
    else {
        return true;
    }
}

bool CompareComplexValues(Image& first_image, Image& second_image, float minimum_ccc, float mask_radius) {

    MyDebugAssertTrue(first_image.is_in_memory, "First image is not in memory");
    MyDebugAssertTrue(second_image.is_in_memory, "Second image is not in memory");
    MyDebugAssertFalse(first_image.is_in_real_space, "First image is in real space");
    MyDebugAssertFalse(second_image.is_in_real_space, "Second image is in real space");
    MyDebugAssertTrue(first_image.HasSameDimensionsAs(&second_image), "Images must have same dimensions");

    // use everything within nyquist (maybe the corners should be checked too?)
    constexpr float low_limit2       = 0.f;
    constexpr float high_limit2      = 0.25f;
    constexpr float signed_cc_limit2 = 0.25f;

    float score = first_image.GetWeightedCorrelationWithImage(second_image, low_limit2, high_limit2, signed_cc_limit2);

    if ( score < minimum_ccc ) {
        wxPrintf("\nFailed CCC is %g\n", score);
        first_image.QuickAndDirtyWriteSlice("first_image.mrc", 1);
        second_image.QuickAndDirtyWriteSlice("second_image.mrc", 1);
        return false;
    }
    else {
        return true;
    }
}

// void SamplesPrintResult(wxString testName, bool result) {

//   wxPrintf("\t%s",testName);
//   result ? wxPrintf(": [Success]\n") : wxPrintf(": [Failed]\n");

// }

void SamplesPrintTestStartMessage(wxString message, bool bold) {
    // If not bold we print underlined
    wxPrintf("\n");
    if ( bold )
        SamplesPrintBold(message);
    else
        SamplesPrintUnderlined(message);
    wxPrintf("\n\n");
}

void SamplesPrintUnderlined(wxString message) {
    if ( OutputIsAtTerminal( ) == true )
        wxPrintf(ANSI_UNDERLINE + message + ANSI_UNDERLINE_OFF);
    else
        wxPrintf("%s", message);
}

void SamplesPrintBold(wxString message) {
    if ( OutputIsAtTerminal( ) == true )
        wxPrintf(ANSI_BOLD + message + ANSI_BOLD_OFF);
    else
        wxPrintf("%s", message);
}

void SamplesPrintResult(bool passed, int line) {

    if ( passed == true ) {
        if ( OutputIsAtTerminal( ) == true )
            wxPrintf(ANSI_COLOR_GREEN "PASSED!" ANSI_COLOR_RESET);
        else
            wxPrintf("PASSED!");
    }
    else {
        if ( OutputIsAtTerminal( ) == true )
            wxPrintf(ANSI_COLOR_RED "FAILED! (Line : %i)" ANSI_COLOR_RESET, line);
        else
            wxPrintf("FAILED! (Line : %i)", line);
        // Removing the exit behavior because I want all tests to run as this is more informative for CI, i.e.
        // multiple fixes can be made in a single go rather than a one at a time approach.
        // exit(1);
    }
}

void SamplesPrintResultCanFail(bool passed, int line) {

    if ( passed == true ) {
        if ( OutputIsAtTerminal( ) == true )
            wxPrintf(ANSI_COLOR_GREEN "PASSED!" ANSI_COLOR_RESET);
        else
            wxPrintf("PASSED!");
    }
    else {
        if ( OutputIsAtTerminal( ) == true )
            wxPrintf(ANSI_COLOR_BLUE "FAILED, BUT SKIPPING! (Line : %i)" ANSI_COLOR_RESET, line);
        else
            wxPrintf("FAILED, BUT SKIPPING! (Line : %i)", line);
    }
}

void SamplesBeginTest(const char* test_name, bool& test_has_passed) {
    int length      = strlen(test_name);
    int blank_space = 45 - length;
    wxPrintf("\n  Testing %s ", test_name);
    test_has_passed = true;

    for ( int counter = 0; counter < blank_space; counter++ ) {
        wxPrintf(" ");
    }

    wxPrintf(": ");
}

void SamplesBeginPrint(const char* test_name) {
    int length      = strlen(test_name);
    int blank_space = 64 - length;
    wxPrintf("  %s ", test_name);

    for ( int counter = 0; counter < blank_space; counter++ ) {
        wxPrintf(" ");
    }

    wxPrintf(": ");
}

FileTracker::~FileTracker( ) {
    // destructor: remove all files written to harddrive.
    Cleanup( );
}

void FileTracker::Cleanup( ) {
    wxPrintf("\nRemoving test files ... \n");

    for ( auto& it : testFiles )
        delete it;
    testFiles.clear( );

    wxPrintf("\ndone!\n");
}

Image GetAbsOfFourierTransformAsRealImage(Image& input_image) {
    Image tmp_img;
    int   pixel_pitch = (input_image.logical_x_dimension + input_image.padding_jump_value) / 2;
    tmp_img.Allocate(pixel_pitch, input_image.logical_y_dimension, input_image.logical_z_dimension, true, true);

    int address_complex = 0;
    int address_real    = 0;
    for ( int k = 0; k < input_image.logical_z_dimension; k++ ) {
        for ( int j = 0; j < input_image.logical_y_dimension; j++ ) {
            for ( int i = 0; i < pixel_pitch; i++ ) {
                tmp_img.real_values[address_real] = abs(input_image.complex_values[address_complex]);
                address_complex++;
                address_real++;
            }
            address_real += tmp_img.padding_jump_value;
        }
    }
    return tmp_img;
}

/**
* @brief Primarily for trouble shooting re-arrangment of Fourier comonents. Without this, an iFFT is taken prior to saving to disk, which is not what we want in some cases.
* @param input_image
* @return Image sized as non-redunant Fourier transform
*/
