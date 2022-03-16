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

bool ProperCompareRealValues(Image& first_image, Image& second_image, float epsilon) {
    bool passed;
    if ( first_image.real_memory_allocated != second_image.real_memory_allocated ) {

        // wxPrintf(" real_memory_allocated values are not the same. [Failed]\n");
        // wxPrintf(" cpu_image.real_memory_allocated ==  %s\n",
        //         std::to_string(first_image.real_memory_allocated));
        // wxPrintf(" resized_host_image.real_memory_allocated ==  %s\n",
        //         std::to_string(second_image.real_memory_allocated));

        passed = false;
    }
    else {

        // print2DArray(first_image);
        // print2DArray(second_image);

        int total_pixels   = 0;
        int unequal_pixels = 0;
        // wxPrintf(" real_memory_allocated values are the same. (%s) Starting loop\n", std::to_string(first_image.real_memory_allocated));
        // wxPrintf(" cpu_image.real_values[0] == (%s)\n", std::to_string(first_image.real_values[0]));
        // wxPrintf(" resized_host_image.real_values[0] == (%s)\n", std::to_string(second_image.real_values[0]));

        int i = 0;
        for ( int z = 0; z < first_image.logical_z_dimension; z++ ) {
            for ( int y = 0; y < first_image.logical_y_dimension; y++ ) {
                for ( int x = 0; x < first_image.logical_x_dimension; x++ ) {
                    if ( std::fabs(first_image.real_values[i] - second_image.real_values[i]) > epsilon ) {
                        unequal_pixels++;
                        if ( unequal_pixels < 5 ) {
                            wxPrintf(" Unequal pixels at position: %s, value 1: %s, value 2: %s.\n", std::to_string(i),
                                     std::to_string(first_image.real_values[i]),
                                     std::to_string(second_image.real_values[i]));
                        }
                        //wxPrintf(" Diff: %f\n", first_image.real_values[i]-second_image.real_values[i]);
                    }
                    total_pixels++;
                    i++;
                }
                i += first_image.padding_jump_value;
            }
        }

        passed = true;
        if ( unequal_pixels > 0 ) {
            int      unequal_percent = 100 * (unequal_pixels / total_pixels);
            wxString err_message     = std::to_string(unequal_pixels) + " out of " +
                                   std::to_string(total_pixels) + "(" +
                                   std::to_string(unequal_percent) +
                                   "%) of pixels are not equal between CPU and GPU "
                                   "images after resizing. [Failed]\n";
            wxPrintf(err_message);

            wxPrintf("Padding values 1: %s, and 2: %s\n",
                     std::to_string(first_image.padding_jump_value),
                     std::to_string(second_image.padding_jump_value));
            passed = false;
        }
    }

    return passed;
}

// void SamplesPrintResult(wxString testName, bool result) {

//   wxPrintf("\t%s",testName);
//   result ? wxPrintf(": [Success]\n") : wxPrintf(": [Failed]\n");

// }

void SamplesPrintTestStartMessage(wxString message, bool bold) {
    // If not bold we print underlined
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
    }

    wxPrintf("\n");
}

void SamplesBeginTest(const char* test_name, bool& test_has_passed) {
    int length      = strlen(test_name);
    int blank_space = 45 - length;
    wxPrintf("  Testing %s ", test_name);
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

    wxPrintf("done!\n");
}
