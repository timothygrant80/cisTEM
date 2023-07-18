#ifndef SRC_PROGRAMS_SAMPLES_COMMON_COMMON_H_
#define SRC_PROGRAMS_SAMPLES_COMMON_COMMON_H_

#include "helper_functions.h"
#include "embedded_test_file.h"
#include "numeric_test_file.h"

extern bool samples_tests_have_all_passed;

inline void TEST(bool result) {
    if ( ! result ) {
        samples_tests_have_all_passed = false;
    }
}

inline wxString CheckForReferenceImages( ) {
    wxString cistem_ref_dir = "";
    // If we are in the dev container the CISTEM_REF_IMAGES variable should be defined, pointing to images we need.
    bool was_found = wxGetEnv(wxString("CISTEM_REF_IMAGES"), &cistem_ref_dir);
    if ( ! was_found ) {
        // If we are not in the dev container, we can't do the tests.
        TEST(false);

        wxPrintf("Failed to resolve the (CISTEM_REF_IMAGES) environment variable.\n", cistem_ref_dir);
        wxPrintf("We can't run the test without images!\n\n");
        exit(-1);
    }

    return cistem_ref_dir;
};

#endif
