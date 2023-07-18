#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "masking.h"

void TestRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting a test to be a runner", false);

    TEST(MyTest(hiv_image_80x80x1_filename, temp_directory));

    SamplesPrintEndMessage( );

    return;
}

bool MyTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Extract slice CPU vs ground truth", passed);

    // Do some things, set the value for passed

    all_passed = passed ? all_passed : false;

    SamplesTestResult(passed);

    return all_passed;
}