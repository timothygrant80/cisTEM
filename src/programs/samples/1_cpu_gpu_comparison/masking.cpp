#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "masking.h"

struct GPUTimer {
    GPUTimer( ) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer( ) {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start( ) {
        cudaEventRecord(start_, 0);
    }

    float seconds( ) {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }

  private:
    cudaEvent_t start_, stop_;
};

void CPUvsGPUMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting CPU vs GPU masking tests:", false);

    TEST(DoCosineMaskingTest(hiv_image_80x80x1_filename, temp_directory));

    SamplesPrintEndMessage( );

    return;
}

bool DoCosineMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory) {

    bool passed     = true;
    bool all_passed = true;

    SamplesBeginTest("Cosine mask real space", passed);

    wxString tmp_img_filename = temp_directory + "/tmp1.mrc";

    MRCFile input_file(hiv_image_80x80x1_filename.ToStdString( ), false);
    MRCFile output_file(tmp_img_filename.ToStdString( ), false);

    Image    cpu_image;
    Image    gpu_host_image;
    GpuImage gpu_image;

    cpu_image.ReadSlice(&input_file, 1);
    gpu_host_image.ReadSlice(&input_file, 1);

    gpu_image.Init(gpu_host_image);
    gpu_image.CopyHostToDevice( );

    float wanted_mask_radius;
    float wanted_mask_edge;
    bool  invert;
    bool  force_mask_value;
    float wanted_mask_value;

    RandomNumberGenerator my_rand(pi_v<float>);

    int n_loops = 1;
    for ( int i = 0; i < n_loops; i++ ) {

        // Make some random parameters.
        wanted_mask_radius = 0.f; // GetUniformRandomSTD(0.0f, cpu_image.logical_x_dimension / 2.0f);
        wanted_mask_edge   = 20.f; //GetUniformRandomSTD(0.0f, 20.0f);
        wanted_mask_value  = 0.f; //GetUniformRandomSTD(0.0f, 1.0f);
        if ( my_rand.GetUniformRandomSTD(0.0f, 1.0f > 0.5f) ) {
            invert = true;
        }
        else {
            invert = false;
        }
        if ( my_rand.GetUniformRandomSTD(0.0f, 1.0f > 0.5f) ) {
            force_mask_value = true;
        }
        else {
            force_mask_value = false;
        }

        // FIXME: for intial run, fix the values.
        invert           = false;
        force_mask_value = false;

        cpu_image.CosineMask(wanted_mask_radius, wanted_mask_edge, invert, force_mask_value, wanted_mask_value);
        // gpu_image.CosineMask(wanted_mask_radius, wanted_mask_edge, invert, force_mask_value, wanted_mask_value);
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    return all_passed;
}
