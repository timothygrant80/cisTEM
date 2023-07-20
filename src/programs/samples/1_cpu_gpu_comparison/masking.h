#ifndef SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_MASKING_COMPARISON_H_
#define SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_MASKING_COMPARISON_H_

// #ifdef ENABLEGPU
// #include <cutensor.h>
// #endif

void CPUvsGPUMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);
bool DoCosineMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);

#endif