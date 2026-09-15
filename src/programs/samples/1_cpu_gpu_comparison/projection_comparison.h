#ifndef SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_PROJECTION_COMPARISON_H_
#define SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_PROJECTION_COMPARISON_H_

void CPUvsGPUProjectionRunner(const std::string& temp_directory);
bool DoCPUvsGPUProjectionTest(const std::string& cistem_ref_dir, const std::string& temp_directory);

#endif