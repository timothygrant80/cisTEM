#ifndef SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_STATISTICAL_OPS_H_
#define SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_STATISTICAL_OPS_H_

template <typename T>
struct MeasuredValue {
    T cpu;
    T gpu;
};

// Test runner
void CPUvsGPUStatisticalOpsRunner(const std::string& hiv_image_80x80x1_filename, std::string& temp_directory);

// Tests for mean, variance etc.
bool DoStatsticalMomentsTests(const std::string& hiv_image_80x80x1_filename, std::string& temp_directory);

// Tests for min/max
bool DoExtremumTests(const std::string& hiv_image_80x80x1_filename, std::string& temp_directory);

#endif