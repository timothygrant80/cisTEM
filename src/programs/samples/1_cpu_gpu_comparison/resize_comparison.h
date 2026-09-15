#ifndef SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_RESIZE_COMPARISON_H_
#define SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_RESIZE_COMPARISON_H_

void CPUvsGPUResizeRunner(std::string hiv_image_80x80x1_filename, std::string temp_directory);
bool DoCPUvsGPURealSpaceResize(std::string hiv_image_80x80x1_filename, std::string temp_directory);
bool DoCPUvsGPUFourierResize(std::string hiv_image_80x80x1_filename, std::string temp_directory);

#endif
