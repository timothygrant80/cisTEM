#ifndef SRC_PROGRAMS_SAMPLES_4_FFTS_SIMPLE_CUFFT_H_
#define SRC_PROGRAMS_SAMPLES_4_FFTS_SIMPLE_CUFFT_H_

void SimpleCuFFTRunner(const std::string& hiv_image_80x80x1_filename, std::string& temp_directory);
bool DoInPlaceR2CandC2R(const std::string& hiv_image_80x80x1_filename, std::string& temp_directory);
bool DoInPlaceR2CandC2RBatched(const std::string& hiv_image_80x80x1_filename, std::string& temp_directory);

#endif