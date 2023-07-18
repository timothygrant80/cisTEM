#ifndef SRC_PROGRAMS_SAMPLES_4_FFTS_SIMPLE_CUFFT_H_
#define SRC_PROGRAMS_SAMPLES_4_FFTS_SIMPLE_CUFFT_H_

void SimpleCuFFTRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);
bool DoInPlaceR2CandC2R(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);
bool DoInPlaceR2CandC2RBatched(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);

#endif