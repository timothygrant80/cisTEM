#ifndef SRC_PROGRAMS_SAMPLES_0_RESAMPLE_H_
#define SRC_PROGRAMS_SAMPLES_0_RESAMPLE_H_

void ResampleRunner(const std::string& temp_directory);
bool DoCTFImageVsTexture(const std::string& cistem_ref_dir, const std::string& temp_directory);
bool DoFourierCropVsLerpResize(const std::string& cistem_ref_dir, const std::string& temp_directory);
bool DoFourierExpandVsLerpResize(const std::string& cistem_ref_dir, const std::string& temp_directory);
bool DoLerpWithCTF(const std::string& cistem_ref_dir, const std::string& temp_directory);

#endif /* SRC_PROGRAMS_SAMPLES_0_RESAMPLE_H_ */