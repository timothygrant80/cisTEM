#ifndef SRC_PROGRAMS_SAMPLES_0_RESAMPLE_H_
#define SRC_PROGRAMS_SAMPLES_0_RESAMPLE_H_

void ResampleRunner(const wxString& temp_directory);
bool DoCTFImageVsTexture(const wxString& cistem_ref_dir, const wxString& temp_directory);
bool DoFourierCropVsLerpResize(const wxString& cistem_ref_dir, const wxString& temp_directory);
bool DoLerpWithCTF(const wxString& cistem_ref_dir, const wxString& temp_directory);

#endif /* SRC_PROGRAMS_SAMPLES_0_RESAMPLE_H_ */