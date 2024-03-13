#ifndef _SRC_PROGRAMS_SAMPLES_5_BATCHED_OPS_BATCHED_CORRELATION_H_
#define _SRC_PROGRAMS_SAMPLES_5_BATCHED_OPS_BATCHED_CORRELATION_H_

class GpuImage;

void BatchedCorrelationRunner(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory);
bool DoBatchedCorrelationTest(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory);

void RunBatchedCorrelation(GpuImage& d_ref_img, GpuImage* d_seq_rotation_cache, int n_search_images, int batch_size, bool test_mirror, float* results, bool is_ground_truth);

#endif