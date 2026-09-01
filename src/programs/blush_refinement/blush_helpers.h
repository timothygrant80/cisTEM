#ifndef _SRC_PROGRAMS_BLUSH_REFINEMENT_BLUSH_HELPERS_H
#define _SRC_PROGRAMS_BLUSH_REFINEMENT_BLUSH_HELPERS_H

#include "../../../include/libtorch/libtorch_push_macros.h"
#include <torch/torch.h>
#include "../../../include/libtorch/libtorch_pop_macros.h"
#include "blush_model.h"
#include <functional>

namespace BlushHelpers {
torch::Tensor get_local_std_dev(torch::Tensor grid, int size);
torch::Tensor make_weight_box(const int& block_size, int margin);
torch::Tensor generate_radial_mask(const int& box_size, const float& radius, const int& mask_edge_width);
void          ApplyBlush(std::vector<float>& input_volume, const std::string& model_filename, const int& box_size, const float& mask_radius, const int& batch_size, const int& max_threads, std::shared_ptr<std::atomic<bool>> stop_flag, std::function<bool(int percentage, long seconds_remaining)> progress_callback);
} // namespace BlushHelpers

#endif