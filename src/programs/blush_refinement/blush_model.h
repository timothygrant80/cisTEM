#ifndef _SRC_PROGRAMS_BLUSH_REFINEMENT_BLUSH_MODEL_H_
#define _SRC_PROGRAMS_BLUSH_REFINEMENT_BLUSH_MODEL_H_

#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/normalization.h>
#include <torch/serialize.h>
#include <torch/torch.h>

struct DoubleConv : torch::nn::Module {
    torch::nn::Sequential conv{nullptr};
    DoubleConv(int in_channels, int out_channels, int mid_channels = -1);
    torch::Tensor forward(torch::Tensor x);
};

struct Up : torch::nn::Module {
    torch::nn::Upsample         up{nullptr};
    torch::nn::ConvTranspose3d  up_ct{nullptr};
    std::shared_ptr<DoubleConv> conv;
    bool                        pad;

    Up(int in_channels, int out_channels, bool trilinear = true, bool pad = false);
    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2);
};

struct BlushModel : torch::nn::Module {

    std::shared_ptr<DoubleConv> inc;
    torch::nn::ModuleList       down{nullptr};
    torch::nn::ModuleList       up{nullptr};
    torch::nn::Conv3d           outc{nullptr};
    torch::nn::MaxPool3d        down_pool{nullptr};

    BlushModel(int in_channels = 2, int out_channels = 2);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor grid, torch::Tensor local_std);
    void                                     load_weights(const std::string& path);
};
#endif