#include "blush_model.h"
#include "../../core/core_headers.h"

#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
namespace F = torch::nn::functional;
// WIP: rebuild the PyTorch architecture in C++ for more customization and potentially efficiency with multi-threading

// Constants
constexpr int  C_BLOCK_SIZE = 64;
constexpr int  C_DEPTH      = 5;
constexpr int  C_WIDTH      = 16;
constexpr bool C_TRILINEAR  = true;
constexpr bool C_MASK_INFER = true;

// Normalization module
using NORM_MODULE = torch::nn::InstanceNorm3d;
using ACTIVATION  = torch::nn::SiLU;
using CONV_3D     = torch::nn::Conv3d;

DoubleConv::DoubleConv(int in_channels, int out_channels, int mid_channels) {
    if ( mid_channels == -1 )
        mid_channels = out_channels;

    auto seq = torch::nn::Sequential( );
    seq->push_back("conv1", torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, mid_channels, 3).padding({1, 1, 1}).kernel_size({3, 3, 3})));
    seq->push_back("norm1", torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(mid_channels).affine(true).momentum(0.1).eps(1e-5)));
    seq->push_back("act1", torch::nn::SiLU( ));
    seq->push_back("conv2", torch::nn::Conv3d(torch::nn::Conv3dOptions(mid_channels, out_channels, 3).padding({1, 1, 1}).kernel_size({3, 3, 3})));
    seq->push_back("norm2", torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(out_channels).affine(true)));
    seq->push_back("act2", torch::nn::SiLU( ));

    conv = register_module("double_conv", seq);
}

torch::Tensor DoubleConv::forward(torch::Tensor x) {
    return conv->forward(x);
}

Up::Up(int in_channels, int out_channels, bool trilinear, bool pad) {
    this->pad = pad;
    if ( trilinear ) {
        up   = register_module("up", torch::nn::Upsample(torch::nn::UpsampleOptions( ).scale_factor(std::vector<double>({2, 2, 2})).mode(torch::kTrilinear).align_corners(true)));
        conv = register_module("conv", std::make_shared<DoubleConv>(in_channels, out_channels, in_channels / 2));
    }

    else {
        up_ct = register_module("up", torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(in_channels, in_channels / 2, 2)));
        conv  = register_module("conv", std::make_shared<DoubleConv>(in_channels, out_channels));
    }
}

torch::Tensor Up::forward(torch::Tensor x1, torch::Tensor x2) {
    if ( up ) {
        x1 = up->forward(x1);
    }
    else if ( up_ct ) {
        x1 = up_ct->forward(x1);
    }

    // Gets difference between dimensions for padding
    if ( pad ) {
        int64_t diffZ = x2.size(2) - x1.size(2);
        int64_t diffY = x2.size(3) - x1.size(3);
        int64_t diffX = x2.size(4) - x1.size(4);
        x1            = F::pad(x1, F::PadFuncOptions({diffX / 2, diffX - diffX / 2, diffY / 2, diffY - diffY / 2, diffZ / 2, diffZ - diffZ / 2}));
    }

    auto x = torch::cat({x2, x1}, /*dim=*/1);
    return conv->forward(x);
}

BlushModel::BlushModel(int in_channels, int out_channels) {
    int factor = (C_TRILINEAR) ? 2 : 1;

    // Initial convolution
    inc = register_module("inc", std::make_shared<DoubleConv>(in_channels, C_WIDTH));

    down_pool = torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2));

    down = register_module("down", torch::nn::ModuleList( )); // torch::nn::ModuleList( ) is a container for storing modules
    up   = register_module("up", torch::nn::ModuleList( )); // torch::nn::ModuleList( ) is a container for storing modules
    // Downsampling layers
    for ( int i = 0; i < C_DEPTH - 1; i++ ) {
        int n = 1 << i; // << is bit shift operator; bit representation of left number is shifted i positions to the right, effectively multiplying 1 by 2^i
        down->push_back(std::make_shared<DoubleConv>(C_WIDTH * n, C_WIDTH * n * 2));
    }
    // Line below, again use bit shift for 2^(DEPTH - 1)
    down->push_back(std::make_shared<DoubleConv>(C_WIDTH * (1 << (C_DEPTH - 1)), C_WIDTH * (1 << C_DEPTH) / factor));

    // Up sampling layers
    for ( int i = 0; i < C_DEPTH - 1; ++i ) {
        int n = 1 << (C_DEPTH - 1 - i); // 2^(C_DEPTH-1-i)
        up->push_back(std::make_shared<Up>(C_WIDTH * n * 2, C_WIDTH * n / factor, C_TRILINEAR));
    }
    up->push_back(std::make_shared<Up>(C_WIDTH * 2, C_WIDTH, C_TRILINEAR));

    // NOTE: In the Python code, the list of Upsample layers are normally added to a ModuleList
    // which handles module registration. But, since I am manually registering modules, what I do
    // for up until this point should be fine

    outc = register_module("outc", torch::nn::Conv3d(torch::nn::Conv3dOptions(C_WIDTH, out_channels, 1)));
}

std::tuple<torch::Tensor, torch::Tensor> BlushModel::forward(torch::Tensor grid, torch::Tensor local_std) {
    auto std           = torch::std(grid, {-1, -2, -3}, /*correction=*/1, /*keepdim=*/true);
    auto mean          = torch::mean(grid, {-1, -2, -3}, /*keepdim=*/true);
    auto grid_standard = (grid - mean) / (std + 1e-12);
    auto input         = torch::cat({grid_standard.unsqueeze(1), local_std.unsqueeze(1)}, /*dim=*/1);

    auto                       nn = inc->forward(input);
    std::vector<torch::Tensor> skip;

    // Downsample path
    for ( int i = 0; i < C_DEPTH; ++i ) {
        skip.push_back(nn);
        // nn = F::max_pool3d(nn, torch::nn::MaxPool3dOptions(2));
        nn = down_pool(nn);
        nn = down[i]->as<DoubleConv>( )->forward(nn);
    }

    // Reverse skip connections
    std::reverse(skip.begin( ), skip.end( ));

    // Upsample path
    for ( int i = 0; i < C_DEPTH; ++i ) {
        nn = up[i]->as<Up>( )->forward(nn, skip[i]);
    }

    nn = outc->forward(nn);

    // Slicing questionable...
    auto output = grid_standard - nn.index({torch::indexing::Slice( ), 0});
    output      = output * (std + 1e-12) + mean;

    torch::Tensor mask_logit = C_MASK_INFER ? nn.slice(1, 1, 2) : torch::Tensor( );
    return {output, mask_logit};
}

void BlushModel::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if ( ! file )
        throw std::runtime_error("Failed to open " + path + " weights file for reading");

    std::unordered_map<std::string, torch::Tensor> params_map;
    for ( auto& pair : named_parameters( ) ) {
        params_map[pair.key( )] = pair.value( );
    }

    std::unordered_map<std::string, torch::Tensor> buffers_map;
    for ( auto& pair : named_buffers( ) ) {
        buffers_map[pair.key( )] = pair.value( );
    }

    while ( file.peek( ) != EOF ) {
        int64_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data( ), name_len);

        torch::Tensor* target_tensor = nullptr;
        auto           it_param      = params_map.find(name);
        if ( it_param != params_map.end( ) ) {
            target_tensor = &it_param->second;
        }
        else {
            auto it_buf = buffers_map.find(name);
            if ( it_buf != buffers_map.end( ) ) {
                target_tensor = &it_buf->second;
            }
            else {
                throw std::runtime_error("Parameter or buffer " + name + " not found in model");
            }
        }

        torch::Tensor& param = *target_tensor;

        int64_t ndims;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
        std::vector<int64_t> shape(ndims);
        for ( int i = 0; i < ndims; ++i ) {
            file.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
        }

        auto expected_shape = param.sizes( );
        if ( shape != expected_shape ) {
            throw std::runtime_error("Shape mismatch for " + name);
        }

        int64_t num_elems = 1;
        for ( auto dim : shape )
            num_elems *= dim;
        file.read(reinterpret_cast<char*>(param.data_ptr( )), num_elems * param.element_size( ));
    }
}