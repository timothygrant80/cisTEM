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

// // Weight box creation
// torch::Tensor make_weight_box(int size, int margin = 4) {
//     margin = margin > 0 ? margin : 1;
//     int s  = size - margin * 2;

//     at::Tensor z = torch::linspace(-s / 2, s / 2, s); //.view({-1, 1, 1, 1});
//     at::Tensor y = torch::linspace(-s / 2, s / 2, s); //.view({1, -1, 1, 1});
//     at::Tensor x = torch::linspace(-s / 2, s / 2, s); //.view({1, 1, -1, 1});

//     at::Tensor r = torch::maximum(torch::abs(x), torch::abs(y));
//     r            = torch::maximum(torch::abs(z), r);
//     r            = torch::cos(r / r.max( ) * (M_PI / 2)) * 0.6 + 0.4;

//     auto w                                                                                            = torch::zeros({size, size, size});
//     w.slice(0, margin, size - margin).slice(1, margin, size - margin).slice(2, margin, size - margin) = r;

//     return w;
// }

/**
 * @brief C++ method of calculating the weight box; will need to compare this against the one above
 * 
 * @param block_size 
 * @param margin 
 * @return at::Tensor 
 */
// at::Tensor cpp_make_weight_box(const int& block_size, int margin) {
//     margin = (margin - 1 > 0) ? margin - 1 : 0;
//     int s  = block_size - (margin * 2);

//     std::cout << "s == " << std::to_string(s) << std::endl;
//     // wxPrintf("s == %i\n", s);

//     // Create evenly spaced 1D tensors; should range from -27 to 27, with increments of...55?
//     torch::Tensor x = torch::linspace(-s / 2, s / 2, s);
//     torch::Tensor y = torch::linspace(-s / 2, s / 2, s);
//     torch::Tensor z = torch::linspace(-s / 2, s / 2, s);

//     // Broadcasting means these tensors can be subsequently expanded to fit the appropriate dimensions as needed
//     // Each of these tensors is different, and later they are sliced together dimensionally
//     // expand broadcasts the dimension with s, so the values of the linspace array are applied in all 3 dimensions
//     // This is meant to replicate numpy.meshgrid functionality
//     at::Tensor xx = x.view({s, 1, 1}).expand({s, s, s});
//     at::Tensor yy = y.view({1, s, 1}).expand({s, s, s});
//     at::Tensor zz = z.view({1, 1, s}).expand({s, s, s});
//     // {
//     //     std::ofstream ofile("xx_values.txt");
//     //     if ( ofile.is_open( ) ) {
//     //         for ( int a = 0; a < xx.size(2); a++ ) {
//     //             for ( int b = 0; b < xx.size(1); b++ ) {
//     //                 for ( int c = 0; c < xx.size(0); c++ ) {
//     //                     ofile << xx[c][b][a].item<float>( ) << std::endl;
//     //                 }
//     //             }
//     //         }
//     //         ofile.close( );
//     //     }
//     //     else {
//     //         wxPrintf("xx_values.txt failed to open.\n");
//     //     }
//     // }

//     // Radial distance; performs pixelwise comparison of the values at each point in all 3 tensors, selecting the one with the largest value at each pixel
//     // The max function actually returns both the raw values being compared along with the indices where the max value came from (including which tensor it came from)
//     //  We want the actual values before applying cosine falloff
//     xx                = xx.abs( );
//     yy                = yy.abs( );
//     zz                = zz.abs( );
//     at::Tensor radius = torch::zeros_like(xx);
//     {
//         // torch::max only accepts 2 args; break apart the comparison of the broadcasted linspace arrays
//         // Life cannot be so easy; I must use nested for to complete this comparison
//         // auto max1 = torch::max(xx.abs( ), yy.abs( ));
//         // radius    = torch::max(max1, zz.max( ));
//         for ( int k = 0; k < xx.size(2); k++ ) {
//             for ( int j = 0; j < xx.size(1); j++ ) {
//                 for ( int i = 0; i < xx.size(0); i++ ) {
//                     float max = std::max(zz[i][j][k].item<float>( ), std::max(xx[i][j][k].item<float>( ), yy[i][j][k].item<float>( )));
//                     // max             = torch::max(max, zz[i][j][k].item<float>( ));
//                     radius[i][j][k] = max;
//                 }
//             }
//         }
//     }
//     // {
//     //     std::ofstream ofile("radius_values.txt");
//     //     if ( ofile.is_open( ) ) {
//     //         for ( int a = 0; a < radius.size(2); a++ ) {
//     //             for ( int b = 0; b < radius.size(1); b++ ) {
//     //                 for ( int c = 0; c < radius.size(0); c++ ) {
//     //                     ofile << radius[c][b][a].item<float>( ) << std::endl;
//     //                 }
//     //             }
//     //         }
//     //         ofile.close( );
//     //     }
//     // }
//     // Cosine transformation; creates a smooth falloff where values decrease the further from the center you go
//     radius = torch::cos(radius / radius.max( ) * (M_PI / 2));

//     // Sets up the output tensor that will slice from each of the tensors
//     torch::Tensor weight_grid = torch::zeros({block_size, block_size, block_size});

//     auto slices = weight_grid.slice(0, margin, block_size - margin).slice(1, margin, block_size - margin).slice(2, margin, block_size - margin);
//     slices.copy_(radius);
//     weight_grid = weight_grid.clamp_min(1e-6); // avoid zeros

//     //Let 's just check what' s going on with the weights_grid...
//     {
//         // First, clear the text file
//         std::ofstream ofs;
//         ofs.open("initial_weights_grid_vals.txt", std::ofstream::trunc | std::ofstream::out);
//         ofs.close( );
//         ofs.open("initial_weights_grid_vals.txt", std::ios::app);
//         if ( ofs.is_open( ) ) {
//             for ( int a = 0; a < weight_grid.size(2); a++ ) {
//                 for ( int b = 0; b < weight_grid.size(1); b++ ) {
//                     for ( int c = 0; c < weight_grid.size(0); c++ ) {
//                         ofs << weight_grid[c][b][a].item<float>( ) << std::endl;
//                     }
//                 }
//             }
//             ofs.close( );
//         }
//         else {
//             std::cout << "initial_weights_grid_vals.txt did not open.\n"
//                       << std::endl;
//             // wxPrintf("initial_weights_grid_vals.txt did not open.\n\n");
//         }
//     }

//     return weight_grid;
// }

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
        throw std::runtime_error("Failed to open file for reading");

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