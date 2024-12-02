// LibTorch includes MUST come first!!
// LibTorch v2.5.0+cpu used to run this code.
#include <torch/nn/functional/conv.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>

// #include "blush_model.h" // includes torch/torch.h
#include "../../core/core_headers.h"

class
        BlushRefinement : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(BlushRefinement)

// Assumes cubic
// Image      generate_radial_mask(int box_size, float radius, int voxel_size); // TODO: maybe implement
at::Tensor get_local_std_dev(torch::Tensor grid, int size = 10, const bool use_dbg = false);

void BlushRefinement::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("Blush_Refinement", 1.0);

    std::string input_volume_filename = my_input->GetFilenameFromUser("Enter input mrc volume name", "The volume that will be denoised via blush refinement", "input.mrc", true);
    std::string input_mask_filename   = my_input->GetFilenameFromUser("Enter mask filename", "The mask that will be used to mask the volume and localized standard deviation.", "mask.mrc", true);
    std::string output_mrc_filename   = my_input->GetStringFromUser("Enter desired name of output mrc volume", "The denoised volume post blush refinement", "output.mrc");
    std::string model_path            = my_input->GetFilenameFromUser("Enter path to converted blush model", "The TorchScript converted form of the blush model", "blush_model.pt", true);

    // TODO: Come up with a better solution for including the weights, one that is automatic; perhaps adding a member to the database schema would be appropriate
    // Best answer I've come up with for now is create a checkbox in the auto-refine (or ab initio, can't remember) with a disabled textctrl that the user can specify a path in;
    // this can then be stored in the database? Idk, worth discussing with Tim if this even works
    // std::string weights_path = my_input->GetFilenameFromUser("Enter path to blush model weights", "Model weights pulled from the blush checkpoint file", "", true);
    bool use_dbg = my_input->GetYesNoFromUser("Use debug print statements?", "For development purposes and tracing program execution,", "No");
    // int         particle_diameter   = my_input->GetIntFromUser("Enter particle diameter", "Diameter of the particle in ", "180", 0, 2000);

    delete my_input;

    my_current_job.Reset(5);
    my_current_job.ManualSetArguments("ttttb", input_volume_filename.c_str( ),
                                      input_mask_filename.c_str( ),
                                      output_mrc_filename.c_str( ),
                                      //   weights_path.c_str( ),
                                      model_path.c_str( ),
                                      use_dbg);
    //   particle_diameter);
}

bool BlushRefinement::DoCalculation( ) {
    std::string input_volume_filename{my_current_job.arguments[0].ReturnStringArgument( )};
    std::string input_mask_filename{my_current_job.arguments[1].ReturnStringArgument( )};
    std::string output_mrc_filename{my_current_job.arguments[2].ReturnStringArgument( )};
    // std::string weights_filename{my_current_job.arguments[2].ReturnStringArgument( )};
    std::string model_filename{my_current_job.arguments[3].ReturnStringArgument( )};
    const bool  use_dbg{my_current_job.arguments[4].ReturnBoolArgument( )};
    // int         particle_diameter{my_current_job.arguments[2].ReturnIntegerArgument( )};

    // Load the model
    // Image                           input_mask; // May not need
    torch::jit::script::Module model;
    // auto                            model = std::make_shared<BlushModel>( );
    MRCFile                         input_mrc(input_volume_filename);
    MRCFile                         input_mask_mrc(input_mask_filename);
    Image                           input_volume;
    Image                           input_mask;
    at::Tensor                      real_values_tensor;
    at::Tensor                      in_std; // locally derived standard deviation -- passed to model
    const int                       batch_size  = 2;
    const int                       in_channels = 2;
    const int                       block_size  = 128;
    std::vector<torch::jit::IValue> inputs; // For passing tensors to model

    wxDateTime overall_start = wxDateTime::Now( );
    wxDateTime overall_finish;
    wxDateTime local_std_dev_start;
    wxDateTime local_std_dev_finish;
    wxDateTime blush_start;
    wxDateTime blush_finish;

    wxPrintf("Reading input and loading volume and input mask...\n");
    input_volume.ReadSlices(&input_mrc, 1, input_mrc.ReturnNumberOfSlices( ));
    if ( use_dbg )
        wxPrintf("Volume read in successfully.\n");

    input_mask.ReadSlices(&input_mask_mrc, 1, input_mask_mrc.ReturnNumberOfSlices( ));
    if ( use_dbg )
        wxPrintf("Mask read in successfully.\n");

    const int     box_size   = input_volume.logical_x_dimension;
    const int64_t num_pixels = std::pow(box_size, 3);
    at::Tensor    blocks;
    {
        std::vector<int64_t> tmp_blocks_vector = {batch_size, in_channels, block_size, block_size, block_size};
        blocks                                 = torch::zeros(tmp_blocks_vector, torch::TensorOptions( ).dtype(torch::kFloat32));
    }

    if ( use_dbg ) {
        wxPrintf("blocks shape: [%ld, %ld, %ld, %ld, %ld]\n\n",
                 blocks.size(0),
                 blocks.size(1),
                 blocks.size(2),
                 blocks.size(3),
                 blocks.size(4));
    }

    try {
        // FIXME: this works for now for testing purposes; but, this would need to be included alongside the project whenever blush was needed
        // model = torch::jit::load("/home/tim/VS_Projects/cisTEM/src/programs/blush_refinement/blush_model.pt");
        wxPrintf("Loading model...\n");
        // std::vector<char> model_data_vec(model_data, model_data + sizeof(model_data) / sizeof(model_data[0]));

        // std::istringstream model_stream(std::string(model_data_vec.begin( ), model_data_vec.end( )));

        model = torch::jit::load(model_filename);

        // torch::load(model, weights_filename);

        // For loading PyTorch model weights
        // std::unordered_map<std::string, torch::Tensor> state_dict;
        // torch::load(state_dict, model_filename);
        // model.load_weights(model_filename);
    } catch ( std::exception& e ) {
        wxPrintf("Failed to load blush model: %s\n", e.what( ));
        return false;
    }

    if ( use_dbg )
        wxPrintf("Blush model successfully loaded.\n");
    // const float pixel_size{input_mrc.ReturnPixelSize( )};

    // torch::Tensor model_voxel_size_tensor{torch::tensor({pixel_size})};
    // torch::Tensor box_size_tensor{torch::tensor({box_size})};

    input_volume.Normalize( );
    input_volume.RemoveFFTWPadding( );
    // FIXME: this may not work as an input; the model expects a 3D array, not a 1D array containing all elements of the 3D
    real_values_tensor = torch::from_blob(input_volume.real_values, {num_pixels}, torch::kFloat32);

    // Check the conversion; does it bust the array?
    if ( use_dbg ) {
        Image tmp_output_vol_test;
        tmp_output_vol_test.CopyFrom(&input_volume);
        std::memcpy(tmp_output_vol_test.real_values, real_values_tensor.data_ptr<float>( ), num_pixels * sizeof(float));
        tmp_output_vol_test.QuickAndDirtyWriteSlices("array_tensor_array_test.mrc", 1, input_mrc.ReturnNumberOfSlices( ), false);
    }
    real_values_tensor = real_values_tensor.view({box_size, box_size, box_size}).contiguous( ); // Shape this as a 3D volume instead of linearized 1D
    // Test if view, contiguous have any impact on the output
    if ( use_dbg ) {
        Image tmp_output_vol_test;
        tmp_output_vol_test.CopyFrom(&input_volume);
        std::memcpy(tmp_output_vol_test.real_values, real_values_tensor.data_ptr<float>( ), num_pixels * sizeof(float));
        tmp_output_vol_test.QuickAndDirtyWriteSlices("view_contiguous_array_tensor_array_test.mrc", 1, input_mrc.ReturnNumberOfSlices( ), false);
    }

    if ( use_dbg )
        wxPrintf("Loaded real_values into tensor; shape is: %ld, %ld, %ld\n", real_values_tensor.size(0), real_values_tensor.size(1), real_values_tensor.size(2));
    // Get standard dev, normalize
    {
        try {
            at::Tensor tmp_real_values_tensor = real_values_tensor.clone( );
            wxPrintf("Calculating localized standard deviation...\n\n\n");
            local_std_dev_start   = wxDateTime::Now( );
            at::Tensor tmp_in_std = get_local_std_dev(tmp_real_values_tensor.unsqueeze(0), 10, use_dbg);

            if ( use_dbg )
                wxPrintf("tmp_in_std shape: %ld, %ld, %ld, %ld\n", tmp_in_std.size(0), tmp_in_std.size(1), tmp_in_std.size(2), tmp_in_std.size(3));

            in_std = tmp_in_std / torch::mean(tmp_in_std); // Necessary if already normalized?
            wxPrintf("Standard deviation computed.\n");
            at::Tensor mean    = torch::mean(real_values_tensor);
            at::Tensor std_dev = torch::std(real_values_tensor);
            // This is just setting zero-mean (normalizing) -- may be able to just use already normalized real_values array in the future
            real_values_tensor = (tmp_real_values_tensor - mean) / (std_dev + 1e-8); // Get to do this because libtorch C++ backend automatically applies tensor ops to whole tensor
        } catch ( std::exception& e ) {
            wxPrintf("Error when getting standard deviation and normalizing the real_values tensor: %s\n", e.what( ));
            return false;
        }
    } // end getting std_dev, normalizing

    local_std_dev_finish           = wxDateTime::Now( );
    wxTimeSpan duration_of_std_dev = local_std_dev_finish.Subtract(local_std_dev_start);
    if ( use_dbg ) {
        // Write out the std dev
        Image tmp_in_std;
        tmp_in_std.CopyFrom(&input_volume);
        at::Tensor unsqueezed_in_std = in_std.clone( ).squeeze(0);
        std::memcpy(tmp_in_std.real_values, in_std.data_ptr<float>( ), num_pixels * sizeof(float));
        tmp_in_std.QuickAndDirtyWriteSlices("std_dev.mrc", 1, input_mrc.ReturnNumberOfSlices( ), false);
    }
    wxPrintf("Duration of calculating localized standard deviation:        %s\n\n", duration_of_std_dev.Format( ));
    if ( use_dbg )
        wxPrintf("real_values tensor acquired\n");

    // Handling the complex_values array is a little more complicated; have to create a tensor with real part, then imaginary part, then stack them
    // at::Tensor complex_tensor;

    // This complex thing will not be needed
    // Get complex tensor
    // {
    //     std::vector<float> real_part;
    //     std::vector<float> imag_part;
    //     // DEBUG:
    //     // wxPrintf("max size of float vector: %zu\n", imag_part.max_size( ));

    //     std::complex<float> cur_val;
    //     input_volume.ForwardFFT( );

    //     // NOTE: any time access of complex_values is needed, limit by real_memory_allocated / 2 for looping over all the pixels
    //     for ( int i = 0; i < input_volume.real_memory_allocated / 2; i++ ) {
    //         cur_val = input_volume.complex_values[i];
    //         real_part.push_back(real(cur_val));
    //         imag_part.push_back(imag(cur_val));
    //         // DEBUG:
    //         // wxPrintf("i == %i\n", i);
    //     }
    //     input_volume.BackwardFFT( );
    //     // Stack real and imaginary parts to make complex tensor
    //     at::Tensor real_tensor = torch::tensor(real_part);
    //     at::Tensor imag_tensor = torch::tensor(imag_part);
    //     complex_tensor         = torch::stack({real_tensor, imag_tensor}, 0);
    // } // End tensorising complex_values

    // DEBUG:
    // wxPrintf("Succesfully converted volume data to tensors\n");

    wxPrintf("Applying mask...\n");
    try {
        input_mask.RemoveFFTWPadding( );
        at::Tensor mask_tensor = torch::from_blob(input_mask.real_values, {num_pixels}, torch::kFloat32);
        mask_tensor            = mask_tensor.view({box_size, box_size, box_size}).contiguous( ); // Shape this as a 3D volume instead of linearized 1D
        real_values_tensor *= mask_tensor;
        in_std *= mask_tensor;
    } catch ( std::exception& e ) {
        wxPrintf("Error generating/applying mask. %s\n", e.what( ));
        return false;
    }

    // Write out masked images to inspect what's going on
    if ( use_dbg ) {
        Image output_masked_volume;
        Image output_masked_std_dev;
        output_masked_volume.CopyFrom(&input_volume);
        output_masked_std_dev.CopyFrom(&input_volume);

        at::Tensor tmp_real_values = real_values_tensor.clone( ).squeeze(0);
        at::Tensor tmp_in_std      = in_std.clone( ).squeeze(0);

        std::memcpy(output_masked_volume.real_values, tmp_real_values.data_ptr<float>( ), num_pixels * sizeof(float));
        std::memcpy(output_masked_std_dev.real_values, tmp_in_std.data_ptr<float>( ), num_pixels * sizeof(float));
        output_masked_volume.QuickAndDirtyWriteSlices("masked_volume.mrc", 1, input_mrc.ReturnNumberOfSlices( ), false);
        output_masked_std_dev.QuickAndDirtyWriteSlices("masked_std_dev.mrc", 1, input_mrc.ReturnNumberOfSlices( ), false);
    }
    // torch::jit:IValue is type-erased
    try {
        real_values_tensor = real_values_tensor.view({1, box_size, box_size, box_size});
    } catch ( std::exception& e ) {
        wxPrintf("Error line 143 calling view on real_values_tensor: %s\n", e.what( ));
    }
    // FIXME: instead of using this test_tensor, actually just get the normalized stddev and pass it alongside the volume
    try {
        in_std = in_std.view({1, box_size, box_size, box_size});
    } catch ( std::exception& e ) {
        wxPrintf("Error line 149 calling view on in_std: %s\n", e.what( ));
        return false;
    }
    if ( use_dbg ) {
        wxPrintf("\nreal_values_tensor shape: [%ld, %ld, %ld, %ld]\n\n",
                 real_values_tensor.size(0),
                 real_values_tensor.size(1),
                 real_values_tensor.size(2),
                 real_values_tensor.size(3));
        wxPrintf("in_std shape = [%ld, %ld, %ld, %ld]\n\n",
                 in_std.size(0),
                 in_std.size(1),
                 in_std.size(2),
                 in_std.size(3));
    }
    // For debugging/knowledge check purposes: index of linearized form of 3D volume: z * box_size^2 + y * box_size + x
    // This should get the final pixel value of the first slice
    // if ( use_dbg ) {
    //     // wxPrintf("real_values_tensor[0][191][191][191]: %f\n", real_values_tensor[0][191][191][191].item<float>( )); // [191][191]
    //     // wxPrintf("original real_values at corresponding 1D position: %f\n", input_volume.real_values[7077887]); // 36863
    //     // wxPrintf("Current padding jump value: %i\n", input_volume.padding_jump_value);
    // }

    // TODO: Create tensor for in_channels?
    try {
        inputs.push_back(real_values_tensor);
        inputs.push_back(in_std);
    } catch ( std::exception& e ) {
        wxPrintf("Error adding tensors to inputs tensor; %s", e.what( ));
        return false;
    }
    // inputs.push_back(complex_tensor);
    // inputs.push_back(model_voxel_size_tensor);
    // inputs.push_back(box_size_tensor);

    if ( use_dbg )
        wxPrintf("Successfully passed all inputs\n");

    // Curious of how the tensors will be returned, given C++ doesn't support multiple returns --
    // recommended way to handle this normally is with a tuple, so hopefully it will be something of that sort
    // std::tuple<at::Tensor, at::Tensor> output;
    at::Tensor output;

    // TODO: implement batch processing in the same way as apply_model in util.py
    try {
        wxPrintf("Running blush model...\n");
        blush_start  = wxDateTime::Now( );
        output       = model.forward(inputs).toTensor( );
        blush_finish = wxDateTime::Now( );
    } catch ( std::exception& e ) {
        wxPrintf("Couldn't run model; exception occurred: %s\n", e.what( ));
        return false;
    }

    wxTimeSpan blush_duration = blush_finish.Subtract(blush_start);
    wxPrintf("Finished running blush model. Total duration:        %s\n", blush_duration.Format( ));
    if ( use_dbg ) {
        wxPrintf("Successfully ran model.forward with inputs\n");
        wxPrintf("\noutput tensor shape: [%ld, %ld, %ld, %ld]\n\n",
                 output.size(0),
                 output.size(1),
                 output.size(2),
                 output.size(3));
        wxPrintf("About to remove the first dimension and make contiguous in memory...\n");
    }

    // ChatGPT suggests that squeeze ops can lead to non-contiguous memory changes; try leaving contiguous in
    output = output.squeeze(0);
    output = output.contiguous( );

    if ( use_dbg ) {
        wxPrintf("About to copy input volume to output volume...\n");
    }
    Image output_volume;
    output_volume.CopyFrom(&input_volume);
    if ( use_dbg )
        wxPrintf("About to copy output tensor to real_values in output volume...\n");
    std::memcpy(output_volume.real_values, output.data_ptr<float>( ), num_pixels * sizeof(float));
    output_volume.AddFFTWPadding( );

    if ( use_dbg )
        wxPrintf("About to write out the denoised output volume\n");
    output_volume.QuickAndDirtyWriteSlices(output_mrc_filename, 1, input_mrc.ReturnNumberOfSlices( ));

    wxPrintf("Blush complete.\n");
    // input_volume.ForwardFFT( );
    // input_volume.Normalize( );
    overall_finish      = wxDateTime::Now( );
    wxTimeSpan duration = overall_finish.Subtract(overall_start);
    wxPrintf("Total blush runtime:         %s\n", duration.Format( ));

    return true;
}

// FIXME: something went wrong: the dimensions of std is wrong somehow
at::Tensor get_local_std_dev(torch::Tensor grid, int size, const bool use_dbg) {
    // Unsqueeze and clone the grid tensor
    if ( use_dbg )
        wxPrintf("Dimensions of grid: %ld, %ld, %ld, %ld\n", grid.size(0), grid.size(1), grid.size(2), grid.size(3));

    at::Tensor new_grid = grid.clone( ).unsqueeze(1);
    at::Tensor grid2    = new_grid.square( );

    if ( use_dbg )
        wxPrintf("Dimensions of new_grid (after unsqueezing grid): %ld, %ld, %ld, %ld, %ld\n", new_grid.size(0), new_grid.size(1), new_grid.size(2), new_grid.size(3), new_grid.size(4));

    // Create the kernel
    at::Tensor ls     = torch::linspace(-1.5, 1.5, 2 * size + 1); // Create tensor of size 2 * size + 1 with evenly spaced values between -1.5 and 1.5
    at::Tensor kernel = torch::exp(-ls.square( )); // Make normal dist (gets rid of negatives (square), applies exponential function(exp))
    kernel /= kernel.sum( );

    // Something going wrong here; kernel shape is [1, 1, 1, 1, 21]
    kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4); // Shape: (1, 1, kernel_size, 1, 1)
    // kernel = kernel.expand({1, 1, 2 * size + 1, 2 * size + 1, 2 * size + 1});

    if ( use_dbg )
        wxPrintf("kernel shape: [%ld, %ld, %ld, %ld, %ld]\n", kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3), kernel.size(4));

    // Set options, then perform 3D convolution
    torch::nn::functional::Conv3dFuncOptions options;
    options.stride(1).padding({size, 0, 0});

    // NOTE: This test using the full 1,1,21,21,21 kernel and doing a single convolution had ~14m:30s runtime; too long, not worth pursuing
    // However, the output seemed a little better, though smearing persisted in the first images
    // if ( use_dbg )
    //     wxPrintf("Convolving new_grid...\n");
    // new_grid = torch::nn::functional::conv3d(new_grid, kernel, options);
    // if ( use_dbg )
    //     wxPrintf("new_grid shape: [%ld, %ld, %ld, %ld, %ld]\n", new_grid.size(0), new_grid.size(1), new_grid.size(2), new_grid.size(3), new_grid.size(4));

    // if ( use_dbg )
    //     wxPrintf("Convolving grid2...\n");
    // grid2 = torch::nn::functional::conv3d(grid2, kernel, options);
    // if ( use_dbg ) {
    //     wxPrintf("grid2 shape: [%ld, %ld, %ld, %ld, %ld]\n", grid2.size(0), grid2.size(1), grid2.size(2), grid2.size(3), grid2.size(4));
    // }
    for ( int i = 0; i < 3; ++i ) {
        if ( use_dbg ) {
            wxPrintf("\nNew convolution started\n");
            wxPrintf("Permuting new_grid\n");
        }

        new_grid = new_grid.permute({0, 1, 4, 2, 3}); // Shift (N, C, D, W, H) then (N, C, H, D, W) and finally (N, C, W, H, D)
        if ( use_dbg )
            wxPrintf("new_grid shape: [%ld, %ld, %ld, %ld, %ld]\n", new_grid.size(0), new_grid.size(1), new_grid.size(2), new_grid.size(3), new_grid.size(4));
        new_grid = new_grid.contiguous( );
        // new_grid = new_grid.permute({0, 1, 2, 3, 4}); // Change shape to (N, C, D, H, W)
        // torch::nn::functional::Conv3dFuncOptions options;
        // options.stride({1, 1, 1});
        // options.stride(1);
        // options.padding(torch::nn::functional::PadMode::SAME);

        if ( use_dbg )
            wxPrintf("Performing 3D convolution on new_grid\n");

        new_grid = torch::nn::functional::conv3d(new_grid, /*weight=*/kernel, options);

        if ( use_dbg )
            wxPrintf("Permuting grid2\n");

        grid2 = grid2.permute({0, 1, 4, 2, 3}); // Same permutation
        grid2 = grid2.contiguous( );
        if ( use_dbg )
            wxPrintf("grid2 shape: [%ld, %ld, %ld, %ld, %ld]\n", grid2.size(0), grid2.size(1), grid2.size(2), grid2.size(3), grid2.size(4));

        if ( use_dbg )
            wxPrintf("Performing 3D convolution on grid2\n");

        grid2 = torch::nn::functional::conv3d(grid2, /*weight=*/kernel, options);
    }

    at::Tensor std = torch::sqrt(torch::clamp(grid2 - new_grid.square( ), 0));
    if ( use_dbg )
        wxPrintf("\nAfter get_local_std_dev (pre squeeze): %ld, %ld, %ld, %ld, %ld\n", std.size(0), std.size(1), std.size(2), std.size(3), std.size(4));
    std = std.squeeze(1); // Remove the second dimension
    if ( use_dbg )
        wxPrintf("After std after squeeze: %ld, %ld, %ld, %ld\n\n", std.size(0), std.size(1), std.size(2), std.size(3));

    return std;
}