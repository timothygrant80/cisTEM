// LibTorch includes MUST come first!!
// LibTorch v2.5.0+cpu used to run this code.
#include <torch/nn/functional/conv.h>
#include <torch/script.h>
// #include <torch/torch.h>
#include <chrono>

#include "../../core/core_headers.h"
#include <numeric>

#include "blush_model.h" // includes torch/torch.h

using namespace torch::indexing;
using torch::Tensor;

class
        BlushRefinement : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

/**
 * @brief Class enabling iteration over subsets of a volume.
 * Theoretically useful for 1D and 2D objects as well. It handles the state
 * of the container and manages ranges.
 * 
 */
class BlockIterator {
  public:
    BlockIterator(const std::tuple<int, int, int>& dims, const int& block_size, const int& strides) : x_index(0), y_index(0), z_index(0), index(0) {
        x_range = GetStartingIndices(std::get<0>(dims), block_size, strides);
        y_range = GetStartingIndices(std::get<1>(dims), block_size, strides);
        z_range = GetStartingIndices(std::get<2>(dims), block_size, strides);
    }

    /**
     * @brief Encapsulates current state of the iteration (i.e. the current block)
     * and controls movement to next block.
     * 
     */
    class Iterator {
      public:
        Iterator(BlockIterator& parent, bool is_end = false) : parent(parent), x_index(0), y_index(0), z_index(0), index(0), is_end(is_end){ };

        // Sort of backward logic; want this to return false when reaching the end, not true; so, we invert
        // is_end within this check.
        bool operator!=(const Iterator& other) {
            return ! is_end;
        }

        std::tuple<int, int, int> operator*( ) {
            return {parent.x_range[x_index], parent.y_range[y_index], parent.z_range[z_index]};
        }

        Iterator& operator++( ) {
            x_index++;
            if ( x_index == parent.x_range.size( ) ) {
                x_index = 0;
                y_index++;
            }

            if ( y_index == parent.y_range.size( ) ) {
                y_index = 0;
                z_index++;
            }

            if ( z_index == parent.z_range.size( ) )
                is_end = true;
            else
                index++;

            return *this;
        }

      private:
        BlockIterator& parent;
        int            x_index, y_index, z_index, index;
        bool           is_end;
    };

    // Begin and end iteration; begin is called when starting the range-based for loop,
    // while end should be called when the loop is complete
    Iterator begin( ) {
        // wxPrintf("\n\n\nBegin is being called...\n\n\n");
        return Iterator(*this, false);
    }

    Iterator end( ) {
        // wxPrintf("\n\n\nEnd is being called...\n\n\n");
        return Iterator(*this, true);
    }

    /**
     * @brief Get the next coords to be used for slices that will be passed to Blush.
     * 
     * @return std::tuple<int, int, int> Contains the coordinates to be used in blocks. If these integers are negative, iterating will cease.
     */
    // std::tuple<int, int, int> next( ) {
    //     if ( ! has_next( ) ) {
    //         return {-1, -1, -1};
    //     }

    //     if ( x_index == x_range.size( ) ) {
    //         x_index = 0;
    //         y_index++;
    //     }
    //     if ( y_index == y_range.size( ) ) {
    //         y_index = 0;
    //         z_index++;
    //     }

    //     std::tuple<int, int, int> coords{x_range[x_index], y_range[y_index], z_range[z_index]};
    //     x_index++;
    //     index++;

    //     return coords;
    // }

  private:
    int              x_index, y_index, z_index, index;
    std::vector<int> x_range, y_range, z_range;

    /**
     * @brief Based on numpy arange; it creates a vector with the specified
     * endpoint based on the desired block size and dimension. For Blush purposes,
     * it will give the starting indices of the blocks that will be processed through
     * the model.
     * 
     * @param dim Size of the dimension in question (usually x, y, or z).
     * @param block_size Size of the block's dimension (x, y, or z).
     * @param strides Amount by which to increment each element.
     * @return std::vector<int> Contains indices for start of each block.
     */
    std::vector<int> GetStartingIndices(const int& dim, const int& block_size, const int& strides) {
        // Difference between image dim and block dim to determine final index
        int              span    = dim - block_size;
        int              n_steps = (span / strides) + 1;
        std::vector<int> r(n_steps);
        for ( int i = 0; i < n_steps; i++ ) {
            r[i] = i * strides;
            // wxPrintf("r[%i] == %i\n", i, r[i]);
        }

        if ( r.back( ) != span )
            r.push_back(span);
        return r;
    };

    /**
     * @brief Check if this iterator has reached the end of the block.
     * 
     * @return true Reached end.
     * @return false Have not yet reached end.
     */
    // bool has_next( ) {
    //     return z_index < z_range.size( );
    // }
};

IMPLEMENT_APP(BlushRefinement)

// Assumes cubic
// Image      generate_radial_mask(int box_size, float radius, int voxel_size); // TODO: maybe implement
Tensor get_local_std_dev(torch::Tensor grid, Image& orig, int size = 10, const bool use_dbg = false);

Tensor                                       make_weight_box(const int& block_size, int margin);
std::vector<std::vector<std::vector<float>>> cubic_zeros_vector(const int& box_size);
void                                         test_weight_box_output(const std::string& txt_filename, const Tensor& cpp_output, const int& weight_box_size);
std::vector<float>                           localized_std_dev(Tensor& real_values, const long& num_pixels, const int& box_size, const int& kernel_size, const int& max_threads);
Tensor                                       slice_tensor(const Tensor input, const int& x, const int& y, const int& z, const int& block_size);
Tensor                                       generate_radial_mask(const int& box_size, const float& radius, const int& mask_edge_width);
void                                         tesnor_slicing_test( );
void                                         write_text_files(Image& volume, Image& mask);
bool                                         test_slicing_of_std_dev_and_real_values(Tensor real_slice, Tensor std_dev_slice, Tensor full_real, Tensor full_std_dev, const int& x, const int& y, const int& z);
void                                         write_real_values_to_text_file(const std::string& ofname, const int& box_size, const bool& is_single_dim, bool overwrite = true, Tensor* tensor_data = nullptr, float* array_data = nullptr);
void                                         compare_text_file_lines(std::string fname1, std::string fname2, std::string ofname, const bool& print_vals = true);
void                                         calculate_average_difference(const std::string& ofname, const int& max_threads, const int& box_size, Tensor* t_data1 = nullptr, Tensor* t_data2 = nullptr, float* a_data1 = nullptr, float* a_data2 = nullptr);
void                                         read_real_values_from_text_file(const std::string& fname, std::unique_ptr<float[]>& array_to_fill, const int& box_size);

void BlushRefinement::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("Blush_Refinement", 1.0);

    std::string input_volume_filename = my_input->GetFilenameFromUser("Enter input mrc volume name", "The volume that will be denoised via blush refinement", "input.mrc", true);
    // std::string input_mask_filename   = my_input->GetFilenameFromUser("Enter mask filename", "The mask that will be used to mask the volume and localized standard deviation.", "mask.mrc", true);
    std::string output_mrc_filename = my_input->GetStringFromUser("Enter desired name of output mrc volume", "The denoised volume post blush refinement", "output.mrc");
    // std::string model_path          = my_input->GetFilenameFromUser("Enter path to converted blush model", "The TorchScript converted form of the blush model", "blush_model.pt", true);

    // TODO: Come up with a better solution for including the weights, one that is automatic; perhaps adding a member to the database schema would be appropriate
    // Best answer I've come up with for now is create a checkbox in the auto-refine (or ab initio, can't remember) with a disabled textctrl that the user can specify a path in;
    // this can then be stored in the database? Idk, worth discussing with Tim if this even works
    // std::string weights_path = my_input->GetFilenameFromUser("Enter path to blush model weights", "Model weights pulled from the blush checkpoint file", "", true);
    bool use_dbg = my_input->GetYesNoFromUser("Use debug print statements?", "For development purposes and tracing program execution,", "No");
    // int         particle_diameter   = my_input->GetIntFromUser("Enter particle diameter", "Diameter of the particle in ", "180", 0, 2000);
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Enter desired number of threads", "Number of threads for parallelizing localized standard deviation calculation", "1", 1, 256);
#endif

    delete my_input;

    my_current_job.Reset(5);
    my_current_job.ManualSetArguments("ttbi", input_volume_filename.c_str( ),
                                      //   input_mask_filename.c_str( ),
                                      output_mrc_filename.c_str( ),
                                      //   weights_path.c_str( ),
                                      //   model_path.c_str( ),
                                      use_dbg,
                                      max_threads);
    //   particle_diameter);
}

// NOTE: all libtorch functionality sorts by and assumes z, y, x rather than x, y, z
bool BlushRefinement::DoCalculation( ) {

    // compare_text_file_lines("real_values.txt", "misc_txts/py_real_values.txt", "misc_txts/comp_real_values.txt");
    // compare_text_file_lines("real_values.txt", "model_input_txts/py_1D_tensor_real_values.txt", "comp_torchscript_0_60_60.txt");
    // compare_text_file_lines("real_values.txt", "model_input_txts/py_3D_tensor_real_values_zyx.txt", "comp_torchscript_0_60_60_zyx.txt");
    // compare_text_file_lines("real_values.txt", "model_input_txts/py_3D_tensor_real_values_xyz.txt", "comp_torchscript_0_60_60_xyz.txt");
    // compare_text_file_lines("model_input_txts/real_vals_sliced_0_60_60.txt", "model_input_txts/python_torchscript_model_input_0_60_60.txt", "comp_torchscript_xyz_0_60_60.txt");
    // return false;

    // std::unique_ptr<float[]> rvarr(new float[64 * 64 * 64]);
    // std::unique_ptr<float[]> isarray(new float[64 * 64 * 64]);
    // Tensor                   rv = torch::zeros({1, 64, 64, 64});
    // Tensor                   sd = torch::zeros({1, 64, 64, 64});
    // read_real_values_from_text_file("model_input_txts/python_torchscript_model_input_0_60_60.txt", rvarr, 64);
    // read_real_values_from_text_file("model_input_txts/python_torchscript_in_std_model_input_0_60_60.txt", isarray, 64);

    // // zyx, so x_ will take on the calculation that is typically used for z
    // for ( int pixel_counter = 0; pixel_counter < 64 * 64 * 64; pixel_counter++ ) {
    //     int x_ = pixel_counter / (64 * 64);
    //     int y_ = (pixel_counter / 64) % 64;
    //     int z_ = pixel_counter % 64;

    //     rv.index({0, z_, y_, x_}) = rvarr[pixel_counter];
    //     sd.index({0, z_, y_, x_}) = isarray[pixel_counter];
    // }
    // write_real_values_to_text_file("python_read_in_rv_0_60_60.txt", 64, false, true, &rv);
    // write_real_values_to_text_file("python_read_in_sd_0_60_60.txt", 64, false, true, &sd);
    // compare_text_file_lines("python_read_in_rv_0_60_60.txt", "model_input_txts/python_torchscript_model_input_0_60_60.txt", "comp_rv_0_60_60.txt");
    // compare_text_file_lines("python_read_in_sd_0_60_60.txt", "model_input_txts/python_torchscript_in_std_model_input_0_60_60.txt", "comp_sd_0_60_60.txt");
    // return false;

    // return false;
    // Minimal Slice issue example:
    // using namespace torch::indexing::Slice;
    using torch::Tensor;
    // {
    //     // 1. Create a multidimensional tensor.
    //     try {
    //         int                      size     = 27;
    //         float*                   tmp_data = new float[size + 100];
    //         std::shared_ptr<float[]> original_data(new float[size]);
    //         std::complex<float>*     complex_values = reinterpret_cast<std::complex<float>*>(tmp_data);
    //         // float* original_data = new float[size];
    //         float sum = 0;
    //         for ( int i = 0; i < size; i++ ) {
    //             // original_data[i] = i + 1;
    //             tmp_data[i]       = i + 1;
    //             complex_values[i] = std::complex<float>(tmp_data[i], 0);
    //             original_data[i]  = tmp_data[i];
    //             sum += original_data[i];
    //         }
    //         Tensor with_additional_complex_ptr = torch::from_blob(tmp_data, {size}, torch::kFloat32);
    //         wxPrintf("mean of complex ptr: %f\n", torch::mean(with_additional_complex_ptr).item<float>( ));

    //         float  mean            = sum / size;
    //         int    box_size        = 3;
    //         Tensor tensorized_data = torch::from_blob(original_data.get( ), {size}, torch::kFloat32);

    //         long   numel        = tensorized_data.numel( );
    //         size_t element_size = tensorized_data.element_size( );
    //         wxPrintf("numel: %lli\n", numel);
    //         wxPrintf("element size: %zu\n", element_size);
    //         wxPrintf("Total mem: %lli\n", numel * element_size);
    //         wxPrintf("Expected mem: %lli\n", size * sizeof(float));
    //         wxPrintf("Reshaping...\n");

    //         tensorized_data = tensorized_data.reshape({box_size, box_size, box_size});
    //         // tensorized_data = tensorized_data.swapaxes(0, 2);
    //         wxPrintf("Looping...\n");
    //         // 2. Print to verify appropriate shape and values.
    //         for ( int k = 0; k < box_size; k++ ) {
    //             for ( int j = 0; j < box_size; j++ ) {
    //                 for ( int i = 0; i < box_size; i++ ) {
    //                     wxPrintf("%f ", tensorized_data.index({i, j, k}).item<float>( ));
    //                 }
    //                 wxPrintf("\n");
    //             }
    //             wxPrintf("\n");
    //         }
    //         float tensor_mean = tensorized_data.mean(torch::kFloat32).item<float>( );
    //         wxPrintf("manual mean: %f; tensor_mean: %f\n", mean, tensor_mean);
    //         delete[] complex_values;
    //         // delete[] original_data;
    //     } catch ( std::exception& e ) {
    //         wxPrintf("Failed to generate sample run: %s\n", e.what( ));
    //         std::cout << "Failed to generate the post-processing grids. " << e.what( ) << std::endl;
    //         // return false;
    //     }
    //     // 3. Slice the tensor.
    //     // 4. Print the values after slicing to see if they are correct.
    //     //
    //     // return true;
    // }

    // tesnor_slicing_test( );
    std::string input_volume_filename{my_current_job.arguments[0].ReturnStringArgument( )};
    // std::string input_mask_filename{my_current_job.arguments[1].ReturnStringArgument( )};
    std::string output_mrc_filename{my_current_job.arguments[1].ReturnStringArgument( )};
    // std::string weights_filename{my_current_job.arguments[2].ReturnStringArgument( )};
    // std::string model_filename{my_current_job.arguments[3].ReturnStringArgument( )};
    const bool use_dbg{my_current_job.arguments[2].ReturnBoolArgument( )};
    const int  max_threads{my_current_job.arguments[3].ReturnIntegerArgument( )};
    // int         particle_diameter{my_current_job.arguments[2].ReturnIntegerArgument( )};

    std::string     model_filename;
    constexpr float model_voxel_size{1.5f}; // The expected voxel/pixel size of inputs to the blush model
    constexpr int   model_block_size{64};
    constexpr int   num_subset_pixels{model_block_size * model_block_size * model_block_size};
    constexpr int   strides{20};
    constexpr int   in_channels{2};
    constexpr int   batch_size{1};

    // QUICK DEBUG: COMPARE COUNT GRID AND INFER GRID FINAL STATE WITH THE FINAL PYTHON STATE
    // compare_text_file_lines("python_infer_grid_final.txt", "infer_grid_vals.txt", "infer_grid_comp.txt");
    // compare_text_file_lines("python_count_grid_final.txt", "count_grid_vals.txt", "count_grid_comp.txt");
    // compare_text_file_lines("cpp_pre_normalized_std_dev1.txt", "cpp_pre_normalized_std_dev2.txt", "cpp_conv_comp.txt", true);
    // compare_text_file_lines("model_output_txts/cpp_output_40_20_80.txt", "model_output_txts/python_output_40_20_80.txt", "model_output_txts/comp_output_40_20_80.txt", true);
    // compare_text_file_lines("model_input_txts/vol_input_block_0_60_60.txt", "model_input_txts/py_vol_input_0_60_60.txt", "model_input_txts/comp_input_block_0_60_60.txt", true);
    // compare_text_file_lines("model_output_txts/python_torchscript_model_output_0_60_60.txt", "model_output_txts/python_output_w_c_inputs_0_60_60.txt", "comp_python_pytorch_torchscript_w_c_inputs_0_60_60.txt");
    // abort( );

    // NOTE: This is crucial for giving improved runtime for tensor ops
    torch::set_num_threads(1);
    // torch::set_num_interop_threads(max_threads);

    // NOTE: These are parameters that are directly relevant to the processing in the model
    // model_voxel_size and model_block_size are values that are predetermined in model parameters; the latter 3
    // could technically be user specified.

    BlushModel model(2, 2);
    try {
        // DEBUG: write out model params for comparison to python layers
        std::ofstream params_txt("cpp_named_params.txt", std::iostream::trunc | std::iostream::out);
        params_txt.close( );
        params_txt.open("cpp_named_params.txt", std::ios::app);
        if ( params_txt.is_open( ) ) {
            for ( auto& param : model.named_parameters( ) ) {
                params_txt << param.key( ) << std::endl;
            }
        }
        else {
            wxPrintf("Error opening cpp_named_params.txt");
        }
        model.load_weights("test_weights_saving_function.dat");
    } catch ( std::exception& e ) {
        wxPrintf("%s\n", e.what( ));
    }

    // Set to evaluation mode:
    model.eval( );

    MRCFile input_mrc(input_volume_filename);
    float   vol_original_pixel_size = input_mrc.ReturnPixelSize( );
    Image   input_volume;

    float scale_factor = vol_original_pixel_size / model_voxel_size;
    // Uh-oh, our new box size isn't even; make it so
    int new_box_size = input_mrc.ReturnXSize( ) * scale_factor;

    // Improve this; let's not assume we always increase; pick the smaller difference
    if ( new_box_size % 2 != 0 ) {
        float poss_voxel_size_1 = (vol_original_pixel_size * new_box_size) / (new_box_size + 1);
        float poss_voxel_size_2 = (vol_original_pixel_size * new_box_size) / (new_box_size - 1);
        if ( abs(poss_voxel_size_1 - model_voxel_size) < abs(poss_voxel_size_2 - model_voxel_size) )
            new_box_size++;
        else
            new_box_size--;
    }

    if ( use_dbg )
        wxPrintf("new_box_size == %i\n", new_box_size);

    // std::vector<torch::jit::IValue> inputs; // For passing tensors to model

    wxDateTime overall_start = wxDateTime::Now( );
    wxDateTime overall_finish;
    wxDateTime local_std_dev_start;
    wxDateTime local_std_dev_finish;
    wxDateTime blush_start;
    wxDateTime blush_finish;

    wxPrintf("Reading input and loading volume and input mask...\n");
    input_volume.ReadSlices(&input_mrc, 1, input_mrc.ReturnNumberOfSlices( ));
    wxPrintf("scale_factor == %f, scale_factor * logical_x == %f\n", scale_factor, scale_factor * float(input_volume.logical_x_dimension));
    if ( use_dbg )
        wxPrintf("Volume read in successfully.\n");

    if ( scale_factor > 1.0f ) {
        input_volume.ForwardFFT( );
        input_volume.Resize(new_box_size, new_box_size, new_box_size);
        input_volume.BackwardFFT( );
    }

    const int                     box_size   = input_volume.logical_x_dimension;
    const long                    num_pixels = std::pow(box_size, 3);
    Tensor                        blocks(at::zeros({batch_size, model_block_size, model_block_size, model_block_size}));
    std::vector<std::vector<int>> coords(batch_size, std::vector<int>(3, 0));

    if ( use_dbg ) {
        wxPrintf("blocks shape: [%ld, %ld, %ld, %ld]\n\n",
                 blocks.size(0),
                 blocks.size(1),
                 blocks.size(2),
                 blocks.size(3));
    }

    // Read traced model
    // {
    //     // NOTE: this presupposes that the traced model will be present within the execution directory. It would
    //     // be useful to find a way to store the traced model in the cisTEM/src/programs/blush_refinement
    //     // and copy it to the build directory during compiling to simplify packaging the model alongside
    //     // the binaries.
    //     wxString model_directory{wxStandardPaths::Get( ).GetExecutablePath( )};
    //     model_directory = model_directory.BeforeLast('/');
    //     model_directory += "/traced_blush_with_script.pt";
    //     model_filename = model_directory.ToStdString( );
    //     // DEBUG:
    //     wxPrintf("\nModel path: %s\n\n", model_filename);
    // }

    if ( use_dbg )
        wxPrintf("Blush model successfully loaded.\n");
    Tensor real_values_tensor = torch::zeros({box_size, box_size, box_size}, torch::kFloat32);

    Tensor in_std;
    in_std = torch::zeros({box_size, box_size, box_size}); // locally derived standard deviation -- passed to model
    input_volume.RemoveFFTWPadding( );

    real_values_tensor = torch::zeros({box_size, box_size, box_size}, torch::kFloat32);
    real_values_tensor = torch::from_blob(input_volume.real_values, {box_size, box_size, box_size}, torch::kFloat32).contiguous( );
    real_values_tensor = real_values_tensor.permute({2, 1, 0}).contiguous( );
    // real_values_tensor = torch::swapaxes(real_values_tensor, 0, 2); // Torch expects z-fastest (i.e., z, y, x organization) rather than x-fastest, which cisTEM uses, so swap those axes
    // write_real_values_to_text_file("reshaped_cpp_vals.txt", box_size, false, true, &real_values_tensor);
    // compare_text_file_lines("reshaped_cpp_vals.txt", "py_tensorized_real_values.txt", "comp_read_in_values.txt");
    // #pragma omp parallel for collapse(3) num_threads(max_threads)
    //     for ( int z = 0; z < box_size; ++z ) {
    //         for ( int y = 0; y < box_size; ++y ) {
    //             for ( int x = 0; x < box_size; ++x ) {
    //                 // x changes fastest in your array
    //                 int idx                     = x + y * box_size + z * box_size * box_size;
    //                 real_values_tensor[x][y][z] = input_volume.real_values[idx];
    //             }
    //         }
    //     }

    // try {
    //     // Compare whole block
    //     write_real_values_to_text_file("full_tensor_cpp.txt", box_size, false, true, &real_values_tensor);
    //     compare_text_file_lines("full_tensor_cpp.txt", "full_tensor_py.py", "new_comp_full_tensor.txt");

    //     // Compare slice
    //     auto slice_of_rv = real_values_tensor.slice(0, 0, 64, 1).slice(1, 0, 64, 1).slice(2, 0, 64, 1);
    //     write_real_values_to_text_file("sliced_tensor_cpp.txt", model_block_size, false, true, &slice_of_rv);
    //     compare_text_file_lines("sliced_tensor_cpp.txt", "sliced_tensor_py.py", "new_comp_slicing.txt");
    // } catch ( std::exception& e ) {
    //     wxPrintf("%s\n", e.what( ));
    // }

    if ( torch::any(torch::isnan(real_values_tensor)).item<bool>( ) ) {
        wxPrintf("This tensor contains NaN values.\n");
    }
    if ( torch::any(torch::isinf(real_values_tensor)).item<bool>( ) ) {
        wxPrintf("Tensor contains inf values.\n");
    }
    // try {
    //     real_values_tensor = real_values_tensor.permute({2, 1, 0}).contiguous( );
    // } catch ( std::exception& e ) {
    //     wxPrintf("Error permuting: %s\n", e.what( ));
    // }

    wxPrintf("Image mean: %f\n", input_volume.ReturnAverageOfRealValues( ));
    wxPrintf("Tensor mean: %f\n", torch::mean(real_values_tensor).item<float>( ));
    wxPrintf("Other tensor mean (.mean()): %f\n", real_values_tensor.mean( ).item<float>( ));

    if ( use_dbg )
        wxPrintf("Loaded real_values into tensor; shape is: %ld, %ld, %ld\n", real_values_tensor.size(0), real_values_tensor.size(1), real_values_tensor.size(2));
    // Get standard dev, normalize

    wxPrintf("Generating post-processing grids.\n");

    Tensor weights_tensor = make_weight_box(model_block_size, 10);
    // wxPrintf("Comparing weights tensor to Python weights tensor...\n");
    // compare_text_file_lines("initial_weights_grid_vals.txt", "py_weight_block.txt", "comp_weights_grid_vals.txt", true);
    std::vector<float> weights_grid(std::pow(model_block_size, 3));
    std::memcpy(weights_grid.data( ), weights_tensor.data_ptr<float>( ), std::pow(model_block_size, 3) * sizeof(float));

    // std::vector<float> infer_grid(num_pixels);
    // std::vector<float> count_grid(num_pixels);
    Tensor infer_grid = torch::zeros({box_size, box_size, box_size}, torch::kFloat32);
    Tensor count_grid = torch::zeros({box_size, box_size, box_size}, torch::kFloat32);

    wxPrintf("Completed generation of post-processing grids.\n");

    float mean    = input_volume.ReturnAverageOfRealValues( );
    float std_dev = sqrt(input_volume.ReturnSumOfSquares( ));
    try {
        Tensor tmp_real_values_tensor = real_values_tensor.clone( );
        wxPrintf("Calculating localized standard deviation...\n\n\n");
        local_std_dev_start = wxDateTime::Now( );
        Tensor tmp_in_std   = get_local_std_dev(tmp_real_values_tensor.unsqueeze(0), input_volume, 10, use_dbg).squeeze(0).contiguous( ).clone( );

        Image tmp(input_volume);
        std::memcpy(tmp.real_values, tmp_in_std.data_ptr<float>( ), num_pixels * sizeof(float));
        tmp.AddFFTWPadding( );
        Tensor t_local_std_dev_mean = torch::empty({1});
        float  local_std_dev_mean   = tmp.ReturnAverageOfRealValues( );
        wxPrintf("local_std_dev Image class mean: %f\n", local_std_dev_mean);
        t_local_std_dev_mean[0] = local_std_dev_mean;
        in_std                  = tmp_in_std / t_local_std_dev_mean;

        Tensor t_mean      = torch::empty({1});
        t_mean[0]          = mean;
        Tensor t_std_dev   = torch::empty({1});
        t_std_dev[0]       = std_dev;
        real_values_tensor = (real_values_tensor - mean) / (std_dev + 1e-8);
    } catch ( std::exception& e ) {
        wxPrintf("Error when getting standard deviation and normalizing the real_values tensor: %s\n", e.what( ));
        local_std_dev_finish           = wxDateTime::Now( );
        wxTimeSpan duration_of_std_dev = local_std_dev_finish.Subtract(local_std_dev_start);
        wxPrintf("Duration of standard deviation calculation:\t\t%s\n", duration_of_std_dev.Format( ));
        return false;
    }

    // Check if the Python and C++ standard deviation tensors match (and real_values_tensor post normalization)
    if ( use_dbg ) {
        // wxPrintf("in_std shape after std dev block: [%ld, %ld, %ld]\n", in_std.size(0), in_std.size(1), in_std.size(2) /*in_std.size(3)*/);
        // write_real_values_to_text_file("in_std_after_norm_cpp.txt", box_size, false, true, &in_std);
        // write_real_values_to_text_file("real_values_tensor_after_in_std_norm_cpp.txt", box_size, false, true, &real_values_tensor);
        // compare_text_file_lines("in_std_after_norm_cpp.txt", "in_std_after_norm_py.txt", "comp_in_std_after_norm.txt");
        // compare_text_file_lines("real_values_tensor_after_in_std_norm_cpp.txt", "real_values_tensor_after_in_std_norm_py.txt", "comp_real_values_tensor_after_in_std_norm.txt");
        // compare_text_file_lines("get_local_std_dev1.txt", "get_local_std_dev2.txt", "dif_iter_local_std_dev_cmp.txt");
    }

    local_std_dev_finish           = wxDateTime::Now( );
    wxTimeSpan duration_of_std_dev = local_std_dev_finish.Subtract(local_std_dev_start);
    if ( use_dbg ) {
        // Write out the std dev
        Image tmp_in_std;
        tmp_in_std.CopyFrom(&input_volume);
        Tensor unsqueezed_in_std = in_std.clone( );
        std::memcpy(tmp_in_std.real_values, in_std.data_ptr<float>( ), num_pixels * sizeof(float));
        tmp_in_std.AddFFTWPadding( );
        tmp_in_std.QuickAndDirtyWriteSlices("original_std_dev.mrc", 1, box_size, true);
    }
    wxPrintf("Duration of calculating localized standard deviation:        %s\n\n", duration_of_std_dev.Format( ));

    wxPrintf("Applying mask...\n");
    Tensor mask_tensor;
    Image  direct_input_mask, in_std_img;
    try {
        direct_input_mask.Allocate(&input_volume);
        const float original_pixel_size = 4.00f, outer_mask_radius = 90.0f;
        direct_input_mask.AddFFTWPadding( );
        int mask_edge_width = int(10.0f / model_voxel_size);
        wxPrintf("mask_edge_width = %i\n", mask_edge_width);
        constexpr float particle_diameter = 180.0f;
        float           radius            = particle_diameter / (2 * model_voxel_size) + mask_edge_width / 2.0f;
        radius                            = std::min(radius, ((new_box_size - mask_edge_width) / 2.0f) + 1);

        direct_input_mask.SetToConstant(1);
        direct_input_mask.CosineMask(radius, mask_edge_width, false, true);

        // Check cosine mask
        if ( use_dbg ) {
            direct_input_mask.QuickAndDirtyWriteSlices("cosine_mask_check.mrc", 1, new_box_size);
        }

        direct_input_mask.RemoveFFTWPadding( );
        mask_tensor = torch::from_blob(direct_input_mask.real_values, {box_size, box_size, box_size}, torch::kFloat32);
        mask_tensor = mask_tensor.permute({2, 1, 0}).contiguous( );
        // mask_tensor         = mask_tensor.reshape({new_box_size, new_box_size, new_box_size});

        // Check if tensor masks work with a single thread
        if ( use_dbg ) {
            Tensor tmp_reals  = real_values_tensor.clone( );
            Tensor tmp_in_std = in_std.clone( );

            tmp_reals *= mask_tensor;
            tmp_in_std *= mask_tensor;

            // wxPrintf("Writing masked tensors...\n");
            // write_real_values_to_text_file("cpp_masked_volume.txt", box_size, false, true, &tmp_reals);
            // write_real_values_to_text_file("cpp_masked_in_std.txt", box_size, false, true, &tmp_in_std);
            // wxPrintf("Comparing masked tensors...\n");
            // compare_text_file_lines("cpp_masked_volume.txt", "python_masked_volume_output.txt", "comp_masked_volume.txt");
            // compare_text_file_lines("cpp_masked_in_std.txt", "python_masked_in_std_output.txt", "comp_masked_in_std.txt");

            // Switch from z-fastest back to x-fastest
            tmp_reals  = tmp_reals.permute({2, 1, 0}).contiguous( );
            tmp_in_std = tmp_in_std.permute({2, 1, 0}).contiguous( );

            // Now copy to cisTEM for writing out
            Image tmp_masked_vol(input_volume);
            Image tmp_masked_in_std(input_volume);

            std::memcpy(tmp_masked_vol.real_values, tmp_reals.data_ptr<float>( ), num_pixels * sizeof(float));
            std::memcpy(tmp_masked_in_std.real_values, tmp_in_std.data_ptr<float>( ), num_pixels * sizeof(float));
            tmp_masked_vol.AddFFTWPadding( );
            tmp_masked_in_std.AddFFTWPadding( );
            tmp_masked_vol.QuickAndDirtyWriteSlices("tensor_masked_volume.mrc", 1, box_size, true);
            tmp_masked_in_std.QuickAndDirtyWriteSlices("tensor_masked_in_std.mrc", 1, box_size, true);
        }

        if ( use_dbg ) {
            std::memcpy(direct_input_mask.real_values, mask_tensor.data_ptr<float>( ), num_pixels * sizeof(float));
            direct_input_mask.AddFFTWPadding( );
            direct_input_mask.QuickAndDirtyWriteSlices("cli_binary_mask.mrc", 1, box_size, true);
            direct_input_mask.RemoveFFTWPadding( ); // To keep consistent with the non-debug state of the Image mask

            // Check means:
            wxPrintf("Mask info:\nTensor mean = %f\ncisTEM mean = %f\n", mask_tensor.mean( ).item<float>( ), direct_input_mask.ReturnAverageOfRealValues( ));
            wxPrintf("mask_tensor dims: [%ld, %ld, %ld]\n", mask_tensor.size(0), mask_tensor.size(1), mask_tensor.size(2));

            // Also check in_std and real_values tensor
            Image tmp_real;
            Image tmp_std;
            tmp_real.Allocate(&input_volume);
            tmp_std.Allocate(&input_volume);

            std::memcpy(tmp_real.real_values, real_values_tensor.data_ptr<float>( ), num_pixels * sizeof(float));
            std::memcpy(tmp_std.real_values, in_std.data_ptr<float>( ), num_pixels * sizeof(float));
            tmp_real.AddFFTWPadding( );
            tmp_std.AddFFTWPadding( );

            tmp_real.QuickAndDirtyWriteSlices("cli_real_vals_before_mask_check.mrc", 1, new_box_size, true);
            tmp_std.QuickAndDirtyWriteSlices("cli_in_std_before_mask_check.mrc", 1, new_box_size, true);
        }

        // direct_input_mask.RemoveFFTWPadding( ); // DON'T remove FFTW padding again; input_volume already had it removed
        wxPrintf("Shape of real_values_tensor: [%ld, %ld, %ld]; and in_std: [%ld, %ld, %ld]\n",
                 real_values_tensor.size(0), real_values_tensor.size(1), real_values_tensor.size(2),
                 in_std.size(0), in_std.size(1), in_std.size(2) /*in_std.size(3)*/);

        // By default, NO PADDING PRESENT

        // Currently, img_in_std is NOT padded, and neither is input_volume; so process the pixels as though this is the case (no padding_jump_value)
        // Apply the binary mask to the volume and localized standard deviation
        // FIXME: This is wonky; I should just directly multiply into the tensor, and use_dbg should memcpy to images
        {
            in_std_img.Allocate(&input_volume);
            std::memcpy(input_volume.real_values, real_values_tensor.data_ptr<float>( ), num_pixels * sizeof(float));
            std::memcpy(in_std_img.real_values, in_std.data_ptr<float>( ), num_pixels * sizeof(float));
            input_volume.AddFFTWPadding( );
            in_std_img.AddFFTWPadding( );
            direct_input_mask.AddFFTWPadding( );
            input_volume.MultiplyPixelWise(direct_input_mask);
            in_std_img.MultiplyPixelWise(direct_input_mask);
            direct_input_mask.RemoveFFTWPadding( );

            if ( use_dbg ) {
                input_volume.QuickAndDirtyWriteSlices("cistem_mask_multiply_real_val.mrc", 1, box_size, true);
                in_std_img.QuickAndDirtyWriteSlices("cistem_mask_multiply_std_dev.mrc", 1, box_size, true);
            }

            input_volume.RemoveFFTWPadding( );
            in_std_img.RemoveFFTWPadding( );
            std::memcpy(real_values_tensor.data_ptr<float>( ), input_volume.real_values, num_pixels * sizeof(float));
            std::memcpy(in_std.data_ptr<float>( ), in_std_img.real_values, num_pixels * sizeof(float));
        }
    } catch ( std::exception& e ) {
        wxPrintf("Error generating/applying mask. %s\n", e.what( ));
        return false;
    }

    // torch::jit:IValue is type-erased
    // First add dim:
    real_values_tensor = real_values_tensor.unsqueeze(0);
    in_std             = in_std.unsqueeze(0);

    (real_values_tensor.is_contiguous( )) ? wxPrintf("real_values_tensor is contiguous\n") : wxPrintf("real_values_tensor is NOT contiguous\n");
    (in_std.is_contiguous( )) ? wxPrintf("in_std is contiguous\n") : wxPrintf("in_std is NOT contiguous\n");
    // Compare against the Python real_values right before passing to the function
    // NOTE: there should be some differences because of the mask
    // if ( use_dbg ) {
    // Tensor tmp_real = real_values_tensor.squeeze(0);
    // Tensor tmp_std  = in_std.squeeze(0);
    // write_real_values_to_text_file("cpp_real_vals_before_model.txt", box_size, false, true, &tmp_real);
    // write_real_values_to_text_file("cpp_local_std_dev_before_model.txt", box_size, false, true, &tmp_std);
    // abort( );
    //     compare_text_file_lines("cpp_real_vals_before_model.txt", "python_masked_volume_output.txt", "real_vals_before_model_comp.txt");
    //     compare_text_file_lines("cpp_local_std_dev_before_model.txt", "python_masked_in_std_output.txt", "local_std_before_model_comp.txt");
    // }

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

    // try {
    //     // FIXME: this works for now for testing purposes; but, this would need to be included alongside the project whenever blush was needed
    //     // model = torch::jit::load("/home/tim/VS_Projects/cisTEM/src/programs/blush_refinement/blush_model.pt");
    //     wxPrintf("Loading model...\n");
    //     model = torch::jit::load(model_filename);
    // } catch ( std::exception& e ) {
    //     wxPrintf("Failed to load blush model: %s\n", e.what( ));
    //     return false;
    // }

    Tensor output_vol_tensor, output_mask_tensor; // output,
    direct_input_mask.AddFFTWPadding( );

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Try passing the entire volume and mask to the model instead of blocking for now; then do the same in Python, so the two can be compared to see if the model is working as anticipated.
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // auto full_volume_passed_output    = model.forward(real_values_tensor, in_std);
    // auto output_of_full_volume_passed = std::get<0>(full_volume_passed_output);
    // output_of_full_volume_passed      = output_of_full_volume_passed.squeeze(0);
    // write_real_values_to_text_file("full_volume_output_cpp.txt", box_size, false, true, &output_of_full_volume_passed);
    // // Compare the volume with the Python values
    // compare_text_file_lines("full_volume_output_cpp.txt", "full_volume_output_py.txt", "comp_full_volume_output.txt");
    // {
    //     Image full_volume_output;
    //     full_volume_output.Allocate(&input_volume);
    //     output_of_full_volume_passed = output_of_full_volume_passed.permute({2, 1, 0}).contiguous( ); // switch back to x-fastest
    //     std::memcpy(full_volume_output.real_values, output_of_full_volume_passed.data_ptr<float>( ), num_pixels * sizeof(float));
    //     full_volume_output.AddFFTWPadding( );
    //     full_volume_output.QuickAndDirtyWriteSlices("full_volume_output_cpp.mrc", 1, box_size, true);
    // }
    // return true;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //     std::ifstream py_vals_txt("full_volume_output_py.txt");
    //     std::string   s;
    //     at::Tensor    python_vals = torch::zeros({num_pixels}, torch::kFloat32);
    //     wxPrintf("Loading Py values...\n");
    // #pragma omp parallel for num_threads(max_threads)
    //     for ( int k = 0; k < box_size; k++ ) {
    //         std::getline(py_vals_txt, s);
    //         std::istringstream in_val(s);
    //         float              val;
    //         in_val >> val;
    //         python_vals[k] = val;
    //     }

    //     python_vals = python_vals.reshape({box_size, box_size, box_size});

    //     std::ofstream comp("comp_full_volume_output.txt", std::ofstream::trunc | std::ofstream::out);
    //     comp.close( );
    //     comp.open("comp_full_volume_output.txt", std::ios::app);
    //     // #pragma omp parallel for num_threads(max_threads) collapse(3)
    //     wxPrintf("Comparing real values...\n");
    //     try {
    //         constexpr float eps              = 1e-2;
    //         torch::Tensor   diff             = (output_of_full_volume_passed - python_vals).abs( );
    //         torch::Tensor   within_tolerance = diff.le(eps);

    //         for ( int x = 0; x < box_size; x++ ) {
    //             for ( int y = 0; y < box_size; y++ ) {
    //                 for ( int z = 0; z < box_size; z++ ) {
    //                     if ( within_tolerance[z][y][x].item<bool>( ) )
    //                         comp << "SAME" << std::endl;
    //                     else
    //                         comp << "DIFFERENT" << std::endl;
    //                 }
    //             }
    //         }
    //         comp.close( );
    //     } catch ( std::exception& e ) {
    //         wxPrintf("%s\n", e.what( ));
    //     }

    // wxPrintf("Comparing full volume outputs....\n");
    // compare_text_file_lines("full_volume_output_cpp.txt", "full_volume_output_py.txt", "comp_full_volume_output.txt");
    // {
    //     output_of_full_volume_passed = torch::swapaxes(output_of_full_volume_passed, 0, 2); // switch back to x-fastest before copying back to cisTEM volume
    //     Image output_volume;
    //     output_volume.Allocate(new_box_size, new_box_size, new_box_size);
    //     if ( use_dbg )
    //         wxPrintf("About to copy output tensor to real_values in output volume...\n");
    //     // Perhaps this memcpy is going wrong
    //     std::memcpy(output_volume.real_values, output_of_full_volume_passed.data_ptr<float>( ), num_pixels * sizeof(float));
    //     output_volume.AddFFTWPadding( );

    //     if ( scale_factor != 1 ) {
    //         output_volume.ForwardFFT( );
    //         output_volume.Resize(input_volume.logical_x_dimension / scale_factor, input_volume.logical_x_dimension / scale_factor, input_volume.logical_x_dimension / scale_factor);
    //         output_volume.Resize(input_mrc.ReturnXSize( ), input_mrc.ReturnXSize( ), input_mrc.ReturnXSize( ));
    //         output_volume.BackwardFFT( );
    //     }

    //     output_volume.QuickAndDirtyWriteSlices(output_mrc_filename, 1, output_volume.logical_x_dimension, true);

    //     wxPrintf("Blush complete.\n");
    //     overall_finish      = wxDateTime::Now( );
    //     wxTimeSpan duration = overall_finish.Subtract(overall_start);
    //     wxPrintf("Total blush runtime:         %s\n", duration.Format( ));
    // }
    // return true;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    try {
        wxPrintf("Running blush model...\n");
        blush_start = wxDateTime::Now( );

        BlockIterator it({new_box_size, new_box_size, new_box_size}, model_block_size, strides);
        int           bi = 0;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // DEBUG:
        std::ofstream out_vals("cpp_means.txt", std::ofstream::trunc | std::ofstream::out);
        out_vals.close( );
        out_vals.open("cpp_tensor_means.txt", std::ios::app);
        std::ofstream cistem_mean_vals("cistem_mean_vals.txt", std::ofstream::trunc | std::ofstream::out);
        cistem_mean_vals.close( );
        cistem_mean_vals.open("cistem_mean_vals.txt", std::ios::app);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // #pragma omp parallel for num_threads(max_threads) default(shared)
        for ( auto it_coords : it ) {
            Tensor real_val_block = torch::zeros({1, model_block_size, model_block_size, model_block_size});
            Tensor std_dev_block  = torch::zeros({1, model_block_size, model_block_size, model_block_size});

            int x = std::get<0>(it_coords);
            int y = std::get<1>(it_coords);
            int z = std::get<2>(it_coords);
            // wxPrintf("x == %i, y == %i, z == %i\n", x, y, z);

            // -1 is the stop value returned by the iterator when all blocks are done
            // If we have -1, then we should just be done.
            if ( x > -1 ) {
                // float manual_mean;
                float cur_array_sum = 0.0f;
                // for ( int k = z; k < z + model_block_size; k++ ) {
                //     for ( int j = y; j < y + model_block_size; j++ ) {
                //         for ( int i = x; i < x + model_block_size; i++ ) {
                //             float cur_array_val = direct_input_mask.ReturnRealPixelFromPhysicalCoord(i, j, k);
                //             cur_array_sum += cur_array_val;
                //             // if ( use_dbg ) {
                //             //     ofarray_slice << cur_array_val << std::endl;
                //             // }
                //         }
                //     }
                // }
                auto  current_slice = mask_tensor.slice(0, z, z + model_block_size, 1).slice(1, y, y + model_block_size, 1).slice(2, x, x + model_block_size, 1);
                float mask_mean     = current_slice.mean( ).item<float>( );
                // manual_mean         = cur_array_sum / std::pow(model_block_size, 3);

                // DEBUG:
                if ( out_vals.is_open( ) )
                    out_vals << mask_mean << std::endl;
                // if ( cistem_mean_vals.is_open( ) )
                //     cistem_mean_vals << manual_mean << std::endl;

                // wxPrintf("mask tensor mean: %f\n", mask_mean);
                // wxPrintf("manual mask mean: %f\n", manual_mean);
                if ( mask_mean < 0.3 ) {
                    // if ( x == 0 && y == 0 && z == 0 ) {
                    //     std::ofstream ofs;
                    //     ofs.open("cpp_throw_out.txt", std::ofstream::out | std::ofstream::trunc);
                    //     ofs.close( );
                    // }
                    // std::ofstream ofile("cpp_throw_out.txt", std::ios::app);
                    // if ( ofile.is_open( ) ) {
                    //     ofile << "THROW" << std::endl;
                    //     ofile.close( );
                    // }
                    continue;
                }
                // else {
                //     // std::ofstream ofile("cpp_throw_out.txt", std::ios::app);
                //     // if ( ofile.is_open( ) ) {
                //     //     ofile << "KEEP" << std::endl;
                //     //     ofile.close( );
                //     // }
                // }

                // tensors have inherent zyx; so x coord must start from k, and z from i (y unaffected)
                // for ( int k = 0; k < model_block_size; k++ ) {
                //     for ( int j = 0; j < model_block_size; j++ ) {
                //         for ( int i = 0; i < model_block_size; i++ ) {
                //             real_val_block.index({0, i, j, k}) = input_volume.ReturnRealPixelFromPhysicalCoord(k + x, j + y, i + z);
                //             std_dev_block.index({0, i, j, k})  = in_std_img.ReturnRealPixelFromPhysicalCoord(k + x, j + y, i + z);
                //             // real_val_block.index({0, i, j, k}) = real_values_tensor.index({0, i + x, j + y, k + z}).item<float>( );
                //             // std_dev_block.index({0, i, j, k})  = in_std.index({0, i + x, j + y, k + z}).item<float>( );
                //         }
                //     }
                // }
                real_val_block = real_values_tensor.slice(1, z, z + model_block_size, 1).slice(2, y, y + model_block_size, 1).slice(3, x, x + model_block_size, 1);
                std_dev_block  = real_values_tensor.slice(1, z, z + model_block_size, 1).slice(2, y, y + model_block_size, 1).slice(3, x, x + model_block_size, 1);
                bi++;
            }

            // NOTE: this will always be done right after the initial loading, so we don't need to worry about bi really
            if ( bi == batch_size ) {
                auto   init_output = model.forward(real_val_block, std_dev_block);
                Tensor output      = std::get<0>(init_output);
                output             = output.detach( );

                // wxPrintf("Output shape: [%ld, %ld, %ld, %ld]\n", output.size(0), output.size(1), output.size(2), output.size(3));

                for ( int i = 0; i < batch_size; i++ ) {
                    // Update inference grid
                    // NOTE: slicing this way makes a "view" of the tensor, which means that the original tensor is modified as well;
                    // thus the final addition to the infer_grid and count_grid tensors will be reflected in the original tensors
                    Tensor infer_grid_slice = infer_grid.slice(0, z, z + model_block_size, 1).slice(1, y, y + model_block_size, 1).slice(2, x, x + model_block_size, 1);
                    auto   update           = output[i] * weights_tensor;
                    infer_grid_slice += update;

                    // Update count grid
                    Tensor count_grid_slice = count_grid.slice(0, z, z + model_block_size, 1).slice(1, y, y + model_block_size, 1).slice(2, x, x + model_block_size, 1);
                    count_grid_slice += weights_tensor;
                }
                // std::vector<float> output_vol_vec(std::pow(model_block_size, 3));
                // std::memcpy(output_vol_vec.data( ), output.data_ptr<float>( ), std::pow(model_block_size, 3));

                // #pragma omp parallel for default(shared) num_threads(max_threads)
                // for ( int pixel_counter = 0; pixel_counter < num_subset_pixels; pixel_counter++ ) {
                //     int Z = pixel_counter / (model_block_size * model_block_size);
                //     int Y = (pixel_counter / model_block_size) % model_block_size;
                //     int X = pixel_counter % model_block_size;

                //     int x_coord = X + x;
                //     int y_coord = Y + y;
                //     int z_coord = Z + z;

                //     // Finds the corresponding subset of the data in the full size volumes
                //     int index = x_coord + X * (y_coord + Y * z_coord);

                //     if ( isnan(output_vol_vec[pixel_counter]) )
                //         output_vol_vec[pixel_counter] = 0;

                //     infer_grid[index] += output_vol_vec[pixel_counter] * weights_grid[index];
                //     // if ( output_vol_vec[pixel_counter] != 0.0f )
                //     // wxPrintf("output_vol_vect[%i] == %f\n", pixel_counter, output_vol_vec[pixel_counter]);
                //     // FIXME: this may be causing problems; mask_grid being a 2D means it will not reach the same index as infer_grid
                //     // Should mask be 3D? Check the single_blush python code to see if it should
                //     // mask_grid[index] = output_vol_vec[pixel_counter] * weights_grid[index];
                //     count_grid[index] += weights_grid[index];
                // }
                bi = 0;
            }
        }

        infer_grid = torch::where(count_grid > 0, infer_grid / count_grid, infer_grid);
        infer_grid = torch::where(count_grid < 1e-1, 0, infer_grid); // Set values where count_grid is less than 0.1 to 0
        infer_grid *= mask_tensor;
        // Check the values of infer grid prior to the update
        // wxPrintf("Printing pre_norm values...\n");

        //         constexpr float c_assume_zero_threshold = 0.1f;

        // #pragma omp parallel for default(shared) num_threads(max_threads)
        //         for ( long pixel_counter = 0; pixel_counter < num_pixels; pixel_counter++ ) {
        //             if ( isnan(infer_grid[pixel_counter]) )
        //                 infer_grid[pixel_counter] = 0;

        //             if ( count_grid[pixel_counter] > 0 ) {
        //                 infer_grid[pixel_counter] /= count_grid[pixel_counter];
        //                 // mask_grid[pixel_counter] /= count_grid[pixel_counter];
        //             }

        //             if ( infer_grid[pixel_counter] < c_assume_zero_threshold )
        //                 infer_grid[pixel_counter] = 0;
        //             // if ( mask_grid[pixel_counter] < c_assume_zero_threshold )
        //             // mask_grid[pixel_counter] = 0;

        //             // Mask out the final inference, and normalize?
        //             infer_grid[pixel_counter] *= direct_input_mask.real_values[pixel_counter];
        //             // mask_grid[pixel_counter] *= direct_input_mask.real_values[pixel_counter];
        //             // wxPrintf("infer_grid[%li] == %f\n", pixel_counter, infer_grid[pixel_counter]);
        //             // FIXME: Perhaps I can get rid of this normalization step, opting instead to call Image::Normalize after copying to Image object
        //             infer_grid[pixel_counter] = infer_grid[pixel_counter] * (std_dev * 1e-8) + mean;
        //         }
        blush_finish              = wxDateTime::Now( );
        wxTimeSpan blush_duration = blush_finish.Subtract(blush_start);
        out_vals.close( );
        cistem_mean_vals.close( );
        wxPrintf("Finished running blush model. Total duration:        %s\n", blush_duration.Format( ));

    }

    catch ( std::exception& e ) {
        wxPrintf("Couldn't run model; exception occurred: %s\n", e.what( ));
        return false;
    }

    // if ( use_dbg ) {
    //     // Check count_grid values
    //     wxPrintf("Writing out count_grid and infer_grid vals...\n");
    //     std::ofstream ofs1("count_grid_vals.txt", std::ofstream::trunc | std::ofstream::out);
    //     std::ofstream ofs2("infer_grid_vals.txt", std::ofstream::trunc | std::ofstream::out);
    //     ofs1.close( );
    //     ofs2.close( );
    //     ofs1.open("count_grid_vals.txt", std::ios::app);
    //     ofs2.open("infer_grid_vals.txt", std::ios::app);
    //     if ( ofs1.is_open( ) && ofs2.is_open( ) ) {
    //         for ( int i = 0; i < num_pixels; i++ ) {
    //             ofs1 << count_grid[i] << std::endl;
    //             ofs2 << infer_grid[i] << std::endl;
    //         }
    //         ofs1.close( );
    //         ofs2.close( );
    //     }
    //     else {
    //         wxPrintf("Failed to open one of the text files.\n");
    //     }
    //     wxPrintf("Finished writing out count_grid and infer_grid vals.\n");
    // }

    // NOTE: Here relion would normally apply additional filtering before writing out the actual volume.
    // However, cisTEM does its own filtering and this should not necessarily be needed. If something goes wrong,
    // add that filtering here.

    // TODO: Update subsequent with post-post-processing (the stuff that comes after apply model)
    // Some is the same like resampling back to original size; but crossover grid (and ), applying max res are new
    wxPrintf("Finishing post-processing...\n");

    // Split the model's output volume and output mask so they can be used appropriately
    // try {
    // if ( use_dbg ) {
    // wxPrintf("output_vol_tensor == [%ld, %ld, %ld, %ld]\n", output_vol_tensor.size(0), output_vol_tensor.size(1), output_vol_tensor.size(2), output_vol_tensor.size(3));
    // wxPrintf("output_mask_tensor == [%ld, %ld, %ld]\n", output_mask_tensor.size(0), output_mask_tensor.size(1), output_mask_tensor.size(2));
    // }
    //
    // } catch ( std::exception& e ) {
    // wxPrintf("Failed to break apart the model output. Exception occurred: %s\n", e.what( ));
    // return false;
    // }

    // Let's verify the dimensions

    // output_vol_tensor = output_vol_tensor.squeeze(0);
    // output_vol_tensor = output_vol_tensor.contiguous( );

    Image output_volume;
    output_volume.Allocate(new_box_size, new_box_size, new_box_size);
    if ( use_dbg )
        wxPrintf("About to copy output tensor to real_values in output volume...\n");
    // Perhaps this memcpy is going wrong
    wxPrintf("Permuting infer_grid to x-fastest...\n");
    infer_grid = infer_grid.permute({2, 1, 0}).contiguous( ); // switch back to x-fastest
    std::memcpy(output_volume.real_values, infer_grid.data_ptr<float>( ), num_pixels * sizeof(float));
    output_volume.AddFFTWPadding( );

    if ( scale_factor != 1 ) {
        output_volume.ForwardFFT( );
        output_volume.Resize(input_volume.logical_x_dimension / scale_factor, input_volume.logical_x_dimension / scale_factor, input_volume.logical_x_dimension / scale_factor);
        output_volume.Resize(input_mrc.ReturnXSize( ), input_mrc.ReturnXSize( ), input_mrc.ReturnXSize( ));
        output_volume.BackwardFFT( );
    }

    output_volume.QuickAndDirtyWriteSlices(output_mrc_filename, 1, output_volume.logical_x_dimension, true);

    wxPrintf("Comparing means from Python and C++...\n");
    // compare_text_file_lines("cpp_tensor_means.txt", "py_tensor_means.txt", "comp_tensor_means.txt");
    // compare_text_file_lines("cistem_mean_vals.txt", "py_tensor_means.txt", "comp_py_tensor_means_w_cistem_means.txt");
    wxPrintf("Blush complete.\n");
    overall_finish      = wxDateTime::Now( );
    wxTimeSpan duration = overall_finish.Subtract(overall_start);
    wxPrintf("Total blush runtime:         %s\n", duration.Format( ));

    return true;
}

/**
 * @brief Returns a torch tensor of a localized standard deviation.
 * 
 * @param grid Volume being passed to blush.
 * @param orig Original image; for debugging purposes.
 * @param size Desired kernel size.
 * @param use_dbg Whether debug prints/writes should be used.
 * @return Tensor 
 */
Tensor get_local_std_dev(torch::Tensor grid, Image& orig, int size, const bool use_dbg) {
    // Unsqueeze and clone the grid tensor
    if ( use_dbg )
        wxPrintf("Dimensions of grid: %ld, %ld, %ld, %ld\n", grid.size(0), grid.size(1), grid.size(2), grid.size(3));

    Tensor new_grid = grid.unsqueeze(1).clone( );
    Tensor grid2    = new_grid.square( );

    if ( use_dbg )
        wxPrintf("Dimensions of new_grid (after unsqueezing grid): %ld, %ld, %ld, %ld, %ld\n", new_grid.size(0), new_grid.size(1), new_grid.size(2), new_grid.size(3), new_grid.size(4));

    // Create the kernel
    Tensor ls     = torch::linspace(-1.5, 1.5, 2 * size + 1); // Create tensor of size 2 * size + 1 with evenly spaced values between -1.5 and 1.5 -- size passed as 10
    Tensor kernel = torch::exp(-ls.square( )).to(new_grid.device( )); // Make normal dist (gets rid of negatives (square), applies exponential function(exp))
    kernel /= kernel.sum( );

    kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4); // Shape: (1, 1, kernel_size, 1, 1)

    if ( use_dbg )
        wxPrintf("kernel shape: [%ld, %ld, %ld, %ld, %ld]\n", kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3), kernel.size(4));

    if ( use_dbg ) {
        // Check the new_grid after the unsqueezing to see if that's causing anything
        Tensor tmp = new_grid.clone( );
        tmp        = tmp.squeeze(0).squeeze(0);

        Image tmp_vol;
        tmp_vol.CopyFrom(&orig);
        std::memcpy(tmp_vol.real_values, tmp.data_ptr<float>( ), std::pow(tmp_vol.logical_x_dimension, 3));
        tmp_vol.AddFFTWPadding( );
        tmp_vol.QuickAndDirtyWriteSlices("cli_before_conv.mrc", 1, tmp_vol.logical_z_dimension);
    }

    // Set options, then perform 3D convolution
    torch::nn::functional::Conv3dFuncOptions options;
    // This should equally pad the directions of the convolution dimension by 10 (i.e. 10 in either direction)

    // NOTE: setting the options like below, and creating pre-padded tensors are both equally viable ways to prepare the
    // tensor for dimension size changes; they give the same result. I will stick with the full padding for clarity's sake
    // options.stride(1).padding({size, 0, 0}); // This is the only way the dimensions are maintained during convolution

    // Pads tensors by size in either direction of x, y, and z dimensions (if size == 10, then padding totals 20)
    new_grid = torch::constant_pad_nd(new_grid, {size, size, size, size, size, size});
    grid2    = torch::constant_pad_nd(grid2, {size, size, size, size, size, size});

    // This convolution will remove the padded dimensions
    for ( int i = 0; i < 3; ++i ) {
        // if ( use_dbg ) {
        // wxPrintf("\nNew convolution started\n");
        // wxPrintf("Permuting new_grid\n");
        // }

        new_grid = new_grid.permute({0, 1, 4, 2, 3}); // Shift (N, C, D, W, H) then (N, C, H, D, W) and finally (N, C, W, H, D)
        // new_grid = new_grid.contiguous( );
        // if ( use_dbg ) {
        //     wxPrintf("new_grid shape: [%ld, %ld, %ld, %ld, %ld]\n", new_grid.size(0), new_grid.size(1), new_grid.size(2), new_grid.size(3), new_grid.size(4));
        //     wxPrintf("Performing 3D convolution on new_grid\n");
        // }

        new_grid = torch::nn::functional::conv3d(new_grid, kernel, options);

        // Check dimensions after convolution
        // if ( use_dbg ) {
        //     wxPrintf("new_grid dims after convolution: [%ld, %ld, %ld, %ld, %ld]\n", new_grid.size(0), new_grid.size(1), new_grid.size(2), new_grid.size(3), new_grid.size(4));
        // }

        // if ( use_dbg )
        //     wxPrintf("Permuting grid2\n");

        grid2 = grid2.permute({0, 1, 4, 2, 3}); // Same permutation
        // grid2 = grid2.contiguous( );
        // if ( use_dbg ) {
        //     wxPrintf("grid2 shape: [%ld, %ld, %ld, %ld, %ld]\n", grid2.size(0), grid2.size(1), grid2.size(2), grid2.size(3), grid2.size(4));
        //     wxPrintf("Performing 3D convolution on grid2\n");
        // }

        grid2 = torch::nn::functional::conv3d(grid2, /*weight=*/kernel, options);
        // Check dimensions after convolution
        // if ( use_dbg )
        //     wxPrintf("grid2 dims after convolution: [%ld, %ld, %ld, %ld, %ld]\n", grid2.size(0), grid2.size(1), grid2.size(2), grid2.size(3), grid2.size(4));
    }

    // Check dimensions of the narrowed new_grid
    // if ( use_dbg ) {
    //     auto sizes = new_grid.sizes( );
    //     wxPrintf("dimensions of narrowed new_grid: ");
    //     for ( int i = 0; i < sizes.size( ); i++ ) {
    //         if ( ! (i == sizes.size( ) - 1) )
    //             wxPrintf("%ld ", sizes[i]);
    //         else {
    //             wxPrintf("%ld\n", sizes[i]);
    //         }
    //     }
    // }
    if ( use_dbg ) {
        // Check the new_grid after the unsqueezing to see if that's causing anything
        Tensor tmp = new_grid.clone( );
        tmp        = tmp.squeeze(0).squeeze(0);
        Image tmp_vol;
        tmp_vol.CopyFrom(&orig);
        std::memcpy(tmp_vol.real_values, tmp.data_ptr<float>( ), std::pow(tmp_vol.logical_x_dimension, 3) * sizeof(float));
        tmp_vol.AddFFTWPadding( );
        tmp_vol.QuickAndDirtyWriteSlices("cli_directly_after_conv_local_std_dev.mrc", 1, tmp_vol.logical_z_dimension, true);
    }

    Tensor in_std = torch::sqrt(torch::clamp(grid2 - new_grid.square( ), 0));

    if ( use_dbg ) {
        // Check the new_grid after the unsqueezing to see if that's causing anything
        Tensor tmp = in_std.clone( );
        tmp        = tmp.squeeze(0).squeeze(0);
        Image tmp_vol;
        tmp_vol.CopyFrom(&orig);
        std::memcpy(tmp_vol.real_values, tmp.data_ptr<float>( ), std::pow(tmp_vol.logical_x_dimension, 3) * sizeof(float));
        tmp_vol.AddFFTWPadding( );
        tmp_vol.QuickAndDirtyWriteSlices("cli_after_squaring_grid.mrc", 1, tmp_vol.logical_z_dimension, true);

        // wxPrintf("\nAfter get_local_std_dev (pre squeeze): %ld, %ld, %ld, %ld, %ld\n", in_std.size(0), in_std.size(1), in_std.size(2), in_std.size(3), in_std.size(4));
    }

    in_std = in_std.squeeze(0);

    // Check dimensions; also write text file to check the values against the Python values
    // if ( use_dbg ) {
    //     wxPrintf("After std after squeeze: %ld, %ld, %ld, %ld\n\n", in_std.size(0), in_std.size(1), in_std.size(2), in_std.size(3));
    // std::ofstream ofs("cpp_pre_normalized_std_dev.txt", std::ofstream::trunc | std::ofstream::out);
    // ofs.close( );
    // ofs.open("cpp_pre_normalized_std_dev.txt", std::ios::app);
    // if ( ofs.is_open( ) ) {
    //     for ( int k = 0; k < in_std.size(3); k++ ) {
    //         for ( int j = 0; j < in_std.size(2); j++ ) {
    //             for ( int i = 0; i < in_std.size(1); i++ ) {
    //                 ofs << in_std[0][i][j][k].item<float>( ) << std::endl;
    //             }
    //         }
    //     }
    //     ofs.close( );
    //     abort( );
    // }
    // else {
    //     wxPrintf("Couldn't open cpp_pre_normalized_std_dev.txt\n");
    //     abort( );
    // }
    // }

    return in_std.clone( );
}

/**
 * @brief Generate a 3D grid of weights where each dimension is of size block_size.
 * 
 * @param block_size The size of segments passed to the model -- comes from model's previously set model_block_size.
 * @param margin The thickness of the border around the distribution generated by 
 * @return Tensor 
 */
Tensor make_weight_box(const int& block_size, int margin) {
    margin = (margin - 1 > 0) ? margin - 1 : 0;
    int s  = block_size - (margin * 2);

    wxPrintf("s == %i\n", s);

    // Create evenly spaced 1D tensors; should range from -27 to 27, with increments of...55?
    torch::Tensor x = torch::linspace(-s / 2, s / 2, s);
    torch::Tensor y = torch::linspace(-s / 2, s / 2, s);
    torch::Tensor z = torch::linspace(-s / 2, s / 2, s);

    // Broadcasting means these tensors can be subsequently expanded to fit the appropriate dimensions as needed
    // Each of these tensors is different, and later they are sliced together dimensionally
    // expand broadcasts the dimension with s, so the values of the linspace array are applied in all 3 dimensions
    // This is meant to replicate numpy.meshgrid functionality
    Tensor xx = x.view({s, 1, 1}).expand({s, s, s});
    Tensor yy = y.view({1, s, 1}).expand({s, s, s});
    Tensor zz = z.view({1, 1, s}).expand({s, s, s});
    // {
    //     std::ofstream ofile("xx_values.txt");
    //     if ( ofile.is_open( ) ) {
    //         for ( int a = 0; a < xx.size(2); a++ ) {
    //             for ( int b = 0; b < xx.size(1); b++ ) {
    //                 for ( int c = 0; c < xx.size(0); c++ ) {
    //                     ofile << xx[c][b][a].item<float>( ) << std::endl;
    //                 }
    //             }
    //         }
    //         ofile.close( );
    //     }
    //     else {
    //         wxPrintf("xx_values.txt failed to open.\n");
    //     }
    // }

    // Radial distance; performs pixelwise comparison of the values at each point in all 3 tensors, selecting the one with the largest value at each pixel
    // The max function actually returns both the raw values being compared along with the indices where the max value came from (including which tensor it came from)
    //  We want the actual values before applying cosine falloff
    xx            = xx.abs( );
    yy            = yy.abs( );
    zz            = zz.abs( );
    Tensor radius = torch::zeros_like(xx);
    {
        // torch::max only accepts 2 args; break apart the comparison of the broadcasted linspace arrays
        // Life cannot be so easy; I must use nested for to complete this comparison
        // auto max1 = torch::max(xx.abs( ), yy.abs( ));
        // radius    = torch::max(max1, zz.max( ));
        for ( int k = 0; k < xx.size(2); k++ ) {
            for ( int j = 0; j < xx.size(1); j++ ) {
                for ( int i = 0; i < xx.size(0); i++ ) {
                    float max = std::max(zz[i][j][k].item<float>( ), std::max(xx[i][j][k].item<float>( ), yy[i][j][k].item<float>( )));
                    // max             = torch::max(max, zz[i][j][k].item<float>( ));
                    radius[i][j][k] = max;
                }
            }
        }
    }
    // {
    //     std::ofstream ofile("radius_values.txt");
    //     if ( ofile.is_open( ) ) {
    //         for ( int a = 0; a < radius.size(2); a++ ) {
    //             for ( int b = 0; b < radius.size(1); b++ ) {
    //                 for ( int c = 0; c < radius.size(0); c++ ) {
    //                     ofile << radius[c][b][a].item<float>( ) << std::endl;
    //                 }
    //             }
    //         }
    //         ofile.close( );
    //     }
    // }
    // Cosine transformation; creates a smooth falloff where values decrease the further from the center you go
    radius = torch::cos(radius / radius.max( ) * (M_PI / 2));

    // Sets up the output tensor that will slice from each of the tensors
    torch::Tensor weight_grid = torch::zeros({block_size, block_size, block_size});

    auto slices = weight_grid.slice(0, margin, block_size - margin).slice(1, margin, block_size - margin).slice(2, margin, block_size - margin);
    slices.copy_(radius);
    weight_grid = weight_grid.clamp_min(1e-6); // avoid zeros

    //Let 's just check what' s going on with the weights_grid...
    // {
    //     // First, clear the text file
    //     std::ofstream ofs;
    //     ofs.open("initial_weights_grid_vals.txt", std::ofstream::trunc | std::ofstream::out);
    //     ofs.close( );
    //     ofs.open("initial_weights_grid_vals.txt", std::ios::app);
    //     if ( ofs.is_open( ) ) {
    //         for ( int a = 0; a < weight_grid.size(2); a++ ) {
    //             for ( int b = 0; b < weight_grid.size(1); b++ ) {
    //                 for ( int c = 0; c < weight_grid.size(0); c++ ) {
    //                     ofs << weight_grid[c][b][a].item<float>( ) << std::endl;
    //                 }
    //             }
    //         }
    //         ofs.close( );
    //     }
    //     else {
    //         wxPrintf("initial_weights_grid_vals.txt did not open.\n\n");
    //     }
    // }

    return weight_grid;
}

// The LibTorch version of the calculation makes the kernel Gaussian, so pixels @ center are weighted more than those at edges
/**
 * @brief Generate a Guassian kernel for calculating the weighted localized standard deviation of the volume.
 * 
 * @param kernel_size The size of the kernel (passed as a literal; commonly 10).
 * @param sigma Standard deviation of the Gaussian distribution.
 * @return std::vector<float> 
 */
std::vector<float> generate_3D_gaussian_kernel(int kernel_size, const float& sigma) {
    // want kernel to be 2 * kernel_size + 1; so size passed is technically the half size
    int                half_kernel_size = (kernel_size - 1) / 2; // becomes 10
    std::vector<float> kernel(std::pow(kernel_size, 3)); // multiply by 3 for 3D
    float              sum = 0.0f;

    // Create kernel
    for ( int z = -half_kernel_size; z <= half_kernel_size; z++ ) {
        for ( int y = -half_kernel_size; y <= half_kernel_size; y++ ) {
            for ( int x = -half_kernel_size; x <= half_kernel_size; x++ ) {
                float gauss_val = std::exp(-0.5f * (std::pow(x / sigma, 2) + std::pow(y / sigma, 2) + std::pow(z / sigma, 2)));
                long  index     = (z + half_kernel_size) * kernel_size * kernel_size + (y + half_kernel_size) * kernel_size + (x + half_kernel_size);
                kernel[index]   = gauss_val;
                sum += gauss_val;
            }
        }
    }

    // TODO: validate whether this is needed or if we shouldn't normalize this fully yet...
    // Normalize the kernel so that it sums to 1
    for ( float& val : kernel ) {
        val /= sum;
    }

    return kernel;
}

// TODO: this needs to be changed to work for 3D
/**
 * @brief Calculate the weighted average of a voxel given a kernel of weights.
 * 
 * @param data The volume that is having it's weighted mean calculated.
 * @param kernel Distribution of weights that is being applied to the data.
 * @param kernel_size The dimension of the (cubic) kernel.
 * @param center_x Current x coordinate for kernel center.
 * @param center_y Current y coordinate for kernel center.
 * @param center_z Current z coordinate for kernel center.
 * @param box_size Dimension of the cubic volume.
 * @return float The calculated weighted average.
 */
float weighted_mean_3D(const std::vector<float>& data, const std::vector<float>& kernel, const int& kernel_size, const int& center_x, const int& center_y, const int& center_z, const int& box_size) {
    float weighted_sum     = 0.0f;
    float weight_sum       = 0.0f;
    int   half_kernel_size = (kernel_size - 1) / 2;

    for ( int dz = -half_kernel_size; dz <= half_kernel_size; dz++ ) {
        for ( int dy = -half_kernel_size; dy <= half_kernel_size; dy++ ) {
            for ( int dx = -half_kernel_size; dx <= half_kernel_size; dx++ ) {
                int x = center_x + dx;
                int y = center_y + dy;
                int z = center_z + dz;

                if ( x >= 0 && x < box_size && y >= 0 && y < box_size && z >= 0 && z < box_size ) {
                    long data_index   = z * box_size * box_size + y * box_size + x;
                    int  kernel_index = (dz + half_kernel_size) * kernel_size * kernel_size + (dy + half_kernel_size) * kernel_size + (dx + half_kernel_size);

                    weighted_sum += data[data_index] * kernel[kernel_index];
                    weight_sum += kernel[kernel_index];
                }
            }
        }
    }

    if ( weight_sum > 0.0f ) {
        return weighted_sum / weight_sum;
    }

    return 0.0f;
}

// Well, let's try a different localized std dev calculation function; one that doesn't rely on convolution
// We'll see how fast it goes and how accurate it seems to be -- may need to re-run Python script that produces their output
// as a validation

// First, make the function; after calling, we'll convert the values to a tensor
std::vector<float> localized_std_dev(Tensor& real_values, const long& num_pixels, const int& box_size, const int& kernel_size, const int& max_threads) {
    // Load tensor into vector
    std::vector<float> grid_vals(num_pixels);
    std::memcpy(grid_vals.data( ), real_values.data_ptr<float>( ), num_pixels * sizeof(float));

    std::vector<float> result(num_pixels);
    // const float        sigma = 1.0f; // Not sure how this should be changed accordingly to match the Python format
    // Before we can start with the mean calculation, we have to generate the kernels
    // std::vector<float> gaussian_kernel = generate_3D_gaussian_kernel(kernel_size, sigma);

    // Generate linspace Guassian using the same method
    Tensor ls     = torch::linspace(-1.5, 1.5, kernel_size); // Create tensor of size 2 * size + 1 with evenly spaced values between -1.5 and 1.5 -- size passed as 10
    Tensor kernel = torch::exp(-ls.square( )).to(real_values.device( )); // Make normal dist (gets rid of negatives (square), applies exponential function(exp))
    kernel /= kernel.sum( );

    kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4);
    kernel = kernel.expand({kernel_size, kernel_size, kernel_size, 1, 1});
    kernel = kernel.squeeze(-1).squeeze(-1);

    std::vector<float> gaussian_kernel(std::pow(kernel_size, 3));
    std::memcpy(gaussian_kernel.data( ), kernel.data_ptr<float>( ), kernel.numel( ) * sizeof(float));

    long pixel_counter = 0;
    int  half_window   = kernel_size / 2;
// TODO: parallelize this loop?
#pragma omp parallel for num_threads(max_threads) default(shared) private(pixel_counter)
    for ( pixel_counter = 0; pixel_counter < num_pixels; pixel_counter++ ) {
        int   x     = (pixel_counter % box_size);
        int   y     = ((pixel_counter / box_size) % box_size);
        int   z     = (pixel_counter / (box_size * box_size));
        int   index = 0, count = 0;
        float weighted_sum = 0.0, weight_sum = 0.0, mean = 0.0, std_dev = 0.0;

        // Gets depth, row, column (don't need to worry about padding jumps -- padding removed)
        mean = weighted_mean_3D(grid_vals, gaussian_kernel, kernel_size, x, y, z, box_size);

        // Next get std_dev
        for ( int dz = -half_window; dz <= half_window; dz++ ) {
            for ( int dy = -half_window; dy <= half_window; dy++ ) {
                for ( int dx = -half_window; dx <= half_window; dx++ ) {
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if ( nx >= 0 && nx < box_size && ny < box_size && nz >= 0 && nz < box_size ) {
                        // Gets depth, row, column
                        index = nz * box_size * box_size + ny * box_size + nx;
                        std_dev += std::pow(grid_vals[index] - mean, 2);
                        count++;
                    }
                }
            }
        }

        float result_to_add;

        if ( count > 0 )
            result_to_add = std::sqrt(std_dev / count);
        else
            result_to_add = 0.0f;

        result[pixel_counter] = result_to_add;
    }

    // wxPrintf("sqrt is: %f\n", std::sqrt(std_dev / count));

    return result;
}

/**
 * @brief Designed strictly for testing the output values of the weight box function in C++ against
 * the original pythonic form in Relion.
 * 
 * @param txt_filename Name of the file containing the weight values generated in Python.
 * @param cpp_output Tensor containing the weight values generated in C++.
 * @param weight_box_size Size of the weight box as needed by the Blush model.
 */
void test_weight_box_output(const std::string& txt_filename, const Tensor& cpp_output, const int& weight_box_size) {
    std::ifstream python_output(txt_filename);
    int           num_differences = 0;
    int           pixel_index     = 0;
    if ( ! python_output.is_open( ) ) {
        wxPrintf("Failed to open the file. Filename: %s\n", txt_filename);
        return;
    }
    else {
        wxPrintf("Opened weight_boxes file.\n");
        // Start reading:
        float cur_element;
        while ( python_output >> cur_element ) {
            // Do the comparison in place
            if ( cpp_output[pixel_index / (weight_box_size * weight_box_size)][(pixel_index / weight_box_size) % weight_box_size][pixel_index % weight_box_size].item<float>( ) - cur_element > 1e-3 ) {
                num_differences++;
            }
            pixel_index++;
        }
    }
    wxPrintf("Total number of weight box differences: %i\n", num_differences);
}

/**
 * @brief Get a particular slice of a tensor for processing on a subset of the full tensor data.
 * It is assumed the tensor will have at least 3 dimensions, and a format starting with x-dimension (x, y, z).
 * 
 * @param input Tensor to be sliced.
 * @param x Starting x dim of slice.
 * @param y Starting y dim of slice.
 * @param z Starting z dim of slice.
 * @param block_size How far from the starting dimension the slice will go.
 * @return Tensor Slice of the input tensor.
 */
Tensor slice_tensor(const Tensor input, const int& x, const int& y, const int& z, const int& block_size) {
    MyDebugAssertTrue(input.dim( ) >= 3, "Must have at least 3 dimensions to proceed.");

    // DEBUG: sanity check
    wxPrintf("Dimensions of input tensor: [%ld, %ld, %ld]\n", input.size(0), input.size(1), input.size(2));
    Tensor output_tensor = torch::zeros({block_size, block_size, block_size}, torch::kFloat32);

    // Assumes x, y, z are final 3 dimensions
    output_tensor = input.slice(input.dim( ) - 3, x, x + block_size).slice(input.dim( ) - 2, y, y + block_size).slice(input.dim( ) - 1, z, z + block_size);
    // DEBUG:
    {
        std::ofstream ofs;
        ofs.open("pre-return-slice-tensor.txt", std::ios::app);
        if ( ofs.is_open( ) ) {
            for ( int k = 0; k < block_size; k++ ) {
                for ( int j = 0; j < block_size; j++ ) {
                    for ( int i = 0; i < block_size; i++ ) {
                        ofs << output_tensor[i][j][k].item<float>( ) << std::endl;
                    }
                }
            }
            ofs.close( );
        }
    }
    // DEBUG: second sanity check:
    wxPrintf("Dimensions of output tensor: [%ld, %ld, %ld]\n", output_tensor.size(0), output_tensor.size(1), output_tensor.size(2));
    return output_tensor;
}

// FIXME: Does NOT give same output as the Python code; it is VERY off
Tensor generate_radial_mask(const int& box_size, const float& radius, const int& mask_edge_width) {
    float  bz2 = box_size / 2.0;
    Tensor ls  = torch::linspace(-bz2, bz2, box_size);
    Tensor r   = torch::stack(torch::meshgrid({ls, ls, ls}), -1);

    // This uses the sqrt of squared distances, so the raw distance; perhaps that's causing problems compared w/ CosineMask?
    r = torch::sqrt(torch::sum(torch::square(r), -1));
    // wxPrintf("About to write out...\n");
    // // wxPrintf("%li\n", r.dim( ));
    // write_real_values_to_text_file("cpp_r_writeout.txt", box_size, false, true, &r);
    // wxPrintf("About to compare...\n");
    // compare_text_file_lines("cpp_r_writeout.txt", "py_r_writeout.txt", "comp_r_writeout.txt");

    // DEBUG: write out r, compare against Python output
    Tensor scale = torch::ones_like(r);

    if ( mask_edge_width > 0 ) {
        int edge_low  = radius - mask_edge_width / 2;
        int edge_high = radius + mask_edge_width / 2;

        // torch::where is a boolean mask function
        scale = torch::where(r > edge_high, 0, scale);
        scale = torch::where((r >= edge_low) & (r <= edge_high), 0.5 + 0.5 * torch::cos(M_PI * (torch::where((r >= edge_low) & (r <= edge_high), r - edge_low, r)) / mask_edge_width), scale);
    }
    else {
        scale = torch::where(r > radius, 0, scale);
    }

    return scale;
}

std::vector<std::vector<std::vector<float>>> cubic_zeros_vector(const int& box_size) {
    return std::vector<std::vector<std::vector<float>>>(box_size, std::vector<std::vector<float>>(box_size, std::vector<float>(box_size, 0)));
}

// NOTE: this slicing method actually works; let's try going back to that implementation later
void tesnor_slicing_test( ) {
    torch::Tensor test1 = torch::ones({192, 192, 192}, torch::kFloat32);
    // Let's try taking a slice of this:
    torch::Tensor test2 = test1.index({torch::indexing::Slice(0, 64), torch::indexing::Slice(0, 64), torch::indexing::Slice(0, 64)});

    // Check the shape and the values:
    auto shape = test2.sizes( );
    for ( int i = 0; i < shape.size( ); i++ ) {
        wxPrintf("test2 shape: %li", shape[i]);
        if ( i < shape.size( ) - 1 ) {
            wxPrintf(", ");
        }
    }
    wxPrintf("\n");
    std::ofstream ofs;
    ofs.open("test2_tensor_values.txt", std::ios::app);
    if ( ofs.is_open( ) ) {
        for ( int k = 0; k < 64; k++ ) {
            for ( int j = 0; j < 64; j++ ) {
                for ( int i = 0; i < 64; i++ ) {
                    ofs << test2[i][j][k].item<float>( ) << std::endl;
                }
            }
        }
        ofs.close( );
    }
    wxPrintf("Finished\n");
}

/**
 * @brief Function for writing out volume and mask that can be used
 * to trace the PyTorch model to generate the TorchScript intermediate.
 * 
 * @param volume De-padded volume that would normally be the input.
 * @param mask De-padded mask that would normally be the input.
 */
void write_text_files(Image& volume, Image& mask) {
    std::ofstream of1;
    std::ofstream of2;
    of1.open("real_values.txt", std::ofstream::trunc | std::ofstream::out);
    of2.open("mask_values.txt", std::ofstream::trunc | std::ofstream::out);
    of1.close( );
    of2.close( );
    of1.open("real_values.txt", std::ios::app);
    of2.open("mask_values.txt", std::ios::app);
    for ( int pixel_counter = 0; pixel_counter < volume.number_of_real_space_pixels; pixel_counter++ ) {
        if ( of1.is_open( ) && of2.is_open( ) ) {
            of1 << volume.real_values[pixel_counter] << std::endl;
            of2 << mask.real_values[pixel_counter] << std::endl;
        }
        else {
            wxPrintf("Error opening the text files.\n");
            abort( );
        }
    }
    of1.close( );
    of2.close( );
    wxPrintf("Finished writing to text files.\n");
    // abort( );
}

bool test_slicing_of_std_dev_and_real_values(Tensor real_slice, Tensor std_dev_slice, Tensor full_real, Tensor full_std_dev, const int& x, const int& y, const int& z) {
    Image test_real;
    Image test_std_dev;
    test_real.Allocate(192, 192, 192);
    test_std_dev.Allocate(192, 192, 192);

    constexpr int threads           = 12;
    constexpr int model_block_size  = 64;
    constexpr int num_subset_pixels = model_block_size * model_block_size * model_block_size;
    bool          passed            = true;

    wxPrintf("Dims of slices: real_slice == [%ld, %ld, %ld]; std_dev_slice == [%ld, %ld, %ld]\n", real_slice.size(0), real_slice.size(1), real_slice.size(2), std_dev_slice.size(0), std_dev_slice.size(1), std_dev_slice.size(2));

#pragma omp parallel for default(shared) num_threads(threads)
    for ( int pixel_counter = 0; pixel_counter < num_subset_pixels; pixel_counter++ ) {
        int Z = pixel_counter / (model_block_size * model_block_size);
        int Y = (pixel_counter / model_block_size) % model_block_size;
        int X = pixel_counter % model_block_size;

        int x_coord = X + x;
        int y_coord = Y + y;
        int z_coord = Z + z;

        // int index = x_coord + X * (y_coord + Y * z_coord);

        // TODO: logic for the comparison
        // Should be comparing the slice tensors against their full counterparts
        // If difference between slice position and corresponding full tensor position is greater than 1e-6 (floating point comparison), then the slices were not copied correctlys
        if ( (real_slice[X][Y][Z].item<float>( ) - full_real[x_coord][y_coord][z_coord].item<float>( ) > 1e-6) || (std_dev_slice[X][Y][Z].item<float>( ) - full_std_dev[x_coord][y_coord][z_coord].item<float>( ) > 1e-6) ) {
            passed = false;
        }
    }

    return passed;
}

void read_real_values_from_text_file(const std::string& fname, std::unique_ptr<float[]>& array_to_fill, const int& box_size) {
    std::ifstream ifs(fname);
    float         cur;
    long          pixel_counter = 0;
    if ( ifs.is_open( ) ) {
        while ( ifs >> cur ) {
            array_to_fill[pixel_counter] = cur;
            pixel_counter++;
        }
    }
    else {
        wxPrintf("Error opening file \"%s\" for reading.\n", fname);
    }
}

/**
 * @brief Write out the real values of an image to a text file.
 * 
 * @param fname Desired filename.
 * @param box_size Cubic box size of the volume.
 * @param is_single_dim Specifies whether processing will be done in 3 dimensions or 1 dimension.
 * @param overwrite Specifies whether it's preferred to clear previous contents from file before writing.
 * @param tensor_data When data is contained in Tensor; nullptr if not using a tensor.
 * @param array_data When data is in 1D linearized form; nullptr if not using float array.
 */
void write_real_values_to_text_file(const std::string& ofname, const int& box_size, const bool& is_single_dim, bool overwrite, Tensor* tensor_data, float* array_data) {
    // 1. Create output filestream
    std::ofstream ofs;

    // 2. Optionally (mostly for debugging/testing) clear the text file when opening
    if ( overwrite ) {
        ofs.open(ofname, std::ofstream::trunc | std::ofstream::out);
        ofs.close( );
    }

    // 3. Open the text file for appending.
    ofs.open(ofname, std::ios::app);
    if ( ! ofs.is_open( ) ) {
        wxPrintf("Failed to open %s.\n", ofname);
    }

    // REMOVE SINGLE DIM LOGIC -- using linearized arrays means just looping directly over the array, don't need to account for single dimensionality
    if ( is_single_dim ) {
        // Only relevant for volumes; if this was a single 2D image, you'd probably get the dimensions from the MRC -- needed especially if not a square
        const int num_pixels = std::pow(box_size, 3);

        if ( array_data ) {
            for ( int i = 0; i < num_pixels; i++ ) {
                ofs << array_data[i] << std::endl;
            }
        }
        else if ( tensor_data ) {
            for ( int i = 0; i < num_pixels; i++ ) {
                ofs << tensor_data->index({i}).item<float>( ) << std::endl;
            }
        }
        else {
            MyDebugPrintWithDetails("Error: array_data and tensor_data are null!\n");
        }
    }
    // NOTE: for simplicity's sake, I'm just assuming array_data is always 1D, and only tensor_data can be 3D
    else {
        if ( tensor_data ) {
            Tensor tensor_copy = tensor_data->clone( );
            auto   s           = tensor_data->sizes( );
            for ( int i = 0; i < s.size( ); i++ ) {
                wxPrintf("%li ", s[i]);
            }
            wxPrintf("\n");
            while ( tensor_copy.dim( ) > 3 ) {
                tensor_copy = tensor_copy.squeeze(0);
            }
            for ( int k = 0; k < box_size; k++ ) {
                for ( int j = 0; j < box_size; j++ ) {
                    for ( int i = 0; i < box_size; i++ ) {
                        ofs << tensor_copy.index({i, j, k}).item<float>( ) << std::endl;
                    }
                }
            }
        }
        else {
            MyDebugPrintWithDetails("Error: tensor data is null!\n");
        }
    }

    ofs.close( );
}

/**
 * @brief Compare line by line the values contained in 2 text files.
 * 
 * @param fname1 Filename 1.
 * @param fname2 Filename 2.
 * @param ofname Output filename.
 * @param print_vals Whether to print the values alongside DIFFERENT or SAME
 */
void compare_text_file_lines(std::string fname1, std::string fname2, std::string ofname, const bool& print_vals) {
    std::ifstream f1, f2;
    f1.open(fname1);
    f2.open(fname2);

    if ( ! f1.is_open( ) ) {
        wxPrintf("ERROR: %s not open!\n", fname1);
    }
    if ( ! f2.is_open( ) ) {
        wxPrintf("ERROR: %s not open!\n", fname2);
    }
    std::ofstream comp_f(ofname, std::ofstream::trunc | std::ofstream::out);
    comp_f.close( );
    comp_f.open(ofname, std::ios::app);

    std::string s1, s2;
    while ( std::getline(f1, s1) && std::getline(f2, s2) ) {
        std::istringstream ss1(s1), ss2(s2);
        double             dev1, dev2;
        if ( ! (ss1 >> dev1) ) {
            wxPrintf("Error reading ss1 (istringstream) into dev1 (float).\n");
            break;
        }
        if ( ! (ss2 >> dev2) ) {
            wxPrintf("Error reading ss2 (istringstream) into dev2 (float).\n");
            break;
        }
        if ( ! print_vals )
            (std::abs(dev1 - dev2) > 1e-2) ? comp_f << "DIFFERENT" << std::endl : comp_f << "SAME" << std::endl;
        else
            (std::abs(dev1 - dev2) > 1e-2) ? comp_f << "DIFFERENT " << dev1 << " | " << dev2 << std::endl : comp_f << "SAME " << dev1 << " | " << dev2 << std::endl;
    }
    wxPrintf("Completed comparison of text files %s and %s.\n", fname1, fname2);
}

void calculate_average_difference(const std::string& ofname, const int& max_threads, const int& box_size, Tensor* t_data1, Tensor* t_data2, float* a_data1, float* a_data2) {
    // Tensor diffs = torch::zeros({box_size, box_size, box_size});
    float sum = 0.0;
#pragma omp parallel for default(shared) num_threads(max_threads) reduction(+ \
                                                                            : sum)
    for ( int p = 0; p < box_size * box_size * box_size; p++ ) {
        if ( t_data1 && t_data2 ) {
            int x = p % box_size;
            int y = (p / box_size) % box_size;
            int z = p / std::pow(box_size, 2);
            // diffs.index({x, y, z}) = t_data1.index({x, y, z}).item<float>( ) - t_data2.index({x, y, z}).item<float>( );
            sum += t_data1->index({x, y, z}).item<float>( ) - t_data2->index({x, y, z}).item<float>( );
        }
        else if ( a_data1 && a_data2 ) {
            sum += a_data1[p] - a_data2[p];
        }
        // else {
        //     MyDebugPrintWithDetails("Error when trying to access the needed data for average difference; no non-null data structures!\n");
        //     break;
        // }
    }
    float mean = sum / std::pow(box_size, 3);

    wxPrintf("Average difference = %f\n\n", mean);
}