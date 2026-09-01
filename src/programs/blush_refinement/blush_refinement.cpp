// LibTorch includes MUST come first!!
// LibTorch v2.5.0+cpu used to run this code.

#include "../../core/core_headers.h"
// #include "../../constants/constants.h"
#include <vector>

#include <wx/process.h>
#include <wx/txtstrm.h>
#include <wx/utils.h> // Needed for wxMilliSleep
#include <mkl_service.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdint>
#include <string>
#include <wx/string.h>
#include <wx/log.h>

/**
 * @brief Used by BlushRefinement class to spawn and communicate with the ipc_blush
 * helper process to isolate the LibTorch library to a binary separate from cisTEM
 * dependencies. Uses a pipe-based transmission which can quickly send serialized
 * data between the two processes, including raw pixel data and parameters necessary
 * for executing the BlushHelpers::ApplyBlush function.
 * 
 */
class TorchShim {
  private:
    pid_t m_pid    = -1;
    int   m_out_fd = -1;
    int   m_in_fd  = -1;

  public:
    // Prevent accidental copying (prevents duplicate closes on pipes)
    TorchShim(const TorchShim&)            = delete;
    TorchShim& operator=(const TorchShim&) = delete;

    // Allow moving if needed
    TorchShim(TorchShim&& other) noexcept
        : m_pid(other.m_pid), m_out_fd(other.m_out_fd), m_in_fd(other.m_in_fd) {
        other.m_pid    = -1;
        other.m_out_fd = -1;
        other.m_in_fd  = -1;
    }

    TorchShim& operator=(TorchShim&& other) noexcept {
        if ( this != &other ) {
            Cleanup( );
            m_pid          = other.m_pid;
            m_out_fd       = other.m_out_fd;
            m_in_fd        = other.m_in_fd;
            other.m_pid    = -1;
            other.m_out_fd = -1;
            other.m_in_fd  = -1;
        }
        return *this;
    }

    void Cleanup( ) {
        if ( m_out_fd != -1 ) {
            close(m_out_fd);
            m_out_fd = -1;
        }
        if ( m_in_fd != -1 ) {
            close(m_in_fd);
            m_in_fd = -1;
        }
        if ( m_pid > 0 ) {
            int status;
            waitpid(m_pid, &status, 0);
            m_pid = -1;
        }
    }

    bool WriteAll(const void* buffer, size_t total_bytes) {
        const char* ptr           = static_cast<const char*>(buffer);
        size_t      bytes_written = 0;
        while ( bytes_written < total_bytes ) {
            ssize_t n = ::write(m_out_fd, ptr + bytes_written, total_bytes - bytes_written);
            if ( n <= 0 ) {
                if ( n < 0 && (errno == EAGAIN || errno == EINTR) ) {
                    continue;
                }
                return false;
            }
            bytes_written += n;
        }
        return true;
    }

    bool ReadAll(void* buffer, size_t total_bytes) {
        char*  ptr        = static_cast<char*>(buffer);
        size_t bytes_read = 0;

        while ( bytes_read < total_bytes ) {
            ssize_t n = ::read(m_in_fd, ptr + bytes_read, total_bytes - bytes_read);
            if ( n <= 0 ) {
                if ( n < 0 && (errno == EAGAIN || errno == EINTR) )
                    continue;
                return false; // EOF or pipe error
            }
            bytes_read += n;
        }
        return true;
    }

    TorchShim(const wxString& executablePath) {
        int in_pipe[2];
        int out_pipe[2];

        if ( pipe(in_pipe) != 0 || pipe(out_pipe) != 0 ) {
            wxLogError("TorchShim error: Failed to create POSIX pipes.");
            return;
        }

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);

        posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);

        posix_spawn_file_actions_addclose(&actions, in_pipe[1]);
        posix_spawn_file_actions_addclose(&actions, out_pipe[0]);

        // wxScopedCharBuffer path_buffer = executablePath.mb_str( );
        std::string   path_buffer = executablePath.ToStdString( );
        char*         argv[]      = {const_cast<char*>(path_buffer.c_str( )), nullptr};
        extern char** environ;

        int status = posix_spawnp(&m_pid, executablePath.mb_str( ), &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions);

        close(in_pipe[0]);
        close(out_pipe[1]);

        if ( status != 0 ) {
            wxLogError("TorchShim error: posix_spawn failed with error code %d", status);
            close(in_pipe[1]);
            close(out_pipe[0]);
            m_pid = -1;
            return;
        }

        m_out_fd = in_pipe[1];
        m_in_fd  = out_pipe[0];
    }

    ~TorchShim( ) {
        Cleanup( );
    }

    bool IsValid( ) const { return m_pid > 0 && m_out_fd != -1 && m_in_fd != -1; }

    std::vector<char> ProcessJob(const std::vector<char>& payload, std::function<bool(int, long)> progress_callback = nullptr) {

        ////////////////////////////////////////////////////////////////////////////
        // DEBUG:
        // std::ofstream dump("/tmp/blush_payload.bin", std::ios::binary);
        // uint32_t len_prefix = payload.size();
        // dump.write(reinterpret_cast<const char*>(&len_prefix), sizeof(len_prefix));
        // dump.write(payload.data(), payload.size());
        // dump.close();
        ////////////////////////////////////////////////////////////////////////////

        if ( ! IsValid( ) ) {
            wxLogError("TorchShim error: Subprocess not running or pipes closed.");
            return { };
        }

        auto     write_start = std::chrono::high_resolution_clock::now( );
        uint32_t len         = payload.size( );
        if ( ! WriteAll(&len, sizeof(len)) ) {
            wxLogError("TorchShim error: Failed to write payload length.");
            return { };
        }
        if ( ! WriteAll(payload.data( ), len) ) {
            wxLogError("TorchShim error: Failed to write full payload.");
            return { };
        }
        auto write_end = std::chrono::high_resolution_clock::now( );
        wxPrintf("[PROFILE] Payload write (%u bytes): %g ms\n", len, std::chrono::duration<double, std::milli>(write_end - write_start).count( ));

        while ( true ) {
            uint32_t msg_type = 0;

            if ( ! ReadAll(&msg_type, sizeof(msg_type)) ) {
                wxLogError("TorchShim error: Failed to read message type from child.");
                return { };
            }

            if ( msg_type == 1 ) {
                int32_t pct = 0;
                int64_t s   = 0;
                if ( ! ReadAll(&pct, sizeof(pct)) || ! ReadAll(&s, sizeof(s)) ) {
                    wxLogError("TorchShim error: Failed to read progress data.");
                    return { };
                }

                if ( progress_callback ) {
                    if ( ! progress_callback(pct, s) ) {
                        wxLogError("TorchShim error: failed to transmit progress data to GUI.");
                        return { };
                    }
                }
            }
            else if ( msg_type == 2 ) {
                uint32_t out_len = 0;
                if ( ! ReadAll(&out_len, sizeof(out_len)) || out_len == 0 ) {
                    wxLogError("TorchShim error: Failed to read output length prefix from child.");
                    return { };
                }

                auto              read_start = std::chrono::high_resolution_clock::now( );
                std::vector<char> result(out_len);
                if ( ! ReadAll(result.data( ), out_len) ) {
                    wxLogError("TorchShim error: Failed to read full output result payload.");
                    return { };
                }
                auto read_end = std::chrono::high_resolution_clock::now( );
                wxPrintf("[PROFILE] Result read (%u bytes): %g ms\n", out_len, std::chrono::duration<double, std::milli>(read_end - read_start).count( ));
                return result;
            }
            else {
                wxLogError("TorchShim error: received unknown message type from blush subprocess.");
                return { };
            }
        }

        wxLogError("TorchShim error: something went wrong and no proper return was reached.");
        return { };
    };
};

#include "blush_helpers.h"
#include <numeric>
#include <chrono>

using namespace torch::indexing;

class
        BlushRefinement : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
    std::vector<char> SerializeBlushJob(const std::string& model_filename, float mask_radius, int box_size, int num_threads, int batch_size, const std::vector<float>& pixel_data, const std::string& log_directory = "");
};

IMPLEMENT_APP(BlushRefinement)

/**
 * @brief For testing the BlushHelpers::ApplyBlush function in the standalone CLI program
 * 
 */
void BlushRefinement::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("Blush_Refinement", 1.0);

    std::string input_volume_filename            = my_input->GetFilenameFromUser("Input mrc filename", "The volume that will be denoised via blush refinement", "input.mrc", true);
    std::string output_mrc_filename              = my_input->GetStringFromUser("Output mrc filename", "The denoised volume post blush refinement", "output.mrc");
    float       desired_mask_radius_in_angstroms = my_input->GetFloatFromUser("Mask radius (angstroms)", "The user desired mask radius since it's not being derived from particle diameter", "20.0", 0.0f);
    float       pixel_size                       = my_input->GetFloatFromUser("Pixel size", "", "1.5", 0.5);
    int         batch_size                       = my_input->GetIntFromUser("Batch size", "Number of 64x64x64 blocks per forward pass to blush model", "1", 1, 10);

    int max_threads = 1;

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Enter desired number of threads", "Number of threads for parallelizing localized standard deviation calculation", "1", 1, 256);
#endif

    // CLI mode: pass empty string to use default ~/.cisTEM/logs/blush/
    // (Logger will expand HOME and create directories)
    std::string log_dir = "";

    delete my_input;

    my_current_job.Reset(7);
    my_current_job.ManualSetArguments("tttffii", input_volume_filename.c_str( ),
                                      output_mrc_filename.c_str( ),
                                      log_dir.c_str( ),
                                      desired_mask_radius_in_angstroms,
                                      pixel_size,
                                      batch_size,
                                      max_threads);
}

bool BlushRefinement::DoCalculation( ) {
    std::string ifname{my_current_job.arguments[0].ReturnStringArgument( )};
    std::string ofname{my_current_job.arguments[1].ReturnStringArgument( )};

    // - CLI mode (is_running_locally): logs go to ~/.cisTEM/logs/blush/
    // - GUI mode: logs go to <project_dir>/Scratch/BlushLogs
    std::string log_dir{my_current_job.arguments[2].ReturnStringArgument( )};

    float mask_radius{my_current_job.arguments[3].ReturnFloatArgument( )};
    float pixel_size{my_current_job.arguments[4].ReturnFloatArgument( )};
    int   batch_size{my_current_job.arguments[5].ReturnIntegerArgument( )};
    int   max_threads{my_current_job.arguments[6].ReturnIntegerArgument( )};
    int   ref_file_idx;
    if ( ! is_running_locally ) {
        ref_file_idx = my_current_job.arguments[7].ReturnIntegerArgument( );
    }

    // Read in the MRC
    MRCFile input_file(ifname);
    Image   input_volume;
    int     box_size{input_file.ReturnXSize( )};
    input_volume.ReadSlices(&input_file, 1, box_size);

    constexpr float MODEL_VOXEL_SIZE{1.5};

    // Scale the pixel data in preparation for the 1.5 pixel size expected by the blush model
    float     scale_factor{1.0f};
    const int original_box_size{input_volume.logical_x_dimension};
    bool      must_resample{false};
    int       new_box_size{input_volume.logical_x_dimension};
    {
        float           wanted_sf{pixel_size / MODEL_VOXEL_SIZE};
        constexpr float tolerance{1e-2f};
        new_box_size = static_cast<int>(std::floor(input_volume.logical_x_dimension * wanted_sf + 0.5f));
        if ( new_box_size % 2 != 0 )
            new_box_size++;

        scale_factor  = static_cast<float>(new_box_size) / static_cast<float>(input_volume.logical_x_dimension);
        must_resample = (std::abs(scale_factor - 1.0f) > tolerance);

        if ( must_resample ) {
            input_volume.ForwardFFT( );
            input_volume.Resize(new_box_size, new_box_size, new_box_size);
            input_volume.BackwardFFT( );
        }
    }

    input_volume.RemoveFFTWPadding( );

    // Determine total number of iterations:
    constexpr int block_size{64};
    constexpr int stride_size{20};

    // This ensures that even a partial batch counts as 1 full iteration
    // This is exactly how many times for ( auto it_coords : it ) loop will run in BlushHelpers::ApplyBlush

    int total_blush_iterations;
    // Account for precise blocking to get appropriate expected iteration count.
    {
        int span    = input_volume.logical_x_dimension - block_size;
        int n_steps = (span / stride_size) + 1;
        int last    = (n_steps - 1) * stride_size;
        if ( last != span )
            n_steps += 1; // account for the pushed-back edge block
        total_blush_iterations = n_steps * n_steps * n_steps;
    }

    ProgressBar* progress_bar;
    if ( is_running_locally ) {
        progress_bar = new ProgressBar(100);
    }
    wxPrintf("\n\nBlushing %s...\n\n", ifname);
    auto stop_flag = std::make_shared<std::atomic<bool>>(false);
    auto startTime = std::chrono::high_resolution_clock::now( );

    // Load pixel data from Image for passage for passage to serialization function
    std::vector<float> pixel_data(new_box_size * new_box_size * new_box_size);
    std::memcpy(pixel_data.data( ), input_volume.real_values, sizeof(float) * new_box_size * new_box_size * new_box_size);

    // Get paths for the blush weights and the blush subprocess, ipc_blush
    std::string exe_path       = wxStandardPaths::Get( ).GetExecutablePath( ).ToStdString( );
    size_t      last_slash_idx = exe_path.rfind('/');
    if ( std::string::npos != last_slash_idx ) {
        exe_path = exe_path.substr(0, last_slash_idx + 1);
    }
    std::string model_filename = exe_path + "blush_weights.dat";
    exe_path += "ipc_blush";

    // Perform serialization with the necessary parameters
    auto              serialize_start = std::chrono::high_resolution_clock::now( );
    std::vector<char> buffer_data     = SerializeBlushJob(model_filename, mask_radius, new_box_size, max_threads, batch_size, pixel_data, log_dir);
    auto              serialize_end   = std::chrono::high_resolution_clock::now( );
    // wxPrintf("[PROFILE] Serialization: %g ms\n", std::chrono::duration<double, std::milli>(serialize_end - serialize_start).count( ));
    TorchShim shim{wxString(exe_path)};
    int       pct_complete = 0;

    wxPrintf("Performing inference...\n");
    if ( shim.IsValid( ) ) {
        JobResult*        intermediate_result;
        float             gui_result_params[3];
        std::vector<char> result_bytes = shim.ProcessJob(buffer_data, [this, &pct_complete, progress_bar, &intermediate_result, &gui_result_params](int pct_increment, long s_rem) {
            if ( this->is_running_locally ) {
                pct_complete += pct_increment;
                std::min(pct_complete, 100);
                progress_bar->Update(pct_complete);
            }
            else {
                // GUI transmission logic
                intermediate_result = new JobResult;
                // wxPrintf("Performing update for %i\n%", pct_complete);
                intermediate_result->job_number = this->my_current_job.job_number;
                gui_result_params[0]            = -1; // indicates that no update should be made GUI-side for the filename
                gui_result_params[1]            = pct_increment;
                gui_result_params[2]            = s_rem;
                intermediate_result->SetResult(3, gui_result_params);
                AddJobToResultQueue(intermediate_result);
            }

            return true;
        });

        wxPrintf("Completed inference.\n");
        const size_t expected_bytes = static_cast<size_t>(new_box_size) * new_box_size * new_box_size * sizeof(float);
        if ( result_bytes.size( ) != expected_bytes ) {
            wxLogError("TorchShim error: expected %zu bytes from subprocess, got %zu\n", expected_bytes, result_bytes.size( ));
            return false;
        }
        std::memcpy(input_volume.real_values, result_bytes.data( ), new_box_size * new_box_size * new_box_size * sizeof(float));
    }

    if ( must_resample ) {
        input_volume.ForwardFFT( );
        input_volume.Resize(original_box_size, original_box_size, original_box_size);
        input_volume.BackwardFFT( );
    }
    input_volume.AddFFTWPadding( );

    auto   endTime  = std::chrono::high_resolution_clock::now( );
    double duration = std::chrono::duration<double>(endTime - startTime).count( );
    // wxPrintf("\n\ntotalForwardTime == %g min\naverage forward time == %g min\n", totalForwardTime / 60, (totalForwardTime / 60) / total_blush_iterations);
    wxPrintf("\nTotal blush run time: %g\n\n", duration);
    delete progress_bar;
    MRCFile ofile(ofname, true);
    input_volume.WriteSlices(&ofile, 1, box_size);
    ofile.SetPixelSizeAndWriteHeader(pixel_size);

    // Now that file is written, send the index of the reference back to the GUI so that it can be updated
    if ( ! is_running_locally ) {
        float* results = new float[3];
        results[0]     = static_cast<float>(ref_file_idx);
        results[1]     = 0;
        results[2]     = 0;
        my_result.SetResult(3, results);
        delete[] results;
    }
    return true;
}

std::vector<char> BlushRefinement::SerializeBlushJob(const std::string& model_filename, float mask_radius, int box_size, int num_threads, int batch_size, const std::vector<float>& pixel_data, const std::string& log_directory) {
    size_t filename_length = model_filename.size( );
    size_t log_dir_length  = log_directory.size( );
    size_t header_size     = sizeof(filename_length) + filename_length + sizeof(log_dir_length) + log_dir_length + sizeof(mask_radius) + sizeof(box_size) + sizeof(num_threads) + sizeof(batch_size);
    size_t pixel_bytes     = pixel_data.size( ) * sizeof(float);

    std::vector<char> payload(header_size + pixel_bytes);
    size_t            offset = 0;

    // Pack the header parameters in the EXACT order the child expects them

    // Model filename (length + data)
    std::memcpy(payload.data( ) + offset, &filename_length, sizeof(filename_length));
    offset += sizeof(filename_length);
    std::memcpy(payload.data( ) + offset, model_filename.data( ), filename_length);
    offset += filename_length;

    // Log directory (length + data) - NEW parameter for production logging
    std::memcpy(payload.data( ) + offset, &log_dir_length, sizeof(log_dir_length));
    offset += sizeof(log_dir_length);
    if ( log_dir_length > 0 ) {
        std::memcpy(payload.data( ) + offset, log_directory.data( ), log_dir_length);
        offset += log_dir_length;
    }

    // Remaining parameters
    std::memcpy(payload.data( ) + offset, &mask_radius, sizeof(float));
    offset += sizeof(float);
    std::memcpy(payload.data( ) + offset, &box_size, sizeof(int));
    offset += sizeof(int);
    std::memcpy(payload.data( ) + offset, &num_threads, sizeof(int));
    offset += sizeof(int);
    std::memcpy(payload.data( ) + offset, &batch_size, sizeof(int));
    offset += sizeof(int);

    // Pack the pixel floats
    std::memcpy(payload.data( ) + offset, pixel_data.data( ), pixel_bytes);

    return payload;
}