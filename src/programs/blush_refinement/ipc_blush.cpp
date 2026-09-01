//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
	ipc_blush is a binary whose sole purpose is to execute functionality provided by a third party library.
	It is spawned using C/C++ classes meant to follow POSIX standards for interprocess communication
	between two different processes. This design choice was made after attempting to integrate LibTorch C++ API
	with existing cisTEM binaries, which causes conflicts between the Intel MKL used by cisTEM, and the MKL
	packaged within LibTorch. The solution was to create a purely standalone binary that only links against
	standard C/C++ libraries and LibTorch, bypassing all unnecessary libraries that the inference logic will
	not use.

	Why spawned with C/C++ POSIX instead of WX?

	The wxProcess class is used by the main GUI thread to spawn subprocesses. However, only the application's
	main thread is allowed to do this. Trying to spawn wxProcess objects from an existing wxProcess hits an
	assert in the wxWidgets library that prevents the secondary subprocess from starting. Using C/C++ POSIX
	bypasses this, allowing the math to be performed in the desired standalone binary. This also simplifies
	the logic for calling blush_refinement from the GUI or the CLI.
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "blush_logger.h"
#include "blush_helpers.h"
#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstring> // Needed for std::memcpy
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <sys/prctl.h>
#include <signal.h>

// #include <mkl_service.h>

void SignalHandler(int sig) {
    BLUSH_LOG_ERROR("FATAL CRASH: OS sent signal %d to child process!", sig);
    exit(1);
}

// Helper to read exactly N bytes from stdin
bool ReadExact(void* buffer, size_t length) {
    uint8_t* ptr        = static_cast<uint8_t*>(buffer);
    size_t   total_read = 0;
    while ( total_read < length ) {
        size_t r = fread(ptr + total_read, 1, length - total_read, stdin);
        if ( r == 0 ) {
            // EOF or pipe broken
            BLUSH_LOG_WARN("ReadExact: EOF or pipe broken (total_read=%zu, expected=%zu)", total_read, length);
            return false;
        }
        total_read += r;
    }
    return true;
}

// Bulletproof binary pipe writer
bool WriteExact(const void* buffer, size_t length) {
    const uint8_t* ptr           = static_cast<const uint8_t*>(buffer);
    size_t         total_written = 0;
    while ( total_written < length ) {
        size_t w = fwrite(ptr + total_written, 1, length - total_written, stdout);
        if ( w == 0 ) {
            return false;
        }
        total_written += w;
    }
    fflush(stdout); // CRITICAL
    return true;
}

int main( ) {
    // Register crash handlers before any logging
    std::signal(SIGSEGV, SignalHandler);
    std::signal(SIGABRT, SignalHandler);
    std::signal(SIGILL, SignalHandler);
    std::signal(SIGFPE, SignalHandler);
    std::signal(SIGTERM, SignalHandler);
    std::signal(SIGINT, SignalHandler);

    /* NOTE: this form of subprocess termination only works on Linux, not any other OS; going with this for now because it simplifies
     * handling process termination without getting into SocketCommunicator logic by tracking the PID for the ipc_blush subprocess
	 * and terminating it as well when the termination signal is received.
	 */
    prctl(PR_SET_PDEATHSIG, SIGTERM);

    while ( true ) {
        auto payload_receive_start = std::chrono::high_resolution_clock::now( );

        // 1. Read the 4-byte length prefix
        uint32_t payload_len = 0;
        if ( ! ReadExact(reinterpret_cast<char*>(&payload_len), sizeof(payload_len)) ) {
            // Parent closed pipe - exit cleanly (this is normal shutdown)
            break;
        }

        // 2. Read the full payload
        std::vector<char> buffer(payload_len);
        if ( ! ReadExact(buffer.data( ), payload_len) ) {
            BLUSH_LOG_ERROR("Failed to read payload data (expected %u bytes)", payload_len);
            break;
        }
        auto   payload_receive_end = std::chrono::high_resolution_clock::now( );
        double payload_receive_ms  = std::chrono::duration<double, std::milli>(payload_receive_end - payload_receive_start).count( );

        // 3. Deserialize parameters in EXACT order they were serialized

        size_t offset = 0;

        // Model filename (length + data)
        if ( offset + sizeof(size_t) > buffer.size( ) ) {
            BLUSH_LOG_ERROR("Buffer too small for model_filename length");
            continue;
        }
        size_t model_filename_length;
        std::memcpy(&model_filename_length, buffer.data( ) + offset, sizeof(size_t));
        offset += sizeof(size_t);

        if ( offset + model_filename_length > buffer.size( ) ) {
            BLUSH_LOG_ERROR("Buffer too small for model_filename data");
            continue;
        }
        std::string model_filename;
        model_filename.resize(model_filename_length);
        std::memcpy(model_filename.data( ), buffer.data( ) + offset, model_filename_length);
        offset += model_filename_length;

        // Log directory (length + data) - NEW parameter for production logging
        if ( offset + sizeof(size_t) > buffer.size( ) ) {
            BLUSH_LOG_ERROR("Buffer too small for log_directory length");
            continue;
        }
        size_t log_dir_length;
        std::memcpy(&log_dir_length, buffer.data( ) + offset, sizeof(size_t));
        offset += sizeof(size_t);

        std::string log_directory;
        if ( log_dir_length > 0 ) {
            if ( offset + log_dir_length > buffer.size( ) ) {
                BLUSH_LOG_ERROR("Buffer too small for log_directory data");
                continue;
            }
            log_directory.resize(log_dir_length);
            std::memcpy(log_directory.data( ), buffer.data( ) + offset, log_dir_length);
            offset += log_dir_length;
        }

// Initialize logger now that we have log directory
// (GUI mode provides project path, CLI mode passes empty string)
#ifdef BLUSH_DEBUG_LOGGING
        BlushLogger::InitializeLogger(log_directory, BlushLogger::Level::INFO);
        BLUSH_LOG_INFO("ipc_blush subprocess started");
        BLUSH_LOG_INFO("[PROFILE] Payload receive (%u bytes): %.1f ms", payload_len, payload_receive_ms);
        BLUSH_LOG_INFO("Model: %s", model_filename.c_str( ));
        if ( ! log_directory.empty( ) ) {
            BLUSH_LOG_INFO("Log directory: %s (GUI mode)", log_directory.c_str( ));
        }
        else {
            BLUSH_LOG_INFO("Using default log directory (CLI mode)");
        }
#endif

        // {
        //     std::string msg = "Successfully loaded model_filename: " + model_filename + "\n";
        //     DbgLog(msg.c_str( ));
        // }
        // Ensure the buffer is large enough to hold all necessary data.
        size_t remaining_header = sizeof(float) + (3 * sizeof(int));
        if ( offset + remaining_header > buffer.size( ) ) {
            continue;
        }

        // float pixel_size;
        float mask_radius;
        int   box_size;
        int   num_threads;
        int   batch_size;

        // std::memcpy(&pixel_size,  buffer.data() + offset, sizeof(float)); offset += sizeof(float);
        std::memcpy(&mask_radius, buffer.data( ) + offset, sizeof(float));
        offset += sizeof(float);
        std::memcpy(&box_size, buffer.data( ) + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&num_threads, buffer.data( ) + offset, sizeof(int));
        offset += sizeof(int);
        std::memcpy(&batch_size, buffer.data( ) + offset, sizeof(int));
        offset += sizeof(int);

        // DbgLog("Successfully parsed remaining serialized data.\n");

        // Calculate remaining bytes for pixel data
        size_t pixel_bytes = buffer.size( ) - offset;

        // Make sure all buffer bytes were fully read
        if ( pixel_bytes % sizeof(float) != 0 ) {
            continue;
        }

        // DbgLog("Verified all bytes read\n");
        size_t num_floats = pixel_bytes / sizeof(float);

        // Pre-allocate the vector before copying!
        std::vector<float> pixel_data(num_floats);
        std::memcpy(pixel_data.data( ), buffer.data( ) + offset, pixel_bytes);

        constexpr int block_size             = 64;
        constexpr int stride_size            = 20;
        const int     total_blush_iterations = pow(((box_size - block_size) / stride_size) + 1, 3);

        auto stop_flag = std::make_shared<std::atomic<bool>>(false);

        // Execute blush with a dummy lambda (no UI progress bar!)
        // std::cerr << "About to execute BlushHelpers::ApplyBlush..." << std::endl;

        enum TransmissionType { PROGRESS = 1,
                                RESULT   = 2 };

        // Lambda to help transmit progress and results back to blush_refinement parent
        auto write_to_parent = [](const void* buf, size_t count) {
            const char* p       = static_cast<const char*>(buf);
            size_t      written = 0;
            while ( written < count ) {
                ssize_t res = write(STDOUT_FILENO, p + written, count - written);
                if ( res <= 0 ) {
                    if ( res < 0 && (errno == EAGAIN || errno == EINTR) )
                        continue;
                    BLUSH_LOG_ERROR("write_to_parent FAILED: requested=%zu written_so_far=%zu errno=%d", count, written, errno);
                    return false;
                }
                written += res;
            }
            return true;
        };

#ifdef BLUSH_DEBUG_LOGGING
        BLUSH_LOG_INFO("Starting ApplyBlush: box_size=%d, mask_radius=%.1f, batch_size=%d, num_threads=%d",
                       box_size, mask_radius, batch_size, num_threads);
        auto inference_start = std::chrono::high_resolution_clock::now( );
#endif

        try {
            BlushHelpers::ApplyBlush(pixel_data, model_filename, box_size, mask_radius, batch_size, num_threads, stop_flag, [&write_to_parent](int percent, long seconds_remaining) {
                uint32_t msg_type = 1; // PROGRESS message type
                int32_t  pct      = static_cast<int32_t>(percent);
                int64_t  s_rem    = static_cast<int64_t>(seconds_remaining);

                // Send the progress data
                if ( ! write_to_parent(&msg_type, sizeof(msg_type)) )
                    return false;
                if ( ! write_to_parent(&pct, sizeof(pct)) )
                    return false;
                if ( ! write_to_parent(&s_rem, sizeof(s_rem)) )
                    return false;

                return true;
            });
#ifdef BLUSH_DEBUG_LOGGING
            auto inference_end = std::chrono::high_resolution_clock::now( );
            BLUSH_LOG_INFO("ApplyBlush completed successfully");
            BLUSH_LOG_INFO("[PROFILE] Model inference: %.1f ms", std::chrono::duration<double, std::milli>(inference_end - inference_start).count( ));
#endif

        } catch ( std::exception& e ) {
            BLUSH_LOG_ERROR("CRITICAL LIBTORCH ERROR: %s", e.what( ));
            break;
        }

        // 4. Send the result back
        // pixel_data is now modified. We need to serialize it back to bytes.
        size_t            out_bytes = pixel_data.size( ) * sizeof(float);
        std::vector<char> result_payload(out_bytes);
        std::memcpy(result_payload.data( ), pixel_data.data( ), out_bytes);
        uint32_t msg_type = 2; // RESULT message type
        uint32_t out_len  = result_payload.size( );

#ifdef BLUSH_DEBUG_LOGGING
        BLUSH_LOG_INFO("Sending RESULT: %u bytes", out_len);
#endif

        auto result_write_start = std::chrono::high_resolution_clock::now( );
        bool ok1                = write_to_parent(&msg_type, sizeof(msg_type));
        bool ok2                = write_to_parent(&out_len, sizeof(out_len));
        bool ok3                = write_to_parent(result_payload.data( ), out_len);
        auto result_write_end   = std::chrono::high_resolution_clock::now( );

        if ( ! (ok1 && ok2 && ok3) ) {
            BLUSH_LOG_ERROR("Failed to send RESULT: ok1=%d ok2=%d ok3=%d", int(ok1), int(ok2), int(ok3));
        }
#ifdef BLUSH_DEBUG_LOGGING
        else {
            BLUSH_LOG_INFO("RESULT sent successfully");
            BLUSH_LOG_INFO("[PROFILE] Result write (%u bytes): %.1f ms", out_len, std::chrono::duration<double, std::milli>(result_write_end - result_write_start).count( ));
        }
#endif
    }
#ifdef BLUSH_DEBUG_LOGGING
    BLUSH_LOG_INFO("ipc_blush exiting normally");
#endif
    return 0;
}