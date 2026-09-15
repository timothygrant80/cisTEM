#ifndef SRC_PROGRAMS_CORE_CORE_HEADERS_H_
#define SRC_PROGRAMS_CORE_CORE_HEADERS_H_

typedef struct Peak {
    float x;
    float y;
    float z;
    float value;
    long  physical_address_within_image;
} Peak;

typedef struct Kernel2D {
    int   pixel_index[4];
    float pixel_weight[4];
} Kernel2D;

typedef struct CurvePoint {
    int   index_m;
    int   index_n;
    float value_m;
    float value_n;
} CurvePoint;

// All the defines set in configure.ac
#include <cistem_config.h>
#ifndef _LARGE_FILE_SOURCE
#define _LARGE_FILE_SOURCE
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "../constants/constants.h"
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <cstdarg>
#include <cfloat>
#include <complex>
#include <iterator>
#include <utility>
#include <vector>
#include <unordered_map>
#include <random>
#include <functional>
#ifdef MKL
// These are in $MKLROOT/include
#include <fftw/fftw3.h>
#include <fftw/fftw3_mkl.h>
#else
// These should'nt be used, but are here for completeness.
// See note on licensing.
#include <fftw3.h>
#endif
#include <math.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include "sqlite/sqlite3.h"
#include <wx/wx.h>
#include <wx/txtstrm.h>
#include <wx/defs.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include <wx/dir.h>
#include <wx/wfstream.h>
#include <wx/tokenzr.h>
#include <wx/textfile.h>
#include <wx/log.h>
#include <wx/regex.h>
#include <wx/xml/xml.h>
#ifdef ENABLE_WEBVIEW
#include <wx/webview.h>
#endif

#include <execinfo.h>

// Prints a backtrace of the calling thread (used by the debug assert macros).
class StackDump {
  public:
    StackDump(const char* argv0 = NULL) { (void)argv0; }

    void Walk(size_t skip = 1) {
        void* frames[128];
        int   count = backtrace(frames, 128);
        printf("Stack dump:\n\n");
        char** symbols = backtrace_symbols(frames, count);
        for ( int frame = int(skip); frame < count; frame++ ) {
            if ( symbols != NULL )
                printf("[%2i] %s\n", frame - int(skip), symbols[frame]);
            else
                printf("[%2i] %p\n", frame - int(skip), frames[frame]);
        }
        printf("\n");
        free(symbols);
    }
};

#include "defines.h"
#include "string_functions.h"
#include "date_time.h"
#include "filesystem_functions.h"
#include "command_line_parser.h"
#include "event_loop.h"
#include "socket_communication_utils/tcp_socket.h"
#include "stopwatch.h"
#include "cistem_parameters.h"
#include "cistem_star_file_reader.h"
#include "assets.h"
#include "asset_group.h"
#include "socket_communication_utils/socket_codes.h"
#include "template_matching.h"
#include "functions.h"
#include "run_command.h"
#include "socket_communication_utils/run_profile.h"
#include "socket_communication_utils/run_profile_manager.h"
#include "socket_communication_utils/job_packager.h"
#include "ctf.h"
#include "curve.h"
#include "abstract_image_file.h"
#include "mrc_header.h"
#include "mrc_file.h"
#include "dm_file.h"
#include "tiff/tiffio.h"
#include "tiff_file.h"
#include "eer_file.h"
#include "image_file.h"
#include "matrix.h"
#include "angles_and_shifts.h"
#include "empirical_distribution.h"
#include "randomnumbergenerator.h"
#include "image.h"
#include "spectrum_image.h"
#include "socket_communication_utils/socket_communicator.h"
#include "userinput.h"
#include "symmetry_matrix.h"
#include "parameter_constraints.h"
#include "resolution_statistics.h"
#include "reconstructed_volume.h"
#include "particle.h"
#include "reconstruct_3d.h"
#include "electron_dose.h"
#include "angular_distribution_histogram.h"
#include "refinement_package.h"
#include "template_matches_package.h"
#include "refinement.h"
#include "classification.h"
#include "classification_selection.h"
#include "database.h"
#include "project.h"
#include "job_tracker.h"
#include "numeric_text_file.h"
#include "progressbar.h"
#include "downhill_simplex.h"
#include "brute_force_search.h"
#include "conjugate_gradient.h"
#include "euler_search.h"
#include "frealign_parameter_file.h"
#include "basic_star_file_reader.h"
#include "particle_finder.h"
#include "myapp.h"
#include "rle3d.h"
#include "local_resolution_estimator.h"
#include "ccl3d.h"
#include "pdb.h"

#ifdef ENABLEGPU
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <cufftXt.h>
#include <npp.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_statistics_functions.h>
#include <npps_arithmetic_and_logical_operations.h>
#include <typeinfo>
#include <limits>
#endif

#include "padded_coordinates.h"

#ifdef MKL
#define MKL_Complex8 std::complex<float>
#include <mkl.h>
#endif

extern RandomNumberGenerator global_random_number_generator;

#ifdef _OPENMP
#include <omp.h>
#endif

/*!
 * Returns the thread number of the current thread, numbered from 0
 */
inline int ReturnThreadNumberOfCurrentThread( ) {
#ifdef _OPENMP
    return omp_get_thread_num( );
#else
    return 0;
#endif
};

inline int ReturnNumberOfThreads( ) {
#ifdef _OPENMP
    return omp_get_num_threads( );
#else
    return 1;
#endif
};

inline bool ReturnInParallelRegionBool( ) {
#ifdef _OPENMP
    return omp_in_parallel( ) == 0 ? false : true;
#else
    return false;
#endif
}

#endif // SRC_PROGRAMS_CORE_CORE_HEADERS_H_
