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
const std::complex<float> I(0.0,1.0);
#include <iterator>
#include <utility>
#include <vector>
#include <random>
#include <functional>
#include <fftw3.h>
#include <math.h>
#include "sqlite/sqlite3.h"
#include <wx/wx.h>
#include <wx/socket.h>
#include <wx/cmdline.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include <wx/dir.h>
#include <wx/wfstream.h>
#include <wx/tokenzr.h>
#include <wx/txtstrm.h>
#include <wx/textfile.h>
#include <wx/regex.h>
#include <wx/stackwalk.h>
#include <wx/xml/xml.h>


class StackDump : public wxStackWalker // so we can give backtraces..
{
public:
    StackDump(const char *argv0)
        : wxStackWalker(argv0)
    {
    }

    virtual void Walk(size_t skip = 1)
    {
        wxPrintf("Stack dump:\n\n");

        wxStackWalker::Walk(skip);
    }

protected:
    virtual void OnStackFrame(const wxStackFrame& frame)
    {
        wxPrintf("[%2i] ", int(frame.GetLevel()));

        wxString name = frame.GetName();
        if ( !name.empty() )
        {
            wxPrintf("%-20.40s", name.mb_str());
        }
        else
        {
            wxPrintf("0x%08lx", (unsigned long)frame.GetAddress());
        }

        if ( frame.HasSourceLocation() )
        {
            wxPrintf("\t%s:%i",
                   frame.GetFileName().mb_str(),
                   int(frame.GetLine()));
        }

        wxPrintf("");

        wxString type, val;
        for ( size_t n = 0; frame.GetParam(n, &type, &name, &val); n++ )
        {
            wxPrintf("\t%s %s = %s\n", type.mb_str(), name.mb_str(), val.mb_str());

        }
        wxPrintf("\n");
    }
};


#include "cistem_parameters.h"
#include "cistem_star_file_reader.h"
#include "defines.h"
#include "assets.h"
#include "asset_group.h"
#include "socket_codes.h"
#include "template_matching.h"
#include "functions.h"
#include "run_profiles.h"
#include "job_packager.h"
#include "ctf.h"
#include "curve.h"
#include "abstract_image_file.h"
#include "mrc_header.h"
#include "mrc_file.h"
#include "dm_file.h"
#include "tiff/tiffio.h"
#include "tiff_file.h"
#include "image_file.h"
#include "matrix.h"
#include "angles_and_shifts.h"
#include "empirical_distribution.h"
#include "randomnumbergenerator.h"
#include "image.h"
#include "socket_communicator.h"
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
#include "json/json_defs.h"
#include "json/jsonwriter.h"
#include "json/jsonreader.h"
#include "json/jsonval.h"
#ifdef EXPERIMENTAL
#include "pdb.h"
#include "water.h"
#endif

#ifdef USEGPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>
#include <npp.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_statistics_functions.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <typeinfo>
#include <limits>
#include <omp.h>
#include "../../gpu/src/DeviceManager.h"
#include "../../gpu/src/ContextManager.h"
#include "../../gpu/src/GpuImage.h"
#include "../../gpu/src/TemplateMatchingCore.h"
#endif





#ifdef MKL
#include <mkl.h>
#endif

extern RandomNumberGenerator global_random_number_generator;
