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
#include "defines.h"
#include "assets.h"
#include "asset_group.h"
#include "socket_codes.h"
#include "userinput.h"
#include "functions.h"
#include "randomnumbergenerator.h"
#include "tiff/tiffio.h"
#include "abstract_image_file.h"
#include "mrc_header.h"
#include "mrc_file.h"
#include "dm_file.h"
#include "tiff_file.h"
#include "image_file.h"
#include "matrix.h"
#include "symmetry_matrix.h"
#include "ctf.h"
#include "curve.h"
#include "angles_and_shifts.h"
#include "parameter_constraints.h"
#include "empirical_distribution.h"
#include "image.h"
#include "resolution_statistics.h"
#include "reconstructed_volume.h"
#include "particle.h"
#include "reconstruct_3d.h"
#include "electron_dose.h"
#include "run_profiles.h"
#include "angular_distribution_histogram.h"
#include "refinement_package.h"
#include "refinement.h"
#include "classification.h"
#include "classification_selection.h"
#include "database.h"
#include "project.h"
#include "job_packager.h"
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

#ifdef MKL
#include <mkl.h>
#endif

extern RandomNumberGenerator global_random_number_generator;

