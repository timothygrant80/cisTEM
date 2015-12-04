typedef struct Peak {
  float x;
  float y;
  float z;
  float value;
} Peak;

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <cstdarg>
#include <cfloat>
#include <complex.h>
#include <fftw3.h>
#include <sqlite3.h>
#include <wx/wx.h>
#include <wx/socket.h>
#include <wx/cmdline.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include "defines.h"
#include "assets.h"
#include "asset_group.h"
#include "socket_codes.h"
#include "userinput.h"
#include "functions.h"
#include "mrc_header.h"
#include "mrc_file.h"
#include "curve.h"
#include "ctf.h"
#include "image.h"
#include "electron_dose.h"
#include "run_profiles.h"
#include "database.h"
#include "project.h"
#include "job_packager.h"
#include "job_tracker.h"
#include "myapp.h"


