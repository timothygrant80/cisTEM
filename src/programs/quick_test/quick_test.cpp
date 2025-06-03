#include <cistem_config.h>

#include "../../core/core_headers.h"
#include "../../constants/constants.h"

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#include "quick_test_gpu.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#include "../../core/scattering_potential.h"

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    void     AddCommandLineOptions( );
    wxString symmetry_symbol;
    bool     my_test_1 = false;
    bool     my_test_2 = true;
    int      idx;

    std::array<wxString, 2> input_starfile_filename;

  private:
};

IMPLEMENT_APP(QuickTestApp)

// Optional command-line stuff
void QuickTestApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("disable-user-input", "Disable interactive user input prompts. Default false");
}

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {
    // This flag allows skipping interactive prompts, useful for automated testing with Copilot.
    if ( command_line_parser.FoundSwitch("disable-user-input") ) {
        std::cout << "Skipping interactive user input as per command line flag." << std::endl;
        return;
    }
    UserInput* my_input = new UserInput("QuickTest", 2.0);

    idx                           = my_input->GetIntFromUser("Index", "", "", 0, 1000);
    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

bool QuickTestApp::DoCalculation( ) {

#ifdef ENABLEGPU
    // DeviceManager gpuDev;
    // gpuDev.ListDevices( );

    // QuickTestGPU quick_test_gpu;
    // quick_test_gpu.callHelloFromGPU(idx);
#endif

    return true;
}
