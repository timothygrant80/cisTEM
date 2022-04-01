#include "../../core/core_headers.h"

class
        GpuDevices : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(GpuDevices)

// override the DoInteractiveUserInput

void GpuDevices::DoInteractiveUserInput( ) {
}

// override the do calculation method which will be what is actually run..

bool GpuDevices::DoCalculation( ) {
    DeviceManager gpuDev;

    wxPrintf("\nGpuDevices is running...\n\n");

    gpuDev.ListDevices( );
    return true;
}
