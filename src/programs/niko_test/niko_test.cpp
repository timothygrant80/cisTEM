#include "../../core/core_headers.h"

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput( ) {
}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation( ) {

    Image t_img;
    t_img.Allocate(512, 512, true);

    for ( int i = 0; i < 10000; i++ ) {
        t_img.ForwardFFT( );
        t_img.BackwardFFT( );
    }
    return true;
}
