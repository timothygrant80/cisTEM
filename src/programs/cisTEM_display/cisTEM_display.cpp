#include "../../core/gui_core_headers.h"

class DisplayApp : public wxApp {
  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
};

IMPLEMENT_APP(DisplayApp)

DisplayFrame* display_frame;

bool DisplayApp::OnInit( ) {
    wxInitAllImageHandlers( );
    display_frame = new DisplayFrame(NULL, wxID_ANY, "cisTEM Display", wxPoint(-1, -1), wxSize(-1, -1), wxDEFAULT_FRAME_STYLE);
    display_frame->Layout( );
    display_frame->Show(true);

    return true;
}

int DisplayApp::OnExit( ) {
    return 0;
}
