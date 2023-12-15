#include "../../core/gui_core_headers.h"

class DisplayApp : public wxApp {
  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);
};

IMPLEMENT_APP(DisplayApp)

DisplayFrame* display_frame;

bool DisplayApp::OnInit( ) {
    wxInitAllImageHandlers( );

    display_frame = new DisplayFrame(NULL, wxID_ANY, "cisTEM Display", wxPoint(-1, -1), wxSize(-1, -1), wxDEFAULT_FRAME_STYLE);
    wxString cmd_full_filename;
    wxString cmd_filename;

    // Check if filenames are present; if so, open them
    if ( argc > 1 ) {
        for ( int i = 1; i < argc; i++ ) {
            cmd_filename = argv[i];
            wxFileName filename(cmd_filename);
            filename.Normalize(wxPATH_NORM_LONG | wxPATH_NORM_DOTS | wxPATH_NORM_TILDE | wxPATH_NORM_ABSOLUTE);
            cmd_full_filename = filename.GetFullPath( );
            display_frame->cisTEMDisplayPanel->OpenFile(cmd_full_filename, cmd_full_filename);
        }
    }

    display_frame->Layout( );
    display_frame->Show(true);
    return true;
}

void DisplayApp::OnInitCmdLine(wxCmdLineParser& parser) {
}

bool DisplayApp::OnCmdLineParsed(wxCmdLineParser& parser) {

    return true;
}

int DisplayApp::OnExit( ) {
    return 0;
}
