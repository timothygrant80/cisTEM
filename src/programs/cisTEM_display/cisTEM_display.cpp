#include "../../core/gui_core_headers.h"
#include "DisplayServer.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <wx/snglinst.h>

// #define IPC_PORT 3456
// #define IPC_SERVICE_NAME "CistemDisplayService"

class MyServer;

class DisplayApp : public wxApp {
  public:
    virtual bool OnInit( );
    virtual int  OnExit( );
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);
    ~DisplayApp( );

  private:
    wxSingleInstanceChecker* m_checker;
};

IMPLEMENT_APP(DisplayApp)

DisplayFrame* display_frame;

DisplayApp::~DisplayApp( ) {
    DisplayServer::GetInstance( ).Stop( );
}

bool DisplayApp::OnInit( ) {

    const wxString name = wxString::Format("cisTEM_Display-%s", wxGetUserId( ));
    m_checker           = new wxSingleInstanceChecker(name);
    if ( m_checker->IsAnotherRunning( ) ) {
        if ( argc > 1 ) {
            int sock = socket(AF_UNIX, SOCK_STREAM, 0);
            if ( sock != -1 ) {
                struct sockaddr_un addr;
                memset(&addr, 0, sizeof(addr));
                addr.sun_family = AF_UNIX;
                strcpy(addr.sun_path, SOCKET_PATH.c_str( ));
                if ( connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0 ) {
                    for ( int i = 1; i < argc; i++ ) {
                        wxFileName filename(argv[i]);
                        filename.Normalize(wxPATH_NORM_LONG | wxPATH_NORM_DOTS | wxPATH_NORM_TILDE | wxPATH_NORM_ABSOLUTE);
                        wxString           cmd_full_filename = filename.GetFullPath( );
                        wxScopedCharBuffer buffer            = cmd_full_filename.ToUTF8( );
                        write(sock, buffer.data( ), buffer.length( ));
                        write(sock, "\n", 1);
                    }
                }
                else {
                    perror("client connect failed\n");
                }
                close(sock);
            }
            return false;
        }
    }
    else {
        SetupSignalHandlers( );
        DisplayServer::GetInstance( ).Start( );
    }

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
    DisplayServer::GetInstance( ).Stop( );
    return 0;
}
