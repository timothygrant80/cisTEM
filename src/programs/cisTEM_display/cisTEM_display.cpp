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
    bool                     new_instance;
    wxArrayString            files_to_open;
    // MyServer*                m_server;
};

static const wxCmdLineEntryDesc display_cmd_line_desc[] = {
        {wxCMD_LINE_SWITCH, "h", "help", "displays help on the command line parameters", wxCMD_LINE_VAL_NONE, wxCMD_LINE_OPTION_HELP},
        {wxCMD_LINE_SWITCH, "n", "new-instance", "force starting a new instance, even if another is already running", wxCMD_LINE_VAL_NONE, 0},
        {wxCMD_LINE_PARAM, nullptr, nullptr, "file(s) to open", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE},
        {wxCMD_LINE_NONE}};

IMPLEMENT_APP(DisplayApp)

DisplayFrame*
        display_frame;

DisplayApp::~DisplayApp( ) {
    DisplayServer::GetInstance( ).Stop( );
}

bool DisplayApp::OnInit( ) {
    if ( ! wxApp::OnInit( ) )
        return false;

    // new_instance = m_parser->Found(wxT("n"));

    const wxString name = wxString::Format("cisTEM_Display-%s", wxGetUserId( ));
    m_checker           = new wxSingleInstanceChecker(name);
    if ( m_checker->IsAnotherRunning( ) && ! new_instance ) {
        if ( argc > 1 ) {
            int sock = socket(AF_UNIX, SOCK_STREAM, 0);
            if ( sock != -1 ) {
                struct sockaddr_un addr;
                memset(&addr, 0, sizeof(addr));
                addr.sun_family = AF_UNIX;
                strcpy(addr.sun_path, SOCKET_PATH.c_str( ));
                if ( connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0 ) {
                    for ( size_t i = 0; i < files_to_open.GetCount( ); i++ ) {
                        wxScopedCharBuffer buffer = files_to_open[i].ToUTF8( );
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
    else if ( m_checker->IsAnotherRunning( ) && new_instance ) {
        // Now make sure we proceed with opening a new application instance
        wxPrintf("Proceding with opening a new instance...\n");
    }
    else {
        SetupSignalHandlers( );
        DisplayServer::GetInstance( ).Start( );
    }

    wxInitAllImageHandlers( );

    display_frame = new DisplayFrame(NULL, wxID_ANY, "cisTEM Display", wxPoint(-1, -1), wxSize(-1, -1), wxDEFAULT_FRAME_STYLE);
    wxString cmd_full_filename;
    wxString cmd_filename;

    for ( int i = 0; i < files_to_open.GetCount( ); i++ ) {
        display_frame->cisTEMDisplayPanel->OpenFile(files_to_open[i], files_to_open[i]);
    }

    display_frame->Layout( );
    display_frame->Show(true);
    return true;
}

void DisplayApp::OnInitCmdLine(wxCmdLineParser& parser) {
    parser.SetDesc(display_cmd_line_desc);
    parser.SetSwitchChars(wxT("-"));
}

bool DisplayApp::OnCmdLineParsed(wxCmdLineParser& parser) {
    new_instance = parser.Found(wxT("n"));

    for ( size_t arg_counter = 0; arg_counter < parser.GetParamCount( ); arg_counter++ ) {
        wxString   cmd_filename = parser.GetParam(arg_counter);
        wxFileName filename(cmd_filename);
        filename.Normalize(wxPATH_NORM_LONG | wxPATH_NORM_DOTS | wxPATH_NORM_TILDE | wxPATH_NORM_ABSOLUTE);
        wxString cmd_full_filename = filename.GetFullPath( );
        files_to_open.Add(cmd_full_filename);
    }
    return true;
}

int DisplayApp::OnExit( ) {
    DisplayServer::GetInstance( ).Stop( );
    return 0;
}
