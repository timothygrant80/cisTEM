#ifndef _src_programs_cisTEM_display_IpcServer_h_
#define _src_programs_cisTEM_display_IpcServer_h_

#pragma once
#include <string>
#include <signal.h>
#include <unistd.h>
#include <wx/wx.h>
#include <wx/event.h>
#include <cstdlib>
#include <pwd.h>

wxDECLARE_EVENT(EVT_SERVER_OPEN_FILE, wxCommandEvent);

class DisplayServer {
  public:
    static DisplayServer& GetInstance( );

    bool Start( );
    void Stop( );

  private:
    int  socket_fd = -1;
    void ServerLoop( );
};

// These two functions will ensure that no matter how the display program
// is closed, the existing socket in /tmp will be deleted

inline const std::string GetUserSocketPath( ) {
    const char* username = getenv("USER");
    if ( ! username ) {
        struct passwd* pw = getpwuid(getuid( ));
        if ( pw )
            username = pw->pw_name;
        else
            username = "unknown";
    }
    return username;
};

inline std::string SOCKET_PATH = ("/tmp/cisTEM_display_ipc_socket_" + GetUserSocketPath( )).c_str( );

inline void CleanupSocketFile(int sig_num) {
    unlink(SOCKET_PATH.c_str( ));
    signal(sig_num, SIG_DFL);
    raise(sig_num);
};

inline void SetupSignalHandlers( ) {
    signal(SIGINT, CleanupSocketFile);
    signal(SIGTERM, CleanupSocketFile);
    signal(SIGQUIT, CleanupSocketFile);
    signal(SIGTSTP, CleanupSocketFile);
};

#endif