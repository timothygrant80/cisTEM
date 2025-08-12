#include "DisplayServer.h"
#include <thread>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cstring>
#include <iostream>
#include <wx/app.h>
#include <wx/event.h>
#include <wx/log.h>
#include <wx/string.h>
#include <wx/thread.h>
#include <wx/window.h>
// #include "DisplayConfig.h"

wxDEFINE_EVENT(EVT_SERVER_OPEN_FILE, wxCommandEvent);

DisplayServer& DisplayServer::GetInstance( ) {
    static DisplayServer instance;
    return instance;
}

bool DisplayServer::Start( ) {
    unlink(SOCKET_PATH.c_str( ));
    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if ( socket_fd < 0 )
        return false;

    sockaddr_un addr{ };
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH.c_str( ), sizeof(addr.sun_path) - 1);
    unlink(SOCKET_PATH.c_str( ));

    if ( bind(socket_fd, (sockaddr*)&addr, sizeof(addr)) < 0 ) {
        perror("bind");
        return false;
    }

    if ( listen(socket_fd, 5) < 0 ) {
        perror("listen");
        return false;
    }
    std::thread([this]( ) { ServerLoop( ); }).detach( );
    return true;
}

void DisplayServer::ServerLoop( ) {
    while ( true ) {
        int client_fd = accept(socket_fd, nullptr, nullptr);
        if ( client_fd < 0 ) {
            perror("accept");
            continue;
        }

        std::string buffer;
        char        tmp[512];
        ssize_t     len;
        while ( (len = read(client_fd, tmp, sizeof(tmp))) > 0 ) {
            buffer.append(tmp, len);
        }
        close(client_fd);
        size_t start = 0;
        while ( true ) {
            size_t pos = buffer.find('\n', start);
            if ( pos == std::string::npos )
                break;

            std::string filename = buffer.substr(start, pos - start);
            start                = pos + 1;


            if ( ! filename.empty( ) ) {
                wxString       message = wxString::FromUTF8(filename.c_str( )).Trim( );
                wxCommandEvent evt(EVT_SERVER_OPEN_FILE, 1000);
                evt.SetString(message);
                wxQueueEvent(wxTheApp->GetTopWindow( )->GetEventHandler( ), new wxCommandEvent(evt));
            }
        }
        close(client_fd);
    }
}

void DisplayServer::Stop( ) {
    if ( socket_fd != -1 ) {
        close(socket_fd);
        unlink(SOCKET_PATH.c_str( ));
    }
}
