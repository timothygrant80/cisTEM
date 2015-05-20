#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include "wx/socket.h"

#include "../../core/core_headers.h"
#include "../../core/socket_codes.h"

#define SERVER_ID 100
#define SOCKET_ID 101

SETUP_SOCKET_CODES

class
MyApp : public wxAppConsole
{

	public:
		virtual bool OnInit();

		wxSocketServer *m_server;


		bool            m_busy;
		bool 			is_connected;

		int             m_numClients;
		int             m_number_watching;

		wxSocketBase *FirstSocket;
		wxSocketBase *SecondSocket;
		wxSocketBase *ThirdSocket;

		bool firstsocketisconnected;
		bool secondsocketisconnected;
		bool thirdsocketisconnected;

		void OnServerEvent(wxSocketEvent& event);
		void OnSocketEvent(wxSocketEvent& event);


};


IMPLEMENT_APP(MyApp)


bool MyApp::OnInit()
{

	wxIPV4address addr;
	addr.Service(3000);

	m_server = new wxSocketServer(addr);

	// We use Ok() here to see if the server is really listening
	if (! m_server->Ok())
	{
		wxPrintf("Could not listen at the specified port !\n\n");
		return false;
	}
	else
	{
		wxPrintf("Server listening.\n\n");
	}

	// Setup the event handler and subscribe to connection events

	m_server->SetEventHandler(*this, SERVER_ID);
	m_server->SetNotify(wxSOCKET_CONNECTION_FLAG);
	m_server->Notify(true);

	this->Connect(SERVER_ID, wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnServerEvent) );
	this->Connect(SOCKET_ID, wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnSocketEvent) );

	m_busy = false;
	m_numClients = 0;
	m_number_watching = 0;

	return true;
}


void MyApp::OnServerEvent(wxSocketEvent& event)
{
	  wxString s = _("OnServerEvent: ");
	  wxSocketBase *sock = NULL;

	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_CONNECTION : s.Append(_("wxSOCKET_CONNECTION\n")); break;
	    default                  : s.Append(_("Unexpected event !\n")); break;
	  }

	  wxPrintf(s);

	  if (m_numClients < 3)
	  {
		  // we have space for another

	      // Accept new connection if there is one in the pending
	      // connections queue, else exit. We use Accept(false) for
	      // non-blocking accept (although if we got here, there
	      // should ALWAYS be a pending connection).

	      sock = m_server->Accept(false);
	      sock->SetFlags(wxSOCKET_WAITALL);//|wxSOCKET_BLOCK);

	      // request identification..
	    	  wxPrintf(" Requesting identification...\n");

   	  	      sock->WriteMsg(socket_please_identify, SOCKET_CODE_SIZE);
   	  	      wxPrintf(" Waiting for reply...\n");
  	  	      sock->WaitForRead(5);


   	  	      if (sock->IsData() == true)
   	  	      {
   	  	    	  	   sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);


	    	  	      if (memcmp(socket_input_buffer, socket_identification, sizeof(socket_input_buffer)) != 0)
	    	  	      {
	    	  	    	 wxPrintf(" Incorrect MSG - Exiting\n\n");
	    	  	       	 // incorrect identification - close the connection..
	    	  	       	 sock->Destroy();
	    	  	       	 sock = NULL;
	    	  	      }
	    	  	      else
	    	  	      {	  wxPrintf(" Correct Identification\n");

	    	  	    	  sock->SetEventHandler(*this, SOCKET_ID);
	    	  	    	  sock->SetNotify(wxSOCKET_INPUT_FLAG | wxSOCKET_LOST_FLAG);
	    	  	    	  sock->Notify(true);
	    	  	      }
   	  	      }
   	  	      else
   	  	      {
	    	   	   wxPrintf(" ...Read Timeout \n\n");
	    	     // time out - close the connection
	    	   	   sock->Destroy();
	    	   	   sock = NULL;
	    	  }


	  }


/*
	  if (sock)
	  {
	   	    // fill the first free connection..

	   	  if (firstsocketisconnected == false)
	   	  {
	   		  FirstSocket = sock;
	   		  wxPrintf("\n   **   New Connection Accepted  on Socket #1 **\n");
	   		  m_numClients++;
	   		  firstsocketisconnected = true;
	   	  }
	   	  else
	   	  if (secondsocketisconnected == false)
	   	  {
	   		  SecondSocket = sock;
	   		  wxPrintf("\n   **   New Connection Accepted  on Socket #2 **\n");
	   		  m_numClients++;
	   		  secondsocketisconnected = true;
	   	  }
	   	  else
	   	  if (thirdsocketisconnected == false)
	   	  {
	   		  ThirdSocket = sock;
	   		  wxPrintf("\n   **   New Connection Accepted  on Socket #3 **\n");
	   		  m_numClients++;
	   		  thirdsocketisconnected = true;
	   	  }
	   	  else
	   	  {
	   		wxPrintf("\n   **   Error: Can't find a free socket, though i think there should be one?   **\n");

	   	  }

	  }
	  else
	  {
		  wxPrintf("\n   **   Error: couldn't accept a new connection, maybe I have 3 connected already?   **\n");
	    return;
	  }

*/


	  //UpdateStatusBar();
}

void MyApp::OnSocketEvent(wxSocketEvent& event)
{
  wxString s = _("OnSocketEvent: ");
  wxSocketBase *sock = event.GetSocket();

  // First, print a message
  switch(event.GetSocketEvent())
  {
    case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
    case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
    default             : s.Append(_("Unexpected event !\n")); break;
  }

  //m_text->AppendText(s);

  wxPrintf(s);

  // Now we process the event
  switch(event.GetSocketEvent())
  {
    case wxSOCKET_INPUT:
    {
      // We disable input events, so that the test doesn't trigger
      // wxSocketEvent again.
      sock->SetNotify(wxSOCKET_LOST_FLAG);
/*
      // Which test are we going to run?

      switch (c)
      {
        case 0xBE:  wxPrintf("0xBE\n\n"); break;//Test1(sock); break;
        case 0xCE:  wxPrintf("0xCE\n\n"); break;//Test2(sock); break;
        case 0xDE:  wxPrintf("0xDE\n\n");break;//Test3(sock); break;
        default:
          wxPrintf("Unknown test id received from client\n\n");
      }

*/
      // Enable input events again.

      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
      break;
    }

    case wxSOCKET_LOST:
    {
        //m_numClients--;

        //wxPrintf("Socket Disconnected!!\n");
        sock->Destroy();

        break;
    }
    default: ;
  }
}

