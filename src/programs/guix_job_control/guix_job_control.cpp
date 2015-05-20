#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include "wx/socket.h"

#include "../../core/core_headers.h"
#include "../../core/socket_codes.h"

#define SERVER_ID 100
#define GUI_SOCKET_ID 101

SETUP_SOCKET_CODES




class
JobControlApp : public wxAppConsole
{
	wxIPV4address gui_address;
	wxIPV4address my_address;
	wxIPV4address master_process_address;

	wxSocketClient *gui_socket;
	bool            gui_socket_is_busy;
	bool 			gui_socket_is_connected;
	bool            gui_panel_is_connected;

	unsigned char   job_code[SOCKET_CODE_SIZE];

	long gui_port;
	long my_port;
	long master_process_port;

	long total_number_of_slaves;
	long number_of_slaves_;

	void LaunchSlaves();
	bool ConnectToGui();
	void OnGuiSocketEvent(wxSocketEvent& event);


	public:
		virtual bool OnInit();



};

IMPLEMENT_APP(JobControlApp)


bool JobControlApp::OnInit()
{

	long counter;

	// set up the parameters for passing the gui address..

	static const wxCmdLineEntryDesc command_line_descriptor[] =
	{
			{ wxCMD_LINE_PARAM, "a", "address", "gui_address", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY },
			{ wxCMD_LINE_PARAM, "p", "port", "gui_port", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_OPTION_MANDATORY },
			{ wxCMD_LINE_PARAM, "j", "job_code", "job_code", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY },
			{ wxCMD_LINE_NONE }
	};


	wxCmdLineParser command_line_parser( command_line_descriptor, argc, argv);

	wxPrintf("\n");
	if (command_line_parser.Parse(true) != 0)
	{
		wxPrintf("\n\n");
		exit(0);
	}

	// get the address and port of the gui (should be command line options).


	if (gui_address.Hostname(command_line_parser.GetParam(0)) == false)
	{
		MyDebugPrint(" Error: Address (%s) - not recognized as an IP or hostname\n\n", command_line_parser.GetParam(0));
		exit(-1);
	};

	if (command_line_parser.GetParam(1).ToLong(&gui_port) == false)
	{
		MyDebugPrint(" Error: Port (%s) - not recognized as a port\n\n", command_line_parser.GetParam(1));
		exit(-1);
	}

	if (command_line_parser.GetParam(2).Len() != SOCKET_CODE_SIZE)
	{
		{
			MyDebugPrint(" Error: Code (%s) - is the incorrect length(%i instead of %i)\n\n", command_line_parser.GetParam(2), command_line_parser.GetParam(2).Len(), SOCKET_CODE_SIZE);
			exit(-1);
		}
	}


	// copy over job code.

	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		job_code[counter] = command_line_parser.GetParam(2).GetChar(counter);
	}


	gui_address.Service(gui_port);

	// Attempt to connect to the gui..

	gui_socket = new wxSocketClient();

	// Setup the event handler and subscribe to most events

	gui_socket->SetEventHandler(*this, GUI_SOCKET_ID);
	gui_socket->SetNotify(wxSOCKET_CONNECTION_FLAG |wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
	gui_socket->Notify(true);
	gui_socket_is_busy = false;
	gui_socket_is_connected = false;
	gui_panel_is_connected = false;

	this->Connect(GUI_SOCKET_ID, wxEVT_SOCKET, wxSocketEventHandler( JobControlApp::OnGuiSocketEvent) );
	MyDebugPrint("\n JOB CONTROL: Trying to connect to %s:%i (timeout = 10 sec) ...\n", gui_address.IPAddress(), gui_address.Service());

	gui_socket->Connect(gui_address, false);
	gui_socket->WaitOnConnect(10);

	if (gui_socket->IsConnected() == false)
	{
	   gui_socket->Close();
	   MyDebugPrint("Failed ! Unable to connect\n");
	   return false;
	}
	else
	{
		MyDebugPrint("Succeeded - Connection established!\n\n");
		gui_socket_is_connected = true;
		return true;
	}
}

void JobControlApp::OnGuiSocketEvent(wxSocketEvent& event)
{
	  wxString s = _("OnSocketEvent: ");
	  wxSocketBase *sock = event.GetSocket();

	  MyDebugAssertTrue(sock == gui_socket, "GUI Socket event from Non GUI socket??");

	  // First, print a message
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	    case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	    default             : s.Append(_("Unexpected event !\n")); break;
	  }

	  //m_text->AppendText(s);

	  MyDebugPrint(s);

	  // Now we process the event
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT:
	    {
	      // We disable input events, so that the test doesn't trigger
	      // wxSocketEvent again.
	      sock->SetNotify(wxSOCKET_LOST_FLAG);
	      sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

	      if (memcmp(socket_input_buffer, socket_please_identify, SOCKET_CODE_SIZE) == 0) // identification
	      {
	    	  // send the job identification to complete the connection
	    	  sock->WriteMsg(job_code, SOCKET_CODE_SIZE);
	      }
	      else
	      if (memcmp(socket_input_buffer, you_are_connected, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
	      {
	    	  MyDebugPrint("JOB CONTROL : Connected to Panel");
	    	  gui_panel_is_connected = true;

	    	  // ask the panel to send job details..

	    	  sock->WriteMsg(send_job_details, SOCKET_CODE_SIZE);



	      }

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

	        wxPrintf("Socket Disconnected!!\n");
	        sock->Destroy();
	        ExitMainLoop();

	        break;
	    }
	    default: ;
	  }
}


