#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include "wx/socket.h"

#include "../../core/core_headers.h"
#include "../../core/socket_codes.h"

#define SERVER_ID 100
#define GUI_SOCKET_ID 101
#define MASTER_SOCKET_ID 102

SETUP_SOCKET_CODES




class
JobControlApp : public wxAppConsole
{
	bool have_assigned_master;
	wxIPV4address gui_address;
	wxIPV4address my_address;
	wxIPV4address master_process_address;

	wxSocketServer *socket_server;
	wxSocketClient *gui_socket;
	wxSocketBase   *master_socket;
	bool           gui_socket_is_busy;
	bool 		   gui_socket_is_connected;
	bool           gui_panel_is_connected;

	unsigned char   job_code[SOCKET_CODE_SIZE];

	long gui_port;
	long master_process_port;

	short int my_port;

	wxString my_port_string;
	wxString my_ip_address;
	wxString master_ip_address;
	wxString master_port;

	long number_of_slaves_already_connected;

	JobPackage my_job_package;

	void SetupServer();
	void LaunchRemoteJob();
	bool ConnectToGui();
	void OnGuiSocketEvent(wxSocketEvent& event);
	void OnMasterSocketEvent(wxSocketEvent& event);
	void OnServerEvent(wxSocketEvent& event);




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


	Bind(wxEVT_SOCKET,wxSocketEventHandler( JobControlApp::OnGuiSocketEvent), this,  GUI_SOCKET_ID );
	MyDebugPrint("\n JOB CONTROL: Trying to connect to %s:%i (timeout = 10 sec) ...\n", gui_address.IPAddress(), gui_address.Service());

	gui_socket->Connect(gui_address, false);
	gui_socket->WaitOnConnect(10);

	if (gui_socket->IsConnected() == false)
	{
	   gui_socket->Close();
	   MyDebugPrint(" JOB CONTROL : Failed ! Unable to connect\n");
	   return false;
	}

	MyDebugPrint(" JOB CONTROL: Succeeded - Connection established!\n\n");
	gui_socket_is_connected = true;


	// setup a server... so that slaves can later contact back.

	SetupServer();

	return true;

}

void JobControlApp::LaunchRemoteJob()
{
	long counter;
	wxIPV4address address;

	//MyDebugPrint("Launching Slaves");

	// for n processes (specified in the job package) we need to launch the specified command, along with our
	// IP address, port and job code..

	wxString execution_command;
	execution_command = my_job_package.command_to_run + " " + my_ip_address + " " + my_port_string + " ";


	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		execution_command += job_code[counter];
	}

	MyDebugPrint("Launching \"%s\" %i times\n", execution_command, my_job_package.number_of_processes);

	for (counter = 0; counter < my_job_package.number_of_processes; counter++)
	{
		wxExecute(execution_command);
	}

	// now we wait for the connections - this is taken care of as server events..

}

void JobControlApp::SetupServer()
{
	wxIPV4address my_address;
	wxIPV4address buffer_address;

	MyDebugPrint("Setting up Server...");

	for (short int current_port = START_PORT; current_port <= END_PORT; current_port++)
	{

		if (current_port == END_PORT)
		{
			wxPrintf("JOB CONTROL : Could not find a valid port !\n\n");
			abort();
		}

		my_port = current_port;
		my_address.Service(my_port);

		socket_server = new wxSocketServer(my_address);

		if (	socket_server->Ok())
		{
			  // setup events for the socket server..

		   	  socket_server->SetEventHandler(*this, SERVER_ID);
		   	  socket_server->SetNotify(wxSOCKET_CONNECTION_FLAG);
		  	  socket_server->Notify(true);

		  	  Bind(wxEVT_SOCKET,wxSocketEventHandler( JobControlApp::OnServerEvent), this,  SERVER_ID);

			  buffer_address.Hostname(wxGetFullHostName()); // hopefully get my ip
			  my_ip_address = buffer_address.IPAddress();
			  my_port_string = wxString::Format("%hi", my_port);


			  break;
		}
		else socket_server->Destroy();

	}

}

void JobControlApp::OnServerEvent(wxSocketEvent& event)
{
	  SETUP_SOCKET_CODES

	  wxString s = _("OnServerEvent: ");
	  wxSocketBase *sock = NULL;

	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_CONNECTION : s.Append(_("wxSOCKET_CONNECTION\n")); break;
	    default                  : s.Append(_("Unexpected event !\n")); break;
	  }

	  //MyDebugPrint(s);

	  // Accept new connection if there is one in the pending
	  // connections queue, else exit. We use Accept(false) for
	  // non-blocking accept (although if we got here, there
	  // should ALWAYS be a pending connection).

	  sock = socket_server->Accept(false);
	  sock->SetFlags(wxSOCKET_WAITALL);//|wxSOCKET_BLOCK);

	  // request identification..
	  //MyDebugPrint(" Requesting identification...");
	  sock->WriteMsg(socket_please_identify, SOCKET_CODE_SIZE);
	  //MyDebugPrint(" Waiting for reply...");
	  sock->WaitForRead(5);

	  if (sock->IsData() == true)
	  {
    	  sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

    	  // does this correspond to our job code?

	    if ((memcmp(socket_input_buffer, job_code, SOCKET_CODE_SIZE) != 0) )
	  	{
	  	  	MyDebugPrint(" Unknown JOB ID - Closing Connection\n");

	  	  	// incorrect identification - close the connection..
		    sock->Destroy();
		    sock = NULL;
		}
		else
		{
			// one of the slaves has connected to us.  If it is the first one then
			// we need to make it the master, tell it to start a socket server
			// and send us the address so we can pass it on to all future slaves.

			// If we have already assigned the master, then we just need to send it
			// the masters address.

			if (have_assigned_master == false)  // we don't have a master, so assign it
			{
				sock->WriteMsg(socket_you_are_the_master, SOCKET_CODE_SIZE);

				// read the ip address and port..

				sock->WaitForRead(20);
				if (sock->IsData() == true)
				{
					master_ip_address = ReceivewxStringFromSocket(sock);
				}
				else
				{
					MyDebugPrintWithDetails(" JOB CONTROL: Read Timeout waiting for ip address!!");
					abort();
				}

				sock->WaitForRead(10);
				if (sock->IsData() == true)
				{
					master_port = ReceivewxStringFromSocket(sock);
				}
				else
				{
					MyDebugPrintWithDetails(" JOB CONTROL: Read Timeout waiting for port!!");
					abort();
				}

				master_socket = sock;
				have_assigned_master = true;

				// setup events on the master..

				Bind(wxEVT_SOCKET, wxSocketEventHandler( JobControlApp::OnMasterSocketEvent), this,  MASTER_SOCKET_ID);

				sock->SetEventHandler(*this, MASTER_SOCKET_ID);
				sock->SetNotify(wxSOCKET_CONNECTION_FLAG |wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
				sock->Notify(true);


			}
			else  // we have a master, tell this slave who it's master is.
			{
				sock->WriteMsg(socket_you_are_a_slave, SOCKET_CODE_SIZE);
				SendwxStringToSocket(&master_ip_address, sock);
				SendwxStringToSocket(&master_port, sock);

				// that should be the end of our interactions with the slave
				// it should disconnect itself, we won't even bother
				// setting up events for it..
			}
		}
	}
	else
	{
		MyDebugPrint(" JOB CONTROL : ...Read Timeout waiting for job ID \n\n");
	 	// time out - close the connection
		sock->Destroy();
		sock = NULL;
	}

}

void JobControlApp::OnMasterSocketEvent(wxSocketEvent& event)
{
	  wxString s = _("JOB CONTROL : OnSocketEvent: ");
	  wxSocketBase *sock = event.GetSocket();

	  MyDebugAssertTrue(sock == master_socket, "Master Socket event from Non Master socket??");

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

		  if (memcmp(socket_input_buffer, socket_send_job_details, SOCKET_CODE_SIZE) == 0)
		  {
			  // send the job details through...

			  MyDebugPrint("JOB CONTROL : Sending Job details to master slave");
			  my_job_package.SendJobPackage(sock);

		  }


	      // Enable input events again.

	      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	      break;
	    }

	    case wxSOCKET_LOST:
	    {

	        wxPrintf("JOB CONTROL : Master Socket Disconnected!!\n");
	        sock->Destroy();
	        ExitMainLoop();

	        break;
	    }
	    default: ;
	  }


}


void JobControlApp::OnGuiSocketEvent(wxSocketEvent& event)
{
	  wxString s = _("JOB CONTROL : OnSocketEvent: ");
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
	      if (memcmp(socket_input_buffer, socket_you_are_connected, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
	      {
	    	  MyDebugPrint("JOB CONTROL : Connected to Panel");
	    	  gui_panel_is_connected = true;

	    	  // ask the panel to send job details..

	    	  MyDebugPrint("JOB CONTROL : Asking for job details");
	    	  sock->WriteMsg(socket_send_job_details, SOCKET_CODE_SIZE);

	      }
	      else
		  if (memcmp(socket_input_buffer, socket_ready_to_send_job_package, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
		  {
			  // receive the job details..

			 // MyDebugPrint("JOB CONTROL : Receiving Job Package");
			  my_job_package.ReceiveJobPackage(sock);
			  MyDebugPrint("JOB CONTROL : Job Package Received, launching slaves");

			  //MyDebugPrint("Filename = %s", my_job_package.jobs[0].arguments[0].string_argument[0]);

			  // Now we have the job, we can launch the slaves..

			  LaunchRemoteJob();


		  }


	      // Enable input events again.

	      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	      break;
	    }

	    case wxSOCKET_LOST:
	    {

	        wxPrintf("JOB CONTROL : Socket Disconnected!!\n");
	        sock->Destroy();
	        ExitMainLoop();

	        break;
	    }
	    default: ;
	  }
}


