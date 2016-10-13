#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include "wx/socket.h"

#include "../../core/core_headers.h"
#include "../../core/socket_codes.h"

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDINFO, wxThreadEvent);

#define SERVER_ID 100
#define GUI_SOCKET_ID 101
#define MASTER_SOCKET_ID 102

SETUP_SOCKET_CODES

class
JobControlApp : public wxAppConsole
{
	wxTimer *connection_timer;
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

	bool ConnectToGui();
	void OnGuiSocketEvent(wxSocketEvent& event);
	void OnMasterSocketEvent(wxSocketEvent& event);
	void OnServerEvent(wxSocketEvent& event);
	void OnConnectionTimer(wxTimerEvent& event);
	void CheckForConnections();
	void SendError(wxString error_to_send);
	void SendInfo(wxString info_to_send);

	void SendJobFinished(int job_number);
	void SendJobResult(JobResult *result_to_send);

	void SendAllJobsFinished();
	void SendNumberofConnections();

	void OnThreadLaunchJob(wxThreadEvent &event);
	void OnThreadSendInfo(wxThreadEvent &event);


	public:
		virtual bool OnInit();



};

class LaunchJobThread : public wxThread
{
	public:
    	LaunchJobThread(JobControlApp *handler, RunProfile wanted_run_profile, wxString wanted_ip_address, wxString wanted_port, const unsigned char *wanted_job_code) : wxThread(wxTHREAD_DETACHED)
		{
    		main_thread_pointer = handler;
    		current_run_profile = wanted_run_profile;
    		ip_address = wanted_ip_address;
    		port_number = wanted_port;

    		for (int counter = 0; counter <= SOCKET_CODE_SIZE; counter++)
    		{
    			job_code[counter] = wanted_job_code[counter];
    		}

		}
    	//~LaunchJobThread();
	protected:

    	JobControlApp *main_thread_pointer;
    	RunProfile current_run_profile;
    	wxString ip_address;
    	wxString port_number;
    	unsigned char job_code[SOCKET_CODE_SIZE];

		void LaunchRemoteJob();
	 	void QueueInfo(wxString info_to_queue);
    	virtual ExitCode Entry();
};


wxThread::ExitCode LaunchJobThread::Entry()
{
	LaunchRemoteJob();
	return (wxThread::ExitCode)0;     // success
}


IMPLEMENT_APP(JobControlApp)


bool JobControlApp::OnInit()
{

	wxPrintf("Running...\n");
	long counter;
	wxIPV4address my_address;

	// set up the parameters for passing the gui address..

	static const wxCmdLineEntryDesc command_line_descriptor[] =
	{
			{ wxCMD_LINE_PARAM, "a", "address", "gui_address", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY },
			{ wxCMD_LINE_PARAM, "p", "port", "gui_port", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_OPTION_MANDATORY },
			{ wxCMD_LINE_PARAM, "j", "job_code", "job_code", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY },
			{ wxCMD_LINE_NONE }
	};


	wxCmdLineParser command_line_parser( command_line_descriptor, argc, argv);

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

	// initialise sockets

	wxSocketBase::Initialize();

	// start the server..

	SetupServer();

	// Attempt to connect to the gui..

	gui_address.Service(gui_port);
	gui_socket = new wxSocketClient();

	// Setup the event handler and subscribe to most events

	gui_socket_is_busy = false;
	gui_socket_is_connected = false;
	gui_panel_is_connected = false;


	wxPrintf("\n JOB CONTROL: Trying to connect to %s:%i (timeout = 30 sec) ...\n", gui_address.IPAddress(), gui_address.Service());

	gui_socket->Connect(gui_address, false);
	gui_socket->WaitOnConnect(30);

	if (gui_socket->IsConnected() == false)
	{
	   gui_socket->Close();
	   MyDebugPrint(" JOB CONTROL : Failed ! Unable to connect\n");
	   return false;
	}

	gui_socket->SetFlags(wxSOCKET_BLOCK | wxSOCKET_WAITALL);
	wxPrintf(" JOB CONTROL: Succeeded - Connection established!\n\n");
	gui_socket_is_connected = true;

	// we can use this socket to get our ip_address

	my_ip_address = ReturnIPAddressFromSocket(gui_socket);
	number_of_slaves_already_connected = 0;

	// subscribe to gui events..

	Bind(wxEVT_SOCKET,wxSocketEventHandler( JobControlApp::OnGuiSocketEvent), this,  GUI_SOCKET_ID );
	gui_socket->SetEventHandler(*this, GUI_SOCKET_ID);
	gui_socket->SetNotify(wxSOCKET_CONNECTION_FLAG |wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
	gui_socket->Notify(true);


	// Job launching event..

	Bind(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB, &JobControlApp::OnThreadLaunchJob, this);
	Bind(wxEVT_COMMAND_MYTHREAD_SENDINFO, &JobControlApp::OnThreadSendInfo, this);

	// Setup the connection timer, to check for connections periodically in case the events get missed..

	Bind(wxEVT_TIMER, wxTimerEventHandler( JobControlApp::OnConnectionTimer ), this);
	connection_timer = new wxTimer(this, 0);
	connection_timer->Start(5000);

	return true;

}

void LaunchJobThread::LaunchRemoteJob()
{
	long counter;
	long command_counter;
	long process_counter;

	wxIPV4address address;

	// for n processes (specified in the job package) we need to launch the specified command, along with our
	// IP address, port and job code..

	wxString executable;
	wxString execution_command;


	if(current_run_profile.controller_address == "")
	{
		executable = current_run_profile.executable_name + " " + ip_address + " " + port_number + " ";
	}
	else
	{
		executable = current_run_profile.executable_name + " " + current_run_profile.controller_address + " " + port_number + " ";
	}

	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		executable += job_code[counter];
	}


	wxMilliSleep(2000);

	for (command_counter = 0; command_counter <  current_run_profile.number_of_run_commands; command_counter++)
	{

		execution_command =  current_run_profile.run_commands[command_counter].command_to_run;
		execution_command.Replace("$command", executable);

		//execution_command += "&";

		QueueInfo(wxString::Format("Job Control : Executing '%s' %i times.", execution_command, current_run_profile.run_commands[command_counter].number_of_copies));

		for (process_counter = 0; process_counter < current_run_profile.run_commands[command_counter].number_of_copies; process_counter++)
		{

			wxMilliSleep( current_run_profile.run_commands[command_counter].delay_time_in_ms);

			wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB);
			test_event->SetString(execution_command);

			wxQueueEvent(main_thread_pointer, test_event);
			//wxExecute(execution_command);
			//system(execution_command.ToUTF8().data());
		}

	}

	// now we wait for the connections - this is taken care of as server events..

}

void  LaunchJobThread::QueueInfo(wxString info_to_queue)
{
	wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SENDINFO);
	test_event->SetString(info_to_queue);

	wxQueueEvent(main_thread_pointer, test_event);
}


void JobControlApp::OnThreadLaunchJob(wxThreadEvent &event)
{
	if (wxExecute(event.GetString()) == -1)
	{
		SendError(wxString::Format("Error: Failed to launch (%s)", event.GetString()));
	}
}

void JobControlApp::OnThreadSendInfo(wxThreadEvent& my_event)
{
	SendInfo(my_event.GetString());
}

void JobControlApp::SendError(wxString error_to_send)
{
//	SETUP_SOCKET_CODES

	// send the error message flag

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(gui_socket, socket_i_have_an_error, SOCKET_CODE_SIZE);
	SendwxStringToSocket(&error_to_send, gui_socket);

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
}

void JobControlApp::SendInfo(wxString info_to_send)
{
//	SETUP_SOCKET_CODES

	// send the error message flag

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(gui_socket, socket_i_have_info, SOCKET_CODE_SIZE);
	SendwxStringToSocket(&info_to_send, gui_socket);

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
}


void JobControlApp::SetupServer()
{
	wxIPV4address my_address;
	wxIPV4address buffer_address;

	//MyDebugPrint("Setting up Server...");

	for (short int current_port = START_PORT; current_port <= END_PORT; current_port++)
	{

		if (current_port == END_PORT)
		{
			wxPrintf("JOB CONTROL : Could not find a valid port !\n\n");
			ExitMainLoop();
			return;
		}

		my_port = current_port;
		my_address.Service(my_port);


		socket_server = new wxSocketServer(my_address, wxSOCKET_BLOCK);

		if (socket_server->IsOk() == true)
		{
			Bind(wxEVT_SOCKET,wxSocketEventHandler( JobControlApp::OnServerEvent), this,  SERVER_ID);

			socket_server->SetEventHandler(*this, SERVER_ID);
			socket_server->SetNotify(wxSOCKET_CONNECTION_FLAG);
			socket_server->Notify(true);

			my_port_string = wxString::Format("%hi", my_port);
			break;
		}
		else
		{
			socket_server->Destroy();
		}

	}

}

void JobControlApp::OnServerEvent(wxSocketEvent& event)
{
	  CheckForConnections();
}

void JobControlApp::CheckForConnections()
{
	  wxSocketBase *sock = NULL;

	  while (1==1) // sometimes, multiple connections only seem to generate one event.. so we keep checking until there are no more connections..
	  {

		  sock = socket_server->Accept(false);

		  if (sock == NULL) break;

		  sock->SetFlags(wxSOCKET_BLOCK | wxSOCKET_WAITALL);

		  // request identification..
		  WriteToSocket(sock, socket_please_identify, SOCKET_CODE_SIZE);
		  sock->WaitForRead(5);

		  if (sock->IsData() == true)
		  {
			  ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);

			  // does this correspond to our job code?

			  if ((memcmp(socket_input_buffer, job_code, SOCKET_CODE_SIZE) != 0) )
			  {
				  SendError("JOB CONTROL : Unknown JOB ID - Closing Connection");

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

					  master_socket = sock;
					  have_assigned_master = true;

					  WriteToSocket(sock, socket_you_are_the_master, SOCKET_CODE_SIZE);

					  // read the ip address and port..

					  sock->WaitForRead(30);
					  if (sock->IsData() == true)
					  {
						  master_ip_address = ReceivewxStringFromSocket(sock);
					  }
					  else
					  {
						  SendError("JOB CONTROL: Read Timeout waiting for ip address");
						  abort();
					  }

					  sock->WaitForRead(30);
					  if (sock->IsData() == true)
					  {
						  master_port = ReceivewxStringFromSocket(sock);
					  }
					  else
					  {
						  SendError("JOB CONTROL: Read Timeout waiting for port");
						  abort();
					  }


					  // setup events on the master..

					  Bind(wxEVT_SOCKET, wxSocketEventHandler( JobControlApp::OnMasterSocketEvent), this,  MASTER_SOCKET_ID);

					  sock->SetEventHandler(*this, MASTER_SOCKET_ID);
					  sock->SetNotify(wxSOCKET_CONNECTION_FLAG |wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
					  sock->Notify(true);

					  number_of_slaves_already_connected++;
					  SendNumberofConnections();

				  }
				  else  // we have a master, tell this slave who it's master is.
				  {
					  WriteToSocket(sock, socket_you_are_a_slave, SOCKET_CODE_SIZE);
					  SendwxStringToSocket(&master_ip_address, sock);
					  SendwxStringToSocket(&master_port, sock);

					  // that should be the end of our interactions with the slave
					  // it should disconnect itself, we won't even bother
					  // setting up events for it..

					  number_of_slaves_already_connected++;
					  SendNumberofConnections();

				  }
			  }
		  }
		  else
		  {
			  SendError("JOB CONTROL : Read Timeout waiting for job ID");
			  // time out - close the connection
			  sock->Destroy();
			  sock = NULL;
		  }
	  }


}

void JobControlApp::OnConnectionTimer(wxTimerEvent& event)
{
	//wxPrintf("Timer Fired\n");
	CheckForConnections();
}

void JobControlApp::OnMasterSocketEvent(wxSocketEvent& event)
{
	//  wxString s = _("JOB CONTROL : OnSocketEvent: ");
	  wxSocketBase *sock = event.GetSocket();

	  MyDebugAssertTrue(sock == master_socket, "Master Socket event from Non Master socket??");

	  // First, print a message
//	  switch(event.GetSocketEvent())
//	  {
//	    case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
//	    case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
//	    default             : s.Append(_("Unexpected event !\n")); break;
//	  }

	  //m_text->AppendText(s);

	  //MyDebugPrint(s);

	  // Now we process the event
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT:
	    {
	      // We disable input events, so that the test doesn't trigger
	      // wxSocketEvent again.

	      sock->SetNotify(wxSOCKET_LOST_FLAG);
	      ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);

		  if (memcmp(socket_input_buffer, socket_send_job_details, SOCKET_CODE_SIZE) == 0)
		  {
			  // send the job details through...

			  my_job_package.SendJobPackage(sock);

		  }
		  else
		  if (memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0) // identification
		  {
			 wxString error_message;
			 error_message = ReceivewxStringFromSocket(sock);

			 // send the error message up the chain..

			 SendError(error_message);
		 }
		 else
		 if (memcmp(socket_input_buffer, socket_i_have_info, SOCKET_CODE_SIZE) == 0) // identification
		 {
			 wxString info_message;
			 info_message = ReceivewxStringFromSocket(sock);

			 // send the error message up the chain..

			 SendInfo(info_message);
		 }
		 else
		 if (memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0) // identification
		 {
			 // which job is finished and how big is the result..

			 JobResult temp_job;
			 temp_job.ReceiveFromSocket(sock);



			 if (temp_job.result_size > 0)
			 {
				 SendJobResult(&temp_job);
			 }

		 }
		 else
		 if (memcmp(socket_input_buffer, socket_job_finished, SOCKET_CODE_SIZE) == 0) // identification
		 {
			 // which job is finished?

			 int finished_job;
			 ReadFromSocket(sock, &finished_job, 4);

			 // send the info to the gui

			 SendJobFinished(finished_job);
		 }
		 else
		 if (memcmp(socket_input_buffer, socket_all_jobs_finished, SOCKET_CODE_SIZE) == 0) // identification
		 {
			 SendAllJobsFinished();
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
	        abort();

	        break;
	    }
	    default: ;
	  }


}

void JobControlApp::SendJobFinished(int job_number)
{
	//SETUP_SOCKET_CODES

	// get the next job..
	gui_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(gui_socket, socket_job_finished, SOCKET_CODE_SIZE);
	WriteToSocket(gui_socket, &job_number, 4);

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
}

void JobControlApp::SendJobResult(JobResult *job_to_send)
{
	//SETUP_SOCKET_CODES

	// sendjobresultcode
	gui_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(gui_socket, socket_job_result, SOCKET_CODE_SIZE);
	job_to_send->SendToSocket(gui_socket);

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
}


void JobControlApp::SendAllJobsFinished()
{
	//	SETUP_SOCKET_CODES

	// get the next job..
	gui_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(gui_socket, socket_all_jobs_finished, SOCKET_CODE_SIZE);

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}

void JobControlApp::SendNumberofConnections()
{
	//SETUP_SOCKET_CODES

	// get the next job..
	gui_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(gui_socket, socket_number_of_connections, SOCKET_CODE_SIZE);
	WriteToSocket(gui_socket, &number_of_slaves_already_connected, 4);

	gui_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	if (number_of_slaves_already_connected == my_job_package.my_profile.ReturnTotalJobs())
	{
		connection_timer->Stop();
		Unbind(wxEVT_TIMER, wxTimerEventHandler( JobControlApp::OnConnectionTimer ), this);
		delete connection_timer;
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

	  //MyDebugPrint(s);

	  // Now we process the event
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT:
	    {
	      // We disable input events, so that the test doesn't trigger
	      // wxSocketEvent again.
	      sock->SetNotify(wxSOCKET_LOST_FLAG);
	      ReadFromSocket(sock, socket_input_buffer, SOCKET_CODE_SIZE);


	      if (memcmp(socket_input_buffer, socket_please_identify, SOCKET_CODE_SIZE) == 0) // identification
	      {
	    	  // send the job identification to complete the connection
	    	  WriteToSocket(sock, job_code, SOCKET_CODE_SIZE);
	      }
	      else
	      if (memcmp(socket_input_buffer, socket_you_are_connected, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
	      {
	    	  gui_panel_is_connected = true;

	    	  // ask the panel to send job details..

	    	  WriteToSocket(sock, socket_send_job_details, SOCKET_CODE_SIZE);


	      }
	      else
		  if (memcmp(socket_input_buffer, socket_ready_to_send_job_package, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
		  {
			  // receive the job details..

			  my_job_package.ReceiveJobPackage(sock);
			  LaunchJobThread *launch_thread = new LaunchJobThread(this, my_job_package.my_profile, my_ip_address, my_port_string, job_code);

			  if ( launch_thread->Run() != wxTHREAD_NO_ERROR )
			  {
				  MyPrintWithDetails("Can't create the launch thread!");
				  delete launch_thread;
				  ExitMainLoop();
				  return;
			  }
		  }
	      else
		  if (memcmp(socket_input_buffer, socket_time_to_die, SOCKET_CODE_SIZE) == 0)
		  {
			  // destroy the server..

			  socket_server->Destroy();

			  // close Gui connection..

			  sock->Destroy();

			  // pass message on to master if we have one..

			  if (have_assigned_master == true)
			  {
				  master_socket->Notify(false);
				  WriteToSocket(master_socket, socket_time_to_die, SOCKET_CODE_SIZE);
				  master_socket->Destroy();
			  }

			  // exit..

			  ExitMainLoop();
			  return;
		  }


	      // Enable input events again.

	      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	      break;
	    }

	    case wxSOCKET_LOST:
	    {

	        wxPrintf("JOB CONTROL : GUI Socket Disconnected!!\n");
	        sock->Destroy();
	        ExitMainLoop();
	        abort();

	        break;
	    }
	    default: ;
	  }
}


