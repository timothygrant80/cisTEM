#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include "wx/socket.h"
#include <wx/evtloop.h>

#include "../../core/core_headers.h"
#include "../../core/socket_codes.h"

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDINFO, wxThreadEvent);

SETUP_SOCKET_CODES

#ifdef __WXOSX__
class
JobControlApp : public wxApp, public SocketCommunicator
#else
class
JobControlApp : public wxAppConsole, public SocketCommunicator
#endif
{
	bool have_assigned_master;
	wxArrayString possible_gui_addresses;

	wxIPV4address active_gui_address;
	wxIPV4address my_address;
	wxIPV4address master_process_address;

	wxSocketClient *gui_socket;
	wxSocketBase   *master_socket;

	int number_of_received_jobs;

	long gui_port;
	long master_process_port;

	bool all_jobs_are_finished;

	short int my_port;
	wxArrayString my_possible_ip_addresses;
	wxString my_port_string;
	wxString my_active_ip_address;
	wxString master_ip_address;
	wxString master_port;

	long number_of_slaves_already_connected;

	// Socket Handling overrides..

	void HandleNewSocketConnection(wxSocketBase *new_connection,  unsigned char *identification_code);
	void HandleSocketYouAreConnected(wxSocketBase *connected_socket);
	void HandleSocketJobPackage(wxSocketBase *connected_socket, JobPackage *received_package);
	void HandleSocketTimeToDie(wxSocketBase *connected_socket);
	void HandleSocketIHaveAnError(wxSocketBase *connected_socket, wxString error_message);
	void HandleSocketIHaveInfo(wxSocketBase *connected_socket, wxString info_message);
	void HandleSocketJobResult(wxSocketBase *connected_socket, JobResult *received_result);
	void HandleSocketJobResultQueue(wxSocketBase *connected_socket, ArrayofJobResults *received_queue);
	//void HandleSocketSendJobDetails(wxSocketBase *connected_socket);
	void HandleSocketJobFinished(wxSocketBase *connected_socket, int finished_job_number);
	void HandleSocketAllJobsFinished(wxSocketBase *connected_socket, long received_timing_in_milliseconds);
	void HandleSocketDisconnect(wxSocketBase *connected_socket);
	void HandleSocketTemplateMatchResultReady(wxSocketBase *connected_socket, int &image_number, float &threshold_used, ArrayOfTemplateMatchFoundPeakInfos &peak_infos, ArrayOfTemplateMatchFoundPeakInfos &peak_changes);

	// end

	void SendError(wxString error_to_send);
	void SendInfo(wxString info_to_send);

	void SendJobFinished(int job_number);
	void SendJobResult(JobResult *result_to_send);
	void SendJobResultQueue(ArrayofJobResults &queue_to_send);

	void SendAllJobsFinished(long total_timing_from_master);
	void SendNumberofConnections();

	void OnThreadLaunchJob(wxThreadEvent &event);
	void OnThreadSendInfo(wxThreadEvent &event);


	public:
		virtual bool OnInit();
		void OnEventLoopEnter(wxEventLoopBase *	loop);



};

class LaunchJobThread : public wxThread
{
	public:
    	LaunchJobThread(JobControlApp *handler, RunProfile wanted_run_profile, wxString wanted_ip_address, wxString wanted_port, const unsigned char *wanted_job_code, long wanted_actual_number_of_jobs) : wxThread(wxTHREAD_DETACHED)
		{
    		main_thread_pointer = handler;
    		current_run_profile = wanted_run_profile;
    		ip_address = wanted_ip_address;
    		port_number = wanted_port;
    		actual_number_of_jobs = wanted_actual_number_of_jobs;

    		for (int counter = 0; counter < SOCKET_CODE_SIZE; counter++)
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
    	long actual_number_of_jobs;
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
	number_of_received_jobs = 0;
	all_jobs_are_finished = false;

	MyDebugPrint("Job Controller: Running...\n");

	// initialise sockets
	wxSocketBase::Initialize();


	return true;

}

void JobControlApp::OnEventLoopEnter(wxEventLoopBase *	loop)
{
	if (loop->IsMain() == true)
	{
		long counter;
		wxIPV4address my_address;
		wxIPV4address junk_address;
		wxString current_address;
		wxArrayString buffer_addresses;
		have_assigned_master = false;

		// Set brother event handler, this is a nastly little hack so that the socket communicator can use the event handler, and it will work whether the "brother" is a console app or gui panel.
		// it needs to be done in all classes that derive from SocketCommunicator

		brother_event_handler = this;

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

		wxStringTokenizer gui_ip_address_tokens(command_line_parser.GetParam(0),",");

		while(gui_ip_address_tokens.HasMoreTokens() == true)
		{
			current_address = gui_ip_address_tokens.GetNextToken();
			possible_gui_addresses.Add(current_address);
			if (junk_address.Hostname(current_address) == false)
			{
				MyDebugPrint(" Error: Address (%s) - not recognized as an IP or hostname\n\n", current_address);
				exit(-1);
			};
		}

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
			current_job_code[counter] = command_line_parser.GetParam(2).GetChar(counter);
		}


		// Attempt to connect to the gui..

		active_gui_address.Service(gui_port);
		gui_socket = new wxSocketClient();
		gui_socket->Notify(false);

		for (counter = 0; counter < possible_gui_addresses.GetCount(); counter++)
		{
			active_gui_address.Hostname(possible_gui_addresses.Item(counter));
			MyDebugPrint("\nJob Controller: Trying to connect to %s:%i (timeout = 4 sec) ...\n", active_gui_address.IPAddress(), active_gui_address.Service());

			gui_socket->Connect(active_gui_address, false);
			gui_socket->WaitOnConnect(20);

			if (gui_socket->IsConnected() == false)
			{
			   gui_socket->Close();
			   MyDebugPrint("Connection Failed.\n\n");
			}
			else
			{
				break;
			}
		}

		if (gui_socket->IsConnected() == false)
		{
		   gui_socket->Close();
		   wxPrintf("All connections Failed! Exiting...\n");
		   ExitMainLoop();
		   return;
		}

		gui_socket->SetFlags(SOCKET_FLAGS);
		MyDebugPrint("Job Controller: Succeeded - Connection established!\n\n");

		number_of_slaves_already_connected = 0;


		// monitor gui socket..

		MonitorSocket(gui_socket);


		// Job launching event..

		Bind(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB, &JobControlApp::OnThreadLaunchJob, this);
		Bind(wxEVT_COMMAND_MYTHREAD_SENDINFO, &JobControlApp::OnThreadSendInfo, this);

	}
}

void LaunchJobThread::LaunchRemoteJob()
{
	long counter;
	long command_counter;
	long process_counter;
	long number_of_commands_to_run;
	long number_of_commands_run = 0;
	long number_to_run_for_this_command;

	wxIPV4address address;

	// for n processes (specified in the job package) we need to launch the specified command, along with our
	// IP address, port and job code..

	wxString executable;
	wxString executable_with_threads;
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



	if (actual_number_of_jobs < current_run_profile.ReturnTotalJobs()) number_of_commands_to_run = actual_number_of_jobs;
	else
	number_of_commands_to_run = current_run_profile.ReturnTotalJobs();

	for (command_counter = 0; command_counter <  current_run_profile.number_of_run_commands; command_counter++)
	{
		if (number_of_commands_to_run - number_of_commands_run < current_run_profile.run_commands[command_counter].number_of_copies) number_to_run_for_this_command = number_of_commands_to_run - number_of_commands_run;
		else number_to_run_for_this_command = current_run_profile.run_commands[command_counter].number_of_copies;

		execution_command =  current_run_profile.run_commands[command_counter].command_to_run;
		executable_with_threads = executable + wxString::Format(" %i", current_run_profile.run_commands[command_counter].number_of_threads_per_copy);
		execution_command.Replace("$command", executable_with_threads);
		execution_command.Replace("$program_name", current_run_profile.executable_name);

		execution_command += "&";

		for (process_counter = 0; process_counter < number_to_run_for_this_command; process_counter++)
		{

			wxMilliSleep( current_run_profile.run_commands[command_counter].delay_time_in_ms);

			if (process_counter == 0) QueueInfo(wxString::Format("Job Control : Executing '%s' %li times.", execution_command, number_to_run_for_this_command));

			//wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_LAUNCHJOB);
			//test_event->SetString(execution_command);

			//wxQueueEvent(main_thread_pointer, test_event);
			//wxExecute(execution_command);
			system(execution_command.ToUTF8().data());
			number_of_commands_run++;
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
	if (wxExecute(event.GetString()) == 0)
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
	WriteToSocket(gui_socket, socket_i_have_an_error, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	SendwxStringToSocket(&error_to_send, gui_socket);
}

void JobControlApp::SendInfo(wxString info_to_send)
{
	WriteToSocket(gui_socket, socket_i_have_info, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	SendwxStringToSocket(&info_to_send, gui_socket);
}

void JobControlApp::SendJobFinished(int job_number)
{
	WriteToSocket(gui_socket, socket_job_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(gui_socket, &job_number, sizeof(int), true, "SendJobNumber", FUNCTION_DETAILS_AS_WXSTRING);
}

void JobControlApp::SendJobResult(JobResult *job_to_send)
{
	WriteToSocket(gui_socket, socket_job_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	job_to_send->SendToSocket(gui_socket);
}

void JobControlApp::SendJobResultQueue(ArrayofJobResults &queue_to_send)
{
	WriteToSocket(gui_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	SendResultQueueToSocket(gui_socket, queue_to_send);
}


void JobControlApp::SendAllJobsFinished(long total_timing_from_master)
{
	WriteToSocket(gui_socket, socket_all_jobs_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(gui_socket, &total_timing_from_master, sizeof(long), true, "SendTotalMillisecondsSpentOnThreads", FUNCTION_DETAILS_AS_WXSTRING);
}

void JobControlApp::SendNumberofConnections()
{
	WriteToSocket(gui_socket, socket_number_of_connections, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(gui_socket, &number_of_slaves_already_connected, 4, true, "SendNumberOfConnections", FUNCTION_DETAILS_AS_WXSTRING);

	int number_of_commands_to_run;

	if (current_job_package.number_of_jobs + 1 < current_job_package.my_profile.ReturnTotalJobs()) number_of_commands_to_run = current_job_package.number_of_jobs + 1;
	else number_of_commands_to_run = current_job_package.my_profile.ReturnTotalJobs();

	if (number_of_slaves_already_connected == number_of_commands_to_run)
	{

		ShutDownServer();
		MyDebugPrint("Socket Server is now shutdown\n");
	}
}


//////////////////////////////////////////////////////////////////////////////////////
//                      SOCKET OVERRIDES  - FIRST SERVER                            //
//////////////////////////////////////////////////////////////////////////////////////

void JobControlApp::HandleNewSocketConnection(wxSocketBase *new_connection, unsigned char *identification_code)
{


	 if (new_connection == NULL) return;

	 if ((memcmp(identification_code, current_job_code, SOCKET_CODE_SIZE) != 0) )
	 {
		 SendError("Unknown Job ID (Job Control), leftover from a previous job? - Closing Connection");
		 new_connection->Destroy(); // should not be monitoring so can just destroy it
		 new_connection = NULL;
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

			 master_socket = new_connection;
			 have_assigned_master = true;

			 WriteToSocket(new_connection, socket_you_are_the_master, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
			 current_job_package.SendJobPackage(new_connection);

			 bool no_error;
			 master_ip_address = ReceivewxStringFromSocket(new_connection, no_error);
			 master_port = ReceivewxStringFromSocket(new_connection, no_error);

			 // monitor the master socket..

			 MonitorSocket(new_connection);

			 number_of_slaves_already_connected++;
			 SendNumberofConnections();

		 }
		 else  // we have a master, tell this slave who it's master is.
		 {
			 WriteToSocket(new_connection, socket_you_are_a_slave, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
			 SendwxStringToSocket(&master_ip_address, new_connection);
			 SendwxStringToSocket(&master_port, new_connection);

			 // that should be the end of our interactions with the slave
			 // it should disconnect itself, we won't even bother
			 // setting up events for it..

			 number_of_slaves_already_connected++;
			 SendNumberofConnections();

		 }
	 }

	 delete [] identification_code;
}

//////////////////////////////////////////////////////////////////////////////////////
//                      THESE SHOULD BE FROM THE GUI                                //
//////////////////////////////////////////////////////////////////////////////////////


void JobControlApp::HandleSocketYouAreConnected(wxSocketBase *connected_socket)
{
	// we are connected to the gui, ask it to send job details..

	WriteToSocket(connected_socket, socket_send_job_details, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
}

void JobControlApp::HandleSocketJobPackage(wxSocketBase *connected_socket, JobPackage *received_package)
{
	// we have a job package from the gui, copy it.. start a server and launch the jobs..

	current_job_package = *received_package;
	delete received_package;

	SetupServer();
	my_port = ReturnServerPort();
	my_port_string = ReturnServerPortString();

	wxString current_address_according_to_gui;
	wxString ip_address_string;
	wxArrayString buffer_addresses = ReturnServerAllIpAddresses();

	current_address_according_to_gui = ReturnIPAddressFromSocket(gui_socket);

	my_possible_ip_addresses.Clear();
	if (current_address_according_to_gui.IsEmpty() == false) my_possible_ip_addresses.Add(current_address_according_to_gui);

	for (int counter = 0; counter < buffer_addresses.GetCount(); counter++)
	{
		if (buffer_addresses.Item(counter) != current_address_according_to_gui) my_possible_ip_addresses.Add(buffer_addresses.Item(counter));
	}

	for (int counter = 0; counter < my_possible_ip_addresses.GetCount(); counter++)
	{
		if (counter != 0) ip_address_string += ",";
	  	ip_address_string += my_possible_ip_addresses.Item(counter);
	}

	LaunchJobThread *launch_thread = new LaunchJobThread(this, current_job_package.my_profile, ip_address_string, my_port_string, current_job_code, current_job_package.number_of_jobs);

	if ( launch_thread->Run() != wxTHREAD_NO_ERROR )
	{
		MyPrintWithDetails("Can't create the launch thread!");
		delete launch_thread;
		ExitMainLoop();
		return;
	}
}

void JobControlApp::HandleSocketTimeToDie(wxSocketBase *connected_socket)
{
	  if (have_assigned_master == true)
	  {
		  WriteToSocket(master_socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
		  StopMonitoringAndDestroySocket(master_socket);

	  }


	  ShutDownServer();

	  // close GUI connection

	  StopMonitoringAndDestroySocket(gui_socket);

	  ShutDownSocketMonitor();

	  // die..

	  ExitMainLoop();
	  exit(0);
	  return;
}

//////////////////////////////////////////////////////////////////////////////////////
//                    THESE SHOULD BE FROM THE MASTER                               //
//////////////////////////////////////////////////////////////////////////////////////

//void JobControlApp::HandleSocketSendJobDetails(wxSocketBase *connected_socket)
//{
	// send it the job package we should have already received from the gui..

//	WriteToSocket(connected_socket, socket_sending_job_package, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
//	current_job_package.SendJobPackage(connected_socket);
//}

void JobControlApp::HandleSocketIHaveAnError(wxSocketBase *connected_socket, wxString error_message)
{
	// pass the error message up to the GUI..

	SendError(error_message);
}

void JobControlApp::HandleSocketIHaveInfo(wxSocketBase *connected_socket, wxString info_message)
{
	// pass the error message up to the GUI..

	SendInfo(info_message);
}

void JobControlApp::HandleSocketJobResult(wxSocketBase *connected_socket, JobResult *received_result)
{
	// pass the result onto the GUI...

	SendJobResult(received_result);

	// delete it..

	delete received_result;
}

void JobControlApp::HandleSocketJobResultQueue(wxSocketBase *connected_socket, ArrayofJobResults *received_queue)
{
	// pass it on and delete..

	SendJobResultQueue(*received_queue);
	number_of_received_jobs += received_queue->GetCount();

	delete received_queue;
}

void JobControlApp::HandleSocketJobFinished(wxSocketBase *connected_socket, int finished_job_number)
{
	// pass on to the gui..

	 SendJobFinished(finished_job_number);
}

void JobControlApp::HandleSocketAllJobsFinished(wxSocketBase *connected_socket, long received_timing_in_milliseconds)
{
	// pass on to the gui..

	SendAllJobsFinished(received_timing_in_milliseconds);
	all_jobs_are_finished = true;

	// don't die, wait for GUI to kill me..
}

void JobControlApp::HandleSocketTemplateMatchResultReady(wxSocketBase *connected_socket, int &image_number, float &threshold_used, ArrayOfTemplateMatchFoundPeakInfos &peak_infos, ArrayOfTemplateMatchFoundPeakInfos &peak_changes)
{
	// pass on to the gui..

	SendTemplateMatchingResultToSocket(gui_socket, image_number, threshold_used, peak_infos, peak_changes);
}

void JobControlApp::HandleSocketDisconnect(wxSocketBase *connected_socket)
{
	// what disconnected..

	if (connected_socket == gui_socket)
	{
		MyDebugPrint("Got a disconnect from the GUI - aborting");

		 if (have_assigned_master == true)
		 {
			 WriteToSocket(master_socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
			 StopMonitoringAndDestroySocket(master_socket);
		 }

		 ShutDownServer();

		 // close GUI connection

		 StopMonitoringAndDestroySocket(gui_socket);

		 ShutDownSocketMonitor();

		 // die..

		 ExitMainLoop();
		 exit(0);
		 return;
	}
	else
	if (connected_socket == master_socket)
	{
		//must be from the master..
		//Have we send all jobs are finished?

		if (all_jobs_are_finished == false) // something went wrong
		{
				MyDebugPrint("Controller got disconnect from master...");
				SendError("Controller received a disconnect from the master before the job was finished..");


				ShutDownServer();

				// close GUI connection

				StopMonitoringAndDestroySocket(gui_socket);
				ShutDownSocketMonitor();

				// die..

				ExitMainLoop();
				exit(0);
				return;
		}

		// otherwise we don't care.. still wait for gui to kill us..
	}
	else // who knows what this is!
	{
		 StopMonitoringAndDestroySocket(connected_socket);
	}

}
