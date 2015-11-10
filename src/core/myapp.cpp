#include "core_headers.h"

bool MyApp::OnInit()
{

	int parse_status;
	int number_of_arguments;

	number_of_dispatched_jobs = 0;
	number_of_finished_jobs = 0;

	long counter;

	// Bind the thread events

	Bind(wxEVT_COMMAND_MYTHREAD_COMPLETED, &MyApp::OnThreadComplete, this);
	Bind(wxEVT_COMMAND_MYTHREAD_SENDERROR, &MyApp::OnThreadSendError, this);


	// Connect to the controller program..
	// set up the parameters for passing the gui address..

	static const wxCmdLineEntryDesc command_line_descriptor[] =
	{
			{ wxCMD_LINE_PARAM, "a", "address", "gui_address", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL },
			{ wxCMD_LINE_PARAM, "p", "port", "gui_port", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL },
			{ wxCMD_LINE_PARAM, "j", "job_code", "job_code", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL },
			{ wxCMD_LINE_NONE }
	};


	wxCmdLineParser command_line_parser( command_line_descriptor, argc, argv);

	wxPrintf("\n");
	parse_status = command_line_parser.Parse(true);
	number_of_arguments = command_line_parser.GetParamCount();

	if (parse_status != 0)
	{
		wxPrintf("\n\n");
		return false;

	}

	// if we have no arguments run interactively.. if we have 3 continue as though we have network info, else error..

	if (number_of_arguments == 0)
	{
		is_running_locally = true;
		DoInteractiveUserInput();
		DoCalculation();
		return false;
	}
	else
	if (number_of_arguments != 3)
	{
		command_line_parser.Usage();
		wxPrintf("\n\n");
		return false;
	}

	is_running_locally = false;


	// get the address and port of the gui (should be command line options).


	if (controller_address.Hostname(command_line_parser.GetParam(0)) == false)
	{
		MyPrintWithDetails(" Error: Address (%s) - not recognized as an IP or hostname\n\n", command_line_parser.GetParam(0));
		exit(-1);
	};

	if (command_line_parser.GetParam(1).ToLong(&controller_port) == false)
	{
		MyPrintWithDetails(" Error: Port (%s) - not recognized as a port\n\n", command_line_parser.GetParam(1));
		exit(-1);
	}

	if (command_line_parser.GetParam(2).Len() != SOCKET_CODE_SIZE)
	{
		{
			MyPrintWithDetails(" Error: Code (%s) - is the incorrect length (%li instead of %i)\n\n", command_line_parser.GetParam(2), command_line_parser.GetParam(2).Len(), SOCKET_CODE_SIZE);
			exit(-1);
		}
	}

	// copy over job code.

	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		job_code[counter] = command_line_parser.GetParam(2).GetChar(counter);
	}


	controller_address.Service(controller_port);

	// Attempt to connect to the controller..

	MyDebugPrint("\n JOB : Trying to connect to %s:%i (timeout = 10 sec) ...\n", controller_address.IPAddress(), controller_address.Service());
	controller_socket = new wxSocketClient();

	// Setup the event handler and subscribe to most events

	controller_socket->SetEventHandler(*this, SOCKET_ID);
	controller_socket->SetNotify(wxSOCKET_CONNECTION_FLAG |wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
	controller_socket->Notify(true);
	is_connected = false;


	Bind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );


	controller_socket->Connect(controller_address, false);
	controller_socket->WaitOnConnect(30);

	if (controller_socket->IsConnected() == false)
	{
	   controller_socket->Close();
	   MyDebugPrint(" JOB : Failed ! Unable to connect\n");
	   return false;
	}

	MyDebugPrint(" JOB : Succeeded - Connection established!\n\n");
	is_connected = true;

	return true;
}

void MyApp::OnOriginalSocketEvent(wxSocketEvent &event)
{
	SETUP_SOCKET_CODES

	 wxString s = _("JOB : OnSocketEvent: ");
	 wxSocketBase *sock = event.GetSocket();

	 MyDebugAssertTrue(sock == controller_socket, "GUI Socket event from Non controller socket??");

		  // First, print a message
	 switch(event.GetSocketEvent())
	 {
	   case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	   case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	   default             : s.Append(_("Unexpected event !\n")); break;
	 }

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
			 if (memcmp(socket_input_buffer, socket_you_are_the_master, SOCKET_CODE_SIZE) == 0) // we are the master.. so setup job control and prepare to recieve connections
		     {
		    	  MyDebugPrint("JOB  : I am the master");

		    	  i_am_the_master = true;

		    	  // redirect socket events appropriately

				  Unbind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );
				  Bind(wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnControllerSocketEvent), this,  SOCKET_ID );

		    	  // we need to start a server so that the slaves can connect..

		    	  SetupServer();

		    	  // I have to send my ip address to the controller..

		    	  my_ip_address = ReturnIPAddressFromSocket(sock);

		    	  SendwxStringToSocket(&my_ip_address, sock);
		    	  SendwxStringToSocket(&my_port_string, sock);

		    	  // ok, now get the job details from the conduit controller

		    	  MyDebugPrint("JOB MASTER : Asking for job details");
		    	  sock->WriteMsg(socket_send_job_details, SOCKET_CODE_SIZE);

		    	  // it should now send a conformation code followed by the package..

		    	  sock->WaitForRead(15);

		    	  if (sock->IsData() == true)
		    	  {
		    		  sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

		    	 	  // is this correct?

		    	 	  if ((memcmp(socket_input_buffer, socket_ready_to_send_job_package, SOCKET_CODE_SIZE) != 0) )
		    	 	  {
		    	 		  MyDebugPrintWithDetails("JOB Master : Oops unidentified message!");

		    	 		  sock->Destroy();
		    	 		  sock = NULL;
		    	 		  abort();
		    	 	  }
		    	 	  else
		    	 	  {

						  // receive the job details..
						  my_job_package.ReceiveJobPackage(sock);
						  MyDebugPrint("JOB Master : Job Package Received");

				    	  // allocate space for socket pointers..
						  number_of_connected_slaves = 0;
						  slave_sockets = new wxSocketBase*[my_job_package.my_profile.ReturnTotalJobs()];
		    	 	  }
		    	  }
		    	  else
		    	  {
		    		  MyDebugPrintWithDetails(" JOB MASTER : ...Read Timeout waiting for job package \n\n");
		    		  // time out - close the connection
		    		  sock->Destroy();
		    		  sock = NULL;
		    		  abort();
		    	  }
		      }
		      else
			  if (memcmp(socket_input_buffer, socket_you_are_a_slave, SOCKET_CODE_SIZE) == 0) // i'm a slave.. who is the master
			  {
				  long received_port;
				  // receive the job details..

				  MyDebugPrint("JOB  : I am a slave");
				  i_am_the_master = false;

			      // get the master_ip_address;

				  master_ip_address = ReceivewxStringFromSocket(sock);
				  master_port_string = ReceivewxStringFromSocket(sock);

				  master_port_string.ToLong(&received_port);
				  master_port = (short int) received_port;


				  // now we drop the connection, and connect to our new master..

				  sock->Destroy();
				  Unbind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );

				  // connect to the new master..

				  controller_socket = new wxSocketClient();
				  controller_address.Hostname(master_ip_address);
				  controller_address.Service(master_port);

				  controller_socket->Connect(controller_address, false);
				  controller_socket->WaitOnConnect(30);

				  if (controller_socket->IsConnected() == false)
				  {
					   controller_socket->Close();
					   MyDebugPrint("JOB : Failed ! Unable to connect\n");

				  }


				  // Setup the event handler and subscribe to most events


				   controller_socket->SetEventHandler(*this, SOCKET_ID);
				   controller_socket->SetNotify(wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
				   controller_socket->Notify(true);


				   Bind(wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnMasterSocketEvent), this,  SOCKET_ID );

				  // we should now be connected to the new master and reacting to events (e.g. first jobs) from it...
			  }
			  else
			  {
				  MyDebugPrintWithDetails("Unknown Message!")
				  abort();
			  }


		      // Enable input events again.

		      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
		      break;
		    }

		    case wxSOCKET_LOST:
		    {

		        wxPrintf("JOB  : Socket Disconnected!!\n");
		        sock->Destroy();
		        ExitMainLoop();

		        break;
		    }
		    default: ;
		  }

}


void MyApp::OnControllerSocketEvent(wxSocketEvent &event)
{
	SETUP_SOCKET_CODES

	  wxString s = _("JOB MASTER : OnControllerSocketEvent: ");
	  wxSocketBase *sock = event.GetSocket();

	  MyDebugAssertTrue(sock == controller_socket, "Controller Socket event from Non Controller socket??");

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

	      if (memcmp(socket_input_buffer, socket_ready_to_send_job_package, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
		  {
	    	  //SPACER DELETE

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

void MyApp::SendNextJobTo(wxSocketBase *socket)
{
	SETUP_SOCKET_CODES

	// if we haven't dispatched all jobs yet, then send it, otherwise tell the slave to die..

	if (number_of_dispatched_jobs < my_job_package.number_of_jobs)
	{
		my_job_package.jobs[number_of_dispatched_jobs].SendJob(socket);
		number_of_dispatched_jobs++;

	}
	else
	{
		socket->WriteMsg(socket_time_to_die, SOCKET_CODE_SIZE);
	}
}

void MyApp::SendJobFinished(int job_number)
{
	MyDebugAssertTrue(i_am_the_master == true, "SendJobFinished called by a slave!");

	SETUP_SOCKET_CODES

	// get the next job..
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_job_finished, SOCKET_CODE_SIZE);
	// send the job number of the current job..
	controller_socket->WriteMsg(&job_number, 4);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);


}

void MyApp::SendAllJobsFinished()
{
	MyDebugAssertTrue(i_am_the_master == true, "SendAllJobsFinished called by a slave!");

	SETUP_SOCKET_CODES

	// get the next job..
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_all_jobs_finished, SOCKET_CODE_SIZE);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}

void MyApp::OnSlaveSocketEvent(wxSocketEvent &event)
{
	SETUP_SOCKET_CODES

	wxString s = _("JOB MASTER: OnSlaveSocketEvent: ");
	wxSocketBase *sock = event.GetSocket();

	//MyDebugAssertTrue(sock == controller_socket, "Master Socket event from Non controller socket??");

	// First, print a message
	switch(event.GetSocketEvent())
	{
	   case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	   case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	   default             : s.Append(_("Unexpected event !\n")); break;
	}

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

			 if (memcmp(socket_input_buffer, socket_send_next_job, SOCKET_CODE_SIZE) == 0) // identification
			 {
				 MyDebugPrint("JOB MASTER : SEND NEXT JOB");

		    	 int finished_job_number;
		    	 sock->ReadMsg(&finished_job_number, 4);

		    	 SendNextJobTo(sock);

		    	 // Send info that the job has finished..

		    	 if (finished_job_number != -1)
		    	 {
		    		 SendJobFinished(finished_job_number);
		    		 number_of_finished_jobs++;
		    		 my_job_package.jobs[finished_job_number].has_been_run = true;

		    		 if (number_of_finished_jobs == my_job_package.number_of_jobs)
		    		 {
		    			 SendAllJobsFinished();

		    			 if (my_job_package.ReturnNumberOfJobsRemaining() != 0)
		    			 {
		    				 SendError("All jobs should be finished, but job package is not empty.");
		    			 }

		    		   	  	  // time to die!

		    			 	 controller_socket->Destroy();
		    			 	 ExitMainLoop();

		    		 }
		    	 }
			 }
			 else
			 if (memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0) // identification
			 {
				 // got an error message..
				 MyDebugPrint("JOB MASTER : Error Message");
				wxString error_message;

				error_message = ReceivewxStringFromSocket(sock);

				// send the error message up the chain..

				SocketSendError(error_message);
			 }



			 // Enable input events again.

			 sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
			 break;
		 }

		 case wxSOCKET_LOST:
		 {
			 if (number_of_dispatched_jobs < my_job_package.number_of_jobs)
			 {
				 SocketSendError("Error: A slave has disconnected before all jobs are finished.");
			 }

		     wxPrintf("JOB Master : a slave socket Disconnected!!\n");

		     sock->Destroy();
		     //ExitMainLoop();

		     break;
		  }
		 default: ;
	}


}

void MyApp::SetupServer()
{

	wxIPV4address my_address;
	wxIPV4address buffer_address;

	MyDebugPrint("Setting up Server...");

	for (short int current_port = START_PORT; current_port <= END_PORT; current_port++)
	{

		if (current_port == END_PORT)
		{
			wxPrintf("JOB MASTER : Could not find a valid port !\n\n");
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

		  Bind(wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnServerEvent), this, SERVER_ID);

		  my_port_string = wxString::Format("%hi", my_port);
/*
		  buffer_address.Hostname(wxGetFullHostName()); // hopefully get my ip
		  my_ip_address = buffer_address.IPAddress();



*/

		  break;
		}
		else socket_server->Destroy();

	}
}

void MyApp::OnServerEvent(wxSocketEvent& event) // this should only be called by the master
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
	 MyDebugPrint(" JOB MASTER : Requesting identification...");
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

	  		// A slave has connected

	  		slave_sockets[number_of_connected_slaves] = sock;
	  		number_of_connected_slaves++;

	  		// tell it is is connected..

	  		sock->WriteMsg(socket_you_are_connected, SOCKET_CODE_SIZE);

	  		// we need to setup events of this socket..

	  		sock->SetEventHandler(*this, SOCKET_ID);
			sock->SetNotify(wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
			sock->Notify(true);

			Unbind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );
			Bind(wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnSlaveSocketEvent), this,  SOCKET_ID );


	  	  }
		}
		else
		{
			MyDebugPrint(" JOB MASTER : ...Read Timeout waiting for job ID \n\n");
		 	// time out - close the connection
			sock->Destroy();
			sock = NULL;
		}


}


void MyApp::OnMasterSocketEvent(wxSocketEvent& event)
{
	SETUP_SOCKET_CODES

	wxString s = _("JOB : OnMasterSocketEvent: ");
	wxSocketBase *sock = event.GetSocket();

	MyDebugAssertTrue(sock == controller_socket, "Master Socket event from Non controller socket??");

	// First, print a message
	switch(event.GetSocketEvent())
	{
	   case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	   case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	   default             : s.Append(_("Unexpected event !\n")); break;
	}

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
				 MyDebugPrint("JOB SLAVE : Sending Identification")
		    	 sock->WriteMsg(job_code, SOCKET_CODE_SIZE);

			 }
			 else
			 if (memcmp(socket_input_buffer, socket_you_are_connected, SOCKET_CODE_SIZE) == 0) // identification
			 {
			   	  // we are connected, request the first job..
				 is_connected = true;
				 MyDebugPrint("JOB SLAVE : Requesting job");
				 controller_socket->WriteMsg(socket_send_next_job, SOCKET_CODE_SIZE);
				 int no_job = -1;
				 controller_socket->WriteMsg(&no_job, 4);

			 }
			 else
			 if (memcmp(socket_input_buffer, socket_ready_to_send_single_job, SOCKET_CODE_SIZE) == 0) // identification
			 {
				 // are we currently running a job?

				 MyDebugAssertTrue(currently_running_a_job == false, "Received a new job, when already running a job!");

				 // recieve job

				 MyDebugPrint("JOB SLAVE : About to receive new job")

				 my_current_job.RecieveJob(sock);

				 // run a thread for this job..

				 MyDebugPrint("JOB SLAVE : New Job, starting thread")

				 MyDebugAssertTrue(work_thread == NULL, "Running a new thread, but old thread is not NULL");

				 work_thread = new CalculateThread(this);

				 if ( work_thread->Run() != wxTHREAD_NO_ERROR )
				 {
				       MyPrintWithDetails("Can't create the thread!");
				       delete work_thread;
				       work_thread = NULL;
				       abort();
				 }


				 MyDebugPrint("JOB SLAVE : Started Thread");



			 }
			 else
			 if (memcmp(socket_input_buffer, socket_time_to_die, SOCKET_CODE_SIZE) == 0) // identification
			 {
			   	  // time to die!

				 sock->Destroy();
	 		     ExitMainLoop();
			 }


			 			 // Enable input events again.

			 sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
			 break;
		 }

		 case wxSOCKET_LOST:
		 {

		     wxPrintf("JOB  : Master Socket Disconnected!!\n");
		     sock->Destroy();
		     ExitMainLoop();

		     break;
		  }
		 default: ;
	}


}

void MyApp::OnThreadComplete(wxThreadEvent& my_event)
{
	SETUP_SOCKET_CODES

	// The compute thread is finished.. get the next job

	// thread should be dead, or nearly dead..

	work_thread = NULL;

	// get the next job..
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_send_next_job, SOCKET_CODE_SIZE);
	// send the job number of the current job..
	controller_socket->WriteMsg(&my_current_job.job_number, 4);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}

void MyApp::OnThreadSendError(wxThreadEvent& my_event)
{
	SocketSendError(my_event.GetString());
	//MyDebugPrint("ThreadSendError");
}

void MyApp::SocketSendError(wxString error_to_send)
{
	SETUP_SOCKET_CODES

	// send the error message flag

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_i_have_an_error, SOCKET_CODE_SIZE);

	SendwxStringToSocket(&error_to_send, controller_socket);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}


void MyApp::SendError(wxString error_to_send)
{
	if (is_running_locally == true)
	{
		wxPrintf("\nError : %s\n", error_to_send);

	}
	else
	if (work_thread != NULL)
	{
		work_thread->QueueError(error_to_send);
	}
	else MyDebugPrint("SendError with null work thread!")
}

// Main execution in this thread..

wxThread::ExitCode CalculateThread::Entry()
{
	bool success = main_thread_pointer->DoCalculation(); // This should be overrided per app..

	wxThreadEvent *my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_COMPLETED);

	if (success == true) my_thread_event->SetInt(1);
	else my_thread_event->SetInt(0);

	wxQueueEvent(main_thread_pointer, my_thread_event);


    return (wxThread::ExitCode)0;     // success
}

void  CalculateThread::QueueError(wxString error_to_queue)
{
	wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SENDERROR);
	test_event->SetString(error_to_queue);

	wxQueueEvent(main_thread_pointer, test_event);
}

CalculateThread::~CalculateThread()
{
    //wxCriticalSectionLocker enter(m_pHandler->m_pThreadCS);
    // the thread is being destroyed; make sure not to leave dangling pointers around
    main_thread_pointer = NULL;
}


