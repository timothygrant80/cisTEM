#include "core_headers.h"

SETUP_SOCKET_CODES

#define THREAD_START_NEXT_JOB 0
#define THREAD_DIE 1
#define THREAD_SLEEP 2

bool MyApp::OnInit()
{


	long counter;
	int parse_status;
	int number_of_arguments;

	thread_next_action = THREAD_SLEEP;

	number_of_dispatched_jobs = 0;
	number_of_finished_jobs = 0;
	number_of_connected_slaves = 0;

	connection_timer = NULL;
	zombie_timer = NULL;


	// Bind the thread events

	Bind(wxEVT_COMMAND_MYTHREAD_COMPLETED, &MyApp::OnThreadComplete, this);
	Bind(wxEVT_COMMAND_MYTHREAD_SENDERROR, &MyApp::OnThreadSendError, this);
	Bind(wxEVT_COMMAND_MYTHREAD_SENDINFO, &MyApp::OnThreadSendInfo, this);
	Bind(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE, &MyApp::OnThreadIntermediateResultAvailable, this);


	// Connect to the controller program..


	// set up the parameters for passing the gui address..
	command_line_parser.SetCmdLine(argc,argv);
	command_line_parser.AddParam("controller_address",wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL);
	command_line_parser.AddParam("controller_port",wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL);
	command_line_parser.AddParam("job_code",wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL);

	// Let the app add options
	AddCommandLineOptions();

	//wxPrintf("\n");
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


	// initialise sockets..

	wxSocketBase::Initialize();

	// Attempt to connect to the controller..

	controller_address.Service(controller_port);
	is_connected = false;

	//MyDebugPrint("\n JOB : Trying to connect to %s:%i (timeout = 30 sec) ...\n", controller_address.IPAddress(), controller_address.Service());
	controller_socket = new wxSocketClient();
	controller_socket->SetFlags(wxSOCKET_BLOCK);

	// Setup the event handler and subscribe to most events

	Bind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );
	controller_socket->SetEventHandler(*this, SOCKET_ID);
	controller_socket->SetNotify(wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
	controller_socket->Notify(true);

	wxSleep(2);

	controller_socket->Connect(controller_address, false);
	controller_socket->WaitOnConnect(120);

	if (controller_socket->IsConnected() == false || controller_socket->IsOk() == false)
	{
	   controller_socket->Close();
	   MyDebugPrint(" JOB : Failed ! Unable to connect\n");
	   return false;
	}

	// we are apparently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
	// system - if the port if valid.  So if we don't get any events from this socket within 120 seconds, we are going to assume something
	// went wrong and die...

	i_am_a_zombie = true;
	Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnZombieTimer ), this, 1);
	zombie_timer = new wxTimer(this, 1);
	zombie_timer->Start(120000, true);




	// go into the event loop

	return true;
}

// Placeholder (to be overridden) function to add options to the command line
void MyApp::AddCommandLineOptions( )
{
	return;
}

void MyApp::OnOriginalSocketEvent(wxSocketEvent &event)
{
	wxSocketBase *sock = event.GetSocket();

	MyDebugAssertTrue(sock == controller_socket, "GUI Socket event from Non controller socket??");

	// if we got here, we have actual communication, and therefore we are not a zombie..

	i_am_a_zombie = false;
	if (zombie_timer != NULL)
	{
		delete zombie_timer;
		zombie_timer = NULL;
	}

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

		    	  // Enable input events again
			      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
			 }
			 else
			 if (memcmp(socket_input_buffer, socket_you_are_the_master, SOCKET_CODE_SIZE) == 0) // we are the master.. so setup job control and prepare to receive connections
		     {
		    	  //MyDebugPrint("JOB  : I am the master");

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

		    	  sock->WriteMsg(socket_send_job_details, SOCKET_CODE_SIZE);

		    	  // it should now send a conformation code followed by the package..

		    	  sock->WaitForRead(10);

		    	  if (sock->IsData() == true)
		    	  {
		    		  sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

		    	 	  // is this correct?

		    	 	  if ((memcmp(socket_input_buffer, socket_ready_to_send_job_package, SOCKET_CODE_SIZE) != 0) )
		    	 	  {
		    	 		  MyDebugPrintWithDetails("JOB Master : Oops unidentified message!");

		    	 		  sock->Destroy();
		    	 		  sock = NULL;
		    	 		  ExitMainLoop();
		    	 		  return;
		    	 	  }
		    	 	  else
		    	 	  {

						  // receive the job details..
						  my_job_package.ReceiveJobPackage(sock);
						  //MyDebugPrint("JOB Master : Job Package Received");

				    	  // allocate space for socket pointers..
						  number_of_connected_slaves = 0;
						  slave_sockets = new wxSocketBase*[my_job_package.my_profile.ReturnTotalJobs()];

						  for (int counter = 0; counter < my_job_package.my_profile.ReturnTotalJobs(); counter++)
						  {
							  slave_sockets[counter] = NULL;
						  }
		    	 	  }
		    	  }
		    	  else
		    	  {
		    		  MyDebugPrintWithDetails(" JOB MASTER : ...Read Timeout waiting for job package \n\n");
		    		  // time out - close the connection
		    		  sock->Destroy();
		    		  sock = NULL;
		    		  ExitMainLoop();
		    		  return;
		    	  }


		    	  // Setup a timer for any straggling connections..

		    	  Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnConnectionTimer ), this, 0);
		    	  connection_timer = new wxTimer(this, 0);
		    	  connection_timer->Start(5000);

		    	  // Enable input events again
			      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

		      }
		      else
			  if (memcmp(socket_input_buffer, socket_you_are_a_slave, SOCKET_CODE_SIZE) == 0) // i'm a slave.. who is the master
			  {
				  long received_port;

				  i_am_the_master = false;

			      // get the master_ip_address;

				  master_ip_address = ReceivewxStringFromSocket(sock);
				  master_port_string = ReceivewxStringFromSocket(sock);

				  master_port_string.ToLong(&received_port);
				  master_port = (short int) received_port;

				  // now we drop the connection, and connect to our new master..

				  Unbind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );
				  sock->Destroy();

				  Bind(wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnMasterSocketEvent), this,  SOCKET_ID );
				  // connect to the new master..

				  controller_socket = new wxSocketClient();
				  controller_address.Hostname(master_ip_address);
				  controller_address.Service(master_port);

				  // Setup the event handler and subscribe to most events
				   controller_socket->SetEventHandler(*this, SOCKET_ID);
				   controller_socket->SetNotify(wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
				   controller_socket->Notify(true);

				  controller_socket->Connect(controller_address, false);
				  controller_socket->WaitOnConnect(120);

				  if (controller_socket->IsConnected() == false)
				  {
					   controller_socket->Close();
					   MyDebugPrint("JOB : Failed ! Unable to connect\n");

				  }

					// Start the worker thread..

				  work_thread = new CalculateThread(this);

				  if ( work_thread->Run() != wxTHREAD_NO_ERROR )
				  {
				       MyPrintWithDetails("Can't create the thread!");
				       delete work_thread;
				       work_thread = NULL;
				       ExitMainLoop();
				 }



				  // we are apparently connected, but this can be a lie = a certain number of connections appear to just be accepted by the operating
				  // system - if the port if valid.  So if we don't get any events from this socket with 30 seconds, we are going to assume something
				  // went wrong and die...

				  i_am_a_zombie = true;
				  Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnZombieTimer ), this, 1);
				  zombie_timer = new wxTimer(this, 1);
				  zombie_timer->Start(120000, true);


			  }
			  else
			  {
				  MyDebugPrintWithDetails("Unknown Message!")
				  ExitMainLoop();
				  return;
			  }

		      break;
		    }

		    case wxSOCKET_LOST:
		    {

		        wxPrintf("JOB  : Socket Disconnected!!\n");
		        sock->Destroy();
		        ExitMainLoop();
		        return;

		        break;
		    }
		    default: ;
		  }

}


void MyApp::OnControllerSocketEvent(wxSocketEvent &event)
{
	//SETUP_SOCKET_CODES

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

	  //MyDebugPrint(s);

	  // Now we process the event
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT:
	    {
	      // We disable input events, so that the test doesn't trigger
	      // wxSocketEvent again.
	      sock->SetNotify(wxSOCKET_LOST_FLAG);
	      sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

	      /*
	      if (memcmp(socket_input_buffer, socket_ready_to_send_job_package, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
		  {
	    	  //SPACER DELETE

		  }*/

	      if (memcmp(socket_input_buffer, socket_time_to_die, SOCKET_CODE_SIZE) == 0) // we are connected to the relevant gui panel.
		  {
	    	  // tell any connected slaves to die. then exit..

	    	  for (int counter = 0; counter < number_of_connected_slaves; counter++)
	    	  {
	    		  if (slave_sockets[counter] != NULL)
	    		  {
	    			  slave_sockets[counter]->WriteMsg(socket_time_to_die, SOCKET_CODE_SIZE);
	    			  slave_sockets[counter]->Destroy();
	  	    		  slave_sockets[counter] = NULL;

	    		  }

	    	  }

	    	  ExitMainLoop();
	    	  return;

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
	        return;

	        break;
	    }
	    default: ;
	  }

}

void MyApp::SendNextJobTo(wxSocketBase *socket)
{
	//SETUP_SOCKET_CODES

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

	//SETUP_SOCKET_CODES

	// send job finished code
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_job_finished, SOCKET_CODE_SIZE);
	// send the job number of the current job..
	controller_socket->WriteMsg(&job_number, 4);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);


}

void MyApp::SendJobResult(JobResult *result)
{
	MyDebugAssertTrue(i_am_the_master == true, "SendJobResult called by a slave!");

	//SETUP_SOCKET_CODES

	// sendjobresultcode

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_job_result, SOCKET_CODE_SIZE);
	result->SendToSocket(controller_socket);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}

void MyApp::SendIntermediateResult(JobResult *result)
{
	MyDebugAssertTrue(i_am_the_master == false, "SendIntermediateResult called by master!");

	//SETUP_SOCKET_CODES

	//wxPrintf("Sending int. Job Slave (%f)\n", result->result_data[0]);
	// sendjobresultcode

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_job_result, SOCKET_CODE_SIZE);
	result->SendToSocket(controller_socket);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}

void MyApp::SendAllJobsFinished()
{
	MyDebugAssertTrue(i_am_the_master == true, "SendAllJobsFinished called by a slave!");

	//SETUP_SOCKET_CODES

	// get the next job..
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_all_jobs_finished, SOCKET_CODE_SIZE);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}

void MyApp::OnSlaveSocketEvent(wxSocketEvent &event)
{
	//SETUP_SOCKET_CODES

	wxString s = _("JOB MASTER: OnSlaveSocketEvent: ");
	wxSocketBase *sock = event.GetSocket();

	 float *result;

	//MyDebugAssertTrue(sock == controller_socket, "Master Socket event from Non controller socket??");

	// First, print a message
	switch(event.GetSocketEvent())
	{
	   case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	   case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	   default             : s.Append(_("Unexpected event !\n")); break;
	}

	//MyDebugPrint(s);

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
				 //MyDebugPrint("JOB MASTER : SEND NEXT JOB");

				 // if there is a result to send on, get it..

				 JobResult temp_result;
		    	 temp_result.ReceiveFromSocket(sock);
		    	 SendNextJobTo(sock);

		    	 // Send info that the job has finished, and if necessary the result..

		    	 if (temp_result.job_number != -1)
		    	 {
		    		 if (temp_result.result_size > 0)
		    		 {
		    			 SendJobResult(&temp_result);
		    		 }
		    		 else // just say job finished..
		    		 {
		    			 SendJobFinished(temp_result.job_number);
		    		 }

		    		 number_of_finished_jobs++;
		    		 my_job_package.jobs[temp_result.job_number].has_been_run = true;

		    		 if (number_of_finished_jobs == my_job_package.number_of_jobs)
		    		 {
		    			 SendAllJobsFinished();

		    			 //wxPrintf("Sending all jobs finished\n");

		    			 if (my_job_package.ReturnNumberOfJobsRemaining() != 0)
		    			 {
		    				 SocketSendError("All jobs should be finished, but job package is not empty.");
		    			 }

		    		   	 // time to die!

		    			 controller_socket->Destroy();
		    			 ExitMainLoop();
		    			 return;

		    		 }
		    	 }
			 }
			 else
			 if (memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0) // identification
			 {
				 // got an error message..
				 //MyDebugPrint("JOB MASTER : Error Message");
				wxString error_message;

				error_message = ReceivewxStringFromSocket(sock);

				// send the error message up the chain..

				SocketSendError(error_message);
			 }
			 else
			 if (memcmp(socket_input_buffer, socket_i_have_info, SOCKET_CODE_SIZE) == 0) // identification
			 {
				 // got an error message..
				 //MyDebugPrint("JOB MASTER : Error Message");
				wxString info_message;

				info_message = ReceivewxStringFromSocket(sock);

				// send the error message up the chain..

				SocketSendInfo(info_message);
			 }
			 if (memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0) // identification
			 {
				 JobResult temp_result;
			     temp_result.ReceiveFromSocket(sock);
			     //wxPrintf("Sending int. Job Master (%f)\n", temp_result.result_data[0]);
			     SendJobResult(&temp_result);
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

		     //wxPrintf("JOB Master : a slave socket Disconnected!!\n");

		     sock->Destroy();
		     //ExitMainLoop();
		     //abort();

		     break;
		  }
		 default: ;
	}


}

void MyApp::SetupServer()
{

	wxIPV4address my_address;
	wxIPV4address buffer_address;

	//MyDebugPrint("Setting up Server...");

	for (short int current_port = START_PORT; current_port <= END_PORT; current_port++)
	{

		if (current_port == END_PORT)
		{
			wxPrintf("JOB MASTER : Could not find a valid port !\n\n");
			ExitMainLoop();
			return;
		}

		my_port = current_port;
		my_address.Service(my_port);

		socket_server = new wxSocketServer(my_address);
		socket_server->SetFlags(wxSOCKET_BLOCK);

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

void MyApp::OnConnectionTimer(wxTimerEvent& event)
{
	CheckForConnections();
}

void MyApp::OnZombieTimer(wxTimerEvent& event)
{
	if (i_am_a_zombie == true)
	{
		// hmm..

		ExitMainLoop();
	}

}

void MyApp::OnServerEvent(wxSocketEvent& event) // this should only be called by the master
{
	 MyDebugAssertTrue(i_am_the_master == true, "Server event called by non master!");
	 CheckForConnections();

}

void MyApp::CheckForConnections()
{
	// SETUP_SOCKET_CODES

	 wxSocketBase *sock = NULL;

	 while (1==1)
	 {
		 sock = socket_server->Accept(false);

		 if (sock == NULL) break;

		 sock->SetFlags(wxSOCKET_BLOCK);//|wxSOCKET_BLOCK);
		 sock->WriteMsg(socket_please_identify, SOCKET_CODE_SIZE);
		 sock->WaitForRead(5);

		 if (sock->IsData() == true)
		 {
		   	  sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

		   	  // does this correspond to our job code?

		   	  if ((memcmp(socket_input_buffer, job_code, SOCKET_CODE_SIZE) != 0) )
		 	  {
		 	  	  	SocketSendError("JOB MASTER : received Unknown JOB ID - Closing Connection");

		 	  	  	// incorrect identification - close the connection..
		 		    sock->Destroy();
		 		    sock = NULL;
		 	  }
		   	  else
		   	  {

		   		// A slave has connected

		   		Unbind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );
		 		Bind(wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnSlaveSocketEvent), this,  SOCKET_ID );

		   		// we need to setup events of this socket..

		   		sock->SetEventHandler(*this, SOCKET_ID);
		 		sock->SetNotify(wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
		 		sock->Notify(true);


		   		slave_sockets[number_of_connected_slaves] = sock;
		   		number_of_connected_slaves++;
		   		//SocketSendInfo(wxString::Format("slave slaves connected = %li", number_of_connected_slaves));

		   		// tell it is is connected..

		   		sock->WriteMsg(socket_you_are_connected, SOCKET_CODE_SIZE);

		 		// if we have them all we can delete the timer..

		 		if (number_of_connected_slaves == my_job_package.my_profile.ReturnTotalJobs() - 1)
		 		{
		 			connection_timer->Stop();
		 			Unbind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnConnectionTimer ), this);
		 			delete connection_timer;
		 			SocketSendInfo("All slaves have re-connected to the master.");
		 		}
		   	  }
		 }
		 else
		 {
		 		SocketSendError("JOB MASTER : ...Read Timeout waiting for job ID");
		 	 	// time out - close the connection
		 		sock->Destroy();
		 		sock = NULL;
		 }


	 }

}


void MyApp::OnMasterSocketEvent(wxSocketEvent& event)
{
	//SETUP_SOCKET_CODES

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

	//MyDebugPrint(s);

	 // if we got here, we are not a zombie..

	 i_am_a_zombie = false;
	 if (zombie_timer != NULL)
	 {
		 delete zombie_timer;
		zombie_timer = NULL;
	 }

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
				 //MyDebugPrint("JOB SLAVE : Sending Identification")
		    	 sock->WriteMsg(job_code, SOCKET_CODE_SIZE);

			 }
			 else
			 if (memcmp(socket_input_buffer, socket_you_are_connected, SOCKET_CODE_SIZE) == 0) // identification
			 {
			   	  // we are connected, request the first job..
				 is_connected = true;
				 //MyDebugPrint("JOB SLAVE : Requesting job");
				 controller_socket->WriteMsg(socket_send_next_job, SOCKET_CODE_SIZE);

				 JobResult temp_result; // dummy result for the initial request - not reallt very nice
				 temp_result.job_number = -1;
				 temp_result.result_size = 0;
				 temp_result.SendToSocket(controller_socket);


			 }
			 else
			 if (memcmp(socket_input_buffer, socket_ready_to_send_single_job, SOCKET_CODE_SIZE) == 0) // identification
			 {
				 // are we currently running a job?

				 MyDebugAssertTrue(currently_running_a_job == false, "Received a new job, when already running a job!");

				 // recieve job

				// MyDebugPrint("JOB SLAVE : About to receive new job")

				 my_current_job.RecieveJob(sock);

				 // run a thread for this job..

				 //MyDebugPrint("JOB SLAVE : New Job, starting thread")

				 //MyDebugAssertTrue(work_thread == NULL, "Running a new thread, but old thread is not NULL");

				 wxMutexLocker *lock = new wxMutexLocker(job_lock);

				 if (lock->IsOk() == true)
				 {
					 MyDebugAssertFalse(thread_next_action = THREAD_START_NEXT_JOB, "Thread action is already start job");
					 thread_next_action = THREAD_START_NEXT_JOB;

				 }
				 else
				 {
				       MyPrintWithDetails("Can't get job lock!");
				 }

				 delete lock;

				 /*
				 work_thread = new CalculateThread(this);

								 if ( work_thread->Run() != wxTHREAD_NO_ERROR )
								 {
								       MyPrintWithDetails("Can't create the thread!");
								       delete work_thread;
								       work_thread = NULL;
								       ExitMainLoop();
								       return;
								 }*/


				 //MyDebugPrint("JOB SLAVE : Started Thread");



			 }
			 else
			 if (memcmp(socket_input_buffer, socket_time_to_die, SOCKET_CODE_SIZE) == 0) // identification
			 {
			   	  // time to die!
				 wxMutexLocker *lock = new wxMutexLocker(job_lock);

				 if (lock->IsOk() == true)
				 {
					 thread_next_action = THREAD_DIE;

				 }
				 else
				 {
				       MyPrintWithDetails("Can't get job lock!");
				 }

				 delete lock;

				 sock->Destroy();
				 wxSleep(5);
			     if (work_thread != NULL) work_thread->Kill();
	 		     ExitMainLoop();
	 		     return;
			 }


			 			 // Enable input events again.

			 sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
			 break;
		 }

		 case wxSOCKET_LOST:
		 {

		     wxPrintf("JOB  : Master Socket Disconnected!!\n");
		     sock->Destroy();

		     if (work_thread != NULL) work_thread->Kill();
		     ExitMainLoop();
		     return;

		     break;
		  }
		 default: ;
	}


}

void MyApp::OnThreadComplete(wxThreadEvent& my_event)
{
	//SETUP_SOCKET_CODES

	// The compute thread is finished.. get the next job
	// thread should be dead, or nearly dead..

	//work_thread = NULL;
	SendAllResultsFromResultQueue();

	// get the next job..
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_send_next_job, SOCKET_CODE_SIZE);

	// if there is a result - send it to the gui..
	my_result.job_number = my_current_job.job_number;
	my_result.SendToSocket(controller_socket);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	// clear the results queue..




}

void MyApp::OnThreadSendError(wxThreadEvent& my_event)
{
	SocketSendError(my_event.GetString());
	//MyDebugPrint("ThreadSendError");
}

void MyApp::OnThreadSendInfo(wxThreadEvent& my_event)
{
	SocketSendInfo(my_event.GetString());
	//MyDebugPrint("ThreadSendError");
}

void MyApp::OnThreadIntermediateResultAvailable(wxThreadEvent& my_event)
{
	SendAllResultsFromResultQueue();
}

void MyApp::SendAllResultsFromResultQueue()
{
	while (1==1)
	{
		JobResult *popped_job = PopJobFromResultQueue();

		if (popped_job == NULL) break;
		else
		{
			SendIntermediateResult(popped_job);
			delete popped_job;
		}
	}
}

void MyApp::SocketSendError(wxString error_to_send)
{
	//SETUP_SOCKET_CODES

	// send the error message flag

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_i_have_an_error, SOCKET_CODE_SIZE);

	SendwxStringToSocket(&error_to_send, controller_socket);
	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}

void MyApp::SocketSendInfo(wxString info_to_send)
{
	//SETUP_SOCKET_CODES

	// send the error message flag

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
	controller_socket->WriteMsg(socket_i_have_info, SOCKET_CODE_SIZE);

	SendwxStringToSocket(&info_to_send, controller_socket);
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
	else
	{
		SocketSendError("SendError with null work thread!");
	}
}

void MyApp::SendInfo(wxString info_to_send)
{
	if (is_running_locally == true)
	{
		wxPrintf("\nInfo : %s\n", info_to_send);

	}
	else
	if (work_thread != NULL)
	{
		work_thread->QueueInfo(info_to_send);
	}
	else
	{
		SocketSendError("SendInfo with null work thread!");
	}
}

void MyApp::AddJobToResultQueue(JobResult * result_to_add)
{
	wxMutexLocker *lock = new wxMutexLocker(job_lock);

	if (lock->IsOk() == true) job_queue.Add(result_to_add);
	else
	{
		MyPrintWithDetails("Can't get job lock!");
	}

	delete lock;

	wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE);


	if (work_thread != NULL)
	{
		work_thread->MarkIntermediateResultAvailable();
	}
	else
	{
		wxPrintf("Work thread is NULL!\n");
	}


}

JobResult * MyApp::PopJobFromResultQueue()
{
	JobResult *popped_job = NULL;

	wxMutexLocker *lock = new wxMutexLocker(job_lock);

	if (lock->IsOk() == true)
	{
		if (job_queue.GetCount() > 0)
		{
			popped_job = job_queue.Detach(0);
		}
	}
	else
	{
		MyPrintWithDetails("Can't get job lock!");
	}

	delete lock;
	return popped_job;
}

// Main execution in this thread..

wxThread::ExitCode CalculateThread::Entry()
{
	int thread_action_copy;

	while (1==1)
	{
		wxMutexLocker *lock = new wxMutexLocker(main_thread_pointer->job_lock);

		if (lock->IsOk() == true)
		{
			thread_action_copy = main_thread_pointer->thread_next_action;
		}
		else
		{

		}

		if (main_thread_pointer->thread_next_action == THREAD_START_NEXT_JOB)
		{
			main_thread_pointer->thread_next_action = THREAD_SLEEP;
		}

		delete lock;

		if (thread_action_copy == THREAD_START_NEXT_JOB)
		{
			bool success = main_thread_pointer->DoCalculation(); // This should be overrided per app..
			wxThreadEvent *my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_COMPLETED);

			if (success == true) my_thread_event->SetInt(1);
			else my_thread_event->SetInt(0);
			wxQueueEvent(main_thread_pointer, my_thread_event);
		}
		else
		if (thread_action_copy == THREAD_SLEEP) wxMilliSleep(100);
		else
		if (thread_action_copy == THREAD_DIE) break;
	}

	fftwf_cleanup(); // this is needed to stop valgrind reporting memory leaks..
    return (wxThread::ExitCode)0;     // success
}

void  CalculateThread::QueueError(wxString error_to_queue)
{
	wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SENDERROR);
	test_event->SetString(error_to_queue);

	wxQueueEvent(main_thread_pointer, test_event);
}

void  CalculateThread::QueueInfo(wxString info_to_queue)
{
	wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SENDINFO);
	test_event->SetString(info_to_queue);

	wxQueueEvent(main_thread_pointer, test_event);
}

void CalculateThread::MarkIntermediateResultAvailable()
{
	wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE);

	wxQueueEvent(main_thread_pointer, test_event);
}

CalculateThread::~CalculateThread()
{
    //wxCriticalSectionLocker enter(m_pHandler->m_pThreadCS);
    // the thread is being destroyed; make sure not to leave dangling pointers around
    main_thread_pointer = NULL;
}


