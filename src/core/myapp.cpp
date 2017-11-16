#include "core_headers.h"

SETUP_SOCKET_CODES

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_ENDING, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SEND_IMAGE_RESULT, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDERROR, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDINFO, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE, wxThreadEvent);

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
	queue_timer = NULL;

	queue_timer_set = false;

	currently_running_a_job = false;

	time_of_last_queue_send = 0;
	number_of_results_sent = 0;

	total_milliseconds_spent_on_threads = 0;

	wxString current_address;
	wxArrayString possible_controller_addresses;
	wxIPV4address junk_address;


	// Bind the thread events

	Bind(wxEVT_COMMAND_MYTHREAD_COMPLETED, &MyApp::OnThreadComplete, this); // Called when DoCalculation finishes
	Bind(wxEVT_COMMAND_MYTHREAD_SENDERROR, &MyApp::OnThreadSendError, this);
	Bind(wxEVT_COMMAND_MYTHREAD_SENDINFO, &MyApp::OnThreadSendInfo, this);
	Bind(wxEVT_COMMAND_MYTHREAD_ENDING, &MyApp::OnThreadEnding, this); // When thread is about to die
	Bind(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE, &MyApp::OnThreadIntermediateResultAvailable, this);
	Bind(wxEVT_COMMAND_MYTHREAD_SEND_IMAGE_RESULT, &MyApp::OnThreadSendImageResult, this);


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
		stopwatch.Start();
		DoCalculation();
		total_milliseconds_spent_on_threads += stopwatch.Time();
		fftwf_cleanup(); // this is needed to stop valgrind reporting memory leaks..
		exit(0);
	}
	else
	if (number_of_arguments != 3)
	{
		command_line_parser.Usage();
		wxPrintf("\n\n");
		return false;
	}

	is_running_locally = false;


	// get the address and port of the job controller (should be command line options).

	wxStringTokenizer controller_ip_address_tokens(command_line_parser.GetParam(0),",");

	while(controller_ip_address_tokens.HasMoreTokens() == true)
	{
		current_address = controller_ip_address_tokens.GetNextToken();
		possible_controller_addresses.Add(current_address);
		if (junk_address.Hostname(current_address) == false)
		{
			MyDebugPrint(" Error: Address (%s) - not recognized as an IP or hostname\n\n", current_address);
			exit(-1);
		};
	}


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

	active_controller_address.Service(controller_port);
	is_connected = false;

	//MyDebugPrint("\n JOB : Trying to connect to %s:%i (timeout = 30 sec) ...\n", controller_address.IPAddress(), controller_address.Service());
	controller_socket = new wxSocketClient();

	controller_socket->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK );

	// Setup the event handler and subscribe to most events

	Bind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );
	controller_socket->SetEventHandler(*this, SOCKET_ID);
	controller_socket->SetNotify(wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
	controller_socket->Notify(true);

	//wxSleep(2);

	for (counter = 0; counter < possible_controller_addresses.GetCount(); counter++)
	{
		active_controller_address.Hostname(possible_controller_addresses.Item(counter));
		controller_socket->Connect(active_controller_address, false);
		controller_socket->WaitOnConnect(4);

		if (controller_socket->IsConnected() == false)
		{
		   controller_socket->Close();
		   //wxPrintf("Connection Failed.\n\n");
		}
		else
		{
			break;
		}
	}


	controller_socket->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK );

	if (controller_socket->IsConnected() == false || controller_socket->IsOk() == false)
	{
	   controller_socket->Close();
	   MyDebugPrint(" JOB : Failed ! Unable to connect\n");
	   return false;
	}

	// we are apparently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
	// system - if the port if valid.  So if we don't get any events from this socket within 10 seconds, we are going to try again...

	number_of_failed_connections = 0;
	i_am_a_zombie = true;
	Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnZombieTimer ), this, 1);
	zombie_timer = new wxTimer(this, 1);
	zombie_timer->StartOnce(10000);

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

	MyDebugAssertTrue(sock == controller_socket, "Original Socket event from Non controller socket??");


	sock->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK);

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
			 ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);

			 if (memcmp(socket_input_buffer, socket_please_identify, SOCKET_CODE_SIZE) == 0) // identification
			 {
		    	  // send the job identification to complete the connection
				  WriteToSocket(sock, job_code, SOCKET_CODE_SIZE);

		    	  // Enable input events again
			      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
			 }
			 else
			 if (memcmp(socket_input_buffer, socket_you_are_the_master, SOCKET_CODE_SIZE) == 0) // we are the master.. so setup job control and prepare to receive connections
		     {
		    	  //MyDebugPrint("JOB  : I am the master");

		    	  i_am_the_master = true;

		    	  // we need to start a server so that the slaves can connect..

		    	  SetupServer();

		    	  // I have to send my ip address to the controller..

		    	  my_ip_address = ReturnIPAddressFromSocket(sock);

		    	  SendwxStringToSocket(&my_ip_address, sock);
		    	  SendwxStringToSocket(&my_port_string, sock);

		    	  // ok, now get the job details from the conduit controller

		    	  WriteToSocket(sock, socket_send_job_details, SOCKET_CODE_SIZE);

		    	  // it should now send a conformation code followed by the package..

		    	//  sock->WaitForRead(10);

		    	//  if (sock->IsData() == true)
		    	  //{
		    		  ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);

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
		    	 // }
		    	  //else
		    	  //{
		    	//	  MyDebugPrintWithDetails(" JOB MASTER : ...Read Timeout waiting for job package \n\n");
		    	//	  // time out - close the connection
		    	//	  sock->Destroy();
		    	//	  sock = NULL;
		    	//	  ExitMainLoop();
		    	//	  return;
		    	  //}


		    	  // Setup a timer for any straggling connections..

		    	  Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnConnectionTimer ), this, 0);
		    	  connection_timer = new wxTimer(this, 0);
		    	  connection_timer->Start(5000);

		    	  // Enable input events again
			      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
		    	  // redirect socket events appropriately

				  Unbind(wxEVT_SOCKET,wxSocketEventHandler( MyApp::OnOriginalSocketEvent), this,  SOCKET_ID );
				  Bind(wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnControllerSocketEvent), this,  SOCKET_ID );

				  // queue timer event

				  Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnQueueTimer ), this, 2);


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

				  controller_socket->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK );

				  active_controller_address.Hostname(master_ip_address);
				  active_controller_address.Service(master_port);

				  // Setup the event handler and subscribe to most events
				  controller_socket->SetEventHandler(*this, SOCKET_ID);
				  controller_socket->SetNotify(wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
				  controller_socket->Notify(true);

				  controller_socket->Connect(active_controller_address, false);
				  controller_socket->WaitOnConnect(10);

				  controller_socket->SetFlags( wxSOCKET_WAITALL | wxSOCKET_BLOCK );

				  if (controller_socket->IsConnected() == false)
				  {
					   controller_socket->Close();
					   MyDebugPrint("JOB : Failed ! Unable to connect\n");

				  }

                  // Start the worker thread..
				  stopwatch.Start();
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


	  sock->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK);

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
	      ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);

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
	    			  long milliseconds_spent_on_slave_thread;
	    			  slave_sockets[counter]->SetNotify(false);
	    			  WriteToSocket(slave_sockets[counter], socket_time_to_die, SOCKET_CODE_SIZE);
	    			  ReadFromSocket(slave_sockets[counter], &milliseconds_spent_on_slave_thread, sizeof(long));
	    			  total_milliseconds_spent_on_threads += milliseconds_spent_on_slave_thread;

	    			  slave_sockets[counter]->Destroy();
	  	    		  slave_sockets[counter] = NULL;

	    		  }

	    	  }

	    	  sock->Destroy();
	    	  ExitMainLoop();
	    	  exit(0);

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

	        // tell the slaves to die..


	    	for (int counter = 0; counter < number_of_connected_slaves; counter++)
	    	{
	    	  if (slave_sockets[counter] != NULL)
	    	  {
	    		  long milliseconds_spent_on_slave_thread;
	    		  slave_sockets[counter]->SetNotify(false);
	    		  WriteToSocket(slave_sockets[counter], socket_time_to_die, SOCKET_CODE_SIZE);
	    		  ReadFromSocket(slave_sockets[counter], &milliseconds_spent_on_slave_thread, sizeof(long));
	    		  total_milliseconds_spent_on_threads += milliseconds_spent_on_slave_thread;

	    		  slave_sockets[counter]->Destroy();
	  	    	  slave_sockets[counter] = NULL;
    		  }

	    	}

	        ExitMainLoop();
	        exit(0);
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
		WriteToSocket(socket, socket_time_to_die, SOCKET_CODE_SIZE);

		// Receive timing for that slave before its thread dies
		long milliseconds_spent_on_slave_thread;
		ReadFromSocket(socket, &milliseconds_spent_on_slave_thread, sizeof(long));
		total_milliseconds_spent_on_threads += milliseconds_spent_on_slave_thread;
		socket->Destroy();
		socket = NULL;

	}
}

void MyApp::SendJobFinished(int job_number)
{
	MyDebugAssertTrue(i_am_the_master == true, "SendJobFinished called by a slave!");

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(controller_socket, socket_job_finished, SOCKET_CODE_SIZE);
	// send the job number of the current job..
	WriteToSocket(controller_socket, &job_number, 4);

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);


}

void MyApp::SendJobResult(JobResult *result)
{
	MyDebugAssertTrue(i_am_the_master == true, "SendJobResult called by a slave!");

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(controller_socket, socket_job_result, SOCKET_CODE_SIZE);
	result->SendToSocket(controller_socket);

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
}

void MyApp::SendJobResultQueue(ArrayofJobResults &queue_to_send)
{
	MyDebugAssertTrue(i_am_the_master == true, "SendJobResultQueue called by a slave!");

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(controller_socket, socket_job_result_queue, SOCKET_CODE_SIZE);
	SendResultQueueToSocket(controller_socket, queue_to_send);

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
}

void MyApp::MasterSendIntenalQueue()
{
//	wxPrintf("(Master) Sending internal queue of %li jobs (%i), sending on to controller\n", job_queue.GetCount(), job_queue.Item(0).job_number);
	SendJobResultQueue(job_queue);
	job_queue.Clear();
	time_of_last_queue_send = time(NULL);
    //wxPrintf("(Master) Queue sent\n");

}


void MyApp::SendAllJobsFinished()
{
	MyDebugAssertTrue(i_am_the_master == true, "SendAllJobsFinished called by a slave!");

	 // we will send all jobs finished - but first we need to ensure we have sent any results in the result queue

	if (job_queue.GetCount() != 0) MasterSendIntenalQueue();

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(controller_socket, socket_all_jobs_finished, SOCKET_CODE_SIZE);
	WriteToSocket(controller_socket, &total_milliseconds_spent_on_threads, sizeof(long));

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

}


// This is when running on the master. It handles events on the socket connected to the slaves
void MyApp::OnSlaveSocketEvent(wxSocketEvent &event)
{

	wxString s = _("JOB MASTER: OnSlaveSocketEvent: ");
	wxSocketBase *sock = event.GetSocket();

	sock->SetFlags( wxSOCKET_WAITALL | wxSOCKET_BLOCK);

	float *result;

	// First, print a message
	/*	switch(event.GetSocketEvent())
	{
	   case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	   case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	   default             : s.Append(_("Unexpected event !\n")); break;
	}*/

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

		if (memcmp(socket_input_buffer, socket_send_next_job, SOCKET_CODE_SIZE) == 0) // identification
		{
			// MyDebugPrint("JOB MASTER : SEND NEXT JOB");

			// if there is a result to send on, get it..
			JobResult temp_result;
			temp_result.ReceiveFromSocket(sock);

			// Send the next job
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

					//wxPrintf("Sending all jobs finished (%li / %i)\n", number_of_finished_jobs, my_job_package.number_of_jobs);

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

			if (sock == NULL) return;
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
		else
		if (memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0) // identification
		{
			JobResult temp_result;
			temp_result.ReceiveFromSocket(sock);
			// wxPrintf("Sending int. Job Master (%f)\n", temp_result.result_data[0]);
			SendJobResult(&temp_result);
		}
		else
		if (memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0) // identification
		{
			ArrayofJobResults temp_array;
			//wxPrintf("(Master) Receieved socket_job_result_queue - receiving queue from slave\n");
			ReceiveResultQueueFromSocket(sock, temp_array);

			// copy these results to our own result queue

			for (int counter = 0; counter < temp_array.GetCount(); counter++)
			{
				job_queue.Add(temp_array.Item(counter));
			}

			if (queue_timer_set == false)
			{
				queue_timer_set = true;
				queue_timer = new wxTimer(this, 2);
				queue_timer->StartOnce(1000);
			}
		}
		else
		if (memcmp(socket_input_buffer, socket_result_with_image_to_write, SOCKET_CODE_SIZE) == 0) // identification
		{
			Image image_to_write;
			int details[3];

			ReadFromSocket(sock, details, sizeof(int) * 3);
			image_to_write.Allocate(details[0], details[1], 1);
			int position_in_stack = details[2];
			ReadFromSocket(sock, image_to_write.real_values, image_to_write.real_memory_allocated * sizeof(float));
			wxString filename_to_write;
			filename_to_write = ReceivewxStringFromSocket(sock);

			if (master_output_file.IsOpen() == false || master_output_file.filename != filename_to_write)
			{
				master_output_file.OpenFile(filename_to_write.ToStdString(), true);
				image_to_write.WriteSlice(&master_output_file, 1); // to setup the file..
			}

			image_to_write.WriteSlice(&master_output_file, position_in_stack);

			float temp_float;
			temp_float = position_in_stack;

			JobResult job_to_queue;
			job_to_queue.SetResult(1, &temp_float);
			job_queue.Add(job_to_queue);


			if (queue_timer_set == false)
			{
				queue_timer_set = true;
				queue_timer = new wxTimer(this, 2);
				queue_timer->StartOnce(1000);
			}
		}

		/*
		else
		if (memcmp(socket_input_buffer, socket_send_timing_for_thread, SOCKET_CODE_SIZE) == 0) // identification
		{
			long milliseconds_spent_on_slave_thread;
			ReadFromSocket(sock, &milliseconds_spent_on_slave_thread, sizeof(long));
			total_milliseconds_spent_on_threads += milliseconds_spent_on_slave_thread;
			wxPrintf("master received this timing: %ld\n",milliseconds_spent_on_slave_thread);
		}
		*/




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

		socket_server->SetFlags( wxSOCKET_WAITALL | wxSOCKET_BLOCK );

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
		number_of_failed_connections++;

		if (number_of_failed_connections >= 5) ExitMainLoop();

		controller_socket->Close();
		controller_socket->Connect(active_controller_address, false);
		controller_socket->WaitOnConnect(120);

		if (controller_socket->IsConnected() == false)
		{
		   controller_socket->Close();
		   //wxPrintf("Connection Failed.\n\n");
		}

		controller_socket->SetFlags(wxSOCKET_WAITALL | wxSOCKET_BLOCK );

		if (controller_socket->IsConnected() == false || controller_socket->IsOk() == false)
		{
		   controller_socket->Close();
		   MyDebugPrint(" JOB : Failed ! Unable to connect\n");
		   ExitMainLoop();
		}

		// once again, we are parently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
		// system - if the port if valid.  So if we don't get any events from this socket within 10 seconds, we are going to try again...

		zombie_timer = new wxTimer(this, 1);
		zombie_timer->StartOnce(10000);
	}
}

void MyApp::OnQueueTimer(wxTimerEvent& event)
{
	//wxPrintf("Queue timer fired\n");
	if (job_queue.GetCount() > 0)
	{
	//	wxPrintf("sending from timer\n");
		MasterSendIntenalQueue();
	}
	queue_timer_set = false;
	delete queue_timer;
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


		 sock->SetFlags( wxSOCKET_WAITALL | wxSOCKET_BLOCK );
		 WriteToSocket(sock, socket_please_identify, SOCKET_CODE_SIZE);

//		 sock->WaitForRead(5);

	//	 if (sock->IsData() == true)
	//	 {
			  ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);

		   	  // does this correspond to our job code?

		   	  if ((memcmp(socket_input_buffer, job_code, SOCKET_CODE_SIZE) != 0) )
		 	  {
		 	  	  	SocketSendError("Received Unknown Job ID (leftover process from a cancelled job?)- Closing Connection");

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
		   		WriteToSocket(sock, socket_you_are_connected, SOCKET_CODE_SIZE);

		 		// if we have them all we can delete the timer..

		   		int number_of_commands_to_run;
	   			if (my_job_package.number_of_jobs + 1 < my_job_package.my_profile.ReturnTotalJobs()) number_of_commands_to_run = my_job_package.number_of_jobs + 1;
	   			else number_of_commands_to_run = my_job_package.my_profile.ReturnTotalJobs();

		 		if (number_of_connected_slaves == number_of_commands_to_run - 1)
		 		{
		 			connection_timer->Stop();
		 			Unbind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnConnectionTimer ), this);
		 			delete connection_timer;
		 			SocketSendInfo("All slaves have re-connected to the master.");
		 		}
		   	  }
		// }
		 //else
		 //{
		 	//	SocketSendError("JOB MASTER : ...Read Timeout waiting for job ID");
		 	// 	// time out - close the connection
		 	//	sock->Destroy();
		 	//	sock = NULL;
		// }


	 }

}


void MyApp::OnMasterSocketEvent(wxSocketEvent& event)
{
	//SETUP_SOCKET_CODES

	wxString s = _("JOB : OnMasterSocketEvent: ");
	wxSocketBase *sock = event.GetSocket();

	sock->SetFlags( wxSOCKET_WAITALL | wxSOCKET_BLOCK );

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
			 ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);

			 if (memcmp(socket_input_buffer, socket_please_identify, SOCKET_CODE_SIZE) == 0) // identification
			 {
		    	  // send the job identification to complete the connection
				 //MyDebugPrint("JOB SLAVE : Sending Identification")
				 WriteToSocket(sock, job_code, SOCKET_CODE_SIZE);
			 }
			 else
			 if (memcmp(socket_input_buffer, socket_you_are_connected, SOCKET_CODE_SIZE) == 0) // identification
			 {
			   	  // we are connected, request the first job..
				 is_connected = true;
				 //MyDebugPrint("JOB SLAVE : Requesting job");
				 WriteToSocket(sock, socket_send_next_job, SOCKET_CODE_SIZE);

				 JobResult temp_result; // dummy result for the initial request - not reallt very nice
				 temp_result.job_number = -1;
				 temp_result.result_size = 0;
				 temp_result.SendToSocket(sock);


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
					 SocketSendError("Job Lock Error!");
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

				 // Timing stuff here
				long milliseconds_spent_by_thread = stopwatch.Time();
				controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
				WriteToSocket(controller_socket, &milliseconds_spent_by_thread, sizeof(long));
				controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

			   	  // time to die!
				 wxMutexLocker *lock = new wxMutexLocker(job_lock);

				 if (lock->IsOk() == true)
				 {
					 thread_next_action = THREAD_DIE;

				 }
				 else
				 {
					 SocketSendError("Job Lock Error!");
				     MyPrintWithDetails("Can't get job lock!");
				 }

				 delete lock;

				 // give the thread some time to die..
				 wxSleep(5);
				 // process thread events in case it has done something
				 Yield(); //(wxEVT_CATEGORY_THREAD);
				 // finish
				 sock->Destroy();
				 if (work_thread != NULL) work_thread->Kill();
	 		     ExitMainLoop();
	 		     exit(0);
	 		     return;
			 }


			 			 // Enable input events again.

			 sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
			 break;
		 }

		 case wxSOCKET_LOST:
		 {

		     wxPrintf("JOB  : Master Socket Disconnected!!\n");
			 wxMutexLocker *lock = new wxMutexLocker(job_lock);

			 if (lock->IsOk() == true)
			 {
				 thread_next_action = THREAD_DIE;
			 }
			 else
			 {
				 SocketSendError("Job Lock Error!");
				 MyPrintWithDetails("Can't get job lock!");
			 }

			 delete lock;

			 // give the thread some time to die..
			 wxSleep(5);

			 // process thread events in case it has done something
			 Yield(); //(wxEVT_CATEGORY_THREAD);

			 // finish
			 sock->Destroy();
			 if (work_thread != NULL) work_thread->Kill();
			 ExitMainLoop();
			 exit(0);
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
	WriteToSocket(controller_socket, socket_send_next_job, SOCKET_CODE_SIZE);

	// if there is a result - send it to the gui..
	my_result.job_number = my_current_job.job_number;
	my_result.SendToSocket(controller_socket);

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	// clear the results queue..
}

void MyApp::OnThreadEnding(wxThreadEvent& my_event)
{
	SendAllResultsFromResultQueue();

	work_thread = NULL;
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
//	wxPrintf("MyApp::Received result available event..\n");
	SendAllResultsFromResultQueue();
}

void MyApp::OnThreadSendImageResult(wxThreadEvent& my_event)
{
	MyDebugAssertTrue(i_am_the_master == false, "OnThreadSendImageResult called by master!");

	Image image_to_send;
	image_to_send = my_event.GetPayload<Image>();
	int position_in_stack = my_event.GetInt();
	wxString filename_to_write = my_event.GetString();
	int details[3];

	details[0] = image_to_send.logical_x_dimension;
	details[1] = image_to_send.logical_y_dimension;
	details[2] = position_in_stack;

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG);

	WriteToSocket(controller_socket, socket_result_with_image_to_write, SOCKET_CODE_SIZE);
	WriteToSocket(controller_socket, details, sizeof(int) * 3);
	WriteToSocket(controller_socket, image_to_send.real_values, image_to_send.real_memory_allocated * sizeof(float));
	SendwxStringToSocket(&filename_to_write, controller_socket);

	controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
}

void MyApp::SendAllResultsFromResultQueue()
{
	// have we sent results within the last second? if so wait 1s

	ArrayofJobResults my_queue_array;

	// we want to pop off all the jobs, and send them in one big lump..

	while (1==1)
	{
		JobResult *popped_job = PopJobFromResultQueue();

		if (popped_job == NULL)
		{
			break;
		}
		else
		{
			my_queue_array.Add(*popped_job);
			delete popped_job;
		}
	}

	// ok, send them all..

	if (my_queue_array.GetCount() > 0)
	{
		if (time(NULL) - time_of_last_queue_send < 1)
		{
			wxSleep(1);
		}

		SendIntermediateResultQueue(my_queue_array);
		time_of_last_queue_send = time(NULL);
	}


}


void MyApp::SendIntermediateResultQueue(ArrayofJobResults &queue_to_send)
{
	MyDebugAssertTrue(i_am_the_master == false, "SendIntermediateResultQueue called by master!");

	if (queue_to_send.GetCount() > 0)
	{
		controller_socket->SetNotify(wxSOCKET_LOST_FLAG);


		WriteToSocket(controller_socket, socket_job_result_queue, SOCKET_CODE_SIZE);
		SendResultQueueToSocket(controller_socket, queue_to_send);

		controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	}


}

void MyApp::SocketSendError(wxString error_to_send)
{
	// send the error message flag

	if (is_running_locally == false)
	{
		controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
		WriteToSocket(controller_socket, socket_i_have_an_error, SOCKET_CODE_SIZE);

		SendwxStringToSocket(&error_to_send, controller_socket);
		controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	}
}

void MyApp::SocketSendInfo(wxString info_to_send)
{
	// send the info message flag

	if (is_running_locally == false)
	{
		controller_socket->SetNotify(wxSOCKET_LOST_FLAG);
		WriteToSocket(controller_socket, socket_i_have_info, SOCKET_CODE_SIZE);

		SendwxStringToSocket(&info_to_send, controller_socket);
		controller_socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	}

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

void MyApp::SendErrorAndCrash(wxString error_to_send)
{
	SendError(error_to_send);
	if (!is_running_locally) wxSleep(2); // wait for the main thread to actually send the error
	abort();
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
//	wxPrintf("Adding Job to result Queue\n");
	wxMutexLocker *lock = new wxMutexLocker(job_lock);

	if (lock->IsOk() == true) job_queue.Add(result_to_add);
	else
	{
		SocketSendError("Job Lock Error!");
		MyPrintWithDetails("Can't get job lock!");
	}

	delete lock;

	if (work_thread != NULL)
	{
//		wxPrintf("MyApp::Marking Intermediate Result Available...\n");
		work_thread->MarkIntermediateResultAvailable();
	}
	else
	{
		wxPrintf("Work thread is NULL!\n");
	}
}

void MyApp::SendProcessedImageResult(Image *image_to_send, int position_in_stack, wxString filename_to_save)
{
	if (work_thread != NULL)
	{
		work_thread->SendProcessedImageResult(image_to_send, position_in_stack, filename_to_save);
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
		SocketSendError("Job Lock Error!");
		MyPrintWithDetails("Can't get job lock!");
	}

	delete lock;
	return popped_job;
}

// Main execution in this thread..

wxThread::ExitCode CalculateThread::Entry()
{
	int thread_action_copy;
	long millis_sleeping = 0;

	while (1==1)
	{
		wxMutexLocker *lock = new wxMutexLocker(main_thread_pointer->job_lock);

		if (lock->IsOk() == true)
		{
			thread_action_copy = main_thread_pointer->thread_next_action;
		}
		else
		{
			QueueError("Job Lock Error!");
		}

		if (main_thread_pointer->thread_next_action == THREAD_START_NEXT_JOB)
		{
			main_thread_pointer->thread_next_action = THREAD_SLEEP;
			millis_sleeping = 0;
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
		if (thread_action_copy == THREAD_SLEEP)
		{
			wxMilliSleep(100);
			millis_sleeping += 100;

			if (millis_sleeping > 10000)
			{
				// we have been waiting for 10 seconds, something probably went wrong - so die.
				wxPrintf("Calculation thread has been waiting for something to do for 10 seconds - going to finish\n");
				QueueError("Calculation thread has been waiting for something to do for 10 seconds - going to finish");
				break;
			}
		}
		else
		if (thread_action_copy == THREAD_DIE) break;
	}

	wxThreadEvent *my_thread_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_ENDING);
	wxQueueEvent(main_thread_pointer, my_thread_event);


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
	//wxPrintf("CalculateThread::Queueing Result Available Event..\n");
}

void CalculateThread::SendProcessedImageResult(Image *image_to_send, int position_in_stack, wxString filename_to_save)
{
	wxThreadEvent *test_event = new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_SEND_IMAGE_RESULT);
	test_event->SetInt(position_in_stack);
	test_event->SetString(filename_to_save);
	test_event->SetPayload(*image_to_send);
	wxQueueEvent(main_thread_pointer, test_event);
}

CalculateThread::~CalculateThread()
{
    //wxCriticalSectionLocker enter(m_pHandler->m_pThreadCS);
    // the thread is being destroyed; make sure not to leave dangling pointers around

	wxMutexLocker *lock = new wxMutexLocker(main_thread_pointer->job_lock);

	if (lock->IsOk() == true)
	{
		main_thread_pointer->work_thread = NULL;
	}
	else
	{
		QueueError("Job Lock Error!");
	}


	delete lock;

	main_thread_pointer = NULL;

}


