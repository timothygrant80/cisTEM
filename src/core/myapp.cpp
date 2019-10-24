#include "core_headers.h"
#include <wx/evtloop.h>

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_ENDING, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SEND_IMAGE_RESULT, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDERROR, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDINFO, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_INTERMEDIATE_RESULT_AVAILABLE, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SEND_PROGRAM_DEFINED_RESULT, ReturnProgramDefinedResultEvent);

#define THREAD_START_NEXT_JOB 0
#define THREAD_DIE 1
#define THREAD_SLEEP 2

bool MyApp::OnInit()
{
	long counter;
	thread_next_action = THREAD_SLEEP;

	number_of_dispatched_jobs = 0;
	number_of_finished_jobs = 0;
	number_of_timing_results_received = 0;

	max_number_of_connected_slaves = 0;

	zombie_timer = NULL;
	queue_timer = NULL;
	queue_timer_set = false;

	controller_socket = NULL;
	master_socket = NULL;

	connected_to_the_master = false;
	currently_running_a_job = false;

	time_of_last_queue_send = 0;
	time_of_last_master_queue_send = 0;
	number_of_results_sent = 0;

	total_milliseconds_spent_on_threads = 0;

	ProgramSpecificInit();

	return true;
}

int MyApp::OnExit()
{
	ProgramSpecificCleanUp();
	return 0;
}

void MyApp::OnEventLoopEnter(wxEventLoopBase *	loop)
{
	if (loop->IsMain() == true)
	{
		// initialise sockets, and set the event handler for SocketCommunicator

		wxSocketBase::Initialize();
		brother_event_handler = this;

		int parse_status;
		int number_of_arguments;
		int counter;
		long temp_long;

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
		Bind(wxEVT_COMMAND_MYTHREAD_SEND_PROGRAM_DEFINED_RESULT, &MyApp::OnThreadSendProgramDefinedResult, this);

		// Connect to the controller program..

		command_line_parser.SetCmdLine(argc,argv);
		command_line_parser.AddParam("controller_address",wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL);
		command_line_parser.AddParam("controller_port",wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL);
		command_line_parser.AddParam("job_code",wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL);
		command_line_parser.AddParam("wanted_number_of_threads",wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL);

		// Let the app add options
		AddCommandLineOptions();

		//wxPrintf("\n");

		parse_status = command_line_parser.Parse(true);
		number_of_arguments = command_line_parser.GetParamCount();

		if (parse_status != 0)
		{
			wxPrintf("\n\n");
			ExitMainLoop();
			exit(0);
			return;
		}

		// if we have no arguments run interactively.. if we have 4 continue as though we have network info, else error..

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
		if (number_of_arguments != 4)
		{
			command_line_parser.Usage();
			wxPrintf("\n\n");
			ExitMainLoop();
			exit(0);
			return;
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

		if (command_line_parser.GetParam(3).ToLong(&temp_long) == false)
		{
			MyPrintWithDetails(" Error: No. of Threads (%s) - not recognized as a number\n\n", command_line_parser.GetParam(3));
			exit(-1);
		}

		number_of_threads_requested_on_command_line = temp_long;
		if (number_of_threads_requested_on_command_line < 1) number_of_threads_requested_on_command_line = 1;

		// copy over job code.

		for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
		{
			current_job_code[counter] = command_line_parser.GetParam(2).GetChar(counter);
		}


		// Attempt to connect to the controller..

		active_controller_address.Service(controller_port);
		is_connected = false;

		//MyDebugPrint("\n JOB : Trying to connect to %s:%i (timeout = 30 sec) ...\n", controller_address.IPAddress(), controller_address.Service());
		controller_socket = new wxSocketClient();
		controller_socket->SetFlags(SOCKET_FLAGS);
		controller_socket->Notify(false);


		for (counter = 0; counter < possible_controller_addresses.GetCount(); counter++)
		{
			active_controller_address.Hostname(possible_controller_addresses.Item(counter));
			controller_socket->Connect(active_controller_address, false);
			controller_socket->WaitOnConnect(30);

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

		controller_socket->SetFlags(SOCKET_FLAGS);

		if (controller_socket->IsConnected() == false || controller_socket->IsOk() == false)
		{
			controller_socket->Close();
			MyDebugPrint(" JOB : Failed ! Unable to connect\n");
			ExitMainLoop();
			exit(0);
			return;
		}

		// Monitor this connection..

		MonitorSocket(controller_socket);

		// we are apparently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
		// system - if the port if valid.  So if we don't get any events from this socket within 10 seconds, we are going to try again...

		number_of_failed_connections = 0;
		i_am_a_zombie = true;

		zombie_timer = new wxTimer(this, 1);
		zombie_timer->StartOnce(20000);
		
		// timer events

		Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnQueueTimer ), this, 2);
		Bind(wxEVT_TIMER, wxTimerEventHandler( MyApp::OnZombieTimer ), this, 1);		
		
		
	}
}

// Placeholder (to be overridden) function to add options to the command line
void MyApp::AddCommandLineOptions( )
{
	return;
}


void MyApp::SendNextJobTo(wxSocketBase *socket)
{
	// if we haven't dispatched all jobs yet, then send it, otherwise tell the slave to die..

	if (number_of_dispatched_jobs < current_job_package.number_of_jobs)
	{
		current_job_package.jobs[number_of_dispatched_jobs].SendJob(socket);
		number_of_dispatched_jobs++;
	}
	else
	{
		WriteToSocket(socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
		// stop monitoring the socket..
		//StopMonitoringSocket(socket); stopped doing this for timings

	}
}

void MyApp::SendJobFinished(int job_number)
{
	//MyDebugAssertTrue(i_am_the_master == true, "SendJobFinished called by a slave!");

	WriteToSocket(controller_socket, socket_job_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	// send the job number of the current job..
	WriteToSocket(controller_socket, &job_number, sizeof(int), true, "SendJobNumber", FUNCTION_DETAILS_AS_WXSTRING);
}

void MyApp::SendJobResult(JobResult *result)
{
	//MyDebugAssertTrue(i_am_the_master == true, "SendJobResult called by a slave!");

	WriteToSocket(controller_socket, socket_job_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	result->SendToSocket(controller_socket);
}

void MyApp::SendJobResultQueue(ArrayofJobResults &queue_to_send)
{
	//MyDebugAssertTrue(i_am_the_master == true, "SendJobResultQueue called by a slave!");

	WriteToSocket(controller_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	SendResultQueueToSocket(controller_socket, queue_to_send);
}

void MyApp::MasterSendIntenalQueue()
{
	SendJobResultQueue(master_job_queue);
	master_job_queue.Clear();
	time_of_last_master_queue_send = time(NULL);
}

void MyApp::SendAllJobsFinished()
{
	//MyDebugAssertTrue(i_am_the_master == true, "SendAllJobsFinished called by a slave!");

	// we will send all jobs finished - but first we need to ensure we have sent any results in the result queue
	// wait for 5 seconds to give slaves times to send in their last jobs..

	wxSleep(1);
	Yield();

	if (master_job_queue.GetCount() != 0) MasterSendIntenalQueue();

	WriteToSocket(controller_socket, socket_all_jobs_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(controller_socket, &total_milliseconds_spent_on_threads, sizeof(long), true, "SendTotalMillisecondsSpentOnThreads", FUNCTION_DETAILS_AS_WXSTRING);
}


void MyApp::OnZombieTimer(wxTimerEvent& event)
{
	if (i_am_a_zombie == true)
	{
		number_of_failed_connections++;

		if (number_of_failed_connections >= 5) ExitMainLoop();

		if (connected_to_the_master == true)
		{
			master_socket->Close();
			master_socket->Connect(active_controller_address, false);
			master_socket->WaitOnConnect(30);

			if (master_socket->IsConnected() == false)
			{
				master_socket->Close();
			}

			master_socket->SetFlags(SOCKET_FLAGS);

			if (master_socket->IsConnected() == false || master_socket->IsOk() == false)
			{
				master_socket->Close();
				MyDebugPrint(" JOB : Failed ! Unable to connect\n");
				ExitMainLoop();
			}

			if (i_am_the_master == false) controller_socket = master_socket;
		}
		else
		{
			controller_socket->Close();
			controller_socket->Connect(active_controller_address, false);
			controller_socket->WaitOnConnect(30);

			if (controller_socket->IsConnected() == false)
			{
				controller_socket->Close();
			}

			controller_socket->SetFlags(SOCKET_FLAGS);

			if (controller_socket->IsConnected() == false || controller_socket->IsOk() == false)
			{
				controller_socket->Close();
				MyDebugPrint(" JOB : Failed ! Unable to connect\n");
				ExitMainLoop();
			}
		}
			// once again, we are aparently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
			// system - if the port if valid.  So if we don't get any events from this socket within 10 seconds, we are going to try again...

		zombie_timer = new wxTimer(this, 1);
		zombie_timer->StartOnce(20000);
	}
}

void MyApp::OnMasterQueueTimer(wxTimerEvent& event)
{
	if (master_job_queue.GetCount() > 0)
	{
		MasterSendIntenalQueue();
	}

	master_queue_timer_set = false;
	delete master_queue_timer;

}

void MyApp::OnQueueTimer(wxTimerEvent& event)
{
	SendAllResultsFromResultQueue();

	queue_timer_set = false;
	delete queue_timer;
}

void MyApp::OnThreadComplete(wxThreadEvent& my_event)
{
	//SETUP_SOCKET_CODES

	// The compute thread is finished.. get the next job
	// thread should be dead, or nearly dead..

	//work_thread = NULL;
	SendAllResultsFromResultQueue();

	// get the next job..
	WriteToSocket(master_socket, socket_send_next_job, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);

	// if there is a result - send it to the gui..
	my_result.job_number = my_current_job.job_number;
	my_result.SendToSocket(master_socket);

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
	//SendAllResultsFromResultQueue();

	if (queue_timer_set == false)
	{
		queue_timer_set = true;
		queue_timer = new wxTimer(this, 2);
		queue_timer->StartOnce(1000);
	}
}

void MyApp::OnThreadSendImageResult(wxThreadEvent& my_event)
{
	//MyDebugAssertTrue(i_am_the_master == false, "OnThreadSendImageResult called by master!");

	Image image_to_send;
	image_to_send = my_event.GetPayload<Image>();
	int position_in_stack = my_event.GetInt();
	wxString filename_to_write = my_event.GetString();
	int details[3];

	details[0] = image_to_send.logical_x_dimension;
	details[1] = image_to_send.logical_y_dimension;
	details[2] = position_in_stack;

	WriteToSocket(master_socket, socket_result_with_image_to_write, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(master_socket, details, sizeof(int) * 3, true, "SendResultImageDetailsFromSlaveToMaster", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(master_socket, image_to_send.real_values, image_to_send.real_memory_allocated * sizeof(float), true, "SendResultImageDataFromSlaveToMaster", FUNCTION_DETAILS_AS_WXSTRING);
	SendwxStringToSocket(&filename_to_write, master_socket);
}

void MyApp::OnThreadSendProgramDefinedResult(ReturnProgramDefinedResultEvent& my_event)
{
	//MyDebugAssertTrue(i_am_the_master == false, "OnThreadSendImageResult called by master!");

	float *array_to_send = my_event.GetResultData();
	long size_of_array = my_event.GetSizeOfResultData();
	int number_of_expected_results =  my_event.GetNumberOfExpectedResults();
	int result_number = my_event.GetResultNumber();

	int details[3];

	details[0] = size_of_array;
	details[1] = result_number;
	details[2] = number_of_expected_results;

	WriteToSocket(master_socket, socket_program_defined_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(master_socket, details, sizeof(int) * 3, true, "SendProgramDefinedResultDetailsFromSlaveToMaster", FUNCTION_DETAILS_AS_WXSTRING);
	WriteToSocket(master_socket, array_to_send, size_of_array * sizeof(float), true, "SendProgramDefinedResultArrayFromSlaveToMaster", FUNCTION_DETAILS_AS_WXSTRING);

	delete [] array_to_send;
}

void MyApp::SendAllResultsFromResultQueue()
{
	// have we sent results within the last second? if so wait 1s

	ArrayofJobResults my_queue_array;

	// we want to pop off all the jobs, and send them in one big lump..

	wxMutexLocker *lock = new wxMutexLocker(job_lock);

	if (lock->IsOk() == true)
	{
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
	}
	else
	{
		SocketSendError("Job Lock Error!");
		MyPrintWithDetails("Can't get job lock!");
	}


	delete lock;

	// ok, send them all..

	if (my_queue_array.GetCount() > 0)
	{
/*		if (time(NULL) - time_of_last_queue_send < 1)
		{
			wxSleep(1);
		}*/

		SendIntermediateResultQueue(my_queue_array);
		time_of_last_queue_send = time(NULL);
	}


}


void MyApp::SendIntermediateResultQueue(ArrayofJobResults &queue_to_send)
{
	//MyDebugAssertTrue(i_am_the_master == false, "SendIntermediateResultQueue called by master!");

	if (queue_to_send.GetCount() > 0)
	{
		WriteToSocket(master_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
		SendResultQueueToSocket(master_socket, queue_to_send);
	}

	time_of_last_queue_send = time(NULL);


}

void MyApp::SocketSendError(wxString error_to_send)
{
	// send the error message flag

	if (is_running_locally == false)
	{
		WriteToSocket(controller_socket, socket_i_have_an_error, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
		SendwxStringToSocket(&error_to_send, controller_socket);
	}
}

void MyApp::SocketSendInfo(wxString info_to_send)
{
	// send the info message flag

	if (is_running_locally == false)
	{
		WriteToSocket(controller_socket, socket_i_have_info, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
		SendwxStringToSocket(&info_to_send, controller_socket);
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
	DEBUG_ABORT;
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

void MyApp::SendProgramDefinedResultToMaster(float *array_to_send, long size_of_array, int result_number, int number_of_expected_results)
{
	if (work_thread != NULL)
	{
		work_thread->SendProgramDefinedResultToMaster(array_to_send, size_of_array, result_number, number_of_expected_results);
	}
	else
	{
		wxPrintf("Work thread is NULL!\n");
	}
}


JobResult * MyApp::PopJobFromResultQueue()  // MAKE SURE THE MUTEX JOB_LOCK IS LOCKED BEFORE CALLING THIS!
{
	JobResult *popped_job = NULL;

	if (job_queue.GetCount() > 0) popped_job = job_queue.Detach(0);
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
			//wxPrintf("Thread next action = %i\n", main_thread_pointer->thread_next_action);
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

			if (millis_sleeping > job_wait_time * 1000)
			{
				// we have been waiting for 10 seconds, something probably went wrong - so die.
				wxPrintf("Calculation thread has been waiting for something to do for %f.2 seconds - going to finish\n", job_wait_time);
				QueueError(wxString::Format("Calculation thread has been waiting for something to do for %f.2 seconds - going to finish", job_wait_time));
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

void CalculateThread::SendProgramDefinedResultToMaster(float  *array_to_send, long size_of_array, int result_number, int number_of_expected_results)
{
	ReturnProgramDefinedResultEvent *test_event = new ReturnProgramDefinedResultEvent(wxEVT_COMMAND_MYTHREAD_SEND_PROGRAM_DEFINED_RESULT);
	test_event->SetResultData(array_to_send);
	test_event->SetSizeOfResultData(size_of_array);
	test_event->SetResultNumber(result_number);
	test_event->SetNumberOfExpectedResults(number_of_expected_results);
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

///////////////////////////////////////////////////////////////////////////////////
//                              SOCKET HANDLING                                  //
///////////////////////////////////////////////////////////////////////////////////

// These should be from guix_job_control :-


//void MyApp::HandleSocketJobPackage(wxSocketBase *connected_socket, JobPackage *received_package)
//{
//	current_job_package = *received_package;
//	delete received_package;
//}

void MyApp::HandleSocketYouAreTheMaster(wxSocketBase *connected_socket, JobPackage *received_package)
{

	current_job_package = *received_package;
	delete received_package;

	// we got real communication, so we are not a zombie
	
	i_am_a_zombie = false;
	if (zombie_timer != NULL)
	{
		delete zombie_timer;
		zombie_timer = NULL;
	}

	i_am_the_master = true;

	// we need to start a server so that the slaves can connect..

	SetupServer();

	my_port = ReturnServerPort();
	my_port_string = ReturnServerPortString();

	my_ip_address = ReturnIPAddressFromSocket(connected_socket);

	// connect myself as a slave..


	master_ip_address = my_ip_address;
	master_port_string = my_port_string;
	master_port = my_port;

	master_socket = new wxSocketClient();
	master_socket->SetFlags(SOCKET_FLAGS);
	master_socket->Notify(false);

	active_controller_address.Hostname("localhost");
	active_controller_address.Service(master_port);

	master_socket->Connect(active_controller_address, false);
	master_socket->WaitOnConnect(30);

	master_socket->SetFlags(SOCKET_FLAGS );

	if (master_socket->IsConnected() == false)
	{
		master_socket->Close();
		MyDebugPrint("JOB : Failed ! Unable to connect\n");
	}

	// otherwise we should be connected.. so start monitoring..

	MonitorSocket(master_socket);

	// Start the worker thread..
	stopwatch.Start();
	work_thread = new CalculateThread(this, GetMaxJobWaitTimeInSeconds());

	if ( work_thread->Run() != wxTHREAD_NO_ERROR )
	{
		MyPrintWithDetails("Can't create the thread!");
		delete work_thread;
		work_thread = NULL;
		//ExitMainLoop();
	}


		// I have to send my ip address to the controller..


	// This is possibly dodgy as it's not being controlled by SocketCommunicator - hopefully it is ok, as this socket is not yet
	// being monitored in the corresponding read in guix_job_control - but it's a possible point of failure.

	SendwxStringToSocket(&my_ip_address, connected_socket);
	SendwxStringToSocket(&my_port_string, connected_socket);

	// ok, now get the job details from the conduit controller

	//WriteToSocket(connected_socket, socket_send_job_details, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);

}

void MyApp::HandleSocketYouAreASlave(wxSocketBase *connected_socket, wxString master_ip_address, wxString master_port_string)
{

	// we got real communication, so we are not a zombie
	
	i_am_a_zombie = false;
	if (zombie_timer != NULL)
	{
		delete zombie_timer;
		zombie_timer = NULL;
	}
	
	long received_port;
	//i_am_the_master = false;

	master_port_string.ToLong(&received_port);
	master_port = (short int) received_port;

	// remove this socket from monitoring and destroy it..

	StopMonitoringAndDestroySocket(connected_socket);

	// connect to the new master..

	master_socket = new wxSocketClient();
	master_socket->SetFlags(SOCKET_FLAGS);
	master_socket->Notify(false);

	active_controller_address.Hostname(master_ip_address);
	active_controller_address.Service(master_port);

	master_socket->Connect(active_controller_address, false);
	master_socket->WaitOnConnect(30);

	master_socket->SetFlags(SOCKET_FLAGS );

	if (master_socket->IsConnected() == false)
	{
		master_socket->Close();
		MyDebugPrint("JOB : Failed ! Unable to connect\n");
	}

	// otherwise we should be connected.. so start monitoring..

	MonitorSocket(master_socket);
	if (i_am_the_master == false) controller_socket = master_socket;

	// Start the worker thread..
	stopwatch.Start();
	work_thread = new CalculateThread(this, GetMaxJobWaitTimeInSeconds());

	if ( work_thread->Run() != wxTHREAD_NO_ERROR )
	{
		MyPrintWithDetails("Can't create the thread!");
		delete work_thread;
		work_thread = NULL;
		ExitMainLoop();
	}
	
	// we are apparently connected again, but this can be a lie = a certain number of connections appear to just be accepted by the operating
	// system - if the port if valid.  So if we don't get any events from this socket with 30 seconds, we are going to assume something
	// went wrong and die...

	i_am_a_zombie = true;
	zombie_timer = new wxTimer(this, 1);
	zombie_timer->StartOnce(20000);
	
}

void MyApp::HandleSocketTimeToDie(wxSocketBase *connected_socket) // This can be sent to a slave or the master, need to check which it is.
{
	if (i_am_the_master == true && connected_socket == controller_socket)
	{
		// tell any connected slaves to die. then exit..

		for (int counter = 0; counter < slave_socket_pointers.GetCount(); counter++)
		{
			WriteToSocket(slave_socket_pointers[counter], socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING );
		}

		slave_socket_pointers.Clear();


	}
	else  //Slave
	{
		 // Timing stuff here
		long milliseconds_spent_by_thread = stopwatch.Time();

		WriteToSocket(master_socket, socket_send_thread_timing, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
		WriteToSocket(master_socket, &milliseconds_spent_by_thread, sizeof(long), true, "SendMillisecondsSpentByThread", FUNCTION_DETAILS_AS_WXSTRING);

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
		StopMonitoringAndDestroySocket(master_socket);
		if (i_am_the_master == false) ShutDownSocketMonitor();

		// give the thread some time to die..
		wxSleep(2);

		// process thread events in case it has done something
		Yield(); //(wxEVT_CATEGORY_THREAD);

		if (work_thread != NULL) work_thread->Kill();

		if (i_am_the_master == false) // don't die if we are also the master
		{
			ExitMainLoop();
			exit(0);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////
//                        FROM SLAVES WHEN I AM THE MASTER                       //
///////////////////////////////////////////////////////////////////////////////////

void MyApp::HandleSocketSendNextJob(wxSocketBase *connected_socket, JobResult *received_result)
{
	SendNextJobTo(connected_socket);

	// Send info that the job has finished, and if necessary the result..

	if (received_result->job_number != -1)
	{
		if (received_result->result_size > 0)
		{
			SendJobResult(received_result);
		}
		else // just say job finished..
		{
			SendJobFinished(received_result->job_number);
		}

		number_of_finished_jobs++;
		current_job_package.jobs[received_result->job_number].has_been_run = true;

		// check if we have all timings, and all results (this is chctedecked in two places - socket send timing and receive results as it is not certain will happen last)

		if (number_of_finished_jobs == current_job_package.number_of_jobs && number_of_timing_results_received == max_number_of_connected_slaves)
		{
			// if we are writing a file, close it..
			if (master_output_file.IsOpen() == true) master_output_file.CloseFile();

			SendAllJobsFinished();

			if (current_job_package.ReturnNumberOfJobsRemaining() != 0)
			{
				SocketSendError("All jobs should be finished, but job package is not empty.");
			}

			// time to die!

			StopMonitoringAndDestroySocket(connected_socket);
			ShutDownSocketMonitor();
			delete received_result;
			ExitMainLoop();
			return;
		}
	}

	delete received_result;
}

void MyApp::HandleSocketIHaveAnError(wxSocketBase *connected_socket, wxString error_message)
{
	SocketSendError(error_message);
}

void MyApp::HandleSocketIHaveInfo(wxSocketBase *connected_socket, wxString info_message)
{
	SocketSendInfo(info_message);
}

void MyApp::HandleSocketJobResult(wxSocketBase *connected_socket, JobResult *received_result)
{
	SendJobResult(received_result);
	delete received_result;
}

void MyApp::HandleSocketJobResultQueue(wxSocketBase *connected_socket, ArrayofJobResults *received_queue)
{
	// copy these results to our own result queue

	for (int counter = 0; counter < received_queue->GetCount(); counter++)
	{
		master_job_queue.Add(received_queue->Item(counter));
	}

	delete received_queue;

	// if there is no timer running, start one.

	if (master_queue_timer_set == false)
	{
		master_queue_timer_set = true;
		master_queue_timer = new wxTimer(this, 3);
		master_queue_timer->StartOnce(1000);
	}
}

void MyApp::HandleSocketResultWithImageToWrite(wxSocketBase *connected_socket, Image *image_to_write, wxString filename_to_write_to, int position_in_stack)
{
	if (master_output_file.IsOpen() == false || master_output_file.filename != filename_to_write_to)
	{
		// if we are writing a file, close it..
		if (master_output_file.IsOpen() == true) master_output_file.CloseFile();
		master_output_file.OpenFile(filename_to_write_to.ToStdString(), true);
		image_to_write->WriteSlice(&master_output_file, 1); // to setup the file..
	}

	image_to_write->WriteSlice(&master_output_file, position_in_stack);
	delete image_to_write;

	float temp_float;
	temp_float = position_in_stack;

	JobResult job_to_queue;
	job_to_queue.SetResult(1, &temp_float);
	master_job_queue.Add(job_to_queue);

	if (master_queue_timer_set == false)
	{
		master_queue_timer_set = true;
		master_queue_timer = new wxTimer(this, 3);
		master_queue_timer->StartOnce(1000);
	}
	else
	{
		if (time(NULL) - time_of_last_master_queue_send > 2)
		{
			// must be a lot of queued event, so the timer is not being called -  send the current result queue anyway so the gui gets updated;

			MasterSendIntenalQueue();
		}
	}
}

void MyApp::HandleSocketProgramDefinedResult(wxSocketBase *connected_socket, float *data_array, int size_of_data_array, int result_number, int number_of_expected_results)
{
	MasterHandleProgramDefinedResult(data_array, size_of_data_array, result_number, number_of_expected_results);
	delete [] data_array;
}

void MyApp::HandleSocketSendThreadTiming(wxSocketBase *connected_socket, long received_timing_in_milliseconds)
{
	total_milliseconds_spent_on_threads += received_timing_in_milliseconds;
	StopMonitoringAndDestroySocket(connected_socket);
	//connected_socket->Destroy();

	number_of_timing_results_received++;

	// check if we have all timings, and all results (this is checked in two places - socket send timing and receive results as it is not certain will happen last)

	if (number_of_finished_jobs == current_job_package.number_of_jobs && number_of_timing_results_received == max_number_of_connected_slaves)
	{
		// if we are writing a file, close it..
		if (master_output_file.IsOpen() == true) master_output_file.CloseFile();

		SendAllJobsFinished();

		if (current_job_package.ReturnNumberOfJobsRemaining() != 0)
		{
			SocketSendError("All jobs should be finished, but job package is not empty.");
		}

		// time to die!


		ShutDownSocketMonitor();
		//controller_socket->Destroy();
		//ExitMainLoop();
		//return;
	}
}

///////////////////////////////////////////////////////////////////////////////////
//              SERVER CONNECTIONS FROM SLAVES WHEN I AM THE MASTER              //
///////////////////////////////////////////////////////////////////////////////////

void MyApp::HandleNewSocketConnection(wxSocketBase *new_connection,  unsigned char *identification_code)
{
	 if (new_connection == NULL) return;

	 if ((memcmp(identification_code, current_job_code, SOCKET_CODE_SIZE) != 0) )
	 {
		 SendError("Unknown Job ID (Job Control), leftover from a previous job? - Closing Connection");
		 new_connection->Destroy(); // we should not be monitoring this socket, just destroy it.
		 new_connection = NULL;
	 }
	 else
	 {
		 // start monitoring this socket..
		 MonitorSocket(new_connection);
		 slave_socket_pointers.Add(new_connection);
		 max_number_of_connected_slaves++;

		 // tell it is is connected..
		 WriteToSocket(new_connection, socket_you_are_connected, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
		 
		int number_of_commands_to_run;
		if (current_job_package.number_of_jobs + 1 < current_job_package.my_profile.ReturnTotalJobs()) number_of_commands_to_run = current_job_package.number_of_jobs + 1;
		else number_of_commands_to_run = current_job_package.my_profile.ReturnTotalJobs();

		if (slave_socket_pointers.GetCount() == number_of_commands_to_run - 1)
 		{
			SocketSendInfo("All slaves have re-connected to the master.");
		}
	 }

	 delete [] identification_code;
}

///////////////////////////////////////////////////////////////////////////////////
//                      FROM THE MASTER WHEN I AM A SLAVE                        //
///////////////////////////////////////////////////////////////////////////////////

// Time to die is above as it could be for slave or master

void MyApp::HandleSocketYouAreConnected(wxSocketBase *connected_socket)
{

	 // if we got here, we are not a zombie..

	 i_am_a_zombie = false;
	 if (zombie_timer != NULL)
	 {
		 delete zombie_timer;
		zombie_timer = NULL;
	 }
	 
	// we are connected, request the first job..
	is_connected = true;

	WriteToSocket(connected_socket, socket_send_next_job, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	JobResult temp_result; // dummy result for the initial request - not reallt very nice
	temp_result.job_number = -1;
	temp_result.result_size = 0;
	temp_result.SendToSocket(connected_socket);
}

void MyApp::HandleSocketReadyToSendSingleJob(wxSocketBase *connected_socket, RunJob *received_job)
{
	MyDebugAssertTrue(currently_running_a_job == false, "Received a new job, when already running a job!");
	my_current_job = *received_job;
	delete received_job;

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
}


void MyApp::HandleSocketDisconnect(wxSocketBase *connected_socket)
{
	if (connected_socket == controller_socket && i_am_the_master == true) // kill everything..
	{
		MyDebugPrint("Master received disconnect from controller");

		for (int counter = 0; counter < slave_socket_pointers.GetCount(); counter++)
		{
			WriteToSocket(slave_socket_pointers[counter], socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING );
			StopMonitoringAndDestroySocket(slave_socket_pointers[counter]);
	    }

		slave_socket_pointers.Clear();

		StopMonitoringAndDestroySocket(controller_socket);
		ShutDownServer();
		ShutDownSocketMonitor();


		ExitMainLoop();
		return;
	}
	else
	if (i_am_the_master == true && connected_socket != master_socket) // a slave died..
	{
		if (number_of_dispatched_jobs < current_job_package.number_of_jobs)
		{
			SocketSendError("Error: A slave has disconnected before all jobs are finished.");
		}

		StopMonitoringAndDestroySocket(connected_socket);
	}
	else // i am a slave and the master died.. time to die
	{
		StopMonitoringAndDestroySocket(connected_socket);
		ShutDownSocketMonitor();

		if (work_thread != NULL) work_thread->Kill();
		ExitMainLoop();
		return;
	}
}
