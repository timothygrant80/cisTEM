#include "core_headers.h"

#define THREAD_START_NEXT_JOB 0
#define THREAD_DIE 1
#define THREAD_SLEEP 2

MyApp::MyApp( ) {
    argc = 0;
    argv = NULL;

    zombie_timer                 = 0;
    zombie_timer_set             = false;
    i_am_a_zombie                = false;
    number_of_failed_connections = 0;
    queue_timer                  = 0;
    queue_timer_set              = false;
    master_queue_timer           = 0;
    master_queue_timer_set       = false;

    total_milliseconds_spent_on_threads = 0;

    controller_socket = NULL;
    master_socket     = NULL;

    is_connected            = false;
    connected_to_the_master = false;
    currently_running_a_job = false;
    controller_port         = 0;

    my_port                                     = 0;
    is_running_locally                          = true;
    number_of_threads_requested_on_command_line = 1;
    i_am_the_master                             = false;
    i_am_a_worker                               = false;
    number_of_results_sent                      = 0;
    number_of_timing_results_received           = 0;
    master_port                                 = 0;
    max_number_of_connected_workers             = 0;
    number_of_dispatched_jobs                   = 0;
    number_of_finished_jobs                     = 0;

    work_thread                    = NULL;
    thread_next_action             = THREAD_SLEEP;
    time_of_last_queue_send        = 0;
    time_of_last_master_queue_send = 0;
}

MyApp::~MyApp( ) {
    if ( work_thread != NULL ) {
        StopWorkThread( );
        delete work_thread;
        work_thread = NULL;
    }
}

int MyApp::Main(int argc_in, char** argv_in) {
    argc = argc_in;
    argv = argv_in;

    if ( OnInit( ) == false )
        return 1;

    OnEventLoopEnter( );

    int exit_code = Run( );

    StopWorkThread( );

    OnExit( );
    return exit_code;
}

bool MyApp::OnInit( ) {
    thread_next_action = THREAD_SLEEP;

    number_of_dispatched_jobs         = 0;
    number_of_finished_jobs           = 0;
    number_of_timing_results_received = 0;

    max_number_of_connected_workers = 0;

    zombie_timer_set       = false;
    queue_timer_set        = false;
    master_queue_timer_set = false;

    controller_socket = NULL;
    master_socket     = NULL;

    connected_to_the_master = false;
    currently_running_a_job = false;

    time_of_last_queue_send        = 0;
    time_of_last_master_queue_send = 0;
    number_of_results_sent         = 0;

    total_milliseconds_spent_on_threads = 0;

    socket_to_worker_job_pointer_hash.clear( );

    inter_thread_message_queue.Post(0);

    ActivateMKLDebugForNonIntelCPU( ); // if not Intel CPU and if using the MKL attempt to set an environment variable that can lead to substanstial speedup.
    ProgramSpecificInit( );

    return true;
}

int MyApp::OnExit( ) {
    ProgramSpecificCleanUp( );
    return 0;
}

void MyApp::StopWorkThread( ) {
    if ( work_thread == NULL )
        return;

    {
        std::lock_guard<std::mutex> lock(job_lock);
        thread_next_action = THREAD_DIE;
    }

    // the thread checks its action every 100 ms while idle; a running DoCalculation cannot be interrupted.
    for ( int waited = 0; waited < 20 && work_thread->HasFinished( ) == false; waited++ )
        SleepForMilliseconds(100);

    if ( work_thread->HasFinished( ) )
        work_thread->Join( );
}

bool MyApp::ConnectSocketToController(TcpSocket* socket) {
    socket->Close( );
    socket->Connect(active_controller_address, int(controller_port), 30);
    return socket->IsConnected( );
}

void MyApp::StartZombieTimer( ) {
    // we are apparently connected, but this can be a lie as a certain number of connections appear to just be accepted by the operating
    // system - if the port if valid.  So if we don't get any events from this socket within 20 seconds, we are going to try again...
    i_am_a_zombie    = true;
    zombie_timer     = StartOnceTimer(20000, [this]( ) { OnZombieTimer( ); });
    zombie_timer_set = true;
}

void MyApp::StopZombieTimer( ) {
    i_am_a_zombie = false;
    if ( zombie_timer_set ) {
        StopTimer(zombie_timer);
        zombie_timer_set = false;
    }
}

void MyApp::OnEventLoopEnter( ) {
    // initialise sockets, and set the event handler for SocketCommunicator

    brother_event_handler = this;

    int  parse_status;
    int  number_of_arguments;
    int  counter;
    long temp_long;

    std::vector<std::string> possible_controller_addresses;

    socket_to_worker_job_pointer_hash.clear( );

    // Connect to the controller program..

    command_line_parser.SetCmdLine(argc, argv);
    command_line_parser.AddParam("controller_address", CMD_LINE_VAL_STRING, CMD_LINE_PARAM_OPTIONAL);
    command_line_parser.AddParam("controller_port", CMD_LINE_VAL_NUMBER, CMD_LINE_PARAM_OPTIONAL);
    command_line_parser.AddParam("job_code", CMD_LINE_VAL_STRING, CMD_LINE_PARAM_OPTIONAL);
    command_line_parser.AddParam("wanted_number_of_threads", CMD_LINE_VAL_NUMBER, CMD_LINE_PARAM_OPTIONAL);

    // Let the app add options
    AddCommandLineOptions( );

    parse_status        = command_line_parser.Parse(true);
    number_of_arguments = int(command_line_parser.GetParamCount( ));

    if ( parse_status != 0 ) {
        Printf("\n\n");
        ExitMainLoop( );
        exit(0);
        return;
    }

    // if we have no arguments run interactively.. if we have 4 continue as though we have network info, else error..

    if ( number_of_arguments == 0 ) {
        is_running_locally = true;
        DoInteractiveUserInput( );
        stopwatch.Start( );
        DoCalculation( );
        total_milliseconds_spent_on_threads += stopwatch.Time( );
        MyInteractiveProgramCleanup( );
        fftwf_cleanup( ); // this is needed to stop valgrind reporting memory leaks..
        exit(0);
    }
    else if ( number_of_arguments != 4 ) {
        command_line_parser.Usage( );
        Printf("\n\n");
        ExitMainLoop( );
        exit(0);
        return;
    }

    is_running_locally = false;

    // get the address and port of the job controller (should be command line options).

    for ( const std::string& current_address : SplitString(command_line_parser.GetParam(0), ",") ) {
        possible_controller_addresses.push_back(current_address);
        if ( HostnameIsValid(current_address) == false ) {
            MyDebugPrint(" Error: Address (%s) - not recognized as an IP or hostname\n\n", current_address);
            exit(-1);
        };
    }

    if ( StringToLong(command_line_parser.GetParam(1), controller_port) == false ) {
        MyPrintWithDetails(" Error: Port (%s) - not recognized as a port\n\n", command_line_parser.GetParam(1));
        exit(-1);
    }

    if ( command_line_parser.GetParam(2).length( ) != SOCKET_CODE_SIZE ) {
        MyPrintWithDetails(" Error: Code (%s) - is the incorrect length (%li instead of %i)\n\n", command_line_parser.GetParam(2), long(command_line_parser.GetParam(2).length( )), SOCKET_CODE_SIZE);
        exit(-1);
    }

    if ( StringToLong(command_line_parser.GetParam(3), temp_long) == false ) {
        MyPrintWithDetails(" Error: No. of Threads (%s) - not recognized as a number\n\n", command_line_parser.GetParam(3));
        exit(-1);
    }

    number_of_threads_requested_on_command_line = int(temp_long);
    if ( number_of_threads_requested_on_command_line < 1 )
        number_of_threads_requested_on_command_line = 1;

    // copy over job code.

    std::string job_code = command_line_parser.GetParam(2);
    for ( counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
        current_job_code[counter] = (unsigned char)job_code[counter];
    }

    // Attempt to connect to the controller..

    is_connected = false;

    controller_socket = new TcpSocket( );

    for ( counter = 0; counter < int(possible_controller_addresses.size( )); counter++ ) {
        active_controller_address = possible_controller_addresses[counter];
        if ( ConnectSocketToController(controller_socket) )
            break;
    }

    if ( controller_socket->IsConnected( ) == false ) {
        controller_socket->Close( );
        MyDebugPrint(" JOB : Failed ! Unable to connect\n");
        ExitMainLoop( );
        exit(0);
        return;
    }

    // Monitor this connection..

    MonitorSocket(controller_socket);

    number_of_failed_connections = 0;
    StartZombieTimer( );
}

// Placeholder (to be overridden) function to add options to the command line
void MyApp::AddCommandLineOptions( ) {
    return;
}

void MyApp::SendNextJobTo(TcpSocket* socket) {
    // if we haven't dispatched all jobs yet, then send it, otherwise tell the worker to die..

    if ( number_of_dispatched_jobs < current_job_package.number_of_jobs ) {
        // See RunJob::SendJob() Doxygen for encoding order specification
        current_job_package.jobs[number_of_dispatched_jobs].SendJob(socket);
        socket_to_worker_job_pointer_hash[socket] = &current_job_package.jobs[number_of_dispatched_jobs];
        number_of_dispatched_jobs++;
    }
    else {
        WriteToSocket(socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
        // stop monitoring the socket..
        //StopMonitoringSocket(socket); stopped doing this for timings

        // Remember that this socket doesn't have a job anymore
        socket_to_worker_job_pointer_hash.erase(socket);
    }
}

void MyApp::SendJobFinished(int job_number) {
    WriteToSocket(controller_socket, socket_job_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
    // send the job number of the current job..
    WriteToSocket(controller_socket, &job_number, sizeof(int), true, "SendJobNumber", FUNCTION_DETAILS_AS_STRING);
}

void MyApp::SendJobResult(JobResult* result) {
    WriteToSocket(controller_socket, socket_job_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
    // See JobResult::SendToSocket() Doxygen for encoding order specification
    result->SendToSocket(controller_socket);
}

void MyApp::SendJobResultQueue(ArrayofJobResults& queue_to_send) {
    WriteToSocket(controller_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
    // See SendResultQueueToSocket() Doxygen for encoding order specification
    SendResultQueueToSocket(controller_socket, queue_to_send);
}

void MyApp::MasterSendIntenalQueue( ) {
    SendJobResultQueue(master_job_queue);
    master_job_queue.clear( );
    time_of_last_master_queue_send = time(NULL);
}

void MyApp::SendAllJobsFinished( ) {
    // we will send all jobs finished - but first we need to ensure we have sent any results in the result queue
    // wait for a second to give workers time to send in their last jobs..

    SleepForSeconds(1);

    if ( master_job_queue.size( ) != 0 )
        MasterSendIntenalQueue( );

    WriteToSocket(controller_socket, socket_all_jobs_finished, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
    WriteToSocket(controller_socket, &total_milliseconds_spent_on_threads, sizeof(long), true, "SendTotalMillisecondsSpentOnThreads", FUNCTION_DETAILS_AS_STRING);
}

void MyApp::OnZombieTimer( ) {
    zombie_timer_set = false;

    if ( i_am_a_zombie == true ) {
        number_of_failed_connections++;

        if ( number_of_failed_connections >= 5 )
            ExitMainLoop( );

        if ( connected_to_the_master == true ) {
            if ( ConnectSocketToController(master_socket) == false ) {
                master_socket->Close( );
                MyDebugPrint(" JOB : Failed ! Unable to connect\n");
                ExitMainLoop( );
            }

            if ( i_am_the_master == false )
                controller_socket = master_socket;
        }
        else {
            if ( ConnectSocketToController(controller_socket) == false ) {
                controller_socket->Close( );
                MyDebugPrint(" JOB : Failed ! Unable to connect\n");
                ExitMainLoop( );
            }
        }

        // once again, we are aparently connected, but this can be a lie, so try again in 20 seconds if nothing arrives..
        StartZombieTimer( );
    }
}

void MyApp::OnMasterQueueTimer( ) {
    if ( master_job_queue.size( ) > 0 ) {
        MasterSendIntenalQueue( );
    }

    master_queue_timer_set = false;
}

void MyApp::OnQueueTimer( ) {
    SendAllResultsFromResultQueue( );

    queue_timer_set = false;
}

void MyApp::OnThreadComplete(bool success) {
    // The compute thread is finished.. get the next job

    SendAllResultsFromResultQueue( );

    // get the next job..
    WriteToSocket(master_socket, socket_send_next_job, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);

    // if there is a result - send it to the gui..
    my_result.job_number = my_current_job.job_number;
    my_result.SendToSocket(master_socket);
}

void MyApp::OnThreadEnding( ) {
    SendAllResultsFromResultQueue( );

    if ( work_thread != NULL ) {
        work_thread->Join( );
        delete work_thread;
        work_thread = NULL;
    }
}

void MyApp::OnThreadSendError(std::string error_message) {
    SocketSendError(error_message);
}

void MyApp::OnThreadSendInfo(std::string info_message) {
    SocketSendInfo(info_message);
}

void MyApp::OnThreadIntermediateResultAvailable( ) {
    if ( queue_timer_set == false ) {
        queue_timer_set = true;
        queue_timer     = StartOnceTimer(1000, [this]( ) { OnQueueTimer( ); });
    }
}

void MyApp::OnThreadSendImageResult(std::shared_ptr<Image> image_to_send, int position_in_stack, std::string filename_to_write) {
    int details[3];

    details[0] = image_to_send->logical_x_dimension;
    details[1] = image_to_send->logical_y_dimension;
    details[2] = position_in_stack;

    WriteToSocket(master_socket, socket_result_with_image_to_write, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
    WriteToSocket(master_socket, details, sizeof(int) * 3, true, "SendResultImageDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_STRING);
    WriteToSocket(master_socket, image_to_send->real_values, image_to_send->real_memory_allocated * sizeof(float), true, "SendResultImageDataFromWorkerToMaster", FUNCTION_DETAILS_AS_STRING);
    SendStringToSocket(filename_to_write, master_socket);

    // post a message to the message queue to allow the calulcation thread to send the next image..
    inter_thread_message_queue.Post(0);
}

void MyApp::OnThreadSendProgramDefinedResult(float* array_to_send, long size_of_array, int result_number, int number_of_expected_results) {
    int details[3];

    details[0] = int(size_of_array);
    details[1] = result_number;
    details[2] = number_of_expected_results;

    WriteToSocket(master_socket, socket_program_defined_result, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
    WriteToSocket(master_socket, details, sizeof(int) * 3, true, "SendProgramDefinedResultDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_STRING);
    WriteToSocket(master_socket, array_to_send, size_of_array * sizeof(float), true, "SendProgramDefinedResultArrayFromWorkerToMaster", FUNCTION_DETAILS_AS_STRING);

    delete[] array_to_send;
}

void MyApp::SendAllResultsFromResultQueue( ) {
    ArrayofJobResults my_queue_array;

    // we want to pop off all the jobs, and send them in one big lump..

    {
        std::lock_guard<std::mutex> lock(job_lock);

        while ( true ) {
            JobResult* popped_job = PopJobFromResultQueue( );

            if ( popped_job == NULL ) {
                break;
            }
            else {
                my_queue_array.push_back(*popped_job);
                delete popped_job;
            }
        }
    }

    // ok, send them all..

    if ( my_queue_array.size( ) > 0 ) {
        SendIntermediateResultQueue(my_queue_array);
        time_of_last_queue_send = time(NULL);
    }
}

void MyApp::SendIntermediateResultQueue(ArrayofJobResults& queue_to_send) {
    if ( queue_to_send.size( ) > 0 ) {
        WriteToSocket(master_socket, socket_job_result_queue, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
        SendResultQueueToSocket(master_socket, queue_to_send);
    }

    time_of_last_queue_send = time(NULL);
}

void MyApp::SocketSendError(std::string error_to_send) {
    // send the error message flag

    if ( is_running_locally == false ) {
        WriteToSocket(controller_socket, socket_i_have_an_error, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
        SendStringToSocket(error_to_send, controller_socket);
    }
}

void MyApp::SocketSendInfo(std::string info_to_send) {
    // send the info message flag

    if ( is_running_locally == false ) {
        WriteToSocket(controller_socket, socket_i_have_info, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
        SendStringToSocket(info_to_send, controller_socket);
    }
}

void MyApp::SendError(std::string error_to_send) {
    if ( is_running_locally == true ) {
        Printf("\nError : %s\n", error_to_send);
    }
    else if ( work_thread != NULL ) {
        work_thread->QueueError(error_to_send);
    }
    else {
        SocketSendError("SendError with null work thread!");
    }
}

void MyApp::SendErrorAndCrash(std::string error_to_send) {
    SendError(error_to_send);
    if ( ! is_running_locally )
        SleepForSeconds(2); // wait for the main thread to actually send the error
    DEBUG_ABORT;
}

void MyApp::SendInfo(std::string info_to_send) {
    if ( is_running_locally == true ) {
        Printf("\nInfo : %s\n", info_to_send);
    }
    else if ( work_thread != NULL ) {
        work_thread->QueueInfo(info_to_send);
    }
    else {
        SocketSendError("SendInfo with null work thread!");
    }
}

void MyApp::AddJobToResultQueue(JobResult* result_to_add) {
    // takes ownership of result_to_add
    {
        std::lock_guard<std::mutex> lock(job_lock);
        job_queue.push_back(*result_to_add);
    }
    delete result_to_add;

    if ( work_thread != NULL ) {
        work_thread->MarkIntermediateResultAvailable( );
    }
    else {
        Printf("Work thread is NULL!\n");
    }
}

void MyApp::SendProcessedImageResult(Image* image_to_send, int position_in_stack, std::string filename_to_save) {
    if ( work_thread != NULL ) {
        work_thread->SendProcessedImageResult(image_to_send, position_in_stack, filename_to_save);
    }
    else {
        Printf("Work thread is NULL!\n");
    }
}

void MyApp::SendProgramDefinedResultToMaster(float* array_to_send, long size_of_array, int result_number, int number_of_expected_results) {
    if ( work_thread != NULL ) {
        work_thread->SendProgramDefinedResultToMaster(array_to_send, size_of_array, result_number, number_of_expected_results);
    }
    else {
        Printf("Work thread is NULL!\n");
    }
}

JobResult* MyApp::PopJobFromResultQueue( ) // MAKE SURE THE MUTEX JOB_LOCK IS LOCKED BEFORE CALLING THIS!
{
    JobResult* popped_job = NULL;

    if ( job_queue.size( ) > 0 ) {
        popped_job = new JobResult(job_queue.front( ));
        job_queue.erase(job_queue.begin( ));
    }
    return popped_job;
}

///////////////////////////////////////////////////////////////////////////////////
//                              CALCULATION THREAD                                //
///////////////////////////////////////////////////////////////////////////////////

CalculateThread::CalculateThread(MyApp* handler, float wanted_job_wait_time) : has_finished(false) {
    main_thread_pointer = handler;
    job_wait_time       = wanted_job_wait_time;
}

CalculateThread::~CalculateThread( ) {
    if ( thread.joinable( ) ) {
        if ( std::this_thread::get_id( ) == thread.get_id( ) )
            thread.detach( );
        else
            thread.join( );
    }
    main_thread_pointer = NULL;
}

bool CalculateThread::Run( ) {
    try {
        thread = std::thread([this]( ) {
            Entry( );
            has_finished = true;
        });
    }
    catch ( const std::system_error& error ) {
        MyPrintWithDetails("Can't create the calculation thread (%s)\n", error.what( ));
        return false;
    }
    return true;
}

void CalculateThread::Join( ) {
    if ( ! thread.joinable( ) )
        return;
    if ( std::this_thread::get_id( ) == thread.get_id( ) )
        thread.detach( );
    else
        thread.join( );
}

// Main execution in this thread..

void CalculateThread::Entry( ) {
    int  thread_action_copy;
    long millis_sleeping = 0;

    while ( true ) {
        {
            std::lock_guard<std::mutex> lock(main_thread_pointer->job_lock);
            thread_action_copy = main_thread_pointer->thread_next_action;

            if ( main_thread_pointer->thread_next_action == THREAD_START_NEXT_JOB ) {
                main_thread_pointer->thread_next_action = THREAD_SLEEP;
                millis_sleeping                         = 0;
            }
        }

        if ( thread_action_copy == THREAD_START_NEXT_JOB ) {
            bool   success = main_thread_pointer->DoCalculation( ); // This should be overrided per app..
            MyApp* app     = main_thread_pointer;
            app->CallAfter([app, success]( ) { app->OnThreadComplete(success); });
        }
        else if ( thread_action_copy == THREAD_SLEEP ) {
            SleepForMilliseconds(100);
            millis_sleeping += 100;

            if ( millis_sleeping > job_wait_time * 1000 ) {
                // we have been waiting for a long time, something probably went wrong - so die.
                Printf("Calculation thread has been waiting for something to do for %f.2 seconds - going to finish\n", job_wait_time);
                QueueError(Format("Calculation thread has been waiting for something to do for %f.2 seconds - going to finish", job_wait_time));
                break;
            }
        }
        else if ( thread_action_copy == THREAD_DIE )
            break;
    }

    MyApp* app = main_thread_pointer;
    app->CallAfter([app]( ) { app->OnThreadEnding( ); });

    fftwf_cleanup( ); // this is needed to stop valgrind reporting memory leaks..
}

void CalculateThread::QueueError(std::string error_to_queue) {
    MyApp* app = main_thread_pointer;
    app->CallAfter([app, error_to_queue]( ) { app->OnThreadSendError(error_to_queue); });
}

void CalculateThread::QueueInfo(std::string info_to_queue) {
    MyApp* app = main_thread_pointer;
    app->CallAfter([app, info_to_queue]( ) { app->OnThreadSendInfo(info_to_queue); });
}

void CalculateThread::MarkIntermediateResultAvailable( ) {
    MyApp* app = main_thread_pointer;
    app->CallAfter([app]( ) { app->OnThreadIntermediateResultAvailable( ); });
}

void CalculateThread::SendProcessedImageResult(Image* image_to_send, int position_in_stack, std::string filename_to_save) {
    char message;
    if ( main_thread_pointer->inter_thread_message_queue.ReceiveTimeout(300000, message) == false ) // timeout after 5 minutes;
    {
        QueueError("Timed out waiting for message queue");
    }

    std::shared_ptr<Image> image_copy = std::make_shared<Image>(*image_to_send);
    MyApp*                 app        = main_thread_pointer;
    app->CallAfter([app, image_copy, position_in_stack, filename_to_save]( ) { app->OnThreadSendImageResult(image_copy, position_in_stack, filename_to_save); });
}

void CalculateThread::SendProgramDefinedResultToMaster(float* array_to_send, long size_of_array, int result_number, int number_of_expected_results) {
    MyApp* app = main_thread_pointer;
    app->CallAfter([app, array_to_send, size_of_array, result_number, number_of_expected_results]( ) { app->OnThreadSendProgramDefinedResult(array_to_send, size_of_array, result_number, number_of_expected_results); });
}

///////////////////////////////////////////////////////////////////////////////////
//                              SOCKET HANDLING                                  //
///////////////////////////////////////////////////////////////////////////////////

void MyApp::HandleSocketYouAreTheMaster(TcpSocket* connected_socket, JobPackage* received_package) {

    current_job_package = *received_package;
    delete received_package;

    // we got real communication, so we are not a zombie

    StopZombieTimer( );

    i_am_the_master = true;

    // we need to start a server so that the workers can connect..

    SetupServer( );

    my_port        = ReturnServerPort( );
    my_port_string = ReturnServerPortString( );

    my_ip_address = ReturnIPAddressFromSocket(connected_socket);

    // connect myself as a worker..

    master_ip_address  = my_ip_address;
    master_port_string = my_port_string;
    master_port        = my_port;

    master_socket = new TcpSocket( );
    master_socket->Connect("localhost", master_port, 30);

    if ( master_socket->IsConnected( ) == false ) {
        master_socket->Close( );
        MyDebugPrint("JOB : Failed ! Unable to connect\n");
    }
    else {
        MonitorSocket(master_socket);

        // Start the worker thread..
        stopwatch.Start( );
        work_thread = new CalculateThread(this, GetMaxJobWaitTimeInSeconds( ));

        if ( work_thread->Run( ) == false ) {
            MyPrintWithDetails("Can't create the thread!");
            delete work_thread;
            work_thread = NULL;
            ExitMainLoop( );
        }
    }

    // I have to send my ip address to the controller..

    // This is possibly dodgy as it's not being controlled by SocketCommunicator - hopefully it is ok, as this socket is not yet
    // being monitored in the corresponding read in the job controller - but it's a possible point of failure.

    SendStringToSocket(my_ip_address, connected_socket);
    SendStringToSocket(my_port_string, connected_socket);
}

void MyApp::HandleSocketYouAreAWorker(TcpSocket* connected_socket, std::string master_ip_address, std::string master_port_string) {

    // we got real communication, so we are not a zombie

    StopZombieTimer( );

    long received_port;

    StringToLong(master_port_string, received_port);
    master_port = (short int)received_port;

    // remove this socket from monitoring and destroy it..

    StopMonitoringAndDestroySocket(connected_socket);
    IfSocketIsAKeySocketSetItToNull(connected_socket);

    // connect to the new master..

    master_socket = new TcpSocket( );

    active_controller_address = master_ip_address;
    controller_port           = master_port;

    master_socket->Connect(active_controller_address, master_port, 30);

    if ( master_socket->IsConnected( ) == false ) {
        master_socket->Close( );
        MyDebugPrint("JOB : Failed ! Unable to connect\n");
    }

    // otherwise we should be connected.. so start monitoring..

    MonitorSocket(master_socket);
    if ( i_am_the_master == false )
        controller_socket = master_socket;
    connected_to_the_master = true;

    // Start the worker thread..
    stopwatch.Start( );
    work_thread = new CalculateThread(this, GetMaxJobWaitTimeInSeconds( ));

    if ( work_thread->Run( ) == false ) {
        MyPrintWithDetails("Can't create the thread!");
        delete work_thread;
        work_thread = NULL;
        ExitMainLoop( );
    }

    // we are apparently connected again, but this can be a lie = a certain number of connections appear to just be accepted by the operating
    // system - if the port if valid.  So if we don't get any events from this socket with 20 seconds, we are going to assume something
    // went wrong and try again...

    StartZombieTimer( );
}

void MyApp::HandleSocketTimeToDie(TcpSocket* connected_socket) // This can be sent to a worker or the master, need to check which it is.
{
    if ( i_am_the_master == true && connected_socket == controller_socket ) {
        // tell any connected workers to die. then exit..

        for ( size_t counter = 0; counter < worker_socket_pointers.size( ); counter++ ) {
            WriteToSocket(worker_socket_pointers[counter], socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
        }

        worker_socket_pointers.clear( );
    }
    else //Worker
    {
        // Timing stuff here
        long milliseconds_spent_by_thread = stopwatch.Time( );

        WriteToSocket(master_socket, socket_send_thread_timing, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
        WriteToSocket(master_socket, &milliseconds_spent_by_thread, sizeof(long), true, "SendMillisecondsSpentByThread", FUNCTION_DETAILS_AS_STRING);

        // time to die!
        {
            std::lock_guard<std::mutex> lock(job_lock);
            thread_next_action = THREAD_DIE;
        }

        StopMonitoringAndDestroySocket(master_socket);
        if ( i_am_the_master == false )
            ShutDownSocketMonitor( );

        // give the thread some time to die..
        SleepForSeconds(2);

        if ( i_am_the_master == false ) // don't die if we are also the master
        {
            ExitMainLoop( );
            exit(0);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////
//                        FROM WORKERS WHEN I AM THE MASTER                       //
///////////////////////////////////////////////////////////////////////////////////

void MyApp::HandleSocketSendNextJob(TcpSocket* connected_socket, JobResult* received_result) {
    SendNextJobTo(connected_socket);

    // Send info that the job has finished, and if necessary the result..

    if ( received_result->job_number != -1 ) {
        if ( received_result->result_size > 0 ) {
            SendJobResult(received_result);
        }
        else // just say job finished..
        {
            SendJobFinished(received_result->job_number);
        }

        number_of_finished_jobs++;
        current_job_package.jobs[received_result->job_number].has_been_run = true;

        // check if we have all timings, and all results (this is checked in two places - socket send timing and receive results as it is not certain which will happen last)

        if ( number_of_finished_jobs == current_job_package.number_of_jobs && number_of_timing_results_received == max_number_of_connected_workers ) {

            SendAllJobsFinished( );

            if ( current_job_package.ReturnNumberOfJobsRemaining( ) != 0 ) {
                SocketSendError("All jobs should be finished, but job package is not empty.");
            }

            // time to die!

            StopMonitoringAndDestroySocket(connected_socket);
            ShutDownSocketMonitor( );
            delete received_result;

            ExitMainLoop( );
            return;
        }
    }

    delete received_result;
}

void MyApp::HandleSocketIHaveAnError(TcpSocket* connected_socket, std::string error_message) {
    SocketSendError(error_message);
}

void MyApp::HandleSocketIHaveInfo(TcpSocket* connected_socket, std::string info_message) {
    SocketSendInfo(info_message);
}

void MyApp::HandleSocketJobResult(TcpSocket* connected_socket, JobResult* received_result) {
    SendJobResult(received_result);
    delete received_result;
}

void MyApp::HandleSocketJobResultQueue(TcpSocket* connected_socket, ArrayofJobResults* received_queue) {
    // copy these results to our own result queue

    for ( size_t counter = 0; counter < received_queue->size( ); counter++ ) {
        master_job_queue.push_back((*received_queue)[counter]);
    }

    delete received_queue;

    // if there is no timer running, start one.

    if ( master_queue_timer_set == false ) {
        master_queue_timer_set = true;
        master_queue_timer     = StartOnceTimer(1000, [this]( ) { OnMasterQueueTimer( ); });
    }
}

void MyApp::HandleSocketResultWithImageToWrite(TcpSocket* connected_socket, std::string filename_to_write_to, int position_in_stack) {
    // The image itself was written by the socket monitor thread; queue the position as a result for the controller.

    float temp_float;
    temp_float = position_in_stack;

    JobResult job_to_queue;
    job_to_queue.SetResult(1, &temp_float);
    master_job_queue.push_back(job_to_queue);

    if ( master_queue_timer_set == false ) {
        master_queue_timer_set = true;
        master_queue_timer     = StartOnceTimer(1000, [this]( ) { OnMasterQueueTimer( ); });
    }
    else {
        if ( time(NULL) - time_of_last_master_queue_send > 2 ) {
            // must be a lot of queued events, so the timer is not being called -  send the current result queue anyway so the gui gets updated;

            MasterSendIntenalQueue( );
        }
    }
}

void MyApp::HandleSocketProgramDefinedResult(TcpSocket* connected_socket, float* data_array, int size_of_data_array, int result_number, int number_of_expected_results) {
    MasterHandleProgramDefinedResult(data_array, size_of_data_array, result_number, number_of_expected_results);
    delete[] data_array;
}

void MyApp::HandleSocketSendThreadTiming(TcpSocket* connected_socket, long received_timing_in_milliseconds) {
    total_milliseconds_spent_on_threads += received_timing_in_milliseconds;
    StopMonitoringAndDestroySocket(connected_socket);

    number_of_timing_results_received++;

    // check if we have all timings, and all results (this is checked in two places - socket send timing and receive results as it is not certain which will happen last)

    if ( number_of_finished_jobs == current_job_package.number_of_jobs && number_of_timing_results_received == max_number_of_connected_workers ) {
        SendAllJobsFinished( );

        if ( current_job_package.ReturnNumberOfJobsRemaining( ) != 0 ) {
            SocketSendError("All jobs should be finished, but job package is not empty.");
        }

        // time to die!

        SleepForSeconds(5);

        delete controller_socket;
        controller_socket = NULL;

        ShutDownServer( );
        ShutDownSocketMonitor( );
        ExitMainLoop( );
        return;
    }
}

///////////////////////////////////////////////////////////////////////////////////
//              SERVER CONNECTIONS FROM WORKERS WHEN I AM THE MASTER              //
///////////////////////////////////////////////////////////////////////////////////

void MyApp::HandleNewSocketConnection(TcpSocket* new_connection, unsigned char* identification_code) {
    if ( new_connection == NULL )
        return;

    if ( (memcmp(identification_code, current_job_code, SOCKET_CODE_SIZE) != 0) ) {
        SendError("Unknown Job ID (Job Control), leftover from a previous job? - Closing Connection");
        delete new_connection; // we should not be monitoring this socket, just destroy it.
        new_connection = NULL;
    }
    else {
        // start monitoring this socket..
        MonitorSocket(new_connection);
        worker_socket_pointers.push_back(new_connection);
        max_number_of_connected_workers++;

        // tell it is is connected..
        WriteToSocket(new_connection, socket_you_are_connected, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);

        int number_of_commands_to_run;
        if ( current_job_package.number_of_jobs + 1 < current_job_package.my_profile.ReturnTotalJobs( ) )
            number_of_commands_to_run = current_job_package.number_of_jobs + 1;
        else
            number_of_commands_to_run = current_job_package.my_profile.ReturnTotalJobs( );

        if ( int(worker_socket_pointers.size( )) == number_of_commands_to_run - 1 ) {
            SocketSendInfo("All workers have re-connected to the master.");
        }
    }

    delete[] identification_code;
}

///////////////////////////////////////////////////////////////////////////////////
//                      FROM THE MASTER WHEN I AM A WORKER                        //
///////////////////////////////////////////////////////////////////////////////////

// Time to die is above as it could be for worker or master

void MyApp::HandleSocketYouAreConnected(TcpSocket* connected_socket) {

    // if we got here, we are not a zombie..

    StopZombieTimer( );

    // we are connected, request the first job..
    is_connected = true;

    WriteToSocket(connected_socket, socket_send_next_job, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
    JobResult temp_result; // dummy result for the initial request - not really very nice
    temp_result.job_number  = -1;
    temp_result.result_size = 0;
    temp_result.SendToSocket(connected_socket);
}

void MyApp::HandleSocketReadyToSendSingleJob(TcpSocket* connected_socket, RunJob* received_job) {
    MyDebugAssertTrue(currently_running_a_job == false, "Received a new job, when already running a job!");
    my_current_job = *received_job;
    delete received_job;

    std::lock_guard<std::mutex> lock(job_lock);
    MyDebugAssertFalse(thread_next_action == THREAD_START_NEXT_JOB, "Thread action is already start job");
    thread_next_action = THREAD_START_NEXT_JOB;
}

void MyApp::HandleSocketDisconnect(TcpSocket* connected_socket) {
    if ( connected_socket == controller_socket && i_am_the_master == true ) // kill everything..
    {
        MyDebugPrint("Master received disconnect from controller");

        for ( size_t counter = 0; counter < worker_socket_pointers.size( ); counter++ ) {
            WriteToSocket(worker_socket_pointers[counter], socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_STRING);
            StopMonitoringAndDestroySocket(worker_socket_pointers[counter]);
            worker_socket_pointers[counter] = NULL;
        }

        worker_socket_pointers.clear( );

        StopMonitoringAndDestroySocket(controller_socket);
        controller_socket = NULL;

        ShutDownServer( );
        ShutDownSocketMonitor( );

        ExitMainLoop( );
        return;
    }
    else if ( i_am_the_master == true && connected_socket != master_socket ) // a worker died..
    {
        if ( number_of_dispatched_jobs < current_job_package.number_of_jobs ) {
            SocketSendError("Error: A worker has disconnected before all jobs are finished.");
            if ( socket_to_worker_job_pointer_hash.count(connected_socket) > 0 )
                SocketSendInfo("The disconnected worker was running a job with the following arguments:\n" + socket_to_worker_job_pointer_hash[connected_socket]->PrintAllArgumentsToString( ));
        }

        StopMonitoringAndDestroySocket(connected_socket);
    }
    else // i am a worker and the master died.. time to die
    {
        StopMonitoringAndDestroySocket(connected_socket);
        if ( i_am_the_master == false )
            ShutDownSocketMonitor( );

        ExitMainLoop( );
        return;
    }
}

// Mainly for when we destroy worker sockets without directly using the array, we want them to be set to NULL;

void MyApp::IfSocketIsAKeySocketSetItToNull(TcpSocket* socket_to_check) {
    for ( size_t counter = 0; counter < worker_socket_pointers.size( ); counter++ ) {
        if ( worker_socket_pointers[counter] == socket_to_check )
            worker_socket_pointers[counter] = NULL;
    }

    if ( controller_socket == socket_to_check )
        controller_socket = NULL;

    if ( master_socket == socket_to_check )
        master_socket = NULL;
}
