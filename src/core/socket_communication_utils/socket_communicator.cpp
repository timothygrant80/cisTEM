#include "../core_headers.h"

///////////////////////////////////////////////////////////////////////////////////
//                          SocketCommunicatorThread                              //
///////////////////////////////////////////////////////////////////////////////////

SocketCommunicatorThread::~SocketCommunicatorThread( ) {
    if ( thread.joinable( ) ) {
        // The owner should have joined; if not, never let a joinable std::thread be destroyed.
        if ( std::this_thread::get_id( ) == thread.get_id( ) )
            thread.detach( );
        else
            thread.join( );
    }
}

bool SocketCommunicatorThread::Run( ) {
    try {
        thread = std::thread([this]( ) {
            Entry( );
            has_finished = true;
        });
    }
    catch ( const std::system_error& error ) {
        MyPrintWithDetails("Error: can't start a thread (%s)\n", error.what( ));
        return false;
    }
    return true;
}

void SocketCommunicatorThread::Join( ) {
    if ( ! thread.joinable( ) )
        return;
    if ( std::this_thread::get_id( ) == thread.get_id( ) )
        thread.detach( );
    else
        thread.join( );
}

///////////////////////////////////////////////////////////////////////////////////
//                              SocketServerThread                                //
///////////////////////////////////////////////////////////////////////////////////

void SocketServerThread::Entry( ) {
    local_copy_server_is_running = false;

    for ( short int current_port = START_PORT; current_port <= END_PORT; current_port++ ) {
        if ( current_port == END_PORT ) {
            wxPrintf("Server: Could not find a valid port !\n\n");
            parent_pointer->server_is_running = false;
            return;
        }

        std::lock_guard<std::mutex> socket_server_lock(parent_pointer->server_mutex);
        if ( socket_server.Listen(current_port) ) {
            my_port             = current_port;
            all_my_ip_addresses = ReturnIPAddress( );
            my_port_string      = wxString::Format("%hi", my_port);

            parent_pointer->server_is_running = true;
            local_copy_server_is_running      = true;
            break;
        }
    }

    // server should be setup ok..

    while ( true ) {
        if ( StopRequested( ) ) {
            std::lock_guard<std::mutex> server_lock(parent_pointer->server_mutex);
            socket_server.Close( );
            parent_pointer->server_is_running = false;
            local_copy_server_is_running      = false;
            return;
        }

        if ( local_copy_server_is_running ) {
            TcpSocket* new_connection = socket_server.Accept(false);
            while ( new_connection != NULL ) {
                // we have a new connection, but we don't know if it has the correct job code.
                // ask it for identification, and add it for monitoring so we can respond when it sends a job code..

                WriteToSocket(new_connection, socket_please_identify, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);

                MyDebugAssertTrue(parent_pointer->brother_event_handler != NULL, "event handler not set for socket communicator!");
                parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::MonitorSocket, parent_pointer, new_connection));

                // check for further connections

                new_connection = socket_server.Accept(false);
            }
        }
        else
            MyPrintWithDetails("Error: Server is not running");

        SleepForMilliseconds(250);
    }
}

SocketServerThread::~SocketServerThread( ) {
    Join( );
    parent_pointer = NULL;
}

///////////////////////////////////////////////////////////////////////////////////
//                              SocketCommunicator                                //
///////////////////////////////////////////////////////////////////////////////////

SocketCommunicator::SocketCommunicator( ) {
    brother_event_handler = NULL;
    server_thread         = NULL;
    socket_monitor_thread = NULL;
    server_is_running     = false;
    monitor_is_running    = false;
}

SocketCommunicator::~SocketCommunicator( ) {
    if ( server_thread != NULL ) {
        server_thread->RequestStop( );
        server_thread->Join( );
        delete server_thread;
    }
    if ( socket_monitor_thread != NULL ) {
        socket_monitor_thread->RequestStop( );
        socket_monitor_thread->Join( );
        delete socket_monitor_thread;
    }
}

bool SocketCommunicator::SetupServer( ) {
    MyDebugAssertTrue(server_thread == NULL, "Error: trying to start a server, but server_thread != NULL");

    server_thread = new SocketServerThread(this);

    if ( server_thread->Run( ) == false ) {
        MyPrintWithDetails("Warning: Can't create the server thread!");
        delete server_thread;
        server_thread = NULL;
        return false;
    }

    // if we got here, server should be up and running soon.. wait a bit and check before carrying on..

    int time_waited = 0;

    while ( time_waited < 50 ) {
        if ( server_is_running == true || server_thread->HasFinished( ) )
            break;
        SleepForMilliseconds(100);
        time_waited++;
    }

    if ( server_is_running == false ) {
        wxPrintf("Warning:: Timed out waiting for server...\n");
        return false;
    }
    return true;
}

void SocketCommunicator::ShutDownServer( ) {
    if ( server_thread != NULL ) {
        server_thread->RequestStop( );
        server_thread->Join( );
        delete server_thread;
        server_thread = NULL;
    }
    server_is_running = false;
}

void SocketCommunicator::ShutDownSocketMonitor( ) {
    SocketClientMonitorThread* thread_to_stop;
    {
        std::lock_guard<std::mutex> lock(monitor_thread_mutex);
        thread_to_stop        = socket_monitor_thread;
        socket_monitor_thread = NULL;
    }

    if ( thread_to_stop != NULL ) {
        thread_to_stop->RequestStop( );
        thread_to_stop->Join( );
        delete thread_to_stop;
    }
    monitor_is_running = false;
}

short int SocketCommunicator::ReturnServerPort( ) {
    if ( server_is_running == true && server_thread != NULL )
        return server_thread->my_port;
    return -1;
}

wxString SocketCommunicator::ReturnServerPortString( ) {
    if ( server_is_running == true && server_thread != NULL )
        return server_thread->my_port_string;
    return "";
}

wxArrayString SocketCommunicator::ReturnServerAllIpAddresses( ) {
    if ( server_is_running == true && server_thread != NULL )
        return server_thread->all_my_ip_addresses;

    wxArrayString blank;
    return blank;
}

void SocketCommunicator::MonitorSocket(TcpSocket* socket_to_monitor) {
    // is the thread already running? If not start it..

    {
        std::lock_guard<std::mutex> lock(monitor_thread_mutex);

        if ( socket_monitor_thread == NULL ) {
            socket_monitor_thread = new SocketClientMonitorThread(this);

            if ( socket_monitor_thread->Run( ) == false ) {
                MyPrintWithDetails("Warning: Can't create the socket monitor thread!");
                delete socket_monitor_thread;
                socket_monitor_thread = NULL;
                DEBUG_ABORT;
            }

            int time_waited = 0;
            while ( time_waited < 50 ) {
                if ( monitor_is_running == true )
                    break;
                SleepForMilliseconds(100);
                time_waited++;
            }

            if ( monitor_is_running == false ) {
                wxPrintf("Warning:: Timed out waiting for socket monitor thead to start...\n");
                DEBUG_ABORT
            }
        }

        // if we get here the thread should be running, or we should be dead.

        std::lock_guard<std::mutex> socket_monitor_lock(add_sockets_mutex);
        socket_monitor_thread->sockets_to_add_next_cycle.push_back(socket_to_monitor);
    }
}

void SocketCommunicator::SetJobCode(unsigned char* code_to_set) {
    for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
        current_job_code[counter] = code_to_set[counter];
    }
}

void SocketCommunicator::StopMonitoringSocket(TcpSocket* socket_to_monitor) {
    std::lock_guard<std::mutex> lock(monitor_thread_mutex);
    if ( socket_monitor_thread != NULL && monitor_is_running == true ) {
        std::lock_guard<std::mutex> monitor_socket_lock(remove_sockets_mutex);
        socket_monitor_thread->sockets_to_remove_next_cycle.push_back(socket_to_monitor);
    }
}

void SocketCommunicator::StopMonitoringAndDestroySocket(TcpSocket* socket_to_monitor) {
    std::lock_guard<std::mutex> lock(monitor_thread_mutex);
    if ( socket_monitor_thread == NULL || monitor_is_running == false )
        return;

    std::lock_guard<std::mutex> monitor_socket_lock(remove_sockets_and_destroy_mutex);
    socket_monitor_thread->sockets_to_remove_and_destroy_next_cycle.push_back(socket_to_monitor);
}

///////////////////////////////////////////////////////////////////////////////////
//                           SocketClientMonitorThread                            //
///////////////////////////////////////////////////////////////////////////////////

SocketClientMonitorThread::~SocketClientMonitorThread( ) {
    Join( );
}

void SocketClientMonitorThread::ReportDisconnectAndForget(int& index) {
    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[index]));
    monitored_sockets.erase(monitored_sockets.begin( ) + index);
    index--;
}

void SocketClientMonitorThread::Entry( ) {
    SETUP_SOCKET_CODES

    parent_pointer->monitor_is_running = true;

    MyDebugPrint("Socket Monitor Thread is Started\n");

    // enter main loop

    while ( true ) {
        int socket_counter;
        int change_counter;
        int number_with_data;

        if ( StopRequested( ) ) {
            // time to die

            // if we are writing a file, close it..
            if ( buffered_output_file.IsOpen( ) == true )
                buffered_output_file.CloseFile( );

            parent_pointer->monitor_is_running = false;

            MyDebugPrint("There are %li sockets being monitored on close\n", long(monitored_sockets.size( )));

            // destroy all connected sockets..

            for ( socket_counter = 0; socket_counter < int(monitored_sockets.size( )); socket_counter++ ) {
                if ( monitored_sockets[socket_counter] != NULL ) {
                    delete monitored_sockets[socket_counter];
                    monitored_sockets[socket_counter] = NULL;
                }
            }

            return;
        }

        // see if we have sockets to add, or remove

        {
            std::lock_guard<std::mutex> socket_monitor_lock(parent_pointer->add_sockets_mutex);
            for ( change_counter = 0; change_counter < int(sockets_to_add_next_cycle.size( )); change_counter++ ) {
                monitored_sockets.push_back(sockets_to_add_next_cycle[change_counter]);
            }
            sockets_to_add_next_cycle.clear( );
        }

        {
            std::lock_guard<std::mutex> socket_monitor_lock(parent_pointer->remove_sockets_mutex);
            for ( change_counter = 0; change_counter < int(sockets_to_remove_next_cycle.size( )); change_counter++ ) {
                for ( socket_counter = 0; socket_counter < int(monitored_sockets.size( )); socket_counter++ ) {
                    if ( sockets_to_remove_next_cycle[change_counter] == monitored_sockets[socket_counter] ) {
                        monitored_sockets.erase(monitored_sockets.begin( ) + socket_counter);
                        socket_counter--;
                    }
                }
            }
            sockets_to_remove_next_cycle.clear( );
        }

        {
            std::lock_guard<std::mutex> socket_monitor_lock(parent_pointer->remove_sockets_and_destroy_mutex);
            for ( change_counter = 0; change_counter < int(sockets_to_remove_and_destroy_next_cycle.size( )); change_counter++ ) {
                bool was_monitored = false;
                for ( socket_counter = 0; socket_counter < int(monitored_sockets.size( )); socket_counter++ ) {
                    if ( sockets_to_remove_and_destroy_next_cycle[change_counter] == monitored_sockets[socket_counter] ) {
                        monitored_sockets.erase(monitored_sockets.begin( ) + socket_counter);
                        socket_counter--;
                        was_monitored = true;
                    }
                }
                if ( was_monitored )
                    delete sockets_to_remove_and_destroy_next_cycle[change_counter];
            }
            sockets_to_remove_and_destroy_next_cycle.clear( );
        }

        // Loop through all the sockets, check them for input and then behave appropriately...
        // It is VERY IMPORTANT that when there is data on a socket - ALL the data is read at once and then sent
        // to the main thread.  This is because the info is sent as an event and so we have no guarantee that
        // the main thread will get it before we check the socket again, at which point we will read again.
        //
        // E.g. if the sender sends "socket_i_have_an_error", we must read the error in the code below and send it to the
        // main thread as a string.  sockets are passed in the overidden functions, but they should only be used for
        // writing - never reading!

        number_with_data = 0;

        for ( socket_counter = 0; socket_counter < int(monitored_sockets.size( )); socket_counter++ ) {
            TcpSocket* current_socket = monitored_sockets[socket_counter];

            // is the socket ok?

            if ( current_socket == NULL || current_socket->IsConnected( ) == false ) {
                // socket is not ok.. pass on a message to the handler and remove it..
                ReportDisconnectAndForget(socket_counter);
                continue;
            }

            // does this socket have data..

            if ( current_socket->WaitForRead(0) == false )
                continue;

            number_with_data++;

            // this socket has data, read the message..

            if ( ReadFromSocket(current_socket, &socket_input_buffer, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING) == false ) {
                // socket is likely dead
                ReportDisconnectAndForget(socket_counter);
                continue;
            }

            // call the relevant function depending on what this message is..

            if ( memcmp(socket_input_buffer, socket_please_identify, SOCKET_CODE_SIZE) == 0 ) {
                // send my job code..

                if ( WriteToSocket(current_socket, socket_sending_identification, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING) == false ) {
                    ReportDisconnectAndForget(socket_counter);
                }
                else if ( WriteToSocket(current_socket, parent_pointer->current_job_code, SOCKET_CODE_SIZE, true, "SendJobCodeIdentifier", FUNCTION_DETAILS_AS_WXSTRING) == false ) {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_sending_identification, SOCKET_CODE_SIZE) == 0 ) {
                unsigned char* received_job_code = new unsigned char[SOCKET_CODE_SIZE];
                // read the job code
                if ( ReadFromSocket(current_socket, received_job_code, SOCKET_CODE_SIZE, true, "SendJobCodeIdentifier", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleNewSocketConnection, parent_pointer, current_socket, received_job_code));

                    // stop monitoring this socket.. we will let the overridden code decide what to do with it based on the identification code..
                    monitored_sockets.erase(monitored_sockets.begin( ) + socket_counter);
                    socket_counter--;
                }
                else {
                    delete[] received_job_code;
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_you_are_connected, SOCKET_CODE_SIZE) == 0 ) {
                parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketYouAreConnected, parent_pointer, current_socket));
            }
            else if ( memcmp(socket_input_buffer, socket_send_job_details, SOCKET_CODE_SIZE) == 0 ) {
                parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketSendJobDetails, parent_pointer, current_socket));
            }
            else if ( memcmp(socket_input_buffer, socket_sending_job_package, SOCKET_CODE_SIZE) == 0 ) {
                JobPackage* temp_package = new JobPackage;
                if ( temp_package->ReceiveJobPackage(current_socket) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobPackage, parent_pointer, current_socket, temp_package));
                }
                else {
                    delete temp_package;
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_you_are_the_master, SOCKET_CODE_SIZE) == 0 ) {
                JobPackage* temp_package = new JobPackage;
                if ( temp_package->ReceiveJobPackage(current_socket) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketYouAreTheMaster, parent_pointer, current_socket, temp_package));
                }
                else {
                    delete temp_package;
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_you_are_a_worker, SOCKET_CODE_SIZE) == 0 ) {
                wxString master_ip_address;
                wxString master_port_string;
                bool     no_error;

                master_ip_address = ReceivewxStringFromSocket(current_socket, no_error);
                if ( no_error == true )
                    master_port_string = ReceivewxStringFromSocket(current_socket, no_error);

                if ( no_error == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketYouAreAWorker, parent_pointer, current_socket, master_ip_address, master_port_string));
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_send_next_job, SOCKET_CODE_SIZE) == 0 ) {
                JobResult* temp_job = new JobResult;
                if ( temp_job->ReceiveFromSocket(current_socket) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketSendNextJob, parent_pointer, current_socket, temp_job));
                }
                else {
                    delete temp_job;
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_time_to_die, SOCKET_CODE_SIZE) == 0 ) {
                parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketTimeToDie, parent_pointer, current_socket));
            }
            else if ( memcmp(socket_input_buffer, socket_ready_to_send_single_job, SOCKET_CODE_SIZE) == 0 ) {
                RunJob* received_job = new RunJob;
                if ( received_job->RecieveJob(current_socket) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketReadyToSendSingleJob, parent_pointer, current_socket, received_job));
                }
                else {
                    delete received_job;
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0 ) {
                wxString error_message;
                bool     no_error;
                error_message = ReceivewxStringFromSocket(current_socket, no_error);

                if ( no_error == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketIHaveAnError, parent_pointer, current_socket, error_message));
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_i_have_info, SOCKET_CODE_SIZE) == 0 ) {
                wxString info_message;
                bool     no_error;
                info_message = ReceivewxStringFromSocket(current_socket, no_error);

                if ( no_error == true )
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketIHaveInfo, parent_pointer, current_socket, info_message));
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_job_finished, SOCKET_CODE_SIZE) == 0 ) {
                int finished_job_number;
                if ( ReadFromSocket(current_socket, &finished_job_number, sizeof(int), true, "SendJobNumber", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobFinished, parent_pointer, current_socket, finished_job_number));
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0 ) {
                int received_number_of_connections;
                if ( ReadFromSocket(current_socket, &received_number_of_connections, sizeof(int), true, "SendNumberOfConnections", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketNumberOfConnections, parent_pointer, current_socket, received_number_of_connections));
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_all_jobs_finished, SOCKET_CODE_SIZE) == 0 ) {
                long received_timing_in_milliseconds;
                if ( ReadFromSocket(current_socket, &received_timing_in_milliseconds, sizeof(long), true, "SendTotalMillisecondsSpentOnThreads", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketAllJobsFinished, parent_pointer, current_socket, received_timing_in_milliseconds));
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0 ) {
                // get the result and pass it on..  remember to delete it in the overriden function.
                JobResult* temp_job = new JobResult;
                if ( temp_job->ReceiveFromSocket(current_socket) == true )
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobResult, parent_pointer, current_socket, temp_job));
                else {
                    delete temp_job;
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0 ) {
                // get the queue and pass it on..  remember to delete it in the overriden function.
                ArrayofJobResults* temp_array = new ArrayofJobResults;
                if ( ReceiveResultQueueFromSocket(current_socket, *temp_array) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobResultQueue, parent_pointer, current_socket, temp_array));
                }
                else {
                    delete temp_array;
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_result_with_image_to_write, SOCKET_CODE_SIZE) == 0 ) {
                Image*   image_to_write = new Image;
                wxString filename_to_write;
                int      position_in_stack;
                bool     ok = false;

                int details[3];
                if ( ReadFromSocket(current_socket, details, sizeof(int) * 3, true, "SendResultImageDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                    image_to_write->Allocate(details[0], details[1], 1, true, false);
                    position_in_stack = details[2];

                    if ( ReadFromSocket(current_socket, image_to_write->real_values, image_to_write->real_memory_allocated * sizeof(float), true, "SendResultImageDataFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                        bool no_error;
                        filename_to_write = ReceivewxStringFromSocket(current_socket, no_error);
                        if ( no_error == true ) {
                            // THIS IS UNUSUAL
                            // The previous implementation received the image, then queued the event (with CallAfter), expecting myApp to handle write it.  This lead to a situation where if the data was
                            // provided much faster than it can be written then the master's memory filled up.  In order to fix that, I am changing it so that image is directly written by the socket communicator.
                            // this is also no ideal, as the master will freeze to all events while writing the image, but we shall see how it goes.

                            parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketResultWithImageToWrite, parent_pointer, current_socket, filename_to_write, position_in_stack));

                            if ( buffered_output_file.IsOpen( ) == false || buffered_output_file.filename != filename_to_write ) {
                                // if we are writing a file, close it..
                                if ( buffered_output_file.IsOpen( ) == true )
                                    buffered_output_file.CloseFile( );
                                buffered_output_file.OpenFile(filename_to_write.ToStdString( ), true);

                                // Setup the file
                                image_to_write->WriteSlice(&buffered_output_file, 1);
                                buffered_output_file.WriteHeader( );
                                buffered_output_file.FlushFile( );
                            }

                            image_to_write->WriteSlice(&buffered_output_file, position_in_stack);
                            ok = true;
                        }
                    }
                }

                delete image_to_write;
                if ( ! ok )
                    ReportDisconnectAndForget(socket_counter);
            }
            else if ( memcmp(socket_input_buffer, socket_program_defined_result, SOCKET_CODE_SIZE) == 0 ) {
                int details[3];

                if ( ReadFromSocket(current_socket, details, sizeof(int) * 3, true, "SendProgramDefinedResultDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                    int size_of_data_array         = details[0];
                    int result_number              = details[1];
                    int number_of_expected_results = details[2];

                    float* data_array = new float[size_of_data_array];

                    if ( ReadFromSocket(current_socket, data_array, size_of_data_array * sizeof(float), true, "SendProgramDefinedResultArrayFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketProgramDefinedResult, parent_pointer, current_socket, data_array, size_of_data_array, result_number, number_of_expected_results));
                    }
                    else {
                        delete[] data_array;
                        ReportDisconnectAndForget(socket_counter);
                    }
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_send_thread_timing, SOCKET_CODE_SIZE) == 0 ) {
                long received_timing_in_milliseconds;

                if ( ReadFromSocket(current_socket, &received_timing_in_milliseconds, sizeof(long), true, "SendMillisecondsSpentByThread", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketSendThreadTiming, parent_pointer, current_socket, received_timing_in_milliseconds));
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
            else if ( memcmp(socket_input_buffer, socket_template_match_result_ready, SOCKET_CODE_SIZE) == 0 ) {
                int                                image_number;
                float                              threshold_used;
                ArrayOfTemplateMatchFoundPeakInfos peak_infos;
                ArrayOfTemplateMatchFoundPeakInfos peak_changes;

                if ( ReceiveTemplateMatchingResultFromSocket(current_socket, image_number, threshold_used, peak_infos, peak_changes) == true ) {
                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketTemplateMatchResultReady, parent_pointer, current_socket, image_number, threshold_used, peak_infos, peak_changes));
                }
                else {
                    ReportDisconnectAndForget(socket_counter);
                }
            }
        }

        if ( number_with_data == 0 )
            SleepForMilliseconds(250);
    }
}
