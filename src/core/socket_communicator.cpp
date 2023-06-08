#include "../core/core_headers.h"

wxThread::ExitCode SocketServerThread::Entry( ) {

    wxIPV4address my_address;
    local_copy_server_is_running = false;

    { // for mutex
        wxMutexLocker should_shutdown_lock(parent_pointer->shutdown_mutex);
        if ( should_shutdown_lock.IsOk( ) == true )
            should_shutdown = false;
        else
            MyPrintWithDetails("Error: Can't get shutdown lock");
    }

    for ( short int current_port = START_PORT; current_port <= END_PORT; current_port++ ) {
        //wxPrintf("looking at port %hi\n", current_port);
        if ( current_port == END_PORT ) {
            wxPrintf("Server: Could not find a valid port !\n\n");

            { // for mutex
                wxMutexLocker server_is_running_lock(parent_pointer->server_is_running_mutex);
                if ( server_is_running_lock.IsOk( ) == true )
                    parent_pointer->server_is_running = false;
                else
                    MyPrintWithDetails("Error: Can't get server is running lock");
            }

            {
                wxMutexLocker socket_server_lock(parent_pointer->server_mutex);
                if ( socket_server_lock.IsOk( ) == true )
                    socket_server = NULL;
                else
                    MyPrintWithDetails("Error: Can't get server lock");
            }
            //Destroy();
            return (wxThread::ExitCode)-1;
        }

        my_port = current_port;
        my_address.Service(my_port);

        wxMutexLocker socket_server_lock(parent_pointer->server_mutex);
        if ( socket_server_lock.IsOk( ) == true ) {
            socket_server = new wxSocketServer(my_address);
            socket_server->Notify(false);
            socket_server->SetFlags(SOCKET_FLAGS);

            if ( socket_server->IsOk( ) ) {
                // setup events for the socket server..

                all_my_ip_addresses = ReturnIPAddress( );
                my_port_string      = wxString::Format("%hi", my_port);

                { // for mutex
                    wxMutexLocker server_is_running_lock(parent_pointer->server_is_running_mutex);
                    if ( server_is_running_lock.IsOk( ) == true )
                        parent_pointer->server_is_running = true;
                    else
                        MyPrintWithDetails("Error: Can't get server is running lock");
                    ;
                }

                local_copy_server_is_running = true;

                //xPrintf("\n\nServer is running on thread, port: %s\n", my_port_string);

                break;
            }
            else {
                socket_server->Destroy( );
                socket_server = NULL;
            }
        }
        else
            MyPrintWithDetails("Error: Can't get server lock");
    }

    //wxPrintf("After loop, port = %hi, string = %s\n", my_port, my_port_string);
    // server should be setup ok..

    while ( 1 == 1 ) {
        wxSocketBase* new_connection;

        if ( TestDestroy( ) == true ) {
            // shutdown the server socket..

            wxMutexLocker server_lock(parent_pointer->server_mutex);

            if ( server_lock.IsOk( ) == true ) {
                socket_server->Close( );
                socket_server->Destroy( );
                socket_server = NULL;

                { // mutex
                    wxMutexLocker server_is_running_lock(parent_pointer->server_is_running_mutex);
                    if ( server_is_running_lock.IsOk( ) == true ) {
                        parent_pointer->server_is_running = false;
                    }
                    else
                        MyPrintWithDetails("Error: Can't get server is running lock");
                    ;
                }

                local_copy_server_is_running = false;

                return (wxThread::ExitCode)0;
            }
            else {
                wxPrintf("Error: Couldn't get server socket lock");
            }
        }

        { // mutex
            //wxMutexLocker socket_server_lock(parent_pointer->server_mutex);

            if ( local_copy_server_is_running == true ) {
                new_connection = socket_server->Accept(false);
                while ( new_connection != NULL ) {
                    // we have a new connection, but we don't know if it has the correct job code.
                    // ask it for identification, and add it for monitoring so we can respond when it sends a job code..

                    new_connection->SetFlags(SOCKET_FLAGS);
                    new_connection->Notify(false);
                    WriteToSocket(new_connection, socket_please_identify, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);

                    MyDebugAssertTrue(parent_pointer->brother_event_handler != NULL, "event handler not set for socket communicator!");
                    if ( new_connection != NULL )
                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::MonitorSocket, parent_pointer, new_connection));

                    // check for further connections

                    new_connection = socket_server->Accept(false);
                }
            }
            else
                MyPrintWithDetails("Error: Server is NULL");
        }

        wxMilliSleep(250);
    }
}

SocketServerThread::~SocketServerThread( ) {
    parent_pointer = NULL;
}

SocketCommunicator::SocketCommunicator( ) {
    brother_event_handler = NULL;
    server_thread         = NULL;

    { // mutex
        wxMutexLocker server_is_running_lock(server_is_running_mutex);
        if ( server_is_running_lock.IsOk( ) == true )
            server_is_running = false;
        else
            MyPrintWithDetails("Error: Can't get server is running lock");
    }

    { // mutex
        wxMutexLocker monitor_is_running_lock(monitor_is_running_mutex);
        if ( monitor_is_running_lock.IsOk( ) == true )
            monitor_is_running = false;
        else
            MyPrintWithDetails("Error: Can't get monitor is running lock");
    }
}

SocketCommunicator::~SocketCommunicator( ) {
}

bool SocketCommunicator::SetupServer( ) {
    // check a server is not already running..
    //	MyDebugAssertFalse(server_is_running, "Error: trying to start a server, but a server is already running");
    MyDebugAssertTrue(server_thread == NULL, "Error: trying to start a server, but server_thread != NULL");

    // start a server thread...

    //wxPrintf("Starting server thread\n");

    server_thread = new SocketServerThread(this);

    if ( server_thread->Run( ) != wxTHREAD_NO_ERROR ) {
        MyPrintWithDetails("Warning: Can't create the server thread!");
        delete server_thread;
        server_thread = NULL;
        return false;
    }

    // if we got here, server should be up and running soon.. wait a bit and check before carryingon..

    int time_waited = 0;

    while ( time_waited < 50 ) {
        { // for mutex
            wxMutexLocker server_is_running_lock(server_is_running_mutex);
            if ( server_is_running_lock.IsOk( ) == true ) {
                if ( server_is_running == true )
                    break;
            }
            else
                MyPrintWithDetails("Error: Can't get server is running lock");
            ;
        }

        wxMilliSleep(100);
        time_waited++;
    }

    if ( time_waited == 50 ) {
        wxPrintf("Warning:: Timed out waiting for server...\n");
        return false;
    }
    else
        return true;
}

void SocketCommunicator::ShutDownServer( ) {
    //	MyDebugAssertTrue(server_is_running == true, "Error: Shutdown Server called, when server not running");
    //	MyDebugAssertTrue(server_thread != NULL, "Error: Shutdown Server called, when server thread = NULL");

    if ( server_thread != NULL )
        server_thread->Delete( );
    server_thread = NULL;
    //wxPrintf("Waiting for server shutdown...\n");

    int time_waited = 0;
    while ( time_waited < 50 ) {
        { // for mutex
            wxMutexLocker server_is_running_lock(server_is_running_mutex);
            if ( server_is_running_lock.IsOk( ) == true ) {
                if ( server_is_running == false )
                    break;
            }
            else
                MyPrintWithDetails("Error: Can't get server is running lock");
            ;
        }

        wxMilliSleep(100);
        time_waited++;
    }

    if ( time_waited == 50 )
        wxPrintf("Timed out waiting for server shutdown\n");
}

void SocketCommunicator::ShutDownSocketMonitor( ) {
    if ( socket_monitor_thread != NULL )
        socket_monitor_thread->Delete( );
    socket_monitor_thread = NULL;

    //wxPrintf("Waiting for server shutdown...\n");

    int time_waited = 0;
    while ( time_waited < 50 ) {
        { // for mutex
            wxMutexLocker monitor_is_running_lock(monitor_is_running_mutex);
            if ( monitor_is_running_lock.IsOk( ) == true ) {
                if ( monitor_is_running == false )
                    break;
            }
            else
                MyPrintWithDetails("Error: Can't get monitor is running lock");
            ;
        }
        wxMilliSleep(100);
        time_waited++;
    }

    if ( time_waited == 50 )
        wxPrintf("Timed out waiting for socket monitor shutdown\n");
}

short int SocketCommunicator::ReturnServerPort( ) {
    wxMutexLocker server_is_running_lock(server_is_running_mutex);
    if ( server_is_running_lock.IsOk( ) == true ) {
        if ( server_is_running == true && server_thread != NULL )
            return server_thread->my_port;
    }
    else
        MyPrintWithDetails("Error: Can't get server is running lock");

    return -1;
}

wxString SocketCommunicator::ReturnServerPortString( ) {
    wxMutexLocker server_is_running_lock(server_is_running_mutex);
    if ( server_is_running_lock.IsOk( ) == true ) {
        if ( server_is_running == true && server_thread != NULL )
            return server_thread->my_port_string;
    }
    else
        MyPrintWithDetails("Error: Can't get server is running lock");

    return "";
}

wxArrayString SocketCommunicator::ReturnServerAllIpAddresses( ) {
    wxMutexLocker server_is_running_lock(server_is_running_mutex);
    if ( server_is_running_lock.IsOk( ) == true ) {
        if ( server_is_running == true && server_thread != NULL )
            return server_thread->all_my_ip_addresses;
    }
    else
        MyPrintWithDetails("Error: Can't get server is running lock");

    wxArrayString blank;
    return blank;
}

void SocketCommunicator::MonitorSocket(wxSocketBase* socket_to_monitor) {
    // is the thread already running?

    //wxPrintf("Monitor Socket called from panel %s...\n", ReturnName());

    if ( monitor_is_running_mutex.Lock( ) != wxMUTEX_NO_ERROR ) {
        MyPrintWithDetails("Error: Can't get monitor is running lock");
    }

    if ( monitor_is_running == false ) {
        monitor_is_running_mutex.Unlock( );
        socket_monitor_thread = new SocketClientMonitorThread(this);

        if ( socket_monitor_thread->Run( ) != wxTHREAD_NO_ERROR ) {
            MyPrintWithDetails("Warning: Can't create the socket monitor thread!");
            delete socket_monitor_thread;
            socket_monitor_thread = NULL;
            DEBUG_ABORT;
        }

        int time_waited = 0;
        while ( time_waited < 50 ) {
            { // for mutex
                wxMutexLocker monitor_is_running_lock(monitor_is_running_mutex);
                if ( monitor_is_running_lock.IsOk( ) == true ) {
                    if ( monitor_is_running == true )
                        break;
                }
                else
                    MyPrintWithDetails("Error: Can't get monitor is running lock");
                ;
            }

            wxMilliSleep(100);
            time_waited++;
        }

        if ( time_waited == 50 ) {
            wxPrintf("Warning:: Timed out waiting for socket monitor thead to start...\n");
            DEBUG_ABORT
        }
    }
    else
        monitor_is_running_mutex.Unlock( );

    // if we get here the thread should be running, or we should be dead.

    { // for mutex
        wxMutexLocker socket_monitor_lock(add_sockets_mutex);
        if ( socket_monitor_lock.IsOk( ) == true ) {
            //socket_to_monitor->SetFlags(SOCKET_FLAGS);
            socket_monitor_thread->sockets_to_add_next_cycle.Add(socket_to_monitor);
        }
        else
            MyPrintWithDetails("Error: Can't get add sockets lock");
    }
}

void SocketCommunicator::SetJobCode(unsigned char* code_to_set) {
    for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ ) {
        current_job_code[counter] = code_to_set[counter];
    }
}

void SocketCommunicator::StopMonitoringSocket(wxSocketBase* socket_to_monitor) {
    if ( monitor_is_running == true ) {
        wxMutexLocker monitor_socket_lock(remove_sockets_mutex);
        if ( monitor_socket_lock.IsOk( ) == true ) {
            socket_monitor_thread->sockets_to_remove_next_cycle.Add(socket_to_monitor);
        }
        else
            MyPrintWithDetails("Error: Can't get remove socket lock");
    }
}

void SocketCommunicator::StopMonitoringAndDestroySocket(wxSocketBase* socket_to_monitor) {
    if ( monitor_is_running == false )
        return;

    wxMutexLocker monitor_socket_lock(remove_sockets_and_destroy_mutex);
    if ( monitor_socket_lock.IsOk( ) == true ) {
        socket_monitor_thread->sockets_to_remove_and_destroy_next_cycle.Add(socket_to_monitor);
    }
    else
        MyPrintWithDetails("Error: Can't get remove and destroy socket lock");
}

SocketClientMonitorThread::~SocketClientMonitorThread( ) {
}

wxThread::ExitCode SocketClientMonitorThread::Entry( ) {
    SETUP_SOCKET_CODES

    { // for mutex
        wxMutexLocker monitor_is_running_lock(parent_pointer->monitor_is_running_mutex);
        if ( monitor_is_running_lock.IsOk( ) == true )
            parent_pointer->monitor_is_running = true;
        else
            MyPrintWithDetails("Error: Can't get server is running lock");
        ;
    }

    MyDebugPrint("Socket Monitor Thread is Started\n");

    // enter main loop

    while ( 1 == 1 ) {
        int socket_counter;
        int change_counter;
        int number_with_data;

        if ( TestDestroy( ) == true ) {
            // time to die

            // if we are writing a file, close it..
            if ( buffered_output_file.IsOpen( ) == true )
                buffered_output_file.CloseFile( );

            { // mutex
                wxMutexLocker monitor_is_running_lock(parent_pointer->monitor_is_running_mutex);
                if ( monitor_is_running_lock.IsOk( ) == true ) {
                    parent_pointer->monitor_is_running = false;
                }
                else
                    MyPrintWithDetails("Error: Can't get monitor is running lock");

                MyDebugPrint("There are %li sockets being monitored on close\n", monitored_sockets.GetCount( ));

                // destroy all connected sockets..

                for ( socket_counter = 0; socket_counter < monitored_sockets.GetCount( ); socket_counter++ ) {
                    if ( monitored_sockets[socket_counter] != NULL ) {
                        monitored_sockets[socket_counter]->Destroy( );
                        monitored_sockets[socket_counter] = NULL;
                    }
                }

            } // end mutex

            return (wxThread::ExitCode)0;
        }

        // see if we have sockets to add, or remove

        { // for mutex
            wxMutexLocker socket_monitor_lock(parent_pointer->add_sockets_mutex);
            if ( socket_monitor_lock.IsOk( ) == true ) {
                for ( change_counter = 0; change_counter < sockets_to_add_next_cycle.GetCount( ); change_counter++ ) {
                    monitored_sockets.Add(sockets_to_add_next_cycle[change_counter]);
                }

                sockets_to_add_next_cycle.Clear( );
            }
            else
                MyPrintWithDetails("Error: Can't get add sockets lock");
        }

        { // for mutex
            wxMutexLocker socket_monitor_lock(parent_pointer->remove_sockets_mutex);
            if ( socket_monitor_lock.IsOk( ) == true ) {
                for ( change_counter = 0; change_counter < sockets_to_remove_next_cycle.GetCount( ); change_counter++ ) {
                    for ( socket_counter = 0; socket_counter < monitored_sockets.GetCount( ); socket_counter++ ) {
                        if ( sockets_to_remove_next_cycle[change_counter] == monitored_sockets[socket_counter] ) {
                            monitored_sockets.RemoveAt(socket_counter);
                            socket_counter--;
                        }
                    }
                }

                sockets_to_remove_next_cycle.Clear( );
            }
            else
                MyPrintWithDetails("Error: Can't get add sockets lock");
        }

        { // for mutex
            wxMutexLocker socket_monitor_lock(parent_pointer->remove_sockets_and_destroy_mutex);
            if ( socket_monitor_lock.IsOk( ) == true ) {
                for ( change_counter = 0; change_counter < sockets_to_remove_and_destroy_next_cycle.GetCount( ); change_counter++ ) {
                    for ( socket_counter = 0; socket_counter < monitored_sockets.GetCount( ); socket_counter++ ) {
                        if ( sockets_to_remove_and_destroy_next_cycle[change_counter] == monitored_sockets[socket_counter] ) {
                            monitored_sockets.RemoveAt(socket_counter);
                            sockets_to_remove_and_destroy_next_cycle[change_counter]->Destroy( );
                            socket_counter--;
                        }
                    }
                }

                sockets_to_remove_and_destroy_next_cycle.Clear( );
            }
            else
                MyPrintWithDetails("Error: Can't get add sockets lock");
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

        { // mutex
            //wxMutexLocker monitor_socket_lock(parent_pointer->monitor_sockets_access_mutex);
            //if (monitor_socket_lock.IsOk() == true)
            {
                //wxPrintf("About to check %li sockets\n",  monitored_sockets.GetCount());

                for ( socket_counter = 0; socket_counter < monitored_sockets.GetCount( ); socket_counter++ ) {
                    // is the socket ok?

                    if ( monitored_sockets[socket_counter]->IsOk( ) == true && monitored_sockets[socket_counter]->IsConnected( ) == true ) {

                        // does this socket have data..

                        if ( monitored_sockets[socket_counter]->WaitForRead(0, 0) == true ) {
                            number_with_data++;
                            // this socket has data, read the message..
                            //wxPrintf("Socket has information on it...\n");
                            if ( ReadFromSocket(monitored_sockets[socket_counter], &socket_input_buffer, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING) == true ) {

                                //wxPrintf("Read %s\n", StringFromSocketCode(socket_input_buffer));
                                // call the relevant function depending on what this message is..

                                if ( memcmp(socket_input_buffer, socket_please_identify, SOCKET_CODE_SIZE) == 0 ) {
                                    // send my job code..

                                    if ( WriteToSocket(monitored_sockets[socket_counter], socket_sending_identification, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING) == false ) {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                    else if ( WriteToSocket(monitored_sockets[socket_counter], parent_pointer->current_job_code, SOCKET_CODE_SIZE, true, "SendJobCodeIdentifier", FUNCTION_DETAILS_AS_WXSTRING) == false ) {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_sending_identification, SOCKET_CODE_SIZE) == 0 ) {
                                    unsigned char* received_job_code = new unsigned char[SOCKET_CODE_SIZE];
                                    // read the job code
                                    if ( ReadFromSocket(monitored_sockets[socket_counter], received_job_code, SOCKET_CODE_SIZE, true, "SendJobCodeIdentifier", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleNewSocketConnection, parent_pointer, monitored_sockets[socket_counter], received_job_code));

                                        // stop monitoring this socket.. we will let the overridden code decide what to do with it based on the identification code..
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_you_are_connected, SOCKET_CODE_SIZE) == 0 ) {
                                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketYouAreConnected, parent_pointer, monitored_sockets[socket_counter]));
                                }
                                else if ( memcmp(socket_input_buffer, socket_send_job_details, SOCKET_CODE_SIZE) == 0 ) {
                                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketSendJobDetails, parent_pointer, monitored_sockets[socket_counter]));
                                }
                                else if ( memcmp(socket_input_buffer, socket_sending_job_package, SOCKET_CODE_SIZE) == 0 ) {
                                    JobPackage* temp_package = new JobPackage;
                                    if ( temp_package->ReceiveJobPackage(monitored_sockets[socket_counter]) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobPackage, parent_pointer, monitored_sockets[socket_counter], temp_package));
                                    }
                                    else {
                                        delete temp_package;

                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_you_are_the_master, SOCKET_CODE_SIZE) == 0 ) {
                                    JobPackage* temp_package = new JobPackage;
                                    if ( temp_package->ReceiveJobPackage(monitored_sockets[socket_counter]) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketYouAreTheMaster, parent_pointer, monitored_sockets[socket_counter], temp_package));
                                    }
                                    else {
                                        delete temp_package;

                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_you_are_a_worker, SOCKET_CODE_SIZE) == 0 ) {
                                    wxString master_ip_address;
                                    wxString master_port_string;

                                    bool no_error;

                                    master_ip_address = ReceivewxStringFromSocket(monitored_sockets[socket_counter], no_error);

                                    if ( no_error == true ) {
                                        master_port_string = ReceivewxStringFromSocket(monitored_sockets[socket_counter], no_error);
                                    }

                                    if ( no_error == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketYouAreAWorker, parent_pointer, monitored_sockets[socket_counter], master_ip_address, master_port_string));
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_send_next_job, SOCKET_CODE_SIZE) == 0 ) {
                                    JobResult* temp_job = new JobResult;
                                    if ( temp_job->ReceiveFromSocket(monitored_sockets[socket_counter]) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketSendNextJob, parent_pointer, monitored_sockets[socket_counter], temp_job));
                                    }
                                    else {
                                        delete temp_job;
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_time_to_die, SOCKET_CODE_SIZE) == 0 ) {
                                    parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketTimeToDie, parent_pointer, monitored_sockets[socket_counter]));
                                }
                                else if ( memcmp(socket_input_buffer, socket_ready_to_send_single_job, SOCKET_CODE_SIZE) == 0 ) {
                                    RunJob* received_job = new RunJob;
                                    if ( received_job->RecieveJob(monitored_sockets[socket_counter]) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketReadyToSendSingleJob, parent_pointer, monitored_sockets[socket_counter], received_job));
                                    }
                                    else {
                                        delete received_job;
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0 ) {
                                    // get the error..

                                    wxString error_message;
                                    bool     no_error;
                                    error_message = ReceivewxStringFromSocket(monitored_sockets[socket_counter], no_error);

                                    if ( no_error == true ) {
                                        // pass the error on..

                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketIHaveAnError, parent_pointer, monitored_sockets[socket_counter], error_message));
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_i_have_info, SOCKET_CODE_SIZE) == 0 ) {
                                    // get the info..

                                    wxString info_message;
                                    bool     no_error;

                                    info_message = ReceivewxStringFromSocket(monitored_sockets[socket_counter], no_error);

                                    // pass the info on..
                                    if ( no_error == true )
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketIHaveInfo, parent_pointer, monitored_sockets[socket_counter], info_message));
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_job_finished, SOCKET_CODE_SIZE) == 0 ) {
                                    // read which job finished and then pass it on to the main thread..

                                    int finished_job_number;
                                    if ( ReadFromSocket(monitored_sockets[socket_counter], &finished_job_number, sizeof(int), true, "SendJobNumber", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobFinished, parent_pointer, monitored_sockets[socket_counter], finished_job_number));
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0 ) {
                                    int received_number_of_connections;
                                    if ( ReadFromSocket(monitored_sockets[socket_counter], &received_number_of_connections, sizeof(int), true, "SendNumberOfConnections", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketNumberOfConnections, parent_pointer, monitored_sockets[socket_counter], received_number_of_connections));
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_all_jobs_finished, SOCKET_CODE_SIZE) == 0 ) {
                                    long received_timing_in_milliseconds;
                                    if ( ReadFromSocket(monitored_sockets[socket_counter], &received_timing_in_milliseconds, sizeof(long), true, "SendTotalMillisecondsSpentOnThreads", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketAllJobsFinished, parent_pointer, monitored_sockets[socket_counter], received_timing_in_milliseconds));
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0 ) {
                                    // get the result and pass it on..  remember to delete it in the overriden function.
                                    JobResult* temp_job = new JobResult;
                                    if ( temp_job->ReceiveFromSocket(monitored_sockets[socket_counter]) == true )
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobResult, parent_pointer, monitored_sockets[socket_counter], temp_job));
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0 ) {
                                    // get the queue and pass it on..  remember to delete it in the overriden function.
                                    ArrayofJobResults* temp_array = new ArrayofJobResults;
                                    if ( ReceiveResultQueueFromSocket(monitored_sockets[socket_counter], *temp_array) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketJobResultQueue, parent_pointer, monitored_sockets[socket_counter], temp_array));
                                    }
                                    else {
                                        delete temp_array;
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_result_with_image_to_write, SOCKET_CODE_SIZE) == 0 ) {
                                    Image*   image_to_write = new Image;
                                    wxString filename_to_write;
                                    int      position_in_stack;

                                    int details[3];
                                    if ( ReadFromSocket(monitored_sockets[socket_counter], details, sizeof(int) * 3, true, "SendResultImageDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                        image_to_write->Allocate(details[0], details[1], 1, true, false);
                                        position_in_stack = details[2];

                                        if ( ReadFromSocket(monitored_sockets[socket_counter], image_to_write->real_values, image_to_write->real_memory_allocated * sizeof(float), true, "SendResultImageDataFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                            bool no_error;
                                            filename_to_write = ReceivewxStringFromSocket(monitored_sockets[socket_counter], no_error);
                                            if ( no_error == true ) {
                                                // THIS IS UNUSUAL
                                                // The previous implementation received the image, then queued the event (with CallAfter), expecting myApp to handle write it.  This lead to a situation where if the data was
                                                // provided much faster than it can be written then the master's memory filled up.  In order to fix that, I am changing it so that image is directly written by the socket communicator.
                                                // this is also no ideal, as the master will freeze to all events while writing the image, but we shall see how it goes.

                                                //parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketResultWithImageToWrite,parent_pointer, monitored_sockets[socket_counter], image_to_write, filename_to_write, position_in_stack));
                                                parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketResultWithImageToWrite, parent_pointer, monitored_sockets[socket_counter], filename_to_write, position_in_stack));

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
                                                delete image_to_write;
                                            }
                                            else {
                                                delete image_to_write;
                                                // socket is not ok.. pass on a message to the handler and remove it..
                                                parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                                monitored_sockets.RemoveAt(socket_counter);
                                                socket_counter--;
                                            }
                                        }
                                        else {
                                            delete image_to_write;
                                            // socket is not ok.. pass on a message to the handler and remove it..
                                            parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                            monitored_sockets.RemoveAt(socket_counter);
                                            socket_counter--;
                                        }
                                    }
                                    else {
                                        delete image_to_write;
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_program_defined_result, SOCKET_CODE_SIZE) == 0 ) {
                                    float* data_array;
                                    int    details[3];

                                    int size_of_data_array;
                                    int result_number;
                                    int number_of_expected_results;

                                    if ( ReadFromSocket(monitored_sockets[socket_counter], details, sizeof(int) * 3, true, "SendProgramDefinedResultDetailsFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {

                                        size_of_data_array         = details[0];
                                        result_number              = details[1];
                                        number_of_expected_results = details[2];

                                        data_array = new float[size_of_data_array];

                                        if ( ReadFromSocket(monitored_sockets[socket_counter], data_array, size_of_data_array * sizeof(float), true, "SendProgramDefinedResultArrayFromWorkerToMaster", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                            parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketProgramDefinedResult, parent_pointer, monitored_sockets[socket_counter], data_array, size_of_data_array, result_number, number_of_expected_results));
                                        }
                                        else {
                                            delete[] data_array;
                                            // socket is not ok.. pass on a message to the handler and remove it..
                                            parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                            monitored_sockets.RemoveAt(socket_counter);
                                            socket_counter--;
                                        }
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_send_thread_timing, SOCKET_CODE_SIZE) == 0 ) {

                                    long received_timing_in_milliseconds;

                                    if ( ReadFromSocket(monitored_sockets[socket_counter], &received_timing_in_milliseconds, sizeof(long), true, "SendMillisecondsSpentByThread", FUNCTION_DETAILS_AS_WXSTRING) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketSendThreadTiming, parent_pointer, monitored_sockets[socket_counter], received_timing_in_milliseconds));
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                                else if ( memcmp(socket_input_buffer, socket_template_match_result_ready, SOCKET_CODE_SIZE) == 0 ) {
                                    int                                image_number;
                                    float                              threshold_used;
                                    ArrayOfTemplateMatchFoundPeakInfos peak_infos;
                                    ArrayOfTemplateMatchFoundPeakInfos peak_changes;

                                    if ( ReceiveTemplateMatchingResultFromSocket(monitored_sockets[socket_counter], image_number, threshold_used, peak_infos, peak_changes) == true ) {
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketTemplateMatchResultReady, parent_pointer, monitored_sockets[socket_counter], image_number, threshold_used, peak_infos, peak_changes));
                                    }
                                    else {
                                        // socket is not ok.. pass on a message to the handler and remove it..
                                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                        monitored_sockets.RemoveAt(socket_counter);
                                        socket_counter--;
                                    }
                                }
                            }
                            else // socket is likely dead
                            {
                                parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                                monitored_sockets.RemoveAt(socket_counter);
                                socket_counter--;
                            }
                        }
                    }
                    else {
                        // socket is not ok.. pass on a message to the handler and remove it..

                        parent_pointer->brother_event_handler->CallAfter(std::bind(&SocketCommunicator::HandleSocketDisconnect, parent_pointer, monitored_sockets[socket_counter]));
                        monitored_sockets.RemoveAt(socket_counter);
                        socket_counter--;
                    }
                }
            }
            //else MyPrintWithDetails("Error: Can't get socket monitor lock");
        } // end mutex

        if ( number_with_data == 0 )
            wxMilliSleep(250);
    }

    return (wxThread::ExitCode)0;
}
