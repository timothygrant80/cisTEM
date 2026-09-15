#ifndef _SRC_CORE_SOCKET_COMMUNICATION_UTILS_SOCKET_COMMUNICATOR_H_
#define _SRC_CORE_SOCKET_COMMUNICATION_UTILS_SOCKET_COMMUNICATOR_H_

/*
 * SocketCommunicator runs two background threads on behalf of an EventLoop-based application
 * (MyApp, the job controller): SocketServerThread accepts worker connections on a free port in
 * [START_PORT, END_PORT], and SocketClientMonitorThread polls every monitored socket, reads each
 * complete message off it and hands the decoded message to the loop thread with
 * brother_event_handler->CallAfter(...), where the virtual HandleSocket* methods run.
 *
 * Sockets are TcpSocket objects owned through raw pointers; StopMonitoringAndDestroySocket()
 * deletes one. Never read from a socket outside the monitor thread.
 */

class SocketServerThread;
class SocketClientMonitorThread;

typedef std::vector<TcpSocket*> ArrayOfSocketPointers;

class SocketCommunicator {
  protected:
    SocketServerThread*        server_thread;
    SocketClientMonitorThread* socket_monitor_thread;

    JobPackage current_job_package;

  public:
    std::atomic<bool> server_is_running;
    std::atomic<bool> monitor_is_running;

    std::mutex server_mutex;
    std::mutex monitor_thread_mutex; // guards socket_monitor_thread creation
    std::mutex add_sockets_mutex;
    std::mutex remove_sockets_mutex;
    std::mutex remove_sockets_and_destroy_mutex;

    EventLoop* brother_event_handler; // THIS MUST BE SET IN THE CONSTRUCTOR OF INHERITED CLASSES!!!

    unsigned char current_job_code[SOCKET_CODE_SIZE];

    SocketCommunicator( );
    virtual ~SocketCommunicator( );

    bool SetupServer( );
    void ShutDownServer( );

    void ShutDownSocketMonitor( );

    short int     ReturnServerPort( );
    wxString      ReturnServerPortString( );
    wxArrayString ReturnServerAllIpAddresses( );

    void MonitorSocket(TcpSocket* socket_to_monitor);
    void StopMonitoringSocket(TcpSocket* socket_to_monitor);
    void StopMonitoringAndDestroySocket(TcpSocket* socket_to_monitor);
    void SetJobCode(unsigned char* code_to_set);

    virtual wxString ReturnName( ) { return "GenericCommunicator"; }

    // the following should be overidden in the inherited classes
    // It is VERY IMPORTANT that data is never read on the passed socket, it should only be written to the socket.
    // All reading should be handled completely in the monitor thread loop.

    virtual void HandleNewSocketConnection(TcpSocket* new_connection, unsigned char* identification_code) { wxPrintf("Warning:: Unhandled Socket Message (HandleNewSocketConnection)\n"); }

    virtual void HandleSocketYouAreConnected(TcpSocket* connected_socket) { wxPrintf("Warning:: Unhandled Socket Message (HandleSocketYouAreConnected)\n"); }

    virtual void HandleSocketSendJobDetails(TcpSocket* connected_socket) { wxPrintf("Warning:: Unhandled Socket Message (HandleSocketSendJobDetails)\n"); }

    virtual void HandleSocketJobPackage(TcpSocket* connected_socket, JobPackage* received_package) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketJobPackage)\n"); }

    virtual void HandleSocketYouAreTheMaster(TcpSocket* connected_socket, JobPackage* received_package) { wxPrintf("Warning:: Unhandled Socket Message (HandleSocketYouAreTheMaster)\n"); }

    virtual void HandleSocketYouAreAWorker(TcpSocket* connected_socket, wxString master_ip_address, wxString master_port_string) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketYouAreTheWorker)\n"); }

    virtual void HandleSocketSendNextJob(TcpSocket* connected_socket, JobResult* received_result) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketSendNextJob)\n"); }

    virtual void HandleSocketTimeToDie(TcpSocket* connected_socket) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketTimeToDie)\n"); }

    virtual void HandleSocketReadyToSendSingleJob(TcpSocket* connected_socket, RunJob* received_job) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketReadyToSendSingleJob)\n"); }

    virtual void HandleSocketIHaveAnError(TcpSocket* connected_socket, wxString error_message) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketIHaveAnError)\n"); }

    virtual void HandleSocketIHaveInfo(TcpSocket* connected_socket, wxString info_message) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketIHaveInfo)\n"); }

    virtual void HandleSocketJobResult(TcpSocket* connected_socket, JobResult* received_result) { wxPrintf("Warning:: Unhandled Socket Message(HAndleSocketJobResult)\n"); }

    virtual void HandleSocketJobResultQueue(TcpSocket* connected_socket, ArrayofJobResults* received_queue) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketJobResultQueue)\n"); }

    virtual void HandleSocketJobFinished(TcpSocket* connected_socket, int finished_job_number) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketJobFinished)\n"); }

    virtual void HandleSocketAllJobsFinished(TcpSocket* connected_socket, long received_timing_in_milliseconds) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketAllJobsFinished)\n"); }

    virtual void HandleSocketNumberOfConnections(TcpSocket* connected_socket, int received_number_of_connections) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketNumberOfConnections)\n"); }

    virtual void HandleSocketResultWithImageToWrite(TcpSocket* connected_socket, wxString filename_to_write_to, int position_in_stack) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketResultWithImageToWrite)\n"); } // The image itself is written by the monitor thread, see socket_communicator.cpp

    virtual void HandleSocketProgramDefinedResult(TcpSocket* connected_socket, float* data_array, int size_of_data_array, int result_number, int number_of_expected_results) { wxPrintf("Warning:: Unhandled Socket Message (HandleSocketProgramDefinedResult)\n"); }

    virtual void HandleSocketSendThreadTiming(TcpSocket* connected_socket, long received_timing_in_milliseconds) { wxPrintf("Warning:: Unhandled Socket Message(HandleSocketSendThreadTiming\n"); }

    virtual void HandleSocketDisconnect(TcpSocket* connected_socket) { wxPrintf("Warning:: Unhandled Socket Disconnect(HandleSocketDisconnect)\n"); }

    virtual void HandleSocketTemplateMatchResultReady(TcpSocket* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes) { wxPrintf("Warning:: Unhandled Socket Message (HandleSocketTemplateMatchResultReady)\n"); }
};

/**
 * Base for the two background threads: Run() starts the thread on Entry(), RequestStop() asks it
 * to finish at its next check, Join() waits for it (or detaches if called from the thread itself).
 */
class SocketCommunicatorThread {
  public:
    SocketCommunicatorThread(SocketCommunicator* handler) : parent_pointer(handler), should_stop(false), has_finished(false) {}

    virtual ~SocketCommunicatorThread( );

    SocketCommunicatorThread(const SocketCommunicatorThread&) = delete;
    SocketCommunicatorThread& operator=(const SocketCommunicatorThread&) = delete;

    bool Run( );
    void RequestStop( ) { should_stop = true; }

    bool StopRequested( ) const { return should_stop; }

    bool HasFinished( ) const { return has_finished; }

    void Join( );

    SocketCommunicator* parent_pointer;

  protected:
    virtual void Entry( ) = 0;

    std::thread       thread;
    std::atomic<bool> should_stop;
    std::atomic<bool> has_finished;
};

class SocketServerThread : public SocketCommunicatorThread {
  public:
    SocketServerThread(SocketCommunicator* handler) : SocketCommunicatorThread(handler), my_port(-1), local_copy_server_is_running(false) {}

    ~SocketServerThread( );

    wxArrayString all_my_ip_addresses;
    wxString      my_port_string;
    short int     my_port;

    TcpServer socket_server;
    bool      local_copy_server_is_running;

  protected:
    void Entry( ) override;
};

class SocketClientMonitorThread : public SocketCommunicatorThread {
  public:
    SocketClientMonitorThread(SocketCommunicator* handler) : SocketCommunicatorThread(handler) {}

    ~SocketClientMonitorThread( );

    ArrayOfSocketPointers monitored_sockets;
    ArrayOfSocketPointers sockets_to_add_next_cycle;
    ArrayOfSocketPointers sockets_to_remove_next_cycle;
    ArrayOfSocketPointers sockets_to_remove_and_destroy_next_cycle;

    MRCFile buffered_output_file; // for writing directly in the HandleSocketResultWithImageToWrite sequence.

  protected:
    void Entry( ) override;

  private:
    // Remove the socket at `index` from monitored_sockets and tell the handler it went away.
    void ReportDisconnectAndForget(int& index);
};

#endif // _SRC_CORE_SOCKET_COMMUNICATION_UTILS_SOCKET_COMMUNICATOR_H_
