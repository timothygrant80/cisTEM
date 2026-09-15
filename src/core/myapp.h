#ifndef _SRC_CORE_MYAPP_H_
#define _SRC_CORE_MYAPP_H_

/*
 * MyApp is the base class of every cisTEM program. Run interactively (no arguments) it asks its
 * questions and calls DoCalculation() on the main thread. Launched by a job controller
 * (controller_address port job_code threads) it connects to the controller, becomes the master
 * or a worker, and runs DoCalculation() for each job it is handed on a CalculateThread while the
 * main thread runs the EventLoop that services sockets and timers.
 *
 * IMPLEMENT_APP(MyProgramApp) defines main() for a program.
 */

typedef std::vector<TcpSocket*>                  ArrayOfSocketBasePointers;
typedef std::unordered_map<TcpSocket*, RunJob*> SocketJobPointerHash;

#define PrintIfLocal(...)                \
    {                                    \
        if ( is_running_lcally == true ) \
            wxPrintf(__VA_ARGS__);       \
    }

class MyApp; // So CalculateThread class knows about it

// The workhorse / calculation thread

class CalculateThread {
  public:
    CalculateThread(MyApp* handler, float wanted_job_wait_time);
    ~CalculateThread( );

    CalculateThread(const CalculateThread&) = delete;
    CalculateThread& operator=(const CalculateThread&) = delete;

    /// Start the thread; false if it could not be created.
    bool Run( );

    /// True once Entry() has returned.
    bool HasFinished( ) const { return has_finished; }

    /// Wait for the thread to finish (detaches instead when called from the thread itself).
    void Join( );

    MyApp* main_thread_pointer;
    float  job_wait_time;
    void   QueueError(wxString error_to_queue);
    void   QueueInfo(wxString info_to_queue);
    void   MarkIntermediateResultAvailable( );
    void   SendProcessedImageResult(Image* image_to_send, int position_in_stack, wxString filename_to_save);
    void   SendProgramDefinedResultToMaster(float* result_to_send, long size_of_result, int result_number, int number_of_expected_results);

  protected:
    void Entry( );

    std::thread       thread;
    std::atomic<bool> has_finished;
};

// The console APP class.. should just deal with events..

class MyApp : public EventLoop,
              public SocketCommunicator {
    int  zombie_timer;
    bool zombie_timer_set;
    bool i_am_a_zombie;
    int  number_of_failed_connections;

    int  queue_timer;
    bool queue_timer_set;

    int  master_queue_timer;
    bool master_queue_timer_set;

    void OnQueueTimer( );
    void OnMasterQueueTimer( );
    void OnZombieTimer( );

    void StartZombieTimer( );
    void StopZombieTimer( );

    virtual float GetMaxJobWaitTimeInSeconds( ) { return 30.0f; }

    ElapsedTimer stopwatch;
    long         total_milliseconds_spent_on_threads;

    // Ask the calculation thread to stop and give it a moment to do so.
    void StopWorkThread( );

  public:
    MyApp( );
    virtual ~MyApp( );

    int    argc;
    char** argv;

    /// The program's main(): OnInit, OnEventLoopEnter, the event loop, OnExit. Used by IMPLEMENT_APP.
    int Main(int argc, char** argv);

    bool         OnInit( );
    int          OnExit( );
    virtual void ProgramSpecificInit( ){ };
    virtual void ProgramSpecificCleanUp( ){ };
    virtual void MyInteractiveProgramCleanup( ){ };
    void         OnEventLoopEnter( );

    // Socket overides

    void HandleNewSocketConnection(TcpSocket* new_connection, unsigned char* identification_code) override;
    void HandleSocketYouAreTheMaster(TcpSocket* connected_socket, JobPackage* received_package) override;
    void HandleSocketYouAreAWorker(TcpSocket* connected_socket, wxString master_ip_address, wxString master_port_string) override;
    void HandleSocketTimeToDie(TcpSocket* connected_socket) override;
    void HandleSocketJobResult(TcpSocket* connected_socket, JobResult* received_result) override;
    void HandleSocketIHaveAnError(TcpSocket* connected_socket, wxString error_message) override;
    void HandleSocketIHaveInfo(TcpSocket* connected_socket, wxString info_message) override;
    void HandleSocketSendNextJob(TcpSocket* connected_socket, JobResult* received_result) override;
    void HandleSocketJobResultQueue(TcpSocket* connected_socket, ArrayofJobResults* received_queue) override;
    void HandleSocketResultWithImageToWrite(TcpSocket* connected_socket, wxString filename_to_write_to, int position_in_stack) override;
    void HandleSocketProgramDefinedResult(TcpSocket* connected_socket, float* data_array, int size_of_data_array, int result_number, int number_of_expected_results) override;
    void HandleSocketSendThreadTiming(TcpSocket* connected_socket, long received_timing_in_milliseconds) override;
    void HandleSocketYouAreConnected(TcpSocket* connected_socket) override;
    void HandleSocketReadyToSendSingleJob(TcpSocket* connected_socket, RunJob* received_job) override;
    void HandleSocketDisconnect(TcpSocket* connected_socket) override;

    void IfSocketIsAKeySocketSetItToNull(TcpSocket* socket_to_check);

    // array for sending back the results - this may be better off being made into an object..

    JobResult         my_result;
    ArrayofJobResults job_queue;
    ArrayofJobResults master_job_queue;

    // socket stuff

    TcpSocket* controller_socket;
    TcpSocket* master_socket;

    bool        is_connected;
    bool        connected_to_the_master;
    bool        currently_running_a_job;
    std::string active_controller_address; // host name or IP the sockets (re)connect to
    long        controller_port;

    // This message queue is currently used for SendProcessedImageResult, the calculation thread will wait until there is a message on this queue before sending the next image.
    // This is done, so that the process does not swarm ahead and fill the memory with processed images.

    // currently any message will allow sending, one message is pre posted in MyApp::OnInit so that the first SendProcessedImageResult will go ahead.

    MessageQueue<char> inter_thread_message_queue;

    short int my_port;
    wxString  my_ip_address;
    wxString  my_port_string;

    bool is_running_locally;
    int  number_of_threads_requested_on_command_line;

    bool i_am_the_master;
    bool i_am_a_worker;

    int number_of_results_sent;
    int number_of_timing_results_received;

    RunJob my_current_job;
    RunJob global_job_parameters;

    wxString  master_ip_address;
    wxString  master_port_string;
    short int master_port;

    long max_number_of_connected_workers; // for the master...
    long number_of_dispatched_jobs;
    long number_of_finished_jobs;

    ArrayOfSocketBasePointers worker_socket_pointers;

    // HashMap to keep track of which socket is currently working on which job
    SocketJobPointerHash socket_to_worker_job_pointer_hash;

    CommandLineParser command_line_parser;

    virtual bool DoCalculation( ) = 0;

    virtual void DoInteractiveUserInput( ) {
        wxPrintf("\n Error: This program cannot be run interactively..\n\n");
        exit(0);
    }

    virtual void AddCommandLineOptions( );

    void SendError(wxString error_message);
    void SendErrorAndCrash(wxString error_message);
    void SendInfo(wxString error_message);
    void SendIntermediateResultQueue(ArrayofJobResults& queue_to_send);

    CalculateThread* work_thread;
    std::mutex       job_lock;
    int              thread_next_action;

    long time_of_last_queue_send;
    long time_of_last_master_queue_send;

    void       AddJobToResultQueue(JobResult*);
    JobResult* PopJobFromResultQueue( );
    void       SendAllResultsFromResultQueue( );
    void       SendProcessedImageResult(Image* image_to_send, int position_in_stack, wxString filename_to_save);
    void       SendProgramDefinedResultToMaster(float* result_to_send, long size_of_result, int result_number, int number_of_expected_results); // can override MasterHandleSpecialResult to do something with this

    // Called on the main thread by the calculation thread (through CallAfter)
    void OnThreadComplete(bool success);
    void OnThreadEnding( );
    void OnThreadSendError(wxString error_message);
    void OnThreadSendInfo(wxString info_message);
    void OnThreadIntermediateResultAvailable( );
    void OnThreadSendImageResult(std::shared_ptr<Image> image_to_send, int position_in_stack, wxString filename_to_write);
    void OnThreadSendProgramDefinedResult(float* array_to_send, long size_of_array, int result_number, int number_of_expected_results);

  private:
    void SendJobFinished(int job_number);
    void SendJobResult(JobResult* result);
    void SendJobResultQueue(ArrayofJobResults& queue_to_send);
    void MasterSendIntenalQueue( );
    void SendAllJobsFinished( );

    virtual void MasterHandleProgramDefinedResult(float* result_array, long array_size, int result_number, int number_of_expected_results) { wxPrintf("warning parent MasterHandleProgramDefinedResult called, you should probably be overriding this!\n"); } // can use this for program specific results by overiding in combination with SendSpecialResultForMaster in the program

    void SocketSendError(wxString error_message);
    void SocketSendInfo(wxString info_message);

    void SendNextJobTo(TcpSocket* socket);

    bool ConnectSocketToController(TcpSocket* socket);
};

// Defines main() for a program whose application class derives from MyApp.
#ifdef IMPLEMENT_APP
#undef IMPLEMENT_APP
#endif
#define IMPLEMENT_APP(AppClass)                 \
    int main(int argc, char** argv) {           \
        AppClass* the_app = new AppClass( );    \
        int       code    = the_app->Main(argc, argv); \
        delete the_app;                         \
        return code;                            \
    }

#endif // _SRC_CORE_MYAPP_H_
