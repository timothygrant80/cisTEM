#define SERVER_ID 100
#define SOCKET_ID 101



class MyApp; // So CalculateThread class knows about it

// The workhorse / calculation thread

class CalculateThread : public wxThread
{
	public:
    	CalculateThread(MyApp *handler) : wxThread(wxTHREAD_DETACHED) { main_thread_pointer = handler; }
    	~CalculateThread();

    	MyApp *main_thread_pointer;
    	void QueueError(wxString error_to_queue);
    	void QueueInfo(wxString info_to_queue);
    	void MarkIntermediateResultAvailable();
    	void SendProcessedImageResult(Image *image_to_send, int position_in_stack, wxString filename_to_save);


	protected:

    	virtual ExitCode Entry();
        long time_sleeping;

};



// The console APP class.. should just deal with events..

class
MyApp : public wxAppConsole
{

		wxTimer *connection_timer;
		wxTimer *zombie_timer;
		wxTimer *queue_timer;

		bool i_am_a_zombie;
		bool queue_timer_set;

		int number_of_failed_connections;
		void CheckForConnections();
		void OnConnectionTimer(wxTimerEvent& event);
		void OnZombieTimer(wxTimerEvent& event);
		void OnQueueTimer(wxTimerEvent& event);

		wxStopWatch stopwatch;
		long total_milliseconds_spent_on_threads;
		MRCFile master_output_file;


	public:
		virtual bool OnInit();

		// array for sending back the results - this may be better off being made into an object..

		JobResult my_result;
		ArrayofJobResults job_queue;

		// socket stuff

		wxSocketClient *controller_socket;
		bool 			is_connected;
		bool            currently_running_a_job;
		wxIPV4address 	active_controller_address;
		long 			controller_port;
		unsigned char   job_code[SOCKET_CODE_SIZE];
		short int my_port;
		wxString my_ip_address;
		wxString my_port_string;

		bool is_running_locally;

		wxSocketServer *socket_server;

		bool i_am_the_master;

		int number_of_results_sent;

		JobPackage my_job_package;
		RunJob my_current_job;
		RunJob global_job_parameters;

		wxString master_ip_address;
		wxString master_port_string;
		short int master_port;

		long number_of_connected_slaves; // for the master...
		long number_of_dispatched_jobs;
		long number_of_finished_jobs;

		wxSocketBase **slave_sockets;  // POINTER TO POINTER..

		wxCmdLineParser command_line_parser;

		virtual bool DoCalculation() = 0;
		virtual void DoInteractiveUserInput() {wxPrintf("\n Error: This program cannot be run interactively..\n\n"); exit(0);}
		virtual void AddCommandLineOptions();

		void SendError(wxString error_message);
		void SendErrorAndCrash(wxString error_message);
		void SendInfo(wxString error_message);
		void SendIntermediateResultQueue(ArrayofJobResults &queue_to_send);

		CalculateThread *work_thread;
		wxMutex job_lock;
		int thread_next_action;

		long time_of_last_queue_send;

		void AddJobToResultQueue(JobResult *);
		JobResult * PopJobFromResultQueue();
		void SendAllResultsFromResultQueue();
		void SendProcessedImageResult(Image *image_to_send, int position_in_stack, wxString filename_to_save);
		private:

		void SendJobFinished(int job_number);
		void SendJobResult(JobResult *result);
		void SendJobResultQueue(ArrayofJobResults &queue_to_send);
		void MasterSendIntenalQueue();
		void SendAllJobsFinished();



		void SocketSendError(wxString error_message);
		void SocketSendInfo(wxString info_message);

		void SetupServer();

		void SendNextJobTo(wxSocketBase *socket);

		void OnOriginalSocketEvent(wxSocketEvent& event);
		void OnMasterSocketEvent(wxSocketEvent& event);
		void OnSlaveSocketEvent(wxSocketEvent& event);
		void OnControllerSocketEvent(wxSocketEvent& event);



		void OnServerEvent(wxSocketEvent& event);
		void OnThreadComplete(wxThreadEvent& my_event);
		void OnThreadEnding(wxThreadEvent& my_event);
		void OnThreadSendError(wxThreadEvent& my_event);
		void OnThreadSendInfo(wxThreadEvent& my_event);
		void OnThreadIntermediateResultAvailable(wxThreadEvent& my_event);
		void OnThreadSendImageResult(wxThreadEvent& my_event);
};

