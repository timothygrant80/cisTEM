#define SERVER_ID 100
#define SOCKET_ID 101

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDERROR, wxThreadEvent);

class MyApp; // So CalculateThread class knows about it

// The workhorse / calculation thread

class CalculateThread : public wxThread
{
	public:
    	CalculateThread(MyApp *handler) : wxThread(wxTHREAD_DETACHED) { main_thread_pointer = handler; }
    	~CalculateThread();

    	MyApp *main_thread_pointer;
    	void QueueError(wxString error_to_queue);

	protected:

    	virtual ExitCode Entry();

};



// The console APP class.. should just deal with events..

class
MyApp : public wxAppConsole
{

	public:
		virtual bool OnInit();

		wxSocketClient *controller_socket;
		bool 			is_connected;
		bool            currently_running_a_job;
		wxIPV4address 	controller_address;
		long 			controller_port;
		unsigned char   job_code[SOCKET_CODE_SIZE];
		short int my_port;
		wxString my_ip_address;
		wxString my_port_string;

		wxSocketServer *socket_server;

		bool i_am_the_master;

		JobPackage my_job_package;
		RunJob my_current_job;

		wxString master_ip_address;
		wxString master_port_string;
		short int master_port;

		long number_of_connected_slaves; // for the master...
		long number_of_dispatched_jobs;
		long number_of_finished_jobs;

		wxSocketBase **slave_sockets;  // POINTER TO POINTER..

		virtual bool DoCalculation() = 0;

		void SendError(wxString error_message);




		 CalculateThread *work_thread;
		private:

		void SendJobFinished(int job_number);
		void SendAllJobsFinished();
		void SocketSendError(wxString error_message);

		void SetupServer();

		void SendNextJobTo(wxSocketBase *socket);

		void OnOriginalSocketEvent(wxSocketEvent& event);
		void OnMasterSocketEvent(wxSocketEvent& event);
		void OnSlaveSocketEvent(wxSocketEvent& event);
		void OnControllerSocketEvent(wxSocketEvent& event);


		void OnServerEvent(wxSocketEvent& event);
		void OnThreadComplete(wxThreadEvent& my_event);
		void OnThreadSendError(wxThreadEvent& my_event);
};

