#define SERVER_ID 100
#define SOCKET_ID 101

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);

class MyApp; // So CalculateThread class knows about it

// The workhorse / calculation thread

class CalculateThread : public wxThread
{
	public:
    	CalculateThread(MyApp *handler) : wxThread(wxTHREAD_DETACHED) { main_thread_pointer = handler; }
    	~CalculateThread();

	protected:

    	virtual ExitCode Entry();
    	MyApp *main_thread_pointer;
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

		virtual void DoCalculation();

		private:

		 CalculateThread *work_thread;

		 void SetupServer();

		 void SendNextJobTo(wxSocketBase *socket);

		 void OnOriginalSocketEvent(wxSocketEvent& event);
		 void OnMasterSocketEvent(wxSocketEvent& event);
		 void OnSlaveSocketEvent(wxSocketEvent& event);
		 void OnControllerSocketEvent(wxSocketEvent& event);


		 void OnServerEvent(wxSocketEvent& event);
		 void OnThreadComplete(wxThreadEvent&);
};

