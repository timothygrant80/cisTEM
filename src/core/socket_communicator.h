class SocketServerThread;
class SocketClientMonitorThread;


wxDEFINE_EVENT(wxEVT_SOCKET_SERVER_EVENT, wxThreadEvent);
WX_DEFINE_ARRAY(wxSocketBase *, ArrayOfSocketPointers);

class SocketCommunicator
{
	protected :


	SocketServerThread *server_thread;
	SocketClientMonitorThread *socket_monitor_thread;

	JobPackage current_job_package;

	public :

	bool server_is_running;
	bool monitor_is_running;

	wxMutex server_mutex;
	wxMutex server_is_running_mutex;
	wxMutex shutdown_mutex;

	wxMutex monitor_is_running_mutex;

	wxMutex add_sockets_mutex;
	wxMutex remove_sockets_mutex;
	wxMutex remove_sockets_and_destroy_mutex;

	wxEvtHandler *brother_event_handler; // THIS MUST BE SET IN THE CONTRUCTOR OF INHERITED CLASSES!!!

	unsigned char current_job_code[SOCKET_CODE_SIZE];

	SocketCommunicator();
	~SocketCommunicator();

	bool SetupServer();
	void ShutDownServer();

	void ShutDownSocketMonitor();

	short int ReturnServerPort();
	wxString ReturnServerPortString();
	wxArrayString ReturnServerAllIpAddresses();

	void MonitorSocket(wxSocketBase *socket_to_monitor);
	void StopMonitoringSocket(wxSocketBase *socket_to_monitor);
	void StopMonitoringAndDestroySocket(wxSocketBase *socket_to_monitor);
	void SetJobCode(unsigned char *code_to_set);

	virtual wxString ReturnName() {return "GenericCommunicator";}

	// the following should be overidden in the inherited classes
	// It is VERY IMPORTANT that data is never read on the passed socket, it should only be written to the socket.
	// All reading should be handled completely in the monitor thread loop.

	virtual void HandleNewSocketConnection(wxSocketBase *new_connection, unsigned char *identification_code) {wxPrintf("Warning:: Unhandled Socket Message (HandleNewSocketConnection)\n");}
	virtual void HandleSocketYouAreConnected(wxSocketBase *connected_socket) {wxPrintf("Warning:: Unhandled Socket Message (HandleSocketYouAreConnected)\n");}
	virtual void HandleSocketSendJobDetails(wxSocketBase *connected_socket) {wxPrintf("Warning:: Unhandled Socket Message (HandleSocketSendJobDetails)\n");}
	virtual void HandleSocketJobPackage(wxSocketBase *connected_socket, JobPackage *received_package) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketJobPackage)\n");}
	virtual void HandleSocketYouAreTheMaster(wxSocketBase *connected_socket, JobPackage *received_package) {wxPrintf("Warning:: Unhandled Socket Message (HandleSocketYouAreTheMaster)\n");}
	virtual void HandleSocketYouAreASlave(wxSocketBase *connected_socket, wxString master_ip_address, wxString master_port_string) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketYouAreTheSlave)\n");}
	virtual void HandleSocketSendNextJob(wxSocketBase *connected_socket, JobResult *received_result) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketSendNextJob)\n");}
	virtual void HandleSocketTimeToDie(wxSocketBase *connected_socket) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketTimeToDie)\n");}
	virtual void HandleSocketReadyToSendSingleJob(wxSocketBase *connected_socket, RunJob *received_job) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketReadyToSendSingleJob)\n");}
	virtual void HandleSocketIHaveAnError(wxSocketBase *connected_socket, wxString error_message) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketIHaveAnError)\n");}
	virtual void HandleSocketIHaveInfo(wxSocketBase *connected_socket, wxString info_message) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketIHaveInfo)\n");}
	virtual void HandleSocketJobResult(wxSocketBase *connected_socket, JobResult *received_result) {wxPrintf("Warning:: Unhandled Socket Message(HAndleSocketJobResult)\n");}
	virtual void HandleSocketJobResultQueue(wxSocketBase *connected_socket, ArrayofJobResults *received_queue) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketJobResultQueue)\n");}
	virtual void HandleSocketJobFinished(wxSocketBase *connected_socket, int finished_job_number) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketJobFinished)\n");}
	virtual void HandleSocketAllJobsFinished(wxSocketBase *connected_socket, long received_timing_in_milliseconds) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketAllJobsFinished)\n");}
	virtual void HandleSocketNumberOfConnections(wxSocketBase *connected_socket, int received_number_of_connections) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketNumberOfConnections)\n");}
	virtual void HandleSocketResultWithImageToWrite(wxSocketBase *connected_socket, Image *image_to_write, wxString filename_to_write_to, int position_in_stack) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketResultWithImageToWrite)\n");}
	virtual void HandleSocketProgramDefinedResult(wxSocketBase *connected_socket, float *data_array, int size_of_data_array, int result_number, int number_of_expected_results) {wxPrintf("Warning:: Unhandled Socket Message (HandleSocketProgramDefinedResult)\n");}
	virtual void HandleSocketSendThreadTiming(wxSocketBase *connected_socket, long received_timing_in_milliseconds) {wxPrintf("Warning:: Unhandled Socket Message(HandleSocketSendThreadTiming\n");}
	virtual void HandleSocketDisconnect(wxSocketBase *connected_socket) {wxPrintf("Warning:: Unhandled Socket Disconnect(HandleSocketDisconnect)\n");}
	virtual void HandleSocketTemplateMatchResultReady(wxSocketBase *connected_socket, int &image_number, float &threshold_used, ArrayOfTemplateMatchFoundPeakInfos &peak_infos, ArrayOfTemplateMatchFoundPeakInfos &peak_changes ) {wxPrintf("Warning:: Unhandled Socket Message (HandleSocketTemplateMatchResultReady)\n");}

};

class SocketServerThread : public wxThread
{
	public:
	SocketServerThread(SocketCommunicator *handler) : wxThread(wxTHREAD_DETACHED) { parent_pointer = handler; should_shutdown = false;}
   	~SocketServerThread();

   	SocketCommunicator *parent_pointer;

   	wxArrayString all_my_ip_addresses;
	wxString my_port_string;
	short int my_port;

	bool should_shutdown;

	wxSocketServer *socket_server;
	bool local_copy_server_is_running;

	protected:

    virtual ExitCode Entry();
};

class SocketClientMonitorThread : public wxThread
{
	public:
	SocketClientMonitorThread(SocketCommunicator *handler) : wxThread(wxTHREAD_DETACHED) { parent_pointer = handler; }
   	~SocketClientMonitorThread();

   	SocketCommunicator *parent_pointer;

   	ArrayOfSocketPointers monitored_sockets;
   	ArrayOfSocketPointers sockets_to_add_next_cycle;
   	ArrayOfSocketPointers sockets_to_remove_next_cycle;
   	ArrayOfSocketPointers sockets_to_remove_and_destroy_next_cycle;


	protected:

    virtual ExitCode Entry();
};
