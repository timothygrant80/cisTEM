#include <wx/socket.h>

#define RUN_LOCAL 0
#define RUN_PBS 1
#define RUN_SSH 2


class ProcessLauncher {

	long process_type;

	long wanted_number_of_processes;
	long number_of_connected_processes;

	wxIPV4address gui_address;
	wxIPV4address my_address;
	wxIPV4address master_process_address;


	long my_port;
	long gui_port;

public:

	ProcessLauncher();
	~ProcessLauncher();

	virtual void LaunchMasterProcess;
	virtual void LaunchProcess;

	virtual void SendGuiMessage;
	virtual void SendMasterMessage;



};
