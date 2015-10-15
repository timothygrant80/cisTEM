/*  \brief  Master Job Controller for the gui, for tracking and communicating with all running jobs.


*/

#define MAX_GUI_JOBS 1000

class GuiJob {



public :

	bool is_active;
	unsigned char job_code[SOCKET_CODE_SIZE];

	JobPanel *parent_panel;
	wxSocketBase *socket;

	wxString launch_command;

	GuiJob();
	GuiJob(JobPanel *wanted_parent_panel);

	void LaunchJob();
	void KillJob();

};

class GuiJobController {


public:

	long job_index_tracker;
	long number_of_running_jobs;

	GuiJob job_list[MAX_GUI_JOBS]; // FIXED NUMBER - BAD!


	GuiJobController();

	long AddJob(JobPanel *wanted_parent_panel, wxString wanted_launch_command);
	//void LaunchJob(unsigned char *job_code,  wxString launch_command);
	bool LaunchJob(GuiJob *job_to_launch);
	long FindFreeJobSlot();
	long ReturnJobNumberFromJobCode(unsigned char *job_code);
	void GenerateJobCode(unsigned char *job_code);


};
