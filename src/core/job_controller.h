/*  \brief  Master Job Controller for the gui, for tracking and communicating with all running jobs.


*/



class Job {



public :

	bool is_active;
	unsigned char job_code[SOCKET_CODE_SIZE];

	void LaunchJob();
	void KillJob();

};

class JobController {


public:

	long job_index_tracker;
	long number_of_running_jobs;

	GuiJob job_list[MAX_GUI_JOBS]; // FIXED NUMBER - BAD!


	GuiJobController();

	long AddJob(JobPanel *wanted_parent_panel);
	void LaunchJob(unsigned char *job_code);
	long FindFreeJobSlot();
	long ReturnJobNumberFromJobCode(unsigned char *job_code);
	void GenerateJobCode(unsigned char *job_code);


};
