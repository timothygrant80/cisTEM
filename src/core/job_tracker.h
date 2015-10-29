typedef struct TimeRemaining {
  int hours;
  int minutes;
  int seconds;
} TimeRemaining;

class JobTracker {

	int total_number_of_jobs;
	int total_running_processes;
	int total_number_of_finished_jobs;
	int last_update_time_total_running_processes;

	long start_time;
	long last_update_time;

	float last_update_seconds_per_job_per_process;
	long time_of_last_remaining_time_call;

	int old_percent_complete;
	TimeRemaining old_time_remaining;
	TimeRemaining time_remaining;

	public :

	JobTracker();
	~JobTracker();

	void StartTracking(int wanted_total_number_of_jobs);
	void AddConnection();
	void MarkJobFinished();
	TimeRemaining ReturnRemainingTime();
	inline int ReturnPercentCompleted() {return int(myround((float(total_number_of_finished_jobs) / float(total_number_of_jobs)) * 100.0)); wxPrintf("Returning %i%\n", int(myround((float(total_number_of_finished_jobs) / float(total_number_of_jobs)) * 100.0)));};

	bool ShouldUpdate();

};
