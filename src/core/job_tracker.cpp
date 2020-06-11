#include "core_headers.h"


JobTracker::JobTracker()
{
	total_number_of_jobs = 0;
	total_running_processes = 0;
	total_number_of_finished_jobs = 0;
	time_of_last_remaining_time_call = 0;

	start_time = 0;
	last_update_time = 0;

	last_update_time_total_running_processes = 0;
	last_update_seconds_per_job_per_process = 0;

	time_remaining.hours = 99;
	time_remaining.minutes = 99;
	time_remaining.seconds = 99;

	old_time_remaining.hours = -1;
	old_time_remaining.minutes = -1;
	old_time_remaining.seconds = -1;

	old_percent_complete = -1;

}
JobTracker::~JobTracker()
{

}

void JobTracker::StartTracking(int wanted_total_number_of_jobs)
{
	total_number_of_jobs = wanted_total_number_of_jobs;
	total_running_processes = 0;
	total_number_of_finished_jobs = 0;

	start_time = time(NULL);
	last_update_time = start_time;
	last_update_seconds_per_job_per_process = -1;
	last_update_time_total_running_processes = 0;

	time_remaining.hours = 99;
	time_remaining.minutes = 99;
	time_remaining.seconds = 99;

	old_time_remaining.hours = -1;
	old_time_remaining.minutes = -1;
	old_time_remaining.seconds = -1;

}

void JobTracker::AddConnection()
{
	total_running_processes++;
}

void JobTracker::MarkJobFinished()
{
	long current_time = time(NULL);
	long seconds_per_job_per_process_since_last_update;
	long time_since_last_update = current_time - last_update_time;
	long total_time = current_time - start_time;
	float naive_time_per_process = (float(total_time) / float(total_number_of_finished_jobs));
/*
	if (last_update_time_total_running_processes == 0)
	{
		// this is the first job, and provides us with an estimate for how long one job takes..

		last_update_seconds_per_job_per_process = time_since_last_update;
	}
	else
	{
		seconds_per_job_per_process_since_last_update = float(time_since_last_update) / float(last_update_time_total_running_processes);

		last_update_seconds_per_job_per_process *= float(total_number_of_finished_jobs);
		last_update_seconds_per_job_per_process += float(seconds_per_job_per_process_since_last_update);
		last_update_seconds_per_job_per_process /= float(total_number_of_finished_jobs + 1);
	}

	//wxPrintf("Job finished new process time = %f seconds\nnaive time = %f seconds\n", last_update_seconds_per_job_per_process, naive_time_per_process);
	total_number_of_finished_jobs++;

	last_update_time_total_running_processes = total_running_processes;
	if (naive_time_per_process < last_update_seconds_per_job_per_process) last_update_seconds_per_job_per_process = naive_time_per_process;*/

	total_number_of_finished_jobs++;
	last_update_seconds_per_job_per_process = naive_time_per_process;
}

TimeRemaining JobTracker::ReturnRemainingTime()
{
	long current_time = time(NULL);

	if (current_time - time_of_last_remaining_time_call >= 1)
	{
			long seconds_remaining = (total_number_of_jobs - total_number_of_finished_jobs) * last_update_seconds_per_job_per_process;

			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);

			time_of_last_remaining_time_call = current_time;
	}

	//wxPrintf("Per process = %f, Returning %ih:%im:%is\n", last_update_seconds_per_job_per_process, time_remaining.hours, time_remaining.minutes, time_remaining.seconds);

	return time_remaining;

}

TimeRemaining JobTracker::ReturnTimeSinceStart()
{
	long current_time = time(NULL);
	long seconds_since_start = current_time - start_time;
	TimeRemaining time_since_start;

	if (seconds_since_start > 3600) time_since_start.hours = seconds_since_start / 3600;
	else time_since_start.hours = 0;

	if (seconds_since_start > 60) time_since_start.minutes = (seconds_since_start / 60) - (time_since_start.hours * 60);
	else time_since_start.minutes = 0;

	time_since_start.seconds = seconds_since_start - ((time_since_start.hours * 60 + time_since_start.minutes) * 60);

	return time_since_start;
}

bool JobTracker::ShouldUpdate()
{
	TimeRemaining new_time_remaining = ReturnRemainingTime();
	int new_percentage_complete = ReturnPercentCompleted();

	bool should_update = false;

	if (old_percent_complete != new_percentage_complete)
	{
		old_percent_complete = new_percentage_complete;
		should_update = true;
	}



	if (new_time_remaining.hours != old_time_remaining.hours || new_time_remaining.minutes != old_time_remaining.minutes || new_time_remaining.seconds != old_time_remaining.seconds)
	{
		old_time_remaining = new_time_remaining;
		should_update = true;
	}


	return should_update;

}

