/*
 * StopWatch.h
 *
 *  Created on: Nov 18, 2019
 *      Author: himesb
 */

class StopWatch {


public:

	enum TimeFormat : int { NANOSECONDS, MICROSECONDS, MILLISECONDS, SECONDS };
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_pt;


	std::vector<std::string> event_names = {};
	std::vector<time_pt> 	 event_times = {};
	std::vector<uint64_t> 	 elapsed_times = {};

	uint64 hrminsec[4] = {0,0,0,0};
	size_t current_index;
	uint64_t null_time;
	bool is_new;
	bool is_set_overall;
	TimeFormat time_fmt = MICROSECONDS;


	StopWatch();
	virtual ~StopWatch();


	// Create or reuse event named "name", start timing. May overlap with other timing events.
	void start(std::string name);

	// Record the elapsed time since last "start" for this event. Add to cummulative time.
	void lap(std::string name);

	// Check to see if an event named "name" already exists, if not, initialize it.
	void check_for_name(std::string name);

	// Print out all event times, including stopwatch overhead time.
	void print_times();

	// Parse time into a more readable hours:minutes:seconds:milliseconds format for display.
	void convert_time(uint64_t microsec);



private:

	uint64_t stop(TimeFormat T, int idx);


	uint64_t ticks(TimeFormat T, const time_pt& start_time, const time_pt& end_time);


};
