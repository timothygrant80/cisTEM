
#include "core_headers.h"


StopWatch::StopWatch()
{
	current_index = 0;
	null_time = 0;
	is_new = false;
	is_set_overall = false;
	
	std::string dummyString = "Stopwatch Overhead";
	event_names.push_back(dummyString);
	event_times.push_back(std::chrono::high_resolution_clock::now());
	elapsed_times.push_back(stop(time_fmt, current_index));
}

StopWatch::~StopWatch()
{
	// Do nothing;
}

void StopWatch::start(std::string name)
{
	
	// We want to record the overall timing
	if (! is_set_overall)
	{
		// Set this first to prevent a recursive death
		is_set_overall = true;
		this->start("Overall");		
	}
	// Record overhead.
	event_times[0] = std::chrono::high_resolution_clock::now();
	// Check to see if the event has been encountered. If not, first create it and set elapsed time to zero. Return the events index.
	check_for_name(name);
	// Record the start time for the event.
	if (! is_new)
	{
		event_times[current_index] = std::chrono::high_resolution_clock::now();
	}

	// Record the elapsed time for the start method.
	elapsed_times[0] += stop(time_fmt, 0);
}

void StopWatch::lap(std::string name)
{
	// Record overhead.
	event_times[0] = std::chrono::high_resolution_clock::now();
	// Check to see if the event has been encountered. If not, first create it and set elapsed time to zero. Return the events index.
	check_for_name(name);
	if (is_new) { wxPrintf("a new event name was encountered when calling Stopwatch::lap(%s) at line %d in file %s\n", name, __LINE__, __FILE__); exit(-1); }
	elapsed_times[current_index] += stop(time_fmt, current_index);
	elapsed_times[0] += stop(time_fmt, 0);
}

void StopWatch::check_for_name(std::string name)
{
	// Either add to an existing event, or create a new one.
	for (size_t iName = 0; iName < event_names.size(); iName++)
	{
		if (event_names[iName] == name)
		{
			current_index = iName;
			is_new = false;
			break;
		}
		else
		{
			is_new = true;
		}

	}

	if (is_new)
	{
		event_names.push_back(name);
		current_index = event_names.size() - 1;
		event_times.push_back(std::chrono::high_resolution_clock::now());
		elapsed_times.push_back(null_time);
	}
}

void StopWatch::print_times()
{
	
	this->lap("Overall");
	
	uint64 total_time = 0;
	uint64 missed_time = 0;
	// It would be nice to have a variable option for printing the time format in a given rep different from what was recorded.
	std::string time_string;
	switch (time_fmt)
	{
		case NANOSECONDS :
			time_string = "ns";
			break;

		case MICROSECONDS :
			time_string = "us";
			break;

		case MILLISECONDS :
			time_string = "ms";
			break;

		case SECONDS :
			time_string = "s";
			break;

	}
	
	// Subtract the overhead from the total time
	elapsed_times[1] = elapsed_times[1] - elapsed_times[0];
	
	for (size_t iName = 0; iName < event_names.size(); iName++)
	{
		if (iName > 1)
		{
			total_time += elapsed_times[iName];
		}
		
	}
	
	missed_time = elapsed_times[0] + elapsed_times[1] - total_time;
		
	wxPrintf("\n\n\t\t---------Timing Results---------\n\n");
	for (size_t iName = 0; iName < event_names.size(); iName++)
	{
		convert_time(elapsed_times[iName]);
		if ( iName == 0)
		{
			wxPrintf("\t\t%-20s : %ld %s\n", event_names[iName], elapsed_times[iName], time_string);

		}
		else 
		{
			wxPrintf("\t\t%-20s : %2.2ld:%2.2ld:%2.2ld:%03ld  %7.2f% \n", event_names[iName], hrminsec[0], hrminsec[1], hrminsec[2], hrminsec[3],100.0f*(float)elapsed_times[iName]/(float)total_time);

		}
	}
	
	convert_time(total_time);
	if (missed_time > 0)
	{
		wxPrintf("\n\t\t%-20s : %2.2ld:%2.2ld:%2.2ld:%03ld  %7.2f% \n\n", "Total counted", hrminsec[0], hrminsec[1], hrminsec[2], hrminsec[3],100.0f*(float)(total_time - missed_time)/(float)total_time);
		convert_time(missed_time);
		wxPrintf("\t\t%-20s : %2.2ld:%2.2ld:%2.2ld:%03ld\n", "Total not-counted", hrminsec[0], hrminsec[1], hrminsec[2], hrminsec[3]);		
	}
	else
	{
		wxPrintf("\n\t\t%-20s : %2.2ld:%2.2ld:%2.2ld:%03ld [ ??.??% ]\n\n", "Total counted", hrminsec[0], hrminsec[1], hrminsec[2], hrminsec[3]);
		wxPrintf("\t\t%-20s : %s\n", "Total not-counted", "was negative indicating overlapping events.");		
	}

	wxPrintf("\n\t\t--------------------------------\n\n");

}


void StopWatch::convert_time(uint64_t microsec)
{
	// Time is stored in microseconds
	uint64 time_rem;
	time_rem = microsec / 3600000000;
	hrminsec[0] = time_rem;

	microsec -= (time_rem * 3600000000);

	time_rem = microsec / 60000000;
	hrminsec[1] = time_rem;

	microsec -= (time_rem * 60000000);

	time_rem = microsec / 1000000;
	hrminsec[2] = time_rem;

	microsec -= (time_rem * 1000000);

	time_rem = microsec / 1000;
	hrminsec[3] = time_rem;

	microsec -= (time_rem * 1000);

}






uint64_t StopWatch::stop(TimeFormat T, int idx)
{
	const auto current_time = std::chrono::high_resolution_clock::now();
	const auto previous_time = event_times[idx];
	return ticks(T, previous_time, current_time);
}

uint64_t StopWatch::ticks(TimeFormat T, const time_pt& start_time, const time_pt& end_time)
{
	const auto duration = end_time - start_time;
	const uint64_t ns_count = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
	uint64_t up;

	switch (T)
	{
		case TimeFormat::MICROSECONDS :
			up = ((ns_count / 100)%10 >= 5) ? 1 : 0;
			up += (ns_count/1000);
			break;


		case TimeFormat::MILLISECONDS :
			up = ((ns_count / 100000)%10 >= 5) ? 1 : 0;
			up += (ns_count/1000000);
			break;

		case TimeFormat::SECONDS :
			up = ((ns_count / 100000000)%10 >= 5) ? 1 : 0;
			up += (ns_count/1000000000);
			break;
	}
	return up;

}

