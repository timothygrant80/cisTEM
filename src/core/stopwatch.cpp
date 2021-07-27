
#include "core_headers.h"

namespace cistem_timer_noop {

// All the other methods are defined as inline in the header to ensure they are optimized out.
StopWatch::StopWatch() {
	// do nothing
}	
StopWatch::~StopWatch() {
	// do nothing
}

}

namespace cistem_timer {

StopWatch::StopWatch()
{
	null_time = 0;
	is_new = false;
	is_set_overall = false;
	is_recording_measured_time = false;
	is_recording_elapsed_time = false;
	is_entry_point = true;
	number_of_events_being_recorded = 0;
	
	current_index = (SpecialIDX)total_elapsed;
	event_names.push_back("Total elapsed");
	event_times.push_back(std::chrono::high_resolution_clock::now());
	elapsed_times.push_back(null_time);

	current_index = (SpecialIDX)total_measured;
	event_names.push_back("Total measured");
	event_times.push_back(std::chrono::high_resolution_clock::now());
	elapsed_times.push_back(null_time);
}

StopWatch::~StopWatch()
{
	// Do nothing;
}


void StopWatch::record_start(std::string name) 
{
	if (is_new)
	{
		event_names.push_back(name);
		current_index = event_names.size() - 1;
		event_times.push_back(std::chrono::high_resolution_clock::now());
		elapsed_times.push_back(null_time);
	}
	else
	{
		event_times[current_index] = std::chrono::high_resolution_clock::now();
	}
	// We've added a new event if the idx is not special
	if (current_index != (SpecialIDX)total_elapsed && current_index != (SpecialIDX)total_measured) {number_of_events_being_recorded++;}
}

void StopWatch::record_end() 
{

	elapsed_times[current_index] += stop(time_fmt, current_index);
	// We've removed an event if the idx is not special
	if (current_index != (SpecialIDX)total_elapsed && current_index != (SpecialIDX)total_measured) {number_of_events_being_recorded--;}
}

void StopWatch::record_elapsed() 
{
	if (is_recording_elapsed_time)
	{
		elapsed_times[(SpecialIDX)total_elapsed] += stop(time_fmt,(SpecialIDX)total_elapsed);
		is_recording_elapsed_time = false;
	}
  else
	{
		event_times[(SpecialIDX)total_elapsed] = std::chrono::high_resolution_clock::now();
		is_recording_elapsed_time = true;
	}
}

void StopWatch::record_measured()
{
	if (is_recording_measured_time)
	{
		if (number_of_events_being_recorded == 0)
		{
			// stop the measured time
			elapsed_times[(SpecialIDX)total_measured] += stop(time_fmt, (SpecialIDX)total_measured);
			is_recording_measured_time = false;
		}
	}
	else
	{
		if (number_of_events_being_recorded > 0)
		{
			// start the measured time
			event_times[(SpecialIDX)total_measured] = std::chrono::high_resolution_clock::now();
			is_recording_measured_time = true;
		}

	}
}

void StopWatch::mark_entry_or_exit_point(bool threadsafe)
{
	if (threadsafe && ReturnThreadNumberOfCurrentThread() != 0) return;
	if (is_entry_point)
	{
		// If we are already recording, we don't need to do anything, but that shouldn't happen'
		if ( ! is_recording_elapsed_time )
		{
			if (! is_set_overall) is_set_overall = true;
			record_elapsed();
		}

		is_entry_point = false;

	}
	else
	{
		// If we are not recording on an exit point, we have a problem.
		if (! is_recording_elapsed_time) { wxPrintf("a exit point was encounterd with no elapsed time being recorded Stopwatch at line %d in file %s\n", __LINE__, __FILE__); exit(-1); }
		record_elapsed();
		is_entry_point = true;
	}

}

void StopWatch::start(std::string name, bool threadsafe)
{
	// Typically we only want to track a single thread, which is default. Override this with threadsafe = false;
	if (threadsafe && ReturnThreadNumberOfCurrentThread() != 0) return;
	// We want to record the overall timing, triggered by the first start() call.
	if (! is_set_overall)
	{
		// 
		is_set_overall = true;
		record_elapsed();
	}

	// Check to see if the event has been encountered. If not, first create it and set elapsed time to zero. Return the events index.
	check_for_name_and_set_current_idx(name);
	record_start(name);	
	record_measured();

}

void StopWatch::lap(std::string name, bool threadsafe)
{
	// Typically we only want to track a single thread, which is default. Override this with threadsafe = false;
	if (threadsafe && ReturnThreadNumberOfCurrentThread() != 0) return;

	// Check to see if the event has been encountered. If not, first create it and set elapsed time to zero. Return the events index.
	check_for_name_and_set_current_idx(name);
	if (is_new) { wxPrintf("a new event name was encountered when calling Stopwatch::lap(%s) at line %d in file %s\n", name, __LINE__, __FILE__); exit(-1); }
	record_end();
	record_measured();

}

void StopWatch::check_for_name_and_set_current_idx(std::string name)
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

}

void StopWatch::print_times(bool threadsafe)
{
		// Typically we only want to track a single thread, which is default. Override this with threadsafe = false;
	if (threadsafe && ReturnThreadNumberOfCurrentThread() != 0) return;
	if (is_recording_elapsed_time) record_elapsed();
	

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
	


		
	wxPrintf("\n\n\t\t---------Timing Results---------\n\n");
	for (size_t iName = 0; iName < event_names.size(); iName++)
	{

		convert_time(elapsed_times[iName]);

		switch( iName )
		{

			case (SpecialIDX)total_elapsed: 
				wxPrintf("\t\t%-32s : %2.2ld:%2.2ld:%2.2ld:%03ld\n", event_names[iName], hrminsec[0], hrminsec[1], hrminsec[2], hrminsec[3]);
				break;
			case (SpecialIDX)total_measured:
				wxPrintf("\t\t%-32s : %2.2ld:%2.2ld:%2.2ld:%03ld  %7.2f% \n", event_names[iName], hrminsec[0], hrminsec[1], hrminsec[2], hrminsec[3],100.0f*(float)elapsed_times[iName]/(float)elapsed_times[(SpecialIDX)total_elapsed]);
				break;
			default:
				wxPrintf("\t\t%-32s : %2.2ld:%2.2ld:%2.2ld:%03ld  %7.2f% \n", event_names[iName], hrminsec[0], hrminsec[1], hrminsec[2], hrminsec[3],100.0f*(float)elapsed_times[iName]/(float)elapsed_times[(SpecialIDX)total_measured]);
				break;
		}

	}
	
	wxPrintf("\n\t\t--------------------------------\n\n");

}


void StopWatch::convert_time(uint64_t microsec)
{
	// Time is stored in microseconds
	uint64_t time_rem;
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
		case TimeFormat::NANOSECONDS :
			up = ns_count;
			break;


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

