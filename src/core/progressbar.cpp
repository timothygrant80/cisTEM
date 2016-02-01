#include "core_headers.h"

ProgressBar::ProgressBar(void)
{
	MyPrintWithDetails("An instance of progress bar has been created with no constructor. This is not allowed.\n");
	abort();
}

ProgressBar::~ProgressBar(void)
{
	// Set to 100% and end the line..

	if (total_number_of_ticks > 1)
	{
		wxPrintf("   100% [==============================] done!          \n");
	}

}

ProgressBar::ProgressBar(long wanted_total_number_of_ticks)
{

	total_number_of_ticks = wanted_total_number_of_ticks;

	// check if there is less than 2 ticks, if so - don't do anything..

	if (wanted_total_number_of_ticks > 1)
	{
		start_time = time(NULL);
		last_update_time = start_time;

		// draw the start state...

		wxPrintf("     0% [                              ] ???h:??m:??s   \r");
		fflush(stdout);
	}
}

void ProgressBar::Update(long current_tick)
{
	// zero or negative ticks are not allowed, and should indicate an error..

	if (current_tick < 1)
	{
		MyPrintWithDetails("Called with a tick value less than 1\n");
		abort();
	}


	// if there is less than 2 ticks we don't do anything..

	if (total_number_of_ticks > 1)
	{
		long seconds_remaining;
		long minutes_remaining;
		long hours_remaining;
		long percent_complete;
		double current_seconds_per_tick;
		long remaining_time;
		long filled_bar_size;
		long current_time;

		current_time = time(NULL);

		// Only proceed if we are less than one second from the previous update time, this
		// is here to stop incredibly fast processes just spending all their time drawing
		// the progress bar again and again.
		//
		// of course if the process is fast enough it will just spend all it's time getting
		// the time, then you will have to restrict what ticks you send..

		if (current_time - last_update_time>= 1)
		{

			last_update_time = current_time;
			percent_complete = long(myround(100. / double(total_number_of_ticks) * double(current_tick)));
			current_seconds_per_tick = double(time(NULL) - start_time) / double(current_tick);
			remaining_time = long(myround((total_number_of_ticks - current_tick) * current_seconds_per_tick));
			filled_bar_size = long(myround(double(percent_complete * .3)));

			if (remaining_time > 3600) hours_remaining = remaining_time / 3600;
			else hours_remaining = 0;

			if (remaining_time > 60) minutes_remaining = (remaining_time / 60) - (hours_remaining * 60);
			else minutes_remaining = 0;

			seconds_remaining = remaining_time - ((hours_remaining * 60 + minutes_remaining) * 60);

			// some sanity checking..

			if (filled_bar_size > 30) filled_bar_size = 30;

			// draw out the bar.. starting with percent complete.

			wxPrintf("   %3li% [",percent_complete);
			/*
			cout.width(3);
			cout << right << percent_complete;
			cout.width(0);
			cout << "% [";
			*/

			for (long position = 0; position < 30; position++)
			{
				if (position < filled_bar_size) wxPrintf("=");
				else wxPrintf(" ");
			}

			wxPrintf("] ");

			// print out ETA..

			if (hours_remaining > 999)
			{
				wxPrintf("999:99:99    \r");
			}
			else
			{
				//cout << right << hours_remaining;
				//cout << "h:";
				wxPrintf("%lih:",hours_remaining);

				// minutes, if less than ten do a preceding 0

				//if (minutes_remaining < 10) cout << "0";
				//cout << minutes_remaining;
				wxPrintf("%02lim",minutes_remaining);

				//cout << "m:";

				// similiar for the seconds

				//if (seconds_remaining < 10) cout << "0";
				//cout << seconds_remaining;
				wxPrintf("%02lis  \r", seconds_remaining);
				fflush(stdout);

				// we are done so flush and CR!

				//cout << "s   \r" << flush ;
			}
		}
	}


}


