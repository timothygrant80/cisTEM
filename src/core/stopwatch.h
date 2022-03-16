/*
 * StopWatch.h
 *
 *  Created on: Nov 18, 2019
 *      Author: himesb
 */

namespace cistem_timer_noop {

class StopWatch {

  public:
    StopWatch( );
    virtual ~StopWatch( );

    // dummy methods
    inline void start(std::string name, bool thread_safe = true) { return; }

    inline void lap(std::string name, bool thread_safe = true) { return; }

    inline void print_times(bool thread_safe = true) { return; }

    inline void mark_entry_or_exit_point(bool thread_safe = true) { return; }
};

} // namespace cistem_timer_noop

namespace cistem_timer {
class StopWatch {

  public:
    StopWatch( );
    virtual ~StopWatch( );

    // Create or reuse event named "name", start timing. May overlap with other timing events.
    void start(std::string name, bool thread_safe = true);

    // Record the elapsed time since last "start" for this event. Add to cummulative time.
    void lap(std::string name, bool thread_safe = true);

    // Print out all event times, including stopwatch overhead time.
    void print_times(bool thread_safe = true);

    // Start or pause the total elapsed time when passing a stopwatch pointer to a method. Place inside the method at the entry and exit point of the method call.
    void mark_entry_or_exit_point(bool thread_safe = true);

  private:
    enum TimeFormat : int { NANOSECONDS,
                            MICROSECONDS,
                            MILLISECONDS,
                            SECONDS };

    enum SpecialIDX : int { total_elapsed  = 0,
                            total_measured = 1 };

    typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_pt;

    int                      number_of_events_being_recorded;
    std::vector<std::string> event_names   = { };
    std::vector<time_pt>     event_times   = { };
    std::vector<uint64_t>    elapsed_times = { };

    uint64_t   hrminsec[4] = {0, 0, 0, 0};
    size_t     current_index;
    uint64_t   null_time;
    bool       is_new;
    bool       is_set_overall;
    bool       is_recording_measured_time;
    bool       is_recording_elapsed_time;
    bool       is_entry_point;
    TimeFormat time_fmt = MICROSECONDS;

    uint64_t stop(TimeFormat T, int idx);
    uint64_t ticks(TimeFormat T, const time_pt& start_time, const time_pt& end_time);

    // Parse time into a more readable hours:minutes:seconds:milliseconds format for display.
    void convert_time(uint64_t microsec);

    // Check to see if an event named "name" already exists, if not, initialize it.
    void check_for_name_and_set_current_idx(std::string name);

    void record_start(std::string name);
    void record_end( );
    void record_elapsed( );
    void record_measured( );
};
} // namespace cistem_timer
