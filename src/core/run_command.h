class RunCommand {

  public:
    RunCommand( );
    ~RunCommand( );

    wxString command_to_run;
    int      number_of_copies;
    int      number_of_threads_per_copy;
    bool     override_total_copies;
    int      overriden_number_of_copies;
    int      delay_time_in_ms;

    bool operator==(const RunCommand& other) const;
    bool operator!=(const RunCommand& other) const;

    void SetCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_number_of_threads_per_copy, bool wanted_override_total_copies, int wanted_overriden_number_of_copies, int wanted_delay_time_in_ms);
};