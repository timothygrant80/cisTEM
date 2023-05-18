class RunProfile {

  public:
    int  id;
    long number_of_run_commands;
    long number_allocated;

  public:
    RunProfile( );
    RunProfile(const RunProfile& obj); // copy contructor
    ~RunProfile( );

    wxString name;
    wxString manager_command;

    wxString gui_address;
    wxString controller_address;

    RunCommand* run_commands;

    wxString executable_name;

    void AddCommand(RunCommand wanted_command);
    void AddCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_number_of_threads_per_copy, bool wanted_override_total_copies, int wanted_overriden_number_of_copies, int wanted_delay_time_in_ms);
    void RemoveCommand(int number_to_remove);
    void RemoveAll( );
    long ReturnTotalJobs( );
    void SubstituteExecutableName(wxString executable_name);

    RunProfile& operator=(const RunProfile& t);
    RunProfile& operator=(const RunProfile* t);
    bool        operator==(const RunProfile& t);
    bool        operator==(const RunProfile* t);
    bool        operator!=(const RunProfile& t);
    bool        operator!=(const RunProfile* t);

    void CheckNumberAndGrow( );
};
