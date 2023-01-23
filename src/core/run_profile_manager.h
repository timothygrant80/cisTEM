class RunProfileManager {

  public:
    int  current_id_number;
    long number_of_run_profiles;
    long number_allocated;

    RunProfile* run_profiles;

    void AddProfile(RunProfile* profile_to_add);
    void AddBlankProfile( );
    void AddDefaultLocalProfile( );
    void RemoveProfile(int number_to_remove);
    void RemoveAllProfiles( );

    RunProfile* ReturnLastProfilePointer( );
    RunProfile* ReturnProfilePointer(int wanted_profile);

    wxString ReturnProfileName(long wanted_profile);
    long     ReturnProfileID(long wanted_profile);
    long     ReturnTotalJobs(long wanted_profile);

    void WriteRunProfilesToDisk(wxString filename, wxArrayInt profiles_to_write);
    bool ImportRunProfilesFromDisk(wxString filename);

    void CheckNumberAndGrow( );

    RunProfileManager( );
    ~RunProfileManager( );
};
