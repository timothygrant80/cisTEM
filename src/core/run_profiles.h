class RunCommand {

public:
	RunCommand();
	~RunCommand();

	wxString command_to_run;
	int number_of_copies;
	int delay_time_in_ms;

	void SetCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_delay_time_in_ms);
};

class RunProfile {

public:

	int id;
	long number_of_run_commands;
	long number_allocated;

	public :

	RunProfile();
	RunProfile( const RunProfile &obj); // copy contructor
	~RunProfile();

	wxString name;
	wxString manager_command;

	wxString gui_address;
	wxString controller_address;

	RunCommand *run_commands;

	wxString executable_name;

	void AddCommand(RunCommand wanted_command);
	void AddCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_delay_time_in_ms);
	void RemoveCommand(int number_to_remove);
	void RemoveAll();
	long ReturnTotalJobs();
	void SubstituteExecutableName(wxString executable_name);

	RunProfile & operator = (const RunProfile &t);
	RunProfile & operator = (const RunProfile *t);
};


class RunProfileManager {

public:

	int current_id_number;
	long number_of_run_profiles;
	long number_allocated;

	RunProfile *run_profiles;

	void AddProfile(RunProfile *profile_to_add);
	void AddBlankProfile();
	void AddDefaultLocalProfile();
	void RemoveProfile(int number_to_remove);
	void RemoveAllProfiles();

	RunProfile * ReturnLastProfilePointer();
	RunProfile * ReturnProfilePointer(int wanted_profile);

	wxString ReturnProfileName(long wanted_profile);
	long ReturnProfileID(long wanted_profile);
	long ReturnTotalJobs(long wanted_profile);


	RunProfileManager();
	~RunProfileManager();


};
