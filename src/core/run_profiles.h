class RunCommand {

public:
	RunCommand();
	~RunCommand();

	wxString command_to_run;
	int number_of_copies;

	void SetCommand(wxString wanted_command, int wanted_number_of_copies);
};

class RunProfile {

public:
	long number_of_run_commands;
	long number_allocated;

	public :

	RunProfile();
	~RunProfile();

	wxString name;
	wxString manager_command;
	RunCommand *run_commands;

	void AddCommand(RunCommand wanted_command);
	void AddCommand(wxString wanted_command, int wanted_number_of_copies);
	void RemoveCommand(int number_to_remove);

	RunProfile & operator = (const RunProfile &t);
};


class RunProfileManager {

public:

	long number_of_run_profiles;
	long number_allocated;

	RunProfile *run_profiles;

	void AddProfile(RunProfile profile_to_add);
	void AddBlankProfile();
	void RemoveProfile(int number_to_remove);
	void RemoveAllProfiles();

	RunProfileManager();
	~RunProfileManager();


};
