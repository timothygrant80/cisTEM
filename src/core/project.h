class Project {



public :


	Database database;

	bool is_open;
	wxString project_name;

	wxFileName project_directory;
	wxFileName movie_asset_directory;
	wxFileName image_asset_directory;
	wxFileName volume_asset_directory;
	wxFileName ctf_asset_directory;
	wxFileName particle_position_asset_directory;
	wxFileName particle_stack_directory;
	wxFileName class_average_directory;

	wxFileName parameter_file_directory;
	wxFileName scratch_directory;

	double total_cpu_hours;
	int total_jobs_run;

	Project();
	~Project();

	void Close(bool remove_lock = true);
	bool CreateNewProject(wxFileName database_file, wxString project_directory, wxString project_name);
	bool OpenProjectFromFile(wxFileName file_to_open);
	bool ReadMasterSettings();
	bool WriteProjectStatisticsToDatabase();


};
