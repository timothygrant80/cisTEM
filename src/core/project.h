class Project {



public :


	Database database;

	bool is_open;
	wxString project_name;

	wxFileName project_directory;
	wxFileName movie_asset_directory;
	wxFileName image_asset_directory;
	wxFileName ctf_asset_directory;

	double total_cpu_hours;
	int total_jobs_run;

	Project();
	~Project();

	void Close();
	bool CreateNewProject(wxFileName database_file, wxString project_directory, wxString project_name);
	bool OpenProjectFromFile(wxFileName file_to_open);
	bool ReadMasterSettings();


};
