class Database {

	bool is_open;

	sqlite3 *sqlite_database;
	wxFileName database_file;

public :

	Database();
	~Database();

	void Close();

	bool CreateNewDatabase(wxFileName database_file);
	bool Open(wxFileName file_to_open);

	bool CreateTable(const char *table_name, const char *column_format, ...);
	bool InsertOrReplace(const char *table_name, const char *column_format, ...);
	bool GetMasterSettings(wxFileName &project_directory, wxString &project_name, int &imported_integer_version, double &total_cpu_hours, int &total_jobs_run);
	bool CreateAllTables();

};
