class Database {

	bool is_open;
	bool in_batch_insert;
	bool in_batch_select;

	sqlite3 *sqlite_database;
	wxFileName database_file;

	sqlite3_stmt *batch_statement;

public :


	int last_return_code;

	Database();
	~Database();

	void Close();

	bool CreateNewDatabase(wxFileName database_file);
	bool Open(wxFileName file_to_open);

	bool CreateTable(const char *table_name, const char *column_format, ...);
	bool DeleteTable(const char *table_name);
	//bool DeleteTable(const char *table_name);
	bool InsertOrReplace(const char *table_name, const char *column_format, ...);
	bool GetMasterSettings(wxFileName &project_directory, wxString &project_name, int &imported_integer_version, double &total_cpu_hours, int &total_jobs_run);
	bool CreateAllTables();

	void BeginBatchInsert(const char *table_name, int number_of_columns, ...);
	void AddToBatchInsert(const char *column_format, ...);
	void EndBatchInsert();

	bool BeginBatchSelect(const char *select_command);
	bool GetFromBatchSelect(const char *column_format, ...);
	void EndBatchSelect();

	bool ExecuteSQL(const char *command);
	int ReturnSingleIntFromSelectCommand(wxString select_command);

	// Get various id numbers and counts

	int ReturnHighestAlignmentID();
	int ReturnHighestAlignmentJobID();
	int ReturnNumberOfPreviousMovieAlignmentsByAssetID(int wanted_asset_id);
	int ReturnNumberOfPreviousCTFEstimationsByAssetID(int wanted_asset_id);

	int ReturnNumberOfAlignmentJobs();

	void GetUniqueAlignmentIDs(int *alignment_job_ids, int number_of_alignmnet_jobs);

	//Convenience insertion functions..

	//void AddSingleMovieAsset(int movie_asset_id,  wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration);

	bool AddOrReplaceRunProfile(RunProfile *profile_to_add);
	bool DeleteRunProfile(int wanted_id);

	void BeginMovieAssetInsert();
	void AddNextMovieAsset(int movie_asset_id,  wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration);
	void EndMovieAssetInsert();

	void BeginImageAssetInsert();
	void AddNextImageAsset(int image_asset_id,  wxString filename, int position_in_stack, int parent_movie_id, int alignment_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration);
	void EndImageAssetInsert();

	// Convenience select functions...

	void BeginAllMovieAssetsSelect();
	MovieAsset GetNextMovieAsset();
	void EndAllMovieAssetsSelect();

	void BeginAllMovieGroupsSelect();
	AssetGroup GetNextMovieGroup();
	void EndAllMovieGroupsSelect();

	void BeginAllImageAssetsSelect();
	ImageAsset GetNextImageAsset();
	void EndAllImageAssetsSelect();

	void BeginAllImageGroupsSelect();
	AssetGroup GetNextImageGroup();
	void EndAllImageGroupsSelect();

	void BeginAllRunProfilesSelect();
	RunProfile GetNextRunProfile();
	void EndAllRunProfilesSelect();

};
