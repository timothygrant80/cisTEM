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
	int ReturnHighestFindCTFID();
	int ReturnHighestFindCTFJobID();
	int ReturnNumberOfPreviousMovieAlignmentsByAssetID(int wanted_asset_id);
	int ReturnNumberOfPreviousCTFEstimationsByAssetID(int wanted_asset_id);

	int ReturnNumberOfAlignmentJobs();
	int ReturnNumberOfCTFEstimationJobs();

	void GetUniqueAlignmentIDs(int *alignment_job_ids, int number_of_alignmnet_jobs);
	void GetUniqueCTFEstimationIDs(int *ctf_estimation_job_ids, int number_of_ctf_estimation_jobs);

	//Convenience insertion functions..

	//void AddSingleMovieAsset(int movie_asset_id,  wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration);

	bool AddOrReplaceRunProfile(RunProfile *profile_to_add);
	bool DeleteRunProfile(int wanted_id);

	void BeginMovieAssetInsert();
	void AddNextMovieAsset(int movie_asset_id,  wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration);
	void EndMovieAssetInsert();

	void BeginImageAssetInsert();
	void AddNextImageAsset(int image_asset_id,  wxString filename, int position_in_stack, int parent_movie_id, int alignment_id, int ctf_estimation_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration);
	void EndImageAssetInsert();

	void BeginParticlePositionAssetInsert();
	void AddNextParticlePositionAsset(long particle_position_asset_id, long parent_image_asset_id, long pick_job_id, double x_position, double y_position);
	void EndParticlePositionAssetInsert() {EndBatchInsert();};

	// Table creation wrappers..


	bool CreateParticlePositionAssetTable() {return CreateTable("PARTICLE_POSITION_ASSETS", "piirr", "PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "PICK_JOB_ID", "X_POSITION", "Y_POSITION");};
	bool CreateParticlePositionGroupListTable() {return  CreateTable("PARTICLE_POSITION_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );};

	bool CreateVolumeAssetTable() {return CreateTable("VOLUME_ASSETS", "pitriii", "VOLUME_ASSET_ID", "RECONSTRUCTION_JOB_ID", "PIXEL_SIZE", "X_SIZE", "Y_SIZE", "Z_SIZE");};
	bool CreateVolumeGroupListTable() {return  CreateTable("VOLUME_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );};

	// Convenience select functions...

	void BeginAllMovieAssetsSelect();
	MovieAsset GetNextMovieAsset();
	void EndAllMovieAssetsSelect(){EndBatchSelect();};

	void BeginAllMovieGroupsSelect();
	AssetGroup GetNextMovieGroup();
	void EndAllMovieGroupsSelect(){EndBatchSelect();};

	void BeginAllImageAssetsSelect();
	ImageAsset GetNextImageAsset();
	void EndAllImageAssetsSelect(){EndBatchSelect();};

	void BeginAllImageGroupsSelect();
	AssetGroup GetNextImageGroup();
	void EndAllImageGroupsSelect(){EndBatchSelect();};

	void BeginAllParticlePositionAssetsSelect();
	ParticlePositionAsset GetNextParticlePositionAsset();
	void EndAllParticlePositionAssetsSelect() {EndBatchSelect();};

	void BeginAllParticlePositionGroupsSelect();
	AssetGroup GetNextParticlePositionGroup();
	void EndAllParticlePositionGroupsSelect(){EndBatchSelect();};

	void BeginAllVolumeAssetsSelect();
	VolumeAsset GetNextVolumeAsset();
	void EndAllVolumeAssetsSelect() {EndBatchSelect();};

	void BeginAllVolumeGroupsSelect();
	AssetGroup GetNextVolumeGroup();
	void EndAllVolumeGroupsSelect(){EndBatchSelect();};

	void BeginAllRunProfilesSelect();
	RunProfile GetNextRunProfile();
	void EndAllRunProfilesSelect(){EndBatchSelect();};

};
