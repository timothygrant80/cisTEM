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
	long ReturnSingleLongFromSelectCommand(wxString select_command);
	double ReturnSingleDoubleFromSelectCommand(wxString select_command);

	// Get various id numbers and counts

	long ReturnHighestRefinementID();
	long ReturnHighestClassificationID();
	int ReturnHighestAlignmentID();
	int ReturnHighestAlignmentJobID();
	int ReturnHighestFindCTFID();
	int ReturnHighestFindCTFJobID();
	int ReturnHighestPickingID();
	int ReturnHighestPickingJobID();
	int ReturnHighestParticlePositionID();
	int ReturnNumberOfPreviousMovieAlignmentsByAssetID(int wanted_asset_id);
	int ReturnNumberOfPreviousCTFEstimationsByAssetID(int wanted_asset_id);
	int ReturnNumberOfPreviousParticlePicksByAssetID(int wanted_asset_id);

	int ReturnNumberOfAlignmentJobs();
	int ReturnNumberOfCTFEstimationJobs();
	int ReturnNumberOfPickingJobs();
	int ReturnNumberOfImageAssetsWithCTFEstimates();

	void GetUniqueAlignmentIDs(int *alignment_job_ids, int number_of_alignmnet_jobs);
	void GetUniqueCTFEstimationIDs(int *ctf_estimation_job_ids, int number_of_ctf_estimation_jobs);
	void GetUniquePickingJobIDs(int *picking_job_ids, int number_of_picking_jobs);
	void GetUniqueIDsOfImagesWithCTFEstimations(int *image_ids, int &number_of_image_ids);

	void GetActiveDefocusValuesByImageID(long wanted_image_id, float &defocus_1, float &defocus_2, float &defocus_angle);

	void AddRefinementPackageAsset(RefinementPackage *asset_to_add);

	wxArrayLong Return2DClassMembers(long wanted_classifiction_id, int wanted_class);

	//Convenience insertion functions..

	//void AddSingleMovieAsset(int movie_asset_id,  wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration);

	bool AddOrReplaceRunProfile(RunProfile *profile_to_add);
	bool DeleteRunProfile(int wanted_id);

	void BeginMovieAssetInsert();
	void AddNextMovieAsset(int movie_asset_id, wxString name, wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration, wxString gain_filename, int super_resolution_factor);
	void EndMovieAssetInsert();

	void BeginImageAssetInsert();
	void AddNextImageAsset(int image_asset_id,  wxString name, wxString filename, int position_in_stack, int parent_movie_id, int alignment_id, int ctf_estimation_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration);
	void EndImageAssetInsert() {EndBatchInsert();};

	void BeginVolumeAssetInsert();
	void AddNextVolumeAsset(int image_asset_id,  wxString name, wxString filename, int reconstruction_job_id, double pixel_size, int x_size, int y_size, int z_size);
	void EndVolumeAssetInsert() {EndBatchInsert();};


	void BeginParticlePositionAssetInsert();
	//void AddNextParticlePositionAsset(int particle_position_asset_id, int parent_image_asset_id, int pick_job_id, double x_position, double y_position);
	void AddNextParticlePositionAsset(const ParticlePositionAsset *asset);
	void EndParticlePositionAssetInsert() {EndBatchInsert();};

	/*
	void BeginAbInitioParticlePositionAssetInsert();
	void AddNextAbInitioParticlePositionAsset(int particle_position_asset_id, int parent_image_asset_id, int pick_job_id, double x_position, double y_position, double peak_height);
	void EndAbInitioParticlePositionAssetInsert() {EndBatchInsert();};
	*/

	// Table creation wrappers..

	bool CreateParticlePickingListTable() {return CreateTable("PARTICLE_PICKING_LIST", "piiiirrrriiiii", "PICKING_ID", "DATETIME_OF_RUN", "PICKING_JOB_ID", "PARENT_IMAGE_ASSET_ID", "PICKING_ALGORITHM","CHARACTERISTIC_RADIUS", "MAXIMUM_RADIUS","THRESHOLD_PEAK_HEIGHT","HIGHEST_RESOLUTION_USED_IN_PICKING","MIN_DIST_FROM_EDGES","AVOID_HIGH_VARIANCE","AVOID_HIGH_LOW_MEAN","NUM_BACKGROUND_BOXES","MANUAL_EDIT");};
	bool CreateParticlePositionAssetTable() {return CreateTable("PARTICLE_POSITION_ASSETS", "piiirrrirrr", "PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "PICKING_ID", "PICK_JOB_ID", "X_POSITION", "Y_POSITION","PEAK_HEIGHT","TEMPLATE_ASSET_ID","TEMPLATE_PSI","TEMPLATE_THETA","TEMPLATE_PHI");};
	bool CreateParticlePositionGroupListTable() {return  CreateTable("PARTICLE_POSITION_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );};
	bool CreateParticlePickingResultsTable(const int &picking_job_id) {return CreateTable(wxString::Format("PARTICLE_PICKING_RESULTS_%i",picking_job_id),"piirrrirrr","POSITION_ID","PICKING_ID","PARENT_IMAGE_ASSET_ID","X_POSITION", "Y_POSITION","PEAK_HEIGHT","TEMPLATE_ASSET_ID","TEMPLATE_PSI","TEMPLATE_THETA","TEMPLATE_PHI");};

	bool CreateImageAssetTable() {return CreateTable("IMAGE_ASSETS", "pttiiiiiirrr", "IMAGE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "ALIGNMENT_ID", "CTF_ESTIMATION_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION");};
	bool CreateMovieAssetTable() {return CreateTable("MOVIE_ASSETS", "pttiiiirrrrti", "MOVIE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION", "GAIN_FILENAME","SUPER_RESOLUTION_FACTOR");};

	bool CreateVolumeAssetTable() {return CreateTable("VOLUME_ASSETS", "pttiriii", "VOLUME_ASSET_ID", "NAME", "FILENAME", "RECONSTRUCTION_JOB_ID", "PIXEL_SIZE", "X_SIZE", "Y_SIZE", "Z_SIZE");};
	bool CreateVolumeGroupListTable() {return  CreateTable("VOLUME_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );};


	bool CreateRefinementPackageAssetTable() {return CreateTable("REFINEMENT_PACKAGE_ASSETS", "pttitrriii", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "STACK_FILENAME", "STACK_BOX_SIZE", "SYMMETRY", "MOLECULAR_WEIGHT", "PARTICLE_SIZE","NUMBER_OF_CLASSES", "NUMBER_OF_REFINEMENTS", "LAST_REFINEMENT_ID");};
	bool CreateRefinementPackageContainedParticlesTable(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", refinement_package_asset_id), "piirrrrrrrr", "ORIGINAL_PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "POSITION_IN_STACK", "X_POSITION", "Y_POSITION", "PIXEL_SIZE", "DEFOCUS_1", "DEFOCUS_2", "DEFOCUS_ANGLE", "SPHERICAL_ABERRATION", "MICROSCOPE_VOLTAGE");};
	bool CreateRefinementPackageCurrent3DReferencesTable(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", refinement_package_asset_id), "pi", "CLASS_NUMBER", "VOLUME_ASSET_ID");};
	bool CreateRefinementPackageRefinementsList(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", refinement_package_asset_id), "pl", "REFINEMENT_NUMBER", "REFINEMENT_ID");};
	bool CreateRefinementPackageClassificationsList(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", refinement_package_asset_id), "pl", "CLASSIFICATION_NUMBER", "CLASSIFICATION_ID");};

	bool CreateRefinementListTable() {return CreateTable("REFINEMENT_LIST", "Pltillllrrrrrrirrrrirrrrirrir", "REFINEMENT_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "REFINEMENT_WAS_IMPORTED_OR_GENERATED", "DATETIME_OF_RUN", "STARTING_REFINEMENT_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "SIGNED_CC_RESOLUTION_LIMIT", "GLOBAL_RESOLUTION_LIMIT", "GLOBAL_MASK_RADIUS", "NUMBER_RESULTS_TO_REFINE", "ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "CLASSIFICATION_RESOLUTION_LIMIT", "SHOULD_FOCUS_CLASSIFY", "SPHERE_X_COORD", "SPHERE_Y_COORD", "SPHERE_Z_COORD", "SPHERE_RADIUS", "SHOULD_REFINE_CTF", "DEFOCUS_SEARCH_RANGE", "DEFOCUS_SEARCH_STEP", "RESOLUTION_STATISTICS_BOX_SIZE", "RESOLUTION_STATISTICS_PIXEL_SIZE");};
	bool CreateRefinementResultTable(const long refinement_id, const int class_number) {return CreateTable(wxString::Format("REFINEMENT_RESULT_%li_%i", refinement_id, class_number), "Prrrrrrrrrrrrr", "POSITION_IN_STACK", "PSI", "THETA", "PHI", "XSHIFT", "YSHIFT", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE", "OCCUPANCY", "LOGP", "SIGMA", "SCORE", "SCORE_CHANGE");};
	bool CreateRefinementReferenceVolumeIDsTable(const long refinement_id) {return CreateTable(wxString::Format("REFINEMENT_REFERENCE_VOLUME_IDS_%li", refinement_id), "pl", "CLASS_NUMBER", "VOLUME_ASSET_ID");};
	bool CreateRefinementResolutionStatisticsTable(const long refinement_id, int class_counter) {return CreateTable(wxString::Format("REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_id, class_counter), "prrrrr", "SHELL", "RESOLUTION", "FSC", "PART_FSC", "PART_SSNR", "REC_SSNR");};

	bool CreateClassificationListTable() {return CreateTable("CLASSIFICATION_LIST", "Plttilllirrrrrrriir", "CLASSIFICATION_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "CLASS_AVERAGE_FILE", "REFINEMENT_WAS_IMPORTED_OR_GENERATED", "DATETIME_OF_RUN", "STARTING_CLASSIFICATION_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "SMOOTHING_FACTOR", "EXCLUDE_BLANK_EDGES", "AUTO_PERCENT_USED", "PERCENT_USED" );	};
	bool CreateClassificationResultTable(const long classification_id) {return CreateTable(wxString::Format("CLASSIFICATION_RESULT_%li", classification_id), "Prrrirr", "POSITION_IN_STACK", "PSI", "XSHIFT", "YSHIFT", "BEST_CLASS", "SIGMA", "LOGP");};


	void DoVacuum() {ExecuteSQL("VACUUM");}

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
	ParticlePositionAsset GetNextParticlePositionAssetFromResults();
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

	void BeginAllRefinementPackagesSelect();
	RefinementPackage* GetNextRefinementPackage();
	void EndAllRefinementPackagesSelect() {EndBatchSelect();};


	// Convenience CTF parameter function
	void GetCTFParameters( const int &ctf_estimation_id, double &acceleration_voltage, double &spherical_aberration, double &amplitude_constrast, double &defocus_1, double &defocus_2, double &defocus_angle, double &additional_phase_shift );

	// Particle position asset management
	void RemoveParticlePositionsWithGivenParentImageIDFromGroup( const int &group_number_following_gui_convention, const int &parent_image_asset_id);
	void RemoveParticlePositionAssetsPickedFromImagesAlsoPickedByGivenPickingJobID( const int &picking_job_id);
	void RemoveParticlePositionAssetsPickedFromImageWithGivenID( const int &parent_image_asset_id );
	void CopyParticleAssetsFromResultsTable(const int &picking_job_id, const int &parent_image_asset_id);
	void AddArrayOfParticlePositionAssetsToResultsTable(const int &picking_job_id, ArrayOfParticlePositionAssets *array_of_assets);
	void AddArrayOfParticlePositionAssetsToAssetsTable(ArrayOfParticlePositionAssets *array_of_assets);
	ArrayOfParticlePositionAssets ReturnArrayOfParticlePositionAssetsFromResultsTable(const int &picking_job_id, const int &parent_image_asset_id);
	ArrayOfParticlePositionAssets ReturnArrayOfParticlePositionAssetsFromAssetsTable(const int &parent_image_asset_id);

	// Particle picking results management
	void RemoveParticlePositionsFromResultsList(const int &picking_job_id, const int &parent_image_asset_id);
	int ReturnPickingIDGivenPickingJobIDAndParentImageID(const int & picking_job_id, const int &parent_image_asset_id);
	void SetManualEditForPickingID(const int &picking_id, const bool wanted_manual_edit);

	void AddRefinement(Refinement *refinement_to_add);
	void UpdateRefinementResolutionStatistics(Refinement *refinement_to_update);

	Refinement *GetRefinementByID(long wanted_refinement_id);

	void AddClassification(Classification *classification_to_add);
	Classification *GetClassificationByID(long wanted_classification_id);




};
