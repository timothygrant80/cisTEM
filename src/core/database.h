class Database {

	bool is_open;
	bool in_batch_insert;
	bool in_batch_select;

	int number_of_active_transactions;

	sqlite3_stmt *batch_statement;
	bool should_do_local_commit; // used for functions to decide whether to do a commit themselves, or to leave it for someone else to commit. if already in a begin, it won't do begin commit.

public :

	sqlite3 *sqlite_database;
	int last_return_code;
	wxFileName database_file;

	Database();
	~Database();

	void Close(bool remove_lock = true);

	wxString ReturnFilename() {return database_file.GetFullPath();};

	bool CreateNewDatabase(wxFileName database_file);
	bool Open(wxFileName file_to_open, bool disable_locking = false);


	inline void Begin()
	{
		if (number_of_active_transactions == 0) ExecuteSQL("BEGIN IMMEDIATE;");// we can start, otherwise, we are already in begin commit
		number_of_active_transactions++;

	//	MyPrintWithDetails("\nBegin %i\n", number_of_active_transactions);
	}

	inline void Commit()
	{
		number_of_active_transactions--;
		if (number_of_active_transactions == 0)	ExecuteSQL("COMMIT;");

		//MyPrintWithDetails("\nCommit %i\n", number_of_active_transactions);
	}

	bool CreateTable(const char *table_name, const char *column_format, ...);
	bool DeleteTable(const char *table_name);
	bool AddColumnToTable(wxString table_name, wxString column_name, wxString column_format, wxString default_value);
	//bool DeleteTable(const char *table_name);
	bool InsertOrReplace(const char *table_name, const char *column_format, ...);
	bool GetMasterSettings(wxFileName &project_directory, wxString &project_name, int &imported_integer_version, double &total_cpu_hours, int &total_jobs_run);
	bool SetProjectStatistics(double &total_cpu_hours, int &total_jobs_run);
	bool CreateAllTables();

	void BeginBatchInsert(const char *table_name, int number_of_columns, ...);
	void AddToBatchInsert(const char *column_format, ...);
	void EndBatchInsert();

	bool BeginBatchSelect(const char *select_command);
	bool GetFromBatchSelect(const char *column_format, ...);
	void EndBatchSelect();

	int ExecuteSQL(const char *command);
	int Prepare(wxString select_command, sqlite3_stmt **current_statement);
	int Step(sqlite3_stmt *current_statement);
	int Finalize(sqlite3_stmt *current_statement);
	void CheckBindCode(int return_code);

	int ReturnSingleIntFromSelectCommand(wxString select_command);
	long ReturnSingleLongFromSelectCommand(wxString select_command);
	double ReturnSingleDoubleFromSelectCommand(wxString select_command);

	wxArrayInt ReturnIntArrayFromSelectCommand(wxString select_command);
	wxArrayLong ReturnLongArrayFromSelectCommand(wxString select_command);
	wxArrayString ReturnStringArrayFromSelectCommand(wxString select_command);

	bool DoesTableExist(wxString table_name);
	bool DoesColumnExist(wxString table_name, wxString column_name);

	void ReturnProcessLockInfo(long &active_process_id, wxString &active_hostname);
	void SetProcessLockInfo(long &active_process_id, wxString &active_hostname);

	long ReturnRefinementIDGivenReconstructionID(long reconstruction_id);

	// Get various id numbers and counts

	long ReturnHighestRefinementID();
	long ReturnHighestStartupID();
	long ReturnHighestReconstructionID();
	long ReturnHighestClassificationID();
	int ReturnHighestAlignmentID();
	int ReturnHighestAlignmentJobID();
	int ReturnHighestFindCTFID();
	int ReturnHighestFindCTFJobID();
	int ReturnHighestPickingID();
	int ReturnHighestPickingJobID();
	int ReturnHighestParticlePositionID();
	long ReturnHighestClassumSelectionID();

	int ReturnHighestTemplateMatchID();
	int ReturnHighestTemplateMatchJobID();
	void SetActiveTemplateMatchJobForGivenImageAssetID(long image_asset, long template_match_job_id);

	int ReturnNumberOfPreviousMovieAlignmentsByAssetID(int wanted_asset_id);
	int ReturnNumberOfPreviousTemplateMatchesByAssetID(int wanted_asset_id);
	int ReturnNumberOfPreviousCTFEstimationsByAssetID(int wanted_asset_id);
	int ReturnNumberOfPreviousParticlePicksByAssetID(int wanted_asset_id);

	int ReturnNumberOfAlignmentJobs();
	int ReturnNumberOfCTFEstimationJobs();
	int ReturnNumberOfTemplateMatchingJobs();
	int ReturnNumberOfPickingJobs();
	int ReturnNumberOfImageAssetsWithCTFEstimates();



	void GetUniqueAlignmentIDs(int *alignment_job_ids, int number_of_alignmnet_jobs);
	void GetUniqueCTFEstimationIDs(int *ctf_estimation_job_ids, int number_of_ctf_estimation_jobs);
	void GetUniqueTemplateMatchIDs(long *template_match_job_ids, int number_of_template_match_jobs);
	void GetUniquePickingJobIDs(int *picking_job_ids, int number_of_picking_jobs);
	void GetUniqueIDsOfImagesWithCTFEstimations(int *image_ids, int &number_of_image_ids);

	void GetMovieImportDefaults(float &voltage, float &spherical_aberration, float &pixel_size, float &exposure_per_frame, bool &movies_are_gain_corrected, wxString &gain_reference_filename, bool &movies_are_dark_corrected, wxString dark_reference_filename, bool &resample_movies, float &desired_pixel_size, bool &correct_mag_distortion, float &mag_distortion_angle, float &mag_distortion_major_scale, float &mag_distortion_minor_scale, bool &protein_is_white, int &eer_super_res_factor, int &eer_frames_per_image);
	void GetImageImportDefaults(float &voltage, float &spherical_aberration, float &pixel_size, bool &protein_is_white);

	void GetActiveDefocusValuesByImageID(long wanted_image_id, float &defocus_1, float &defocus_2, float &defocus_angle, float &phase_shift, float &amplitude_contrast, float &tilt_angle, float &tilt_axis);

	void AddRefinementPackageAsset(RefinementPackage *asset_to_add);

	wxArrayLong Return2DClassMembers(long wanted_classifiction_id, int wanted_class);
	int ReturnNumberOf2DClassMembers(long wanted_classification_id, int wanted_class_number);

	//Convenience insertion functions..

	//void AddSingleMovieAsset(int movie_asset_id,  wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration);

	void AddOrReplaceRunProfile(RunProfile *profile_to_add);
	void DeleteRunProfile(int wanted_id);

	void BeginMovieAssetInsert();
	void AddNextMovieAsset(int movie_asset_id,  wxString name, wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration, wxString gain_filename, wxString dark_reference, double output_binning_factor, int correct_mag_distortion, float mag_distortion_angle, float mag_distortion_major_scale, float mag_distortion_minor_scale, int protein_is_white, int eer_super_res_factor, int eer_frames_per_image);
	void EndMovieAssetInsert();

	void BeginMovieAssetMetadataInsert();
	void AddNextMovieAssetMetadata(MovieMetadataAsset asset);
	void EndMovieAssetMetadataInsert();

	void UpdateNumberOfFramesForAMovieAsset(int movie_asset_id, int new_number_of_frames);

	void BeginImageAssetInsert();
	void AddNextImageAsset(int image_asset_id,  wxString name, wxString filename, int position_in_stack, int parent_movie_id, int alignment_id, int ctf_estimation_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration, int protein_is_white);
	void EndImageAssetInsert() {EndBatchInsert();};

	void BeginVolumeAssetInsert();
	void AddNextVolumeAsset(int image_asset_id,  wxString name, wxString filename, int reconstruction_job_id, double pixel_size, int x_size, int y_size, int z_size, wxString half_map_1_filename, wxString half_map_2_filename);
	void EndVolumeAssetInsert() {EndBatchInsert();};

#ifdef EXPERIMENTAL
	void BeginAtomicCoordinatesAssetInsert();
	void AddNextAtomicCoordinatesAsset(int image_asset_id,  wxString name, wxString filename, int simulation_3d_job_id, double pixel_size, int x_size, int y_size, int z_size);
	void EndAtomicCoordinatesAssetInsert() {EndBatchInsert();};
#endif
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

	bool CreateProcessLockTable() {return CreateTable("PROCESS_LOCK", "plt", "NUMBER", "ACTIVE_PROCESS", "ACTIVE_HOST");};

	bool CreateParticlePickingListTable() {return CreateTable("PARTICLE_PICKING_LIST", "piiiirrrriiiii", "PICKING_ID", "DATETIME_OF_RUN", "PICKING_JOB_ID", "PARENT_IMAGE_ASSET_ID", "PICKING_ALGORITHM","CHARACTERISTIC_RADIUS", "MAXIMUM_RADIUS","THRESHOLD_PEAK_HEIGHT","HIGHEST_RESOLUTION_USED_IN_PICKING","MIN_DIST_FROM_EDGES","AVOID_HIGH_VARIANCE","AVOID_HIGH_LOW_MEAN","NUM_BACKGROUND_BOXES","MANUAL_EDIT");};
	bool CreateParticlePositionAssetTable() {return CreateTable("PARTICLE_POSITION_ASSETS", "piiirrrirrr", "PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "PICKING_ID", "PICK_JOB_ID", "X_POSITION", "Y_POSITION","PEAK_HEIGHT","TEMPLATE_ASSET_ID","TEMPLATE_PSI","TEMPLATE_THETA","TEMPLATE_PHI");};
	bool CreateParticlePositionGroupListTable() {return  CreateTable("PARTICLE_POSITION_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );};
	bool CreateParticlePickingResultsTable(const int &picking_job_id) {return CreateTable(wxString::Format("PARTICLE_PICKING_RESULTS_%i",picking_job_id),"piirrrirrr","POSITION_ID","PICKING_ID","PARENT_IMAGE_ASSET_ID","X_POSITION", "Y_POSITION","PEAK_HEIGHT","TEMPLATE_ASSET_ID","TEMPLATE_PSI","TEMPLATE_THETA","TEMPLATE_PHI");};

	bool CreateImageAssetTable() {return CreateTable("IMAGE_ASSETS", "pttiiiiiirrri", "IMAGE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "ALIGNMENT_ID", "CTF_ESTIMATION_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION","PROTEIN_IS_WHITE");};
	bool CreateMovieAssetTable() {return CreateTable("MOVIE_ASSETS", "pttiiiirrrrttrirrriii", "MOVIE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION", "GAIN_FILENAME", "DARK_FILENAME", "OUTPUT_BINNING_FACTOR", "CORRECT_MAG_DISTORTION", "MAG_DISTORTION_ANGLE", "MAG_DISTORTION_MAJOR_SCALE", "MAG_DISTORTION_MINOR_SCALE", "PROTEIN_IS_WHITE", "EER_SUPER_RES_FACTOR", "EER_FRAMES_PER_IMAGE");};

	bool CreateVolumeAssetTable() {return CreateTable("VOLUME_ASSETS", "pttiriiitt", "VOLUME_ASSET_ID", "NAME", "FILENAME", "RECONSTRUCTION_JOB_ID", "PIXEL_SIZE", "X_SIZE", "Y_SIZE", "Z_SIZE", "HALF_MAP1_FILENAME", "HALF_MAP2_FILENAME");};
	bool CreateVolumeGroupListTable() {return  CreateTable("VOLUME_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );};

#ifdef EXPERIMENTAL
  bool CreateAtomicCoordinatesAssetTable() {return CreateTable("ATOMICCOORDINATES_ASSETS", "pttiriii", "ATOMICCOORDINATES_ASSET_ID", "NAME", "FILENAME", "SIMULATION_3D_JOB_ID", "PIXEL_SIZE", "X_SIZE", "Y_SIZE", "Z_SIZE");};
	bool CreateAtomicCoordinatesGroupListTable() {return  CreateTable("ATOMICCOORDINATES_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );};
  // TODO: I'm not sure if the following sections that are related to refinement will also be needed for AtomicCoordinates objects, though I may be given we do refinement....just not refinement packages.
#endif

	bool CreateRefinementPackageAssetTable() {return CreateTable("REFINEMENT_PACKAGE_ASSETS", "pttirtrriiii", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "STACK_FILENAME", "STACK_BOX_SIZE", "OUTPUT_PIXEL_SIZE", "SYMMETRY", "MOLECULAR_WEIGHT", "PARTICLE_SIZE","NUMBER_OF_CLASSES", "NUMBER_OF_REFINEMENTS", "LAST_REFINEMENT_ID", "STACK_HAS_WHITE_PROTEIN");};
	bool CreateRefinementPackageContainedParticlesTable(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", refinement_package_asset_id), "piirrrrrrrrrri", "ORIGINAL_PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "POSITION_IN_STACK", "X_POSITION", "Y_POSITION", "PIXEL_SIZE", "DEFOCUS_1", "DEFOCUS_2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "SPHERICAL_ABERRATION", "MICROSCOPE_VOLTAGE", "AMPLITUDE_CONTRAST","ASSIGNED_SUBSET");};
	bool CreateRefinementPackageCurrent3DReferencesTable(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", refinement_package_asset_id), "pi", "CLASS_NUMBER", "VOLUME_ASSET_ID");};
	bool CreateRefinementPackageRefinementsList(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", refinement_package_asset_id), "pl", "REFINEMENT_NUMBER", "REFINEMENT_ID");};
	bool CreateRefinementPackageClassificationsList(const long refinement_package_asset_id) {return CreateTable(wxString::Format("REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", refinement_package_asset_id), "pl", "CLASSIFICATION_NUMBER", "CLASSIFICATION_ID");};

	bool CreateRefinementListTable() {return CreateTable("REFINEMENT_LIST", "Pltillllirr", "REFINEMENT_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "RESOLUTION_STATISTICS_ARE_GENERATED", "DATETIME_OF_RUN", "STARTING_REFINEMENT_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "RESOLUTION_STATISTICS_BOX_SIZE", "RESOLUTION_STATISTICS_PIXEL_SIZE", "PERCENT_USED");};

	bool CreateRefinementDetailsTable(const long refinement_id) {return CreateTable(wxString::Format("REFINEMENT_DETAILS_%li", refinement_id), "plrrrrrrirrrrirrrrirrrrlliiilrrir",		"CLASS_NUMBER", "REFERENCE_VOLUME_ASSET_ID", "LOW_RESOLUTION_LIMIT",
																																														"HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "SIGNED_CC_RESOLUTION_LIMIT",
																																														"GLOBAL_RESOLUTION_LIMIT", "GLOBAL_MASK_RADIUS", "NUMBER_RESULTS_TO_REFINE",
																																														"ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y",
																																														"CLASSIFICATION_RESOLUTION_LIMIT", "SHOULD_FOCUS_CLASSIFY", "SPHERE_X_COORD",
																																														"SPHERE_Y_COORD", "SPHERE_Z_COORD", "SPHERE_RADIUS", "SHOULD_REFINE_CTF", "DEFOCUS_SEARCH_RANGE",
																																														"DEFOCUS_SEARCH_STEP", "AVERAGE_OCCUPANCY", "ESTIMATED_RESOLUTION", "RECONSTRUCTED_VOLUME_ASSET_ID",
																																														"RECONSTRUCTION_ID", "SHOULD_AUTOMASK", "SHOULD_REFINE_INPUT_PARAMS", "SHOULD_USE_SUPPLIED_MASK",
																																														"MASK_ASSET_ID", "MASK_EDGE_WIDTH", "OUTSIDE_MASK_WEIGHT", "SHOULD_LOWPASS_OUTSIDE_MASK", "MASK_FILTER_RESOLUTION");};
	bool CreateMovieMetadataTable() { return CreateTable("MOVIE_ASSETS_METADATA", "Plttrrrrrrrl", "MOVIE_ASSET_METADATA_ID", "MOVIE_ASSET_ID", "METADATA_SOURCE", "CONTENT_JSON", "TILT_ANGLE", "STAGE_POSITION_X", "STAGE_POSITION_Y", "STAGE_POSITION_Z", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y", "EXPOSURE_DOSE", "ACQUISITION_TIME"); }
	bool CreateTemplateMatchingResultsTable() { return CreateTable("TEMPLATE_MATCH_LIST", "Ptllilllitrrrrrrrrrrrrrrrrrrrrrrittttttttttt", "TEMPLATE_MATCH_ID", "JOB_NAME", "DATETIME_OF_RUN", "TEMPLATE_MATCH_JOB_ID", "JOB_TYPE_CODE", "INPUT_TEMPLATE_MATCH_ID", "IMAGE_ASSET_ID", "REFERENCE_VOLUME_ASSET_ID", "IS_ACTIVE", "USED_SYMMETRY", "USED_PIXEL_SIZE", "USED_VOLTAGE", "USED_SPHERICAL_ABERRATION", "USED_AMPLITUDE_CONTRAST", "USED_DEFOCUS1", "USED_DEFOCUS2", "USED_DEFOCUS_ANGLE", "USED_PHASE_SHIFT", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "OUT_OF_PLANE_ANGULAR_STEP", "IN_PLANE_ANGULAR_STEP", "DEFOCUS_SEARCH_RANGE", "DEFOCUS_STEP", "PIXEL_SIZE_SEARCH_RANGE", "PIXEL_SIZE_STEP", "REFINEMENT_THRESHOLD", "USED_THRESHOLD", "REF_BOX_SIZE_IN_ANGSTROMS", "MASK_RADIUS", "MIN_PEAK_RADIUS", "XY_CHANGE_THRESHOLD", "EXCLUDE_ABOVE_XY_THRESHOLD", "MIP_OUTPUT_FILE", "SCALED_MIP_OUTPUT_FILE", "AVG_OUTPUT_FILE", "STD_OUTPUT_FILE", "PSI_OUTPUT_FILE", "THETA_OUTPUT_FILE", "PHI_OUTPUT_FILE", "DEFOCUS_OUTPUT_FILE", "PIXEL_SIZE_OUTPUT_FILE", "HISTOGRAM_OUTPUT_FILE", "PROJECTION_RESULT_OUTPUT_FILE");}
	bool CreateTemplateMatchPeakListTable(const long template_match_job_id) {return CreateTable(wxString::Format("TEMPLATE_MATCH_PEAK_LIST_%li", template_match_job_id), "prrrrrrrr", "PEAK_NUMBER", "X_POSITION", "Y_POSITION", "PSI", "THETA", "PHI", "DEFOCUS", "PIXEL_SIZE", "PEAK_HEIGHT");}
	bool CreateTemplateMatchPeakChangeListTable(const long template_match_job_id) {return CreateTable(wxString::Format("TEMPLATE_MATCH_PEAK_CHANGE_LIST_%li", template_match_job_id), "prrrrrrrrii", "PEAK_NUMBER", "X_POSITION", "Y_POSITION", "PSI", "THETA", "PHI", "DEFOCUS", "PIXEL_SIZE", "PEAK_HEIGHT", "ORIGINAL_PEAK_NUMBER", "NEW_PEAK_NUMBER");}

	bool CreateRefinementResultTable(const long refinement_id, const int class_number) {return CreateTable(wxString::Format("REFINEMENT_RESULT_%li_%i", refinement_id, class_number), "Prrrrrrrrrrrrrirrrrrrrri", "POSITION_IN_STACK", "PSI", "THETA", "PHI", "XSHIFT", "YSHIFT", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "OCCUPANCY", "LOGP", "SIGMA", "SCORE", "IMAGE_IS_ACTIVE", "PIXEL_SIZE", "MICROSCOPE_VOLTAGE", "MICROSCOPE_CS", "AMPLITUDE_CONTRAST", "BEAM_TILT_X", "BEAM_TILT_Y", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y", "ASSIGNED_SUBSET");};
	bool CreateRefinementResolutionStatisticsTable(const long refinement_id, int class_number) {return CreateTable(wxString::Format("REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_id, class_number), "prrrrr", "SHELL", "RESOLUTION", "FSC", "PART_FSC", "PART_SSNR", "REC_SSNR");};
	bool CreateRefinementAngularDistributionTable(const long refinement_id, const int class_number) { return CreateTable(wxString::Format("REFINEMENT_ANGULAR_DISTRIBUTION_%li_%i", refinement_id, class_number), "pr", "BIN_NUMBER", "NUMBER_IN_BIN");}

	bool CreateClassificationListTable() {return CreateTable("CLASSIFICATION_LIST", "Plttilllirrrrrrriir", "CLASSIFICATION_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "CLASS_AVERAGE_FILE", "REFINEMENT_WAS_IMPORTED_OR_GENERATED", "DATETIME_OF_RUN", "STARTING_CLASSIFICATION_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "SMOOTHING_FACTOR", "EXCLUDE_BLANK_EDGES", "AUTO_PERCENT_USED", "PERCENT_USED" );	};
	bool CreateClassificationResultTable(const long classification_id) {return CreateTable(wxString::Format("CLASSIFICATION_RESULT_%li", classification_id), "Prrrirrrrrrrrrrrrrr", "POSITION_IN_STACK", "PSI", "XSHIFT", "YSHIFT", "BEST_CLASS", "SIGMA", "LOGP", "PIXEL_SIZE", "VOLTAGE", "CS", "AMPLITUDE_CONTRAST", "DEFOCUS_1", "DEFOCUS_2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "BEAM_TILT_X", "BEAM_TILT_Y", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y" );};

	bool CreateClassificationSelectionListTable() {return CreateTable("CLASSIFICATION_SELECTION_LIST", "Ptlllii", "SELECTION_ID", "SELECTION_NAME", "CREATION_DATE", "REFINEMENT_PACKAGE_ID", "CLASSIFICATION_ID", "NUMBER_OF_CLASSES", "NUMBER_OF_SELECTIONS");};
	bool CreateClassificationSelectionTable(const long selection_id) {return CreateTable(wxString::Format("CLASSIFICATION_SELECTION_%li", selection_id), "pl", "SELECTION_NUMBER", "CLASS_AVERAGE_NUMBER");};

	bool CreateMovieImportDefaultsTable() {return CreateTable("MOVIE_IMPORT_DEFAULTS", "prrrrititirirrriii", "NUMBER", "VOLTAGE", "SPHERICAL_ABERRATION", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "MOVIES_ARE_GAIN_CORRECTED", "GAIN_REFERENCE_FILENAME", "MOVIES_ARE_DARK_CORRECTED", "DARK_REFERENCE_FILENAME", "RESAMPLE_MOVIES", "DESIRED_PIXEL_SIZE", "CORRECT_MAG_DISTORTION", "MAG_DISTORTION_ANGLE", "MAG_DISTORTION_MAJOR_SCALE", "MAG_DISTORTION_MINOR_SCALE", "PROTEIN_IS_WHITE", "EER_SUPER_RES_FACTOR", "EER_FRAMES_PER_IMAGE" );};
	bool CreateImageImportDefaultsTable() {return CreateTable("IMAGE_IMPORT_DEFAULTS", "prrri", "NUMBER", "VOLTAGE", "SPHERICAL_ABERRATION", "PIXEL_SIZE", "PROTEIN_IS_WHITE");};

	bool CreateStartupListTable() { return CreateTable("STARTUP_LIST", "Pltiirriirrrir", "STARTUP_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "NUMBER_OF_STARTS", "NUMBER_OF_CYCLES", "INITIAL_RES_LIMIT", "FINAL_RES_LIMIT", "AUTO_MASK", "AUTO_PERCENT_USED", "INITIAL_PERCENT_USED", "FINAL_PERCENT_USED", "MASK_RADIUS", "APPLY_LIKELIHOOD_BLURRING", "SMOOTHING_FACTOR");};
	bool CreateStartupResultTable(const long startup_id) {return CreateTable(wxString::Format("STARTUP_RESULT_%li", startup_id), "pl", "CLASS_NUMBER", "VOLUME_ASSET_ID");};

	bool CreateReconstructionListTable() { return CreateTable("RECONSTRUCTION_LIST", "Plltrrrriiiiril", "RECONSTRUCTION_ID", "REFINEMENT_PACKAGE_ID", "REFINEMENT_ID", "NAME", "INNER_MASK_RADIUS", "OUTER_MASK_RADIUS", "RESOLUTION_LIMIT", "SCORE_WEIGHT_CONVERSION", "SHOULD_ADJUST_SCORES", "SHOULD_CROP_IMAGES", "SHOULD_SAVE_HALF_MAPS", "SHOULD_LIKELIHOOD_BLUR", "SMOOTHING_FACTOR", "CLASS_NUMBER", "VOLUME_ASSET_ID");}



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

#ifdef EXPERIMENTAL
	void BeginAllAtomicCoordinatesAssetsSelect();
	AtomicCoordinatesAsset GetNextAtomicCoordinatesAsset();
	void EndAllAtomicCoordinatesAssetsSelect() {EndBatchSelect();};
#endif

	void BeginAllVolumeGroupsSelect();
	AssetGroup GetNextVolumeGroup();
	void EndAllVolumeGroupsSelect(){EndBatchSelect();};

	void BeginAllRunProfilesSelect();
	RunProfile GetNextRunProfile();
	void EndAllRunProfilesSelect(){EndBatchSelect();};

	void BeginAllRefinementPackagesSelect();
	RefinementPackage* GetNextRefinementPackage();
	void EndAllRefinementPackagesSelect() {EndBatchSelect();};

	void AddStartupJob(long startup_job_id, long refinement_package_asset_id, wxString name, int number_of_starts, int number_of_cycles, float initial_res_limit, float final_res_limit, bool auto_mask, bool auto_percent_used,  float initial_percent_used, float final_percent_used, float mask_radius, bool apply_blurring, float smoothing_factor, wxArrayLong result_volume_ids);
	void AddReconstructionJob(long reconstruction_id, long refinement_package_asset_id, long refinement_id, wxString name, float inner_mask_radius, float outer_mask_radius, float resolution_limit, float score_weight_conversion, bool should_adjust_score, bool should_crop_images, bool should_save_half_maps, bool should_likelihood_blur, float smoothing_factor, int class_number, long volume_asset_id);
	void GetReconstructionJob(long wanted_reconstruction_id, long &refinement_package_asset_id, long &refinement_id, wxString &name, float &inner_mask_radius, float &outer_mask_radius, float &resolution_limit, float &score_weight_conversion, bool &should_adjust_score, bool &should_crop_images, bool &should_save_half_maps, bool &should_likelihood_blur, float &smoothing_factor, int &class_number, long &volume_asset_id);

	// Convenience CTF parameter function
	void GetCTFParameters( const int &ctf_estimation_id, double &acceleration_voltage, double &spherical_aberration, double &amplitude_constrast, double &defocus_1, double &defocus_2, double &defocus_angle, double &additional_phase_shift, double &iciness );

	void AddCTFIcinessColumnIfNecessary();

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

	void AddTemplateMatchingResult(long wanted_template_match_id, TemplateMatchJobResults &job_details);

	TemplateMatchJobResults GetTemplateMatchingResultByID(long wanted_template_match_id);

	void AddRefinementAngularDistribution(AngularDistributionHistogram &histogram_to_add, long refinement_id, int class_number);
	void CopyRefinementAngularDistributions(long refinement_id_to_copy, long refinement_id_to_copy_to, int wanted_class_number);
	void  GetRefinementAngularDistributionHistogramData(long wanted_refinement_id, int wanted_class_number, AngularDistributionHistogram &histogram_to_fill);


	Refinement *GetRefinementByID(long wanted_refinement_id, bool include_particle_info = true);

	void AddClassification(Classification *classification_to_add);
	Classification *GetClassificationByID(long wanted_classification_id);

	void AddClassificationSelection(ClassificationSelection *classification_selection_to_add);
	//ClassificationSelection *GetClassificationSelectionByID(long wanted_selection_id);

};


class BeginCommitLocker  // just call begin in the contructor, and commit in the destructor, if it hasn't already been called.
{
	Database *active_database;
	bool already_sent_commit;

public:
	BeginCommitLocker(Database *wanted_database);
	~BeginCommitLocker();
	void Commit();
};
