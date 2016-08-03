#include "core_headers.h"

Database::Database()
{
	last_return_code = -1;
	is_open = false;
	sqlite_database = NULL;

	in_batch_insert = false;
	in_batch_select = false;
	batch_statement = NULL;

}

Database::~Database()
{
	Close();
}

bool Database::ExecuteSQL(const char *command)
{
	char *error_message = NULL;
	int return_code = sqlite3_exec(sqlite_database, command, NULL, 0, &error_message);

	if( return_code != SQLITE_OK )
	{
	   	MyPrintWithDetails("SQL Error: %s\nTrying to execute the following command :-\n\n%s\n", error_message, command);
	    sqlite3_free(error_message);
	    return false;
	}

	return true;
}

int Database::ReturnSingleIntFromSelectCommand(wxString select_command)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	int return_code;
	sqlite3_stmt *current_statement;
	int value;

	return_code = sqlite3_prepare_v2(sqlite_database, select_command, select_command.Length() + 1, &current_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(current_statement);

	if (return_code != SQLITE_DONE && return_code != SQLITE_ROW)
	{
		MyPrintWithDetails("SQL Return Code: %i\n", return_code);
	}

	value = sqlite3_column_int(current_statement, 0);

	sqlite3_finalize(current_statement);

	return value;
}

double Database::ReturnSingleDoubleFromSelectCommand(wxString select_command)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	int return_code;
	sqlite3_stmt *current_statement;
	double value;

	return_code = sqlite3_prepare_v2(sqlite_database, select_command, select_command.Length() + 1, &current_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(current_statement);

	if (return_code != SQLITE_DONE && return_code != SQLITE_ROW)
	{
		MyPrintWithDetails("SQL Return Code: %i\n", return_code);
	}

	value = sqlite3_column_double(current_statement, 0);

	sqlite3_finalize(current_statement);

	return value;
}

int Database::ReturnHighestAlignmentID()
{
	return ReturnSingleIntFromSelectCommand("SELECT MAX(ALIGNMENT_ID) FROM MOVIE_ALIGNMENT_LIST");
}

int Database::ReturnHighestFindCTFID()
{
	return ReturnSingleIntFromSelectCommand("SELECT MAX(CTF_ESTIMATION_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

int Database::ReturnHighestPickingID()
{
	// We return 0 if the particle_picking_list is empty
	return ReturnSingleIntFromSelectCommand("SELECT COALESCE(MAX(PICKING_ID),0) FROM PARTICLE_PICKING_LIST");
}

int Database::ReturnHighestParticlePositionID()
{
	// Note: we can't just look for the maximum position_id in the latest picking results table, since the user is free to add new particle positions
	// to old results tables
	int number_of_picking_jobs = ReturnNumberOfPickingJobs();
	int max_position_id = -1;
	int current_max_position_id = -1;
	int number_of_non_empty_tables = 0;
	wxString sql_query = "select max(";
	for (int counter = 1; counter <= number_of_picking_jobs; counter++)
	{
		if (ReturnSingleIntFromSelectCommand(wxString::Format("select count(*) from particle_picking_results_%i",counter)) > 0)
		{
			if (number_of_non_empty_tables > 0)
			{
				sql_query += ",";
			}
			number_of_non_empty_tables ++;
			sql_query += wxString::Format("(select max(position_id) from particle_picking_results_%i)",counter);
		}
	}
	sql_query += ")";
	return ReturnSingleIntFromSelectCommand(sql_query);
}


int Database::ReturnNumberOfPreviousMovieAlignmentsByAssetID(int wanted_asset_id)
{
	return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM MOVIE_ALIGNMENT_LIST WHERE MOVIE_ASSET_ID = %i", wanted_asset_id));
}

int Database::ReturnNumberOfPreviousCTFEstimationsByAssetID(int wanted_asset_id)
{
	return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID = %i", wanted_asset_id));
}


int Database::ReturnNumberOfPreviousParticlePicksByAssetID(int wanted_asset_id)
{
	return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM PARTICLE_PICKING_LIST WHERE PARENT_IMAGE_ASSET_ID = %i", wanted_asset_id));
}

int Database::ReturnHighestAlignmentJobID()
{
	return ReturnSingleIntFromSelectCommand("SELECT MAX(ALIGNMENT_JOB_ID) FROM MOVIE_ALIGNMENT_LIST");
}

int Database::ReturnHighestFindCTFJobID()
{
	return ReturnSingleIntFromSelectCommand("SELECT MAX(CTF_ESTIMATION_JOB_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

int Database::ReturnHighestPickingJobID()
{
	return ReturnSingleIntFromSelectCommand("SELECT MAX(PICKING_JOB_ID) FROM PARTICLE_PICKING_LIST");
}

int Database::ReturnNumberOfAlignmentJobs()
{
	return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT ALIGNMENT_JOB_ID) FROM MOVIE_ALIGNMENT_LIST");
}

int Database::ReturnNumberOfCTFEstimationJobs()
{
	return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT CTF_ESTIMATION_JOB_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

int Database::ReturnNumberOfPickingJobs()
{
	return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT PICKING_JOB_ID) FROM PARTICLE_PICKING_LIST");
}

void Database::GetUniqueAlignmentIDs(int *alignment_job_ids, int number_of_alignmnet_jobs)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	bool more_data;

	more_data = BeginBatchSelect("SELECT DISTINCT ALIGNMENT_JOB_ID FROM MOVIE_ALIGNMENT_LIST") == true;

	for (int counter = 0; counter < number_of_alignmnet_jobs; counter++)
	{
		if (more_data == false)
		{
			MyPrintWithDetails("Unexpected end of select command");
			abort();
		}

		more_data = GetFromBatchSelect("i", &alignment_job_ids[counter]);

	}

	EndBatchSelect();
}


void Database::GetUniquePickingJobIDs(int *picking_job_ids, int number_of_picking_jobs)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	bool more_data;

	more_data = BeginBatchSelect("SELECT DISTINCT PICKING_JOB_ID FROM PARTICLE_PICKING_LIST") == true;

	for (int counter = 0; counter < number_of_picking_jobs; counter++)
	{
		if (more_data == false)
		{
			MyPrintWithDetails("Unexpected end of select command");
			abort();
		}

		more_data = GetFromBatchSelect("i", &picking_job_ids[counter]);

	}

	EndBatchSelect();
}


void Database::GetUniqueCTFEstimationIDs(int *ctf_estimation_job_ids, int number_of_ctf_estimation_jobs)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	bool more_data;

	more_data = BeginBatchSelect("SELECT DISTINCT CTF_ESTIMATION_JOB_ID FROM ESTIMATED_CTF_PARAMETERS") == true;

	for (int counter = 0; counter < number_of_ctf_estimation_jobs; counter++)
	{
		if (more_data == false)
		{
			MyPrintWithDetails("Unexpected end of select command");
			abort();
		}

		more_data = GetFromBatchSelect("i", &ctf_estimation_job_ids[counter]);

	}

	EndBatchSelect();
}

int Database::ReturnNumberOfImageAssetsWithCTFEstimates()
{
	return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT IMAGE_ASSET_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

void Database::GetUniqueIDsOfImagesWithCTFEstimations(int *image_ids, int &number_of_image_ids)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	bool more_data;

	more_data = BeginBatchSelect("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");

	for (int counter = 0; counter < number_of_image_ids; counter++)
	{
		if (more_data == false)
		{
			MyPrintWithDetails("Unexpected end of select command");
			abort();
		}

		more_data = GetFromBatchSelect("i", &image_ids[counter]);

	}

	EndBatchSelect();
}

void Database::GetCTFParameters( const int &ctf_estimation_id, double &acceleration_voltage, double &spherical_aberration, double &amplitude_constrast, double &defocus_1, double &defocus_2, double &defocus_angle, double &additional_phase_shift )
{
	MyDebugAssertTrue(is_open,"Database not open");


	bool more_data;

	more_data = BeginBatchSelect(wxString::Format("SELECT VOLTAGE, SPHERICAL_ABERRATION, AMPLITUDE_CONTRAST, DEFOCUS1, DEFOCUS2, DEFOCUS_ANGLE, ADDITIONAL_PHASE_SHIFT FROM ESTIMATED_CTF_PARAMETERS WHERE CTF_ESTIMATION_ID=%i", ctf_estimation_id));

	if (more_data)
	{
		GetFromBatchSelect("rrrrrrr",&acceleration_voltage,&spherical_aberration,&amplitude_constrast,&defocus_1,&defocus_2,&defocus_angle,&additional_phase_shift);
	}
	else
	{
		MyPrintWithDetails("Unexpected end of select command\n");
		abort();
	}

	EndBatchSelect();

}


bool Database::CreateNewDatabase(wxFileName wanted_database_file)
{
	int return_code;

	// is project already open?

	if (is_open == true)
	{
		MyPrintWithDetails("Attempting to create a new database, but there is already an open project");
		return false;
	}

	// does the database file exist?

	if (wanted_database_file.Exists())
	{
		MyPrintWithDetails("Attempting to create a new database, but the file already exists");
		return false;
	}

	// make the path absolute..

	wanted_database_file.MakeAbsolute();

	return_code = sqlite3_open_v2(wanted_database_file.GetFullPath().ToUTF8().data(), &sqlite_database, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);

    if( return_code )
    {
    	MyPrintWithDetails("Can't open database: %s\n%s\n", database_file.GetFullPath().ToUTF8().data(), sqlite3_errmsg(sqlite_database));
        return false;
    }

	// update project class details..

	database_file = wanted_database_file;
	is_open = true;

    return true;
}

bool Database::Open(wxFileName file_to_open)
{
	int return_code;

	// is project already open?

	if (is_open == true)
	{
		MyPrintWithDetails("Attempting to open a database, but there is already an open project");
		return false;
	}

	// does the database file exist?

	if (file_to_open.Exists() == false)
	{
		MyPrintWithDetails("Attempting to open a new database, but the file does not exist");
		return false;
	}


	return_code = sqlite3_open_v2(file_to_open.GetFullPath().ToUTF8().data(), &sqlite_database, SQLITE_OPEN_READWRITE, NULL);

	if( return_code )
	{
		MyPrintWithDetails("Can't open database: %s\n%s\n", database_file.GetFullPath().ToUTF8().data(), sqlite3_errmsg(sqlite_database));
	    return false;
	}

	database_file = file_to_open;
	is_open = true;

	return true;


}

bool Database::DeleteTable(const char *table_name)
{
	wxString sql_command = "DROP TABLE IF EXISTS ";
	sql_command += table_name;

	return ExecuteSQL(sql_command.ToUTF8().data());
}

bool Database::CreateTable(const char *table_name, const char *column_format, ...)
{
	int return_code;
	char *error_message = NULL;

	wxString sql_command;

	int number_of_columns = strlen(column_format);

	int current_column = 0;

	sql_command = "CREATE TABLE IF NOT EXISTS ";
	sql_command += table_name;
	sql_command += "(";

	va_list args;
	va_start(args, column_format);

	while (*column_format != '\0')
	{
		current_column++;

		if (*column_format == 't') // text
		{
			sql_command += va_arg(args, const char *);
			sql_command += " TEXT";
		}
		else
		if (*column_format == 'r') // real
		{
			sql_command += va_arg(args, const char *);
			sql_command += " REAL";
		}
		else
		if (*column_format == 'i') // integer
		{
			sql_command += va_arg(args, const char *);
			sql_command += " INTEGER";
		}
		else
		if (*column_format == 'p') // integer
		{
			sql_command += va_arg(args, const char *);
			sql_command += " INTEGER PRIMARY KEY";
		}
		else
		{
			MyPrintWithDetails("Error: Unknown format character!\n");
		}

		if (current_column < number_of_columns) sql_command += ", ";
		else sql_command += " );";

		 ++column_format;
	}

	va_end(args);

	return_code = sqlite3_exec(sqlite_database, sql_command.ToUTF8().data(), NULL, 0, &error_message);

    if( return_code != SQLITE_OK )
    {
    	MyPrintWithDetails("SQL Error: %s\n%s", error_message, sql_command);
        sqlite3_free(error_message);
        abort();
    }

    return true;

}

bool Database::CreateAllTables()
{
	bool success;

	success = CreateTable("MASTER_SETTINGS", "pttiri", "NUMBER", "PROJECT_DIRECTORY", "PROJECT_NAME", "CURRENT_VERSION", "TOTAL_CPU_HOURS", "TOTAL_JOBS_RUN");
	success = CreateTable("RUNNING_JOBS", "pti", "JOB_NUMBER", "JOB_CODE", "MANAGER_IP_ADDRESS");
	success = CreateTable("RUN_PROFILES", "ptttti", "RUN_PROFILE_ID", "PROFILE_NAME", "MANAGER_RUN_COMMAND", "GUI_ADDRESS", "CONTROLLER_ADDRESS", "COMMANDS_ID");
	success = CreateTable("MOVIE_ASSETS", "ptiiiirrrr", "MOVIE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION");
	success = CreateTable("MOVIE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );
	success = CreateTable("MOVIE_ALIGNMENT_LIST", "piiitrrrrrriiriiiii", "ALIGNMENT_ID", "DATETIME_OF_RUN", "ALIGNMENT_JOB_ID", "MOVIE_ASSET_ID", "OUTPUT_FILE", "VOLTAGE", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "PRE_EXPOSURE_AMOUNT", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER", "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS", "HORIZONTAL_MASK", "VERTICAL_MASK" );
	success = CreateTable("IMAGE_ASSETS", "ptiiiiiirrr", "IMAGE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "ALIGNMENT_ID", "CTF_ESTIMATION_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION");
	success = CreateTable("IMAGE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );
	success = CreateParticlePickingListTable();
	success = CreateParticlePositionAssetTable();
	success = CreateParticlePositionGroupListTable();
	success = CreateVolumeAssetTable();
	success = CreateVolumeGroupListTable();


	success = CreateTable("ESTIMATED_CTF_PARAMETERS", "piiiirrrrirrrrririrrrrrrrrrrti", "CTF_ESTIMATION_ID", "CTF_ESTIMATION_JOB_ID", "DATETIME_OF_RUN", "IMAGE_ASSET_ID", "ESTIMATED_ON_MOVIE_FRAMES", "VOLTAGE", "SPHERICAL_ABERRATION", "PIXEL_SIZE", "AMPLITUDE_CONTRAST", "BOX_SIZE", "MIN_RESOLUTION", "MAX_RESOLUTION", "MIN_DEFOCUS", "MAX_DEFOCUS", "DEFOCUS_STEP", "RESTRAIN_ASTIGMATISM", "TOLERATED_ASTIGMATISM", "FIND_ADDITIONAL_PHASE_SHIFT", "MIN_PHASE_SHIFT", "MAX_PHASE_SHIFT", "PHASE_SHIFT_STEP", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE", "ADDITIONAL_PHASE_SHIFT", "SCORE", "DETECTED_RING_RESOLUTION", "DETECTED_ALIAS_RESOLUTION", "OUTPUT_DIAGNOSTIC_FILE","NUMBER_OF_FRAMES_AVERAGED");
	return success;
}

bool Database::InsertOrReplace(const char *table_name, const char *column_format, ...)
{
	int number_of_columns = strlen(column_format);

	int current_column = 0;
	int return_code;
	char *error_message = NULL;

	wxString sql_command;

	sql_command = "INSERT OR REPLACE INTO  ";
	sql_command += table_name;
	sql_command += "(";

	va_list args;
	va_start(args, column_format);

	for (current_column = 1; current_column <= number_of_columns; current_column++)
	{
		sql_command += va_arg(args, const char *);

		if (current_column == number_of_columns) sql_command += " ) ";
		else sql_command += ", ";
	}

	sql_command += "VALUES (";
	current_column = 0;

	while (*column_format != '\0')
	{

		current_column++;

		if (*column_format == 't') // text
		{
			sql_command += "'";
			sql_command += va_arg(args, const char *);
			sql_command += "'";
		}
		else
		if (*column_format == 'r') // real
		{
			sql_command += wxString::Format("%f", va_arg(args, double));
		}
		else
		if (*column_format == 'i' || *column_format == 'p') // integer
		{
			sql_command += wxString::Format("%i",  va_arg(args, int));
		}
		else
		{
			MyPrintWithDetails("Error: Unknown format character!\n");
			abort();
		}

		if (current_column < number_of_columns) sql_command += ", ";
		else sql_command += " );";

		 ++column_format;
	}

	va_end(args);

	return_code = sqlite3_exec(sqlite_database, sql_command.ToUTF8().data(), NULL, 0, &error_message);

    if( return_code != SQLITE_OK )
    {
    	MyPrintWithDetails("SQL Error: %s\n\nFor :-\n%s", error_message, sql_command);
        sqlite3_free(error_message);
        return false;
    }

    return true;


}

bool Database::GetMasterSettings(wxFileName &project_directory, wxString &project_name, int &imported_integer_version, double &total_cpu_hours, int &total_jobs_run)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	sqlite3_stmt *sqlite_statement;
	int return_code;
	wxString sql_command = "select * from MASTER_SETTINGS;";

	return_code = sqlite3_prepare_v2(sqlite_database, sql_command.ToUTF8().data(), sql_command.Length() + 1, &sqlite_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(sqlite_statement);

	if (return_code != SQLITE_ROW)
	{
		MyPrintWithDetails("SQL Return Code: %i\n", return_code);
		return false;
	}

	project_directory = sqlite3_column_text(sqlite_statement, 1);
	project_name = sqlite3_column_text(sqlite_statement, 2);
	imported_integer_version = sqlite3_column_int(sqlite_statement, 3);
	total_cpu_hours = sqlite3_column_double(sqlite_statement, 4);
	total_jobs_run = sqlite3_column_int(sqlite_statement, 5);

	sqlite3_finalize(sqlite_statement);
	return true;
}

void Database::Close()
{
	if (is_open == true)
	{
		int return_code = sqlite3_close(sqlite_database);
		MyDebugAssertTrue(return_code == SQLITE_OK, "SQL close error, return code : %i\n", return_code );
	}

	is_open = false;
	sqlite_database = NULL;

}

void Database::BeginBatchInsert(const char *table_name, int number_of_columns, ...)
{
	MyDebugAssertTrue(is_open == true, "database not open!");
	MyDebugAssertTrue(in_batch_insert == false, "Starting batch insert but already in batch insert mode");
	MyDebugAssertTrue(in_batch_select == false, "Starting batch insert but already in batch select mode");

	wxString sql_command;
	int counter;
	int return_code;
	char *error_message = NULL;

	in_batch_insert = true;

	va_list args;
	va_start(args, number_of_columns);

	sql_command = "BEGIN IMMEDIATE;";
	last_return_code = sqlite3_exec(sqlite_database, sql_command.ToUTF8().data(), NULL, 0, &error_message);


	if (last_return_code != SQLITE_OK)
	{
		MyPrintWithDetails("SQL Error : %s\n", error_message);
		sqlite3_free(error_message);
	}

	sql_command = "INSERT OR REPLACE INTO ";
	sql_command += table_name;
	sql_command += " (";

	for (counter = 1; counter <= number_of_columns; counter++)
	{
		sql_command += va_arg(args, const char *);

		if (counter < number_of_columns) sql_command += ",";
		else sql_command += ") ";
	}

	va_end(args);

	sql_command += "VALUES (";

	for (counter = 1; counter <= number_of_columns; counter++)
	{
		if (counter < number_of_columns) sql_command += "?,";
		else sql_command += "?); ";
	}

	return_code = sqlite3_prepare_v2(sqlite_database, sql_command.ToUTF8().data(), sql_command.Length(), &batch_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\nSQL command : %s", return_code, sql_command.ToUTF8().data() );
}

void Database::AddToBatchInsert(const char *column_format, ...)
{
	int argument_counter = 0;
	const char * text_pointer;
	int return_code;
	va_list args;
	va_start(args, column_format);

	while (*column_format != '\0')
	{
		argument_counter++;

		if (*column_format == 't') // text
		{
			text_pointer = va_arg(args,const char *);
			return_code = sqlite3_bind_text(batch_statement, argument_counter, text_pointer, strlen(text_pointer), SQLITE_STATIC);
			MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );
		}
		else
		if (*column_format == 'r') // real
		{
			return_code = sqlite3_bind_double(batch_statement, argument_counter, va_arg(args, double));
			MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );
		}
		else
		if (*column_format == 'i') // integer
		{
			return_code = sqlite3_bind_int(batch_statement, argument_counter, va_arg(args, int));
			MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );
		}
		else
		if (*column_format == 'l') // long
		{
			return_code = sqlite3_bind_int64(batch_statement, argument_counter, va_arg(args, long));
			MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

		}
		else
		{
			MyPrintWithDetails("Error: Unknown format character!\n");
		}

		 ++column_format;
	}

	va_end(args);

	return_code = sqlite3_step(batch_statement);
	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

    return_code = sqlite3_clear_bindings(batch_statement);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_reset(batch_statement);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

}

void Database::EndBatchInsert()
{
	int return_code;
	char *error_message = NULL;
	wxString sql_command;

	sql_command += "COMMIT;";

	return_code = sqlite3_exec(sqlite_database, sql_command.ToUTF8().data(), NULL, 0, &error_message);

    if( return_code != SQLITE_OK )
    {
    	MyPrintWithDetails("SQL Error: %s\n", error_message);
        sqlite3_free(error_message);
    }

    sqlite3_finalize(batch_statement);
    in_batch_insert = false;
}

void Database::BeginMovieAssetInsert()
{
	BeginBatchInsert("MOVIE_ASSETS", 10, "MOVIE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION");
}

void Database::AddNextMovieAsset(int movie_asset_id,  wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration)
{
	AddToBatchInsert("itiiiirrrr", movie_asset_id, filename.ToUTF8().data(), position_in_stack, x_size, y_size, number_of_frames, voltage, pixel_size, dose_per_frame, spherical_aberration);
}

/*
void Database::AddMovieAsset(MovieAsset *asset_to_add)
{
	AddGroupInsert("itiiiirrrr", asset_to_add->asset_id, filename.ToUTF8().data(), position_in_stack, x_size, y_size, number_of_frames, voltage, pixel_size, dose_per_frame, spherical_aberration);
}
*/

void Database::EndMovieAssetInsert()
{
	EndBatchInsert();
}


void Database::BeginImageAssetInsert()
{
	BeginBatchInsert("IMAGE_ASSETS", 11, "IMAGE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "ALIGNMENT_ID", "CTF_ESTIMATION_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION");
}

void Database::AddNextImageAsset(int image_asset_id,  wxString filename, int position_in_stack, int parent_movie_id, int alignment_id, int ctf_estimation_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration)
{
	AddToBatchInsert("itiiiiiirrr", image_asset_id, filename.ToUTF8().data(), position_in_stack, parent_movie_id, alignment_id, ctf_estimation_id, x_size, y_size, pixel_size, voltage, spherical_aberration);
}

/*
void Database::BeginAbInitioParticlePositionAssetInsert()
{
	BeginBatchInsert("PARTICLE_POSITION_ASSETS", 6, "PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "PICKING_ID", "X_POSITION", "Y_POSITION","PEAK_HEIGHT");
}

void Database::AddNextAbInitioParticlePositionAsset(int particle_position_asset_id, int parent_image_asset_id, int pick_job_id, double x_position, double y_position, double peak_height )
{
	AddToBatchInsert("iiirrr", particle_position_asset_id, parent_image_asset_id, pick_job_id, x_position, y_position, peak_height);
}
*/

void Database::BeginParticlePositionAssetInsert()
{
	BeginBatchInsert("particle_position_assets", 11, "particle_position_asset_id", "parent_image_asset_id", "picking_id", "pick_job_id", "x_position", "y_position", "peak_height", "template_asset_id", "template_psi", "template_theta", "template_phi");
}

void Database::AddNextParticlePositionAsset(const ParticlePositionAsset *asset)
{
	AddToBatchInsert("iiiirrrirrr", asset->asset_id, asset->parent_id, asset->picking_id, asset->pick_job_id, asset->x_position, asset->y_position, asset->peak_height, asset->parent_template_id, asset->template_psi, asset->template_theta, asset->template_phi);
}



bool Database::BeginBatchSelect(const char *select_command)
{
	MyDebugAssertTrue(is_open == true, "database not open!");
	MyDebugAssertTrue(in_batch_insert == false, "Starting batch select but already in batch insert mode");
	MyDebugAssertTrue(in_batch_select == false, "Starting batch select but already in batch select mode");


	in_batch_select = true;
	int return_code;

	return_code = sqlite3_prepare_v2(sqlite_database, select_command, strlen(select_command) + 1, &batch_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i, Command = %s\n", return_code, select_command );

	last_return_code = sqlite3_step(batch_statement);

	if (last_return_code != SQLITE_DONE && last_return_code != SQLITE_ROW)
	{
		MyPrintWithDetails("SQL Return Code: %i\n", last_return_code);
	}

	if (last_return_code != SQLITE_DONE) return true;
	else return false;
	//return true;
}

bool Database::GetFromBatchSelect(const char *column_format, ...)
{

	MyDebugAssertTrue(is_open == true, "database not open!");
	MyDebugAssertTrue(in_batch_insert == false, "in batch select but batch insert is true");
	MyDebugAssertTrue(in_batch_select == true, "in batch select but batch select is false");
	MyDebugAssertTrue(last_return_code == SQLITE_ROW, "get from batch select, but return code is not SQLITE_ROW");

	int argument_counter = -1;

	va_list args;
	va_start(args, column_format);

	while (*column_format != '\0')
	{
		argument_counter++;

		if (*column_format == 't') // text
		{
			va_arg(args, wxString *)[0] = sqlite3_column_text(batch_statement, argument_counter);
		}
		else
		if (*column_format == 'f') // filename
		{
			va_arg(args, wxFileName *)[0] = sqlite3_column_text(batch_statement, argument_counter);
		}
		else
		if (*column_format == 'r') // real
		{
			va_arg(args, double *)[0] = sqlite3_column_double(batch_statement, argument_counter);
		}
		else
		if (*column_format == 'i') // integer
		{
			va_arg(args, int *)[0] = sqlite3_column_int(batch_statement, argument_counter);
		}
		else
		if (*column_format == 'l') // long
		{
			va_arg(args, long *)[0] = sqlite3_column_int64(batch_statement, argument_counter);
		}
		else
		{
			MyPrintWithDetails("Error: Unknown format character!\n");
		}

		 ++column_format;
	}

	va_end(args);

	last_return_code = sqlite3_step(batch_statement);
	MyDebugAssertTrue(last_return_code == SQLITE_OK || last_return_code == SQLITE_ROW || last_return_code == SQLITE_DONE, "SQL error, return code : %i\n", last_return_code );

	if (last_return_code == SQLITE_DONE) return false;
	else return true;

}

void Database::EndBatchSelect()
{
	sqlite3_finalize(batch_statement);
	in_batch_select = false;
}

void Database::BeginAllMovieAssetsSelect()
{
	BeginBatchSelect("SELECT * FROM MOVIE_ASSETS;");
}

void Database::BeginAllMovieGroupsSelect()
{
	BeginBatchSelect("SELECT * FROM MOVIE_GROUP_LIST;");
}

void Database::BeginAllImageAssetsSelect()
{
	BeginBatchSelect("SELECT * FROM IMAGE_ASSETS;");
}

void Database::BeginAllImageGroupsSelect()
{
	BeginBatchSelect("SELECT * FROM IMAGE_GROUP_LIST;");
}

void Database::BeginAllParticlePositionAssetsSelect()
{
	BeginBatchSelect("SELECT * FROM PARTICLE_POSITION_ASSETS;");
}

void Database::BeginAllVolumeAssetsSelect()
{
	BeginBatchSelect("SELECT * FROM VOLUME_ASSETS;");
}

void Database::BeginAllParticlePositionGroupsSelect()
{
	BeginBatchSelect("SELECT * FROM PARTICLE_POSITION_GROUP_LIST;");
}

void Database::BeginAllVolumeGroupsSelect()
{
	BeginBatchSelect("SELECT * FROM VOLUME_GROUP_LIST;");
}




void Database::BeginAllRunProfilesSelect()
{
	BeginBatchSelect("SELECT * FROM RUN_PROFILES;");
}

RunProfile Database::GetNextRunProfile()
{
	RunProfile temp_profile;
	int profile_table_number;
	int return_code;
	wxString profile_sql_select_command;
	sqlite3_stmt *list_statement = NULL;

	GetFromBatchSelect("itttti", &temp_profile.id, &temp_profile.name, &temp_profile.manager_command, &temp_profile.gui_address, &temp_profile.controller_address, &profile_table_number);

	// now we fill from the specific group table.

	profile_sql_select_command = wxString::Format("SELECT * FROM RUN_PROFILE_COMMANDS_%i", profile_table_number);

	return_code = sqlite3_prepare_v2(sqlite_database, profile_sql_select_command.ToUTF8().data(), profile_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
		temp_profile.AddCommand(sqlite3_column_text(list_statement, 1), sqlite3_column_int(list_statement, 2), sqlite3_column_int(list_statement, 3));
		return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);
	return temp_profile;
}

AssetGroup Database::GetNextMovieGroup()
{
	AssetGroup temp_group;
	int group_table_number;
	int return_code;
	wxString group_sql_select_command;
	sqlite3_stmt *list_statement = NULL;

	GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

	// now we fill from the specific group table.

	group_sql_select_command = wxString::Format("SELECT * FROM MOVIE_GROUP_%i", group_table_number);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
			temp_group.AddMember(sqlite3_column_int(list_statement, 1));
			return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);
	return temp_group;
}

AssetGroup Database::GetNextImageGroup()
{
	AssetGroup temp_group;
	int group_table_number;
	int return_code;
	wxString group_sql_select_command;
	sqlite3_stmt *list_statement = NULL;

	GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

	// now we fill from the specific group table.

	group_sql_select_command = wxString::Format("SELECT * FROM IMAGE_GROUP_%i", group_table_number);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
			temp_group.AddMember(sqlite3_column_int(list_statement, 1));
			return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);
	return temp_group;
}

AssetGroup Database::GetNextParticlePositionGroup()
{
	AssetGroup temp_group;
	int group_table_number;
	int return_code;
	wxString group_sql_select_command;
	sqlite3_stmt *list_statement = NULL;

	GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

	// now we fill from the specific group table.

	group_sql_select_command = wxString::Format("SELECT * FROM PARTICLE_POSITION_GROUP_%i", group_table_number);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
			temp_group.AddMember(sqlite3_column_int(list_statement, 1));
			return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);
	return temp_group;
}

AssetGroup Database::GetNextVolumeGroup()
{
	AssetGroup temp_group;
	int group_table_number;
	int return_code;
	wxString group_sql_select_command;
	sqlite3_stmt *list_statement = NULL;

	GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

	// now we fill from the specific group table.

	group_sql_select_command = wxString::Format("SELECT * FROM VOLUME_GROUP_%i", group_table_number);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
			temp_group.AddMember(sqlite3_column_int(list_statement, 1));
			return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);
	return temp_group;
}

void Database::RemoveParticlePositionsFromResultsList(const int &picking_job_id, const int &parent_image_asset_id)
{
	ExecuteSQL(wxString::Format("delete from particle_picking_results_%i where parent_image_asset_id = %i",picking_job_id,parent_image_asset_id));
}

int Database::ReturnPickingIDGivenPickingJobIDAndParentImageID(const int & picking_job_id, const int &parent_image_asset_id)
{
	return ReturnSingleIntFromSelectCommand(wxString::Format("select distinct picking_id from particle_picking_results_%i where parent_image_asset_id = %i",picking_job_id, parent_image_asset_id));
}

void Database::SetManualEditForPickingID(const int &picking_id, const bool wanted_manual_edit)
{
	int manual_edit_value = 0;
	if (wanted_manual_edit) manual_edit_value = 1;
	ExecuteSQL(wxString::Format("update particle_picking_list set manual_edit=%i where picking_id=%i",manual_edit_value,picking_id));
}

void Database::RemoveParticlePositionsWithGivenParentImageIDFromGroup( const int &group_number_following_gui_convention, const int &parent_image_asset_id)
{
	ExecuteSQL(wxString::Format("delete from particle_position_group_%i where exists(select 1 from particle_position_assets where particle_position_assets.parent_image_asset_id = %i AND particle_position_group_%i.particle_position_asset_id = particle_position_assets.particle_position_asset_id)",group_number_following_gui_convention-1,parent_image_asset_id,group_number_following_gui_convention-1));
}

void Database::RemoveParticlePositionAssetsPickedFromImagesAlsoPickedByGivenPickingJobID( const int &picking_job_id)
{
	ExecuteSQL(wxString::Format("delete from particle_position_assets where exists(select 1 from particle_picking_list where particle_picking_list.picking_job_id = %i AND particle_picking_list.parent_image_asset_id = particle_position_assets.parent_image_asset_id)",picking_job_id));
}

void Database::RemoveParticlePositionAssetsPickedFromImageWithGivenID( const int &parent_image_asset_id )
{
	ExecuteSQL(wxString::Format("delete from particle_position_assets where parent_image_asset_id = %i",parent_image_asset_id));
}

void Database::CopyParticleAssetsFromResultsTable(const int &picking_job_id, const int &parent_image_asset_id)
{
	ExecuteSQL(wxString::Format("insert into particle_position_assets select particle_picking_results_%i.position_id, %i, particle_picking_results_%i.picking_id, %i, particle_picking_results_%i.x_position, particle_picking_results_%i.y_position, particle_picking_results_%i.peak_height, particle_picking_results_%i.template_asset_id, particle_picking_results_%i.template_psi, particle_picking_results_%i.template_theta, particle_picking_results_%i.template_phi from particle_picking_results_%i where particle_picking_results_%i.parent_image_asset_id = %i",
												                                          picking_job_id,  parent_image_asset_id,     picking_job_id,     picking_job_id,           picking_job_id,                         picking_job_id,                              picking_job_id,                            picking_job_id,                             picking_job_id,                                picking_job_id,                        picking_job_id,                              picking_job_id,                    picking_job_id,                  parent_image_asset_id));
}

void Database::AddArrayOfParticlePositionAssetsToResultsTable(const int &picking_job_id, ArrayOfParticlePositionAssets *array_of_assets)
{
	BeginBatchInsert(wxString::Format("particle_picking_results_%i",picking_job_id), 10, "position_id", "picking_id", "parent_image_asset_id", "x_position", "y_position", "peak_height", "template_asset_id", "template_psi", "template_theta", "template_phi");

	ParticlePositionAsset *asset;
	for (size_t counter = 0; counter < array_of_assets->GetCount(); counter ++ )
	{
		asset = & array_of_assets->Item(counter);
		AddToBatchInsert("iiirrrirrr", asset->asset_id, asset->picking_id, asset->parent_id, asset->x_position, asset->y_position, asset->peak_height, asset->parent_template_id, asset->template_psi, asset->template_theta, asset->template_phi);
	}

	EndBatchInsert();

}

void Database::AddArrayOfParticlePositionAssetsToAssetsTable(ArrayOfParticlePositionAssets *array_of_assets)
{
	BeginParticlePositionAssetInsert();
	ParticlePositionAsset *asset;
	for (size_t counter = 0; counter < array_of_assets->GetCount(); counter ++ )
	{
		asset = & array_of_assets->Item(counter);
		AddNextParticlePositionAsset(asset);
	}

	EndBatchInsert();
}

ArrayOfParticlePositionAssets Database::ReturnArrayOfParticlePositionAssetsFromResultsTable(const int &picking_job_id, const int &parent_image_asset_id)
{
	ArrayOfParticlePositionAssets array_of_assets;
	array_of_assets.Clear();
	BeginBatchSelect(wxString::Format("select * from particle_picking_results_%i where parent_image_asset_id = %i",picking_job_id,parent_image_asset_id));
	while (last_return_code == SQLITE_ROW)
	{
		array_of_assets.Add(GetNextParticlePositionAssetFromResults());
	}
	EndBatchSelect();
	return array_of_assets;
}

MovieAsset Database::GetNextMovieAsset()
{
	MovieAsset temp_asset;

	GetFromBatchSelect("ifiiiirrrr", &temp_asset.asset_id, &temp_asset.filename, &temp_asset.position_in_stack, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.number_of_frames, &temp_asset.microscope_voltage, &temp_asset.pixel_size, &temp_asset.dose_per_frame, &temp_asset.spherical_aberration);
	temp_asset.total_dose = temp_asset.dose_per_frame * temp_asset.number_of_frames;
	return temp_asset;
}


ImageAsset Database::GetNextImageAsset()
{
	ImageAsset temp_asset;

	GetFromBatchSelect("ifiiiiiirrr", &temp_asset.asset_id, &temp_asset.filename, &temp_asset.position_in_stack, &temp_asset.parent_id, &temp_asset.alignment_id, &temp_asset.ctf_estimation_id, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.pixel_size, &temp_asset.microscope_voltage,  &temp_asset.spherical_aberration);
	return temp_asset;
}

ParticlePositionAsset Database::GetNextParticlePositionAsset()
{
	ParticlePositionAsset temp_asset;
	GetFromBatchSelect("iiiirrrirrr", &temp_asset.asset_id, &temp_asset.parent_id, &temp_asset.picking_id, &temp_asset.pick_job_id, &temp_asset.x_position, &temp_asset.y_position,&temp_asset.peak_height,&temp_asset.parent_template_id,&temp_asset.template_psi,&temp_asset.template_theta,&temp_asset.template_phi);
	return temp_asset;
}

ParticlePositionAsset Database::GetNextParticlePositionAssetFromResults()
{
	ParticlePositionAsset temp_asset;
	GetFromBatchSelect("iiirrrirrr", &temp_asset.asset_id, &temp_asset.picking_id,  &temp_asset.parent_id, &temp_asset.x_position, &temp_asset.y_position,&temp_asset.peak_height,&temp_asset.parent_template_id,&temp_asset.template_psi,&temp_asset.template_theta,&temp_asset.template_phi);
	return temp_asset;
}


VolumeAsset Database::GetNextVolumeAsset()
{
	VolumeAsset temp_asset;
	GetFromBatchSelect("fiiriii", &temp_asset.filename, &temp_asset.asset_id, &temp_asset.reconstruction_job_id, &temp_asset.pixel_size, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.z_size);
	return temp_asset;
}


bool Database::AddOrReplaceRunProfile(RunProfile *profile_to_add)
{

	InsertOrReplace("RUN_PROFILES", "ptttti", "RUN_PROFILE_ID", "PROFILE_NAME", "MANAGER_RUN_COMMAND", "GUI_ADDRESS", "CONTROLLER_ADDRESS", "COMMANDS_ID", profile_to_add->id, profile_to_add->name.ToUTF8().data(), profile_to_add->manager_command.ToUTF8().data(), profile_to_add->gui_address.ToUTF8().data(), profile_to_add->controller_address.ToUTF8().data(), profile_to_add->id);
	DeleteTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id));
	CreateTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id), "ptii", "COMMANDS_NUMBER", "COMMAND_STRING", "NUMBER_OF_COPIES", "DELAY_TIME_IN_MS");

	for (int counter = 0; counter < profile_to_add->number_of_run_commands; counter++)
	{
		InsertOrReplace(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id), "ptii", "COMMANDS_NUMBER", "COMMAND_STRING", "NUMBER_OF_COPIES", "DELAY_TIME_IN_MS", counter, profile_to_add->run_commands[counter].command_to_run.ToUTF8().data(), profile_to_add->run_commands[counter].number_of_copies, profile_to_add->run_commands[counter].delay_time_in_ms);
	}
}

bool Database::DeleteRunProfile(int wanted_id)
{
	ExecuteSQL(wxString::Format("DELETE FROM RUN_PROFILES WHERE RUN_PROFILE_ID=%i", wanted_id).ToUTF8().data());
	DeleteTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", wanted_id));
}

