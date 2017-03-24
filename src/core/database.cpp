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

long Database::ReturnSingleLongFromSelectCommand(wxString select_command)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	int return_code;
	sqlite3_stmt *current_statement;
	long value;

	return_code = sqlite3_prepare_v2(sqlite_database, select_command, select_command.Length() + 1, &current_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(current_statement);

	if (return_code != SQLITE_DONE && return_code != SQLITE_ROW)
	{
		MyPrintWithDetails("SQL Return Code: %i\n", return_code);
	}

	value = sqlite3_column_int64(current_statement, 0);

	sqlite3_finalize(current_statement);

	return value;
}

void Database::GetActiveDefocusValuesByImageID(long wanted_image_id, float &defocus_1, float &defocus_2, float &defocus_angle)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	int return_code;
	sqlite3_stmt *current_statement;
	int value;
	wxString select_command = wxString::Format("SELECT DEFOCUS1, DEFOCUS2, DEFOCUS_ANGLE FROM ESTIMATED_CTF_PARAMETERS, IMAGE_ASSETS WHERE ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_ID=IMAGE_ASSETS.CTF_ESTIMATION_ID AND IMAGE_ASSETS.IMAGE_ASSET_ID=%li;", wanted_image_id);

	return_code = sqlite3_prepare_v2(sqlite_database, select_command, select_command.Length() + 1, &current_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(current_statement);

	if (return_code != SQLITE_DONE && return_code != SQLITE_ROW)
	{
		MyPrintWithDetails("SQL Return Code: %i\n", return_code);
	}

	defocus_1 = float(sqlite3_column_double(current_statement, 0));
	defocus_2 = float(sqlite3_column_double(current_statement, 1));
	defocus_angle = float(sqlite3_column_double(current_statement, 2));

	sqlite3_finalize(current_statement);
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

long Database::ReturnHighestRefinementID()
{
	return ReturnSingleLongFromSelectCommand("SELECT MAX(REFINEMENT_ID) FROM REFINEMENT_LIST");
}

long Database::ReturnHighestClassificationID()
{
	return ReturnSingleLongFromSelectCommand("SELECT MAX(CLASSIFICATION_ID) FROM CLASSIFICATION_LIST");
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
	if (number_of_non_empty_tables > 0)
	{
		return ReturnSingleIntFromSelectCommand(sql_query);
	}
	else
	{
		return 0;
	}
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

long Database::ReturnHighestClassumSelectionID()
{
	return ReturnSingleLongFromSelectCommand("SELECT MAX(SELECTION_ID) FROM CLASSIFICATION_SELECTION_LIST");
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
		if (*column_format == 'l') // integer
		{
			sql_command += va_arg(args, const char *);
			sql_command += " INTEGER";
		}
		else
		if (*column_format == 'p' || *column_format == 'P') // integer
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
	//success = CreateTable("MOVIE_ASSETS", "ptiiiirrrr", "MOVIE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION");
	success = CreateMovieAssetTable();
	success = CreateImageAssetTable();
	success = CreateTable("MOVIE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );
	success = CreateTable("MOVIE_ALIGNMENT_LIST", "piiitrrrrrriiriiiii", "ALIGNMENT_ID", "DATETIME_OF_RUN", "ALIGNMENT_JOB_ID", "MOVIE_ASSET_ID", "OUTPUT_FILE", "VOLTAGE", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "PRE_EXPOSURE_AMOUNT", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER", "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS", "HORIZONTAL_MASK", "VERTICAL_MASK" );

	success = CreateTable("IMAGE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );
	success = CreateParticlePickingListTable();
	success = CreateParticlePositionAssetTable();
	success = CreateParticlePositionGroupListTable();
	success = CreateVolumeAssetTable();
	success = CreateVolumeGroupListTable();
	success = CreateRefinementPackageAssetTable();
	success = CreateRefinementListTable();
	success = CreateClassificationListTable();
	success = CreateClassificationSelectionListTable();


	success = CreateTable("ESTIMATED_CTF_PARAMETERS", "piiiirrrrirrrrririrrrrrrrrrrtii", "CTF_ESTIMATION_ID", "CTF_ESTIMATION_JOB_ID", "DATETIME_OF_RUN", "IMAGE_ASSET_ID", "ESTIMATED_ON_MOVIE_FRAMES", "VOLTAGE", "SPHERICAL_ABERRATION", "PIXEL_SIZE", "AMPLITUDE_CONTRAST", "BOX_SIZE", "MIN_RESOLUTION", "MAX_RESOLUTION", "MIN_DEFOCUS", "MAX_DEFOCUS", "DEFOCUS_STEP", "RESTRAIN_ASTIGMATISM", "TOLERATED_ASTIGMATISM", "FIND_ADDITIONAL_PHASE_SHIFT", "MIN_PHASE_SHIFT", "MAX_PHASE_SHIFT", "PHASE_SHIFT_STEP", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE", "ADDITIONAL_PHASE_SHIFT", "SCORE", "DETECTED_RING_RESOLUTION", "DETECTED_ALIAS_RESOLUTION", "OUTPUT_DIAGNOSTIC_FILE","NUMBER_OF_FRAMES_AVERAGED","LARGE_ASTIGMATISM_EXPECTED");
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
		if (*column_format == 'l' || *column_format == 'P') // long
		{
			sql_command += wxString::Format("%li",  va_arg(args, long));
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
    	abort();
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
	MyDebugAssertTrue(is_open == true, "da1414 tabase not open!");
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
	BeginBatchInsert("MOVIE_ASSETS", 17, "MOVIE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION","GAIN_FILENAME","OUTPUT_BINNING_FACTOR", "CORRECT_MAG_DISTORTION", "MAG_DISTORTION_ANGLE", "MAG_DISTORTION_MAJOR_SCALE", "MAG_DISTORTION_MINOR_SCALE");
}

void Database::AddNextMovieAsset(int movie_asset_id,  wxString name, wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration, wxString gain_filename, double output_binning_factor, int correct_mag_distortion, float mag_distortion_angle, float mag_distortion_major_scale, float mag_distortion_minor_scale)
{
	AddToBatchInsert("ittiiiirrrrtrirrr", movie_asset_id, name.ToUTF8().data(), filename.ToUTF8().data(), position_in_stack, x_size, y_size, number_of_frames, voltage, pixel_size, dose_per_frame, spherical_aberration,gain_filename.ToUTF8().data(), output_binning_factor, correct_mag_distortion, mag_distortion_angle, mag_distortion_major_scale, mag_distortion_minor_scale);
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
	BeginBatchInsert("IMAGE_ASSETS", 12, "IMAGE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "ALIGNMENT_ID", "CTF_ESTIMATION_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION");
}

void Database::BeginVolumeAssetInsert()
{
	BeginBatchInsert("VOLUME_ASSETS", 8, "VOLUME_ASSET_ID", "NAME", "FILENAME", "RECONSTRUCTION_JOB_ID", "PIXEL_SIZE", "X_SIZE", "Y_SIZE", "Z_SIZE");
}

void Database::AddNextVolumeAsset(int image_asset_id,  wxString name, wxString filename, int reconstruction_job_id, double pixel_size, int x_size, int y_size, int z_size)
{
	AddToBatchInsert("ittiriii", image_asset_id, name.ToUTF8().data(), filename.ToUTF8().data(), reconstruction_job_id, pixel_size, x_size, y_size, z_size);
}

void Database::AddNextImageAsset(int image_asset_id,  wxString name, wxString filename, int position_in_stack, int parent_movie_id, int alignment_id, int ctf_estimation_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration)
{
	AddToBatchInsert("ittiiiiiirrr", image_asset_id, name.ToUTF8().data(), filename.ToUTF8().data(), position_in_stack, parent_movie_id, alignment_id, ctf_estimation_id, x_size, y_size, pixel_size, voltage, spherical_aberration);
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
		if (*column_format == 's') // single (float)
		{
			double temp_double =  sqlite3_column_double(batch_statement, argument_counter);
			va_arg(args, float *)[0] = float(temp_double);
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

void Database::BeginAllRefinementPackagesSelect()
{
	BeginBatchSelect("SELECT * FROM REFINEMENT_PACKAGE_ASSETS;");}




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



RefinementPackage*  Database::GetNextRefinementPackage()
{
	RefinementPackage *temp_package;
	RefinementPackageParticleInfo temp_info;

	temp_package = new RefinementPackage;

	int return_code;


	wxString group_sql_select_command;
	sqlite3_stmt *list_statement = NULL;

	GetFromBatchSelect("lttitrriii", &temp_package->asset_id, &temp_package->name, &temp_package->stack_filename, &temp_package->stack_box_size, &temp_package->symmetry, &temp_package->estimated_particle_weight_in_kda, &temp_package->estimated_particle_size_in_angstroms, &temp_package->number_of_classes, &temp_package->number_of_run_refinments, &temp_package->last_refinment_id);

	// particles

	group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", temp_package->asset_id);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
		temp_info.original_particle_position_asset_id = sqlite3_column_int64(list_statement, 0);
		temp_info.parent_image_id = sqlite3_column_int64(list_statement, 1);
		temp_info.position_in_stack = sqlite3_column_int64(list_statement, 2);
		temp_info.x_pos = sqlite3_column_double(list_statement, 3);
		temp_info.y_pos = sqlite3_column_double(list_statement, 4);
		temp_info.pixel_size = sqlite3_column_double(list_statement, 5);
		temp_info.defocus_1 = sqlite3_column_double(list_statement, 6);
		temp_info.defocus_2 = sqlite3_column_double(list_statement, 7);
		temp_info.defocus_angle = sqlite3_column_double(list_statement, 8);
		temp_info.phase_shift = sqlite3_column_double(list_statement, 9);
		temp_info.spherical_aberration = sqlite3_column_double(list_statement, 10);
		temp_info.microscope_voltage = sqlite3_column_double(list_statement, 11);

		temp_package->contained_particles.Add(temp_info);

		return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);

	// 3d references

	group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", temp_package->asset_id);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
		temp_package->references_for_next_refinement.Add(sqlite3_column_int64(list_statement, 1));
		return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);


	// refinement list

	group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", temp_package->asset_id);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
		temp_package->refinement_ids.Add(sqlite3_column_int64(list_statement, 1));
		return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);


	// classification list

	group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", temp_package->asset_id);

	return_code = sqlite3_prepare_v2(sqlite_database, group_sql_select_command.ToUTF8().data(), group_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
		temp_package->classification_ids.Add(sqlite3_column_int64(list_statement, 1));
		return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );

	sqlite3_finalize(list_statement);



	return temp_package;

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

ArrayOfParticlePositionAssets Database::ReturnArrayOfParticlePositionAssetsFromAssetsTable(const int &parent_image_asset_id)
{
	ArrayOfParticlePositionAssets array_of_assets;
	array_of_assets.Clear();
	BeginBatchSelect(wxString::Format("select * from particle_position_assets where parent_image_asset_id = %i",parent_image_asset_id));
	while (last_return_code == SQLITE_ROW)
	{
		array_of_assets.Add(GetNextParticlePositionAsset());
	}
	EndBatchSelect();
	return array_of_assets;
}

wxArrayLong  Database::Return2DClassMembers(long wanted_classifiction_id, int wanted_class_number)
{
	wxArrayLong class_members;
	long temp_long;

	BeginBatchSelect(wxString::Format("select POSITION_IN_STACK from classification_result_%li where BEST_CLASS = %i", wanted_classifiction_id, wanted_class_number));

	while (last_return_code == SQLITE_ROW)
	{
		GetFromBatchSelect("l", &temp_long);
		class_members.Add(temp_long);
	}

	EndBatchSelect();

	return class_members;


}

MovieAsset Database::GetNextMovieAsset()
{
	MovieAsset temp_asset;
	int correct_mag_distortion;

	GetFromBatchSelect("itfiiiirrrrtrirrr", &temp_asset.asset_id, &temp_asset.asset_name, &temp_asset.filename, &temp_asset.position_in_stack, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.number_of_frames, &temp_asset.microscope_voltage, &temp_asset.pixel_size, &temp_asset.dose_per_frame, &temp_asset.spherical_aberration, & temp_asset.gain_filename, & temp_asset.output_binning_factor, &correct_mag_distortion, &temp_asset.mag_distortion_angle, &temp_asset.mag_distortion_major_scale, &temp_asset.mag_distortion_minor_scale);
	temp_asset.correct_mag_distortion = correct_mag_distortion;
	temp_asset.total_dose = temp_asset.dose_per_frame * temp_asset.number_of_frames;
	return temp_asset;
}


ImageAsset Database::GetNextImageAsset()
{
	ImageAsset temp_asset;

	GetFromBatchSelect("itfiiiiiirrr", &temp_asset.asset_id, &temp_asset.asset_name, &temp_asset.filename, &temp_asset.position_in_stack, &temp_asset.parent_id, &temp_asset.alignment_id, &temp_asset.ctf_estimation_id, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.pixel_size, &temp_asset.microscope_voltage,  &temp_asset.spherical_aberration);
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
	GetFromBatchSelect("itflriii", &temp_asset.asset_id, &temp_asset.asset_name, &temp_asset.filename, &temp_asset.reconstruction_job_id, &temp_asset.pixel_size, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.z_size);
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

void Database::AddRefinementPackageAsset(RefinementPackage *asset_to_add)
{
	InsertOrReplace("REFINEMENT_PACKAGE_ASSETS", "Pttitrriii", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "STACK_FILENAME", "STACK_BOX_SIZE", "SYMMETRY", "MOLECULAR_WEIGHT", "PARTICLE_SIZE", "NUMBER_OF_CLASSES", "NUMBER_OF_REFINEMENTS", "LAST_REFINEMENT_ID", asset_to_add->asset_id, asset_to_add->name.ToUTF8().data(), asset_to_add->stack_filename.ToUTF8().data(), asset_to_add->stack_box_size, asset_to_add->symmetry.ToUTF8().data(), asset_to_add->estimated_particle_weight_in_kda, asset_to_add->estimated_particle_size_in_angstroms, asset_to_add->number_of_classes, asset_to_add->number_of_run_refinments, asset_to_add->last_refinment_id);
	CreateRefinementPackageContainedParticlesTable(asset_to_add->asset_id);
	CreateRefinementPackageCurrent3DReferencesTable(asset_to_add->asset_id);
	CreateRefinementPackageRefinementsList(asset_to_add->asset_id);
	CreateRefinementPackageClassificationsList(asset_to_add->asset_id);


	BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", asset_to_add->asset_id), 12 , "ORIGINAL_PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "POSITION_IN_STACK", "X_POSITION", "Y_POSITION", "PIXEL_SIZE", "DEFOCUS_1", "DEFOCUS_2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "SPHERICAL_ABERRATION", "MICROSCOPE_VOLTAGE");

	for (long counter = 0; counter < asset_to_add->contained_particles.GetCount(); counter++)
	{

		AddToBatchInsert("lllrrrrrrrrr", asset_to_add->contained_particles.Item(counter).original_particle_position_asset_id, asset_to_add->contained_particles.Item(counter).parent_image_id, asset_to_add->contained_particles.Item(counter).position_in_stack, asset_to_add->contained_particles.Item(counter).x_pos, asset_to_add->contained_particles.Item(counter).y_pos, asset_to_add->contained_particles.Item(counter).pixel_size, asset_to_add->contained_particles.Item(counter).defocus_1, asset_to_add->contained_particles.Item(counter).defocus_2, asset_to_add->contained_particles.Item(counter).defocus_angle, asset_to_add->contained_particles.Item(counter).phase_shift, asset_to_add->contained_particles.Item(counter).spherical_aberration, asset_to_add->contained_particles.Item(counter).microscope_voltage);
	}

	EndBatchInsert();


	BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", asset_to_add->asset_id), 2 , "CLASS_NUMBER", "VOLUME_ASSET_ID");

	for (long counter = 0; counter < asset_to_add->references_for_next_refinement.GetCount(); counter++)
	{

		AddToBatchInsert("ll", counter + 1, asset_to_add->references_for_next_refinement.Item(counter));
	}

	EndBatchInsert();

	BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", asset_to_add->asset_id), 2 , "REFINEMENT_NUMBER", "REFINEMENT_ID");

	for (long counter = 0; counter < asset_to_add->refinement_ids.GetCount(); counter++)
	{

		AddToBatchInsert("ll", counter + 1, asset_to_add->refinement_ids.Item(counter));
	}

	EndBatchInsert();

	BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", asset_to_add->asset_id), 2 , "CLASSIFICATION_NUMBER", "CLASSIFICATION_ID");

	for (long counter = 0; counter < asset_to_add->classification_ids.GetCount(); counter++)
	{

		AddToBatchInsert("ll", counter + 1, asset_to_add->classification_ids.Item(counter));
	}

	EndBatchInsert();

}

void Database::AddRefinement(Refinement *refinement_to_add)
{
	int class_counter;
	long counter;

	InsertOrReplace("REFINEMENT_LIST", "Pltillllrrrrrrirrrrirrrrirrir", "REFINEMENT_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "REFINEMENT_WAS_IMPORTED_OR_GENERATED", "DATETIME_OF_RUN", "STARTING_REFINEMENT_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "SIGNED_CC_RESOLUTION_LIMIT", "GLOBAL_RESOLUTION_LIMIT", "GLOBAL_MASK_RADIUS", "NUMBER_RESULTS_TO_REFINE", "ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "CLASSIFICATION_RESOLUTION_LIMIT", "SHOULD_FOCUS_CLASSIFY", "SPHERE_X_COORD", "SPHERE_Y_COORD", "SPHERE_Z_COORD", "SPHERE_RADIUS", "SHOULD_REFINE_CTF", "DEFOCUS_SEARCH_RANGE", "DEFOCUS_SEARCH_STEP", "RESOLUTION_STATISTICS_BOX_SIZE", "RESOLUTION_STATISTICS_PIXEL_SIZE", refinement_to_add->refinement_id, refinement_to_add->refinement_package_asset_id, refinement_to_add->name.ToUTF8().data(), refinement_to_add->refinement_was_imported_or_generated, refinement_to_add->datetime_of_run.GetAsDOS(), refinement_to_add->starting_refinement_id, refinement_to_add->number_of_particles, refinement_to_add->number_of_classes, refinement_to_add->low_resolution_limit, refinement_to_add->high_resolution_limit, refinement_to_add->mask_radius, refinement_to_add->signed_cc_resolution_limit, refinement_to_add->global_resolution_limit, refinement_to_add->global_mask_radius, refinement_to_add->number_results_to_refine, refinement_to_add->angular_search_step, refinement_to_add->search_range_x, refinement_to_add->search_range_y, refinement_to_add->classification_resolution_limit, refinement_to_add->should_focus_classify, refinement_to_add->sphere_x_coord, refinement_to_add->sphere_y_coord, refinement_to_add->sphere_z_coord, refinement_to_add->sphere_radius, refinement_to_add->should_refine_ctf, refinement_to_add->defocus_search_range, refinement_to_add->defocus_search_step, refinement_to_add->resolution_statistics_box_size, refinement_to_add->resolution_statistics_pixel_size);

	for (class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++)
	{
		CreateRefinementResultTable(refinement_to_add->refinement_id, class_counter);
		CreateRefinementResolutionStatisticsTable(refinement_to_add->refinement_id, class_counter);
	}

	CreateRefinementReferenceVolumeIDsTable(refinement_to_add->refinement_id);

	BeginBatchInsert(wxString::Format("REFINEMENT_REFERENCE_VOLUME_IDS_%li", refinement_to_add->refinement_id), 2 , "CLASS_NUMBER", "VOLUME_ASSET_ID");

	for ( counter = 0; counter < refinement_to_add->reference_volume_ids.GetCount(); counter++)
	{
		AddToBatchInsert("ll", counter, refinement_to_add->reference_volume_ids[counter]);
	}

	EndBatchInsert();

	for (class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++)
	{
		BeginBatchInsert(wxString::Format("REFINEMENT_RESULT_%li_%i", refinement_to_add->refinement_id, class_counter), 15 ,"POSITION_IN_STACK", "PSI", "THETA", "PHI", "XSHIFT", "YSHIFT", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "OCCUPANCY", "LOGP", "SIGMA", "SCORE", "SCORE_CHANGE");

		for (counter = 0; counter < refinement_to_add->number_of_particles; counter++)
		{
			AddToBatchInsert("lrrrrrrrrrrrrrr", refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].position_in_stack, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].psi, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].theta, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].phi, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].xshift, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].yshift, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].defocus1, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].defocus2, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].defocus_angle, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].phase_shift, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].occupancy, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].logp, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].sigma, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].score, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].score_change);
		}

		EndBatchInsert();
	}

	for (class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++)
	{

		BeginBatchInsert(wxString::Format("REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_to_add->refinement_id, class_counter), 6 ,"SHELL", "RESOLUTION", "FSC", "PART_FSC", "PART_SSNR", "REC_SSNR");

		for ( counter = 0; counter <= refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.number_of_points; counter++)
		{
			AddToBatchInsert("lrrrrr", counter, refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_x[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_SSNR.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.rec_SSNR.data_y[counter]);
		}

		EndBatchInsert();
	}

}


void Database::UpdateRefinementResolutionStatistics(Refinement *refinement_to_add)
{
	int class_counter;
	long counter;

	ExecuteSQL("BEGIN");

	for (class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++)
	{
		ExecuteSQL(wxString::Format("DROP TABLE REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_to_add->refinement_id, class_counter));
	}

	ExecuteSQL("COMMIT");

	for (class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++)
	{
		CreateRefinementResolutionStatisticsTable(refinement_to_add->refinement_id, class_counter);

		BeginBatchInsert(wxString::Format("REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_to_add->refinement_id, class_counter), 6 ,"SHELL", "RESOLUTION", "FSC", "PART_FSC", "PART_SSNR", "REC_SSNR");

		for ( counter = 0; counter <= refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.number_of_points; counter++)
		{
			AddToBatchInsert("lrrrrr", counter, refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_x[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_SSNR.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.rec_SSNR.data_y[counter]);
		}

		EndBatchInsert();
	}

}

Refinement *Database::GetRefinementByID(long wanted_refinement_id)
{
	wxString sql_select_command;
	int return_code;
	sqlite3_stmt *list_statement = NULL;
	Refinement *temp_refinement = new Refinement;
	RefinementResult temp_result;
	int class_counter;

	float temp_resolution;
	float temp_fsc;
	float temp_part_fsc;
	float temp_part_ssnr;
	float temp_rec_ssnr;

	bool more_data;

	ClassRefinementResults junk_class_results;
	RefinementResult junk_result;

	// general data

	sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_LIST WHERE REFINEMENT_ID=%li", wanted_refinement_id);

	return_code = sqlite3_prepare_v2(sqlite_database, sql_select_command.ToUTF8().data(), sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\nSQL Command : %s\n", return_code , sql_select_command);

	return_code = sqlite3_step(list_statement);


	temp_refinement->refinement_id = sqlite3_column_int64(list_statement, 0);
	temp_refinement->refinement_package_asset_id = sqlite3_column_int64(list_statement, 1);
	temp_refinement->name = sqlite3_column_text(list_statement, 2);
	temp_refinement->refinement_was_imported_or_generated = sqlite3_column_int(list_statement, 3);
	temp_refinement->datetime_of_run.SetFromDOS((unsigned long) sqlite3_column_int64(list_statement, 4));
	temp_refinement->starting_refinement_id = sqlite3_column_int64(list_statement, 5);
	temp_refinement->number_of_particles = sqlite3_column_int64(list_statement, 6);
	temp_refinement->number_of_classes = sqlite3_column_int(list_statement, 7);
	temp_refinement->low_resolution_limit = sqlite3_column_double(list_statement, 8);
	temp_refinement->high_resolution_limit = sqlite3_column_double(list_statement, 9);
	temp_refinement->mask_radius = sqlite3_column_double(list_statement, 10);
	temp_refinement->signed_cc_resolution_limit = sqlite3_column_double(list_statement, 11);
	temp_refinement->global_resolution_limit = sqlite3_column_double(list_statement, 12);
	temp_refinement->global_mask_radius = sqlite3_column_double(list_statement, 13);
	temp_refinement->number_results_to_refine = sqlite3_column_int(list_statement, 14);
	temp_refinement->angular_search_step = sqlite3_column_double(list_statement, 15);
	temp_refinement->search_range_x = sqlite3_column_double(list_statement, 16);
	temp_refinement->search_range_y = sqlite3_column_double(list_statement, 17);
	temp_refinement->classification_resolution_limit = sqlite3_column_double(list_statement, 18);
	temp_refinement->should_focus_classify = sqlite3_column_int(list_statement, 19);
	temp_refinement->sphere_x_coord = sqlite3_column_double(list_statement, 20);
	temp_refinement->sphere_y_coord = sqlite3_column_double(list_statement, 21);
	temp_refinement->sphere_z_coord = sqlite3_column_double(list_statement, 22);
	temp_refinement->sphere_radius = sqlite3_column_double(list_statement, 23);
	temp_refinement->should_refine_ctf = sqlite3_column_int(list_statement, 24);
	temp_refinement->defocus_search_range = sqlite3_column_double(list_statement, 25);
	temp_refinement->defocus_search_step = sqlite3_column_double(list_statement, 26);
	temp_refinement->resolution_statistics_box_size = sqlite3_column_int(list_statement, 27);
	temp_refinement->resolution_statistics_pixel_size = sqlite3_column_double(list_statement, 28);

	sqlite3_finalize(list_statement);

	// volume_ids..


	// 3d references
	sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_REFERENCE_VOLUME_IDS_%li", temp_refinement->refinement_id);
	return_code = sqlite3_prepare_v2(sqlite_database, sql_select_command.ToUTF8().data(), sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
		temp_refinement->reference_volume_ids.Add(sqlite3_column_int64(list_statement, 1));
		return_code = sqlite3_step(list_statement);
	}

	MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code );
	sqlite3_finalize(list_statement);

	// now get all the parameters..

	temp_refinement->class_refinement_results.Alloc(temp_refinement->number_of_classes);
	temp_refinement->class_refinement_results.Add(junk_class_results, temp_refinement->number_of_classes);

	for (class_counter = 0; class_counter < temp_refinement->number_of_classes; class_counter++)
	{
		temp_refinement->class_refinement_results[class_counter].particle_refinement_results.Alloc(temp_refinement->number_of_particles);
		sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_RESULT_%li_%i", temp_refinement->refinement_id, class_counter + 1);

		more_data = BeginBatchSelect(sql_select_command);

		while (more_data == true)
		{
			more_data = GetFromBatchSelect("lssssssssssssss", &temp_result.position_in_stack,
					                                                                              &temp_result.psi,
																								  &temp_result.theta,
																								  &temp_result.phi,
																								  &temp_result.xshift,
																								  &temp_result.yshift,
																								  &temp_result.defocus1,
																								  &temp_result.defocus2,
																								  &temp_result.defocus_angle,
																								  &temp_result.phase_shift,
																								  &temp_result.occupancy,
																								  &temp_result.logp,
																								  &temp_result.sigma,
																								  &temp_result.score,
																								  &temp_result.score_change);

			temp_refinement->class_refinement_results[class_counter].particle_refinement_results.Add(temp_result);
		}

		EndBatchSelect();

	}

	// resolution statistics

	for (class_counter = 0; class_counter < temp_refinement->number_of_classes; class_counter++)
	{
		temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.Init(temp_refinement->resolution_statistics_pixel_size, temp_refinement->resolution_statistics_box_size);

		sql_select_command = wxString::Format("SELECT RESOLUTION, FSC, PART_FSC, PART_SSNR, REC_SSNR FROM REFINEMENT_RESOLUTION_STATISTICS_%li_%i", temp_refinement->refinement_id, class_counter + 1);
		more_data = BeginBatchSelect(sql_select_command);

		while (more_data == true)
		{
			more_data = GetFromBatchSelect("sssss", &temp_resolution, &temp_fsc, &temp_part_fsc, &temp_part_ssnr, &temp_rec_ssnr);

			temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.AddPoint(temp_resolution, temp_fsc);
			temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_FSC.AddPoint(temp_resolution, temp_part_fsc);
			temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.AddPoint(temp_resolution, temp_part_ssnr);
			temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.rec_SSNR.AddPoint(temp_resolution, temp_rec_ssnr);

		}


		EndBatchSelect();

	}


	return temp_refinement;

}

void Database::AddClassification(Classification *classification_to_add)
{
	MyDebugAssertTrue(classification_to_add->number_of_particles == classification_to_add->classification_results.GetCount(), "Number of results does not equal number of particles in this classification");
	long counter;

	InsertOrReplace("CLASSIFICATION_LIST", "Plttilllirrrrrrriir", "CLASSIFICATION_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "CLASS_AVERAGE_FILE", "REFINEMENT_WAS_IMPORTED_OR_GENERATED", "DATETIME_OF_RUN", "STARTING_CLASSIFICATION_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "SMOOTHING_FACTOR", "EXCLUDE_BLANK_EDGES", "AUTO_PERCENT_USED", "PERCENT_USED", classification_to_add->classification_id, classification_to_add->refinement_package_asset_id, classification_to_add->name.ToUTF8().data(), classification_to_add->class_average_file.ToUTF8().data(), classification_to_add->classification_was_imported_or_generated, classification_to_add->datetime_of_run.GetAsDOS(), classification_to_add->starting_classification_id, classification_to_add->number_of_particles, classification_to_add->number_of_classes, classification_to_add->low_resolution_limit, classification_to_add->high_resolution_limit, classification_to_add->mask_radius, classification_to_add->angular_search_step, classification_to_add->search_range_x, classification_to_add->search_range_y, classification_to_add->smoothing_factor, classification_to_add->exclude_blank_edges, classification_to_add->auto_percent_used, classification_to_add->percent_used);
	CreateClassificationResultTable(classification_to_add->classification_id);

	BeginBatchInsert(wxString::Format("CLASSIFICATION_RESULT_%li", classification_to_add->classification_id), 7 , "POSITION_IN_STACK", "PSI", "XSHIFT", "YSHIFT", "BEST_CLASS", "SIGMA", "LOGP");

	for ( counter = 0; counter < classification_to_add->classification_results.GetCount(); counter++)
	{
		AddToBatchInsert("lrrrirr", classification_to_add->classification_results[counter].position_in_stack, classification_to_add->classification_results[counter].psi, classification_to_add->classification_results[counter].xshift, classification_to_add->classification_results[counter].yshift, classification_to_add->classification_results[counter].best_class, classification_to_add->classification_results[counter].sigma, classification_to_add->classification_results[counter].logp);
	}

	EndBatchInsert();
}

Classification *Database::GetClassificationByID(long wanted_classification_id)
{
	wxString sql_select_command;
	int return_code;
	sqlite3_stmt *list_statement = NULL;
	Classification *temp_classification = new Classification;
	bool more_data;
	long records_retrieved = 0;

	ClassificationResult junk_result;

	// general data

	sql_select_command = wxString::Format("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=%li", wanted_classification_id);
	return_code = sqlite3_prepare_v2(sqlite_database, sql_select_command.ToUTF8().data(), sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\nSQL Command : %s\n", return_code , sql_select_command);
	return_code = sqlite3_step(list_statement);

	temp_classification->classification_id = sqlite3_column_int64(list_statement, 0);
	temp_classification->refinement_package_asset_id = sqlite3_column_int64(list_statement, 1);
	temp_classification->name = sqlite3_column_text(list_statement, 2);
	temp_classification->class_average_file = sqlite3_column_text(list_statement, 3);
	temp_classification->classification_was_imported_or_generated = sqlite3_column_int(list_statement, 4);
	temp_classification->datetime_of_run.SetFromDOS((unsigned long) sqlite3_column_int64(list_statement, 5));
	temp_classification->starting_classification_id = sqlite3_column_int64(list_statement, 6);
	temp_classification->number_of_particles = sqlite3_column_int64(list_statement, 7);
	temp_classification->number_of_classes = sqlite3_column_int(list_statement, 8);
	temp_classification->low_resolution_limit = sqlite3_column_double(list_statement, 9);
	temp_classification->high_resolution_limit = sqlite3_column_double(list_statement, 10);
	temp_classification->mask_radius = sqlite3_column_double(list_statement, 11);
	temp_classification->angular_search_step = sqlite3_column_double(list_statement, 12);
	temp_classification->search_range_x = sqlite3_column_double(list_statement, 13);
	temp_classification->search_range_y = sqlite3_column_double(list_statement, 14);
	temp_classification->smoothing_factor = sqlite3_column_double(list_statement, 15);
	temp_classification->exclude_blank_edges = sqlite3_column_int(list_statement, 16);
	temp_classification->auto_percent_used = sqlite3_column_int(list_statement, 17);
	temp_classification->percent_used = sqlite3_column_double(list_statement, 18);

	sqlite3_finalize(list_statement);

	// now get all the parameters..

	temp_classification->classification_results.Alloc(temp_classification->number_of_particles);

	sql_select_command = wxString::Format("SELECT * FROM CLASSIFICATION_RESULT_%li", temp_classification->classification_id);
	wxPrintf("Select command = %s\n", sql_select_command.ToUTF8().data());
	more_data = BeginBatchSelect(sql_select_command);

	while (more_data == true)
	{

		more_data = GetFromBatchSelect("lsssiss", 	&junk_result.position_in_stack,
													&junk_result.psi,
													&junk_result.xshift,
													&junk_result.yshift,
													&junk_result.best_class,
													&junk_result.sigma,
													&junk_result.logp);

		temp_classification->classification_results.Add(junk_result);
		records_retrieved++;

		wxPrintf("Got info for particle %li\n", junk_result.position_in_stack);
	}

	MyDebugAssertTrue(records_retrieved == temp_classification->number_of_particles, "No of Retrieved Results != No of Particles");

	EndBatchSelect();
	return temp_classification;
}

void Database::AddClassificationSelection(ClassificationSelection *classification_selection_to_add)
{
	InsertOrReplace("CLASSIFICATION_SELECTION_LIST", "ltlllii", "SELECTION_ID", "SELECTION_NAME", "CREATION_DATE", "REFINEMENT_PACKAGE_ID", "CLASSIFICATION_ID", "NUMBER_OF_CLASSES", "NUMBER_OF_SELECTIONS", classification_selection_to_add->selection_id, classification_selection_to_add->name.ToUTF8().data(), classification_selection_to_add->creation_date.GetAsDOS(), classification_selection_to_add->refinement_package_asset_id, classification_selection_to_add->classification_id, classification_selection_to_add->number_of_classes, classification_selection_to_add->number_of_selections);
	CreateClassificationSelectionTable(classification_selection_to_add->selection_id);

	BeginBatchInsert(wxString::Format("CLASSIFICATION_SELECTION_%li", classification_selection_to_add->selection_id), 1, "CLASS_AVERAGE_NUMBER");

	for (int counter = 0; counter < classification_selection_to_add->selections.GetCount(); counter++)
	{
		AddToBatchInsert("l", classification_selection_to_add->selections.Item(counter));
	}

	EndBatchInsert();
}

