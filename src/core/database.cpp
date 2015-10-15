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
    	MyPrintWithDetails("SQL Error: %s\n", error_message);
        sqlite3_free(error_message);
        return false;
    }

    return true;

}

bool Database::CreateAllTables()
{
	bool success;

	success = CreateTable("MASTER_SETTINGS", "pttiri", "NUMBER", "PROJECT_DIRECTORY", "PROJECT_NAME", "CURRENT_VERSION", "TOTAL_CPU_HOURS", "TOTAL_JOBS_RUN");
	success = CreateTable("RUNNING_JOBS", "pti", "JOB_NUMBER", "JOB_CODE", "MANAGER_IP_ADDRESS");
	success = CreateTable("RUN_PROFILES", "ptti", "RUN_PROFILE_ID", "PROFILE_NAME", "MANAGER_RUN_COMMAND", "COMMANDS_ID");
	success = CreateTable("MOVIE_ASSETS", "ptiiiirrrr", "MOVIE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION");
	success = CreateTable("MOVIE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );
	success = CreateTable("MOVIE_ALIGNMENT_LIST", "piiirrrrriiriiiii", "ALIGNMENT_NUMBER", "DATETIME_OF_RUN", "MOVIE_ASSET_ID", "ALIGNMENT_ID", "VOLTAGE", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER", "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS", "HORIZONTAL_MASK", "VERTICAL_MASK" );
	success = CreateTable("IMAGE_ASSETS", "ptiiiirrr", "IMAGE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION");
	success = CreateTable("IMAGE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );
	success = CreateTable("ESTIMATED_IMAGE_CTF_PARAMETERS", "piirrrrirrrrrrirrrrrr", "CTF_ESTIMATION_NUMBER", "DATETIME_OF_RUN", "IMAGE_ASSET_ID", "VOLTAGE", "SPHERICAL_ABERRATION", "PIXEL_SIZE", "AMPLITUDE_CONTRAST", "SPECTRUM_SIZE", "MIN_RESOLUTION", "MAX_RESOLUTION", "MIN_DEFOCUS", "MAX_DEFOCUS", "DEFOCUS_STEP", "TOLERATED_ASTIGMATISM", "FIND_ADDITIONAL_PHASE_SHIFT", "MIN_PHASE_SHIFT", "MAX_PHASE_SHIFT", "PHASE_SHIFT_STEP", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE");

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

	return_code = sqlite3_prepare_v2(sqlite_database, sql_command.ToUTF8().data(), strlen(sql_command.ToUTF8().data()) + 1, &sqlite_statement, NULL);
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

	sql_command = "BEGIN;";
	last_return_code = sqlite3_exec(sqlite_database, sql_command.ToUTF8().data(), NULL, 0, &error_message);


	if (last_return_code != SQLITE_OK)
	{
		MyPrintWithDetails("SQL Error : %s\n", error_message);
		sqlite3_free(error_message);
	}

	sql_command = "INSERT INTO ";
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
	BeginBatchInsert("IMAGE_ASSETS", 9, "IMAGE_ASSET_ID", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION");
}

void Database::AddNextImageAsset(int image_asset_id,  wxString filename, int position_in_stack, int parent_movie_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration)
{
	AddToBatchInsert("itiiiirrr", image_asset_id, filename.ToUTF8().data(), position_in_stack, parent_movie_id, x_size, y_size, pixel_size, voltage, spherical_aberration);
}

void Database::EndImageAssetInsert()
{
	EndBatchInsert();
}


void Database::BeginBatchSelect(const char *select_command)
{
	MyDebugAssertTrue(is_open == true, "database not open!");
	MyDebugAssertTrue(in_batch_insert == false, "Starting batch select but already in batch insert mode");
	MyDebugAssertTrue(in_batch_select == false, "Starting batch select but already in batch select mode");


	in_batch_select = true;
	int return_code;

	return_code = sqlite3_prepare_v2(sqlite_database, select_command, strlen(select_command) + 1, &batch_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	last_return_code = sqlite3_step(batch_statement);

	if (last_return_code != SQLITE_DONE && last_return_code != SQLITE_ROW)
	{
		MyPrintWithDetails("SQL Return Code: %i\n", last_return_code);
		//return false;
	}

	//return true;
}

void Database::GetFromBatchSelect(const char *column_format, ...)
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
		{
			MyPrintWithDetails("Error: Unknown format character!\n");
		}

		 ++column_format;
	}

	va_end(args);

	last_return_code = sqlite3_step(batch_statement);
	MyDebugAssertTrue(last_return_code == SQLITE_OK || last_return_code == SQLITE_ROW || last_return_code == SQLITE_DONE, "SQL error, return code : %i\n", last_return_code );

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

	GetFromBatchSelect("itti", &temp_profile.id, &temp_profile.name, &temp_profile.manager_command, &profile_table_number);

	// now we fill from the specific group table.

	profile_sql_select_command = wxString::Format("SELECT * FROM RUN_PROFILE_COMMANDS_%i", profile_table_number);

	return_code = sqlite3_prepare_v2(sqlite_database, profile_sql_select_command.ToUTF8().data(), profile_sql_select_command.Length() + 1, &list_statement, NULL);
	MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code );

	return_code = sqlite3_step(list_statement);

	while (  return_code == SQLITE_ROW)
	{
		temp_profile.AddCommand(sqlite3_column_text(list_statement, 1), sqlite3_column_int(list_statement, 2));
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

	GetFromBatchSelect("ifiiiirrr", &temp_asset.asset_id, &temp_asset.filename, &temp_asset.position_in_stack, &temp_asset.parent_id, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.pixel_size, &temp_asset.microscope_voltage,  &temp_asset.spherical_aberration);
	return temp_asset;
}


void Database::EndAllMovieGroupsSelect()
{
	EndBatchSelect();
}

void Database::EndAllMovieAssetsSelect()
{
	EndBatchSelect();

}

void Database::EndAllImageGroupsSelect()
{
	EndBatchSelect();
}

void Database::EndAllImageAssetsSelect()
{
	EndBatchSelect();

}

void Database::EndAllRunProfilesSelect()
{
	EndBatchSelect();

}

bool Database::AddOrReplaceRunProfile(RunProfile *profile_to_add)
{

	InsertOrReplace("RUN_PROFILES", "ptti", "RUN_PROFILE_ID", "PROFILE_NAME", "MANAGER_RUN_COMMAND", "COMMANDS_ID", profile_to_add->id, profile_to_add->name.ToUTF8().data(), profile_to_add->manager_command.ToUTF8().data(), profile_to_add->id);
	DeleteTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id));
	CreateTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id), "pti", "COMMANDS_NUMBER", "COMMAND_STRING", "NUMBER_OF_COPIES");

	for (int counter = 0; counter < profile_to_add->number_of_run_commands; counter++)
	{
		InsertOrReplace(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id), "pti", "COMMANDS_NUMBER", "COMMAND_STRING", "NUMBER_OF_COPIES", counter, profile_to_add->run_commands[counter].command_to_run.ToUTF8().data(), profile_to_add->run_commands[counter].number_of_copies);
	}
}

bool Database::DeleteRunProfile(int wanted_id)
{
	ExecuteSQL(wxString::Format("DELETE FROM RUN_PROFILES WHERE RUN_PROFILE_ID=%i", wanted_id).ToUTF8().data());
	DeleteTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", wanted_id));
}

