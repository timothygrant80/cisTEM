#include "core_headers.h"

Database::Database()
{
	is_open = false;
	sqlite_database = NULL;
}

Database::~Database()
{
	if (is_open == true)
	{
		sqlite3_close(sqlite_database);
	}
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
		if (*column_format == 'i') // integer
		{
			sql_command += wxString::Format("%i",  va_arg(args, int));
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

bool Database::GetMasterSettings(wxFileName &project_directory, wxString &project_name, int &imported_integer_version, double &total_cpu_hours, int &total_jobs_run)
{
	MyDebugAssertTrue(is_open == true, "database not open!");

	sqlite3_stmt *sqlite_statement;
	int return_code;
	wxString sql_command = "select * from MASTER_SETTINGS;";

	sqlite3_prepare_v2(sqlite_database, sql_command.ToUTF8().data(), strlen(sql_command.ToUTF8().data()) + 1, &sqlite_statement, NULL);
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

	sqlite3_free(sqlite_statement);
	return true;
}

void Database::Close()
{
	if (is_open == true)
	{
		sqlite3_close(sqlite_database);
	}

	is_open = false;
	sqlite_database = NULL;

}
