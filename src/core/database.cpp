#include "core_headers.h"
#include "database_schema.h"

Database::Database( ) {
    last_return_code = -1;
    is_open          = false;
    sqlite_database  = NULL;

    in_batch_insert = false;
    in_batch_select = false;
    batch_statement = NULL;

    should_do_local_commit = false;

    number_of_active_transactions = 0;

    sqlite3_config(SQLITE_CONFIG_SINGLETHREAD); // we only need access from a single thread right now, and this should be a little faster as it avoids mutex checks;
}

Database::~Database( ) {
    Close( );
}

int Database::ExecuteSQL(const char* command) {
    char* error_message = NULL;
    int   return_code   = sqlite3_exec(sqlite_database, command, NULL, 0, &error_message);

    if ( return_code != SQLITE_OK ) {
        MyPrintWithDetails("SQL Error: %s\nTrying to execute the following command :-\n\n%s\n", error_message, command);
        sqlite3_free(error_message);

        if ( return_code != SQLITE_LOCKED )
            DEBUG_ABORT;
    }

    return return_code;
}

int Database::Prepare(wxString sql_command, sqlite3_stmt** current_statement) {
    int return_code;
    return_code = sqlite3_prepare_v2(sqlite_database, sql_command.ToUTF8( ).data( ), sql_command.Length( ) + 1, current_statement, NULL);

    if ( return_code != SQLITE_OK ) {
        MyPrintWithDetails("SQL Error: %s\nTrying to execute the following command :-\n\n%s\n", sqlite3_errmsg(sqlite_database), sql_command);
        if ( return_code != SQLITE_LOCKED )
            DEBUG_ABORT;
    }

    return return_code;
}

int Database::Step(sqlite3_stmt* current_statement) {
    int return_code;
    return_code = sqlite3_step(current_statement);

    if ( return_code != SQLITE_DONE && return_code != SQLITE_ROW ) {
        MyPrintWithDetails("SQL Error: %s\n\n", sqlite3_errstr(return_code));
        if ( return_code != SQLITE_LOCKED )
            DEBUG_ABORT;
    }

    return return_code;
}

int Database::Finalize(sqlite3_stmt* current_statement) {
    int return_code;
    return_code = sqlite3_finalize(current_statement);

    if ( return_code != SQLITE_OK ) {
        MyPrintWithDetails("SQL Error: %s\n\n", sqlite3_errmsg(sqlite_database));
        if ( return_code != SQLITE_LOCKED )
            DEBUG_ABORT;
    }

    return return_code;
}

void Database::CheckBindCode(int return_code) {
    if ( return_code != SQLITE_OK ) {
        MyPrintWithDetails("SQL Error: %s\n\n", sqlite3_errmsg(sqlite_database));
        if ( return_code != SQLITE_LOCKED )
            DEBUG_ABORT;
    }
}

int Database::ReturnSingleIntFromSelectCommand(wxString select_command) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    int           value;

    Prepare(select_command, &current_statement);
    return_code = Step(current_statement);
    value       = sqlite3_column_int(current_statement, 0);

    Finalize(current_statement);

    return value;
}

wxArrayInt Database::ReturnIntArrayFromSelectCommand(wxString select_command) {
    wxArrayInt ints_to_return;

    MyDebugAssertTrue(is_open == true, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    int           current_value;

    Prepare(select_command, &current_statement);
    //Step(current_statement);

    return_code = Step(current_statement);

    while ( return_code == SQLITE_ROW ) {
        current_value = sqlite3_column_int(current_statement, 0);
        ints_to_return.Add(current_value);
        return_code = Step(current_statement);
    }

    Finalize(current_statement);

    return ints_to_return;
}

wxArrayLong Database::ReturnLongArrayFromSelectCommand(wxString select_command) {
    wxArrayLong longs_to_return;

    MyDebugAssertTrue(is_open == true, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    long          current_value;

    Prepare(select_command, &current_statement);
    //Step(current_statement);

    return_code = Step(current_statement);

    while ( return_code == SQLITE_ROW ) {
        current_value = sqlite3_column_int64(current_statement, 0);
        longs_to_return.Add(current_value);
        return_code = Step(current_statement);
    }

    Finalize(current_statement);

    return longs_to_return;
}

long Database::ReturnRefinementIDGivenReconstructionID(long reconstruction_id) {
    return ReturnSingleLongFromSelectCommand(wxString::Format("SELECT REFINEMENT_ID FROM RECONSTRUCTION_LIST WHERE RECONSTRUCTION_ID=%li", reconstruction_id));
}

wxArrayString Database::ReturnStringArrayFromSelectCommand(wxString select_command) {
    wxArrayString strings_to_return;

    MyDebugAssertTrue(is_open == true, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    wxString      current_value;

    Prepare(select_command, &current_statement);

    return_code = Step(current_statement);

    while ( return_code == SQLITE_ROW ) {
        current_value = sqlite3_column_text(current_statement, 0);
        strings_to_return.Add(current_value);
        return_code = Step(current_statement);
    }

    Finalize(current_statement);

    return strings_to_return;
}

long Database::ReturnSingleLongFromSelectCommand(wxString select_command) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    long          value;

    Prepare(select_command, &current_statement);
    Step(current_statement);

    value = sqlite3_column_int64(current_statement, 0);

    Finalize(current_statement);

    return value;
}

void Database::GetActiveDefocusValuesByImageID(long wanted_image_id, float& defocus_1, float& defocus_2, float& defocus_angle, float& phase_shift, float& amplitude_contrast, float& tilt_angle, float& tilt_axis) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    int           value;
    wxString      select_command = wxString::Format("SELECT DEFOCUS1, DEFOCUS2, DEFOCUS_ANGLE, ADDITIONAL_PHASE_SHIFT, AMPLITUDE_CONTRAST, TILT_ANGLE, TILT_AXIS FROM ESTIMATED_CTF_PARAMETERS, IMAGE_ASSETS WHERE ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_ID=IMAGE_ASSETS.CTF_ESTIMATION_ID AND IMAGE_ASSETS.IMAGE_ASSET_ID=%li;", wanted_image_id);

    Prepare(select_command, &current_statement);
    Step(current_statement);

    defocus_1          = float(sqlite3_column_double(current_statement, 0));
    defocus_2          = float(sqlite3_column_double(current_statement, 1));
    defocus_angle      = float(sqlite3_column_double(current_statement, 2));
    phase_shift        = float(sqlite3_column_double(current_statement, 3));
    amplitude_contrast = float(sqlite3_column_double(current_statement, 4));
    tilt_angle         = float(sqlite3_column_double(current_statement, 5));
    tilt_axis          = float(sqlite3_column_double(current_statement, 6));

    Finalize(current_statement);
}

double Database::ReturnSingleDoubleFromSelectCommand(wxString select_command) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    double        value;

    Prepare(select_command, &current_statement);
    Step(current_statement);

    value = sqlite3_column_double(current_statement, 0);

    Finalize(current_statement);

    return value;
}

long Database::ReturnHighestRefinementID( ) {
    return ReturnSingleLongFromSelectCommand("SELECT MAX(REFINEMENT_ID) FROM REFINEMENT_LIST");
}

long Database::ReturnHighestStartupID( ) {
    return ReturnSingleLongFromSelectCommand("SELECT MAX(STARTUP_ID) FROM STARTUP_LIST");
}

long Database::ReturnHighestReconstructionID( ) {
    return ReturnSingleLongFromSelectCommand("SELECT MAX(RECONSTRUCTION_ID) FROM RECONSTRUCTION_LIST");
}

long Database::ReturnHighestClassificationID( ) {
    return ReturnSingleLongFromSelectCommand("SELECT MAX(CLASSIFICATION_ID) FROM CLASSIFICATION_LIST");
}

int Database::ReturnHighestAlignmentID( ) {
    return ReturnSingleIntFromSelectCommand("SELECT MAX(ALIGNMENT_ID) FROM MOVIE_ALIGNMENT_LIST");
}

int Database::ReturnHighestTemplateMatchID( ) {
    return ReturnSingleIntFromSelectCommand("SELECT MAX(TEMPLATE_MATCH_ID) FROM TEMPLATE_MATCH_LIST");
}

int Database::ReturnHighestFindCTFID( ) {
    return ReturnSingleIntFromSelectCommand("SELECT MAX(CTF_ESTIMATION_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

int Database::ReturnHighestPickingID( ) {
    // We return 0 if the particle_picking_list is empty
    return ReturnSingleIntFromSelectCommand("SELECT COALESCE(MAX(PICKING_ID),0) FROM PARTICLE_PICKING_LIST");
}

int Database::ReturnHighestParticlePositionID( ) {
    // Note: we can't just look for the maximum position_id in the latest picking results table, since the user is free to add new particle positions
    // to old results tables
    int      number_of_picking_jobs     = ReturnNumberOfPickingJobs( );
    int      max_position_id            = -1;
    int      current_max_position_id    = -1;
    int      number_of_non_empty_tables = 0;
    wxString sql_query                  = "select max(";
    for ( int counter = 1; counter <= number_of_picking_jobs; counter++ ) {
        if ( ReturnSingleIntFromSelectCommand(wxString::Format("select count(*) from particle_picking_results_%i", counter)) > 0 ) {
            if ( number_of_non_empty_tables > 0 ) {
                sql_query += ",";
            }
            number_of_non_empty_tables++;
            sql_query += wxString::Format("(select max(position_id) from particle_picking_results_%i)", counter);
        }
    }
    sql_query += ")";
    if ( number_of_non_empty_tables > 0 ) {
        return ReturnSingleIntFromSelectCommand(sql_query);
    }
    else {
        return 0;
    }
}

int Database::ReturnNumberOfPreviousMovieAlignmentsByAssetID(int wanted_asset_id) {
    return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM MOVIE_ALIGNMENT_LIST WHERE MOVIE_ASSET_ID = %i", wanted_asset_id));
}

int Database::ReturnNumberOfPreviousTemplateMatchesByAssetID(int wanted_asset_id) {
    return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM TEMPLATE_MATCH_LIST WHERE IMAGE_ASSET_ID = %i", wanted_asset_id));
}

int Database::ReturnNumberOfPreviousCTFEstimationsByAssetID(int wanted_asset_id) {
    return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID = %i", wanted_asset_id));
}

int Database::ReturnNumberOfPreviousParticlePicksByAssetID(int wanted_asset_id) {
    return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM PARTICLE_PICKING_LIST WHERE PARENT_IMAGE_ASSET_ID = %i", wanted_asset_id));
}

int Database::ReturnHighestAlignmentJobID( ) {
    return ReturnSingleIntFromSelectCommand("SELECT MAX(ALIGNMENT_JOB_ID) FROM MOVIE_ALIGNMENT_LIST");
}

int Database::ReturnHighestTemplateMatchJobID( ) {
    return ReturnSingleIntFromSelectCommand("SELECT MAX(TEMPLATE_MATCH_JOB_ID) FROM TEMPLATE_MATCH_LIST");
}

int Database::ReturnHighestFindCTFJobID( ) {
    return ReturnSingleIntFromSelectCommand("SELECT MAX(CTF_ESTIMATION_JOB_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

int Database::ReturnHighestPickingJobID( ) {
    return ReturnSingleIntFromSelectCommand("SELECT MAX(PICKING_JOB_ID) FROM PARTICLE_PICKING_LIST");
}

long Database::ReturnHighestClassumSelectionID( ) {
    return ReturnSingleLongFromSelectCommand("SELECT MAX(SELECTION_ID) FROM CLASSIFICATION_SELECTION_LIST");
}

int Database::ReturnNumberOfAlignmentJobs( ) {
    return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT ALIGNMENT_JOB_ID) FROM MOVIE_ALIGNMENT_LIST");
}

int Database::ReturnNumberOfCTFEstimationJobs( ) {
    return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT CTF_ESTIMATION_JOB_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

int Database::ReturnNumberOfTemplateMatchingJobs( ) {
    return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT TEMPLATE_MATCH_JOB_ID) FROM TEMPLATE_MATCH_LIST");
}

int Database::ReturnNumberOfPickingJobs( ) {
    return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT PICKING_JOB_ID) FROM PARTICLE_PICKING_LIST");
}

void Database::GetUniqueAlignmentIDs(int* alignment_job_ids, int number_of_alignmnet_jobs) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    bool more_data;

    more_data = BeginBatchSelect("SELECT DISTINCT ALIGNMENT_JOB_ID FROM MOVIE_ALIGNMENT_LIST") == true;

    for ( int counter = 0; counter < number_of_alignmnet_jobs; counter++ ) {
        if ( more_data == false ) {
            MyPrintWithDetails("Unexpected end of select command");
            DEBUG_ABORT;
        }

        more_data = GetFromBatchSelect("i", &alignment_job_ids[counter]);
    }

    EndBatchSelect( );
}

void Database::GetUniquePickingJobIDs(int* picking_job_ids, int number_of_picking_jobs) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    bool more_data;

    more_data = BeginBatchSelect("SELECT DISTINCT PICKING_JOB_ID FROM PARTICLE_PICKING_LIST") == true;

    for ( int counter = 0; counter < number_of_picking_jobs; counter++ ) {
        if ( more_data == false ) {
            MyPrintWithDetails("Unexpected end of select command");
            DEBUG_ABORT;
        }

        more_data = GetFromBatchSelect("i", &picking_job_ids[counter]);
    }

    EndBatchSelect( );
}

void Database::GetUniqueCTFEstimationIDs(int* ctf_estimation_job_ids, int number_of_ctf_estimation_jobs) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    bool more_data;

    more_data = BeginBatchSelect("SELECT DISTINCT CTF_ESTIMATION_JOB_ID FROM ESTIMATED_CTF_PARAMETERS") == true;

    for ( int counter = 0; counter < number_of_ctf_estimation_jobs; counter++ ) {
        if ( more_data == false ) {
            MyPrintWithDetails("Unexpected end of select command");
            DEBUG_ABORT;
        }

        more_data = GetFromBatchSelect("i", &ctf_estimation_job_ids[counter]);
    }

    EndBatchSelect( );
}

void Database::GetUniqueTemplateMatchIDs(std::vector<long>& template_match_job_ids, int number_of_template_match_jobs) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    bool more_data;

    more_data = BeginBatchSelect("SELECT DISTINCT TEMPLATE_MATCH_JOB_ID FROM TEMPLATE_MATCH_LIST") == true;

    for ( int counter = 0; counter < number_of_template_match_jobs; counter++ ) {
        // FIXME: This is a weird mix. Shouldn't it be either a debug assert, or a runtime assert?
        // Here we always check and print the message, but only abort in debug mode.
        if ( more_data == false ) {
            MyPrintWithDetails("Unexpected end of select command");
            DEBUG_ABORT;
        }
        // Add in a dummy value so the containers size is correct.
        template_match_job_ids.emplace_back(0);
        more_data = GetFromBatchSelect_NoChar(&template_match_job_ids[counter]);
    }

    EndBatchSelect( );
}

int Database::ReturnNumberOfImageAssetsWithCTFEstimates( ) {
    return ReturnSingleIntFromSelectCommand("SELECT COUNT(DISTINCT IMAGE_ASSET_ID) FROM ESTIMATED_CTF_PARAMETERS");
}

void Database::GetUniqueIDsOfImagesWithCTFEstimations(int* image_ids, int& number_of_image_ids) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    bool more_data;

    more_data = BeginBatchSelect("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");

    for ( int counter = 0; counter < number_of_image_ids; counter++ ) {
        if ( more_data == false ) {
            MyPrintWithDetails("Unexpected end of select command");
            DEBUG_ABORT;
        }

        more_data = GetFromBatchSelect("i", &image_ids[counter]);
    }

    EndBatchSelect( );
}

void Database::GetCTFParameters(const int& ctf_estimation_id, double& acceleration_voltage, double& spherical_aberration, double& amplitude_constrast, double& defocus_1, double& defocus_2, double& defocus_angle, double& additional_phase_shift, double& iciness) {
    MyDebugAssertTrue(is_open, "Database not open");

    bool more_data;

    more_data = BeginBatchSelect(wxString::Format("SELECT VOLTAGE, SPHERICAL_ABERRATION, AMPLITUDE_CONTRAST, DEFOCUS1, DEFOCUS2, DEFOCUS_ANGLE, ADDITIONAL_PHASE_SHIFT, ICINESS FROM ESTIMATED_CTF_PARAMETERS WHERE CTF_ESTIMATION_ID=%i", ctf_estimation_id));

    if ( more_data ) {
        GetFromBatchSelect("rrrrrrrr", &acceleration_voltage, &spherical_aberration, &amplitude_constrast, &defocus_1, &defocus_2, &defocus_angle, &additional_phase_shift, &iciness);
    }
    else {
        MyPrintWithDetails("Unexpected end of select command\n");
        DEBUG_ABORT;
    }

    EndBatchSelect( );
}

void Database::AddCTFIcinessColumnIfNecessary( ) {
    MyDebugAssertTrue(is_open, "Database not open");

    if ( DoesColumnExist("ESTIMATED_CTF_PARAMETERS", "ICINESS") ) {
        MyDebugPrint("Iciness column exists in estimated_ctf_parameters\n");
    }
    else {
        MyDebugPrint("Need to create iciness column in estimated_ctf_parameters\n");
        AddColumnToTable("ESTIMATED_CTF_PARAMETERS", "ICINESS", "r", "0.0");
    }
}

bool Database::CreateNewDatabase(wxFileName wanted_database_file) {
    int return_code;

    // is project already open?

    if ( is_open == true ) {
        MyPrintWithDetails("Attempting to create a new database, but there is already an open project");
        return false;
    }

    // does the database file exist?

    if ( wanted_database_file.Exists( ) ) {
        MyPrintWithDetails("Attempting to create a new database, but the file already exists");
        return false;
    }

    // make the path absolute..

    wanted_database_file.MakeAbsolute( );

    return_code = sqlite3_open_v2(wanted_database_file.GetFullPath( ).ToUTF8( ).data( ), &sqlite_database, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, "unix-dotfile");

    if ( return_code ) {
        MyPrintWithDetails("Can't open database: %s\n%s\n", database_file.GetFullPath( ).ToUTF8( ).data( ), sqlite3_errmsg(sqlite_database));
        return false;
    }

    //ExecuteSQL("PRAGMA main.locking_mode=EXCLUSIVE;");
    //ExecuteSQL("PRAGMA main.temp_store=MEMORY;");
    //ExecuteSQL("PRAGMA main.synchronous=NORMAL;");
    //ExecuteSQL("PRAGMA main.page_size=4096;");
    //ExecuteSQL("PRAGMA main.cache_size=10000;");
    //ExecuteSQL("PRAGMA main.journal_mode=WAL;");

    // if we get here, we should have the database open, with an exclusive lock

    database_file = wanted_database_file;
    is_open       = true;

    return true;
}

bool Database::Open(wxFileName file_to_open, bool disable_locking) {
    int return_code;

    // is project already open?

    if ( is_open == true ) {
        MyPrintWithDetails("Attempting to open a database, but there is already an open project");
        return false;
    }

    // does the database file exist?

    if ( file_to_open.Exists( ) == false ) {
        MyPrintWithDetails("Attempting to open a new database, but the file does not exist");
        return false;
    }

    if ( disable_locking == true )
        return_code = sqlite3_open_v2(file_to_open.GetFullPath( ).ToUTF8( ).data( ), &sqlite_database, SQLITE_OPEN_READWRITE, "unix-none");
    else
        return_code = sqlite3_open_v2(file_to_open.GetFullPath( ).ToUTF8( ).data( ), &sqlite_database, SQLITE_OPEN_READWRITE, "unix-dotfile");

    if ( return_code ) {
        MyPrintWithDetails("Can't open database: %s\n%s\n", database_file.GetFullPath( ).ToUTF8( ).data( ), sqlite3_errmsg(sqlite_database));
        return false;
    }

    //	ExecuteSQL("PRAGMA main.locking_mode=EXCLUSIVE;");
    //	ExecuteSQL("PRAGMA main.temp_store=MEMORY;");
    //	ExecuteSQL("PRAGMA main.synchronous=NORMAL;");
    //	ExecuteSQL("PRAGMA main.page_size=4096;");
    //	ExecuteSQL("PRAGMA main.cache_size=10000;");
    //	ExecuteSQL("PRAGMA main.journal_mode=WAL;");

    // if we get here, we should have the database open, with an exclusive lock

    database_file = file_to_open;
    is_open       = true;

    // We used to check here whether all tables exists, this has been moved now
    // to the GUI upon project opening
    // CreateAllTables();

    return true;
}

/**
 * @brief Make backup copy of existing database. Currently only in use when
 * database schema changes are detected and a schema update is necessary.
 * 
 * @param backup_db Filename of backup database.
 * @return true If all backup is created successfully.
 * @return false If backup fails.
 */
bool Database::CopyDatabase(wxFileName backup_db) {
    sqlite3*        destination;
    sqlite3_backup* backup;
    int             return_code;
    bool            must_open_source_db; // source database should already be open; we will check to be safe

    // Check if source database is already open (it should be); if not, open it
    must_open_source_db = ! (this->is_open);
    if ( must_open_source_db ) {
        return_code = sqlite3_open_v2(this->database_file.GetFullPath( ).ToUTF8( ).data( ), &this->sqlite_database, SQLITE_OPEN_READWRITE, "unix-dotfile");
        if ( return_code != SQLITE_OK ) {
            MyPrintWithDetails("Cannot open source database: %s\n", sqlite3_errmsg(this->sqlite_database));
            sqlite3_close(this->sqlite_database);
            return false;
        }
    }

    // Open backup database
    return_code = sqlite3_open_v2(backup_db.GetFullPath( ).ToUTF8( ).data( ), &destination, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, "unix-dotfile");
    if ( return_code != SQLITE_OK ) {
        MyPrintWithDetails("Cannot open destination database: %s\n", sqlite3_errmsg(destination));
        sqlite3_close(destination);
        return false;
    }

    // Get backup database ready for backing up
    backup = sqlite3_backup_init(destination, "main", this->sqlite_database, "main");
    if ( ! backup ) {
        MyPrintWithDetails("Backup failed: %s\n", sqlite3_errmsg(destination));
        if ( must_open_source_db )
            sqlite3_close(this->sqlite_database);
        sqlite3_close(destination);
        return false;
    }

    // Perform backup
    wxPrintf("Backing up...\n");
    while ( return_code == SQLITE_OK ) {
        return_code = sqlite3_backup_step(backup, 5);
    }

    sqlite3_backup_finish(backup);

    // Ensure backup finished properly
    if ( return_code != SQLITE_DONE ) {
        MyPrintWithDetails("Backup failed %s\n", sqlite3_errmsg(destination));
        return false;
    }
    else {
        wxPrintf("Backup completed successfully\n");
    }

    if ( must_open_source_db )
        sqlite3_close(this->sqlite_database);
    sqlite3_close(destination);
    return true;
}

bool Database::DeleteTable(const char* table_name) {
    wxString sql_command = "DROP TABLE IF EXISTS ";
    sql_command += table_name;

    return ExecuteSQL(sql_command.ToUTF8( ).data( ));
}

bool Database::AddColumnToTable(wxString table_name, wxString column_name, wxString column_format, wxString default_value) {
    wxString sql_command = "ALTER TABLE " + table_name + " ADD COLUMN " + column_name + " ";
    if ( column_format.IsSameAs("t", false) ) {
        sql_command += " TEXT";
    }
    else if ( column_format.IsSameAs("r", false) ) {
        sql_command += " REAL";
    }
    else if ( column_format.IsSameAs("i", false) || column_format.IsSameAs("l", false) ) {
        sql_command += " INTEGER";
    }

    sql_command += " DEFAULT " + default_value;

    int return_code = ExecuteSQL(sql_command);

    if ( return_code != SQLITE_OK ) {
        DEBUG_ABORT;
    }

    return true;
}

bool Database::CreateTable(const char* table_name, const char* column_format, ...) {
    int   return_code;
    char* error_message = NULL;

    wxString sql_command;

    int number_of_columns = strlen(column_format);

    int current_column = 0;

    sql_command = "CREATE TABLE IF NOT EXISTS ";
    sql_command += table_name;
    sql_command += "(";

    va_list args;
    va_start(args, column_format);

    while ( *column_format != '\0' ) {
        current_column++;

        if ( *column_format == 't' ) // text
        {
            sql_command += va_arg(args, const char*);
            sql_command += " TEXT";
        }
        else if ( *column_format == 'r' ) // real
        {
            sql_command += va_arg(args, const char*);
            sql_command += " REAL";
        }
        else if ( *column_format == 'i' ) // integer
        {
            sql_command += va_arg(args, const char*);
            sql_command += " INTEGER";
        }
        else if ( *column_format == 'l' ) // integer
        {
            sql_command += va_arg(args, const char*);
            sql_command += " INTEGER";
        }
        else if ( *column_format == 'p' || *column_format == 'P' ) // integer
        {
            sql_command += va_arg(args, const char*);
            sql_command += " INTEGER PRIMARY KEY";
        }
        else {
            MyPrintWithDetails("Error: Unknown format character!\n");
        }

        if ( current_column < number_of_columns )
            sql_command += ", ";
        else
            sql_command += " );";

        ++column_format;
    }

    va_end(args);

    return_code = ExecuteSQL(sql_command.ToUTF8( ).data( ));

    if ( return_code != SQLITE_OK ) {
        DEBUG_ABORT;
    }

    return true;
}

bool Database::CreateTable(const char* table_name, const char* column_format, std::vector<wxString> columns) {
    int return_code;
    int col_counter;

    wxString sql_command;

    int number_of_columns = strlen(column_format);
    MyDebugAssertTrue(number_of_columns == columns.size( ), "Column formt string length unequal to number of columns\n");

    sql_command = "CREATE TABLE IF NOT EXISTS ";
    sql_command += table_name;
    sql_command += "(";

    for ( col_counter = 0; col_counter < columns.size( ); col_counter++ ) {
        sql_command += columns[col_counter];
        sql_command += map_type_char_to_sqlite_string(column_format[col_counter]);

        if ( col_counter < number_of_columns - 1 )
            sql_command += ", ";
        else
            sql_command += " );";
    }

    return_code = ExecuteSQL(sql_command.ToUTF8( ).data( ));

    if ( return_code != SQLITE_OK ) {
        DEBUG_ABORT;
    }

    return true;
}

bool Database::CreateAllTables( ) {
    using namespace database_schema;
    bool success;

    BeginCommitLocker active_locker(this);

    for ( TableData& table : static_tables ) {
        success = CreateTable(std::get<TABLE_NAME>(table), std::get<TABLE_TYPES>(table), std::get<TABLE_COLUMNS>(table));
        CheckSuccess(success);
    }

    return success;
}

bool Database::InsertOrReplace(const char* table_name, const char* column_format, ...) {
    int number_of_columns = strlen(column_format);

    int   current_column = 0;
    int   return_code;
    char* error_message = NULL;

    wxString sql_command;
    wxString temp_string;

    sql_command = "INSERT OR REPLACE INTO  ";
    sql_command += table_name;
    sql_command += "(";

    va_list args;
    va_start(args, column_format);

    for ( current_column = 1; current_column <= number_of_columns; current_column++ ) {
        sql_command += va_arg(args, const char*);

        if ( current_column == number_of_columns )
            sql_command += " ) ";
        else
            sql_command += ", ";
    }

    sql_command += "VALUES (";
    current_column = 0;

    while ( *column_format != '\0' ) {

        current_column++;

        if ( *column_format == 't' ) // text
        {
            sql_command += "'";
            temp_string = va_arg(args, const char*);
            //escape apostrophes
            temp_string.Replace("'", "''");
            sql_command += temp_string;
            sql_command += "'";
        }
        else if ( *column_format == 'r' ) // real
        {
            sql_command += wxString::Format("%f", va_arg(args, double));
        }
        else if ( *column_format == 'i' || *column_format == 'p' ) // integer
        {
            sql_command += wxString::Format("%i", va_arg(args, int));
        }
        else if ( *column_format == 'l' || *column_format == 'P' ) // long
        {
            sql_command += wxString::Format("%li", va_arg(args, long));
        }
        else {
            MyPrintWithDetails("Error: Unknown format character!\n");
            DEBUG_ABORT;
        }

        if ( current_column < number_of_columns )
            sql_command += ", ";
        else
            sql_command += " );";

        ++column_format;
    }

    va_end(args);

    // escape apostrophes;
    return_code = ExecuteSQL(sql_command.ToUTF8( ).data( ));

    if ( return_code != SQLITE_OK ) {
        DEBUG_ABORT;
    }

    return true;
}

bool Database::GetMasterSettings(wxFileName& project_directory, wxString& project_name, int& imported_integer_version, double& total_cpu_hours, int& total_jobs_run, wxString& cistem_version_text, cistem::workflow::Enum& current_workflow) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    sqlite3_stmt* sqlite_statement;
    wxString      sql_command = "select * from MASTER_SETTINGS;";

    Prepare(sql_command, &sqlite_statement);
    Step(sqlite_statement);

    project_directory        = sqlite3_column_text(sqlite_statement, 1);
    project_name             = sqlite3_column_text(sqlite_statement, 2);
    imported_integer_version = sqlite3_column_int(sqlite_statement, 3);
    total_cpu_hours          = sqlite3_column_double(sqlite_statement, 4);
    total_jobs_run           = sqlite3_column_int(sqlite_statement, 5);
    cistem_version_text      = sqlite3_column_text(sqlite_statement, 6);
    current_workflow         = static_cast<cistem::workflow::Enum>(sqlite3_column_int(sqlite_statement, 7));

    Finalize(sqlite_statement);
    return true;
}

bool Database::SetProjectStatistics(double& total_cpu_hours, int& total_jobs_run) {
    MyDebugAssertTrue(is_open == true, "database not open!");
    MyDebugAssertTrue(total_cpu_hours >= 0.0, "Oops, negative total number of CPU hours: %f", total_cpu_hours);

    ExecuteSQL(wxString::Format("UPDATE MASTER_SETTINGS SET TOTAL_JOBS_RUN = %i, TOTAL_CPU_HOURS = %f, CISTEM_VERSION_TEXT = '%s'", total_jobs_run, float(total_cpu_hours), CISTEM_VERSION_TEXT));
    return true;
}

bool Database::DoesTableExist(wxString table_name) {
    MyDebugAssertTrue(is_open == true, "database not open!");
    return (bool)ReturnSingleIntFromSelectCommand(wxString::Format("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='%s';", table_name));
}

bool Database::DoesColumnExist(wxString table_name, wxString column_name) {
    MyDebugAssertTrue(is_open, "database not open!");

    int           return_code;
    sqlite3_stmt* current_statement;
    wxString      sql_command = "SELECT " + column_name + " FROM " + table_name + " LIMIT 0";

    return_code = sqlite3_prepare_v2(sqlite_database, sql_command.ToUTF8( ).data( ), sql_command.Length( ) + 1, &current_statement, NULL);

    Finalize(current_statement);

    return return_code == SQLITE_OK;
}

void Database::GetMovieImportDefaults(float& voltage, float& spherical_aberration, float& pixel_size, float& exposure_per_frame, bool& movies_are_gain_corrected, wxString& gain_reference_filename, bool& movies_are_dark_corrected, wxString dark_reference_filename, bool& resample_movies, float& desired_pixel_size, bool& correct_mag_distortion, float& mag_distortion_angle, float& mag_distortion_major_scale, float& mag_distortion_minor_scale, bool& protein_is_white, int& eer_super_res_factor, int& eer_frames_per_image) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    sqlite3_stmt* sqlite_statement;
    int           return_code;
    wxString      sql_command = "select * FROM MOVIE_IMPORT_DEFAULTS;";

    Prepare(sql_command, &sqlite_statement);
    Step(sqlite_statement);

    voltage                    = sqlite3_column_double(sqlite_statement, 1);
    spherical_aberration       = sqlite3_column_double(sqlite_statement, 2);
    pixel_size                 = sqlite3_column_double(sqlite_statement, 3);
    exposure_per_frame         = sqlite3_column_double(sqlite_statement, 4);
    movies_are_gain_corrected  = sqlite3_column_int(sqlite_statement, 5);
    gain_reference_filename    = sqlite3_column_text(sqlite_statement, 6);
    movies_are_dark_corrected  = sqlite3_column_int(sqlite_statement, 7);
    dark_reference_filename    = sqlite3_column_text(sqlite_statement, 8);
    resample_movies            = sqlite3_column_int(sqlite_statement, 9);
    desired_pixel_size         = sqlite3_column_double(sqlite_statement, 10);
    correct_mag_distortion     = sqlite3_column_int(sqlite_statement, 11);
    mag_distortion_angle       = sqlite3_column_double(sqlite_statement, 12);
    mag_distortion_major_scale = sqlite3_column_double(sqlite_statement, 13);
    mag_distortion_minor_scale = sqlite3_column_double(sqlite_statement, 14);
    protein_is_white           = sqlite3_column_int(sqlite_statement, 15);
    eer_super_res_factor       = sqlite3_column_int(sqlite_statement, 16);
    eer_frames_per_image       = sqlite3_column_int(sqlite_statement, 17);

    Finalize(sqlite_statement);
}

void Database::GetImageImportDefaults(float& voltage, float& spherical_aberration, float& pixel_size, bool& protein_is_white) {
    MyDebugAssertTrue(is_open == true, "database not open!");

    sqlite3_stmt* sqlite_statement;
    int           return_code;
    wxString      sql_command = "select * FROM IMAGE_IMPORT_DEFAULTS;";

    Prepare(sql_command, &sqlite_statement);
    Step(sqlite_statement);

    voltage              = sqlite3_column_double(sqlite_statement, 1);
    spherical_aberration = sqlite3_column_double(sqlite_statement, 2);
    pixel_size           = sqlite3_column_double(sqlite_statement, 3);
    protein_is_white     = sqlite3_column_int(sqlite_statement, 4);

    Finalize(sqlite_statement);
}

void Database::Close(bool remove_lock) {
    if ( is_open == true ) {
        BeginCommitLocker active_locker(this);
        // drop any ownership..
        if ( remove_lock == true ) {
            long my_process_id       = wxGetProcessId( );
            long database_process_id = ReturnSingleLongFromSelectCommand("SELECT ACTIVE_PROCESS FROM PROCESS_LOCK");

            if ( my_process_id != database_process_id && my_process_id != 0 && database_process_id > 0 ) {
                wxPrintf("\n\nError: Active process ID != my process ID, leaving the process lock in place.\n\n");
            }
            else {
                DeleteTable("PROCESS_LOCK");
                CreateProcessLockTable( );
            }
        }

        ExecuteSQL("PRAGMA optimize");

        active_locker.Commit( ); // Force commit

        // here the begin commit should be 0..

        if ( number_of_active_transactions != 0 )
            MyPrintWithDetails("Warning: Transaction number (%i) is not 0 upon close!\n", number_of_active_transactions);

        int return_code = sqlite3_close(sqlite_database);
        MyDebugAssertTrue(return_code == SQLITE_OK, "SQL close error, return code : %i\n", return_code);
    }

    is_open         = false;
    sqlite_database = NULL;
}

void Database::BeginBatchInsert(const char* table_name, int number_of_columns, ...) {
    MyDebugAssertTrue(is_open == true, "database not open!");
    MyDebugAssertTrue(in_batch_insert == false, "Starting batch insert but already in batch insert mode");
    MyDebugAssertTrue(in_batch_select == false, "Starting batch insert but already in batch select mode");

    wxString sql_command;
    int      counter;
    int      return_code;
    char*    error_message = NULL;

    in_batch_insert = true;

    va_list args;
    va_start(args, number_of_columns);

    // add a begin

    Begin( );

    sql_command = "INSERT OR REPLACE INTO ";
    sql_command += table_name;
    sql_command += " (";

    for ( counter = 1; counter <= number_of_columns; counter++ ) {
        sql_command += va_arg(args, const char*);

        if ( counter < number_of_columns )
            sql_command += ",";
        else
            sql_command += ") ";
    }

    va_end(args);

    sql_command += "VALUES (";

    for ( counter = 1; counter <= number_of_columns; counter++ ) {
        if ( counter < number_of_columns )
            sql_command += "?,";
        else
            sql_command += "?); ";
    }

    Prepare(sql_command, &batch_statement);
}

void Database::AddToBatchInsert(const char* column_format, ...) {
    int         argument_counter = 0;
    const char* text_pointer;
    int         return_code;
    va_list     args;
    va_start(args, column_format);

    while ( *column_format != '\0' ) {
        argument_counter++;

        if ( *column_format == 't' ) // text
        {
            text_pointer = va_arg(args, const char*);
            return_code  = sqlite3_bind_text(batch_statement, argument_counter, text_pointer, strlen(text_pointer), SQLITE_STATIC);
            CheckBindCode(return_code);
        }
        else if ( *column_format == 'r' ) // real
        {
            return_code = sqlite3_bind_double(batch_statement, argument_counter, va_arg(args, double));
            CheckBindCode(return_code);
        }
        else if ( *column_format == 'i' ) // integer
        {
            return_code = sqlite3_bind_int(batch_statement, argument_counter, va_arg(args, int));
            CheckBindCode(return_code);
        }
        else if ( *column_format == 'l' ) // long
        {
            return_code = sqlite3_bind_int64(batch_statement, argument_counter, va_arg(args, long));
            CheckBindCode(return_code);
        }
        else {
            MyPrintWithDetails("Error: Unknown format character!\n");
        }

        ++column_format;
    }

    va_end(args);

    return_code = Step(batch_statement);
    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    return_code = sqlite3_clear_bindings(batch_statement);
    MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code);

    return_code = sqlite3_reset(batch_statement);
    MyDebugAssertTrue(return_code == SQLITE_OK, "SQL error, return code : %i\n", return_code);
}

void Database::EndBatchInsert( ) {
    int   return_code;
    char* error_message = NULL;

    Commit( );

    Finalize(batch_statement);
    in_batch_insert = false;
}

void Database::BeginMovieAssetInsert( ) {
    BeginBatchInsert("MOVIE_ASSETS", 21, "MOVIE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "X_SIZE", "Y_SIZE", "NUMBER_OF_FRAMES", "VOLTAGE", "PIXEL_SIZE", "DOSE_PER_FRAME", "SPHERICAL_ABERRATION", "GAIN_FILENAME", "DARK_FILENAME", "OUTPUT_BINNING_FACTOR", "CORRECT_MAG_DISTORTION", "MAG_DISTORTION_ANGLE", "MAG_DISTORTION_MAJOR_SCALE", "MAG_DISTORTION_MINOR_SCALE", "PROTEIN_IS_WHITE", "EER_SUPER_RES_FACTOR", "EER_FRAMES_PER_IMAGE");
}

void Database::AddNextMovieAsset(int movie_asset_id, wxString name, wxString filename, int position_in_stack, int x_size, int y_size, int number_of_frames, double voltage, double pixel_size, double dose_per_frame, double spherical_aberration, wxString gain_filename, wxString dark_filename, double output_binning_factor, int correct_mag_distortion, float mag_distortion_angle, float mag_distortion_major_scale, float mag_distortion_minor_scale, int protein_is_white, int eer_super_res_factor, int eer_frames_per_image) {
    AddToBatchInsert("ittiiiirrrrttrirrriii", movie_asset_id, name.ToUTF8( ).data( ), filename.ToUTF8( ).data( ), position_in_stack, x_size, y_size, number_of_frames, voltage, pixel_size, dose_per_frame, spherical_aberration, gain_filename.ToUTF8( ).data( ), dark_filename.ToUTF8( ).data( ), output_binning_factor, correct_mag_distortion, mag_distortion_angle, mag_distortion_major_scale, mag_distortion_minor_scale, protein_is_white, eer_super_res_factor, eer_frames_per_image);
}

/*
void Database::AddMovieAsset(MovieAsset *asset_to_add)
{
	AddGroupInsert("itiiiirrrr", asset_to_add->asset_id, filename.ToUTF8().data(), position_in_stack, x_size, y_size, number_of_frames, voltage, pixel_size, dose_per_frame, spherical_aberration);
}
*/

void Database::EndMovieAssetInsert( ) {
    EndBatchInsert( );
}

void Database::BeginMovieAssetMetadataInsert( ) {
    BeginBatchInsert("MOVIE_ASSETS_METADATA", 11, "MOVIE_ASSET_ID", "METADATA_SOURCE", "CONTENT_JSON", "TILT_ANGLE", "STAGE_POSITION_X", "STAGE_POSITION_Y", "STAGE_POSITION_Z", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y", "EXPOSURE_DOSE", "ACQUISITION_TIME");
}

void Database::AddNextMovieAssetMetadata(MovieMetadataAsset asset) {
    AddToBatchInsert("lttrrrrrrrl", asset.movie_asset_id,
                     asset.metadata_source.ToUTF8( ).data( ),
                     asset.content_json.ToUTF8( ).data( ),
                     asset.tilt_angle,
                     asset.stage_position_x,
                     asset.stage_position_y,
                     asset.stage_position_z,
                     asset.image_shift_x,
                     asset.image_shift_y,
                     asset.exposure_dose,
                     (asset.acquisition_time.IsValid( )) ? asset.acquisition_time.GetAsDOS( ) : -1);
}

void Database::EndMovieAssetMetadataInsert( ) {
    EndBatchInsert( );
}

void Database::UpdateNumberOfFramesForAMovieAsset(int movie_asset_id, int new_number_of_frames) {
    ExecuteSQL(wxString::Format("UPDATE MOVIE_ASSETS SET NUMBER_OF_FRAMES = %i WHERE MOVIE_ASSET_ID = %i", new_number_of_frames, movie_asset_id));
}

void Database::BeginImageAssetInsert( ) {
    BeginBatchInsert("IMAGE_ASSETS", 13, "IMAGE_ASSET_ID", "NAME", "FILENAME", "POSITION_IN_STACK", "PARENT_MOVIE_ID", "ALIGNMENT_ID", "CTF_ESTIMATION_ID", "X_SIZE", "Y_SIZE", "PIXEL_SIZE", "VOLTAGE", "SPHERICAL_ABERRATION", "PROTEIN_IS_WHITE");
}

void Database::BeginVolumeAssetInsert( ) {
    BeginBatchInsert("VOLUME_ASSETS", 10, "VOLUME_ASSET_ID", "NAME", "FILENAME", "RECONSTRUCTION_JOB_ID", "PIXEL_SIZE", "X_SIZE", "Y_SIZE", "Z_SIZE", "HALF_MAP1_FILENAME", "HALF_MAP2_FILENAME");
}

void Database::AddNextVolumeAsset(int image_asset_id, wxString name, wxString filename, int reconstruction_job_id, double pixel_size, int x_size, int y_size, int z_size, wxString half_map_1_filename, wxString half_map_2_filename) {
    AddToBatchInsert("ittiriiitt", image_asset_id, name.ToUTF8( ).data( ), filename.ToUTF8( ).data( ), reconstruction_job_id, pixel_size, x_size, y_size, z_size, half_map_1_filename.ToUTF8( ).data( ), half_map_2_filename.ToUTF8( ).data( ));
}

#ifdef EXPERIMENTAL
void Database::BeginAtomicCoordinatesAssetInsert( ) {
    BeginBatchInsert("ATOMIC_COORDINATES_ASSETS", 11, "ATOMIC_COORDINATES_ASSET_ID", "NAME", "FILENAME", "SIMULATION_3D_JOB_ID", "X_SIZE", "Y_SIZE", "Z_SIZE", "PDB_ID", "PDB_AVG_BFACTOR", "PDB_STD_BFACTOR", "EFFECTIVE_WEIGHT");
}

void Database::AddNextAtomicCoordinatesAsset(const AtomicCoordinatesAsset* asset) {
    AddToBatchInsert("ittiiiitrrr", asset->asset_id, asset->asset_name.ToUTF8( ).data( ), asset->filename.GetFullPath( ).ToUTF8( ).data( ),
                     asset->simulation_3d_job_id, asset->x_size, asset->y_size, asset->z_size,
                     asset->pdb_id.ToUTF8( ).data( ), asset->pdb_avg_bfactor, asset->pdb_std_bfactor, asset->effective_weight);
}
#endif

void Database::AddNextImageAsset(int image_asset_id, wxString name, wxString filename, int position_in_stack, int parent_movie_id, int alignment_id, int ctf_estimation_id, int x_size, int y_size, double voltage, double pixel_size, double spherical_aberration, int protein_is_white) {
    AddToBatchInsert("ittiiiiiirrri", image_asset_id, name.ToUTF8( ).data( ), filename.ToUTF8( ).data( ), position_in_stack, parent_movie_id, alignment_id, ctf_estimation_id, x_size, y_size, pixel_size, voltage, spherical_aberration, protein_is_white);
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

void Database::BeginParticlePositionAssetInsert( ) {
    BeginBatchInsert("particle_position_assets", 11, "particle_position_asset_id", "parent_image_asset_id", "picking_id", "pick_job_id", "x_position", "y_position", "peak_height", "template_asset_id", "template_psi", "template_theta", "template_phi");
}

void Database::AddNextParticlePositionAsset(const ParticlePositionAsset* asset) {
    AddToBatchInsert("iiiirrrirrr", asset->asset_id, asset->parent_id, asset->picking_id, asset->pick_job_id, asset->x_position, asset->y_position, asset->peak_height, asset->parent_template_id, asset->template_psi, asset->template_theta, asset->template_phi);
}

bool Database::BeginBatchSelect(const char* select_command) {
    MyDebugAssertTrue(is_open == true, "database not open!");
    MyDebugAssertTrue(in_batch_insert == false, "Starting batch select but already in batch insert mode");
    MyDebugAssertTrue(in_batch_select == false, "Starting batch select but already in batch select mode");

    in_batch_select = true;

    Prepare(select_command, &batch_statement);
    last_return_code = Step(batch_statement);

    if ( last_return_code != SQLITE_DONE )
        return true;
    else
        return false;
}

bool Database::GetFromBatchSelect(const char* column_format, ...) {

    MyDebugAssertTrue(is_open == true, "database not open!");
    MyDebugAssertTrue(in_batch_insert == false, "in batch select but batch insert is true");
    MyDebugAssertTrue(in_batch_select == true, "in batch select but batch select is false");
    MyDebugAssertTrue(last_return_code == SQLITE_ROW, "get from batch select, but return code is not SQLITE_ROW");

    int argument_counter = -1;

    va_list args;
    va_start(args, column_format);

    while ( *column_format != '\0' ) {
        argument_counter++;

        if ( *column_format == 't' ) // text
        {
            va_arg(args, wxString*)[0] = sqlite3_column_text(batch_statement, argument_counter);
        }
        else if ( *column_format == 'f' ) // filename
        {
            va_arg(args, wxFileName*)[0] = sqlite3_column_text(batch_statement, argument_counter);
        }
        else if ( *column_format == 'r' ) // real
        {
            va_arg(args, double*)[0] = sqlite3_column_double(batch_statement, argument_counter);
        }
        else if ( *column_format == 'i' ) // integer
        {
            va_arg(args, int*)[0] = sqlite3_column_int(batch_statement, argument_counter);
        }
        else if ( *column_format == 's' ) // single (float)
        {
            double temp_double      = sqlite3_column_double(batch_statement, argument_counter);
            va_arg(args, float*)[0] = float(temp_double);
        }
        else if ( *column_format == 'l' ) // long
        {
            va_arg(args, long*)[0] = sqlite3_column_int64(batch_statement, argument_counter);
        }
        else {
            MyPrintWithDetails("Error: Unknown format character!\n");
        }

        ++column_format;
    }

    va_end(args);

    last_return_code = Step(batch_statement);

    if ( last_return_code == SQLITE_DONE )
        return false;
    else
        return true;
}

void Database::EndBatchSelect( ) {
    Finalize(batch_statement);
    in_batch_select = false;
}

template <bool flag = false>
inline void static_ProcessBatchSelectElement_no_match( ) { static_assert(flag, "no matching type!"); }

template <typename T>
void Database::ProcessBatchSelectElement(T* ptr, int& argument_counter) {

    if constexpr ( std::is_same_v<T, wxString> ) {
        ptr[0] = sqlite3_column_text(batch_statement, argument_counter);
    }
    else if constexpr ( std::is_same_v<T, wxFileName> ) {
        ptr[0] = sqlite3_column_text(batch_statement, argument_counter);
    }
    else if constexpr ( std::is_same_v<T, double> ) {
        ptr[0] = sqlite3_column_double(batch_statement, argument_counter);
    }
    else if constexpr ( std::is_same_v<T, int> ) {
        ptr[0] = sqlite3_column_int(batch_statement, argument_counter);
    }
    else if constexpr ( std::is_same_v<T, float> ) {
        double temp_double = sqlite3_column_double(batch_statement, argument_counter);
        ptr[0]             = float(temp_double);
    }
    else if constexpr ( std::is_same_v<T, long> ) {
        ptr[0] = sqlite3_column_int64(batch_statement, argument_counter);
    }
    else {
        static_ProcessBatchSelectElement_no_match( );
    }
    argument_counter++;
}

/**
 * @brief This function takes a variable number of arguments, that are all pointers to data that
 * will be fetched from the database. Rather than passing a char* literal to decode the types,
 * we use a fold expresssion to call the ProcessBatchSelectElement function for each argument.
 * The type of each pointer is checked there and the appropriate sqlite3 function is called, incrementing
 * the argument counter.
 * 
 * @tparam Args 
 * @param args 
 * @return true 
 * @return false 
 */
template <class... Args>
bool Database::GetFromBatchSelect_NoChar(Args... args) {

    MyDebugAssertTrue(is_open == true, "database not open!");
    MyDebugAssertTrue(in_batch_insert == false, "in batch select but batch insert is true");
    MyDebugAssertTrue(in_batch_select == true, "in batch select but batch select is false");
    MyDebugAssertTrue(last_return_code == SQLITE_ROW, "get from batch select, but return code is not SQLITE_ROW");

    int argument_counter = 0;

    (ProcessBatchSelectElement(args, argument_counter), ...);

    last_return_code = Step(batch_statement);

    if ( last_return_code == SQLITE_DONE )
        return false;
    else
        return true;
}

void Database::BeginAllMovieAssetsSelect( ) {
    BeginBatchSelect("SELECT * FROM MOVIE_ASSETS;");
}

void Database::BeginAllMovieGroupsSelect( ) {
    BeginBatchSelect("SELECT * FROM MOVIE_GROUP_LIST;");
}

void Database::BeginAllImageAssetsSelect( ) {
    BeginBatchSelect("SELECT * FROM IMAGE_ASSETS;");
}

void Database::BeginAllImageGroupsSelect( ) {
    BeginBatchSelect("SELECT * FROM IMAGE_GROUP_LIST;");
}

void Database::BeginAllParticlePositionAssetsSelect( ) {
    BeginBatchSelect("SELECT * FROM PARTICLE_POSITION_ASSETS;");
}

void Database::BeginAllVolumeAssetsSelect( ) {
    BeginBatchSelect("SELECT * FROM VOLUME_ASSETS;");
}

#ifdef EXPERIMENTAL
void Database::BeginAllAtomicCoordinatesAssetsSelect( ) {
    BeginBatchSelect("SELECT * FROM ATOMIC_COORDINATES_ASSETS;");
}
#endif

void Database::BeginAllParticlePositionGroupsSelect( ) {
    BeginBatchSelect("SELECT * FROM PARTICLE_POSITION_GROUP_LIST;");
}

void Database::BeginAllVolumeGroupsSelect( ) {
    BeginBatchSelect("SELECT * FROM VOLUME_GROUP_LIST;");
}

void Database::BeginAllRefinementPackagesSelect( ) {
    BeginBatchSelect("SELECT * FROM REFINEMENT_PACKAGE_ASSETS;");
}

void Database::BeginAllRunProfilesSelect( ) {
    BeginBatchSelect("SELECT * FROM RUN_PROFILES;");
}

RunProfile Database::GetNextRunProfile( ) {
    RunProfile    temp_profile;
    int           profile_table_number;
    int           return_code;
    wxString      profile_sql_select_command;
    sqlite3_stmt* list_statement = NULL;

    GetFromBatchSelect("itttti", &temp_profile.id, &temp_profile.name, &temp_profile.manager_command, &temp_profile.gui_address, &temp_profile.controller_address, &profile_table_number);

    // now we fill from the specific group table.

    profile_sql_select_command = wxString::Format("SELECT * FROM RUN_PROFILE_COMMANDS_%i", profile_table_number);

    Prepare(profile_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_profile.AddCommand(sqlite3_column_text(list_statement, 1), sqlite3_column_int(list_statement, 2), sqlite3_column_int(list_statement, 3), bool(sqlite3_column_int(list_statement, 4)), sqlite3_column_int(list_statement, 5), sqlite3_column_int(list_statement, 6));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);
    return temp_profile;
}

AssetGroup Database::GetNextMovieGroup( ) {
    AssetGroup    temp_group;
    int           group_table_number;
    int           return_code;
    wxString      group_sql_select_command;
    sqlite3_stmt* list_statement = NULL;

    GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

    // now we fill from the specific group table.

    group_sql_select_command = wxString::Format("SELECT * FROM MOVIE_GROUP_%i", group_table_number);

    Prepare(group_sql_select_command, &list_statement);

    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_group.AddMember(sqlite3_column_int(list_statement, 1));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);
    return temp_group;
}

AssetGroup Database::GetNextImageGroup( ) {
    AssetGroup    temp_group;
    int           group_table_number;
    int           return_code;
    wxString      group_sql_select_command;
    sqlite3_stmt* list_statement = NULL;

    GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

    // now we fill from the specific group table.

    group_sql_select_command = wxString::Format("SELECT * FROM IMAGE_GROUP_%i", group_table_number);

    Prepare(group_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_group.AddMember(sqlite3_column_int(list_statement, 1));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);
    return temp_group;
}

RefinementPackage* Database::GetNextRefinementPackage( ) {
    RefinementPackage*            temp_package;
    RefinementPackageParticleInfo temp_info;

    temp_package = new RefinementPackage;

    int return_code;
    int temp_int;

    wxString      group_sql_select_command;
    sqlite3_stmt* list_statement = NULL;

    GetFromBatchSelect("lttistrriiii", &temp_package->asset_id, &temp_package->name, &temp_package->stack_filename, &temp_package->stack_box_size, &temp_package->output_pixel_size, &temp_package->symmetry, &temp_package->estimated_particle_weight_in_kda, &temp_package->estimated_particle_size_in_angstroms, &temp_package->number_of_classes, &temp_package->number_of_run_refinments, &temp_package->last_refinment_id, &temp_int);
    temp_package->stack_has_white_protein = temp_int;

    // particles

    group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li ORDER BY POSITION_IN_STACK", temp_package->asset_id);

    Prepare(group_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_info.original_particle_position_asset_id = sqlite3_column_int64(list_statement, 0);
        temp_info.parent_image_id                     = sqlite3_column_int64(list_statement, 1);
        temp_info.position_in_stack                   = sqlite3_column_int64(list_statement, 2);
        temp_info.x_pos                               = sqlite3_column_double(list_statement, 3);
        temp_info.y_pos                               = sqlite3_column_double(list_statement, 4);
        temp_info.pixel_size                          = sqlite3_column_double(list_statement, 5);
        temp_info.defocus_1                           = sqlite3_column_double(list_statement, 6);
        temp_info.defocus_2                           = sqlite3_column_double(list_statement, 7);
        temp_info.defocus_angle                       = sqlite3_column_double(list_statement, 8);
        temp_info.phase_shift                         = sqlite3_column_double(list_statement, 9);
        temp_info.spherical_aberration                = sqlite3_column_double(list_statement, 10);
        temp_info.microscope_voltage                  = sqlite3_column_double(list_statement, 11);
        temp_info.amplitude_contrast                  = sqlite3_column_double(list_statement, 12);
        temp_info.assigned_subset                     = sqlite3_column_int(list_statement, 13);

        temp_package->contained_particles.Add(temp_info);

        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);

    // 3d references

    group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", temp_package->asset_id);

    Prepare(group_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_package->references_for_next_refinement.Add(sqlite3_column_int64(list_statement, 1));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);

    // refinement list

    group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", temp_package->asset_id);

    Prepare(group_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_package->refinement_ids.Add(sqlite3_column_int64(list_statement, 1));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);

    // classification list

    group_sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", temp_package->asset_id);

    Prepare(group_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_package->classification_ids.Add(sqlite3_column_int64(list_statement, 1));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);

    return temp_package;
}

AssetGroup Database::GetNextParticlePositionGroup( ) {
    AssetGroup    temp_group;
    int           group_table_number;
    int           return_code;
    wxString      group_sql_select_command;
    sqlite3_stmt* list_statement = NULL;

    GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

    // now we fill from the specific group table.

    group_sql_select_command = wxString::Format("SELECT * FROM PARTICLE_POSITION_GROUP_%i", group_table_number);

    Prepare(group_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_group.AddMember(sqlite3_column_int(list_statement, 1));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);
    return temp_group;
}

AssetGroup Database::GetNextVolumeGroup( ) {
    AssetGroup    temp_group;
    int           group_table_number;
    int           return_code;
    wxString      group_sql_select_command;
    sqlite3_stmt* list_statement = NULL;

    GetFromBatchSelect("iti", &temp_group.id, &temp_group.name, &group_table_number);

    // now we fill from the specific group table.

    group_sql_select_command = wxString::Format("SELECT * FROM VOLUME_GROUP_%i", group_table_number);

    Prepare(group_sql_select_command, &list_statement);
    return_code = Step(list_statement);

    while ( return_code == SQLITE_ROW ) {
        temp_group.AddMember(sqlite3_column_int(list_statement, 1));
        return_code = Step(list_statement);
    }

    MyDebugAssertTrue(return_code == SQLITE_DONE, "SQL error, return code : %i\n", return_code);

    Finalize(list_statement);
    return temp_group;
}

void Database::RemoveParticlePositionsFromResultsList(const int& picking_job_id, const int& parent_image_asset_id) {
    ExecuteSQL(wxString::Format("delete from particle_picking_results_%i where parent_image_asset_id = %i", picking_job_id, parent_image_asset_id));
}

int Database::ReturnPickingIDGivenPickingJobIDAndParentImageID(const int& picking_job_id, const int& parent_image_asset_id) {
    return ReturnSingleIntFromSelectCommand(wxString::Format("select distinct picking_id from particle_picking_results_%i where parent_image_asset_id = %i", picking_job_id, parent_image_asset_id));
}

void Database::SetManualEditForPickingID(const int& picking_id, const bool wanted_manual_edit) {
    int manual_edit_value = 0;
    if ( wanted_manual_edit )
        manual_edit_value = 1;
    ExecuteSQL(wxString::Format("update particle_picking_list set manual_edit=%i where picking_id=%i", manual_edit_value, picking_id));
}

void Database::RemoveParticlePositionsWithGivenParentImageIDFromGroup(const int& group_number_following_gui_convention, const int& parent_image_asset_id) {
    ExecuteSQL(wxString::Format("delete from particle_position_group_%i where exists(select 1 from particle_position_assets where particle_position_assets.parent_image_asset_id = %i AND particle_position_group_%i.particle_position_asset_id = particle_position_assets.particle_position_asset_id)", group_number_following_gui_convention, parent_image_asset_id, group_number_following_gui_convention));
}

void Database::RemoveParticlePositionAssetsPickedFromImagesAlsoPickedByGivenPickingJobID(const int& picking_job_id) {
    ExecuteSQL(wxString::Format("delete from particle_position_assets where exists(select 1 from particle_picking_list where particle_picking_list.picking_job_id = %i AND particle_picking_list.parent_image_asset_id = particle_position_assets.parent_image_asset_id)", picking_job_id));
}

void Database::RemoveParticlePositionAssetsPickedFromImageWithGivenID(const int& parent_image_asset_id) {
    ExecuteSQL(wxString::Format("delete from particle_position_assets where parent_image_asset_id = %i", parent_image_asset_id));
}

void Database::CopyParticleAssetsFromResultsTable(const int& picking_job_id, const int& parent_image_asset_id) {
    ExecuteSQL(wxString::Format("insert into particle_position_assets select particle_picking_results_%i.position_id, %i, particle_picking_results_%i.picking_id, %i, particle_picking_results_%i.x_position, particle_picking_results_%i.y_position, particle_picking_results_%i.peak_height, particle_picking_results_%i.template_asset_id, particle_picking_results_%i.template_psi, particle_picking_results_%i.template_theta, particle_picking_results_%i.template_phi from particle_picking_results_%i where particle_picking_results_%i.parent_image_asset_id = %i",
                                picking_job_id, parent_image_asset_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, picking_job_id, parent_image_asset_id));
}

void Database::AddArrayOfParticlePositionAssetsToResultsTable(const int& picking_job_id, ArrayOfParticlePositionAssets* array_of_assets) {
    BeginBatchInsert(wxString::Format("particle_picking_results_%i", picking_job_id), 10, "position_id", "picking_id", "parent_image_asset_id", "x_position", "y_position", "peak_height", "template_asset_id", "template_psi", "template_theta", "template_phi");

    ParticlePositionAsset* asset;
    for ( size_t counter = 0; counter < array_of_assets->GetCount( ); counter++ ) {
        asset = &array_of_assets->Item(counter);
        AddToBatchInsert("iiirrrirrr", asset->asset_id, asset->picking_id, asset->parent_id, asset->x_position, asset->y_position, asset->peak_height, asset->parent_template_id, asset->template_psi, asset->template_theta, asset->template_phi);
    }

    EndBatchInsert( );
}

void Database::AddArrayOfParticlePositionAssetsToAssetsTable(ArrayOfParticlePositionAssets* array_of_assets) {
    BeginParticlePositionAssetInsert( );
    ParticlePositionAsset* asset;
    for ( size_t counter = 0; counter < array_of_assets->GetCount( ); counter++ ) {
        asset = &array_of_assets->Item(counter);
        AddNextParticlePositionAsset(asset);
    }

    EndBatchInsert( );
}

ArrayOfParticlePositionAssets Database::ReturnArrayOfParticlePositionAssetsFromResultsTable(const int& picking_job_id, const int& parent_image_asset_id) {
    ArrayOfParticlePositionAssets array_of_assets;
    array_of_assets.Clear( );
    BeginBatchSelect(wxString::Format("select * from particle_picking_results_%i where parent_image_asset_id = %i", picking_job_id, parent_image_asset_id));
    while ( last_return_code == SQLITE_ROW ) {
        array_of_assets.Add(GetNextParticlePositionAssetFromResults( ));
    }
    EndBatchSelect( );
    return array_of_assets;
}

ArrayOfParticlePositionAssets Database::ReturnArrayOfParticlePositionAssetsFromAssetsTable(const int& parent_image_asset_id) {
    ArrayOfParticlePositionAssets array_of_assets;
    array_of_assets.Clear( );
    BeginBatchSelect(wxString::Format("select * from particle_position_assets where parent_image_asset_id = %i", parent_image_asset_id));
    while ( last_return_code == SQLITE_ROW ) {
        array_of_assets.Add(GetNextParticlePositionAsset( ));
    }
    EndBatchSelect( );
    return array_of_assets;
}

int Database::ReturnNumberOf2DClassMembers(long wanted_classification_id, int wanted_class_number) {
    return ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM CLASSIFICATION_RESULT_%li WHERE BEST_CLASS = %i", wanted_classification_id, wanted_class_number));
}

wxArrayLong Database::Return2DClassMembers(long wanted_classifiction_id, int wanted_class_number) {
    wxArrayLong class_members;
    long        temp_long;

    BeginBatchSelect(wxString::Format("select POSITION_IN_STACK from classification_result_%li where BEST_CLASS = %i", wanted_classifiction_id, wanted_class_number));

    while ( last_return_code == SQLITE_ROW ) {
        GetFromBatchSelect("l", &temp_long);
        class_members.Add(temp_long);
    }

    EndBatchSelect( );

    return class_members;
}

MovieAsset Database::GetNextMovieAsset( ) {
    MovieAsset temp_asset;
    int        correct_mag_distortion;

    GetFromBatchSelect("itfiiiirrrrttrirrriii", &temp_asset.asset_id, &temp_asset.asset_name, &temp_asset.filename, &temp_asset.position_in_stack, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.number_of_frames, &temp_asset.microscope_voltage, &temp_asset.pixel_size, &temp_asset.dose_per_frame, &temp_asset.spherical_aberration, &temp_asset.gain_filename, &temp_asset.dark_filename, &temp_asset.output_binning_factor, &correct_mag_distortion, &temp_asset.mag_distortion_angle, &temp_asset.mag_distortion_major_scale, &temp_asset.mag_distortion_minor_scale, &temp_asset.protein_is_white, &temp_asset.eer_super_res_factor, &temp_asset.eer_frames_per_image);
    temp_asset.correct_mag_distortion = correct_mag_distortion;
    temp_asset.total_dose             = temp_asset.dose_per_frame * temp_asset.number_of_frames;
    return temp_asset;
}

ImageAsset Database::GetNextImageAsset( ) {
    ImageAsset temp_asset;

    GetFromBatchSelect("itfiiiiiirrri", &temp_asset.asset_id, &temp_asset.asset_name, &temp_asset.filename, &temp_asset.position_in_stack, &temp_asset.parent_id, &temp_asset.alignment_id, &temp_asset.ctf_estimation_id, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.pixel_size, &temp_asset.microscope_voltage, &temp_asset.spherical_aberration, &temp_asset.protein_is_white);
    return temp_asset;
}

ParticlePositionAsset Database::GetNextParticlePositionAsset( ) {
    ParticlePositionAsset temp_asset;
    GetFromBatchSelect("iiiirrrirrr", &temp_asset.asset_id, &temp_asset.parent_id, &temp_asset.picking_id, &temp_asset.pick_job_id, &temp_asset.x_position, &temp_asset.y_position, &temp_asset.peak_height, &temp_asset.parent_template_id, &temp_asset.template_psi, &temp_asset.template_theta, &temp_asset.template_phi);
    return temp_asset;
}

ParticlePositionAsset Database::GetNextParticlePositionAssetFromResults( ) {
    ParticlePositionAsset temp_asset;
    GetFromBatchSelect("iiirrrirrr", &temp_asset.asset_id, &temp_asset.picking_id, &temp_asset.parent_id, &temp_asset.x_position, &temp_asset.y_position, &temp_asset.peak_height, &temp_asset.parent_template_id, &temp_asset.template_psi, &temp_asset.template_theta, &temp_asset.template_phi);
    return temp_asset;
}

VolumeAsset Database::GetNextVolumeAsset( ) {
    VolumeAsset temp_asset;
    GetFromBatchSelect("itflriiitt", &temp_asset.asset_id, &temp_asset.asset_name, &temp_asset.filename, &temp_asset.reconstruction_job_id, &temp_asset.pixel_size, &temp_asset.x_size, &temp_asset.y_size, &temp_asset.z_size, &temp_asset.half_map_1_filename, &temp_asset.half_map_2_filename);
    return temp_asset;
}

#ifdef EXPERIMENTAL

AtomicCoordinatesAsset Database::GetNextAtomicCoordinatesAsset( ) {
    AtomicCoordinatesAsset temp_asset;
    // Note: no distinction between single and double (s/r) seems to be made in writing to the DB, based on format strings, yet when reading it must be correct.
    //
    GetFromBatchSelect("itfliiitsss", &temp_asset.asset_id, &temp_asset.asset_name, &temp_asset.filename, &temp_asset.simulation_3d_job_id,
                       &temp_asset.x_size, &temp_asset.y_size, &temp_asset.z_size, &temp_asset.pdb_id,
                       &temp_asset.pdb_avg_bfactor, &temp_asset.pdb_std_bfactor, &temp_asset.effective_weight);
    return temp_asset;
}
#endif

void Database::AddOrReplaceRunProfile(RunProfile* profile_to_add) {

    BeginCommitLocker active_locker(this);
    InsertOrReplace("RUN_PROFILES", "ptttti", "RUN_PROFILE_ID", "PROFILE_NAME", "MANAGER_RUN_COMMAND", "GUI_ADDRESS", "CONTROLLER_ADDRESS", "COMMANDS_ID", profile_to_add->id, profile_to_add->name.ToUTF8( ).data( ), profile_to_add->manager_command.ToUTF8( ).data( ), profile_to_add->gui_address.ToUTF8( ).data( ), profile_to_add->controller_address.ToUTF8( ).data( ), profile_to_add->id);
    DeleteTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id));
    CreateTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id), "ptiiiii", "COMMANDS_NUMBER", "COMMAND_STRING", "NUMBER_OF_COPIES", "NUMBER_OF_THREADS_PER_COPY", "OVERRIDE_TOTAL_NUMBER_OF_COPIES", "OVERIDDEN_TOTAL_NUMBER_OF_COPIES", "DELAY_TIME_IN_MS");

    for ( int counter = 0; counter < profile_to_add->number_of_run_commands; counter++ ) {
        InsertOrReplace(wxString::Format("RUN_PROFILE_COMMANDS_%i", profile_to_add->id), "ptiiiii", "COMMANDS_NUMBER", "COMMAND_STRING", "NUMBER_OF_COPIES", "NUMBER_OF_THREADS_PER_COPY", "OVERRIDE_TOTAL_NUMBER_OF_COPIES", "OVERIDDEN_TOTAL_NUMBER_OF_COPIES", "DELAY_TIME_IN_MS", counter, profile_to_add->run_commands[counter].command_to_run.ToUTF8( ).data( ), profile_to_add->run_commands[counter].number_of_copies, profile_to_add->run_commands[counter].number_of_threads_per_copy, int(profile_to_add->run_commands[counter].override_total_copies), profile_to_add->run_commands[counter].overriden_number_of_copies, profile_to_add->run_commands[counter].delay_time_in_ms);
    }
}

void Database::DeleteRunProfile(int wanted_id) {
    BeginCommitLocker active_locker(this);
    ExecuteSQL(wxString::Format("DELETE FROM RUN_PROFILES WHERE RUN_PROFILE_ID=%i", wanted_id).ToUTF8( ).data( ));
    DeleteTable(wxString::Format("RUN_PROFILE_COMMANDS_%i", wanted_id));
}

void Database::AddRefinementPackageAsset(RefinementPackage* asset_to_add) {
    BeginCommitLocker active_locker(this);
    int               temp_int = asset_to_add->stack_has_white_protein;
    InsertOrReplace("REFINEMENT_PACKAGE_ASSETS", "Pttirtrriiii", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "STACK_FILENAME", "STACK_BOX_SIZE", "OUTPUT_PIXEL_SIZE", "SYMMETRY", "MOLECULAR_WEIGHT", "PARTICLE_SIZE", "NUMBER_OF_CLASSES", "NUMBER_OF_REFINEMENTS", "LAST_REFINEMENT_ID", "STACK_HAS_WHITE_PROTEIN", asset_to_add->asset_id, asset_to_add->name.ToUTF8( ).data( ), asset_to_add->stack_filename.ToUTF8( ).data( ), asset_to_add->stack_box_size, asset_to_add->output_pixel_size, asset_to_add->symmetry.ToUTF8( ).data( ), asset_to_add->estimated_particle_weight_in_kda, asset_to_add->estimated_particle_size_in_angstroms, asset_to_add->number_of_classes, asset_to_add->number_of_run_refinments, asset_to_add->last_refinment_id, temp_int);
    CreateRefinementPackageContainedParticlesTable(asset_to_add->asset_id);
    CreateRefinementPackageCurrent3DReferencesTable(asset_to_add->asset_id);
    CreateRefinementPackageRefinementsList(asset_to_add->asset_id);
    CreateRefinementPackageClassificationsList(asset_to_add->asset_id);

    BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", asset_to_add->asset_id), 14, "ORIGINAL_PARTICLE_POSITION_ASSET_ID", "PARENT_IMAGE_ASSET_ID", "POSITION_IN_STACK", "X_POSITION", "Y_POSITION", "PIXEL_SIZE", "DEFOCUS_1", "DEFOCUS_2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "SPHERICAL_ABERRATION", "MICROSCOPE_VOLTAGE", "AMPLITUDE_CONTRAST", "ASSIGNED_SUBSET");

    for ( long counter = 0; counter < asset_to_add->contained_particles.GetCount( ); counter++ ) {

        AddToBatchInsert("lllrrrrrrrrrri", asset_to_add->contained_particles.Item(counter).original_particle_position_asset_id, asset_to_add->contained_particles.Item(counter).parent_image_id, asset_to_add->contained_particles.Item(counter).position_in_stack, asset_to_add->contained_particles.Item(counter).x_pos, asset_to_add->contained_particles.Item(counter).y_pos, asset_to_add->contained_particles.Item(counter).pixel_size, asset_to_add->contained_particles.Item(counter).defocus_1, asset_to_add->contained_particles.Item(counter).defocus_2, asset_to_add->contained_particles.Item(counter).defocus_angle, asset_to_add->contained_particles.Item(counter).phase_shift, asset_to_add->contained_particles.Item(counter).spherical_aberration, asset_to_add->contained_particles.Item(counter).microscope_voltage, asset_to_add->contained_particles.Item(counter).amplitude_contrast, asset_to_add->contained_particles.Item(counter).assigned_subset);
    }

    EndBatchInsert( );

    BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", asset_to_add->asset_id), 2, "CLASS_NUMBER", "VOLUME_ASSET_ID");

    for ( long counter = 0; counter < asset_to_add->references_for_next_refinement.GetCount( ); counter++ ) {

        AddToBatchInsert("ll", counter + 1, asset_to_add->references_for_next_refinement.Item(counter));
    }

    EndBatchInsert( );

    BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", asset_to_add->asset_id), 2, "REFINEMENT_NUMBER", "REFINEMENT_ID");

    for ( long counter = 0; counter < asset_to_add->refinement_ids.GetCount( ); counter++ ) {

        AddToBatchInsert("ll", counter + 1, asset_to_add->refinement_ids.Item(counter));
    }

    EndBatchInsert( );

    BeginBatchInsert(wxString::Format("REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li", asset_to_add->asset_id), 2, "CLASSIFICATION_NUMBER", "CLASSIFICATION_ID");

    for ( long counter = 0; counter < asset_to_add->classification_ids.GetCount( ); counter++ ) {

        AddToBatchInsert("ll", counter + 1, asset_to_add->classification_ids.Item(counter));
    }

    EndBatchInsert( );
}

void Database::AddStartupJob(long startup_job_id, long refinement_package_asset_id, wxString name, int number_of_starts, int number_of_cycles, float initial_res_limit, float final_res_limit, bool auto_mask, bool auto_percent_used, float initial_percent_used, float final_percent_used, float mask_radius, bool apply_blurring, float smoothing_factor, wxArrayLong result_volume_ids) {
    BeginCommitLocker active_locker(this);
    InsertOrReplace("STARTUP_LIST", "Pltiirriirrrir", "STARTUP_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "NUMBER_OF_STARTS", "NUMBER_OF_CYCLES", "INITIAL_RES_LIMIT", "FINAL_RES_LIMIT", "AUTO_MASK", "AUTO_PERCENT_USED", "INITIAL_PERCENT_USED", "FINAL_PERCENT_USED", "MASK_RADIUS", "APPLY_LIKELIHOOD_BLURRING", "SMOOTHING_FACTOR", startup_job_id, refinement_package_asset_id, name.ToUTF8( ).data( ), number_of_starts, number_of_cycles, initial_res_limit, final_res_limit, int(auto_mask), int(auto_percent_used), initial_percent_used, final_percent_used, mask_radius, int(apply_blurring), smoothing_factor);
    CreateStartupResultTable(startup_job_id);

    for ( int class_counter = 0; class_counter < result_volume_ids.GetCount( ); class_counter++ ) {
        InsertOrReplace(wxString::Format("STARTUP_RESULT_%li", startup_job_id), "pl", "CLASS_NUMBER", "VOLUME_ASSET_ID", class_counter + 1, result_volume_ids[class_counter]);
    }
}

void Database::AddReconstructionJob(long reconstruction_id, long refinement_package_asset_id, long refinement_id, wxString name, float inner_mask_radius, float outer_mask_radius, float resolution_limit, float score_weight_conversion, bool should_adjust_score, bool should_crop_images, bool should_save_half_maps, bool should_likelihood_blur, float smoothing_factor, int class_number, long volume_asset_id) {
    InsertOrReplace("RECONSTRUCTION_LIST", "Plltrrrriiiiril", "RECONSTRUCTION_ID", "REFINEMENT_PACKAGE_ID", "REFINEMENT_ID", "NAME", "INNER_MASK_RADIUS", "OUTER_MASK_RADIUS", "RESOLUTION_LIMIT", "SCORE_WEIGHT_CONVERSION", "SHOULD_ADJUST_SCORES", "SHOULD_CROP_IMAGES", "SHOULD_SAVE_HALF_MAPS", "SHOULD_LIKELIHOOD_BLUR", "SMOOTHING_FACTOR", "CLASS_NUMBER", "VOLUME_ASSET_ID", reconstruction_id, refinement_package_asset_id, refinement_id, name.ToUTF8( ).data( ), inner_mask_radius, outer_mask_radius, resolution_limit, score_weight_conversion, int(should_adjust_score), int(should_crop_images), int(should_save_half_maps), int(should_likelihood_blur), smoothing_factor, class_number, volume_asset_id);
}

void Database::GetReconstructionJob(long wanted_reconstruction_id, long& refinement_package_asset_id, long& refinement_id, wxString& name, float& inner_mask_radius, float& outer_mask_radius, float& resolution_limit, float& score_weight_conversion, bool& should_adjust_score, bool& should_crop_images, bool& should_save_half_maps, bool& should_likelihood_blur, float& smoothing_factor, int& class_number, long& volume_asset_id) {
    wxString      sql_select_command;
    sqlite3_stmt* list_statement = NULL;
    int           return_code;

    sql_select_command = wxString::Format("SELECT * FROM RECONSTRUCTION_LIST WHERE RECONSTRUCTION_ID=%li", wanted_reconstruction_id);
    Prepare(sql_select_command, &list_statement);

    return_code = Step(list_statement);

    refinement_package_asset_id = sqlite3_column_int64(list_statement, 1);
    refinement_id               = sqlite3_column_int64(list_statement, 2);
    name                        = sqlite3_column_text(list_statement, 3);
    inner_mask_radius           = sqlite3_column_double(list_statement, 4);
    outer_mask_radius           = sqlite3_column_double(list_statement, 5);
    resolution_limit            = sqlite3_column_double(list_statement, 6);
    score_weight_conversion     = sqlite3_column_double(list_statement, 7);
    should_adjust_score         = sqlite3_column_int(list_statement, 8);
    should_crop_images          = sqlite3_column_int(list_statement, 9);
    should_save_half_maps       = sqlite3_column_int(list_statement, 10);
    should_likelihood_blur      = sqlite3_column_int(list_statement, 11);
    smoothing_factor            = sqlite3_column_double(list_statement, 12);
    class_number                = sqlite3_column_int(list_statement, 13);
    volume_asset_id             = sqlite3_column_int64(list_statement, 14);

    Finalize(list_statement);
}

void Database::AddTemplateMatchingResult(long wanted_template_match_id, TemplateMatchJobResults& job_details) {

    int peak_counter;

    InsertOrReplace("TEMPLATE_MATCH_LIST", "Ptllillltrrrrrrrrrrrrrrrrrrrrrrittttttttttt", "TEMPLATE_MATCH_ID", "JOB_NAME", "DATETIME_OF_RUN", "TEMPLATE_MATCH_JOB_ID", "JOB_TYPE_CODE", "INPUT_TEMPLATE_MATCH_ID", "IMAGE_ASSET_ID", "REFERENCE_VOLUME_ASSET_ID", "USED_SYMMETRY", "USED_PIXEL_SIZE", "USED_VOLTAGE", "USED_SPHERICAL_ABERRATION", "USED_AMPLITUDE_CONTRAST", "USED_DEFOCUS1", "USED_DEFOCUS2", "USED_DEFOCUS_ANGLE", "USED_PHASE_SHIFT", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "OUT_OF_PLANE_ANGULAR_STEP", "IN_PLANE_ANGULAR_STEP", "DEFOCUS_SEARCH_RANGE", "DEFOCUS_STEP", "PIXEL_SIZE_SEARCH_RANGE", "PIXEL_SIZE_STEP", "REFINEMENT_THRESHOLD", "USED_THRESHOLD", "REF_BOX_SIZE_IN_ANGSTROMS", "MASK_RADIUS", "MIN_PEAK_RADIUS", "XY_CHANGE_THRESHOLD", "EXCLUDE_ABOVE_XY_THRESHOLD", "MIP_OUTPUT_FILE", "SCALED_MIP_OUTPUT_FILE", "AVG_OUTPUT_FILE", "STD_OUTPUT_FILE", "PSI_OUTPUT_FILE", "THETA_OUTPUT_FILE", "PHI_OUTPUT_FILE", "DEFOCUS_OUTPUT_FILE", "PIXEL_SIZE_OUTPUT_FILE", "HISTOGRAM_OUTPUT_FILE", "PROJECTION_RESULT_OUTPUT_FILE", wanted_template_match_id, job_details.job_name.ToUTF8( ).data( ), job_details.datetime_of_run, job_details.job_id, job_details.job_type, job_details.input_job_id, job_details.image_asset_id, job_details.ref_volume_asset_id, job_details.symmetry.ToUTF8( ).data( ), job_details.pixel_size, job_details.voltage, job_details.spherical_aberration, job_details.amplitude_contrast, job_details.defocus1, job_details.defocus2, job_details.defocus_angle, job_details.phase_shift, job_details.low_res_limit, job_details.high_res_limit, job_details.out_of_plane_step, job_details.in_plane_step, job_details.defocus_search_range, job_details.defocus_step, job_details.pixel_size_search_range, job_details.pixel_size_step, job_details.refinement_threshold, job_details.used_threshold, job_details.reference_box_size_in_angstroms, job_details.mask_radius, job_details.min_peak_radius, job_details.xy_change_threshold, int(job_details.exclude_above_xy_threshold), job_details.mip_filename.ToUTF8( ).data( ), job_details.scaled_mip_filename.ToUTF8( ).data( ), job_details.avg_filename.ToUTF8( ).data( ), job_details.std_filename.ToUTF8( ).data( ), job_details.psi_filename.ToUTF8( ).data( ), job_details.theta_filename.ToUTF8( ).data( ), job_details.phi_filename.ToUTF8( ).data( ), job_details.defocus_filename.ToUTF8( ).data( ), job_details.pixel_size_filename.ToUTF8( ).data( ), job_details.histogram_filename.ToUTF8( ).data( ), job_details.projection_result_filename.ToUTF8( ).data( ));

    CreateTemplateMatchPeakListTable(wanted_template_match_id);

    BeginBatchInsert(wxString::Format("TEMPLATE_MATCH_PEAK_LIST_%li", wanted_template_match_id), 9, "PEAK_NUMBER", "X_POSITION", "Y_POSITION", "PSI", "THETA", "PHI", "DEFOCUS", "PIXEL_SIZE", "PEAK_HEIGHT");

    for ( peak_counter = 1; peak_counter <= job_details.found_peaks.GetCount( ); peak_counter++ ) {
        AddToBatchInsert("irrrrrrrr", peak_counter, job_details.found_peaks[peak_counter - 1].x_pos, job_details.found_peaks[peak_counter - 1].y_pos, job_details.found_peaks[peak_counter - 1].psi, job_details.found_peaks[peak_counter - 1].theta, job_details.found_peaks[peak_counter - 1].phi, job_details.found_peaks[peak_counter - 1].defocus, job_details.found_peaks[peak_counter - 1].pixel_size, job_details.found_peaks[peak_counter - 1].peak_height);
    }

    EndBatchInsert( );

    CreateTemplateMatchPeakChangeListTable(wanted_template_match_id);

    BeginBatchInsert(wxString::Format("TEMPLATE_MATCH_PEAK_CHANGE_LIST_%li", wanted_template_match_id), 11, "PEAK_NUMBER", "X_POSITION", "Y_POSITION", "PSI", "THETA", "PHI", "DEFOCUS", "PIXEL_SIZE", "PEAK_HEIGHT", "ORIGINAL_PEAK_NUMBER", "NEW_PEAK_NUMBER");

    for ( peak_counter = 1; peak_counter <= job_details.peak_changes.GetCount( ); peak_counter++ ) {
        AddToBatchInsert("irrrrrrrrii", peak_counter, job_details.peak_changes[peak_counter - 1].x_pos, job_details.peak_changes[peak_counter - 1].y_pos, job_details.peak_changes[peak_counter - 1].psi, job_details.peak_changes[peak_counter - 1].theta, job_details.peak_changes[peak_counter - 1].phi, job_details.peak_changes[peak_counter - 1].defocus, job_details.peak_changes[peak_counter - 1].pixel_size, job_details.peak_changes[peak_counter - 1].peak_height, job_details.peak_changes[peak_counter - 1].original_peak_number, job_details.peak_changes[peak_counter - 1].new_peak_number);
    }

    EndBatchInsert( );
}

/** 
 * Returns the Template Match ID for one result of a given Job
*/

long Database::GetTemplateMatchIdForGivenJobId(long wanted_template_match_job_id) {
    wxString sql_select_command = wxString::Format("SELECT TEMPLATE_MATCH_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID=%li", wanted_template_match_job_id);
    return ReturnSingleLongFromSelectCommand(sql_select_command);
}

TemplateMatchJobResults Database::GetTemplateMatchingResultByID(long wanted_template_match_id) {
    TemplateMatchJobResults    temp_result;
    TemplateMatchFoundPeakInfo temp_peak_info;

    wxString      sql_select_command;
    int           return_code;
    long          template_match_id;
    sqlite3_stmt* list_statement = NULL;
    bool          more_data;
    sql_select_command = wxString::Format("SELECT * FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_ID=%li", wanted_template_match_id);
    Prepare(sql_select_command, &list_statement);
    return_code = Step(list_statement);

    template_match_id               = sqlite3_column_int64(list_statement, 0);
    temp_result.job_name            = sqlite3_column_text(list_statement, 1);
    temp_result.datetime_of_run     = sqlite3_column_int64(list_statement, 2);
    temp_result.job_id              = sqlite3_column_int64(list_statement, 3);
    temp_result.job_type            = sqlite3_column_int(list_statement, 4);
    temp_result.input_job_id        = sqlite3_column_int64(list_statement, 5);
    temp_result.image_asset_id      = sqlite3_column_int64(list_statement, 6);
    temp_result.ref_volume_asset_id = sqlite3_column_int64(list_statement, 7);

    // number 8 is "IS ACTIVE" which i am not recording in the class

    temp_result.symmetry                        = sqlite3_column_text(list_statement, 9);
    temp_result.pixel_size                      = sqlite3_column_double(list_statement, 10);
    temp_result.voltage                         = sqlite3_column_double(list_statement, 11);
    temp_result.spherical_aberration            = sqlite3_column_double(list_statement, 12);
    temp_result.amplitude_contrast              = sqlite3_column_double(list_statement, 13);
    temp_result.defocus1                        = sqlite3_column_double(list_statement, 14);
    temp_result.defocus2                        = sqlite3_column_double(list_statement, 15);
    temp_result.defocus_angle                   = sqlite3_column_double(list_statement, 16);
    temp_result.phase_shift                     = sqlite3_column_double(list_statement, 17);
    temp_result.low_res_limit                   = sqlite3_column_double(list_statement, 18);
    temp_result.high_res_limit                  = sqlite3_column_double(list_statement, 19);
    temp_result.out_of_plane_step               = sqlite3_column_double(list_statement, 20);
    temp_result.in_plane_step                   = sqlite3_column_double(list_statement, 21);
    temp_result.defocus_search_range            = sqlite3_column_double(list_statement, 22);
    temp_result.defocus_step                    = sqlite3_column_double(list_statement, 23);
    temp_result.pixel_size_search_range         = sqlite3_column_double(list_statement, 24);
    temp_result.pixel_size_step                 = sqlite3_column_double(list_statement, 25);
    temp_result.refinement_threshold            = sqlite3_column_double(list_statement, 26);
    temp_result.used_threshold                  = sqlite3_column_double(list_statement, 27);
    temp_result.reference_box_size_in_angstroms = sqlite3_column_double(list_statement, 28);
    temp_result.mask_radius                     = sqlite3_column_double(list_statement, 29);
    temp_result.min_peak_radius                 = sqlite3_column_double(list_statement, 30);
    temp_result.xy_change_threshold             = sqlite3_column_double(list_statement, 31);
    temp_result.exclude_above_xy_threshold      = bool(sqlite3_column_int(list_statement, 32));
    temp_result.mip_filename                    = sqlite3_column_text(list_statement, 33);
    temp_result.scaled_mip_filename             = sqlite3_column_text(list_statement, 34);
    temp_result.avg_filename                    = sqlite3_column_text(list_statement, 35);
    temp_result.std_filename                    = sqlite3_column_text(list_statement, 36);
    temp_result.psi_filename                    = sqlite3_column_text(list_statement, 37);
    temp_result.theta_filename                  = sqlite3_column_text(list_statement, 38);
    temp_result.phi_filename                    = sqlite3_column_text(list_statement, 39);
    temp_result.defocus_filename                = sqlite3_column_text(list_statement, 40);
    temp_result.pixel_size_filename             = sqlite3_column_text(list_statement, 41);
    temp_result.histogram_filename              = sqlite3_column_text(list_statement, 42);
    temp_result.projection_result_filename      = sqlite3_column_text(list_statement, 43);

    Finalize(list_statement);

    // now get all the peaks
    sql_select_command = wxString::Format("SELECT * FROM TEMPLATE_MATCH_PEAK_LIST_%li", template_match_id);
    more_data          = BeginBatchSelect(sql_select_command);

    int peak_number;
    while ( more_data == true ) {
        more_data = GetFromBatchSelect("issssssss", &peak_number,
                                       &temp_peak_info.x_pos,
                                       &temp_peak_info.y_pos,
                                       &temp_peak_info.psi,
                                       &temp_peak_info.theta,
                                       &temp_peak_info.phi,
                                       &temp_peak_info.defocus,
                                       &temp_peak_info.pixel_size,
                                       &temp_peak_info.peak_height);

        temp_result.found_peaks.Add(temp_peak_info);
    }

    EndBatchSelect( );

    // now all the changes..

    // now get all the peaks
    sql_select_command = wxString::Format("SELECT * FROM TEMPLATE_MATCH_PEAK_CHANGE_LIST_%li", template_match_id);
    more_data          = BeginBatchSelect(sql_select_command);

    while ( more_data == true ) {

        more_data = GetFromBatchSelect("issssssssii", &peak_number,
                                       &temp_peak_info.x_pos,
                                       &temp_peak_info.y_pos,
                                       &temp_peak_info.psi,
                                       &temp_peak_info.theta,
                                       &temp_peak_info.phi,
                                       &temp_peak_info.defocus,
                                       &temp_peak_info.pixel_size,
                                       &temp_peak_info.peak_height,
                                       &temp_peak_info.original_peak_number,
                                       &temp_peak_info.new_peak_number);

        temp_result.peak_changes.Add(temp_peak_info);
    }

    EndBatchSelect( );

    return temp_result;
}

void Database::SetActiveTemplateMatchJobForGivenImageAssetID(long image_asset, long template_match_job_id) {
    BeginCommitLocker active_locker(this);
    ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET IS_ACTIVE=0 WHERE IMAGE_ASSET_ID=%li", image_asset));
    ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET IS_ACTIVE=1 WHERE IMAGE_ASSET_ID=%li AND TEMPLATE_MATCH_JOB_ID=%li", image_asset, template_match_job_id));
}

void Database::AddRefinement(Refinement* refinement_to_add) {
    int   class_counter;
    long  counter;
    bool  should_commit = false;
    float estimated_resolution;

    BeginCommitLocker active_locker(this);

    InsertOrReplace("REFINEMENT_LIST", "Pltillllirr", "REFINEMENT_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "RESOLUTION_STATISTICS_ARE_GENERATED", "DATETIME_OF_RUN", "STARTING_REFINEMENT_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "RESOLUTION_STATISTICS_BOX_SIZE", "RESOLUTION_STATISTICS_PIXEL_SIZE", "PERCENT_USED", refinement_to_add->refinement_id, refinement_to_add->refinement_package_asset_id, refinement_to_add->name.ToUTF8( ).data( ), refinement_to_add->resolution_statistics_are_generated, refinement_to_add->datetime_of_run.GetAsDOS( ), refinement_to_add->starting_refinement_id, refinement_to_add->number_of_particles, refinement_to_add->number_of_classes, refinement_to_add->resolution_statistics_box_size, refinement_to_add->resolution_statistics_pixel_size, refinement_to_add->percent_used);

    for ( class_counter = 0; class_counter < refinement_to_add->number_of_classes; class_counter++ ) {
        CreateRefinementResultTable(refinement_to_add->refinement_id, class_counter + 1);
        CreateRefinementResolutionStatisticsTable(refinement_to_add->refinement_id, class_counter + 1);
        CreateRefinementDetailsTable(refinement_to_add->refinement_id);

        if ( refinement_to_add->resolution_statistics_are_generated == true )
            estimated_resolution = 0.0;
        else
            estimated_resolution = refinement_to_add->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution( );

        InsertOrReplace(wxString::Format("REFINEMENT_DETAILS_%li", refinement_to_add->refinement_id), "ilrrrrrrirrrrirrrrirrrrlliiilrrir", "CLASS_NUMBER", "REFERENCE_VOLUME_ASSET_ID", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "SIGNED_CC_RESOLUTION_LIMIT", "GLOBAL_RESOLUTION_LIMIT", "GLOBAL_MASK_RADIUS", "NUMBER_RESULTS_TO_REFINE", "ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "CLASSIFICATION_RESOLUTION_LIMIT", "SHOULD_FOCUS_CLASSIFY", "SPHERE_X_COORD", "SPHERE_Y_COORD", "SPHERE_Z_COORD", "SPHERE_RADIUS", "SHOULD_REFINE_CTF", "DEFOCUS_SEARCH_RANGE", "DEFOCUS_SEARCH_STEP", "AVERAGE_OCCUPANCY", "ESTIMATED_RESOLUTION", "RECONSTRUCTED_VOLUME_ASSET_ID", "RECONSTRUCTION_ID", "SHOULD_AUTOMASK", "SHOULD_REFINE_INPUT_PARAMS", "SHOULD_USE_SUPPLIED_MASK", "MASK_ASSET_ID", "MASK_EDGE_WIDTH", "OUTSIDE_MASK_WEIGHT", "SHOULD_LOWPASS_OUTSIDE_MASK", "MASK_FILTER_RESOLUTION", class_counter + 1, refinement_to_add->reference_volume_ids[class_counter], refinement_to_add->class_refinement_results[class_counter].low_resolution_limit, refinement_to_add->class_refinement_results[class_counter].high_resolution_limit, refinement_to_add->class_refinement_results[class_counter].mask_radius, refinement_to_add->class_refinement_results[class_counter].signed_cc_resolution_limit, refinement_to_add->class_refinement_results[class_counter].global_resolution_limit, refinement_to_add->class_refinement_results[class_counter].global_mask_radius, refinement_to_add->class_refinement_results[class_counter].number_results_to_refine, refinement_to_add->class_refinement_results[class_counter].angular_search_step, refinement_to_add->class_refinement_results[class_counter].search_range_x, refinement_to_add->class_refinement_results[class_counter].search_range_y, refinement_to_add->class_refinement_results[class_counter].classification_resolution_limit, refinement_to_add->class_refinement_results[class_counter].should_focus_classify, refinement_to_add->class_refinement_results[class_counter].sphere_x_coord, refinement_to_add->class_refinement_results[class_counter].sphere_y_coord, refinement_to_add->class_refinement_results[class_counter].sphere_z_coord, refinement_to_add->class_refinement_results[class_counter].sphere_radius, refinement_to_add->class_refinement_results[class_counter].should_refine_ctf, refinement_to_add->class_refinement_results[class_counter].defocus_search_range, refinement_to_add->class_refinement_results[class_counter].defocus_search_step, refinement_to_add->class_refinement_results[class_counter].average_occupancy, estimated_resolution, refinement_to_add->class_refinement_results[class_counter].reconstructed_volume_asset_id, refinement_to_add->class_refinement_results[class_counter].reconstruction_id, refinement_to_add->class_refinement_results[class_counter].should_auto_mask, refinement_to_add->class_refinement_results[class_counter].should_refine_input_params, refinement_to_add->class_refinement_results[class_counter].should_use_supplied_mask, refinement_to_add->class_refinement_results[class_counter].mask_asset_id, refinement_to_add->class_refinement_results[class_counter].mask_edge_width, refinement_to_add->class_refinement_results[class_counter].outside_mask_weight, refinement_to_add->class_refinement_results[class_counter].should_low_pass_filter_mask, refinement_to_add->class_refinement_results[class_counter].filter_resolution);
    }

    for ( class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++ ) {
        BeginBatchInsert(wxString::Format("REFINEMENT_RESULT_%li_%i", refinement_to_add->refinement_id, class_counter), 24, "POSITION_IN_STACK", "PSI", "THETA", "PHI", "XSHIFT", "YSHIFT", "DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "OCCUPANCY", "LOGP", "SIGMA", "SCORE", "IMAGE_IS_ACTIVE", "PIXEL_SIZE", "MICROSCOPE_VOLTAGE", "MICROSCOPE_CS", "AMPLITUDE_CONTRAST", "BEAM_TILT_X", "BEAM_TILT_Y", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y", "ASSIGNED_SUBSET");

        for ( counter = 0; counter < refinement_to_add->number_of_particles; counter++ ) {
            AddToBatchInsert("lrrrrrrrrrrrrrirrrrrrrri", refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].position_in_stack, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].psi, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].theta, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].phi, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].xshift, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].yshift, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].defocus1, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].defocus2, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].defocus_angle, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].phase_shift, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].occupancy, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].logp, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].sigma, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].score, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].image_is_active, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].pixel_size, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].microscope_voltage_kv, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].microscope_spherical_aberration_mm, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].amplitude_contrast, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].beam_tilt_x, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].beam_tilt_y, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].image_shift_x, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].image_shift_y, refinement_to_add->class_refinement_results[class_counter - 1].particle_refinement_results[counter].assigned_subset);
        }

        EndBatchInsert( );
    }

    for ( class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++ ) {

        BeginBatchInsert(wxString::Format("REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_to_add->refinement_id, class_counter), 6, "SHELL", "RESOLUTION", "FSC", "PART_FSC", "PART_SSNR", "REC_SSNR");

        for ( counter = 0; counter <= refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.number_of_points; counter++ ) {
            AddToBatchInsert("lrrrrr", counter, refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_x[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_SSNR.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.rec_SSNR.data_y[counter]);
        }

        EndBatchInsert( );
    }
}

void Database::UpdateRefinementResolutionStatistics(Refinement* refinement_to_add) {
    int  class_counter;
    long counter;

    BeginCommitLocker active_locker(this);

    for ( class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++ ) {
        ExecuteSQL(wxString::Format("DROP TABLE REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_to_add->refinement_id, class_counter));
    }

    for ( class_counter = 1; class_counter <= refinement_to_add->number_of_classes; class_counter++ ) {
        CreateRefinementResolutionStatisticsTable(refinement_to_add->refinement_id, class_counter);

        BeginBatchInsert(wxString::Format("REFINEMENT_RESOLUTION_STATISTICS_%li_%i", refinement_to_add->refinement_id, class_counter), 6, "SHELL", "RESOLUTION", "FSC", "PART_FSC", "PART_SSNR", "REC_SSNR");

        for ( counter = 0; counter <= refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.number_of_points; counter++ ) {
            AddToBatchInsert("lrrrrr", counter, refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_x[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_FSC.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.part_SSNR.data_y[counter], refinement_to_add->class_refinement_results[class_counter - 1].class_resolution_statistics.rec_SSNR.data_y[counter]);
        }

        EndBatchInsert( );
    }
}

Refinement* Database::GetRefinementByID(long wanted_refinement_id, bool include_particle_info) {
    wxString         sql_select_command;
    int              return_code;
    sqlite3_stmt*    list_statement  = NULL;
    Refinement*      temp_refinement = new Refinement;
    RefinementResult temp_result;
    int              class_counter;
    long             current_reference_volume;
    long             number_of_active_images;

    float temp_resolution;
    float temp_fsc;
    float temp_part_fsc;
    float temp_part_ssnr;
    float temp_rec_ssnr;

    bool more_data;

    ClassRefinementResults junk_class_results;
    RefinementResult       junk_result;

    // general data

    sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_LIST WHERE REFINEMENT_ID=%li", wanted_refinement_id);
    Prepare(sql_select_command, &list_statement);

    return_code = Step(list_statement);

    temp_refinement->refinement_id                       = sqlite3_column_int64(list_statement, 0);
    temp_refinement->refinement_package_asset_id         = sqlite3_column_int64(list_statement, 1);
    temp_refinement->name                                = sqlite3_column_text(list_statement, 2);
    temp_refinement->resolution_statistics_are_generated = sqlite3_column_int(list_statement, 3);
    temp_refinement->datetime_of_run.SetFromDOS((unsigned long)sqlite3_column_int64(list_statement, 4));
    temp_refinement->starting_refinement_id           = sqlite3_column_int64(list_statement, 5);
    temp_refinement->number_of_particles              = sqlite3_column_int64(list_statement, 6);
    temp_refinement->number_of_classes                = sqlite3_column_int(list_statement, 7);
    temp_refinement->resolution_statistics_box_size   = sqlite3_column_int(list_statement, 8);
    temp_refinement->resolution_statistics_pixel_size = sqlite3_column_double(list_statement, 9);
    temp_refinement->percent_used                     = sqlite3_column_double(list_statement, 10);

    Finalize(list_statement);

    // now get all the parameters..

    temp_refinement->class_refinement_results.Alloc(temp_refinement->number_of_classes);
    temp_refinement->class_refinement_results.Add(junk_class_results, temp_refinement->number_of_classes);

    // now the details..

    sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_DETAILS_%li", wanted_refinement_id);
    Prepare(sql_select_command, &list_statement);

    for ( class_counter = 0; class_counter < temp_refinement->number_of_classes; class_counter++ ) {
        return_code = Step(list_statement);
        temp_refinement->reference_volume_ids.Add(sqlite3_column_int64(list_statement, 1));
        temp_refinement->class_refinement_results[class_counter].low_resolution_limit            = sqlite3_column_double(list_statement, 2);
        temp_refinement->class_refinement_results[class_counter].high_resolution_limit           = sqlite3_column_double(list_statement, 3);
        temp_refinement->class_refinement_results[class_counter].mask_radius                     = sqlite3_column_double(list_statement, 4);
        temp_refinement->class_refinement_results[class_counter].signed_cc_resolution_limit      = sqlite3_column_double(list_statement, 5);
        temp_refinement->class_refinement_results[class_counter].global_resolution_limit         = sqlite3_column_double(list_statement, 6);
        temp_refinement->class_refinement_results[class_counter].global_mask_radius              = sqlite3_column_double(list_statement, 7);
        temp_refinement->class_refinement_results[class_counter].number_results_to_refine        = sqlite3_column_int(list_statement, 8);
        temp_refinement->class_refinement_results[class_counter].angular_search_step             = sqlite3_column_double(list_statement, 9);
        temp_refinement->class_refinement_results[class_counter].search_range_x                  = sqlite3_column_double(list_statement, 10);
        temp_refinement->class_refinement_results[class_counter].search_range_y                  = sqlite3_column_double(list_statement, 11);
        temp_refinement->class_refinement_results[class_counter].classification_resolution_limit = sqlite3_column_double(list_statement, 12);
        temp_refinement->class_refinement_results[class_counter].should_focus_classify           = sqlite3_column_int(list_statement, 13);
        temp_refinement->class_refinement_results[class_counter].sphere_x_coord                  = sqlite3_column_double(list_statement, 14);
        temp_refinement->class_refinement_results[class_counter].sphere_y_coord                  = sqlite3_column_double(list_statement, 15);
        temp_refinement->class_refinement_results[class_counter].sphere_z_coord                  = sqlite3_column_double(list_statement, 16);
        temp_refinement->class_refinement_results[class_counter].sphere_radius                   = sqlite3_column_double(list_statement, 17);
        temp_refinement->class_refinement_results[class_counter].should_refine_ctf               = sqlite3_column_int(list_statement, 18);
        temp_refinement->class_refinement_results[class_counter].defocus_search_range            = sqlite3_column_double(list_statement, 19);
        temp_refinement->class_refinement_results[class_counter].defocus_search_step             = sqlite3_column_double(list_statement, 20);
        temp_refinement->class_refinement_results[class_counter].average_occupancy               = sqlite3_column_double(list_statement, 21);
        temp_refinement->class_refinement_results[class_counter].estimated_resolution            = sqlite3_column_double(list_statement, 22);
        temp_refinement->class_refinement_results[class_counter].reconstructed_volume_asset_id   = sqlite3_column_int64(list_statement, 23);
        temp_refinement->class_refinement_results[class_counter].reconstruction_id               = sqlite3_column_int64(list_statement, 24);
        temp_refinement->class_refinement_results[class_counter].should_auto_mask                = sqlite3_column_int(list_statement, 25);
        temp_refinement->class_refinement_results[class_counter].should_refine_input_params      = sqlite3_column_int(list_statement, 26);
        temp_refinement->class_refinement_results[class_counter].should_use_supplied_mask        = sqlite3_column_int(list_statement, 27);
        temp_refinement->class_refinement_results[class_counter].mask_asset_id                   = sqlite3_column_int64(list_statement, 28);
        temp_refinement->class_refinement_results[class_counter].mask_edge_width                 = sqlite3_column_double(list_statement, 29);
        temp_refinement->class_refinement_results[class_counter].outside_mask_weight             = sqlite3_column_double(list_statement, 30);
        temp_refinement->class_refinement_results[class_counter].should_low_pass_filter_mask     = sqlite3_column_int(list_statement, 31);
        temp_refinement->class_refinement_results[class_counter].filter_resolution               = sqlite3_column_double(list_statement, 32);
    }

    Finalize(list_statement);

    if ( include_particle_info == true ) {
        for ( class_counter = 0; class_counter < temp_refinement->number_of_classes; class_counter++ ) {
            temp_refinement->class_refinement_results[class_counter].particle_refinement_results.Alloc(temp_refinement->number_of_particles);
            sql_select_command = wxString::Format("SELECT * FROM REFINEMENT_RESULT_%li_%i", temp_refinement->refinement_id, class_counter + 1);

            more_data = BeginBatchSelect(sql_select_command);

            temp_refinement->class_refinement_results[class_counter].average_occupancy = 0.0f;
            number_of_active_images                                                    = 0;

            while ( more_data == true ) {
                more_data = GetFromBatchSelect("lsssssssssssssissssssssi", &temp_result.position_in_stack,
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
                                               &temp_result.image_is_active,
                                               &temp_result.pixel_size,
                                               &temp_result.microscope_voltage_kv,
                                               &temp_result.microscope_spherical_aberration_mm,
                                               &temp_result.amplitude_contrast,
                                               &temp_result.beam_tilt_x,
                                               &temp_result.beam_tilt_y,
                                               &temp_result.image_shift_x,
                                               &temp_result.image_shift_y,
                                               &temp_result.assigned_subset);

                temp_refinement->class_refinement_results[class_counter].particle_refinement_results.Add(temp_result);

                if ( temp_result.image_is_active >= 0.0 ) {
                    temp_refinement->class_refinement_results[class_counter].average_occupancy += temp_result.occupancy;
                    number_of_active_images++;
                }
            }
            if ( number_of_active_images > 0 )
                temp_refinement->class_refinement_results[class_counter].average_occupancy /= float(number_of_active_images);
            EndBatchSelect( );
        }
    }

    // resolution statistics

    for ( class_counter = 0; class_counter < temp_refinement->number_of_classes; class_counter++ ) {
        temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.Init(temp_refinement->resolution_statistics_pixel_size, temp_refinement->resolution_statistics_box_size);

        sql_select_command = wxString::Format("SELECT RESOLUTION, FSC, PART_FSC, PART_SSNR, REC_SSNR FROM REFINEMENT_RESOLUTION_STATISTICS_%li_%i", temp_refinement->refinement_id, class_counter + 1);
        more_data          = BeginBatchSelect(sql_select_command);

        while ( more_data == true ) {
            more_data = GetFromBatchSelect("sssss", &temp_resolution, &temp_fsc, &temp_part_fsc, &temp_part_ssnr, &temp_rec_ssnr);

            temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.AddPoint(temp_resolution, temp_fsc);
            temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_FSC.AddPoint(temp_resolution, temp_part_fsc);
            temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.AddPoint(temp_resolution, temp_part_ssnr);
            temp_refinement->class_refinement_results[class_counter].class_resolution_statistics.rec_SSNR.AddPoint(temp_resolution, temp_rec_ssnr);
        }

        EndBatchSelect( );
    }

    return temp_refinement;
}

void Database::AddClassification(Classification* classification_to_add) {
    MyDebugAssertTrue(classification_to_add->number_of_particles == classification_to_add->classification_results.GetCount( ), "Number of results does not equal number of particles in this classification");
    long counter;

    BeginCommitLocker active_locker(this);

    InsertOrReplace("CLASSIFICATION_LIST", "Plttilllirrrrrrriir", "CLASSIFICATION_ID", "REFINEMENT_PACKAGE_ASSET_ID", "NAME", "CLASS_AVERAGE_FILE", "REFINEMENT_WAS_IMPORTED_OR_GENERATED", "DATETIME_OF_RUN", "STARTING_CLASSIFICATION_ID", "NUMBER_OF_PARTICLES", "NUMBER_OF_CLASSES", "LOW_RESOLUTION_LIMIT", "HIGH_RESOLUTION_LIMIT", "MASK_RADIUS", "ANGULAR_SEARCH_STEP", "SEARCH_RANGE_X", "SEARCH_RANGE_Y", "SMOOTHING_FACTOR", "EXCLUDE_BLANK_EDGES", "AUTO_PERCENT_USED", "PERCENT_USED", classification_to_add->classification_id, classification_to_add->refinement_package_asset_id, classification_to_add->name.ToUTF8( ).data( ), classification_to_add->class_average_file.ToUTF8( ).data( ), classification_to_add->classification_was_imported_or_generated, classification_to_add->datetime_of_run.GetAsDOS( ), classification_to_add->starting_classification_id, classification_to_add->number_of_particles, classification_to_add->number_of_classes, classification_to_add->low_resolution_limit, classification_to_add->high_resolution_limit, classification_to_add->mask_radius, classification_to_add->angular_search_step, classification_to_add->search_range_x, classification_to_add->search_range_y, classification_to_add->smoothing_factor, classification_to_add->exclude_blank_edges, classification_to_add->auto_percent_used, classification_to_add->percent_used);
    CreateClassificationResultTable(classification_to_add->classification_id);

    BeginBatchInsert(wxString::Format("CLASSIFICATION_RESULT_%li", classification_to_add->classification_id), 19, "POSITION_IN_STACK", "PSI", "XSHIFT", "YSHIFT", "BEST_CLASS", "SIGMA", "LOGP", "PIXEL_SIZE", "VOLTAGE", "CS", "AMPLITUDE_CONTRAST", "DEFOCUS_1", "DEFOCUS_2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "BEAM_TILT_X", "BEAM_TILT_Y", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y");

    for ( counter = 0; counter < classification_to_add->classification_results.GetCount( ); counter++ ) {
        AddToBatchInsert("lrrrirrrrrrrrrrrrrr", classification_to_add->classification_results[counter].position_in_stack,
                         classification_to_add->classification_results[counter].psi,
                         classification_to_add->classification_results[counter].xshift,
                         classification_to_add->classification_results[counter].yshift,
                         classification_to_add->classification_results[counter].best_class,
                         classification_to_add->classification_results[counter].sigma,
                         classification_to_add->classification_results[counter].logp,
                         classification_to_add->classification_results[counter].pixel_size,
                         classification_to_add->classification_results[counter].microscope_voltage_kv,
                         classification_to_add->classification_results[counter].microscope_spherical_aberration_mm,
                         classification_to_add->classification_results[counter].amplitude_contrast,
                         classification_to_add->classification_results[counter].defocus_1,
                         classification_to_add->classification_results[counter].defocus_2,
                         classification_to_add->classification_results[counter].defocus_angle,
                         classification_to_add->classification_results[counter].phase_shift,
                         classification_to_add->classification_results[counter].beam_tilt_x,
                         classification_to_add->classification_results[counter].beam_tilt_y,
                         classification_to_add->classification_results[counter].image_shift_x,
                         classification_to_add->classification_results[counter].image_shift_y);
    }

    EndBatchInsert( );
}

Classification* Database::GetClassificationByID(long wanted_classification_id) {
    wxString        sql_select_command;
    int             return_code;
    sqlite3_stmt*   list_statement      = NULL;
    Classification* temp_classification = new Classification;
    bool            more_data;
    long            records_retrieved = 0;

    ClassificationResult junk_result;

    // general data

    sql_select_command = wxString::Format("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=%li", wanted_classification_id);
    Prepare(sql_select_command, &list_statement);
    return_code = Step(list_statement);

    temp_classification->classification_id                        = sqlite3_column_int64(list_statement, 0);
    temp_classification->refinement_package_asset_id              = sqlite3_column_int64(list_statement, 1);
    temp_classification->name                                     = sqlite3_column_text(list_statement, 2);
    temp_classification->class_average_file                       = sqlite3_column_text(list_statement, 3);
    temp_classification->classification_was_imported_or_generated = sqlite3_column_int(list_statement, 4);
    temp_classification->datetime_of_run.SetFromDOS((unsigned long)sqlite3_column_int64(list_statement, 5));
    temp_classification->starting_classification_id = sqlite3_column_int64(list_statement, 6);
    temp_classification->number_of_particles        = sqlite3_column_int64(list_statement, 7);
    temp_classification->number_of_classes          = sqlite3_column_int(list_statement, 8);
    temp_classification->low_resolution_limit       = sqlite3_column_double(list_statement, 9);
    temp_classification->high_resolution_limit      = sqlite3_column_double(list_statement, 10);
    temp_classification->mask_radius                = sqlite3_column_double(list_statement, 11);
    temp_classification->angular_search_step        = sqlite3_column_double(list_statement, 12);
    temp_classification->search_range_x             = sqlite3_column_double(list_statement, 13);
    temp_classification->search_range_y             = sqlite3_column_double(list_statement, 14);
    temp_classification->smoothing_factor           = sqlite3_column_double(list_statement, 15);
    temp_classification->exclude_blank_edges        = sqlite3_column_int(list_statement, 16);
    temp_classification->auto_percent_used          = sqlite3_column_int(list_statement, 17);
    temp_classification->percent_used               = sqlite3_column_double(list_statement, 18);

    Finalize(list_statement);

    // now get all the parameters..

    temp_classification->classification_results.Alloc(temp_classification->number_of_particles);

    sql_select_command = wxString::Format("SELECT * FROM CLASSIFICATION_RESULT_%li", temp_classification->classification_id);
    //wxPrintf("Select command = %s\n", sql_select_command.ToUTF8().data());
    more_data = BeginBatchSelect(sql_select_command);

    while ( more_data == true ) {

        more_data = GetFromBatchSelect("lsssissssssssssssss", &junk_result.position_in_stack,
                                       &junk_result.psi,
                                       &junk_result.xshift,
                                       &junk_result.yshift,
                                       &junk_result.best_class,
                                       &junk_result.sigma,
                                       &junk_result.logp,
                                       &junk_result.pixel_size,
                                       &junk_result.microscope_voltage_kv,
                                       &junk_result.microscope_spherical_aberration_mm,
                                       &junk_result.amplitude_contrast,
                                       &junk_result.defocus_1,
                                       &junk_result.defocus_2,
                                       &junk_result.defocus_angle,
                                       &junk_result.phase_shift,
                                       &junk_result.beam_tilt_x,
                                       &junk_result.beam_tilt_y,
                                       &junk_result.image_shift_x,
                                       &junk_result.image_shift_y);

        temp_classification->classification_results.Add(junk_result);
        records_retrieved++;

        //		wxPrintf("Got info for particle %li\n", junk_result.position_in_stack);
    }

    MyDebugAssertTrue(records_retrieved == temp_classification->number_of_particles, "No of Retrieved Results != No of Particles");

    EndBatchSelect( );
    return temp_classification;
}

void Database::AddClassificationSelection(ClassificationSelection* classification_selection_to_add) {
    BeginCommitLocker active_locker(this);
    InsertOrReplace("CLASSIFICATION_SELECTION_LIST", "ltlllii", "SELECTION_ID", "SELECTION_NAME", "CREATION_DATE", "REFINEMENT_PACKAGE_ID", "CLASSIFICATION_ID", "NUMBER_OF_CLASSES", "NUMBER_OF_SELECTIONS", classification_selection_to_add->selection_id, classification_selection_to_add->name.ToUTF8( ).data( ), classification_selection_to_add->creation_date.GetAsDOS( ), classification_selection_to_add->refinement_package_asset_id, classification_selection_to_add->classification_id, classification_selection_to_add->number_of_classes, classification_selection_to_add->number_of_selections);
    CreateClassificationSelectionTable(classification_selection_to_add->selection_id);

    BeginBatchInsert(wxString::Format("CLASSIFICATION_SELECTION_%li", classification_selection_to_add->selection_id), 1, "CLASS_AVERAGE_NUMBER");

    for ( int counter = 0; counter < classification_selection_to_add->selections.GetCount( ); counter++ ) {
        AddToBatchInsert("l", classification_selection_to_add->selections.Item(counter));
    }

    EndBatchInsert( );
}

void Database::ReturnProcessLockInfo(long& active_process_id, wxString& active_hostname) {
    if ( ReturnSingleIntFromSelectCommand(wxString::Format("select count(*) from PROCESS_LOCK")) != 1 ) {
        active_process_id = -1;
        active_hostname   = "";
    }
    else {
        sqlite3_stmt* list_statement     = NULL;
        wxString      sql_select_command = "SELECT * FROM PROCESS_LOCK";
        Prepare(sql_select_command, &list_statement);
        Step(list_statement);

        active_process_id = sqlite3_column_int64(list_statement, 1);
        active_hostname   = sqlite3_column_text(list_statement, 2);

        Finalize(list_statement);
    }
}

void Database::SetProcessLockInfo(long& active_process_id, wxString& active_hostname) {
    BeginCommitLocker active_locker(this);
    DeleteTable("PROCESS_LOCK");
    CreateProcessLockTable( );
    InsertOrReplace("PROCESS_LOCK", "plt", "NUMBER", "ACTIVE_PROCESS", "ACTIVE_HOST", 1, active_process_id, active_hostname.ToUTF8( ).data( ));
}

void Database::AddRefinementAngularDistribution(AngularDistributionHistogram& histogram_to_add, long refinement_id, int class_number) {
    BeginCommitLocker active_locker(this);
    CreateRefinementAngularDistributionTable(refinement_id, class_number);
    BeginBatchInsert(wxString::Format("REFINEMENT_ANGULAR_DISTRIBUTION_%li_%i", refinement_id, class_number), 2, "BIN_NUMBER", "NUMBER_IN_BIN");

    for ( int bin_counter = 0; bin_counter < histogram_to_add.histogram_data.GetCount( ); bin_counter++ ) {
        AddToBatchInsert("ir", bin_counter, histogram_to_add.histogram_data[bin_counter]);
    }

    EndBatchInsert( );
}

void Database::CopyRefinementAngularDistributions(long refinement_id_to_copy, long refinement_id_to_copy_to, int wanted_class_number) {
    CreateRefinementAngularDistributionTable(refinement_id_to_copy_to, wanted_class_number);
    ExecuteSQL(wxString::Format("INSERT INTO REFINEMENT_ANGULAR_DISTRIBUTION_%li_%i SELECT * FROM REFINEMENT_ANGULAR_DISTRIBUTION_%li_%i", refinement_id_to_copy_to, wanted_class_number, refinement_id_to_copy, wanted_class_number).ToUTF8( ).data( ));
}

void Database::GetRefinementAngularDistributionHistogramData(long wanted_refinement_id, int wanted_class_number, AngularDistributionHistogram& histogram_to_fill) // must be correct size
{
    float temp_float;
    bool  more_data;
    int   bin_counter = 0;

    more_data = BeginBatchSelect(wxString::Format("SELECT NUMBER_IN_BIN FROM REFINEMENT_ANGULAR_DISTRIBUTION_%li_%i", wanted_refinement_id, wanted_class_number));

    while ( more_data == true ) {
        more_data                                     = GetFromBatchSelect("s", &temp_float);
        histogram_to_fill.histogram_data[bin_counter] = temp_float; //fabsf(global_random_number_generator.GetUniformRandom() * 50);
        bin_counter++;
    }

    EndBatchSelect( );
}

bool Database::UpdateSchema(ColumnChanges columns, UpdateProgressTracker* progress_bar, unsigned long total_num_rows, int normalized_increments) {
    using namespace database_schema;
    CreateAllTables( );
    char          format;
    wxString      column_format;
    int           col_counter;
    unsigned long increments_processed    = 0;
    int           current_progress        = 0;
    int           previous_progress       = 0;
    bool          should_update_text      = false; // Modified in UpdateProgressTracker::OnUpdateProgress
    bool          output_pixel_size_added = false;

    // Assitive lambda function used in helping to update the progress bar
    auto calculate_current_percentage = [&increments_processed, &total_num_rows, &normalized_increments]( ) -> int {
        double percent_completion = (double(increments_processed) / double(total_num_rows)) * normalized_increments;
        return static_cast<int>(percent_completion);
    };

    for ( ColumnChange& column : columns ) {
        format        = std::get<COLUMN_CHANGE_TYPE>(column);
        column_format = map_type_char_to_sqlite_string(format);
        ExecuteSQL(wxString::Format("ALTER TABLE %s ADD COLUMN %s %s;", std::get<COLUMN_CHANGE_TABLE>(column), std::get<COLUMN_CHANGE_NAME>(column), column_format));

        // checks for specific problem :-
        // output pixel size was not added onto the end, and it needs to have a value set.

        if ( std::get<COLUMN_CHANGE_NAME>(column) == "OUTPUT_PIXEL_SIZE" )
            output_pixel_size_added = true;

        if ( progress_bar ) {
            increments_processed++; // Not actually updating rows, but altering tables; but, need one variable for tracking
            current_progress = calculate_current_percentage( );

            if ( current_progress > previous_progress ) {
                progress_bar->OnUpdateProgress(current_progress, "Making column changes...", should_update_text);
                previous_progress = current_progress;
            }
        }
    }
    should_update_text = true;

    UpdateVersion( );

    // do the more complicated post work..

    if ( output_pixel_size_added ) {
        // Grab IDs first, or else we can't properly update progress bar
        wxArrayInt ref_pkg_ids = ReturnIntArrayFromSelectCommand("select REFINEMENT_PACKAGE_ASSET_ID from REFINEMENT_PACKAGE_ASSETS");
        ExecuteSQL("drop table if exists cistem_schema_update_temp_table");
        ExecuteSQL("alter table REFINEMENT_PACKAGE_ASSETS rename to cistem_schema_update_temp_table");
        CreateAllTables( ); // should now be correct order but blank..

        std::vector<wxString> all_columns;

        for ( TableData& table : static_tables ) {

            if ( std::get<TABLE_NAME>(table) == "REFINEMENT_PACKAGE_ASSETS" ) {
                for ( col_counter = 0; col_counter < std::get<TABLE_COLUMNS>(table).size( ); col_counter++ ) {
                    all_columns.push_back(std::get<TABLE_COLUMNS>(table)[col_counter]);
                }
            }
        }

        wxString sql_command;
        sql_command = "insert into REFINEMENT_PACKAGE_ASSETS select ";

        for ( col_counter = 0; col_counter < all_columns.size( ); col_counter++ ) {
            sql_command += all_columns[col_counter];

            if ( col_counter < all_columns.size( ) - 1 )
                sql_command += ", ";
            else
                sql_command += " from cistem_schema_update_temp_table;";
        }

        ExecuteSQL(sql_command);
        ExecuteSQL("drop table if exists cistem_schema_update_temp_table");

        // now we need to go and set this value to the output pixel size..
        double current_pixel_size;
        int    num_particles = 0;

        for ( int refinement_package_counter = 0; refinement_package_counter < ref_pkg_ids.GetCount( ); refinement_package_counter++ ) {
            current_pixel_size = ReturnSingleDoubleFromSelectCommand(wxString::Format("select pixel_size from refinement_package_contained_particles_%i", ref_pkg_ids[refinement_package_counter]));
            ExecuteSQL(wxString::Format("update refinement_package_assets set output_pixel_size = %f where refinement_package_asset_id = %i", current_pixel_size, ref_pkg_ids[refinement_package_counter]));

            // Update loading bar
            if ( progress_bar ) {
                num_particles = ReturnSingleIntFromSelectCommand(wxString::Format("select COUNT(*) from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", ref_pkg_ids[refinement_package_counter]));
                increments_processed += num_particles;
                current_progress = calculate_current_percentage( );

                if ( current_progress > previous_progress ) {
                    progress_bar->OnUpdateProgress(current_progress, "Updating refinement package(s) pixel size...", should_update_text);
                    previous_progress = current_progress;
                }
            }
        }
        should_update_text = true;
        // Next, make sure pixel size, aberration, voltage, and amplitude contrast are being updated where needed
        // This ensures updates from beta databases are successful
        {
            double     classification_held_pixel_size      = 0.0;
            double     refinement_held_pixel_size          = 0.0;
            double     contained_particles_pixel_size      = 0.0;
            double     aberration                          = 0.0;
            double     voltage                             = 0.0;
            double     amplitude_contrast                  = 0.0;
            double     defocus_1                           = 0.0;
            double     defocus_2                           = 0.0;
            double     defocus_angle                       = 0.0;
            double     phase_shift                         = 0.0;
            int        corresponding_refinement_package_id = 0;
            int        refinement_number_of_classes        = 0;
            wxArrayInt classification_ids                  = ReturnIntArrayFromSelectCommand(wxString::Format("select CLASSIFICATION_ID from CLASSIFICATION_LIST"));
            wxArrayInt refinement_ids                      = ReturnIntArrayFromSelectCommand(wxString::Format("select REFINEMENT_ID from REFINEMENT_LIST"));

            // Use the refinement_package_asset_id from each table to update the needed variables
            // First do classification results
            for ( int classification_result_counter = 0; classification_result_counter < classification_ids.GetCount( ); classification_result_counter++ ) {
                corresponding_refinement_package_id = ReturnSingleIntFromSelectCommand(wxString::Format("select REFINEMENT_PACKAGE_ASSET_ID from CLASSIFICATION_LIST where CLASSIFICATION_ID = %i", classification_ids[classification_result_counter]));
                classification_held_pixel_size      = ReturnSingleDoubleFromSelectCommand(wxString::Format("select PIXEL_SIZE from CLASSIFICATION_RESULT_%i", classification_ids[classification_result_counter]));
                contained_particles_pixel_size      = ReturnSingleDoubleFromSelectCommand(wxString::Format("select PIXEL_SIZE from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));

                // If pixel size doesn't match, other parameters probably don't -- fix classification tables; fill all NULL columns.
                if ( contained_particles_pixel_size != classification_held_pixel_size ) {
                    aberration         = ReturnSingleDoubleFromSelectCommand(wxString::Format("select SPHERICAL_ABERRATION from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                    voltage            = ReturnSingleDoubleFromSelectCommand(wxString::Format("select MICROSCOPE_VOLTAGE from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                    amplitude_contrast = ReturnSingleDoubleFromSelectCommand(wxString::Format("select AMPLITUDE_CONTRAST from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                    defocus_1          = ReturnSingleDoubleFromSelectCommand(wxString::Format("select DEFOCUS_1 from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                    defocus_2          = ReturnSingleDoubleFromSelectCommand(wxString::Format("select DEFOCUS_2 from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                    defocus_angle      = ReturnSingleDoubleFromSelectCommand(wxString::Format("select DEFOCUS_ANGLE from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                    phase_shift        = ReturnSingleDoubleFromSelectCommand(wxString::Format("select PHASE_SHIFT from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));

                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set PIXEL_SIZE = %f", classification_ids[classification_result_counter], contained_particles_pixel_size));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set CS = %f", classification_ids[classification_result_counter], aberration));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set AMPLITUDE_CONTRAST = %f", classification_ids[classification_result_counter], amplitude_contrast));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set VOLTAGE = %f", classification_ids[classification_result_counter], voltage));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set DEFOCUS_1 = %f", classification_ids[classification_result_counter], defocus_1));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set DEFOCUS_2 = %f", classification_ids[classification_result_counter], defocus_2));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set DEFOCUS_ANGLE = %f", classification_ids[classification_result_counter], defocus_angle));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set PHASE_SHIFT = %f", classification_ids[classification_result_counter], phase_shift));

                    // Then fill in the columns that would remain 0.0 as the values didn't exist in the beta version
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set BEAM_TILT_X = 0.0 where BEAM_TILT_X is null", classification_ids[classification_result_counter]));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set BEAM_TILT_Y = 0.0 where BEAM_TILT_Y is null", classification_ids[classification_result_counter]));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set IMAGE_SHIFT_X = 0.0 where IMAGE_SHIFT_X is null", classification_ids[classification_result_counter]));
                    ExecuteSQL(wxString::Format("update CLASSIFICATION_RESULT_%i set IMAGE_SHIFT_Y = 0.0 where IMAGE_SHIFT_Y is null", classification_ids[classification_result_counter]));
                }

                // Update the loading bar
                if ( progress_bar ) {
                    num_particles = ReturnSingleIntFromSelectCommand(wxString::Format("select COUNT(*) from CLASSIFICATION_RESULT_%i", classification_ids[classification_result_counter]));
                    increments_processed += num_particles;
                    current_progress = calculate_current_percentage( );
                    if ( current_progress > previous_progress ) {
                        progress_bar->OnUpdateProgress(current_progress, "Updating classification result(s)...", should_update_text);
                        previous_progress = current_progress;
                    }
                }
            }
            should_update_text = true;

            // Now do refinement results; first loop over the refinement_ids, then loop over the number of classes.
            for ( int refinement_result_counter = 0; refinement_result_counter < refinement_ids.GetCount( ); refinement_result_counter++ ) {
                refinement_number_of_classes = ReturnSingleIntFromSelectCommand(wxString::Format("select NUMBER_OF_CLASSES from REFINEMENT_LIST where REFINEMENT_ID = %i", refinement_ids[refinement_result_counter])) + 1;
                for ( int refinement_result_class_counter = 1; refinement_result_class_counter < refinement_number_of_classes; refinement_result_class_counter++ ) {
                    corresponding_refinement_package_id = ReturnSingleIntFromSelectCommand(wxString::Format("select REFINEMENT_PACKAGE_ASSET_ID from REFINEMENT_LIST where REFINEMENT_ID = %i", refinement_ids[refinement_result_counter]));
                    refinement_held_pixel_size          = ReturnSingleDoubleFromSelectCommand(wxString::Format("select PIXEL_SIZE from REFINEMENT_RESULT_%i_%i", refinement_ids[refinement_result_counter], refinement_result_class_counter));
                    contained_particles_pixel_size      = ReturnSingleDoubleFromSelectCommand(wxString::Format("select PIXEL_SIZE from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));

                    // If pixel size doesn't match, other parameters probably don't; repeat above process for refinement result tables
                    if ( contained_particles_pixel_size != classification_held_pixel_size ) {
                        aberration         = ReturnSingleDoubleFromSelectCommand(wxString::Format("select SPHERICAL_ABERRATION from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                        voltage            = ReturnSingleDoubleFromSelectCommand(wxString::Format("select MICROSCOPE_VOLTAGE from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));
                        amplitude_contrast = ReturnSingleDoubleFromSelectCommand(wxString::Format("select AMPLITUDE_CONTRAST from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", corresponding_refinement_package_id));

                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set PIXEL_SIZE = %f", refinement_ids[refinement_result_counter], refinement_result_class_counter, contained_particles_pixel_size));
                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set MICROSCOPE_CS = %f", refinement_ids[refinement_result_counter], refinement_result_class_counter, aberration));
                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set AMPLITUDE_CONTRAST = %f", refinement_ids[refinement_result_counter], refinement_result_class_counter, amplitude_contrast));
                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set MICROSCOPE_VOLTAGE = %f", refinement_ids[refinement_result_counter], refinement_result_class_counter, voltage));

                        // Then fill in the columns that would remain 0.0 as the values didn't exist in the beta version
                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set BEAM_TILT_X = 0.0 where BEAM_TILT_X is null", refinement_ids[refinement_result_counter], refinement_result_class_counter));
                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set BEAM_TILT_Y = 0.0 where BEAM_TILT_Y is null", refinement_ids[refinement_result_counter], refinement_result_class_counter));
                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set IMAGE_SHIFT_X = 0.0 where IMAGE_SHIFT_X is null", refinement_ids[refinement_result_counter], refinement_result_class_counter));
                        ExecuteSQL(wxString::Format("update REFINEMENT_RESULT_%i_%i set IMAGE_SHIFT_Y = 0.0 where IMAGE_SHIFT_Y is null", refinement_ids[refinement_result_counter], refinement_result_class_counter));
                    }

                    // Update the loading bar
                    if ( progress_bar ) {
                        num_particles = ReturnSingleIntFromSelectCommand(wxString::Format("select COUNT(*) from REFINEMENT_RESULT_%i_%i", refinement_ids[refinement_result_counter], refinement_result_class_counter));
                        increments_processed += num_particles;
                        current_progress = calculate_current_percentage( );

                        if ( current_progress > previous_progress ) {
                            progress_bar->OnUpdateProgress(current_progress, "Updating refinement result(s)...", should_update_text);
                            previous_progress = current_progress;
                        }
                    }
                }
            }
        }

        // Now update the last 3 columns of the estimated ctf parameters table
        {
            ExecuteSQL(wxString::Format("update ESTIMATED_CTF_PARAMETERS set ICINESS = 0.0 where ICINESS is null"));
            ExecuteSQL(wxString::Format("update ESTIMATED_CTF_PARAMETERS set TILT_ANGLE = 0.0 where TILT_ANGLE is null"));
            ExecuteSQL(wxString::Format("update ESTIMATED_CTF_PARAMETERS set TILT_AXIS = 0.0 where TILT_AXIS is null"));
        }
    }
    return true;
}

bool Database::UpdateVersion( ) {
    ExecuteSQL(wxString::Format("UPDATE MASTER_SETTINGS SET CURRENT_VERSION = %i, CISTEM_VERSION_TEXT = '%s'", INTEGER_DATABASE_VERSION, CISTEM_VERSION_TEXT));
    return true;
}

std::pair<Database::TableChanges, Database::ColumnChanges> Database::CheckSchema( ) {
    using namespace database_schema;
    MyDebugAssertTrue(is_open == true, "database not open!");
    TableChanges  missing_tables;
    ColumnChanges missing_columns;
    // Check Static Tables
    wxArrayString return_strings;
    int           count;
    int           counter;
    int           col_counter;
    for ( TableData& table : static_tables ) {

        return_strings = ReturnStringArrayFromSelectCommand(wxString::Format("SELECT name FROM sqlite_master WHERE type='table' AND name='%s';", std::get<0>(table)));
        if ( return_strings.IsEmpty( ) ) {
            missing_tables.push_back(std::get<TABLE_NAME>(table));
            continue;
        }
        for ( col_counter = 0; col_counter < std::get<TABLE_COLUMNS>(table).size( ); col_counter++ ) {
            auto& column = std::get<TABLE_COLUMNS>(table)[col_counter];
            char  type   = std::get<TABLE_TYPES>(table)[col_counter];

            count = ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) AS CNTREC FROM pragma_table_info('%s') WHERE name='%s';", std::get<0>(table), column));
            if ( count < 1 ) {
                missing_columns.push_back(ColumnChange(std::get<TABLE_NAME>(table), column, type));
            }
        }
    }

    for ( TableData& table : dynamic_tables ) {

        return_strings = ReturnStringArrayFromSelectCommand(wxString::Format("SELECT name FROM sqlite_master WHERE type='table' AND name  LIKE '%s_%';", std::get<0>(table)));
        for ( counter = 0; counter < return_strings.GetCount( ); counter++ ) {
            // Make sure it is not any of the static columns that happen to match
            if ( any_of(static_tables.begin( ), static_tables.end( ), [&](TableData& table) { return return_strings[counter].IsSameAs(std::get<0>(table)); }) ) {
                continue;
            }
            for ( col_counter = 0; col_counter < std::get<TABLE_COLUMNS>(table).size( ); col_counter++ ) {
                auto& column = std::get<TABLE_COLUMNS>(table)[col_counter];
                char  type   = std::get<TABLE_TYPES>(table)[col_counter];
                count        = ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) AS CNTREC FROM pragma_table_info('%s') WHERE name='%s';", return_strings[counter], column));
                if ( count < 1 ) {
                    missing_columns.push_back(ColumnChange(return_strings[counter], column, type));
                }
            }
        }
    }

    return std::pair<TableChanges, ColumnChanges>(missing_tables, missing_columns);
}

BeginCommitLocker::BeginCommitLocker(Database* wanted_database) {
    active_database     = wanted_database;
    already_sent_commit = false;
    active_database->Begin( );
}

BeginCommitLocker::~BeginCommitLocker( ) {
    if ( already_sent_commit == false )
        active_database->Commit( );
}

void BeginCommitLocker::Commit( ) {
    MyDebugAssertTrue(already_sent_commit == false, "Commiting multiple times!");
    already_sent_commit = true;
    active_database->Commit( );
}
