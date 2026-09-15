#include "../core_headers.h"

// wxTextFile replacements: it read a whole file into memory as a list of lines (with the line
// terminators stripped) and wrote it back out with the native terminator, which on Linux is "\n".

static std::vector<std::string> ReadFileAsLines(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream            input_stream(filename);
    std::string              line;

    while ( std::getline(input_stream, line) ) {
        if ( ! line.empty( ) && line.back( ) == '\r' )
            line.pop_back( );
        lines.push_back(line);
    }

    return lines;
}

// Returns "" past the end of the file rather than reading out of bounds; a caller that reaches
// that point has a truncated file and will fail its next format check.
static std::string ReturnNextLine(const std::vector<std::string>& lines, size_t& current_line) {
    if ( current_line >= lines.size( ) )
        return std::string( );
    return lines[current_line++];
}

static void AddLine(std::ofstream& output_stream, const std::string& line) {
    output_stream << line << "\n";
}

RunProfileManager::RunProfileManager( ) {
    current_id_number      = 0;
    number_of_run_profiles = 0;
    run_profiles           = new RunProfile[5];
    number_allocated       = 5;
}

RunProfileManager::~RunProfileManager( ) {
    delete[] run_profiles;
}

void RunProfileManager::AddProfile(RunProfile* profile_to_add) {
    // check we have enough memory
    CheckNumberAndGrow( );

    // Should be fine for memory, so just add one.

    run_profiles[number_of_run_profiles] = profile_to_add;
    number_of_run_profiles++;

    if ( profile_to_add->id > current_id_number )
        current_id_number = profile_to_add->id;
}

void RunProfileManager::AddBlankProfile( ) {
    // check we have enough memory
    CheckNumberAndGrow( );

    current_id_number++;
    run_profiles[number_of_run_profiles].id                     = current_id_number;
    run_profiles[number_of_run_profiles].name                   = "New Profile";
    run_profiles[number_of_run_profiles].number_of_run_commands = 0;
    run_profiles[number_of_run_profiles].manager_command        = "$command";
    run_profiles[number_of_run_profiles].gui_address            = "";
    run_profiles[number_of_run_profiles].controller_address     = "";

    run_profiles[number_of_run_profiles].AddCommand("$command", 2, 1, false, 0, 10);

    number_of_run_profiles++;
}

void RunProfileManager::AddDefaultLocalProfile( ) {
    // check we have enough memory
    CheckNumberAndGrow( );

    std::error_code error_code;
    std::string     execution_command = std::filesystem::read_symlink("/proc/self/exe", error_code).string( );
    execution_command                 = BeforeLast(execution_command, '/');
    execution_command += "/$command";

    current_id_number++;
    run_profiles[number_of_run_profiles].id                     = current_id_number;
    run_profiles[number_of_run_profiles].name                   = "Default Local";
    run_profiles[number_of_run_profiles].number_of_run_commands = 0;
    run_profiles[number_of_run_profiles].manager_command        = execution_command;
    run_profiles[number_of_run_profiles].gui_address            = "";
    run_profiles[number_of_run_profiles].controller_address     = "";

    int number_of_cores = int(std::thread::hardware_concurrency( ));
    if ( number_of_cores < 1 )
        number_of_cores = 1;
    number_of_cores++;

    run_profiles[number_of_run_profiles].AddCommand(execution_command, number_of_cores, 1, false, 0, 10);
    number_of_run_profiles++;

    bool make_recon = false;
    if ( make_recon ) {
        current_id_number++;
        run_profiles[number_of_run_profiles].id                     = current_id_number;
        run_profiles[number_of_run_profiles].name                   = "Default Reconstruction";
        run_profiles[number_of_run_profiles].number_of_run_commands = 0;
        run_profiles[number_of_run_profiles].manager_command        = execution_command;
        run_profiles[number_of_run_profiles].gui_address            = "";
        run_profiles[number_of_run_profiles].controller_address     = "";

        int number_of_cores = int(std::thread::hardware_concurrency( ));
        if ( number_of_cores < 1 )
            number_of_cores = 1;
        number_of_cores++;

        run_profiles[number_of_run_profiles].AddCommand(execution_command, 1, (number_of_cores / 2), false, 0, 10);
        number_of_run_profiles++;
    }
}

RunProfile* RunProfileManager::ReturnLastProfilePointer( ) {
    return &run_profiles[number_of_run_profiles - 1];
}

RunProfile* RunProfileManager::ReturnProfilePointer(int wanted_profile) {
    return &run_profiles[wanted_profile];
}

void RunProfileManager::RemoveProfile(int number_to_remove) {
    MyDebugAssertTrue(number_to_remove >= 0 && number_to_remove < number_of_run_profiles, "Error: Trying to remove a profile that doesnt't exist");

    for ( long counter = number_to_remove; counter < number_of_run_profiles - 1; counter++ ) {
        run_profiles[counter] = run_profiles[counter + 1];
    }

    number_of_run_profiles--;
}

void RunProfileManager::RemoveAllProfiles( ) {
    number_of_run_profiles = 0;

    if ( number_allocated > 100 ) {
        delete[] run_profiles;
        number_allocated = 100;
        run_profiles     = new RunProfile[number_allocated];
    }
}

std::string RunProfileManager::ReturnProfileName(long wanted_profile) {
    return run_profiles[wanted_profile].name;
}

long RunProfileManager::ReturnProfileID(long wanted_profile) {
    return run_profiles[wanted_profile].id;
}

long RunProfileManager::ReturnTotalJobs(long wanted_profile) {
    return run_profiles[wanted_profile].ReturnTotalJobs( );
}

void RunProfileManager::WriteRunProfilesToDisk(std::string filename, std::vector<int> profiles_to_write) {
    std::ofstream output_file(filename, std::ios::out | std::ios::trunc);

    int profile_counter;
    int command_counter;

    AddLine(output_file, Format("number_of_profiles=%i", int(profiles_to_write.size())));

    for ( profile_counter = 0; profile_counter < int(profiles_to_write.size( )); profile_counter++ ) {
        AddLine(output_file, Format("profile_%i_name=\"%s\"", profile_counter, run_profiles[profiles_to_write[profile_counter]].name));
        AddLine(output_file, Format("profile_%i_manager_command=\"%s\"", profile_counter, run_profiles[profiles_to_write[profile_counter]].manager_command));
        AddLine(output_file, Format("profile_%i_gui_address=\"%s\"", profile_counter, run_profiles[profiles_to_write[profile_counter]].gui_address));
        AddLine(output_file, Format("profile_%i_controller_address=\"%s\"", profile_counter, run_profiles[profiles_to_write[profile_counter]].controller_address));
        AddLine(output_file, Format("profile_%i_number_of_run_commands=%li", profile_counter, run_profiles[profiles_to_write[profile_counter]].number_of_run_commands));

        for ( command_counter = 0; command_counter < run_profiles[profiles_to_write[profile_counter]].number_of_run_commands; command_counter++ ) {
            AddLine(output_file, Format("profile_%i_command_%i_command_to_run=\"%s\"", profile_counter, command_counter, run_profiles[profiles_to_write[profile_counter]].run_commands[command_counter].command_to_run));
            AddLine(output_file, Format("profile_%i_command_%i_number_of_copies=%i", profile_counter, command_counter, run_profiles[profiles_to_write[profile_counter]].run_commands[command_counter].number_of_copies));
            AddLine(output_file, Format("profile_%i_command_%i_number_of_threads_per_copy=%i", profile_counter, command_counter, run_profiles[profiles_to_write[profile_counter]].run_commands[command_counter].number_of_threads_per_copy));
            AddLine(output_file, Format("profile_%i_command_%i_should_override_total_number_of_copies=%i", profile_counter, command_counter, int(run_profiles[profiles_to_write[profile_counter]].run_commands[command_counter].override_total_copies)));
            AddLine(output_file, Format("profile_%i_command_%i_overriden_total_number_of_copies=%i", profile_counter, command_counter, run_profiles[profiles_to_write[profile_counter]].run_commands[command_counter].overriden_number_of_copies));
            AddLine(output_file, Format("profile_%i_command_%i_delay_time_in_ms=%i", profile_counter, command_counter, run_profiles[profiles_to_write[profile_counter]].run_commands[command_counter].delay_time_in_ms));
        }
    }

    output_file.close( );
}

bool RunProfileManager::ImportRunProfilesFromDisk(std::string filename) {
    std::vector<std::string> input_file;
    size_t                   current_line = 0;

    int profile_counter;
    int command_counter;

    std::string line_buffer;
    std::string buffer_command_to_run;
    long     buffer_number_of_run_commands;
    long     buffer_number_of_copies;
    long     buffer_number_of_threads;
    long     buffer_override_total_jobs;
    long     buffer_overriden_total_jobs;
    long     buffer_delay_time_in_ms;
    long     number_of_profiles;
    bool     success;

    if ( DoesFileExist(filename) == false )
        return false;

    input_file = ReadFileAsLines(filename);

    // get number of profiles..

    line_buffer = ReturnNextLine(input_file, current_line);

    if ( ReplaceAll(line_buffer, "number_of_profiles=", "") != 1 )
        return false;
    success = StringToLong(line_buffer, number_of_profiles);
    if ( success == false )
        return false;

    RunProfile profiles_buffer[number_of_profiles];

    for ( profile_counter = 0; profile_counter < number_of_profiles; profile_counter++ ) {
        line_buffer = ReturnNextLine(input_file, current_line);
        if ( ReplaceAll(line_buffer, Format("profile_%i_name=\"", profile_counter), "") != 1 )
            return false;
        Trim(line_buffer);
        profiles_buffer[profile_counter].name = Truncate(line_buffer, line_buffer.length( ) - 1);

        line_buffer = ReturnNextLine(input_file, current_line);
        if ( ReplaceAll(line_buffer, Format("profile_%i_manager_command=\"", profile_counter), "") != 1 )
            return false;
        Trim(line_buffer);
        profiles_buffer[profile_counter].manager_command = Truncate(line_buffer, line_buffer.length( ) - 1);

        line_buffer = ReturnNextLine(input_file, current_line);
        if ( ReplaceAll(line_buffer, Format("profile_%i_gui_address=\"", profile_counter), "") != 1 )
            return false;
        Trim(line_buffer);
        profiles_buffer[profile_counter].gui_address = Truncate(line_buffer, line_buffer.length( ) - 1);

        line_buffer = ReturnNextLine(input_file, current_line);
        if ( ReplaceAll(line_buffer, Format("profile_%i_controller_address=\"", profile_counter), "") != 1 )
            return false;
        Trim(line_buffer);
        profiles_buffer[profile_counter].controller_address = Truncate(line_buffer, line_buffer.length( ) - 1);

        line_buffer = ReturnNextLine(input_file, current_line);
        if ( ReplaceAll(line_buffer, Format("profile_%i_number_of_run_commands=", profile_counter), "") != 1 )
            return false;
        success = StringToLong(Trim(line_buffer), buffer_number_of_run_commands);
        if ( success == false )
            return false;

        for ( command_counter = 0; command_counter < buffer_number_of_run_commands; command_counter++ ) {
            line_buffer = ReturnNextLine(input_file, current_line);
            if ( ReplaceAll(line_buffer, Format("profile_%i_command_%i_command_to_run=\"", profile_counter, command_counter), "") != 1 )
                return false;
            Trim(line_buffer);
            buffer_command_to_run = Truncate(line_buffer, line_buffer.length( ) - 1);

            line_buffer = ReturnNextLine(input_file, current_line);
            if ( ReplaceAll(line_buffer, Format("profile_%i_command_%i_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = StringToLong(Trim(line_buffer), buffer_number_of_copies);
            if ( success == false )
                return false;

            line_buffer = ReturnNextLine(input_file, current_line);
            if ( ReplaceAll(line_buffer, Format("profile_%i_command_%i_number_of_threads_per_copy=", profile_counter, command_counter), "") != 1 )
                return false;
            success = StringToLong(Trim(line_buffer), buffer_number_of_threads);
            if ( success == false )
                return false;

            line_buffer = ReturnNextLine(input_file, current_line);
            if ( ReplaceAll(line_buffer, Format("profile_%i_command_%i_should_override_total_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = StringToLong(Trim(line_buffer), buffer_override_total_jobs);
            if ( success == false )
                return false;

            line_buffer = ReturnNextLine(input_file, current_line);
            if ( ReplaceAll(line_buffer, Format("profile_%i_command_%i_overriden_total_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = StringToLong(Trim(line_buffer), buffer_overriden_total_jobs);
            if ( success == false )
                return false;

            line_buffer = ReturnNextLine(input_file, current_line);
            if ( ReplaceAll(line_buffer, Format("profile_%i_command_%i_delay_time_in_ms=", profile_counter, command_counter), "") != 1 )
                return false;
            success = StringToLong(Trim(line_buffer), buffer_delay_time_in_ms);
            if ( success == false )
                return false;

            profiles_buffer[profile_counter].AddCommand(buffer_command_to_run, int(buffer_number_of_copies), int(buffer_number_of_threads), bool(buffer_override_total_jobs), int(buffer_overriden_total_jobs), int(buffer_delay_time_in_ms));
        }
    }

    // actually add them..

    for ( profile_counter = 0; profile_counter < number_of_profiles; profile_counter++ ) {
        profiles_buffer[profile_counter].id = current_id_number;
        AddProfile(&profiles_buffer[profile_counter]);
        profiles_buffer[profile_counter].id = current_id_number++;
    }

    return true;
}

void RunProfileManager::CheckNumberAndGrow( ) {
    RunProfile* buffer;
    if ( number_of_run_profiles >= number_allocated ) {
        // reallocate..

        if ( number_allocated < 100 )
            number_allocated *= 2;
        else
            number_allocated += 100;

        buffer = new RunProfile[number_allocated];

        for ( long counter = 0; counter < number_of_run_profiles; counter++ ) {
            buffer[counter] = run_profiles[counter];
        }

        delete[] run_profiles;
        run_profiles = buffer;
    }
}