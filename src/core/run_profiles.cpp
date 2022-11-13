#include "core_headers.h"

RunCommand::RunCommand( ) {
    command_to_run             = "";
    number_of_copies           = 0;
    delay_time_in_ms           = 0;
    number_of_threads_per_copy = 0;
    override_total_copies      = false;
    overriden_number_of_copies = true;
}

RunCommand::~RunCommand( ) {
}

void RunCommand::SetCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_number_of_threads_per_copy, bool wanted_override_total_copies, int wanted_overriden_number_of_copies, int wanted_delay_time_in_ms) {
    command_to_run             = wanted_command;
    number_of_copies           = wanted_number_of_copies;
    number_of_threads_per_copy = wanted_number_of_threads_per_copy;
    override_total_copies      = wanted_override_total_copies;
    overriden_number_of_copies = false;
    delay_time_in_ms           = wanted_delay_time_in_ms;
}

RunProfile::RunProfile( ) {
    id                     = 1;
    number_of_run_commands = 0;

    manager_command  = "$command";
    run_commands     = new RunCommand[5];
    number_allocated = 5;

    executable_name    = "";
    gui_address        = "";
    controller_address = "";
}

RunProfile::RunProfile(const RunProfile& obj) // copy contructor
{
    id                     = obj.id;
    name                   = obj.name;
    number_of_run_commands = obj.number_of_run_commands;
    manager_command        = obj.manager_command;
    number_allocated       = obj.number_allocated;

    executable_name = obj.executable_name;

    gui_address        = obj.gui_address;
    controller_address = obj.controller_address;

    run_commands = new RunCommand[number_allocated];

    for ( long counter = 0; counter < number_of_run_commands; counter++ ) {
        run_commands[counter] = obj.run_commands[counter];
    }
}

RunProfile::~RunProfile( ) {
    delete[] run_commands;
}

void RunProfile::CheckNumberAndGrow( ) {
    RunCommand* buffer;
    if ( number_of_run_commands >= number_allocated ) {
        // reallocate..

        if ( number_allocated < 100 )
            number_allocated *= 2;
        else
            number_allocated += 100;

        buffer = new RunCommand[number_allocated];

        for ( long counter = 0; counter < number_of_run_commands; counter++ ) {
            buffer[counter] = run_commands[counter];
        }

        delete[] run_commands;
        run_commands = buffer;
    }
}

void RunProfile::AddCommand(RunCommand wanted_command) {
    // check we have enough memory

    CheckNumberAndGrow( );

    // Should be fine for memory, so just add one.

    run_commands[number_of_run_commands] = wanted_command;
    number_of_run_commands++;
}

void RunProfile::AddCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_number_of_threads_per_copy, bool wanted_override_total_copies, int wanted_overriden_number_of_copies, int wanted_delay_time_in_ms) {
    // check we have enough memory

    CheckNumberAndGrow( );

    // Should be fine for memory, so just add one.

    run_commands[number_of_run_commands].command_to_run             = wanted_command;
    run_commands[number_of_run_commands].number_of_copies           = wanted_number_of_copies;
    run_commands[number_of_run_commands].number_of_threads_per_copy = wanted_number_of_threads_per_copy;
    run_commands[number_of_run_commands].override_total_copies      = wanted_override_total_copies;
    run_commands[number_of_run_commands].overriden_number_of_copies = wanted_overriden_number_of_copies;
    run_commands[number_of_run_commands].delay_time_in_ms           = wanted_delay_time_in_ms;
    number_of_run_commands++;
}

void RunProfile::RemoveCommand(int number_to_remove) {
    MyDebugAssertTrue(number_to_remove >= 0 && number_to_remove < number_of_run_commands, "Error: Trying to remove a command that doesnt't exist");

    for ( long counter = number_to_remove; counter < number_of_run_commands - 1; counter++ ) {
        run_commands[counter] = run_commands[counter + 1];
    }

    number_of_run_commands--;
}

void RunProfile::RemoveAll( ) {
    number_of_run_commands = 0;
}

long RunProfile::ReturnTotalJobs( ) {
    long total_jobs = 0;

    for ( long counter = 0; counter < number_of_run_commands; counter++ ) {
        if ( run_commands[counter].override_total_copies == true ) {
            total_jobs += run_commands[counter].overriden_number_of_copies;
        }
        else {
            total_jobs += run_commands[counter].number_of_copies;
        }
    }

    return total_jobs;
}

void RunProfile::SubstituteExecutableName(wxString executable_name) {
    for ( long counter = 0; counter < number_of_run_commands; counter++ ) {
        run_commands[counter].command_to_run.Replace("$command", executable_name);
    }
}

RunProfile& RunProfile::operator=(const RunProfile& t) {
    // Check for self assignment
    if ( this != &t ) {
        if ( this->number_allocated != t.number_allocated ) {
            delete[] this->run_commands;

            this->run_commands     = new RunCommand[t.number_allocated];
            this->number_allocated = t.number_allocated;
        }

        this->id                     = t.id;
        this->name                   = t.name;
        this->number_of_run_commands = t.number_of_run_commands;
        this->manager_command        = t.manager_command;
        this->executable_name        = t.executable_name;
        this->gui_address            = t.gui_address;
        this->controller_address     = t.controller_address;

        for ( long counter = 0; counter < t.number_of_run_commands; counter++ ) {
            this->run_commands[counter] = t.run_commands[counter];
        }
    }

    return *this;
}

RunProfile& RunProfile::operator=(const RunProfile* t) {
    // Check for self assignment
    if ( this != t ) {
        if ( this->number_allocated != t->number_allocated ) {
            delete[] this->run_commands;

            this->run_commands     = new RunCommand[t->number_allocated];
            this->number_allocated = t->number_allocated;
        }

        this->id                     = t->id;
        this->name                   = t->name;
        this->number_of_run_commands = t->number_of_run_commands;
        this->manager_command        = t->manager_command;
        this->executable_name        = t->executable_name;
        this->gui_address            = t->gui_address;
        this->controller_address     = t->controller_address;

        for ( long counter = 0; counter < t->number_of_run_commands; counter++ ) {
            this->run_commands[counter] = t->run_commands[counter];
        }
    }

    return *this;
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

    profile_to_add->CheckNumberAndGrow( );

    // Should be fine for memory, so just add one.

    run_profiles[number_of_run_profiles] = profile_to_add;
    number_of_run_profiles++;

    if ( profile_to_add->id > current_id_number )
        current_id_number = profile_to_add->id;
}

void RunProfileManager::AddBlankProfile( ) {
    // check we have enough memory

    run_profiles->CheckNumberAndGrow( );

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

    run_profiles->CheckNumberAndGrow( );






    wxString execution_command = wxStandardPaths::Get( ).GetExecutablePath( );
    execution_command          = execution_command.BeforeLast('/');
    execution_command += "/$command";

    current_id_number++;
    run_profiles[number_of_run_profiles].id                     = current_id_number;
    run_profiles[number_of_run_profiles].name                   = "Default Local";
    run_profiles[number_of_run_profiles].number_of_run_commands = 0;
    run_profiles[number_of_run_profiles].manager_command        = execution_command;
    run_profiles[number_of_run_profiles].gui_address            = "";
    run_profiles[number_of_run_profiles].controller_address     = "";

    int number_of_cores = wxThread::GetCPUCount( );
    if ( number_of_cores == -1 )
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

        int number_of_cores = wxThread::GetCPUCount( );
        if ( number_of_cores == -1 )
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

wxString RunProfileManager::ReturnProfileName(long wanted_profile) {
    return run_profiles[wanted_profile].name;
}

long RunProfileManager::ReturnProfileID(long wanted_profile) {
    return run_profiles[wanted_profile].id;
}

long RunProfileManager::ReturnTotalJobs(long wanted_profile) {
    return run_profiles[wanted_profile].ReturnTotalJobs( );
}

void RunProfileManager::WriteRunProfilesToDisk(wxString filename, wxArrayInt profiles_to_write) {
    wxTextFile output_file;

    int profile_counter;
    int command_counter;

    if ( DoesFileExist(filename) == true ) {
        output_file.Open(filename);
        output_file.Clear( );
    }
    else {
        output_file.Create(filename);
    }

    output_file.AddLine(wxString::Format("number_of_profiles=%i", int(profiles_to_write.GetCount( ))));

    for ( profile_counter = 0; profile_counter < profiles_to_write.GetCount( ); profile_counter++ ) {
        output_file.AddLine(wxString::Format("profile_%i_name=\"%s\"", profile_counter, run_profiles[profiles_to_write.Item(profile_counter)].name));
        output_file.AddLine(wxString::Format("profile_%i_manager_command=\"%s\"", profile_counter, run_profiles[profiles_to_write.Item(profile_counter)].manager_command));
        output_file.AddLine(wxString::Format("profile_%i_gui_address=\"%s\"", profile_counter, run_profiles[profiles_to_write.Item(profile_counter)].gui_address));
        output_file.AddLine(wxString::Format("profile_%i_controller_address=\"%s\"", profile_counter, run_profiles[profiles_to_write.Item(profile_counter)].controller_address));
        output_file.AddLine(wxString::Format("profile_%i_number_of_run_commands=%li", profile_counter, run_profiles[profiles_to_write.Item(profile_counter)].number_of_run_commands));

        for ( command_counter = 0; command_counter < run_profiles[profiles_to_write.Item(profile_counter)].number_of_run_commands; command_counter++ ) {
            output_file.AddLine(wxString::Format("profile_%i_command_%i_command_to_run=\"%s\"", profile_counter, command_counter, run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].command_to_run));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_number_of_copies=%i", profile_counter, command_counter, run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].number_of_copies));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_number_of_threads_per_copy=%i", profile_counter, command_counter, run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].number_of_threads_per_copy));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_should_override_total_number_of_copies=%i", profile_counter, command_counter, int(run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].override_total_copies)));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_overriden_total_number_of_copies=%i", profile_counter, command_counter, run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].overriden_number_of_copies));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_delay_time_in_ms=%i", profile_counter, command_counter, run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].delay_time_in_ms));
        }
    }

    output_file.Write( );
    output_file.Close( );
}

bool RunProfileManager::ImportRunProfilesFromDisk(wxString filename) {
    wxTextFile input_file;

    int profile_counter;
    int command_counter;

    wxString line_buffer;
    wxString buffer_command_to_run;
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

    input_file.Open(filename);

    // get number of profiles..

    line_buffer = input_file.GetFirstLine( );

    if ( line_buffer.Replace("number_of_profiles=", "") != 1 )
        return false;
    success = line_buffer.ToLong(&number_of_profiles);
    if ( success == false )
        return false;

    RunProfile profiles_buffer[number_of_profiles];

    for ( profile_counter = 0; profile_counter < number_of_profiles; profile_counter++ ) {
        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_name=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].name = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_manager_command=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].manager_command = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_gui_address=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].gui_address = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_controller_address=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].controller_address = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_number_of_run_commands=", profile_counter), "") != 1 )
            return false;
        success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_number_of_run_commands);
        if ( success == false )
            return false;

        wxPrintf("Got here\n");

        for ( command_counter = 0; command_counter < buffer_number_of_run_commands; command_counter++ ) {
            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_command_to_run=\"", profile_counter, command_counter), "") != 1 )
                return false;
            buffer_command_to_run = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_number_of_copies);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_number_of_threads_per_copy=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_number_of_threads);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_should_override_total_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_override_total_jobs);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_overriden_total_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_overriden_total_jobs);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_delay_time_in_ms=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_delay_time_in_ms);
            if ( success == false )
                return false;

            profiles_buffer[profile_counter].AddCommand(buffer_command_to_run, int(buffer_number_of_copies), int(buffer_number_of_threads), bool(buffer_override_total_jobs), int(buffer_overriden_total_jobs), int(buffer_delay_time_in_ms));
        }
    }

    // actually add them..

    for ( profile_counter = 0; profile_counter < number_of_profiles; profile_counter++ ) {
        profiles_buffer[profile_counter].id = current_id_number;
        AddProfile(&profiles_buffer[profile_counter]);
        main_frame->current_project.database.AddOrReplaceRunProfile(ReturnLastProfilePointer( ));
        profiles_buffer[profile_counter].id = current_id_number++;
    }

    return true;
}