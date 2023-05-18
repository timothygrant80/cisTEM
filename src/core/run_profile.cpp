#include "core_headers.h"

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

bool RunProfile::operator==(const RunProfile& t) {
    if ( this == &t )
        return true;

    if ( this->name != t.name )
        return false;
    if ( this->number_of_run_commands != t.number_of_run_commands )
        return false;
    if ( this->manager_command != t.manager_command )
        return false;
    if ( this->executable_name != t.executable_name )
        return false;
    if ( this->gui_address != t.gui_address )
        return false;
    if ( this->controller_address != t.controller_address )
        return false;

    for ( long counter = 0; counter < t.number_of_run_commands; counter++ ) {
        if ( this->run_commands[counter] != t.run_commands[counter] )
            return false;
    }

    return true;
}

bool RunProfile::operator!=(const RunProfile& t) {
    return ! (*this == t);
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

bool RunProfile::operator==(const RunProfile* t) {
    if ( this == t )
        return true;

    if ( this->name != t->name )
        return false;
    if ( this->number_of_run_commands != t->number_of_run_commands )
        return false;
    if ( this->manager_command != t->manager_command )
        return false;
    if ( this->executable_name != t->executable_name )
        return false;
    if ( this->gui_address != t->gui_address )
        return false;
    if ( this->controller_address != t->controller_address )
        return false;

    for ( long counter = 0; counter < t->number_of_run_commands; counter++ ) {
        if ( this->run_commands[counter] != t->run_commands[counter] )
            return false;
    }

    return true;
}

bool RunProfile::operator!=(const RunProfile* t) {
    return ! (*this == t);
}
