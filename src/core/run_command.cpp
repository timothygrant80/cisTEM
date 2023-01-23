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

bool RunCommand::operator==(const RunCommand& other) const {
    if ( command_to_run != other.command_to_run ) {
        return false;
    }
    if ( number_of_copies != other.number_of_copies ) {
        return false;
    }
    if ( delay_time_in_ms != other.delay_time_in_ms ) {
        return false;
    }
    if ( number_of_threads_per_copy != other.number_of_threads_per_copy ) {
        return false;
    }
    if ( override_total_copies != other.override_total_copies ) {
        return false;
    }
    if ( overriden_number_of_copies != other.overriden_number_of_copies ) {
        return false;
    }

    return true;
}

bool RunCommand::operator!=(const RunCommand& other) const {
    return ! (*this == other);
}

void RunCommand::SetCommand(wxString wanted_command, int wanted_number_of_copies, int wanted_number_of_threads_per_copy, bool wanted_override_total_copies, int wanted_overriden_number_of_copies, int wanted_delay_time_in_ms) {
    command_to_run             = wanted_command;
    number_of_copies           = wanted_number_of_copies;
    number_of_threads_per_copy = wanted_number_of_threads_per_copy;
    override_total_copies      = wanted_override_total_copies;
    overriden_number_of_copies = false;
    delay_time_in_ms           = wanted_delay_time_in_ms;
}
