#include "command_line_parser.h"
#include "string_functions.h"

#include <algorithm>
#include <cstdio>

CommandLineParser::CommandLineParser( ) {
}

CommandLineParser::CommandLineParser(int argc, char** argv) {
    SetCmdLine(argc, argv);
}

void CommandLineParser::SetCmdLine(int argc, char** argv) {
    arguments_.clear( );
    program_name_ = argc > 0 ? ReturnFileNameFromArgv0(argv[0]) : std::string( );
    for ( int i = 1; i < argc; i++ )
        arguments_.push_back(argv[i]);
}

std::string CommandLineParser::ReturnFileNameFromArgv0(const char* argv0) {
    std::string name(argv0);
    size_t      slash = name.rfind('/');
    return slash == std::string::npos ? name : name.substr(slash + 1);
}

void CommandLineParser::AddSwitch(const std::string& short_name, const std::string& long_name, const std::string& description) {
    Entry entry{ENTRY_SWITCH, short_name, long_name, description, CMD_LINE_VAL_NONE, CMD_LINE_PARAM_OPTIONAL, false, CMD_SWITCH_NOT_FOUND, "", 0, 0.0};
    named_entries_.push_back(entry);
}

void CommandLineParser::AddOption(const std::string& short_name, const std::string& long_name, const std::string& description, CommandLineValueType type, int flags) {
    Entry entry{ENTRY_OPTION, short_name, long_name, description, type, flags, false, CMD_SWITCH_NOT_FOUND, "", 0, 0.0};
    named_entries_.push_back(entry);
}

void CommandLineParser::AddParam(const std::string& name, CommandLineValueType type, int flags) {
    Entry entry{ENTRY_PARAM, "", name, "", type, flags, false, CMD_SWITCH_NOT_FOUND, "", 0, 0.0};
    parameter_entries_.push_back(entry);
}

CommandLineParser::Entry* CommandLineParser::FindByShortName(const std::string& name) {
    for ( Entry& entry : named_entries_ )
        if ( ! entry.short_name.empty( ) && entry.short_name == name )
            return &entry;
    return nullptr;
}

CommandLineParser::Entry* CommandLineParser::FindByLongName(const std::string& name) {
    for ( Entry& entry : named_entries_ )
        if ( ! entry.long_name.empty( ) && entry.long_name == name )
            return &entry;
    return nullptr;
}

const CommandLineParser::Entry* CommandLineParser::FindNamedEntry(const std::string& name) const {
    for ( const Entry& entry : named_entries_ )
        if ( (! entry.short_name.empty( ) && entry.short_name == name) || (! entry.long_name.empty( ) && entry.long_name == name) )
            return &entry;
    return nullptr;
}

std::string CommandLineParser::TypeName(CommandLineValueType type) {
    switch ( type ) {
        case CMD_LINE_VAL_STRING:
            return "str";
        case CMD_LINE_VAL_NUMBER:
            return "num";
        case CMD_LINE_VAL_DOUBLE:
            return "double";
        default:
            return "";
    }
}

bool CommandLineParser::StoreOptionValue(Entry& entry, const std::string& text, std::vector<std::string>& errors) {
    std::string display = entry.long_name.empty( ) ? "-" + entry.short_name : "--" + entry.long_name;
    entry.string_value  = text;
    switch ( entry.type ) {
        case CMD_LINE_VAL_NUMBER:
            if ( ! StringToLong(text, entry.long_value) ) {
                errors.push_back("'" + text + "' is not a valid number for option '" + display + "'");
                return false;
            }
            break;
        case CMD_LINE_VAL_DOUBLE:
            if ( ! StringToDouble(text, entry.double_value) ) {
                errors.push_back("'" + text + "' is not a valid number for option '" + display + "'");
                return false;
            }
            break;
        default:
            break;
    }
    entry.found = true;
    return true;
}

int CommandLineParser::Parse(bool show_usage) {
    std::vector<std::string> errors;
    bool                     help_requested = false;

    for ( Entry& entry : named_entries_ ) {
        entry.found        = false;
        entry.switch_state = CMD_SWITCH_NOT_FOUND;
    }
    parameter_values_.clear( );

    bool only_parameters_follow = false;

    for ( size_t i = 0; i < arguments_.size( ); i++ ) {
        const std::string& argument = arguments_[i];

        if ( ! only_parameters_follow && argument == "--" ) {
            only_parameters_follow = true;
            continue;
        }

        if ( ! only_parameters_follow && argument.size( ) > 1 && argument[0] == '-' && argument != "-" ) {
            bool        is_long = argument.size( ) > 2 && argument[1] == '-';
            std::string body    = argument.substr(is_long ? 2 : 1);
            std::string name    = body;
            std::string value;
            bool        has_inline_value = false;
            bool        negated          = false;

            size_t equals = body.find('=');
            if ( equals != std::string::npos ) {
                name             = body.substr(0, equals);
                value            = body.substr(equals + 1);
                has_inline_value = true;
            }

            if ( name == "h" || name == "help" ) {
                help_requested = true;
                continue;
            }

            Entry* entry = nullptr;
            if ( is_long ) {
                entry = FindByLongName(name);
                if ( entry == nullptr && ! has_inline_value && ! name.empty( ) && name.back( ) == '-' ) {
                    entry = FindByLongName(name.substr(0, name.size( ) - 1));
                    if ( entry != nullptr && entry->kind == ENTRY_SWITCH )
                        negated = true;
                    else
                        entry = nullptr;
                }
            }
            else {
                entry = FindByShortName(name);
                if ( entry == nullptr && ! has_inline_value ) {
                    // "-j4" : a short option immediately followed by its value
                    for ( Entry& candidate : named_entries_ ) {
                        if ( candidate.kind == ENTRY_OPTION && ! candidate.short_name.empty( ) && StartsWith(name, candidate.short_name) ) {
                            entry            = &candidate;
                            value            = name.substr(candidate.short_name.size( ));
                            name             = candidate.short_name;
                            has_inline_value = true;
                            break;
                        }
                    }
                }
                if ( entry == nullptr && ! has_inline_value && ! name.empty( ) && name.back( ) == '-' ) {
                    entry = FindByShortName(name.substr(0, name.size( ) - 1));
                    if ( entry != nullptr && entry->kind == ENTRY_SWITCH )
                        negated = true;
                    else
                        entry = nullptr;
                }
            }

            if ( entry == nullptr ) {
                errors.push_back("Unknown option '" + argument + "'");
                continue;
            }

            if ( entry->kind == ENTRY_SWITCH ) {
                if ( has_inline_value ) {
                    errors.push_back("Switch '" + argument + "' does not take a value");
                    continue;
                }
                entry->found        = true;
                entry->switch_state = negated ? CMD_SWITCH_OFF : CMD_SWITCH_ON;
                continue;
            }

            // an option with a value
            if ( ! has_inline_value ) {
                if ( i + 1 >= arguments_.size( ) ) {
                    errors.push_back("Option '" + argument + "' requires a value");
                    continue;
                }
                value = arguments_[++i];
            }
            StoreOptionValue(*entry, value, errors);
            continue;
        }

        // a positional parameter
        parameter_values_.push_back(argument);
    }

    if ( ! help_requested ) {
        // check mandatory options
        for ( const Entry& entry : named_entries_ ) {
            if ( entry.kind == ENTRY_OPTION && (entry.flags & CMD_LINE_OPTION_MANDATORY) && ! entry.found ) {
                std::string display = entry.long_name.empty( ) ? "-" + entry.short_name : "--" + entry.long_name;
                errors.push_back("The required option '" + display + "' was not specified");
            }
        }

        // check parameters: count and type
        size_t required = 0;
        bool   multiple = false;
        for ( const Entry& entry : parameter_entries_ ) {
            if ( ! (entry.flags & CMD_LINE_PARAM_OPTIONAL) )
                required++;
            if ( entry.flags & CMD_LINE_PARAM_MULTIPLE )
                multiple = true;
        }
        if ( parameter_values_.size( ) < required )
            errors.push_back(Format("Expected at least %zu parameter(s), got %zu", required, parameter_values_.size( )));
        if ( ! multiple && parameter_values_.size( ) > parameter_entries_.size( ) )
            errors.push_back(Format("Unexpected parameter '%s'", parameter_values_[parameter_entries_.size( )]));

        for ( size_t i = 0; i < parameter_values_.size( ) && i < parameter_entries_.size( ); i++ ) {
            const Entry& entry = parameter_entries_[i];
            long         junk_long;
            double       junk_double;
            if ( entry.type == CMD_LINE_VAL_NUMBER && ! StringToLong(parameter_values_[i], junk_long) )
                errors.push_back("'" + parameter_values_[i] + "' is not a valid number for parameter '" + entry.long_name + "'");
            else if ( entry.type == CMD_LINE_VAL_DOUBLE && ! StringToDouble(parameter_values_[i], junk_double) )
                errors.push_back("'" + parameter_values_[i] + "' is not a valid number for parameter '" + entry.long_name + "'");
        }
    }

    for ( const std::string& error : errors )
        fprintf(stderr, "%s\n", error.c_str( ));

    if ( help_requested ) {
        if ( show_usage )
            Usage( );
        return -1;
    }
    if ( ! errors.empty( ) ) {
        if ( show_usage )
            Usage( );
        return int(errors.size( ));
    }
    return 0;
}

std::string CommandLineParser::GetUsageString( ) const {
    std::string usage;
    if ( ! logo_.empty( ) )
        usage += logo_ + "\n";

    usage += "Usage: " + program_name_;
    for ( const Entry& entry : named_entries_ ) {
        std::string display = entry.short_name.empty( ) ? "--" + entry.long_name : "-" + entry.short_name;
        std::string item    = display;
        if ( entry.kind == ENTRY_OPTION )
            item += " <" + TypeName(entry.type) + ">";
        bool optional = ! (entry.kind == ENTRY_OPTION && (entry.flags & CMD_LINE_OPTION_MANDATORY));
        usage += optional ? " [" + item + "]" : " " + item;
    }
    for ( const Entry& entry : parameter_entries_ ) {
        std::string item = entry.long_name.empty( ) ? "<" + TypeName(entry.type) + ">" : entry.long_name;
        if ( entry.flags & CMD_LINE_PARAM_MULTIPLE )
            item += "...";
        usage += (entry.flags & CMD_LINE_PARAM_OPTIONAL) ? " [" + item + "]" : " " + item;
    }
    usage += "\n";

    size_t                   widest = 0;
    std::vector<std::string> names;
    for ( const Entry& entry : named_entries_ ) {
        std::string name;
        if ( ! entry.short_name.empty( ) )
            name += "-" + entry.short_name;
        if ( ! entry.long_name.empty( ) ) {
            if ( ! name.empty( ) )
                name += ", ";
            name += "--" + entry.long_name;
        }
        if ( entry.kind == ENTRY_OPTION )
            name += "=<" + TypeName(entry.type) + ">";
        names.push_back(name);
        widest = std::max(widest, name.size( ));
    }
    for ( size_t i = 0; i < named_entries_.size( ); i++ )
        usage += Format("  %-*s  %s\n", int(widest), names[i], named_entries_[i].description);
    for ( const Entry& entry : parameter_entries_ )
        if ( ! entry.description.empty( ) )
            usage += Format("  %-*s  %s\n", int(widest), entry.long_name, entry.description);
    usage += "  -h, --help" + std::string(widest > 10 ? widest - 10 : 0, ' ') + "  show this help message\n";
    return usage;
}

void CommandLineParser::Usage( ) const {
    fputs(GetUsageString( ).c_str( ), stderr);
}

bool CommandLineParser::Found(const std::string& name) const {
    const Entry* entry = FindNamedEntry(name);
    return entry != nullptr && entry->found;
}

bool CommandLineParser::Found(const std::string& name, std::string* value) const {
    const Entry* entry = FindNamedEntry(name);
    if ( entry == nullptr || ! entry->found || entry->kind != ENTRY_OPTION )
        return false;
    if ( value != nullptr )
        *value = entry->string_value;
    return true;
}

bool CommandLineParser::Found(const std::string& name, long* value) const {
    const Entry* entry = FindNamedEntry(name);
    if ( entry == nullptr || ! entry->found || entry->kind != ENTRY_OPTION )
        return false;
    if ( value != nullptr ) {
        if ( entry->type == CMD_LINE_VAL_NUMBER )
            *value = entry->long_value;
        else if ( entry->type == CMD_LINE_VAL_DOUBLE )
            *value = long(entry->double_value);
        else if ( ! StringToLong(entry->string_value, *value) )
            return false;
    }
    return true;
}

bool CommandLineParser::Found(const std::string& name, double* value) const {
    const Entry* entry = FindNamedEntry(name);
    if ( entry == nullptr || ! entry->found || entry->kind != ENTRY_OPTION )
        return false;
    if ( value != nullptr ) {
        if ( entry->type == CMD_LINE_VAL_DOUBLE )
            *value = entry->double_value;
        else if ( entry->type == CMD_LINE_VAL_NUMBER )
            *value = double(entry->long_value);
        else if ( ! StringToDouble(entry->string_value, *value) )
            return false;
    }
    return true;
}

CommandLineSwitchState CommandLineParser::FoundSwitch(const std::string& name) const {
    const Entry* entry = FindNamedEntry(name);
    if ( entry == nullptr || ! entry->found )
        return CMD_SWITCH_NOT_FOUND;
    return entry->switch_state;
}

std::string CommandLineParser::GetParam(size_t index) const {
    return index < parameter_values_.size( ) ? parameter_values_[index] : std::string( );
}
