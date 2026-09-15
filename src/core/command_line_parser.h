#ifndef _SRC_CORE_COMMAND_LINE_PARSER_H_
#define _SRC_CORE_COMMAND_LINE_PARSER_H_

/*
 * A small command-line parser with the subset of the wxCmdLineParser interface cisTEM uses:
 * positional parameters, options taking a value (--name=value, --name value, -n value, -nvalue)
 * and switches (--flag, -f; --flag- turns a switch explicitly off). "-h" / "--help" print the
 * usage text and make Parse() return -1.
 */

#include <string>
#include <vector>

enum CommandLineValueType {
    CMD_LINE_VAL_NONE = 0,
    CMD_LINE_VAL_STRING,
    CMD_LINE_VAL_NUMBER, // long
    CMD_LINE_VAL_DOUBLE
};

enum CommandLineEntryFlags {
    CMD_LINE_PARAM_OPTIONAL   = 1, // for parameters and options: may be omitted (options are optional by default)
    CMD_LINE_OPTION_MANDATORY = 2, // for options: must be given
    CMD_LINE_PARAM_MULTIPLE   = 4 // for the last parameter: may be repeated
};

enum CommandLineSwitchState {
    CMD_SWITCH_OFF       = -1, // given as "--name-"
    CMD_SWITCH_NOT_FOUND = 0,
    CMD_SWITCH_ON        = 1
};

class CommandLineParser {
  public:
    CommandLineParser( );
    CommandLineParser(int argc, char** argv);

    void SetCmdLine(int argc, char** argv);

    /// Sets the text printed before the option list by Usage().
    void SetLogo(const std::string& logo) { logo_ = logo; }

    void AddSwitch(const std::string& short_name, const std::string& long_name = "", const std::string& description = "");
    void AddLongSwitch(const std::string& long_name, const std::string& description = "") { AddSwitch("", long_name, description); }

    void AddOption(const std::string& short_name, const std::string& long_name, const std::string& description = "", CommandLineValueType type = CMD_LINE_VAL_STRING, int flags = CMD_LINE_PARAM_OPTIONAL);
    void AddLongOption(const std::string& long_name, const std::string& description = "", CommandLineValueType type = CMD_LINE_VAL_STRING, int flags = CMD_LINE_PARAM_OPTIONAL) { AddOption("", long_name, description, type, flags); }

    void AddParam(const std::string& name = "", CommandLineValueType type = CMD_LINE_VAL_STRING, int flags = 0);

    /**
     * Parse the command line given to SetCmdLine. Returns 0 on success, -1 if help was requested
     * and a positive count of errors otherwise. With show_usage the usage text is printed in the
     * last two cases, and the specific errors always go to stderr.
     */
    int Parse(bool show_usage = true);

    std::string GetUsageString( ) const;
    void        Usage( ) const;

    /// True if the named switch or option (short or long name) was given.
    bool Found(const std::string& name) const;
    /// Value of a string option; false if not given.
    bool Found(const std::string& name, std::string* value) const;
    /// Value of a number option; false if not given.
    bool Found(const std::string& name, long* value) const;
    /// Value of a double option; false if not given.
    bool Found(const std::string& name, double* value) const;

    CommandLineSwitchState FoundSwitch(const std::string& name) const;

    size_t      GetParamCount( ) const { return parameter_values_.size( ); }
    std::string GetParam(size_t index = 0) const;

  private:
    enum EntryKind { ENTRY_SWITCH,
                     ENTRY_OPTION,
                     ENTRY_PARAM };

    struct Entry {
        EntryKind            kind;
        std::string          short_name;
        std::string          long_name;
        std::string          description;
        CommandLineValueType type;
        int                  flags;

        bool                   found;
        CommandLineSwitchState switch_state;
        std::string            string_value;
        long                   long_value;
        double                 double_value;
    };

    const Entry* FindNamedEntry(const std::string& name) const;
    Entry*       FindByShortName(const std::string& name);
    Entry*       FindByLongName(const std::string& name);
    bool         StoreOptionValue(Entry& entry, const std::string& text, std::vector<std::string>& errors);
    static std::string TypeName(CommandLineValueType type);
    static std::string ReturnFileNameFromArgv0(const char* argv0);

    std::vector<std::string> arguments_; // without argv[0]
    std::string              program_name_;
    std::string              logo_;
    std::vector<Entry>       named_entries_;
    std::vector<Entry>       parameter_entries_;
    std::vector<std::string> parameter_values_;
};

#endif // _SRC_CORE_COMMAND_LINE_PARSER_H_
