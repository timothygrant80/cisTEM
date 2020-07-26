#define INTEGER_DATABASE_VERSION 2
#define CISTEM_VERSION_TEXT "2.0.0-alpha"
#define START_PORT 3000
#define END_PORT 5000
// Define PI constants
#define PI 3.14159265359
#define PIf 3.14159265359f
#define PISQ 9.869604401089
#define PISQf 9.869604401089f
#define SOCKET_FLAGS wxSOCKET_WAITALL | wxSOCKET_BLOCK
//#define SOCKET_FLAGS wxSOCKET_WAITALL

// From Table 2.2 DeGraff
#define RELATIVISTIC_VOLTAGE_100 109784.0f // Volts
#define RELATIVISTIC_VOLTAGE_200 239139.0f
#define RELATIVISTIC_VOLTAGE_300 388062.0f
#define LORENTZ_FACTOR_100 1.196f // M/restMass or 1/sqrt(1-v^2/c^2)
#define LORENTZ_FACTOR_200 1.391f
#define LORENTZ_FACTOR_300 1.587f
#define ELECTRON_REST_MASS 510998.0f // eV

// Used in pdb.h simulate.h
#define NUMBER_OF_ATOM_TYPES 22

#define MINIMUM_BEAM_TILT_SIGNIFICANCE_SCORE 10.0f

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_UNDERLINE     "\e[4m"
#define ANSI_UNDERLINE_OFF "\e[24m"
#define ANSI_BLINK_SLOW "\x1b[5m"
#define ANSI_BLINK_OFF "\x1b[25m"

#define SCALED_IMAGE_SIZE 1200

#define CheckSuccess(bool_to_check) if (bool_to_check == false) {return false;};
#define FUNCTION_DETAILS_AS_WXSTRING wxString::Format("%s (%s:%i)",__PRETTY_FUNCTION__,__FILE__,__LINE__)
#define MyPrintfGreen(...)  {wxPrintf(ANSI_COLOR_GREEN); wxPrintf(__VA_ARGS__); wxPrintf(ANSI_COLOR_RESET);}
#define MyPrintfCyan(...)  {wxPrintf(ANSI_COLOR_CYAN); wxPrintf(__VA_ARGS__); wxPrintf(ANSI_COLOR_RESET);}
#define MyPrintfRed(...)  {wxPrintf(ANSI_COLOR_RED); wxPrintf(__VA_ARGS__); wxPrintf(ANSI_COLOR_RESET);}

#ifdef DEBUG
#define MyDebugPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); StackDump dump(NULL); dump.Walk(2);}
#define MyPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);StackDump dump(NULL); dump.Walk(2);}
#define MyDebugPrint(...)	{wxPrintf(__VA_ARGS__); wxPrintf("\n");}
#define MyDebugAssertTrue(cond, msg, ...) {if ((cond) != true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define MyDebugAssertFalse(cond, msg, ...) {if ((cond) == true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define DEBUG_ABORT {StackDump dump(NULL); dump.Walk(1); abort();}
#else
#define MyPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);}
#define MyDebugPrintWithDetails(...)
#define MyDebugPrint(...)
#define MyDebugAssertTrue(cond, msg, ...)
#define MyDebugAssertFalse(cond, msg, ...)
#define DEBUG_ABORT exit(-1);
#endif

// Asserts that must be true/false always.
#define MyAssertTrue(cond, msg, ...) {if ((cond) != true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define MyAssertFalse(cond, msg, ...) {if ((cond) == true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}

WX_DECLARE_OBJARRAY(float, wxArrayFloat);
WX_DECLARE_OBJARRAY(bool, wxArrayBool);

