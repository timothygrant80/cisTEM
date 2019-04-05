#define INTEGER_DATABASE_VERSION 2
#define CISTEM_VERSION_TEXT "2.0.0-beta"
#define START_PORT 3000
#define END_PORT 5000
// Define PI constants
#define PI 3.14159265359
#define PIf 3.14159265359f

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

#define MyPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);}
#define MyPrintfGreen(...)  {wxPrintf(ANSI_COLOR_GREEN); wxPrintf(__VA_ARGS__); wxPrintf(ANSI_COLOR_RESET);}
#define MyPrintfCyan(...)  {wxPrintf(ANSI_COLOR_CYAN); wxPrintf(__VA_ARGS__); wxPrintf(ANSI_COLOR_RESET);}
#define MyPrintfRed(...)  {wxPrintf(ANSI_COLOR_RED); wxPrintf(__VA_ARGS__); wxPrintf(ANSI_COLOR_RESET);}

#ifdef DEBUG
#define MyDebugPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);}
#define MyDebugPrint(...)	{wxPrintf(__VA_ARGS__); wxPrintf("\n");}
#define MyDebugAssertTrue(cond, msg, ...) {if ((cond) != true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define MyDebugAssertFalse(cond, msg, ...) {if ((cond) == true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define DEBUG_ABORT abort();
#else
#define MyDebugPrintWithDetails(...)
#define MyDebugPrint(...)
#define MyDebugAssertTrue(cond, msg, ...)
#define MyDebugAssertFalse(cond, msg, ...)
#define DEBUG_ABORT exit(-1);
#endif

WX_DECLARE_OBJARRAY(float, wxArrayFloat);
WX_DECLARE_OBJARRAY(bool, wxArrayBool);

