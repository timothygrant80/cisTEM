#define INTEGER_DATABASE_VERSION 1
#define START_PORT 3000
#define END_PORT 5000
#define PI 3.141592

#define MyPrintWithDetails(...)	wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);

#ifdef DEBUG
#define MyDebugPrintWithDetails(...)	wxLogDebug(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);
#define MyDebugPrint(...)	wxLogDebug(__VA_ARGS__); wxPrintf("\n");
#define MyDebugAssertTrue(cond, msg, ...) if ((cond) != true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); abort();}
#define MyDebugAssertFalse(cond, msg, ...) if ((cond) == true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); abort();}
#else
#define MyDebugPrintWithDetails(...)
#define MyDebugPrint(...)
#define MyDebugAssertTrue(cond, msg)
#define MyDebugAssertFalse(cond, msg)
#endif
