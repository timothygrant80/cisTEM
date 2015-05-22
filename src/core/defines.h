#define MyPrintWithDetails(...)	wxLogDebug(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);

#ifdef DEBUG
#define MyDebugPrintWithDetails(...)	wxLogDebug(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);
#define MyDebugPrint(...)	wxLogDebug(__VA_ARGS__); wxPrintf("\n");
#define MyDebugAssertTrue(cond, msg)  if ((cond) != true) { printf("Error! %s\nfailed Assert at %s:%i\n%s\n", msg, __FILE__,__LINE__,__PRETTY_FUNCTION__); exit(-1);}
#define MyDebugAssertFalse(cond, msg)  if ((cond) == true) { printf("Error! %s\nfailed Assert at %s:%i\n%s\n", msg, __FILE__,__LINE__,__PRETTY_FUNCTION__); exit(-1);}
#else
#define MyDebugPrintWithDetails(...)
#define MyDebugPrint(...)
#define MyDebugAssertTrue(cond, msg)
#define MyDebugAssertFalse(cond, msg)
#endif
