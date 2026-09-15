#ifndef _src_core_defines_h_
#define _src_core_defines_h_

// clang-format off

#include "../constants/constants.h"

#define INTEGER_DATABASE_VERSION 2
#define START_PORT 3000
#define END_PORT 5000
// Define PI constants
#define PI 3.14159265359
#define PIf 3.14159265359f
#define PISQ 9.869604401089
#define PISQf 9.869604401089f

// Types of noise distributions
namespace cistem {

  enum NoiseType : int { UNIFORM, GAUSSIAN, POISSON, EXPONENTIAL, GAMMA };

}

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
#define ANSI_BOLD "\033[1m"
#define ANSI_BOLD_OFF "\033[0m"


#define SCALED_IMAGE_SIZE 1200

#define CheckSuccess(bool_to_check) if (bool_to_check == false) {return false;};
#define FUNCTION_DETAILS_AS_STRING Format("%s (%s:%i)",__PRETTY_FUNCTION__,__FILE__,__LINE__)

#define MyPrintfGreen(...)  {Printf(ANSI_COLOR_GREEN); Printf(__VA_ARGS__); Printf(ANSI_COLOR_RESET);}
#define MyPrintfCyan(...)  {Printf(ANSI_COLOR_CYAN); Printf(__VA_ARGS__); Printf(ANSI_COLOR_RESET);}
#define MyPrintfRed(...)  {Printf(ANSI_COLOR_RED); Printf(__VA_ARGS__); Printf(ANSI_COLOR_RESET);}

#ifdef DEBUG
#define MyDebugWarnThreadSafety(cond, ...) {if (cond) { FPrintf(stderr, "Warning: Potential thread safety issue detected, this object should be declared private or explicitly constructed in thread parallel region %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); }}
#define MyDebugPrintWithDetails(...)	{Printf(__VA_ARGS__); Printf("From %s:%i\n%s\n\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); StackDump dump(NULL); dump.Walk(2);}
#define MyPrintWithDetails(...)	{Printf(__VA_ARGS__); Printf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);StackDump dump(NULL); dump.Walk(2);}
#define MyDebugPrint(...)	{Printf(__VA_ARGS__); Printf("\n");}
#define MyDebugAssertTrue(cond, msg, ...) {if ((cond) != true) { Printf("\n" msg, ##__VA_ARGS__); Printf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define MyDebugAssertFalse(cond, msg, ...) {if ((cond) == true) { Printf("\n" msg, ##__VA_ARGS__); Printf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define DEBUG_ABORT {StackDump dump(NULL); dump.Walk(1); abort();}
#else
#define MyDebugWarnThreadSafety(cond, ...) 
#define MyPrintWithDetails(...)	{Printf(__VA_ARGS__); Printf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);}
#define MyDebugPrintWithDetails(...)
#define MyDebugPrint(...)
#define MyDebugAssertTrue(cond, msg, ...)
#define MyDebugAssertFalse(cond, msg, ...)
#define DEBUG_ABORT exit(-1);
#endif

// Asserts that must be true/false always.
#define MyAssertTrue(cond, msg, ...) {if ((cond) != true) { Printf("\n" msg, ##__VA_ARGS__); Printf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define MyAssertFalse(cond, msg, ...) {if ((cond) == true) { Printf("\n" msg, ##__VA_ARGS__); Printf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
// clang-format on

#endif /* defines.h */