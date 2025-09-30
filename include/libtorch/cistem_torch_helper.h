#ifndef _INCLUDE_LIBTORCH_CISTEM_TORCH_HELPER_H_
#define _INCLUDE_LIBTORCH_CISTEM_TORCH_HELPER_H_

// This header provides a clean interface for including LibTorch in cisTEM code.
// It handles macro conflicts between cisTEM/wxWidgets and LibTorch headers.
//
// Usage:
//   #ifdef cisTEM_USING_LIBTORCH
//   #include "libtorch/cistem_torch_helper.h"
//   #endif

#ifdef cisTEM_USING_LIBTORCH

// Save all macros that conflict with LibTorch
// N_ is from wxWidgets translation.h (i18n)
// Type macros (NONE, TEXT, INTEGER, FLOAT, BOOL, LONG, DOUBLE, CHAR) are from core/defines.h
#pragma push_macro("N_")
#pragma push_macro("NONE")
#pragma push_macro("TEXT")
#pragma push_macro("INTEGER")
#pragma push_macro("FLOAT")
#pragma push_macro("BOOL")
#pragma push_macro("LONG")
#pragma push_macro("DOUBLE")
#pragma push_macro("CHAR")

// Undefine conflicting macros
#undef N_
#undef NONE
#undef TEXT
#undef INTEGER
#undef FLOAT
#undef BOOL
#undef LONG
#undef DOUBLE
#undef CHAR

// Include LibTorch headers
#include <torch/torch.h>

// Restore all macros for use in cisTEM code
#pragma pop_macro("CHAR")
#pragma pop_macro("DOUBLE")
#pragma pop_macro("LONG")
#pragma pop_macro("BOOL")
#pragma pop_macro("FLOAT")
#pragma pop_macro("INTEGER")
#pragma pop_macro("TEXT")
#pragma pop_macro("NONE")
#pragma pop_macro("N_")

#endif // cisTEM_USING_LIBTORCH

#endif // _INCLUDE_LIBTORCH_CISTEM_TORCH_HELPER_H_