#ifndef _INCLUDE_LIBTORCH_LIBTORCH_PUSH_MACROS_H_
#define _INCLUDE_LIBTORCH_LIBTORCH_PUSH_MACROS_H_

/**
 * @brief Saves wxWidgets macros that conflict with LibTorch macros
 * to a stack, and undefines them to enable successful compiling with 
 * LibTorch. To view the directives that restore the macros, see 
 * include/libtorch/libtorch_pop_macros.h. 
 * 
 * Having these push/undef/pop operations in separate headers allows the inclusion of 
 * multiple header files that might be needed when utilizing LibTorch
 * libraries.
 * 
 */
#ifdef cisTEM_USING_LIBTORCH
/* Push conflicting macros to a stack for later restoration. */
#pragma push_macro("N_")
#pragma push_macro("UNCHECKED")
#pragma push_macro("NONE")
#pragma push_macro("TEXT")
#pragma push_macro("INTEGER")
#pragma push_macro("FLOAT")
#pragma push_macro("BOOL")
#pragma push_macro("LONG")
#pragma push_macro("DOUBLE")
#pragma push_macro("CHAR")

/* Undef the conflicting macros */
#undef N_
#undef UNCHECKED
#undef NONE
#undef TEXT
#undef INTEGE
#undef FLOAT
#undef BOOL
#undef LONG
#undef DOUBLE

#endif // cisTEM_USING_LIBTORCH
#endif // _INCLUDE_LIBTORCH_LIBTORCH_PUSH_MACROS_H_