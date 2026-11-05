#ifndef _INCLUDE_LIBTORCH_LIBTORCH_POP_MACROS_H_
#define _INCLUDE_LIBTORCH_LIBTORCH_POP_MACROS_H_

#ifdef cisTEM_USING_LIBTORCH

/**
 * @brief Restores the macro definitions that conflict between cisTEM
 * and the LibTorch library. See include/libtorch/libtorch_push_macros.h for the 
 * corresponding push and undef operations that remove them to avoid
 * conflicts.
 * 
 * Having these push/undef/pop operations in separate headers allows the inclusion of 
 * multiple header files that might be needed when utilizing LibTorch 
 * libraries.
 * 
 */
#pragma pop_macro("CHAR")
#pragma pop_macro("DOUBLE")
#pragma pop_macro("LONG")
#pragma pop_macro("BOOL")
#pragma pop_macro("FLOAT")
#pragma pop_macro("INTEGER")
#pragma pop_macro("TEXT")
#pragma pop_macro("NONE")
#pragma pop_macro("N_")
#pragma pop_macro("UNCHECKED")

#endif // cisTEM_USING_LIBTORCH
#endif // _INCLUDE_LIBTORCH_LIBTORCH_POP_MACROS_H_