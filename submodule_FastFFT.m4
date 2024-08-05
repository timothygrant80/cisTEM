# Setup use of FastFFT submodule


AC_DEFUN([submodule_FastFFT],
[

FastFFT_CXX_FLAGS=""  
FastFFT_CUDA_FLAGS=""

# git submodule add https://github.com/bHimes/FastFFT.git src/ext/FastFFT

# I was originally compiling the library separately, but let's hope we don't need to do that.
# # Check to see if the location of the FastFFT library is define  
# AC_CHECK_FILE([$FastFFT_DIR/lib/libFastFFT_static.a], [want_FastFFT="yes"], [] )


# If a dev has pulled in the FastFFT submodule, it will be enabled by default. It may be explicitly disabled with configure options. Check these first.
use_FastFFT="yes"
FastFFT_DEFINES=""

# Also introduce synchronizing debug level in FastFFT if it is in place in cisTEM
if test "x$want_gpu_debug" = "xyes"; then
    fft_debug_level=4
else
    fft_debug_level=0
fi

AC_ARG_ENABLE(FastFFT, AS_HELP_STRING([--disable-FastFFT],[Do not use the FastFFT library, even if submodule is present]), [
    if test "$enableval" = no; then
        use_FastFFT="no"
        AC_MSG_NOTICE([Not using the FastFFT Library b/c --disable-FastFFT is configured.])      
    else
        AC_MSG_ERROR([FastFFT is enabled by default, if present. Specifying --enable-FastFFT breaks the configuration. If you want to disable FastFFT, please configure with --disable-FastFFT])
    fi
], 
[
    # Okay, this has not been disabled (explicitly) so let's see if the submodule is in place.
    AC_CHECK_FILE("$TOPSRCDIR/include/FastFFT/include/FastFFT.h",[use_FastFFT="yes"],[use_FastFFT="no"])
    if test "x$use_FastFFT" = "xyes"; then
        # using $CISTEM_CONFIG_DIR/$TOPSRCDIR/ is a bit of a jenky way to get an absolute path. I'm doing this
        # because TOPSRCDIR defined by autotools srcdir is generally ../.. so a file in src/core would be okay to include, but 
        # a file in src/program/program name would not find the header.
        # FastFFT_INCLUDES="$FastFFT_INCLUDES -I$CISTEM_CONFIG_DIR/$TOPSRCDIR/src/ext/FastFFT/include"
        AC_DEFINE(cisTEM_USING_FastFFT, [], [Use the FasFFT library for GPU FFTs where appropriate.])
        AC_MSG_NOTICE([Using the FastFFT Library.])
        AC_DEFINE([CUFFTDX_DISABLE_RUNTIME_ASSERTS], [], [Define the CUFFTDX_DISABLE_RUNTIME_ASSERTS flag])
       

        # Generally, you probably shouldn't need to use these through cisTEM, but they do need to be defined.
        # We can't use AC_DEFINE here because the definitions are placed in cistem_config.h which is NOT included in FastFFT      

        # AC_DEFINE_UNQUOTED(FFT_DEBUG_LEVEL, $fft_debug_level, [Define the FFT_DEBUG_LEVEL flag])
        # AC_DEFINE(FFT_DEBUG_STAGE, [3], [Define the FFT_DEBUG_STAGE flag])
        FastFFT_DEFINES="$FastFFT_DEFINES -DFFT_DEBUG_LEVEL=$fft_debug_level -DFFT_DEBUG_STAGE=8"
        FastFFT_CUDA_FLAGS=" --extended-lambda --Wext-lambda-captures-this --expt-relaxed-constexpr"

    else
        AC_MSG_NOTICE([Not using the FastFFT Library b/c $TOPSRCDIR/include/FastFFT/include/FastFFT.h was not found.])
    fi   
])

AM_CONDITIONAL([ENABLE_FASTFFT_AM], [test "x$use_FastFFT" = "xyes"])

AC_SUBST(FastFFT_CUDA_FLAGS)
AC_SUBST(FastFFT_CXX_FLAGS)
])
