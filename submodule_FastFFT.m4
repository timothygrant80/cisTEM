# Setup use of FastFFT submodule


AC_DEFUN([submodule_FastFFT],
[

# git submodule add https://github.com/bHimes/FastFFT.git src/ext/FastFFT

# I was originally compiling the library separately, but let's hope we don't need to do that.
# # Check to see if the location of the FastFFT library is define  
# AC_CHECK_FILE([$FastFFT_DIR/lib/libFastFFT_static.a], [want_FastFFT="yes"], [] )


# If a dev has pulled in the FastFFT submodule, it will be enabled by default. It may be explicitly disabled with configure options. Check these first.
use_FastFFT="yes"
FastFFT_FLAGS=""
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
    AC_CHECK_FILE("$TOPSRCDIR/src/ext/FastFFT/include/FastFFT.h",[use_FastFFT="yes"],[use_FastFFT="no"])
    if test "x$use_FastFFT" = "xyes"; then
        # using $CISTEM_CONFIG_DIR/$TOPSRCDIR/ is a bit of a jenky way to get an absolute path. I'm doing this
        # because TOPSRCDIR defined by autotools srcdir is generally ../.. so a file in src/core would be okay to include, but 
        # a file in src/program/program name would not find the header.
        SUBMODULE_INCLUDES="$SUBMODULE_INCLUDES -I$CISTEM_CONFIG_DIR/$TOPSRCDIR/src/ext/FastFFT/include"
        AC_DEFINE(ENABLE_FastFFT, [], [Use the FasFFT library for GPU FFTs where appropriate.])
        AC_MSG_NOTICE([Using the FastFFT Library.])
        AC_DEFINE([CUFFTDX_DISABLE_RUNTIME_ASSERTS], [], [Define the CUFFTDX_DISABLE_RUNTIME_ASSERTS flag])
        # Generally, you probably shouldn't need to use these through cisTEM, but they do need to be defined.
        # TODO send these as a flag only where needed.
        FastFFT_FLAGS="-DFFT_DEBUG_LEVEL=0 -DDEBUG_FFT_STAGE=8" #-DHEAVYERRORCHECKING_FFT        
    else
        AC_MSG_NOTICE([Not using the FastFFT Library b/c $TOPSRCDIR/src/ext/FastFFT/include/FastFFT.h was not found.])
    fi   
])


])