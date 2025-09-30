# LibTorch configuration for cisTEM
#
# This macro configures LibTorch support for machine learning-based tools
# such as blush regularization for cryo-EM density map denoising.
#
# LibTorch libraries are dynamically linked (even in static builds) and
# bundled with the distribution using RPATH for easy deployment.
#
# Configuration:
#   --enable-libtorch               : Enable LibTorch support (opt-in, ML tools available)
#   LIBTORCH_ROOT=<path>            : Specify LibTorch installation path (default: /opt/libtorch)
#
# Output variables:
#   LIBTORCH_CXX_FLAGS              : C++ flags for LibTorch include paths
#   LIBTORCH_LIBS                   : Libraries to link (-ltorch -ltorch_cpu -lc10)
#   LIBTORCH_RPATH                  : RPATH flags for runtime library location
#   use_libtorch                    : "yes" or "no" indicating if LibTorch is available
#
# Preprocessor defines:
#   cisTEM_USING_LIBTORCH           : Defined when LibTorch is available
#
# Automake conditionals:
#   ENABLE_LIBTORCH_AM              : Set to true when LibTorch is enabled

AC_DEFUN([AX_LIBTORCH],
[
    use_libtorch="no"
    LIBTORCH_CXX_FLAGS=""
    LIBTORCH_LIBS=""
    LIBTORCH_RPATH=""

    # Check if user wants to enable libtorch (opt-in)
    AC_ARG_ENABLE(libtorch,
        AS_HELP_STRING([--enable-libtorch], [Use LibTorch for ML-based tools @<:@default=no@:>@]),
        [AS_IF([test "x$enableval" = "xyes"],
               [use_libtorch="yes"
                AC_MSG_NOTICE([LibTorch support requested by user])],
               [AS_IF([test "x$enableval" = "xno"],
                      [AC_MSG_ERROR([LibTorch is disabled by default. Specifying --disable-libtorch breaks the configuration. If you want to enable LibTorch, please configure with --enable-libtorch])])])],
        [AC_MSG_NOTICE([LibTorch support not requested (use --enable-libtorch to enable)])])

    AS_IF([test "x$use_libtorch" = "xyes"],
    [
        # Check if LIBTORCH_ROOT is set, otherwise try /opt/libtorch
        AS_IF([test "x$LIBTORCH_ROOT" = "x"],
              [LIBTORCH_ROOT="/opt/libtorch"])

        AC_MSG_NOTICE([Checking for LibTorch in $LIBTORCH_ROOT])

        # Check if libtorch exists (torch.h is in csrc/api/include subdirectory)
        AC_CHECK_FILE(["$LIBTORCH_ROOT/include/torch/csrc/api/include/torch/torch.h"],
        [
            HAVE_LIBTORCH="yes"
            AC_MSG_NOTICE([LibTorch found at $LIBTORCH_ROOT])
        ],
        [
            HAVE_LIBTORCH="no"
            use_libtorch="no"
            AC_MSG_WARN([LibTorch not found at $LIBTORCH_ROOT. ML-based tools will not be available.])
            AC_MSG_WARN([To enable LibTorch: set LIBTORCH_ROOT=/path/to/libtorch or install to /opt/libtorch])
        ])

        AS_IF([test "x$HAVE_LIBTORCH" = "xyes"],
        [
            # Define preprocessor macro for conditional compilation
            AC_DEFINE([cisTEM_USING_LIBTORCH], [], [Use LibTorch for ML-based tools])

            # Set include paths as flags (not directly modifying CPPFLAGS/CXXFLAGS)
            # Following FastFFT pattern: programs that need LibTorch will use LIBTORCH_CXX_FLAGS
            LIBTORCH_CXX_FLAGS="-I${LIBTORCH_ROOT}/include -I${LIBTORCH_ROOT}/include/torch/csrc/api/include"

            # Warn about static linking issues
            AS_IF([test "x$static_link" = "xtrue"],
                  [AC_MSG_WARN([Static linking with LibTorch is not recommended by PyTorch developers.])
                   AC_MSG_WARN([LibTorch will be dynamically linked even in static build mode.])])

            # Always use dynamic linking for libtorch (even in static builds)
            # Order matters: torch depends on torch_cpu, which depends on c10
            LIBTORCH_LIBS="-L${LIBTORCH_ROOT}/lib -ltorch -ltorch_cpu -lc10"

            # Set RPATH for runtime library location
            # This allows the executable to find libraries relative to its location
            # Enables bundling the .so files with the distribution
            # $ORIGIN is a special variable that expands to the directory containing the executable
            LIBTORCH_RPATH="-Wl,-rpath,'\$\$ORIGIN/lib' -Wl,-rpath,'\$\$ORIGIN/../lib' -Wl,-rpath,'${LIBTORCH_ROOT}/lib'"

            AC_MSG_NOTICE([LibTorch configuration:])
            AC_MSG_NOTICE([  LIBTORCH_ROOT      = $LIBTORCH_ROOT])
            AC_MSG_NOTICE([  LIBTORCH_CXX_FLAGS = $LIBTORCH_CXX_FLAGS])
            AC_MSG_NOTICE([  LIBTORCH_LIBS      = $LIBTORCH_LIBS])
            AC_MSG_NOTICE([  LIBTORCH_RPATH     = $LIBTORCH_RPATH])
        ])
    ])

    # Set automake conditional for Makefile.am
    AM_CONDITIONAL([ENABLE_LIBTORCH_AM], [test "x$use_libtorch" = "xyes"])

    # Substitute variables for use in Makefile.am
    AC_SUBST(LIBTORCH_CXX_FLAGS)
    AC_SUBST(LIBTORCH_LIBS)
    AC_SUBST(LIBTORCH_RPATH)
])