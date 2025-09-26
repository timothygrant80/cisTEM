AC_INIT(cisTEM, [2.0.0-alpha])
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

# The include is to capture <cistem_config.h> where all symbols defined by AC_DEFINE | AC_DEFINE_UNQUOTED placed.
# This is needed to ensure they are passed to nvcc and also cleans up the output of make.
# Note: autoheader breaks if all three args are not present in AC_DEFINE
# Note2: autoheader is v. picky, no space before a seperator "," e.g. [VARNAME], is okay but [VARNAME] , is not.
INPUT_CPPFLAGS="$CPPFLAGS"
INPUT_CXXFLAGS="$CXXFLAGS"

AC_PROG_CC([icc gcc clang])
AC_PROG_CXX([icpc g++ clang++])

AM_PROG_CC_C_O

AC_PROG_INSTALL
LT_PREREQ([2.4])
LT_INIT([dlopen])
AC_LANG(C++)

# Set this for the gpu makefile hack
# TODO: I don't think this is still needed, but I'm not sure.
TOPSRCDIR=$srcdir
CISTEM_CONFIG_DIR=`pwd`
if test "x$TOPSRCDIR" = "x." ; then
	TOPSRCDIR=`pwd`
fi

# don't let the AC_PROG_CXX set default flags, set them back to input..

CXXFLAGS="$INPUT_CXXFLAGS -I$TOPSRCDIR/../include"
CPPFLAGS="$INPUT_CPPFLAGS -I$TOPSRCDIR/../include"

# set up initial flags
AC_MSG_NOTICE([Setting up initial flags])
AC_MSG_NOTICE([CXXFLAGS = $CXXFLAGS])
AC_MSG_NOTICE([CPPFLAGS = $CPPFLAGS])

#
# We need to know about static linking prior to setting CUDA flags 
#
# NOTE: the variable name "static_link" is tested in ax_cuda.m4 and must be consistent w/ the name here.
static_link="false"
AC_ARG_ENABLE(staticmode, AS_HELP_STRING([--enable-staticmode],[Link statically [default=no]]),[
  if test "$enableval" = yes; then
  static_link="true"
  WXCONFIGFLAGS="--static=yes"
  fi
])

  
AC_MSG_NOTICE(static_link = $static_link)
AM_CONDITIONAL([STATIC_LINKING_AM],[test x$static_link = xtrue])

#
# Check CUDA at the top, b/c it needs at least c++14 so we are now using c++17, let cpu only builds stay on c++11 until other devs are okay with modernizing.
#
want_cuda="no"
AX_CUDA 
CPP_STANDARD=17

# 
# Check for extra requested programs beyond main functionality and testing
# Default is to build all programs.
#   --disable-build-all will override default behavior and build only those programs needed for the GUI to run and for testing
#   --enable-build-program will add the additional program(s)
#     these flags have no meaning unles --disable-build-all is set
#
NON_ESSENTIAL_PROGRAMS_TO_BE_COMPILED

# if test "$want_cuda" = "yes" ; then
#   CPP_STANDARD=17
# else
#   CPP_STANDARD=11
# fi

# if test "x$CXX" = "xclang++"; then
#   CPP_STANDARD=17
# fi

# check to see if compiler warnings are desired.
# no value or --enable-warnings result in warnings on, --disable-warnings results in warnings off
AC_ARG_ENABLE([warnings], AS_HELP_STRING([--disable-warnings], [Default=false]), 
[
	# If flag do this
	if test "x$enableval" = "xno" ; then
		WARNINGS_ON=""		
	else
		if test "x$CXX" = "xicpc"; then		
			WARNINGS_ON="-w2"
		else
			WARNINGS_ON="-Wall"	
		fi		
	fi
], 
[ 
		# If no flag do this
		if test "x$CXX" = "xicpc"; then		
			WARNINGS_ON="-w2"
		else
			WARNINGS_ON="-Wall"	
		fi	
])

# Noop: Code not yet ported from downstream
AC_ARG_ENABLE(multiple-global-refinements, AS_HELP_STRING([--disable-multiple-global-refinements],[Only do one global search in AutoRefine3d [default=no]]),[
	if test "x$enableval" = "xno" ; then
		AC_DEFINE([DISABLE_MUTLI_GLOBAL_REFINEMENTS], [ ], [Disable multiple rounds of global refinement in AutoRefine3d])
  fi
]) # No action needed if flag is not present as multiple global refinements is the default.

# NOTE: Ofast, also for gcc, adds more aggressive optimizations, than O3 and also sets -fno-protect-parens which we do not want.
# Unlike gcc, -fno-protect-parens is not a default for icpc, so we need to add it here.
# TODO: test  with -ipo (may take longer to compile but iterprocess optimization may worth it)
# -fprotect-parens
if test "x$CXX" = "xicpc"; then		
	CXXFLAGS="$CXXFLAGS -O3 -no-prec-div -no-prec-sqrt -fprotect-parens $WARNINGS_ON -std=c++${CPP_STANDARD} -wd1125"
	CPPFLAGS="$CPPFLAGS -O3 -no-prec-div -no-prec-sqrt -fprotect-parens $WARNINGS_ON"
  # This may only be needed for dynamic linking
  
  LIBS="$LIBS -lz"
else 
  if test "x$CXX" = "xclang++"; then
    # Clang needs the -no-c++11-narrowing suppression to compile
	  CXXFLAGS="$CXXFLAGS -m64 -funroll-loops -O3 -pipe $WARNINGS_ON  -std=c++${CPP_STANDARD} -Wno-c++11-narrowing" 		
	  CPPFLAGS="$CPPFLAGS -m64 -funroll-loops -O3 -pipe $WARNINGS_ON "
  else 
	  CXXFLAGS="$CXXFLAGS -m64 -funroll-loops -O3 -fprotect-parens -pipe $WARNINGS_ON -fexpensive-optimizations -std=c++${CPP_STANDARD}" 		
	  CPPFLAGS="$CPPFLAGS -m64 -funroll-loops -O3 -fprotect-parens -pipe $WARNINGS_ON -fexpensive-optimizations"

	  #sqlite needs dl on gcc
	
    LIBS="$LIBS -ldl -lz"
  fi
fi

#
# What kind of OS are we on?
#
AC_CANONICAL_HOST

build_linux=no
build_windows=no
build_mac=no

# Detect the target system
case "${host_os}" in
    linux*)
        build_linux=yes
        ;;
    cygwin*|mingw*)
        build_windows=yes
        ;;
    darwin*)
        build_mac=yes
        ;;
    *)
        # Syntax highlighting wasn't seeing the end quote in VSCODE w/  inline msg
        msg="OS $host_os is not supported"
        AC_MSG_ERROR([$msg])
        ;;
esac

# Pass the conditionals to automake
AM_CONDITIONAL([LINUX_AM], [test "$build_linux" = "yes"])
AM_CONDITIONAL([WINDOWS_AM], [test "$build_windows" = "yes"])
AM_CONDITIONAL([OSX_AM], [test "$build_mac" = "yes"])

# Use all available processor instruction sets - this defaults to no (I think) if the option is not flagged. See issue #296 on github
instruction_set=""
AC_ARG_ENABLE(latest-instruction-set, AS_HELP_STRING([--enable-latest-instruction-set],[Use the latest available CPU instruction set, only affects Intel compiler [default=no]]),[
  if test "$enableval" = yes; then
	instruction_set=" -xHost -wd10382"
  fi
]) #, [instruction_set=" -xHost -wd10382"]) if we wanted default to be yes, this fourth arg should be included.

if test "x$CXX" = "xicpc"; then	
	CXXFLAGS="$CXXFLAGS $instruction_set"
fi

# MKL

use_mkl=1
AC_ARG_ENABLE(mkl, AS_HELP_STRING([--disable-mkl],[Do not use the Intel MKL]), [
 if test "$enableval" = no; then
    # LINK_MKL_NOT_FFTW3
    AC_MSG_NOTICE([     ])
    AC_MSG_ERROR([You must use the Intel mkl in cisTEM. See FFT_LICENSE_NOTE in the root directory for details.])
    use_mkl=0
 fi
])




# wx-config 

WXCONFIG=wx-config
AC_ARG_WITH(wx-config,
AS_HELP_STRING([--with-wx-config=FILE],[Use the given path to wx-config when determining
                          wxWidgets configuration; defaults to "wx-config"]),
[
    if test "$withval" != "yes" -a "$withval" != ""; then
        WXCONFIG=$withval
    fi
])





# debugmode

AC_ARG_ENABLE(debugmode, AS_HELP_STRING([--enable-debugmode],[Compile in debug mode [default=no]]),
[
  if test "$enableval" = yes; then
    AC_DEFINE([DEBUG], [], [Define the debug flag])
    if test "x$CXX" = "xicpc"; then
        CXXFLAGS="-O2 -debug -no-prec-div -no-prec-sqrt -fprotect-parens $WARNINGS_ON -std=c++${CPP_STANDARD} -wd1125"
        CPPFLAGS="-O2 -debug -no-prec-div -no-prec-sqrt -fprotect-parens $WARNINGS_ON"
    else
        CPPFLAGS="-O2 -g $WARNINGS_ON"
        CXXFLAGS="-O2 -g -std=c++${CPP_STANDARD} $WARNINGS_ON"
    fi
  else
    AC_DEFINE([NDEBUG], [], [Define the nodebug flag for either g++ or clang builds])
  fi
 ], 
[
    # Action if not given
    AC_DEFINE([NDEBUG], [], [Define the nodebug flag for either g++ or clang builds])
])
  

###############################
# BEGIN CUDA CHECKS AND SETS
###############################

# This also has an effect in submodule_FastFFT on the FastFFT_FLAGS
want_gpu_debug="no"
AC_ARG_ENABLE(gpu-debug, AS_HELP_STRING([--enable-gpu-debug],[Compile heavy synchronous checking [default=no]]),[
  if test "$enableval" = yes; then
    AC_DEFINE([ENABLE_GPU_DEBUG],[], [use the gpu or not])
  	AC_MSG_NOTICE([Compiling with synchronizing debug checks for GPU code])	
    want_gpu_debug="yes"
  fi
  ]) 
  
# Call the m4 macro to check and setup for FastFFT if present and requested. Must be run after AX_CUDA
submodule_FastFFT

if test "x$want_cuda" = "xyes" ; then

    # ~Runtime about 85% with K3 images, but mips are different - total detection is ~the same. 
    # Determined this is likely just numerical error. Enabling by default.
    #    --disable-feature is equivalent to --enable-feature=no. 
    use_rotated_tm=1
    AC_ARG_ENABLE(rotated-tm, AS_HELP_STRING([--disable-rotated-tm],[Place power of 2 sized axis on X by rotating 90 deg in template matching for speed [default=no]]),[
        if test "$enableval" = no; then
            use_rotated_tm=0
        fi
    ])

    if test "x$use_rotated_tm" = "x1"; then
        AC_DEFINE([ROTATEFORSPEED], [], [Rotate image for faster FFT in template matching])
        AC_MSG_NOTICE([Enabling rotated-tm]);
    fi

    AC_MSG_NOTICE([Adding the CUDA includes Setting up NVCC initial flags])
    CUDA_CPPFLAGS="$FastFFT_CUDA_FLAGS $FastFFT_DEFINES $CUDA_CFLAGS $NVCCFLAGS"
    CUDA_CXXFLAGS="$FastFFT_CUDA_FLAGS $FastFFT_DEFINES $CUDA_CFLAGS $NVCCFLAGS"
    
    CXXFLAGS="$CXXFLAGS $use_gpu_cache_hints $CUDA_CFLAGS"
    CPPFLAGS="$CPPFLAGS $use_gpu_cache_hints $CUDA_CFLAGS"
    AC_MSG_NOTICE([Using CUDA_CXXFLAGS=$CUDA_CXXFLAGS])
    AC_MSG_NOTICE([Using CUDA_LIBS=$CUDA_LIBS])

    # This does include any gpu functionality directly, but is used primarily in the GUI to do things like
    # showing radio buttons to  use/not use gpu functionality that is enabled in the called program.
    AC_DEFINE([SHOW_CISTEM_GPU_OPTIONS], [], [Enable gpu codepath])
fi

##############################
# END CUDA CHECKS AND SETS
##############################

# Define WANT_CISTEM_GPU_AM to be used in the Makefile on a per-program basis to then define ENABLEGPU or not.
AM_CONDITIONAL([WANT_CISTEM_GPU_AM], [test "$want_cuda" = "yes"])

# fftw

AC_ARG_WITH(fftw-dir,
AS_HELP_STRING([--with-fftw-dir=DIR],[Declare the root directory of fftw3, if its 
                          current location is not system accessible ]),
[
    if test "$withval" != "yes" -a "$withval" != ""; then
      # LINK_MKL_NOT_FFTW3
      AC_MSG_NOTICE([     ])
      AC_MSG_ERROR([You must use the Intel mkl in cisTEM. See FFT_LICENSE_NOTE in the root directory for details.])
 
      CPPFLAGS="$CPPFLAGS -I$withval/include -L$withval/lib"
      CXXFLAGS="$CXXFLAGS -I$withval/include -L$withval/lib"
      LDFLAGS="$LDFLAGS -L$withval/lib"
    fi
])


# Use fine grained profiling

want_profiling="no"  
AC_ARG_ENABLE(profiling, AS_HELP_STRING([--enable-profiling],[Include fine-grained profiling [default=no]]),[
  if test "$enableval" = yes; then
    AC_DEFINE([CISTEM_PROFILING], [], [Define the profiling flag])
  	want_profiling="true"
  	AC_MSG_NOTICE([Compiling with fine-grained profiling])	
  fi
  ]) 

AM_CONDITIONAL([ENABLEPROFILING_AM], [test x$want_profiling = xtrue])

AC_ARG_ENABLE(fp16-particlestacks, AS_HELP_STRING([--enable-fp16-particlestacks],[Save particle stacks in reduced precision. Use with caution  [default=no]]),[
  if test "x$enableval" = "xyes"; then
    AC_DEFINE([USE_FP16_PARTICLE_STACKS], [], [Save particle stacks in reduced precision])
  	AC_MSG_NOTICE([Compiling with reduced precision particle stacks])	
  fi
  ]) 

AC_ARG_ENABLE(fp16-tm-outputs, AS_HELP_STRING([--disable-fp16-tm-outputs],[Disable saving template matching outputs in reduced precision ]),[
  if test "x$enableval" = "xno"; then
  	AC_MSG_NOTICE([Not Compiling with reduced precision template matching outputs])	
  else
    AC_DEFINE([USE_FP16_TM_OUTPUTS], [], [Save template matching outputs in reduced precision])
    AC_MSG_NOTICE([Compiling with reduced precision template matching outputs])	
  fi
  ],
  [
  AC_DEFINE([USE_FP16_TM_OUTPUTS], [], [Save template matching outputs in reduced precision])
  AC_MSG_NOTICE([Compiling with reduced precision template matching outputs])
  ])


  

want_webview="no"
AC_ARG_ENABLE(wx-webview, AS_HELP_STRING([--enable-wx-webview],[Compile with experimental webview panel]),[
  if test "$enableval" = yes; then
    AC_DEFINE([ENABLE_WEBVIEW], [], [Define the webview flag])
  	want_webview="true"
  	AC_MSG_NOTICE([Compiling with webview enabled])	
  fi
 ]) 
AM_CONDITIONAL([ENABLE_WEBVIEW_AM], [test x$want_webview = xtrue])


# experimental
compile_experimental="false"
AC_ARG_ENABLE(experimental, AS_HELP_STRING([--enable-experimental],[Compile with experimental [default=no]]),[
  if test "$enableval" = yes; then
    AC_DEFINE([EXPERIMENTAL], [], [Define the experimental flag])
    if test "x$CXX" = "xicpc"; then 
        CXXFLAGS="$CXXFLAGS -wd1595"
    fi
  compile_experimental="true"
  fi
  ])
 
#rigorous socket check 

AC_ARG_ENABLE(rigorous-sockets, AS_HELP_STRING([--enable-rigorous-sockets],[Use rigorous socket checking [default=no]]),[
  if test "$enableval" = yes; then
    AC_DEFINE([RIGOROUS_SOCKETS],[], [Define the rigorous sockets flag])
  	AC_MSG_NOTICE([Compiling with rigorous socket checking])	
  fi
  ])
 
AM_CONDITIONAL([EXPERIMENTAL_AM],[test x$compile_experimental = xtrue])

# OpenMP
AC_ARG_ENABLE(openmp, AS_HELP_STRING([--enable-openmp],[Compile with OpenMP threading [default=no]]),[
  if test "$enableval" = yes; then
  AC_OPENMP([],AC_MSG_ERROR("You asked for OpenMP, but I was not able to figure out how to compile programs that use OpenMP"))
  CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"
  fi
  ])
  


#sqlite3-dir

#AC_ARG_WITH(sqlite3-dir,
#[[  --with-sqlite3-dir=DIR  Declare the root directory of sqlite3, if its
#                          current location is not system accessible ]],
#[
#    if test "$withval" != "yes" -a "$withval" != ""; then
#        CPPFLAGS="$CPPFLAGS -I$withval/include -L$withval/lib"
#        CXXFLAGS="$CXXFLAGS -I$withval/include -L$withval/lib"
#        LDFLAGS="$LDFLAGS -L$withval/lib"
#    fi
#])

wxversion=0

AC_DEFUN([WXTEST],
[
        AC_REQUIRE([AC_PROG_AWK])
        AC_MSG_CHECKING([wxWidgets version])
        if wxversion=`$WXCONFIG --version`; then
                AC_MSG_RESULT([$wxversion])
                AC_DEFINE_UNQUOTED([wx_is_available], 1, [Define to 1 if wx is available])
        else
                AC_MSG_RESULT([not found])
                AC_DEFINE_UNQUOTED([wx_is_available], 0, [Define to 0 if wx is unavailable])
        fi])

# Call WXTEST func
WXTEST

# Verify minimus requires
vers=`echo $wxversion | $AWK 'BEGIN { FS = "."; } { printf "% d", ($1  1000 + $2)  1000 + $3;}'`
if test -n "$vers" && test "$vers" -ge 310001004; then

    # TODO: Get all define symbols from WX into cistem_config.h
    # wx_cxx_defines="`$($WXCONFIG $WXCONFIGFLAGS --cxxflags) | $AWK 'BEGIN{output_string=""}{ for (i = 1; i <= NF; i++) if($i ~ /^-D/) output_string=output_string" "$i}END{print output_string}'`"   

        WX_CPPFLAGS="`$WXCONFIG $WXCONFIGFLAGS --cppflags`"
        WX_CXXFLAGS="`$WXCONFIG $WXCONFIGFLAGS --cxxflags | sed -e 's/-fno-exceptions//'`"
        WX_LIBS="`$WXCONFIG $WXCONFIGFLAGS --libs richtext,std,aui`"

        if test "$build_mac" =  "yes"; then
          WX_CPPFLAGS_BASE="`$WXCONFIG $WXCONFIGFLAGS --cppflags`"
          WX_CXXFLAGS_BASE="`$WXCONFIG $WXCONFIGFLAGS --cxxflags | sed -e 's/-fno-exceptions//'`"
          WX_LIBS_BASE="`$WXCONFIG $WXCONFIGFLAGS --libs richtext,std,aui,xml`"
        else
          WX_CPPFLAGS_BASE="`$WXCONFIG $WXCONFIGFLAGS --cppflags base`"
          WX_CXXFLAGS_BASE="`$WXCONFIG $WXCONFIGFLAGS --cxxflags base | sed -e 's/-fno-exceptions//'`"
          WX_LIBS_BASE="`$WXCONFIG $WXCONFIGFLAGS --libs base,net,xml`"
        fi
        AC_MSG_NOTICE([wxWidgets libsbase $WX_LIBS])       
        AC_MSG_NOTICE([wxWidgets libsbase $WX_LIBS_BASE])
        wx_is_available=1

    # Record whether we are on the stable (3.0.4/5) or development (3.1.5) branch
    if test -n "$vers" && test "$vers" -gt 310001005; then
        AC_DEFINE([WXWIDGETS_3_DEV], [], [Define to 1 if wxWidgets version is 3.1.5 or greater])
    else
        AC_DEFINE([WXWIDGETS_3_0_5], [], [Define to 1 if wxWidgets version is 3.0.4 or greater])
    fi

    # Record which gtk version we are using
    GTK_CONFIG=`$WXCONFIG --selected-config | $AWK 'BEGIN{FS="-"}{if($1=="gtk2") print 2; else print 3}'`
    # if test -n "$GTK_CONFIG" -ne 2 -a -n "$GTK_CONFIG" -ne 3; then
    #     AC_MSG_ERROR("Can't determine GTK version")
    # fi
    AC_DEFINE_UNQUOTED([GTK_VERSION], $GTK_CONFIG, [Define to 2 if gtk2, 3 if gtk3])

else
        AC_MSG_ERROR("Only wxWidgets version 3.0.4+ is currently supported.")
        wx_is_available=0
fi



# Link the MKL with a variety of threading options

use_mkl_threads="no"
use_gnu_threads="no"
use_mkl_sequential="yes"
AC_ARG_ENABLE(mkl-threads, AS_HELP_STRING([--enable-mkl-threads],[Compile with MKL threading using Intel threads [default=no]]),[
    use_mkl_threads="yes"
    use_mkl_sequential="no"

])
AC_ARG_ENABLE(gnu-threads, AS_HELP_STRING([--enable-gnu-threads],[Compile with MKL threading using Intel threads [default=no]]),[
    use_gnu_threads="yes"
    use_mkl_sequential="no"

])

# Make sure we have the environmental variables set up
AC_CHECK_FILE("$MKLROOT/include/mkl.h",
[   ],
[AC_MSG_ERROR([MKLROOT is not set correctly])
])

   


  AC_DEFINE([MKL], [], [Use the MKL])
# Compiler options are the same for all threading options (as long as it is gcc)
AC_DEFINE([MKL_ILP64], [], [Use the MKL ILP64 interface])


# Configure MKL based on threading options
if test "x$CXX" = "xicpc"; then
  if test "x$use_mkl_threads" = "xyes"; then
    CPPFLAGS="$CPPFLAGS -qmkl=parallel -I"${MKLROOT}/include" "
    CXXFLAGS="$CXXFLAGS -qmkl=parallel -I"${MKLROOT}/include" "
    LIBS="$LIBS -qmkl=parallel "
  elif test "x$use_gnu_threads" = "xyes"; then
    # Intel compiler with GNU OpenMP threads
    CPPFLAGS="$CPPFLAGS -I"${MKLROOT}/include" "
    CXXFLAGS="$CXXFLAGS -I"${MKLROOT}/include" "
    # Let the manual linking below handle GNU threads
  else
    # Sequential (default)
    CPPFLAGS="$CPPFLAGS -qmkl=sequential -I"${MKLROOT}/include" "
    CXXFLAGS="$CXXFLAGS -qmkl=sequential -I"${MKLROOT}/include" "
    LIBS="$LIBS -qmkl=sequential "
  fi
else
  CPPFLAGS="$CPPFLAGS -I"${MKLROOT}/include" "
  CXXFLAGS="$CXXFLAGS -I"${MKLROOT}/include" "
fi

LDFLAGS="$LDFLAGS -L"${MKLROOT}/lib/intel64" "


MKL_LIBS=""
LIBTOOL_FLAGS=""
if test "x$use_mkl_threads" = "xyes" -o "x$use_gnu_threads" = "xyes"; then

    # I added the proper options here along when updating with sequential, which is what cisTEM has always used.
    # To actually use threaded FFTs, we would need to:
    # For multithreaded plans, use normal sequence of calls to the fftw_init_threads() and fftw_plan_with_nthreads() functions (refer to FFTW documentation).
    AC_MSG_NOTICE([      ]) 
    # AC_MSG_ERROR([ Threading with the MKL is not fully supported yet. ])
    # We can't have both mkl-threads and gnu-threads
    if test "x$use_mkl_threads" = "xyes" -a "x$use_gnu_threads" = "xyes"; then
        AC_MSG_ERROR("You can't have both --enable-mkl-threads and --enable-gnu-threads")
    fi

    # Okay, we have only one of them, so we can use the mkl-threads or gnu-threads
    if test "x$use_mkl_threads" = "xyes"; then
        if test "x$static_link" = "xtrue" ; then          
            MKL_LIBS="$MKL_LIBS -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread  "
        else
            MKL_LIBS="$MKL_LIBS -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread "
        fi
    else
        if test "x$static_link" = "xtrue" ; then
            MKL_LIBS="$MKL_LIBS -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a  ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread "
        else
            MKL_LIBS="$MKL_LIBS -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread "
        fi
    fi
else    
    # Use MKL seqential
    AC_MSG_NOTICE([ Using the MKL sequential interface. ])

    
    if test "x$static_link" = "xtrue" ; then
      AC_MSG_NOTICE([   Using static linking for the MKL. ])

        # |-Wl*-group|/opt/*mkl* 
        LIBTOOL_FLAGS="--preserve-dup-deps "
        circular_deps="${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a "
        MKL_LIBS=" $MKL_LIBS ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${circular_deps}  ${circular_deps}  ${circular_deps} -lpthread -lm -ldl"

        # LIBS="$LIBS ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a "
    else
        MKL_LIBS="$MKL_LIBS -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread "
    fi    
    
fi


# Even though we do not link the FFTW3 libraries, the Intel MKL uses one of FFTW3's headers to maintain parity while defining a wrapper. 
# AC_CHECK_HEADER(fftw3.h, , AC_MSG_ERROR("Can't find fftw3.h"))
# AC_CHECK_LIB([mkl_intel_lp64],[fftwf_malloc],[HAVE_MKL="yes"],[HAVE_MKL="no"],[-lmkl_core -lmkl_sequential -lpthread])

# if test "x$use_mkl" = "x0"; then
# 	HAVE_MKL="no"
# fi

# if test "x$HAVE_MKL" = "xyes"; then

# 	AC_MSG_NOTICE(Using Intel MKL for fast Fourier transforms)


# 	AC_CHECK_LIB(fftw3f, fftwf_malloc, [LIBS="$LIBS -lfftw3f"],AC_MSG_ERROR("Can't find FFTW3's libraries. Please check your installation of FFTW3."))
	# 	AC_MSG_NOTICE(Using FFTW for Fourier transforms)

# fi

#If we're goind a static build with the intel compiler, let's have Intel libraries linked statically also
# It's importnat that -static-intel be specified after -mkl, otherwise mkl will be linked dynamically
if test "x$static_link" = "xtrue"; then
 if test "x$CXX" = "xicpc"; then
 	LDFLAGS="$LDFLAGS -static-intel -qopenmp-link=static"
 	# There is no static library for libcilkrts. Suppress this warning.
 	LDFLAGS="$LDFLAGS -wd10237"
 else
	LDFLAGS="$LDFLAGS -static"
 fi
fi

# tiff (if we're not using wx's builtin tiff support)
#
# Using AC_CHECK_LIB fails when we have a static build of wx in a non-canonical place. I think the following should be fine.
# AC_CHECK_LIB([wxtiff-3.0],[TIFFOpen],[WX_LIBS_BASE="-lwxtiff-3.0 $WX_LIBS_BASE"],[wxtiff=0],`$WXCONFIG --libs base`)
wxtiff=`$WXCONFIG --libs base | grep -c lwxtiff-3.0`
AC_MSG_NOTICE([wxtiff = $wxtiff])
if test "x$wxtiff" = "x0"; then
  AC_SEARCH_LIBS([TIFFOpen],[tiff],[],[AC_MSG_ERROR(Could not find your installation of the TIFF library)])
else 
  WX_LIBS_BASE="-lwxtiff-3.0 $WX_LIBS_BASE"
fi

# make it so we can turn off gui

#AC_ARG_ENABLE(textonly, [  --enable-textonly       Do not compile wxWidgets applications [[default=no]]],[
#  if test "$enableval" = yes; then
#  wx_is_available=0
#  fi  
#  ])


#if test "$wx_is_available" = 0; then

#echo
#echo "**    WxWidgets not available, GUI program will not be compiled **"
#echo

#fi


# What SVN revision are we using? Inner if clause will not work if svnversion is not callable
AC_CHECK_PROG(SVB_CHECK,svnversion,yes)
if test "x$SVN_CHECK" == "xyes" ; then
  define([svnversion], esyscmd([sh -c "svnversion -n"]))dnl
  if test "x$svnversion" != "xUnversioned directory"; then
    AC_DEFINE(CISTEM_SVN_REV, "svnversion", [SVN Revision])
  fi
fi

AC_CHECK_PROG(GIT_CHECK,git,yes)
# If we have git, it must not be too old, e.g. 1.8 is still the default on older OS's like CentOS 7 
is_git_new_enough=`git version | awk '{if($3 >= 2) print yes; else print no}'`
if test "xis_git_new_enough" == "xno" ; then
  	AC_MSG_NOTICE([your version of git is too old, you can updated on Centos 7 with sudo yum -y install https://packages.endpoint.com/rhel/7/os/x86_64/endpoint-repo-1.7-1.x86_64.rpm
  	])
else

set_git_versioning_defaults="no"
AC_CHECK_FILE("$TOPSRCDIR/.git/HEAD",
  [
    if test "x$GIT_CHECK" = "xyes"; then 
      # All the args require the extra double quotes to be properly formated later on.
      
      # Get the current branch, which could have multiple refs, which is why we need the awk (*) marks the active branch.
      AC_DEFINE_UNQUOTED([CISTEM_CURRENT_BRANCH],["`git  branch | awk '/^\*/{print $2}'`"], [get the current branch])
      # Get the first reachable tag - number of commits since that tag - short commit hash
      version_name="`cat $TOPSRCDIR/VERSION | $AWK  '{print $1}'`"
      version_commit="`cat $TOPSRCDIR/VERSION | $AWK  '{print $2}'`"
      is_dirty="`git status --short | awk '{if($0!="" && FNR==1) print "-dirty"}'`"
      AC_DEFINE_UNQUOTED([CISTEM_VERSION_TEXT],["`git rev-list ${version_commit}..HEAD | awk -v NAME=$version_name -v ISDIRTY=$is_dirty 'BEGIN{n=0}{n=n+1}{if(FNR==1) H=$1}END{print NAME"-"n-1"-"substr(H,1,7)ISDIRTY}'`"], [current version text])
      # Get both an easily parsed date, as well as a more readable date
      AC_DEFINE_UNQUOTED([CISTEM_TIME_YYMMDDHHMMSS],["`git show --no-patch --no-notes --pretty='%cd' --date=format:%y%m%d%H%M%S`"], [time of build complete])
      AC_DEFINE_UNQUOTED([CISTEM_TIME_READABLE],["`git show --no-patch --no-notes --pretty='%ci' --date=format:%y%m%d%H%M%S`"], [time of build readable]) 
      
    else
      set_git_versioning_defaults="yes"

    fi
  ], 
  [
    set_git_versioning_defaults="yes"
  ])
fi

# Confirm we found the proper git versioning information. This is primarly intended for developers who are actively using multiple versions
# or who are distributing test debug builds to users, such that they may easily confirm what version they are using from the 
# overview panel. 
if test "x$set_git_versioning_defaults" = "xyes" ; then
  # Check to see if the user has specified the versioning information.
  if test "x$CISTEM_VERSION_TEXT" = "x" ; then
    AC_MSG_NOTICE([CISTEM_VERSION_TEXT not specified, using default])
    AC_DEFINE(CISTEM_VERSION_TEXT, "undefined", [undefined_version]) 
  fi
  if test "x$CISTEM_TIME_YYMMDDHHMMSS" = "x" ; then
    AC_MSG_NOTICE([CISTEM_TIME_YYMMDDHHMMSS not specified, using default])
    AC_DEFINE(CISTEM_TIME_YYMMDDHHMMSS, "undefined", [undefined_time])
  fi
  if test "x$CISTEM_TIME_READABLE" = "x" ; then
    AC_MSG_NOTICE([CISTEM_TIME_READABLE not specified, using default])
    AC_DEFINE(CISTEM_TIME_READABLE, "undefined", [undefined_time_readable])
  fi
  if test "x$CISTEM_CURRENT_BRANCH" = "x" ; then
    AC_MSG_NOTICE([CISTEM_CURRENT_BRANCH not specified, using default])
    AC_DEFINE(CISTEM_CURRENT_BRANCH, "undefined", [undefined_branch])
  fi
fi


CXXFLAGS="$CXXFLAGS -I$CISTEM_CONFIG_DIR"
CPPFLAGS="$CPPFLAGS -I$CISTEM_CONFIG_DIR"
WX_CPPFLAGS="$WX_CPPFLAGS -I$CISTEM_CONFIG_DIR"
WX_CPPFLAGS_BASE="$WX_CPPFLAGS_BASE -I$CISTEM_CONFIG_DIR"

# Make sure the host compiler gets all flags when called from nvcc
CUDA_TO_CPP="`echo $CPPFLAGS $WX_CPPFLAGS | sed -e 's/\s\+/,/g' | awk '{print "-Xcompiler " $0}'`"
CUDA_TO_CXX="`echo $CXXFLAGS $WX_CXXFLAGS | sed -e 's/\s\+/,/g' | awk '{print "-Xcompiler " $0}'`"
CUDA_CPPFLAGS="$CUDA_CPPFLAGS $CUDA_TO_CPP "
CUDA_CXXFLAGS="$CUDA_CXXFLAGS $CUDA_TO_CXX "



AC_SUBST(WX_LIBS)
AC_SUBST(WX_CPPFLAGS)
AC_SUBST(WX_LIBS_BASE)
AC_SUBST(WX_CPPFLAGS_BASE)
AC_SUBST(CUDA_CXXFLAGS)
AC_SUBST(CUDA_CPPFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(libFastFFT_OBJECTS)
AC_SUBST(MKL_LIBS)
AC_SUBST(LIBTOOL_FLAGS)



  AC_MSG_NOTICE([Using CUDA_CXXFLAGS=$CUDA_CXXFLAGS])
  AC_MSG_NOTICE([WX_LIBS_BASE=$WX_LIBS_BASE])
  AC_MSG_NOTICE([WX_LIBS=$WX_LIBS])
  AC_MSG_NOTICE([LIBS=$LIBS])
  AC_MSG_NOTICE([MKL_LIBS=$MKL_LIBS])
  AC_MSG_NOTICE([libFastFFT_OBJECTS=$libFastFFT_OBJECTS])

AC_CONFIG_FILES([Makefile src/Makefile])
AC_CONFIG_HEADERS([cistem_config.h:cistem_config.in])
AC_OUTPUT

# Test placeholder for smudge/clean filters


# SA INSERT 1b
# "" 
# SA INSERT 1e
