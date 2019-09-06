AC_INIT(cisTEM, [2.0.0-alpha])
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

INPUT_CPPFLAGS="$CPPFLAGS"
INPUT_CXXFLAGS="$CXXFLAGS"

AC_PROG_CC([icc gcc])
AC_PROG_CXX([icpc g++ gcc])

AM_PROG_CC_C_O

AC_PROG_INSTALL
AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL
AC_LANG(C++)

# don't let the AC_PROG_CXX set default flags, set them back to input..

CXXFLAGS="$INPUT_CXXFLAGS"
CPPFLAGS="$INPUT_CPPFLAGS"

# set up initial flags
AC_MSG_NOTICE([Setting up initial flags])


# check to see if compiler warnings are desired.
WALLON=""
AC_ARG_WITH([warnings-on], [Enable compiler warnings], [WALLON="-Wall"])


if test "x$CXX" = "xicpc"; then		
	CXXFLAGS="$CXXFLAGS -O3 -no-prec-div -no-prec-sqrt -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -std=c++11"
	CPPFLAGS="$CPPFLAGS -O3 -no-prec-div -no-prec-sqrt -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE"
else
	CXXFLAGS="$CXXFLAGS -DNDEBUG -funroll-loops -O3 -pipe $WALLON -fexpensive-optimizations -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -std=c++11"		
	CPPFLAGS="$CPPFLAGS -DNDEBUG -funroll-loops -O3 -pipe $WALLON -fexpensive-optimizations -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE"

	#sqlite needs dl on gcc
	
	LIBS="$LIBS -ldl"
fi

want_cuda="no"
###############################
# BEGIN CUDA CHECKS AND SETS
###############################

AC_MSG_NOTICE([Checking for gpu request and vars])
AC_ARG_WITH([cuda], AS_HELP_STRING([--with-cuda@<:@=yes|no|DIR@:>@], [prefix where cuda is installed (default=no)]),
[
	with_cuda=$withval
	if test "$withval" = "no"
	then
		want_cuda="no"
	elif test "$withval" = "yes"
	then
		want_cuda="yes"
	else
		want_cuda="yes"
		cuda_home_path=$withval
	fi
], [ want_cuda="no"] )


if test "$want_cuda" = "yes" 
then
	# Set conditional and run macro
	
	AX_CUDA
	
    AC_MSG_NOTICE([Adding the CUDA includes Setting up NVCC initial flags])
    CPPFLAGS="$CPPFLAGS $CUDA_CFLAGS -DUSEGPU"
    CXXFLAGS="$CXXFLAGS $CUDA_CFLAGS -DUSEGPU"	
    LIBS="$LIBS $CUDA_LIBS"	
fi

##############################
# END CUDA CHECKS AND SETS
##############################

AM_CONDITIONAL([USEGPU_AM], [test "$want_cuda" = "yes"])

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
        AC_MSG_ERROR(["OS $host_os is not supported"])
        ;;
esac
# Pass the conditionals to automake
AM_CONDITIONAL([LINUX_AM], [test "$build_linux" = "yes"])
AM_CONDITIONAL([WINDOWS_AM], [test "$build_windows" = "yes"])
AM_CONDITIONAL([OSX_AM], [test "$build_mac" = "yes"])

# Use all available processor instruction sets
AC_ARG_ENABLE(latest-instruction-set, [  --enable-latest-instruction-set     Use latest available CPU instruction set [[default=yes]]],[
  if test "$enableval" = yes; then
	if test "x$CXX" = "xicpc"; then	
		CXXFLAGS="$CXXFLAGS -xHost"
	fi
  fi
  ])

# MKL

AC_ARG_ENABLE(mkl, [  --disable-mkl           Do not use the Intel MKL ],[
  if test "$enableval" = no; then
        use_mkl=0
  fi])


# Static linking

static_link="false"

AC_ARG_ENABLE(staticmode, [  --enable-staticmode     Link statically [[default=no]]],[
  if test "$enableval" = yes; then
	static_link="true"
	WXCONFIGFLAGS="--static=yes"
  fi
  ])

	
AC_MSG_NOTICE(static_link = $static_link)
AM_CONDITIONAL([STATIC_LINKING_AM],[test x$static_link = xtrue])


# wx-config 

WXCONFIG=wx-config
AC_ARG_WITH(wx-config,
[[  --with-wx-config=FILE   Use the given path to wx-config when determining
                          wxWidgets configuration; defaults to "wx-config"]],
[
    if test "$withval" != "yes" -a "$withval" != ""; then
        WXCONFIG=$withval
    fi
])

# MKL

use_mkl=1

AC_ARG_ENABLE(mkl, [  --disable-mkl           Do not use the Intel MKL ],[
  if test "$enableval" = no; then
  	use_mkl=0
  fi])

# debugmode

AC_ARG_ENABLE(debugmode, [  --enable-debugmode      Compile in debug mode [[default=no]]],[
  if test "$enableval" = yes; then
  CPPFLAGS="-O2 -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -DDEBUG $WALLON $INPUT_CPPFLAGS"
  CXXFLAGS="-O2 -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -DDEBUG -std=c++11 $WALLON $INPUT_CXXFLAGS"
  fi
  ])
  
  

# fftw

AC_ARG_WITH(fftw-dir,
[[  --with-fftw-dir=DIR     Declare the root directory of fftw3, if its 
                          current location is not system accessible ]],
[
    if test "$withval" != "yes" -a "$withval" != ""; then
        CPPFLAGS="$CPPFLAGS -I$withval/include -L$withval/lib"
        CXXFLAGS="$CXXFLAGS -I$withval/include -L$withval/lib"
        LDFLAGS="$LDFLAGS -L$withval/lib"
    fi
])

# experimental

AC_ARG_ENABLE(experimental, [  --enable-experimental  Compile with experimental [[default=no]]],[
  if test "$enableval" = yes; then
  CPPFLAGS="$CPPFLAGS -DEXPERIMENTAL"
  CXXFLAGS="$CXXFLAGS -DEXPERIMENTAL"
  compile_experimental="true"
  fi
  ])
 
#rigorous socket check 

AC_ARG_ENABLE(rigorous-sockets, [  --enable-rigorous-sockets  Use rigorous socket checking [[default=no]]],[
  if test "$enableval" = yes; then
  CPPFLAGS="$CPPFLAGS -DRIGOROUS_SOCKET_CHECK"
  CXXFLAGS="$CXXFLAGS -DRIGOROUS_SOCKET_CHECK"  
  fi
  ])
 
AM_CONDITIONAL([EXPERIMENTAL_AM],[test x$compile_experimental = xtrue])

# OpenMP
AC_ARG_ENABLE(openmp, [  --enable-openmp  Compile with OpenMP threading [[default=no]]],[
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
if test -n "$vers" && test "$vers" -ge 3000004; then
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
                
        wx_is_available=1
else
        AC_MSG_ERROR("Only wxWidgets version 3.0.4+ is currently supported.")
        wx_is_available=0
fi

# Do FFTW3 Checks

# Check whether Intel's MKL is available. If not, link against FFTW.
AC_CHECK_HEADER(fftw3.h, , AC_MSG_ERROR("Can't find fftw3.h"))
AC_CHECK_LIB([mkl_intel_lp64],[fftwf_malloc],[HAVE_MKL="yes"],[HAVE_MKL="no"],[-lmkl_core -lmkl_sequential -lpthread])

if test "x$use_mkl" = "x0"; then
	HAVE_MKL="no"
fi

if test "x$HAVE_MKL" = "xyes"; then

	AC_MSG_NOTICE(Using Intel MKL for fast Fourier transforms)
	
	CXXFLAGS="$CXXFLAGS -DMKL"
	CPPFLAGS="$CPPFLAGS -DMKL"

	if test "x$CXX" = "xicpc"; then
		CPPFLAGS="$CPPFLAGS -mkl=sequential"
		LIBS="$LIBS -mkl=sequential"
	else
		LDFLAGS="$LDFLAGS -L\$(MKLROOT)/lib/intel64"
		if test "x$static_link" = "xtrue"; then
			AC_MSG_ERROR("Static builds with Intel MKL are not supported when using the GNU compiler. Please try with the Intel compiler instead.")
	    else
#	    	LIBS="$LIBS -lmkl_rt"
	    	LIBS="$LIBS -lmkl_sequential -lmkl_core -lmkl_intel_lp64"
	    fi
	fi
else

	AC_CHECK_LIB(fftw3f, fftwf_malloc, [LIBS="$LIBS -lfftw3f"],AC_MSG_ERROR("Can't find FFTW3's libraries. Please check your installation of FFTW3."))
	AC_MSG_NOTICE(Using FFTW for Fourier transforms)

fi

#If we're goind a static build with the intel compiler, let's have Intel libraries linked statically also
# It's importnat that -static-intel be specified after -mkl, otherwise mkl will be linked dynamically
if test "x$static_link" = "xtrue"; then
 if test "x$CXX" = "xicpc"; then
 	LDFLAGS="$LDFLAGS -static-intel"
 else
	LDFLAGS="$LDFLAGS -static"
 fi
fi

# tiff (if we're not using wx's builtin tiff support)
#
AC_CHECK_LIB([wxtiff-3.0],[TIFFOpen],[WX_LIBS_BASE="-lwxtiff-3.0 $WX_LIBS_BASE"],[wxtiff=0],`$WXCONFIG --libs base`)
if test "x$wxtiff" = "x0"; then
  AC_SEARCH_LIBS([TIFFOpen],[tiff],[],[AC_MSG_ERROR(Could not find your installation of the TIFF library)])
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



AC_SUBST(WX_LIBS)
AC_SUBST(WX_CPPFLAGS)
AC_SUBST(WX_LIBS_BASE)
AC_SUBST(WX_CPPFLAGS_BASE)

AC_CONFIG_FILES([Makefile src/Makefile])

AC_OUTPUT
