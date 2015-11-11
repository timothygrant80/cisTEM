AC_INIT(ProjectX, 0.1)
AM_INIT_AUTOMAKE
AC_CONFIG_MACRO_DIR([m4])

INPUT_CPPFLAGS="$CXXFLAGS"
INPUT_CXXFLAGS="$CXXFLAGS"

AC_PROG_CXX([icpc g++ gcc])
AC_PROG_CXX
AC_PROG_INSTALL
AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL
AC_LANG(C++)

# don't let the AC_PROG_CXX set default flags, set them back to input..

CXXFLAGS="$INPUT_CXXFLAGS"
CPPFLAGS="$INPUT_CPPFLAGS"

# set up initial flags

if test "x$CXX" = "xicpc"; then
		CXXFLAGS="$CXXFLAGS -O3 -no-prec-div -xHost -no-prec-sqrt -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE"
		CPPFLAGS="$CPPFLAGS -O3 -no-prec-div -xHost -no-prec-sqrt -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE"		
	else
		CXXFLAGS="$CXXFLAGS -DNDEBUG -funroll-loops -O3 -Wall -pipe -fexpensive-optimizations -ffast-math -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE"		
		CPPFLAGS="$CPPFLAGS -DNDEBUG -funroll-loops -O3 -Wall -pipe -fexpensive-optimizations -ffast-math -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE"
fi

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

# debugmode


AC_ARG_ENABLE(debugmode, [  --enable-debugmode      Compile in debug mode [[default=no]]],[
  if test "$enableval" = yes; then
  CPPFLAGS="-O2 -g -pg -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -DDEBUG"
  CXXFLAGS="-O2 -g -pg -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -DDEBUG"
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

#sqlite3-dir

AC_ARG_WITH(sqlite3-dir,
[[  --with-sqlite3-dir=DIR     Declare the root directory of sqlite3, if its
                          current location is not system accessible ]],
[
    if test "$withval" != "yes" -a "$withval" != ""; then
        CPPFLAGS="$CPPFLAGS -I$withval/include -L$withval/lib"
        CXXFLAGS="$CXXFLAGS -I$withval/include -L$withval/lib"
        LDFLAGS="$LDFLAGS -L$withval/lib"
    fi
])

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
vers=`echo $wxversion | $AWK 'BEGIN { FS = "."; } { printf "% d", ($1 * 1000 + $2) * 1000 + $3;}'`
if test -n "$vers" && test "$vers" -ge 3000000; then
        WX_CPPFLAGS="`$WXCONFIG --cppflags`"
        WX_CXXFLAGS="`$WXCONFIG --cxxflags | sed -e 's/-fno-exceptions//'`"
        WX_LIBS="`$WXCONFIG --libs richtext,std,aui`"

        WX_CPPFLAGS_BASE="`$WXCONFIG --cppflags base`"
        WX_CXXFLAGS_BASE="`$WXCONFIG --cxxflags base | sed -e 's/-fno-exceptions//'`"
        WX_LIBS_BASE="`$WXCONFIG --libs base,net`"
        wx_is_available=1
else
        AC_MSG_NOTICE("wxWidgets version > 3.0.0 required. wxWidgets disabled")
        wx_is_available=0
fi

# Do FFTW3 Checks

# Check whether Intel's MKL is available. If not, link against FFTW.
AC_CHECK_HEADER(fftw3.h, , AC_MSG_ERROR("Can't find fftw3.h"))
AC_CHECK_LIB([mkl_rt],[fftwf_malloc],[HAVE_MKL="yes"],[HAVE_MKL="no"])

if test "x$HAVE_MKL" = "xyes"; then

	AC_MSG_NOTICE(Will use Intel MKL rather than FFTW for fast Fourier transforms)
	
	if test "x$CXX" = "xicpc"; then
		CPPFLAGS="$CPPFLAGS -mkl"
		LIBS="$LIBS -mkl"
	else
       		LIBS="$LIBS -lmkl_rt"
       		LDFLAGS="$LDFLAGS -L\$(MKLROOT)/lib/intel64"
	fi
else

	AC_CHECK_LIB(fftw3f, fftw_import_wisdom_from_file, [LIBS="$LIBS -lfftw3f"],AC_MSG_ERROR("Can't find FFTW3's libraries. Please check your installation of FFTW3."))

fi

# SQLITE 3 Checks

AC_CHECK_HEADER(sqlite3.h, , AC_MSG_ERROR("Can't find sqlite3.h. Please check your installation of SQLITE3."))
AC_CHECK_LIB(sqlite3, sqlite3_open, [LIBS="$LIBS -lsqlite3"],AC_MSG_ERROR("Can't find SQLITE3's libraries. Please check your installation of SQLITE3."))

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

#AM_CONDITIONAL([WX_AVAILABLE], test "$wx_is_available" != 0)
AC_SUBST(WX_LIBS)
AC_SUBST(WX_CPPFLAGS)
AC_SUBST(WX_LIBS_BASE)
AC_SUBST(WX_CPPFLAGS_BASE)


AC_OUTPUT(Makefile src/Makefile)
