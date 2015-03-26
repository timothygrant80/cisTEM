AC_INIT(ProjectX, 0.1)
AM_INIT_AUTOMAKE
AC_CONFIG_MACRO_DIR([m4])

CPPFLAGS=$CPPFLAGS
CXXFLAGS=$CXFLAGS
CFLAGS=$CFLAGS

AC_PROG_CXX
AC_PROG_INSTALL
AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL
AC_LANG(C++)

#AC_ARG_VAR(MPICXX,[MPI C++ compiler command])
#AC_CHECK_PROGS(MPICXX, mpic++ mpicxx mpiCC hcp mpxlC_r mpxlC mpCC cmpic++, $CXX)

#MPI_CPPFLAGS="`$MPICXX -showme:compile`"
#MPI_LIBS="`$MPICXX -showme:link`"

CPPFLAGS="$CPPFLAGS -DNDEBUG -funroll-loops -O3 -Wall -pipe -fexpensive-optimizations -ffast-math -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_GNU_SOURCE"
CXXFLAGS="$CXXFLAGS -DNDEBUG -funroll-loops -O3 -Wall -pipe -fexpensive-optimizations -ffast-math -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_GNU_SOURCE"
   

AC_ARG_ENABLE(debugmode, [  --enable-debugmode      Compile in debug mode [[default=no]]],[
  if test "$enableval" = yes; then
  CPPFLAGS="-g -pg -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_GNU_SOURCE -DDEBUG"
  CXXFLAGS="-g -pg -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_GNU_SOURCE -DDEBUG"
  fi
  ])

#MPI_CPPFLAGS="$MPI_CPPFLAGS $CPPFLAGS"


WXCONFIG=wx-config
AC_ARG_WITH(wx-config,
[[  --with-wx-config=FILE   Use the given path to wx-config when determining
                          wxWidgets configuration; defaults to "wx-config"]],
[
    if test "$withval" != "yes" -a "$withval" != ""; then
        WXCONFIG=$withval
    fi
])

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
	WX_LIBS="`$WXCONFIG --libs std,aui`"
	wx_is_available=1
else
        AC_MSG_ERROR("wxWidgets version > 3.0.0 required.")
	wx_is_available=0
fi

## Is this wx 3?
#if test -n "$vers" && test "$vers" -ge 3000000; then
#        WX_CPPFLAGS="$WX_CPPFLAGS -DWX_3"
#        WX_CXXFLAGS="$WX_CXXFLAGS -DWX_3"
#fi


# Do FFTW3 Checks

AC_CHECK_HEADER(fftw3.h, , AC_MSG_ERROR("Can't find fftw3.h. Please check your installation of FFTW3."))
AC_CHECK_LIB(fftw3, fftw_import_wisdom_from_file, [LIBS="$LIBS -lfftw3"],AC_MSG_ERROR("Can't find FFTW3's libraries. Please check your installation of FFTW3."))

# MPI Checks..

#lets see if we can compile a sample MPI program

#ORIGINAL_LD_FLAGS=$LDFLAGS
#ORIGINAL_CPPFLAGS=$CPPFLAGS
#ORIGINAL_CXXFLAGS=$CXXFLAGS
#ORIGINAL_LIBS=$LIBS

#CXXFLAGS="$MPI_CPPFLAGS"
#CPPFLAGS="$MPI_CPPFLAGS"
#LIBS="$MPI_LIBS"

#AC_MSG_CHECKING([mpi compilation])
#AC_LINK_IFELSE(  [AC_LANG_PROGRAM([[#include <mpi.h>]],
#                                    [[MPI_Finalize();]])],
#                   [
#                     AC_MSG_RESULT([passed])
#                     MPI_COMPILER=1   
#                     ORIGINAL_CPPFLAGS="$ORIGINAL_CPPFLAGS -DWITH_MPI"
#                     ORIGINAL_CXXFLAGS="$ORIGINAL_CXXFLAGS -DWITH_MPI"
#                   ],
#                   [
#                     AC_MSG_RESULT([failed])
#                     MPI_COMPILER=0
#                   ]
#               )

#CPPFLAGS="$ORIGINAL_CPPFLAGS"
#CXXFLAGS="$ORIGINAL_CXXFLAGS"
#LIBS="$ORIGINAL_LIBS"

#AC_ARG_ENABLE(textonly, [  --enable-textonly       Do not compile wxWidgets applications [[default=no]]],[
#  if test "$enableval" = yes; then
#  wx_is_available=0
#  fi  
#  ])


#if test "$MPI_COMPILER" = 0; then 

#echo
#echo "**    MPI not available, MPI programs will not be compiled **"
#echo

#fi

#if test "$wx_is_available" = 0; then

#echo
#echo "**    WxWidgets not available, GUI programs will not be compiled **"
#echo

#fi



#AM_CONDITIONAL([COMPILE_MPI], test "$MPI_COMPILER" != 0)
#AM_CONDITIONAL([WX_AVAILABLE], test "$wx_is_available" != 0)

#AC_SUBST(MPI_COMPILER)
#AC_SUBST(MPI_LIBS)
#AC_SUBST(MPI_LDFLAGS)
#AC_SUBST(MPI_CPPFLAGS)
AC_SUBST(WX_LIBS)
AC_SUBST(WX_CPPFLAGS)

AC_OUTPUT(Makefile src/Makefile)
