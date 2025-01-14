# This file was originally based on ideas in:
# ax_cuda.m4: An m4 macro to detect and configure Cuda
# Copyright © 2008 Frederic Chateau <frederic.chateau@cea.fr>
#
# It substantially different in purpose and implementation and as such is not a derivative work.


AC_DEFUN([AX_CUDA],
[

# Default install for cuda
default_cuda_home_path="/usr/local/cuda"

msg="Checking for static linking ($static_link) of CUDA libs"
AC_MSG_NOTICE([$msg])


# In Thrust 1.10 c++11 is deprecated. Not using those libs now, so squash warnings, but we should consider switching to a newer standard
AC_DEFINE(THRUST_IGNORE_DEPRECATED_CPP_11, [], true)
AC_MSG_NOTICE([Checking for gpu request and vars])
AC_ARG_WITH([cuda], AS_HELP_STRING([--with-cuda@<:@=yes|no|DIR@:>@], [prefix where cuda is installed (default=no)]),
[
	with_cuda=$withval
	if test "$withval" = "yes" ; then
		want_cuda="yes"
		cuda_home_path=$default_cuda_home_path
	else 
		if test "$withval" = "no" ; then
			want_cuda="no"
		else
			want_cuda="yes"
			cuda_home_path=$withval
		fi
	fi
	
], [ want_cuda="no"] )

if test "$want_cuda" = "yes" ; then

	# check that nvcc compiler is in the path
	if test -n "$cuda_home_path"
	then
	    nvcc_search_dirs="$PATH$PATH_SEPARATOR$cuda_home_path/bin"
	else
	    nvcc_search_dirs=$PATH
	fi

	AC_PATH_PROG([NVCC], [nvcc], [], [$nvcc_search_dirs])
	if test -n "$NVCC"
	then
		have_nvcc="yes"
	else
		have_nvcc="no"
	fi

	# test if nvcc version is >= 2.3
	NVCC_VERSION=`$NVCC --version | grep release | awk 'gsub(/,/, "") {print [$]5}'`
	AC_MSG_RESULT([nvcc version : $NVCC_VERSION $NVCC_VERSION])
  # I don't like relying on parsing strings, but this works fine for cuda 8-11.3
  is_cuda_ge_11=`echo $NVCC_VERSION | awk '{if([$]1 < 11.0) print 0 ; else print 1}'`
	
 	# we'll only use 64 bit arch
  	libdir=lib64

	# set CUDA flags for static compilation. This is required for cufft callbacks.
    if test "x$static_link" == "xtrue"
	then
      AC_MSG_NOTICE([static linking of cuda libs])
	  CUDA_CFLAGS="-I$cuda_home_path/include "
      CUDA_LIBS="-L$cuda_home_path/$libdir -lcufft_static -lnppial_static -lnppist_static -lnppc_static -lnppidei_static -lnppitc_static -lcurand_static -lculibos -lcudart_static -lrt"
	else
      AC_MSG_NOTICE([dynamic linking of cuda libs])
      CUDA_CFLAGS="-I$cuda_home_path/include "
	  CUDA_LIBS="-L$cuda_home_path/$libdir -lcufft_static -lnppial -lnppist -lnppc -lnppidei -lnppitc -lcurand -lculibos -lcudart -lrt"
      # Note: lcutensor requires lcublasLt and it must be listed *after* it in the link line.
      # CUDA_LIBS="-L/opt/cuTensor/lib/cistem_version -L$cuda_home_path/$libdir -lcufft_static -lnppial -lnppist -lnppc -lnppidei -lnppitc -lcurand  -lcutensor -lcublasLt -lculibos -lcudart -lrt"
	fi


	saved_CPPFLAGS=$CPPFLAGS
	saved_LIBS=$LIBS
  	saved_CUDA_LIBS=$CUDA_LIBS

	# Env var CUDA_DRIVER_LIB_PATH can be used to set an alternate driver library path
	# this is usefull when building on a host where only toolkit (nvcc) is installed
	# and not driver. Driver libs must be placed in some location specified by this var.
	if test -n "$CUDA_DRIVER_LIB_PATH"
	then
	    CUDA_LIBS+=" -L$CUDA_DRIVER_LIB_PATH -lcuda"
	else
	    CUDA_LIBS+=" -lcuda"
	fi

	CPPFLAGS="$CPPFLAGS $CUDA_CFLAGS"
	LIBS="$LIBS $CUDA_LIBS"

	AC_LANG_PUSH(C)
	AC_MSG_CHECKING([for Cuda headers])
  	AC_MSG_NOTICE([cuda path is $cuda_home_path])
	AC_COMPILE_IFELSE(
	[
		AC_LANG_PROGRAM([@%:@include <cuda.h>], [])
	],
	[
		have_cuda_headers="yes"
		AC_MSG_RESULT([yes])
	],
	[
		have_cuda_headers="no"
		AC_MSG_RESULT([not found])
	])


	AC_LANG_POP(C)

	CPPFLAGS=$saved_CPPFLAGS
	LIBS=$saved_LIBS
  	CUDA_LIBS=$saved_CUDA_LIBS
	


		have_cuda="yes"

fi

# This is the code that will be generated at compile time and should be specified for the most used gpu 
# TODO: export target_arch to link against pre-built FastFFT that has the same target arch
target_arch=""
AC_ARG_WITH([target-gpu-arch], AS_HELP_STRING([--with-target-gpu-arch@<:@=70,75,80,86,89,90@:>@], [Primary architecture to compile for (default=86)]),
[
	if test "$withval" = "90" ; then target_arch=90
	elif  test "$withval" = "89" ; then target_arch=89
	elif  test "$withval" = "86" ; then target_arch=86 
	elif  test "$withval" = "80" ; then target_arch=80
	elif  test "$withval" = "75" ; then target_arch=75
	elif  test "$withval" = "70" ; then target_arch=70
		else
		AC_MSG_ERROR([Requested target-gpu-arch must be in 70,75,80,86,89,90 not $withval])
	fi
	
], [ target_arch="86"] )
AC_MSG_NOTICE([target gpu architecture is sm$target_arch])

# Default nvcc flags
NVCCFLAGS=" -ccbin $CXX"
NVCCFLAGS+=" --gpu-architecture=sm_$target_arch -gencode=arch=compute_$target_arch,code=compute_$target_arch"

# This is the oldest arch that will have JIT-able code g
oldest_arch=""
AC_ARG_WITH([oldest-gpu-arch], AS_HELP_STRING([--with-oldest-gpu-arch@<:@=70,75,80,86,89,90:>@], [Oldest architecture make compatible for (default=80)]),
[
	if test "$withval" = "90" ; then oldest_arch=90
	elif  test "$withval" = "89" ; then oldest_arch=89
	elif  test "$withval" = "86" ; then oldest_arch=86 
	elif  test "$withval" = "80" ; then oldest_arch=80
	elif  test "$withval" = "75" ; then oldest_arch=75
	elif  test "$withval" = "70" ; then oldest_arch=70
		else
		AC_MSG_ERROR([Requested target-oldest_arch must be in 70,75,80,86,89 not $withval])
	fi
	
], [ oldest_arch="80"] )
AC_MSG_NOTICE([oldest gpu architecture is sm$oldest_arch])

if test "$oldest_arch" -gt "$target_arch" ; then 
	AC_MSG_ERROR([Requested target-oldest_arch is greater than the target arch.]) 
else

	current_arch="70"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -ne "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi	
	
	current_arch="75"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -ne "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi	
	
	current_arch="80"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -ne "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi		

	current_arch="86"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -ne "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi	

	current_arch="89"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -ne "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi	

	current_arch="90"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -ne "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi	
		
fi

if test "x$is_cuda_ge_11" == "x1" ; then
  AC_MSG_NOTICE([CUDA >= 11.0, enabling --extra-device-vectorization])
  NVCCFLAGS+=" --extra-device-vectorization -std=c++17 --expt-relaxed-constexpr --threads=8 --split-compile=8 " 
else
  NVCCFLAGS+=" -std=c++11" 
fi
  
#--extra-device-vectorization
# -Xcompiler= -DGPU -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_DLFCN_H=1"
NVCCFLAGS+=" --default-stream per-thread -m64 -O3 --use_fast_math  -Xptxas --warn-on-local-memory-usage,--warn-on-spills,--warn-on-double-precision-use,--generate-line-info "

AC_ARG_ENABLE(gpu-cache-hints, AS_HELP_STRING([--disable-gpu-cache-hints],[Do not use the intrinsics for cache hints]),[
  if test "$enableval" = no; then
  	NVCCFLAGS+=" -Xcompiler= -DDISABLECACHEHINTS"
  	AC_MSG_NOTICE([Disabling cache hint intrinsics requiring CUDA 11 or newer])  	
  fi])
  
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(NVCCFLAGS)
])
