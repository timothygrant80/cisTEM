#!/bin/bash
#
# Prepare a ctffind release
#
# Alexis Rohou, 2016
#
# Last modified 6 Jun 2017 (adapt to Genentech)
#
# Before running this script:
# - update version in this script
# - update version in AC_INIT at beginning of configure.ac
# - update version in ctffind.cpp, near line 4 (const std::string ctffind_version = "4.1.3";)
# - update NEWS, ChangeLog
#
# This script should be run on an older node on the cluster, to ensure compatibility. I think we want glibc <= 2.18
# % bsub -n 8 -R "span[hosts=1]" -x -Is -XF bash
#
version=4.1.10
svn_loc="https://github.com/ngrigorieff/cisTEM/trunk"
svn_rev="HEAD"
svn_rev="563"
#configure_flags="--with-wx-config=/groups/grigorieff/home/grantt/Apps/wxWidgets3_cluster_static/bin/wx-config --disable-debugmode --enable-staticmode --enable-mkl CC=icc CXX=icpc "
configure_flags="--disable-debugmode --enable-staticmode --enable-mkl CC=icc CXX=icpc "
configure_flags_no_latest=" --disable-latest-instruction-set ${configure_flags}"
number_of_cores=8
installation_username="rohoua"
installation_prefix="$HOME/software/ctffind/${version}"
temp_dir="$(mktemp -d)"
temp_dir="$HOME/scratch"

# Check system's libc version
function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }
libc_version="$(ldd --version | awk '/libc/ {print $NF}')"
if version_gt $libc_version "2.18"; then
  echo "Oops. Current system has libc $libc_version, but we need 2.18 or below"
  exit
else
  echo "OK. Current system has libc $libc_version, which is not greater than 2.18"
fi


# Override for tests on laptop
if [[ $(hostname) == "uroy" ]]; then
  installation_username="rohoua"
  installation_prefix="$HOME/work/software/ctffind_${version}"
  configure_flags="--disable-debugmode --enable-staticmode --enable-mkl CC=gcc CXX=g++ "
  configure_flags_no_latest=" --disable-latest-instruction-set ${configure_flags}"
fi

if [ "${USER}" != "${installation_username}" ]; then
  echo "This script must be run as user ${installation_username}"
  exit
fi

cd $temp_dir
test -d ctffind_rel_prep || mkdir ctffind_rel_prep
cd ctffind_rel_prep
test -d $version && rm -fr $version
test -d $version || mkdir $version
cd $version


# The line below is needed for su to work
unset SHELL

# Check out ctffind from SVN
echo "Checking out from SVN..."
svn co -r ${svn_rev} ${svn_loc} . > checkout.log 2>&1


rm -f config.guess config.sub depcomp install-sh missing
cd ctffind_standalone

# If necessary, patch configure.ac
: <<'COMMENTED_OUT'
patch <<eof
Index: configure.ac
===================================================================
--- configure.ac	(revision 333)
+++ configure.ac	(working copy)
@@ -205,7 +205,7 @@
 #
 AC_CHECK_LIB([wxtiff-3.0],[TIFFOpen],[WX_LIBS_BASE="-lwxtiff-3.0 \$WX_LIBS_BASE"],[wxtiff=0],`\$WXCONFIG --libs base`)
 if test "x\$wxtiff" = "x0"; then
-  AC_SEARCH_LIBS([TIFFOpen],[tiff],[],[AC_MSG_ERROR(Could not find your installation of the TIFF library)])
+  AC_SEARCH_LIBS([TIFFOpen],[tiff],[],[AC_MSG_ERROR(Could not find your installation of the TIFF library)],[-ljpeg -lz])
 fi

 # make it so we can turn off gui
eof
COMMENTED_OUT

# make sure we are using the latest Intel compiler, wx, etc
module unload apps/lsf/prod
module load apps/intel
module load apps/wxwidgets
module load apps/tiff
module load apps/jpeg
module load apps/zlib

# Prepare for building
source ../regenerate_project.b

# Configure
echo "Configuring..."
./configure $configure_flags --prefix ${installation_prefix} > configure.log 2>&1

# Prepare and copy tar ball
echo "Preparing source tar ball..."
export TAR_OPTIONS="--owner=0 --group=0 --numeric-owner"
make dist > make_dist.log 2>&1
rm -f ctffind-${version}.tar.gz
autoreconf
make dist > make_dist_2.log 2>&1
mkdir -p ${installation_prefix}
cp -p ctffind*tar.gz ${installation_prefix}/

# Build
echo "Building..."
make -j ${number_of_cores} > make.log 2>&1


# Install
echo "Installing ctffind to ${installation_prefix}..."
make install > make_install.log 2>&1


# Prepare and copy binary tar ball
echo "Preparing binary tar ball..."
remember_dir=$(pwd)
cd ${installation_prefix}
tar -czf ctffind-${version}-linux64.tar.gz bin/ctffind*
cd ${remember_dir}

# Now let's do another build without the latest instruction set. Should be more compatible
echo "Preparing a build for older processors..."
cd $temp_dir/ctffind_rel_prep/${version}/ctffind_standalone
make distclean
echo "Configuring..."
./configure $configure_flags_no_latest  --prefix ${installation_prefix} --bindir ${installation_prefix}/bin_compat > configure.log 2>&1
echo "Building..."
make -j ${number_of_cores} > make.log 2>&1
echo "Installing"
make install > make_install.log 2>&1
echo "Preparing binary tar ball..."
cd ${installation_prefix}
tar -czf ctffind-${version}-linux64-compat.tar.gz bin_compat/ctffind*
cd ${remember_dir}

# Now let's do another build without the latest instruction set and with debug. Should be more compatible but slower.
echo "Preparing a build for older processors with debug..."
cd $temp_dir/ctffind_rel_prep/${version}/ctffind_standalone
make distclean
echo "Configuring..."
./configure $configure_flags_no_latest  --prefix ${installation_prefix} --enable-debugmode --bindir ${installation_prefix}/bin_compat_dbg > configure.log 2>&1
echo "Building..."
make -j ${number_of_cores} > make.log 2>&1
echo "Installing"
make install > make_install.log 2>&1
echo "Preparing binary tar ball..."
cd ${installation_prefix}
tar -czf ctffind-${version}-linux64-compat-dbg.tar.gz bin_compat_dbg/ctffind*
cd ${remember_dir}


# All done
echo "All done. Check ${installation_prefix}"
