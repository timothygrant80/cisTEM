#!/bin/bash
#
# Prepare a ctffind release
#
# Alexis Rohou, 2016
#
# Last modified 06 Jul 2016
#
# Before running this script:
# - update version in this script
# - update version in AC_INIT at beginning of configure.ac
# - update version in ctffind.cpp, near line 4 (const std::string ctffind_version = "4.1.3";)
# - update NEWS, ChangeLog
#
# This script should be run on the Janelia cluster rather than workstations to ensure compatibility
#
version=4.1.6
svn_loc="svn+ssh://praha.hhmi.org/groups/grigorieff/home/grantt/Apps/svnrepos/projectx"
configure_flags="--with-wx-config=/groups/grigorieff/home/grantt/Apps/wxWidgets3_cluster_static/bin/wx-config --disable-debugmode --enable-staticmode --enable-mkl CC=icc CXX=icpc "
configure_flags_no_latest=" --disable-latest-instruction-set ${configure_flags}"
number_of_cores=8
installation_username="grigoriefflab"
installation_prefix="$GWARE/ctffind_${version}"


# Override for tests on laptop
if [[ $(hostname) == "stitt" ]]; then
  installation_username="rohoua"
  installation_prefix="$HOME/work/software/ctffind_${version}"
  SCRATCH="/tmp"
  configure_flags="--disable-debugmode --enable-staticmode --enable-mkl CC=icc CXX=icpc "
  configure_flags_no_latest=" --disable-latest-instruction-set ${configure_flags}"
fi

if [ "${USER}" != "${installation_username}" ]; then
  echo "This script must be run as user ${installation_username}"
  exit
fi

cd $SCRATCH
test -d ctffind_rel_prep || mkdir ctffind_rel_prep
cd ctffind_rel_prep
test -d $version && rm -fr $version
test -d $version || mkdir $version
cd $version


# The line below is needed for su to work
unset SHELL

# Check out ctffind from SVN
if [[ $(hostname) == "stitt" ]]; then
  echo "Copying from workspace..."
  cp -rp ~/work/workspace/ProjectX/* . > cp.log 2>&1
else
  echo "Checking out from SVN..."
  #svn co ${svn_loc}/tags/ctffind_${version} . > checkout.log 2>&1
  svn co ${svn_loc} . > checkout.log 2>&1
fi

#libtoolize --force > libtoolize.log 2>&1
source regenerate_project.b
cd ctffind_standalone

# make sure we are using the latest Intel compiler
if [ -f /usr/local/INTEL2016.sh ]; then
  . /usr/local/INTEL2016.sh
fi
# Prepare for building
source ../regenerate_project.b
#libtoolize --force > libtoolize.log 2>&1
#aclocal > aclocal.log 2>&1
#autoreconf > autoreconf.log 2>&1


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
tar -czf ctffind-${version}-linux64.tar.gz bin/*
cd ${remember_dir}

# Now let's do another build without the latest instruction set. Should be more compatible
echo "Preparing a build for older processors..."
cd $SCRATCH/ctffind_rel_prep/${version}/ctffind_standalone
make distclean
echo "Configuring..."
./configure $configure_flags_no_latest  --prefix ${installation_prefix} --bindir ${installation_prefix}/bin_compat > configure.log 2>&1
echo "Building..."
make -j ${number_of_cores} > make.log 2>&1
echo "Installing"
make install > make_install.log 2>&1
echo "Preparing binary tar ball..."
cd ${installation_prefix}
tar -czf ctffind-${version}-linux64-compat.tar.gz bin_compat/*
cd ${remember_dir}

# Now let's do another build without the latest instruction set and with debug. Should be more compatible but slower.
echo "Preparing a build for older processors with debug..."
cd $SCRATCH/ctffind_rel_prep/${version}/ctffind_standalone
make distclean
echo "Configuring..."
./configure $configure_flags_no_latest  --prefix ${installation_prefix} --enable-debugmode --bindir ${installation_prefix}/bin_compat_dbg > configure.log 2>&1
echo "Building..."
make -j ${number_of_cores} > make.log 2>&1
echo "Installing"
make install > make_install.log 2>&1
echo "Preparing binary tar ball..."
cd ${installation_prefix}
tar -czf ctffind-${version}-linux64-compat-dbg.tar.gz bin_compat_dbg/*
cd ${remember_dir}


# All done
echo "All done. Check ${installation_prefix}"
