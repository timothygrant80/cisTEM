#!/bin/bash

n_threads=${1}

wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2

for compiler in gcc icpc clang; do
    mkdir -p /opt/WX/${compiler} 
    tar -xf /tmp/wxwidgets.tar.bz2 -C /opt/WX/${compiler}
    case $compiler in
        gcc)
            CC=gcc
            CXX=g++
            ;;
        icpc)
            CC=icc
            CXX=icpc
            . /opt/intel/oneapi/setvars.sh
            ;;
        clang)
            CC=clang
            CXX=clang++
            ;;
    esac


    cd /opt/WX/${compiler}/wxWidgets-3.0.5
    ./configure \
    --disable-precomp-headers \
    --prefix=/opt/WX/${compiler} \
    --with-libnotify=no --disable-shared \
    --without-gtkprint \
    --with-libjpeg=builtin \
    --with-libpng=builtin \
    --with-libtiff=builtin \
    --with-zlib=builtin \
    --with-expat=builtin \
    --disable-compat28 \
    --without-liblzma \
    --without-libjbig \
    --with-gtk=2 \
    --disable-sys-libs 

    make -j${n_threads}
    make install
    make clean

    rm -rf /opt/WX/${compiler}/wxWidgets-3.0.5

    # First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
    tf=`tempfile` && cp /opt/WX/${compiler}/include/wx-3.0/wx/longlong.h /opt/WX/${compiler}/include/wx-3.0/wx/longlong.h.orig
    awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /opt/WX/${compiler}/include/wx-3.0/wx/longlong.h.orig > $tf 
    mv $tf /opt/WX/${compiler}/include/wx-3.0/wx/longlong.h 
    chmod a+r /opt/WX/${compiler}/include/wx-3.0/wx/longlong.h

done