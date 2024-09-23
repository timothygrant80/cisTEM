#!/bin/bash

# Called from the top layer Dockerfile
# Used to make conditionals easier

cd /opt/WX/gcc-dynamic/wxWidgets-3.0.5 && make install && make clean
cd /opt/WX
rm -rf /opt/WX/gcc-dynamic/wxWidgets-3.0.5

# First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
tf=`tempfile` && cp /opt/WX/gcc-dynamic/include/wx-3.0/wx/longlong.h /opt/WX/gcc-dynamic/include/wx-3.0/wx/longlong.h.orig && \
    awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /opt/WX/gcc-dynamic/include/wx-3.0/wx/longlong.h.orig > $tf && \
    mv $tf /opt/WX/gcc-dynamic/include/wx-3.0/wx/longlong.h && \
    chmod a+r /opt/WX/gcc-dynamic/include/wx-3.0/wx/longlong.h



