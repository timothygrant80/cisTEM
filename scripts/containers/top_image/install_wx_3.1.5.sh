#!/bin/bash

# Called from the top layer Dockerfile
# Used to make conditionals easier

# Install wxWidgets 3.1.5 - this is used when webview is built (and must be configured as such TODO add check)
wget -qO - https://repos.codelite.org/CodeLite.asc | apt-key add - && \
apt-add-repository 'deb https://repos.codelite.org/wx3.1.5/ubuntu/ groovy universe' && \
apt-get update && apt-get install -y \
libwxbase3.1-0-unofficial \
libwxbase3.1unofficial-dev \
libwxgtk3.1-0-unofficial \
libwxgtk3.1unofficial-dev \
wx3.1-headers \
wx-common \
libwxgtk-media3.1-0-unofficial \
libwxgtk-media3.1unofficial-dev \
libwxgtk-webview3.1-0-unofficial \
libwxgtk-webview3.1unofficial-dev \
libwxgtk-webview3.1-0-unofficial-dbg \
libwxbase3.1-0-unofficial-dbg \
libwxgtk3.1-0-unofficial-dbg \
libwxgtk-media3.1-0-unofficial-dbg \
&& rm -rf /var/lib/apt/lists/*

# First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
tf=`tempfile` && cp /usr/include/wx-3.1-unofficial/wx/longlong.h /usr/include/wx-3.1-unofficial/wx/longlong.h.orig && \
    awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /usr/include/wx-3.1-unofficial/wx/longlong.h.orig > $tf && \
    mv $tf /usr/include/wx-3.1-unofficial/wx/longlong.h && \
    chmod a+r /usr/include/wx-3.1-unofficial/wx/longlong.h