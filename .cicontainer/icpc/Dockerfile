FROM ubuntu:20.04

# /groups/himesb/.config/Code\ -\ Insiders/User/globalStorage/ms-vscode-remote.remote-containers/cli-bin/code-insiders build --image-name bhimesbhimes/cistem_build_env:latest ./.devcontainer_build

# For compiling wxWidgets
ARG n_threads=16
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/New_York


RUN apt-get update && apt-get -y install tzdata 


RUN apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

# Install useful ubuntu packages libgtk2.0-dev
RUN apt-get --allow-releaseinfo-change update && apt-get install -y \
    libtool autoconf autotools-dev \
    libfftw3-dev libtiff-dev software-properties-common libffi-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev  \
    libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget libgtk2.0-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-empt the intel function that checks intel or not at runtime so that optimal codepaths may be used on AMD procs.
RUN mkdir /opt/intel && echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/intel/fakeIntel.c && \
    gcc -shared -fPIC -o /opt/intel/libfakeIntel.so /opt/intel/fakeIntel.c 

ENV LD_PRELOAD=/opt/intel/libfakeIntel.so    

# Get the MKL and intel compiler: note, this is 19G by default, will try to determine minimal set needed huge waste but works well enough for now, final size is ~5gb

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2021.4.0 intel-oneapi-mkl-2021.4.0 intel-oneapi-mkl-devel-2021.4.0 \
    && cd /opt/intel/oneapi/ && rm -rf debugger/ conda_channel/ mkl/latest/lib/intel64/*_sycl* \
    && rm -rf /var/lib/apt/lists/* 


#Need newer git for Github actions
RUN add-apt-repository ppa:git-core/ppa -y && apt-get --allow-releaseinfo-change update && apt-get install -y git

RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2 && \
    echo 'Building wxWidgets but deferring install' && \
    mkdir -p /opt/WX/intel-static && mkdir -p /opt/WX/intel-dynamic && \
    tar -xf /tmp/wxwidgets.tar.bz2 -C /opt/WX/intel-static && \
    rm -rf /tmp/wxwidgets.tar.bz2 


RUN . /opt/intel/oneapi/setvars.sh && \
    cd /opt/WX/intel-static/wxWidgets-3.0.5 && \
    CXX=icpc CC=icc ./configure --disable-precomp-headers --prefix=/opt/WX/intel-static --with-libnotify=no --disable-shared \
    --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin \
    --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs  && \
    make -j$n_threads  && make install && make clean && \
    rm -rf /opt/WX/intel-static/wxWidgets-3.0.5

# First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
RUN tf=`tempfile` && cp /opt/WX/intel-static/include/wx-3.0/wx/longlong.h /opt/WX/intel-static/include/wx-3.0/wx/longlong.h.orig && \
    awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /opt/WX/intel-static/include/wx-3.0/wx/longlong.h.orig > $tf && \
    mv $tf /opt/WX/intel-static/include/wx-3.0/wx/longlong.h && \
    chmod a+r /opt/WX/intel-static/include/wx-3.0/wx/longlong.h




