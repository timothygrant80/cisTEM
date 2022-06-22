FROM ubuntu:20.04

# /groups/himesb/.config/Code\ -\ Insiders/User/globalStorage/ms-vscode-remote.remote-containers/cli-bin/code-insiders build --image-name bhimesbhimes/cistem_build_env:latest ./.devcontainer_build

# For compiling wxWidgets
ARG n_threads=12
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/New_York
ARG GCC_VER=11

RUN apt-get update && apt-get -y install tzdata


RUN apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

RUN apt-get update && apt-get install -y sudo

RUN useradd -ms /bin/bash cisTEMx
RUN echo "cisTEMx:cisTEMx" | chpasswd && adduser cisTEMx sudo

# Install useful ubuntu packages
RUN apt-get --allow-releaseinfo-change update && apt-get install -y libgtk2.0-dev \
    libtool autoconf autotools-dev nano gedit meld cmake \
    fftw3-dev libtiff-dev software-properties-common libffi-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev libtiff-dev \
    libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget \
    build-essential git xauth zip unzip parallel sqlite3 python3 python3-pip curl  \
    && rm -rf /var/lib/apt/lists/*

# Install newest gcc
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get --allow-releaseinfo-change  update && apt install -y gcc-${GCC_VER} g++-${GCC_VER} \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VER} 100 --slave \
    /usr/bin/g++ g++ /usr/bin/g++-${GCC_VER} --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VER} \
    && rm -rf /var/lib/apt/lists/*


# Get the MKL and intel compiler: note, this is 19G by default, will try to determine minimal set needed huge waste but works well enough for now, final size is ~5gb
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2021.4.0 intel-oneapi-mkl-2021.4.0 intel-oneapi-mkl-devel-2021.4.0 \
    && cd /opt/intel/oneapi/ && rm -rf debugger/ conda_channel/ mkl/latest/lib/intel64/*_sycl* \
    && rm -rf /var/lib/apt/lists/* 

# Minimize the MKL disk footprint
# cd  /opt/intel/oneapi/mkl/latest/lib/intel64/ && \
# rm -f *.a libmkl_sycl*
# Install wxWidgets 3.0.5 for multiple compilers and static linking with only gtk2 deps


RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2 && \
    echo 'Installing wxWidgets' && \
    tar -xf /tmp/wxwidgets.tar.bz2 -C /tmp && \
    cd /tmp/wxWidgets-3.0.5  && \
    mkdir /opt/WX 

# Here for the record if you want to build and link static binaries to avoid the cointainerized distribution
# RUN . /opt/intel/oneapi/setvars.sh  &&  cd /tmp/wxWidgets-3.0.5 && CXX=icpc CC=icc CXXFLAGS=-fPIC CFLAGS=-fPIC  ./configure --disable-precomp-headers --prefix=/opt/WX/intel-static --with-libnotify=no --disable-shared \
#     --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin \
#     --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs  && \
#     make -j$n_threads && \
#     make install && make clean 

# Now build with out static
RUN . /opt/intel/oneapi/setvars.sh  && cd /tmp/wxWidgets-3.0.5 && CXX=icpc CC=icc CXXFLAGS=-fPIC CFLAGS=-fPIC  ./configure --disable-precomp-headers --prefix=/opt/WX/intel-dynamic --with-libnotify=no  \
    --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin \
    --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs  && \
    make -j$n_threads && \
    make install && \
    rm -rf /tmp/wxWidgets-3.0.5 /tmp/wxwidgets.tar.bz2


# # Install wxWidgets 3.1.5 - this is used when webview is built (and must be configured as such TODO add check)
# RUN wget -qO - https://repos.codelite.org/CodeLite.asc | apt-key add - && \
#     apt-add-repository 'deb https://repos.codelite.org/wx3.1.5/ubuntu/ groovy universe' && \
#     apt-get update && apt-get install -y \
#     libwxbase3.1-0-unofficial \
#     libwxbase3.1unofficial-dev \
#     libwxgtk3.1-0-unofficial \
#     libwxgtk3.1unofficial-dev \
#     wx3.1-headers \
#     wx-common \
#     libwxgtk-media3.1-0-unofficial \
#     libwxgtk-media3.1unofficial-dev \
#     libwxgtk-webview3.1-0-unofficial \ 
#     libwxgtk-webview3.1unofficial-dev \ 
#     libwxgtk-webview3.1-0-unofficial-dbg \ 
#     libwxbase3.1-0-unofficial-dbg \
#     libwxgtk3.1-0-unofficial-dbg \
#     libwxgtk-media3.1-0-unofficial-dbg \
#     && rm -rf /var/lib/apt/lists/*


# Install wxFormbuilder
RUN cd /tmp && wget https://github.com/wxFormBuilder/wxFormBuilder/releases/download/v3.10.0/wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && \
    apt-get update && apt install -y ./wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && rm wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Install clang format
# 14 - these goa t the end of /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
    apt-get update && apt-get install -y clang-format-14 -y && \
    cd /usr/bin && ln -s clang-format-14 clang-format \
    && rm -rf /var/lib/apt/lists/*

# Install Node 16
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*  && \
    npm install -g pnpm

# Install pytorch
RUN cd /tmp && wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && \
    mv libtorch /opt
# Include the lib path in LD_RUN_PATH so on linking, the correct path is known 
ENV LD_RUN_PATH=/opt/libtorch/lib:${LD_RUN_PATH}

# Install CUDA
RUN cd /tmp && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uNiD5k9fkMiC6JfR6Gqi5LthMe0VA_h0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uNiD5k9fkMiC6JfR6Gqi5LthMe0VA_h0" -O cuda_11.6.0_510.39.01_linux.run && rm -rf /tmp/cookies.txt && \
    sh cuda_11.6.0_510.39.01_linux.run --silent --toolkit && \
    rm cuda_11.6.0_510.39.01_linux.run && \
    # We currently don't need a bunch of things in the cuda toolkit, which installed is something like ~6.4Gb, this reduces to 1.9
    cd /usr/local/cuda && rm -rf nsight* extras && \
    cd lib64 && rm libcublas* libcuspars* libcusolver*

RUN echo 'alias lt="ls -lrth"' >> /home/cisTEMx/.bashrc && \
    echo 'alias dU="du -ch --max-depth=1 | sort -h"' >> /home/cisTEMx/.bashrc && \
    echo 'source /opt/intel/oneapi/setvars.sh' >> /home/cisTEMx/.bashrc && \
    echo 'bind "set bell-style none"' >> /home/cisTEMx/.bashrc

# # First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
# RUN tf=`tempfile` && cp /usr/include/wx-3.1-unofficial/wx/longlong.h /usr/include/wx-3.1-unofficial/wx/longlong.h.orig && \
#     awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /usr/include/wx-3.1-unofficial/wx/longlong.h.orig > $tf && \
#     mv $tf /usr/include/wx-3.1-unofficial/wx/longlong.h && \
#     chmod a+r /usr/include/wx-3.1-unofficial/wx/longlong.h

# First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
RUN tf=`tempfile` && cp /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h.orig && \
    awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h.orig > $tf && \
    mv $tf /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h && \
    chmod a+r /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h

# RUN tf=`tempfile` && cp /opt/WX/intel-static/include/wx-3.0/wx/longlong.h /opt/WX/intel-static/include/wx-3.0/wx/longlong.h.orig && \
#     awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /opt/WX/intel-static/include/wx-3.0/wx/longlong.h.orig > $tf && \
#     mv $tf /opt/WX/intel-static/include/wx-3.0/wx/longlong.h && \
#     chmod a+r /opt/WX/intel-static/include/wx-3.0/wx/longlong.h

# # First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
# RUN tf=`tempfile` && cp /usr/include/wx-3.1-unofficial/wx/longlong.h /usr/include/wx-3.1-unofficial/wx/longlong.h.orig && \
#     awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /usr/include/wx-3.1-unofficial/wx/longlong.h.orig > $tf && \
#     mv $tf /usr/include/wx-3.1-unofficial/wx/longlong.h && \
#     chmod a+r /usr/include/wx-3.1-unofficial/wx/longlong.h    

USER cisTEMx
WORKDIR /home/cisTEMx

