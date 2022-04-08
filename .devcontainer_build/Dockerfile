FROM ubuntu:20.04

# /groups/himesb/.config/Code\ -\ Insiders/User/globalStorage/ms-vscode-remote.remote-containers/cli-bin/code-insiders build --image-name bhimesbhimes/cistem_build_env:latest ./.devcontainer_build

# For compiling wxWidgets
ARG n_threads=24
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

RUN apt-get --allow-releaseinfo-change update && apt-get install -y libgtk2.0-dev \
    libtool autoconf autotools-dev nano gedit meld cmake \
    fftw3-dev libtiff-dev software-properties-common libffi-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev libtiff-dev \
    libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget \
    build-essential git xauth zip unzip parallel \
    && rm -rf /var/lib/apt/lists/*



# wx Formbuilder deps For focal
# wxWidgets with gtk3 compiles fine but has weird sizing issues. Can prob be sorted easily, but for now, stick with 2.
# libgtk-3-dev libwxgtk3.0-gtk3-dev 

# Getting a 404?
#RUN cd /tmp && wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run && \
#    sh cuda_11.6.2_510.47.03_linux.run --silent --toolkit && rm -rf cuda_11.6.2_510.47.03_linux.run
RUN cd /tmp && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uNiD5k9fkMiC6JfR6Gqi5LthMe0VA_h0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uNiD5k9fkMiC6JfR6Gqi5LthMe0VA_h0" -O cuda_11.6.0_510.39.01_linux.run && rm -rf /tmp/cookies.txt && \
    sh cuda_11.6.0_510.39.01_linux.run --silent --toolkit && \
    rm cuda_11.6.0_510.39.01_linux.run

# We currently don't need a bunch of things in the cuda toolkit, which installed is something like ~6.4Gb, this reduces to 1.9
RUN cd /usr/local/cuda && rm -rf nsight* extras && \
    cd lib64 && rm libcublas* libcuspars* libcusolver*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get --allow-releaseinfo-change  update && apt install -y gcc-${GCC_VER} g++-${GCC_VER} \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VER} 100 --slave \
    /usr/bin/g++ g++ /usr/bin/g++-${GCC_VER} --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VER} \
    && rm -rf /var/lib/apt/lists/*

# Need newer git for Github actions (requires > git 2.18 so for ubuntu > 20.04 this is standard )
# RUN add-apt-repository ppa:git-core/ppa -y && apt-get --allow-releaseinfo-change update && apt-get install -y git \
#     && rm -rf /var/lib/apt/lists/*


RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2 && \
    echo 'Installing wxWidgets' && \
    tar -xf /tmp/wxwidgets.tar.bz2 -C /tmp && \
    cd /tmp/wxWidgets-3.0.5  && \
    CXXFLAGS=-fPIC CFLAGS=-fPIC  ./configure --disable-precomp-headers --prefix=/usr/local --with-libnotify=no --disable-shared \
    --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin \
    --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs  && \
    make -j$n_threads && \
    make install && \
    rm -rf /tmp/wxWidgets-3.0.5 /tmp/wxwidgets.tar.bz2


RUN cd /tmp && wget https://github.com/wxFormBuilder/wxFormBuilder/releases/download/v3.10.0/wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && \
    apt-get update && apt install -y ./wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && rm wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb

# First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include
RUN tf=`tempfile` && cp /usr/local/include/wx-3.0/wx/longlong.h /usr/local/include/wx-3.0/wx/longlong.h.orig && \
    awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /usr/local/include/wx-3.0/wx/longlong.h.orig > $tf && \
    mv $tf /usr/local/include/wx-3.0/wx/longlong.h && \
    chmod a+r /usr/local/include/wx-3.0/wx/longlong.h

# Let's also grab pytorch
RUN cd /tmp && wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && \
    mv libtorch /opt

# Include the lib path in LD_RUN_PATH so on linking, the correct path is known 
ENV LD_RUN_PATH=/opt/libtorch/lib:${LD_RUN_PATH}
# cisTEMx@derp:~$ g++ -L/opt/libtorch/lib -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include test_torch.cpp -o tt -ltorch -lc10 -ltorch_cpu -ltorch_cuda_cppFH

# clang format
# 14 - these goa t the end of /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
RUN apt-get update && apt-get install clang-format-14 -y && \
    cd /usr/bin && ln -s clang-format-14 clang-format

USER cisTEMx
WORKDIR /home/cisTEMx
