FROM ubuntu:20.04

ARG n_threads=16
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/New_York
ARG GCC_VER=11

COPY get_and_build_wx.sh /tmp/get_and_build_wx.sh
RUN apt-get update && apt-get -y install tzdata 


RUN apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

RUN apt-get --allow-releaseinfo-change update && apt-get install -y \
    libgtk2.0-dev libgtk-3-dev \
    libtool autoconf autotools-dev nano gedit meld cmake \
    libfftw3-dev libtiff-dev software-properties-common libffi-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev libtiff-dev \
    libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget \
    build-essential git xauth zip unzip parallel sqlite3 python3 python3-pip curl gdb  \
    && rm -rf /var/lib/apt/lists/*
# Install clang format
# 14 - these goa t the end of /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-get update && apt-get install -y clang-format-14 clang-14 clang-tidy-14 && \
    cd /usr/bin && ln -s clang-format-14 clang-format && ln -s clang-14 clang \
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

RUN /tmp/get_and_build_wx.sh




