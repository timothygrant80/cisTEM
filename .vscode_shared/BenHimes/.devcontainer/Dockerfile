FROM ubuntu:20.04

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

RUN apt-get --allow-releaseinfo-change update && apt-get install -y gtk2.0-dev \
    fftw3-dev libtiff-dev software-properties-common libffi-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev libtiff-dev \
    libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run && \
    sh cuda_11.6.2_510.47.03_linux.run --silent --toolkit 

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get --allow-releaseinfo-change  update && apt install -y gcc-${GCC_VER} g++-${GCC_VER} \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VER} 100 --slave \
    /usr/bin/g++ g++ /usr/bin/g++-${GCC_VER} --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VER} \
    && rm -rf /var/lib/apt/lists/*

#Need newer git for Github actions
RUN add-apt-repository ppa:git-core/ppa -y && apt-get --allow-releaseinfo-change update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*


RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2 && \
    echo 'Installing wxWidgets' && \
    tar -xf /tmp/wxwidgets.tar.bz2 -C /tmp && \
    cd /tmp/wxWidgets-3.0.5  && \
    CXXFLAGS=-fPIC CFLAGS=-fPIC ./configure --disable-precomp-headers --prefix=/usr/local --with-libnotify=no --disable-shared --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs && \
    make -j$n_threads && \
    make install

# For modules we want gcc > 1

# get teh cuda toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb && \
    sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb && \
    sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub && \
    sudo apt-get --allow-releaseinfo-change update  && \
    sudo apt-get -y install cuda

USER cisTEMx
WORKDIR /home/cisTEMx
