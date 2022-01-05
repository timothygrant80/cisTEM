FROM intel/oneapi-hpckit

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y gtk2.0-dev \
        fftw3-dev libtiff-dev software-properties-common libffi-dev \
        libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev libtiff-dev \
        libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget 

#Need newer git for Github actions
RUN add-apt-repository ppa:git-core/ppa -y && apt-get --allow-releaseinfo-change update && apt-get install -y git

RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2 && \
    echo 'Installing wxWidgets' && \
    tar -xf /tmp/wxwidgets.tar.bz2 -C /tmp && \
    cd /tmp/wxWidgets-3.0.5  && \
    CXX=/opt/intel/oneapi/compiler/2021.4.0/linux/bin/intel64/icpc CC=/opt/intel/oneapi/compiler/2021.4.0/linux/bin/intel64/icc CXXFLAGS=-fPIC CFLAGS=-fPIC ./configure --disable-precomp-headers --prefix=/usr/local --with-libnotify=no --disable-shared --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs && \
    make -j4 && \
    make install
