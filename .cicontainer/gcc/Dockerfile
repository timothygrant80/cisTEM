FROM dockcross/manylinux-x64

 # Fix yum. CentOS 5 is end-of-life as of 2017-03-31.
RUN sed -i \
    '/^mirrorlist=/d; s|^#\(baseurl=http://\).*/\([^/]*\)/\$basearch/$|\1vault.centos.org/5.11/\2/$basearch|' \
    /etc/yum.repos.d/{CentOS-Base.repo,libselinux.repo}

RUN yum update -y \
    && yum install -y \
    wget \
    fftw-devel \
    gtk2-devel \
    && yum clean all

RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2 && \
    echo 'Installing wxWidgets' && \
    tar -xf /tmp/wxwidgets.tar.bz2 -C /tmp && \
    cd /tmp/wxWidgets-3.0.5  && \
    CXXFLAGS=-fPIC CFLAGS=-fPIC ./configure --disable-precomp-headers --prefix=/usr/local --with-libnotify=no --disable-shared --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs && \
    make -j4 && \
    make install
