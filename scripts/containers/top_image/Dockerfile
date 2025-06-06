# syntax=docker/dockerfile:1
FROM fake_repo

# To allow a variety of common options, arguments may be passed to the build_container script, 
# which have the side-effect that this top-level layer is rebuilt without cache every time.
#
# Generally, the thinking is that all top-level additions are relatively cheap to build, although this may
# not be quite true for WX.

# Options are still somewhat restricted
#   build-type = linking is static or dynamic
#   build-compiler = icpc or g++
#   build-wx-version = old (3.0.5) or new (3.1.5) NOTE: we probably want to test 3.2 instead
#   build-npm = false [ default ] or true if specified
#   build-ref-images = false [ default ] or true if specified
ARG n_threads=12
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/New_York
ARG GCC_VER=11
# note "-" in the variable seems to break the conditional statements.
ARG build_type="static"
ARG build_compiler="icpc"
ARG build_wx_version="stable"
ARG build_npm="false"
ARG build_ref_images="false"

SHELL ["/bin/bash", "-c"]
# some rebuild comment
ENV CISTEM_REF_IMAGES=/cisTEMdev/cistem_reference_images

# Install wxWidgets
COPY install_wx_3.1.5.sh install_node_16.sh /tmp/


# If we do not do this, we want to link the static or dynamic libs from /opt/WX to /usr/bin so we don't need to set it on configure lines and also so that wxformbuilder can find them
RUN echo "checking for wxWidgets 3.1.5" && if [[ "x${build_wx_version}" == "xdev" ]]; then /tmp/install_wx_3.1.5.sh  ; else echo "linking the (${build_type}) wx-config system wide"  && ln -sf /opt/WX/icc-${build_type}/bin/wx-config /usr/bin/wx-config ; fi

# Get reference images for testing and debugging
RUN mkdir -p /opt && pip3 install gdown toml mrcfile numpy matplotlib scipy && cd /opt && gdown --fuzzy https://drive.google.com/file/d/12OiZIkfm4YF61lJo5-EEVc264pYD13OJ/view?usp=sharing && tar -xjvf FastFFT_forBuild.tar.bz2 && rm FastFFT_forBuild.tar.bz2 && mv FastFFT_forBuild /opt/FastFFT
RUN if [[ "x${build_ref_images}" == "xtrue" ]]; then mkdir -p /cisTEMdev && pip3 install gdown toml mrcfile numpy matplotlib scipy && cd /cisTEMdev && gdown --fuzzy https://drive.google.com/file/d/197sE_pO4FWmjCo0zlqRJXAxN2BLbmHS_/view?usp=sharing && tar -xjvf cistem_reference_images_fp32.tar.bz2 && rm cistem_reference_images_fp32.tar.bz2 ;fi


# Will this work with wx 3.0.5?

# Install wxFormbuilder
# Note, this will ignore the stable dynamic wx at /opt/WX/intel-dynamic and install libwxbase3.0-0v5 and libwxgtk3.0-gtk3-0v5 and libwxgtk-media3.0-gtk3-0v5
RUN cd /opt/WX && \
    apt-get update && apt install -y ./wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && rm wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# # Install Node 16
RUN echo "build npm" && if [[ "x${build_npm}" == "xtrue" ]] ; then /tmp/install_node_16.sh ; fi

# TODO: this flag doesn't exist in the build script. Relocating from the base image to the top image
RUN if [[ "x${build_pytorch}" == "xtrue" ]]; then cd /tmp && wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip && unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && mv libtorch /opt ; fi


# Include the lib path in LD_RUN_PATH so on linking, the correct path is known 
# TODO: could this be avoided as the installation? Will there be any problems if used without the actual install
# Only set LD_RUN_PATH for non-static builds to avoid warnings
RUN if [[ "x${build_type}" != "xstatic" ]]; then echo "export LD_RUN_PATH=/opt/libtorch/lib:\${LD_RUN_PATH}" >> /etc/profile.d/ld_run_path.sh; fi

# TODO: if adding dynamic linking for the cuda libs, we can save a bunch of space in the image by removing these large static libs
# Use the basename bit to ensure no rm -rf foibles with root dir in empyt string case
# RUN ls /usr/local/cuda/lib64/lib*_static.a | grep -v cufft_static.a | while read a; do rm -rf /usr/local/cuda/lib64/$(basename $a); done && \
#     rm -rf /usr/local/cuda/lib64/libcufft_static_nocallback.a



USER cisTEMdev
WORKDIR /home/cisTEMdev

