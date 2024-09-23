#!/bin/bash

build_wx_version=$1

if [[ "x${build_wx_version}" == "x" ]]; then
    echo "build_wx_version not set"
    exit 1
fi

case $build_wx_version in
    gcc_ci)
        echo "build_wx_version set to gcc for CI"
        /tmp/install_ci_gcc_wx_3.0.5.sh
        echo "linking the (${build_type}) wx-config system wide"  && ln -sf /opt/WX/gcc-dynamic/bin/wx-config /usr/bin/wx-config
        ;;
    clang_ci)
        echo "build_wx_version set to clang for CI"
        /tmp/install_ci_clang_wx_3.0.5.sh
        echo "linking the (${build_type}) wx-config system wide"  && ln -sf /opt/WX/clang-dynamic/bin/wx-config /usr/bin/wx-config
        ;;
    icpc_ci)
        echo "build_wx_version set to icpc for CI"
        /tmp/install_wx_3.0.5_static.sh
        # Prune bloat from this CI container
        rm -rf /usr/local/cuda/*
         echo "linking the (${build_type}) wx-config system wide"  && ln -sf /opt/WX/intel-static/bin/wx-config /usr/bin/wx-config
        ;;
    dev)
        echo "build_wx_version set to dev"
        /tmp/install_wx_3.1.5.sh 
        # This is installed with package manager so no need to link wx-config
        ;;
    stable)
        echo "build_wx_version set to stable"
        /tmp/install_wx_3.0.5_static.sh
        echo "linking the (${build_type}) wx-config system wide"  && ln -sf /opt/WX/intel-static/bin/wx-config /usr/bin/wx-config
        ;;
    *)
        echo "build_wx_version set to unknown"
        exit 1
        ;;
esac

