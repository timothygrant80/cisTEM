#!/usr/bin/env bash
set -euo pipefail


apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev
    
cd /usr/src
curl -O https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz
tar -xf Python-3.10.15.tgz
cd Python-3.10.15
./configure --enable-optimizations --enable-shared --with-ensurepip=install
make -j"$(nproc)"
make altinstall
ldconfig

ln -sf /usr/local/bin/python3.10 /usr/local/bin/python
ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3
ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip
ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3

