#!/bin/bash

set -eo pipefail

cd /tmp && \
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcpu.zip -O libtorch.zip && \
unzip -q libtorch.zip && \
rm libtorch.zip && \
mv libtorch /opt/libtorch 