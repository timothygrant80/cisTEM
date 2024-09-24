#!/bin/bash

docker build --tag cistemdashorg/cistem_build_env:ci_base_cpu_only ./ 

# docker push cistemdashorg/cistem_build_env:ci_base_cpu_only