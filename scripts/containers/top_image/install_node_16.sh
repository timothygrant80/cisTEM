#!/bin/bash

# Called from the top layer Dockerfile
# Used to make conditionals easier

curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*  && \
    npm install -g pnpm