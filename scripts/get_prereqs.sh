#!/bin/bash

# You should probably review this and just use as notes. As it is, this script does things like remove an existing docker installation, which may not be something you want.
# exit 0

# Probably still work through the forking mechanism, which people will set up on github
# https://github.com/bHimes/cisTEMx.git

# Once you have your fork, pull down the repo locally. 
# It seems build times can be strongly impacted by disk i/o, consider this when choosing where you put your local repo.
git clone git@github.com:bHimes/cisTEMx.git

# Prep to insall docker. If you have an older version on your system, uncomment the next line to remove it.
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update

# These should already be on lab system
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add dockers official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add docker repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# install
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# add yourself to the docker group
usermod -aG docker $USER
newgrp docker # this is nice so you don't have to logout/in

# Install vscode (I use code-insiders, this is up to you)
sudo apt install software-properties-common apt-transport-https 
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"

sudo apt install code-insiders



