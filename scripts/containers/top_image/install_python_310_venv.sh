#!/usr/bin/env bash
set -euo pipefail

# Install prerequisites and add deadsnakes PPA (non-interactive)
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends software-properties-common ca-certificates gnupg
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

# Install Python 3.10 and venv tooling
apt-get install -y --no-install-recommends \
  python3.10 python3.10-venv python3.10-distutils python3.10-dev

# Create venv at /opt/venv with Python 3.10
python3.10 -m venv /opt/venv

# Optional: pre-upgrade pip/setuptools/wheel inside the venv
/opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# Make venv world-readable/executable so non-root can use it (adjust as needed)
chmod -R a+rX /opt/venv
