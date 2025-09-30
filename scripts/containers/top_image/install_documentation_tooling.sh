#!/bin/bash
set -e  # Exit on any error

# install_documentation_deps.sh - Install all documentation system dependencies
# Designed for Ubuntu Docker containers
# Assumes: Python 3.10 venv at /opt/venv, clang-14 already installed in base image

echo "🚀 Installing Documentation System Dependencies"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verify we're using the venv
print_status "Verifying Python venv..."
if [[ "$VIRTUAL_ENV" != "/opt/venv" ]]; then
    print_error "VIRTUAL_ENV not set to /opt/venv (current: $VIRTUAL_ENV)"
    exit 1
fi

print_status "Using Python: $(which python) ($(python --version))"
print_status "Using pip: $(which pip) ($(pip --version))"

# Update package lists
print_status "Updating package lists..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq

# Install ONLY additional dependencies not in base image
print_status "Installing additional build dependencies for documentation..."
apt-get install -y \
    libyaml-dev \
    libopenjp2-7-dev \
    libwebp-dev \
    libclang-14-dev \
    libclang-common-14-dev \
    libclang1-14

# Install documentation-specific Python packages using venv pip
print_status "Installing Python documentation packages..."

# Core MkDocs and extensions
pip install \
    "mkdocs>=1.5.0" \
    "mkdocs-material>=9.0.0" \
    "mkdocstrings[python]>=0.20.0" \
    "mkdocs-git-revision-date-localized-plugin" \
    "mkdocs-minify-plugin" \
    "mkdocs-autorefs" \
    "pymdown-extensions"

# AST parsing and C++ documentation
pip install \
    "libclang>=16.0.0" \
    "pyyaml>=6.0" \
    "jinja2>=3.0.0" \
    "click>=8.0.0"

# Additional useful packages
pip install \
    "pre-commit>=3.0.0" \
    "pytest>=7.0.0" \
    "black" \
    "flake8" \
    "mypy" \
    "isort"

# JSON and data processing
pip install \
    "jsonschema>=4.0.0" \
    "pydantic>=2.0.0" \
    "requests>=2.28.0"

# Install Git LFS (for large documentation assets) - Note: git already installed in base image
print_status "Installing Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

# Configure Git for container use (Note: git already configured in base image, but safe to re-configure)
print_status "Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global safe.directory '*'

# Set up environment variables for libclang
print_status "Setting up environment variables..."
export LLVM_CONFIG=/usr/bin/llvm-config-14
export CLANG_LIBRARY_PATH=/usr/lib/llvm-14/lib
export LIBCLANG_PATH=/usr/lib/llvm-14/lib/libclang.so.1

# Add environment variables to .bashrc for cisTEMdev user
cat >> /home/cisTEMdev/.bashrc << 'EOF'

# Documentation system environment variables
export LLVM_CONFIG=/usr/bin/llvm-config-14
export CLANG_LIBRARY_PATH=/usr/lib/llvm-14/lib
export LIBCLANG_PATH=/usr/lib/llvm-14/lib/libclang.so.1
EOF

# Test the installation
print_status "Testing installation..."

# Test Python packages (using venv python)
python -c "
import sys
packages = [
    'mkdocs', 'material', 'mkdocstrings',
    'yaml', 'jinja2', 'clang.cindex', 'jsonschema'
]
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError as e:
        print(f'❌ {pkg}: {e}')
        failed.append(pkg)

if failed:
    print(f'\n⚠️  Failed to import: {failed}')
    sys.exit(1)
else:
    print('\n🎉 All Python packages successfully imported!')
"

# Test clang (already installed in base image, symlinks exist)
print_status "Testing Clang installation..."
clang --version || print_warning "Clang test failed"

# Test mkdocs (installed via venv pip)
print_status "Testing MkDocs installation..."
mkdocs --version || print_warning "MkDocs test failed"

# Clean up to reduce image size
print_status "Cleaning up to reduce image size..."
apt-get autoremove -y
apt-get autoclean
rm -rf /var/lib/apt/lists/*
pip cache purge

print_success "Documentation system dependencies installed successfully!"
print_status "Available tools:"
echo "  - Python $(python --version | cut -d' ' -f2) (venv at $VIRTUAL_ENV)"
echo "  - Clang $(clang --version | head -n1 | cut -d' ' -f3)"
echo "  - MkDocs $(mkdocs --version)"
echo "  - Git $(git --version | cut -d' ' -f3)"

print_status "Libclang environment variables added to .bashrc"
print_status "Ready for documentation development! 🚀"
