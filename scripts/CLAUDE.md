# Build System and Scripts Guidelines for cisTEM

This file provides guidance for working with cisTEM's build system and utility scripts.

## Build System Overview

cisTEM uses GNU Autotools as the primary build system, with CMake as an alternative. The build system handles complex dependencies including Intel MKL, CUDA, and wxWidgets.

## Autotools Build Process

### Project Regeneration
After modifying build system files:
```bash
# Required after changes to:
# - configure.ac
# - m4/*.m4
# - Makefile.am files
./regenerate_project.b
```

### Configuration and Building
```bash
# Create build directory
mkdir -p build/intel-debug-static
cd build/intel-debug-static

# Configure with common options
CC=icc CXX=icpc ../../configure \
    --enable-debugmode \
    --with-wx-config=/opt/WX/icc-static/bin/wx-config \
    --enable-staticmode \
    --enable-openmp

# Build
make -j16
```

### Common Configure Options
- `--enable-debugmode` - Debug build with assertions
- `--enable-staticmode` - Static linking
- `--enable-gpu` - Enable CUDA support
- `--enable-experimental` - Include experimental features
- `--enable-openmp` - OpenMP parallelization
- `--with-cuda=/usr/local/cuda` - CUDA installation path
- `--with-wx-config=/path/to/wx-config` - wxWidgets configuration

## VS Code Integration

### Task Configuration
Build tasks are defined in `.vscode/tasks.json`:
- `Configure cisTEM DEBUG build` - Run configure
- `BUILD cisTEM DEBUG` - Compile the project
- Various compiler/configuration combinations

### After Making Changes
Always prompt the user to build:
```
"Would you like me to build the project to verify these changes?"
```

## Docker Development Environment

### Container Architecture
```
scripts/containers/
├── base_container/      # Base OS and dependencies
└── top_container/       # Development tools and environment
```

### Container Management
```bash
# Regenerate containers after Dockerfile changes
./regenerate_containers.sh

# The script handles:
# - Building base and top containers
# - Setting up development environment
# - Configuring VS Code integration
```

## Utility Scripts

### Project Scripts
- `regenerate_project.b` - Regenerate autotools files
- `regenerate_containers.sh` - Rebuild Docker containers
- `scripts/testing/run_tests.sh` - Execute test suite

### Build Helper Scripts
Located in `scripts/build/`:
- Helper scripts for different build configurations
- Compiler setup scripts
- Dependency verification

## Adding New Source Files

### Updating Makefile.am
When adding new source files:
```makefile
# In src/gui/Makefile.am
cisTEM_SOURCES += \
    MyNewPanel.cpp \
    MyNewPanel.h

# In src/programs/new_program/Makefile.am
new_program_SOURCES = \
    new_program.cpp \
    ../../core/core_headers.h
```

After updating Makefile.am:
```bash
./regenerate_project.b
# Then reconfigure and rebuild
```

## Testing Scripts

### Running Tests
```bash
# Unit tests
./build/src/unit_test_runner

# Console tests
./build/src/console_test

# Functional tests
./build/src/samples_functional_testing
```

### CI Integration
GitHub Actions workflows in `.github/workflows/`:
- Define test matrices
- Specify compiler configurations
- Run automated tests

## Performance Scripts

### Profiling Tools
Scripts for performance analysis:
```bash
# Intel VTune profiling
scripts/profile/run_vtune.sh program_name

# Memory usage analysis
scripts/profile/check_memory.sh program_name
```

## Common Issues and Solutions

### Dependency Issues
- MKL not found: Check `MKLROOT` environment variable
- wxWidgets issues: Verify `wx-config` path
- CUDA problems: Ensure CUDA toolkit is installed

### Build Failures
- Run `make clean` before rebuilding after configuration changes
- Delete build directory for clean rebuild
- Check compiler versions match requirements

### Parallel Build Issues
- Some targets may have race conditions
- Use `make -j1` for debugging build issues
- Report parallel build failures for fixing

## Best Practices

1. **Always regenerate after build system changes** using `regenerate_project.b`
2. **Use separate build directories** for different configurations
3. **Keep build scripts simple** and well-documented
4. **Test scripts on clean checkout** to ensure reproducibility
5. **Document dependencies** in scripts
6. **Use absolute paths** in scripts when possible
7. **Add error checking** to all scripts
8. **Maintain backward compatibility** in build scripts