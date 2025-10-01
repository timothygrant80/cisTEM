#!/bin/bash

# The default user profile is in .vscode_shared/CistemDev, we link to that for vscode settings. 
# If you are using some custom settings, you can change this to your own profile at .vscode_shared/YourProfile

if [[ ! -L .vscode ]]; then
    ln -s .vscode_shared/CistemDev .vscode
fi

# The container is split into a base and top layer:

#BASE:
    # You should NOT alter the base environment unless there is a change you think everyone needs to incorporate. This is okay to do, but must be 
    # tested and submitted as a PR to the base container defintion in scripts/containers/base.

    # The shared organization repo name is in .vscode_shared/CistemDev/CONTAINER_REPO_NAME and the current version tag is in .vscode_shared/CistemDev/CONTAINER_VERSION_BASE.

#TOP:
    # The default build environment is to use wxWidgets stable (3.0.5), icpc, and to link statically.

    # You may easisly switch between icpc and gcc, static/dynamic, and wxWidgets stable/development (3.1.5) by building your own top layer
    # using scripts/containers/build_container.sh top --args. 
        # See script for args.

# somewhere near vscode 1.98 the devcontainers extension stopped recognizing softlinks to .devcontainer.json
# this is acknowedged as a bug (https://github.com/microsoft/vscode-remote-release/issues/10536)
# As a workaround we create a softlink to .devcontainer.json in the current directory
mkdir -p .devcontainer
cd  .devcontainer
if [[ ! -L .devcontainer.json ]] ; then
    ln -s ../.vscode/devcontainer.json .devcontainer.json
fi
cd ..

# Install clang-format-14 pre-commit hook
if [ -f scripts/install_clang_format_hook.sh ]; then
    echo "Installing clang-format-14 pre-commit hook..."
    ./scripts/install_clang_format_hook.sh
fi