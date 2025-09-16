#!/bin/bash

# For claude code.

# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# in lieu of restarting the shell
. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 22

# Verify the Node.js version:
node -v # Should print "v22.19.0".

# Verify npm version:
npm -v # Should print "10.9.3".

# Install the Claude Code CLI globally using npm:
npm install -g @anthropic-ai/claude-code