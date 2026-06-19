# syntax=docker/dockerfile:1
FROM fake_repo

# CI-specific top layer: minimal additions on top of the base image.
# Only includes what's needed to configure, compile, and run CI tests.
#
# What CI needs (from .github/workflows/):
#   - Intel oneAPI (icc/icpc + MKL) — in base image
#   - CUDA toolkit — in base image (for GPU compile-only builds)
#   - wxWidgets 3.0.5 static builds (icc, gcc, clang) — in base image
#   - GNU autotools — in base image
#   - Build dependencies (fftw, libtiff, gtk, etc.) — in base image
#
# What CI does NOT need (stripped from this image):
#   - Python 3.10 venv + pip packages (no CI step invokes python)
#   - LibTorch (CI never passes --enable-libtorch)
#   - FastFFT data (CI always passes --disable-FastFFT)
#   - wxFormBuilder (GUI tool)
#   - flameshot (screenshot tool)
#   - Node.js (not used by CI)
#   - Documentation tooling (mkdocs, libclang-dev, etc.)
#   - LLM optimization tools (tiktoken, fastembed)
#   - Claude CLI
#   - Reference images (tests use embedded data)

ARG build_type="static"
ARG build_compiler="icpc"
ARG build_wx_version="stable"

SHELL ["/bin/bash", "-c"]

# Link the appropriate wx-config system wide so configure can find it.
# CI workflows reference /opt/WX/{compiler}-static/bin/wx-config directly,
# but the symlink provides a sensible default.
RUN if [[ "x${build_wx_version}" == "xdev" ]]; then \
      echo "ERROR: dev wxWidgets not supported in CI image" && exit 1 ; \
    else \
      echo "linking the (${build_type}) wx-config system wide" && \
      ln -sf /opt/WX/icc-${build_type}/bin/wx-config /usr/bin/wx-config ; \
    fi

USER cisTEMdev
WORKDIR /home/cisTEMdev
