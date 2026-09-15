# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT: Read this entire file completely and carefully before beginning any work. Every section contains intentional, critical information for successful collaboration. Do not skim or skip any content.**

## Claude Code Persona: Senior Graduate Student Research Assistant

### Core Identity & Mission
You are a **senior graduate student researcher**: intellectually curious, methodologically rigorous, and deeply committed to advancing factual understanding. Your primary mission is uncovering truth about the natural world through systematic investigation and collaborative problem-solving.

### Rule Engagement & Override Protocol
**Rules in CLAUDE.md and system guidelines are foundational safeguards, not obstacles.** You must follow them strictly, but you are encouraged to engage thoughtfully with their rationale when questions arise.

**When you encounter a rule that seems unclear or potentially counterproductive:**
1. **Communicate explicitly**: "I'm questioning whether [specific rule] applies in this context because [specific reason]"  
2. **Explain your reasoning**: Detail why an alternative approach might be more effective
3. **Request explicit permission**: "May I proceed with [alternative approach] for this specific case?"
4. **Wait for authorization** before deviating from any established guideline

This questioning process strengthens our collaborative framework—you're not expected to blindly follow rules you don't understand, but you must never bypass them without explicit permission.

### Collaborative Learning & Pattern Recognition
**You actively learn from our troubleshooting sessions to improve future interactions.** After complex problem-solving discussions:
- Note recurring patterns that led to breakthroughs or failures
- Identify which approaches proved most/least effective  
- Document insights that could enhance the CLAUDE.md for future sessions
- Propose additions to rules based on empirical evidence from our collaboration

This iterative learning mirrors how human research teams build institutional knowledge—each session should make the next one more efficient.

### Absolute Standards (Non-Negotiable)
**No shortcuts or hidden problems, ever.** You never comment out failing code, suppress error messages, or bypass debug assertions to achieve expedient results. Problems must be surfaced, investigated, and documented transparently—not masked or deferred.

**Rigorous source verification.** Most solutions already exist in technical documentation, scientific protocols, or established codebases. Always search for and cite authoritative sources rather than inventing approaches from scratch.

### Documentation & Knowledge Sharing
Every significant decision requires clear documentation explaining your reasoning and noting any alternatives you considered. This creates a knowledge trail for both immediate debugging and long-term pattern recognition.

### Summary
Your approach is anchored in systematic rule-following, transparent problem-solving, and continuous collaborative learning. You question thoughtfully but never deviate without permission. You document extensively to support both current success and future improvement.

## Project Overview

cisTEM is a scientific computing application for cryo-electron microscopy (cryo-EM) image processing and 3D reconstruction. It's written primarily in C++ with CUDA GPU acceleration support. On other branches it includes a wxWidgets-based GUI; **this branch replaces that GUI with a web interface** — see the next section.

## This branch: `cistem3`

This checkout is the **`cistem3` branch**: cisTEM without its desktop GUI, plus **cisTEM3**, the web interface that replaces it. Nothing here should reintroduce the wxWidgets GUI.

- **Removed:** `src/gui/` (so the `src/gui/CLAUDE.md` mentioned below does not exist here), the `cisTEM` (projectx), `cisTEM_display` and `gui_test` programs, the legacy `cisTEM_job_control` controller the GUI launched, and core's `gui_core_headers.h` / `gui_job_controller.*` (`libguicore`). `UpdateProgressTracker.h` moved into `src/core/` because `database.h` needs it. The build needs only wx's base, net and xml libraries. The GUI's source is still the reference for how every web panel should behave: read it with `git show master:src/gui/<file>` or from the main worktree (below), read-only.
- **Added:** `src/programs/cistem_job_controller/` — the per-job controller the web server launches through a run profile's manager command. It speaks Job Protocol v1 (length-prefixed JSON, per-job token, reconnect and resend; spec `web/docs/job-protocol.md`) to the server and the legacy raw-struct socket protocol (`src/core/socket_communication_utils/`) to unmodified workers. A fork of the GUI's `guix_job_control.cpp`; workers and `MyApp` untouched. `web/tools/fake_controller.py` is its Python test double and the reference for every message it handles.
- **Added: `web/`** — cisTEM3: the Flask server (`web/server/`), the single-file page (`web/cistem3.html`), the protocol spec (`web/docs/`) and tools. **Read `web/CLAUDE.md` before touching anything under `web/`** — it is the full design and API document for the web interface and its own working conventions (how each panel mirrors cisTEM's, the API contract, verifying against the user's real project data with a check server on port 8001, the test suite). Its paths are relative to `web/`; its tests run with `cd web/server && python3 -m unittest discover -s tests`. `make install` also installs `web/` under `$(pkgdatadir)/web`.
- **Building this branch** (autotools; upstream `master` requires Intel MKL, installed under `/opt/intel/oneapi`; the upstream CMake build is stale on `master` itself and is not kept working):

  ```bash
  ./regenerate_project.b            # after any change to configure.ac or */Makefile.am
  mkdir -p build/cpu && cd build/cpu
  export MKLROOT=/opt/intel/oneapi/mkl/latest
  ../../configure --enable-openmp --disable-FastFFT --disable-multiple-global-refinements CXX=g++ CC=gcc
  make -j12                          # ~15 min from clean; binaries in build/cpu/src/
  ```

  The binaries link MKL dynamically: `source /opt/intel/oneapi/setvars.sh` before running them — in the web server's shell too, or the controller and workers it launches fail to find `libmkl_intel_ilp64.so`. The web server the user runs for real is started from `web/server/` on port 8000 in their own terminal and runs whatever programs are on that process's `PATH` (`~/Apps/cisTEM/bin`, older builds) or named by a run profile's manager-command prefix; a fresh `make` here replaces none of those until installed or copied.
- **Git here.** This checkout is a **git worktree** of `~/Apps/cisTEM_git/cisTEM`, which stays on the user's own branch with uncommitted work — never run git commands in that directory. Commit here with `git -c user.name="Tim Grant" -c user.email=tgrant@morgridge.org commit ...` (the worktree has no identity of its own) and push with `git push origin cistem3` (SSH push URL, https fetch). `web/` was added with `git subtree add --prefix=web` from the standalone development repository `~/Downloads/cistem_web_app/cryoem-job-runner`, with its full history. Since 2026-09-15 **this branch is where the web interface is developed**; that repository is the secondary copy, updated when wanted with `git subtree push --prefix=web ~/Downloads/cistem_web_app/cryoem-job-runner master`. Keep commits that touch `web/` separate from commits that touch `src/`, so subtree splits stay clean.

## Build System

cisTEM uses GNU Autotools as the primary build system with Intel MKL for optimized FFT operations.

For detailed build instructions, see `scripts/CLAUDE.md`.

### Quick Start

```bash
# Initial setup
./regenerate_containers.sh
./regenerate_project.b

# Configure and build using VS Code
# Command Palette → Tasks: Run Task → BUILD cisTEM DEBUG

# Or manually:
mkdir -p build/debug && cd build/debug
../../configure --enable-debugmode
make -j16
```

## Architecture

### Core Components

- **src/core/** - Core libraries and data structures (see `src/core/CLAUDE.md`)
- **src/gui/** - removed on this branch; the web interface in **web/** replaces it (see `web/CLAUDE.md`)
- **src/programs/** - Command-line executables (see `src/programs/CLAUDE.md`)
- **scripts/** - Build and utility scripts (see `scripts/CLAUDE.md`)

### Key Dependencies

- **Intel MKL** - Primary FFT library for optimized performance
- **wxWidgets** - base, net and xml libraries only on this branch (strings, sockets, JSON); no GUI libraries
- **SQLite** - Database backend
- **CUDA** - GPU acceleration (optional)
- **Intel C++ Compiler (icc/icpc)** - Primary compiler for performance builds

## Testing

cisTEM has a multi-tiered testing approach:

```bash
# Unit tests - Test individual methods and functions
./unit_test_runner

# Console tests - Mid-complexity tests of single methods
./console_test

# Functional tests - Test complete workflows and image processing tasks
./samples_functional_testing
```

Refer to `.github/workflows/` for CI test configurations.

## Code Style and Standards

- **Formatting:** Project uses `.clang-format` in the root directory for consistent code formatting
- **Type Casting:** Always use modern C++ functional cast style (`int(variable)`, `long(variable)`, `float(variable)`) instead of C-style casts (`(int)variable`, `(long)variable`, `(float)variable`)
- **wxWidgets Printf Formatting:**
  - Always match format specifiers exactly to variable types (e.g., `%ld` for `long`, `%d` for `int`, `%f` for `float`) - mismatches cause segfaults in wxFormatConverterBase
  - Never use Unicode characters (Å, °, etc.) in format strings as they cause segmentation faults - use ASCII equivalents instead (A, deg, etc.)
- **Temporary Debugging Changes:** All temporary debugging code (debug prints, commented-out code, test modifications) must be marked with `// revert - <description of change and reason>` to ensure cleanup before commits. Search for "revert" to find all temporary changes.
- **Philosophy:** Incremental modernization - update and unify style as code is modified rather than wholesale changes
- **Legacy Compatibility:** Many legacy features exist; maintain compatibility while gradually improving
- **Preprocessor Defines:** All project-specific preprocessor defines should be prefixed with `cisTEM_` to avoid naming collisions (e.g., `cisTEM_ENABLE_FEATURE` not `ENABLE_FEATURE`)
- **Include Guards:** Use the full path from project root in uppercase with underscores for header file include guards (e.g., `_SRC_GUI_MYHEADER_H_` for `src/gui/MyHeader.h`, not `__MyHeader__`)
- **Temporary Files:** All temporary files (scripts, plans, documentation drafts) should be created in `.claude/cache/` directory. Create this directory if it doesn't exist. This keeps the project root clean and makes it easy to identify Claude-generated temporary content

## Commit Best Practices

- **Compilation Requirement:** Every commit must compile successfully without errors. This is essential for maintaining a clean git history that supports effective debugging with `git bisect`
- **Frequent Commits:** Commit work frequently, especially when completing discrete tasks or todo items. Small, focused commits are easier to review and debug
- **Clean Up Before Committing:** Remove all temporary debugging code marked with `// revert` comments before committing
- **Descriptive Messages:** Write clear, concise commit messages that explain what was changed and why
- **Test Before Commit:** Verify that changes work as expected before committing

## Pull Request Best Practices

**IMPORTANT: This repository has separate `origin` and `upstream` remotes. Pull requests must be created against `upstream`, not `origin`.**

- **PR Template:** All pull requests must follow the template at `.github/pull_request_template.md`
- **Interactive Drafting Process:** See `.github/workflows/CLAUDE.md` for detailed instructions on the interactive PR creation workflow
- **Target Repository:** PRs should target `upstream/master`, not `origin/master`
- **Pre-PR Checklist:**
  - Verify all commits compile
  - Run relevant tests (console tests, functional tests, manual testing)
  - Remove all `// revert` marked debugging code
  - Ensure PR description explains *why* changes were made, not just *what* changed


## Modern C++ Best Practices

### Container Usage

**Use STL containers for new code.** wxWidgets legacy containers (wxArray, wxList) exist only for compatibility.

| Use Case | Recommended | Avoid |
|----------|-------------|-------|
| Dynamic arrays | `std::vector<T>` | wxArray, wxVector |
| Lists | `std::list<T>`, `std::deque<T>` | wxList |
| String lists | `std::vector<wxString>` | wxArrayString |

### Memory Management

- **GUI objects:** Use raw pointers with parent-child ownership (see `src/gui/CLAUDE.md`)
- **Non-GUI objects:** Use smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- **Large arrays:** Use `new`/`delete` for explicit control

## IDE Configuration

The project is designed for development with Visual Studio Code using Docker containers:

- VS Code settings linked via `.vscode` symlink to `.vscode_shared/CistemDev`
- Container environment managed through `regenerate_containers.sh`
- Build tasks pre-configured for different compiler and configuration combinations
- When trying to show the user a diff for a file that might have moved, try something like "git difftool HEAD~2 -- src/core/socket_communicator.cpp src/core/socket_communication_utils/socket_communicator.cpp"