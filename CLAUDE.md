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

cisTEM is a scientific computing application for cryo-electron microscopy (cryo-EM) image processing and 3D reconstruction. It's written primarily in C++ with CUDA GPU acceleration support and includes both command-line programs and a wxWidgets-based GUI.

## Build System

cisTEM uses GNU Autotools as the primary build system with Intel MKL for optimized FFT operations.

### Developer Build Process

For development builds, follow this sequence from the project root:

1. **Initial setup after clean install:**

   ```bash
   ./regenerate_containers.sh
   ./regenerate_project.b
   ```

2. **Configure and build using VS Code tasks:**
   - Use VS Code Command Palette → Tasks: Run Task
   - Default profiles:
     - `Configure cisTEM DEBUG build` (only needed if build system files changed: configure.ac, *.m4, Makefile.am)
     - `BUILD cisTEM DEBUG`
   - **Note:** If you modify configure.ac, any .m4 files, or Makefile.am, run `./regenerate_project.b`, then configure, then build

3. **Manual build process:**

   ```bash
   # Example debug build with Intel compiler and GPU support
   mkdir -p build/intel-gpu-debug-static
   cd build/intel-gpu-debug-static
   CC=icc CXX=icpc ../../configure --enable-debugmode --enable-gpu-debug \
     --with-wx-config=/opt/WX/icc-static/bin/wx-config \
     --enable-staticmode --with-cuda=/usr/local/cuda \
     --enable-experimental --enable-openmp
   make -j8
   ```

### CMake (Alternative)

```bash
mkdir build && cd build
cmake -DBUILD_STATIC_BINARIES=ON -DBUILD_EXPERIMENTAL_FEATURES=OFF ..
make -j$(nproc)
```

Build options for CMake:

- `BUILD_STATIC_BINARIES=ON/OFF` - Static vs dynamic linking
- `BUILD_EXPERIMENTAL_FEATURES=ON/OFF` - Include experimental code
- `BUILD_OpenMP=ON/OFF` - Enable OpenMP multithreading

### Docker Development Environment
The project uses a Docker container for cross-platform development. Container definitions are in `scripts/containers/` with base and top layer architecture.

## Architecture

### Core Components

- **src/core/** - Core libraries and data structures
  - Image processing classes (`image.h`, `mrc_file.h`)
  - Mathematical utilities (`matrix.h`, `functions.h`)
  - Database interface (SQLite integration)
  - GPU acceleration headers and CUDA code

- **src/gui/** - wxWidgets-based graphical interface
  - Main application framework
  - Panel components for different workflows
  - Icon resources and UI elements

- **src/programs/** - Command-line executables
  - Individual processing programs (ctffind, unblur, refine3d, etc.)
  - Each program is self-contained with its own main()

### Key Dependencies

- **Intel MKL** - Primary FFT library for optimized performance
- **FFTW** - Alternative FFT library (maintained for portability but not officially supported due to restrictive licensing)
- **wxWidgets** - GUI framework (typically 3.0.5 stable)
- **LibTIFF** - TIFF image file support
- **SQLite** - Database backend
- **CUDA** - GPU acceleration (optional)
- **Intel C++ Compiler (icc/icpc)** - Primary compiler for performance builds

## Development Commands

### Testing

cisTEM has a multi-tiered testing approach:

```bash
# Unit tests - Test individual methods and functions
./unit_test_runner

# Console tests - Mid-complexity tests of single methods
./console_test

# Functional tests - Test complete workflows and image processing tasks
./samples_functional_testing

# Quick test executable
./quick_test
```

**Testing hierarchy:**

- `unit_test_runner` - Basic unit tests for core functionality
- `console_test` - Intermediate complexity, testing individual methods with embedded test data
- `samples_functional_testing` - Full workflow tests simulating real image processing tasks

Refer to `.github/workflows/` for CI test configurations and current testing priorities.

### GPU Development

The project includes CUDA code for GPU acceleration. GPU-related files are primarily in:

- Core extensions for GPU operations
- Specialized GPU kernels for image processing
- CUDA FFT implementations

### Code Structure Notes

- Most core functionality is in header-only or heavily templated C++ code
- Image processing uses custom Image class with MRC file format support
- Database schema is defined for project management
- Extensive use of wxWidgets for cross-platform GUI components
- Legacy features mean style isn't fully coherent, but the project aims to unify as code is modified

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

## wxWidgets Best Practices for cisTEM

### Memory Management
- **Widget Ownership:** Create widgets with clear parent-child relationships. Parent widgets automatically delete their children.
- **Avoid Complex Member Widgets:** Don't use `std::unique_ptr` or member variables for widgets that may outlive workflow switches. Instead, create them locally in dialogs.
- **Dialog-Scoped Resources:** For temporary UI elements (like queue managers), create them as children of dialogs rather than panel members.

Example:
```cpp
// GOOD: Dialog owns the widget
wxDialog* dialog = new wxDialog(parent, ...);
MyWidget* widget = new MyWidget(dialog);  // Dialog will delete it

// AVOID: Complex lifecycle management
class Panel {
    std::unique_ptr<MyWidget> persistent_widget;  // Risky during workflow switches
};
```

### Database Access Patterns
- **Defer Database Operations:** Never access the database in constructors, especially during workflow switching when `main_frame` might be invalid.
- **Use Lazy Loading:** Implement a flag-based approach for database operations.

Example:
```cpp
// GOOD: Lazy loading pattern
class MyWidget {
    bool needs_database_load = true;

    void OnFirstUse() {
        if (needs_database_load && main_frame && main_frame->current_project.is_open) {
            LoadFromDatabase();
            needs_database_load = false;
        }
    }
};
```

### Workflow Switching Robustness
- **Design for Destruction:** Panels are destroyed and recreated during workflow switches. Don't assume persistence.
- **State in Database:** Keep complex state in the database rather than in memory.
- **Avoid Destructor Logic:** Don't put complex logic in destructors; wxWidgets handles most cleanup automatically.

### Build System Tips
- **Parallel Builds:** Use `make -j16` (or available thread count) for faster compilation
- **Force Rebuilds:** Delete dependency files to force rebuild: `rm gui/.deps/cisTEM-TargetFile.*`
- **VS Code Quirks:** Git diffs may need manual refresh in VS Code after file changes

## Environment Variables

- `WX_CONFIG` - Path to wx-config for specifying wxWidgets installation
- CUDA environment variables for GPU builds
- Various build flags configured in `.vscode/tasks.json`

## IDE Configuration

The project is designed for development with Visual Studio Code using Docker containers:

- VS Code settings linked via `.vscode` symlink to `.vscode_shared/CistemDev`
- Container environment managed through `regenerate_containers.sh`
- Build tasks pre-configured for different compiler and configuration combinations

## Template Matching Queue Development Patterns

### Static Members for Cross-Dialog Persistence
When implementing features that need to persist across dialog instances (like queues), use static members rather than complex lifecycle management:
```cpp
// In header
static std::deque<QueueItem> execution_queue;
static long currently_running_id;

// In cpp file - define static members
std::deque<QueueItem> QueueManager::execution_queue;
long QueueManager::currently_running_id = -1;
```

### Job Completion Tracking Pattern
For async job tracking in panels, store the job ID when starting and check it in completion callbacks:
```cpp
// In job panel header
long running_queue_job_id;  // -1 if not from queue

// In job start
running_queue_job_id = job.template_match_id;

// In ProcessAllJobsFinished or similar
if (running_queue_job_id > 0) {
    UpdateQueueStatus(running_queue_job_id, "complete");
    running_queue_job_id = -1;
}
```

This pattern allows proper status updates without tight coupling between components.

## wxWidgets Modern C++ Best Practices

### Container Selection

**Use STL containers for new code.** wxWidgets legacy containers (wxArray, wxList, wxVector) exist only for compatibility and should not be used in new development.

| Use Case | Recommended | Avoid |
|----------|-------------|-------|
| Dynamic arrays | `std::vector<T>` | wxArray, wxVector |
| Lists | `std::list<T>`, `std::deque<T>` | wxList |
| String lists | `std::vector<wxString>` | wxArrayString |
| Maps | `std::map`, `std::unordered_map` | wxHashMap |

**Rationale:** STL containers are more efficient, safer with run-time checks, and integrate better with modern C++ features.

### Memory Management for wxWidgets Objects

#### GUI Objects (wxWindow-derived)

**Use raw pointers with parent-child ownership model:**

```cpp
// GOOD: Parent manages child lifetime
wxDialog* dialog = new wxDialog(parent, ...);
wxButton* button = new wxButton(dialog, ...);  // Dialog will delete button

// AVOID: Smart pointers with GUI objects
std::unique_ptr<wxDialog> dialog;  // Risk of double-deletion
```

**Key Rules:**

- Always specify a parent for wxWindow-derived objects
- Parents automatically delete their children
- Use `Destroy()` method, not `delete` for top-level windows
- Never use smart pointers with wxWindow objects (causes double-deletion crashes)

#### Non-GUI Objects

**Use smart pointers freely:**

```cpp
// GOOD: Smart pointers for data structures
std::unique_ptr<DataProcessor> processor = std::make_unique<DataProcessor>();
std::shared_ptr<ImageData> shared_data = std::make_shared<ImageData>();
```

### Static Members for Persistent State

For features that need to persist across dialog instances:

```cpp
// In header
class QueueManager {
    static std::deque<QueueItem> execution_queue;  // Survives dialog recreation
};

// In cpp file
std::deque<QueueItem> QueueManager::execution_queue;  // Define static member
```

### Build Configuration

- Enable STL support: `--enable-std_containers` or `wxUSE_STD_CONTAINERS=1`
- Modern wxWidgets (3.3+) implements legacy containers using STL internally
- C++11 is minimum requirement, C++14/17 features supported

### Best Practices Summary

- **Data structures:** Use STL containers, not wx legacy containers
- **GUI objects:** Raw pointers with parent-child model
- **Non-GUI objects:** Smart pointers recommended
- **Persistence:** Static members for cross-dialog state
- **Memory safety:** Let wxWidgets handle GUI lifecycle, use RAII for everything else
