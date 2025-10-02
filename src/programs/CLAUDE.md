# Command-Line Program Development Guidelines for cisTEM

This file provides guidance for developing and maintaining cisTEM's command-line programs.

## Program Architecture

Each cisTEM program is a self-contained executable that performs a specific image processing task. Programs are designed to be independent of the GUI and database, allowing them to run standalone or be called from the GUI.

### Standard Program Structure
```cpp
#include "../../core/core_headers.h"

class MyProgram : public MyApp {
public:
    bool DoCalculation();
    void DoInteractiveUserInput();

private:
    // Program-specific parameters
    float pixel_size;
    int box_size;
    wxString input_filename;
};

IMPLEMENT_APP(MyProgram)

bool MyProgram::DoCalculation() {
    // Main processing logic
    return true;
}

void MyProgram::DoInteractiveUserInput() {
    // Interactive parameter collection
}
```

## Parameter Handling

### UserInput Framework
Use the UserInput class for consistent parameter collection:
```cpp
UserInput my_input("ProgramName", version);

// Add parameter definitions
my_input.AddParameter("PARAMETER_NAME", "Input filename", "input.mrc", MRC_FILENAME);
my_input.AddParameter("BOX_SIZE", "Box size in pixels", "256");
my_input.AddParameter("PIXEL_SIZE", "Pixel size in Angstroms", "1.0");

// Check for command-line arguments
if (!my_input.CheckForHelp(argc, argv)) {
    // Collect parameters
    if (my_input.CheckForDoubleClick(argc, argv)) {
        // Interactive mode
        DoInteractiveUserInput();
    } else {
        // Command-line mode
        my_input.GetParameters(argc, argv);
    }
}
```

### Command-Line Argument Changes
When modifying command-line arguments, maintain backward compatibility:
```cpp
// Example: Replacing MAX_SEARCH_SIZE define with CLI argument
// OLD: #define MAX_SEARCH_SIZE 500
// NEW: Add as parameter with sensible default
my_input.AddParameter("MAX_SEARCH_SIZE", "Maximum search size", "500");
```

## Progress Reporting

### ProgressBar Usage
For long-running operations, provide progress feedback:
```cpp
ProgressBar my_progress_bar(number_of_steps);

for (int step = 0; step < number_of_steps; step++) {
    // Do work
    ProcessStep(step);

    // Update progress
    my_progress_bar.Update(step + 1);
}
```

### Console Output Guidelines
- Use `wxPrintf()` for normal output
- Use `SendInfo()` for important status messages
- Use `SendError()` for error conditions
- Avoid excessive output in loops

## File I/O Patterns

### Input File Validation
Always validate input files before processing:
```cpp
if (!DoesFileExist(input_filename)) {
    SendError(wxString::Format("Input file %s does not exist", input_filename));
    return false;
}

MRCFile input_file(input_filename.ToStdString(), false);
if (!input_file.is_valid) {
    SendError("Invalid MRC file");
    return false;
}
```

### Output File Handling
Check for existing files and handle appropriately:
```cpp
if (DoesFileExist(output_filename) && !overwrite) {
    SendError(wxString::Format("Output file %s already exists", output_filename));
    return false;
}
```

### Results Output
Programs output results directly to files, not databases:
```cpp
// Write results to MRC files
MRCFile output_file(output_filename.ToStdString(), true);
result_image.WriteSlices(&output_file, 1, result_image.logical_z_dimension);

// Write metadata to text files
NumericTextFile results_file(results_filename, OPEN_TO_WRITE);
results_file.WriteCommentLine("# Column 1: Image number");
results_file.WriteCommentLine("# Column 2: Defocus 1 (Angstroms)");
results_file.WriteLine(image_number, defocus1, defocus2);
```

## Common Program Types

### Image Processing Programs
Programs that process individual images or stacks:
- `ctffind` - CTF estimation
- `unblur` - Motion correction
- `resample` - Image resampling

### 3D Processing Programs
Programs that work with 3D volumes:
- `refine3d` - 3D refinement
- `reconstruct3d` - 3D reconstruction
- `project3d` - Generate 2D projections

### Utility Programs
Helper programs for specific tasks:
- `merge_star` - Merge STAR files
- `remove_duplicates` - Remove duplicate particles
- `apply_mask` - Apply masks to images

## Testing Programs

### Quick Test Pattern
For rapid development testing:
```cpp
// In programs/quick_test/quick_test.cpp
if (test_type == "my_new_test") {
    // Test your new functionality
    Image test_image;
    test_image.Allocate(256, 256, 1);

    // Run your algorithm
    MyNewAlgorithm(test_image);

    // Verify results
    wxPrintf("Test completed successfully\n");
}
```

## Performance Optimization

### OpenMP Parallelization
Use OpenMP for parallel processing:
```cpp
#pragma omp parallel for schedule(dynamic)
for (long particle = 0; particle < number_of_particles; particle++) {
    // Process each particle independently
    ProcessParticle(particle);
}
```

### Memory Management
Be mindful of memory usage with large datasets:
```cpp
// Process in chunks for large datasets
const int chunk_size = 1000;
for (int start = 0; start < total_images; start += chunk_size) {
    int end = std::min(start + chunk_size, total_images);
    ProcessImageChunk(start, end);
}
```

## Error Handling

### Graceful Failure
Programs should fail gracefully with informative messages:
```cpp
try {
    // Main processing
    if (!DoCalculation()) {
        SendError("Calculation failed");
        return false;
    }
} catch (std::exception& e) {
    SendError(wxString::Format("Fatal error: %s", e.what()));
    return false;
}
```

## Integration with GUI

### Socket Communication
When called from GUI, programs communicate via sockets:
```cpp
if (is_running_locally == false) {
    // Set up socket communication with GUI
    JobResult my_result;
    my_result.result_size = 1;
    my_result.result[0] = final_resolution;

    // Send result to GUI
    SendJobResult(&my_result);
}
```

### Independence Principle
**Important:** Programs must function without GUI or database:
- Accept all parameters via command line
- Read input from files, not database
- Write output to files, not database
- GUI reads program output files and updates database

This separation ensures programs can be:
- Run standalone for testing
- Called from scripts or pipelines
- Used with other workflow managers

## Best Practices Summary

1. **Maintain independence** from GUI and database
2. **Use consistent parameter naming** across related programs
3. **Provide meaningful default values** for all parameters
4. **Validate all inputs** before processing
5. **Report progress** for long-running operations
6. **Handle errors gracefully** with informative messages
7. **Write results atomically** to avoid partial outputs
8. **Document algorithm parameters** in help text
9. **Test with edge cases** (empty files, single particle, etc.)