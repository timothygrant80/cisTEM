# Core Library Development Guidelines for cisTEM

This file provides guidance for working with cisTEM's core computational libraries and data structures.

## Architecture Overview

The core library provides fundamental image processing, mathematical operations, and data management functionality used throughout cisTEM.

### Key Components
- **Image Processing:** `image.h`, `mrc_file.h`, `tiff_file.h`
- **Mathematical Operations:** `matrix.h`, `functions.h`, `numerical_recipes.h`
- **Database Interface:** `database.h`, `project.h`
- **GPU Acceleration:** GPU-specific headers and CUDA implementations
- **FFT Operations:** MKL and FFTW wrappers

## Image Class Best Practices

### Memory Management
The Image class manages large memory blocks:
```cpp
Image my_image;
my_image.Allocate(x_size, y_size, z_size);
// Methods internally use debug assertions to ensure allocation
```

### MRC File Format
cisTEM primarily uses MRC (Medical Research Council) format for electron microscopy data:
```cpp
// Reading MRC files
MRCFile input_file(filename, false);  // false = read mode
Image my_image;
my_image.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices());

// Writing MRC files
MRCFile output_file(filename, true);  // true = write mode
my_image.WriteSlices(&output_file, 1, my_image.logical_z_dimension);
```

## Mathematical Operations

### Coordinate Systems
cisTEM uses Fourier space conventions common in cryo-EM:
- Real space: Origin at corner (0,0,0)
- Fourier space: DC component at (0,0,0) after FFT
- Physical coordinates: Often centered with respect to box center

### FFT Library Usage
**Intel MKL is the primary FFT library:**
```cpp
// Forward FFT
my_image.ForwardFFT();  // Converts real to complex

// Inverse FFT
my_image.BackwardFFT(); // Converts complex to real
```

## Database Operations

### Thread Safety
Database operations are NOT thread-safe. Use appropriate locking:
```cpp
// Use database mutex for multi-threaded access
std::lock_guard<std::mutex> lock(database_mutex);
database.ExecuteSQL(query);
```

### Transaction Management
Use transactions for multiple related operations:
```cpp
database.Begin();
try {
    // Multiple database operations
    database.ExecuteSQL(query1);
    database.ExecuteSQL(query2);
    database.Commit();
} catch (...) {
    database.Rollback();
    throw;
}
```

## GPU Development Patterns

### CUDA Integration
GPU code follows specific patterns for memory management:
```cpp
#ifdef ENABLEGPU
    if (use_gpu) {
        // GPU-specific implementation
        GpuImage gpu_image;
        gpu_image.CopyFrom(cpu_image);
        gpu_image.ForwardFFT();
    } else {
        // CPU fallback
        cpu_image.ForwardFFT();
    }
#else
    // CPU-only build
    cpu_image.ForwardFFT();
#endif
```

## Performance Considerations

### OpenMP Usage
Many core operations are parallelized with OpenMP:
```cpp
#pragma omp parallel for
for (long pixel = 0; pixel < number_of_pixels; pixel++) {
    // Parallel processing
    // Avoid race conditions on shared data
}
```

## Testing Patterns

### Unit Testing
Core functionality should have comprehensive unit tests:
```cpp
// In unit_test_programs/
TEST_CASE("Image::ForwardFFT") {
    Image test_image;
    test_image.Allocate(64, 64, 1);

    // Set up test data
    test_image.SetToConstant(1.0f);

    // Test operation
    test_image.ForwardFFT();

    // Verify results
    REQUIRE(test_image.is_in_real_space == false);
    REQUIRE(abs(test_image.complex_values[0]) > 0);
}
```

### Console Testing
For more complex scenarios, use console_test:
```cpp
// Test individual methods with embedded test data
if (test_number == IMAGE_FFT_TEST) {
    Image test_image;
    // Complex test scenario
    RunFFTTest(test_image);
}
```

## Common Core Files

### Essential Headers
- `src/core/core_headers.h` - Includes all core functionality
- `src/core/assets.h` - Asset management classes
- `src/core/image.h` - Primary image processing class
- `src/core/electron_dose.h` - Dose weighting calculations
- `src/core/ctf.h` - Contrast transfer function

### Utility Classes
- `src/core/progressbar.h` - Console progress reporting
- `src/core/randomnumbergenerator.h` - Random number generation
- `src/core/curve.h` - 1D curve fitting and interpolation
- `src/core/angles_and_shifts.h` - Euler angle conversions

## Error Handling

### Assertions vs Exceptions
- Use `MyDebugAssertTrue()` and `MyDebugAssertFalse()` for development-time checks
- Use exceptions for runtime errors that can be recovered
- Never suppress assertions or errors to hide problems

**Note:** Current assertion implementation uses macros. Print formatting may still incur a cost even in release builds. This should be fixed with templated versions in the future.

```cpp
// Development assertions
MyDebugAssertTrue(image.is_in_memory, "Image must be allocated");
MyDebugAssertFalse(error_condition, "Error condition should not occur");

// Runtime error handling
if (!file.OpenFile(filename, false)) {
    throw std::runtime_error("Cannot open file: " + filename);
}
```

## Code Style Guidelines

### Using Declarations and Type Aliases

**`using` declarations should be scoped as narrowly as possible:**

```cpp
// ❌ AVOID: Global scope using declarations
using MyType = cistem::fundamental_type::Enum;

void MyFunction() {
    MyType value = MyType::integer_t;  // Pollutes global namespace
}

// ✅ GOOD: Function-scoped using declarations
void MyFunction() {
    using MyType = cistem::fundamental_type::Enum;
    MyType value = MyType::integer_t;  // Scoped to function
}

// ✅ ACCEPTABLE: Class-scoped (only if used extensively throughout class)
class MyClass {
    using MyType = cistem::fundamental_type::Enum;

    void Method1() {
        MyType value = MyType::integer_t;
    }

    void Method2() {
        MyType value = MyType::float_t;
    }
};

// ✅ BEST: Use full type when only used a few times
void MyFunction() {
    cistem::fundamental_type::Enum value = cistem::fundamental_type::integer_t;
}
```

**Rationale:**
- Global `using` declarations pollute the namespace for all files that include the header
- Function-scoped declarations keep type aliases local and clear
- Class-scoped declarations are acceptable when a type is used extensively throughout a class
- Full type names are preferred when brevity doesn't significantly improve readability

**Static Assertions for Type Safety:**
When using function-scoped `using` declarations for type aliases, add static assertions to verify critical type properties:
```cpp
// ✅ BEST: Function-scoped using with compile-time safety check
bool JobPackage::SendJobPackage(TcpSocket* socket) {
    using c_ft = cistem::fundamental_type::Enum;
    static_assert(sizeof(c_ft) == sizeof(uint8_t),
                  "fundamental_type::Enum must match uint8_t size for safe casting in wire protocol");

    // Now safe to use c_ft throughout function
    c_ft type_descriptor = c_ft::integer_t;
    // ...
}
```

This pattern combines readability (short alias) with safety (compile-time verification), ensuring type assumptions don't break during refactoring.

**Legacy Code:**
Most cisTEM code has been updated to use properly-scoped `using` declarations. If you encounter global `using` declarations in older files, refactor them to function or class scope when modernizing those files.

## Best Practices Summary

1. **Use debug assertions** to verify preconditions in methods
2. **Use appropriate FFT normalization** for your algorithm
3. **Lock database access** in multi-threaded contexts
4. **Profile performance-critical code** with Intel VTune
5. **Write comprehensive tests** for new core functionality
6. **Document mathematical algorithms** with references to papers
7. **Consider GPU acceleration** for computationally intensive operations
8. **Maintain backward compatibility** with existing file formats
9. **Scope `using` declarations narrowly** - function > class > never global