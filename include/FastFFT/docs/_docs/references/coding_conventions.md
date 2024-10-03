# Coding Conventions

## Key structures

Defined in src/FastFFT.h

### Device pointers

- All transorms are out of place, such that twice the memory needed for a single transform is allocated. The second half is referred to as the "buffer."
- position space and momentum space is a bit of a misnomer here as a partial transform is no longer in position space, but not fully in momentum space.
- The templated parameters refer to the input data precision and the wanted compute precision.

```c++

  template<typename I, typename C>
  struct DevicePointers {
    // Use this to catch unsupported input/ compute types and throw exception.
    int* position_space = nullptr;
    int* position_space_buffer = nullptr;
    int* momentum_space = nullptr;
    int* momentum_space_buffer = nullptr;
    int* image_to_search = nullptr;
  };

  // Input real, compute single-precision
  template<>
  struct DevicePointers<float*, float*> {
    float*  position_space;
    float*  position_space_buffer;
    float2* momentum_space;
    float2* momentum_space_buffer;
    float2* image_to_search;

  };
```

### Device properties


```c++
  typedef
  struct __align__(32) _DeviceProps {
    int device_id;
    int device_arch;
    int max_shared_memory_per_block;
    int max_shared_memory_per_SM;
    int max_registers_per_block;
    int max_persisting_L2_cache_size;
  } DeviceProps;
```

### FFT Size

This refers to the 1d FFT size and is used in determining which templated kernels to use.

```c++
  typedef 
  struct __align__(8) _FFT_Size {
    // Following Sorensen & Burrus 1993 for clarity
    short N; // N : 1d FFT size
    short L; // L : number of non-zero output/input points 
    short P; // P >= L && N % P == 0 : The size of the sub-FFT used to compute the full transform. Currently also must be a power of 2.
    short Q; // Q = N/P : The number of sub-FFTs used to compute the full transform
} FFT_Size;
```

### Memory Offsets

These are passed to every kernel and the meanings are

```{TODO} Review these for consistency then write the description.
```

```c++
  typedef
	struct __align__(8) _Offsets{
    unsigned short shared_input;
    unsigned short shared_output;
    unsigned short physical_x_input;
    unsigned short physical_x_output;
  } Offsets;
```

### Launch parameters


```c++
  typedef 
  struct __align__(64) _LaunchParams{
    int Q;
    float twiddle_in;
    dim3 gridDims;
    dim3 threadsPerBlock;
    Offsets mem_offsets;
  } LaunchParams;
```

