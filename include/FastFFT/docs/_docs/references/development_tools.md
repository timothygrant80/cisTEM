# Development Tools

Project based tools to help with debugging.

## Cuda error checking (kernels)

In src/FastFFT.cuh the HEAVYERRORCHECKING_FFT should be defined whenever defining new code or debugging. This enables the pre/postcheck macros which should be placed before and after any kernel invocations. As you can see in the code snippet below, when active, these macros force a stream synchronization after the kernel, which is asynchronous w.r.t the host. The precheck is to catch any errors that may have already happend, but were not caught by a missing postcheck, or came from an API call (next section.)

```{warning}
Without these guards, it can be very difficult to know where errors are actually coming from!
```

```c++
// When defined Turns on synchronization based checking for all FFT kernels as well as cudaErr macros
#define HEAVYERRORCHECKING_FFT

// Note we are using std::cerr b/c the wxWidgets apps running in cisTEM are capturing std::cout
// If I leave cudaErr blank when HEAVYERRORCHECKING_FFT is not defined, I get some reports/warnings about unused or unreferenced variables. I suspect the performance hit is very small so just leave this on.
// The real cost is in the synchronization of in pre/postcheck.
#define cudaErr(error) { auto status = static_cast<cudaError_t>(error); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << " :-> "; MyFFTPrintWithDetails("");} };

#ifndef HEAVYERRORCHECKING_FFT 
#define postcheck 
#define precheck 
#else
#define postcheck { cudaErr(cudaPeekAtLastError()); cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread); cudaErr(error) }
#define precheck { cudaErr(cudaGetLastError()) }
#endif

inline void checkCudaErr(cudaError_t err) 
{ 
  if (err != cudaSuccess) 
  { 
    std::cerr << cudaGetErrorString(err) << " :-> " << std::endl;
    MyFFTPrintWithDetails(" ");
  } 
};
```

## Cuda error checking (API)

calls to any cuda or cuda library API should be enclosed with cudaErr(), which will check for errors and print them to std::cerr.

## Simple printing macros

Old school, yes. Effective, also yes. Print out a message, what line and file the statement came from.

```c++
#define MyFFTPrintWithDetails(...) {std::cerr << __VA_ARGS__  << " From: " << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl;}

```

## Runtime asserts

As much as possible is statically checked. This is not always possible, so runtime asserts are used to catch bugs.

```c++
#define MyFFTRunTimeAssertTrue(cond, msg, ...) {if ((cond) != true) { std::cerr << msg   << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}
#define MyFFTRunTimeAssertFalse(cond, msg, ...) {if ((cond) == true) { std::cerr << msg  << std::endl << " Failed Assert at "  << __FILE__  << " " << __LINE__  << " " << __PRETTY_FUNCTION__ << std::endl; exit(-1);}}

```

## Partial transform checkpoints

Because the size and order of the data are changed inbetween different steps of a multi-dimensional trasnform, isolating bugs requires some method to do so. Right now, this is achieved by setting the debug_stage manually in build/Makefile.

```c++
# For testing/debugging it is convenient to execute and have print functions for partial transforms.
# These will go directly in the kernels and also in the helper Image.cuh definitions for PrintArray.
# The number refers to the number of 1d FFTs performed, 
# Fwd 0, 1, 2, 3( none, x, z, original y)
# 4 intermediate ops, like conj multiplication
# Inv 5, 6, 7 ( original y, z, x)
debug_stage=3
```

```{TODO} it would be nice if this were taken from an env variable
```

## Other printing based debug tools

FastFFT::PrintVector()

- print vectors of different type and number of elements
- can be used directly, however, it is generally used indirectly via either

FastFFT::PrintState()

```c++
  void PrintState()
  {
    std::cout << "================================================================" << std::endl; 
    std::cout << "Device Properties: " << std::endl;
    std::cout << "================================================================" << std::endl; 

    std::cout << "Device idx: " << device_properties.device_id << std::endl;
    std::cout << "max_shared_memory_per_block: " << device_properties.max_shared_memory_per_block << std::endl;
    std::cout << "max_shared_memory_per_SM: " << device_properties.max_shared_memory_per_SM << std::endl;
    std::cout << "max_registers_per_block: " << device_properties.max_registers_per_block << std::endl;
    std::cout << "max_persisting_L2_cache_size: " << device_properties.max_persisting_L2_cache_size << std::endl;
    std::cout << std::endl;

    std::cout << "State Variables:\n" << std::endl;
        // std::cerr << "is_in_memory_device_pointer " << is_in_memory_device_pointer << std::endl; // FIXME: switched to is_pointer_in_device_memory(d_ptr.buffer_1) defined in FastFFT.cuh
    std::cout << "is_in_buffer_memory " << is_in_buffer_memory << std::endl;
    std::cout << "is_fftw_padded_input " << is_fftw_padded_input << std::endl;
    std::cout << "is_fftw_padded_output " << is_fftw_padded_output << std::endl;
    std::cout << "is_real_valued_input " << IsAllowedRealType<InputType>  << std::endl;
    std::cout << "is_set_input_params " << is_set_input_params << std::endl;
    std::cout << "is_set_output_params " << is_set_output_params << std::endl;
    std::cout << "is_size_validated " << is_size_validated << std::endl;
    std::cout << std::endl;

    std::cout << "Size variables:\n" << std::endl;
    std::cout << "transform_size.N " << transform_size.N << std::endl;
    std::cout << "transform_size.L " << transform_size.L << std::endl;
    std::cout << "transform_size.P " << transform_size.P << std::endl;
    std::cout << "transform_size.Q " << transform_size.Q << std::endl;
    std::cout << "fwd_dims_in.x,y,z "; PrintVectorType(fwd_dims_in); std::cout << std::endl;
    std::cout << "fwd_dims_out.x,y,z " ; PrintVectorType(fwd_dims_out); std::cout<< std::endl;
    std::cout << "inv_dims_in.x,y,z " ; PrintVectorType(inv_dims_in); std::cout<< std::endl;
    std::cout << "inv_dims_out.x,y,z " ; PrintVectorType(inv_dims_out); std::cout<< std::endl;
    std::cout << std::endl;

    std::cout << "Misc:\n" << std::endl;
    std::cout << "compute_memory_wanted " << compute_memory_wanted << std::endl;
    std::cout << "memory size to copy " << memory_size_to_copy << std::endl;
    std::cout << "fwd_size_change_type " << SizeChangeName[fwd_size_change_type] << std::endl;
    std::cout << "inv_size_change_type " << SizeChangeName[inv_size_change_type] << std::endl;
    std::cout << "transform stage complete " << transform_stage_completed << std::endl;
    std::cout << "input_origin_type " << OriginTypeName[input_origin_type] << std::endl;
    std::cout << "output_origin_type " << OriginTypeName[output_origin_type] << std::endl;
    
  }; // PrintState()
  ```

  or

FastFFT::PrintLaunchParameters()

```c++
  void PrintLaunchParameters(LaunchParams LP)
  {
    std::cout << "Launch parameters: " << std::endl;
    std::cout << "  Threads per block: ";
    PrintVectorType(LP.threadsPerBlock);
    std::cout << "  Grid dimensions: ";
    PrintVectorType(LP.gridDims);
    std::cout << "  Q: " << LP.Q << std::endl;
    std::cout << "  Twiddle in: " << LP.twiddle_in << std::endl;
    std::cout << "  shared input: " << LP.mem_offsets.shared_input << std::endl;
    std::cout << "  shared output (memlimit in r2c): " << LP.mem_offsets.shared_output << std::endl;
    std::cout << "  physical_x_input: " << LP.mem_offsets.physical_x_input << std::endl;
    std::cout << "  physical_x_output: " << LP.mem_offsets.physical_x_output << std::endl;

  };
```
