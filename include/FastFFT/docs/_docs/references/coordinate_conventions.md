# Coordinate System Conventions

## Layout

### Summary

Similar to the Image and GpuImage class in *cis*TEM (link) the FourierTransformer class provides functionality to map multi-dimensional arrays onto linear memory. 
#### Logical dimesion 
- refer to the extents (number of elements) of the data, without respect to any padding.
- a consumer of the FourierTransformer class only ever worries about input and output logical dimensions.

#### Physical dimension
- refers to the extents (number of elements) of the data in memory.
  - for R2C, even out of place transforms, only the positive half of the physical x dimension is stored. 
  - while the constructor asks whether the input data are padded or packed, for the time being, only cuFFT/FFTW padding are handled.
- all references to transpose and permutation of axes in the kernel names refer to the physical coordinate system, which remains fixed through the lifetime of the object.

### Offsets

These are used for allocating dynamic shared memory and mapping from linear memory to higher dimensional constructs.

- shared_input is used when the input data are to be transformed multiple times with different twiddle factors, within a kernel. Generally, this is for zero padded ffts.
- shared_output is used to coalesce output that are calculated with strides, again general for zero padded ffts, where the stride will be Q = N/P (see definitions for [FFT size](fft-size-label))
- physical_x_input[output] refers to the number of elements along the x-axis in physical memory, i.e. the fast contiguous dimension. May include padding. 
```{note}
This often does not relate to the logical dimensions, e.g. in a 2D R2C kernel ending with "XY" the physical_x_input is the input logical x dimension + padding, while the physical_x_output will be the logical y input dimension due to the implicit transpose.
```
defintion:
```c++
typedef
	struct __align__(8) _Offsets {
    unsigned short shared_input;
    unsigned short shared_output;
    unsigned short physical_x_input;
    unsigned short physical_x_output;
} Offsets;
```

(fft-size-label)=
#### FFT size

Describes the one-dimensional size of the FFT, and relates the information containing vs zero valued sizes. Note that the non-zero values are currently only handled for consecutive indices, however, this is not a requirement in the transform decomposition, and for movie alignment in particular.

```{note}
Currently only power of 2 sizes are supported and N must be divisible by P and P == L. Checked at runtime.
```

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

### Indexing (mapping) operations

#### ```int GetSharedMemPaddedIndex(const int index)```

Returns the index in the shared memory buffer that corresponds to the given index in the input data.

#### ```unsigned int GetSharedMemPaddedIndex(const unsigned int index)```

Returns the index in the shared memory buffer that corresponds to the given index in the input data.

#### ```unsigned int Return1DFFTAddress(const unsigned int pixel_pitch)```

Returns the address in memory for each 1D FFT along the fast axis of a multi-dimensional array.

#### ```unsigned int Return1DFFTAddress_strided_Z(const unsigned int pixel_pitch)```

Returns the address in memory for each 1D FFT along the fast axis of a multi-dimensional array, where each block performs XZ_STRIDE ffts that are indexed into using threadIdx.z. These compose a 2D tile taken along the XZ plane which will in most cases be output to a tile in the ZX plane (transposed). The goal is for partial coalsecing on the write op.

#### ```unsigned int ReturnZplane(const unsigned int NX, const unsigned int NY)```

Returns the address in memory of a given plane in a 3D array, determined by blockIdx.z.

#### ```unsigned int Return1DFFTAddress_Z(const unsigned int NY)```
```{todo} 
unused, remove?
```

#### ```unsigned int Return1DFFTColumn_XYZ_transpose(const unsigned int NX)```
Returns a 1D address for any of the ffts making up a 2D tile in the physical XY plane. Depends on tIdx.z.Similar to ```Return1DFFTAddress_strided_Z``` but assuming the XZ axes are already transposed, i.e. the tile is in the transformed (ZY)' plane

#### ```unsigned int Return1DFFTAddress_XZ_transpose(const unsigned int X)```

Returns the address in the transposed output array, swapping XZ axes.

#### ```unsigned int Return1DFFTAddress_XZ_transpose_strided_Z(const unsigned int IDX)```

Returns the address in the transposed output array, swapping XZ axes. Depends on XZ_STRIDE and tIdx.z. Used in partial coalsecing of batched transforms.

#### ```unsigned int Return1DFFTAddress_YZ_transpose_strided_Z(const unsigned int IDX)```
Returns the output address in the physical XZ plane corresponding to the transformed (YZ)' plane. 

#### ```unsigned int Return1DFFTColumn_XZ_to_XY()```
Called by ```Return1DFFTAddress_YX()``` seems to not be used, FIXME.

#### ```unsigned int Return1DFFTAddress_YX()```
 seems to not be used, FIXME.
 



