
# FastFFT

---

(MS-label)=

## Abstract

The Fast Fourier transform (FFT) is one of the most widely used and heavily optimized digital signal processing algorithms. Many of the image processing algorithms used in cryo-Electron Microscopy rely on the FFT to accelerate convolution operations central to image alignment and also in reconstruction algorithms that operate most accurately in Fourier space. FFT libraries like FFTW and cuFFT provide routines for highly-optimized general purpose multi-dimensional FFTs; however, their generality comes at a cost of performance in several use-cases where only a subset of the input or output points are required. We demonstrate that algorithms based on the transform decomposition approach are well suited to the memory hierarchy of modern GPUs by implementing them in CUDA/C++ using the cufftdx header-only library. The results have practical implications, accelerating several key image processing algorithms by factors of 3-10x over those built using Nvidia’s general purpose FFT library cuFFT. These include movie-frame alignment, image resampling via Fourier cropping, 2d and 3d template matching, 3d reconstruction, and subtomogram averaging and alignment.

## Introduction

The Discrete Fourier Transform (DFT) and linear filtering, *e.g.* convolution, are among the most common operations in digital signal processing. It is therefore assumed that the reader has a basic familiarity with Fourier Analysis and it's applications in their respective fields; we will focus here on digital image processing for convenience. For a detailed introduction to the reader is referred to the free book by Smith {cite:p}`smith_mathematics_2008`. There are many ways to implement the computation of the DFT, the most simple of which can be understood as a matrix multiplication, which scales in computational complexity as O(n^2). In order improve the performance of the DFT, several algorithms have been devised, that collectively may be called the Fast Fourier Transform (FFT). All of these implement some level of recursion, as described below, which in turn reduces the computational complexity to O(nlog(n)). The FFT additionally improves performance by reducing the total memory footprint of the calculation, allowing for better utilization of high-performance, low capacity memory cache. In addition to increasing computational speed, this reduction in the total number of floating point operations also leads to reduced roundoff error, making the end result more accurate. The goal of many popular FFT library's is to provide optimal routines for FFT's of any size that are also adapted to the vagarys of the specific computer hardware they are being run on ⚠️. Most papers describing FFT algorithms focus on computational complexity from the perspective of the number of arithmetic operations  ⚠️, however, cache coherency and limited on chip memory ultimately reduce the efficiency of large FFTs (⚠️ TODO: Intro figure 1). For certain applications, particularly research that requires high performance computing resources, some generality may be exchanged for even more performance, resulting in substantial reduction of financial, and environmental, resources. At the extreme end of this tradeoff between generality and performance are hardware build specifically for a single algorithm, (Add something here about asics and fpgas and ANTON ⚠️) By incorporating prior information about specific image processing algorithms, we present optimized two and three dimensional FFT routines that blah blah blahh⚠️.

### Background

#### Examples from cryo-EM

⚠️ I think it makes sense to lead with the specific use cases we will be having to test and implement the desired improvements, from the perspective of cryo-EM. From there, providing a few examples of parallels, probably in CNNs etc and only then going into specifics about the implementation.

#### The discrete Fourier Transform

The Fourier Transform is a mathematical operation that maps an input function into a dual space; for example a function of time into a function of frequency. These dual spaces are sometimes referred to as position and momentum space, real and Fourier space, image space and k-space etc. Given that a "real-space" function can be complex valued, we will use the position and momentum space nomenclature to avoid ambiguity.

The discrete Fourier Transform (DFT) extends the operation of the Fourier Transform to a band-limited sequence of evenly spaced samples of a continuous function. In one dimension, it is defined for a sequence of N samples $x(n)$ as: 

% This produces a labelled eqution in jupyter book that will at least render the math in vscode preview, just without the label.
$$ X(k) = \sum_{n=0}^{N-1} x(n) \exp\left( \frac{-2\pi i k}{N}n \right) $$ (dft-1d-equation)

```{note}
*Throughout the text we use lower/upper case to refer to position/momemtum space variables.*
```


The DFT is fully seperable when calculated with respect to a Cartesian coordintate system. For example, for an M x N array, the DFT is defined as:

$$ X(k_m,k_n) = \sum_{m=0}^{M-1} \left[ \sum_{n=0}^{N-1} x(m,n) \exp\left( \frac{-2\pi i k_n}{N} n \right) \right] \exp\left( \frac{-2\pi i k_m}{M} m \right) $$ (dft-2d-equation)

From this equation, it should be clear in the most simple cases the 2D DFT can be calculated by first calculating the 1D FFT for each column and then each row, resulting in $ M \times N $ 1D DFTs. This seperability extends to higher dimensions, and is what permits us to exploit regions of the input that are known to be zero.

In addition to being seperable, the DFT has several other important properties:


```{TODO} list properties (And brief example with a few citations, preferably specific to cryo-EM where each is capitalized on.)
- linearity
- Parsevals
- Convolution theorem
- sinc interpolation
- Fourier Slice theorem
```
  
#### The Fast (Discrete) Fourier Transform

In looking at [⚠️ DFT equation above] it is clear that the DFT requires $ O(N^2) $
 complex exponential function evaluations, multiplications, and additions. The fast Fourier Transform (FFT) reduces the compuational complexity to $ O(Nlog_2{N}) $
 with the most efficient algorithm, the split-radix FFT requiring just $ 4Nlog_2{N} - 6N  $⚠️. The Cooley-Tukey algorithm {cite:p}`cooley_algorithm_1965` was published little more than a decade after the first digitial computers became available. As is often the case in science, their discovery was really a re-discovery; the divide and conquer approach that underpins the FFT was already known to Gauss as early as 1805, predating Fourier's work itself! {cite:p}`heideman_gauss_1985` 

% This won't display properly in vscode preview, it is an inset block quote with offset author attribution.
 ```{epigraph}

This story of the FFT can be used to give one incentive to investigate not
only new and novel approaches, but to occasionally look over old papers and see the variety of tricks and clever ideas which were used when computing was, by itself, a laborious chore which gave clever people great incentive to develop efficient methods. Perhaps among the ideas discarded before the days of electronic computers, we may find more seeds of new
algorithms.

-- James W. Cooley {cite}`cooley_re-discovery_1987`

```

This present work itself follows from this same spirit of re-discovery; presently with respect to ideas discarded before the days of efficient graphics processing units (GPUs), rather than electronic computers on a whole.

⚠️ Segue to include notes from FFTW - before last PP, something something FFTW is an example of dev since then - as those authors note, pruning something something, note on arithmetic vs caches (cite actual FFTW paper) something something.

#### Exploiting Zero Values (prior information)

The simplest approach to avoiding redundant calculations and memory transfers in calculating a multi-dimensional FFT can be realized if the algorithm is made aware of null rows or columns.

```{TODO} list properties (And brief example with a few citations, preferably specific to cryo-EM where each is capitalized on.)
- Concept, reduce ops, but especially i/o
- Mention pruning
- Introduce transform decomposition (Sorensen)
- Pictorial explanation for the major benefactors, also list estimate of ops. 
  - Movie alignment
  - 2D TM
  - 3D TM
  - Subtomogram averaging
```


## Theory

### The DFT and FFT


The FFT is useful in image analysis as it can help to isolate repetitive features of a given size. It is also useful in signal processing as it can be used to perform convolution operations. Multidimensional FFTs are fully seperable, such that in the simpliest case, FFTs in higher dimensions are composed of a sequence of 1D FFTs. For example, in two dimensions, the FFT of a 2D image is composed of the 1D FFTs of the rows and columns. A naive implementation is compuatationally ineffecient due to the strided memory access pattern⚠️. One solution is to transpose the data after the FFT and then transpose the result back ⚠️. This requires extra trips to and from main memory. Another solution is to decompose the 2D transform using vector-radix FFTs ⚠️.

#### Summary of GPU memory hierarchy as it pertains to this work

Working with a subset of input or output points for an FFT exposes several possibilites to exploit fine grained control of the memory hierarchy available on modern Nvidia GPUs. 


#### Graphic to illustrate how the implicit transpose and real-space shifting reduces memory access for forward padding

#### Maths for further reduced memory access and computation using transform decomposition on sparse inputs

#### Maths for partial outputs

#### 



## Results

### Basic performance

#### Comparing 2D FFTs using cufftdx w/o reordering

By design, the cufft library from Nvidia returns an FFT in the natural order [TODO check term] which requires two transpose operations, which in many cases seem to be optimized to reduce global memory access via vector-radix transforms. This means that the numbers we compare here are not quite apples to apples, as the result from FastFFT is transposed in memory in the Fourier domain.

##### Table 1: cufft/FastFFT runtime FFT -> iFFT pairs

| 2D square size | sm70 | sm86 (hack) |
|----|----|----|
| 64  |  1.23 | 0.93 |
| 128 | 1.36 | 1.00 |
| 256 | 1.22 | 1.08 |
| 512 | 0.92 | 0.89 |
| 1024| 0.69 | 0.67 |
| 2048| 0.95 | 0.96 |
| 4096| 1.10 | 1.12 |

🍍 The sm86 hack (v0.3.1) is going to be an underestimate.

🍍 None of the kernels are even remotely optimized at this point, they have only been assembled and tested to pass expected behavior for FFTs of constant functions, unit impulse functions, and basic convolution ops.

🍍 The relative decrease in performance on re-org seems to  be partially due to less optimization in the kernels at compile time (kernels themselves did not change) and given the trend with size, mainly due to overhead in launch? The question is whether the trade off in ease in reading the code is worth it.

:plus: Add in results for using the decomposition method for non-powers of two, more computation but fewer local memory accesses vs Bluesteins.

#### Comparing 2D FFT based convolution

##### Table 2: cuFFT/FastFFT runtime for zero padded convolution


| 2D input | 2x size |  4096 | <- sm70 sm86 -> |2x size |  4096 |
| --- | ---- | ---- | --- | ---- | ---- |
| 32 | 2.32 | 2.46 | | 2.19 | 2.64 |
| 64 | 2.54 | 2.59 | | 2.63 | 2.72 |
| 128 | 2.25 | 2.52 | | 2.34 | 2.73 |
| 256 | 1.57 | 2.45 |   | 1.62 | 2.70 |
| 512 | 1.33 | 2.46 | | 1.33 | 2.62 |
| 1024 | 1.85 | 2.25 | | 1.79 | 2.43 |
| 2048 | 1.95 | 1.95 | | 2.10| 2.10|

*with a generic functor*

| 2D input | 2x size |  4096 | <- sm70 sm86 -> |2x size |  4096 |
| --- | ---- | ---- | --- | ---- | ---- |
| 32 | 2.18 | 2.50 | | 2.64 | 2.63 |
| 64 | 2.60 | 2.61 | | 2.44 | 2.71 |
| 128 | 2.28 | 2.62 | | 2.30 | 2.72 |
| 256 | 1.59 | 2.60 |   | 1.63 | 2.71 |
| 512 | 1.34 | 2.48 | | 1.32 | 2.63 |
| 1024 | 1.90 | 2.30 | | 1.80 | 2.46 |
| 2048 | 1.95 | 1.95 | | 2.10| 2.10|

*with cufftdx-1.0.0 general release*

```{note}
The fft detail records are being forwarede to sm70 from sm86, whereas in my modification of vs 0.3.1 used in the numbers above, I forwarded sm86 to sm80.
```

| 2D input | 2x size |  4096 | <- sm70 sm86 -> |2x size |  4096 |
| --- | ---- | ---- | --- | ---- | ---- |
| 32 | 2.40 | 2.47 | | 2.16 | 2.57 |
| 64 | 2.60 | 2.56 | | 2.37 | 2.66 |
| 128 | 2.39 | 2.58 | | 2.39 | 2.67 |
| 256 | 1.93 | 2.55 |   | 1.87 | 2.64 |
| 512 | 1.48 | 2.46 | | 1.50 | 2.55 |
| 1024 | 1.96 | 2.28 | | 1.88 | 2.36 |
| 2048 | 1.95 | 1.95 | | 2.05| 2.05|

##### Table 3: cuFFT/FastFFT runtime (sm70) zero padded convolution of input size (top row), where only (bottom row) pixel sq. is needed for output

| | 32   |64   | 128 | 256 | 512 | 1024| 2048| 4096|
|---- | ----    | ---- |---- |---- |---- |---- |---- |---- |
| 16 |               1.8   | 1.75| 1.88| 1.75|1.36 | 0.93| 1.7 | 2.01   |
| 32 |                  | 1.56| 1.87| 1.72|1.37 | 0.92| 1.69 | 2.00   |
| 64 |                  | | 1.82| 1.69|1.34 | 0.91| 1.68 | 1.99   |
| 128|                  | | | 1.65|1.31 | 0.89| 1.64 | 1.95   |
| 256|                  | | | |1.24 | 0.84| 1.57 | 1.91   |
| 512|                  | | | | | 0.98| 1.48 | 1.84   |
| 1024|                 | | | | | | 1.32 | 1.69   |
| 2048|                 | | | | | |  | 1.48   |

##### Table 4: cuFFT/FastFFT runtime (sm86 hack) zero padded convolution of input size (top row), where only (bottom row) pixel sq. is needed for output

| | 32   |64   | 128 | 256 | 512 | 1024| 2048| 4096|
|---- | ----    | ---- |---- |---- |---- |---- |---- |---- |
| 16 |               1.49   | 1.58| 1.92| 1.46|1.34 | 1.02| 1.75 | 2.02   |
| 32 |                  | 1.50| 1.91| 1.45|1.33 | 1.02| 1.75 | 2.02   |
| 64 |                  | | 1.87| 1.42|1.31 | 1.01| 1.75 | 2.02   |
| 128|                  | | | 1.39|1.27 | 0.99| 1.71 | 2.00   |
| 256|                  | | | |1.21 | 0.93| 1.64 | 1.96   |
| 512|                  | | | | | 0.86| 1.53 | 1.89   |
| 1024|                 | | | | | | 1.35 | 1.74   |
| 2048|                 | | | | | |  | 1.50   |

🍍 None of the kernels are optimized at this point, they have only been assembled and tested to pass expected behavior for FFTs of constant functions, unit impulse functions, and basic convolution ops.

🍍 See note on previous table. The relative perf hit is not nearly as dramatic as in the previous table; however it is still about 10% which is a tough pill to swallow.

##### Table 3: cuFFT/FastFFT runtime for FFT + iFFT pairs


| 3D cubic size | Unbatched |  partial coalescing trick | <-sm70 sm86 (hack)-> | partial coalescing trick |
|----|----| ----- |----| ----- |
| 16 | 0.99 | 1.5 || 1.12|
| 32 | 0.93 | 1.0 || 1.08|
| 64  |  0.55 | 0.78 || 0.84|
| 128 | 0.71 |  0.96* ||0.93| 
| 256 | 0.47 | 0.99 || 1.03 |
| 512 | 0.23 | 0.97 || 1.07|

* This is with a partial coalesced stride of 8, while the others were best at 16. Along with Q I'll make both of these parameters compile time template parameters rather than fixed constants.

#### Table 4: Current 3D bottlenecks 512^3 (C - coalesced mem access)
| kernel | time (ms) | Load | Store 
|----|----|----|----|
| R2C_NONE_XZ | 3.27 | C | N|
| C2C_NONE_XYZ| 2.29 | C | N|
| C2C_fwd_NONE | 1.76 | C | C|
| C2C_inv_NONE_YZ | 2.27 | C | N|
|C2C_NONE_XYZ | 2.22 | C | N |
|C2R_NONE | 1.8 | C | C |

* XZ -> XZ axes transposed on write op
* XYZ -> All axes permuted on write op
* YZ -> YZ (physical XZ) axes transposed on write op

- Movie alignment expense (Pre/post process and alignment percentages.)

  :soon: 
  
#### Input movies are sparse (speed-up)

The largest speed up for a subset of input points is had when those non-zero points are contiguous, however, the reduced memory transfers even for non-contiguous values is still dope. The low-exposure imaging conditions used in HR-cryoEM generally results in ~10% non-zeros. ⚠️ Something about EER format.

#### Movies have a predictable shift range

The shifts between subsequent exposures in cryoEM movie frames is relatively small, on the order of tens of Angstroms (usually 5-20 pixels) in early frames, down to angstroms (0.5-2 pixels) in later frames. Using this prior knowledge, we can use only a limited subset of the output pixels, dramatically decreasing runtime in the translational search.

#### 2D template matching

Typical use case is to search an image of size ~ 4K x 4K over ~ 2 million cross correlations, with a template that is ~ 384-512 pixels square. Each of these searches requires padding the template to the size of the search image, taking the FFT, conj mul and iFFT. We currently optimize this by transforming the search image in single-precision, storing in FP16 to improve reads. Convert from FP16 and conj mul in single precision using cuFFT callbacks on the load, convert back to FP16 on the store with a store call back. Here we improve on this by ....

#### 3D template matching

Pad something like 128 --> 512^3 over X chunks. Far fewer orientations than 2DTM (number). 

#### Subtomogram averaging

While the total zero padding is less (1.5x to 2x) than 3D template matching, there are typically many more cross-correlations in a given project. Like movie alignment, subtomogram averaging also benefits in both directions, as the translational search allows for a strictly limited set of output points to find the shifts. These combined lead to X speedup.


## Discussion


## Conclusion

The FFT has many applications in signal and image processing, such that improving its efficiency for even a subset of problems can still have a broad impact. Here we demonstrate how fairly old algorithmic ideas, transform decomposition, can be mapped to new computational hardware to achieve substantial computational acceleration. The use cases presented in the results have typical run-times on the order of hours to days, such that even fractional speed ups, much less 2-5 fold acceleration are substantial. Gimme some oreos now.


## References

```{bibliography}
:style: unsrt
```

