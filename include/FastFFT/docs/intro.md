# Welcome to your source for info about *Fast*FFT

*Fast*FFT (Fast Fast Fourier Transforms) _n_ - 


## Project Summary

hansen_dynamic-signaling-team_2010
#### Goals:

This project aims to accelerate a specific subset of Fast Fourier Transforms (FFTs) especially important image processing, my particular flavor of which is  high-resolution phase-contrast cryo-Electron Microscopy. The gist is to take advantage of the cufftdx library from Nvidia, currently in early access, as well as algorithmic ideas from *Sorensen et. al* {cite:p}`sorensen_efficient_1993`. I became aware of [VkFFT](https://github.com/DTolm/VkFFT) after I had finished my initial [proof of principle](https://github.com/bHimes/cisTEM_downstream_bah/blob/DFT/src/gpu/DFTbyDecomposition.cu) experiments with cufftdx in Oct 2020. It may be entirely possible to do the same things using that library, though I haven't had a chance to look through the source.

An additional experiment tied to this project, is the development of the [**scientific manuscript**](MS-label) "live" alongside the development of the code. While most of my projects are pretty safe from being scooped, this one is decidedly "scoopable" as it should have broad impact, and also shouldn't be to hard to implement once the info is presented.

#### Design:

The FourierTransformer class, in the FastFFT namespace may be used in your cpp/cuda project by including the *header.* The basic usage is as follows:

- The input/output data pointers, size and type are set. (analagous to the cufftXtMakePlanMany)
- Class methods are provided to handle memory allocation on the gpu as well as transfer to/from host/device.
- Simple methods for FwdFFT, InvFFT, CrossCorrelation etc. are public and call the correct combination of substranforms based on the data and padding/trimming wanted.


## Request new documentation or report bugs

```{margin} ***Formatting TIP***
Use the **write** and **preview** tabs to see how the markdown you are editing in your github issue will look after submitting.
```
TODO add the these templates in the FastFFT repo

<!-- TODO add the these templates in the FastFFT repo
If you do not find the info you need, please request it via [How Do I ... request](https://github.com/bHimes/cisTEM_docs/issues/new?assignees=&labels=documentation&template=how-do-i-do----.md&title=)
If you find a bug, like a broken link, please report it [using this form](https://github.com/bHimes/cisTEM_docs/issues/new?assignees=&labels=bug&template=bug_report.md&title=) -->








