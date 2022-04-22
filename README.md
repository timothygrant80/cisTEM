[![cisTEMx release](https://github.com/bHimes/cisTEMx/actions/workflows/release_build.yml/badge.svg)](https://github.com/bHimes/cisTEMx/actions/workflows/release_build.yml) [![cisTEMx debug](https://github.com/bHimes/cisTEMx/actions/workflows/debug_build.yml/badge.svg)](https://github.com/bHimes/cisTEMx/actions/workflows/debug_build.yml)

# *cis*TEM
[*cis*TEM](https://cistem.org) is user-friendly software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. It was originally developed by Tim Grant, Alexis Rohou and Nikolaus Grigorieff and comprises a number of tools to process image data including movies, micrographs and stacks of single-particle images, implementing a complete “pipeline” of processing steps to obtain high-resolution single-particle reconstructions. cisTEM is distributed under the [Janelia Research Campus Software License](http://license.janelia.org/license/) License and pre-compiled binaries can be downloaded from [cistem.org](https://cistem.org). For best performance, we recommend downloading and using the pre-compiled binaries, rather than compiling the source code. New users are encouraged to follow the [tutorial](https://cistem.org/documentation#tab-1-1), which provides a quick way to become familiar with the most important functions of cisTEM.

## Features
* Complete single-particle analysis pipeline, from raw movie frame alignemnt to final sharpened map ready for model building
* Single-window, easy-to-use, friendly, and cross-platform graphical user interface (GUI)
* Processing of tilted micrographs or movies
* Ab-inito 3D reconstruction (in the absence of any pre-existing map or model)
* Per-particle defocus refinement
* Beam tilt estimation and correction
* Ewald sphere correction
* Compatibility with other processing packages:
	* Import and export to/from RELION (including 3.1)
	* Import of WARP directories as cisTEM projects [command-line only]
	* Import and export to/from FREALIGN
* Ability to merge data with different pixel sizes, and/or from different microscopes
* Particle subtraction and symmetry expansion [command-line only]
* High-resolution template matching
* Dark & gain estimation and correction from dataset [command-line only]

## Installation

### Pre-compiled binaries
Pre-compiled binaries for Linux are available from [*cis*TEM.org](https://cisTEM.org).

### Building from source
We use GNU autotools to configure and build cisTEM. You should be able to do:
<pre>
git clone https://github.com/timothygrant80/cistem.git
cd cisTEM
mkdir build
cd build
../configure
make 
make install
</pre>

## Contributing to *cis*TEM
We welcome pull requests!

The recommended IDE for developing with cisTEM is Microsoft Visual Studio Code. Our main target OS is Linux - for convenience, the [cistem\_dev\_env Docker image](https://hub.docker.com/repository/docker/arohou/cistem_dev_env) can be used for cross-platform development. For detailed documentation on how to setup vscode with this Docker, please consult the [Docker hub page](https://hub.docker.com/repository/docker/arohou/cistem_dev_env).

## Getting help
Please address questions and requests to help to the [_cis_TEM forum](https://cistem.org/forum).

## Known issues
Please note that the repository version of cisTEM is likely unstable, and features are not guaranteed to work.  For this reason we recommend sticking to the released versions of cisTEM. 

For a list of currently known issues, please see the issue page on [Github](https://github.com/timothygrant80/cisTEM/issues/).

## License and distribution
Recent versions of cisTEM (including the 2.0+ releases) will soon be available under the [Janelia Research Campus Software License](http://license.janelia.org/license/).

cisTEM 1.0 is also distributed under [Janelia Research Campus Software License](http://license.janelia.org/license/).
