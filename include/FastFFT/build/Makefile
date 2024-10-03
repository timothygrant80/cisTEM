
NVCC=nvcc
# TODO: test with gcc and clang
NVCC_FLAGS=-ccbin=g++ -t 8 
# TODO: static may need to be added later or independently just for benchmarks, while libs to link through python are probably going to need to be dynamic.
NVCC_FLAGS+=--cudart=static
NVCC_FLAGS+=--default-stream per-thread -m64 -O3 --use_fast_math --extra-device-vectorization --extended-lambda --Wext-lambda-captures-this -std=c++17 -Xptxas --warn-on-local-memory-usage,--warn-on-spills, --generate-line-info -Xcompiler=-std=c++17
# Experimental flag to allow relaxed constexpr host call in device code
NVCC_FLAGS+=--expt-relaxed-constexpr
# For testing, particularly until the cufftdx::BlockDim() operator is enabled
NVCC_FLAGS+=-DCUFFTDX_DISABLE_RUNTIME_ASSERTS

# Don't explicitly instantiate the 3d versions unless requested
NVCC_FLAGS+=-DFastFFT_3d_instantiation

# Gencode arguments, only supporting Volta or newer
# SMS ?= 70 75 80 86
# In initial dev, only compile for 70 or 86 depending on which workstation I'm on, b/c it is faster.
SMS ?=  86

ifeq (${FastFFT_lto},1)
$(info "Building with LTO")
GENCODE_FLAGS := -gencode arch=compute_86,code=lto_86
else
$(info "Building without LTO")
GENCODE_FLAGS := -gencode arch=compute_86,code=sm_86
endif


# TODO:
# Lets first clean this up to have several targets, one for the library, one for the tests and move the helper Image object to the tests.
# Lets then get set lto as an option using an environment (or cli) variable. Print this out and alter the binary name.
# For now, just build with 86 for speed.

# FIXME: this is set in my own bashrc. Until cufftdx comes out of early release, this is fine.
CUFFTDX_INCLUDE_DIR := ../include
$(info $$CUFFTDX_INCLUDE_DIR is [${CUFFTDX_INCLUDE_DIR}])

# TODO: check on cuda version (building with 11.3.1 r/n)
CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include

# TODO: Changing the debug flags will not force re compilation
DEBUG_FLAGS :=  
# Debug level determines various asserts and print macros defined in FastFFT.cuh These should only be set when building tests and developing.
ifeq (${FFT_DEBUG_LEVEL},)
debug_level=0
$(info $$FFT_DEBUG_LEVEL is not defined, setting to 0)
else
debug_level=${FFT_DEBUG_LEVEL}
$(info $$debug_level is  [${FFT_DEBUG_LEVEL}])
endif
DEBUG_FLAGS+=-DFFT_DEBUG_LEVEL=${debug_level}

# For testing/debugging it is convenient to execute and have print functions for partial transforms.
# These will go directly in the kernels and also in the helper Image.cuh definitions for PrintArray.
# The number refers to the number of 1d FFTs performed, 
# Fwd 0, 1, 2, 3( none, x, z, original y)
# 4 intermediate ops, like conj multiplication
# Inv 5, 6, 7 ( original y, z, x)

ifeq (${FFT_DEBUG_STAGE},)
debug_stage=8
$(info $$FFT_DEBUG_STAGE is not defined, setting to 8)
else
debug_stage=${FFT_DEBUG_STAGE}
$(info $$debug_stage is  [${FFT_DEBUG_STAGE}])
endif
DEBUG_FLAGS+=-DFFT_DEBUG_STAGE=${debug_stage}

# Track what the last build did so that the pre-commit hook can skip re-building if there is no debug included.
remove_debug_file=0
ifeq (${debug_stage}, 8) 
	ifneq (${debug_level},0)
		remove_debug_file=1
	endif
else
	remove_debug_file=1
endif

ifeq (${FastFFT_sync_checking},1)
# If HEAVYERRORCHECKING_FFT is not already asked for, then add it anytime debug_stage < 8 (partial FFTs)
	DEBUG_FLAGS+=-DHEAVYERRORCHECKING_FFT
else
	ifeq (${debug_level}, 4)
		DEBUG_FLAGS+=-DHEAVYERRORCHECKING_FFT
	endif
endif



# External libraries used for testing
EXTERNAL_LIBS= -lfftw3f -lcufft_static -lculibos -lcudart_static -lrt

TEST_BUILD_DIR=tests
TEST_SRC_DIR=../src/tests
# Get all the test source files and remove cu extension
# TODO: selective targets
TEST_TARGETS=$(patsubst %.cu,$(TEST_BUILD_DIR)/%,$(notdir $(wildcard $(TEST_SRC_DIR)/*.cu)))
TEST_DEPS=$(wildcard $(TEST_SRC_DIR)/*.cuh)

$(info $$TEST_TARGETS is [${TEST_TARGETS}])

$(info )
all: $(TEST_TARGETS)
ifeq (${remove_debug_file}, 0)
	touch built_without_debug.txt
else
	rm -f built_without_debug.txt
endif

$(TEST_BUILD_DIR)/%: $(TEST_BUILD_DIR)/%.o lib/Image.o libFastFFT.a
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< lib/Image.o -L$(PWD) -lFastFFT $(EXTERNAL_LIBS)
# $(NVCC) $(NVCC_FLAGS) -dlto -arch=sm_86 -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) -o $@ $^ -lFastFFT $(EXTERNAL_LIBS)


$(TEST_BUILD_DIR)/%.o: $(TEST_SRC_DIR)/%.cu $(TEST_DEPS)
	@mkdir -p tests
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) -o $@ $<

# tests/test1: tests/linkedTest.o tests/test.o lib/Image.o libFastFFT.a
# 	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ -L$(PWD) -lFastFFT $(EXTERNAL_LIBS)
# # $(NVCC) $(NVCC_FLAGS) -dlto -arch=sm_86 -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) -o $@ $^ -lFastFFT $(EXTERNAL_LIBS)

# tests/linkedTest.o: tests/test.o lib/Image.o 
# 	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) --device-link -o $@ $^ 
# # $(NVCC) $(NVCC_FLAGS) -dlto -arch=sm_86 -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) --device-link -o $@ $^ 

# tests/test.o: ../src/tests/test.cu
# 	@mkdir -p tests
# 	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) -o $@ $<

lib/Image.o: ../src/fastfft/Image.cu ../src/fastfft/Image.cuh
	@mkdir -p lib
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) -o $@ $< 
	


# Build the library, using static linking for now to make sure we're set to go with cisTEM
libFastFFT.a: lib/FastFFT.o 
	$(NVCC) $(NVCC_FLAGS) --lib $(GENCODE_FLAGS) $(DEBUG_FLAGS) -o $@ $^ $(EXTERNAL_LIBS)

lib/FastFFT.o: ../src/fastfft/FastFFT.cu ../include/FastFFT.cuh
	@mkdir -p lib
	$(NVCC) -dc $(NVCC_FLAGS) $(GENCODE_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(DEBUG_FLAGS) -o $@ $<  
 


# remove all but the executables.
.PHONY: clean
clean:
	rm -f tests/*.o lib/*.o *.a *.o
	rm -f $(TEST_TARGETS)


# clean out everything.
.PHONY: clobber
clobber: clean
	rm -f libFastFFT.a


.DEFAULT_GOAL := all
