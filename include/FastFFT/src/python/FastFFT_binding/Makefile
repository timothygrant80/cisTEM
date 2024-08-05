# TODO: nvc++ ?
NVCC=nvcc

# TODO: test with gcc and clang
NVCC_FLAGS=-ccbin=g++ -t 4 


NVCC_FLAGS+=--default-stream per-thread -m64 -O3 --use_fast_math --extra-device-vectorization -std=c++17 -Xptxas --warn-on-local-memory-usage,--warn-on-spills, --generate-line-info -Xcompiler=-fPIC,-shared,-O3,-Wall,-std=c++17
# Experimental flag to allow relaxed constexpr host call in device code
NVCC_FLAGS+=--expt-relaxed-constexpr
# For testing, particularly until the cufftdx::BlockDim() operator is enabled
NVCC_FLAGS+=-DCUFFTDX_DISABLE_RUNTIME_ASSERTS

# Gencode arguments, only supporting Volta or newer
# SMS ?= 70 75 80 86
# In initial dev, only compile for 70 or 86 depending on which workstation I'm on, b/c it is faster.
SMS ?= 70 86



ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))


# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif # GENCODE_FLAGS

# FIXME: this is set in my own bashrc. Until cufftdx comes out of early release, this is fine.
CUFFTDX_INCLUDE_DIR := $(FastFFT_cufftdx_dir)/include
$(info $$CUFFTDX_INCLUDE_DIR is [${CUFFTDX_INCLUDE_DIR}])

# TODO: check on cuda version (building with 11.3.1 r/n)
CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include


SRCS := FastFFT_python_binding.cu
# Needs to match the first arg to PYBIND11_MODULE(fastfft_test, m)
TARGETS := FastFFT.so
OBJS := FastFFT.o

# from $(python -m pybind11 --includes)
PYTHON_INCLUDES=-I/usr/include/python3.8 -I/groups/himesb/.myPyBind11/lib/python3.8/site-packages/pybind11/include

# $(TARGETS): $(SRCS)
# 	$(COMPILER) $(CXX_FLAGS) $(PYTHON_INCLUDES) $^ -o $@

# Track what the last build did so that the pre-commit hook can skip re-building if there is no debug included.

ifeq (${debug_kernel},) 
    NVCC_FLAGS+=-DFFT_DEBUG_STAGE=8 -DFFT_DEBUG_LEVEL=0
else
	NVCC_FLAGS+=-DHEAVYERRORCHECKING_FFT -DFFT_DEBUG_LEVEL=4 -DDEBUG_FFT_STAGE=8
endif

$(TARGETS): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(GENCODE_FLAGS) -o $@ $<  -lculibos -lcudart -lrt

$(OBJS): $(SRCS)
	$(NVCC) -dc $(NVCC_FLAGS) -I$(CUFFTDX_INCLUDE_DIR) $(GENCODE_FLAGS)  $(PYTHON_INCLUDES) -o $@ $<

clean:
	rm -f *.so *.o


