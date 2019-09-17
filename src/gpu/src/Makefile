.RECIPEPREFIX=+++ 
CUDA_PATH ?= "/groups/grigorieff/home/himesb/thirdParty/cuda-10.1.2/cuda-toolkit"

HOST_COMPILER ?= icpc
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) 
NVCCFLAGS   := 
INCLUDES    :=
LDFLAGS     :=
DEFINES     :=

# simplePrintf.cu ContextManager.cu
SOURCES= Histogram.cu TemplateMatchingCore.cu DeviceManager.cu GpuImage.cu 
OBJECTS=$(SOURCES:.cu=.o)

$(info $$SOURCES is [${SOURCES}])
$(info $$OBJECTS is [${OBJECTS}])


# Forces stream synchronization after all kernels and asynchronous API calls (where wrapped) for proper debugging/ timing.
DEFINES += -DHEAVY_ERROR_CHECKING=true        


NVCCFLAGS += --default-stream per-thread -m64 
NVCCFLAGS += -Xptxas --warn-on-double-precision-use,--warn-on-local-memory-usage,--warn-on-spills --generate-line-info
NVCCFLAGS += -Xcompiler= -std=c++11 -Xlinker -lculibos
NVCCFLAGS += -lfftw3f,-lwx_gtk2u_richtext-3.0,-lwx_gtk2u_aui-3.0,-lwx_gtk2u_xrc-3.0,-lwx_gtk2u_html-3.0,-lwx_gtk2u_qa-3.0,lwx_gtk2u_adv-3.0,-lwx_gtk2u_core-3.0,-lwx_baseu_xml-3.0,-lwx_baseu_net-3.0,-lwx_baseu-3.0,-ltiff,-pthread,-fopenmp --gpu-architecture=compute_70 --gpu-code=sm_70,compute_70

INCLUDES += -I../include -I/groups/grigorieff/himesb/thirdParty/src/fftw-3.3.8/include -I/groups/grigorieff/home/himesb/thirdParty/wxWidgets/lib/wx/include/gtk2-unicode-3.0 -I/groups/grigorieff/home/himesb/thirdParty/wxWidgets/include/wx-3.0 -I/groups/grigorieff/home/himesb/thirdParty/cuda-10.0/samples/7_CUDALibraries/common/UtilNPP -I/groups/grigorieff/home/himesb/thirdParty/cuda-10.0/samples/common/inc

LDFLAGS += --cudart=static -lcufft_static -lcudart_static,-lcublas_static,-lcurand_static,-lnppc_static,-lnppial_static,-lnppist_static -lculibos -L${CUDA_BIN_PATH}/lib64 -L${CUDA_BIN_PATH}/nvvm/lib64 -L/groups/grigorieff/home/himesb/thirdParty/wxWidgets/lib -L/groups/grigorieff/himesb/thirdParty/src/fftw-3.3.8/lib

DEFINES += -D _FILE_OFFSET_BITS=64 -DWXUSINGDLL  -DGPU   -D__WXGTK__ -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_DLFCN_H=1 -Dwx_is_available=1 -DwxUSE_GUI=0

SMS ?= 70 75

all: $(OBJECTS) link.o ../lib/libcisTEMGPU.a

$(OBJECTS):$(SOURCES)
+++$(NVCC) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) --device-c $^

link.o : $(OBJECTS)
+++$(NVCC) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) --device-link  $^ --output-file $@ $(LDFLAGS)

../lib/libcisTEMGPU.a: link.o $(OBJECTS) 
+++$(NVCC) $(NVCCFLAGS) --lib --output-file $@ $^







 
