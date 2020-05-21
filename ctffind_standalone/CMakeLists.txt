project(ctffind)
cmake_minimum_required(VERSION 2.8)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


#
# Setup findFFTW
#
configure_file(downloadFindFFTW.cmake.in findFFTW-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "CMake step for findFFTW failed: ${result}")
    else()
    message("CMake step for findFFTW completed (${result}).")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "Build step for findFFTW failed: ${result}")
endif()

set(findFFTW_DIR ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${findFFTW_DIR}")
	
#
# Let's look for single-precision FFTW
#
find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB)
include_directories(${FFTW_INCLUDE_DIRS})

#
# Let's look for wxWidgets (we can rely on cmake's built-in)
#
set(wxWidgets_USE_DEBUG false)	
find_package(wxWidgets REQUIRED COMPONENTS core base std net xml)
include(${wxWidgets_USE_FILE})
include_directories(${wxWidgets_INCLUDE_DIRS})

#
# Let's look for LibTIFF
#
find_package(TIFF REQUIRED)
include_directories(${TIFF_INCLUDE_DIRS})

add_subdirectory(src)
