message("In the src/program/ctffind/ CMakeLists file")
add_executable(ctffind ctffind.cpp)
add_dependencies(ctffind cisTEM_core)

target_link_libraries(ctffind cisTEM_core)
target_link_libraries(ctffind fftw3f)
target_link_libraries(ctffind ${wxWidgets_LIBRARIES})
target_link_libraries(ctffind ${TIFF_LIBRARIES})