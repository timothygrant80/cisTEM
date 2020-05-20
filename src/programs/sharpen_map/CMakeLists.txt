add_executable(sharpen_map sharpen_map.cpp)
add_dependencies(sharpen_map cisTEM_core)

target_link_libraries(sharpen_map cisTEM_core)
target_link_libraries(sharpen_map fftw3f)
target_link_libraries(sharpen_map ${wxWidgets_LIBRARIES})
target_link_libraries(sharpen_map ${TIFF_LIBRARIES})
