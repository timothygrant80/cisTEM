add_executable(reconstruct3d reconstruct3d.cpp)
add_dependencies(reconstruct3d cisTEM_core)

target_link_libraries(reconstruct3d cisTEM_core)
target_link_libraries(reconstruct3d fftw3f)
target_link_libraries(reconstruct3d ${wxWidgets_LIBRARIES})
target_link_libraries(reconstruct3d ${TIFF_LIBRARIES})
