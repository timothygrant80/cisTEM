add_executable(refine3d refine3d.cpp)
add_dependencies(refine3d cisTEM_core)

target_link_libraries(refine3d cisTEM_core)
target_link_libraries(refine3d fftw3f)
target_link_libraries(refine3d ${wxWidgets_LIBRARIES})
target_link_libraries(refine3d ${TIFF_LIBRARIES})
