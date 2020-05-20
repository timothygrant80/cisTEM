add_executable(unblur unblur.cpp)
add_dependencies(unblur cisTEM_core)

target_link_libraries(unblur cisTEM_core)
target_link_libraries(unblur fftw3f)
target_link_libraries(unblur ${wxWidgets_LIBRARIES})
target_link_libraries(unblur ${TIFF_LIBRARIES})
