message("In the src/program/projectx/ CMakeLists file")
add_executable(cisTEM projectx.cpp)

add_dependencies(cisTEM cisTEM_gui_core)
add_dependencies(cisTEM cisTEM_gui)

target_link_libraries(cisTEM cisTEM_core)
target_link_libraries(cisTEM cisTEM_gui_core)
target_link_libraries(cisTEM cisTEM_gui)
target_link_libraries(cisTEM fftw3f)
target_link_libraries(cisTEM ${wxWidgets_LIBRARIES})
target_link_libraries(cisTEM ${TIFF_LIBRARIES})