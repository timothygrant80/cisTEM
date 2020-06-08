add_executable(cisTEM projectx.cpp)

add_dependencies(cisTEM cisTEM_core)
add_dependencies(cisTEM cisTEM_gui_core)
add_dependencies(cisTEM cisTEM_gui)

target_link_libraries(cisTEM    cisTEM_core 
                                cisTEM_gui_core
                                cisTEM_gui
                                )


install(TARGETS cisTEM
RUNTIME DESTINATION bin)