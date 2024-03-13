add_executable(cisTEM_display cisTEM_display.cpp)

add_dependencies(cisTEM_display cisTEM_core)
add_dependencies(cisTEM_display cisTEM_gui_core)
add_dependencies(cisTEM_display cisTEM_gui)

target_link_libraries(cisTEM_display   cisTEM_core 
                                cisTEM_gui_core
                                cisTEM_gui
                                )


install(TARGETS cisTEM_display
RUNTIME DESTINATION bin)