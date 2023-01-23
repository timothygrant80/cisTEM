add_executable(gui_test gui_test.cpp)

add_dependencies(gui_test cisTEM_core)
add_dependencies(gui_test cisTEM_gui_core)

target_link_libraries(gui_test    cisTEM_core 
                                cisTEM_gui_core)


install(TARGETS gui_test
RUNTIME DESTINATION bin)