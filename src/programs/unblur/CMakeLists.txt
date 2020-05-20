add_executable(unblur unblur.cpp)
add_dependencies(unblur cisTEM_core)

target_link_libraries(unblur cisTEM_core)
