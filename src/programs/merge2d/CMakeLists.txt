add_executable(merge2d merge2d.cpp)
add_dependencies(merge2d cisTEM_core)

target_link_libraries(merge2d cisTEM_core)

install(TARGETS merge2d
        RUNTIME DESTINATION bin)
