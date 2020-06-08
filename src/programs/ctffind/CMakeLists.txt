add_executable(ctffind ctffind.cpp)
add_dependencies(ctffind cisTEM_core)

target_link_libraries(ctffind cisTEM_core)

install(TARGETS ctffind
        RUNTIME DESTINATION bin)
