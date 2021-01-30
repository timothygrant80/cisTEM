add_executable(convert_eer_to_mrc convert_eer_to_mrc.cpp)
add_dependencies(convert_eer_to_mrc cisTEM_core)

target_link_libraries(convert_eer_to_mrc cisTEM_core)

install(TARGETS convert_eer_to_mrc
        RUNTIME DESTINATION bin)
