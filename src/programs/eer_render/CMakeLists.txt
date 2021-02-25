add_executable(eer_render eer_render.cpp)
add_dependencies(eer_render cisTEM_core)

target_link_libraries(eer_render cisTEM_core)

install(TARGETS eer_rendere
        RUNTIME DESTINATION bin)
