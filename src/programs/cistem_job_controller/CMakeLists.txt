add_executable(cistem_job_controller cistem_job_controller.cpp)

add_dependencies(cistem_job_controller cisTEM_core)

target_link_libraries(cistem_job_controller    cisTEM_core)

install(TARGETS cistem_job_controller
        RUNTIME DESTINATION bin)
