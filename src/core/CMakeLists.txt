add_library(cisTEM_gui_core STATIC gui_job_controller.cpp)

add_library(cisTEM_core STATIC 
	functions.cpp
	globals.cpp
	randomnumbergenerator.cpp
	userinput.cpp
	angles_and_shifts.cpp
	parameter_constraints.cpp
	assets.cpp
	asset_group.cpp
	curve.cpp
	image.cpp
	electron_dose.cpp
	matrix.cpp
	symmetry_matrix.cpp
	abstract_image_file.cpp
	mrc_header.cpp
	mrc_file.cpp
	tiff_file.cpp
	dm_file.cpp
	eer_file.cpp
	image_file.cpp
	sqlite/sqlite3.c
	database.cpp
	project.cpp
	reconstruct_3d.cpp
	reconstructed_volume.cpp
	resolution_statistics.cpp
	particle.cpp
	job_packager.cpp
	job_tracker.cpp
	run_profiles.cpp
	myapp.cpp
	empirical_distribution.cpp
	ctf.cpp
	numeric_text_file.cpp
	progressbar.cpp
	downhill_simplex.cpp
	brute_force_search.cpp
	va04.cpp
	conjugate_gradient.cpp
	euler_search.cpp
	frealign_parameter_file.cpp
	basic_star_file_reader.cpp
	refinement_package.cpp
	refinement.cpp
	classification.cpp
	particle_finder.cpp
	classification_selection.cpp
	rle3d.cpp
	angular_distribution_histogram.cpp
	local_resolution_estimator.cpp 
	cistem_parameters.cpp
	cistem_star_file_reader.cpp
	socket_communicator.cpp
	json/jsonval.cpp
	json/jsonreader.cpp
	json/jsonwriter.cpp
	stopwatch.cpp
	ccl3d.cpp
	template_matching.cpp
)

if (BUILD_EXPERIMENTAL_FEATURES)
	add_library(cisTEM_core_experimental STATIC
		water.cpp 
		pdb.cpp 
	)
endif(BUILD_EXPERIMENTAL_FEATURES)

target_link_libraries(cisTEM_core ${CMAKE_DL_LIBS})
target_link_libraries(cisTEM_core ${FFTW_LIBRARIES})
target_link_libraries(cisTEM_core ${wxWidgets_LIBRARIES})
target_link_libraries(cisTEM_core ${TIFF_LIBRARIES})

target_link_libraries(cisTEM_gui_core ${wxWidgets_LIBRARIES})

if (BUILD_EXPERIMENTAL_FEATURES) 
	add_dependencies(cisTEM_core cisTEM_core_experimental)
	target_link_libraries(cisTEM_core cisTEM_core_experimental)
endif(BUILD_EXPERIMENTAL_FEATURES)
