noinst_HEADERS = core/asset_group.h \
                 core/core_headers.h \
                 core/defines.h \
                 core/functions.h \
                 core/image.h \
                 core/mrc_file.h \
                 core/mrc_header.h \
                 gui/AlignMoviesPanel.h \
                 gui/ErrorDialog.h \
                 gui/MainFrame.h \
                 gui/mathplot.h \
                 gui/MyMovieAssetPanel.h \
                 gui/MovieImportDialog.h \
                 gui/ProjectX_gui.h \
                 gui/icons/action_icon.cpp \
                 gui/icons/assets_icon.cpp \
                 gui/icons/ctf_icon.cpp \
                 gui/icons/image_icon.cpp \
                 gui/icons/movie_align_icon.cpp \
                 gui/icons/movie_icon.cpp \
                 gui/icons/overview_icon.cpp 

noinst_LIBRARIES = libcore.a libguicore.a

bin_PROGRAMS = ProjectX unblur guix_job_control console_test


libcore_a_SOURCES =    core/functions.cpp \
                       core/assets.cpp \
                       core/asset_group.cpp \
                       core/image.cpp \
                       core/mrc_file.cpp \
                       core/mrc_header.cpp \
                       core/database.cpp \
                       core/project.cpp \
                       core/job_packager.cpp \
                       core/job_tracker.cpp \
                       core/run_profiles.cpp \
                       core/myapp.cpp

                       
libcore_a_CXXFLAGS = $(WX_CPPFLAGS_BASE)
libcore_a_CPPFLAGS = $(WX_CPPFLAGS_BASE)

libguicore_a_SOURCES =  core/gui_job_controller.cpp

libguicore_a_CXXFLAGS = $(WX_CPPFLAGS)
libguicore_a_CPPFLAGS = $(WX_CPPFLAGS)



ProjectX_SOURCES = programs/projectx/projectx.cpp gui/job_panel.cpp gui/ProjectX_gui.cpp gui/MainFrame.cpp gui/MyAssetParentPanel.cpp gui/MyMovieAssetPanel.cpp gui/MyImageAssetPanel.cpp gui/MovieImportDialog.cpp gui/MyImageImportDialog.cpp gui/ErrorDialog.cpp gui/AlignMoviesPanel.cpp gui/MyRunProfilesPanel.cpp gui/mathplot.cpp gui/MyAddRunCommandDialog.cpp gui/MyNewProjectWizard.cpp
ProjectX_CXXFLAGS = $(WX_CPPFLAGS)
ProjectX_CPPFLAGS = $(WX_CPPFLAGS)
ProjectX_LDFLAGS = $(WX_LIBS)
ProjectX_LDADD = libguicore.a libcore.a

guix_job_control_SOURCES = programs/guix_job_control/guix_job_control.cpp 
guix_job_control_CXXFLAGS = $(WX_CPPFLAGS_BASE)
guix_job_control_CPPFLAGS = $(WX_CPPFLAGS_BASE)
guix_job_control_LDFLAGS = $(WX_LIBS_BASE)
guix_job_control_LDADD = libcore.a

console_test_SOURCES = programs/console_test/console_test.cpp 
console_test_CXXFLAGS = $(WX_CPPFLAGS_BASE)
console_test_CPPFLAGS = $(WX_CPPFLAGS_BASE)
console_test_LDFLAGS = $(WX_LIBS_BASE)
console_test_LDADD = libcore.a


unblur_SOURCES = programs/unblur/unblur.cpp
unblur_CXXFLAGS = $(WX_CPPFLAGS_BASE)
unblur_CPPFLAGS = $(WX_CPPFLAGS_BASE)
unblur_LDFLAGS = $(WX_LIBS_BASE)
unblur_LDADD = libcore.a



