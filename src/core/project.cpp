#include "core_headers.h"

Project::Project( ) {

    is_open         = false;
    total_cpu_hours = 0;
    total_jobs_run  = 0;

    project_name      = "";
    project_directory = "";
}

Project::~Project( ) {
}

bool Project::CreateNewProject(wxFileName wanted_database_file, wxString wanted_project_directory, wxString wanted_project_name) {
    int      return_code;
    wxString directory_string;
    bool     success;

    // is project already open?

    if ( is_open == true ) {
        MyPrintWithDetails("Attempting to create a new project, but there is already an open project");
        return false;
    }

    if ( wanted_project_name.IsEmpty( ) == true ) {
        MyDebugPrintWithDetails("Attempting to create a new project, but the project name is blank");
        return false;
    }

    if ( wanted_project_directory.IsEmpty( ) == true ) {
        MyDebugPrintWithDetails("Attempting to create a new project, but the project dir is blank");
        return false;
    }

    success = database.CreateNewDatabase(wanted_database_file);
    CheckSuccess(success);
    success = database.CreateAllTables( );
    CheckSuccess(success);

    project_name      = wanted_project_name;
    project_directory = wanted_project_directory;

    // create sub folders..

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets";
    wxFileName::Mkdir(directory_string);

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Movies";
    movie_asset_directory = directory_string;
    wxFileName::Mkdir(movie_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Images";
    image_asset_directory = directory_string;
    wxFileName::Mkdir(image_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Volumes";
    volume_asset_directory = directory_string;
    wxFileName::Mkdir(volume_asset_directory.GetFullPath( ));

#ifdef EXPERIMENTAL
    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/TemplateMatching";
    template_matching_asset_directory = directory_string;
    if ( wxDir::Exists(template_matching_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(template_matching_asset_directory.GetFullPath( ));
#endif

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/PhaseDifferenceImages";
    phase_difference_asset_directory = directory_string;
    wxFileName::Mkdir(phase_difference_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/CTF";
    ctf_asset_directory = directory_string;
    wxFileName::Mkdir(ctf_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/ParticlePosition";
    particle_position_asset_directory = directory_string;
    wxFileName::Mkdir(particle_position_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/ParticleStacks";
    particle_stack_directory = directory_string;
    wxFileName::Mkdir(particle_stack_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/ClassAverages";
    class_average_directory = directory_string;
    wxFileName::Mkdir(class_average_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Parameters";
    parameter_file_directory = directory_string;
    wxFileName::Mkdir(parameter_file_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Scratch";
    scratch_directory = directory_string;
    wxFileName::Mkdir(scratch_directory.GetFullPath( ));

    // sub directories

    directory_string = image_asset_directory.GetFullPath( );
    directory_string += "/Spectra";
    wxFileName::Mkdir(directory_string);

    directory_string = image_asset_directory.GetFullPath( );
    directory_string += "/Scaled";
    wxFileName::Mkdir(directory_string);

    directory_string = volume_asset_directory.GetFullPath( );
    directory_string += "/OrthViews";
    wxFileName::Mkdir(directory_string);

    total_cpu_hours = 0;
    total_jobs_run  = 0;

    // set master settings..

    if ( database.InsertOrReplace("MASTER_SETTINGS", "ittirit", "NUMBER", "PROJECT_DIRECTORY", "PROJECT_NAME", "CURRENT_VERSION", "TOTAL_CPU_HOURS", "TOTAL_JOBS_RUN", "CISTEM_VERSION_TEXT", 1, project_directory.GetFullPath( ).ToUTF8( ).data( ), project_name.ToUTF8( ).data( ), INTEGER_DATABASE_VERSION, total_cpu_hours, total_jobs_run, CISTEM_VERSION_TEXT) == false )
        return false;

    is_open = true;

    return true;
}

bool Project::OpenProjectFromFile(wxFileName file_to_open) {
    bool     success;
    wxString directory_string;

    // is project already open?

    if ( is_open == true ) {
        MyPrintWithDetails("Attempting to create a new project, but there is already an open project");
        return false;
    }

    success = database.Open(file_to_open);
    CheckSuccess(success);
    success = ReadMasterSettings( );
    CheckSuccess(success);

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Movies";
    movie_asset_directory = directory_string;
    if ( wxDir::Exists(movie_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(movie_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Images";
    image_asset_directory = directory_string;
    if ( wxDir::Exists(image_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(image_asset_directory.GetFullPath( ));

#ifdef EXPERIMENTAL
    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/TemplateMatching";
    template_matching_asset_directory = directory_string;
    if ( wxDir::Exists(template_matching_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(template_matching_asset_directory.GetFullPath( ));
#endif

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/PhaseDifferenceImages";
    phase_difference_asset_directory = directory_string;
    if ( wxDir::Exists(phase_difference_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(phase_difference_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Volumes";
    volume_asset_directory = directory_string;
    if ( wxDir::Exists(volume_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(volume_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/CTF";
    ctf_asset_directory = directory_string;
    if ( wxDir::Exists(ctf_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(ctf_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/ParticlePosition";
    particle_position_asset_directory = directory_string;
    if ( wxDir::Exists(particle_position_asset_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(particle_position_asset_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/ParticleStacks";
    particle_stack_directory = directory_string;
    if ( wxDir::Exists(particle_stack_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(particle_stack_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/ClassAverages";
    class_average_directory = directory_string;
    if ( wxDir::Exists(class_average_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(class_average_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Assets/Parameters";
    parameter_file_directory = directory_string;
    if ( wxDir::Exists(parameter_file_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(parameter_file_directory.GetFullPath( ));

    directory_string = project_directory.GetFullPath( );
    directory_string += "/Scratch";
    scratch_directory = directory_string;
    if ( wxDir::Exists(scratch_directory.GetFullPath( )) == false )
        wxFileName::Mkdir(scratch_directory.GetFullPath( ));

    // sub directories

    directory_string = image_asset_directory.GetFullPath( );
    directory_string += "/Spectra";
    if ( wxDir::Exists(directory_string) == false )
        wxFileName::Mkdir(directory_string);

    directory_string = image_asset_directory.GetFullPath( );
    directory_string += "/Scaled";
    if ( wxDir::Exists(directory_string) == false )
        wxFileName::Mkdir(directory_string);

    directory_string = volume_asset_directory.GetFullPath( );
    directory_string += "/OrthViews";
    if ( wxDir::Exists(directory_string) == false )
        wxFileName::Mkdir(directory_string);

    is_open = true;

    return success;
}

bool Project::ReadMasterSettings( ) {
    bool success;

    int imported_integer_version;

    //MyDebugAssertTrue(is_open == true, "Project not open!");

    success = database.GetMasterSettings(project_directory, project_name, integer_database_version, total_cpu_hours, total_jobs_run, cistem_version_text, current_workflow);

    if ( success == true ) {
        //MyDebugAssertTrue(imported_integer_version == INTEGER_DATABASE_VERSION, "Database version numbers are different!");
    }

    return success;
}

void Project::WriteProjectStatisticsToDatabase( ) {
    database.SetProjectStatistics(total_cpu_hours, total_jobs_run);
}

void Project::Close(bool remove_lock, bool update_statistics) {
    if ( update_statistics )
        WriteProjectStatisticsToDatabase( );
    database.UpdateVersion( );
    database.Close(remove_lock);

    is_open         = false;
    total_cpu_hours = 0;
    total_jobs_run  = 0;

    project_name      = "";
    project_directory = "";
}
