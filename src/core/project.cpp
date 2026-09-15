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

bool Project::CreateNewProject(std::filesystem::path wanted_database_file, std::string wanted_project_directory, std::string wanted_project_name) {
    int      return_code;
    std::string directory_string;
    bool     success;

    // is project already open?

    if ( is_open == true ) {
        MyPrintWithDetails("Attempting to create a new project, but there is already an open project");
        return false;
    }

    if ( wanted_project_name.empty() == true ) {
        MyDebugPrintWithDetails("Attempting to create a new project, but the project name is blank");
        return false;
    }

    if ( wanted_project_directory.empty() == true ) {
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

    directory_string = project_directory.string();
    directory_string += "/Assets";
    MakeDirectory(directory_string);

    directory_string = project_directory.string();
    directory_string += "/Assets/Movies";
    movie_asset_directory = directory_string;
    MakeDirectory(movie_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/Images";
    image_asset_directory = directory_string;
    MakeDirectory(image_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/Volumes";
    volume_asset_directory = directory_string;
    MakeDirectory(volume_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/TemplateMatching";
    template_matching_asset_directory = directory_string;
    if ( DirectoryExists(template_matching_asset_directory.string()) == false )
        MakeDirectory(template_matching_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/PhaseDifferenceImages";
    phase_difference_asset_directory = directory_string;
    MakeDirectory(phase_difference_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/CTF";
    ctf_asset_directory = directory_string;
    MakeDirectory(ctf_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/ParticlePosition";
    particle_position_asset_directory = directory_string;
    MakeDirectory(particle_position_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/ParticleStacks";
    particle_stack_directory = directory_string;
    MakeDirectory(particle_stack_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/ClassAverages";
    class_average_directory = directory_string;
    MakeDirectory(class_average_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/Parameters";
    parameter_file_directory = directory_string;
    MakeDirectory(parameter_file_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Scratch";
    scratch_directory = directory_string;
    MakeDirectory(scratch_directory.string());

    // sub directories

    directory_string = image_asset_directory.string();
    directory_string += "/Spectra";
    MakeDirectory(directory_string);

    directory_string = image_asset_directory.string();
    directory_string += "/Scaled";
    MakeDirectory(directory_string);

    directory_string = volume_asset_directory.string();
    directory_string += "/OrthViews";
    MakeDirectory(directory_string);

    total_cpu_hours = 0;
    total_jobs_run  = 0;

    // set master settings..

    if ( database.InsertOrReplace("MASTER_SETTINGS", "ittirit", "NUMBER", "PROJECT_DIRECTORY", "PROJECT_NAME", "CURRENT_VERSION", "TOTAL_CPU_HOURS", "TOTAL_JOBS_RUN", "CISTEM_VERSION_TEXT", 1, project_directory.string().c_str(), project_name.c_str(), INTEGER_DATABASE_VERSION, total_cpu_hours, total_jobs_run, CISTEM_VERSION_TEXT) == false )
        return false;

    is_open = true;

    return true;
}

bool Project::OpenProjectFromFile(std::filesystem::path file_to_open) {
    bool     success;
    std::string directory_string;

    // is project already open?

    if ( is_open == true ) {
        MyPrintWithDetails("Attempting to create a new project, but there is already an open project");
        return false;
    }

    success = database.Open(file_to_open);
    CheckSuccess(success);
    success = ReadMasterSettings( );
    CheckSuccess(success);

    directory_string = project_directory.string();
    directory_string += "/Assets/Movies";
    movie_asset_directory = directory_string;
    if ( DirectoryExists(movie_asset_directory.string()) == false )
        MakeDirectory(movie_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/Images";
    image_asset_directory = directory_string;
    if ( DirectoryExists(image_asset_directory.string()) == false )
        MakeDirectory(image_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/TemplateMatching";
    template_matching_asset_directory = directory_string;
    if ( DirectoryExists(template_matching_asset_directory.string()) == false )
        MakeDirectory(template_matching_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/PhaseDifferenceImages";
    phase_difference_asset_directory = directory_string;
    if ( DirectoryExists(phase_difference_asset_directory.string()) == false )
        MakeDirectory(phase_difference_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/Volumes";
    volume_asset_directory = directory_string;
    if ( DirectoryExists(volume_asset_directory.string()) == false )
        MakeDirectory(volume_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/CTF";
    ctf_asset_directory = directory_string;
    if ( DirectoryExists(ctf_asset_directory.string()) == false )
        MakeDirectory(ctf_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/ParticlePosition";
    particle_position_asset_directory = directory_string;
    if ( DirectoryExists(particle_position_asset_directory.string()) == false )
        MakeDirectory(particle_position_asset_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/ParticleStacks";
    particle_stack_directory = directory_string;
    if ( DirectoryExists(particle_stack_directory.string()) == false )
        MakeDirectory(particle_stack_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/ClassAverages";
    class_average_directory = directory_string;
    if ( DirectoryExists(class_average_directory.string()) == false )
        MakeDirectory(class_average_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Assets/Parameters";
    parameter_file_directory = directory_string;
    if ( DirectoryExists(parameter_file_directory.string()) == false )
        MakeDirectory(parameter_file_directory.string());

    directory_string = project_directory.string();
    directory_string += "/Scratch";
    scratch_directory = directory_string;
    if ( DirectoryExists(scratch_directory.string()) == false )
        MakeDirectory(scratch_directory.string());

    // sub directories

    directory_string = image_asset_directory.string();
    directory_string += "/Spectra";
    if ( DirectoryExists(directory_string) == false )
        MakeDirectory(directory_string);

    directory_string = image_asset_directory.string();
    directory_string += "/Scaled";
    if ( DirectoryExists(directory_string) == false )
        MakeDirectory(directory_string);

    directory_string = volume_asset_directory.string();
    directory_string += "/OrthViews";
    if ( DirectoryExists(directory_string) == false )
        MakeDirectory(directory_string);

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
