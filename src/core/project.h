#include "../constants/constants.h"

class Project {

  public:
    Database database;

    bool     is_open;
    std::string project_name;

    std::filesystem::path project_directory;
    std::filesystem::path movie_asset_directory;
    std::filesystem::path image_asset_directory;
    std::filesystem::path template_matching_asset_directory;
    std::filesystem::path phase_difference_asset_directory;
    std::filesystem::path volume_asset_directory;
    std::filesystem::path ctf_asset_directory;
    std::filesystem::path particle_position_asset_directory;
    std::filesystem::path particle_stack_directory;
    std::filesystem::path class_average_directory;

    std::filesystem::path parameter_file_directory;
    std::filesystem::path scratch_directory;

    double total_cpu_hours;
    int    total_jobs_run;

    int      integer_database_version;
    std::string cistem_version_text;
    std::string current_workflow; // It would be better to connect this somehow to main_frame.current_workflow or vice-versa

    Project( );
    ~Project( );

    void Close(bool remove_lock = true, bool update_statistics = true);
    bool CreateNewProject(std::filesystem::path database_file, std::string project_directory, std::string project_name);
    bool OpenProjectFromFile(std::filesystem::path file_to_open);
    bool ReadMasterSettings( );
    void WriteProjectStatisticsToDatabase( );

    inline bool RecordCurrentWorkflowInDB(std::string workflow) { return database.RecordCurrentWorkflowInDB(workflow); }
};
