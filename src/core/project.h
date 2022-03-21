#include "cistem_constants.h"

class Project {

  public:
    Database database;

    bool     is_open;
    wxString project_name;

    wxFileName project_directory;
    wxFileName movie_asset_directory;
    wxFileName image_asset_directory;
    wxFileName template_matching_asset_directory;
    wxFileName phase_difference_asset_directory;
    wxFileName volume_asset_directory;
    wxFileName ctf_asset_directory;
    wxFileName particle_position_asset_directory;
    wxFileName particle_stack_directory;
    wxFileName class_average_directory;

    wxFileName parameter_file_directory;
    wxFileName scratch_directory;

    double total_cpu_hours;
    int    total_jobs_run;

    int                    integer_database_version;
    wxString               cistem_version_text;
    cistem::workflow::Enum current_workflow;

    Project( );
    ~Project( );

    void Close(bool remove_lock = true, bool update_statistics = true);
    bool CreateNewProject(wxFileName database_file, wxString project_directory, wxString project_name);
    bool OpenProjectFromFile(wxFileName file_to_open);
    bool ReadMasterSettings( );
    void WriteProjectStatisticsToDatabase( );

    inline bool RecordCurrentWorkflowInDB(cistem::workflow::Enum workflow) { return database.RecordCurrentWorkflowInDB(workflow); }
};
