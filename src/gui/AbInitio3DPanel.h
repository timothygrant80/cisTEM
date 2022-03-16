#ifndef __Abinitio3DPanel__
#define __Abinitio3DPanel__

class AbInitio3DPanel;

class AbInitioManager {
  public:
    AbInitioManager( );

    bool             start_with_reconstruction;
    AbInitio3DPanel* my_parent;

    long current_job_starttime;
    long time_of_last_update;
    int  number_of_generated_3ds;

    int running_job_type;
    int number_of_rounds_to_run;
    int number_of_rounds_run;

    long current_job_id;

    long current_refinement_package_asset_id;
    long current_input_refinement_id;
    long current_output_refinement_id;

    long number_of_received_particle_results;
    long number_of_expected_results;

    wxArrayFloat align_sym_best_correlations;
    wxArrayFloat align_sym_best_x_rots;
    wxArrayFloat align_sym_best_y_rots;
    wxArrayFloat align_sym_best_z_rots;
    wxArrayFloat align_sym_best_x_shifts;
    wxArrayFloat align_sym_best_y_shifts;
    wxArrayFloat align_sym_best_z_shifts;
    bool         apply_symmetry;

    RefinementPackage* active_refinement_package;
    Refinement*        input_refinement;
    Refinement*        output_refinement;

    ClassificationSelection active_classification_selection;

    RunProfile active_refinement_run_profile;
    RunProfile active_reconstruction_run_profile;

    bool active_always_apply_symmetry;

    wxString active_symmetry_string;

    //float startup_percent_used;
    float current_high_res_limit;
    float next_high_res_limit;

    float start_percent_used;
    float end_percent_used;
    float symmetry_start_percent_used;
    float symmetry_end_percent_used;

    float active_start_percent_used;
    float active_end_percent_used;
    float current_percent_used;

    float active_start_res;
    float active_end_res;

    bool active_should_automask;
    bool active_auto_set_percent_used;

    float active_inner_mask_radius;
    float active_global_mask_radius;

    float active_search_range_x;
    float active_search_range_y;

    bool active_should_apply_blurring;
    bool active_smoothing_factor;
    bool active_use_classums;

    int active_images_per_class;
    int active_number_of_2d_classes;

    int number_of_starts_to_run;
    int number_of_starts_run;

    wxArrayString current_reference_filenames;

    float    stack_bin_factor;
    wxString active_stack_filename;
    float    active_pixel_size;
    bool     stack_has_been_precomputed;

    void SetParent(AbInitio3DPanel* wanted_parent);

    void BeginRefinementCycle( );
    void CycleRefinement( );

    void UpdatePlotPanel( );

    void SetupPrepareStackJob( );
    void RunPrepareStackJob( );

    void SetupAlignSymmetryJob( );
    void RunAlignSymmetryJob( );

    void SetupRefinementJob( );
    void SetupReconstructionJob( );
    void SetupMerge3dJob( );

    void SetupInitialReconstructionJob( );
    void SetupInitialMerge3dJob( );

    void RunInitialReconstructionJob( );
    void RunInitialMerge3dJob( );

    void RunRefinementJob( );
    void RunReconstructionJob( );
    void RunMerge3dJob( );

    void ProcessJobResult(JobResult* result_to_process);
    void ProcessAllJobsFinished( );

    void OnMaskerThreadComplete( );
    void DoMasking( );
};

class AbInitio3DPanel : public AbInitio3DPanelParent {
    friend class AbInitioManager;

  protected:
    // Handlers for events.

    AbInitioManager my_abinitio_manager;

    void OnUpdateUI(wxUpdateUIEvent& event);
    void OnExpertOptionsToggle(wxCommandEvent& event);
    void OnInfoURL(wxTextUrlEvent& event);
    void TerminateButtonClick(wxCommandEvent& event);
    void FinishButtonClick(wxCommandEvent& event);
    void StartRefinementClick(wxCommandEvent& event);
    void ResetAllDefaultsClick(wxCommandEvent& event);

  public:
    wxStopWatch stopwatch;

    long time_of_last_result_update;
    bool refinement_package_combo_is_dirty;
    bool run_profiles_are_dirty;
    bool classification_selections_are_dirty;

    long selected_refinement_package;

    int           number_of_resampled_volumes_recieved;
    long          current_startup_id;
    wxArrayString resampled_volume_filenames;

    wxString old_symmetry;
    int      old_number_of_classes;
    int      old_class_selection;
    bool     old_automask_value;

    int active_orth_thread_id;
    int active_mask_thread_id;
    int active_sym_thread_id;

    int next_thread_id;

    bool running_job;

    AbInitio3DPanel(wxWindow* parent);

    wxString ReturnName( ) { return "AbInitioPanel"; }

    void WriteInfoText(wxString text_to_write);
    void WriteErrorText(wxString text_to_write);
    void WriteWarningText(wxString text_to_write);
    void WriteBlueText(wxString text_to_write);

    // overridden socket methods..

    void OnSocketJobResultMsg(JobResult& received_result);
    void OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue);
    void SetNumberConnectedText(wxString wanted_text);
    void SetTimeRemainingText(wxString wanted_text);
    void OnSocketAllJobsFinished( );

    void OnMaskerThreadComplete(wxThreadEvent& my_event);
    void OnOrthThreadComplete(ReturnProcessedImageEvent& my_event);
    void OnVolumeResampled(ReturnProcessedImageEvent& my_event);
    void OnImposeSymmetryThreadComplete(wxThreadEvent& event);

  public:
    void Reset( );
    void SetDefaults( );
    void SetInfo( );
    void FillRefinementPackagesComboBox( );
    void FillRunProfileComboBoxes( );
    void NewRefinementPackageSelected( );

    void OnRefinementPackageComboBox(wxCommandEvent& event);
    void OnMethodChange(wxCommandEvent& event);

    void TakeLastStartClicked(wxCommandEvent& event);
    void TakeCurrentClicked(wxCommandEvent& event);

    void TakeCurrent( );
    void TakeLastStart( );
};

class ResampleVolumeThread : public wxThread {
  public:
    ResampleVolumeThread(wxWindow* wanted_parent_window, wxString wanted_input_volume, int wanted_box_size, float wanted_output_pixel_size, int wanted_class_number) : wxThread(wxTHREAD_DETACHED) {
        parent_window     = wanted_parent_window;
        input_volume      = wanted_input_volume;
        box_size          = wanted_box_size;
        output_pixel_size = wanted_output_pixel_size;
        class_number      = wanted_class_number;
    }

    wxWindow* parent_window;
    wxString  input_volume;
    int       box_size;
    float     output_pixel_size;
    int       class_number;

  protected:
    ExitCode Entry( );
};

class ImposeAlignmentAndSymmetryThread : public wxThread {
  public:
    ImposeAlignmentAndSymmetryThread(wxWindow* wanted_parent_window, wxArrayString wanted_input_volumes, wxArrayString wanted_output_volumes, wxArrayFloat wanted_x_rots, wxArrayFloat wanted_y_rots, wxArrayFloat wanted_z_rots, wxArrayFloat wanted_x_shifts, wxArrayFloat wanted_y_shifts, wxArrayFloat wanted_z_shifts, wxString wanted_symmetry, int wanted_thread_id = -1) {
        parent_window  = wanted_parent_window;
        input_volumes  = wanted_input_volumes;
        output_volumes = wanted_output_volumes;
        x_rots         = wanted_x_rots;
        y_rots         = wanted_y_rots;
        z_rots         = wanted_z_rots;
        x_shifts       = wanted_x_shifts;
        y_shifts       = wanted_y_shifts;
        z_shifts       = wanted_z_shifts;
        symmetry       = wanted_symmetry;
        thread_id      = wanted_thread_id;
    }

    wxWindow*     parent_window;
    wxArrayString input_volumes;
    wxArrayString output_volumes;
    wxArrayFloat  x_rots;
    wxArrayFloat  y_rots;
    wxArrayFloat  z_rots;
    wxArrayFloat  x_shifts;
    wxArrayFloat  y_shifts;
    wxArrayFloat  z_shifts;
    wxString      symmetry;
    int           thread_id;

  protected:
    ExitCode Entry( );
};

#endif
