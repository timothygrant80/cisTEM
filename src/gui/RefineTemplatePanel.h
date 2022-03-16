#ifndef __RefineTemplatePanel__
#define __RefineTemplatePanel__

class RefineTemplatePanel : public RefineTemplatePanelParent {
    long my_job_id;

    JobTracker my_job_tracker;

    bool running_job;
    bool all_input_images_have_match_template_result;

    Image    result_image;
    wxBitmap result_bitmap;

    wxArrayString input_image_filenames;

    float ref_box_size_in_pixels;

    AssetGroup active_group;

    ArrayOfTemplateMatchJobResults cached_results;
    TemplateMatchJobResults        current_input_result;

  public:
    RefineTemplatePanel(wxWindow* parent);

    bool group_combo_is_dirty;
    bool run_profiles_are_dirty;
    bool volumes_are_dirty;
    long time_of_last_result_update;

    long expected_number_of_results;
    long number_of_received_results;
    long current_job_starttime;
    long time_of_last_update;

    // methods
    void WriteResultToDataBase( );
    void OnUpdateUI(wxUpdateUIEvent& event);
    void FillGroupComboBox( );
    void FillRunProfileComboBox( );
    void StartEstimationClick(wxCommandEvent& event);
    void FinishButtonClick(wxCommandEvent& event);
    void TerminateButtonClick(wxCommandEvent& event);
    void ResetAllDefaultsClick(wxCommandEvent& event);
    void OnGroupComboBox(wxCommandEvent& event);

    void OnSocketJobResultMsg(JobResult& received_result);
    void OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue);
    void SetNumberConnectedText(wxString wanted_text);
    void SetTimeRemainingText(wxString wanted_text);
    void OnSocketAllJobsFinished( );
    void HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes);
    bool CheckGroupHasTemplateMatchRunDone( );

    //void Refresh();
    void SetInfo( );
    void OnInfoURL(wxTextUrlEvent& event);

    void WriteInfoText(wxString text_to_write);
    void WriteErrorText(wxString text_to_write);

    void ProcessResult(JobResult* result_to_process);
    void ProcessAllJobsFinished( );
    void UpdateProgressBar( );

    void Reset( );
    void ResetDefaults( );
};

#endif
