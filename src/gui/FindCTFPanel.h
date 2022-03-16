#ifndef __FindCTFPanel__
#define __FindCTFPanel__

class MyFindCTFPanel : public FindCTFPanel {
    long my_job_id;

    int length_of_process_number;

    JobTracker my_job_tracker;

    bool running_job;

    Image    result_image;
    wxBitmap result_bitmap;

    AssetGroup active_group;

  public:
    MyFindCTFPanel(wxWindow* parent);
    JobResult* buffered_results;

    bool group_combo_is_dirty;
    bool run_profiles_are_dirty;
    long time_of_last_result_update;

    // methods

    void OnMovieRadioButton(wxCommandEvent& event);
    void OnImageRadioButton(wxCommandEvent& event);
    void OnFindAdditionalPhaseCheckBox(wxCommandEvent& event);
    void OnRestrainAstigmatismCheckBox(wxCommandEvent& event);
    void WriteResultToDataBase( );
    void OnExpertOptionsToggle(wxCommandEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);
    void FillGroupComboBox( );
    void FillRunProfileComboBox( );
    void StartEstimationClick(wxCommandEvent& event);
    void FinishButtonClick(wxCommandEvent& event);
    void TerminateButtonClick(wxCommandEvent& event);
    //void Refresh();
    void SetInfo( );
    void OnInfoURL(wxTextUrlEvent& event);
    void EnableMovieProcessingIfAppropriate( );

    void OnSocketJobResultMsg(JobResult& received_result);
    void SetNumberConnectedText(wxString wanted_text);
    void SetTimeRemainingText(wxString wanted_text);
    void OnSocketAllJobsFinished( );

    void WriteInfoText(wxString text_to_write);
    void WriteErrorText(wxString text_to_write);

    void ProcessResult(JobResult* result_to_process);
    void ProcessAllJobsFinished( );
    void UpdateProgressBar( );

    void Reset( );
    void ResetDefaults( );
};

#endif // __AlignMoviesPanel__
