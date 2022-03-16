class MyMovieAlignResultsPanel : public MovieAlignResultsPanel {
  public:
    MyMovieAlignResultsPanel(wxWindow* parent);

    void JunkMe(wxCommandEvent& event);
    void OnDefineFilterClick(wxCommandEvent& event);
    void OnAddToGroupClick(wxCommandEvent& event);
    void OnRemoveFromGroupClick(wxCommandEvent& event);
    void OnNextButtonClick(wxCommandEvent& event);
    void OnPreviousButtonClick(wxCommandEvent& event);
    void OnAddAllToGroupClick(wxCommandEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);
    void OnShowTypeRadioBoxChange(wxCommandEvent& event);
    void OnJobDetailsToggle(wxCommandEvent& event);

    void OnAllMoviesSelect(wxCommandEvent& event);
    void OnByFilterSelect(wxCommandEvent& event);
    void OnCharHook(wxKeyEvent& event);

    int GetFilter( );

    void OnValueChanged(wxDataViewEvent& event);
    void DrawCurveAndFillDetails(int row, int column);
    int  ReturnRowFromAssetID(int asset_id, int start_location = 0);
    void FillBasedOnSelectCommand(wxString wanted_command);
    void Clear( );

    void FillGroupComboBox( );

    int* alignment_job_ids;
    int  number_of_alignmnet_jobs;
    int* per_row_asset_id;
    int* per_row_array_position;
    int  number_of_assets;

    int selected_row;
    int selected_column;

    bool doing_panel_fill;

    bool is_dirty;
    bool group_combo_is_dirty;

    wxString current_fill_command;
};
