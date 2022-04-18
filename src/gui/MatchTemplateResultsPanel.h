
class MatchTemplateResultsPanel : public MatchTemplateResultsPanelParent {
  public:
    MatchTemplateResultsPanel(wxWindow* parent);
    // The results and actions panels need to talk to each other.
    friend class MatchTemplatePanel;

    void OnDefineFilterClick(wxCommandEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);
    void OnValueChanged(wxDataViewEvent& event);
    void FillGroupComboBox( );
    void FillBasedOnSelectCommand(wxString wanted_command, int wanted_volume_asset_id = -1);
    int  ReturnRowFromAssetID(int asset_id, int start_location = 0);
    void OnAddAllToGroupClick(wxCommandEvent& event);
    void FillResultsPanelAndDetails(int row, int column);
    void OnNextButtonClick(wxCommandEvent& event);
    void OnPreviousButtonClick(wxCommandEvent& event);
    void OnAddToGroupClick(wxCommandEvent& event);
    void OnRemoveFromGroupClick(wxCommandEvent& event);
    void OnJobDetailsToggle(wxCommandEvent& event);
    void OnTemplateComboBox(wxCommandEvent& event);
    void Clear( );

    int  GetFilter( );
    void OnAllImagesSelect(wxCommandEvent& event);
    void OnByFilterSelect(wxCommandEvent& event);
    void OnCharHook(wxKeyEvent& event);

    long* template_match_job_ids;
    int   number_of_template_match_ids;
    int*  per_row_asset_id;
    int*  per_row_array_position;
    int   number_of_assets;

    int selected_row;
    int selected_column;

    bool doing_panel_fill;

    bool is_dirty;
    bool group_combo_is_dirty;
    bool volumes_are_dirty;

    wxString current_fill_command;
};
