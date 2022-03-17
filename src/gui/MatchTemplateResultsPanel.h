extern MatchTemplatePanel* match_template_panel;

class MatchTemplateResultsPanel : public MatchTemplateResultsPanelParent {
  public:
    MatchTemplateResultsPanel(wxWindow* parent);

    /*
		void OnDefineFilterClick( wxCommandEvent& event );
		void OnAddToGroupClick( wxCommandEvent& event );
		void OnNextButtonClick( wxCommandEvent& event );
		void OnPreviousButtonClick( wxCommandEvent& event );

		void OnShowTypeRadioBoxChange(wxCommandEvent& event);
		void OnJobDetailsToggle( wxCommandEvent& event );

		void OnAllMoviesSelect( wxCommandEvent& event );
		void OnByFilterSelect( wxCommandEvent& event );

		int GetFilter();



		int ReturnRowFromAssetID(int asset_id, int start_location = 0);

		void Clear();
*/
    void OnDefineFilterClick(wxCommandEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);
    void OnValueChanged(wxDataViewEvent& event);
    void FillGroupComboBox( );
    void FillBasedOnSelectCommand(wxString wanted_command);
    int  ReturnRowFromAssetID(int asset_id, int start_location = 0);
    void OnAddAllToGroupClick(wxCommandEvent& event);
    void FillResultsPanelAndDetails(int row, int column);
    void OnNextButtonClick(wxCommandEvent& event);
    void OnPreviousButtonClick(wxCommandEvent& event);
    void OnAddToGroupClick(wxCommandEvent& event);
    void OnRemoveFromGroupClick(wxCommandEvent& event);
    void OnJobDetailsToggle(wxCommandEvent& event);
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

    wxString current_fill_command;

    static int OnHeaderClick( ) {
        // Rather than access members directly, create a method in MatchTemplatePanel to do this stuff.
        match_template_panel->SymmetryComboBox->SetSelection(10);

        // First we'll want to check to see if all the results match the expected and whether or not their job_status is SUCCESS.

        // If they are all complete we have nothing to do except update the database to say this is ACTIVE

        // If the are not all complete, let's disable all input in the actions panel, and then enable the "re-run" button. The user should be able to untick this.
        return 1;
    }
};
