class MyRefinementResultsPanel : public RefinementResultsPanel
{


	public:

	bool is_dirty;
	bool input_params_are_dirty;

	long refinement_id_of_buffered_refinement;
	Refinement *currently_displayed_refinement;
	Refinement *buffered_full_refinement;

	MyRefinementResultsPanel( wxWindow* parent );
	void FillRefinementPackageComboBox(void);
	void FillInputParametersComboBox(void);
	void FillAngles(int wanted_class);
	void DrawOrthViews();

	void OnUpdateUI( wxUpdateUIEvent& event );

	void OnRefinementPackageComboBox( wxCommandEvent& event );
	void OnInputParametersComboBox( wxCommandEvent& event );
	void OnDisplayTabChange(wxAuiNotebookEvent& event);
	void OnJobDetailsToggle( wxCommandEvent& event );

	int current_class;

	void OnClassComboBoxChange( wxCommandEvent& event );
	void AngularPlotPopupClick(wxCommandEvent& event);
	void PopupParametersClick(wxCommandEvent& event);

	void UpdateCachedRefinement();
	void UpdateBufferedFullRefinement();

	void WriteJobInfo(int wanted_class);
	void ClearJobInfo();




};

