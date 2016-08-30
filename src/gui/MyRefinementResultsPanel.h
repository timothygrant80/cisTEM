class MyRefinementResultsPanel : public RefinementResultsPanel
{


	public:

	bool is_dirty;

	MyRefinementResultsPanel( wxWindow* parent );
	void FillRefinementPackageComboBox(void);
	void FillInputParametersComboBox(void);
	void FillParameterListCtrl();
	void FillAngles();

	void OnUpdateUI( wxUpdateUIEvent& event );

	void OnRefinementPackageComboBox( wxCommandEvent& event );
	void OnInputParametersComboBox( wxCommandEvent& event );

	int current_class;

	void OnPlotButtonClick( wxCommandEvent& event );
	void OnClassComboBoxChange( wxCommandEvent& event );


};

