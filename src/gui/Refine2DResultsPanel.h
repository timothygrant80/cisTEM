#ifndef __Refine2DResultsPanel__
#define __Refine2DResultsPanel__

class
Refine2DResultsPanel : public Refine2DResultsPanelParent
{

public:
	Refine2DResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~Refine2DResultsPanel();

	void FillRefinementPackageComboBox(void);
	void FillInputParametersComboBox(void);

	void OnUpdateUI( wxUpdateUIEvent& event );
	void OnRefinementPackageComboBox( wxCommandEvent& event );
	void OnInputParametersComboBox( wxCommandEvent& event );

	void OnAddButtonClick( wxCommandEvent& event );
	void OnDeleteButtonClick( wxCommandEvent& event );
	void OnRenameButtonClick( wxCommandEvent& event );
	void OnClearButtonClick( wxCommandEvent& event );
	void OnInvertButtonClick( wxCommandEvent& event );
	void OnCopyOtherButtonClick( wxCommandEvent& event );
	void OnJobDetailsToggle( wxCommandEvent& event );

	void OnClassumRightClick( wxMouseEvent& event );
	void OnClassumLeftClick( wxMouseEvent& event );

	void WriteJobInfo(long wanted_classification_id);

	void FillSelectionManagerListCtrl(bool select_latest = false);

	void OnDeselected( wxListEvent& event );
	void OnSelected( wxListEvent& event );
	void OnActivated( wxListEvent& event );
	void OnEndLabelEdit( wxListEvent& event );

	void Clear();
	void ClearJobInfo();



	bool refinement_package_combo_is_dirty;
	bool input_params_combo_is_dirty;
	bool classification_selections_are_dirty;
	int selected_class;
};


enum
{
	Toolbar_New,
	Toolbar_Delete
};


#endif

