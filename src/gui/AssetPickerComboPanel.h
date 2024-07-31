#ifndef __ASSETPICKERCOMBO_PANEL_H__
#define __ASSETPICKERCOMBO_PANEL_H__

class AssetPickerComboPanel : public AssetPickerComboPanelParent {
  public:
    AssetPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    virtual ~AssetPickerComboPanel( );
    void ParentPopUpSelectorClicked(wxCommandEvent& event);

    void OnSize(wxSizeEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);
    void OnNextButtonClick(wxCommandEvent& event);
    void OnPreviousButtonClick(wxCommandEvent& event);

    void SetSelection(int wanted_selection) { AssetComboBox->SetSelection(wanted_selection); }

    void SetSelectionWithEvent(int wanted_selection) { AssetComboBox->SetSelectionWithEvent(wanted_selection); }

    int ReturnSelection( ) { return AssetComboBox->GetSelection( ); }

    int GetSelection( ) { return AssetComboBox->GetSelection( ); }

    int GetCount( ) { return AssetComboBox->GetCount( ); }

    void Clear( ) { AssetComboBox->Clear( ); }

    void ChangeValue(wxString value_to_set) { AssetComboBox->ChangeValue(value_to_set); }

    virtual void GetAssetFromPopup( );
    //virtual bool FillComboBox() = 0;
};

class ClassSelectionPickerComboPanel : public AssetPickerComboPanel {
  public:
    ClassSelectionPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(bool always_select_latest = false, long wanted_refinement_package_id = -1) { return AssetComboBox->FillWithClassAverageSelections(always_select_latest, wanted_refinement_package_id); }
};

class VolumeAssetPickerComboPanel : public AssetPickerComboPanel {
  public:
    VolumeAssetPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(bool include_generate_from_params = false, bool always_select_latest = false) { return AssetComboBox->FillWithVolumeAssets(include_generate_from_params, always_select_latest); }
};

#ifdef EXPERIMENTAL
class AtomicCoordinatesAssetPickerComboPanel : public AssetPickerComboPanel {
  public:
    AtomicCoordinatesAssetPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(bool include_generate_from_params = false, bool always_select_latest = false) { return AssetComboBox->FillWithAtomicCoordinatesAssets(include_generate_from_params, always_select_latest); }
};
#endif

class RefinementPackagePickerComboPanel : public AssetPickerComboPanel {
  public:
    RefinementPackagePickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox( ) { return AssetComboBox->FillWithRefinementPackages( ); }
};

class RefinementPickerComboPanel : public AssetPickerComboPanel {
  public:
    RefinementPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(long wanted_refinement_package, bool always_select_latest = false) { return AssetComboBox->FillWithRefinements(wanted_refinement_package, always_select_latest); }
};

class ClassificationPickerComboPanel : public AssetPickerComboPanel {
  public:
    ClassificationPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(long wanted_refinement_package, bool include_new_classification, bool always_select_latest = false) { return AssetComboBox->FillWithClassifications(wanted_refinement_package, include_new_classification, always_select_latest); }
};

class ImageGroupPickerComboPanel : public AssetPickerComboPanel {
  public:
    ImageGroupPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(bool include_all_images_group) { return AssetComboBox->FillWithImageGroups(include_all_images_group); }
};

class TMJobPickerComboPanel : public AssetPickerComboPanel {
  public:
    TMJobPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox( ) { return AssetComboBox->FillWithTMJobs( ); }
};

class TMPackagePickerComboPanel : public AssetPickerComboPanel {
  public:
    TMPackagePickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox( ) { return AssetComboBox->FillWithTMPackages( ); }
};

class MovieGroupPickerComboPanel : public AssetPickerComboPanel {
  public:
    MovieGroupPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(bool include_all_movies_group) { return AssetComboBox->FillWithMovieGroups(include_all_movies_group); }
};

class ImagesPickerComboPanel : public AssetPickerComboPanel {
  public:
    ImagesPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    bool FillComboBox(long wanted_image_group) { return AssetComboBox->FillWithImages(wanted_image_group); }
};

class AssetPickerListCtrl : public wxListCtrl {
  public:
    AssetPickerListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    virtual wxString OnGetItemText(long item, long column) const;
};

#endif
