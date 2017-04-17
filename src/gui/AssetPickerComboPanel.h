#ifndef __ASSETPICKERCOMBO_PANEL_H__
#define	__ASSETPICKERCOMBO_PANEL_H__

class AssetPickerComboPanel : public AssetPickerComboPanelParent
{
public :

	AssetPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	void ParentPopUpSelectorClicked(wxCommandEvent& event);

	int ReturnSelection() {return AssetComboBox->GetSelection();};
	virtual void GetAssetFromPopup() = 0;
	virtual void FillComboBox() = 0;

};

class VolumeAssetPickerComboPanel : public AssetPickerComboPanel
{
public:
	VolumeAssetPickerComboPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	void GetAssetFromPopup();
	void FillComboBox();
};


#endif
