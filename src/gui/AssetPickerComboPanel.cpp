#include "../core/gui_core_headers.h"

extern MyVolumeAssetPanel *volume_asset_panel;

AssetPickerComboPanel::AssetPickerComboPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: AssetPickerComboPanelParent(parent, id, pos, size, style)
{
	wxLogNull *suppress_png_warnings = new wxLogNull;
	#include "icons/window_plus_icon_16.cpp"
	wxBitmap niko_picture1_bmp = wxBITMAP_PNG_FROM_DATA(window_plus_icon_16);
	WindowSelectButton->SetBitmap(niko_picture1_bmp);

	Bind(wxEVT_BUTTON, &AssetPickerComboPanel::ParentPopUpSelectorClicked, this);

}

void AssetPickerComboPanel::ParentPopUpSelectorClicked(wxCommandEvent& event)
{
	if (AssetComboBox->GetCount() > 1) GetAssetFromPopup();
}


VolumeAssetPickerComboPanel::VolumeAssetPickerComboPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: AssetPickerComboPanel(parent, id, pos, size, style)
{

}

void VolumeAssetPickerComboPanel::GetAssetFromPopup()
{
	int counter;
	ListCtrlDialog *picker_dialog = new ListCtrlDialog(this, wxID_ANY, "Select a Volume Asset");

	picker_dialog->MyListCtrl->InsertColumn(0, "Volume Asset", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

	for (counter = 0; counter < volume_asset_panel->all_assets_list->number_of_assets; counter++)
	{
		picker_dialog->MyListCtrl->InsertItem(counter, volume_asset_panel->ReturnAssetName(counter), counter);
	}

	if (volume_asset_panel->all_assets_list->number_of_assets > 0)
	{
		if (AssetComboBox->GetSelection() == -1)
		{
			picker_dialog->MyListCtrl->SetItemState(0, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
			picker_dialog->MyListCtrl->EnsureVisible(0);
		}
		else
		{
			picker_dialog->MyListCtrl->SetItemState(AssetComboBox->GetSelection(), wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
			picker_dialog->MyListCtrl->EnsureVisible(AssetComboBox->GetSelection());
		}
	}

	int client_width;
	int client_height;
	int current_width;

	picker_dialog->MyListCtrl->GetClientSize(&client_width, &client_height);
	picker_dialog->MyListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);

	current_width = picker_dialog->MyListCtrl->GetColumnWidth(0);

	if (client_width > current_width) picker_dialog->MyListCtrl->SetColumnWidth(0, client_width);

	if (picker_dialog->ShowModal() == wxID_OK)
	{
		int selected_item = picker_dialog->MyListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);

		if (selected_item != -1)
		{
			AssetComboBox->SetSelection(selected_item);

		}
	}

	picker_dialog->Destroy();
}

void VolumeAssetPickerComboPanel::FillComboBox()
{
	AssetComboBox->FillWithVolumeAssets();
}
