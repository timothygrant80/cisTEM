#include "../core/gui_core_headers.h"

extern MyVolumeAssetPanel *volume_asset_panel;

MyVolumeChooserDialog::MyVolumeChooserDialog( wxWindow* parent )
:
VolumeChooserDialog( parent )
{
	selected_volume_id = -1;
	selected_volume_name = "Generate from params.";
	ComboBox->Append("Generate from params.");
	AppendVolumeAssetsToComboBox(ComboBox);
}

void MyVolumeChooserDialog::OnCancelClick( wxCommandEvent& event )
{
	Destroy();
}

void MyVolumeChooserDialog::OnRenameClick( wxCommandEvent& event )
{
	if (ComboBox->GetSelection() == 0)
	{
		selected_volume_id = -1;
		selected_volume_name = "Generate from params.";
	}
	else
	{
		selected_volume_id = volume_asset_panel->ReturnAssetID(ComboBox->GetSelection() - 1);
		selected_volume_name = volume_asset_panel->ReturnAssetName(ComboBox->GetSelection() - 1);
	}

	EndModal(wxID_OK);
}

