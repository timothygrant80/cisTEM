#include "../core/gui_core_headers.h"

extern AtomicCoordinatesAssetPanel* atomic_coordinates_asset_panel;

AtomicCoordinatesChooserDialog::AtomicCoordinatesChooserDialog(wxWindow* parent)
    : AtomicCoordinatesChooserDialogParent(parent) {
    selected_volume_id   = -1;
    selected_volume_name = "Generate from params.";
    ComboBox->FillComboBox(true);
}

void AtomicCoordinatesChooserDialog::OnCancelClick(wxCommandEvent& event) {
    Destroy( );
}

void AtomicCoordinatesChooserDialog::OnRenameClick(wxCommandEvent& event) {
    if ( ComboBox->GetSelection( ) == 0 ) {
        selected_volume_id   = -1;
        selected_volume_name = "Generate from params.";
    }
    else {
        selected_volume_id   = atomic_coordinates_asset_panel->ReturnAssetID(ComboBox->GetSelection( ) - 1);
        selected_volume_name = atomic_coordinates_asset_panel->ReturnAssetName(ComboBox->GetSelection( ) - 1);
    }

    EndModal(wxID_OK);
}
