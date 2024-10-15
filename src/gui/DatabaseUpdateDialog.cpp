#include "DatabaseUpdateDialog.h"

DatabaseUpdateDialog::DatabaseUpdateDialog(wxWindow* parent, wxString db_changes) : DatabaseUpdateDialogParent(parent) {
    schema_changes = db_changes;
    SchemaChangesTextCtrl->AppendText(db_changes);
}

void DatabaseUpdateDialog::OnButtonClicked(wxCommandEvent& event) {
    int button_id = event.GetId( );

    enum UpdateOptions { CANCEL            = 1,
                         UPDATE_ONLY       = 2,
                         BACKUP_AND_UPDATE = 3 };

    switch ( button_id ) {
        case wxID_CANCEL:
            EndModal(CANCEL);
            break;
        case wxID_UPDATE_ONLY:
            EndModal(UPDATE_ONLY);
            break;
        case wxID_BACKUP_AND_UPDATE:
            EndModal(BACKUP_AND_UPDATE);
            break;
        default:
            EndModal(CANCEL);
            break;
    }
}