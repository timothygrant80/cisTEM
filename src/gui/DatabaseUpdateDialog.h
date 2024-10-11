#ifndef _SRC_GUI_DATABASE_UDPATE_DIALOG_H_
#define _SRC_GUI_DATABASE_UPDATE_DIALOG_H_

#include "ProjectX_gui_main.h"

class DatabaseUpdateDialog : public DatabaseUpdateDialogParent {
  public:
    // DatabaseUpdateDialog(wxWindow* parent);
    DatabaseUpdateDialog(wxWindow* parent, wxString db_changes);

    void OnButtonClicked(wxCommandEvent& event);

  private:
    wxString schema_changes;
};
#endif