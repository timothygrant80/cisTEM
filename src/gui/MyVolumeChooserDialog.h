#ifndef __MyVolumeChooserDialog__
#define __MyVolumeChooserDialog__

#include "ProjectX_gui.h"

class MyVolumeChooserDialog : public VolumeChooserDialog {

  public:
    long     selected_volume_id;
    wxString selected_volume_name;

    MyVolumeChooserDialog(wxWindow* parent);
    virtual void OnCancelClick(wxCommandEvent& event);
    virtual void OnRenameClick(wxCommandEvent& event);
};

#endif
