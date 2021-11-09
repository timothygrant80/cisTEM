#ifndef __AtomicCoordinatesChooserDialog__
#define __AtomicCoordinatesChooserDialog__

#include "ProjectX_gui.h"

class AtomicCoordinatesChooserDialog : public AtomicCoordinatesChooserDialogParent
{

	public :

	long selected_volume_id;
	wxString selected_volume_name;

  AtomicCoordinatesChooserDialog (wxWindow *parent);
	virtual void OnCancelClick( wxCommandEvent& event );
	virtual void OnRenameClick( wxCommandEvent& event );
};

#endif

