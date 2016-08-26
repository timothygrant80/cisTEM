#ifndef __MyRenameDialog__
#define __MyRenameDialog__

#include "ProjectX_gui.h"

class MyRenameDialog : public RenameDialog
{
	friend class MyAssetParentPanel;

public :

	wxArrayLong selected_assets_array_position;

	MyRenameDialog (wxWindow *parent);
	virtual void OnCancelClick( wxCommandEvent& event );
	virtual void OnRenameClick( wxCommandEvent& event );
	void SizeAndPosition();
};

#endif

