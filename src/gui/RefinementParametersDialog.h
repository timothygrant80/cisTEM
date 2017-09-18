#ifndef __RefinementParametersDialog__
#define __RefinementParametersDialog__

class RefinementParametersDialog : public RefinementParametersDialogParent
{

public :

	int current_class;
	RefinementParametersDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxDEFAULT_DIALOG_STYLE);
	void OnCloseButtonClick(wxCommandEvent &event);
	void OnSaveButtonClick(wxCommandEvent &event);
	void OnSelectionChange(wxCommandEvent &event);
};

#endif

