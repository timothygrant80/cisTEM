#ifndef __PopupTextDialog__
#define __PopupTextDialog__

class PopupTextDialog : public PopupTextDialogParent
{

public :

	PopupTextDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxDEFAULT_DIALOG_STYLE);
	void OnCopyToClipboardClick( wxCommandEvent& event );
	void OnCloseButtonClick(wxCommandEvent &event);
	void OnSaveButtonClick(wxCommandEvent &event);
};

#endif

