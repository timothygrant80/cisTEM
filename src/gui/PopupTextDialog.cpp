#include "../core/gui_core_headers.h"


PopupTextDialog::PopupTextDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos, const wxSize &size, long style)
:
PopupTextDialogParent( parent, id, title, pos, size, style)
{

}

void PopupTextDialog::OnCopyToClipboardClick( wxCommandEvent& event )
{
	if (wxTheClipboard->Open())
	{
	    wxTheClipboard->SetData( new wxTextDataObject(OutputTextCtrl->GetValue()) );
	    wxTheClipboard->Close();
	}
}
