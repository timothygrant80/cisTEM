#include "../core/gui_core_headers.h"


PopupTextDialog::PopupTextDialog (wxWindow *parent, wxWindowID id, const wxString &title, const wxPoint &pos, const wxSize &size, long style)
:
PopupTextDialogParent( parent, id, title, pos, size, style)
{
	ClipBoardButton->SetBitmap(wxArtProvider::GetBitmap(wxART_COPY));
}

void PopupTextDialog::OnCopyToClipboardClick( wxCommandEvent& event )
{
	if (wxTheClipboard->Open())
	{
	    wxTheClipboard->SetData( new wxTextDataObject(OutputTextCtrl->GetValue()) );
	    wxTheClipboard->Close();
	}
}
void PopupTextDialog::OnSaveButtonClick( wxCommandEvent& event )
{
	ProperOverwriteCheckSaveDialog *saveFileDialog;
	saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save txt file"), "TXT files (*.txt)|*.txt", ".txt");
	if (saveFileDialog->ShowModal() == wxID_CANCEL)
	{
		saveFileDialog->Destroy();
		return;
	}

	// save the file then..

	OutputTextCtrl->SaveFile(saveFileDialog->ReturnProperPath());
	saveFileDialog->Destroy();
}

void PopupTextDialog::OnCloseButtonClick(wxCommandEvent &event)
{
	Destroy();
}
