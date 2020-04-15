#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;

MyPhenixSettingsPanel::MyPhenixSettingsPanel( wxWindow* parent )
:
PhenixSettingsPanel( parent )
{
	buffer_phenix_path = "";
}

void MyPhenixSettingsPanel::OnPhenixPathTextChanged( wxCommandEvent& event )
{
	buffer_phenix_path = PhenixPathTextCtrl->GetValue();
	if (buffer_phenix_path.Find("bin") ==  wxNOT_FOUND) //later: find --> endswith
	{
		PhenixPathErrorStaticText->SetLabel("Path must be to a bin directory");
	}
	else
	{
		PhenixPathErrorStaticText->SetLabel("");
//		main_frame->current_project.database.UpdatePhenixPath(buffer_phenix_path);
	}
	event.Skip();
}

void MyPhenixSettingsPanel::OnPhenixPathBrowseButtonClick( wxCommandEvent& event )
{
	wxDirDialog openDirDialog(NULL, "Choose Phenix bin directory", "", wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

	if (openDirDialog.ShowModal() == wxID_OK)
	{
		PhenixPathTextCtrl->SetValue(openDirDialog.GetPath());
	}
	event.Skip();
}

void MyPhenixSettingsPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	event.Skip();
}
