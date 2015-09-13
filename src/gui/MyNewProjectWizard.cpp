#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;

MyNewProjectWizard::MyNewProjectWizard( wxWindow* parent )
:
NewProjectWizard( parent )
{

	// Set the inital path to the user's home directory..

	ParentDirTextCtrl->SetValue(wxStandardPaths::Get().GetUserConfigDir());

	// Generate project path..

	GenerateProjectPath();


}

void MyNewProjectWizard::OnBrowseButtonClick( wxCommandEvent& event )
{

	wxDirDialog dlg(NULL, "Choose input directory", ParentDirTextCtrl->GetValue(),  wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

	if (dlg.ShowModal() == wxID_OK)
	{
		ParentDirTextCtrl->SetValue(dlg.GetPath());
	}

}


void MyNewProjectWizard::GenerateProjectPath()
{
	wxString current_project_path = ParentDirTextCtrl->GetValue();

	if (current_project_path.EndsWith("/") == false) current_project_path += "/";
	current_project_path += ProjectNameTextCtrl->GetValue();
	ProjectPathTextCtrl->ChangeValue(current_project_path);


}

void MyNewProjectWizard::OnProjectTextChange( wxCommandEvent& event )
{

	GenerateProjectPath();
	CheckProjectPath();
}

void MyNewProjectWizard::OnParentDirChange( wxCommandEvent& event )
{
	GenerateProjectPath();
	CheckProjectPath();

}

void MyNewProjectWizard::OnProjectPathChange(wxCommandEvent& event)
{
	int number_of_directories;
	wxString path_string = "/";

	wxFileName current_dirname = wxFileName::DirName(ProjectPathTextCtrl->GetValue());

	// the last directory is the project name..

	number_of_directories = current_dirname.GetDirCount();
	wxArrayString all_directories = current_dirname.GetDirs();

	if (number_of_directories > 1)
	{
		for (int counter = 0; counter < number_of_directories - 1; counter++)
		{
			path_string += all_directories.Item(counter);
			path_string += "/";
		}
	}

	ParentDirTextCtrl->ChangeValue(path_string);
	ProjectNameTextCtrl->ChangeValue(all_directories.Last());

	CheckProjectPath();

}

void MyNewProjectWizard::CheckProjectPath()
{

	wxFileName current_dirname = wxFileName::DirName(ProjectPathTextCtrl->GetValue());

	// the last directory is the project name..

	if (current_dirname.Exists()) ErrorText->SetLabel("Error: Directory Exists");
	else ErrorText->SetLabel("");



}

void MyNewProjectWizard::OnFinished(  wxWizardEvent& event  )
{
	// create the directory if it doesn't exist (which it shouldn't)

	wxFileName current_dirname = wxFileName::DirName(ProjectPathTextCtrl->GetValue());

	if (current_dirname.Exists())
	{
		MyDebugPrintWithDetails("Directory should not already exist, and does!\n");
	}
	else current_dirname.Mkdir();

	wxString wanted_database_file = ProjectPathTextCtrl->GetValue();
	if (wanted_database_file.EndsWith("/") == false) wanted_database_file += "/";
	wanted_database_file += ProjectNameTextCtrl->GetValue();
	wanted_database_file += ".db";

	main_frame->current_project.CreateNewProject(wanted_database_file, ProjectPathTextCtrl->GetValue(), ProjectNameTextCtrl->GetValue());

}
