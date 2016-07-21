//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyAddRunCommandDialog::MyAddRunCommandDialog( MyRunProfilesPanel *parent)
:
AddRunCommandDialog( parent )
{
	my_parent = parent;

}

void MyAddRunCommandDialog::OnOKClick( wxCommandEvent& event )
{
	ProcessResult();
}

void MyAddRunCommandDialog::OnCancelClick( wxCommandEvent& event )
{
	Destroy();
}

void MyAddRunCommandDialog::OnEnter( wxCommandEvent& event )
{
	ProcessResult();
}

void MyAddRunCommandDialog::ProcessResult()
{

	// check that the text box contains $command..

	if (CommandTextCtrl->GetValue().Find("$command") ==  wxNOT_FOUND)
	{
		ErrorStaticText->Show(true);
		Layout();
		Fit();
	}
	else
	{
		if (my_parent->buffer_profile.run_commands[my_parent->selected_command].command_to_run == CommandTextCtrl->GetValue() && my_parent->buffer_profile.run_commands[my_parent->selected_command].number_of_copies == NumberCopiesSpinCtrl->GetValue() && my_parent->buffer_profile.run_commands[my_parent->selected_command].delay_time_in_ms == DelayTimeSpinCtrl->GetValue())
		{
			// nothing changed just exit..

			Destroy();
		}
		else
		{
			//Update and exit..

			my_parent->buffer_profile.run_commands[my_parent->selected_command].command_to_run = CommandTextCtrl->GetValue();
			my_parent->buffer_profile.run_commands[my_parent->selected_command].number_of_copies = NumberCopiesSpinCtrl->GetValue();
			my_parent->buffer_profile.run_commands[my_parent->selected_command].delay_time_in_ms = DelayTimeSpinCtrl->GetValue();
			my_parent->FillCommandsBox();
			my_parent->command_panel_has_changed = true;
			Destroy();
		}
	}
}
