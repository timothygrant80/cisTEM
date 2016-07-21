//#include "core_headers.h"
#include "gui_core_headers.h"

extern MyMainFrame *main_frame;

GuiJobController::GuiJobController()
{
	job_index_tracker = 0;
	number_of_running_jobs = 0;
}

GuiJob::GuiJob()
{
	is_active = false;
	socket = NULL;
	parent_panel = NULL;
	launch_command = "guix_job_control";
}

GuiJob::GuiJob(JobPanel *wanted_parent_panel)
{
	is_active = false;
	socket = NULL;
	parent_panel = wanted_parent_panel;
	launch_command = "guix_job_control";

}

long GuiJobController::AddJob(JobPanel *wanted_parent_panel, wxString wanted_launch_command, wxString wanted_gui_address)
{
	// check if there is a free slot

	long new_index = FindFreeJobSlot();

	if (new_index == -1)
	{
		wxPrintf("\n\nTerminal Error: No free job slots\n\n");
		abort();
	}

	// set the job_info

	job_list[new_index].is_active = true;
	job_list[new_index].parent_panel = wanted_parent_panel;
	job_list[new_index].launch_command = wanted_launch_command;
	job_list[new_index].gui_address = wanted_gui_address;

	GenerateJobCode(job_list[new_index].job_code);

	// Launch the job..

	if (LaunchJob(&job_list[new_index]) == true) return new_index;
	else return -1;

}

bool GuiJobController::LaunchJob(GuiJob *job_to_launch)
{
	long counter;

	wxString execution_command;
	wxString executable;

	if (job_to_launch->gui_address == "")
	{
		executable = "guix_job_control " + main_frame->my_ip_address + " " + main_frame->my_port_string + " ";
	}
	else
	{
		executable = "guix_job_control " + job_to_launch->gui_address + " " + main_frame->my_port_string + " ";
	}


	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		executable += job_to_launch->job_code[counter];
	}

	execution_command = job_to_launch->launch_command;
	execution_command.Replace("$command", executable);
	execution_command += "&";

	job_to_launch->parent_panel->WriteInfoText("Launching Job...\n(" + execution_command + ")\n");

	/*
	if (wxExecute(execution_command) == -1)
	{
		job_to_launch->parent_panel->WriteErrorText("Failed: Error launching job!\n\n");
		return false;
	}
	else
	{
		job_to_launch->parent_panel->WriteInfoText("Success: Job launched!\n\n");
	}*/

	system(execution_command.ToUTF8().data());

	return true;
}

void GuiJobController::GenerateJobCode(unsigned char *job_code)
{
	srand (time(NULL));

	long counter;
	unsigned char temp_code[SOCKET_CODE_SIZE];

	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		temp_code[counter] = (unsigned char) (rand() % 9 + 48);
		//if (temp_code[counter] == 92) temp_code[counter] = 123; // slashes screw things up
	}

	// this is extremely unlikely, but just in case the job code is already being used, we do this check..

	while (ReturnJobNumberFromJobCode(temp_code) != -1)
	{
		for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
		{
			temp_code[counter] = (unsigned char) (rand() % 9 + 48);
			//if (temp_code[counter] == 92) temp_code[counter] = 123; // slashes screw things up
			//if (temp_code[counter] == 96) temp_code[counter] = 125;
		}
	}

	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		job_code[counter] = temp_code[counter];
	}

}


long GuiJobController::FindFreeJobSlot()
{
	long found_index = -1;
	long counter;

	for (counter = 0; counter < MAX_GUI_JOBS; counter++)
	{
		if (job_list[counter].is_active == false)
		{
			found_index = counter;
			break;
		}
	}

	return found_index;

}

long GuiJobController::ReturnJobNumberFromJobCode(unsigned char *job_code)
{
	long counter;

	for (counter = 0; counter < MAX_GUI_JOBS; counter++)
	{
		if (memcmp(job_list[counter].job_code, job_code, SOCKET_CODE_SIZE) == 0 && job_list[counter].is_active == true)
		{
			return counter;
		}
	}

	return -1;
}

void GuiJobController::KillJob(int job_to_kill)
{
	SETUP_SOCKET_CODES

	if (job_list[job_to_kill].is_active == true)
	{
		if (job_list[job_to_kill].socket != NULL)
		{
			job_list[job_to_kill].socket->Notify(false);
			job_list[job_to_kill].socket->WriteMsg(socket_time_to_die, SOCKET_CODE_SIZE);
			job_list[job_to_kill].socket->Destroy();
		}
		job_list[job_to_kill].socket = NULL;
		job_list[job_to_kill].is_active = false;
	}
}
