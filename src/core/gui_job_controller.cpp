#include "core_headers.h"
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
}

GuiJob::GuiJob(JobPanel *wanted_parent_panel)
{
	is_active = false;
	socket = NULL;
	parent_panel = wanted_parent_panel;

}

long GuiJobController::AddJob(JobPanel *wanted_parent_panel)
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

	GenerateJobCode(job_list[new_index].job_code);

	// Launch the job..

	LaunchJob(job_list[new_index].job_code);

	// send back the index

	return new_index;

}

void GuiJobController::LaunchJob(unsigned char *job_code)
{
	long counter;

	wxString execution_command;
	execution_command = "guix_job_control " + main_frame->my_ip_address + " " + main_frame->my_port_string + " ";


	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		execution_command += job_code[counter];
	}


	MyDebugPrint("Launching %s\n", execution_command);

	wxExecute(execution_command);

}

void GuiJobController::GenerateJobCode(unsigned char *job_code)
{
	srand (time(NULL));

	long counter;
	unsigned char temp_code[SOCKET_CODE_SIZE];

	for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
	{
		temp_code[counter] = (unsigned char) (rand() % 74 + 48);
		if (temp_code[counter] == 92) temp_code[counter] = 123; // slashes screw things up
	}

	// this is extremely unlikely, but just in case the job code is already being used, we do this check..

	while (ReturnJobNumberFromJobCode(temp_code) != -1)
	{
		for (counter = 0; counter < SOCKET_CODE_SIZE; counter++)
		{
			temp_code[counter] = (unsigned char) (rand() % 74 + 48);
			if (temp_code[counter] == 92) temp_code[counter] = 123; // slashes screw things up
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
