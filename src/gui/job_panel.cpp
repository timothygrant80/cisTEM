#include "../core/gui_core_headers.h"

JobPanel::JobPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	brother_event_handler = this;
	my_job_id = -1;
}

JobPanel::~JobPanel()
{


}

void JobPanel::HandleSocketSendJobDetails(wxSocketBase *connected_socket)
{
	// send the job package
	WriteToSocket(connected_socket, socket_sending_job_package, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
	current_job_package.SendJobPackage(connected_socket);
}

void JobPanel::HandleSocketIHaveAnError(wxSocketBase *connected_socket, wxString error_message)
{
	WriteErrorText(error_message);
}

void JobPanel::HandleSocketIHaveInfo(wxSocketBase *connected_socket, wxString info_message)
{
	WriteInfoText(info_message);
}

void JobPanel::HandleSocketJobFinished(wxSocketBase *connected_socket, int finished_job_number)
{
	my_job_tracker.MarkJobFinished();
	OnSocketJobFinished(finished_job_number);
}

void JobPanel::HandleSocketJobResult(wxSocketBase *connected_socket, JobResult *received_result)
{
	JobResult temp_result;
	temp_result = *received_result;
	delete received_result;
	OnSocketJobResultMsg(temp_result);
}

void JobPanel::HandleSocketJobResultQueue(wxSocketBase *connected_socket, ArrayofJobResults *received_queue)
{
	ArrayofJobResults temp_queue;
	temp_queue = *received_queue;
	delete received_queue;
	OnSocketJobResultQueueMsg(temp_queue);
}

void JobPanel::HandleSocketNumberOfConnections(wxSocketBase *connected_socket, int received_number_of_connections)
{
	my_job_tracker.AddConnection();
	int total_processes = current_job_package.my_profile.ReturnTotalJobs();
	if (current_job_package.number_of_jobs  < current_job_package.my_profile.ReturnTotalJobs()) total_processes = current_job_package.number_of_jobs;
	else total_processes =  current_job_package.my_profile.ReturnTotalJobs();

	int length_of_process_number;

	if (received_number_of_connections == total_processes) WriteInfoText(wxString::Format("All %i processes are connected.", received_number_of_connections));

	if (length_of_process_number == 6) SetNumberConnectedText(wxString::Format("%6i / %6i processes connected.", received_number_of_connections, total_processes));
	else
	if (length_of_process_number == 5) SetNumberConnectedText(wxString::Format("%5i / %5i processes connected.", received_number_of_connections, total_processes));
	else
	if (length_of_process_number == 4) SetNumberConnectedText(wxString::Format("%4i / %4i processes connected.", received_number_of_connections, total_processes));
	else
	if (length_of_process_number == 3) SetNumberConnectedText(wxString::Format("%3i / %3i processes connected.", received_number_of_connections, total_processes));
	else
	if (length_of_process_number == 2) SetNumberConnectedText(wxString::Format("%2i / %2i processes connected.", received_number_of_connections, total_processes));
	else
	SetNumberConnectedText(wxString::Format("%1i / %1i processes connected.", received_number_of_connections, total_processes));
}

void JobPanel::HandleSocketAllJobsFinished(wxSocketBase *connected_socket, long received_timing_in_milliseconds)
{
	MyDebugAssertTrue(received_timing_in_milliseconds >= 0,"Oops. Got negative timing from controller: %li\n",received_timing_in_milliseconds);
	MyDebugAssertTrue(main_frame->current_project.total_cpu_hours + received_timing_in_milliseconds / 3600000.0 >= main_frame->current_project.total_cpu_hours,"Oops. Double overflow when summing hours spent on project. Total number before adding: %f. Timing from controller: %li",main_frame->current_project.total_cpu_hours,received_timing_in_milliseconds);
	main_frame->current_project.total_cpu_hours += received_timing_in_milliseconds / 3600000.0;
	MyDebugAssertTrue(main_frame->current_project.total_cpu_hours >= 0.0,"Negative total_cpu_hour: %f %li",main_frame->current_project.total_cpu_hours,received_timing_in_milliseconds);
	main_frame->current_project.total_jobs_run += my_job_tracker.total_number_of_jobs;

	// Update project statistics in the database
	main_frame->current_project.WriteProjectStatisticsToDatabase();

	// stop monitoring the socket

	StopMonitoringSocket(connected_socket);
	wxMilliSleep(300);


	OnSocketAllJobsFinished();
}

void JobPanel::SetNumberConnectedTextToZeroAndStartTracking()
{
	long number_of_refinement_processes;
	if (current_job_package.number_of_jobs < current_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = current_job_package.number_of_jobs;
	else number_of_refinement_processes =  current_job_package.my_profile.ReturnTotalJobs();

	int length_of_process_number;

	if (number_of_refinement_processes >= 100000) length_of_process_number = 6;
	else
	if (number_of_refinement_processes >= 10000) length_of_process_number = 5;
	else
	if (number_of_refinement_processes >= 1000) length_of_process_number = 4;
	else
	if (number_of_refinement_processes >= 100) length_of_process_number = 3;
	else
	if (number_of_refinement_processes >= 10) length_of_process_number = 2;
	else
	length_of_process_number = 1;

	if (length_of_process_number == 6) SetNumberConnectedText(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
	else
	if (length_of_process_number == 5) SetNumberConnectedText(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
	else
	if (length_of_process_number == 4) SetNumberConnectedText(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
	else
	if (length_of_process_number == 3) SetNumberConnectedText(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
	else
	if (length_of_process_number == 2) SetNumberConnectedText(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));
	else
	SetNumberConnectedText(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));

	SetTimeRemainingText("Time Remaining : ???h:??m:??s");
	Layout();
	running_job = true;
	my_job_tracker.StartTracking(current_job_package.number_of_jobs);
}

void JobPanel::HandleSocketDisconnect(wxSocketBase *connected_socket)
{
	WriteErrorText("Error: Controller Disconnected - shutting down job.");
	//master has disconnected..
	main_frame->job_controller.KillJobIfSocketExists(connected_socket);

}


