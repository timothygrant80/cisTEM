#ifndef __JOB_PANEL_H__
#define __JOB_PANEL_H__

#include <wx/panel.h>
#include "wx/socket.h"


class JobPanel : public wxPanel, public SocketCommunicator
{
	protected:

	JobTracker my_job_tracker;

	bool running_job;
	long my_job_id;

	public:

	virtual void HandleSocketSendJobDetails(wxSocketBase *connected_socket);
	virtual void HandleSocketIHaveAnError(wxSocketBase *connected_socket, wxString error_message);
	virtual void HandleSocketIHaveInfo(wxSocketBase *connected_socket, wxString info_message);
	virtual void HandleSocketJobFinished(wxSocketBase *connected_socket, int finished_job_number);
	virtual void HandleSocketJobResult(wxSocketBase *connected_socket, JobResult *received_result);
	virtual void HandleSocketJobResultQueue(wxSocketBase *connected_socket, ArrayofJobResults *received_queue);
	virtual void HandleSocketNumberOfConnections(wxSocketBase *connected_socket, int received_number_of_connections);
	virtual void HandleSocketAllJobsFinished(wxSocketBase *connected_socket, long received_timing_in_milliseconds);
	virtual void HandleSocketDisconnect(wxSocketBase *connected_socket);


	virtual void UpdateJobDetails(const char *updateinfo) { }
	virtual void SetNumberConnectedTextToZeroAndStartTracking();

	// make these functions pure virtual before release..

	virtual void WriteInfoText(wxString text_to_write) {}
	virtual void WriteErrorText(wxString text_to_write) {}

	virtual void OnSocketJobResultMsg(JobResult &received_result) {}
	virtual void OnSocketJobResultQueueMsg(ArrayofJobResults &received_queue) {}
	virtual void OnSocketJobFinished(int finished_job_number) {}

	virtual void SetNumberConnectedText(wxString wanted_text) {}
	virtual void SetTimeRemainingText(wxString wanted_text) {}

	virtual void OnSocketAllJobsFinished() {}

	JobPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 869,566 ), long style = wxTAB_TRAVERSAL );
	~JobPanel();

};

#endif
