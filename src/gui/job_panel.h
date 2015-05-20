#ifndef __JOB_PANEL_H__
#define __JOB_PANEL_H__

#include <wx/panel.h>
#include "wx/socket.h"


class JobPanel : public wxPanel
{
	protected:



	public:

	virtual void UpdateJobDetails(const char *updateinfo) { }
	virtual void OnJobSocketEvent(wxSocketEvent& event) {wxPrintf("JobPanel\n\n");}

		JobPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 869,566 ), long style = wxTAB_TRAVERSAL );
		~JobPanel();

};

#endif
