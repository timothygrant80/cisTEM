#include "job_panel.h"

JobPanel::JobPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	  Bind(wxEVT_SOCKET, &JobPanel::OnJobSocketEvent, this);
}

JobPanel::~JobPanel()
{
	Unbind(wxEVT_SOCKET, &JobPanel::OnJobSocketEvent, this);

}
