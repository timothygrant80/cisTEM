#ifndef __MainFrame__
#define __MainFrame__

/** Implementing MainFrame */
class MyMainFrame : public MainFrame
{
	public:
		/** Constructor */
		MyMainFrame( wxWindow* parent );
	//// end generated class members

		wxSocketServer *socket_server;

		wxTreeItemId tree_root;
		wxTreeItemId movie_branch;

		GuiJobController job_controller;

		wxString my_ip_address;
		wxString my_port_string;

		short int my_port;

		void SetupServer();
		void RecalculateAssetBrowser(void);
		void OnCollapseAll( wxCommandEvent& event );
		void OnMenuBookChange( wxListbookEvent& event );

		void OnServerEvent(wxSocketEvent& event);
		void OnSocketEvent(wxSocketEvent& event);
		//LaunchJob(JobPanel *parent_panel, )

};

#endif // __MainFrame__
