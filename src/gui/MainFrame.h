#ifndef __MainFrame__
#define __MainFrame__

/** Implementing MainFrame */
class MyMainFrame : public MainFrame
{
	public:
		/** Constructor */
		MyMainFrame( wxWindow* parent );
		~MyMainFrame();

	//// end generated class members

		wxSocketServer *socket_server;

		wxTreeItemId tree_root;
		wxTreeItemId movie_branch;

		GuiJobController job_controller;

		wxString my_ip_address;
		wxString my_port_string;

		Project current_project;

		short int my_port;

		void SetupServer();
		void RecalculateAssetBrowser(void);
		void OnCollapseAll( wxCommandEvent& event );
		void OnMenuBookChange( wxBookCtrlEvent& event );

		void OnFileNewProject( wxCommandEvent& event );
		void OnFileOpenProject( wxCommandEvent& event );
		void OnFileExit( wxCommandEvent& event );
		void OnFileCloseProject( wxCommandEvent& event );
		void OnFileMenuUpdate( wxUpdateUIEvent& event );

		void OnServerEvent(wxSocketEvent& event);
		void OnSocketEvent(wxSocketEvent& event);

		void DirtyEverything();
		void DirtyMovieGroups();
		void DirtyImageGroups();
		void DirtyParticlePositionGroups();
		void DirtyRunProfiles();

		//LaunchJob(JobPanel *parent_panel, )

};

#endif // __MainFrame__
