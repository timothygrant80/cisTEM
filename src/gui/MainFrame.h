#ifndef __MainFrame__
#define __MainFrame__

/** Implementing MainFrame */
class MyMainFrame : public MainFrame
{
		bool is_fullscreen;
	public:
		/** Constructor */
		MyMainFrame( wxWindow* parent );
		~MyMainFrame();

	//// end generated class members

		wxSocketServer *socket_server;

		wxTreeItemId tree_root;
		wxTreeItemId movie_branch;

		GuiJobController job_controller;

		wxArrayString all_my_ip_addresses;
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

		void OnExportCoordinatesToImagic ( wxCommandEvent & event );
		void OnExportToFrealign( wxCommandEvent & event );
		void OnExportToRelion( wxCommandEvent & event );

		void OnCharHook( wxKeyEvent& event );

		void OnServerEvent(wxSocketEvent& event);
		void OnSocketEvent(wxSocketEvent& event);

		void DirtyEverything();
		void DirtyMovieGroups();
		void DirtyImageGroups();
		void DirtyParticlePositionGroups();
		void DirtyRunProfiles();
		void DirtyRefinementPackages();
		void DirtyInputParameters();

		//LaunchJob(JobPanel *parent_panel, )

};

#endif // __MainFrame__
