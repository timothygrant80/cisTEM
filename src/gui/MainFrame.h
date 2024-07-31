#ifndef _gui_MainFrame_h_
#define _gui_MainFrame_h_

/** Implementing MainFrame */
class MyMainFrame : public MainFrame, public SocketCommunicator {
    bool                   is_fullscreen;
    cistem::workflow::Enum current_workflow;
    cistem::workflow::Enum previous_workflow;

  public:
    /** Constructor */
    MyMainFrame(wxWindow* parent);
    ~MyMainFrame( );

    //// end generated class members

    wxTreeItemId tree_root;
    wxTreeItemId movie_branch;

    GuiJobController job_controller;

    wxArrayString all_my_ip_addresses;
    wxString      my_port_string;

    Project current_project;

    short int my_port;

    virtual wxString ReturnName( ) { return "MainFrame"; }

    void RecalculateAssetBrowser(void);
    void OnCollapseAll(wxCommandEvent& event);
    void OnMenuBookChange(wxBookCtrlEvent& event);

    void OnFileNewProject(wxCommandEvent& event);
    void OnFileOpenProject(wxCommandEvent& event);
    void OnFileExit(wxCommandEvent& event);
    void OnFileCloseProject(wxCommandEvent& event);
    void OnFileMenuUpdate(wxUpdateUIEvent& event);

    void OnHelpLaunch(wxCommandEvent& event);
    void OnAboutLaunch(wxCommandEvent& event);

    void OnExportCoordinatesToImagic(wxCommandEvent& event);
    void OnExportToFrealign(wxCommandEvent& event);
    void OnExportToRelion(wxCommandEvent& event);

    void OpenProject(wxString project_filename);
    void GetFileAndOpenProject( );
    void StartNewProject( );

    void OnCharHook(wxKeyEvent& event);

    //	void OnServerEvent(wxSocketEvent& event);
    //	void OnSocketEvent(wxSocketEvent& event);

    // Socket Handling overrides..

    void HandleNewSocketConnection(wxSocketBase* new_connection, unsigned char* identification_code);

    // end socket

    void DirtyEverything( );
    void DirtyMovieGroups( );
    void DirtyImageGroups( );
    void DirtyVolumes( );
#ifdef EXPERIMENTAL
    void DirtyAtomicCoordinates( );
#endif
    void DirtyParticlePositionGroups( );
    void DirtyRunProfiles( );
    void DirtyRefinementPackages( );
    void DirtyRefinements( );
    void DirtyClassificationSelections( );
    void DirtyClassifications( );
    void DirtyTemplateMatchesPackages( );
    void ResetAllPanels( );

    void ClearScratchDirectory( );
    void ClearStartupScratch( );
    void ClearRefine2DScratch( );
    void ClearRefine3DScratch( );
    void ClearAutoRefine3DScratch( );
    void ClearGenerate3DScratch( );
    void ClearRefineCTFScratch( );

    wxString ReturnScratchDirectory( );
    wxString ReturnStartupScratchDirectory( );
    wxString ReturnRefine2DScratchDirectory( );
    wxString ReturnRefine3DScratchDirectory( );
    wxString ReturnAutoRefine3DScratchDirectory( );
    wxString ReturnGenerate3DScratchDirectory( );
    wxString ReturnRefineCTFScratchDirectory( );

    bool MigrateProject(wxString old_project_directory, wxString new_project_directory);

    template <class FrameTypeFrom, class FrameTypeTo>
    void UpdateWorkflow(FrameTypeFrom* input_frame, FrameTypeTo* output_frame, wxString frame_name);

    void SetSingleParticleWorkflow(bool triggered_by_gui_event = false);
    void SetTemplateMatchingWorkflow(bool triggered_by_gui_event = false);

    void OnSingleParticleWorkflow(wxCommandEvent& event);
    void OnTemplateMatchingWorkflow(wxCommandEvent& event);

    inline cistem::workflow::Enum ReturnCurrentWorkflow( ) { return current_workflow; };

    inline cistem::workflow::Enum ReturnPreviousWorkflow( ) { return previous_workflow; };

    inline void ManuallyUpdateWorkflowMenuCheckBox( ) {

        if ( current_workflow == cistem::workflow::single_particle ) {
            WorkflowMenu->Check(WorkflowSingleParticle->GetId( ), true);
        }

        if ( current_workflow == cistem::workflow::template_matching ) {
            WorkflowMenu->Check(WorkflowTemplateMatching->GetId( ), true);
        }
    }

    //LaunchJob(JobPanel *parent_panel, )
};

#endif // _gui_MainFrame_h_
