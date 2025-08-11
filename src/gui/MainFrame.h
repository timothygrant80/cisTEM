#ifndef _gui_MainFrame_h_
#define _gui_MainFrame_h_
#include "UpdateProgressTracker.h"
#include "workflows/WorkflowRegistry.h"
#include <memory>

/** Implementing MainFrame */
class MyMainFrame : public MainFrame, public SocketCommunicator, public UpdateProgressTracker {
    bool is_fullscreen;
    // cistem::workflow::Enum current_workflow;
    wxString               current_workflow = "";
    cistem::workflow::Enum previous_workflow;

  public:
    /** Constructor */
    MyMainFrame(wxWindow* parent);
    ~MyMainFrame( );

    std::shared_ptr<WorkflowRegistry( )> registry;

    //// end generated class members

    wxTreeItemId tree_root;
    wxTreeItemId movie_branch;

    GuiJobController job_controller;

    wxArrayString all_my_ip_addresses;
    wxString      my_port_string;

    Project current_project;

    short int my_port;

    virtual wxString ReturnName( ) { return "MainFrame"; }

    // For schema update only; could be more generalized/re-used when needed; overrides from UpdateProgressTracker
    void OnUpdateProgress(int progress, wxString new_msg, bool& should_update_text) override;
    void OnCompletion( ) override;

    void RecalculateAssetBrowser(void);
    void OnCollapseAll(wxCommandEvent& event);
    void OnMenuBookChange(wxBookCtrlEvent& event);

    void OnFileNewProject(wxCommandEvent& event);
    void OnFileOpenProject(wxCommandEvent& event);
    void OnFileExit(wxCommandEvent& event);
    void OnFileCloseProject(wxCommandEvent& event);
    void OnFileMenuUpdate(wxUpdateUIEvent& event);
    void OnWorkflowMenuSelection(wxCommandEvent& event);

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
    void DirtyAtomicCoordinates( );
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
    void SetActionsPanelChild(wxWindow* panel);
    void SwitchWorkflowPanels(const wxString& workflow_name);

    // void OnSingleParticleWorkflow(wxCommandEvent& event);
    // void OnTemplateMatchingWorkflow(wxCommandEvent& event);

    inline wxString ReturnCurrentWorkflow( ) {
        for ( int workflow_option_counter = 0; workflow_option_counter < 0; workflow_option_counter++ ) {
            if ( WorkflowMenu->IsChecked(workflow_option_counter) ) {
                current_workflow = WorkflowMenu->GetLabel(workflow_option_counter);
            }
        }

        // Should not get here, but in case we do, just assume SPA
        if ( current_workflow.IsEmpty( ) )
            current_workflow = "Single Particle";
        if ( current_project.database.is_open )
            current_project.RecordCurrentWorkflowInDB(current_workflow);
        return current_workflow;
    };

    inline cistem::workflow::Enum ReturnPreviousWorkflow( ) { return previous_workflow; };

    inline void ManuallyUpdateWorkflowMenuCheckBox( ) {

        // if ( current_workflow == cistem::workflow::single_particle ) {
        // WorkflowMenu->Check(WorkflowSingleParticle->GetId( ), true);
        // }
        //
        // if ( current_workflow == cistem::workflow::template_matching ) {
        // WorkflowMenu->Check(WorkflowTemplateMatching->GetId( ), true);
        // }
        if ( current_workflow.Cmp("Single Particle") ) {
            WorkflowMenu->Check(WorkflowMenu->FindItem("Single Particle"), true);
            WorkflowMenu->Check(WorkflowMenu->FindItem("Template Matching"), false);
        }
        else if ( current_workflow.Cmp("Template Matching") ) {
            WorkflowMenu->Check(WorkflowMenu->FindItem("Template Matching"), true);
            WorkflowMenu->Check(WorkflowMenu->FindItem("Single Particle"), false);
        }
    }

    //LaunchJob(JobPanel *parent_panel, )

  private:
    // Only used in schema update at this point
    OneSecondProgressDialog* update_progress_dialog;
    void                     UpdateDatabase(std::pair<Database::TableChanges, Database::ColumnChanges>& schema_comparison);
    ActionsPanelParent*      actions_panel_child;
};

#endif // _gui_MainFrame_h_
