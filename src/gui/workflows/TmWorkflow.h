#ifndef _SRC_GUI_WORKFLOWS_TM_WORKFLOW_H_
#define _SRC_GUI_WORKFLOWS_TM_WORKFLOW_H_

#include "../ActionsPanelTm.h"
#include "Icons.h"

// GLOBAL PANEL POINTERS: Similar to SpaWorkflow.h, these globals are managed carefully.
//
// TEMPLATE MATCHING SPECIFIC PANELS:
// - match_template_panel: The main template matching configuration panel
// - refine_template_panel: For refining template matches
// - match_template_results_panel: For displaying match results
//
// SHARED PANELS (also used in Single Particle):
// - align_movies_panel, findctf_panel: Pre-processing panels
// - generate_3d_panel, sharpen_3d_panel: 3D reconstruction panels
//
// CRITICAL: ActionsPanelTm destructor must nullify only the panels it creates.
// Some panels might be shared or managed elsewhere, so we only clean up what we own.

extern ActionsPanelParent*        actions_panel;
extern MyAlignMoviesPanel*        align_movies_panel;
extern MyFindCTFPanel*            findctf_panel;
extern MatchTemplatePanel*        match_template_panel;
extern MatchTemplateResultsPanel* match_template_results_panel;
extern RefineTemplatePanel*       refine_template_panel;
extern Generate3DPanel*           generate_3d_panel;
extern Sharpen3DPanel*            sharpen_3d_panel;
extern wxImageList*               ActionsTmBookIconImages;

// #include "../ResultsPanelTm.h"

class TmWorkflow {
    TmWorkflow( );
};

/**
 * @brief Serves to register the template matching workflow
 * with the GUI and enable conditional rendering of panel
 * contents specific to template matching.
 * 
 */
struct TmWorkflowRegister {
    TmWorkflowRegister( ) {
        WorkflowDefinition def;
        def.name               = "Template Matching";
        def.createActionsPanel = [](wxWindow* parent) {
            ActionsPanelTm* actions_panel_tm = new ActionsPanelTm(parent);
            // Don't set the global actions_panel here - it will be set by the caller

            // PANEL CREATION: Create Template Matching specific panels.
            // Note: These replace any existing Single Particle panels with the same names.
            // The old panels are destroyed first (handled by ActionsPanelSpa destructor if coming from SPA).
            // ActionsPanelTm destructor will nullify these pointers when switching away from TM.
            align_movies_panel = new MyAlignMoviesPanel(actions_panel_tm->ActionsBook);
            findctf_panel = new MyFindCTFPanel(actions_panel_tm->ActionsBook);
            match_template_panel = new MatchTemplatePanel(actions_panel_tm->ActionsBook);
            refine_template_panel = new RefineTemplatePanel(actions_panel_tm->ActionsBook);
            generate_3d_panel = new Generate3DPanel(actions_panel_tm->ActionsBook);
            sharpen_3d_panel = new Sharpen3DPanel(actions_panel_tm->ActionsBook);

            if ( ! actions_panel_tm->ActionsBook->GetImageList( ) ) {
                actions_panel_tm->ActionsBook->AssignImageList(GetActionsTmBookIconImages( ));
            }

            actions_panel_tm->ActionsBook->AddPage(align_movies_panel, "Align Movies", true, 0);
            actions_panel_tm->ActionsBook->AddPage(findctf_panel, "Find CTF", false, 1);
            actions_panel_tm->ActionsBook->AddPage(match_template_panel, "Match Templates", false, 2);
            actions_panel_tm->ActionsBook->AddPage(refine_template_panel, "Refine Template", false, 3);
            actions_panel_tm->ActionsBook->AddPage(generate_3d_panel, "Generate 3D", false, 4);
            actions_panel_tm->ActionsBook->AddPage(sharpen_3d_panel, "Sharpen 3D", false, 5);

            return actions_panel_tm;
        };
        // TODO: define a results panel function as well
        WorkflowRegistry::Instance( ).RegisterWorkflow(def);
        // [](wxWindow* parent) { return new ResultsPanelTm(parent); }});
    }
};

static TmWorkflowRegister register_tm_workflow;
#endif