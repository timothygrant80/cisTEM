#ifndef _SRC_GUI_WORKFLOWS_TM_WORKFLOW_H_
#define _SRC_GUI_WORKFLOWS_TM_WORKFLOW_H_

#include "../ActionsPanelTm.h"
#include "Icons.h"

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
            actions_panel                    = static_cast<ActionsPanelParent*>(actions_panel_tm);
            align_movies_panel               = new MyAlignMoviesPanel(actions_panel->ActionsBook);
            findctf_panel                    = new MyFindCTFPanel(actions_panel->ActionsBook);
            match_template_results_panel     = new MatchTemplateResultsPanel(actions_panel->ActionsBook);
            match_template_panel             = new MatchTemplatePanel(actions_panel->ActionsBook);
            refine_template_panel            = new RefineTemplatePanel(actions_panel->ActionsBook);
            generate_3d_panel                = new Generate3DPanel(actions_panel->ActionsBook);
            sharpen_3d_panel                 = new Sharpen3DPanel(actions_panel->ActionsBook);

            actions_panel->ActionsBook->AssignImageList(GetActionsTmBookIconImages( ));
            actions_panel->ActionsBook->AddPage(align_movies_panel, "Align Movies", true, 0);
            actions_panel->ActionsBook->AddPage(findctf_panel, "Find CTF", false, 1);
            actions_panel->ActionsBook->AddPage(match_template_panel, "Match Templates", false, 2);
            actions_panel->ActionsBook->AddPage(refine_template_panel, "Refine Template", false, 3);
            actions_panel->ActionsBook->AddPage(match_template_results_panel, "MT Results", false, 2);
            actions_panel->ActionsBook->AddPage(generate_3d_panel, "Generate 3D", false, 4);
            actions_panel->ActionsBook->AddPage(sharpen_3d_panel, "Sharpen 3D", false, 5);

            return actions_panel;
        };
        // TODO: define a results panel function as well
        WorkflowRegistry::Instance( ).RegisterWorkflow(def);
        // [](wxWindow* parent) { return new ResultsPanelTm(parent); }});
    }
};

static TmWorkflowRegister register_tm_workflow;
#endif