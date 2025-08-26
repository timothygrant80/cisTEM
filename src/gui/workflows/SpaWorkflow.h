#ifndef _SRC_GUI_WORKFLOWS_SPA_WORKFLOW_H_
#define _SRC_GUI_WORKFLOWS_SPA_WORKFLOW_H_

#include "../ActionsPanelSpa.h"
#include "Icons.h"

// These are all defined in projectx.cpp; we'll use them here so that
// the panel will be fully instantiated any time the workflow changes.
extern MyAlignMoviesPanel*   align_movies_panel;
extern MyFindCTFPanel*       findctf_panel;
extern MyFindParticlesPanel* findparticles_panel;
extern MyRefine2DPanel*      classification_panel;
extern MyRefine3DPanel*      refine_3d_panel;
extern RefineCTFPanel*       refine_ctf_panel;
extern AutoRefine3DPanel*    auto_refine_3d_panel;
extern AbInitio3DPanel*      ab_initio_3d_panel;
extern Generate3DPanel*      generate_3d_panel;
extern Sharpen3DPanel*       sharpen_3d_panel;
extern wxImageList*          ActionsSpaBookIconImages;

// #include "../ResultsPanelSpa.h"

class SpaWorkflow {
    SpaWorkflow( );
};

/**
 * @brief Serves to register the single particle workflow
 * with the GUI and enable conditional rendering of panel
 * contents specific to single particle analysis.
 * 
 */
struct SpaWorkflowRegister {
    SpaWorkflowRegister( ) {
        // TODO: also add the results panel creation here
        WorkflowDefinition def;
        def.name               = "Single Particle";
        def.createActionsPanel = [](wxWindow* parent) {
            ActionsPanelSpa* actions_panel = new ActionsPanelSpa(parent);
            align_movies_panel             = new MyAlignMoviesPanel(actions_panel->ActionsBook);
            findctf_panel                  = new MyFindCTFPanel(actions_panel->ActionsBook);
            findparticles_panel            = new MyFindParticlesPanel(actions_panel->ActionsBook);
            classification_panel           = new MyRefine2DPanel(actions_panel->ActionsBook);
            refine_3d_panel                = new MyRefine3DPanel(actions_panel->ActionsBook);
            refine_ctf_panel               = new RefineCTFPanel(actions_panel->ActionsBook);
            auto_refine_3d_panel           = new AutoRefine3DPanel(actions_panel->ActionsBook);
            ab_initio_3d_panel             = new AbInitio3DPanel(actions_panel->ActionsBook);
            generate_3d_panel              = new Generate3DPanel(actions_panel->ActionsBook);
            sharpen_3d_panel               = new Sharpen3DPanel(actions_panel->ActionsBook);

            if ( ! actions_panel->ActionsBook->GetImageList( ) ) {
                actions_panel->ActionsBook->AssignImageList(GetActionsSpaBookIconImages( ));
            }

            actions_panel->ActionsBook->AddPage(align_movies_panel, "Align Movies", true, 0);
            actions_panel->ActionsBook->AddPage(findctf_panel, "Find CTF", false, 1);
            actions_panel->ActionsBook->AddPage(findparticles_panel, "Find Particles", false, 2);
            actions_panel->ActionsBook->AddPage(classification_panel, "2D Classify", false, 3);
            actions_panel->ActionsBook->AddPage(ab_initio_3d_panel, "Ab-Initio 3D", false, 4);
            actions_panel->ActionsBook->AddPage(auto_refine_3d_panel, "Auto Refine", false, 5);
            actions_panel->ActionsBook->AddPage(refine_3d_panel, "Manual Refine", false, 6);
            actions_panel->ActionsBook->AddPage(refine_ctf_panel, "Refine CTF", false, 7);
            actions_panel->ActionsBook->AddPage(generate_3d_panel, "Generate 3D", false, 8);
            actions_panel->ActionsBook->AddPage(sharpen_3d_panel, "Sharpen 3D", false, 9);

            return actions_panel;
        };
        // TODO: define a results panel function as well
        WorkflowRegistry::Instance( ).RegisterWorkflow(def);
    }
};

static SpaWorkflowRegister register_spa_workflow;

#endif