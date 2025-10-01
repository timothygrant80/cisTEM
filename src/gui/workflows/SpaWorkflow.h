#ifndef _SRC_GUI_WORKFLOWS_SPA_WORKFLOW_H_
#define _SRC_GUI_WORKFLOWS_SPA_WORKFLOW_H_

#include "../ActionsPanelSpa.h"
#include "Icons.h"

// GLOBAL PANEL POINTERS: These are all defined in projectx.cpp and used throughout the application.
//
// IMPORTANT LIFECYCLE MANAGEMENT:
// - These pointers are shared globally across the entire application
// - They are created when a workflow is activated (see createActionsPanel lambda below)
// - They are destroyed when switching workflows (handled by ActionsPanelSpa destructor)
// - The destructor MUST set these to nullptr to prevent dangling pointer access
//
// WHY GLOBALS?
// - Historical design: The application was originally single-workflow
// - Many parts of the codebase expect direct access to these panels
// - Refactoring to eliminate globals would require extensive changes
//
// SAFETY PROTOCOL:
// 1. Create panels in workflow registration (below)
// 2. Destroy panels when switching workflows (automatic via wxWidgets)
// 3. Nullify pointers in destructor (prevents segfaults)
// 4. Check for null before access (in Dirty*() methods and elsewhere)
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

            // PANEL CREATION: Create all workflow-specific panels as children of ActionsBook.
            // These panels will be automatically destroyed when actions_panel is destroyed.
            // The ActionsPanelSpa destructor will handle nullifying the global pointers.
            align_movies_panel   = new MyAlignMoviesPanel(actions_panel->ActionsBook);
            findctf_panel        = new MyFindCTFPanel(actions_panel->ActionsBook);
            findparticles_panel  = new MyFindParticlesPanel(actions_panel->ActionsBook);
            classification_panel = new MyRefine2DPanel(actions_panel->ActionsBook);
            refine_3d_panel      = new MyRefine3DPanel(actions_panel->ActionsBook);
            refine_ctf_panel     = new RefineCTFPanel(actions_panel->ActionsBook);
            auto_refine_3d_panel = new AutoRefine3DPanel(actions_panel->ActionsBook);
            ab_initio_3d_panel   = new AbInitio3DPanel(actions_panel->ActionsBook);
            generate_3d_panel    = new Generate3DPanel(actions_panel->ActionsBook);
            sharpen_3d_panel     = new Sharpen3DPanel(actions_panel->ActionsBook);

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