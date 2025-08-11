#ifndef _SRC_GUI_WORKFLOW_REGISTRY_H_
#define _SRC_GUI_WORKFLOW_REGISTRY_H_

#include <wx/string.h>
#include <map>
#include <vector>

/**
 * @brief Manages workflow definitions so that rendering in the GUI is more DRY between
 * workflows and centralizing the logic for different workflows.
 * 
 */
struct WorkflowDefinition {
    wxString                           name;
    std::function<wxPanel*(wxWindow*)> createActionsPanel;
    // std::function<wxPanel*(wxWindow*)> createResultsPanel;
};

class WorkflowRegistry {
  public:
    // using PanelFactory = std::function<wxPanel*(wxWindow*)>;

    inline static WorkflowRegistry& Instance( ) {
        static WorkflowRegistry instance;
        return instance;
    }

    inline void RegisterWorkflow(const WorkflowDefinition def) {
        factories[def.name] = def;
    };

    wxPanel* CreateActionsPanel(const wxString& name, wxWindow* parent) {
        return factories[name].createActionsPanel(parent);
    };

    // wxPanel* CreateResultsPanel(const wxString& name, wxWindow* parent) {
    //     return factories[name].createResultsPanel(parent);
    // }

    inline std::vector<wxString> GetWorkflowNames( ) const {
        std::vector<wxString> names;
        for ( const auto& pair : factories )
            names.push_back(pair.first);
        return names;
    };

  private:
    std::map<wxString, WorkflowDefinition> factories;
};

#endif