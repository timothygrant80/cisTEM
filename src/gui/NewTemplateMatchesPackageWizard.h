#ifndef __NewTemplateMatchesPackageWizard__
#define __NewTemplateMatchesPackageWizard__

#include "ProjectX_gui.h"

class NewTemplateMatchesPackageWizard;

class TemplateMatchesWizardPage : public wxWizardPage {
    NewTemplateMatchesPackageWizard* wizard_pointer;

  public:
    TemplateMatchesWizardPanel* my_panel;

    TemplateMatchesWizardPage(NewTemplateMatchesPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~TemplateMatchesWizardPage( );
    void SelectionChanged(wxCommandEvent& event);

    wxWizardPage* GetNext( ) const { return NULL; };

    wxWizardPage* GetPrev( ) const { return NULL; };
};

class NewTemplateMatchesPackageWizard : public NewTemplateMatchesPackageWizardParent {
  public:
    NewTemplateMatchesPackageWizard(wxWindow* parent);
    ~NewTemplateMatchesPackageWizard( );

    TemplateMatchesWizardPage* page;

    void DisableNextButton( );
    void EnableNextButton( );

    void OnFinished(wxWizardEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);
    void PageChanging(wxWizardEvent& event);
    void PageChanged(wxWizardEvent& event);

    long       num_selected_matches;
    wxArrayInt tm_ids;
    wxArrayInt image_ids;
};

#endif
