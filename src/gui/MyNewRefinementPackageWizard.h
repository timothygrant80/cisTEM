#ifndef __MyNewRefinmentPackageWizard__
#define __MyNewRefinmentPackageWizard__

#include "ProjectX_gui.h"

class MyNewRefinementPackageWizard;

// this is a shitty little class for holding class selections

class NewRefinementPackageWizardClassSelection {
  public:
    wxArrayBool class_selection;
};

WX_DECLARE_OBJARRAY(NewRefinementPackageWizardClassSelection, ArrayofNewRefinementPackageWizardClassSelection);

class TemplateWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    TemplateWizardPanel* my_panel;

    TemplateWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~TemplateWizardPage( );

    void RefinementPackageChanged(wxCommandEvent& event);

    wxWizardPage* GetNext( ) const;

    wxWizardPage* GetPrev( ) const { return NULL; };
};

class InputParameterWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    InputParameterWizardPanel* my_panel;

    InputParameterWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class InputTemplateMatchesPackageWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    InputTemplateMatchesPackageWizardPanel* my_panel;

    InputTemplateMatchesPackageWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class ParticleGroupWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    ParticleGroupWizardPanel* my_panel;

    ParticleGroupWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class BoxSizeWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    BoxSizeWizardPanel* my_panel;

    BoxSizeWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class NumberofClassesWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    NumberofClassesWizardPanel* my_panel;

    NumberofClassesWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~NumberofClassesWizardPage( );

    void NumberClassesChanged(wxSpinEvent& event);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class InitialReferencesWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    InitialReferenceSelectWizardPanel* my_panel;
    wxBoxSizer*                        main_sizer;

    InitialReferencesWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;

    void CreatePanel( );
};

class SymmetryWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    SymmetryWizardPanel* my_panel;

    SymmetryWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class MolecularWeightWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    MolecularWeightWizardPanel* my_panel;

    MolecularWeightWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class LargestDimensionWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    LargestDimensionWizardPanel* my_panel;

    LargestDimensionWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class ClassesSetupWizardPageA : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    ClassesSetupWizardPanelA* my_panel;

    ClassesSetupWizardPageA(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~ClassesSetupWizardPageA( );

    void CarryOverYesButtonChanged(wxCommandEvent& event);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class ClassesSetupWizardPageB : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    ClassesSetupWizardPanelB* my_panel;

    ClassesSetupWizardPageB(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~ClassesSetupWizardPageB( );

    void        ClassSelectionChanged(wxListEvent& event);
    wxArrayBool ReturnSelectedClasses( );

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class ClassesSetupWizardPageC : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    ClassesSetupWizardPanelC* my_panel;

    void OldClassListCtrlDeSelected(wxListEvent& event);
    void OldClassListCtrlSelected(wxListEvent& event);
    void NewClassSelectionChanged(wxListEvent& event);

    int  ReturnSelectedNewClass( );
    void DrawClassSelections( );
    void UpdateCurrentClassSelectionsText( );

    bool       IsAtLeastOneOldClassSelectedForEachNewClass( );
    wxArrayInt ReturnReferencesForClass(int wanted_class);

    ArrayofNewRefinementPackageWizardClassSelection current_class_selections;

    ClassesSetupWizardPageC(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~ClassesSetupWizardPageC( );

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class ClassesSetupWizardPageD : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    ClassesSetupWizardPanelD* my_panel;

    ClassesSetupWizardPageD(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class ClassesSetupWizardPageE : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    ClassesSetupWizardPanelE* my_panel;

    ClassesSetupWizardPageE(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class ClassSelectionWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    ClassSelectionWizardPanel* my_panel;

    ClassSelectionWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class RecentrePicksWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    RecentrePicksWizardPanel* my_panel;

    RecentrePicksWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~RecentrePicksWizardPage( );

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class RemoveDuplicatesWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    RemoveDuplicatesWizardPanel* my_panel;

    RemoveDuplicatesWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~RemoveDuplicatesWizardPage( );

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class RemoveDuplicateThresholdWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    RemoveDuplicateThresholdWizardPanel* my_panel;

    RemoveDuplicateThresholdWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);
    ~RemoveDuplicateThresholdWizardPage( );

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class OutputPixelSizeWizardPage : public wxWizardPage {
    MyNewRefinementPackageWizard* wizard_pointer;

  public:
    OutputPixelSizeWizardPanel* my_panel;

    OutputPixelSizeWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap = wxNullBitmap);

    wxWizardPage* GetNext( ) const;
    wxWizardPage* GetPrev( ) const;
};

class MyNewRefinementPackageWizard : public NewRefinementPackageWizard {
  public:
    MyNewRefinementPackageWizard(wxWindow* parent);
    ~MyNewRefinementPackageWizard( );

    TemplateWizardPage*                    template_page;
    InputParameterWizardPage*              parameter_page;
    InputTemplateMatchesPackageWizardPage* template_matches_page;
    ParticleGroupWizardPage*               particle_group_page;
    NumberofClassesWizardPage*             number_of_classes_page;
    BoxSizeWizardPage*                     box_size_page;
    InitialReferencesWizardPage*           initial_reference_page;
    SymmetryWizardPage*                    symmetry_page;
    MolecularWeightWizardPage*             molecular_weight_page;
    LargestDimensionWizardPage*            largest_dimension_page;
    ClassSelectionWizardPage*              class_selection_page;
    OutputPixelSizeWizardPage*             output_pixel_size_page;

    RecentrePicksWizardPage*            recentre_picks_page;
    RemoveDuplicatesWizardPage*         remove_duplicate_picks_page;
    RemoveDuplicateThresholdWizardPage* remove_duplicate_picks_threshold_page;

    ClassesSetupWizardPageA* class_setup_pageA;
    ClassesSetupWizardPageB* class_setup_pageB;
    ClassesSetupWizardPageC* class_setup_pageC;
    ClassesSetupWizardPageD* class_setup_pageD;
    ClassesSetupWizardPageE* class_setup_pageE;

    void DisableNextButton( );
    void EnableNextButton( );

    void OnFinished(wxWizardEvent& event);
    void OnUpdateUI(wxUpdateUIEvent& event);
    void PageChanging(wxWizardEvent& event);
    void PageChanged(wxWizardEvent& event);

    wxArrayInt ReturnIDsOfActiveImages(ArrayOfRefinmentPackageParticleInfos& particle_info_buffer);
};

#endif
