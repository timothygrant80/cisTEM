#ifndef __MyNewRefinmentPackageWizard__
#define __MyNewRefinmentPackageWizard__

#include "ProjectX_gui.h"

class MyNewRefinementPackageWizard;

class TemplateWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	TemplateWizardPanel *my_panel;

 	 TemplateWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);
 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const {return NULL;};

};

class InputParameterWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	InputParameterWizardPanel *my_panel;

 	 InputParameterWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);
 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;

};

class ParticleGroupWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	ParticleGroupWizardPanel *my_panel;

 	 ParticleGroupWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;

};

class BoxSizeWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	BoxSizeWizardPanel *my_panel;

	BoxSizeWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;

};

class NumberofClassesWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	NumberofClassesWizardPanel *my_panel;

	NumberofClassesWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;

};

class InitialReferencesWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	InitialReferenceSelectWizardPanel *my_panel;

	InitialReferencesWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;

	 void CreatePanel();

};

class SymmetryWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	SymmetryWizardPanel *my_panel;

	SymmetryWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;
};

class MolecularWeightWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	MolecularWeightWizardPanel *my_panel;

	MolecularWeightWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;
};

class LargestDimensionWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	LargestDimensionWizardPanel *my_panel;

	LargestDimensionWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;
};



class ClassesSetupWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;

	public:

	ClassesSetupWizardPanel *my_panel;

	ClassesSetupWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;

};

class ClassSelectionWizardPage : public wxWizardPage
{
	MyNewRefinementPackageWizard *wizard_pointer;


	public:

	ClassSelectionWizardPanel *my_panel;

	ClassSelectionWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap=wxNullBitmap);

 	 wxWizardPage * GetNext () const;
	 wxWizardPage * GetPrev () const;

};

class MyNewRefinementPackageWizard : public NewRefinementPackageWizard
{
public:

		MyNewRefinementPackageWizard( wxWindow* parent );
		~MyNewRefinementPackageWizard();

		TemplateWizardPage *template_page;
		InputParameterWizardPage *parameter_page;
		ParticleGroupWizardPage *particle_group_page;
		NumberofClassesWizardPage *number_of_classes_page;
		BoxSizeWizardPage *box_size_page;
		ClassesSetupWizardPage *class_setup_page;
		InitialReferencesWizardPage *initial_reference_page;
		SymmetryWizardPage *symmetry_page;
		MolecularWeightWizardPage *molecular_weight_page;
		LargestDimensionWizardPage *largest_dimension_page;
		ClassSelectionWizardPage *class_selection_page;

		void DisableNextButton();
		void EnableNextButton();

		void OnFinished( wxWizardEvent& event );
		void OnUpdateUI(wxUpdateUIEvent& event);
		void PageChanging(wxWizardEvent& event);
		void PageChanged(wxWizardEvent& event);
	
};


#endif
