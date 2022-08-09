#ifndef __COMBINEREFINEMENTPACKAGESWIZARD_H_
#define __COMBINEREFINEMENTPACKAGESWIZARD_H_



class CombineRefinementPackagesWizard;

class PackageSelectionPage : public wxWizardPage
{
	CombineRefinementPackagesWizard *wizard_pointer;

public:

	PackageSelectionPanel *package_selection_panel;

	PackageSelectionPage (CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap=wxNullBitmap);
	~PackageSelectionPage ();

	wxWizardPage * GetNext () const;
	wxWizardPage * GetPrev () const {return NULL;};
};




class CombinedClassSelectionPage : public wxWizardPage
{
	CombineRefinementPackagesWizard *wizard_pointer;

public:
	CombinedClassSelectionPanel *combined_class_selection_panel;

	CombinedClassSelectionPage (CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap=wxNullBitmap);
	~CombinedClassSelectionPage ();

	wxStaticText* ClassText;

	wxWizardPage * GetNext () const;
	wxWizardPage * GetPrev () const;

};

class RefinementSelectPage : public wxWizardPage
{
	CombineRefinementPackagesWizard *wizard_pointer;

public:
	CombinedPackageRefinementPanel *combined_package_refinement_selection_panel;

	RefinementSelectPage(CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap=wxNullBitmap);
	~RefinementSelectPage();

	wxWizardPage * GetNext() const;
	wxWizardPage * GetPrev() const;
};

class VolumeSelectionPage : public wxWizardPage
{
	CombineRefinementPackagesWizard *wizard_pointer;

public:

	VolumeSelectionPage(CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap=wxNullBitmap);
	~VolumeSelectionPage();

	wxWizardPage * GetNext() const;
	wxWizardPage * GetPrev() const;

};

class CombineRefinementPackagesWizard : public CombineRefinementPackagesWizardParent
{
public:

	CombineRefinementPackagesWizard(wxWindow* parent);
	~CombineRefinementPackagesWizard();
	void OnUpdateUI (wxUpdateUIEvent& event);
	void OnCancelClick( wxWizardEvent& event );
	void OnFinished( wxWizardEvent& event );
	void DisableNextButton();
	void EnableNextButton();

	void PageChanging(wxWizardEvent& event);
	void PageChanged(wxWizardEvent& event);

	PackageSelectionPage *package_selection_page;
	CombinedClassSelectionPage *combined_class_selection_page;
	RefinementSelectPage *refinement_selection_page;

private:

	bool CheckIfDuplicate (int comparison_original_particle_position_asset_id, RefinementPackage* combined_package);
	int number_of_visits;
	int checked_counter;
	bool refinements_page_has_been_visited;
	bool classes_page_has_been_visited = false;
	bool volume_selection_page_has_been_visited = false;
	bool imported_params_found;
	bool classes_selected;
	wxArrayString refinement_names;
};


#endif
