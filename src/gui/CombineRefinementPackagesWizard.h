#ifndef __COMBINEREFINEMENTPACKAGESWIZARD_H_
#define __COMBINEREFINEMENTPACKAGESWIZARD_H_


#include "ProjectX_gui.h"
#include "my_controls.h"

class CombineRefinementPackagesWizard; //: public CombineRefinementPackagesWizardParent
//{
/*public:

	wxArrayString refinement_name;
	wxArrayString packages_to_combine_filenames;

	CombineRefinementPackagesWizard ( MyRefinementPackageAssetPanel *parent );

	void OnCancelClick( wxCommandEvent& event );
	void OnFinished( wxCommandEvent& event );
	void ImportedPackageCheck(wxUpdateUIEvent& event);

	private:
	//int binarySearch (RefinementPackage* combined_package, long lower_bound, long upper_bound, long original_asset_particle_position);
	//void AddParticleAt (RefinementPackage* refinement_package, RefinementPackageParticleInfo particle_to_add, int index, long size);
	bool CheckIfDuplicate(int comparison_original_particle_position_asset_id, RefinementPackage* combined_package);


};*/

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

class CombinedPackageItemPanel : public CombinedPackageItemPicker
{
	public:
	CombinedPackageItemPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~CombinedPackageItemPanel();
	void FillClassSelectionBox(int package_number);
	void FillRefinementSelectionBox();

	private:
	bool has_random_parameters;

};


class CombinedClassSelectionPage : public wxWizardPage
{
	CombineRefinementPackagesWizard *wizard_pointer;

public:
	CombinedClassSelectionPanel *combined_class_selection_panel;

	CombinedClassSelectionPage (CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap=wxNullBitmap);
	~CombinedClassSelectionPage ();

	wxWizardPage * GetNext () const; // this will have to be changed
	wxWizardPage * GetPrev () const;
};

class ItemSelectPanel : public wxPanel
{
public:

	CombinedPackageItemPanel* ItemSelectBoxPanel;
	wxStaticText* ClassText;
	wxBoxSizer* MainSizer;
	wxBoxSizer* bSizer989;

	ItemSelectPanel (wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~ItemSelectPanel();

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
	RefinementSelectPage *refinement_select_page;

private:

	bool CheckIfDuplicate (int comparison_original_particle_position_asset_id, RefinementPackage* combined_package);
	int number_of_visits;
	int checked_counter;
	int selected_refinement_id;
	bool refinements_page_has_been_visited;
	bool imported_params_found;
	bool classes_selected;
	wxArrayString refinement_names;
};
#endif
