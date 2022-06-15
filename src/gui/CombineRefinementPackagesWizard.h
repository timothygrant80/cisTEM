#ifndef __COMBINEREFINEMENTPACKAGESWIZARD_H_
#define __COMBINEREFINEMENTPACKAGESWIZARD_H_


#include "ProjectX_gui.h"
#include "AssetPickerComboPanel.h"

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

class CombinedPackageClassPanel : public CombinedPackageClassPicker
{
	public:
	CombinedPackageClassPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~CombinedPackageClassPanel();
	void FillSelectionBox(int package_number);
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

class ClassSelectPanel : public wxPanel
{
public:

	CombinedPackageClassPanel* ClassSelectBoxPanel;
	wxStaticText* ClassText;
	wxBoxSizer* MainSizer;
	wxBoxSizer* bSizer989;

	ClassSelectPanel (wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
	~ClassSelectPanel();

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

private:

	bool CheckIfDuplicate (int comparison_original_particle_position_asset_id, RefinementPackage* combined_package);
	int number_of_visits;
	int checked_counter;
};
#endif
