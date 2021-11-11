//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;

CombineRefinementPackagesDialog::CombineRefinementPackagesDialog(MyRefinementPackageAssetPanel *parent)
:
CombineRefinementPackagesDialogParent( parent )
{
	my_parent = parent;

	wxArrayString refinement_name;

	for (int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	{
		refinement_name.Add(refinement_package_asset_panel->all_refinement_packages[counter].name);
	}
	RefinementPackagesListCtrl->InsertItems(refinement_name, 0);

}
void CombineRefinementPackagesDialog::OnCancelClick( wxCommandEvent& event )
{
	EndModal(0);
	Destroy();
}

void CombineRefinementPackagesDialog::OnCombineClick( wxCommandEvent& event )
{
	EndModal(0);
	Destroy();
}
