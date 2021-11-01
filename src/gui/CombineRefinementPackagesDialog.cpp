//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"


CombineRefinementPackagesDialog::CombineRefinementPackagesDialog(MyRefinementPackageAssetPanel *parent)
:
CombineRefinementPackagesDialogParent( parent )
{
	my_parent = parent;


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


