#ifndef __CombineRefinementPackagesDialog__
#define __CombineRefinementPackagesDialog__

#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/listbox.h>

class CombineRefinementPackagesDialog : public CombineRefinementPackagesDialogParent
{
	MyRefinementPackageAssetPanel *my_parent;

	public:

	wxArrayString refinement_name;

	CombineRefinementPackagesDialog( MyRefinementPackageAssetPanel *parent );

	void OnCancelClick( wxCommandEvent& event );
	void OnCombineClick( wxCommandEvent& event );

};
#endif /* __CombineRefinementPackagesDialog__ */
