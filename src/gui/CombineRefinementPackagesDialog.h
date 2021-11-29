#ifndef __CombineRefinementPackagesDialog__
#define __CombineRefinementPackagesDialog__

#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/listbox.h>
#include <wx/checklst.h>

class CombineRefinementPackagesDialog : public CombineRefinementPackagesDialogParent
{
	public:

	wxArrayString refinement_name;
	wxArrayString packages_to_combine_filenames;

	CombineRefinementPackagesDialog( MyRefinementPackageAssetPanel *parent );

	void OnCancelClick( wxCommandEvent& event );
	void OnCombineClick( wxCommandEvent& event );

};
#endif /* __CombineRefinementPackagesDialog__ */
