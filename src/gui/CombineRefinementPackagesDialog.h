#ifndef __CombineRefinementPackagesDialog__
#define __CombineRefinementPackagesDialog__

class CombineRefinementPackagesDialog : public CombineRefinementPackagesDialogParent
{
	MyRefinementPackageAssetPanel *my_parent;

	public:



	CombineRefinementPackagesDialog( MyRefinementPackageAssetPanel *parent );

	void OnCancelClick( wxCommandEvent& event );
	void OnCombineClick( wxCommandEvent& event );
};
#endif /* __CombineRefinementPackagesDialog__ */
