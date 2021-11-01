#ifndef __MyRefinementPackageAssetPanel__
#define __MyRefinementPackageAssetPanel__


class MyRefinementPackageAssetPanel : public RefinementPackageAssetPanel
{
	public:

		bool is_dirty;
		long current_asset_number;
		long selected_refinement_package;
		bool should_veto_motion;

		MyRefinementPackageAssetPanel( wxWindow* parent );
		ArrayOfRefinementPackages all_refinement_packages;

		long ReturnArrayPositionFromAssetID(long wanted_asset_id);
		void OnCreateClick( wxCommandEvent& event );
		void OnRenameClick( wxCommandEvent& event );
		void OnDeleteClick( wxCommandEvent& event );
		void OnImportClick( wxCommandEvent& event );
		void OnExportClick( wxCommandEvent& event );
		void OnCombineClick( wxCommandEvent& event);
		void OnUpdateUI(wxUpdateUIEvent& event);
		void OnDisplayStackButton( wxCommandEvent& event );

		void AddAsset(RefinementPackage *refinement_package );

		void Reset();

		void FillRefinementPackages();

		void ImportAllFromDatabase();

		// mouse vetos

		void MouseVeto( wxMouseEvent& event );
		void MouseCheckPackagesVeto( wxMouseEvent& event );
		void MouseCheckParticlesVeto( wxMouseEvent& event );
		void VetoInvalidMouse( wxListCtrl *wanted_list, wxMouseEvent& event );
		void OnMotion(wxMouseEvent& event);
		void OnPackageFocusChange( wxListEvent& event );
		void OnPackageActivated( wxListEvent& event );
		void OnBeginEdit( wxListEvent& event );
		void OnEndEdit( wxListEvent& event );
		void OnVolumeListItemActivated( wxListEvent& event );
		void ReDrawActiveReferences();

		void RemoveVolumeFromAllRefinementPackages(long wanted_volume_asset_id);
		void RemoveImageFromAllRefinementPackages(long wanted_image_asset_id);

	//	Refinement* ReturnPointerToRefinementByRefinementID(long wanted_id);
		ShortRefinementInfo* ReturnPointerToShortRefinementInfoByRefinementID(long wanted_id);
		void ImportAllRefinementInfosFromDatabase();
		ArrayofShortRefinementInfos all_refinement_short_infos;
		//ArrayofRefinements all_refinements;

		ShortClassificationInfo* ReturnPointerToShortClassificationInfoByClassificationID(long wanted_id);
		void ImportAllClassificationInfosFromDatabase();
		ArrayofShortClassificationInfos all_classification_short_infos;

		void ImportAllClassificationSelectionsFromDatabase();
		ArrayofClassificationSelections all_classification_selections;


};


#endif // __MyRefinementPackageAssetPanel__
