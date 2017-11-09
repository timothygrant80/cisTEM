extern MyMainFrame *main_frame;

class MyAssetParentPanel : public AssetParentPanel
{

		friend class GroupDropTarget;
		friend class RenameDialog;

private:



		long highlighted_item;

		bool name_is_being_edited;
		bool should_veto_motion;


		virtual void ImportAssetClick( wxCommandEvent& event ) = 0;
		//virtual void ExportAssetClick( wxCommandEvent& event ) = 0;

		void RemoveAssetClick( wxCommandEvent& event );
		void RenameAssetClick( wxCommandEvent& event);
		void OnDisplayButtonClick( wxCommandEvent& event );

		void RemoveAllAssetsClick( wxCommandEvent& event );
		void AddSelectedAssetClick( wxCommandEvent& event );

		void NewGroupClick( wxCommandEvent& event );
		void RemoveGroupClick( wxCommandEvent& event );
		void RenameGroupClick( wxCommandEvent& event );
		void InvertGroupClick( wxCommandEvent& event );
		void NewFromParentClick( wxCommandEvent& event );
		virtual void EnableNewFromParentButton();

		void OnUpdateUI( wxUpdateUIEvent& event );
		void MouseVeto( wxMouseEvent& event );
		void OnMotion( wxMouseEvent& event );
		void VetoInvalidMouse( wxListCtrl *wanted_list, wxMouseEvent& event );
		void MouseCheckGroupsVeto( wxMouseEvent& event );
		void MouseCheckContentsVeto( wxMouseEvent& event );
		void OnGroupActivated( wxListEvent& event );
		void OnAssetActivated( wxListEvent& event );

		virtual void DisplaySelectedItems();

		virtual int ShowDeleteMessageDialog() = 0;
		virtual int ShowDeleteAllMessageDialog() = 0;

		virtual void CompletelyRemoveAsset(long wanted_asset) = 0;
		virtual void RemoveAssetFromDatabase(long wanted_asset) = 0;
		virtual void RenameAsset(long wanted_asset, wxString wanted_name) = 0;
		virtual void RemoveFromGroupInDatabase(int wanted_group, int wanted_asset_id) = 0;
		virtual void RemoveAllFromDatabase() = 0;
		virtual void RemoveAllGroupMembersFromDatabase(int wanted_group_id) = 0;
		virtual void AddGroupToDatabase(int wanted_group_id, const char * wanted_group_name, int wanted_list_id) = 0;
		virtual void RemoveGroupFromDatabase(int wanted_group_id) = 0;
		virtual void RenameGroupInDatabase(int wanted_group_id, const char *wanted_name) = 0;
		virtual void FillAssetSpecificContentsList() = 0;
		virtual void ImportAllFromDatabase() = 0;
		virtual void DirtyGroups() = 0;
		virtual void DoAfterDeletionCleanup() = 0;

	public:

		AssetGroupList *all_groups_list;
		AssetList *all_assets_list;

		long selected_group;
		long selected_content;

		int current_asset_number;
		int current_group_number;

		bool is_dirty;

		MyAssetParentPanel( wxWindow* parent );
		~MyAssetParentPanel();

		virtual void InsertGroupMemberToDatabase(int wanted_group, int wanted_asset) = 0;
		virtual void InsertArrayofGroupMembersToDatabase(long wanted_group, wxArrayLong *wanted_array, OneSecondProgressDialog *progress_dialog = NULL) = 0;
		long ReturnGroupMember(long wanted_group, long wanted_member);

		void FillContentsList();
		void AddAsset(Asset *asset_to_add);
		void AddContentItemToGroup(long wanted_group, long wanted_content_item);
		void AddArrayItemToGroup(long wanted_group, long wanted_array_item);
		void DeleteArrayItemFromGroup(long wanted_group, long wanted_array_item);
		void AddArrayofArrayItemsToGroup(long wanted_group, wxArrayLong *array_of_wanted_items, OneSecondProgressDialog *progress_dialog = NULL);

		void OnGroupFocusChange( wxListEvent& event );
		void OnContentsSelected( wxListEvent& event );
		void OnBeginEdit( wxListEvent& event );
		void OnEndEdit( wxListEvent& event );
		void OnBeginContentsDrag( wxListEvent& event );

		void SizeGroupColumn();
		void SizeContentsColumn(int column_number);

		unsigned long ReturnNumberOfAssets();
		unsigned long ReturnNumberOfGroups();

		bool DragOverGroups(wxCoord x, wxCoord y);

		void SetGroupName(long wanted_group, wxString wanted_name);

		wxString ReturnGroupName(long wanted_group);
		int ReturnGroupID(long wanted_group);
		wxString ReturnAssetShortFilename(long wanted_asset);
		wxString ReturnAssetLongFilename(long wanted_asset);

		long ReturnGroupSize(long wanted_group);
		int ReturnGroupMemberID(long wanted_group, long wanted_member);
		void RemoveAssetFromGroups(long wanted_asset, bool dirty_groups = true);
		void FillGroupList();

		void SetSelectedGroup(long wanted_group);

		//void CheckActiveButtons();

		int ReturnArrayPositionFromParentID(int wanted_id);
		int ReturnArrayPositionFromAssetID(int wanted_id);
		int ReturnAssetID(int wanted_asset);
		wxString ReturnAssetName(long wanted_asset);
		long ReturnParentAssetID(long wanted_asset);

		//bool IsFileAnAsset(wxFileName file_to_check) = 0;
		virtual Asset* ReturnAssetPointer(long wanted_asset) = 0;
		virtual wxString ReturnItemText(long item, long column) const  = 0;


		void Reset();


		virtual void UpdateInfo() = 0;

};


class GroupDropTarget : public wxDropTarget
{

	friend class MyAssetParentPanel;

	private:

		wxListCtrl *my_owner;
		wxTextDataObject *my_data;
		MyAssetParentPanel *my_panel;

	public:
    	GroupDropTarget(wxListCtrl *owner, MyAssetParentPanel *asset_panel);

    virtual bool OnDrop(wxCoord x, wxCoord y);//, const wxString& dropped_text);
    virtual wxDragResult OnData(wxCoord x, wxCoord y, wxDragResult defResult);
    virtual wxDragResult OnDragOver (wxCoord x, wxCoord y, wxDragResult defResult);
    virtual void OnLeave();
};
