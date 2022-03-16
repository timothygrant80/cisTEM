class AtomicCoordinatesAssetPanel : public MyAssetPanelParent {

  protected:
    void DirtyGroups( ) { main_frame->DirtyParticlePositionGroups( ); };

  public:
    AtomicCoordinatesAssetPanel(wxWindow* parent);
    ~AtomicCoordinatesAssetPanel( );

    void ImportAssetClick(wxCommandEvent& event);

    void RemoveAssetFromDatabase(long wanted_asset);
    void RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id);
    void InsertGroupMemberToDatabase(int wanted_group, int wanted_asset);
    void InsertArrayofGroupMembersToDatabase(long wanted_group, wxArrayLong* wanted_array, OneSecondProgressDialog* progress_dialog = NULL);
    void RemoveAllFromDatabase( );
    void RemoveAllGroupMembersFromDatabase(int wanted_group_id);
    void AddGroupToDatabase(int wanted_group_id, const char* wanted_group_name, int wanted_list_id);
    void RemoveGroupFromDatabase(int wanted_group_id);
    void RenameGroupInDatabase(int wanted_group_id, const char* wanted_name);
    void ImportAllFromDatabase( );
    void FillAssetSpecificContentsList( );
    void UpdateInfo( );

    AtomicCoordinatesAsset* ReturnAssetPointer(long wanted_asset);
    wxString                ReturnItemText(long item, long column) const;
    void                    RenameAsset(long wanted_asset, wxString wanted_name);
    bool                    IsFileAnAsset(wxFileName file_to_check);

    int  ShowDeleteMessageDialog( );
    int  ShowDeleteAllMessageDialog( );
    void CompletelyRemoveAsset(long wanted_asset);
    void CompletelyRemoveAssetByID(long wanted_asset_id);
    void DoAfterDeletionCleanup( );
};
