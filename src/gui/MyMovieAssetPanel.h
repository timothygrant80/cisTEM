class MyMovieAssetPanel : public MyAssetPanelParent {

  protected:
    void DirtyGroups( ) { main_frame->DirtyMovieGroups( ); };

  public:
    MyMovieAssetPanel(wxWindow* parent);
    ~MyMovieAssetPanel( );

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
    void RenameAsset(long wanted_asset, wxString wanted_name);
    void UpdateInfo( );

    bool     IsFileAnAsset(wxFileName file_to_check);
    double   ReturnAssetPixelSize(long wanted_asset);
    double   ReturnAssetAccelerationVoltage(long wanted_asset);
    double   ReturnAssetDosePerFrame(long wanted_asset);
    double   ReturnAssetPreExposureAmount(long wanted_asset);
    float    ReturnAssetSphericalAbberation(long wanted_asset);
    bool     ReturnAssetProteinIsWhite(long wanted_asset);
    int      ReturnAssetEerFramesPerImage(long wanted_asset);
    int      ReturnAssetEerSuperResFactor(long wanted_asset);
    int      ReturnAssetID(long wanted_asset);
    wxString ReturnAssetGainFilename(long wanted_asset);
    wxString ReturnAssetDarkFilename(long wanted_asset);
    float    ReturnAssetBinningFactor(long wanted_asset);
    bool     ReturnCorrectMagDistortion(long wanted_asset);
    float    ReturnMagDistortionAngle(long wanted_asset);
    float    ReturnMagDistortionMajorScale(long wanted_asset);
    float    ReturnMagDistortionMinorScale(long wanted_asset);

    MovieAsset* ReturnAssetPointer(long wanted_asset);
    wxString    ReturnItemText(long item, long column) const;

    int  ShowDeleteMessageDialog( );
    int  ShowDeleteAllMessageDialog( );
    void CompletelyRemoveAsset(long wanted_asset);
    void CompletelyRemoveAssetByID(long wanted_asset_id);
    void DoAfterDeletionCleanup( );
};
