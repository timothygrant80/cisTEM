class AssetGroup {

	long number_allocated;



  public:
	
	AssetGroup();
	AssetGroup( const AssetGroup &obj); // copy contructor
	~AssetGroup();


	int id;
	long *members;
	long number_of_members;
	wxString name;

	void SetName(wxString wanted_name);
	void AddMember(long number_to_add);
	void RemoveMember(long number_to_remove);
	void RemoveAll();

	void CopyFrom(AssetGroup *other_group);

	long FindMember(long member_to_find);

	AssetGroup & operator = (const AssetGroup &t);
	AssetGroup & operator = (const AssetGroup *t);

};


class AssetGroupList {

	long number_allocated;

public:

	AssetGroupList();
	~AssetGroupList();

	long number_of_groups;
	AssetGroup *groups;

	long ReturnNumberOfGroups();

	void AddGroup(wxString name);
	void AddGroup(AssetGroup *group_to_add);

	void RemoveGroup(long number_to_remove);
	void AddMemberToGroup(long wanted_group_number, long member_to_add);
	long ReturnGroupMember(long wanted_group_number, long wanted_member);
	void RemoveAssetFromExtraGroups(long wanted_asset);
	void ShiftMembersDueToAssetRemoval(long number_to_shift_after);
	void RemoveAll();
};



