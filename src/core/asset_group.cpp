#include "core_headers.h"

//#include "gui_core_headers.h"

AssetGroup::AssetGroup( ) {
    // start off with 15 members;
    id                = -1;
    name              = wxEmptyString;
    number_of_members = 0;
    number_allocated  = 15;
    members           = new long[15];
    can_be_picked     = false;
}

AssetGroup::~AssetGroup( ) {
    delete[] members;
}

AssetGroup::AssetGroup(const AssetGroup& obj) // copy constructor
{
    id                = obj.id;
    name              = obj.name;
    number_of_members = obj.number_of_members;
    number_allocated  = obj.number_allocated;

    members = new long[number_allocated];

    for ( long counter = 0; counter < number_of_members; counter++ ) {
        members[counter] = obj.members[counter];
    }

    can_be_picked = obj.can_be_picked;
}

AssetGroup& AssetGroup::operator=(const AssetGroup& t) {
    // Check for self assignment

    if ( this != &t ) {
        if ( this->number_allocated != t.number_allocated ) {
            delete[] this->members;

            this->members          = new long[t.number_allocated];
            this->number_allocated = t.number_allocated;
        }

        this->id                = t.id;
        this->number_of_members = t.number_of_members;
        this->name              = t.name;

        for ( long counter = 0; counter < t.number_of_members; counter++ ) {
            this->members[counter] = t.members[counter];
        }

        can_be_picked = t.can_be_picked;
    }

    return *this;
}

AssetGroup& AssetGroup::operator=(const AssetGroup* t) {
    // Check for self assignment

    if ( this != t ) {
        if ( this->number_allocated != t->number_allocated ) {
            delete[] this->members;

            this->members          = new long[t->number_allocated];
            this->number_allocated = t->number_allocated;
        }

        this->id                = t->id;
        this->number_of_members = t->number_of_members;
        this->name              = t->name;

        for ( long counter = 0; counter < t->number_of_members; counter++ ) {
            this->members[counter] = t->members[counter];
        }

        can_be_picked = t->can_be_picked;
    }

    return *this;
}

void AssetGroup::SetName(wxString wanted_name) {
    name = wanted_name;
}

void AssetGroup::RemoveAll( ) {
    number_of_members = 0;

    if ( number_allocated > 100 ) {
        delete[] members;
        number_allocated = 100;
        members          = new long[number_allocated];
    }
}

void AssetGroup::AddMember(long number_to_add) {
    // check we have enough memory

    long* buffer;

    if ( number_of_members >= number_allocated ) {
        // reallocate..

        if ( number_allocated < 10000 )
            number_allocated *= 2;
        else
            number_allocated += 10000;

        buffer = new long[number_allocated];

        for ( long counter = 0; counter < number_of_members; counter++ ) {
            buffer[counter] = members[counter];
        }

        delete[] members;
        members = buffer;
    }

    // Should be fine for memory, so just add one.

    members[number_of_members] = number_to_add;
    number_of_members++;
}

void AssetGroup::CopyFrom(AssetGroup* other_group) {
    RemoveAll( );
    id            = other_group->id;
    name          = other_group->name;
    can_be_picked = other_group->can_be_picked;

    for ( long counter = 0; counter < other_group->number_of_members; counter++ ) {
        AddMember(other_group->members[counter]);
    }
}

void AssetGroup::RemoveMember(long number_to_remove) // number to remove is the array position IN THE GROUP - this is confusing, as in Add member, it is the array position from all assets
{
    if ( number_to_remove < 0 || number_to_remove >= number_of_members ) {
        wxPrintf("Error! Trying to add to remove an image that doesn't exist\n\n");
        exit(-1);
    }

    for ( long counter = number_to_remove; counter < number_of_members - 1; counter++ ) {
        members[counter] = members[counter + 1];
    }

    number_of_members--;
}

long AssetGroup::FindMember(long member_to_find) {
    long found_position = -1;

    for ( long counter = 0; counter < number_of_members; counter++ ) {
        if ( members[counter] == member_to_find ) {
            found_position = counter;
            break;
        }
    }

    return found_position;
}

AssetGroupList::AssetGroupList( ) {
    // start with 5 - All movies, and space for extra.

    number_allocated = 5;
    number_of_groups = 1;

    groups = new AssetGroup[5];

    //groups[0].SetName("All Movies");
}

void AssetGroupList::AddMemberToGroup(long wanted_group_number, long member_to_add) {
    if ( wanted_group_number < 0 || wanted_group_number >= number_of_groups ) {
        wxPrintf("Error! Trying to add to a group that does not exist\n\n");
        exit(-1);
    }

    groups[wanted_group_number].AddMember(member_to_add);
}

long AssetGroupList::ReturnGroupMember(long wanted_group_number, long wanted_member) {
    if ( wanted_group_number >= number_of_groups || wanted_group_number < 0 ) {
        MyDebugPrintWithDetails("Trying to request a group that doesn't exist (%li)", wanted_group_number);
        return -1;
    }

    if ( wanted_member < 0 || wanted_member >= groups[wanted_group_number].number_of_members ) {
        MyDebugPrintWithDetails("Trying to request a member that doesn't exist (%li; number of members in group %li is %li)", wanted_member, wanted_group_number, groups[wanted_group_number].number_of_members);
        return -1;
    }

    return groups[wanted_group_number].members[wanted_member];
}

AssetGroupList::~AssetGroupList( ) {

    delete[] groups;
}

// Removes the specified member from all extra (not all movies) groups.

void AssetGroupList::RemoveAssetFromExtraGroups(long wanted_asset) {
    long found_position;

    for ( long counter = 1; counter < number_of_groups; counter++ ) {
        found_position = groups[counter].FindMember(wanted_asset);

        if ( found_position != -1 )
            groups[counter].RemoveMember(found_position);
    }
}

void AssetGroupList::ShiftMembersDueToAssetRemoval(long wanted_asset) {
    long group_counter;
    long member_counter;

    for ( group_counter = 1; group_counter < number_of_groups; group_counter++ ) {
        for ( member_counter = 0; member_counter < groups[group_counter].number_of_members; member_counter++ ) {
            if ( groups[group_counter].members[member_counter] > wanted_asset )
                groups[group_counter].members[member_counter]--;
        }
    }
}

void AssetGroupList::AddGroup(wxString name) {
    // check we have enough memory

    AssetGroup* buffer;

    if ( number_of_groups >= number_allocated ) {
        // reallocate..

        if ( number_allocated < 1000 )
            number_allocated *= 2;
        else
            number_allocated += 1000;

        buffer = new AssetGroup[number_allocated];

        for ( long counter = 0; counter < number_of_groups; counter++ ) {
            buffer[counter].CopyFrom(&groups[counter]);
        }

        delete[] groups;
        groups = buffer;
    }

    // Should be fine for memory, so just add one.

    groups[number_of_groups].SetName(name);
    groups[number_of_groups].RemoveAll( );
    number_of_groups++;
    // this is in case it used to exist.
}

void AssetGroupList::AddGroup(AssetGroup* group_to_add) {
    // check we have enough memory

    AssetGroup* buffer;

    if ( number_of_groups >= number_allocated ) {
        // reallocate..

        if ( number_allocated < 1000 )
            number_allocated *= 2;
        else
            number_allocated += 1000;

        buffer = new AssetGroup[number_allocated];

        for ( long counter = 0; counter < number_of_groups; counter++ ) {
            buffer[counter].CopyFrom(&groups[counter]);
        }

        delete[] groups;
        groups = buffer;
    }

    // Should be fine for memory, so just add one.

    groups[number_of_groups] = group_to_add;
    number_of_groups++;
}

void AssetGroupList::RemoveGroup(long number_to_remove) {
    //wxPrintf("Removing group #%li\n", number_to_remove);

    if ( number_to_remove < 0 || number_to_remove >= number_of_groups ) {
        wxPrintf("Error! Trying to remove a group that does not exist\n\n");
        exit(-1);
    }

    for ( long counter = number_to_remove; counter < number_of_groups - 1; counter++ ) {
        groups[counter].CopyFrom(&groups[counter + 1]);
    }

    number_of_groups--;
}

long AssetGroupList::ReturnNumberOfGroups( ) {
    return number_of_groups;
}

void AssetGroupList::RemoveAll( ) {

    for ( long counter = 0; counter < number_of_groups; counter++ ) {
        groups[counter].RemoveAll( );
    }

    number_of_groups = 1;
}
