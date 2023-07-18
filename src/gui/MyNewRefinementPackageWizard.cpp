//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofNewRefinementPackageWizardClassSelection);

#ifdef USE_FP16_PARTICLE_STACKS
#warning "Using half-precision particle stacks"
#endif

extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;
extern MyParticlePositionAssetPanel*  particle_position_asset_panel;
extern MyImageAssetPanel*             image_asset_panel;
extern MyVolumeAssetPanel*            volume_asset_panel;

static int wxCMPFUNC_CONV SortByParentImageID(RefinementPackageParticleInfo** a, RefinementPackageParticleInfo** b) // function for sorting the classum selections by parent_image_id - this makes cutting them out more efficient
{
    if ( (*a)->parent_image_id > (*b)->parent_image_id )
        return 1;
    else if ( (*a)->parent_image_id < (*b)->parent_image_id )
        return -1;
    else {
        if ( (*a)->original_particle_position_asset_id > (*b)->original_particle_position_asset_id )
            return 1;
        else if ( (*a)->original_particle_position_asset_id < (*b)->original_particle_position_asset_id )
            return -1;
        else
            return 0;
    }
};

MyNewRefinementPackageWizard::MyNewRefinementPackageWizard(wxWindow* parent)
    : NewRefinementPackageWizard(parent) {
    template_page          = new TemplateWizardPage(this);
    parameter_page         = new InputParameterWizardPage(this);
    particle_group_page    = new ParticleGroupWizardPage(this);
    number_of_classes_page = new NumberofClassesWizardPage(this);
    box_size_page          = new BoxSizeWizardPage(this);

    initial_reference_page = new InitialReferencesWizardPage(this);
    symmetry_page          = new SymmetryWizardPage(this);
    molecular_weight_page  = new MolecularWeightWizardPage(this);
    largest_dimension_page = new LargestDimensionWizardPage(this);
    class_selection_page   = new ClassSelectionWizardPage(this);

    recentre_picks_page                   = new RecentrePicksWizardPage(this);
    remove_duplicate_picks_page           = new RemoveDuplicatesWizardPage(this);
    remove_duplicate_picks_threshold_page = new RemoveDuplicateThresholdWizardPage(this);

    output_pixel_size_page = new OutputPixelSizeWizardPage(this);

    class_setup_pageA = new ClassesSetupWizardPageA(this);
    class_setup_pageB = new ClassesSetupWizardPageB(this);
    class_setup_pageC = new ClassesSetupWizardPageC(this);
    class_setup_pageD = new ClassesSetupWizardPageD(this);
    class_setup_pageE = new ClassesSetupWizardPageE(this);

    GetPageAreaSizer( )->Add(template_page);
    //GetPageAreaSizer()->Add(particle_group_page);
    //GetPageAreaSizer()->Add(number_of_classes_page);
    //GetPageAreaSizer()->Add(box_size_page);

    Bind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(MyNewRefinementPackageWizard::OnUpdateUI), this);
}

MyNewRefinementPackageWizard::~MyNewRefinementPackageWizard( ) {
    /*
	delete template_page;
	delete particle_group_page;
	delete number_of_classes_page;
	delete box_size_page;
	delete class_setup_page;
	delete initial_reference_page;
	delete symmetry_page;
	delete molecular_weight_page;
	delete largest_dimension_page;
	delete particle_group_page;
	delete class_selection_page;
*/
    Unbind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(MyNewRefinementPackageWizard::OnUpdateUI), this);
}

void MyNewRefinementPackageWizard::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( GetCurrentPage( ) == template_page )
        EnableNextButton( );
    else if ( GetCurrentPage( ) == particle_group_page ) {
        if ( particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetCount( ) > 0 ) {
            if ( particle_position_asset_panel->ReturnGroupSize(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection( )) > 0 )
                EnableNextButton( );
            else
                DisableNextButton( );
        }
        else
            DisableNextButton( );
    }
    else if ( GetCurrentPage( ) == box_size_page ) {
        EnableNextButton( );
    }
    else if ( GetCurrentPage( ) == number_of_classes_page )
        EnableNextButton( );
    else if ( GetCurrentPage( ) == class_setup_pageA )
        EnableNextButton( );
    else if ( GetCurrentPage( ) == class_setup_pageB ) {
        if ( class_setup_pageB->my_panel->ClassListCtrl->GetSelectedItemCount( ) > 0 )
            EnableNextButton( );
        else
            DisableNextButton( );
    }
    else if ( GetCurrentPage( ) == class_setup_pageC ) {
        if ( class_setup_pageC->IsAtLeastOneOldClassSelectedForEachNewClass( ) == true )
            EnableNextButton( );
        else
            DisableNextButton( );
    }
    else if ( GetCurrentPage( ) == symmetry_page ) {
        wxString symmetry = symmetry_page->my_panel->SymmetryComboBox->GetValue( );
        if ( IsAValidSymmetry(&symmetry) == true )
            EnableNextButton( );
        else
            DisableNextButton( );
    }
    else if ( GetCurrentPage( ) == molecular_weight_page )
        EnableNextButton( );
    else if ( GetCurrentPage( ) == largest_dimension_page )
        EnableNextButton( );
    else if ( GetCurrentPage( ) == parameter_page )
        EnableNextButton( );
    else if ( GetCurrentPage( ) == class_selection_page ) {
        if ( class_selection_page->my_panel->SelectionListCtrl->GetSelectedItemCount( ) > 0 ) {
            int  total_selections = 0;
            long item             = -1;

            for ( ;; ) {
                item = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
                if ( item == -1 )
                    break;

                total_selections += refinement_package_asset_panel->all_classification_selections.Item(item).number_of_selections;

                if ( total_selections > 0 )
                    break;
            }

            if ( total_selections > 0 )
                EnableNextButton( );
            else
                DisableNextButton( );
        }
        else
            DisableNextButton( );
    }
}

void MyNewRefinementPackageWizard::DisableNextButton( ) {
    wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
    if ( win )
        win->Enable(false);
}

void MyNewRefinementPackageWizard::EnableNextButton( ) {
    wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
    if ( win )
        win->Enable(true);
}

void MyNewRefinementPackageWizard::PageChanging(wxWizardEvent& event) {
}

void MyNewRefinementPackageWizard::PageChanged(wxWizardEvent& event) {

    if ( event.GetPage( ) == template_page ) {
        if ( template_page->my_panel->InfoText->has_autowrapped == false ) {
            template_page->Freeze( );
            template_page->my_panel->InfoText->AutoWrap( );
            template_page->Layout( );
            template_page->Thaw( );
        }
    }
    else if ( event.GetPage( ) == parameter_page ) {
        parameter_page->Freeze( );

        if ( parameter_page->my_panel->InfoText->has_autowrapped == false ) {

            parameter_page->my_panel->InfoText->AutoWrap( );
            parameter_page->Layout( );
        }

        //	wxPrintf("filling\n");
        parameter_page->my_panel->GroupComboBox->Clear( );
        for ( int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3).refinement_ids.GetCount( ); counter++ ) {
            parameter_page->my_panel->GroupComboBox->Append(refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3).refinement_ids[counter])->name);
        }
        //	 wxPrintf("filled\n");

        parameter_page->my_panel->GroupComboBox->SetSelection(parameter_page->my_panel->GroupComboBox->GetCount( ) - 1);
        parameter_page->Thaw( );
    }
    if ( event.GetPage( ) == particle_group_page ) {
        if ( particle_group_page->my_panel->InfoText->has_autowrapped == false ) {
            particle_group_page->Freeze( );
            particle_group_page->my_panel->InfoText->AutoWrap( );
            particle_group_page->Layout( );
            particle_group_page->Thaw( );
        }
    }
    else if ( event.GetPage( ) == number_of_classes_page ) {
        if ( number_of_classes_page->my_panel->InfoText->has_autowrapped == false ) {
            number_of_classes_page->Freeze( );
            number_of_classes_page->my_panel->InfoText->AutoWrap( );
            number_of_classes_page->Layout( );
            number_of_classes_page->Thaw( );
        }
    }
    else if ( event.GetPage( ) == box_size_page ) {
        if ( box_size_page->my_panel->InfoText->has_autowrapped == false ) {
            box_size_page->Freeze( );
            box_size_page->my_panel->InfoText->AutoWrap( );
            box_size_page->Layout( );
            box_size_page->Thaw( );
        }

        if ( template_page->my_panel->GroupComboBox->GetSelection( ) > 1 && box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( ) == 1 ) {
            RefinementPackage* template_package = &refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
            box_size_page->my_panel->BoxSizeSpinCtrl->SetValue(template_package->stack_box_size);
        }
        else if ( box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( ) == 1 ) {
            // do an intelligent default..

            if ( template_page->my_panel->GroupComboBox->GetSelection( ) == 0 ) // new, we should do largest_dimension * 2, scaled up to factorizable.
            {
                // i need to know the pixel size, which is not a simple thing really. For now, lets just make a guess.
                // lets just take the pixel size of the first included particle..

                float pixel_size_guess;
                float current_largest_dimension = largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( );

                ParticlePositionAsset* first_particle       = particle_position_asset_panel->ReturnAssetPointer(particle_position_asset_panel->ReturnGroupMember(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection( ), 0));
                ImageAsset*            first_particle_image = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(first_particle->parent_id));

                pixel_size_guess = first_particle_image->pixel_size;
                current_largest_dimension /= pixel_size_guess;

                box_size_page->my_panel->BoxSizeSpinCtrl->SetValue(ReturnClosestFactorizedUpper(int(current_largest_dimension * 2.2), 3, true));
            }
            else // from class selection
            { // take the stack size of the refinement package of the first selected class selection

                long                                                                                                                     item                             = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
                MyDebugAssertTrue(item != -1, "Ooops, there is no selected classification selection - that shouldn't be possible?") long parent_refinement_package_id     = refinement_package_asset_panel->all_classification_selections.Item(item).refinement_package_asset_id;
                long                                                                                                                     parent_refinement_array_position = refinement_package_asset_panel->ReturnArrayPositionFromAssetID(parent_refinement_package_id);

                box_size_page->my_panel->BoxSizeSpinCtrl->SetValue(refinement_package_asset_panel->all_refinement_packages[parent_refinement_array_position].stack_box_size);
            }
        }
    }
    else if ( event.GetPage( ) == output_pixel_size_page ) {
        if ( output_pixel_size_page->my_panel->InfoText->has_autowrapped == false ) {
            output_pixel_size_page->Freeze( );
            output_pixel_size_page->my_panel->InfoText->AutoWrap( );
            output_pixel_size_page->Layout( );
            output_pixel_size_page->Thaw( );
        }

        if ( template_page->my_panel->GroupComboBox->GetSelection( ) > 1 && box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( ) == 1 ) {
            RefinementPackage* template_package = &refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
            output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ChangeValueFloat(template_package->output_pixel_size);
        }
        else if ( output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ReturnValue( ) == 0.0f ) {
            // do an intelligent default..

            if ( template_page->my_panel->GroupComboBox->GetSelection( ) == 0 ) // new, we should do largest_dimension * 2, scaled up to factorizable.
            {
                // i need to know the pixel size, which is not a simple thing really. For now, lets just make a guess.
                // lets just take the pixel size of the first included particle..

                float pixel_size_guess;
                float current_largest_dimension = largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( );

                ParticlePositionAsset* first_particle       = particle_position_asset_panel->ReturnAssetPointer(particle_position_asset_panel->ReturnGroupMember(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection( ), 0));
                ImageAsset*            first_particle_image = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(first_particle->parent_id));

                output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ChangeValueFloat(first_particle_image->pixel_size);
            }
            else // from class selection
            { // take the stack size of the refinement package of the first selected class selection

                long                                                                                                                     item                             = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
                MyDebugAssertTrue(item != -1, "Ooops, there is no selected classification selection - that shouldn't be possible?") long parent_refinement_package_id     = refinement_package_asset_panel->all_classification_selections.Item(item).refinement_package_asset_id;
                long                                                                                                                     parent_refinement_array_position = refinement_package_asset_panel->ReturnArrayPositionFromAssetID(parent_refinement_package_id);

                output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ChangeValueFloat(refinement_package_asset_panel->all_refinement_packages[parent_refinement_array_position].output_pixel_size);
            }
        }
    }
    else if ( event.GetPage( ) == initial_reference_page ) {

        int       counter;
        wxWindow* window_pointer;
        initial_reference_page->Freeze( );

        //initial_reference_page->my_panel->Destroy();
        //initial_reference_page->CreatePanel();

        initial_reference_page->my_panel->InfoText->AutoWrap( );
        initial_reference_page->my_panel->ScrollSizer->Clear(true);

        for ( counter = 1; counter <= number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue( ); counter++ ) {
            ClassVolumeSelectPanel* panel1 = new ClassVolumeSelectPanel(initial_reference_page->my_panel->ScrollWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
            panel1->ClassText->SetLabel(wxString::Format("Class #%2i :", counter));
            panel1->class_number = counter;
            initial_reference_page->my_panel->ScrollSizer->Add(panel1, 0, wxEXPAND | wxALL, 5);
        }

        initial_reference_page->Layout( );
        initial_reference_page->Thaw( );
    }
    else if ( event.GetPage( ) == symmetry_page ) {
        if ( symmetry_page->my_panel->InfoText->has_autowrapped == false ) {
            symmetry_page->Freeze( );
            symmetry_page->my_panel->InfoText->AutoWrap( );
            symmetry_page->Layout( );
            symmetry_page->Thaw( );
        }

        if ( template_page->my_panel->GroupComboBox->GetSelection( ) > 1 && symmetry_page->my_panel->SymmetryComboBox->GetValue( ) == "0" ) {
            RefinementPackage* template_package = &refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
            symmetry_page->my_panel->SymmetryComboBox->SetValue(template_package->symmetry);
        }
        else if ( symmetry_page->my_panel->SymmetryComboBox->GetValue( ) == "0" ) {
            if ( template_page->my_panel->GroupComboBox->GetSelection( ) == 1 ) // take the value of the first selected classum selections refinement package
            {
                long                                                                                                                     item                             = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
                MyDebugAssertTrue(item != -1, "Ooops, there is no selected classification selection - that shouldn't be possible?") long parent_refinement_package_id     = refinement_package_asset_panel->all_classification_selections.Item(item).refinement_package_asset_id;
                long                                                                                                                     parent_refinement_array_position = refinement_package_asset_panel->ReturnArrayPositionFromAssetID(parent_refinement_package_id);

                symmetry_page->my_panel->SymmetryComboBox->SetValue(refinement_package_asset_panel->all_refinement_packages[parent_refinement_array_position].symmetry);
            }
            else
                symmetry_page->my_panel->SymmetryComboBox->SetSelection(0);
        }
    }
    else if ( event.GetPage( ) == molecular_weight_page ) {
        if ( molecular_weight_page->my_panel->InfoText->has_autowrapped == false ) {
            molecular_weight_page->Freeze( );
            molecular_weight_page->my_panel->InfoText->AutoWrap( );
            molecular_weight_page->Layout( );
            molecular_weight_page->Thaw( );
        }

        if ( template_page->my_panel->GroupComboBox->GetSelection( ) > 1 && molecular_weight_page->my_panel->MolecularWeightTextCtrl->ReturnValue( ) == 0.0 ) {
            RefinementPackage* template_package = &refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
            molecular_weight_page->my_panel->MolecularWeightTextCtrl->ChangeValueFloat(template_package->estimated_particle_weight_in_kda);
        }
        else if ( molecular_weight_page->my_panel->MolecularWeightTextCtrl->ReturnValue( ) == 0.0 ) {
            if ( template_page->my_panel->GroupComboBox->GetSelection( ) == 1 ) // take the value of the first selected classum selections refinement package
            {
                long                                                                                                                     item                             = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
                MyDebugAssertTrue(item != -1, "Ooops, there is no selected classification selection - that shouldn't be possible?") long parent_refinement_package_id     = refinement_package_asset_panel->all_classification_selections.Item(item).refinement_package_asset_id;
                long                                                                                                                     parent_refinement_array_position = refinement_package_asset_panel->ReturnArrayPositionFromAssetID(parent_refinement_package_id);

                molecular_weight_page->my_panel->MolecularWeightTextCtrl->ChangeValueFloat(refinement_package_asset_panel->all_refinement_packages[parent_refinement_array_position].estimated_particle_weight_in_kda);
            }
            else
                molecular_weight_page->my_panel->MolecularWeightTextCtrl->ChangeValueFloat(300.0f);
        }
    }
    else if ( event.GetPage( ) == largest_dimension_page ) {
        if ( largest_dimension_page->my_panel->InfoText->has_autowrapped == false ) {
            largest_dimension_page->Freeze( );
            largest_dimension_page->my_panel->InfoText->AutoWrap( );
            largest_dimension_page->Layout( );
            largest_dimension_page->Thaw( );
        }

        if ( template_page->my_panel->GroupComboBox->GetSelection( ) > 1 && largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( ) == 0.0 ) {
            RefinementPackage* template_package = &refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
            largest_dimension_page->my_panel->LargestDimensionTextCtrl->ChangeValueFloat(template_package->estimated_particle_size_in_angstroms);
        }
        else if ( largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( ) == 0.0 ) {
            if ( template_page->my_panel->GroupComboBox->GetSelection( ) == 1 ) // take the value of the first selected classum selections refinement package
            {
                long                                                                                                                     item                             = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
                MyDebugAssertTrue(item != -1, "Ooops, there is no selected classification selection - that shouldn't be possible?") long parent_refinement_package_id     = refinement_package_asset_panel->all_classification_selections.Item(item).refinement_package_asset_id;
                long                                                                                                                     parent_refinement_array_position = refinement_package_asset_panel->ReturnArrayPositionFromAssetID(parent_refinement_package_id);

                largest_dimension_page->my_panel->LargestDimensionTextCtrl->ChangeValueFloat(refinement_package_asset_panel->all_refinement_packages[parent_refinement_array_position].estimated_particle_size_in_angstroms);
            }
            else
                largest_dimension_page->my_panel->LargestDimensionTextCtrl->ChangeValueFloat(150.0f);
        }
    }
    else if ( event.GetPage( ) == class_selection_page ) {
        if ( class_selection_page->my_panel->InfoText->has_autowrapped == false ) {
            class_selection_page->Freeze( );
            class_selection_page->my_panel->InfoText->AutoWrap( );
            class_selection_page->Layout( );
            class_selection_page->Thaw( );
        }
    }
    else if ( event.GetPage( ) == class_setup_pageA ) {
        if ( class_setup_pageA->my_panel->InfoText->has_autowrapped == false ) {
            class_setup_pageA->Freeze( );
            class_setup_pageA->my_panel->InfoText->AutoWrap( );
            class_setup_pageA->Layout( );
            class_setup_pageA->Thaw( );
        }
    }
    else if ( event.GetPage( ) == class_setup_pageB ) {
        if ( class_setup_pageB->my_panel->InfoText->has_autowrapped == false ) {
            class_setup_pageB->Freeze( );
            class_setup_pageB->my_panel->InfoText->AutoWrap( );
            class_setup_pageB->Layout( );
            class_setup_pageB->Thaw( );
        }

        // we need to fill with the appopriate classess.

        if ( class_setup_pageB->my_panel->ClassListCtrl->GetItemCount( ) == 0 ) {

            class_setup_pageB->my_panel->Freeze( );
            class_setup_pageB->my_panel->ClassListCtrl->ClearAll( );
            class_setup_pageB->my_panel->ClassListCtrl->InsertColumn(0, wxT("Class No."), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
            class_setup_pageB->my_panel->ClassListCtrl->InsertColumn(1, wxT("Average Occupancy"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
            class_setup_pageB->my_panel->ClassListCtrl->InsertColumn(2, wxT("Est. Resolution"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);

            ShortRefinementInfo* selected_refinement = refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection( ) - 3].refinement_ids[parameter_page->my_panel->GroupComboBox->GetSelection( )]);
            wxPrintf("refinement_id = %li, number_of_occ = %li\n", selected_refinement->refinement_id, selected_refinement->average_occupancy.GetCount( ));
            for ( int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection( ) - 3].number_of_classes; counter++ ) {
                class_setup_pageB->my_panel->ClassListCtrl->InsertItem(counter, wxString::Format("Class #%2i", counter + 1));
                class_setup_pageB->my_panel->ClassListCtrl->SetItem(counter, 1, wxString::Format("%.2f %%", selected_refinement->average_occupancy[counter]));

                if ( selected_refinement->estimated_resolution[counter] <= 0 )
                    class_setup_pageB->my_panel->ClassListCtrl->SetItem(counter, 2, wxString::Format(wxT("N/A"), selected_refinement->estimated_resolution[counter]));
                else
                    class_setup_pageB->my_panel->ClassListCtrl->SetItem(counter, 2, wxString::Format(wxT("%.2f Å"), selected_refinement->estimated_resolution[counter]));
            }

            class_setup_pageB->my_panel->Thaw( );
        }
    }
    else if ( event.GetPage( ) == class_setup_pageC ) {
        if ( class_setup_pageC->my_panel->InfoText->has_autowrapped == false ) {
            class_setup_pageC->Freeze( );
            class_setup_pageC->my_panel->InfoText->AutoWrap( );
            class_setup_pageC->Layout( );
            class_setup_pageC->Thaw( );
        }

        // we need to hold class selections, but we don't want to clear/redraw them them if they are valid (the user may have clicked fwd/bck and come back)
        // if something changes that will affect this panel, it should clear current_class_selections, here we check if it is clear and if so set it up.

        if ( class_setup_pageC->current_class_selections.GetCount( ) == 0 ) {
            NewRefinementPackageWizardClassSelection temp_classes_to_select_from;
            int                                      number_of_classes_to_select_from;

            // how many classes do we have to select from. this depends on whether we are carrying over all classes or not.

            if ( class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false ) {
                number_of_classes_to_select_from = class_setup_pageB->my_panel->ClassListCtrl->GetSelectedItemCount( );
            }
            else {
                number_of_classes_to_select_from = refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection( ) - 3].refinement_ids[parameter_page->my_panel->GroupComboBox->GetSelection( )])->number_of_classes;
            }

            temp_classes_to_select_from.class_selection.Add(false, number_of_classes_to_select_from);
            class_setup_pageC->current_class_selections.Add(temp_classes_to_select_from, number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue( ));

            class_setup_pageC->my_panel->Freeze( );
            class_setup_pageC->my_panel->NewClassListCtrl->ClearAll( );
            class_setup_pageC->my_panel->NewClassListCtrl->InsertColumn(0, wxT("New Class No."), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
            class_setup_pageC->my_panel->NewClassListCtrl->InsertColumn(1, wxT("Old Class Selections"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);

            for ( int counter = 0; counter < number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue( ); counter++ ) {
                class_setup_pageC->my_panel->NewClassListCtrl->InsertItem(counter, wxString::Format("Class #%2i", counter + 1));
            }

            class_setup_pageC->my_panel->NewClassListCtrl->SetItemState(0, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
            class_setup_pageC->DrawClassSelections( );
            class_setup_pageC->my_panel->Thaw( );
        }
    }
    else if ( event.GetPage( ) == class_setup_pageD ) {
        if ( class_setup_pageD->my_panel->InfoText->has_autowrapped == false ) {
            class_setup_pageD->Freeze( );
            class_setup_pageD->my_panel->InfoText->AutoWrap( );
            class_setup_pageD->Layout( );
            class_setup_pageD->Thaw( );
        }
    }
    else if ( event.GetPage( ) == class_setup_pageE ) {
        if ( class_setup_pageE->my_panel->InfoText->has_autowrapped == false ) {
            class_setup_pageE->Freeze( );
            class_setup_pageE->my_panel->InfoText->AutoWrap( );
            class_setup_pageE->Layout( );
            class_setup_pageE->Thaw( );
        }
    }
    else if ( event.GetPage( ) == recentre_picks_page ) {
        if ( recentre_picks_page->my_panel->InfoText->has_autowrapped == false ) {
            recentre_picks_page->Freeze( );
            recentre_picks_page->my_panel->InfoText->AutoWrap( );
            recentre_picks_page->Layout( );
            recentre_picks_page->Thaw( );
        }
    }
    else if ( event.GetPage( ) == remove_duplicate_picks_page ) {
        if ( remove_duplicate_picks_page->my_panel->InfoText->has_autowrapped == false ) {
            remove_duplicate_picks_page->Freeze( );
            remove_duplicate_picks_page->my_panel->InfoText->AutoWrap( );
            remove_duplicate_picks_page->Layout( );
            remove_duplicate_picks_page->Thaw( );
        }
    }
    else if ( event.GetPage( ) == remove_duplicate_picks_threshold_page ) {
        if ( remove_duplicate_picks_threshold_page->my_panel->InfoText->has_autowrapped == false ) {
            remove_duplicate_picks_threshold_page->Freeze( );
            remove_duplicate_picks_threshold_page->my_panel->InfoText->AutoWrap( );
            remove_duplicate_picks_threshold_page->Layout( );
            remove_duplicate_picks_threshold_page->Thaw( );
        }

        if ( remove_duplicate_picks_threshold_page->my_panel->DuplicatePickThresholdTextCtrl->ReturnValue( ) == 0.0 ) {
            remove_duplicate_picks_threshold_page->my_panel->DuplicatePickThresholdTextCtrl->ChangeValueFloat(largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( ) * 0.25f);
        }
    }
}

// given an array of refinement package particle infos, return an array of interegers with unique parent image IDs
wxArrayInt MyNewRefinementPackageWizard::ReturnIDsOfActiveImages(ArrayOfRefinmentPackageParticleInfos& particle_info_buffer) {
    wxArrayInt active_images;
    long       particle_counter;
    int        second_counter;

    // get all images we are using..

    for ( particle_counter = 0; particle_counter < particle_info_buffer.GetCount( ); particle_counter++ ) {
        if ( particle_info_buffer[particle_counter].parent_image_id != -1 && particle_info_buffer[particle_counter].parent_image_id != -2 ) {
            active_images.Add(particle_info_buffer[particle_counter].parent_image_id);

            for ( second_counter = particle_counter + 1; second_counter < particle_info_buffer.GetCount( ); second_counter++ ) {
                if ( particle_info_buffer[second_counter].parent_image_id == particle_info_buffer[particle_counter].parent_image_id )
                    particle_info_buffer[second_counter].parent_image_id = -2;
            }
        }
    }

    //wxPrintf("There are %li active images\n", active_images.GetCount());
    return active_images;
}

void MyNewRefinementPackageWizard::OnFinished(wxWizardEvent& event) {
    int  class_counter;
    long particle_counter;
    long counter;
    long item;
    long parent_classification_id;

    // cut out the particles if necessary..

    RefinementPackage*            temp_refinement_package = new RefinementPackage;
    RefinementPackageParticleInfo temp_particle_info;

    ClassRefinementResults junk_class_results;
    RefinementResult       junk_result;
    Refinement             temp_refinement;

    RefinementPackage* parent_refinement_package_link;

    wxArrayLong current_images;

    if ( template_page->my_panel->GroupComboBox->GetSelection( ) == 0 ) // This is a new package
    {
        long number_of_particles;
        number_of_particles = particle_position_asset_panel->ReturnGroupSize(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection( ));

        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Refinement Package", "Creating Refinement Package...", number_of_particles * 2, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

        temp_refinement_package->name                     = wxString::Format("Refinement Package #%li", refinement_package_asset_panel->current_asset_number);
        temp_refinement_package->number_of_classes        = number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue( );
        temp_refinement_package->stack_has_white_protein  = false;
        temp_refinement_package->number_of_run_refinments = 0;

        temp_refinement.number_of_classes                = temp_refinement_package->number_of_classes;
        temp_refinement.number_of_particles              = number_of_particles;
        temp_refinement.name                             = "Random Parameters";
        temp_refinement.resolution_statistics_box_size   = box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( );
        temp_refinement.resolution_statistics_pixel_size = output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ReturnValue( );
        temp_refinement.refinement_package_asset_id      = refinement_package_asset_panel->current_asset_number + 1;

        long current_particle_parent_image_id = 0;
        long current_loaded_image_id          = -1;
        long position_in_stack                = 0;

        int current_x_pos;
        int current_y_pos;

        float average_value_at_edges;
        float image_defocus_1;
        float image_defocus_2;
        float image_defocus_angle;
        float image_phase_shift;
        float image_amplitude_contrast;
        float image_tilt_angle;
        float image_tilt_axis;

        // for tilt calculations
        AnglesAndShifts rotation_angle;
        float           image_x_position;
        float           image_y_position;
        float           x_rotated;
        float           y_rotated;
        float           tilt_based_height;

        ImageAsset*            current_image_asset             = NULL;
        ParticlePositionAsset* current_particle_position_asset = NULL;
        Image                  current_image;
        Image                  cut_particle;

        wxFileName output_stack_filename = main_frame->current_project.particle_stack_directory.GetFullPath( ) + wxString::Format("/particle_stack_%li.mrc", refinement_package_asset_panel->current_asset_number);

        // specific package setup..

        temp_refinement_package->stack_box_size    = box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( );
        temp_refinement_package->output_pixel_size = output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ReturnValue( );
        temp_refinement_package->stack_filename    = output_stack_filename.GetFullPath( );
        temp_refinement_package->symmetry          = symmetry_page->my_panel->SymmetryComboBox->GetValue( ).Upper( );
        ;
        temp_refinement_package->estimated_particle_weight_in_kda     = molecular_weight_page->my_panel->MolecularWeightTextCtrl->ReturnValue( );
        temp_refinement_package->estimated_particle_size_in_angstroms = largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( );

        // setup the 3ds

        wxWindowList all_children = initial_reference_page->my_panel->ScrollWindow->GetChildren( );

        ClassVolumeSelectPanel* panel_pointer;

        for ( counter = 0; counter < all_children.GetCount( ); counter++ ) {
            if ( all_children.Item(counter)->GetData( )->GetClassInfo( )->GetClassName( ) == wxString("wxPanel") ) {
                panel_pointer = reinterpret_cast<ClassVolumeSelectPanel*>(all_children.Item(counter)->GetData( ));

                if ( panel_pointer->VolumeComboBox->GetSelection( ) == 0 ) {
                    temp_refinement_package->references_for_next_refinement.Add(-1);
                }
                else {
                    temp_refinement_package->references_for_next_refinement.Add(volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(panel_pointer->VolumeComboBox->GetSelection( ) - 1)->asset_id);
                }
            }
        }

        // size the box..

        cut_particle.Allocate(box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( ), box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( ), 1);

        // open the output stack

        MRCFile output_stack(output_stack_filename.GetFullPath( ).ToStdString( ), true);

        // setup the refinement..

        long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
        temp_refinement_package->refinement_ids.Add(refinement_id);

        temp_refinement.refinement_id                       = refinement_id;
        temp_refinement.resolution_statistics_are_generated = true;
        temp_refinement.SizeAndFillWithEmpty(number_of_particles, temp_refinement_package->number_of_classes);

        //temp_refinement.class_refinement_results.Alloc(temp_refinement_package->number_of_classes);

        //temp_refinement.class_refinement_results.Add(junk_class_results, temp_refinement_package->number_of_classes);

        for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
            temp_refinement.class_refinement_results[class_counter].average_occupancy = 100.0 / temp_refinement_package->number_of_classes;
        }

        for ( counter = 0; counter < number_of_particles; counter++ ) {
            // current particle, what image is it from?

            current_particle_position_asset  = particle_position_asset_panel->ReturnAssetPointer(particle_position_asset_panel->ReturnGroupMember(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection( ), counter));
            current_particle_parent_image_id = current_particle_position_asset->parent_id;

            if ( current_loaded_image_id != current_particle_parent_image_id ) {
                // load it..

                current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(current_particle_parent_image_id));
                current_image.QuickAndDirtyReadSlice(current_image_asset->filename.GetFullPath( ).ToStdString( ), 1);

                // take out weird values..

                current_image.ReplaceOutliersWithMean(6);

                current_loaded_image_id = current_particle_parent_image_id;
                average_value_at_edges  = current_image.ReturnAverageOfRealValuesOnEdges( );

                // we have to get the defocus stuff from the database..

                main_frame->current_project.database.GetActiveDefocusValuesByImageID(current_particle_parent_image_id, image_defocus_1, image_defocus_2, image_defocus_angle, image_phase_shift, image_amplitude_contrast, image_tilt_angle, image_tilt_axis);
            }

            // do the cutting..

            position_in_stack++;

            current_x_pos = myround(current_particle_position_asset->x_position / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_x;
            current_y_pos = myround(current_particle_position_asset->y_position / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_y;

            current_image.ClipInto(&cut_particle, average_value_at_edges, false, 1.0, current_x_pos, current_y_pos, 0);
            cut_particle.ZeroFloatAndNormalize( );
            if ( current_image_asset->protein_is_white )
                cut_particle.InvertRealValues( );
            cut_particle.WriteSlice(&output_stack, position_in_stack);

            // set the contained particles..

            temp_particle_info.spherical_aberration                = current_image_asset->spherical_aberration;
            temp_particle_info.microscope_voltage                  = current_image_asset->microscope_voltage;
            temp_particle_info.parent_image_id                     = current_particle_parent_image_id;
            temp_particle_info.pixel_size                          = current_image_asset->pixel_size;
            temp_particle_info.position_in_stack                   = position_in_stack;
            temp_particle_info.defocus_angle                       = image_defocus_angle;
            temp_particle_info.phase_shift                         = image_phase_shift;
            temp_particle_info.amplitude_contrast                  = image_amplitude_contrast;
            temp_particle_info.x_pos                               = current_particle_position_asset->x_position;
            temp_particle_info.y_pos                               = current_particle_position_asset->y_position;
            temp_particle_info.original_particle_position_asset_id = current_particle_position_asset->asset_id;

            if ( image_tilt_angle == 0.0f && image_tilt_axis == 0.0f ) {
                temp_particle_info.defocus_1 = image_defocus_1;
                temp_particle_info.defocus_2 = image_defocus_2;
            }
            else // calculate a defocus value based on tilt..
            {
                rotation_angle.GenerateRotationMatrix2D(image_tilt_axis);
                image_x_position = current_x_pos * current_image_asset->pixel_size;
                image_y_position = current_y_pos * current_image_asset->pixel_size;

                rotation_angle.euler_matrix.RotateCoords2D(image_x_position, image_y_position, x_rotated, y_rotated);
                tilt_based_height            = y_rotated * tanf(deg_2_rad(image_tilt_angle));
                temp_particle_info.defocus_1 = image_defocus_1 + tilt_based_height;
                temp_particle_info.defocus_2 = image_defocus_2 + tilt_based_height;

                //wxPrintf("axis = %f, angle = %f, height = %f\n", image_tilt_axis, image_tilt_angle, tilt_based_height);
            }

            temp_refinement_package->contained_particles.Add(temp_particle_info);

            for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].position_in_stack = counter + 1;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus1          = temp_particle_info.defocus_1;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus2          = temp_particle_info.defocus_2;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus_angle     = image_defocus_angle;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phase_shift       = image_phase_shift;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].logp              = 0.0;

                if ( temp_refinement_package->number_of_classes == 1 )
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
                else
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = fabsf(global_random_number_generator.GetUniformRandom( ) * (200.0f / float(temp_refinement_package->number_of_classes)));

                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phi                                = global_random_number_generator.GetUniformRandom( ) * 180.0;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].theta                              = rad_2_deg(acosf(2.0f * fabsf(global_random_number_generator.GetUniformRandom( )) - 1.0f));
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].psi                                = global_random_number_generator.GetUniformRandom( ) * 180.0;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].score                              = 0.0;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_is_active                    = 1;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].sigma                              = 1.0;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].pixel_size                         = current_image_asset->pixel_size;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].microscope_voltage_kv              = current_image_asset->microscope_voltage;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].microscope_spherical_aberration_mm = current_image_asset->spherical_aberration;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].amplitude_contrast                 = image_amplitude_contrast;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].beam_tilt_x                        = 0.0f;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].beam_tilt_y                        = 0.0f;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_shift_x                      = 0.0f;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_shift_y                      = 0.0f;
            }

            my_dialog->Update(counter + 1);
        }

        /*
		 * Now that we know about all the particles, we also know about all the micrographs
		 * and we can decide how to distribute particles between the two half datasets/maps
		 */
        {
            ArrayOfRefinmentPackageParticleInfos particle_info_buffer  = temp_refinement_package->contained_particles;
            wxArrayInt                           active_images         = ReturnIDsOfActiveImages(particle_info_buffer);
            int                                  number_of_micrographs = active_images.Count( );
            int                                  current_subset;

            if ( number_of_particles < 500 ) {
                // With so few particles, we have to go even/odd
                current_subset = 1;
                for ( counter = 0; counter < number_of_particles; counter++ ) {
                    temp_refinement_package->contained_particles[counter].assigned_subset = current_subset;
                    for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
                        temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].assigned_subset = current_subset;
                    }
                    if ( current_subset == 1 ) {
                        current_subset = 2;
                    }
                    else {
                        current_subset = 1;
                    }
                    my_dialog->Update(number_of_particles + counter);
                }
            }
            else if ( number_of_micrographs < 20 ) {
                // We have don't have many micrographs, so splitting by micrograph may not be safe
                // Let's just split the stack into ~10 chunks
                for ( counter = 0; counter < number_of_particles; counter++ ) {
                    if ( counter / (number_of_particles / 10) % 2 ) {
                        current_subset = 1;
                    }
                    else {
                        current_subset = 2;
                    }
                    temp_refinement_package->contained_particles[counter].assigned_subset = current_subset;
                    for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
                        temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].assigned_subset = current_subset;
                    }
                    my_dialog->Update(number_of_particles + counter);
                }
            }
            else {
                // We have enough particles and enough micrographs, so let's split by micrograph, since this guarantees that any duplicate particle picks won't mess with the FSC (though it means the half-datasets wil likely not be equal-sized)
                for ( counter = 0; counter < number_of_particles; counter++ ) {
                    // Let's assume that the parent image IDs are approximately random (e.g. the user didn't purposfully remove all odd-numbered image assets)
                    if ( temp_refinement_package->contained_particles[counter].parent_image_id % 2 ) {
                        current_subset = 1;
                    }
                    else {
                        current_subset = 2;
                    }
                    temp_refinement_package->contained_particles[counter].assigned_subset = current_subset;
                    for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
                        temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].assigned_subset = current_subset;
                    }
                    my_dialog->Update(number_of_particles + counter);
                }
            }
        } // end of work on assigned_subset

        my_dialog->Destroy( );
    }
    else if ( template_page->my_panel->GroupComboBox->GetSelection( ) == 1 ) // 2D Classums
    {
        long         number_of_particles;
        long         parent_refinement_package_id;
        long         parent_refinement_array_position;
        wxArrayFloat particle_total_shifts_squared;
        float        particle_x_shift;
        float        particle_y_shift;

        Classification* current_classification;

        ArrayOfRefinmentPackageParticleInfos class_average_particle_infos;
        wxArrayLong                          class_average_particle_parent_refinement_packages_array_position;

        OneSecondProgressDialog* my_progress_dialog = new OneSecondProgressDialog("Sorting out which contained particles to copy over", "Reading particles from database...", 100, this, wxPD_APP_MODAL);

        item = -1;
        for ( ;; ) {
            item = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
            if ( item == -1 )
                break;

            current_images.Clear( );

            // for each selection we need to extract out the relevant particle position assets..
            parent_refinement_package_id     = refinement_package_asset_panel->all_classification_selections.Item(item).refinement_package_asset_id;
            parent_refinement_array_position = refinement_package_asset_panel->ReturnArrayPositionFromAssetID(parent_refinement_package_id);
            parent_refinement_package_link   = &refinement_package_asset_panel->all_refinement_packages.Item(parent_refinement_array_position);
            parent_classification_id         = refinement_package_asset_panel->all_classification_selections.Item(item).classification_id;

            if ( recentre_picks_page->my_panel->ReCentreYesButton->GetValue( ) == true )
                current_classification = main_frame->current_project.database.GetClassificationByID(parent_classification_id);

            for ( counter = 0; counter < refinement_package_asset_panel->all_classification_selections.Item(item).selections.GetCount( ); counter++ ) {
                current_images = main_frame->current_project.database.Return2DClassMembers(parent_classification_id, refinement_package_asset_panel->all_classification_selections.Item(item).selections.Item(counter));
                my_progress_dialog->Pulse( );

                // copy out all the relevant particle positions

                for ( particle_counter = 0; particle_counter < current_images.GetCount( ); particle_counter++ ) {
                    class_average_particle_infos.Add(parent_refinement_package_link->ReturnParticleInfoByPositionInStack(current_images.Item(particle_counter)));
                    class_average_particle_parent_refinement_packages_array_position.Add(parent_refinement_array_position);

                    if ( recentre_picks_page->my_panel->ReCentreYesButton->GetValue( ) == true ) {

                        particle_x_shift = current_classification->ReturnXShiftByPositionInStack(current_images.Item(particle_counter));
                        particle_y_shift = current_classification->ReturnYShiftByPositionInStack(current_images.Item(particle_counter));

                        class_average_particle_infos[class_average_particle_infos.GetCount( ) - 1].x_pos -= particle_x_shift;
                        class_average_particle_infos[class_average_particle_infos.GetCount( ) - 1].y_pos -= particle_y_shift;

                        particle_total_shifts_squared.Add(powf(particle_x_shift, 2) + powf(particle_y_shift, 2));
                    }
                }
            }

            if ( recentre_picks_page->my_panel->ReCentreYesButton->GetValue( ) == true )
                delete current_classification;
        }

        if ( recentre_picks_page->my_panel->ReCentreYesButton->GetValue( ) == true && remove_duplicate_picks_page->my_panel->RemoveDuplicateYesButton->GetValue( ) == true ) // remove duplicates..
        {

            ArrayOfRefinmentPackageParticleInfos particle_info_buffer;
            particle_info_buffer                               = class_average_particle_infos;
            wxArrayInt                           active_images = ReturnIDsOfActiveImages(particle_info_buffer);
            ArrayOfRefinmentPackageParticleInfos particle_infos_for_current_image;
            ArrayOfRefinmentPackageParticleInfos current_duplicate_particles;
            wxArrayInt                           original_array_locations;
            int                                  second_counter;
            int                                  image_counter;
            int                                  number_removed             = 0;
            float                                threshold_distance_squared = powf(remove_duplicate_picks_threshold_page->my_panel->DuplicatePickThresholdTextCtrl->ReturnValue( ), 2);

            //ArrayOfRefinmentPackageParticleInfos duplicate_debug;

            // we are going to loop image by image, as duplicates must be on the same image, and there should probably never be enough particles on any one image to make this that slow..

            OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Refinement Package", "Removing Duplicates...", active_images.GetCount( ), this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);
            particle_info_buffer               = class_average_particle_infos;
            class_average_particle_infos.Clear( ); // we will fill it back up without duplicates.

            for ( image_counter = 0; image_counter < active_images.GetCount( ); image_counter++ ) {
                // extract all the infos in this image..
                particle_infos_for_current_image.Clear( );
                original_array_locations.Clear( );

                for ( particle_counter = 0; particle_counter < particle_info_buffer.GetCount( ); particle_counter++ ) {
                    if ( particle_info_buffer[particle_counter].parent_image_id == active_images[image_counter] ) {
                        particle_infos_for_current_image.Add(particle_info_buffer[particle_counter]);
                        original_array_locations.Add(particle_counter);
                    }
                }

                // ok remove all the duplicates..

                for ( particle_counter = 0; particle_counter < particle_infos_for_current_image.GetCount( ); particle_counter++ ) {
                    if ( particle_infos_for_current_image[particle_counter].parent_image_id == -1 )
                        continue; // can't do this if we don't have parent info..

                    for ( second_counter = particle_counter + 1; second_counter < particle_infos_for_current_image.GetCount( ); second_counter++ ) {
                        if ( particle_infos_for_current_image[second_counter].parent_image_id == -1 )
                            continue; // can't do this if we don't have parent info..

                        if ( powf(particle_infos_for_current_image[particle_counter].x_pos - particle_infos_for_current_image[second_counter].x_pos, 2) + powf(particle_infos_for_current_image[particle_counter].y_pos - particle_infos_for_current_image[second_counter].y_pos, 2) < threshold_distance_squared ) {
                            // so these two particles are too close.. which one has the smallest shift, we will keep that one as that was most likely to be the best original pick..
                            //			duplicate_debug.Add(particle_infos_for_current_image[particle_counter]);
                            //			duplicate_debug.Add(particle_infos_for_current_image[second_counter]);

                            if ( particle_total_shifts_squared[original_array_locations[particle_counter]] < particle_total_shifts_squared[original_array_locations[second_counter]] ) {
                                // keep the first one, should be a simple case of just deleting the second, subtract one from second counter to compensate.

                                particle_infos_for_current_image.RemoveAt(second_counter);
                                original_array_locations.RemoveAt(second_counter);
                                second_counter--;
                                number_removed++;
                            }
                            else {
                                // ok this is a bit more complicated, we want to keep the second...

                                particle_infos_for_current_image.RemoveAt(particle_counter);
                                original_array_locations.RemoveAt(particle_counter);
                                particle_counter--;
                                number_removed++;
                                break;
                            }
                        }
                    }
                }

                // ok, now we should have removed all duplicates, so add them back in to the original list.

                for ( particle_counter = 0; particle_counter < particle_infos_for_current_image.GetCount( ); particle_counter++ ) {
                    class_average_particle_infos.Add(particle_infos_for_current_image[particle_counter]);
                }

                my_dialog->Update(image_counter + 1);
            }

            // we need to add any particles with no parent image back in as well (i.e. they came from an imported stack, otherwise they won't have been included)

            for ( particle_counter = 0; particle_counter < particle_info_buffer.GetCount( ); particle_counter++ ) {
                if ( particle_info_buffer[particle_counter].parent_image_id == -1 )
                    class_average_particle_infos.Add(particle_info_buffer[particle_counter]);
            }

            /*
			class_average_particle_infos.Clear();
			for (particle_counter = 0; particle_counter < duplicate_debug.GetCount(); particle_counter++)
			{
				class_average_particle_infos.Add(duplicate_debug[particle_counter]);
			}
*/

            my_dialog->Destroy( );
            //wxPrintf("Removed a %i of %li picks\n", number_removed, particle_info_buffer.GetCount());

        } // end of test for recenter & remove duplicates

        number_of_particles = class_average_particle_infos.GetCount( );
        class_average_particle_infos.Sort(SortByParentImageID);
        my_progress_dialog->Destroy( );

        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Refinement Package", "Creating Refinement Package...", number_of_particles, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

        temp_refinement_package->name                     = wxString::Format("Refinement Package #%li", refinement_package_asset_panel->current_asset_number);
        temp_refinement_package->number_of_classes        = number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue( );
        temp_refinement_package->stack_has_white_protein  = false;
        temp_refinement_package->number_of_run_refinments = 0;

        temp_refinement.number_of_classes                = temp_refinement_package->number_of_classes;
        temp_refinement.number_of_particles              = number_of_particles;
        temp_refinement.name                             = "Random Parameters";
        temp_refinement.resolution_statistics_box_size   = box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( );
        temp_refinement.resolution_statistics_pixel_size = output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ReturnValue( );
        temp_refinement.refinement_package_asset_id      = refinement_package_asset_panel->current_asset_number + 1;

        long current_particle_parent_image_id = 0;
        long current_loaded_image_id          = -1;
        long position_in_stack                = 0;

        int current_x_pos;
        int current_y_pos;

        float average_value_at_edges;
        float image_defocus_1;
        float image_defocus_2;
        float image_defocus_angle;
        float image_phase_shift;
        float image_amplitude_contrast;
        float image_tilt_angle;
        float image_tilt_axis;

        // for tilt calculations
        AnglesAndShifts rotation_angle;
        float           image_x_position;
        float           image_y_position;
        float           x_rotated;
        float           y_rotated;
        float           tilt_based_height;

        long    currently_opened_refinement_package_particle_stack = -1;
        MRCFile input_stack_file;

        ImageAsset*            current_image_asset             = NULL;
        ParticlePositionAsset* current_particle_position_asset = NULL;

        Image stack_image; // in case we are reading from a stack
        Image current_image; // for reading images to cut from
        Image cut_particle; // the cut particle

        wxFileName output_stack_filename = main_frame->current_project.particle_stack_directory.GetFullPath( ) + wxString::Format("/particle_stack_%li.mrc", refinement_package_asset_panel->current_asset_number);

        // specific package setup..

        temp_refinement_package->stack_box_size    = box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( );
        temp_refinement_package->output_pixel_size = output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ReturnValue( );
        temp_refinement_package->stack_filename    = output_stack_filename.GetFullPath( );
        temp_refinement_package->symmetry          = symmetry_page->my_panel->SymmetryComboBox->GetValue( ).Upper( );
        ;
        temp_refinement_package->estimated_particle_weight_in_kda     = molecular_weight_page->my_panel->MolecularWeightTextCtrl->ReturnValue( );
        temp_refinement_package->estimated_particle_size_in_angstroms = largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( );

        // setup the 3ds

        wxWindowList            all_children = initial_reference_page->my_panel->ScrollWindow->GetChildren( );
        ClassVolumeSelectPanel* panel_pointer;

        for ( counter = 0; counter < all_children.GetCount( ); counter++ ) {
            if ( all_children.Item(counter)->GetData( )->GetClassInfo( )->GetClassName( ) == wxString("wxPanel") ) {
                panel_pointer = reinterpret_cast<ClassVolumeSelectPanel*>(all_children.Item(counter)->GetData( ));

                if ( panel_pointer->VolumeComboBox->GetSelection( ) == 0 ) {
                    temp_refinement_package->references_for_next_refinement.Add(-1);
                }
                else {
                    temp_refinement_package->references_for_next_refinement.Add(volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(panel_pointer->VolumeComboBox->GetSelection( ) - 1)->asset_id);
                }
            }
        }

        // size the box..

        cut_particle.Allocate(box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( ), box_size_page->my_panel->BoxSizeSpinCtrl->GetValue( ), 1);

        // open the output stack

        MRCFile output_stack(output_stack_filename.GetFullPath( ).ToStdString( ), true);
#ifdef USE_FP16_PARTICLE_STACKS
        output_stack.SetOutputToFP16( );
#endif
        // setup the refinement..

        long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
        temp_refinement_package->refinement_ids.Add(refinement_id);

        temp_refinement.refinement_id                       = refinement_id;
        temp_refinement.resolution_statistics_are_generated = true;
        temp_refinement.SizeAndFillWithEmpty(number_of_particles, temp_refinement_package->number_of_classes);

        for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
            temp_refinement.class_refinement_results[class_counter].average_occupancy = 100.0 / temp_refinement_package->number_of_classes;
        }

        for ( counter = 0; counter < number_of_particles; counter++ ) {
            // current particle, what image is it from?

            current_particle_parent_image_id = class_average_particle_infos.Item(counter).parent_image_id;

            if ( current_particle_parent_image_id == -1 ) // we don't have a parent image, we will have to use the stack image..
            {
                if ( currently_opened_refinement_package_particle_stack != class_average_particle_parent_refinement_packages_array_position[counter] ) {
                    input_stack_file.OpenFile(refinement_package_asset_panel->all_refinement_packages[class_average_particle_parent_refinement_packages_array_position[counter]].stack_filename.ToStdString( ), false);
                    currently_opened_refinement_package_particle_stack = class_average_particle_parent_refinement_packages_array_position[counter];
                }

                stack_image.ReadSlice(&input_stack_file, class_average_particle_infos[counter].position_in_stack);

                if ( refinement_package_asset_panel->all_refinement_packages[class_average_particle_parent_refinement_packages_array_position[counter]].stack_has_white_protein == true ) {
                    // invert..

                    stack_image.InvertRealValues( );
                    stack_image.ZeroFloatAndNormalize( );
                }

                // pad into the new box..

                position_in_stack++;

                stack_image.ClipInto(&cut_particle, stack_image.ReturnAverageOfRealValuesOnEdges( ));

                temp_particle_info.spherical_aberration = class_average_particle_infos[counter].spherical_aberration;
                temp_particle_info.microscope_voltage   = class_average_particle_infos[counter].microscope_voltage;
                temp_particle_info.parent_image_id      = -1;
                temp_particle_info.pixel_size           = class_average_particle_infos[counter].pixel_size;
                temp_particle_info.position_in_stack    = position_in_stack;
                temp_particle_info.defocus_1            = class_average_particle_infos[counter].defocus_1;
                temp_particle_info.defocus_2            = class_average_particle_infos[counter].defocus_2;
                temp_particle_info.defocus_angle        = class_average_particle_infos[counter].defocus_angle;
                temp_particle_info.phase_shift          = class_average_particle_infos[counter].phase_shift;
                temp_particle_info.amplitude_contrast   = class_average_particle_infos[counter].amplitude_contrast;
                temp_particle_info.assigned_subset      = class_average_particle_infos[counter].assigned_subset;

                // fill in the refinement data..

                for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].position_in_stack = counter + 1;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus1          = class_average_particle_infos[counter].defocus_1;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus2          = class_average_particle_infos[counter].defocus_2;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus_angle     = class_average_particle_infos[counter].defocus_angle;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phase_shift       = class_average_particle_infos[counter].phase_shift;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].logp              = 0.0;

                    if ( temp_refinement_package->number_of_classes == 1 )
                        temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
                    else
                        temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = fabsf(global_random_number_generator.GetUniformRandom( ) * (200.0f / float(temp_refinement_package->number_of_classes)));

                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phi                                = global_random_number_generator.GetUniformRandom( ) * 180.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].theta                              = rad_2_deg(acosf(2.0f * fabsf(global_random_number_generator.GetUniformRandom( )) - 1.0f));
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].psi                                = global_random_number_generator.GetUniformRandom( ) * 180.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].score                              = 0.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_is_active                    = 1;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].sigma                              = 1.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].pixel_size                         = class_average_particle_infos[counter].pixel_size;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].microscope_voltage_kv              = class_average_particle_infos[counter].microscope_voltage;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].microscope_spherical_aberration_mm = class_average_particle_infos[counter].spherical_aberration;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].amplitude_contrast                 = class_average_particle_infos[counter].amplitude_contrast;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].beam_tilt_x                        = 0.0f;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].beam_tilt_y                        = 0.0f;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_shift_x                      = 0.0f;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_shift_y                      = 0.0f;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].assigned_subset                    = class_average_particle_infos[counter].assigned_subset;
                }
            }
            else // we can cut it out of the image..
            {
                if ( current_loaded_image_id != current_particle_parent_image_id ) {
                    // load it..

                    current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(current_particle_parent_image_id));
                    current_image.QuickAndDirtyReadSlice(current_image_asset->filename.GetFullPath( ).ToStdString( ), 1);
                    if ( current_image_asset->protein_is_white )
                        current_image.InvertRealValues( );
                    current_image.ReplaceOutliersWithMean(6);
                    current_loaded_image_id = current_particle_parent_image_id;
                    average_value_at_edges  = current_image.ReturnAverageOfRealValuesOnEdges( );

                    // we have to get the defocus stuff from the database..

                    main_frame->current_project.database.GetActiveDefocusValuesByImageID(current_particle_parent_image_id, image_defocus_1, image_defocus_2, image_defocus_angle, image_phase_shift, image_amplitude_contrast, image_tilt_angle, image_tilt_axis);
                }

                // do the cutting..

                position_in_stack++;

                current_x_pos = myround(class_average_particle_infos.Item(counter).x_pos / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_x;
                current_y_pos = myround(class_average_particle_infos.Item(counter).y_pos / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_y;
                current_image.ClipInto(&cut_particle, average_value_at_edges, false, 1.0, current_x_pos, current_y_pos, 0);

                temp_particle_info.spherical_aberration = current_image_asset->spherical_aberration;
                temp_particle_info.microscope_voltage   = current_image_asset->microscope_voltage;
                temp_particle_info.parent_image_id      = current_particle_parent_image_id;
                temp_particle_info.pixel_size           = current_image_asset->pixel_size;
                temp_particle_info.position_in_stack    = position_in_stack;
                temp_particle_info.defocus_angle        = image_defocus_angle;
                temp_particle_info.phase_shift          = image_phase_shift;
                temp_particle_info.amplitude_contrast   = image_amplitude_contrast;
                temp_particle_info.assigned_subset      = class_average_particle_infos[counter].assigned_subset;

                if ( image_tilt_angle == 0.0f && image_tilt_axis == 0.0f ) {
                    temp_particle_info.defocus_1 = image_defocus_1;
                    temp_particle_info.defocus_2 = image_defocus_2;
                }
                else // calculate a defocus value based on tilt..
                {
                    rotation_angle.GenerateRotationMatrix2D(image_tilt_axis);
                    image_x_position = current_x_pos * current_image_asset->pixel_size;
                    image_y_position = current_y_pos * current_image_asset->pixel_size;

                    rotation_angle.euler_matrix.RotateCoords2D(image_x_position, image_y_position, x_rotated, y_rotated);
                    tilt_based_height            = y_rotated * tanf(deg_2_rad(image_tilt_angle));
                    temp_particle_info.defocus_1 = image_defocus_1 + tilt_based_height;
                    temp_particle_info.defocus_2 = image_defocus_2 + tilt_based_height;
                }

                // fill in the refinement data..

                for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].position_in_stack = counter + 1;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus1          = temp_particle_info.defocus_1;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus2          = temp_particle_info.defocus_2;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus_angle     = image_defocus_angle;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phase_shift       = image_phase_shift;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].logp              = 0.0;

                    if ( temp_refinement_package->number_of_classes == 1 )
                        temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
                    else
                        temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = fabsf(global_random_number_generator.GetUniformRandom( ) * (200.0f / float(temp_refinement_package->number_of_classes)));

                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phi = global_random_number_generator.GetUniformRandom( ) * 180.0;
                    //temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].theta = global_random_number_generator.GetUniformRandom() * 180.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].theta                              = rad_2_deg(acosf(2.0f * fabsf(global_random_number_generator.GetUniformRandom( )) - 1.0f));
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].psi                                = global_random_number_generator.GetUniformRandom( ) * 180.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].score                              = 0.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_is_active                    = 1;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].sigma                              = 1.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].sigma                              = 1.0;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].pixel_size                         = current_image_asset->pixel_size;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].microscope_voltage_kv              = current_image_asset->microscope_voltage;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].microscope_spherical_aberration_mm = current_image_asset->spherical_aberration;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].amplitude_contrast                 = image_amplitude_contrast;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].beam_tilt_x                        = 0.0f;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].beam_tilt_y                        = 0.0f;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_shift_x                      = 0.0f;
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].image_shift_y                      = 0.0f;

                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].assigned_subset = class_average_particle_infos[counter].assigned_subset;
                }
            }

            cut_particle.ZeroFloatAndNormalize( );
            cut_particle.WriteSlice(&output_stack, position_in_stack);

            // set the contained particles..

            temp_particle_info.x_pos                               = class_average_particle_infos.Item(counter).x_pos;
            temp_particle_info.y_pos                               = class_average_particle_infos.Item(counter).y_pos;
            temp_particle_info.original_particle_position_asset_id = class_average_particle_infos.Item(counter).original_particle_position_asset_id;

            temp_refinement_package->contained_particles.Add(temp_particle_info);
            my_dialog->Update(counter + 1);
        }

        my_dialog->Destroy( );
    }
    else // based on previous refinement package
    {

        int   best_class;
        int   input_class_counter;
        float highest_occupancy;

        RefinementPackage* template_refinement_package = &refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
        Refinement*        refinement_to_copy          = main_frame->current_project.database.GetRefinementByID(refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection( ) - 3].refinement_ids[parameter_page->my_panel->GroupComboBox->GetSelection( )]);

        // this stuff should always apply..

        temp_refinement_package->name                     = wxString::Format("Refinement Package #%li", refinement_package_asset_panel->current_asset_number);
        temp_refinement_package->number_of_classes        = number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue( );
        temp_refinement_package->number_of_run_refinments = 0;
        temp_refinement.resolution_statistics_box_size    = template_refinement_package->stack_box_size;
        temp_refinement.resolution_statistics_pixel_size  = output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ReturnValue( );

        temp_refinement_package->stack_box_size                       = template_refinement_package->stack_box_size;
        temp_refinement_package->output_pixel_size                    = output_pixel_size_page->my_panel->OutputPixelSizeTextCtrl->ReturnValue( );
        temp_refinement_package->stack_has_white_protein              = template_refinement_package->stack_has_white_protein;
        temp_refinement_package->symmetry                             = symmetry_page->my_panel->SymmetryComboBox->GetValue( ).Upper( );
        temp_refinement_package->estimated_particle_weight_in_kda     = molecular_weight_page->my_panel->MolecularWeightTextCtrl->ReturnValue( );
        temp_refinement_package->estimated_particle_size_in_angstroms = largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue( );

        long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID( ) + 1;
        temp_refinement_package->refinement_ids.Add(refinement_id);
        temp_refinement.refinement_id     = refinement_id;
        temp_refinement.number_of_classes = temp_refinement_package->number_of_classes;

        temp_refinement.name                        = "Starting Parameters";
        temp_refinement.refinement_package_asset_id = refinement_package_asset_panel->current_asset_number + 1;

        // initial references..
        wxWindowList all_children = initial_reference_page->my_panel->ScrollWindow->GetChildren( );

        ClassVolumeSelectPanel* panel_pointer;
        for ( counter = 0; counter < all_children.GetCount( ); counter++ ) {
            if ( all_children.Item(counter)->GetData( )->GetClassInfo( )->GetClassName( ) == wxString("wxPanel") ) {
                panel_pointer = reinterpret_cast<ClassVolumeSelectPanel*>(all_children.Item(counter)->GetData( ));

                if ( panel_pointer->VolumeComboBox->GetSelection( ) == 0 ) {
                    temp_refinement_package->references_for_next_refinement.Add(-1);
                }
                else {
                    temp_refinement_package->references_for_next_refinement.Add(volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(panel_pointer->VolumeComboBox->GetSelection( ) - 1)->asset_id);
                }
            }
        }

        // lets make a list of the particles we are going to take

        wxArrayLong particles_to_take;

        if ( class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == true || template_refinement_package->number_of_classes == 1 ) // All particles
        {
            for ( particle_counter = 0; particle_counter < template_refinement_package->contained_particles.GetCount( ); particle_counter++ ) {
                particles_to_take.Add(particle_counter);
            }
        }
        else // Selection
        {
            // which classes are selected to carry over?

            wxArrayBool is_class_selected = class_setup_pageB->ReturnSelectedClasses( );

            for ( particle_counter = 0; particle_counter < template_refinement_package->contained_particles.GetCount( ); particle_counter++ ) {
                // work out which class has the highest occupancy, then check if that class is selected to carry particles over

                best_class        = 0;
                highest_occupancy = -FLT_MAX;

                for ( class_counter = 0; class_counter < refinement_to_copy->number_of_classes; class_counter++ ) {
                    //wxPrintf("class counter = %i, particle_counter = %li\n", class_counter, particle_counter);
                    //if (refinement_to_copy->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy > highest_occupancy)
                    if ( refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(class_counter, template_refinement_package->contained_particles[particle_counter].position_in_stack).occupancy > highest_occupancy ) {
                        highest_occupancy = refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(class_counter, template_refinement_package->contained_particles[particle_counter].position_in_stack).occupancy;
                        best_class        = class_counter;
                    }
                }

                if ( is_class_selected[best_class] == true )
                    particles_to_take.Add(particle_counter);
            }
        }

        long number_of_particles            = particles_to_take.GetCount( );
        temp_refinement.number_of_particles = number_of_particles;
        OneSecondProgressDialog* my_dialog  = new OneSecondProgressDialog("Refinement Package", "Creating Refinement Package...", number_of_particles, this);
        temp_refinement.SizeAndFillWithEmpty(number_of_particles, temp_refinement.number_of_classes);

        MRCFile* input_stack;
        MRCFile* output_stack;
        Image    image_for_new_stack;

        if ( class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == true ) // taking over all particles, don't need to make a new stack
        {
            temp_refinement_package->stack_filename = template_refinement_package->stack_filename;
        }
        else // we are going to make a new stack..
        {
            wxFileName output_stack_filename        = main_frame->current_project.particle_stack_directory.GetFullPath( ) + wxString::Format("/particle_stack_%li.mrc", refinement_package_asset_panel->current_asset_number);
            temp_refinement_package->stack_filename = output_stack_filename.GetFullPath( );

            // open the input/output stack

            input_stack  = new MRCFile(template_refinement_package->stack_filename.ToStdString( ), false);
            output_stack = new MRCFile(output_stack_filename.GetFullPath( ).ToStdString( ), true);
        }

        for ( particle_counter = 0; particle_counter < number_of_particles; particle_counter++ ) {
            temp_particle_info = template_refinement_package->contained_particles[particles_to_take[particle_counter]];

            if ( class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false )
                temp_particle_info.position_in_stack = particle_counter + 1;

            temp_refinement_package->contained_particles.Add(temp_particle_info);

            // do we have to write to a new stack?

            if ( class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false ) // yes we do
            {
                image_for_new_stack.ReadSlice(input_stack, template_refinement_package->contained_particles[particles_to_take[particle_counter]].position_in_stack);
                image_for_new_stack.WriteSlice(output_stack, particle_counter + 1);
            }

            for ( class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++ ) {
                RefinementResult active_result;

                // set the active result for this class..

                if ( template_refinement_package->number_of_classes == 1 )
                    active_result = refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(0, template_refinement_package->contained_particles[particles_to_take[particle_counter]].position_in_stack); //&refinement_to_copy->class_refinement_results[0].particle_refinement_results[particles_to_take[particle_counter]]; // only option
                else {
                    // so does this class have more than one input
                    wxArrayInt selected_input_classes = class_setup_pageC->ReturnReferencesForClass(class_counter);

                    if ( selected_input_classes.GetCount( ) == 1 ) // there is only one class, so easy..
                    {
                        //active_result = &refinement_to_copy->class_refinement_results[selected_input_classes[0]].particle_refinement_results[particles_to_take[particle_counter]];
                        active_result = refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(selected_input_classes[0], template_refinement_package->contained_particles[particles_to_take[particle_counter]].position_in_stack);
                    }
                    else // so we have multiple classes, are we taking best occupancy or random?
                    {
                        if ( class_setup_pageD->my_panel->BestOccupancyRadioButton->GetValue( ) == true ) // best_occupancy
                        {
                            highest_occupancy = -FLT_MAX;

                            for ( input_class_counter = 0; input_class_counter < selected_input_classes.GetCount( ); input_class_counter++ ) {
                                //if (refinement_to_copy->class_refinement_results[selected_input_classes[input_class_counter]].particle_refinement_results[particles_to_take[particle_counter]].occupancy > highest_occupancy)
                                if ( refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(selected_input_classes[input_class_counter], template_refinement_package->contained_particles[particles_to_take[particle_counter]].position_in_stack).occupancy > highest_occupancy ) {
                                    //highest_occupancy = refinement_to_copy->class_refinement_results[selected_input_classes[input_class_counter]].particle_refinement_results[particles_to_take[particle_counter]].occupancy;
                                    highest_occupancy = refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(selected_input_classes[input_class_counter], template_refinement_package->contained_particles[particles_to_take[particle_counter]].position_in_stack).occupancy;
                                    best_class        = selected_input_classes[input_class_counter];
                                }
                            }

                            //active_result = &refinement_to_copy->class_refinement_results[best_class].particle_refinement_results[particles_to_take[particle_counter]];
                            active_result = refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(best_class, template_refinement_package->contained_particles[particles_to_take[particle_counter]].position_in_stack);
                        }
                        else // random
                        {
                            // choose a random class

                            //active_result = &refinement_to_copy->class_refinement_results[selected_input_classes[myroundint(fabsf(global_random_number_generator.GetUniformRandom() * selected_input_classes.GetCount()))]].particle_refinement_results[particles_to_take[particle_counter]];
                            int current_class = selected_input_classes[myroundint(fabsf(global_random_number_generator.GetUniformRandom( ) * (selected_input_classes.GetCount( ) - 1)))];
                            refinement_to_copy->ReturnRefinementResultByClassAndPositionInStack(current_class, template_refinement_package->contained_particles[particles_to_take[particle_counter]].position_in_stack);
                        }
                    }
                }

                if ( class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false )
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].position_in_stack = particle_counter + 1;
                else
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].position_in_stack = active_result.position_in_stack;

                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus1      = active_result.defocus1;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus2      = active_result.defocus2;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].defocus_angle = active_result.defocus_angle;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].phase_shift   = active_result.phase_shift;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp          = active_result.logp;

                // did the user select randomise occupancies?

                if ( class_setup_pageE->my_panel->RandomiseOccupanciesRadioButton->GetValue( ) == true && temp_refinement_package->number_of_classes > 1 ) {
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy = fabsf(global_random_number_generator.GetUniformRandom( ) * (200.0f / float(temp_refinement_package->number_of_classes)));
                }
                else {
                    temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy = 100.0;
                }

                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].phi                                = active_result.phi;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].theta                              = active_result.theta;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].psi                                = active_result.psi;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].xshift                             = active_result.xshift;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].yshift                             = active_result.yshift;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].score                              = active_result.score;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_is_active                    = active_result.image_is_active;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma                              = active_result.sigma;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].pixel_size                         = active_result.pixel_size;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].microscope_voltage_kv              = active_result.microscope_voltage_kv;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = active_result.microscope_spherical_aberration_mm;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].amplitude_contrast                 = active_result.amplitude_contrast;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].beam_tilt_x                        = active_result.beam_tilt_x;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].beam_tilt_y                        = active_result.beam_tilt_y;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_shift_x                      = active_result.image_shift_x;
                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].image_shift_y                      = active_result.image_shift_y;

                temp_refinement.class_refinement_results[class_counter].particle_refinement_results[particle_counter].assigned_subset = active_result.assigned_subset;
            }

            my_dialog->Update(particle_counter + 1);
        }

        my_dialog->Destroy( );
        delete refinement_to_copy;

        if ( class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false ) {
            delete input_stack;
            delete output_stack;
        }
    }

    // now we should have a refinement package and refinement all generated, we just need to add them to the panels / database

    main_frame->current_project.database.Begin( );
    refinement_package_asset_panel->AddAsset(temp_refinement_package);

    for ( class_counter = 0; class_counter < temp_refinement.number_of_classes; class_counter++ ) {
        temp_refinement.class_refinement_results[class_counter].class_resolution_statistics.Init(temp_refinement_package->output_pixel_size, temp_refinement.resolution_statistics_box_size);
        temp_refinement.class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(temp_refinement_package->estimated_particle_weight_in_kda);
    }

    //wxPrintf("Ref ID = %li\n", temp_refinement.reference_volume_ids.Item(0));
    main_frame->current_project.database.AddRefinement(&temp_refinement);

    ArrayofAngularDistributionHistograms all_histograms = temp_refinement.ReturnAngularDistributions(temp_refinement_package->symmetry);
    for ( class_counter = 1; class_counter <= temp_refinement.number_of_classes; class_counter++ ) {
        main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[class_counter - 1], temp_refinement.refinement_id, class_counter);
    }

    ShortRefinementInfo temp_info;
    temp_info = temp_refinement;

    refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);
    main_frame->current_project.database.Commit( );
    //delete temp_refinement_package;
}

////////////////

// TEMPLATE PAGE

/////////////////

TemplateWizardPage::TemplateWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    Freeze( );
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new TemplateWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
    // my_panel->InfoText->AutoWrap();

    // my_panel->GroupComboBox->Freeze();
    my_panel->GroupComboBox->ChangeValue("");
    my_panel->GroupComboBox->Clear( );

    my_panel->GroupComboBox->Append("New Refinement Package");
    my_panel->GroupComboBox->Append("Create From 2D Class Average Selection");
    my_panel->GroupComboBox->Append("--------------------------------------------------------");

    for ( int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount( ); counter++ ) {
        my_panel->GroupComboBox->Append(refinement_package_asset_panel->all_refinement_packages.Item(counter).name);
    }

    /*long first_group_to_include = 0;
	if (!include_all_images_group) first_group_to_include = 1;

	for (long counter = first_group_to_include; counter < image_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		GroupComboBox->Append(image_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), image_asset_panel->ReturnGroupSize(counter)) + ")");

	}

	if (GroupComboBox->GetCount() > 0) GroupComboBox->SetSelection(0);*/

    my_panel->GroupComboBox->SetSelection(0);
    //my_panel->GroupComboBox->Thaw();
    Thaw( );
    /*int width, height;
	my_panel->GetClientSize(&width, &height);
	my_panel->InfoText->Wrap(height);
	my_panel->Fit();*/

    my_panel->GroupComboBox->Bind(wxEVT_COMBOBOX, wxCommandEventHandler(TemplateWizardPage::RefinementPackageChanged), this);
}

TemplateWizardPage::~TemplateWizardPage( ) {
    //my_panel->GroupComboBox->Unbind(wxEVT_COMBOBOX, wxComboEventHandler( TemplateWizardPage::RefinementPackageChanged), this);
}

void TemplateWizardPage::RefinementPackageChanged(wxCommandEvent& event) {
    wizard_pointer->class_setup_pageB->my_panel->ClassListCtrl->ClearAll( );
    wizard_pointer->class_setup_pageC->current_class_selections.Clear( );
}

wxWizardPage* TemplateWizardPage::GetNext( ) const {
    // wxPrintf("Template Next\n");
    if ( my_panel->GroupComboBox->GetSelection( ) == 0 )
        return wizard_pointer->particle_group_page;
    else if ( my_panel->GroupComboBox->GetSelection( ) == 1 )
        return wizard_pointer->class_selection_page;
    else
        return wizard_pointer->parameter_page;
}

////////////////

// INPUT PARAMETERS PAGE

/////////////////

InputParameterWizardPage::InputParameterWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    Freeze( );
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new InputParameterWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    my_panel->GroupComboBox->ChangeValue("");
    my_panel->GroupComboBox->Clear( );

    Thaw( );
}

wxWizardPage* InputParameterWizardPage::GetPrev( ) const {
    return wizard_pointer->template_page;
}

wxWizardPage* InputParameterWizardPage::GetNext( ) const {
    // wxPrintf("Template Next\n");

    return wizard_pointer->molecular_weight_page;
}

//////////////////////

// Particle Group Page

//////////////////////

ParticleGroupWizardPage::ParticleGroupWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new ParticleGroupWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
    //	my_panel->InfoText->AutoWrap();

    FillParticlePositionsGroupComboBox(my_panel->ParticlePositionsGroupComboBox);
}

wxWizardPage* ParticleGroupWizardPage::GetPrev( ) const {
    // wxPrintf("Particle Prev\n");
    return wizard_pointer->template_page;
}

wxWizardPage* ParticleGroupWizardPage::GetNext( ) const {
    // wxPrintf("Particle Next\n");
    //return wizard_pointer->box_size_page;
    return wizard_pointer->molecular_weight_page;
}

//////////////////////////

// BOX SIZE PAGE

////////////////////////////

BoxSizeWizardPage::BoxSizeWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new BoxSizeWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
    //my_panel->InfoText->AutoWrap();
}

wxWizardPage* BoxSizeWizardPage::GetPrev( ) const {
    //if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() == 0) return wizard_pointer->particle_group_page;
    //else
    //return wizard_pointer->class_selection_page;

    // if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() != 1) return wizard_pointer->largest_dimension_page;
    //else
    //if (wizard_pointer->recentre_picks_page->my_panel->ReCentreYesButton->GetValue() == false) return wizard_pointer->recentre_picks_page;
    //else
    //if (wizard_pointer->remove_duplicate_picks_page->my_panel->RemoveDuplicateYesButton->GetValue() == false) return wizard_pointer->remove_duplicate_picks_page;
    //else return wizard_pointer->remove_duplicate_picks_threshold_page;

    return wizard_pointer->symmetry_page;
}

wxWizardPage* BoxSizeWizardPage::GetNext( ) const {
    //  wxPrintf("Box Next\n");
    return wizard_pointer->output_pixel_size_page;
}

//////////////////////////

// Output pixel size page

////////////////////////////

OutputPixelSizeWizardPage::OutputPixelSizeWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new OutputPixelSizeWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    my_panel->OutputPixelSizeTextCtrl->SetPrecision(4);
    //my_panel->InfoText->AutoWrap();
}

wxWizardPage* OutputPixelSizeWizardPage::GetPrev( ) const {
    //if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() == 0) return wizard_pointer->particle_group_page;
    //else
    //return wizard_pointer->class_selection_page;

    /*if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() != 1) return wizard_pointer->largest_dimension_page;
 	  else
 	  if (wizard_pointer->recentre_picks_page->my_panel->ReCentreYesButton->GetValue() == false) return wizard_pointer->recentre_picks_page;
 	  else
 	  if (wizard_pointer->remove_duplicate_picks_page->my_panel->RemoveDuplicateYesButton->GetValue() == false) return wizard_pointer->remove_duplicate_picks_page;
 	  else return wizard_pointer->remove_duplicate_picks_threshold_page;*/

    if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) > 1 )
        return wizard_pointer->symmetry_page;
    else
        return wizard_pointer->box_size_page;
}

wxWizardPage* OutputPixelSizeWizardPage::GetNext( ) const {
    if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) == 1 )
        return wizard_pointer->recentre_picks_page;
    else
        return wizard_pointer->number_of_classes_page;
}

//////////////////////////

// Molecular weight PAGE

////////////////////////////

MolecularWeightWizardPage::MolecularWeightWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new MolecularWeightWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
    //	my_panel->InfoText->AutoWrap();
}

wxWizardPage* MolecularWeightWizardPage::GetPrev( ) const {
    //  wxPrintf("Box Prev\n");
    //	   if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() > 1) return wizard_pointer->parameter_page;
    //   else return wizard_pointer->box_size_page;
    if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) > 1 )
        return wizard_pointer->parameter_page;
    else if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) == 1 )
        return wizard_pointer->class_selection_page;
    else
        return wizard_pointer->particle_group_page;
}

wxWizardPage* MolecularWeightWizardPage::GetNext( ) const {
    //  wxPrintf("Box Next\n");
    return wizard_pointer->largest_dimension_page;
}

//////////////////////////

// largest dimension PAGE

////////////////////////////

LargestDimensionWizardPage::LargestDimensionWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new LargestDimensionWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
    //	my_panel->InfoText->AutoWrap();
}

wxWizardPage* LargestDimensionWizardPage::GetPrev( ) const {
    //  wxPrintf("Box Prev\n");
    return wizard_pointer->molecular_weight_page;
}

wxWizardPage* LargestDimensionWizardPage::GetNext( ) const {
    /*if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() == 1) return wizard_pointer->recentre_picks_page;
    	else
  	   	if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() > 1) return wizard_pointer->symmetry_page;
    	else return wizard_pointer->output_pixel_size_page;
    	*/

    return wizard_pointer->symmetry_page;
}

//////////////////////////

// symmetry PAGE

////////////////////////////

SymmetryWizardPage::SymmetryWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new SymmetryWizardPanel(this);
    my_panel->SymmetryComboBox->Append("C1");
    my_panel->SymmetryComboBox->Append("C2");
    my_panel->SymmetryComboBox->Append("C3");
    my_panel->SymmetryComboBox->Append("C4");
    my_panel->SymmetryComboBox->Append("D2");
    my_panel->SymmetryComboBox->Append("D3");
    my_panel->SymmetryComboBox->Append("D4");
    my_panel->SymmetryComboBox->Append("I");
    my_panel->SymmetryComboBox->Append("I2");
    my_panel->SymmetryComboBox->Append("O");
    my_panel->SymmetryComboBox->Append("T");
    my_panel->SymmetryComboBox->Append("T2");
    my_panel->SymmetryComboBox->ChangeValue("0");

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
    //	my_panel->InfoText->AutoWrap();
}

wxWizardPage* SymmetryWizardPage::GetPrev( ) const {
    //  wxPrintf("Box Prev\n");
    return wizard_pointer->largest_dimension_page;
}

wxWizardPage* SymmetryWizardPage::GetNext( ) const {
    //  wxPrintf("Box Next\n");
    //   	 return wizard_pointer->number_of_classes_page;

    if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) > 1 )
        return wizard_pointer->output_pixel_size_page;
    else
        return wizard_pointer->box_size_page;
}

/////////////////////////

//  NUMBER OF CLASSES PAGE

/////////////////////////////

NumberofClassesWizardPage::NumberofClassesWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new NumberofClassesWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
    my_panel->InfoText->AutoWrap( );

    my_panel->NumberOfClassesSpinCtrl->Bind(wxEVT_SPINCTRL, wxSpinEventHandler(NumberofClassesWizardPage::NumberClassesChanged), this);
}

NumberofClassesWizardPage::~NumberofClassesWizardPage( ) {
    my_panel->NumberOfClassesSpinCtrl->Unbind(wxEVT_SPINCTRL, wxSpinEventHandler(NumberofClassesWizardPage::NumberClassesChanged), this);
}

void NumberofClassesWizardPage::NumberClassesChanged(wxSpinEvent& event) {
    wizard_pointer->class_setup_pageC->current_class_selections.Clear( );
    wizard_pointer->class_setup_pageB->my_panel->ClassListCtrl->ClearAll( );
}

wxWizardPage* NumberofClassesWizardPage::GetNext( ) const {
    //if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() == 0) return NULL;
    //else return wizard_pointer->class_setup_page;

    // wxPrintf("Number classes Next\n");
    if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) > 1 ) {
        RefinementPackage* input_package = &refinement_package_asset_panel->all_refinement_packages.Item(wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) - 3);

        if ( input_package->number_of_classes == 1 ) // if there is only 1 input class, there is no fancy class setup so we can just skip to initial references
        {
            return wizard_pointer->initial_reference_page;
        }
        else // we need to know how to setup the classes
        {
            return wizard_pointer->class_setup_pageA;
        }
    }
    else
        return wizard_pointer->initial_reference_page;
}

wxWizardPage* NumberofClassesWizardPage::GetPrev( ) const {
    if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) == 1 ) {
        if ( wizard_pointer->remove_duplicate_picks_page->my_panel->RemoveDuplicateYesButton->GetValue( ) == false )
            return wizard_pointer->remove_duplicate_picks_page;
        else
            return wizard_pointer->remove_duplicate_picks_threshold_page;
    }
    else
        return wizard_pointer->output_pixel_size_page;
}

// ///////////////////////////

// INITIAL REFERNECES
/////

InitialReferencesWizardPage::InitialReferencesWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    CreatePanel( );

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Add(my_panel);
    main_sizer->Fit(this);
    //my_panel->InfoText->AutoWrap();
}

void InitialReferencesWizardPage::CreatePanel( ) {
    my_panel = new InitialReferenceSelectWizardPanel(this);
}

wxWizardPage* InitialReferencesWizardPage::GetNext( ) const {
    //	 wxPrintf("Initial Next\n");
    return NULL;
}

wxWizardPage* InitialReferencesWizardPage::GetPrev( ) const {
    // wxPrintf("Initial Prev\n");
    if ( wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) > 1 ) {
        RefinementPackage* input_package = &refinement_package_asset_panel->all_refinement_packages.Item(wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
        if ( input_package->number_of_classes == 1 )
            return wizard_pointer->number_of_classes_page;
        else
            return wizard_pointer->class_setup_pageE;
    }
    else
        return wizard_pointer->number_of_classes_page;
}

////////////////////////////

//  Classes Selection Page

/////////////////////////////

ClassSelectionWizardPage::ClassSelectionWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new ClassSelectionWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    int counter;
    int old_width;
    int current_width;

    Freeze( );

    my_panel->SelectionListCtrl->ClearAll( );
    my_panel->SelectionListCtrl->InsertColumn(0, wxT("Selection"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    my_panel->SelectionListCtrl->InsertColumn(1, wxT("Creation Date"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    my_panel->SelectionListCtrl->InsertColumn(2, wxT("Refinement Package"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    my_panel->SelectionListCtrl->InsertColumn(3, wxT("No. Selected"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);

    for ( counter = 0; counter < refinement_package_asset_panel->all_classification_selections.GetCount( ); counter++ ) {
        RefinementPackage* parent_package = &refinement_package_asset_panel->all_refinement_packages.Item(refinement_package_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_classification_selections.Item(counter).refinement_package_asset_id));

        my_panel->SelectionListCtrl->InsertItem(counter, refinement_package_asset_panel->all_classification_selections.Item(counter).name);
        my_panel->SelectionListCtrl->SetItem(counter, 1, refinement_package_asset_panel->all_classification_selections.Item(counter).creation_date.FormatISOCombined(' '));
        my_panel->SelectionListCtrl->SetItem(counter, 2, parent_package->name);
        my_panel->SelectionListCtrl->SetItem(counter, 3, wxString::Format("%i/%i", refinement_package_asset_panel->all_classification_selections.Item(counter).number_of_selections, refinement_package_asset_panel->all_classification_selections.Item(counter).number_of_classes));
    }

    for ( counter = 0; counter < my_panel->SelectionListCtrl->GetColumnCount( ); counter++ ) {
        old_width = my_panel->SelectionListCtrl->GetColumnWidth(counter);
        my_panel->SelectionListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE);
        current_width = my_panel->SelectionListCtrl->GetColumnWidth(counter);

        if ( old_width > current_width )
            my_panel->SelectionListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE_USEHEADER);
    }

    Thaw( );
}

wxWizardPage* ClassSelectionWizardPage::GetNext( ) const {
    //wxPrintf("Classes Next\n");
    return wizard_pointer->molecular_weight_page;
}

wxWizardPage* ClassSelectionWizardPage::GetPrev( ) const {
    return wizard_pointer->template_page;
}

////////////////////////////

//  Classes Setup Page A

/////////////////////////////

ClassesSetupWizardPageA::ClassesSetupWizardPageA(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new ClassesSetupWizardPanelA(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    my_panel->CarryOverYesButton->Bind(wxEVT_RADIOBUTTON, wxCommandEventHandler(ClassesSetupWizardPageA::CarryOverYesButtonChanged), this);
}

ClassesSetupWizardPageA::~ClassesSetupWizardPageA( ) {
    my_panel->CarryOverYesButton->Unbind(wxEVT_RADIOBUTTON, wxCommandEventHandler(ClassesSetupWizardPageA::CarryOverYesButtonChanged), this);
}

void ClassesSetupWizardPageA::CarryOverYesButtonChanged(wxCommandEvent& event) {
    wizard_pointer->class_setup_pageB->my_panel->ClassListCtrl->ClearAll( );
    wizard_pointer->class_setup_pageC->current_class_selections.Clear( );
}

wxWizardPage* ClassesSetupWizardPageA::GetNext( ) const {
    if ( my_panel->CarryOverYesButton->GetValue( ) == false ) {
        // we are carrying over all particles.  Do we need to work out how the classes should be assigned?
        // e.g. if there is only one input class, there is no setup to do..
        return wizard_pointer->class_setup_pageB;
        RefinementPackage* input_package = &refinement_package_asset_panel->all_refinement_packages.Item(wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) - 3);
    }
    else
        return wizard_pointer->class_setup_pageC;
}

wxWizardPage* ClassesSetupWizardPageA::GetPrev( ) const {
    return wizard_pointer->number_of_classes_page;
}

////////////////////////////

//  Classes Setup Page B

/////////////////////////////

ClassesSetupWizardPageB::ClassesSetupWizardPageB(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new ClassesSetupWizardPanelB(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    my_panel->ClassListCtrl->Bind(wxEVT_LIST_ITEM_SELECTED, wxListEventHandler(ClassesSetupWizardPageB::ClassSelectionChanged), this);
    my_panel->ClassListCtrl->Bind(wxEVT_LIST_ITEM_DESELECTED, wxListEventHandler(ClassesSetupWizardPageB::ClassSelectionChanged), this);
}

ClassesSetupWizardPageB::~ClassesSetupWizardPageB( ) {
    my_panel->ClassListCtrl->Unbind(wxEVT_LIST_ITEM_SELECTED, wxListEventHandler(ClassesSetupWizardPageB::ClassSelectionChanged), this);
    my_panel->ClassListCtrl->Unbind(wxEVT_LIST_ITEM_DESELECTED, wxListEventHandler(ClassesSetupWizardPageB::ClassSelectionChanged), this);
}

void ClassesSetupWizardPageB::ClassSelectionChanged(wxListEvent& event) {
    wizard_pointer->class_setup_pageC->current_class_selections.Clear( );
}

wxArrayBool ClassesSetupWizardPageB::ReturnSelectedClasses( ) {
    wxArrayBool selected_classes;
    selected_classes.Add(false, my_panel->ClassListCtrl->GetItemCount( ));

    long itemIndex = -1;

    while ( (itemIndex = my_panel->ClassListCtrl->GetNextItem(itemIndex, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED)) != wxNOT_FOUND ) {
        if ( itemIndex != -1 )
            selected_classes[itemIndex] = true;
    }

    return selected_classes;
}

wxWizardPage* ClassesSetupWizardPageB::GetNext( ) const {
    return wizard_pointer->class_setup_pageC;
}

wxWizardPage* ClassesSetupWizardPageB::GetPrev( ) const {
    return wizard_pointer->class_setup_pageA;
}

////////////////////////////

//  Classes Setup Page C

/////////////////////////////

ClassesSetupWizardPageC::ClassesSetupWizardPageC(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new ClassesSetupWizardPanelC(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    my_panel->OldClassListCtrl->Bind(wxEVT_LIST_ITEM_SELECTED, wxListEventHandler(ClassesSetupWizardPageC::OldClassListCtrlSelected), this);
    my_panel->OldClassListCtrl->Bind(wxEVT_LIST_ITEM_DESELECTED, wxListEventHandler(ClassesSetupWizardPageC::OldClassListCtrlDeSelected), this);
    my_panel->NewClassListCtrl->Bind(wxEVT_LIST_ITEM_SELECTED, wxListEventHandler(ClassesSetupWizardPageC::NewClassSelectionChanged), this);
    my_panel->NewClassListCtrl->Bind(wxEVT_LIST_ITEM_DESELECTED, wxListEventHandler(ClassesSetupWizardPageC::NewClassSelectionChanged), this);
}

ClassesSetupWizardPageC::~ClassesSetupWizardPageC( ) {
    my_panel->OldClassListCtrl->Unbind(wxEVT_LIST_ITEM_SELECTED, wxListEventHandler(ClassesSetupWizardPageC::OldClassListCtrlSelected), this);
    my_panel->OldClassListCtrl->Unbind(wxEVT_LIST_ITEM_DESELECTED, wxListEventHandler(ClassesSetupWizardPageC::OldClassListCtrlDeSelected), this);
    my_panel->NewClassListCtrl->Unbind(wxEVT_LIST_ITEM_SELECTED, wxListEventHandler(ClassesSetupWizardPageC::NewClassSelectionChanged), this);
    my_panel->NewClassListCtrl->Unbind(wxEVT_LIST_ITEM_DESELECTED, wxListEventHandler(ClassesSetupWizardPageC::NewClassSelectionChanged), this);
}

wxArrayInt ClassesSetupWizardPageC::ReturnReferencesForClass(int wanted_class) {
    int counter;

    wxArrayInt  current_references;
    wxArrayBool is_class_selected;
    wxArrayInt  selected_classes;

    if ( wizard_pointer->class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false ) {
        is_class_selected = wizard_pointer->class_setup_pageB->ReturnSelectedClasses( );

        for ( counter = 0; counter < is_class_selected.GetCount( ); counter++ ) {
            if ( is_class_selected[counter] == true )
                selected_classes.Add(counter);
        }
    }
    else {
        for ( counter = 0; counter < current_class_selections[wanted_class].class_selection.GetCount( ); counter++ ) {
            selected_classes.Add(counter);
        }
    }

    for ( counter = 0; counter < current_class_selections[wanted_class].class_selection.GetCount( ); counter++ ) {
        if ( current_class_selections[wanted_class].class_selection[counter] == true ) {
            current_references.Add(selected_classes[counter]);
        }
    }

    return current_references;
}

void ClassesSetupWizardPageC::DrawClassSelections( ) {
    int selected_class = ReturnSelectedNewClass( );

    my_panel->Freeze( );
    my_panel->OldClassListCtrl->ClearAll( );
    my_panel->OldClassListCtrl->InsertColumn(0, wxT("Class No."), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    my_panel->OldClassListCtrl->InsertColumn(1, wxT("Avg. Occ."), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    my_panel->OldClassListCtrl->InsertColumn(2, wxT("Est. Res."), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);

    if ( selected_class >= 0 ) {

        int                  counter;
        ShortRefinementInfo* selected_refinement = refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages[wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) - 3].refinement_ids[wizard_pointer->parameter_page->my_panel->GroupComboBox->GetSelection( )]);

        if ( wizard_pointer->class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false ) {
            // we need to work out which classes are being carried over..
            counter  = 0;
            int item = -1;
            for ( ;; ) {
                item = wizard_pointer->class_setup_pageB->my_panel->ClassListCtrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
                if ( item == -1 )
                    break;

                my_panel->OldClassListCtrl->InsertItem(counter, wxString::Format("Class #%2i", item + 1));
                my_panel->OldClassListCtrl->SetItem(counter, 1, wxString::Format("%.2f %%", selected_refinement->average_occupancy[item]));
                if ( selected_refinement->estimated_resolution[counter] <= 0 )
                    my_panel->OldClassListCtrl->SetItem(counter, 2, wxString::Format(wxT("N/A"), selected_refinement->estimated_resolution[item]));
                else
                    my_panel->OldClassListCtrl->SetItem(counter, 2, wxString::Format(wxT("%.2f Å"), selected_refinement->estimated_resolution[item]));

                if ( current_class_selections[selected_class].class_selection[counter] == true )
                    my_panel->OldClassListCtrl->SetItemState(counter, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
                counter++;
            }
        }
        else {
            // we are carrying over all classes..

            for ( counter = 0; counter < selected_refinement->number_of_classes; counter++ ) {
                my_panel->OldClassListCtrl->InsertItem(counter, wxString::Format("Class #%2i", counter + 1));
                my_panel->OldClassListCtrl->SetItem(counter, 1, wxString::Format("%.2f %%", selected_refinement->average_occupancy[counter]));
                if ( selected_refinement->estimated_resolution[counter] <= 0 )
                    my_panel->OldClassListCtrl->SetItem(counter, 2, wxString::Format(wxT("N/A"), selected_refinement->estimated_resolution[counter]));
                else
                    my_panel->OldClassListCtrl->SetItem(counter, 2, wxString::Format(wxT("%.2f Å"), selected_refinement->estimated_resolution[counter]));

                if ( current_class_selections[selected_class].class_selection[counter] == true )
                    my_panel->OldClassListCtrl->SetItemState(counter, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
            }
        }
    }

    my_panel->Thaw( );
}

void ClassesSetupWizardPageC::UpdateCurrentClassSelectionsText( ) {
    int selected_new_class = ReturnSelectedNewClass( );

    wxString selection_text = "";

    for ( int class_counter = 0; class_counter < current_class_selections[selected_new_class].class_selection.GetCount( ); class_counter++ ) {
        if ( current_class_selections[selected_new_class].class_selection[class_counter] == true ) {
            if ( selection_text.IsEmpty( ) == true )
                selection_text += wxString::Format("%i", class_counter + 1);
            else
                selection_text += wxString::Format(", %i", class_counter + 1);
        }
    }

    my_panel->NewClassListCtrl->SetItem(selected_new_class, 1, selection_text);
}

void ClassesSetupWizardPageC::OldClassListCtrlDeSelected(wxListEvent& event) {
    int selected_new_class = ReturnSelectedNewClass( );
    if ( selected_new_class >= 0 ) {
        current_class_selections[selected_new_class].class_selection[event.GetIndex( )] = false;
        UpdateCurrentClassSelectionsText( );
    }
}

void ClassesSetupWizardPageC::OldClassListCtrlSelected(wxListEvent& event) {
    int selected_new_class = ReturnSelectedNewClass( );
    if ( selected_new_class >= 0 ) {
        current_class_selections[selected_new_class].class_selection[event.GetIndex( )] = true;
        UpdateCurrentClassSelectionsText( );
    }
}

void ClassesSetupWizardPageC::NewClassSelectionChanged(wxListEvent& event) {
    DrawClassSelections( );
}

bool ClassesSetupWizardPageC::IsAtLeastOneOldClassSelectedForEachNewClass( ) {
    if ( current_class_selections.GetCount( ) == 0 )
        return false;
    else {
        bool class_is_empty = false;
        int  new_class_counter;
        int  old_class_counter;

        for ( new_class_counter = 0; new_class_counter < current_class_selections.GetCount( ); new_class_counter++ ) {
            class_is_empty = true;

            for ( old_class_counter = 0; old_class_counter < current_class_selections[new_class_counter].class_selection.GetCount( ); old_class_counter++ ) {
                if ( current_class_selections[new_class_counter].class_selection[old_class_counter] == true )
                    class_is_empty = false;
            }

            if ( class_is_empty == true )
                return false;
        }
    }

    // if we got here there must be at least one selected;

    return true;
}

int ClassesSetupWizardPageC::ReturnSelectedNewClass( ) {
    //wxPrintf("selected new class = %i\n", my_panel->NewClassListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED));
    return my_panel->NewClassListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
}

wxWizardPage* ClassesSetupWizardPageC::GetNext( ) const {
    return wizard_pointer->class_setup_pageD;
}

wxWizardPage* ClassesSetupWizardPageC::GetPrev( ) const {
    if ( wizard_pointer->class_setup_pageA->my_panel->CarryOverYesButton->GetValue( ) == false )
        return wizard_pointer->class_setup_pageB;
    else
        return wizard_pointer->class_setup_pageA;
}

////////////////////////////

//  Classes Setup Page D

/////////////////////////////

ClassesSetupWizardPageD::ClassesSetupWizardPageD(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new ClassesSetupWizardPanelD(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
}

wxWizardPage* ClassesSetupWizardPageD::GetNext( ) const {
    return wizard_pointer->class_setup_pageE;
}

wxWizardPage* ClassesSetupWizardPageD::GetPrev( ) const {
    return wizard_pointer->class_setup_pageC;
}

////////////////////////////

//  Classes Setup Page E

/////////////////////////////

ClassesSetupWizardPageE::ClassesSetupWizardPageE(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new ClassesSetupWizardPanelE(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
}

wxWizardPage* ClassesSetupWizardPageE::GetNext( ) const {
    return wizard_pointer->initial_reference_page;
}

wxWizardPage* ClassesSetupWizardPageE::GetPrev( ) const {
    RefinementPackage* input_package = &refinement_package_asset_panel->all_refinement_packages.Item(wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection( ) - 3);

    if ( input_package->number_of_classes == 1 ) // if there is only 1 input class, there is no fancy class setup so we can just skip to whether to randomise occupancies
    {
        return wizard_pointer->number_of_classes_page;
    }
    else
        return wizard_pointer->class_setup_pageD;
}

////////////////////////////

//  ReCentre Picks Page

/////////////////////////////

RecentrePicksWizardPage::RecentrePicksWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new RecentrePicksWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
}

RecentrePicksWizardPage::~RecentrePicksWizardPage( ) {
}

wxWizardPage* RecentrePicksWizardPage::GetNext( ) const {
    if ( my_panel->ReCentreYesButton->GetValue( ) == false ) {
        return wizard_pointer->number_of_classes_page;
    }
    else
        return wizard_pointer->remove_duplicate_picks_page;
}

wxWizardPage* RecentrePicksWizardPage::GetPrev( ) const {
    return wizard_pointer->output_pixel_size_page;
}

////////////////////////////

//  Remove Duplicates Picks Page

/////////////////////////////

RemoveDuplicatesWizardPage::RemoveDuplicatesWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new RemoveDuplicatesWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
}

RemoveDuplicatesWizardPage::~RemoveDuplicatesWizardPage( ) {
}

wxWizardPage* RemoveDuplicatesWizardPage::GetNext( ) const {
    if ( my_panel->RemoveDuplicateYesButton->GetValue( ) == false ) {
        return wizard_pointer->number_of_classes_page;
    }
    else
        return wizard_pointer->remove_duplicate_picks_threshold_page;
}

wxWizardPage* RemoveDuplicatesWizardPage::GetPrev( ) const {
    return wizard_pointer->recentre_picks_page;
}

////////////////////////////

//  Remove Duplicates Pick Threshold Page

/////////////////////////////

RemoveDuplicateThresholdWizardPage::RemoveDuplicateThresholdWizardPage(MyNewRefinementPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new RemoveDuplicateThresholdWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);
}

RemoveDuplicateThresholdWizardPage::~RemoveDuplicateThresholdWizardPage( ) {
}

wxWizardPage* RemoveDuplicateThresholdWizardPage::GetNext( ) const {
    return wizard_pointer->number_of_classes_page;
}

wxWizardPage* RemoveDuplicateThresholdWizardPage::GetPrev( ) const {
    return wizard_pointer->remove_duplicate_picks_page;
}
