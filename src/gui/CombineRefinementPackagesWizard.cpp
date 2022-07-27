#include "../core/gui_core_headers.h"
#include <wx/arrimpl.cpp>
#include <wx/filename.h>


extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyVolumeAssetPanel *volume_asset_panel;
//extern MyNewRefinementPackageWizard *refinement_package_wizard;


CombineRefinementPackagesWizard::CombineRefinementPackagesWizard(wxWindow *parent)
: CombineRefinementPackagesWizardParent( parent )
{

	number_of_visits = 0;
	imported_params_found = false;
	classes_selected = false;
	refinements_page_has_been_visited = false;
	checked_counter = refinement_package_asset_panel->all_refinement_packages.GetCount();

	package_selection_page = new PackageSelectionPage(this);
	combined_class_selection_page = new CombinedClassSelectionPage(this);
	refinement_selection_page = new RefinementSelectPage(this);
	//volume_selection_page = new VolumeSelectionPage(this);


	GetPageAreaSizer()->Add(package_selection_page);
	Bind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler( CombineRefinementPackagesWizard::OnUpdateUI), this);
}

CombineRefinementPackagesWizard::~CombineRefinementPackagesWizard()
{
	Unbind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler( CombineRefinementPackagesWizard::OnUpdateUI), this);
}

void CombineRefinementPackagesWizard::OnUpdateUI(wxUpdateUIEvent& event)
{
	checked_counter = 0;
	if (GetCurrentPage() == package_selection_page)
	{
		package_selection_page->package_selection_panel->ImportedParamsWarning->Hide();
		for (int i = 0; i < refinement_package_asset_panel->all_refinement_packages.GetCount(); i++)
		{
			if (package_selection_page->package_selection_panel->RefinementPackagesCheckListBox->IsChecked(i) == true ) // Search checked packages for particles from imported params
			{
				for (int j = 0; j < refinement_package_asset_panel->all_refinement_packages[i].contained_particles.GetCount(); j++)
				{
					if (refinement_package_asset_panel->all_refinement_packages[i].contained_particles[j].parent_image_id == -1)
					{
						package_selection_page->package_selection_panel->RemoveDuplicatesCheckbox->Enable(false);
						package_selection_page->package_selection_panel->ImportedParamsWarning->Show();
						package_selection_page->package_selection_panel->RemoveDuplicatesCheckbox->Update();
						imported_params_found = true;
						break;
					}
					else
					{
						package_selection_page->package_selection_panel->RemoveDuplicatesCheckbox->Enable();
						imported_params_found = false;
					}
				}
				if (imported_params_found == true) break;
			}
			else { // This isn't checked, so have the box enabled by default
				package_selection_page->package_selection_panel->RemoveDuplicatesCheckbox->Enable();
				package_selection_page->package_selection_panel->ImportedParamsWarning->Hide();
			}
		}

		for (int i = 0; i < refinement_package_asset_panel->all_refinement_packages.GetCount(); i++)
		{
			if (package_selection_page->package_selection_panel->RefinementPackagesCheckListBox->IsChecked(i) == true )
			{
				checked_counter++;
			}
		}
		if (!(package_selection_page->package_selection_panel->MolecularWeightTextCtrl->ReturnValue() > 0.00
				&& (package_selection_page->package_selection_panel->LargestDimensionTextCtrl->ReturnValue() > 0.00) && (checked_counter >= 2))) // Don't advance the page unless the user has input usable package parameters
		{
			DisableNextButton();
		}
		else EnableNextButton();
	}

	if (GetCurrentPage() == combined_class_selection_page)
	{
		wxWindowList all_children = combined_class_selection_page->combined_class_selection_panel->CombinedClassScrollWindow->GetChildren();
		CombinedPackageClassSelectionPanel *panel_pointer;
		for (int i = 0; i < all_children.GetCount(); i++)
		{
			panel_pointer = reinterpret_cast <CombinedPackageClassSelectionPanel *> (all_children.Item(i)->GetData());
			if (panel_pointer->ClassComboBox->GetSelection() == -1) // Don't let user continue without selecting classes for each package
			{
				classes_selected = false;
				break;
			}
			else
			{
				classes_selected = true;
			}
		}
		if (classes_selected == true) EnableNextButton();
		else DisableNextButton();
	}
}


void CombineRefinementPackagesWizard::OnCancelClick( wxWizardEvent& event )
{
	EndModal(0);
	Destroy();
}

void CombineRefinementPackagesWizard::DisableNextButton()
{
	wxWindow *win = wxWindow::FindWindowById(wxID_FORWARD);
	if(win) win->Enable(false);
}

void CombineRefinementPackagesWizard::EnableNextButton()
{
	wxWindow *win = wxWindow::FindWindowById(wxID_FORWARD);
	if(win) win->Enable(true);
}

void CombineRefinementPackagesWizard::PageChanged(wxWizardEvent& event)
{
	wxArrayString refinement_package_names;
	if (event.GetPage() == package_selection_page && number_of_visits < 1)
	{
		if (refinement_package_asset_panel->all_refinement_packages.GetCount() > 0)
		{
			package_selection_page->package_selection_panel->MolecularWeightTextCtrl->SetValue(wxString::Format(wxT("%.2f"), refinement_package_asset_panel->all_refinement_packages[0].estimated_particle_weight_in_kda));
			package_selection_page->package_selection_panel->LargestDimensionTextCtrl->SetValue(wxString::Format(wxT("%.2f"), refinement_package_asset_panel->all_refinement_packages[0].estimated_particle_size_in_angstroms));
		}
		number_of_visits++;
		wxString no_packages_message = "No packages currently exist to combine";
		package_selection_page->Freeze();
		for (int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
		{
			for (int counter_2 = refinement_package_asset_panel->all_refinement_packages.GetCount() -1; counter_2 >= 0; counter_2--)
			{
				if (counter == counter_2 )
				{
					continue;  // Don't compare package stack_box_size of a package with itself
				}
				else if (refinement_package_asset_panel->all_refinement_packages[counter].stack_box_size == refinement_package_asset_panel->all_refinement_packages[counter_2].stack_box_size && refinement_package_asset_panel->all_refinement_packages.GetCount() > 2)
				{
					refinement_package_names.Add(refinement_package_asset_panel->all_refinement_packages[counter].name);
					break;
				}
				else if (refinement_package_asset_panel->all_refinement_packages[counter].stack_box_size == refinement_package_asset_panel->all_refinement_packages[counter_2].stack_box_size && refinement_package_asset_panel->all_refinement_packages.GetCount() == 2)
				{
					refinement_package_names.Add(refinement_package_asset_panel->all_refinement_packages[counter].name);
					break;
				}
			}
		}

		if (refinement_package_names.IsEmpty())
		{
			refinement_package_names.Add(no_packages_message);
			DisableNextButton();
		}
		package_selection_page->package_selection_panel->RefinementPackagesCheckListBox->InsertItems(refinement_package_names, 0);

	}
	if (event.GetPage() == combined_class_selection_page)
	{
		if (classes_page_has_been_visited == true) combined_class_selection_page->combined_class_selection_panel->CombinedClassScrollWindow->DestroyChildren();

		combined_class_selection_page->Freeze();


		for (int i = 0; i < refinement_package_asset_panel->all_refinement_packages.GetCount(); i++)
		{
			if (package_selection_page->package_selection_panel->RefinementPackagesCheckListBox->IsChecked(i)) // If the refinement package is checked, add a selection box for it.
			{
				CombinedPackageClassSelectionPanel *panel1 = new CombinedPackageClassSelectionPanel(combined_class_selection_page->combined_class_selection_panel->CombinedClassScrollWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
				panel1->ClassText->SetLabel("Class from " + refinement_package_asset_panel->all_refinement_packages[i].name + ": ");
				panel1->ClassText->Wrap(300);
				panel1->FillComboBox(i);
				combined_class_selection_page->combined_class_selection_panel->CombinedClassScrollSizer->Add(panel1, 0, wxALL | wxEXPAND, 5);
			}
		}

		classes_page_has_been_visited = true;
		combined_class_selection_page->Layout();
		combined_class_selection_page->Thaw();
	}

	//TODO: change this code to fit new build
	wxString refinement_name;
	if (event.GetPage() == refinement_selection_page)
	{
		refinement_selection_page->Freeze();

		if (refinements_page_has_been_visited == true) refinement_selection_page->combined_package_refinement_selection_panel->CombinedRefinementScrollWindow->DestroyChildren();

		for (int i = 0; i < refinement_package_asset_panel->all_refinement_packages.GetCount(); i++)
		{
			if (package_selection_page->package_selection_panel->RefinementPackagesCheckListBox->IsChecked(i))
			{
				CombinedPackageRefinementSelectPanel *panel1 = new CombinedPackageRefinementSelectPanel(refinement_selection_page->combined_package_refinement_selection_panel->CombinedRefinementScrollWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
				panel1->RefinementText->SetLabel("Refinement from " + refinement_package_asset_panel->all_refinement_packages[i].name + ": ");
				panel1->RefinementText->Wrap(300);
				panel1->RefinementComboBox->FillComboBox(i, false);
				refinement_selection_page->combined_package_refinement_selection_panel->CombinedRefinementScrollSizer->Add(panel1, 0, wxEXPAND | wxALL, 5);
			}
		}

		refinement_selection_page->combined_package_refinement_selection_panel->SelectRefinementText->Wrap(this->GetClientSize().GetWidth());
		refinement_selection_page->Layout();
		refinement_selection_page->Thaw();
		refinements_page_has_been_visited = true;
	}

	/*if (event.GetPage() == volume_selection_page)
	{
		volume_selection_page->Freeze();

		if (volume_selection_page_has_been_visited == true) volume_selection_page->volume_selection_panel->CombinedVolumeScrollWindow->DestroyChildren();

		CombinedPackageVolumeSelectPanel *panel1 = new CombinedPackageVolumeSelectPanel(volume_selection_page->volume_selection_panel->CombinedVolumeScrollWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
		panel1->VolumeText->SetLabel("Select a volume to use: ");
		panel1->VolumeComboBox->FillComboBox(true, true);
		volume_selection_page->volume_selection_panel->CombinedVolumeScrollSizer->Add(panel1, 0, wxEXPAND | wxALL, 5);

		volume_selection_page->volume_selection_panel->SelectVolumeText->Wrap(this->GetClientSize().GetWidth());
		volume_selection_page->Layout();
		volume_selection_page->Thaw();
		volume_selection_page_has_been_visited = true;
	}*/
}


void CombineRefinementPackagesWizard::PageChanging(wxWizardEvent& event)
{

}


void CombineRefinementPackagesWizard::OnFinished( wxWizardEvent& event )
{
	int counter = 0;
	int class_counter = 0;
	long output_particle_counter = 0;
	long input_particle_counter = 0;
	bool is_duplicate = false;
	double smallest_pixel_size = 0.0;
	RefinementPackageParticleInfo temp_sorting_info;
	RefinementResult temp_sorting_result;

	ArrayOfRefinementPackages array_of_packages_to_combine;
	RefinementPackage *temp_combined_refinement_package = new RefinementPackage;
	RefinementPackageParticleInfo temp_combined_particle_info;

	Refinement *temp_combined_refinement = new Refinement;
	wxArrayString packages_to_combine_filenames;

	wxFileName combined_stacks_filename = main_frame->current_project.particle_stack_directory.GetFullPath() + wxString::Format("/particle_stack_%li.mrc", refinement_package_asset_panel->current_asset_number);
	wxString mrc_filename = combined_stacks_filename.GetFullPath();
	MRCFile combined_stacks_file(combined_stacks_filename.GetFullPath().ToStdString(), true);
	Image image_from_previous_stack;


	for (counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	{
		if (package_selection_page->package_selection_panel->RefinementPackagesCheckListBox->IsChecked(counter) == true)
		{
			output_particle_counter = output_particle_counter + refinement_package_asset_panel->all_refinement_packages[counter].contained_particles.GetCount();
		}
	}
	temp_combined_refinement->number_of_particles = output_particle_counter;
	temp_combined_refinement->SizeAndFillWithEmpty(output_particle_counter, 1);
	for (counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	{
		if (package_selection_page->package_selection_panel->RefinementPackagesCheckListBox->IsChecked(counter) == true)
		{
			array_of_packages_to_combine.Add(refinement_package_asset_panel->all_refinement_packages[counter]);  // Add package to a separate array
		}
		else continue; // package is not checked, do not do anything with it

		if (array_of_packages_to_combine.GetCount() == 1)  // To begin, assign package params
		{
			temp_combined_refinement_package->stack_filename = combined_stacks_filename.GetFullPath();
			temp_combined_refinement_package->symmetry = package_selection_page->package_selection_panel->SymmetryComboBox->GetValue();
			temp_combined_refinement_package->estimated_particle_weight_in_kda = package_selection_page->package_selection_panel->MolecularWeightTextCtrl->ReturnValue();
			temp_combined_refinement_package->estimated_particle_size_in_angstroms = package_selection_page->package_selection_panel->LargestDimensionTextCtrl->ReturnValue();
			temp_combined_refinement_package->stack_box_size = array_of_packages_to_combine[0].stack_box_size;
			temp_combined_refinement_package->number_of_classes = 1;

			temp_combined_refinement_package->number_of_run_refinments = 0;
			temp_combined_refinement_package->lowest_resolution_of_intial_parameter_generated_3ds = refinement_package_asset_panel->all_refinement_packages[counter].lowest_resolution_of_intial_parameter_generated_3ds;
			temp_combined_refinement_package->stack_has_white_protein = refinement_package_asset_panel->all_refinement_packages[counter].stack_has_white_protein;

			long new_refinement_id = main_frame->current_project.database.ReturnHighestRefinementID() + 1;
			temp_combined_refinement_package->refinement_ids.Add(new_refinement_id);
			//temp_combined_refinement_package->last_refinment_id = -1;
			temp_combined_refinement_package->references_for_next_refinement.Add(-1);
			temp_combined_refinement_package->asset_id = refinement_package_asset_panel->all_refinement_packages.GetCount() + 1;  // In database, does index of the list start at 0 or 1? i.e., should this be + 1?
			temp_combined_refinement_package->name = wxString::Format("Refinement Package #%li", refinement_package_asset_panel->current_asset_number);

			temp_combined_refinement->name = wxString::Format("Combined Parameters - ID #%li", new_refinement_id);
			temp_combined_refinement->resolution_statistics_box_size = temp_combined_refinement_package->stack_box_size;
			temp_combined_refinement->starting_refinement_id = new_refinement_id;
			temp_combined_refinement->percent_used = 100.0f;
			temp_combined_refinement->refinement_id = new_refinement_id;
			temp_combined_refinement->datetime_of_run = wxDateTime::Now();
			temp_combined_refinement->number_of_classes = 1;
			temp_combined_refinement->refinement_package_asset_id = temp_combined_refinement_package->asset_id;
		}
	}
	// Make sure the smallest pixel size is being used in the combined package
	for (counter = 0; counter < array_of_packages_to_combine.GetCount(); counter++)
	{
		smallest_pixel_size = array_of_packages_to_combine[0].output_pixel_size;
		if (array_of_packages_to_combine[counter].output_pixel_size < smallest_pixel_size)
		{
			smallest_pixel_size = array_of_packages_to_combine[counter].output_pixel_size;
		}
	}
	temp_combined_refinement_package->output_pixel_size = smallest_pixel_size;
	temp_combined_refinement->resolution_statistics_pixel_size = smallest_pixel_size;

	// Now loop through the existing MRC filenames to get to the files; open, read through, then write each particle to new MRC file, close.
	output_particle_counter = 0;  // Return output_partice_counter to 0 before executing the read/write stack functions
	wxWindowList all_children = refinement_selection_page->combined_package_refinement_selection_panel->CombinedRefinementScrollWindow->GetChildren(); // Get the window's children for pulling user selected classes
	CombinedPackageRefinementSelectPanel *panel_pointer;
	wxArrayLong corresponding_refinement_ids[all_children.GetCount()];
	//TODO: check over how this is being pulled; may not want to go this route
	for (int i = 0; i < all_children.GetCount(); i++)
	{
		if (all_children.Item(i)->GetData()->GetClassInfo()->GetClassName() == wxString("wxPanel"))
		{
			panel_pointer = reinterpret_cast <CombinedPackageRefinementSelectPanel *> (all_children.Item(i)->GetData());
			for (int j = 0; j < refinement_package_asset_panel->all_refinement_packages.GetCount(); j++)
			{
				// If this package is the correct package, then check the selection and grab the proper refinement_id
				if (panel_pointer->RefinementText->GetLabel().Contains(refinement_package_asset_panel->all_refinement_packages[j].name))
				{

					if(panel_pointer->RefinementComboBox->GetSelection() == 0)
						corresponding_refinement_ids->Add(refinement_package_asset_panel->all_refinement_packages[j].refinement_ids[refinement_package_asset_panel->all_refinement_packages[i].refinement_ids.GetCount() - 1]);

					else
					{
						corresponding_refinement_ids->Add(refinement_package_asset_panel->all_refinement_packages[j].refinement_ids[panel_pointer->RefinementComboBox->GetSelection()]);
					}
				}
			}
		}
	}

	wxWindowList class_page_children = combined_class_selection_page->combined_class_selection_panel->CombinedClassScrollWindow->GetChildren();
	CombinedPackageClassSelectionPanel *panel_pointer2;
	int package_classes[all_children.GetCount()];
	for (int i = 0; i < class_page_children.GetCount(); i++)
	{
		if (all_children.Item(i)->GetData()->GetClassInfo()->GetClassName() == wxString("wxPanel"))
		{
			panel_pointer2 = reinterpret_cast <CombinedPackageClassSelectionPanel *> (class_page_children.Item(i)->GetData());
			package_classes[i] = panel_pointer2->ClassComboBox->GetSelection();
		}
	}

/*	wxWindowList volume_page_child = volume_selection_page->volume_selection_panel->CombinedVolumeScrollWindow->GetChildren();
	CombinedPackageVolumeSelectPanel *panel_pointer3;
	panel_pointer3 = reinterpret_cast <CombinedPackageVolumeSelectPanel *> (volume_page_child.Item(0)->GetData());
	if (panel_pointer3->VolumeComboBox->GetSelection() == 0)
	{
		temp_combined_refinement_package->references_for_next_refinement.Add(-1);
	}
	else
	{
		temp_combined_refinement_package->references_for_next_refinement.Add(volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(panel_pointer3->VolumeComboBox->GetSelection() - 1)->asset_id);
	}
	*/

	for (counter = 0; counter < array_of_packages_to_combine.GetCount(); counter++)
	{
		MRCFile input_file(array_of_packages_to_combine[counter].stack_filename.ToStdString(), false);
		Refinement *old_refinement = main_frame->current_project.database.GetRefinementByID(corresponding_refinement_ids->Item(counter));

		for (input_particle_counter = 0; input_particle_counter < array_of_packages_to_combine[counter].contained_particles.GetCount(); input_particle_counter++)
		{
			image_from_previous_stack.ReadSlice(&input_file, input_particle_counter + 1);
			image_from_previous_stack.WriteSlice(&combined_stacks_file, output_particle_counter + 1); // Write from PREVIOUS stack

			temp_combined_particle_info = array_of_packages_to_combine[counter].contained_particles[input_particle_counter];  // Get the infos

			if (package_selection_page->package_selection_panel->RemoveDuplicatesCheckbox->IsEnabled() == true && package_selection_page->package_selection_panel->RemoveDuplicatesCheckbox->IsChecked() == true) // User wants to remove duplicates; is this not working?
			{
				is_duplicate = CheckIfDuplicate(temp_combined_particle_info.original_particle_position_asset_id, temp_combined_refinement_package);

				if (is_duplicate == true)
				{
					continue;  // Do not add duplicate particles
				}

				else // Add the particle if not a duplicate
				{
					temp_combined_refinement_package->contained_particles.Add(temp_combined_particle_info);  // Puts first info into new package
					temp_combined_refinement_package->contained_particles[output_particle_counter].position_in_stack = output_particle_counter + 1;
					temp_combined_refinement_package->contained_particles[output_particle_counter].original_particle_position_asset_id = output_particle_counter + 1; // Question of whether this will be needed here; it probably is

					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].position_in_stack = output_particle_counter + 1;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].defocus1 = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].defocus1;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].defocus2 = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].defocus2;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].defocus_angle = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].defocus_angle;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].phase_shift = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].phase_shift;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].logp = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].logp;

					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].occupancy = 100.0;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].phi = global_random_number_generator.GetUniformRandom() * 180.0;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].theta = rad_2_deg(acosf(2.0f * fabsf(global_random_number_generator.GetUniformRandom()) - 1.0f));
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].psi = global_random_number_generator.GetUniformRandom() * 180.0;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].score = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].score;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].image_is_active = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].image_is_active;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].sigma = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].sigma;

					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].pixel_size = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].pixel_size;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].microscope_voltage_kv = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].microscope_voltage_kv;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].microscope_spherical_aberration_mm = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].microscope_spherical_aberration_mm;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].amplitude_contrast = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].amplitude_contrast;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].beam_tilt_x = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].beam_tilt_x;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].beam_tilt_y = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].beam_tilt_y;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].image_shift_x = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].image_shift_x;
					temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].image_shift_y = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].image_shift_y;
				}
			}

			else  // user does not want to remove duplicates
			{
				if (temp_combined_refinement_package->contained_particles.GetCount() == 0)
				{
					temp_combined_refinement->resolution_statistics_pixel_size = old_refinement->resolution_statistics_pixel_size;
					temp_combined_refinement->resolution_statistics_box_size = old_refinement->resolution_statistics_box_size;
				}

				temp_combined_refinement_package->contained_particles.Add(temp_combined_particle_info);  // Puts info into new package
				temp_combined_refinement_package->contained_particles[output_particle_counter].position_in_stack = output_particle_counter + 1;
				temp_combined_refinement_package->contained_particles[output_particle_counter].original_particle_position_asset_id = output_particle_counter + 1;  // Appears this is necessary when combining and not wanting to remove duplicates; at least this is resolved

				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].position_in_stack = output_particle_counter + 1;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].defocus1 = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].defocus1;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].defocus2 = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].defocus2;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].defocus_angle = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].defocus_angle;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].phase_shift = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].phase_shift;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].logp = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].logp;

				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].occupancy = 100.0;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].phi = global_random_number_generator.GetUniformRandom() * 180.0;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].theta = rad_2_deg(acosf(2.0f * fabsf(global_random_number_generator.GetUniformRandom()) - 1.0f));
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].psi = global_random_number_generator.GetUniformRandom() * 180.0;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].score = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].score;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].image_is_active = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].image_is_active;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].sigma = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].sigma;

				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].pixel_size = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].pixel_size;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].microscope_voltage_kv = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].microscope_voltage_kv;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].microscope_spherical_aberration_mm = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].microscope_spherical_aberration_mm;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].amplitude_contrast = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].amplitude_contrast;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].beam_tilt_x = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].beam_tilt_x;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].beam_tilt_y = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].beam_tilt_y;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].image_shift_x = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].image_shift_x;
				temp_combined_refinement->class_refinement_results[class_counter].particle_refinement_results[output_particle_counter].image_shift_y = old_refinement->class_refinement_results[package_classes[counter]].particle_refinement_results[input_particle_counter].image_shift_y;
			}
			output_particle_counter++;
		}
		input_file.CloseFile();
	}

	combined_stacks_file.CloseFile();

	main_frame->current_project.database.Begin(); // Have to add the newly combined package and its refinement to the database
	refinement_package_asset_panel->AddAsset(temp_combined_refinement_package);
	main_frame->current_project.database.AddRefinementPackageAsset(temp_combined_refinement_package);

	for (class_counter = 0; class_counter < temp_combined_refinement->number_of_classes; class_counter++)
	{
		temp_combined_refinement->class_refinement_results[class_counter].class_resolution_statistics.Init(temp_combined_refinement_package->output_pixel_size, temp_combined_refinement->resolution_statistics_box_size);
		temp_combined_refinement->class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(temp_combined_refinement_package->estimated_particle_weight_in_kda);
	}

	main_frame->current_project.database.AddRefinement(temp_combined_refinement);
	ArrayofAngularDistributionHistograms all_histograms = temp_combined_refinement->ReturnAngularDistributions(temp_combined_refinement_package->symmetry);

	for (class_counter = 1; class_counter <= temp_combined_refinement->number_of_classes; class_counter++)
	{
		main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[class_counter - 1], temp_combined_refinement->refinement_id, class_counter);
	}
	ShortRefinementInfo temp_info;
	temp_info = temp_combined_refinement;

	refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);
	main_frame->current_project.database.Commit();
	Destroy();
}


bool CombineRefinementPackagesWizard::CheckIfDuplicate (int comparison_original_particle_position_asset_id, RefinementPackage* combined_package)
{
	for (int i = 0; i < combined_package->contained_particles.GetCount(); i++)
	{
		if (combined_package->contained_particles[i].original_particle_position_asset_id == comparison_original_particle_position_asset_id)
		{
			return true;
		}
	}
	return false;
}

////////////////////////////

// SELECT PACKAGES PAGE

////////////////////////////

PackageSelectionPage::PackageSelectionPage(CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap)
: wxWizardPage(parent, bitmap)
{
	Freeze();
	wizard_pointer = parent;
	wxBoxSizer* main_sizer;
	package_selection_panel = new PackageSelectionPanel(this);

	main_sizer = new wxBoxSizer(wxVERTICAL);
	this->SetSizer(main_sizer);
	main_sizer->Fit(this);
	main_sizer->Add(package_selection_panel);
	Thaw();

}

PackageSelectionPage::~PackageSelectionPage()
{

}

wxWizardPage *  PackageSelectionPage::GetNext () const
{
	return wizard_pointer->combined_class_selection_page;
}


////////////////////////////

// SELECT CLASSES PAGE

////////////////////////////

CombinedClassSelectionPage::CombinedClassSelectionPage(CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap)
: wxWizardPage(parent, bitmap)
{
	Freeze();
	wizard_pointer = parent;
	wxBoxSizer* bSizer686;
	combined_class_selection_panel = new CombinedClassSelectionPanel(this);

	bSizer686 = new wxBoxSizer(wxVERTICAL);
	this->SetSizer(bSizer686);
	bSizer686->Fit(this);
	bSizer686->Add(combined_class_selection_panel);
	Thaw();

}

CombinedClassSelectionPage::~CombinedClassSelectionPage()
{

}

wxWizardPage * CombinedClassSelectionPage::GetNext () const
{
	return wizard_pointer->refinement_selection_page;
}

wxWizardPage * CombinedClassSelectionPage::GetPrev () const
{
	return wizard_pointer->package_selection_page;
}


void CombinedPackageClassSelectionPanel::FillComboBox(long wanted_refinement_package)
{
	for (int i = 1; i <= refinement_package_asset_panel->all_refinement_packages[wanted_refinement_package].number_of_classes; i++)
	{
		this->ClassComboBox->Append(wxString::Format("%i", i));
		this->ClassComboBox->SetSelection(0);
	}
}

////////////////////////////

// SELECT REFINEMENTS PAGE

////////////////////////////

RefinementSelectPage::RefinementSelectPage(CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap)
: wxWizardPage(parent, bitmap)
{
	Freeze();
	wizard_pointer = parent;
	wxBoxSizer* bSizer686;
	combined_package_refinement_selection_panel = new CombinedPackageRefinementPanel(this);

	bSizer686 = new wxBoxSizer(wxVERTICAL);
	this->SetSizer(bSizer686);
	bSizer686->Fit(this);
	bSizer686->Add(combined_package_refinement_selection_panel);
	Thaw();
}

RefinementSelectPage::~RefinementSelectPage()
{

}

wxWizardPage * RefinementSelectPage::GetNext() const
{
	return NULL;
}

wxWizardPage * RefinementSelectPage::GetPrev() const
{
	return wizard_pointer->combined_class_selection_page;
}


/*VolumeSelectionPage::VolumeSelectionPage(CombineRefinementPackagesWizard *parent, const wxBitmap &bitmap)
: wxWizardPage(parent, bitmap)
{
	Freeze();
	wizard_pointer = parent;
	wxBoxSizer* bSizer686;
	volume_selection_panel = new CombinedPackageVolumePanel(this);

	bSizer686 = new wxBoxSizer(wxVERTICAL);
	this->SetSizer(bSizer686);
	bSizer686->Fit(this);
	bSizer686->Add(volume_selection_panel);
	Thaw();
}

VolumeSelectionPage::~VolumeSelectionPage()
{

}

wxWizardPage * VolumeSelectionPage::GetNext() const
{
	return NULL;
}

wxWizardPage * VolumeSelectionPage::GetPrev() const
{
	return wizard_pointer->refinement_selection_page;
}*/

