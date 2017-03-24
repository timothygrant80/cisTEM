//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyParticlePositionAssetPanel *particle_position_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyVolumeAssetPanel *volume_asset_panel;


static int wxCMPFUNC_CONV SortByParentImageID( RefinementPackageParticleInfo **a, RefinementPackageParticleInfo **b) // function for sorting the classum selections by parent_image_id - this makes cutting them out more efficient
{
	if ((*a)->parent_image_id > (*b)->parent_image_id) return 1;
	else
	if ((*a)->parent_image_id < (*b)->parent_image_id) return -1;
	else
	{
		if ((*a)->original_particle_position_asset_id > (*b)->original_particle_position_asset_id) return 1;
		else
		if ((*a)->original_particle_position_asset_id < (*b)->original_particle_position_asset_id) return -1;
		else
		return 0;
	}
};

MyNewRefinementPackageWizard::MyNewRefinementPackageWizard( wxWindow* parent )
:
NewRefinementPackageWizard( parent )
{
	template_page = new TemplateWizardPage(this);
	parameter_page = new InputParameterWizardPage(this);
	particle_group_page = new ParticleGroupWizardPage(this);
	number_of_classes_page = new NumberofClassesWizardPage(this);
	box_size_page = new BoxSizeWizardPage(this);
	class_setup_page = new ClassesSetupWizardPage(this);
	initial_reference_page = new InitialReferencesWizardPage(this);
	symmetry_page = new SymmetryWizardPage(this);
	molecular_weight_page = new MolecularWeightWizardPage(this);
	largest_dimension_page = new LargestDimensionWizardPage(this);
	class_selection_page = new ClassSelectionWizardPage(this);



	GetPageAreaSizer()->Add(template_page);
	//GetPageAreaSizer()->Add(particle_group_page);
	//GetPageAreaSizer()->Add(number_of_classes_page);
	//GetPageAreaSizer()->Add(box_size_page);

	Bind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MyNewRefinementPackageWizard::OnUpdateUI), this);

}

MyNewRefinementPackageWizard::~MyNewRefinementPackageWizard()
{
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
	Unbind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MyNewRefinementPackageWizard::OnUpdateUI), this);


}

void MyNewRefinementPackageWizard::OnUpdateUI(wxUpdateUIEvent& event)
{
	if (GetCurrentPage() == template_page) EnableNextButton();
	else
	if (GetCurrentPage() == particle_group_page)
	{
	 if (particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetCount() > 0)
	 {
		if (particle_position_asset_panel->ReturnGroupSize(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection()) > 0) EnableNextButton();
		else DisableNextButton();
	 }
	 else DisableNextButton();
	}
	else
	if (GetCurrentPage() == box_size_page) EnableNextButton();
	else
	if (GetCurrentPage() == number_of_classes_page) EnableNextButton();
	else
	if (GetCurrentPage() == class_setup_page) EnableNextButton();
	else
	if (GetCurrentPage() == symmetry_page)
	{
		wxString symmetry = symmetry_page->my_panel->SymmetryComboBox->GetValue();
		if (IsAValidSymmetry(&symmetry) == true) EnableNextButton();
		else DisableNextButton();
	}
	else
	if (GetCurrentPage() == molecular_weight_page) EnableNextButton();
	else
	if (GetCurrentPage() == largest_dimension_page) EnableNextButton();
	else
	if (GetCurrentPage() == parameter_page) EnableNextButton();
	else
	if (GetCurrentPage() == class_selection_page)
	{
		if (class_selection_page->my_panel->SelectionListCtrl->GetSelectedItemCount() > 0)
		{
			int total_selections = 0;
			long item = -1;

			for ( ;; )
			{
				item = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
				if ( item == -1 )
				break;

				total_selections += refinement_package_asset_panel->all_classification_selections.Item(item).number_of_selections;

				if (total_selections > 0) break;
			}

			if (total_selections > 0) EnableNextButton();
			else DisableNextButton();
		}
		else DisableNextButton();


	}
}


void MyNewRefinementPackageWizard::DisableNextButton()
{
	wxWindow *win = wxWindow::FindWindowById(wxID_FORWARD);
	if(win) win->Enable(false);
}

void MyNewRefinementPackageWizard::EnableNextButton()
{
	wxWindow *win = wxWindow::FindWindowById(wxID_FORWARD);
	if(win) win->Enable(true);
}

void MyNewRefinementPackageWizard::PageChanging(wxWizardEvent& event)
{

}

void MyNewRefinementPackageWizard::PageChanged(wxWizardEvent& event)
{

	if (event.GetPage() == template_page)
	{
		if (template_page->my_panel->InfoText->has_autowrapped == false)
		{
			template_page->Freeze();
			template_page->my_panel->InfoText->AutoWrap();
			template_page->Layout();
			template_page->Thaw();
		}
	}
	else
	if (event.GetPage() == parameter_page)
	{
		parameter_page->Freeze();

		if (parameter_page->my_panel->InfoText->has_autowrapped == false)
		{

			parameter_page->my_panel->InfoText->AutoWrap();
			parameter_page->Layout();
		}

	//	wxPrintf("filling\n");
	 	 for (int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection() - 3).refinement_ids.GetCount(); counter++)
	 	 {
	 		parameter_page->my_panel->GroupComboBox->Append (refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages.Item(template_page->my_panel->GroupComboBox->GetSelection() - 3).refinement_ids[counter])->name);
	 	 }
	 //	 wxPrintf("filled\n");


	    parameter_page->my_panel->GroupComboBox->SetSelection(parameter_page->my_panel->GroupComboBox->GetCount() - 1);
		parameter_page->Thaw();
	}
	if (event.GetPage() == particle_group_page)
	{
		if (particle_group_page->my_panel->InfoText->has_autowrapped == false)
		{
			particle_group_page->Freeze();
			particle_group_page->my_panel->InfoText->AutoWrap();
			particle_group_page->Layout();
			particle_group_page->Thaw();
		}
	}
	else
	if (event.GetPage() == number_of_classes_page)
	{
		if (number_of_classes_page->my_panel->InfoText->has_autowrapped == false)
		{
			number_of_classes_page->Freeze();
			number_of_classes_page->my_panel->InfoText->AutoWrap();
			number_of_classes_page->Layout();
			number_of_classes_page->Thaw();
		}
	}
	if (event.GetPage() == box_size_page)
	{
		if (box_size_page->my_panel->InfoText->has_autowrapped == false)
		{
			box_size_page->Freeze();
			box_size_page->my_panel->InfoText->AutoWrap();
			box_size_page->Layout();
			box_size_page->Thaw();
		}
	}
	else
	if (event.GetPage() == class_setup_page)
	{
		//if (box_size_page->my_panel->InfoText->has_autowrapped == false) class_setup_page->my_panel->InfoText->AutoWrap();
	}
	else
	if (event.GetPage() == initial_reference_page)
	{

		int counter;
		wxWindow *window_pointer;

		initial_reference_page->my_panel->Destroy();
		initial_reference_page->CreatePanel();
		initial_reference_page->my_panel->InfoText->AutoWrap();

		for (counter = 1; counter <= number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue(); counter++)
		{
			ClassVolumeSelectPanel *panel1 = new ClassVolumeSelectPanel( initial_reference_page->my_panel->ScrollWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
			panel1->ClassText->SetLabel(wxString::Format("Class #%2i :", counter));
			panel1->class_number = counter;
			initial_reference_page->my_panel->GridSizer->Add( panel1, 0, wxEXPAND | wxALL, 5 );
		}
	}
	else
	if (event.GetPage() == symmetry_page)
	{
		if (symmetry_page->my_panel->InfoText->has_autowrapped == false)
		{
			symmetry_page->Freeze();
			symmetry_page->my_panel->InfoText->AutoWrap();
			symmetry_page->Layout();
			symmetry_page->Thaw();
		}
	}
	else
	if (event.GetPage() == molecular_weight_page)
	{
		if (molecular_weight_page->my_panel->InfoText->has_autowrapped == false)
		{
			molecular_weight_page->Freeze();
			molecular_weight_page->my_panel->InfoText->AutoWrap();
			molecular_weight_page->Layout();
			molecular_weight_page->Thaw();
		}
	}
	else
	if (event.GetPage() == largest_dimension_page)
	{
		if (largest_dimension_page->my_panel->InfoText->has_autowrapped == false)
		{
			largest_dimension_page->Freeze();
			largest_dimension_page->my_panel->InfoText->AutoWrap();
			largest_dimension_page->Layout();
			largest_dimension_page->Thaw();
		}
	}
	else
	if (event.GetPage() == class_selection_page)
	{
		if (class_selection_page->my_panel->InfoText->has_autowrapped == false)
		{
			class_selection_page->Freeze();
			class_selection_page->my_panel->InfoText->AutoWrap();
			class_selection_page->Layout();
			class_selection_page->Thaw();
		}

	}




}


void MyNewRefinementPackageWizard::OnFinished( wxWizardEvent& event )
{
	int class_counter;
	long particle_counter;
	long counter;
	long item;
	long parent_classification_id;

	// cut out the particles if necessary..

	RefinementPackage *temp_refinement_package = new RefinementPackage;
	RefinementPackageParticleInfo temp_particle_info;


	ClassRefinementResults junk_class_results;
	RefinementResult junk_result;
	Refinement temp_refinement;

	RefinementPackage *parent_refinement_package_link;

	wxArrayLong current_images;
	ArrayOfRefinmentPackageParticleInfos class_average_particle_infos;



	if (template_page->my_panel->GroupComboBox->GetSelection() < 2) // This is a new package or from classums
	{
		long number_of_particles;
		if (template_page->my_panel->GroupComboBox->GetSelection() == 0) // completely new..
		{
			number_of_particles = particle_position_asset_panel->ReturnGroupSize(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection());
		}
		else // from classums..
		{
			OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Sorting out which contained particles to copy over", "Reading particles from database...", 100, this, wxPD_APP_MODAL);

			item = -1;
			for ( ;; )
			{
				item = class_selection_page->my_panel->SelectionListCtrl->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
				if ( item == -1 )
				break;

				current_images.Clear();

				// for each selection we need to extract out the relevant particle position assets..

				parent_refinement_package_link = &refinement_package_asset_panel->all_refinement_packages.Item(refinement_package_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_classification_selections.Item(item).refinement_package_asset_id));
				parent_classification_id = refinement_package_asset_panel->all_classification_selections.Item(item).classification_id;

				for (counter = 0; counter < refinement_package_asset_panel->all_classification_selections.Item(item).selections.GetCount(); counter++)
				{
					current_images = main_frame->current_project.database.Return2DClassMembers(parent_classification_id, refinement_package_asset_panel->all_classification_selections.Item(item).selections.Item(counter));
					my_progress_dialog->Pulse();

					// copy out all the relevant particle positions

					for (particle_counter = 0; particle_counter < current_images.GetCount(); particle_counter++)
					{
						class_average_particle_infos.Add(parent_refinement_package_link->ReturnParticleInfoByPositionInStack(current_images.Item(particle_counter)));
					}
				}
			}

			number_of_particles = class_average_particle_infos.GetCount();
			class_average_particle_infos.Sort(SortByParentImageID);
			my_progress_dialog->Destroy();
		}

		OneSecondProgressDialog *my_dialog = new OneSecondProgressDialog ("Refinement Package", "Creating Refinement Package...", number_of_particles, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE| wxPD_APP_MODAL);

		temp_refinement_package->name = wxString::Format("Refinement Package #%li", refinement_package_asset_panel->current_asset_number);
		temp_refinement_package->number_of_classes = number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue();
		temp_refinement_package->number_of_run_refinments = 0;

		temp_refinement.number_of_classes = temp_refinement_package->number_of_classes;
		temp_refinement.number_of_particles = number_of_particles;
		temp_refinement.name = "Random Parameters";
		temp_refinement.resolution_statistics_box_size = box_size_page->my_panel->BoxSizeSpinCtrl->GetValue();
		temp_refinement.refinement_package_asset_id = refinement_package_asset_panel->current_asset_number + 1;

		long current_particle_parent_image_id = 0;
		long current_loaded_image_id = -1;
		long position_in_stack = 0;


		int current_x_pos;
		int current_y_pos;

		float average_value_at_edges;
		float image_defocus_1;
		float image_defocus_2;
		float image_defocus_angle;

		ImageAsset *current_image_asset = NULL;
		ParticlePositionAsset *current_particle_position_asset = NULL;
		Image current_image;
		Image cut_particle;

		wxFileName output_stack_filename = main_frame->current_project.particle_stack_directory.GetFullPath() + wxString::Format("/particle_stack_%li.mrc", refinement_package_asset_panel->current_asset_number);

		// specific package setup..

		temp_refinement_package->stack_box_size = box_size_page->my_panel->BoxSizeSpinCtrl->GetValue();
		temp_refinement_package->stack_filename = output_stack_filename.GetFullPath();
		temp_refinement_package->symmetry = symmetry_page->my_panel->SymmetryComboBox->GetValue();
		temp_refinement_package->estimated_particle_weight_in_kda = molecular_weight_page->my_panel->MolecularWeightTextCtrl->ReturnValue();
		temp_refinement_package->estimated_particle_size_in_angstroms = largest_dimension_page->my_panel->LargestDimensionTextCtrl->ReturnValue();

		// setup the 3ds

		wxWindowList all_children = initial_reference_page->my_panel->ScrollWindow->GetChildren();

		ClassVolumeSelectPanel *panel_pointer;

		for (counter = 0; counter <  all_children.GetCount(); counter++)
		{
			if (all_children.Item(counter)->GetData()->GetClassInfo()->GetClassName() == wxString("wxPanel"))
			{
				panel_pointer = reinterpret_cast <ClassVolumeSelectPanel *> (all_children.Item(counter)->GetData());

				if (panel_pointer->VolumeComboBox->GetCurrentSelection() == 0)
				{
					temp_refinement_package->references_for_next_refinement.Add(-1);
				}
				else
				{
					temp_refinement_package->references_for_next_refinement.Add(volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(panel_pointer->VolumeComboBox->GetCurrentSelection() - 1)->asset_id);
				}
			}
		}


		// size the box..

		cut_particle.Allocate(box_size_page->my_panel->BoxSizeSpinCtrl->GetValue(),box_size_page->my_panel->BoxSizeSpinCtrl->GetValue(),1);

		// open the output stack

		MRCFile output_stack(output_stack_filename.GetFullPath().ToStdString(), true);

		// setup the refinement..

		long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID() + 1;
		temp_refinement_package->refinement_ids.Add(refinement_id);

		temp_refinement.refinement_id = refinement_id;
		temp_refinement.refinement_was_imported_or_generated = true;

		temp_refinement.class_refinement_results.Alloc(temp_refinement_package->number_of_classes);

		temp_refinement.class_refinement_results.Add(junk_class_results, temp_refinement_package->number_of_classes);

		for (class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++)
		{
			temp_refinement.class_refinement_results[class_counter].particle_refinement_results.Alloc(number_of_particles);
			temp_refinement.class_refinement_results[class_counter].particle_refinement_results.Add(junk_result, number_of_particles);
		}

		for (counter = 0; counter < number_of_particles; counter++)
		{
			// current particle, what image is it from?

			if (template_page->my_panel->GroupComboBox->GetSelection() == 0) // completely new..
			{
				current_particle_position_asset = particle_position_asset_panel->ReturnAssetPointer(particle_position_asset_panel->ReturnGroupMember(particle_group_page->my_panel->ParticlePositionsGroupComboBox->GetSelection(), counter));
				current_particle_parent_image_id = current_particle_position_asset->parent_id;
			}
			else
			{
				current_particle_parent_image_id = class_average_particle_infos.Item(counter).parent_image_id;
			}


			if (current_loaded_image_id != current_particle_parent_image_id)
			{
				// load it..

				current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(current_particle_parent_image_id));
				current_image.QuickAndDirtyReadSlice(current_image_asset->filename.GetFullPath().ToStdString(), 1);
				current_loaded_image_id = current_particle_parent_image_id;
				average_value_at_edges = current_image.ReturnAverageOfRealValuesOnEdges();

				// we have to get the defocus stuff from the database..

				main_frame->current_project.database.GetActiveDefocusValuesByImageID(current_particle_parent_image_id, image_defocus_1, image_defocus_2, image_defocus_angle);
			}

			// do the cutting..


			position_in_stack++;

			if (template_page->my_panel->GroupComboBox->GetSelection() == 0) // completely new..
			{
				current_x_pos = myround(current_particle_position_asset->x_position / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_x;
				current_y_pos = myround(current_particle_position_asset->y_position / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_y;
			}
			else
			{
				current_x_pos = myround(class_average_particle_infos.Item(counter).x_pos / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_x;
				current_y_pos = myround(class_average_particle_infos.Item(counter).y_pos / current_image_asset->pixel_size) - current_image.physical_address_of_box_center_y;
			}

			current_image.ClipInto(&cut_particle, average_value_at_edges, false, 1.0, current_x_pos, current_y_pos, 0);
			cut_particle.ZeroFloatAndNormalize();
			cut_particle.WriteSlice(&output_stack, position_in_stack);

			// set the contained particles..

			temp_particle_info.spherical_aberration = current_image_asset->spherical_aberration;
			temp_particle_info.microscope_voltage = current_image_asset->microscope_voltage;
			temp_particle_info.parent_image_id = current_particle_parent_image_id;
			temp_particle_info.pixel_size = current_image_asset->pixel_size;
			temp_particle_info.position_in_stack = position_in_stack;
			temp_particle_info.defocus_1 = image_defocus_1;
			temp_particle_info.defocus_2 = image_defocus_2;
			temp_particle_info.defocus_angle = image_defocus_angle;

			if (template_page->my_panel->GroupComboBox->GetSelection() == 0) // completely new..
			{
				temp_particle_info.x_pos = current_particle_position_asset->x_position;
				temp_particle_info.y_pos = current_particle_position_asset->y_position;
				temp_particle_info.original_particle_position_asset_id = current_particle_position_asset->asset_id;
			}
			else
			{
				temp_particle_info.x_pos = class_average_particle_infos.Item(counter).x_pos;
				temp_particle_info.y_pos = class_average_particle_infos.Item(counter).y_pos;
				temp_particle_info.original_particle_position_asset_id =  class_average_particle_infos.Item(counter).original_particle_position_asset_id;
			}

			temp_refinement_package->contained_particles.Add(temp_particle_info);

			for (class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++)
			{
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].position_in_stack = counter + 1;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus1 = image_defocus_1;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus2 = image_defocus_2;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus_angle = image_defocus_angle;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].logp = 0.0;

				if (temp_refinement_package->number_of_classes == 1) temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
				else temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = fabsf(global_random_number_generator.GetUniformRandom() * 100.0);

				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phi = global_random_number_generator.GetUniformRandom() * 180.0;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].theta = global_random_number_generator.GetUniformRandom() * 180.0;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].psi = global_random_number_generator.GetUniformRandom() * 180.0;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].score = 0.0;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].score_change = 0.0;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].sigma = 1.0;

			}

			my_dialog->Update(counter + 1);
		}

		my_dialog->Destroy();

	}
	else
	{

		long number_of_particles = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles.GetCount();
		OneSecondProgressDialog *my_dialog = new OneSecondProgressDialog ("Refinement Package", "Creating Refinement Package...", number_of_particles, this);

		wxPrintf("Asking for ref pkg = %i\n", template_page->my_panel->GroupComboBox->GetSelection() - 3);
		wxPrintf("Asking for parameter = %i\n", parameter_page->my_panel->GroupComboBox->GetSelection());
		wxPrintf("Ref ID = %li\n", refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].refinement_ids[parameter_page->my_panel->GroupComboBox->GetSelection()]);

		Refinement *refinement_to_copy = main_frame->current_project.database.GetRefinementByID(refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].refinement_ids[parameter_page->my_panel->GroupComboBox->GetSelection()]);

		temp_refinement_package->name = wxString::Format("Refinement Package #%li", refinement_package_asset_panel->current_asset_number);
		temp_refinement_package->number_of_classes = number_of_classes_page->my_panel->NumberOfClassesSpinCtrl->GetValue();
		temp_refinement_package->number_of_run_refinments = 0;
		temp_refinement.resolution_statistics_box_size = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].stack_box_size;

		temp_refinement_package->stack_box_size = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].stack_box_size;
		temp_refinement_package->stack_filename = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].stack_filename;
		temp_refinement_package->symmetry = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].symmetry;
		temp_refinement_package->estimated_particle_weight_in_kda = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].estimated_particle_weight_in_kda;
		temp_refinement_package->estimated_particle_size_in_angstroms = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].estimated_particle_size_in_angstroms;


		long refinement_id = main_frame->current_project.database.ReturnHighestRefinementID() + 1;
		temp_refinement_package->refinement_ids.Add(refinement_id);
		temp_refinement.refinement_id = refinement_id;
		temp_refinement.number_of_classes = temp_refinement_package->number_of_classes;
		temp_refinement.number_of_particles = number_of_particles;
		temp_refinement.name = "Starting Parameters";
		temp_refinement.refinement_package_asset_id = refinement_package_asset_panel->current_asset_number + 1;

		temp_refinement.SizeAndFillWithEmpty(number_of_particles, temp_refinement.number_of_classes);

		wxWindowList all_children = initial_reference_page->my_panel->ScrollWindow->GetChildren();

		// fill the references box

		ClassVolumeSelectPanel *panel_pointer;
		for (long counter = 0; counter <  all_children.GetCount(); counter++)
		{
			if (all_children.Item(counter)->GetData()->GetClassInfo()->GetClassName() == wxString("wxPanel"))
			{
				panel_pointer = reinterpret_cast <ClassVolumeSelectPanel *> (all_children.Item(counter)->GetData());

				if (panel_pointer->VolumeComboBox->GetCurrentSelection() == 0)
				{
					temp_refinement_package->references_for_next_refinement.Add(-1);
				}
				else
				{
					temp_refinement_package->references_for_next_refinement.Add(volume_asset_panel->all_assets_list->ReturnVolumeAssetPointer(panel_pointer->VolumeComboBox->GetCurrentSelection() - 1)->asset_id);
				}
			}
		}


		for (long counter = 0; counter < number_of_particles; counter++)
		{
			// current particle, what image is it from?

			// set the contained particles..

			temp_particle_info.spherical_aberration = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].spherical_aberration;
			temp_particle_info.microscope_voltage = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].microscope_voltage;
			temp_particle_info.original_particle_position_asset_id = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].original_particle_position_asset_id;
			temp_particle_info.parent_image_id = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].parent_image_id;
			temp_particle_info.pixel_size = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].pixel_size;
			temp_particle_info.position_in_stack = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].position_in_stack;
			temp_particle_info.x_pos = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].x_pos;
			temp_particle_info.y_pos = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].y_pos;
			temp_particle_info.defocus_1 = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].defocus_1;
			temp_particle_info.defocus_2 = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].defocus_2;
			temp_particle_info.defocus_angle = refinement_package_asset_panel->all_refinement_packages[template_page->my_panel->GroupComboBox->GetSelection() - 3].contained_particles[counter].defocus_angle;

			temp_refinement_package->contained_particles.Add(temp_particle_info);

			for (class_counter = 0; class_counter < temp_refinement_package->number_of_classes; class_counter++)
			{
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].position_in_stack = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].position_in_stack;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus1 = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].defocus1;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus2 = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].defocus2;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].defocus_angle = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].defocus_angle;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].logp = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].logp;

				if (temp_refinement_package->number_of_classes == 1) temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = 100.0;
				else temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].occupancy = fabsf(global_random_number_generator.GetUniformRandom() * 100.0);

				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].phi = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].phi;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].theta = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].theta;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].psi = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].psi;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].xshift = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].xshift;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].yshift = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].yshift;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].score = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].score;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].score_change = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].score_change;
				temp_refinement.class_refinement_results[class_counter].particle_refinement_results[counter].sigma = refinement_to_copy->class_refinement_results[0].particle_refinement_results[counter].sigma;

				}

				my_dialog->Update(counter + 1);
			}

		my_dialog->Destroy();
		delete refinement_to_copy;

	}

	// now we have to add the refinment_package to the panel..

	refinement_package_asset_panel->AddAsset(temp_refinement_package);
	temp_refinement.resolution_statistics_pixel_size = temp_particle_info.pixel_size;

	for (class_counter = 0; class_counter < temp_refinement.number_of_classes; class_counter++)
	{
		temp_refinement.class_refinement_results[class_counter].class_resolution_statistics.Init(temp_particle_info.pixel_size, temp_refinement.resolution_statistics_box_size);
		temp_refinement.class_refinement_results[class_counter].class_resolution_statistics.GenerateDefaultStatistics(temp_refinement_package->estimated_particle_weight_in_kda);

	}

	main_frame->current_project.database.AddRefinement(&temp_refinement);

	ShortRefinementInfo temp_info;
	temp_info.refinement_id = temp_refinement.refinement_id;
	temp_info.name = temp_refinement.name;
	temp_info.number_of_classes = temp_refinement.number_of_classes;
	temp_info.number_of_particles = temp_refinement.number_of_particles;
	temp_info.refinement_package_asset_id = temp_refinement.refinement_package_asset_id;

	refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);
	//delete temp_refinement_package;


}


////////////////

// TEMPLATE PAGE

/////////////////

 TemplateWizardPage::TemplateWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
 : wxWizardPage(parent, bitmap)
 {
	 Freeze();
	 wizard_pointer = parent;
	 wxBoxSizer* main_sizer;
	 my_panel = new TemplateWizardPanel(this);

	 main_sizer = new wxBoxSizer( wxVERTICAL );
	 this->SetSizer(main_sizer);
	 main_sizer->Fit(this);
	 main_sizer->Add(my_panel);
	// my_panel->InfoText->AutoWrap();

	// my_panel->GroupComboBox->Freeze();
	 my_panel->GroupComboBox->ChangeValue("");
	 my_panel->GroupComboBox->Clear();

	 my_panel->GroupComboBox->Append ("New Refinement Package");
	 my_panel->GroupComboBox->Append ("Create From 2D Class Average Selection");
	 my_panel->GroupComboBox->Append ("--------------------------------------------------------");

	 for (int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	 {
		my_panel->GroupComboBox->Append (refinement_package_asset_panel->all_refinement_packages.Item(counter).name);
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
	Thaw();
	/*int width, height;
	my_panel->GetClientSize(&width, &height);
	my_panel->InfoText->Wrap(height);
	my_panel->Fit();*/
}



 wxWizardPage *  TemplateWizardPage::GetNext () const
 {
	// wxPrintf("Template Next\n");
	 if (my_panel->GroupComboBox->GetSelection() == 0) return wizard_pointer->particle_group_page;
	 else
	 if (my_panel->GroupComboBox->GetSelection() == 1) return wizard_pointer->class_selection_page;
	 else
	 return wizard_pointer->parameter_page;

 }


 ////////////////

 // INPUT PARAMETERS PAGE

 /////////////////

  InputParameterWizardPage::InputParameterWizardPage(MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
  : wxWizardPage(parent, bitmap)
  {
 	 Freeze();
 	 wizard_pointer = parent;
 	 wxBoxSizer* main_sizer;
 	 my_panel = new InputParameterWizardPanel(this);

 	 main_sizer = new wxBoxSizer( wxVERTICAL );
 	 this->SetSizer(main_sizer);
 	 main_sizer->Fit(this);
 	 main_sizer->Add(my_panel);

 	 my_panel->GroupComboBox->ChangeValue("");
 	 my_panel->GroupComboBox->Clear();


 	 Thaw();

 }

  wxWizardPage *  InputParameterWizardPage::GetPrev () const
  {
	  return wizard_pointer->template_page;

  }

  wxWizardPage *  InputParameterWizardPage::GetNext () const
  {
 	// wxPrintf("Template Next\n");

	 return wizard_pointer->molecular_weight_page;


  }


 //////////////////////

 // Particle Group Page

 //////////////////////

 ParticleGroupWizardPage::ParticleGroupWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
 : wxWizardPage(parent, bitmap)
 {
	wizard_pointer = parent;
	wxBoxSizer* main_sizer;
	my_panel = new ParticleGroupWizardPanel(this);

	main_sizer = new wxBoxSizer( wxVERTICAL );
	this->SetSizer(main_sizer);
	main_sizer->Fit(this);
	main_sizer->Add(my_panel);
//	my_panel->InfoText->AutoWrap();

	FillParticlePositionsGroupComboBox(my_panel->ParticlePositionsGroupComboBox);

 }

 wxWizardPage *  ParticleGroupWizardPage::GetPrev () const
 {
	// wxPrintf("Particle Prev\n");
	 return wizard_pointer->template_page;
 }

 wxWizardPage *  ParticleGroupWizardPage::GetNext () const
  {
	// wxPrintf("Particle Next\n");
	 return wizard_pointer->box_size_page;
  }

 //////////////////////////

 // BOX SIZE PAGE

 ////////////////////////////

 BoxSizeWizardPage::BoxSizeWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
  : wxWizardPage(parent, bitmap)
  {
 	wizard_pointer = parent;
 	wxBoxSizer* main_sizer;
 	my_panel = new BoxSizeWizardPanel(this);

 	main_sizer = new wxBoxSizer( wxVERTICAL );
	this->SetSizer(main_sizer);
	main_sizer->Fit(this);
 	main_sizer->Add(my_panel);
	//my_panel->InfoText->AutoWrap();
  }

  wxWizardPage *  BoxSizeWizardPage::GetPrev () const
  {
	  if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() == 0) return wizard_pointer->particle_group_page;
	  else
	  return wizard_pointer->class_selection_page;


  }

  wxWizardPage *  BoxSizeWizardPage::GetNext () const
  {
	//  wxPrintf("Box Next\n");
  	 return wizard_pointer->molecular_weight_page;
  }

  //////////////////////////

  // Molecular weight PAGE

  ////////////////////////////

  MolecularWeightWizardPage::MolecularWeightWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
   : wxWizardPage(parent, bitmap)
   {
  	wizard_pointer = parent;
  	wxBoxSizer* main_sizer;
  	my_panel = new MolecularWeightWizardPanel(this);

  	main_sizer = new wxBoxSizer( wxVERTICAL );
 	this->SetSizer(main_sizer);
 	main_sizer->Fit(this);
  	main_sizer->Add(my_panel);
 //	my_panel->InfoText->AutoWrap();
   }

   wxWizardPage *  MolecularWeightWizardPage::GetPrev () const
   {
 	//  wxPrintf("Box Prev\n");
	   if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() > 1) return wizard_pointer->parameter_page;
	   else return wizard_pointer->box_size_page;
   }

   wxWizardPage *  MolecularWeightWizardPage::GetNext () const
   {
 	//  wxPrintf("Box Next\n");
   	 return wizard_pointer->largest_dimension_page;
   }

   //////////////////////////

   // largest dimension PAGE

   ////////////////////////////

   LargestDimensionWizardPage::LargestDimensionWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
    : wxWizardPage(parent, bitmap)
    {
   	wizard_pointer = parent;
   	wxBoxSizer* main_sizer;
   	my_panel = new LargestDimensionWizardPanel(this);

   	main_sizer = new wxBoxSizer( wxVERTICAL );
  	this->SetSizer(main_sizer);
  	main_sizer->Fit(this);
   	main_sizer->Add(my_panel);
  //	my_panel->InfoText->AutoWrap();
    }

    wxWizardPage *  LargestDimensionWizardPage::GetPrev () const
    {
  	//  wxPrintf("Box Prev\n");
   	 return wizard_pointer->molecular_weight_page;
    }

    wxWizardPage *  LargestDimensionWizardPage::GetNext () const
    {
  	//  wxPrintf("Box Next\n");
    	 return wizard_pointer->symmetry_page;
    }




  //////////////////////////

  // symmetry PAGE

  ////////////////////////////

  SymmetryWizardPage::SymmetryWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
   : wxWizardPage(parent, bitmap)
   {
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
  	my_panel->SymmetryComboBox->SetSelection(0);

  	main_sizer = new wxBoxSizer( wxVERTICAL );
 	this->SetSizer(main_sizer);
 	main_sizer->Fit(this);
  	main_sizer->Add(my_panel);
 //	my_panel->InfoText->AutoWrap();
   }

   wxWizardPage *  SymmetryWizardPage::GetPrev () const
   {
 	//  wxPrintf("Box Prev\n");
  	 return wizard_pointer->largest_dimension_page;
   }

   wxWizardPage *  SymmetryWizardPage::GetNext () const
   {
 	//  wxPrintf("Box Next\n");
   	 return wizard_pointer->number_of_classes_page;
   }




 /////////////////////////

 //  NUMBER OF CLASSES PAGE

 /////////////////////////////

 NumberofClassesWizardPage::NumberofClassesWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
 : wxWizardPage(parent, bitmap)
 {
	wizard_pointer = parent;
	wxBoxSizer* main_sizer;
	my_panel = new NumberofClassesWizardPanel(this);

	main_sizer = new wxBoxSizer( wxVERTICAL );
	this->SetSizer(main_sizer);
	main_sizer->Fit(this);
	main_sizer->Add(my_panel);
	my_panel->InfoText->AutoWrap();
 }


 wxWizardPage *  NumberofClassesWizardPage::GetNext () const
 {
	 //if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() == 0) return NULL;
	 //else return wizard_pointer->class_setup_page;

	// wxPrintf("Number classes Next\n");
	if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() > 1) return wizard_pointer->class_setup_page;
	else return wizard_pointer->initial_reference_page;
 }



 wxWizardPage *  NumberofClassesWizardPage::GetPrev () const
 {
	 return wizard_pointer->symmetry_page;
 }

 // ///////////////////////////

 // INITIAL REFERNECES
 /////

 InitialReferencesWizardPage::InitialReferencesWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
 : wxWizardPage(parent, bitmap)
 {
	wizard_pointer = parent;
	wxBoxSizer* main_sizer;
	CreatePanel();

	main_sizer = new wxBoxSizer( wxVERTICAL );
	this->SetSizer(main_sizer);
	main_sizer->Fit(this);
	main_sizer->Add(my_panel);
	my_panel->InfoText->AutoWrap();

 }

 void InitialReferencesWizardPage::CreatePanel()
 {
	 my_panel = new InitialReferenceSelectWizardPanel(this);
 }


 wxWizardPage *  InitialReferencesWizardPage::GetNext () const
 {
//	 wxPrintf("Initial Next\n");
	 return NULL;
 }



 wxWizardPage *  InitialReferencesWizardPage::GetPrev () const
 {
	// wxPrintf("Initial Prev\n");
	if (wizard_pointer->template_page->my_panel->GroupComboBox->GetSelection() > 1) return wizard_pointer->class_setup_page;
	else return wizard_pointer->number_of_classes_page;
 }


 ////////////////////////////

 //  Classes Selection Page

 /////////////////////////////

 ClassSelectionWizardPage::ClassSelectionWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
 : wxWizardPage(parent, bitmap)
 {
	wizard_pointer = parent;
	wxBoxSizer* main_sizer;
	my_panel = new ClassSelectionWizardPanel(this);

	main_sizer = new wxBoxSizer( wxVERTICAL );
	this->SetSizer(main_sizer);
	main_sizer->Fit(this);
	main_sizer->Add(my_panel);

	int counter;
	int old_width;
	int current_width;

	Freeze();

	my_panel->SelectionListCtrl->ClearAll();
	my_panel->SelectionListCtrl->InsertColumn(0, wxT("Selection"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	my_panel->SelectionListCtrl->	InsertColumn(1, wxT("Creation Date"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	my_panel->SelectionListCtrl->	InsertColumn(2, wxT("Refinement Package"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	my_panel->SelectionListCtrl->InsertColumn(3, wxT("No. Selected"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );

	for (counter = 0; counter < refinement_package_asset_panel->all_classification_selections.GetCount(); counter++)
	{
		RefinementPackage *parent_package = &refinement_package_asset_panel->all_refinement_packages.Item(refinement_package_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_classification_selections.Item(counter).refinement_package_asset_id));

		my_panel->SelectionListCtrl->InsertItem(counter, refinement_package_asset_panel->all_classification_selections.Item(counter).name);
		my_panel->SelectionListCtrl->SetItem (counter, 1, refinement_package_asset_panel->all_classification_selections.Item(counter).creation_date.FormatISOCombined(' '));
		my_panel->SelectionListCtrl->SetItem (counter, 2, parent_package->name);
		my_panel->SelectionListCtrl->SetItem (counter, 3, wxString::Format("%i/%i", refinement_package_asset_panel->all_classification_selections.Item(counter).number_of_selections, refinement_package_asset_panel->all_classification_selections.Item(counter).number_of_classes));
	}

	for (counter = 0; counter < my_panel->SelectionListCtrl->GetColumnCount(); counter++)
	{
		old_width = my_panel->SelectionListCtrl->GetColumnWidth(counter);
		my_panel->SelectionListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE);
		current_width = my_panel->SelectionListCtrl->GetColumnWidth(counter);

		if (old_width > current_width) my_panel->SelectionListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE_USEHEADER);
	}

	Thaw();
 }


 wxWizardPage *  ClassSelectionWizardPage::GetNext () const
 {
	 //wxPrintf("Classes Next\n");
	 return wizard_pointer->box_size_page;

 }



 wxWizardPage *  ClassSelectionWizardPage::GetPrev () const
 {
	 return wizard_pointer->template_page;
 }


 ////////////////////////////

 //  Classes Setup Page

 /////////////////////////////

 ClassesSetupWizardPage::ClassesSetupWizardPage (MyNewRefinementPackageWizard *parent, const wxBitmap &bitmap)
 : wxWizardPage(parent, bitmap)
 {
	wizard_pointer = parent;
	wxBoxSizer* main_sizer;
	my_panel = new ClassesSetupWizardPanel(this);

	main_sizer = new wxBoxSizer( wxVERTICAL );
	this->SetSizer(main_sizer);
	main_sizer->Fit(this);
	main_sizer->Add(my_panel);
 }


 wxWizardPage *  ClassesSetupWizardPage::GetNext () const
 {
	return wizard_pointer->initial_reference_page;
 }



 wxWizardPage *  ClassesSetupWizardPage::GetPrev () const
 {
	 return wizard_pointer->number_of_classes_page;
 }
