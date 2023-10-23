//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyImageAssetPanel*                image_asset_panel;
extern MyVolumeAssetPanel*               volume_asset_panel;
extern TemplateMatchesPackageAssetPanel* template_matches_package_asset_panel;

NewTemplateMatchesPackageWizard::NewTemplateMatchesPackageWizard(wxWindow* parent)
    : NewTemplateMatchesPackageWizardParent(parent) {
    page = new TemplateMatchesWizardPage(this);

    GetPageAreaSizer( )->Add(page);
    num_selected_matches = 0;
    //GetPageAreaSizer()->Add(particle_group_page);
    //GetPageAreaSizer()->Add(number_of_classes_page);
    //GetPageAreaSizer()->Add(box_size_page);

    Bind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(NewTemplateMatchesPackageWizard::OnUpdateUI), this);
}

NewTemplateMatchesPackageWizard::~NewTemplateMatchesPackageWizard( ) {
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
    Unbind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(NewTemplateMatchesPackageWizard::OnUpdateUI), this);
}

void NewTemplateMatchesPackageWizard::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( num_selected_matches == 0 )
        DisableNextButton( );
    else
        EnableNextButton( );
}

void NewTemplateMatchesPackageWizard::DisableNextButton( ) {
    wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
    if ( win )
        win->Enable(false);
}

void NewTemplateMatchesPackageWizard::EnableNextButton( ) {
    wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
    if ( win )
        win->Enable(true);
}

void NewTemplateMatchesPackageWizard::PageChanging(wxWizardEvent& event) {
}

void NewTemplateMatchesPackageWizard::PageChanged(wxWizardEvent& event) {

    if ( event.GetPage( ) == page ) {
        if ( page->my_panel->InfoText->has_autowrapped == false ) {
            page->Freeze( );
            page->my_panel->InfoText->AutoWrap( );
            page->Layout( );
            page->Thaw( );
        }
    }
}

void NewTemplateMatchesPackageWizard::OnFinished(wxWizardEvent& event) {
    cisTEMParameters output_params;
    output_params.parameters_to_write.SetActiveParameters(PSI | THETA | PHI | ORIGINAL_X_POSITION | ORIGINAL_Y_POSITION | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | SCORE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y | ORIGINAL_IMAGE_FILENAME | PIXEL_SIZE);
    output_params.PreallocateMemoryAndBlank(num_selected_matches);

    long     match_counter = 0;
    int      current_image_id, current_tm_id;
    bool     more_data;
    wxString image_filename;
    double   x_shift, y_shift, psi, theta, phi, defocus, score;
    double   acceleration_volatage, spherical_aberration, amplitude_contrast, defocus_1, defocus_2, defocus_angle, additional_phase_shift, iciness;
    for ( long image_counter = 0; image_counter < image_ids.GetCount( ); image_counter++ ) {
        current_image_id = image_ids[image_counter];
        current_tm_id    = tm_ids[image_counter];
        main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT * FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=%i", current_image_id));
        ImageAsset temp_asset = main_frame->current_project.database.GetNextImageAsset( );
        main_frame->current_project.database.EndBatchSelect( );
        image_filename = temp_asset.filename.GetFullPath( );
        main_frame->current_project.database.GetCTFParameters(temp_asset.ctf_estimation_id, acceleration_volatage, spherical_aberration, amplitude_contrast, defocus_1, defocus_2, defocus_angle, additional_phase_shift, iciness);
        more_data = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT X_POSITION, Y_POSITION, PSI, THETA, PHI, DEFOCUS, PEAK_HEIGHT from TEMPLATE_MATCH_PEAK_LIST_%i", current_tm_id));
        while ( more_data ) {
            output_params.all_parameters[match_counter].original_image_filename            = image_filename;
            output_params.all_parameters[match_counter].microscope_voltage_kv              = acceleration_volatage;
            output_params.all_parameters[match_counter].microscope_spherical_aberration_mm = spherical_aberration;
            output_params.all_parameters[match_counter].amplitude_contrast                 = amplitude_contrast;
            output_params.all_parameters[match_counter].defocus_angle                      = defocus_angle;
            output_params.all_parameters[match_counter].phase_shift                        = additional_phase_shift;
            more_data                                                                      = main_frame->current_project.database.GetFromBatchSelect("rrrrrrr", &x_shift, &y_shift, &psi, &theta, &phi, &defocus, &score);
            output_params.all_parameters[match_counter].original_x_position                = x_shift;
            output_params.all_parameters[match_counter].original_y_position                = y_shift;
            output_params.all_parameters[match_counter].psi                                = psi;
            output_params.all_parameters[match_counter].theta                              = theta;
            output_params.all_parameters[match_counter].phi                                = phi;
            output_params.all_parameters[match_counter].defocus_1                          = defocus_1 + defocus;
            output_params.all_parameters[match_counter].defocus_2                          = defocus_2 + defocus;
            output_params.all_parameters[match_counter].score                              = score;
            output_params.all_parameters[match_counter].pixel_size                         = temp_asset.pixel_size;
            match_counter++;
        }
        main_frame->current_project.database.EndBatchSelect( );
    }
    MyDebugPrint("Writing %li matches to cisTEM star file", match_counter);
    int                    id                = main_frame->current_project.database.ReturnHighestTemplateMatchesPackageID( ) + 1;
    std::string            starfile_filename = std::string((main_frame->current_project.template_matching_asset_directory.GetFullPath( ) + wxString::Format("/template_matches_package_%i.star", id)).mb_str( ));
    TemplateMatchesPackage temp_package;
    temp_package.name                  = wxString::Format("Template Matches Package %i", id);
    temp_package.starfile_filename     = starfile_filename;
    temp_package.contained_match_count = match_counter;
    temp_package.asset_id              = id;
    MyDebugPrint("Doing actual write");
    main_frame->current_project.database.AddTemplateMatchesPackageAsset(&temp_package);
    output_params.WriteTocisTEMStarFile(starfile_filename, -1, -1, 0, -1);
    template_matches_package_asset_panel->ImportAllFromDatabase( );
    //main_frame->DirtyTemplateMatchesPackages( );
}

////////////////

// TEMPLATE MATCHES PAGE

/////////////////

TemplateMatchesWizardPage::TemplateMatchesWizardPage(NewTemplateMatchesPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    Freeze( );
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new TemplateMatchesWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    my_panel->ImageGroupComboBox->FillComboBox(true);
    my_panel->TMJobComboBox->FillComboBox( );
    Thaw( );

    my_panel->ImageGroupComboBox->Bind(wxEVT_COMBOBOX, wxCommandEventHandler(TemplateMatchesWizardPage::SelectionChanged), this);
    my_panel->TMJobComboBox->Bind(wxEVT_COMBOBOX, wxCommandEventHandler(TemplateMatchesWizardPage::SelectionChanged), this);
}

TemplateMatchesWizardPage::~TemplateMatchesWizardPage( ) {
    my_panel->ImageGroupComboBox->Unbind(wxEVT_COMBOBOX, wxCommandEventHandler(TemplateMatchesWizardPage::SelectionChanged), this);
    my_panel->TMJobComboBox->Unbind(wxEVT_COMBOBOX, wxCommandEventHandler(TemplateMatchesWizardPage::SelectionChanged), this);
}

void TemplateMatchesWizardPage::SelectionChanged(wxCommandEvent& event) {
    int tm_sel = my_panel->TMJobComboBox->GetSelection( );
    int im_sel = my_panel->ImageGroupComboBox->GetSelection( );
    if ( tm_sel == -1 )
        return;
    int  tm_job   = my_panel->TMJobComboBox->AssetComboBox->associated_ids[tm_sel];
    int  im_group = my_panel->ImageGroupComboBox->AssetComboBox->associated_ids[im_sel];
    bool more_data;
    wizard_pointer->tm_ids.Clear( );
    wizard_pointer->image_ids.Clear( );
    if ( im_group > 0 )
        more_data = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT TEMPLATE_MATCH_ID, TEMPLATE_MATCH_LIST.IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST INNER JOIN IMAGE_GROUP_%i ON TEMPLATE_MATCH_LIST.IMAGE_ASSET_ID = IMAGE_GROUP_%i.IMAGE_ASSET_ID WHERE TEMPLATE_MATCH_JOB_ID=%i", im_group, im_group, tm_job));
    else
        more_data = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT TEMPLATE_MATCH_ID, TEMPLATE_MATCH_LIST.IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID=%i", tm_job));
    while ( more_data == true ) {
        int tm_id, image_id;
        more_data = main_frame->current_project.database.GetFromBatchSelect("ii", &tm_id, &image_id);
        wizard_pointer->tm_ids.Add(tm_id);
        wizard_pointer->image_ids.Add(image_id);
    }
    main_frame->current_project.database.EndBatchSelect( );
    // Iterate through the template matches and count the matches
    wizard_pointer->num_selected_matches = 0;
    for ( int i = 0; i < wizard_pointer->tm_ids.GetCount( ); i++ ) {
        wizard_pointer->num_selected_matches += main_frame->current_project.database.ReturnSingleLongFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM TEMPLATE_MATCH_PEAK_LIST_%i", wizard_pointer->tm_ids[i]));
    }
    my_panel->InfoText->SetLabel(wxString::Format("You have selected %li template matches across %li images from job %i", wizard_pointer->num_selected_matches, wizard_pointer->tm_ids.GetCount( ), tm_job));
}
