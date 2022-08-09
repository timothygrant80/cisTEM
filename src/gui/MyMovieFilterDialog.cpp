//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyFilterDialog::MyFilterDialog(wxWindow* parent)
    : FilterDialog(parent) {
}

void MyFilterDialog::OnCancelClick(wxCommandEvent& event) {
    Destroy( );
}

void MyFilterDialog::OnFilterClick(wxCommandEvent& event) {
    BuildSearchCommand( );
    EndModal(wxID_OK);
}

void MyFilterDialog::SizeAndPosition( ) {
    wxSize filter_input_size = FilterBoxSizer->GetMinSize( );
    wxSize sort_input_size   = SortSizer->GetMinSize( );

    int frame_width;
    int frame_height;
    int frame_position_x;
    int frame_position_y;

    main_frame->GetClientSize(&frame_width, &frame_height);
    main_frame->GetPosition(&frame_position_x, &frame_position_y);

    int total_height = filter_input_size.y + sort_input_size.y;

    filter_input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
    sort_input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);

    if ( total_height > frame_height * 0.9 ) {
        filter_input_size.y -= (total_height - frame_height * 0.9);
    }

    FilterScrollPanel->SetMinSize(filter_input_size);
    FilterScrollPanel->SetSize(filter_input_size);

    SortScrollPanel->SetMinSize(sort_input_size);
    SortScrollPanel->SetSize(sort_input_size);

    SetMaxSize(wxSize(frame_width, frame_height * 0.91));
    Layout( );
    MainBoxSizer->Fit(this);

    int dialog_height;
    int dialog_width;

    // ok so how big is this dialog now?

    GetSize(&dialog_width, &dialog_height);

    int new_x_pos = (frame_position_x + (frame_width / 2) - (dialog_width / 2));
    int new_y_pos = (frame_position_y + (frame_height / 2) - (dialog_height / 2));

    Move(new_x_pos, new_y_pos);
}

// Movies//////////////////////////////////

MyMovieFilterDialog::MyMovieFilterDialog(wxWindow* parent)
    : MyFilterDialog(parent) {
    // add the filter checkboxes..

    asset_id_filter = new IntegerFilterItem("Asset ID", FilterScrollPanel);
    FilterBoxSizer->Add(asset_id_filter, 1, wxEXPAND | wxALL, 5);

    alignment_id_filter = new IntegerFilterItem("Alignment ID", FilterScrollPanel);
    FilterBoxSizer->Add(alignment_id_filter, 1, wxEXPAND | wxALL, 5);

    date_of_run_filter = new DateFilterItem("Date of Run", FilterScrollPanel);
    FilterBoxSizer->Add(date_of_run_filter, 1, wxEXPAND | wxALL, 5);

    job_id_filter = new IntegerFilterItem("Job ID", FilterScrollPanel);
    FilterBoxSizer->Add(job_id_filter, 0, wxEXPAND | wxALL, 5);

    voltage_filter = new FloatFilterItem("Voltage", FilterScrollPanel);
    FilterBoxSizer->Add(voltage_filter, 0, wxEXPAND | wxALL, 5);

    pixel_size_filter = new FloatFilterItem("Pixel Size", FilterScrollPanel);
    FilterBoxSizer->Add(pixel_size_filter, 0, wxEXPAND | wxALL, 5);

    exposure_per_frame_filter = new FloatFilterItem("Exposure Per-Frame", FilterScrollPanel);
    FilterBoxSizer->Add(exposure_per_frame_filter, 01, wxEXPAND | wxALL, 5);

    pre_exposure_filter = new FloatFilterItem("Pre-Exposure Amount", FilterScrollPanel);
    FilterBoxSizer->Add(pre_exposure_filter, 0, wxEXPAND | wxALL, 5);

    // Add the sort combo boxes..

    AssetIDRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("AssetID"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(AssetIDRadioButton, 0, wxALL, 5);

    AlignmentIDRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Alignment ID"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(AlignmentIDRadioButton, 0, wxALL, 5);

    DateOfRunRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Date of Run"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(DateOfRunRadioButton, 0, wxALL, 5);

    JobIDRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Job ID"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(JobIDRadioButton, 0, wxALL, 5);

    VoltageRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Voltage"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(VoltageRadioButton, 0, wxALL, 5);

    PixelSizeRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Pixel Size"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(PixelSizeRadioButton, 0, wxALL, 5);

    ExposureRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Exposure"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(ExposureRadioButton, 0, wxALL, 5);

    PreExposureRadioButton = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Pre-Exposure"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(PreExposureRadioButton, 0, wxALL, 5);

    // resize..

    SizeAndPosition( );
}

void MyMovieFilterDialog::BuildSearchCommand( ) {

    int number_checked       = ReturnNumberChecked( );
    int number_accounted_for = 0;

    search_command = "SELECT DISTINCT MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID FROM MOVIE_ALIGNMENT_LIST,IMAGE_ASSETS WHERE MOVIE_ALIGNMENT_LIST.ALIGNMENT_ID=IMAGE_ASSETS.ALIGNMENT_ID AND IMAGE_ASSETS.PARENT_MOVIE_ID=MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID";

    if ( number_checked > 0 ) {
        search_command += " AND";

        if ( asset_id_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID BETWEEN %i AND %i", asset_id_filter->GetLowValue( ), asset_id_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( alignment_id_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.ALIGNMENT_ID BETWEEN %i AND %i", alignment_id_filter->GetLowValue( ), alignment_id_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( date_of_run_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.DATETIME_OF_RUN BETWEEN %li AND %li", date_of_run_filter->GetLowValue( ), date_of_run_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( job_id_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.ALIGNMENT_JOB_ID BETWEEN %f AND %f", job_id_filter->GetLowValue( ), job_id_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( voltage_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.VOLTAGE BETWEEN %f AND %f", voltage_filter->GetLowValue( ), voltage_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( pixel_size_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.PIXEL_SIZE BETWEEN %f AND %f", pixel_size_filter->GetLowValue( ), pixel_size_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( exposure_per_frame_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.EXPOSURE_PER_FRAME BETWEEN %f AND %f", exposure_per_frame_filter->GetLowValue( ), exposure_per_frame_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( pre_exposure_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" MOVIE_ALIGNMENT_LIST.PRE_EXPOSURE_AMOUNT BETWEEN %f AND %f", exposure_per_frame_filter->GetLowValue( ), exposure_per_frame_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }
    }

    // do the ordering

    if ( AssetIDRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID";
    else if ( AlignmentIDRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.ALIGNMENT_ID";
    else if ( DateOfRunRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.DATETIME_OF_RUN";
    else if ( JobIDRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.ALIGNMENT_JOB_ID";
    else if ( VoltageRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.VOLTAGE";
    else if ( PixelSizeRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.PIXEL_SIZE";
    else if ( ExposureRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.EXPOSURE_PER_FRAME";
    else if ( PreExposureRadioButton->GetValue( ) == true )
        search_command += " ORDER BY MOVIE_ALIGNMENT_LIST.PRE_EXPOSURE_AMOUNT";

    //SELECT MOVIE_ASSET_ID, ALIGNMENT_JOB_ID FROM MOVIE_ALIGNMENT_LIST, IMAGE_ASSETS WHERE MOVIE_ALIGNMENT_LIST.ALIGNMENT_ID=IMAGE_ASSETS.ALIGNMENT_ID AND IMAGE_ASSETS.PARENT_MOVIE_ID=MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID;");
}

int MyMovieFilterDialog::ReturnNumberChecked( ) {
    int number_checked = 0;

    if ( asset_id_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( alignment_id_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( date_of_run_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( job_id_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( voltage_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( pixel_size_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( exposure_per_frame_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( pre_exposure_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;

    return number_checked;
}

/// CTF

MyCTFFilterDialog::MyCTFFilterDialog(wxWindow* parent)
    : MyFilterDialog(parent) {
    SetTitle("Filter / Sort Images");

    // add the filter checkboxes..

    score_filter = new FloatFilterItem("Score", FilterScrollPanel);
    FilterBoxSizer->Add(score_filter, 1, wxEXPAND | wxALL, 5);

    ring_resolution_filter = new FloatFilterItem("Detected Fit Res.", FilterScrollPanel);
    FilterBoxSizer->Add(ring_resolution_filter, 1, wxEXPAND | wxALL, 5);

    alias_resolution_filter = new FloatFilterItem("Detected Alias Res.", FilterScrollPanel);
    FilterBoxSizer->Add(alias_resolution_filter, 1, wxEXPAND | wxALL, 5);

    additional_phase_shift_filter = new FloatFilterItem("Phase Shift", FilterScrollPanel);
    FilterBoxSizer->Add(additional_phase_shift_filter, 1, wxEXPAND | wxALL, 5);

    iciness_filter = new FloatFilterItem("Iciness", FilterScrollPanel);
    FilterBoxSizer->Add(iciness_filter, 1, wxEXPAND | wxALL, 5);

    defocus_filter = new FloatFilterItem("Defocus", FilterScrollPanel);
    FilterBoxSizer->Add(defocus_filter, 1, wxEXPAND | wxALL, 5);

    astigmatism_filter = new FloatFilterItem("Astigmatism", FilterScrollPanel);
    FilterBoxSizer->Add(astigmatism_filter, 1, wxEXPAND | wxALL, 5);

    astigmatism_angle_filter = new FloatFilterItem("Astigmatism Angle", FilterScrollPanel);
    FilterBoxSizer->Add(astigmatism_angle_filter, 1, wxEXPAND | wxALL, 5);

    ctf_tilt_angle_filter = new FloatFilterItem("Tilt Angle", FilterScrollPanel);
    FilterBoxSizer->Add(ctf_tilt_angle_filter, 1, wxEXPAND | wxALL, 5);

    ctf_tilt_axis_filter = new FloatFilterItem("Tilt Axis", FilterScrollPanel);
    FilterBoxSizer->Add(ctf_tilt_axis_filter, 1, wxEXPAND | wxALL, 5);

    asset_id_filter = new IntegerFilterItem("Asset ID", FilterScrollPanel);
    FilterBoxSizer->Add(asset_id_filter, 1, wxEXPAND | wxALL, 5);

    estimation_id_filter = new IntegerFilterItem("Estimation ID", FilterScrollPanel);
    FilterBoxSizer->Add(estimation_id_filter, 1, wxEXPAND | wxALL, 5);

    date_of_run_filter = new DateFilterItem("Date of Run", FilterScrollPanel);
    FilterBoxSizer->Add(date_of_run_filter, 1, wxEXPAND | wxALL, 5);

    job_id_filter = new IntegerFilterItem("Job ID", FilterScrollPanel);
    FilterBoxSizer->Add(job_id_filter, 1, wxEXPAND | wxALL, 5);

    voltage_filter = new FloatFilterItem("Voltage", FilterScrollPanel);
    FilterBoxSizer->Add(voltage_filter, 1, wxEXPAND | wxALL, 5);

    spherical_aberration_filter = new FloatFilterItem("Spherical Aberration", FilterScrollPanel);
    FilterBoxSizer->Add(spherical_aberration_filter, 1, wxEXPAND | wxALL, 5);

    pixel_size_filter = new FloatFilterItem("Pixel Size", FilterScrollPanel);
    FilterBoxSizer->Add(pixel_size_filter, 1, wxEXPAND | wxALL, 5);

    amplitude_contrast_filter = new FloatFilterItem("Amplitude Contrast", FilterScrollPanel);
    FilterBoxSizer->Add(amplitude_contrast_filter, 1, wxEXPAND | wxALL, 5);

    box_size_filter = new IntegerFilterItem("Box Size", FilterScrollPanel);
    FilterBoxSizer->Add(box_size_filter, 1, wxEXPAND | wxALL, 5);

    min_resolution_filter = new FloatFilterItem("Min. Resolution", FilterScrollPanel);
    FilterBoxSizer->Add(min_resolution_filter, 1, wxEXPAND | wxALL, 5);

    max_resolution_filter = new FloatFilterItem("Max. Resolution", FilterScrollPanel);
    FilterBoxSizer->Add(max_resolution_filter, 1, wxEXPAND | wxALL, 5);

    min_defocus_filter = new FloatFilterItem("Min. Defocus", FilterScrollPanel);
    FilterBoxSizer->Add(min_defocus_filter, 1, wxEXPAND | wxALL, 5);

    max_defocus_filter = new FloatFilterItem("Max. Defocus", FilterScrollPanel);
    FilterBoxSizer->Add(max_defocus_filter, 1, wxEXPAND | wxALL, 5);

    defocus_step_filter = new FloatFilterItem("Defocus Step", FilterScrollPanel);
    FilterBoxSizer->Add(defocus_step_filter, 1, wxEXPAND | wxALL, 5);

    tolerated_astigmatism_filter = new FloatFilterItem("Tolerated Astig.", FilterScrollPanel);
    FilterBoxSizer->Add(tolerated_astigmatism_filter, 1, wxEXPAND | wxALL, 5);

    min_phase_shift_filter = new FloatFilterItem("Min. Phase Shift", FilterScrollPanel);
    FilterBoxSizer->Add(min_phase_shift_filter, 1, wxEXPAND | wxALL, 5);

    max_phase_shift_filter = new FloatFilterItem("Max. Phase Shift", FilterScrollPanel);
    FilterBoxSizer->Add(max_phase_shift_filter, 1, wxEXPAND | wxALL, 5);

    phase_shift_step_filter = new FloatFilterItem("Phase Shift Step", FilterScrollPanel);
    FilterBoxSizer->Add(phase_shift_step_filter, 1, wxEXPAND | wxALL, 5);

    // Add the sort combo boxes..

    asset_id_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Asset ID"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(asset_id_radio, 0, wxALL, 5);

    estimation_id_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Estimation ID"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(estimation_id_radio, 0, wxALL, 5);

    date_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Date of Run"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(date_radio, 0, wxALL, 5);

    job_id_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Job ID"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(job_id_radio, 0, wxALL, 5);

    defocus_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Defocus"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(defocus_radio, 0, wxALL, 5);

    astigmatism_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Astigmatism"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(astigmatism_radio, 0, wxALL, 5);

    astigmatism_angle_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Astigmatism Angle"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(astigmatism_angle_radio, 0, wxALL, 5);

    ctf_tilt_angle_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Tilt Angle"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(ctf_tilt_angle_radio, 0, wxALL, 5);

    ctf_tilt_axis_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Tilt Axis"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(ctf_tilt_axis_radio, 0, wxALL, 5);

    score_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Score"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(score_radio, 0, wxALL, 5);

    ring_resolution_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Detected Fit Res."), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(ring_resolution_radio, 0, wxALL, 5);

    alias_resolution_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Detected Alias Res."), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(alias_resolution_radio, 0, wxALL, 5);

    additional_phase_shift_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Phase Shift"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(additional_phase_shift_radio, 0, wxALL, 5);

    iciness_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Iciness"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(iciness_radio, 0, wxALL, 5);

    voltage_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Voltage"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(voltage_radio, 0, wxALL, 5);

    spherical_aberration_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Spherical Aberration"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(spherical_aberration_radio, 0, wxALL, 5);

    pixel_size_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Pixel Size"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(pixel_size_radio, 0, wxALL, 5);

    amplitude_contrast_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Amplitude Contrast"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(amplitude_contrast_radio, 0, wxALL, 5);

    box_size_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Box Size"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(box_size_radio, 0, wxALL, 5);

    min_resolution_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Min. Resolution"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(min_resolution_radio, 0, wxALL, 5);

    max_resolution_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Max. Resolution"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(max_resolution_radio, 0, wxALL, 5);

    min_defocus_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Min. Defocus"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(min_defocus_radio, 0, wxALL, 5);

    max_defocus_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Max. Defocus"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(max_defocus_radio, 0, wxALL, 5);

    defocus_step_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Defocus Step"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(defocus_step_radio, 0, wxALL, 5);

    tolerated_astigmatism_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Tolerated Astigmatism"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(tolerated_astigmatism_radio, 0, wxALL, 5);

    min_phase_shift_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Min. Phase Shift"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(min_phase_shift_radio, 0, wxALL, 5);

    max_phase_shift_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Max. Phase Shift"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(max_phase_shift_radio, 0, wxALL, 5);

    phase_shift_step_radio = new wxRadioButton(SortScrollPanel, wxID_ANY, wxT("Phase Shift Step"), wxDefaultPosition, wxDefaultSize, 0);
    SortSizer->Add(phase_shift_step_radio, 0, wxALL, 5);

    // resize..

    SizeAndPosition( );
}

void MyCTFFilterDialog::BuildSearchCommand( ) {

    int number_checked       = ReturnNumberChecked( );
    int number_accounted_for = 0;

    search_command = "SELECT DISTINCT ESTIMATED_CTF_PARAMETERS.IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS,IMAGE_ASSETS WHERE IMAGE_ASSETS.CTF_ESTIMATION_ID=ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_ID";

    if ( number_checked > 0 ) {
        search_command += " AND";

        if ( asset_id_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.IMAGE_ASSET_ID BETWEEN %i AND %i", asset_id_filter->GetLowValue( ), asset_id_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( estimation_id_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_ID BETWEEN %i AND %i", estimation_id_filter->GetLowValue( ), estimation_id_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( date_of_run_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.DATETIME_OF_RUN BETWEEN %li AND %li", date_of_run_filter->GetLowValue( ), date_of_run_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( job_id_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_JOB_ID BETWEEN %i AND %i", job_id_filter->GetLowValue( ), job_id_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( defocus_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" (ESTIMATED_CTF_PARAMETERS.DEFOCUS1+ESTIMATED_CTF_PARAMETERS.DEFOCUS2)/2 BETWEEN %f AND %f", defocus_filter->GetLowValue( ), defocus_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( astigmatism_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ABS(ESTIMATED_CTF_PARAMETERS.DEFOCUS1-ESTIMATED_CTF_PARAMETERS.DEFOCUS2) BETWEEN %f AND %f", astigmatism_filter->GetLowValue( ), astigmatism_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( score_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.SCORE BETWEEN %f AND %f", score_filter->GetLowValue( ), score_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( ring_resolution_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.DETECTED_RING_RESOLUTION BETWEEN %f AND %f", ring_resolution_filter->GetLowValue( ), ring_resolution_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( alias_resolution_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.DETECTED_ALIAS_RESOLUTION BETWEEN %f AND %f", alias_resolution_filter->GetLowValue( ), alias_resolution_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( additional_phase_shift_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.ADDITIONAL_PHASE_SHIFT BETWEEN %f AND %f", additional_phase_shift_filter->GetLowValue( ), additional_phase_shift_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( iciness_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.ICINESS BETWEEN %f AND %f", iciness_filter->GetLowValue( ), iciness_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( voltage_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.VOLTAGE BETWEEN %f AND %f", voltage_filter->GetLowValue( ), voltage_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( spherical_aberration_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.SPHERICAL_ABERRATION BETWEEN %f AND %f", spherical_aberration_filter->GetLowValue( ), spherical_aberration_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( pixel_size_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.PIXEL_SIZE BETWEEN %f AND %f", pixel_size_filter->GetLowValue( ), pixel_size_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( amplitude_contrast_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.AMPLITUDE_CONTRAST BETWEEN %f AND %f", amplitude_contrast_filter->GetLowValue( ), amplitude_contrast_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( box_size_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.BOX_SIZE BETWEEN %i AND %i", box_size_filter->GetLowValue( ), box_size_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( min_resolution_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.MIN_RESOLUTION BETWEEN %f AND %f", min_resolution_filter->GetLowValue( ), min_resolution_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( max_resolution_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.MAX_RESOLUTION BETWEEN %f AND %f", max_resolution_filter->GetLowValue( ), max_resolution_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( min_defocus_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.MIN_DEFOCUS BETWEEN %f AND %f", min_defocus_filter->GetLowValue( ), min_defocus_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( max_defocus_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.MAX_DEFOCUS BETWEEN %f AND %f", max_defocus_filter->GetLowValue( ), max_defocus_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( defocus_step_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.DEFOCUS_STEP BETWEEN %f AND %f", defocus_step_filter->GetLowValue( ), defocus_step_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( tolerated_astigmatism_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.TOLERATED_ASTIGMATISM BETWEEN %f AND %f", tolerated_astigmatism_filter->GetLowValue( ), tolerated_astigmatism_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( min_phase_shift_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.MIN_PHASE_SHIFT BETWEEN %f AND %f", min_phase_shift_filter->GetLowValue( ), min_phase_shift_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( max_phase_shift_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.MAX_PHASE_SHIFT BETWEEN %f AND %f", max_phase_shift_filter->GetLowValue( ), max_phase_shift_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( phase_shift_step_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.PHASE_SHIFT_STEP BETWEEN %f AND %f", phase_shift_step_filter->GetLowValue( ), phase_shift_step_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( ctf_tilt_angle_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.TILT_ANGLE BETWEEN %f AND %f", ctf_tilt_angle_filter->GetLowValue( ), ctf_tilt_angle_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }

        if ( ctf_tilt_axis_filter->field_checkbox->IsChecked( ) == true ) {
            search_command += wxString::Format(" ESTIMATED_CTF_PARAMETERS.TILT_AXIS BETWEEN %f AND %f", ctf_tilt_axis_filter->GetLowValue( ), ctf_tilt_axis_filter->GetHighValue( ));
            number_accounted_for++;

            if ( number_accounted_for < number_checked )
                search_command += " AND";
        }
    }

    // do the ordering

    if ( asset_id_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.IMAGE_ASSET_ID";
    else if ( estimation_id_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_ID";
    else if ( date_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.DATETIME_OF_RUN";
    else if ( job_id_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_JOB_ID";
    else if ( defocus_radio->GetValue( ) == true )
        search_command += " ORDER BY (ESTIMATED_CTF_PARAMETERS.DEFOCUS1+ESTIMATED_CTF_PARAMETERS.DEFOCUS2)/2";
    else if ( astigmatism_radio->GetValue( ) == true )
        search_command += " ORDER BY ABS(ESTIMATED_CTF_PARAMETERS.DEFOCUS1-ESTIMATED_CTF_PARAMETERS.DEFOCUS2)";
    else if ( astigmatism_angle_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.DEFOCUS_ANGLE";
    else if ( score_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.SCORE";
    else if ( ring_resolution_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.DETECTED_RING_RESOLUTION";
    else if ( alias_resolution_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.DETECTED_ALIAS_RESOLUTION";
    else if ( additional_phase_shift_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.ADDITIONAL_PHASE_SHIFT";
    else if ( iciness_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.ICINESS";
    else if ( voltage_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.VOLTAGE";
    else if ( spherical_aberration_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.SPHERICAL_ABERRATION";
    else if ( pixel_size_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.PIXEL_SIZE";
    else if ( amplitude_contrast_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.AMPLITUDE_CONTRAST";
    else if ( box_size_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.BOX_SIZE";
    else if ( min_resolution_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.MIN_RESOLUTION";
    else if ( max_resolution_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.MAX_RESOLUTION";
    else if ( min_defocus_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.MIN_DEFOCUS";
    else if ( max_defocus_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.MAX_DEFOCUS";
    else if ( defocus_step_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.DEFOCUS_STEP";
    else if ( tolerated_astigmatism_radio->GetValue( ) == true )
        search_command += " ESTIMATED_CTF_PARAMETERS.ORDER BY TOLERATED_ASTIGMATISM";
    else if ( min_phase_shift_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.MIN_PHASE_SHIFT";
    else if ( max_phase_shift_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.MAX_PHASE_SHIFT";
    else if ( phase_shift_step_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.PHASE_SHIFT_STEP";
    else if ( ctf_tilt_angle_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.TILT_ANGLE";
    else if ( ctf_tilt_axis_radio->GetValue( ) == true )
        search_command += " ORDER BY ESTIMATED_CTF_PARAMETERS.TILT_AXIS";
}

int MyCTFFilterDialog::ReturnNumberChecked( ) {
    int number_checked = 0;

    if ( asset_id_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( estimation_id_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( date_of_run_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( job_id_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( defocus_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( astigmatism_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( astigmatism_angle_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( score_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( ring_resolution_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( alias_resolution_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( additional_phase_shift_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( iciness_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( voltage_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( spherical_aberration_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( pixel_size_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( amplitude_contrast_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( box_size_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( min_resolution_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( max_resolution_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( min_defocus_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( max_defocus_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( defocus_step_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( tolerated_astigmatism_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( min_phase_shift_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( max_phase_shift_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( phase_shift_step_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( ctf_tilt_angle_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;
    if ( ctf_tilt_axis_filter->field_checkbox->IsChecked( ) == true )
        number_checked++;

    return number_checked;
}
