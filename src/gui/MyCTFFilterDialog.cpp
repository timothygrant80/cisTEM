//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"


MyFilterDialog::MyFilterDialog( wxWindow* parent )
:
FilterDialog( parent )
{
	SetTitle("Filter / Sort Images");
}


void MyFilterDialog::SizeAndPosition()
{
	wxSize filter_input_size = FilterBoxSizer->GetMinSize();
	wxSize sort_input_size = SortSizer->GetMinSize();

	int frame_width;
	int frame_height;
	int frame_position_x;
	int frame_position_y;

	main_frame->GetSize(&frame_width, &frame_height);
	main_frame->GetPosition(&frame_position_x, &frame_position_y);

	int total_height = filter_input_size.y + sort_input_size.y;

	filter_input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	sort_input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);

	if (total_height > frame_height)
	{
		filter_input_size.y -= (total_height - frame_height);
	}

	FilterScrollPanel->SetMinSize(filter_input_size);
	FilterScrollPanel->SetSize(filter_input_size);

	SortScrollPanel->SetMinSize(sort_input_size);
	SortScrollPanel->SetSize(sort_input_size);

	SetMaxSize(wxSize(frame_width, frame_height));
	Layout();
	MainBoxSizer->Fit(this);

	int dialog_height;
	int dialog_width;

	// ok so how big is this dialog now?

	GetSize(&dialog_width, &dialog_height);

	int new_x_pos = (frame_position_x + (frame_width / 2) - (dialog_width / 2));
	int new_y_pos = (frame_position_y + (frame_height / 2) - (dialog_height / 2));

	Move(new_x_pos, new_y_pos);
}

MyMovieFilterDialog::MyMovieFilterDialog( wxWindow* parent )
:
MyFilterDialog( parent )
{
	// add the filter checkboxes..

	asset_id_filter = new IntegerFilterItem("Asset ID", FilterScrollPanel);
	FilterBoxSizer->Add(asset_id_filter,  1, wxEXPAND | wxALL, 5 );

	estimation_id_filter = new IntegerFilterItem("Estimation ID", FilterScrollPanel);
	FilterBoxSizer->Add(estimation_id_filter,  1, wxEXPAND | wxALL, 5 );

	date_of_run_filter = new IntegerFilterItem("Date of Run", FilterScrollPanel);
	FilterBoxSizer->Add(date_of_run_filter,  1, wxEXPAND | wxALL, 5 );

	job_id_filter = new IntegerFilterItem("Job ID", FilterScrollPanel);
	FilterBoxSizer->Add(job_id_filter,  1, wxEXPAND | wxALL, 5 );

	FloatFilterItem *defocus;
	FloatFilterItem *astigmatism;
	FloatFilterItem *astigmatism_angle;
	FloatFilterItem *score;
	FloatFilterItem *ring_resolution;
	FloatFilterItem *alias_resolution;
	FloatFilterItem *additional_phase_shift;
	FloatFilterItem *voltage_filter;
	FloatFilterItem *spherical_aberation_filter;
	FloatFilterItem *pixel_size_filter;
	FloatFilterItem *amplitude_contrast_filter;
	IntegerFilterItem *box_size_filter;
	FloatFilterItem *min_resolution_filter;
	FloatFilterItem *max_resolution_filter;
	FloatFilterItem *min_defocus_filter;
	FloatFilterItem *max_defocus_filter;
	FloatFilterItem *defocus_step_filter;
	FloatFilterItem *tolerated_astigmatism_filter;
	FloatFilterItem *min_phase_shift_filter;
	FloatFilterItem *max_phase_shift_filter;
	FloatFilterItem *phase_shift_step_filter;

	wxRadioButton *asset_id_radio;
	wxRadioButton *estimation_id_radio;
	wxRadioButton *date_radio;
	wxRadioButton *job_id_radio;
	wxRadioButton *defocus_radio;
	wxRadioButton *astigmatism_radio;
	wxRadioButton *astigmatism_angle_radio;
	wxRadioButton *score_radio;
	wxRadioButton *ring_resolution_radio;
	wxRadioButton *alias_resolution_radio;
	wxRadioButton *additional_phase_shift_radio;
	wxRadioButton *voltage_radio;
	wxRadioButton *spherical_abeeration_radio;
	wxRadioButton *pixel_size_radio;
	wxRadioButton *amplitude_contrast_radio;
	wxRadioButton *box_size_radio;
	wxRadioButton *min_resolution_radio;
	wxRadioButton *max_resolution_radio;
	wxRadioButton *min_defocus_radio;
	wxRadioButton *max_defocus_radio;
	wxRadioButton *defocus_step_radio;
	wxRadioButton *tolerated_astigmatism_radio;
	wxRadioButton *min_phase_shift_radio;
	wxRadioButton *max_phase_shift_radio;
	wxRadioButton *phase_shift_step_radio;
	///

	asset_id_filter = new IntegerFilterItem("Asset ID", FilterScrollPanel);
	FilterBoxSizer->Add(asset_id_filter,  1, wxEXPAND | wxALL, 5 );

	alignment_id_filter = new IntegerFilterItem("Alignment ID", FilterScrollPanel);
	FilterBoxSizer->Add(alignment_id_filter,  1, wxEXPAND | wxALL, 5 );

	date_of_run_filter = new DateFilterItem("Date of Run", FilterScrollPanel);
	FilterBoxSizer->Add(date_of_run_filter,  1, wxEXPAND | wxALL, 5);

	job_id_filter = new IntegerFilterItem("Job ID", FilterScrollPanel);
	FilterBoxSizer->Add(job_id_filter,  0, wxEXPAND | wxALL, 5);

	voltage_filter = new FloatFilterItem("Voltage", FilterScrollPanel);
	FilterBoxSizer->Add(voltage_filter,  0, wxEXPAND | wxALL, 5);

	pixel_size_filter = new FloatFilterItem("Pixel Size", FilterScrollPanel);
	FilterBoxSizer->Add(pixel_size_filter,  0, wxEXPAND | wxALL, 5);

	exposure_per_frame_filter = new FloatFilterItem("Exposure Per-Frame", FilterScrollPanel);
	FilterBoxSizer->Add(exposure_per_frame_filter,  01, wxEXPAND | wxALL, 5);

	pre_exposure_filter = new FloatFilterItem("Pre-Exposure Amount", FilterScrollPanel);
	FilterBoxSizer->Add(pre_exposure_filter,  0, wxEXPAND | wxALL, 5);


	// Add the sort combo boxes..


	AssetIDRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("AssetID"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( AssetIDRadioButton, 0, wxALL, 5 );

	AlignmentIDRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("Alignment ID"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( AlignmentIDRadioButton, 0, wxALL, 5 );

	DateOfRunRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("Date of Run"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( DateOfRunRadioButton, 0, wxALL, 5 );

	JobIDRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("Job ID"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( JobIDRadioButton, 0, wxALL, 5 );

	VoltageRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("Voltage"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( VoltageRadioButton, 0, wxALL, 5 );

	PixelSizeRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("Pixel Size"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( PixelSizeRadioButton, 0, wxALL, 5 );

	ExposureRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("Exposure"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( ExposureRadioButton, 0, wxALL, 5 );

	PreExposureRadioButton = new wxRadioButton( SortScrollPanel, wxID_ANY, wxT("Pre-Exposure"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( PreExposureRadioButton, 0, wxALL, 5 );

	// for testing..
/*
	FloatFilterItem *my_item = new FloatFilterItem("Junk", FilterScrollPanel);
	FilterBoxSizer->Add(my_item,  0, wxEXPAND | wxALL, 5);

	FloatFilterItem *my_item1 = new FloatFilterItem("Junk", FilterScrollPanel);
		FilterBoxSizer->Add(my_item1,  0, wxEXPAND | wxALL, 5);


		FloatFilterItem *my_item2 = new FloatFilterItem("Junk", FilterScrollPanel);
			FilterBoxSizer->Add(my_item2,  0, wxEXPAND | wxALL, 5);


			FloatFilterItem *my_item3 = new FloatFilterItem("Junk", FilterScrollPanel);
				FilterBoxSizer->Add(my_item3,  0, wxEXPAND | wxALL, 5);


				FloatFilterItem *my_item4 = new FloatFilterItem("Junk", FilterScrollPanel);
					FilterBoxSizer->Add(my_item4,  0, wxEXPAND | wxALL, 5);


					FloatFilterItem *my_item5 = new FloatFilterItem("Junk", FilterScrollPanel);
						FilterBoxSizer->Add(my_item5,  0, wxEXPAND | wxALL, 5);


						FloatFilterItem *my_item6 = new FloatFilterItem("Junk", FilterScrollPanel);
							FilterBoxSizer->Add(my_item6,  0, wxEXPAND | wxALL, 5);


							FloatFilterItem *my_item7 = new FloatFilterItem("Junk", FilterScrollPanel);
								FilterBoxSizer->Add(my_item7,  0, wxEXPAND | wxALL, 5);


								FloatFilterItem *my_item8 = new FloatFilterItem("Junk", FilterScrollPanel);
									FilterBoxSizer->Add(my_item8,  0, wxEXPAND | wxALL, 5);


									FloatFilterItem *my_item9 = new FloatFilterItem("Junk", FilterScrollPanel);
										FilterBoxSizer->Add(my_item9,  0, wxEXPAND | wxALL, 5);*/


	// resize..

	SizeAndPosition();


}

void MyMovieFilterDialog::BuildSearchCommand()
{

	int number_checked = ReturnNumberChecked();
	int number_accounted_for = 0;

	search_command = "SELECT DISTINCT MOVIE_ASSET_ID FROM MOVIE_ALIGNMENT_LIST";

	if (number_checked > 0)
	{
		search_command += " WHERE";

		if (asset_id_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" MOVIE_ASSET_ID BETWEEN %i AND %i", asset_id_filter->GetLowValue(), asset_id_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}

		if (alignment_id_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" ALIGNMENT_ID BETWEEN %i AND %i", alignment_id_filter->GetLowValue(), alignment_id_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}

		if (date_of_run_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" DATETIME_OF_RUN BETWEEN %i AND %i", date_of_run_filter->GetLowValue(), date_of_run_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}

		if (job_id_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" ALIGNMENT_JOB_ID BETWEEN %i AND %i", job_id_filter->GetLowValue(), job_id_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}

		if (voltage_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" VOLTAGE BETWEEN %i AND %i", voltage_filter->GetLowValue(), voltage_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}

		if (pixel_size_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" PIXEL_SIZE BETWEEN %i AND %i", pixel_size_filter->GetLowValue(), pixel_size_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}

		if (exposure_per_frame_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" EXPOSURE_PER_FRAME BETWEEN %i AND %i", exposure_per_frame_filter->GetLowValue(), exposure_per_frame_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}

		if (pre_exposure_filter->field_checkbox->IsChecked() == true)
		{
			search_command += wxString::Format(" PRE_EXPOSURE_AMOUNT BETWEEN %i AND %i", exposure_per_frame_filter->GetLowValue(), exposure_per_frame_filter->GetHighValue());
			number_accounted_for++;

			if (number_accounted_for < number_checked) search_command += " AND";
		}


	}

	// do the ordering

	if (AssetIDRadioButton->GetValue() == true) search_command += " ORDER BY MOVIE_ASSET_ID";
	else
	if (AlignmentIDRadioButton->GetValue() == true) search_command += " ORDER BY ALIGNMENT_ID";
	else
	if (DateOfRunRadioButton->GetValue() == true) search_command += " ORDER BY DATETIME_OF_RUN";
	else
	if (JobIDRadioButton->GetValue() == true) search_command += " ORDER BY ALIGNMENT_JOB_ID";
	else
	if (VoltageRadioButton->GetValue() == true) search_command += " ORDER BY VOLTAGE";
	else
	if (PixelSizeRadioButton->GetValue() == true) search_command += " ORDER BY PIXEL_SIZE";
	else
	if (ExposureRadioButton->GetValue() == true) search_command += " ORDER BY EXPOSURE_PER_FRAME";
	else
	if (PreExposureRadioButton->GetValue() == true) search_command += " ORDER BY PRE_EXPOSURE_AMOUNT";



			//SELECT MOVIE_ASSET_ID, ALIGNMENT_JOB_ID FROM MOVIE_ALIGNMENT_LIST, IMAGE_ASSETS WHERE MOVIE_ALIGNMENT_LIST.ALIGNMENT_ID=IMAGE_ASSETS.ALIGNMENT_ID AND IMAGE_ASSETS.PARENT_MOVIE_ID=MOVIE_ALIGNMENT_LIST.MOVIE_ASSET_ID;");



}

int MyMovieFilterDialog::ReturnNumberChecked()
{
	int number_checked = 0;

	if (asset_id_filter->field_checkbox->IsChecked() == true) number_checked++;
	if (alignment_id_filter->field_checkbox->IsChecked() == true) number_checked++;
	if (date_of_run_filter->field_checkbox->IsChecked() == true) number_checked++;
	if (job_id_filter->field_checkbox->IsChecked() == true) number_checked++;
	if (voltage_filter->field_checkbox->IsChecked() == true) number_checked++;
	if (pixel_size_filter->field_checkbox->IsChecked() == true) number_checked++;
	if (exposure_per_frame_filter->field_checkbox->IsChecked() == true) number_checked++;
	if (pre_exposure_filter->field_checkbox->IsChecked() == true) number_checked++;

	return number_checked;
}



void MyMovieFilterDialog::OnCancelClick( wxCommandEvent& event )
{
	Destroy();
}

void MyMovieFilterDialog::OnFilterClick( wxCommandEvent& event )
{
	BuildSearchCommand();
	EndModal(wxID_OK);

}

