#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyMovieFilterDialog::MyMovieFilterDialog( wxWindow* parent )
:
FilterDialog( parent )
{
	// add the filter checkboxes..

	asset_id_filter = new IntegerFilterItem("Asset ID", this);
	FilterBoxSizer->Add(asset_id_filter,  1, wxEXPAND | wxALL, 5 );

	alignment_id_filter = new IntegerFilterItem("Alignment ID", this);
	FilterBoxSizer->Add(alignment_id_filter,  1, wxEXPAND | wxALL, 5 );

	date_of_run_filter = new DateFilterItem("Date of Run", this);
	FilterBoxSizer->Add(date_of_run_filter,  1, wxEXPAND | wxALL, 5);

	job_id_filter = new IntegerFilterItem("Job ID", this);
	FilterBoxSizer->Add(job_id_filter,  0, wxEXPAND | wxALL, 5);

	voltage_filter = new FloatFilterItem("Voltage", this);
	FilterBoxSizer->Add(voltage_filter,  0, wxEXPAND | wxALL, 5);

	pixel_size_filter = new FloatFilterItem("Pixel Size", this);
	FilterBoxSizer->Add(pixel_size_filter,  0, wxEXPAND | wxALL, 5);

	exposure_per_frame_filter = new FloatFilterItem("Exposure Per-Frame", this);
	FilterBoxSizer->Add(exposure_per_frame_filter,  01, wxEXPAND | wxALL, 5);

	pre_exposure_filter = new FloatFilterItem("Pre-Exposure Amount", this);
	FilterBoxSizer->Add(pre_exposure_filter,  0, wxEXPAND | wxALL, 5);

	// Add the sort combo boxes..

	AssetIDRadioButton = new wxRadioButton( this, wxID_ANY, wxT("AssetID"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( AssetIDRadioButton, 0, wxALL, 5 );

	AlignmentIDRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Alignment ID"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( AlignmentIDRadioButton, 0, wxALL, 5 );

	DateOfRunRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Date of Run"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( DateOfRunRadioButton, 0, wxALL, 5 );

	JobIDRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Job ID"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( JobIDRadioButton, 0, wxALL, 5 );

	VoltageRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Voltage"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( VoltageRadioButton, 0, wxALL, 5 );

	PixelSizeRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Pixel Size"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( PixelSizeRadioButton, 0, wxALL, 5 );

	ExposureRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Exposure"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( ExposureRadioButton, 0, wxALL, 5 );

	PreExposureRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Pre-Exposure"), wxDefaultPosition, wxDefaultSize, 0 );
	SortSizer->Add( PreExposureRadioButton, 0, wxALL, 5 );


	// resize..

	MainBoxSizer->Fit(this);
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

