#ifndef __MyMovieFilterDialog__
#define __MyMovieFilterDialog__

#include "ProjectX_gui.h"



class MyMovieFilterDialog : public FilterDialog
{
	public:

		MyMovieFilterDialog( wxWindow* parent );

		void OnCancelClick( wxCommandEvent& event );
		void OnFilterClick( wxCommandEvent& event );
		int ReturnNumberChecked();
		void BuildSearchCommand();


		wxString search_command;

		IntegerFilterItem *asset_id_filter;
		IntegerFilterItem *alignment_id_filter;
		DateFilterItem *date_of_run_filter;
		IntegerFilterItem *job_id_filter;
		FloatFilterItem *voltage_filter;
		FloatFilterItem *pixel_size_filter;
		FloatFilterItem *exposure_per_frame_filter;
		FloatFilterItem *pre_exposure_filter;

		wxRadioButton* AssetIDRadioButton;
		wxRadioButton* AlignmentIDRadioButton;
		wxRadioButton* DateOfRunRadioButton;
		wxRadioButton* JobIDRadioButton;
		wxRadioButton* VoltageRadioButton;
		wxRadioButton* PixelSizeRadioButton;
		wxRadioButton* ExposureRadioButton;
		wxRadioButton* PreExposureRadioButton;
};

#endif // __MyMovieFilterDialog__
