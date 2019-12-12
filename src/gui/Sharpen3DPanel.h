#ifndef __Sharpen3DPanel__
#define __Sharpen3DPanel__


class Sharpen3DPanel : public Sharpen3DPanelParent
{
	void OnInfoURL( wxTextUrlEvent& event );
	void OnUpdateUI( wxUpdateUIEvent& event );
	void OnUseMaskCheckBox( wxCommandEvent& event );
	void OnVolumeComboBox( wxCommandEvent& event );
	void OnRunButtonClick( wxCommandEvent& event );
	void OnSaveResultClick( wxCommandEvent& event );
	void OnImportResultClick( wxCommandEvent& event );
	void SetInfo();
	void OnSharpenThreadComplete(ReturnSharpeningResultsEvent& my_event);
	void OnGuageTimer(wxTimerEvent& event);

	wxTimer *guage_timer;

	public:

	bool volumes_are_dirty;
	bool have_a_result_in_memory;
	bool running_a_job;

	Image *active_result;
	int active_thread_id;
	int next_thread_id;
	bool auto_mask_value;

	Refinement *active_refinement;
	int active_class;

	Sharpen3DPanel( wxWindow* parent );

	void FillVolumePanels();
	void Reset();
	void ResetDefaults();
};

class SharpenMapThread : public wxThread
{
	public:
	SharpenMapThread(wxWindow *parent, wxString wanted_map_filename, float wanted_pixel_size, float wanted_resolution_limit, bool wanted_invert_hand, float wanted_inner_mask_radius, float wanted_outer_mask_radius, float wanted_start_res_for_whitening, float wanted_additional_low_bfactor, float wanted_additional_high_bfactor, float wanted_filter_edge, wxString wanted_input_mask_filename, ResolutionStatistics *wanted_input_resolution_statistics, float wanted_statistics_scale_factor, bool wanted_correct_sinc, bool wanted_should_auto_mask, int wanted_thread_id) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		map_filename = wanted_map_filename;
		pixel_size = wanted_pixel_size;
		resolution_limit = wanted_resolution_limit;
		invert_hand = wanted_invert_hand;
		inner_mask_radius = wanted_inner_mask_radius;
		outer_mask_radius = wanted_outer_mask_radius;
		start_res_for_whitening = wanted_start_res_for_whitening;
		additional_low_bfactor = wanted_additional_low_bfactor;
		additional_high_bfactor = wanted_additional_high_bfactor;
		filter_edge = wanted_filter_edge;
		input_mask_filename = wanted_input_mask_filename;
		input_resolution_statistics = wanted_input_resolution_statistics;
		statistics_scale_factor = wanted_statistics_scale_factor;
		correct_sinc = wanted_correct_sinc;
		thread_id = wanted_thread_id;
		should_auto_mask = wanted_should_auto_mask;
	}

	protected:

	wxWindow *main_thread_pointer;

	wxString map_filename;
	float pixel_size;
	float resolution_limit;
	bool invert_hand;
	float inner_mask_radius;
	float outer_mask_radius;
	float start_res_for_whitening;
	float additional_low_bfactor;
	float additional_high_bfactor;
	float filter_edge;
	wxString input_mask_filename;
	ResolutionStatistics *input_resolution_statistics;
	bool should_auto_mask;
	float statistics_scale_factor;
	bool correct_sinc;
	int thread_id;

    virtual ExitCode Entry();
};


#endif
