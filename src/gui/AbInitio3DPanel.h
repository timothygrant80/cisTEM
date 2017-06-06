#ifndef __Abinitio3DPanel__
#define __Abinitio3DPanel__

class AbInitio3DPanel;

class AbInitioManager
{
public:

	AbInitioManager();

	bool start_with_reconstruction;
	AbInitio3DPanel *my_parent;

	long current_job_starttime;
	long time_of_last_update;
	int number_of_generated_3ds;

	int running_job_type;
	int number_of_rounds_to_run;
	int number_of_rounds_run;
	long current_job_id;

	long current_refinement_package_asset_id;
	long current_input_refinement_id;
	long current_output_refinement_id;

	long number_of_received_particle_results;

	Refinement *input_refinement;
	Refinement *output_refinement;


	//float startup_percent_used;
	float current_high_res_limit;
	float next_high_res_limit;

	float start_percent_used;
	float end_percent_used;
	float current_percent_used;

	int number_of_starts_to_run;
	int number_of_starts_run;

	wxArrayString current_reference_filenames;

	void SetParent(AbInitio3DPanel *wanted_parent);

	void BeginRefinementCycle();
	void CycleRefinement();

	void UpdatePlotPanel();

	void SetupRefinementJob();
	void SetupReconstructionJob();
	void SetupMerge3dJob();

	void SetupInitialReconstructionJob();
	void SetupInitialMerge3dJob();

	void RunInitialReconstructionJob();
	void RunInitialMerge3dJob();

	void RunRefinementJob();
	void RunReconstructionJob();
	void RunMerge3dJob();

	void ProcessJobResult(JobResult *result_to_process);
	void ProcessAllJobsFinished();

	void OnMaskerThreadComplete();
	void DoMasking();



};


class AbInitio3DPanel : public AbInitio3DPanelParent
{
	friend class AbInitioManager;

protected:
	// Handlers for events.

	AbInitioManager my_abinitio_manager;

	void OnUpdateUI( wxUpdateUIEvent& event );
	void OnExpertOptionsToggle( wxCommandEvent& event );
	void OnInfoURL( wxTextUrlEvent& event );
	void TerminateButtonClick( wxCommandEvent& event );
	void FinishButtonClick( wxCommandEvent& event );
	void StartRefinementClick( wxCommandEvent& event );
	void ResetAllDefaultsClick( wxCommandEvent& event );

public:

	long time_of_last_result_update;
	bool refinement_package_combo_is_dirty;
	bool run_profiles_are_dirty;

	long my_job_id;
	long selected_refinement_package;

	JobPackage my_job_package;
	JobTracker my_job_tracker;

	bool running_job;


	AbInitio3DPanel( wxWindow* parent );

	void WriteInfoText(wxString text_to_write);
	void WriteErrorText(wxString text_to_write);
	void WriteWarningText(wxString text_to_write);
	void WriteBlueText(wxString text_to_write);

	void OnJobSocketEvent(wxSocketEvent& event);
	void OnMaskerThreadComplete(wxThreadEvent& my_event);
	void OnOrthThreadComplete(MyOrthDrawEvent& my_event);

	int length_of_process_number;

public:

	void SetDefaults();
	void SetInfo();
	void FillRefinementPackagesComboBox();
	void FillRunProfileComboBoxes();
	void NewRefinementPackageSelected();

	void OnRefinementPackageComboBox( wxCommandEvent& event );

	void TakeLastStartClicked( wxCommandEvent& event );
	void TakeCurrentClicked( wxCommandEvent& event );

	void TakeCurrent();
	void TakeLastStart();

};

class MaskerThread : public wxThread
{
	public:
	MaskerThread(AbInitio3DPanel *parent, wxArrayString wanted_input_files, wxArrayString wanted_output_files, float wanted_pixel_size, float wanted_mask_resolution, float wanted_mask_radius) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		input_files = wanted_input_files;
		output_files = wanted_output_files;
		pixel_size = wanted_pixel_size;
		mask_resolution = wanted_mask_resolution;
		mask_radius = wanted_mask_radius;
	}

	protected:

	AbInitio3DPanel *main_thread_pointer;
	wxArrayString input_files;
	wxArrayString output_files;
	float pixel_size;
	float mask_resolution;
	float mask_radius;

    virtual ExitCode Entry();
};


#endif
