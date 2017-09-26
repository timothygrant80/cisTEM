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

	RefinementPackage *active_refinement_package;
	Refinement *input_refinement;
	Refinement *output_refinement;

	RunProfile active_refinement_run_profile;
	RunProfile active_reconstruction_run_profile;

	//float startup_percent_used;
	float current_high_res_limit;
	float next_high_res_limit;

	float start_percent_used;
	float end_percent_used;

	float active_start_percent_used;
	float active_end_percent_used;
	float current_percent_used;

	float active_start_res;
	float active_end_res;

	bool active_should_automask;
	bool active_auto_set_percent_used;

	float active_inner_mask_radius;
	float active_global_mask_radius;

	float active_search_range_x;
	float active_search_range_y;

	bool active_should_apply_blurring;
	bool active_smoothing_factor;

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
	void OnOrthThreadComplete(ReturnProcessedImageEvent& my_event);

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


#endif
