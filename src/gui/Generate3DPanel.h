#ifndef __Generate3DPanel__
#define __Generage3DPanel__


class Generate3DPanel : public Generate3DPanelParent
{

	protected:
		// Handlers for Refine3DPanel events.
		void OnUpdateUI( wxUpdateUIEvent& event );
		void OnExpertOptionsToggle( wxCommandEvent& event );
		void OnInfoURL( wxTextUrlEvent& event );
		void TerminateButtonClick( wxCommandEvent& event );
		void FinishButtonClick( wxCommandEvent& event );
		void StartReconstructionClick( wxCommandEvent& event );
		void ResetAllDefaultsClick( wxCommandEvent& event );

		// overridden socket methods..

		void OnSocketJobResultMsg(JobResult &received_result);
		void OnSocketJobResultQueueMsg(ArrayofJobResults &received_queue);
		void SetNumberConnectedText(wxString wanted_text);
		void SetTimeRemainingText(wxString wanted_text);
		void OnSocketAllJobsFinished();

		int length_of_process_number;
		//RefinementManager my_refinement_manager;

		int active_orth_thread_id;
		int next_thread_id;

	public:

		long time_of_last_result_update;
		long number_of_received_particle_results;
		long number_of_expected_results;
		bool refinement_package_combo_is_dirty;
		bool run_profiles_are_dirty;
		bool input_params_combo_is_dirty;

		JobResult *buffered_results;

		long current_job_id;
		long selected_refinement_package;

		long time_of_last_update;
		long current_job_starttime;

		wxArrayString output_filenames;


		//int length_of_process_number;

		JobTracker my_job_tracker;

		bool running_job;

		// active

		int running_job_type;

		RefinementPackage *active_refinement_package;
		RunProfile active_reconstruction_run_profile;

		float active_mask_radius;
		float active_inner_mask_radius;
		float active_resolution_limit_rec;
		float active_score_weight_conversion;
		float active_score_threshold;
		bool active_adjust_scores;
		bool active_crop_images;
		bool active_save_half_maps;
		bool active_update_statistics;
		bool active_apply_ewald_correction;
		bool active_apply_inverse_hand;

		Refinement *input_refinement;


		//

		void Reset();
		void SetDefaults();
		Generate3DPanel( wxWindow* parent );
		void SetInfo();

		void WriteInfoText(wxString text_to_write);
		void WriteErrorText(wxString text_to_write);
		void WriteBlueText(wxString text_to_write);

		void SetupReconstructionJob();
		void SetupMerge3dJob();
		void RunReconstructionJob();
		void RunMerge3dJob();

		void FillRefinementPackagesComboBox();
		void FillRunProfileComboBoxes();
		void FillInputParamsComboBox();
		void NewRefinementPackageSelected();

		void OnRefinementPackageComboBox( wxCommandEvent& event );
		void OnInputParametersComboBox( wxCommandEvent& event );

		void OnOrthThreadComplete(ReturnProcessedImageEvent& my_event);

		void ProcessJobResult(JobResult *result_to_process);
		void ProcessAllJobsFinished();

};

#endif // __MyRefine3DPanel__
