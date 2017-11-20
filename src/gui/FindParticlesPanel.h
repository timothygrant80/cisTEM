#ifndef __FindParticlesPanel__
#define __FindParticlesPanel__


class MyFindParticlesPanel : public FindParticlesPanel
{
		long my_job_id;

		int length_of_process_number;

		JobPackage my_job_package;
		JobTracker my_job_tracker;

		bool running_job;

		//Image result_image;
		//wxBitmap result_bitmap;

		AssetGroup active_group;

public:

		MyFindParticlesPanel( wxWindow* parent );
		JobResult *buffered_results;

		bool group_combo_is_dirty;
		bool run_profiles_are_dirty;
		long time_of_last_result_update;


		// methods
		void Reset();
		void ResetDefaults();

		void OnGroupComboBox( wxCommandEvent& event );
		void OnImageComboBox( wxCommandEvent& event );
		void WriteResultToDataBase();
		void ProcessAllJobsFinished();
		void OnExpertOptionsToggle( wxCommandEvent& event );
		void OnUpdateUI( wxUpdateUIEvent& event );
		void FillGroupComboBox();
		void FillImageComboBox();
		void FillPickingAlgorithmComboBox();
		void OnPickingAlgorithmComboBox( wxCommandEvent& event );
		void FillRunProfileComboBox();
		void StartPickingClick( wxCommandEvent& event );
		void FinishButtonClick( wxCommandEvent& event );
		void TerminateButtonClick( wxCommandEvent& event );
		void OnAutoPickRefreshCheckBox( wxCommandEvent& event );
		void OnSetMinimumDistanceFromEdgesCheckBox( wxCommandEvent & event );
		void OnTestOnCurrentMicrographButtonClick( wxCommandEvent & event );
		void OnMaximumParticleRadiusNumericTextEnter( wxCommandEvent& event );
		void OnCharacteristicParticleRadiusNumericTextEnter( wxCommandEvent& event );
		void OnThresholdPeakHeightNumericTextEnter( wxCommandEvent& event );
		void CheckWhetherGroupsCanBePicked();
		void SetAllUserParametersForParticleFinder();
		void DrawResultsFromParticleFinder();
		int ReturnDefaultMinimumDistanceFromEdges();
		void ShowPickingParametersPanel();

		void OnNewCharacteristicParticleRadius();
		void OnNewThresholdPeakHeight();
		void OnNewMaximumParticleRadius();
		void OnNewHighestResolution();

		void OnCharacteristicParticleRadiusNumericTextKillFocus( wxFocusEvent & event );
		void OnCharacteristicParticleRadiusNumericTextSetFocus( wxFocusEvent & event );
		void OnThresholdPeakHeightNumericTextKillFocus( wxFocusEvent & event );
		void OnThresholdPeakHeightNumericTextSetFocus( wxFocusEvent & event );
		void OnMaximumParticleRadiusNumericTextKillFocus( wxFocusEvent& event);
		void OnMaximumParticleRadiusNumericTextSetFocus( wxFocusEvent& event);

		void OnHighestResolutionNumericTextEnter( wxCommandEvent & event );
		void OnHighestResolutionNumericKillFocus( wxFocusEvent & event );
		void OnHighestResolutionNumericSetFocus( wxFocusEvent & event );

		void OnMinimumDistanceFromEdgesSpinCtrl( wxSpinEvent& event );
		void OnAvoidHighVarianceAreasCheckBox( wxCommandEvent& event );
		void OnAvoidAbnormalLocalMeanAreasCheckBox( wxCommandEvent& event );
		void OnNumberOfBackgroundBoxesSpinCtrl( wxSpinEvent& event );
		void OnAlgorithmToFindBackgroundChoice( wxCommandEvent& event );

	//void Refresh();
		void SetInfo();
		void OnInfoURL(wxTextUrlEvent& event);

		ArrayOfParticlePositionAssets ParticlePositionsFromJobResults(JobResult *job_result, const int &parent_image_id, const int &picking_job_id, const int &picking_id, const int &starting_asset_id);

		void WriteInfoText(wxString text_to_write);
		void WriteErrorText(wxString text_to_write);

		void ProcessResult(JobResult *result_to_process, const int &wanted_job_number = -1);
		void UpdateProgressBar();

		virtual void OnJobSocketEvent(wxSocketEvent& event);

		//
		enum particle_picking_algorithms { ab_initio, number_of_picking_algorithms };
		wxString ReturnNameOfPickingAlgorithm( const int wanted_algorithm );

		int ReturnNumberOfJobsCurrentlyRunning();

		int number_of_particles_picked;

private:
		ParticleFinder particle_finder;

		float 	value_on_focus_float;


};

#endif // __FindParticlesPanel__
