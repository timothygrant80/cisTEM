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

public:

		MyFindParticlesPanel( wxWindow* parent );
		JobResult *buffered_results;

		bool group_combo_is_dirty;
		bool run_profiles_are_dirty;
		long time_of_last_result_update;

		// methods

		void WriteResultToDataBase();
		void OnExpertOptionsToggle( wxCommandEvent& event );
		void OnUpdateUI( wxUpdateUIEvent& event );
		void FillGroupComboBox();
		void FillPickingAlgorithmComboBox();
		void OnPickingAlgorithmComboBox( wxCommandEvent& event );
		void FillRunProfileComboBox();
		void StartPickingClick( wxCommandEvent& event );
		void FinishButtonClick( wxCommandEvent& event );
		void TerminateButtonClick( wxCommandEvent& event );
		void OnAutoPickRefreshCheckBox( wxCommandEvent& event );
		void CheckWhetherGroupsCanBePicked();
	//void Refresh();
		void SetInfo();
		void OnInfoURL(wxTextUrlEvent& event);

		ArrayOfParticlePositionAssets ParticlePositionsFromJobResults(JobResult *job_result, const int &parent_image_id, const int &picking_job_id, const int &picking_id, const int &starting_asset_id);

		void WriteInfoText(wxString text_to_write);
		void WriteErrorText(wxString text_to_write);

		void ProcessResult(JobResult *result_to_process);
		//void UpdateProgressBar();

		virtual void OnJobSocketEvent(wxSocketEvent& event);

		//
		enum particle_picking_algorithms { ab_initio, number_of_picking_algorithms };
		wxString ReturnNameOfPickingAlgorithm( const int wanted_algorithm );



};

#endif // __FindParticlesPanel__
