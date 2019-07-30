#ifndef __MatchTemplatePanel__
#define __MatchTemplatePanel__


class MatchTemplatePanel : public MatchTemplateParentPanel
{
		long my_job_id;


		JobTracker my_job_tracker;

		bool running_job;

		Image result_image;
		wxBitmap result_bitmap;

		AssetGroup active_group;

public:

		MatchTemplatePanel( wxWindow* parent );
		JobResult *buffered_results;

		bool group_combo_is_dirty;
		bool run_profiles_are_dirty;
		bool volumes_are_dirty;
		long time_of_last_result_update;

		long expected_number_of_results;
		long number_of_received_results;
		long current_job_starttime;
		long time_of_last_update;

		// methods
		void WriteResultToDataBase();
		void OnExpertOptionsToggle( wxCommandEvent& event );
		void OnUpdateUI( wxUpdateUIEvent& event );
		void FillGroupComboBox();
		void FillRunProfileComboBox();
		void StartEstimationClick( wxCommandEvent& event );
		void FinishButtonClick( wxCommandEvent& event );
		void TerminateButtonClick( wxCommandEvent& event );

		void OnSocketJobResultMsg(JobResult &received_result);
		void OnSocketJobResultQueueMsg(ArrayofJobResults &received_queue);
		void SetNumberConnectedText(wxString wanted_text);
		void SetTimeRemainingText(wxString wanted_text);
		void OnSocketAllJobsFinished();

	//void Refresh();
		void SetInfo();
		void OnInfoURL(wxTextUrlEvent& event);
		void OnGroupComboBox(wxCommandEvent &event);

		void WriteInfoText(wxString text_to_write);
		void WriteErrorText(wxString text_to_write);

		void ProcessResult(JobResult *result_to_process);
		void ProcessAllJobsFinished();
		void UpdateProgressBar();

		virtual void OnJobSocketEvent(wxSocketEvent& event);
		void Reset();
		void ResetDefaults();


};

#endif
