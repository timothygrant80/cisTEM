#ifndef __AlignMoviesPanel__
#define __AlignMoviesPanel__

//// end generated include

/** Implementing AlignMoviesPanel */
class MyAlignMoviesPanel : public AlignMoviesPanel
{

		bool show_expert_options;
		long my_job_id;
		int length_of_process_number;
		JobPackage my_job_package;
		JobTracker my_job_tracker;
		bool running_job;
		AssetGroup active_group;



public:
		/** Constructor */
		MyAlignMoviesPanel( wxWindow* parent );
	//// end generated class members

		bool graph_is_hidden;
		long time_of_last_graph_update;

		//mpInfoCoords    *nfo;

		JobResult *buffered_results;

		bool group_combo_is_dirty;
		bool run_profiles_are_dirty;

		// methods

		void WriteResultToDataBase();
		void OnExpertOptionsToggle( wxCommandEvent& event );
		void OnUpdateUI( wxUpdateUIEvent& event );
		void FillGroupComboBox();
		void FillRunProfileComboBox();
		void StartAlignmentClick( wxCommandEvent& event );
		void FinishButtonClick( wxCommandEvent& event );
		void TerminateButtonClick( wxCommandEvent& event );
		void Refresh();
		void SetInfo();
		void OnInfoURL(wxTextUrlEvent& event);

		void WriteInfoText(wxString text_to_write);
		void WriteErrorText(wxString text_to_write);

		void ProcessResult(JobResult *result_to_process);
		void ProcessAllJobsFinished();
		void UpdateProgressBar();

		virtual void OnJobSocketEvent(wxSocketEvent& event);


};

#endif // __AlignMoviesPanel__
