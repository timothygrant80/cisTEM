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


public:
		/** Constructor */
		MyAlignMoviesPanel( wxWindow* parent );
	//// end generated class members


		std::vector<double> current_accumulated_dose_data;
		std::vector<double> current_x_movement_data;
		std::vector<double> current_y_movement_data;

		mpWindow        *current_plot_window;
		mpInfoLegend    *legend;

		mpFXYVector* current_x_shift_vector_layer;
		mpFXYVector* current_y_shift_vector_layer;

		bool graph_is_hidden;

		long time_of_last_graph_update;

		//mpInfoCoords    *nfo;


		// methods

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

		void ProcessResult(float *result, int result_size, int job_number);
		void UpdateProgressBar();

		virtual void OnJobSocketEvent(wxSocketEvent& event);


};

#endif // __AlignMoviesPanel__
