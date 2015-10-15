#ifndef __AlignMoviesPanel__
#define __AlignMoviesPanel__

//// end generated include

/** Implementing AlignMoviesPanel */
class MyAlignMoviesPanel : public AlignMoviesPanel
{

		bool show_expert_options;
		long my_job_id;
		JobPackage my_job_package;
		bool running_job;


public:
		/** Constructor */
		MyAlignMoviesPanel( wxWindow* parent );
	//// end generated class members

		mpWindow        *plot_window;
		//mpInfoCoords    *nfo;

		std::vector<double> accumulated_dose_data;
		std::vector<double> average_movement_data;



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

		void WriteInfoText(wxString text_to_write);
		void WriteErrorText(wxString text_to_write);

		virtual void OnJobSocketEvent(wxSocketEvent& event);


};

#endif // __AlignMoviesPanel__
