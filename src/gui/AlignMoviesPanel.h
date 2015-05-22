#ifndef __AlignMoviesPanel__
#define __AlignMoviesPanel__

//// end generated include

/** Implementing AlignMoviesPanel */
class MyAlignMoviesPanel : public AlignMoviesPanel
{

		bool show_expert_options;
		long my_job_id;
		JobPackage my_job_package;

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
		void OnStartAlignmentButtonUpdateUI( wxUpdateUIEvent& event );
		void FillGroupComboBox();
		void StartAlignmentClick( wxCommandEvent& event );

		virtual void OnJobSocketEvent(wxSocketEvent& event);
};

#endif // __AlignMoviesPanel__
