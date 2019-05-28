#ifndef __MyPhenixSettingsPanel__
#define __MyPhenixSettingsPanel__


/** Implementing PhenixSettingsPanel */
class MyPhenixSettingsPanel : public PhenixSettingsPanel
{
	public:
//		PhenixSettings phenix_settings;
		wxString buffer_phenix_path;

		/** Constructor */
		MyPhenixSettingsPanel( wxWindow* parent );
		//// end generated class members

		void OnPhenixPathTextChanged( wxCommandEvent& event );
		void OnPhenixPathBrowseButtonClick( wxCommandEvent& event );
		void OnUpdateUI( wxUpdateUIEvent& event );

};

#endif // __MyPhenixSettingsPanel__
