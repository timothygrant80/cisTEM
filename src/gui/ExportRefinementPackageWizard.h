#ifndef __ExportRefinementPackageWizard__
#define __ExportRefinementPackageWizard__

class ExportRefinementPackageWizard : public ExportRefinementPackageWizardParent
{
	protected:
		void OnStackBrowseButtonClick( wxCommandEvent& event );
		void OnMetaBrowseButtonClick( wxCommandEvent& event );
	public:
		/** Constructor */
		ExportRefinementPackageWizard( wxWindow* parent );
		~ExportRefinementPackageWizard();
	//// end generated class members

		void OnFinished(  wxWizardEvent& event  );
		void OnPageChanged(  wxWizardEvent& event  );
		void OnPathChange( wxCommandEvent& event );
		void OnUpdateUI(wxUpdateUIEvent& event);

		void OnParamsComboBox( wxCommandEvent& event );


		void CheckPaths();

		void inline DisableNextButton()
		{
			wxWindow *win = wxWindow::FindWindowById(wxID_FORWARD);
			if(win) win->Enable(false);
		}

		void inline EnableNextButton()
		{
			wxWindow *win = wxWindow::FindWindowById(wxID_FORWARD);
			if(win) win->Enable(true);
		}

		RefinementPackage *current_package;

	
};

#endif
