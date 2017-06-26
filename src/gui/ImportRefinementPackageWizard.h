#ifndef __ImportRefinementPackageWizard__
#define __ImportRefinementPackageWizard__

class ImportRefinementPackageWizard : public ImportRefinementPackageWizardParent
{
	protected:
		void OnStackBrowseButtonClick( wxCommandEvent& event );
		void OnMetaBrowseButtonClick( wxCommandEvent& event );
	public:
		/** Constructor */
		ImportRefinementPackageWizard( wxWindow* parent );
	//// end generated class members

		void OnFinished(  wxWizardEvent& event  );
		void OnPageChanged(  wxWizardEvent& event  );
		void OnPathChange( wxCommandEvent& event );
		void OnUpdateUI(wxUpdateUIEvent& event);


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

	
};

#endif
