///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/statline.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/listctrl.h>
#include <wx/button.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/splitter.h>
#include <wx/spinctrl.h>
#include <wx/checkbox.h>
#include <wx/dialog.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class RunProfilesPanel
///////////////////////////////////////////////////////////////////////////////
class RunProfilesPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline12;
		wxSplitterWindow* m_splitter5;
		wxPanel* ProfilesPanel;
		wxListCtrl* ProfilesListBox;
		wxButton* AddProfileButton;
		wxButton* RenameProfileButton;
		wxButton* RemoveProfileButton;
		wxButton* DuplicateProfileButton;
		wxStaticLine* m_staticline26;
		wxButton* ImportButton;
		wxButton* ExportButton;
		wxPanel* CommandsPanel;
		wxStaticLine* m_staticline15;
		wxStaticText* m_staticText34;
		wxStaticText* NumberProcessesStaticText;
		wxStaticText* m_staticText36;
		wxTextCtrl* ManagerTextCtrl;
		wxStaticText* CommandErrorStaticText;
		wxStaticText* m_staticText65;
		wxStaticText* GuiAddressStaticText;
		wxButton* GuiAutoButton;
		wxButton* ControllerSpecifyButton;
		wxStaticText* m_staticText67;
		wxStaticText* ControllerAddressStaticText;
		wxButton* ControllerAutoButton;
		wxButton* m_button38;
		wxStaticText* m_staticText70;
		wxListCtrl* CommandsListBox;
		wxButton* AddCommandButton;
		wxButton* EditCommandButton;
		wxButton* RemoveCommandButton;
		wxButton* CommandsSaveButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnProfileDClick( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnProfileLeftDown( wxMouseEvent& event ) { event.Skip(); }
		virtual void MouseVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnEndProfileEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnProfilesListItemActivated( wxListEvent& event ) { event.Skip(); }
		virtual void OnProfilesFocusChange( wxListEvent& event ) { event.Skip(); }
		virtual void OnAddProfileClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameProfileClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRemoveProfileClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDuplicateProfileClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImportButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ManagerTextChanged( wxCommandEvent& event ) { event.Skip(); }
		virtual void GuiAddressAutoClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void GuiAddressSpecifyClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ControllerAddressAutoClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ControllerAddressSpecifyClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCommandDClick( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnCommandLeftDown( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnCommandsActivated( wxListEvent& event ) { event.Skip(); }
		virtual void OnCommandsFocusChange( wxListEvent& event ) { event.Skip(); }
		virtual void AddCommandButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void EditCommandButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RemoveCommandButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void CommandsSaveButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		RunProfilesPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 940,517 ), long style = wxHSCROLL|wxTAB_TRAVERSAL|wxVSCROLL, const wxString& name = wxEmptyString );

		~RunProfilesPanel();

		void m_splitter5OnIdle( wxIdleEvent& )
		{
			m_splitter5->SetSashPosition( 349 );
			m_splitter5->Disconnect( wxEVT_IDLE, wxIdleEventHandler( RunProfilesPanel::m_splitter5OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class AddRunCommandDialog
///////////////////////////////////////////////////////////////////////////////
class AddRunCommandDialog : public wxDialog
{
	private:

	protected:
		wxStaticText* m_staticText45;
		wxStaticText* m_staticText46;
		wxStaticText* ThreadsPerCopySpinCtrl;
		wxStaticText* m_staticText58;
		wxStaticLine* m_staticline166;
		wxStaticText* ErrorStaticText;
		wxStaticLine* m_staticline14;
		wxButton* OKButton;
		wxButton* CancelButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOverrideCheckbox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOKClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCancelClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxTextCtrl* CommandTextCtrl;
		wxSpinCtrl* NumberCopiesSpinCtrl;
		wxSpinCtrl* NumberThreadsSpinCtrl;
		wxSpinCtrl* DelayTimeSpinCtrl;
		wxCheckBox* OverrideCheckBox;
		wxSpinCtrl* OverridenNoCopiesSpinCtrl;

		AddRunCommandDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Enter Command..."), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~AddRunCommandDialog();

};

