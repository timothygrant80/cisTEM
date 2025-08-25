///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/gdicmn.h>
#include <wx/listbook.h>
#include <wx/listctrl.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/menu.h>
#include <wx/frame.h>
#include <wx/statline.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/statbmp.h>
#include <wx/stattext.h>
#include <wx/hyperlink.h>
#include <wx/button.h>
#include <wx/dialog.h>
#include <wx/textctrl.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class MainFrame
///////////////////////////////////////////////////////////////////////////////
class MainFrame : public wxFrame
{
	private:

	protected:
		wxPanel* LeftPanel;
		wxMenuBar* m_menubar1;
		wxMenu* FileMenu;
		wxMenu* WorkflowMenu;
		wxMenu* HelpMenu;

		// Virtual event handlers, override them in your derived class
		virtual void OnMenuBookChange( wxListbookEvent& event ) { event.Skip(); }
		virtual void OnFileMenuUpdate( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnFileNewProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileOpenProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileCloseProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileExit( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHelpLaunch( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAboutLaunch( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxListbook* MenuBook;

		MainFrame( wxWindow* parent, wxWindowID id = wxID_OPEN, const wxString& title = wxT("cisTEM"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1366,768 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL|wxWANTS_CHARS );

		~MainFrame();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ActionsPanelParent
///////////////////////////////////////////////////////////////////////////////
class ActionsPanelParent : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline3;

		// Virtual event handlers, override them in your derived class
		virtual void OnActionsBookPageChanged( wxListbookEvent& event ) { event.Skip(); }


	public:
		wxListbook* ActionsBook;

		ActionsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ActionsPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class SettingsPanel
///////////////////////////////////////////////////////////////////////////////
class SettingsPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline3;

		// Virtual event handlers, override them in your derived class
		virtual void OnSettingsBookPageChanged( wxListbookEvent& event ) { event.Skip(); }


	public:
		wxListbook* SettingsBook;

		SettingsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~SettingsPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ResultsPanel
///////////////////////////////////////////////////////////////////////////////
class ResultsPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline3;

		// Virtual event handlers, override them in your derived class
		virtual void OnResultsBookPageChanged( wxListbookEvent& event ) { event.Skip(); }


	public:
		wxListbook* ResultsBook;

		ResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ResultsPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class AssetsPanel
///////////////////////////////////////////////////////////////////////////////
class AssetsPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline68;

		// Virtual event handlers, override them in your derived class
		virtual void OnAssetsBookPageChanged( wxListbookEvent& event ) { event.Skip(); }


	public:
		wxListbook* AssetsBook;

		AssetsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~AssetsPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ExperimentalPanel
///////////////////////////////////////////////////////////////////////////////
class ExperimentalPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline68;

		// Virtual event handlers, override them in your derived class
		virtual void OnExperimentalBookPageChanged( wxListbookEvent& event ) { event.Skip(); }


	public:
		wxListbook* ExperimentalBook;

		ExperimentalPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ExperimentalPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class OverviewPanel
///////////////////////////////////////////////////////////////////////////////
class OverviewPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline2;
		wxPanel* WelcomePanel;

		// Virtual event handlers, override them in your derived class
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }


	public:
		wxRichTextCtrl* InfoText;

		OverviewPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~OverviewPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class AboutDialog
///////////////////////////////////////////////////////////////////////////////
class AboutDialog : public wxDialog
{
	private:

	protected:
		wxStaticLine* m_staticline131;
		wxStaticLine* m_staticline130;
		wxStaticText* m_staticText611;
		wxStaticText* m_staticText605;
		wxStaticText* m_staticText606;
		wxStaticText* m_staticText607;
		wxStaticText* m_staticText608;
		wxStaticText* m_staticText609;
		wxStaticText* m_staticText610;
		wxStaticText* m_staticText613;
		wxStaticText* m_staticText614;
		wxStaticText* m_staticText615;
		wxHyperlinkCtrl* m_hyperlink1;
		wxStaticText* m_staticText617;
		wxHyperlinkCtrl* m_hyperlink2;
		wxStaticLine* m_staticline129;
		wxButton* m_button141;

	public:
		wxStaticBitmap* LogoBitmap;
		wxStaticText* VersionStaticText;
		wxStaticText* BuildDateText;

		AboutDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("About cisTEM"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE );

		~AboutDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class DatabaseUpdateDialogParent
///////////////////////////////////////////////////////////////////////////////
class DatabaseUpdateDialogParent : public wxDialog
{
	private:

	protected:
		enum
		{
			wxID_UPDATE_ONLY = 1000,
			wxID_BACKUP_AND_UPDATE
		};

		wxStaticText* UpdateText;
		wxTextCtrl* SchemaChangesTextCtrl;
		wxStaticLine* m_staticline10;
		wxButton* CancelButton;
		wxButton* UpdateButton;
		wxButton* BackupUpdateButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnButtonClicked( wxCommandEvent& event ) { event.Skip(); }


	public:

		DatabaseUpdateDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Differences in Database Schema Detected!"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~DatabaseUpdateDialogParent();

};

