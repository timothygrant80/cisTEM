///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class DisplayPanel;

#include <wx/gdicmn.h>
#include <wx/toolbar.h>
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
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/button.h>
#include <wx/dialog.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class DisplayPanelParent
///////////////////////////////////////////////////////////////////////////////
class DisplayPanelParent : public wxPanel
{
	private:

	protected:
		wxBoxSizer* MainSizer;
		wxToolBar* Toolbar;

		// Virtual event handlers, override them in your derived class
		virtual void OnMiddleUp( wxMouseEvent& event ) { event.Skip(); }


	public:

		DisplayPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~DisplayPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class DisplayFrameParent
///////////////////////////////////////////////////////////////////////////////
class DisplayFrameParent : public wxFrame
{
	private:

	protected:
		wxMenuBar* m_menubar2;
		wxMenu* DisplayFileMenu;
		wxMenuItem* DisplayFileOpen;
		wxMenuItem* DisplayCloseTab;
		wxMenuItem* SelectOpenTxt;
		wxMenuItem* SelectSaveTxt;
		wxMenuItem* SelectSaveTxtAs;
		wxMenuItem* DisplayExit;
		wxMenu* DisplayLabelMenu;
		wxMenuItem* LabelLocationNumber;
		wxMenu* DisplaySelectMenu;
		wxMenuItem* SelectImageSelectionMode;
		wxMenuItem* SelectCoordsSelectionMode;
		wxMenuItem* SelectInvertSelection;
		wxMenuItem* SelectClearSelection;
		wxMenu* DisplayOptionsMenu;
		wxMenu* OptionsSetPointSize;
		wxMenuItem* CoordSize3;
		wxMenuItem* CoordSize5;
		wxMenuItem* CoordSize7;
		wxMenuItem* CoordSize10;
		wxMenuItem* OptionsSingleImageMode;
		wxMenuItem* OptionsShowSelectionDistances;
		wxMenuItem* OptionsShowResolution;
		wxMenu* DisplayHelpMenu;
		wxMenuItem* HelpAbout;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnFileOpenClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCloseTabClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenTxtClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveTxtClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveTxtAsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExitClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLocationNumberClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImageSelectionModeClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCoordsSelectionModeClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInvertSelectionClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnClearSelectionClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSize3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSize5( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSize7( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSize10( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSingleImageModeClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowSelectionDistancesClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowResolution( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDocumentationClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxBoxSizer* bSizer631;
		DisplayPanel* cisTEMDisplayPanel;

		DisplayFrameParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("cisTEM Display"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~DisplayFrameParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class DisplayManualDialogParent
///////////////////////////////////////////////////////////////////////////////
class DisplayManualDialogParent : public wxDialog
{
	private:

	protected:
		wxBoxSizer* MainSizer;
		wxStaticLine* m_staticline58;
		wxStaticText* m_staticText315;
		wxTextCtrl* minimum_text_ctrl;
		wxStaticText* m_staticText316;
		wxTextCtrl* maximum_text_ctrl;
		wxStaticText* m_staticText317;
		wxToolBar* Toolbar;
		wxStaticLine* m_staticline61;
		wxCheckBox* histogram_checkbox;
		wxCheckBox* live_checkbox;
		wxStaticLine* m_staticline63;
		wxButton* m_button94;
		wxButton* m_button95;

		// Virtual event handlers, override them in your derived class
		virtual void OnClose( wxCloseEvent& event ) { event.Skip(); }
		virtual void OnLeftDown( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnMotion( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnPaint( wxPaintEvent& event ) { event.Skip(); }
		virtual void OnRightDown( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnLowChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHistogramCheck( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRealtimeCheck( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnButtonOK( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnButtonCancel( wxCommandEvent& event ) { event.Skip(); }


	public:

		DisplayManualDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Manual Grey Settings"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE );

		~DisplayManualDialogParent();

};

