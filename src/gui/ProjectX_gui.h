///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Feb 20 2015)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTX_GUI_H__
#define __PROJECTX_GUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class JobPanel;

#include "job_panel.h"
#include <wx/gdicmn.h>
#include <wx/listbook.h>
#include <wx/listctrl.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/treectrl.h>
#include <wx/button.h>
#include <wx/statbox.h>
#include <wx/splitter.h>
#include <wx/frame.h>
#include <wx/statline.h>
#include <wx/stattext.h>
#include <wx/checkbox.h>
#include <wx/combobox.h>
#include <wx/textctrl.h>
#include <wx/dialog.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/statbmp.h>
#include <wx/tglbtn.h>
#include <wx/spinctrl.h>
#include <wx/gauge.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class MainFrame
///////////////////////////////////////////////////////////////////////////////
class MainFrame : public wxFrame 
{
	private:
	
	protected:
		wxSplitterWindow* MainSplitter;
		wxPanel* LeftPanel;
		wxPanel* RightPanel;
		wxButton* m_button12;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnMenuBookChange( wxListbookEvent& event ) { event.Skip(); }
		virtual void OnCollapseAll( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxListbook* MenuBook;
		wxTreeCtrl* AssetTree;
		
		MainFrame( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("ProjectX!"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,800 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~MainFrame();
		
		void MainSplitterOnIdle( wxIdleEvent& )
		{
			MainSplitter->SetSashPosition( 1000 );
			MainSplitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( MainFrame::MainSplitterOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class ActionsPanel
///////////////////////////////////////////////////////////////////////////////
class ActionsPanel : public wxPanel 
{
	private:
	
	protected:
		wxStaticLine* m_staticline3;
	
	public:
		wxListbook* ActionsBook;
		
		ActionsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~ActionsPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class AssetsPanel
///////////////////////////////////////////////////////////////////////////////
class AssetsPanel : public wxPanel 
{
	private:
	
	protected:
		wxStaticLine* m_staticline4;
	
	public:
		wxListbook* AssetsBook;
		
		AssetsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~AssetsPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class OverviewPanel
///////////////////////////////////////////////////////////////////////////////
class OverviewPanel : public wxPanel 
{
	private:
	
	protected:
		wxStaticLine* m_staticline2;
		wxStaticText* m_staticText1;
	
	public:
		
		OverviewPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~OverviewPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class ImageAssetPanel
///////////////////////////////////////////////////////////////////////////////
class ImageAssetPanel : public wxPanel 
{
	private:
	
	protected:
		wxCheckBox* m_checkBox2;
		wxListCtrl* m_listCtrl1;
	
	public:
		
		ImageAssetPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~ImageAssetPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class MovieImportDialog
///////////////////////////////////////////////////////////////////////////////
class MovieImportDialog : public wxDialog 
{
	private:
	
	protected:
		wxListCtrl* PathListCtrl;
		wxButton* m_button10;
		wxButton* m_button11;
		wxButton* ClearButton;
		wxStaticLine* m_staticline7;
		wxStaticText* m_staticText19;
		wxComboBox* VoltageCombo;
		wxStaticText* m_staticText20;
		wxTextCtrl* PixelSizeText;
		wxStaticText* m_staticText21;
		wxTextCtrl* CsText;
		wxStaticText* m_staticText22;
		wxTextCtrl* DoseText;
		wxStaticLine* m_staticline8;
		wxButton* m_button13;
		wxButton* ImportButton;
		
		// Virtual event handlers, overide them in your derived class
		virtual void AddFilesClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddDirectoryClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClearClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTextKeyPress( wxKeyEvent& event ) { event.Skip(); }
		virtual void TextChanged( wxCommandEvent& event ) { event.Skip(); }
		virtual void CancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ImportClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		MovieImportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Import Movies"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 484,539 ), long style = wxCLOSE_BOX ); 
		~MovieImportDialog();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class MovieAssetPanel
///////////////////////////////////////////////////////////////////////////////
class MovieAssetPanel : public wxPanel 
{
	private:
	
	protected:
		wxStaticLine* m_staticline5;
		wxSplitterWindow* m_splitter2;
		wxPanel* m_panel4;
		wxStaticText* m_staticText18;
		wxPanel* m_panel3;
		wxStaticText* m_staticText22;
		wxListCtrl* ContentsListBox;
		wxStaticText* m_staticText19;
		wxButton* ImportMovie;
		wxButton* RemoveSelectedMovieButton;
		wxButton* RemoveAllMoviesButton;
		wxStaticText* m_staticText20;
		wxButton* m_button21;
		wxButton* RemoveGroupButton;
		wxButton* AddSelectedButton;
		wxStaticLine* m_staticline6;
		wxStaticText* m_staticText24;
		wxStaticText* FilenameText;
		wxStaticText* m_staticText4;
		wxStaticText* NumberOfFramesText;
		wxStaticText* m_staticText7;
		wxStaticText* TotalDoseText;
		wxStaticText* m_staticText9;
		wxStaticText* PixelSizeText;
		wxStaticText* m_staticText6;
		wxStaticText* DosePerFrameText;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnBeginEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnEndEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnGroupFocusChange( wxListEvent& event ) { event.Skip(); }
		virtual void OnBeginContentsDrag( wxListEvent& event ) { event.Skip(); }
		virtual void OnContentsSelected( wxListEvent& event ) { event.Skip(); }
		virtual void ImportMovieClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RemoveMovieClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RemoveAllClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void NewGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RemoveGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddSelectedClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxListCtrl* GroupListBox;
		
		MovieAssetPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 940,517 ), long style = wxHSCROLL|wxTAB_TRAVERSAL|wxVSCROLL ); 
		~MovieAssetPanel();
		
		void m_splitter2OnIdle( wxIdleEvent& )
		{
			m_splitter2->SetSashPosition( 200 );
			m_splitter2->Disconnect( wxEVT_IDLE, wxIdleEventHandler( MovieAssetPanel::m_splitter2OnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class ErrorDialog
///////////////////////////////////////////////////////////////////////////////
class ErrorDialog : public wxDialog 
{
	private:
	
	protected:
		wxStaticBitmap* m_bitmap1;
		wxStaticText* m_staticText25;
		wxButton* m_button23;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnClickOK( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxTextCtrl* ErrorText;
		
		ErrorDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 514,500 ), long style = wxDEFAULT_DIALOG_STYLE ); 
		~ErrorDialog();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class AlignMoviesPanel
///////////////////////////////////////////////////////////////////////////////
class AlignMoviesPanel : public JobPanel
{
	private:
	
	protected:
		wxStaticLine* m_staticline12;
		wxStaticText* m_staticText21;
		wxComboBox* GroupComboBox;
		wxToggleButton* m_toggleBtn2;
		wxStaticLine* m_staticline10;
		wxPanel* ExpertPanel;
		wxStaticText* m_staticText43;
		wxStaticText* m_staticText24;
		wxTextCtrl* m_textCtrl5;
		wxStaticText* m_staticText40;
		wxTextCtrl* m_textCtrl10;
		wxStaticText* m_staticText44;
		wxCheckBox* m_checkBox4;
		wxCheckBox* m_checkBox5;
		wxStaticText* m_staticText45;
		wxStaticText* m_staticText46;
		wxTextCtrl* m_textCtrl11;
		wxStaticText* m_staticText47;
		wxSpinCtrl* m_spinCtrl5;
		wxStaticText* m_staticText48;
		wxStaticText* m_staticText49;
		wxSpinCtrl* m_spinCtrl6;
		wxCheckBox* m_checkBox11;
		wxStaticText* m_staticText50;
		wxSpinCtrl* m_spinCtrl7;
		wxStaticText* m_staticText51;
		wxSpinCtrl* m_spinCtrl8;
		wxPanel* GraphPanel;
		wxBoxSizer* GraphSizer;
		wxStaticLine* m_staticline11;
		wxComboBox* m_comboBox3;
		wxGauge* m_gauge4;
		wxButton* StartAlignmentButton;
		wxStaticText* m_staticText52;
		wxTextCtrl* m_textCtrl7;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartAlignmentClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnStartAlignmentButtonUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		
	
	public:
		
		AlignMoviesPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 869,566 ), long style = wxTAB_TRAVERSAL ); 
		~AlignMoviesPanel();
	
};

#endif //__PROJECTX_GUI_H__
