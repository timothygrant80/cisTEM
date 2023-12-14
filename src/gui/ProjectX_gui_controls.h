///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class AngularDistributionPlotPanel;
class AssetPickerListCtrl;
class MemoryComboBox;
class NoFocusBitmapButton;
class NumericTextCtrl;
class PlotCurvePanel;
class PlotFSCPanel;
class RefinementParametersListCtrl;

#include <wx/string.h>
#include <wx/combobox.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/bmpbuttn.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/statline.h>
#include <wx/scrolwin.h>
#include <wx/dialog.h>
#include <wx/toolbar.h>
#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/statbmp.h>
#include <wx/listctrl.h>
#include <wx/choice.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class AssetPickerComboPanelParent
///////////////////////////////////////////////////////////////////////////////
class AssetPickerComboPanelParent : public wxPanel
{
	private:

	protected:

		// Virtual event handlers, override them in your derived class
		virtual void OnSize( wxSizeEvent& event ) { event.Skip(); }
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnPreviousButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxBoxSizer* bSizer436;
		MemoryComboBox* AssetComboBox;
		wxBoxSizer* bSizer494;
		NoFocusBitmapButton* PreviousButton;
		NoFocusBitmapButton* NextButton;
		NoFocusBitmapButton* WindowSelectButton;

		AssetPickerComboPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~AssetPickerComboPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class FilterDialog
///////////////////////////////////////////////////////////////////////////////
class FilterDialog : public wxDialog
{
	private:

	protected:
		wxBoxSizer* MainBoxSizer;
		wxStaticText* m_staticText64;
		wxStaticLine* m_staticline18;
		wxScrolledWindow* FilterScrollPanel;
		wxBoxSizer* FilterBoxSizer;
		wxStaticLine* m_staticline19;
		wxScrolledWindow* SortScrollPanel;
		wxGridSizer* SortSizer;
		wxStaticLine* m_staticline21;
		wxButton* CancelButton;
		wxButton* FilterButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnCancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFilterClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxStaticText* m_staticText81;

		FilterDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Filter / Sort Movies"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE );

		~FilterDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class FSCPanel
///////////////////////////////////////////////////////////////////////////////
class FSCPanel : public wxPanel
{
	private:

	protected:
		wxBoxSizer* TitleSizer;
		wxStaticText* TitleStaticText;
		wxStaticText* EstimatedResolutionLabel;
		wxStaticText* EstimatedResolutionText;
		wxStaticLine* m_staticline104;
		wxStaticLine* m_staticline52;
		PlotFSCPanel* PlotPanel;

		// Virtual event handlers, override them in your derived class
		virtual void SaveImageClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void PopupTextClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		NoFocusBitmapButton* SaveButton;
		NoFocusBitmapButton* FSCDetailsButton;

		FSCPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~FSCPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class DisplayPanelParent
///////////////////////////////////////////////////////////////////////////////
class DisplayPanelParent : public wxPanel
{
	private:

	protected:
		wxBoxSizer* MainSizer;
		wxToolBar* Toolbar;

	public:

		DisplayPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~DisplayPanelParent();

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

		// Virtual event handlers, override them in your derived class
		virtual void OnClickOK( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxTextCtrl* ErrorText;

		ErrorDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 514,500 ), long style = wxDEFAULT_DIALOG_STYLE );

		~ErrorDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ListCtrlDialog
///////////////////////////////////////////////////////////////////////////////
class ListCtrlDialog : public wxDialog
{
	private:

	protected:
		wxButton* m_button108;
		wxButton* m_button109;

	public:
		AssetPickerListCtrl* MyListCtrl;

		ListCtrlDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE );

		~ListCtrlDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class PopupTextDialogParent
///////////////////////////////////////////////////////////////////////////////
class PopupTextDialogParent : public wxDialog
{
	private:

	protected:
		wxButton* CloseButton;
		wxButton* ClipBoardButton;
		wxButton* m_button146;

		// Virtual event handlers, override them in your derived class
		virtual void OnCloseButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCopyToClipboardClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxTextCtrl* OutputTextCtrl;

		PopupTextDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxCAPTION|wxCLOSE_BOX|wxRESIZE_BORDER );

		~PopupTextDialogParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class LargeAngularPlotDialogParent
///////////////////////////////////////////////////////////////////////////////
class LargeAngularPlotDialogParent : public wxDialog
{
	private:

	protected:
		wxButton* CloseButton;
		wxButton* ClipBoardButton;
		wxButton* SaveButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnCloseButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCopyToClipboardClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		AngularDistributionPlotPanel* AngularPlotPanel;

		LargeAngularPlotDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1000,800 ), long style = wxCAPTION|wxCLOSE_BOX|wxRESIZE_BORDER );

		~LargeAngularPlotDialogParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RefinementParametersDialogParent
///////////////////////////////////////////////////////////////////////////////
class RefinementParametersDialogParent : public wxDialog
{
	private:

	protected:
		wxToolBar* ClassToolBar;
		wxStaticText* m_staticText831;
		wxStaticLine* m_staticline137;
		RefinementParametersListCtrl* ParameterListCtrl;
		wxButton* CloseButton;
		wxButton* SaveButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnCloseButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		RefinementParametersDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1000,800 ), long style = wxCAPTION|wxCLOSE_BOX|wxRESIZE_BORDER );

		~RefinementParametersDialogParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class DistributionPlotDialogParent
///////////////////////////////////////////////////////////////////////////////
class DistributionPlotDialogParent : public wxDialog
{
	private:

	protected:
		wxStaticText* m_staticText64711;
		NumericTextCtrl* UpperBoundYNumericCtrl;
		wxStaticText* m_staticText6472;
		NumericTextCtrl* LowerBoundYNumericCtrl;
		wxChoice* DataSeriesToPlotChoice;
		wxStaticText* m_staticText647;
		NumericTextCtrl* LowerBoundXNumericCtrl;
		wxStaticText* m_staticText6471;
		NumericTextCtrl* UpperBoundXNumericCtrl;
		wxButton* SaveTXTButton;
		wxButton* SavePNGButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpperBoundYKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnUpperBoundYSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnUpperBoundYTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLowerBoundYKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnLowerBoundYSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnLowerBoundYTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDataSeriesToPlotChoice( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLowerBoundXKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnLowerBoundXSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnLowerBoundXTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnUpperBoundXKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnUpperBoundXSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnUpperBoundXTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveTXTButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSavePNGButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		PlotCurvePanel* PlotCurvePanelInstance;

		DistributionPlotDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 575,462 ), long style = wxDEFAULT_DIALOG_STYLE );

		~DistributionPlotDialogParent();

};

