///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class BitmapPanel;
class DisplayPanel;
class MemoryComboBox;
class MovieGroupPickerComboPanel;
class ResultsDataViewListCtrl;
class UnblurResultsPanel;

#include "job_panel.h"
#include <wx/statline.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/radiobut.h>
#include <wx/button.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/tglbtn.h>
#include <wx/sizer.h>
#include <wx/dataview.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/combobox.h>
#include <wx/splitter.h>
#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/spinctrl.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class MovieAlignResultsPanel
///////////////////////////////////////////////////////////////////////////////
class MovieAlignResultsPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline25;
		wxSplitterWindow* m_splitter4;
		wxPanel* m_panel13;
		wxRadioButton* AllMoviesButton;
		wxRadioButton* ByFilterButton;
		wxButton* FilterButton;
		wxStaticLine* m_staticline76;
		wxToggleButton* JobDetailsToggleButton;
		ResultsDataViewListCtrl* ResultDataView;
		wxButton* PreviousButton;
		wxButton* AddAllToGroupButton;
		wxButton* NextButton;
		wxPanel* RightPanel;
		wxPanel* JobDetailsPanel;
		wxFlexGridSizer* InfoSizer;
		wxStaticText* m_staticText72;
		wxStaticText* AlignmentIDStaticText;
		wxStaticText* m_staticText74;
		wxStaticText* DateOfRunStaticText;
		wxStaticText* m_staticText93;
		wxStaticText* TimeOfRunStaticText;
		wxStaticText* m_staticText83;
		wxStaticText* VoltageStaticText;
		wxStaticText* m_staticText78;
		wxStaticText* PixelSizeStaticText;
		wxStaticText* m_staticText82;
		wxStaticText* ExposureStaticText;
		wxStaticText* m_staticText96;
		wxStaticText* PreExposureStaticText;
		wxStaticText* m_staticText85;
		wxStaticText* MinShiftStaticText;
		wxStaticText* m_staticText87;
		wxStaticText* MaxShiftStaticText;
		wxStaticText* m_staticText89;
		wxStaticText* TerminationThresholdStaticText;
		wxStaticText* m_staticText91;
		wxStaticText* MaxIterationsStaticText;
		wxStaticText* m_staticText79;
		wxStaticText* BfactorStaticText;
		wxStaticText* m_staticText95;
		wxStaticText* ExposureFilterStaticText;
		wxStaticText* m_staticText99;
		wxStaticText* RestorePowerStaticText;
		wxStaticText* m_staticText101;
		wxStaticText* MaskCrossStaticText;
		wxStaticText* m_staticText103;
		wxStaticText* HorizontalMaskStaticText;
		wxStaticText* m_staticText105;
		wxStaticText* VerticalMaskStaticText;
		wxStaticText* m_staticText1051;
		wxStaticText* IncludeAllFramesStaticText;
		wxStaticText* m_staticText1052;
		wxStaticText* FirstFrameStaticText;
		wxStaticText* m_staticText1053;
		wxStaticText* LastFrameStaticText;
		UnblurResultsPanel* ResultPanel;
		wxButton* DeleteFromGroupButton;
		wxButton* AddToGroupButton;
		MemoryComboBox* GroupComboBox;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnAllMoviesSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnByFilterSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDefineFilterClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreviousButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddAllToGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRemoveFromGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddToGroupClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		MovieAlignResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1165,564 ), long style = wxTAB_TRAVERSAL|wxWANTS_CHARS, const wxString& name = wxEmptyString );

		~MovieAlignResultsPanel();

		void m_splitter4OnIdle( wxIdleEvent& )
		{
			m_splitter4->SetSashPosition( 450 );
			m_splitter4->Disconnect( wxEVT_IDLE, wxIdleEventHandler( MovieAlignResultsPanel::m_splitter4OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class UnblurResultsPanelParent
///////////////////////////////////////////////////////////////////////////////
class UnblurResultsPanelParent : public wxPanel
{
	private:

	protected:
		wxBoxSizer* MainSizer;
		wxSplitterWindow* m_splitter13;
		wxPanel* m_panel80;
		wxBoxSizer* SplitSizer;
		wxSplitterWindow* m_splitter14;
		wxPanel* m_panel82;
		wxStaticText* m_staticText372;
		wxStaticLine* m_staticline73;
		wxPanel* m_panel83;
		wxStaticText* m_staticText373;
		wxStaticLine* m_staticline74;
		wxPanel* PlotPanel;
		wxBoxSizer* GraphSizer;
		wxPanel* m_panel81;
		wxStaticText* m_staticText371;
		wxStaticLine* m_staticline72;

	public:
		wxStaticText* SpectraNyquistStaticText;
		BitmapPanel* SpectraPanel;
		wxStaticText* FilenameStaticText;
		DisplayPanel* ImageDisplayPanel;

		UnblurResultsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 698,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~UnblurResultsPanelParent();

		void m_splitter13OnIdle( wxIdleEvent& )
		{
			m_splitter13->SetSashPosition( 542 );
			m_splitter13->Disconnect( wxEVT_IDLE, wxIdleEventHandler( UnblurResultsPanelParent::m_splitter13OnIdle ), NULL, this );
		}

		void m_splitter14OnIdle( wxIdleEvent& )
		{
			m_splitter14->SetSashPosition( 0 );
			m_splitter14->Disconnect( wxEVT_IDLE, wxIdleEventHandler( UnblurResultsPanelParent::m_splitter14OnIdle ), NULL, this );
		}

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
		wxToggleButton* ExpertToggleButton;
		wxStaticLine* m_staticline10;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText43;
		wxStaticText* m_staticText24;
		wxTextCtrl* minimum_shift_text;
		wxStaticText* m_staticText40;
		wxTextCtrl* maximum_shift_text;
		wxStaticText* m_staticText44;
		wxCheckBox* dose_filter_checkbox;
		wxCheckBox* restore_power_checkbox;
		wxStaticText* m_staticText45;
		wxStaticText* m_staticText46;
		wxTextCtrl* termination_threshold_text;
		wxStaticText* m_staticText47;
		wxSpinCtrl* max_iterations_spinctrl;
		wxStaticText* m_staticText48;
		wxStaticText* m_staticText49;
		wxSpinCtrl* bfactor_spinctrl;
		wxCheckBox* mask_central_cross_checkbox;
		wxStaticText* horizontal_mask_static_text;
		wxSpinCtrl* horizontal_mask_spinctrl;
		wxStaticText* vertical_mask_static_text;
		wxSpinCtrl* vertical_mask_spinctrl;
		wxStaticText* m_staticText481;
		wxCheckBox* include_all_frames_checkbox;
		wxStaticText* first_frame_static_text;
		wxSpinCtrl* first_frame_spin_ctrl;
		wxStaticText* last_frame_static_text;
		wxSpinCtrl* last_frame_spin_ctrl;
		wxCheckBox* SaveScaledSumCheckbox;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
		UnblurResultsPanel* GraphPanel;
		wxPanel* InfoPanel;
		wxRichTextCtrl* InfoText;
		wxStaticLine* m_staticline11;
		wxPanel* ProgressPanel;
		wxStaticText* NumberConnectedText;
		wxGauge* ProgressBar;
		wxStaticText* TimeRemainingText;
		wxStaticLine* m_staticline60;
		wxButton* FinishButton;
		wxButton* CancelAlignmentButton;
		wxPanel* StartPanel;
		wxStaticText* RunProfileText;
		MemoryComboBox* RunProfileComboBox;
		wxButton* StartAlignmentButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartAlignmentClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		MovieGroupPickerComboPanel* GroupComboBox;

		AlignMoviesPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 927,653 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~AlignMoviesPanel();

};

