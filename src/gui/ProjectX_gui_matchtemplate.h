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
class ImageGroupPickerComboPanel;
class MemoryComboBox;
class NoFocusBitmapButton;
class NumericTextCtrl;
class PlotCurvePanel;
class ResultsDataViewListCtrl;
class ShowTemplateMatchResultsPanel;
class VolumeAssetPickerComboPanel;

#include "job_panel.h"
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/statline.h>
#include <wx/bmpbuttn.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/listctrl.h>
#include <wx/panel.h>
#include <wx/splitter.h>
#include <wx/radiobut.h>
#include <wx/tglbtn.h>
#include <wx/dataview.h>
#include <wx/combobox.h>
#include <wx/textctrl.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/checkbox.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class ShowTemplateMatchResultsPanelParent
///////////////////////////////////////////////////////////////////////////////
class ShowTemplateMatchResultsPanelParent : public wxPanel
{
	private:

	protected:
		wxSplitterWindow* m_splitter16;
		wxPanel* m_panel87;
		wxSplitterWindow* m_splitter15;
		wxPanel* m_panel89;
		wxStaticText* PeakTableStaticText;
		wxStaticLine* m_staticline148;
		wxStaticLine* m_staticline82;
		wxPanel* BottomPanel;
		wxStaticText* SurvivalHistogramText;
		wxStaticLine* m_staticline81;
		wxPanel* PeakChangesPanel;
		wxPanel* m_panel86;
		wxStaticText* m_staticText394;
		wxStaticText* ImageFileText;
		wxStaticLine* m_staticline86;

		// Virtual event handlers, override them in your derived class
		virtual void OnSavePeaksClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		NoFocusBitmapButton* SaveButton;
		wxListCtrl* PeakListCtrl;
		PlotCurvePanel* HistogramPlotPanel;
		wxListCtrl* ChangesListCtrl;
		DisplayPanel* ImageDisplayPanel;

		ShowTemplateMatchResultsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 952,539 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ShowTemplateMatchResultsPanelParent();

		void m_splitter16OnIdle( wxIdleEvent& )
		{
			m_splitter16->SetSashPosition( 700 );
			m_splitter16->Disconnect( wxEVT_IDLE, wxIdleEventHandler( ShowTemplateMatchResultsPanelParent::m_splitter16OnIdle ), NULL, this );
		}

		void m_splitter15OnIdle( wxIdleEvent& )
		{
			m_splitter15->SetSashPosition( 0 );
			m_splitter15->Disconnect( wxEVT_IDLE, wxIdleEventHandler( ShowTemplateMatchResultsPanelParent::m_splitter15OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class MatchTemplateResultsPanelParent
///////////////////////////////////////////////////////////////////////////////
class MatchTemplateResultsPanelParent : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline25;
		wxSplitterWindow* m_splitter4;
		wxPanel* m_panel13;
		wxRadioButton* AllImagesButton;
		wxRadioButton* ByFilterButton;
		wxButton* FilterButton;
		wxStaticLine* m_staticline77;
		wxToggleButton* JobDetailsToggleButton;
		ResultsDataViewListCtrl* ResultDataView;
		wxButton* PreviousButton;
		wxButton* AddAllToGroupButton;
		wxButton* NextButton;
		wxPanel* RightPanel;
		wxPanel* JobDetailsPanel;
		wxStaticText* JobTitleStaticText;
		wxFlexGridSizer* InfoSizer;
		wxStaticText* m_staticText72;
		wxStaticText* JobIDStaticText;
		wxStaticText* m_staticText74;
		wxStaticText* DateOfRunStaticText;
		wxStaticText* m_staticText93;
		wxStaticText* TimeOfRunStaticText;
		wxStaticText* m_staticText788;
		wxStaticText* RefVolumeIDStaticText;
		wxStaticText* m_staticText790;
		wxStaticText* SymmetryStaticText;
		wxStaticText* m_staticText78;
		wxStaticText* PixelSizeStaticText;
		wxStaticText* m_staticText83;
		wxStaticText* VoltageStaticText;
		wxStaticText* m_staticText82;
		wxStaticText* CsStaticText;
		wxStaticText* m_staticText96;
		wxStaticText* AmplitudeContrastStaticText;
		wxStaticText* m_staticText85;
		wxStaticText* Defocus1StaticText;
		wxStaticText* m_staticText792;
		wxStaticText* Defocus2StaticText;
		wxStaticText* m_staticText794;
		wxStaticText* DefocusAngleStaticText;
		wxStaticText* m_staticText796;
		wxStaticText* PhaseShiftStaticText;
		wxStaticText* m_staticText87;
		wxStaticText* LowResLimitStaticText;
		wxStaticText* m_staticText89;
		wxStaticText* HighResLimitStaticText;
		wxStaticText* m_staticText91;
		wxStaticText* OOPAngluarStepStaticText;
		wxStaticText* m_staticText79;
		wxStaticText* IPAngluarStepStaticText;
		wxStaticText* m_staticText798;
		wxStaticText* DefocusRangeStaticText;
		wxStaticText* m_staticText95;
		wxStaticText* DefocusStepStaticText;
		wxStaticText* LargeAstigExpectedLabel;
		wxStaticText* PixelSizeRangeStaticText;
		wxStaticText* m_staticText99;
		wxStaticText* PixelSizeStepStaticText;
		wxStaticText* m_staticText872;
		wxStaticText* MinPeakRadiusStaticText;
		wxStaticText* m_staticText874;
		wxStaticText* ShiftThresholdStaticText;
		wxStaticText* m_staticText876;
		wxStaticText* IgnoreShiftedPeaksStaticText;
		wxStaticLine* m_staticline30;
		ShowTemplateMatchResultsPanel* ResultPanel;
		wxButton* DeleteFromGroupButton;
		wxButton* AddToGroupButton;
		MemoryComboBox* GroupComboBox;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnAllImagesSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnByFilterSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDefineFilterClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreviousButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddAllToGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRemoveFromGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddToGroupClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		MatchTemplateResultsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 895,557 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~MatchTemplateResultsPanelParent();

		void m_splitter4OnIdle( wxIdleEvent& )
		{
			m_splitter4->SetSashPosition( 450 );
			m_splitter4->Disconnect( wxEVT_IDLE, wxIdleEventHandler( MatchTemplateResultsPanelParent::m_splitter4OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class MatchTemplatePanelParent
///////////////////////////////////////////////////////////////////////////////
class MatchTemplatePanelParent : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline149;
		wxPanel* InputPanel;
		wxStaticText* m_staticText262;
		wxStaticText* m_staticText478;
		VolumeAssetPickerComboPanel* ReferenceSelectPanel;
		wxStaticText* PleaseEstimateCTFStaticText;
		wxStaticLine* m_staticline151;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText201;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* m_staticText189;
		NumericTextCtrl* OutofPlaneStepNumericCtrl;
		wxStaticText* m_staticText190;
		NumericTextCtrl* InPlaneStepNumericCtrl;
		wxStaticText* m_staticText190211;
		NumericTextCtrl* HighResolutionLimitNumericCtrl;
		wxStaticText* m_staticText19021;
		wxStaticText* m_staticText698;
		wxRadioButton* DefocusSearchYesRadio;
		wxRadioButton* DefocusSearchNoRadio;
		wxStaticText* DefocusRangeStaticText;
		NumericTextCtrl* DefocusSearchRangeNumericCtrl;
		wxStaticText* DefocusStepStaticText;
		NumericTextCtrl* DefocusSearchStepNumericCtrl;
		wxStaticText* m_staticText699;
		wxRadioButton* PixelSizeSearchYesRadio;
		wxRadioButton* PixelSizeSearchNoRadio;
		wxStaticText* PixelSizeRangeStaticText;
		NumericTextCtrl* PixelSizeSearchRangeNumericCtrl;
		wxStaticText* PixelSizeStepStaticText;
		NumericTextCtrl* PixelSizeSearchStepNumericCtrl;
		wxStaticText* m_staticText857;
		wxStaticText* m_staticText849;
		NumericTextCtrl* MinPeakRadiusNumericCtrl;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
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
		wxCheckBox* UseGpuCheckBox;
		wxButton* StartEstimationButton;
		wxCheckBox* ResumeRunCheckBox;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartEstimationClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResumeRunCheckBoxOnCheckBox( wxCommandEvent& event ) { event.Skip(); }


	public:
		ImageGroupPickerComboPanel* GroupComboBox;
		wxComboBox* SymmetryComboBox;
		ShowTemplateMatchResultsPanel* ResultsPanel;

		MatchTemplatePanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,731 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~MatchTemplatePanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RefineTemplatePanelParent
///////////////////////////////////////////////////////////////////////////////
class RefineTemplatePanelParent : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline149;
		wxPanel* InputPanel;
		wxStaticText* m_staticText262;
		wxStaticText* m_staticText478;
		VolumeAssetPickerComboPanel* ReferenceSelectPanel;
		wxStaticText* InputErrorText;
		wxStaticLine* m_staticline151;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText847;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* m_staticText849;
		NumericTextCtrl* MinPeakRadiusNumericCtrl;
		wxStaticText* m_staticText846;
		NumericTextCtrl* PeakSelectionThresholdNumericCtrl;
		wxStaticText* m_staticText848;
		NumericTextCtrl* PeakPlottingThresholdNumericCtrl;
		wxStaticText* mask_radius;
		wxRadioButton* RemoveShiftedPeaksYesRadio;
		wxRadioButton* RemoveShiftedPeaksNoRadio;
		wxStaticText* ShiftThresholdStaticText;
		NumericTextCtrl* PeakChangeThresholdNumericTextCtrl;
		wxStaticText* m_staticText201;
		wxStaticText* m_staticText852;
		NumericTextCtrl* MaskRadiusNumericTextCtrl;
		wxStaticText* m_staticText189;
		NumericTextCtrl* OutofPlaneStepNumericCtrl;
		wxStaticText* m_staticText190;
		NumericTextCtrl* InPlaneStepNumericCtrl;
		wxStaticText* m_staticText190211;
		NumericTextCtrl* HighResolutionLimitNumericCtrl;
		wxStaticText* m_staticText19021;
		wxStaticText* m_staticText698;
		wxRadioButton* DefocusSearchYesRadio;
		wxRadioButton* DefocusSearchNoRadio;
		wxStaticText* DefocusRangeStaticText;
		NumericTextCtrl* DefocusSearchRangeNumericCtrl;
		wxStaticText* DefocusStepStaticText;
		NumericTextCtrl* DefocusSearchStepNumericCtrl;
		wxStaticText* m_staticText699;
		wxRadioButton* PixelSizeSearchYesRadio;
		wxRadioButton* PixelSizeSearchNoRadio;
		wxStaticText* PixelSizeRangeStaticText;
		NumericTextCtrl* PixelSizeSearchRangeNumericCtrl;
		wxStaticText* PixelSizeStepStaticText;
		NumericTextCtrl* PixelSizeSearchStepNumericCtrl;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
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
		wxButton* StartEstimationButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartEstimationClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		ImageGroupPickerComboPanel* GroupComboBox;
		wxComboBox* SymmetryComboBox;
		ShowTemplateMatchResultsPanel* ResultsPanel;

		RefineTemplatePanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,731 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RefineTemplatePanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RefineTemplateDevPanelParent
///////////////////////////////////////////////////////////////////////////////
class RefineTemplateDevPanelParent : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline149;
		wxPanel* InputPanel;
		wxStaticText* m_staticText262;
		wxStaticText* m_staticText478;
		VolumeAssetPickerComboPanel* ReferenceSelectPanel;
		wxStaticText* InputErrorText;
		wxStaticLine* m_staticline151;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText847;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* m_staticText849;
		NumericTextCtrl* MinPeakRadiusNumericCtrl;
		wxStaticText* m_staticText846;
		NumericTextCtrl* PeakSelectionThresholdNumericCtrl;
		wxStaticText* m_staticText848;
		NumericTextCtrl* PeakPlottingThresholdNumericCtrl;
		wxStaticText* mask_radius;
		wxRadioButton* RemoveShiftedPeaksYesRadio;
		wxRadioButton* RemoveShiftedPeaksNoRadio;
		wxStaticText* ShiftThresholdStaticText;
		NumericTextCtrl* PeakChangeThresholdNumericTextCtrl;
		wxStaticText* m_staticText201;
		wxStaticText* m_staticText852;
		NumericTextCtrl* MaskRadiusNumericTextCtrl;
		wxStaticText* m_staticText189;
		NumericTextCtrl* OutofPlaneStepNumericCtrl;
		wxStaticText* m_staticText190;
		NumericTextCtrl* InPlaneStepNumericCtrl;
		wxStaticText* m_staticText190211;
		NumericTextCtrl* HighResolutionLimitNumericCtrl;
		wxStaticText* m_staticText698;
		wxRadioButton* DefocusSearchYesRadio;
		wxRadioButton* DefocusSearchNoRadio;
		wxStaticText* DefocusRangeStaticText;
		NumericTextCtrl* DefocusSearchRangeNumericCtrl;
		wxStaticText* DefocusStepStaticText;
		NumericTextCtrl* DefocusSearchStepNumericCtrl;
		wxStaticText* m_staticText699;
		wxRadioButton* AstigmatismSearchYesRadio;
		wxRadioButton* AstigmatismSearchNoRadio;
		wxStaticText* AstigmatismConstraint;
		NumericTextCtrl* PixelSizeSearchRangeNumericCtrl;
		wxStaticText* m_staticText6992;
		wxRadioButton* BeamTiltSearchYesRadio1;
		wxRadioButton* BeamTiltSearchNoRadio1;
		wxStaticText* m_staticText6991;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
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
		wxButton* StartEstimationButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartEstimationClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		ImageGroupPickerComboPanel* GroupComboBox;
		ShowTemplateMatchResultsPanel* ResultsPanel;

		RefineTemplateDevPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,731 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RefineTemplateDevPanelParent();

};

