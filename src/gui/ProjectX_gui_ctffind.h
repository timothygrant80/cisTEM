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
class CTF1DPanel;
class DisplayCTFRefinementResultsPanel;
class DisplayPanel;
class ImageGroupPickerComboPanel;
class MemoryComboBox;
class MyFSCPanel;
class NoFocusBitmapButton;
class NumericTextCtrl;
class PlotCurvePanel;
class ReferenceVolumesListControlRefinement;
class RefinementPackagePickerComboPanel;
class RefinementPickerComboPanel;
class ResultsDataViewListCtrl;
class ShowCTFResultsPanel;
class VolumeAssetPickerComboPanel;

#include "job_panel.h"
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/statline.h>
#include <wx/panel.h>
#include <wx/sizer.h>
#include <wx/splitter.h>
#include <wx/radiobut.h>
#include <wx/button.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/bmpbuttn.h>
#include <wx/tglbtn.h>
#include <wx/dataview.h>
#include <wx/combobox.h>
#include <wx/checkbox.h>
#include <wx/textctrl.h>
#include <wx/listctrl.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/spinctrl.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class ShowCTFResultsPanelParent
///////////////////////////////////////////////////////////////////////////////
class ShowCTFResultsPanelParent : public wxPanel
{
	private:

	protected:
		wxSplitterWindow* m_splitter16;
		wxPanel* m_panel87;
		wxSplitterWindow* m_splitter15;
		wxPanel* m_panel88;
		wxStaticText* m_staticText377;
		wxStaticLine* m_staticline81;
		wxPanel* m_panel89;
		wxStaticText* m_staticText378;
		wxStaticLine* m_staticline82;
		wxPanel* m_panel86;
		wxStaticText* m_staticText379;
		wxStaticLine* m_staticline78;
		wxStaticText* m_staticText380;
		wxStaticText* Defocus1Text;
		wxStaticText* m_staticText389;
		wxStaticText* ScoreText;
		wxStaticText* m_staticText382;
		wxStaticText* Defocus2Text;
		wxStaticText* m_staticText391;
		wxStaticText* FitResText;
		wxStaticText* m_staticText384;
		wxStaticText* AngleText;
		wxStaticText* m_staticText393;
		wxStaticText* AliasResText;
		wxStaticText* m_staticText386;
		wxStaticText* PhaseShiftText;
		wxStaticText* IcinessLabel;
		wxStaticText* IcinessStaticText;
		wxStaticText* m_staticText866;
		wxStaticText* TiltAngleStaticText;
		wxStaticText* m_staticText868;
		wxStaticText* TiltAxisStaticText;
		wxStaticText* m_staticText8681;
		wxStaticText* ThicknessStaticText;
		wxStaticLine* m_staticline83;
		wxStaticText* m_staticText394;
		wxStaticText* ImageFileText;
		wxStaticLine* m_staticline86;

	public:
		BitmapPanel* CTF2DResultsPanel;
		CTF1DPanel* CTFPlotPanel;
		DisplayPanel* ImageDisplayPanel;

		ShowCTFResultsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 952,539 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ShowCTFResultsPanelParent();

		void m_splitter16OnIdle( wxIdleEvent& )
		{
			m_splitter16->SetSashPosition( 700 );
			m_splitter16->Disconnect( wxEVT_IDLE, wxIdleEventHandler( ShowCTFResultsPanelParent::m_splitter16OnIdle ), NULL, this );
		}

		void m_splitter15OnIdle( wxIdleEvent& )
		{
			m_splitter15->SetSashPosition( 0 );
			m_splitter15->Disconnect( wxEVT_IDLE, wxIdleEventHandler( ShowCTFResultsPanelParent::m_splitter15OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class FindCTFResultsPanel
///////////////////////////////////////////////////////////////////////////////
class FindCTFResultsPanel : public wxPanel
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
		wxFlexGridSizer* InfoSizer;
		wxStaticText* m_staticText72;
		wxStaticText* EstimationIDStaticText;
		wxStaticText* m_staticText74;
		wxStaticText* DateOfRunStaticText;
		wxStaticText* m_staticText93;
		wxStaticText* TimeOfRunStaticText;
		wxStaticText* m_staticText83;
		wxStaticText* VoltageStaticText;
		wxStaticText* m_staticText82;
		wxStaticText* CsStaticText;
		wxStaticText* m_staticText78;
		wxStaticText* PixelSizeStaticText;
		wxStaticText* m_staticText781;
		wxStaticText* ResampleStaticText;
		wxStaticText* m_staticText7811;
		wxStaticText* PixelSizeTargetStaticText;
		wxStaticText* m_staticText78111;
		wxStaticText* EstimateTiltStaticText;
		wxStaticText* m_staticText96;
		wxStaticText* AmplitudeContrastStaticText;
		wxStaticText* m_staticText85;
		wxStaticText* BoxSizeStaticText;
		wxStaticText* m_staticText87;
		wxStaticText* MinResStaticText;
		wxStaticText* m_staticText89;
		wxStaticText* MaxResStaticText;
		wxStaticText* m_staticText91;
		wxStaticText* MinDefocusStaticText;
		wxStaticText* m_staticText79;
		wxStaticText* MaxDefocusStaticText;
		wxStaticText* m_staticText95;
		wxStaticText* DefocusStepStaticText;
		wxStaticText* LargeAstigExpectedLabel;
		wxStaticText* LargeAstigExpectedStaticText;
		wxStaticText* m_staticText99;
		wxStaticText* RestrainAstigStaticText;
		wxStaticText* ToleratedAstigLabel;
		wxStaticText* ToleratedAstigStaticText;
		wxStaticText* NumberOfAveragedFramesLabel;
		wxStaticText* NumberOfAveragedFramesStaticText;
		wxStaticText* m_staticText103;
		wxStaticText* AddtionalPhaseShiftStaticText;
		wxStaticText* MinPhaseShiftLabel;
		wxStaticText* MinPhaseShiftStaticText;
		wxStaticText* MaxPhaseShiftLabel;
		wxStaticText* MaxPhaseshiftStaticText;
		wxStaticText* PhaseShiftStepLabel;
		wxStaticText* PhaseShiftStepStaticText;
		wxStaticText* PhaseShiftStepLabel1;
		wxStaticText* EstimateThicknessStaticText;
		wxStaticText* ThicknessLabel1;
		wxStaticText* Thickness1DStaticText;
		wxStaticText* ThicknessLabel2;
		wxStaticText* Thickness2DStaticText;
		wxStaticText* ThicknessLabel3;
		wxStaticText* ThicknessMinResText;
		wxStaticText* ThicknessLabel4;
		wxStaticText* ThicknessMaxResStaticText;
		wxStaticText* ThicknessLabel5;
		wxStaticText* ThicknessNoDecayStaticText;
		wxStaticText* ThicknessLabel6;
		wxStaticText* ThicknessDownweightNodesStaticText;
		wxStaticLine* m_staticline30;
		ShowCTFResultsPanel* ResultPanel;
		wxButton* DeleteFromGroupButton;
		wxButton* AddToGroupButton;
		MemoryComboBox* GroupComboBox;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnAllMoviesSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnByFilterSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDefineFilterClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPlotResultsButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreviousButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddAllToGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRemoveFromGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddToGroupClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		NoFocusBitmapButton* PlotResultsButton;

		FindCTFResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 895,557 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~FindCTFResultsPanel();

		void m_splitter4OnIdle( wxIdleEvent& )
		{
			m_splitter4->SetSashPosition( 450 );
			m_splitter4->Disconnect( wxEVT_IDLE, wxIdleEventHandler( FindCTFResultsPanel::m_splitter4OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class RefineCTFPanelParent
///////////////////////////////////////////////////////////////////////////////
class RefineCTFPanelParent : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline12;
		wxPanel* InputParamsPanel;
		wxStaticText* m_staticText262;
		wxStaticText* m_staticText263;
		wxCheckBox* UseMaskCheckBox;
		VolumeAssetPickerComboPanel* MaskSelectPanel;
		wxStaticLine* m_staticline52;
		wxCheckBox* RefineBeamTiltCheckBox;
		wxStaticLine* m_staticline142;
		wxCheckBox* RefineCTFCheckBox;
		wxStaticText* HiResLimitStaticText;
		NumericTextCtrl* HighResolutionLimitTextCtrl;
		wxStaticLine* m_staticline143;
		wxToggleButton* ExpertToggleButton;
		wxStaticLine* m_staticline101;
		ReferenceVolumesListControlRefinement* Active3DReferencesListCtrl;
		wxStaticText* PleaseCreateRefinementPackageText;
		wxStaticLine* m_staticline10;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* m_staticText202;
		wxStaticText* NoMovieFramesStaticText;
		NumericTextCtrl* LowResolutionLimitTextCtrl;
		wxStaticText* m_staticText196;
		NumericTextCtrl* MaskRadiusTextCtrl;
		wxStaticText* m_staticText331;
		NumericTextCtrl* InnerMaskRadiusTextCtrl;
		wxStaticText* m_staticText317;
		NumericTextCtrl* SignedCCResolutionTextCtrl;
		wxStaticText* DefocusSearchRangeStaticText;
		NumericTextCtrl* DefocusSearchRangeTextCtrl;
		wxStaticText* DefocusSearchStepStaticText;
		NumericTextCtrl* DefocusSearchStepTextCtrl;
		wxStaticText* m_staticText329;
		wxStaticText* m_staticText332;
		NumericTextCtrl* ScoreToWeightConstantTextCtrl;
		wxStaticText* m_staticText335;
		wxRadioButton* AdjustScoreForDefocusYesRadio;
		wxRadioButton* AdjustScoreForDefocusNoRadio;
		wxStaticText* m_staticText333;
		NumericTextCtrl* ReconstructionScoreThreshold;
		wxStaticText* m_staticText334;
		NumericTextCtrl* ReconstructionResolutionLimitTextCtrl;
		wxStaticText* m_staticText336;
		wxRadioButton* AutoCropYesRadioButton;
		wxRadioButton* AutoCropNoRadioButton;
		wxStaticText* m_staticText363;
		wxRadioButton* ApplyBlurringYesRadioButton;
		wxRadioButton* ApplyBlurringNoRadioButton;
		wxStaticText* SmoothingFactorStaticText;
		NumericTextCtrl* SmoothingFactorTextCtrl;
		wxStaticText* m_staticText405;
		wxStaticText* AutoMaskStaticText;
		wxRadioButton* AutoMaskYesRadioButton;
		wxRadioButton* AutoMaskNoRadioButton;
		wxStaticText* MaskEdgeStaticText;
		NumericTextCtrl* MaskEdgeTextCtrl;
		wxStaticText* MaskWeightStaticText;
		NumericTextCtrl* MaskWeightTextCtrl;
		wxStaticText* LowPassYesNoStaticText;
		wxRadioButton* LowPassMaskYesRadio;
		wxRadioButton* LowPassMaskNoRadio;
		wxStaticText* FilterResolutionStaticText;
		NumericTextCtrl* MaskFilterResolutionText;
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
		MemoryComboBox* RefinementRunProfileComboBox;
		wxStaticText* RunProfileText1;
		MemoryComboBox* ReconstructionRunProfileComboBox;
		wxButton* StartRefinementButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnUseMaskCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighResLimitChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnVolumeListItemActivated( wxListEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAutoMaskButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartRefinementClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		RefinementPackagePickerComboPanel* RefinementPackageComboBox;
		RefinementPickerComboPanel* InputParametersComboBox;
		DisplayCTFRefinementResultsPanel* ShowRefinementResultsPanel;

		RefineCTFPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1285,635 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RefineCTFPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class FindCTFPanel
///////////////////////////////////////////////////////////////////////////////
class FindCTFPanel : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline12;
		wxStaticText* m_staticText21;
		wxToggleButton* ExpertToggleButton;
		wxStaticLine* m_staticline10;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText202;
		wxStaticText* m_staticText186;
		wxRadioButton* MovieRadioButton;
		wxRadioButton* ImageRadioButton;
		wxStaticText* NoMovieFramesStaticText;
		wxSpinCtrl* NoFramesToAverageSpinCtrl;
		wxStaticText* m_staticText188;
		wxSpinCtrl* BoxSizeSpinCtrl;
		wxStaticText* m_staticText196;
		NumericTextCtrl* AmplitudeContrastNumericCtrl;
		wxStaticText* TiltStaticText;
		wxRadioButton* SearchTiltYesRadio;
		wxRadioButton* SearchTiltNoRadio;
		wxStaticText* m_staticText201;
		wxStaticText* m_staticText189;
		NumericTextCtrl* MinResNumericCtrl;
		wxStaticText* m_staticText190;
		NumericTextCtrl* MaxResNumericCtrl;
		wxStaticText* m_staticText191;
		NumericTextCtrl* LowDefocusNumericCtrl;
		wxStaticText* m_staticText192;
		NumericTextCtrl* HighDefocusNumericCtrl;
		wxStaticText* m_staticText194;
		NumericTextCtrl* DefocusStepNumericCtrl;
		wxCheckBox* LargeAstigmatismExpectedCheckBox;
		wxCheckBox* RestrainAstigmatismCheckBox;
		wxStaticText* ToleratedAstigmatismStaticText;
		NumericTextCtrl* ToleratedAstigmatismNumericCtrl;
		wxStaticText* m_staticText200;
		wxCheckBox* AdditionalPhaseShiftCheckBox;
		wxStaticText* MinPhaseShiftStaticText;
		NumericTextCtrl* MinPhaseShiftNumericCtrl;
		wxStaticText* MaxPhaseShiftStaticText;
		NumericTextCtrl* MaxPhaseShiftNumericCtrl;
		wxStaticText* PhaseShiftStepStaticText;
		NumericTextCtrl* PhaseShiftStepNumericCtrl;
		wxStaticText* m_staticText2001;
		wxCheckBox* FitNodesCheckBox;
		wxCheckBox* FitNodes1DCheckBox;
		wxCheckBox* FitNodes2DCheckBox;
		wxStaticText* FitNodesMinResStaticText;
		NumericTextCtrl* FitNodesMinResNumericCtrl;
		wxStaticText* FitNodesMaxResStaticText;
		NumericTextCtrl* FitNodesMaxResNumericCtrl;
		wxCheckBox* FitNodesRoundedSquareCheckBox;
		wxCheckBox* FitNodesWeightsCheckBox;
		wxStaticText* m_staticText20011;
		wxCheckBox* FilterLowresSignalCheckBox;
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
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMovieRadioButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImageRadioButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLargeAstigmatismExpectedCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRestrainAstigmatismCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFindAdditionalPhaseCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFitNodesCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartEstimationClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		ImageGroupPickerComboPanel* GroupComboBox;
		ShowCTFResultsPanel* CTFResultsPanel;

		FindCTFPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,731 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~FindCTFPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class DisplayCTFRefinementResultsPanelParent
///////////////////////////////////////////////////////////////////////////////
class DisplayCTFRefinementResultsPanelParent : public wxPanel
{
	private:

	protected:
		wxStaticText* OrthText;
		wxStaticLine* m_staticline109;

	public:
		wxSplitterWindow* LeftRightSplitter;
		wxPanel* LeftPanel;
		wxSplitterWindow* TopBottomSplitter;
		wxPanel* TopPanel;
		wxStaticText* DefocusChangeText;
		wxStaticLine* DefocusPlotLine;
		PlotCurvePanel* DefocusHistorgramPlotPanel;
		wxPanel* BottomPanel;
		MyFSCPanel* FSCResultsPanel;
		wxPanel* RightPanel;
		DisplayPanel* ShowOrthDisplayPanel;
		wxPanel* RoundPlotPanel;

		DisplayCTFRefinementResultsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 952,539 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~DisplayCTFRefinementResultsPanelParent();

		void LeftRightSplitterOnIdle( wxIdleEvent& )
		{
			LeftRightSplitter->SetSashPosition( 600 );
			LeftRightSplitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( DisplayCTFRefinementResultsPanelParent::LeftRightSplitterOnIdle ), NULL, this );
		}

		void TopBottomSplitterOnIdle( wxIdleEvent& )
		{
			TopBottomSplitter->SetSashPosition( 0 );
			TopBottomSplitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( DisplayCTFRefinementResultsPanelParent::TopBottomSplitterOnIdle ), NULL, this );
		}

};

