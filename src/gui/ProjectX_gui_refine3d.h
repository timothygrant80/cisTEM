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
class AngularDistributionPlotPanelHistogram;
class DisplayPanel;
class DisplayRefinementResultsPanel;
class MemoryComboBox;
class MyFSCPanel;
class NoFocusBitmapButton;
class NumericTextCtrl;
class PlotCurvePanel;
class ReferenceVolumesListControlRefinement;
class RefinementPackagePickerComboPanel;
class RefinementPickerComboPanel;
class VolumeAssetPickerComboPanel;

#include "job_panel.h"
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/panel.h>
#include <wx/sizer.h>
#include <wx/statline.h>
#include <wx/bmpbuttn.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/button.h>
#include <wx/splitter.h>
#include <wx/tglbtn.h>
#include <wx/aui/auibook.h>
#include <wx/checkbox.h>
#include <wx/radiobut.h>
#include <wx/spinctrl.h>
#include <wx/textctrl.h>
#include <wx/listctrl.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/combobox.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class RefinementResultsPanel
///////////////////////////////////////////////////////////////////////////////
class RefinementResultsPanel : public wxPanel
{
	private:

	protected:
		wxSplitterWindow* m_splitter7;
		wxPanel* m_panel49;
		wxStaticText* m_staticText284;
		wxStaticText* m_staticText285;
		wxSplitterWindow* m_splitter16;
		wxPanel* m_panel124;
		MyFSCPanel* FSCPlotPanel;
		wxPanel* m_panel125;
		wxStaticText* m_staticText708;
		wxStaticLine* m_staticline132;
		wxStaticLine* m_staticline131;
		AngularDistributionPlotPanelHistogram* AngularPlotPanel;
		wxPanel* RightPanel;
		wxToggleButton* JobDetailsToggleButton;
		wxStaticLine* m_staticline133;
		wxPanel* JobDetailsPanel;
		wxFlexGridSizer* InfoSizer;
		wxStaticText* m_staticText72;
		wxStaticText* RefinementIDStaticText;
		wxStaticText* m_staticText74;
		wxStaticText* DateOfRunStaticText;
		wxStaticText* m_staticText93;
		wxStaticText* TimeOfRunStaticText;
		wxStaticText* m_staticText785;
		wxStaticText* PercentUsedStaticText;
		wxStaticText* m_staticText83;
		wxStaticText* ReferenceVolumeIDStaticText;
		wxStaticText* m_staticText82;
		wxStaticText* ReferenceRefinementIDStaticText;
		wxStaticText* m_staticText85;
		wxStaticText* LowResLimitStaticText;
		wxStaticText* m_staticText87;
		wxStaticText* HighResLimitStaticText;
		wxStaticText* m_staticText89;
		wxStaticText* MaskRadiusStaticText;
		wxStaticText* m_staticText777;
		wxStaticText* SignedCCResLimitStaticText;
		wxStaticText* m_staticText779;
		wxStaticText* GlobalResLimitStaticText;
		wxStaticText* m_staticText781;
		wxStaticText* GlobalMaskRadiusStaticText;
		wxStaticText* m_staticText783;
		wxStaticText* NumberResultsRefinedStaticText;
		wxStaticText* m_staticText91;
		wxStaticText* AngularSearchStepStaticText;
		wxStaticText* m_staticText79;
		wxStaticText* SearchRangeXStaticText;
		wxStaticText* m_staticText99;
		wxStaticText* SearchRangeYStaticText;
		wxStaticText* m_staticText95;
		wxStaticText* ClassificationResLimitStaticText;
		wxStaticText* LargeAstigExpectedLabel;
		wxStaticText* ShouldFocusClassifyStaticText;
		wxStaticText* ToleratedAstigLabel;
		wxStaticText* SphereXCoordStaticText;
		wxStaticText* NumberOfAveragedFramesLabel;
		wxStaticText* SphereYCoordStaticText;
		wxStaticText* m_staticText787;
		wxStaticText* SphereZCoordStaticText;
		wxStaticText* m_staticText789;
		wxStaticText* SphereRadiusStaticText;
		wxStaticText* m_staticText791;
		wxStaticText* ShouldRefineCTFStaticText;
		wxStaticText* m_staticText793;
		wxStaticText* DefocusSearchRangeStaticText;
		wxStaticText* m_staticText795;
		wxStaticText* DefocusSearchStepStaticText;
		wxStaticText* m_staticText797;
		wxStaticText* ShouldAutoMaskStaticText;
		wxStaticText* m_staticText799;
		wxStaticText* RefineInputParamsStaticText;
		wxStaticText* m_staticText801;
		wxStaticText* UseSuppliedMaskStaticText;
		wxStaticText* m_staticText803;
		wxStaticText* MaskAssetIDStaticText;
		wxStaticText* m_staticText805;
		wxStaticText* MaskEdgeWidthStaticText;
		wxStaticText* m_staticText807;
		wxStaticText* MaskOutsideWeightStaticText;
		wxStaticText* m_staticText809;
		wxStaticText* ShouldFilterOutsideMaskStaticText;
		wxStaticText* m_staticText811;
		wxStaticText* MaskFilterResolutionStaticText;
		wxStaticText* m_staticText813;
		wxStaticText* ReconstructionIDStaticText;
		wxStaticText* m_staticText815;
		wxStaticText* InnerMaskRadiusStaticText;
		wxStaticText* m_staticText817;
		wxStaticText* OuterMaskRadiusStaticText;
		wxStaticText* m_staticText820;
		wxStaticText* ResolutionCutOffStaticText;
		wxStaticText* Score;
		wxStaticText* ScoreWeightConstantStaticText;
		wxStaticText* m_staticText823;
		wxStaticText* AdjustScoresStaticText;
		wxStaticText* m_staticText825;
		wxStaticText* ShouldCropImagesStaticText;
		wxStaticText* m_staticText827;
		wxStaticText* ShouldLikelihoodBlurStaticText;
		wxStaticText* m_staticText829;
		wxStaticText* SmoothingFactorStaticText;
		wxStaticLine* m_staticline30;
		DisplayPanel* OrthPanel;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void PopupParametersClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AngularPlotPopupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }


	public:
		RefinementPackagePickerComboPanel* RefinementPackageComboBox;
		RefinementPickerComboPanel* InputParametersComboBox;
		NoFocusBitmapButton* ParametersDetailButton;
		NoFocusBitmapButton* AngularPlotDetailsButton;

		RefinementResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1007,587 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RefinementResultsPanel();

		void m_splitter7OnIdle( wxIdleEvent& )
		{
			m_splitter7->SetSashPosition( 900 );
			m_splitter7->Disconnect( wxEVT_IDLE, wxIdleEventHandler( RefinementResultsPanel::m_splitter7OnIdle ), NULL, this );
		}

		void m_splitter16OnIdle( wxIdleEvent& )
		{
			m_splitter16->SetSashPosition( 350 );
			m_splitter16->Disconnect( wxEVT_IDLE, wxIdleEventHandler( RefinementResultsPanel::m_splitter16OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class DisplayRefinementResultsPanelParent
///////////////////////////////////////////////////////////////////////////////
class DisplayRefinementResultsPanelParent : public wxPanel
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
		wxStaticText* AngularPlotText;
		wxStaticLine* AngularPlotLine;
		AngularDistributionPlotPanel* AngularPlotPanel;
		wxPanel* BottomPanel;
		MyFSCPanel* FSCResultsPanel;
		wxPanel* RightPanel;
		DisplayPanel* ShowOrthDisplayPanel;
		wxPanel* RoundPlotPanel;

		DisplayRefinementResultsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 952,539 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~DisplayRefinementResultsPanelParent();

		void LeftRightSplitterOnIdle( wxIdleEvent& )
		{
			LeftRightSplitter->SetSashPosition( 600 );
			LeftRightSplitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( DisplayRefinementResultsPanelParent::LeftRightSplitterOnIdle ), NULL, this );
		}

		void TopBottomSplitterOnIdle( wxIdleEvent& )
		{
			TopBottomSplitter->SetSashPosition( 0 );
			TopBottomSplitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( DisplayRefinementResultsPanelParent::TopBottomSplitterOnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassificationPlotPanelParent
///////////////////////////////////////////////////////////////////////////////
class ClassificationPlotPanelParent : public wxPanel
{
	private:

	protected:
		wxPanel* SigmaPanel;
		wxPanel* LikelihoodPanel;
		wxPanel* MobilityPanel;

	public:
		wxAuiNotebook* my_notebook;

		ClassificationPlotPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ClassificationPlotPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class Refine3DPanel
///////////////////////////////////////////////////////////////////////////////
class Refine3DPanel : public JobPanel
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
		wxRadioButton* LocalRefinementRadio;
		wxRadioButton* GlobalRefinementRadio;
		wxStaticText* NoCycleStaticText;
		wxSpinCtrl* NumberRoundsSpinCtrl;
		wxStaticText* HiResLimitStaticText;
		NumericTextCtrl* HighResolutionLimitTextCtrl;
		wxToggleButton* ExpertToggleButton;
		wxStaticLine* m_staticline101;
		ReferenceVolumesListControlRefinement* Active3DReferencesListCtrl;
		wxStaticText* PleaseCreateRefinementPackageText;
		wxStaticLine* m_staticline10;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText318;
		wxButton* ResetAllDefaultsButton;
		wxCheckBox* RefinePsiCheckBox;
		wxCheckBox* RefineThetaCheckBox;
		wxCheckBox* RefinePhiCheckBox;
		wxCheckBox* RefineXShiftCheckBox;
		wxCheckBox* RefineYShiftCheckBox;
		wxCheckBox* RefineOccupanciesCheckBox;
		wxFlexGridSizer* fgSizer1;
		wxStaticText* m_staticText202;
		wxStaticText* NoMovieFramesStaticText;
		NumericTextCtrl* LowResolutionLimitTextCtrl;
		wxStaticText* m_staticText196;
		NumericTextCtrl* MaskRadiusTextCtrl;
		wxStaticText* m_staticText331;
		NumericTextCtrl* InnerMaskRadiusTextCtrl;
		wxStaticText* m_staticText317;
		NumericTextCtrl* SignedCCResolutionTextCtrl;
		wxStaticText* m_staticText362;
		NumericTextCtrl* PercentUsedTextCtrl;
		wxStaticText* m_staticText201;
		wxStaticText* GlobalMaskRadiusStaticText;
		NumericTextCtrl* GlobalMaskRadiusTextCtrl;
		wxStaticText* NumberToRefineStaticText;
		wxSpinCtrl* NumberToRefineSpinCtrl;
		wxStaticText* AlsoRefineInputStaticText1;
		wxRadioButton* AlsoRefineInputYesRadio;
		wxRadioButton* AlsoRefineInputNoRadio;
		wxStaticText* AngularStepStaticText;
		NumericTextCtrl* AngularStepTextCtrl;
		wxStaticText* SearchRangeXStaticText;
		NumericTextCtrl* SearchRangeXTextCtrl;
		wxStaticText* SearchRangeYStaticText;
		NumericTextCtrl* SearchRangeYTextCtrl;
		wxStaticText* m_staticText200;
		wxStaticText* MinPhaseShiftStaticText;
		NumericTextCtrl* ClassificationHighResLimitTextCtrl;
		wxStaticText* PhaseShiftStepStaticText;
		wxRadioButton* SphereClassificatonYesRadio;
		wxRadioButton* SphereClassificatonNoRadio;
		wxStaticText* SphereXStaticText;
		NumericTextCtrl* SphereXTextCtrl;
		wxStaticText* SphereYStaticText;
		NumericTextCtrl* SphereYTextCtrl;
		wxStaticText* SphereZStaticText;
		NumericTextCtrl* SphereZTextCtrl;
		wxStaticText* SphereRadiusStaticText;
		NumericTextCtrl* SphereRadiusTextCtrl;
		wxStaticText* m_staticText323;
		wxStaticText* m_staticText324;
		wxRadioButton* RefineCTFYesRadio;
		wxRadioButton* RefineCTFNoRadio;
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
		wxStaticText* AutoCenterStaticText;
		wxRadioButton* AutoCenterYesRadioButton;
		wxRadioButton* AutoCenterNoRadioButton;
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
		DisplayRefinementResultsPanel* ShowRefinementResultsPanel;

		Refine3DPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1285,635 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~Refine3DPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class Sharpen3DPanelParent
///////////////////////////////////////////////////////////////////////////////
class Sharpen3DPanelParent : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText262;
		wxCheckBox* UseMaskCheckBox;
		VolumeAssetPickerComboPanel* MaskSelectPanel;
		wxStaticLine* m_staticline129;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText1006;
		wxStaticText* m_staticText1007;
		NumericTextCtrl* FlattenFromTextCtrl;
		wxStaticText* m_staticText10081;
		NumericTextCtrl* CutOffResTextCtrl;
		wxStaticText* m_staticText638;
		NumericTextCtrl* AdditionalLowBFactorTextCtrl;
		wxStaticText* m_staticText600;
		NumericTextCtrl* AdditionalHighBFactorTextCtrl;
		wxStaticText* m_staticText10111;
		NumericTextCtrl* FilterEdgeWidthTextCtrl;
		wxStaticText* UseFSCWeightingStaticText;
		wxRadioButton* UseFSCWeightingYesButton;
		wxRadioButton* UseFSCWeightingNoButton;
		wxStaticText* SSNRScaleFactorText;
		NumericTextCtrl* SSNRScaleFactorTextCtrl;
		wxStaticLine* m_staticline136;
		wxStaticText* m_staticText202;
		wxStaticText* InnerMaskRadiusStaticText;
		NumericTextCtrl* InnerMaskRadiusTextCtrl;
		wxStaticText* OuterMaskRadiusStaticText;
		NumericTextCtrl* OuterMaskRadiusTextCtrl;
		wxStaticText* UseAutoMaskingStaticText;
		wxRadioButton* UseAutoMaskingYesButton;
		wxRadioButton* UseAutoMaskingNoButton;
		wxStaticText* m_staticText671;
		wxStaticText* m_staticText642;
		wxRadioButton* InvertHandednessYesButton;
		wxRadioButton* InvertHandednessNoButton;
		wxStaticText* m_staticText6721;
		wxRadioButton* CorrectGriddingYesButton;
		wxRadioButton* CorrectGriddingNoButton;
		wxStaticLine* m_staticline138;
		wxStaticText* m_staticText699;
		wxStaticLine* m_staticline137;
		PlotCurvePanel* GuinierPlot;
		wxStaticLine* m_staticline135;
		DisplayPanel* ResultDisplayPanel;
		wxPanel* InfoPanel;
		wxRichTextCtrl* InfoText;
		wxStaticLine* m_staticline130;
		wxButton* RunJobButton;
		wxGauge* ProgressGuage;
		wxButton* ImportResultButton;
		wxButton* SaveResultButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnUseMaskCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAutoMaskButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void OnRunButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImportResultClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveResultClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		VolumeAssetPickerComboPanel* VolumeComboBox;

		Sharpen3DPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1124,1252 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~Sharpen3DPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class Generate3DPanelParent
///////////////////////////////////////////////////////////////////////////////
class Generate3DPanelParent : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline12;
		wxPanel* InputParamsPanel;
		wxStaticText* m_staticText262;
		wxStaticText* m_staticText263;
		wxToggleButton* ExpertToggleButton;
		wxStaticText* PleaseCreateRefinementPackageText;
		wxStaticLine* m_staticline10;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText329;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* m_staticText196;
		NumericTextCtrl* MaskRadiusTextCtrl;
		wxStaticText* m_staticText331;
		NumericTextCtrl* InnerMaskRadiusTextCtrl;
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
		wxStaticText* m_staticText628;
		wxRadioButton* SaveHalfMapsYesButton;
		wxRadioButton* SaveHalfMapsNoButton;
		wxStaticText* m_staticText631;
		wxRadioButton* OverwriteStatisticsYesButton;
		wxRadioButton* OverwriteStatisticsNoButton;
		wxStaticText* m_staticText879;
		wxRadioButton* ApplyEwaldSphereCorrectionYesButton;
		wxRadioButton* ApplyEwaldSphereCorrectionNoButton;
		wxStaticText* ApplyInverseHandLabelText;
		wxRadioButton* ApplyEwaldInverseHandYesButton;
		wxRadioButton* ApplyEwaldInverseHandNoButton;
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
		wxStaticText* RunProfileText1;
		MemoryComboBox* ReconstructionRunProfileComboBox;
		wxButton* StartReconstructionButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartReconstructionClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		RefinementPackagePickerComboPanel* RefinementPackageComboBox;
		RefinementPickerComboPanel* InputParametersComboBox;
		DisplayRefinementResultsPanel* ShowRefinementResultsPanel;

		Generate3DPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1285,635 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~Generate3DPanelParent();

};

