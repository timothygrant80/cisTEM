///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.1-0-g8feb16b)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class AbInitioPlotPanel;
class AngularDistributionPlotPanel;
class AngularDistributionPlotPanelHistogram;
class AssetPickerListCtrl;
class AutoWrapStaticText;
class BitmapPanel;
class CTF1DPanel;
class ClassSelectionPickerComboPanel;
class ClassificationPickerComboPanel;
class ClassificationPlotPanel;
class ClassificationSelectionListCtrl;
class ContainedParticleListControl;
class ContentsList;
class DisplayCTFRefinementResultsPanel;
class DisplayPanel;
class DisplayRefinementResultsPanel;
class ImageGroupPickerComboPanel;
class ImagesPickerComboPanel;
class MemoryComboBox;
class MovieGroupPickerComboPanel;
class MyFSCPanel;
class NoFocusBitmapButton;
class NumericTextCtrl;
class PickingBitmapPanel;
class PickingResultsDisplayPanel;
class PlotCurvePanel;
class PlotFSCPanel;
class ReferenceVolumesListControl;
class ReferenceVolumesListControlRefinement;
class RefinementPackageListControl;
class RefinementPackagePickerComboPanel;
class RefinementParametersListCtrl;
class RefinementPickerComboPanel;
class ResultsDataViewListCtrl;
class ShowCTFResultsPanel;
class ShowTemplateMatchResultsPanel;
class UnblurResultsPanel;
class VolumeAssetPickerComboPanel;

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
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/menu.h>
#include <wx/frame.h>
#include <wx/combobox.h>
#include <wx/bmpbuttn.h>
#include <wx/button.h>
#include <wx/statline.h>
#include <wx/stattext.h>
#include <wx/radiobut.h>
#include <wx/spinctrl.h>
#include <wx/tglbtn.h>
#include <wx/textctrl.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/splitter.h>
#include <wx/checkbox.h>
#include <wx/dataview.h>
#include <wx/choice.h>
#include <wx/dialog.h>
#include <wx/filepicker.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );
#include <wx/statbox.h>
#include <wx/toolbar.h>
#include <wx/aui/auibook.h>
#include <wx/statbmp.h>
#include <wx/hyperlink.h>

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
		wxMenuItem* WorkflowSingleParticle;
		wxMenuItem* WorkflowTemplateMatching;
		wxMenu* HelpMenu;

		// Virtual event handlers, override them in your derived class
		virtual void OnMenuBookChange( wxListbookEvent& event ) { event.Skip(); }
		virtual void OnFileMenuUpdate( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnFileNewProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileOpenProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileCloseProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileExit( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSingleParticleWorkflow( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTemplateMatchingWorkflow( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHelpLaunch( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAboutLaunch( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxListbook* MenuBook;

		MainFrame( wxWindow* parent, wxWindowID id = wxID_OPEN, const wxString& title = wxT("cisTEM"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1366,768 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL|wxWANTS_CHARS );

		~MainFrame();

};

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
		MemoryComboBox* AssetComboBox;
		NoFocusBitmapButton* PreviousButton;
		NoFocusBitmapButton* NextButton;
		NoFocusBitmapButton* WindowSelectButton;

		AssetPickerComboPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~AssetPickerComboPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class AbInitio3DPanelParent
///////////////////////////////////////////////////////////////////////////////
class AbInitio3DPanelParent : public JobPanel
{
	private:

	protected:
		wxStaticLine* TopStaticLine;
		wxStaticText* ImageOrClassAverageStaticText;
		wxRadioButton* ImageInputRadioButton;
		wxRadioButton* ClassAverageInputRadioButton;
		wxStaticLine* BottomStaticLine;
		wxPanel* InputParamsPanel;
		wxStaticText* InputRefinementPackageStaticText;
		wxStaticText* InputClassificationSelectionStaticText;
		wxStaticLine* m_staticline52;
		wxStaticText* NoClassesStaticText;
		wxSpinCtrl* NumberClassesSpinCtrl;
		wxStaticText* m_staticText415;
		wxSpinCtrl* NumberStartsSpinCtrl;
		wxStaticLine* m_staticline144;
		wxStaticText* SymmetryStaticText;
		wxComboBox* SymmetryComboBox;
		wxStaticText* m_staticText264;
		wxSpinCtrl* NumberRoundsSpinCtrl;
		wxToggleButton* ExpertToggleButton;
		wxStaticText* PleaseCreateRefinementPackageText;
		wxStaticLine* m_staticline141;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText531;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* NoMovieFramesStaticText;
		NumericTextCtrl* InitialResolutionLimitTextCtrl;
		wxStaticText* m_staticText196;
		NumericTextCtrl* FinalResolutionLimitTextCtrl;
		wxStaticText* GlobalMaskRadiusStaticText;
		NumericTextCtrl* GlobalMaskRadiusTextCtrl;
		wxStaticText* InnerMaskRadiusStaticText1;
		NumericTextCtrl* InnerMaskRadiusTextCtrl;
		wxStaticText* SearchRangeXStaticText;
		NumericTextCtrl* SearchRangeXTextCtrl;
		wxStaticText* SearchRangeYStaticText;
		NumericTextCtrl* SearchRangeYTextCtrl;
		wxStaticText* UseAutoMaskingStaticText;
		wxRadioButton* AutoMaskYesRadio;
		wxRadioButton* AutoMaskNoRadio;
		wxStaticText* m_staticText4151;
		wxRadioButton* AutoPercentUsedYesRadio;
		wxRadioButton* AutoPercentUsedNoRadio;
		wxStaticText* InitialPercentUsedStaticText;
		NumericTextCtrl* StartPercentUsedTextCtrl;
		wxStaticText* FinalPercentUsedStaticText;
		NumericTextCtrl* EndPercentUsedTextCtrl;
		wxStaticText* AlwaysApplySymmetryStaticText;
		wxRadioButton* AlwaysApplySymmetryYesButton;
		wxRadioButton* AlwaysApplySymmetryNoButton;
		wxStaticText* m_staticText532;
		wxStaticText* m_staticText363;
		wxRadioButton* ApplyBlurringYesRadioButton;
		wxRadioButton* ApplyBlurringNoRadioButton;
		wxStaticText* SmoothingFactorStaticText;
		NumericTextCtrl* SmoothingFactorTextCtrl;
		wxStaticText* m_staticText662;
		wxStaticText* m_staticText663;
		wxSpinCtrl* ImagesPerClassSpinCtrl;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
		AbInitioPlotPanel* PlotPanel;
		wxPanel* InfoPanel;
		wxRichTextCtrl* InfoText;
		wxStaticText* m_staticText377;
		wxStaticLine* m_staticline91;
		wxStaticLine* m_staticline11;
		wxPanel* ProgressPanel;
		wxStaticText* NumberConnectedText;
		wxGauge* ProgressBar;
		wxStaticText* TimeRemainingText;
		wxStaticLine* m_staticline60;
		wxButton* FinishButton;
		wxButton* CancelAlignmentButton;
		wxStaticLine* CurrentLineOne;
		wxButton* TakeCurrentResultButton;
		wxStaticLine* CurrentLineTwo;
		wxButton* TakeLastStartResultButton;
		wxPanel* StartPanel;
		wxStaticText* RunProfileText;
		MemoryComboBox* RefinementRunProfileComboBox;
		wxStaticText* RunProfileText1;
		MemoryComboBox* ReconstructionRunProfileComboBox;
		wxButton* StartRefinementButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnMethodChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TakeCurrentClicked( wxCommandEvent& event ) { event.Skip(); }
		virtual void TakeLastStartClicked( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartRefinementClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		RefinementPackagePickerComboPanel* RefinementPackageComboBox;
		ClassSelectionPickerComboPanel* ClassSelectionComboBox;
		wxPanel* OrthResultsPanel;
		DisplayPanel* ShowOrthDisplayPanel;

		AbInitio3DPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1631,686 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~AbInitio3DPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class Refine2DPanel
///////////////////////////////////////////////////////////////////////////////
class Refine2DPanel : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline12;
		wxPanel* InputParamsPanel;
		wxStaticText* m_staticText262;
		wxStaticText* m_staticText263;
		wxStaticLine* m_staticline57;
		wxStaticText* m_staticText3321;
		wxSpinCtrl* NumberClassesSpinCtrl;
		wxStaticText* m_staticText264;
		wxSpinCtrl* NumberRoundsSpinCtrl;
		wxToggleButton* ExpertToggleButton;
		wxStaticText* PleaseCreateRefinementPackageText;
		wxStaticLine* m_staticline10;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
		ClassificationPlotPanel* PlotPanel;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText318;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* NoMovieFramesStaticText;
		NumericTextCtrl* LowResolutionLimitTextCtrl;
		wxStaticText* m_staticText188;
		NumericTextCtrl* HighResolutionLimitStartTextCtrl;
		wxStaticText* m_staticText1881;
		NumericTextCtrl* HighResolutionLimitFinishTextCtrl;
		wxStaticText* m_staticText196;
		NumericTextCtrl* MaskRadiusTextCtrl;
		wxStaticText* AngularStepStaticText;
		NumericTextCtrl* AngularStepTextCtrl;
		wxStaticText* SearchRangeXStaticText;
		NumericTextCtrl* MaxSearchRangeTextCtrl;
		wxStaticText* m_staticText330;
		NumericTextCtrl* SmoothingFactorTextCtrl;
		wxStaticText* PhaseShiftStepStaticText;
		wxRadioButton* ExcludeBlankEdgesYesRadio;
		wxRadioButton* ExcludeBlankEdgesNoRadio;
		wxStaticText* m_staticText334;
		wxRadioButton* AutoPercentUsedRadioYes;
		wxRadioButton* SpherAutoPercentUsedRadioNo;
		wxStaticText* PercentUsedStaticText;
		NumericTextCtrl* PercentUsedTextCtrl;
		wxStaticText* m_staticText650;
		wxRadioButton* AutoMaskRadioYes;
		wxRadioButton* AutoMaskRadioNo;
		wxStaticText* m_staticText651;
		wxRadioButton* AutoCentreRadioYes;
		wxRadioButton* AutoCentreRadioNo;
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
		wxButton* StartRefinementButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighResLimitChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartClassificationClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		RefinementPackagePickerComboPanel* RefinementPackageComboBox;
		ClassificationPickerComboPanel* InputParametersComboBox;
		DisplayPanel* ResultDisplayPanel;

		Refine2DPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1363,691 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~Refine2DPanel();

};

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
/// Class Refine2DResultsPanelParent
///////////////////////////////////////////////////////////////////////////////
class Refine2DResultsPanelParent : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline67;
		wxSplitterWindow* m_splitter7;
		wxPanel* LeftPanel;
		wxStaticText* m_staticText284;
		wxStaticText* m_staticText285;
		wxStaticLine* m_staticline66;
		wxPanel* JobDetailsPanel;
		wxFlexGridSizer* InfoSizer;
		wxStaticText* m_staticText72;
		wxStaticText* ClassificationIDStaticText;
		wxStaticText* m_staticText74;
		wxStaticText* DateOfRunStaticText;
		wxStaticText* m_staticText93;
		wxStaticText* TimeOfRunStaticText;
		wxStaticText* m_staticText83;
		wxStaticText* RefinementPackageIDStaticText;
		wxStaticText* m_staticText82;
		wxStaticText* StartClassificationIDStaticText;
		wxStaticText* m_staticText78;
		wxStaticText* NumberClassesStaticText;
		wxStaticText* m_staticText96;
		wxStaticText* NumberParticlesStaticText;
		wxStaticText* m_staticText85;
		wxStaticText* LowResLimitStaticText;
		wxStaticText* m_staticText87;
		wxStaticText* HighResLimitStaticText;
		wxStaticText* m_staticText89;
		wxStaticText* MaskRadiusStaticText;
		wxStaticText* m_staticText91;
		wxStaticText* AngularSearchStepStaticText;
		wxStaticText* m_staticText79;
		wxStaticText* SearchRangeXStaticText;
		wxStaticText* m_staticText95;
		wxStaticText* SmoothingFactorStaticText;
		wxStaticText* LargeAstigExpectedLabel;
		wxStaticText* ExcludeBlankEdgesStaticText;
		wxStaticText* m_staticText99;
		wxStaticText* SearchRangeYStaticText;
		wxStaticText* ToleratedAstigLabel;
		wxStaticText* AutoPercentUsedStaticText;
		wxStaticText* NumberOfAveragedFramesLabel;
		wxStaticText* PercentUsedStaticText;
		wxStaticLine* m_staticline30;
		DisplayPanel* ClassumDisplayPanel;
		wxPanel* m_panel49;
		wxStaticText* m_staticText321;
		wxToggleButton* JobDetailsToggleButton;
		wxStaticLine* m_staticline671;
		wxPanel* SelectionPanel;
		ClassificationSelectionListCtrl* SelectionManagerListCtrl;
		wxButton* AddButton;
		wxButton* DeleteButton;
		wxButton* RenameButton;
		wxButton* CopyOtherButton;
		wxStaticLine* m_staticline64;
		wxButton* ClearButton;
		wxButton* InvertButton;
		wxStaticText* ClassNumberStaticText;
		wxStaticLine* m_staticline65;
		DisplayPanel* ParticleDisplayPanel;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnBeginLabelEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnEndLabelEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnActivated( wxListEvent& event ) { event.Skip(); }
		virtual void OnDeselected( wxListEvent& event ) { event.Skip(); }
		virtual void OnSelected( wxListEvent& event ) { event.Skip(); }
		virtual void OnAddButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDeleteButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCopyOtherButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnClearButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInvertButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		RefinementPackagePickerComboPanel* RefinementPackageComboBox;
		ClassificationPickerComboPanel* InputParametersComboBox;

		Refine2DResultsPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1269,471 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~Refine2DResultsPanelParent();

		void m_splitter7OnIdle( wxIdleEvent& )
		{
			m_splitter7->SetSashPosition( -800 );
			m_splitter7->Disconnect( wxEVT_IDLE, wxIdleEventHandler( Refine2DResultsPanelParent::m_splitter7OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class PickingResultsDisplayPanelParent
///////////////////////////////////////////////////////////////////////////////
class PickingResultsDisplayPanelParent : public wxPanel
{
	private:

	protected:
		wxCheckBox* CirclesAroundParticlesCheckBox;
		wxCheckBox* ScaleBarCheckBox;
		wxCheckBox* HighPassFilterCheckBox;
		wxCheckBox* LowPassFilterCheckBox;
		NumericTextCtrl* LowResFilterTextCtrl;
		wxStaticText* LowAngstromStatic;
		wxCheckBox* WienerFilterCheckBox;
		wxStaticLine* m_staticline831;
		wxStaticText* ImageIDStaticText;
		wxStaticText* DefocusStaticText;
		wxStaticText* IcinessStaticText;
		wxStaticText* NumberOfPicksStaticText;
		wxStaticLine* m_staticline8311;
		wxStaticLine* m_staticline26;

		// Virtual event handlers, override them in your derived class
		virtual void OnCirclesAroundParticlesCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnScaleBarCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighPassFilterCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLowPassFilterCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLowPassKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnLowPassEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnWienerFilterCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnUndoButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRedoButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		PickingBitmapPanel* PickingResultsImagePanel;
		wxButton* UndoButton;
		wxButton* RedoButton;

		PickingResultsDisplayPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1123,360 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~PickingResultsDisplayPanelParent();

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
		wxStaticText* IcinessStaticText;
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
/// Class PickingResultsPanel
///////////////////////////////////////////////////////////////////////////////
class PickingResultsPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline25;
		wxSplitterWindow* m_splitter4;
		wxPanel* m_panel13;
		wxRadioButton* AllImagesButton;
		wxRadioButton* ByFilterButton;
		wxButton* FilterButton;
		wxStaticLine* m_staticline85;
		wxToggleButton* JobDetailsToggleButton;
		ResultsDataViewListCtrl* ResultDataView;
		wxButton* PreviousButton;
		wxButton* NextButton;
		wxPanel* RightPanel;
		wxPanel* JobDetailsPanel;
		wxFlexGridSizer* InfoSizer;
		wxStaticText* m_staticText72;
		wxStaticText* PickIDStaticText;
		wxStaticText* m_staticText74;
		wxStaticText* DateOfRunStaticText;
		wxStaticText* m_staticText93;
		wxStaticText* TimeOfRunStaticText;
		wxStaticText* m_staticText83;
		wxStaticText* AlgorithmStaticText;
		wxStaticText* m_staticText82;
		wxStaticText* ManualEditStaticText;
		wxStaticText* m_staticText78;
		wxStaticText* ThresholdStaticText;
		wxStaticText* m_staticText96;
		wxStaticText* MaximumRadiusStaticText;
		wxStaticText* m_staticText85;
		wxStaticText* CharacteristicRadiusStaticText;
		wxStaticText* m_staticText87;
		wxStaticText* HighestResStaticText;
		wxStaticText* m_staticText89;
		wxStaticText* MinEdgeDistStaticText;
		wxStaticText* m_staticText91;
		wxStaticText* AvoidHighVarStaticText;
		wxStaticText* m_staticText79;
		wxStaticText* AvoidHighLowMeanStaticText;
		wxStaticText* m_staticText95;
		wxStaticText* NumBackgroundBoxesStaticText;
		wxStaticLine* m_staticline30;
		wxStaticText* m_staticText469;
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
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRemoveFromGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddToGroupClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		PickingResultsDisplayPanel* ResultDisplayPanel;

		PickingResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1309,557 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~PickingResultsPanel();

		void m_splitter4OnIdle( wxIdleEvent& )
		{
			m_splitter4->SetSashPosition( 450 );
			m_splitter4->Disconnect( wxEVT_IDLE, wxIdleEventHandler( PickingResultsPanel::m_splitter4OnIdle ), NULL, this );
		}

};

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
/// Class FindParticlesPanel
///////////////////////////////////////////////////////////////////////////////
class FindParticlesPanel : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline12;
		wxStaticText* m_staticText21;
		wxStaticText* PickingAlgorithStaticText;
		wxComboBox* PickingAlgorithmComboBox;
		wxToggleButton* ExpertToggleButton;
		wxStaticText* PleaseEstimateCTFStaticText;
		wxStaticLine* m_staticline10;
		wxSplitterWindow* FindParticlesSplitterWindow;
		wxPanel* LeftPanel;
		wxScrolledWindow* PickingParametersPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText196;
		NumericTextCtrl* ExclusionRadiusNumericCtrl;
		wxStaticText* CharacteristicParticleRadiusStaticText;
		NumericTextCtrl* TemplateRadiusNumericCtrl;
		wxStaticText* ThresholdPeakHeightStaticText1;
		NumericTextCtrl* ThresholdPeakHeightNumericCtrl;
		wxCheckBox* AvoidLowVarianceAreasCheckBox;
		NumericTextCtrl* LowVarianceThresholdNumericCtrl;
		wxCheckBox* AvoidHighVarianceAreasCheckBox;
		NumericTextCtrl* HighVarianceThresholdNumericCtrl;
		wxStaticLine* m_staticline106;
		wxStaticText* m_staticText440;
		wxButton* TestOnCurrentMicrographButton;
		wxCheckBox* AutoPickRefreshCheckBox;
		wxPanel* ExpertOptionsPanel;
		wxBoxSizer* ExpertInputSizer;
		wxStaticLine* m_staticline35;
		wxStaticText* ExpertOptionsStaticText;
		wxStaticText* HighestResolutionStaticText;
		NumericTextCtrl* HighestResolutionNumericCtrl;
		wxCheckBox* SetMinimumDistanceFromEdgesCheckBox;
		wxSpinCtrl* MinimumDistanceFromEdgesSpinCtrl;
		wxCheckBox* m_checkBox9;
		wxSpinCtrl* NumberOfTemplateRotationsSpinCtrl;
		wxCheckBox* AvoidAbnormalLocalMeanAreasCheckBox;
		wxStaticText* m_staticText170;
		wxSpinCtrl* NumberOfBackgroundBoxesSpinCtrl;
		wxStaticText* m_staticText169;
		wxChoice* AlgorithmToFindBackgroundChoice;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
		wxPanel* RightPanel;
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
		wxButton* StartPickingButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnPickingAlgorithmComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExclusionRadiusNumericTextKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnExclusionRadiusNumericTextSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnExclusionRadiusNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTemplateRadiusNumericTextKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnTemplateRadiusNumericTextSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnTemplateRadiusNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnThresholdPeakHeightNumericTextKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnThresholdPeakHeightNumericTextSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnThresholdPeakHeightNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAvoidLowVarianceAreasCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLowVarianceThresholdNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAvoidHighVarianceAreasCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighVarianceThresholdNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTestOnCurrentMicrographButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAutoPickRefreshCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighestResolutionNumericKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnHighestResolutionNumericSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnHighestResolutionNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSetMinimumDistanceFromEdgesCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMinimumDistanceFromEdgesSpinCtrl( wxSpinEvent& event ) { event.Skip(); }
		virtual void OnAvoidAbnormalLocalMeanAreasCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNumberOfBackgroundBoxesSpinCtrl( wxSpinEvent& event ) { event.Skip(); }
		virtual void OnAlgorithmToFindBackgroundChoice( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartPickingClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		ImageGroupPickerComboPanel* GroupComboBox;
		ImagesPickerComboPanel* ImageComboBox;
		PickingResultsDisplayPanel* PickingResultsPanel;

		FindParticlesPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,731 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~FindParticlesPanel();

		void FindParticlesSplitterWindowOnIdle( wxIdleEvent& )
		{
			FindParticlesSplitterWindow->SetSashPosition( 350 );
			FindParticlesSplitterWindow->Disconnect( wxEVT_IDLE, wxIdleEventHandler( FindParticlesPanel::FindParticlesSplitterWindowOnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class VolumeImportDialog
///////////////////////////////////////////////////////////////////////////////
class VolumeImportDialog : public wxDialog
{
	private:

	protected:
		wxListCtrl* PathListCtrl;
		wxButton* m_button10;
		wxButton* m_button11;
		wxButton* ClearButton;
		wxStaticLine* m_staticline7;
		wxStaticText* m_staticText20;
		wxTextCtrl* PixelSizeText;
		wxStaticLine* m_staticline8;
		wxButton* m_button13;
		wxButton* ImportButton;

		// Virtual event handlers, override them in your derived class
		virtual void AddFilesClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddDirectoryClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClearClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTextKeyPress( wxKeyEvent& event ) { event.Skip(); }
		virtual void TextChanged( wxCommandEvent& event ) { event.Skip(); }
		virtual void CancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ImportClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		VolumeImportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Import Images"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 484,539 ), long style = wxCLOSE_BOX );

		~VolumeImportDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class AtomicCoordinatesImportDialogParent
///////////////////////////////////////////////////////////////////////////////
class AtomicCoordinatesImportDialogParent : public wxDialog
{
	private:

	protected:
		wxListCtrl* PathListCtrl;
		wxButton* m_button10;
		wxButton* m_button11;
		wxButton* ClearButton;
		wxStaticLine* m_staticline7;
		wxStaticLine* m_staticline8;
		wxButton* m_button13;
		wxButton* ImportButton;

		// Virtual event handlers, override them in your derived class
		virtual void AddFilesClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddDirectoryClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClearClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void CancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ImportClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		AtomicCoordinatesImportDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Import PDBx/mmCIF"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 484,539 ), long style = wxCLOSE_BOX );

		~AtomicCoordinatesImportDialogParent();

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
		wxButton* s;
		wxStaticLine* m_staticline7;
		wxStaticText* m_staticText19;
		wxComboBox* VoltageCombo;
		wxStaticText* m_staticText21;
		wxTextCtrl* CsText;
		wxStaticText* EerSuperResFactorStaticText;
		wxChoice* EerSuperResFactorChoice;
		wxStaticText* PixelSizeStaticText;
		wxTextCtrl* PixelSizeText;
		wxStaticText* EerNumberOfFramesStaticText;
		wxSpinCtrl* EerNumberOfFramesSpinCtrl;
		wxStaticText* ExposurePerFrameStaticText;
		wxTextCtrl* DoseText;
		wxCheckBox* ApplyDarkImageCheckbox;
		wxFilePickerCtrl* DarkFilePicker;
		wxCheckBox* ApplyGainImageCheckbox;
		wxFilePickerCtrl* GainFilePicker;
		wxCheckBox* ResampleMoviesCheckBox;
		wxStaticText* DesiredPixelSizeStaticText;
		NumericTextCtrl* DesiredPixelSizeTextCtrl;
		wxCheckBox* CorrectMagDistortionCheckBox;
		wxStaticText* DistortionAngleStaticText;
		NumericTextCtrl* DistortionAngleTextCtrl;
		wxStaticText* MajorScaleStaticText;
		NumericTextCtrl* MajorScaleTextCtrl;
		wxStaticText* MinorScaleStaticText;
		NumericTextCtrl* MinorScaleTextCtrl;
		wxCheckBox* MoviesHaveInvertedContrast;
		wxCheckBox* SkipFullIntegrityCheck;
		wxCheckBox* ImportMetadataCheckbox;
		wxStaticLine* m_staticline8;
		wxButton* m_button13;
		wxButton* ImportButton;

		// Virtual event handlers, override them in your derived class
		virtual void AddFilesClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddDirectoryClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClearClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTextKeyPress( wxKeyEvent& event ) { event.Skip(); }
		virtual void TextChanged( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMoviesAreGainCorrectedCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnGainFilePickerChanged( wxFileDirPickerEvent& event ) { event.Skip(); }
		virtual void OnResampleMoviesCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCorrectMagDistortionCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMoviesHaveInvertedContrastCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSkipFullIntegrityCheckCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void CancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ImportClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		MovieImportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Import Movies"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxCLOSE_BOX );

		~MovieImportDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ImageImportDialog
///////////////////////////////////////////////////////////////////////////////
class ImageImportDialog : public wxDialog
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
		wxStaticText* m_staticText21;
		wxTextCtrl* CsText;
		wxStaticText* m_staticText20;
		wxTextCtrl* PixelSizeText;
		wxCheckBox* SaveScaledSumCheckbox;
		wxCheckBox* ImagesHaveInvertedContrast;
		wxStaticLine* m_staticline8;
		wxButton* m_button13;
		wxButton* ImportButton;

		// Virtual event handlers, override them in your derived class
		virtual void AddFilesClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddDirectoryClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClearClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTextKeyPress( wxKeyEvent& event ) { event.Skip(); }
		virtual void TextChanged( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImagesHaveInvertedContrastCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void CancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void ImportClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		ImageImportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Import Images"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 484,559 ), long style = wxCLOSE_BOX );

		~ImageImportDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class AssetPanelParent
///////////////////////////////////////////////////////////////////////////////
class AssetPanelParent : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline5;
		wxSplitterWindow* SplitterWindow;
		wxPanel* LeftPanel;
		wxStaticText* m_staticText18;
		wxButton* AddGroupButton;
		wxButton* RenameGroupButton;
		wxButton* RemoveGroupButton;
		wxButton* InvertGroupButton;
		wxButton* NewFromParentButton;
		wxPanel* m_panel3;
		wxStaticText* AssetTypeText;
		wxButton* ImportAsset;
		wxButton* RemoveSelectedAssetButton;
		wxButton* RemoveAllAssetsButton;
		wxButton* RenameAssetButton;
		wxButton* AddSelectedAssetButton;
		wxButton* DisplayButton;
		wxStaticLine* m_staticline6;
		wxStaticText* Label0Title;
		wxStaticText* Label0Text;
		wxStaticText* Label1Title;
		wxStaticText* Label1Text;
		wxStaticText* Label2Title;
		wxStaticText* Label2Text;
		wxStaticText* Label3Title;
		wxStaticText* Label3Text;
		wxStaticText* Label4Title;
		wxStaticText* Label4Text;
		wxStaticText* Label5Title;
		wxStaticText* Label5Text;
		wxStaticText* Label6Title;
		wxStaticText* Label6Text;
		wxStaticText* Label7Title;
		wxStaticText* Label7Text;
		wxStaticText* Label8Title;
		wxStaticText* Label8Text;
		wxStaticText* Label9Title;
		wxStaticText* Label9Text;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void MouseCheckGroupsVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void MouseVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnBeginEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnEndEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnGroupActivated( wxListEvent& event ) { event.Skip(); }
		virtual void OnGroupFocusChange( wxListEvent& event ) { event.Skip(); }
		virtual void NewGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RenameGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RemoveGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void InvertGroupClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void NewFromParentClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void MouseCheckContentsVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnBeginContentsDrag( wxListEvent& event ) { event.Skip(); }
		virtual void OnAssetActivated( wxListEvent& event ) { event.Skip(); }
		virtual void OnContentsSelected( wxListEvent& event ) { event.Skip(); }
		virtual void OnMotion( wxMouseEvent& event ) { event.Skip(); }
		virtual void ImportAssetClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RemoveAssetClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RemoveAllAssetsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void RenameAssetClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddSelectedAssetClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDisplayButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxListCtrl* GroupListBox;
		ContentsList* ContentsListBox;

		AssetPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1094,668 ), long style = wxHSCROLL|wxTAB_TRAVERSAL|wxVSCROLL, const wxString& name = wxEmptyString );

		~AssetPanelParent();

		void SplitterWindowOnIdle( wxIdleEvent& )
		{
			SplitterWindow->SetSashPosition( 405 );
			SplitterWindow->Disconnect( wxEVT_IDLE, wxIdleEventHandler( AssetPanelParent::SplitterWindowOnIdle ), NULL, this );
		}

};

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

///////////////////////////////////////////////////////////////////////////////
/// Class AutoRefine3DPanelParent
///////////////////////////////////////////////////////////////////////////////
class AutoRefine3DPanelParent : public JobPanel
{
	private:

	protected:
		wxStaticLine* m_staticline12;
		wxPanel* InputParamsPanel;
		wxStaticText* m_staticText262;
		RefinementPackagePickerComboPanel* RefinementPackageSelectPanel;
		wxStaticText* m_staticText478;
		VolumeAssetPickerComboPanel* ReferenceSelectPanel;
		wxCheckBox* UseMaskCheckBox;
		VolumeAssetPickerComboPanel* MaskSelectPanel;
		wxStaticLine* m_staticline54;
		wxStaticText* InitialResLimitStaticText;
		NumericTextCtrl* HighResolutionLimitTextCtrl;
		wxToggleButton* ExpertToggleButton;
		wxStaticText* PleaseCreateRefinementPackageText;
		wxStaticLine* m_staticline105;
		wxScrolledWindow* ExpertPanel;
		wxBoxSizer* InputSizer;
		wxButton* ResetAllDefaultsButton;
		wxStaticText* m_staticText202;
		wxStaticText* NoMovieFramesStaticText;
		NumericTextCtrl* LowResolutionLimitTextCtrl;
		wxStaticText* m_staticText196;
		NumericTextCtrl* MaskRadiusTextCtrl;
		wxStaticText* m_staticText330;
		NumericTextCtrl* InnerMaskRadiusTextCtrl;
		wxStaticText* m_staticText201;
		wxStaticText* GlobalMaskRadiusStaticText;
		NumericTextCtrl* GlobalMaskRadiusTextCtrl;
		wxStaticText* NumberToRefineStaticText;
		wxSpinCtrl* NumberToRefineSpinCtrl;
		wxStaticText* SearchRangeXStaticText;
		NumericTextCtrl* SearchRangeXTextCtrl;
		wxStaticText* SearchRangeYStaticText;
		NumericTextCtrl* SearchRangeYTextCtrl;
		wxStaticText* m_staticText329;
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
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAutoMaskButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartRefinementClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		DisplayRefinementResultsPanel* ShowRefinementResultsPanel;

		AutoRefine3DPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1216,660 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~AutoRefine3DPanelParent();

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

///////////////////////////////////////////////////////////////////////////////
/// Class NewProjectWizard
///////////////////////////////////////////////////////////////////////////////
class NewProjectWizard : public wxWizard
{
	private:

	protected:
		wxStaticText* m_staticText41;
		wxTextCtrl* ProjectNameTextCtrl;
		wxStaticText* m_staticText42;
		wxTextCtrl* ParentDirTextCtrl;
		wxButton* m_button24;
		wxStaticText* m_staticText45;
		wxTextCtrl* ProjectPathTextCtrl;
		wxStaticText* ErrorText;

		// Virtual event handlers, override them in your derived class
		virtual void OnFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnProjectTextChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnParentDirChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnBrowseButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnProjectPathChange( wxCommandEvent& event ) { event.Skip(); }


	public:

		NewProjectWizard( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Create New Project"), const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;

		~NewProjectWizard();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ExportRefinementPackageWizardParent
///////////////////////////////////////////////////////////////////////////////
class ExportRefinementPackageWizardParent : public wxWizard
{
	private:

	protected:
		wxStaticText* m_staticText46511;
		wxStaticLine* m_staticline10511;
		RefinementPickerComboPanel* ParameterSelectPanel;
		wxStaticLine* m_staticline123;
		wxComboBox* ClassComboBox;
		wxStaticText* m_staticText4651;
		wxStaticLine* m_staticline1051;
		wxRadioButton* FrealignRadioButton;
		wxRadioButton* RelionRadioButton;
		wxRadioButton* Relion3RadioButton;
		wxStaticText* m_staticText4741;
		wxStaticLine* m_staticline1061;
		wxStaticText* m_staticText411;
		wxTextCtrl* ParticleStackFileTextCtrl;
		wxButton* m_button2411;
		wxStaticText* MetaFilenameStaticText;
		wxTextCtrl* MetaDataFileTextCtrl;
		wxButton* m_button242;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnPageChanged( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnPageChanging( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnPathChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnStackBrowseButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMetaBrowseButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		ExportRefinementPackageWizardParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;

		~ExportRefinementPackageWizardParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ImportRefinementPackageWizardParent
///////////////////////////////////////////////////////////////////////////////
class ImportRefinementPackageWizardParent : public wxWizard
{
	private:

	protected:
		wxStaticText* m_staticText4651;
		wxStaticLine* m_staticline1051;
		wxRadioButton* cisTEMRadioButton;
		wxRadioButton* RelionRadioButton;
		wxRadioButton* FrealignRadioButton;
		wxStaticText* m_staticText474;
		wxStaticLine* m_staticline106;
		wxStaticText* m_staticText41;
		wxTextCtrl* ParticleStackFileTextCtrl;
		wxButton* m_button241;
		wxStaticText* MetaFilenameStaticText;
		wxTextCtrl* MetaDataFileTextCtrl;
		wxButton* m_button24;
		wxStaticText* m_staticText476;
		wxStaticLine* m_staticline107;
		wxStaticText* PixelSizeTextCtrlLabel;
		NumericTextCtrl* PixelSizeTextCtrl;
		wxStaticText* MicroscopeVoltageTextCtrlLabel;
		NumericTextCtrl* MicroscopeVoltageTextCtrl;
		wxStaticText* m_staticText479;
		NumericTextCtrl* SphericalAberrationTextCtrl;
		wxStaticText* AmplitudeContrastTextCtrlLabel;
		NumericTextCtrl* AmplitudeContrastTextCtrl;
		wxStaticText* m_staticText459;
		wxStaticText* m_staticText460;
		wxStaticText* m_staticText214;
		wxStaticText* m_staticText462;
		wxRadioButton* BlackProteinRadioButton;
		wxRadioButton* WhiteProteinRadioButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnPageChanged( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnPageChanging( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnPathChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnStackBrowseButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMetaBrowseButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxComboBox* SymmetryComboBox;
		NumericTextCtrl* MolecularWeightTextCtrl;
		NumericTextCtrl* LargestDimensionTextCtrl;

		ImportRefinementPackageWizardParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;

		~ImportRefinementPackageWizardParent();

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

///////////////////////////////////////////////////////////////////////////////
/// Class RenameDialog
///////////////////////////////////////////////////////////////////////////////
class RenameDialog : public wxDialog
{
	private:

	protected:
		wxBoxSizer* MainBoxSizer;
		wxStaticText* m_staticText246;
		wxStaticLine* m_staticline18;
		wxScrolledWindow* RenameScrollPanel;
		wxStaticLine* m_staticline19;
		wxButton* CancelButton;
		wxButton* RenameButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnCancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxBoxSizer* RenameBoxSizer;

		RenameDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Rename Assets"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxDEFAULT_DIALOG_STYLE );

		~RenameDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class NewRefinementPackageWizard
///////////////////////////////////////////////////////////////////////////////
class NewRefinementPackageWizard : public wxWizard
{
	private:

	protected:

		// Virtual event handlers, override them in your derived class
		virtual void OnFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void PageChanged( wxWizardEvent& event ) { event.Skip(); }
		virtual void PageChanging( wxWizardEvent& event ) { event.Skip(); }


	public:

		NewRefinementPackageWizard( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Create New Refinement Package"), const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;

		~NewRefinementPackageWizard();

};

///////////////////////////////////////////////////////////////////////////////
/// Class VolumeChooserDialog
///////////////////////////////////////////////////////////////////////////////
class VolumeChooserDialog : public wxDialog
{
	private:

	protected:
		wxBoxSizer* MainBoxSizer;
		wxStaticText* m_staticText246;
		wxStaticLine* m_staticline18;
		wxStaticLine* m_staticline19;
		wxButton* CancelButton;
		wxButton* SetButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnCancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		VolumeAssetPickerComboPanel* ComboBox;

		VolumeChooserDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Select new reference"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~VolumeChooserDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class AtomicCoordinatesChooserDialogParent
///////////////////////////////////////////////////////////////////////////////
class AtomicCoordinatesChooserDialogParent : public wxDialog
{
	private:

	protected:
		wxBoxSizer* MainBoxSizer;
		wxStaticText* m_staticText246;
		wxStaticLine* m_staticline18;
		wxStaticLine* m_staticline19;
		wxButton* CancelButton;
		wxButton* SetButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnCancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		VolumeAssetPickerComboPanel* ComboBox;

		AtomicCoordinatesChooserDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Select new reference"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~AtomicCoordinatesChooserDialogParent();

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
/// Class ParticlePositionExportDialog
///////////////////////////////////////////////////////////////////////////////
class ParticlePositionExportDialog : public wxDialog
{
	private:

	protected:
		wxPanel* m_panel38;
		wxComboBox* GroupComboBox;
		wxDirPickerCtrl* DestinationDirectoryPickerCtrl;
		wxStaticText* WarningText;
		wxButton* CancelButton;
		wxButton* ExportButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnDirChanged( wxFileDirPickerEvent& event ) { event.Skip(); }
		virtual void OnCancelButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		ParticlePositionExportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Export particle positions"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~ParticlePositionExportDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class FrealignExportDialog
///////////////////////////////////////////////////////////////////////////////
class FrealignExportDialog : public wxDialog
{
	private:

	protected:
		wxPanel* m_panel38;
		wxComboBox* GroupComboBox;
		wxStaticText* m_staticText202;
		wxSpinCtrl* DownsamplingFactorSpinCtrl;
		wxStaticText* m_staticText203;
		wxSpinCtrl* BoxSizeSpinCtrl;
		wxCheckBox* NormalizeCheckBox;
		wxCheckBox* FlipCTFCheckBox;
		wxFilePickerCtrl* OutputImageStackPicker;
		wxStaticText* WarningText;
		wxButton* CancelButton;
		wxButton* ExportButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnFlipCTFCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOutputImageStackFileChanged( wxFileDirPickerEvent& event ) { event.Skip(); }
		virtual void OnCancelButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		FrealignExportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Export to Frealign"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~FrealignExportDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RelionExportDialog
///////////////////////////////////////////////////////////////////////////////
class RelionExportDialog : public wxDialog
{
	private:

	protected:
		wxPanel* m_panel38;
		wxComboBox* GroupComboBox;
		wxStaticText* m_staticText202;
		wxSpinCtrl* DownsamplingFactorSpinCtrl;
		wxStaticText* m_staticText203;
		wxSpinCtrl* BoxSizeSpinCtrl;
		wxCheckBox* NormalizeCheckBox;
		wxStaticText* particleRadiusStaticText;
		wxTextCtrl* particleRadiusTextCtrl;
		wxCheckBox* FlipCTFCheckBox;
		wxFilePickerCtrl* OutputImageStackPicker;
		wxStaticText* FileNameStaticText;
		wxStaticText* WarningText;
		wxButton* CancelButton;
		wxButton* ExportButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnNormalizeCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFlipCTFCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOutputImageStackFileChanged( wxFileDirPickerEvent& event ) { event.Skip(); }
		virtual void OnCancelButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		RelionExportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Export to Relion"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~RelionExportDialog();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RefinementPackageAssetPanel
///////////////////////////////////////////////////////////////////////////////
class RefinementPackageAssetPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline52;
		wxSplitterWindow* m_splitter11;
		wxPanel* m_panel50;
		wxStaticText* m_staticText313;
		wxButton* CreateButton;
		wxButton* RenameButton;
		wxButton* DeleteButton;
		wxStaticLine* m_staticline122;
		wxButton* ImportButton;
		wxButton* ExportButton;
		RefinementPackageListControl* RefinementPackageListCtrl;
		wxPanel* m_panel51;
		wxStaticText* ContainedParticlesStaticText;
		ContainedParticleListControl* ContainedParticlesListCtrl;
		wxStaticText* m_staticText230;
		wxButton* DisplayStackButton;
		ReferenceVolumesListControl* Active3DReferencesListCtrl;
		wxStaticLine* m_staticline53;
		wxStaticText* m_staticText319;
		wxStaticText* StackFileNameText;
		wxStaticText* m_staticText210;
		wxStaticText* StackBoxSizeText;
		wxStaticText* m_staticText315;
		wxStaticText* NumberofClassesText;
		wxStaticText* m_staticText279;
		wxStaticText* SymmetryText;
		wxStaticText* m_staticText281;
		wxStaticText* MolecularWeightText;
		wxStaticText* m_staticText283;
		wxStaticText* LargestDimensionText;
		wxStaticText* m_staticText317;
		wxStaticText* NumberofRefinementsText;
		wxStaticText* m_staticText212;
		wxStaticText* LastRefinementIDText;

		// Virtual event handlers, override them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnCreateClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDeleteClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImportClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void MouseCheckPackagesVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void MouseVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnBeginEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnEndEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnPackageActivated( wxListEvent& event ) { event.Skip(); }
		virtual void OnPackageFocusChange( wxListEvent& event ) { event.Skip(); }
		virtual void MouseCheckParticlesVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnDisplayStackButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnVolumeListItemActivated( wxListEvent& event ) { event.Skip(); }


	public:

		RefinementPackageAssetPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 770,428 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RefinementPackageAssetPanel();

		void m_splitter11OnIdle( wxIdleEvent& )
		{
			m_splitter11->SetSashPosition( 600 );
			m_splitter11->Disconnect( wxEVT_IDLE, wxIdleEventHandler( RefinementPackageAssetPanel::m_splitter11OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class TemplateWizardPanel
///////////////////////////////////////////////////////////////////////////////
class TemplateWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214;

	public:
		wxComboBox* GroupComboBox;
		AutoWrapStaticText* InfoText;

		TemplateWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~TemplateWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class InputParameterWizardPanel
///////////////////////////////////////////////////////////////////////////////
class InputParameterWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214;

	public:
		wxComboBox* GroupComboBox;
		AutoWrapStaticText* InfoText;

		InputParameterWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~InputParameterWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassSelectionWizardPanel
///////////////////////////////////////////////////////////////////////////////
class ClassSelectionWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText416;

	public:
		wxListCtrl* SelectionListCtrl;
		AutoWrapStaticText* InfoText;

		ClassSelectionWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ClassSelectionWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class SymmetryWizardPanel
///////////////////////////////////////////////////////////////////////////////
class SymmetryWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214;

	public:
		wxComboBox* SymmetryComboBox;
		AutoWrapStaticText* InfoText;

		SymmetryWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~SymmetryWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class MolecularWeightWizardPanel
///////////////////////////////////////////////////////////////////////////////
class MolecularWeightWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214;

	public:
		NumericTextCtrl* MolecularWeightTextCtrl;
		AutoWrapStaticText* InfoText;

		MolecularWeightWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~MolecularWeightWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class InitialReferenceSelectWizardPanel
///////////////////////////////////////////////////////////////////////////////
class InitialReferenceSelectWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline108;

	public:
		wxBoxSizer* MainSizer;
		wxStaticText* TitleText;
		wxScrolledWindow* ScrollWindow;
		wxBoxSizer* ScrollSizer;
		AutoWrapStaticText* InfoText;

		InitialReferenceSelectWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~InitialReferenceSelectWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class LargestDimensionWizardPanel
///////////////////////////////////////////////////////////////////////////////
class LargestDimensionWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214;

	public:
		NumericTextCtrl* LargestDimensionTextCtrl;
		AutoWrapStaticText* InfoText;

		LargestDimensionWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~LargestDimensionWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class OutputPixelSizeWizardPanel
///////////////////////////////////////////////////////////////////////////////
class OutputPixelSizeWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214;

	public:
		NumericTextCtrl* OutputPixelSizeTextCtrl;
		AutoWrapStaticText* InfoText;

		OutputPixelSizeWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~OutputPixelSizeWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ParticleGroupWizardPanel
///////////////////////////////////////////////////////////////////////////////
class ParticleGroupWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText2141;

	public:
		wxComboBox* ParticlePositionsGroupComboBox;
		AutoWrapStaticText* InfoText;

		ParticleGroupWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ParticleGroupWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class BoxSizeWizardPanel
///////////////////////////////////////////////////////////////////////////////
class BoxSizeWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;

	public:
		wxSpinCtrl* BoxSizeSpinCtrl;
		AutoWrapStaticText* InfoText;

		BoxSizeWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~BoxSizeWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class NumberofClassesWizardPanel
///////////////////////////////////////////////////////////////////////////////
class NumberofClassesWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214111;

	public:
		wxSpinCtrl* NumberOfClassesSpinCtrl;
		AutoWrapStaticText* InfoText;

		NumberofClassesWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~NumberofClassesWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassesSetupWizardPanelA
///////////////////////////////////////////////////////////////////////////////
class ClassesSetupWizardPanelA : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;
		wxRadioButton* m_radioBtn40;

	public:
		wxRadioButton* CarryOverYesButton;
		AutoWrapStaticText* InfoText;

		ClassesSetupWizardPanelA( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ClassesSetupWizardPanelA();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RecentrePicksWizardPanel
///////////////////////////////////////////////////////////////////////////////
class RecentrePicksWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;
		wxRadioButton* m_radioBtn40;

	public:
		wxRadioButton* ReCentreYesButton;
		AutoWrapStaticText* InfoText;

		RecentrePicksWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RecentrePicksWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RemoveDuplicatesWizardPanel
///////////////////////////////////////////////////////////////////////////////
class RemoveDuplicatesWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;
		wxRadioButton* m_radioBtn40;

	public:
		wxRadioButton* RemoveDuplicateYesButton;
		AutoWrapStaticText* InfoText;

		RemoveDuplicatesWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RemoveDuplicatesWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class RemoveDuplicateThresholdWizardPanel
///////////////////////////////////////////////////////////////////////////////
class RemoveDuplicateThresholdWizardPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText214;

	public:
		NumericTextCtrl* DuplicatePickThresholdTextCtrl;
		AutoWrapStaticText* InfoText;

		RemoveDuplicateThresholdWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RemoveDuplicateThresholdWizardPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassesSetupWizardPanelB
///////////////////////////////////////////////////////////////////////////////
class ClassesSetupWizardPanelB : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;
		wxStaticLine* m_staticline103;

	public:
		wxListCtrl* ClassListCtrl;
		AutoWrapStaticText* InfoText;

		ClassesSetupWizardPanelB( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ClassesSetupWizardPanelB();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassesSetupWizardPanelC
///////////////////////////////////////////////////////////////////////////////
class ClassesSetupWizardPanelC : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;
		wxStaticLine* m_staticline104;

	public:
		wxListCtrl* NewClassListCtrl;
		wxListCtrl* OldClassListCtrl;
		AutoWrapStaticText* InfoText;

		ClassesSetupWizardPanelC( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ClassesSetupWizardPanelC();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassesSetupWizardPanelD
///////////////////////////////////////////////////////////////////////////////
class ClassesSetupWizardPanelD : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;
		wxRadioButton* m_radioBtn40;

	public:
		wxRadioButton* BestOccupancyRadioButton;
		AutoWrapStaticText* InfoText;

		ClassesSetupWizardPanelD( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ClassesSetupWizardPanelD();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassesSetupWizardPanelE
///////////////////////////////////////////////////////////////////////////////
class ClassesSetupWizardPanelE : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText21411;
		wxRadioButton* m_radioBtn40;

	public:
		wxRadioButton* RandomiseOccupanciesRadioButton;
		AutoWrapStaticText* InfoText;

		ClassesSetupWizardPanelE( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~ClassesSetupWizardPanelE();

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
/// Class AbInitioPlotPanelParent
///////////////////////////////////////////////////////////////////////////////
class AbInitioPlotPanelParent : public wxPanel
{
	private:

	protected:
		wxPanel* SigmaPanel;

	public:
		wxAuiNotebook* my_notebook;

		AbInitioPlotPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~AbInitioPlotPanelParent();

};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassumSelectionCopyFromDialogParent
///////////////////////////////////////////////////////////////////////////////
class ClassumSelectionCopyFromDialogParent : public wxDialog
{
	private:

	protected:
		wxStaticText* m_staticText414;
		wxStaticLine* m_staticline72;
		wxListCtrl* SelectionListCtrl;
		wxStaticLine* m_staticline71;
		wxButton* OkButton;
		wxButton* CancelButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnOKButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCancelButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:

		ClassumSelectionCopyFromDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~ClassumSelectionCopyFromDialogParent();

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

