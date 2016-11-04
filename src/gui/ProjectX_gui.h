///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jan 30 2016)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTX_GUI_H__
#define __PROJECTX_GUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class AngularDistributionPlotPanel;
class AutoWrapStaticText;
class BitmapPanel;
class CTF1DPanel;
class ContainedParticleListControl;
class ContentsList;
class JobPanel;
class MyFSCPanel;
class NumericTextCtrl;
class PickingBitmapPanel;
class PickingResultsDisplayPanel;
class PlotFSCPanel;
class ReferenceVolumesListControl;
class RefinementPackageListControl;
class RefinementParametersListCtrl;
class ResultsDataViewListCtrl;
class ShowCTFResultsPanel;
class UnblurResultsPanel;

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
#include <wx/stattext.h>
#include <wx/combobox.h>
#include <wx/button.h>
#include <wx/splitter.h>
#include <wx/radiobut.h>
#include <wx/statline.h>
#include <wx/checkbox.h>
#include <wx/dataview.h>
#include <wx/tglbtn.h>
#include <wx/textctrl.h>
#include <wx/dialog.h>
#include <wx/spinctrl.h>
#include <wx/choice.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/statbmp.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );
#include <wx/statbox.h>
#include <wx/filepicker.h>

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
		wxMenu* ExportMenu;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnMenuBookChange( wxListbookEvent& event ) { event.Skip(); }
		virtual void OnFileMenuUpdate( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnFileNewProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileOpenProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileCloseProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileExit( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportCoordinatesToImagic( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportToFrealign( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxListbook* MenuBook;
		
		MainFrame( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("cisTEM"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1366,768 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~MainFrame();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class RefinementResultsPanel
///////////////////////////////////////////////////////////////////////////////
class RefinementResultsPanel : public wxPanel 
{
	private:
	
	protected:
		wxSplitterWindow* m_splitter7;
		wxPanel* m_panel48;
		wxStaticText* m_staticText284;
		wxStaticText* m_staticText285;
		RefinementParametersListCtrl* ParameterListCtrl;
		wxPanel* m_panel49;
		MyFSCPanel* FSCPlotPanel;
		wxButton* PlotAngleButton;
		AngularDistributionPlotPanel* AngularPlotPanel;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnRefinementPackageComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInputParametersComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPlotButtonClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxComboBox* RefinementPackageComboBox;
		wxComboBox* InputParametersComboBox;
		
		RefinementResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 659,471 ), long style = wxTAB_TRAVERSAL ); 
		~RefinementResultsPanel();
		
		void m_splitter7OnIdle( wxIdleEvent& )
		{
			m_splitter7->SetSashPosition( 550 );
			m_splitter7->Disconnect( wxEVT_IDLE, wxIdleEventHandler( RefinementResultsPanel::m_splitter7OnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class ShowCTFResultsParentPanel
///////////////////////////////////////////////////////////////////////////////
class ShowCTFResultsParentPanel : public wxPanel 
{
	private:
	
	protected:
		wxRadioButton* FitType2DRadioButton;
		wxRadioButton* FitType1DRadioButton;
		wxStaticLine* m_staticline26;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnFitTypeRadioButton( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		BitmapPanel* CTF2DResultsPanel;
		CTF1DPanel* CTFPlotPanel;
		
		ShowCTFResultsParentPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~ShowCTFResultsParentPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class PickingResultsDisplayParentPanel
///////////////////////////////////////////////////////////////////////////////
class PickingResultsDisplayParentPanel : public wxPanel 
{
	private:
	
	protected:
		wxCheckBox* CirclesAroundParticlesCheckBox;
		wxCheckBox* HighPassFilterCheckBox;
		wxCheckBox* LowPassFilterCheckBox;
		wxCheckBox* ScaleBarCheckBox;
		wxStaticLine* m_staticline26;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnCirclesAroundParticlesCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighPassFilterCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLowPassFilterCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnScaleBarCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnUndoButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRedoButtonClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxButton* UndoButton;
		wxButton* RedoButton;
		PickingBitmapPanel* PickingResultsImagePanel;
		
		PickingResultsDisplayParentPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~PickingResultsDisplayParentPanel();
	
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
		ResultsDataViewListCtrl* ResultDataView;
		wxButton* PreviousButton;
		wxButton* NextButton;
		wxPanel* RightPanel;
		wxToggleButton* JobDetailsToggleButton;
		wxStaticLine* m_staticline28;
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
		wxStaticLine* m_staticline30;
		ShowCTFResultsPanel* ResultPanel;
		wxButton* AddToGroupButton;
		wxComboBox* GroupComboBox;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnAllMoviesSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnByFilterSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDefineFilterClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreviousButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddToGroupClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		FindCTFResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 895,557 ), long style = wxTAB_TRAVERSAL ); 
		~FindCTFResultsPanel();
		
		void m_splitter4OnIdle( wxIdleEvent& )
		{
			m_splitter4->SetSashPosition( 500 );
			m_splitter4->Disconnect( wxEVT_IDLE, wxIdleEventHandler( FindCTFResultsPanel::m_splitter4OnIdle ), NULL, this );
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
		ResultsDataViewListCtrl* ResultDataView;
		wxButton* PreviousButton;
		wxButton* NextButton;
		wxPanel* RightPanel;
		wxToggleButton* JobDetailsToggleButton;
		wxStaticLine* m_staticline28;
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
		wxButton* AddToGroupButton;
		wxComboBox* GroupComboBox;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnAllMoviesSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnByFilterSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDefineFilterClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreviousButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddToGroupClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		PickingResultsDisplayPanel* ResultDisplayPanel;
		
		PickingResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 895,557 ), long style = wxTAB_TRAVERSAL ); 
		~PickingResultsPanel();
		
		void m_splitter4OnIdle( wxIdleEvent& )
		{
			m_splitter4->SetSashPosition( 500 );
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
		ResultsDataViewListCtrl* ResultDataView;
		wxButton* PreviousButton;
		wxButton* NextButton;
		wxPanel* RightPanel;
		wxToggleButton* JobDetailsToggleButton;
		wxStaticLine* m_staticline28;
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
		UnblurResultsPanel* ResultPanel;
		wxButton* AddToGroupButton;
		wxComboBox* GroupComboBox;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnAllMoviesSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnByFilterSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDefineFilterClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreviousButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNextButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnJobDetailsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAddToGroupClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		MovieAlignResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 895,557 ), long style = wxTAB_TRAVERSAL ); 
		~MovieAlignResultsPanel();
		
		void m_splitter4OnIdle( wxIdleEvent& )
		{
			m_splitter4->SetSashPosition( 500 );
			m_splitter4->Disconnect( wxEVT_IDLE, wxIdleEventHandler( MovieAlignResultsPanel::m_splitter4OnIdle ), NULL, this );
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
/// Class SettingsPanel
///////////////////////////////////////////////////////////////////////////////
class SettingsPanel : public wxPanel 
{
	private:
	
	protected:
		wxStaticLine* m_staticline3;
	
	public:
		wxListbook* SettingsBook;
		
		SettingsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
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
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnResultsBookPageChanged( wxListbookEvent& event ) { event.Skip(); }
		
	
	public:
		wxListbook* ResultsBook;
		
		ResultsPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~ResultsPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class AssetsPanel
///////////////////////////////////////////////////////////////////////////////
class AssetsPanel : public wxPanel 
{
	private:
	
	protected:
		wxStaticLine* m_staticline24;
	
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
		wxStaticText* m_staticText279;
	
	public:
		
		OverviewPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~OverviewPanel();
	
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
		wxStaticText* m_staticText20;
		wxTextCtrl* PixelSizeText;
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
		
		// Virtual event handlers, overide them in your derived class
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
/// Class FindParticlesPanel
///////////////////////////////////////////////////////////////////////////////
class FindParticlesPanel : public JobPanel
{
	private:
	
	protected:
		wxStaticLine* m_staticline12;
		wxStaticText* m_staticText21;
		wxComboBox* GroupComboBox;
		wxStaticText* m_staticText211;
		wxComboBox* PickingAlgorithmComboBox;
		wxToggleButton* ExpertToggleButton;
		wxStaticText* PleaseEstimateCTFStaticText;
		wxStaticLine* m_staticline10;
		wxSplitterWindow* FindParticlesSplitterWindow;
		wxPanel* LeftPanel;
		wxScrolledWindow* PickingParametersPanel;
		wxBoxSizer* InputSizer;
		wxStaticText* m_staticText196;
		NumericTextCtrl* MaximumParticleRadiusNumericCtrl;
		wxStaticText* CharacteristicParticleRadiusStaticText;
		NumericTextCtrl* CharacteristicParticleRadiusNumericCtrl;
		wxStaticText* ThresholdPeakHeightStaticText1;
		NumericTextCtrl* ThresholdPeakHeightNumericCtrl;
		wxButton* TestOnCurrentMicrographButton;
		wxCheckBox* AutoPickRefreshCheckBox;
		wxComboBox* ImageComboBox;
		wxPanel* ExpertOptionsPanel;
		wxBoxSizer* ExpertInputSizer;
		wxStaticLine* m_staticline35;
		wxStaticText* ExpertOptionsStaticText;
		wxStaticText* HighestResolutionStaticText;
		NumericTextCtrl* HighestResolutionNumericCtrl;
		wxCheckBox* SetMinimumDistanceFromEdgesCheckBox;
		wxSpinCtrl* MinimumDistanceFromEdgesSpinCtrl;
		wxCheckBox* m_checkBox8;
		wxCheckBox* m_checkBox9;
		wxSpinCtrl* NumberOfTemplateRotationsSpinCtrl;
		wxCheckBox* AvoidHighVarianceAreasCheckBox;
		wxCheckBox* AvoidAbnormalLocalMeanAreasCheckBox;
		wxCheckBox* ShowEstimatedBackgroundSpectrumCheckBox;
		wxCheckBox* ShowPositionsOfBackgroundBoxesCheckBox;
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
		wxGauge* ProgressBar;
		wxStaticText* NumberConnectedText;
		wxStaticText* TimeRemainingText;
		wxButton* CancelAlignmentButton;
		wxButton* FinishButton;
		wxPanel* StartPanel;
		wxStaticText* RunProfileText;
		wxComboBox* RunProfileComboBox;
		wxButton* StartPickingButton;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnGroupComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPickingAlgorithmComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMaximumParticleRadiusNumericTextKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnMaximumParticleRadiusNumericTextSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnMaximumParticleRadiusNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCharacteristicParticleRadiusNumericTextKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnCharacteristicParticleRadiusNumericTextSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnCharacteristicParticleRadiusNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnThresholdPeakHeightNumericTextKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnThresholdPeakHeightNumericTextSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnThresholdPeakHeightNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTestOnCurrentMicrographButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAutoPickRefreshCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImageComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighestResolutionNumericKillFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnHighestResolutionNumericSetFocus( wxFocusEvent& event ) { event.Skip(); }
		virtual void OnHighestResolutionNumericTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSetMinimumDistanceFromEdgesCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMinimumDistanceFromEdgesSpinCtrl( wxSpinEvent& event ) { event.Skip(); }
		virtual void OnAvoidHighVarianceAreasCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAvoidAbnormalLocalMeanAreasCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNumberOfBackgroundBoxesSpinCtrl( wxSpinEvent& event ) { event.Skip(); }
		virtual void OnAlgorithmToFindBackgroundChoice( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartPickingClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		PickingResultsDisplayPanel* PickingResultsPanel;
		
		FindParticlesPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,731 ), long style = wxTAB_TRAVERSAL ); 
		~FindParticlesPanel();
		
		void FindParticlesSplitterWindowOnIdle( wxIdleEvent& )
		{
			FindParticlesSplitterWindow->SetSashPosition( 528 );
			FindParticlesSplitterWindow->Disconnect( wxEVT_IDLE, wxIdleEventHandler( FindParticlesPanel::FindParticlesSplitterWindowOnIdle ), NULL, this );
		}
	
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
		
		ImageImportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Import Images"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 484,539 ), long style = wxCLOSE_BOX ); 
		~ImageImportDialog();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class AssetParentPanel
///////////////////////////////////////////////////////////////////////////////
class AssetParentPanel : public wxPanel 
{
	private:
	
	protected:
		wxStaticLine* m_staticline5;
		wxSplitterWindow* m_splitter2;
		wxPanel* m_panel4;
		wxStaticText* m_staticText18;
		wxButton* AddGroupButton;
		wxButton* RenameGroupButton;
		wxButton* RemoveGroupButton;
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
		
		// Virtual event handlers, overide them in your derived class
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
		virtual void MouseCheckContentsVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnBeginContentsDrag( wxListEvent& event ) { event.Skip(); }
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
		
		AssetParentPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1094,668 ), long style = wxHSCROLL|wxTAB_TRAVERSAL|wxVSCROLL ); 
		~AssetParentPanel();
		
		void m_splitter2OnIdle( wxIdleEvent& )
		{
			m_splitter2->SetSashPosition( 300 );
			m_splitter2->Disconnect( wxEVT_IDLE, wxIdleEventHandler( AssetParentPanel::m_splitter2OnIdle ), NULL, this );
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
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateIU( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnProfileDClick( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnProfileLeftDown( wxMouseEvent& event ) { event.Skip(); }
		virtual void MouseVeto( wxMouseEvent& event ) { event.Skip(); }
		virtual void OnEndProfileEdit( wxListEvent& event ) { event.Skip(); }
		virtual void OnProfilesListItemActivated( wxListEvent& event ) { event.Skip(); }
		virtual void OnProfilesFocusChange( wxListEvent& event ) { event.Skip(); }
		virtual void OnAddProfileClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameProfileClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRemoveProfileClick( wxCommandEvent& event ) { event.Skip(); }
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
		
		RunProfilesPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 940,517 ), long style = wxHSCROLL|wxTAB_TRAVERSAL|wxVSCROLL ); 
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
		wxComboBox* GroupComboBox;
		wxToggleButton* ExpertToggleButton;
		wxStaticLine* m_staticline10;
		wxPanel* ExpertPanel;
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
		wxStaticText* m_staticText50;
		wxSpinCtrl* horizontal_mask_spinctrl;
		wxStaticText* m_staticText51;
		wxSpinCtrl* vertical_mask_spinctrl;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
		UnblurResultsPanel* GraphPanel;
		wxPanel* InfoPanel;
		wxRichTextCtrl* InfoText;
		wxStaticLine* m_staticline11;
		wxPanel* ProgressPanel;
		wxGauge* ProgressBar;
		wxStaticText* NumberConnectedText;
		wxStaticText* TimeRemainingText;
		wxButton* CancelAlignmentButton;
		wxButton* FinishButton;
		wxPanel* StartPanel;
		wxStaticText* RunProfileText;
		wxComboBox* RunProfileComboBox;
		wxButton* StartAlignmentButton;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartAlignmentClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		AlignMoviesPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 927,653 ), long style = wxTAB_TRAVERSAL ); 
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
		wxStaticText* m_staticText262;
		wxComboBox* RefinementPackageComboBox;
		wxStaticText* m_staticText263;
		wxComboBox* InputParametersComboBox;
		wxRadioButton* LocalRefinementRadio;
		wxRadioButton* GlobalRefinementRadio;
		wxStaticText* m_staticText264;
		wxSpinCtrl* NumberRoundsSpinCtrl;
		wxToggleButton* ExpertToggleButton;
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
		wxStaticText* m_staticText202;
		wxStaticText* NoMovieFramesStaticText;
		NumericTextCtrl* LowResolutionLimitTextCtrl;
		wxStaticText* m_staticText188;
		NumericTextCtrl* HighResolutionLimitTextCtrl;
		wxStaticText* m_staticText196;
		NumericTextCtrl* MaskRadiusTextCtrl;
		wxStaticText* m_staticText317;
		NumericTextCtrl* SignedCCResolutionTextCtrl;
		wxStaticText* m_staticText201;
		wxStaticText* GlobalMaskRadiusStaticText;
		NumericTextCtrl* GlobalMaskRadiusTextCtrl;
		wxStaticText* NumberToRefineStaticText;
		wxSpinCtrl* NumberToRefineSpinCtrl;
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
		wxStaticText* m_staticText330;
		NumericTextCtrl* ReconstructionInnerRadiusTextCtrl;
		wxStaticText* m_staticText331;
		NumericTextCtrl* ReconstructionOuterRadiusTextCtrl;
		wxStaticText* m_staticText332;
		NumericTextCtrl* ScoreToBFactorConstantTextCtrl;
		wxStaticText* m_staticText335;
		wxRadioButton* AdjustScoreForDefocusYesRadio;
		wxRadioButton* AdjustScoreForDefocusNoRadio;
		wxStaticText* m_staticText333;
		NumericTextCtrl* ReconstructioScoreThreshold;
		wxStaticText* m_staticText334;
		NumericTextCtrl* ReconstructionResolutionLimitTextCtrl;
		wxStaticText* m_staticText336;
		wxRadioButton* AutoCropYesRadioButton;
		wxRadioButton* AutoCropNoRadioButton;
		wxPanel* OutputTextPanel;
		wxTextCtrl* output_textctrl;
		wxPanel* InfoPanel;
		wxRichTextCtrl* InfoText;
		AngularDistributionPlotPanel* AngularPlotPanel;
		wxStaticLine* m_staticline11;
		wxPanel* ProgressPanel;
		wxGauge* ProgressBar;
		wxStaticText* NumberConnectedText;
		wxStaticText* TimeRemainingText;
		wxButton* CancelAlignmentButton;
		wxButton* FinishButton;
		wxPanel* StartPanel;
		wxStaticText* RunProfileText;
		wxComboBox* RefinementRunProfileComboBox;
		wxStaticText* RunProfileText1;
		wxComboBox* ReconstructionRunProfileComboBox;
		wxButton* StartRefinementButton;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnRefinementPackageComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInputParametersComboBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResetAllDefaultsClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnHighResLimitChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartRefinementClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		MyFSCPanel* FSCResultsPanel;
		
		Refine3DPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,686 ), long style = wxTAB_TRAVERSAL ); 
		~Refine3DPanel();
	
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
/// Class FindCTFPanel
///////////////////////////////////////////////////////////////////////////////
class FindCTFPanel : public JobPanel
{
	private:
	
	protected:
		wxStaticLine* m_staticline12;
		wxStaticText* m_staticText21;
		wxComboBox* GroupComboBox;
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
		wxGauge* ProgressBar;
		wxStaticText* NumberConnectedText;
		wxStaticText* TimeRemainingText;
		wxButton* CancelAlignmentButton;
		wxButton* FinishButton;
		wxPanel* StartPanel;
		wxStaticText* RunProfileText;
		wxComboBox* RunProfileComboBox;
		wxButton* StartEstimationButton;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnExpertOptionsToggle( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMovieRadioButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnImageRadioButton( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLargeAstigmatismExpectedCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRestrainAstigmatismCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFindAdditionalPhaseCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartEstimationClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		ShowCTFResultsPanel* CTFResultsPanel;
		
		FindCTFPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1200,731 ), long style = wxTAB_TRAVERSAL ); 
		~FindCTFPanel();
	
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
		wxStaticText* m_staticText58;
		wxStaticText* ErrorStaticText;
		wxStaticLine* m_staticline14;
		wxButton* OKButton;
		wxButton* CancelButton;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOKClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCancelClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxTextCtrl* CommandTextCtrl;
		wxSpinCtrl* NumberCopiesSpinCtrl;
		wxSpinCtrl* DelayTimeSpinCtrl;
		
		AddRunCommandDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Enter Command..."), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE ); 
		~AddRunCommandDialog();
	
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
		
		// Virtual event handlers, overide them in your derived class
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
		
		// Virtual event handlers, overide them in your derived class
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
		
		// Virtual event handlers, overide them in your derived class
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
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnCancelClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxComboBox* ComboBox;
		
		VolumeChooserDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Select new reference"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 251,153 ), long style = wxDEFAULT_DIALOG_STYLE ); 
		~VolumeChooserDialog();
	
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
		
		// Virtual event handlers, overide them in your derived class
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
		
		// Virtual event handlers, overide them in your derived class
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
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnFlipCTFCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOutputImageStackFileChanged( wxFileDirPickerEvent& event ) { event.Skip(); }
		virtual void OnCancelButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportButtonClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		FrealignExportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Export to Frealign"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE ); 
		~FrealignExportDialog();
	
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
		RefinementPackageListControl* RefinementPackageListCtrl;
		wxPanel* m_panel51;
		wxStaticText* m_staticText314;
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
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnCreateClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRenameClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDeleteClick( wxCommandEvent& event ) { event.Skip(); }
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
		
		RefinementPackageAssetPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 770,428 ), long style = wxTAB_TRAVERSAL ); 
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
		
		TemplateWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
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
		
		InputParameterWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
		~InputParameterWizardPanel();
	
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
		
		SymmetryWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
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
		
		MolecularWeightWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
		~MolecularWeightWizardPanel();
	
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
		
		LargestDimensionWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
		~LargestDimensionWizardPanel();
	
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
		
		ParticleGroupWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
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
		
		BoxSizeWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
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
		
		NumberofClassesWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
		~NumberofClassesWizardPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class InitialReferenceSelectWizardPanel
///////////////////////////////////////////////////////////////////////////////
class InitialReferenceSelectWizardPanel : public wxPanel 
{
	private:
	
	protected:
	
	public:
		wxBoxSizer* MainSizer;
		wxStaticText* TitleText;
		wxScrolledWindow* ScrollWindow;
		wxGridSizer* GridSizer;
		AutoWrapStaticText* InfoText;
		
		InitialReferenceSelectWizardPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
		~InitialReferenceSelectWizardPanel();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class ClassesSetupWizardPanel 
///////////////////////////////////////////////////////////////////////////////
class ClassesSetupWizardPanel  : public wxPanel 
{
	private:
	
	protected:
		wxStaticText* m_staticText232;
	
	public:
		
		ClassesSetupWizardPanel ( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,400 ), long style = wxTAB_TRAVERSAL ); 
		~ClassesSetupWizardPanel ();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class FSCPanel
///////////////////////////////////////////////////////////////////////////////
class FSCPanel : public wxPanel 
{
	private:
	
	protected:
		wxBoxSizer* TitleSizer;
		wxStaticText* m_staticText280;
		wxStaticText* EstimatedResolutionLabel;
		wxStaticText* EstimatedResolutionText;
		wxStaticLine* m_staticline52;
		PlotFSCPanel* PlotPanel;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnClassComboBoxChange( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxComboBox* ClassComboBox;
		
		FSCPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL ); 
		~FSCPanel();
	
};

#endif //__PROJECTX_GUI_H__
