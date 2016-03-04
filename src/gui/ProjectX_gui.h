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
class JobPanel;
class NumericTextCtrl;
class ResultsDataViewListCtrl;
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
#include <wx/statline.h>
#include <wx/radiobut.h>
#include <wx/button.h>
#include <wx/dataview.h>
#include <wx/stattext.h>
#include <wx/combobox.h>
#include <wx/splitter.h>
#include <wx/textctrl.h>
#include <wx/dialog.h>
#include <wx/statbmp.h>
#include <wx/tglbtn.h>
#include <wx/checkbox.h>
#include <wx/spinctrl.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/scrolwin.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );

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
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnMenuBookChange( wxListbookEvent& event ) { event.Skip(); }
		virtual void OnFileMenuUpdate( wxUpdateUIEvent& event ) { event.Skip(); }
		virtual void OnFileNewProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileOpenProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileCloseProject( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFileExit( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxListbook* MenuBook;
		
		MainFrame( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("ProjectX"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1366,768 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~MainFrame();
	
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
		wxStaticText* m_staticText20;
		wxTextCtrl* PixelSizeText;
		wxStaticText* m_staticText21;
		wxTextCtrl* CsText;
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
		
		ImageImportDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Import Movies"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 484,539 ), long style = wxCLOSE_BOX ); 
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
		wxListCtrl* ContentsListBox;
		wxButton* ImportAsset;
		wxButton* RemoveSelectedAssetButton;
		wxButton* RemoveAllAssetsButton;
		wxButton* AddSelectedAssetButton;
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
		virtual void AddSelectedAssetClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		wxListCtrl* GroupListBox;
		
		AssetParentPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 940,517 ), long style = wxHSCROLL|wxTAB_TRAVERSAL|wxVSCROLL ); 
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
			m_splitter5->SetSashPosition( 300 );
			m_splitter5->Disconnect( wxEVT_IDLE, wxIdleEventHandler( RunProfilesPanel::m_splitter5OnIdle ), NULL, this );
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
		wxPanel* ResultsPanel;
		wxStaticText* m_staticText203;
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
		virtual void OnRestrainAstigmatismCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnFindAdditionalPhaseCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnInfoURL( wxTextUrlEvent& event ) { event.Skip(); }
		virtual void TerminateButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void FinishButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void StartEstimationClick( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
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
/// Class FilterDialog
///////////////////////////////////////////////////////////////////////////////
class FilterDialog : public wxDialog 
{
	private:
	
	protected:
		wxBoxSizer* MainBoxSizer;
		wxStaticText* m_staticText64;
		wxStaticLine* m_staticline18;
		wxBoxSizer* FilterBoxSizer;
		wxStaticLine* m_staticline19;
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

#endif //__PROJECTX_GUI_H__
