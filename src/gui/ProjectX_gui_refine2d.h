///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class ClassificationPickerComboPanel;
class ClassificationPlotPanel;
class ClassificationSelectionListCtrl;
class DisplayPanel;
class MemoryComboBox;
class NumericTextCtrl;
class RefinementPackagePickerComboPanel;

#include "job_panel.h"
#include <wx/statline.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/panel.h>
#include <wx/sizer.h>
#include <wx/spinctrl.h>
#include <wx/tglbtn.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/radiobut.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/combobox.h>
#include <wx/listctrl.h>
#include <wx/splitter.h>

///////////////////////////////////////////////////////////////////////////

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

