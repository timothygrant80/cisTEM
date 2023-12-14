///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class AbInitioPlotPanel;
class ClassSelectionPickerComboPanel;
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
#include <wx/radiobut.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/spinctrl.h>
#include <wx/combobox.h>
#include <wx/tglbtn.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/button.h>
#include <wx/textctrl.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/aui/auibook.h>

///////////////////////////////////////////////////////////////////////////

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

