///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class DisplayRefinementResultsPanel;
class MemoryComboBox;
class NumericTextCtrl;
class RefinementPackagePickerComboPanel;
class VolumeAssetPickerComboPanel;

#include "job_panel.h"
#include <wx/statline.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/panel.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>
#include <wx/textctrl.h>
#include <wx/tglbtn.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/button.h>
#include <wx/spinctrl.h>
#include <wx/radiobut.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>
#include <wx/combobox.h>

///////////////////////////////////////////////////////////////////////////

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
		wxFlexGridSizer* fgSizer1;
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

