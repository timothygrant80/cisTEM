///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class ImageGroupPickerComboPanel;
class ImagesPickerComboPanel;
class MemoryComboBox;
class NumericTextCtrl;
class PickingBitmapPanel;
class PickingResultsDisplayPanel;
class ResultsDataViewListCtrl;

#include "job_panel.h"
#include <wx/panel.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/checkbox.h>
#include <wx/textctrl.h>
#include <wx/stattext.h>
#include <wx/sizer.h>
#include <wx/combobox.h>
#include <wx/statline.h>
#include <wx/button.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/radiobut.h>
#include <wx/tglbtn.h>
#include <wx/dataview.h>
#include <wx/splitter.h>
#include <wx/spinctrl.h>
#include <wx/choice.h>
#include <wx/scrolwin.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/gauge.h>

///////////////////////////////////////////////////////////////////////////

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
		virtual void OnScalingChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnUndoButtonClick( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnRedoButtonClick( wxCommandEvent& event ) { event.Skip(); }


	public:
		PickingBitmapPanel* PickingResultsImagePanel;
		wxComboBox* ScalingComboBox;
		wxStaticText* ImageScalingText;
		wxButton* UndoButton;
		wxButton* RedoButton;

		PickingResultsDisplayPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1123,360 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~PickingResultsDisplayPanelParent();

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

