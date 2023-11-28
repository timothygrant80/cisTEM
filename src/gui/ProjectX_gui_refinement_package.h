///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class AutoWrapStaticText;
class ContainedParticleListControl;
class NumericTextCtrl;
class ReferenceVolumesListControl;
class RefinementPackageListControl;
class RefinementPickerComboPanel;

#include <wx/string.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/stattext.h>
#include <wx/statline.h>
#include <wx/panel.h>
#include <wx/combobox.h>
#include <wx/sizer.h>
#include <wx/radiobut.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/statbox.h>
#include <wx/spinctrl.h>
#include <wx/checkbox.h>
#include <wx/filepicker.h>
#include <wx/dialog.h>
#include <wx/listctrl.h>
#include <wx/splitter.h>
#include <wx/checklst.h>
#include <wx/scrolwin.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class CombineRefinementPackagesWizardParent
///////////////////////////////////////////////////////////////////////////////
class CombineRefinementPackagesWizardParent : public wxWizard
{
	private:

	protected:

		// Virtual event handlers, override them in your derived class
		virtual void OnFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void PageChanged( wxWizardEvent& event ) { event.Skip(); }
		virtual void PageChanging( wxWizardEvent& event ) { event.Skip(); }


	public:

		CombineRefinementPackagesWizardParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Combine Refinement Packages"), const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;

		~CombineRefinementPackagesWizardParent();

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
		wxButton* CombineButton;
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
		virtual void OnCombineClick( wxCommandEvent& event ) { event.Skip(); }
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

		RefinementPackageAssetPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~RefinementPackageAssetPanel();

		void m_splitter11OnIdle( wxIdleEvent& )
		{
			m_splitter11->SetSashPosition( 600 );
			m_splitter11->Disconnect( wxEVT_IDLE, wxIdleEventHandler( RefinementPackageAssetPanel::m_splitter11OnIdle ), NULL, this );
		}

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
/// Class PackageSelectionPanel
///////////////////////////////////////////////////////////////////////////////
class PackageSelectionPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* ErrorStaticText;
		wxStaticLine* m_staticline158;
		wxStaticText* m_staticText798;
		wxStaticText* m_staticText792;
		wxStaticText* m_staticText214;
		wxStaticText* m_staticText2141;

		// Virtual event handlers, override them in your derived class
		virtual void PackageClassSelection( wxCommandEvent& event ) { event.Skip(); }


	public:
		wxCheckListBox* RefinementPackagesCheckListBox;
		wxCheckBox* RemoveDuplicatesCheckbox;
		wxStaticText* ImportedParamsWarning;
		wxComboBox* SymmetryComboBox;
		NumericTextCtrl* MolecularWeightTextCtrl;
		NumericTextCtrl* LargestDimensionTextCtrl;

		PackageSelectionPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 700,500 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~PackageSelectionPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class CombinedClassSelectionPanel
///////////////////////////////////////////////////////////////////////////////
class CombinedClassSelectionPanel : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText883;
		wxStaticLine* m_staticline187;

	public:
		wxScrolledWindow* CombinedClassScrollWindow;
		wxBoxSizer* CombinedClassScrollSizer;

		CombinedClassSelectionPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 700,500 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~CombinedClassSelectionPanel();

};

///////////////////////////////////////////////////////////////////////////////
/// Class CombinedPackageRefinementPanel
///////////////////////////////////////////////////////////////////////////////
class CombinedPackageRefinementPanel : public wxPanel
{
	private:

	protected:
		wxStaticLine* m_staticline187;

	public:
		wxStaticText* SelectRefinementText;
		wxScrolledWindow* CombinedRefinementScrollWindow;
		wxBoxSizer* CombinedRefinementScrollSizer;

		CombinedPackageRefinementPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 700,500 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~CombinedPackageRefinementPanel();

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

