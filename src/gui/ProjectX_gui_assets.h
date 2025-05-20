///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
class ContainedParticleListControl;
class ContentsList;
class NumericTextCtrl;
class TemplateMatchesPackageListControl;
class VolumeAssetPickerComboPanel;

#include <wx/listctrl.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/button.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/sizer.h>
#include <wx/statline.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/dialog.h>
#include <wx/spinctrl.h>
#include <wx/combobox.h>
#include <wx/choice.h>
#include <wx/checkbox.h>
#include <wx/filepicker.h>
#include <wx/panel.h>
#include <wx/splitter.h>
#include <wx/scrolwin.h>
#include <wx/statbox.h>

///////////////////////////////////////////////////////////////////////////

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
		virtual void OnUpdateUI( wxUpdateUIEvent& event ) { event.Skip(); }
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
/// Class ResampleDialogParent
///////////////////////////////////////////////////////////////////////////////
class ResampleDialogParent : public wxDialog
{
	private:

	protected:
		wxStaticText* ResampleInfoText;
		wxStaticLine* m_staticline6;
		wxStaticText* m_staticText29;
		wxSpinCtrl* BoxSizeSpinCtrl;
		wxStaticText* NewPixelSizeText;
		wxStaticLine* m_staticline19;
		wxButton* CancelButton;
		wxButton* OKButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnBoxSizeSpinCtrl( wxSpinEvent& event ) { event.Skip(); }
		virtual void OnBoxSizeTextEnter( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCancel( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOK( wxCommandEvent& event ) { event.Skip(); }


	public:

		ResampleDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Resample Asset"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~ResampleDialogParent();

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
/// Class TemplateMatchesPackageAssetPanelParent
///////////////////////////////////////////////////////////////////////////////
class TemplateMatchesPackageAssetPanelParent : public wxPanel
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
		TemplateMatchesPackageListControl* RefinementPackageListCtrl;
		wxPanel* m_panel51;
		wxStaticText* ContainedParticlesStaticText;
		ContainedParticleListControl* ContainedParticlesListCtrl;
		wxStaticLine* m_staticline53;
		wxStaticText* m_staticText319;
		wxTextCtrl* StarFileNameText;

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


	public:

		TemplateMatchesPackageAssetPanelParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~TemplateMatchesPackageAssetPanelParent();

		void m_splitter11OnIdle( wxIdleEvent& )
		{
			m_splitter11->SetSashPosition( 600 );
			m_splitter11->Disconnect( wxEVT_IDLE, wxIdleEventHandler( TemplateMatchesPackageAssetPanelParent::m_splitter11OnIdle ), NULL, this );
		}

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
		wxBoxSizer* bSizer28;
		wxButton* ImportAsset;
		wxButton* RemoveSelectedAssetButton;
		wxButton* RemoveAllAssetsButton;
		wxButton* RenameAssetButton;
		wxButton* AddSelectedAssetButton;
		wxButton* DisplayButton;
		wxButton* ResampleButton;
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
		virtual void OnResampleClick( wxCommandEvent& event ) { event.Skip(); }


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

