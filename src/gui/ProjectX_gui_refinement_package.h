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
class NumericTextCtrl;
class ReferenceVolumesListControl;
class RefinementPackageListControl;

#include <wx/string.h>
#include <wx/combobox.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/sizer.h>
#include <wx/statbox.h>
#include <wx/stattext.h>
#include <wx/spinctrl.h>
#include <wx/checkbox.h>
#include <wx/filepicker.h>
#include <wx/button.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/panel.h>
#include <wx/dialog.h>
#include <wx/textctrl.h>
#include <wx/statline.h>
#include <wx/listctrl.h>
#include <wx/splitter.h>

///////////////////////////////////////////////////////////////////////////

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
		wxButton* BinButton;
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
		virtual void OnBinClick( wxCommandEvent& event ) { event.Skip(); }
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
/// Class BinningDialogParent
///////////////////////////////////////////////////////////////////////////////
class BinningDialogParent : public wxDialog
{
	private:

	protected:
		wxStaticText* BinningInfoText;
		wxStaticLine* m_staticline6;
		wxStaticText* m_staticText29;
		NumericTextCtrl* DesiredPixelSizeTextCtrl;
		wxStaticText* ActualPixelSizeText;
		wxStaticText* ActualBoxSizeText;
		wxButton* CancelButton;
		wxButton* OKButton;

		// Virtual event handlers, override them in your derived class
		virtual void OnCancel( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOK( wxCommandEvent& event ) { event.Skip(); }


	public:

		BinningDialogParent( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Bin Particle Stack"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_DIALOG_STYLE );

		~BinningDialogParent();

};

