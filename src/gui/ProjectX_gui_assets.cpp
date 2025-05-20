///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "AssetPickerComboPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_assets.h"

///////////////////////////////////////////////////////////////////////////

VolumeImportDialog::VolumeImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 100, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );

	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );

	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );

	ClearButton = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( ClearButton, 33, wxALL, 5 );


	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );

	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );

	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxVERTICAL );


	bSizer26->Add( bSizer55, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );

	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );

	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );

	bSizer33->Add( ImportButton, 0, wxALL, 5 );


	bSizer26->Add( bSizer33, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer26 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( VolumeImportDialog::OnUpdateUI ) );
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ClearClick ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( VolumeImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( VolumeImportDialog::TextChanged ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ImportClick ), NULL, this );
}

VolumeImportDialog::~VolumeImportDialog()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( VolumeImportDialog::OnUpdateUI ) );
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ClearClick ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( VolumeImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( VolumeImportDialog::TextChanged ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ImportClick ), NULL, this );

}

ResampleDialogParent::ResampleDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer31;
	bSizer31 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxVERTICAL );

	ResampleInfoText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxST_NO_AUTORESIZE );
	ResampleInfoText->Wrap( -1 );
	bSizer34->Add( ResampleInfoText, 1, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );


	bSizer31->Add( bSizer34, 1, wxALIGN_CENTER_HORIZONTAL|wxALL|wxEXPAND, 5 );

	m_staticline6 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer31->Add( m_staticline6, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText29 = new wxStaticText( this, wxID_ANY, wxT("Desired Box Size:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText29->Wrap( -1 );
	bSizer33->Add( m_staticText29, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	BoxSizeSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 32, 1600, 192 );
	bSizer33->Add( BoxSizeSpinCtrl, 0, wxALL, 5 );


	bSizer31->Add( bSizer33, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxHORIZONTAL );

	NewPixelSizeText = new wxStaticText( this, wxID_ANY, wxT("New Pixel Size: "), wxDefaultPosition, wxDefaultSize, 0 );
	NewPixelSizeText->Wrap( -1 );
	bSizer38->Add( NewPixelSizeText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer31->Add( bSizer38, 0, wxALIGN_CENTER_HORIZONTAL, 5 );

	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer31->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxHORIZONTAL );

	CancelButton = new wxButton( this, wxID_CANCEL, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32->Add( CancelButton, 0, wxALL, 5 );

	OKButton = new wxButton( this, wxID_OK, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32->Add( OKButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );


	bSizer31->Add( bSizer32, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_TOP|wxALL, 5 );


	this->SetSizer( bSizer31 );
	this->Layout();
	bSizer31->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	BoxSizeSpinCtrl->Connect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( ResampleDialogParent::OnBoxSizeSpinCtrl ), NULL, this );
	BoxSizeSpinCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ResampleDialogParent::OnBoxSizeTextEnter ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ResampleDialogParent::OnCancel ), NULL, this );
	OKButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ResampleDialogParent::OnOK ), NULL, this );
}

ResampleDialogParent::~ResampleDialogParent()
{
	// Disconnect Events
	BoxSizeSpinCtrl->Disconnect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( ResampleDialogParent::OnBoxSizeSpinCtrl ), NULL, this );
	BoxSizeSpinCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ResampleDialogParent::OnBoxSizeTextEnter ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ResampleDialogParent::OnCancel ), NULL, this );
	OKButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ResampleDialogParent::OnOK ), NULL, this );

}

AtomicCoordinatesImportDialogParent::AtomicCoordinatesImportDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 100, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );

	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );

	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );

	ClearButton = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( ClearButton, 33, wxALL, 5 );


	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );

	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );


	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );

	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );

	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );

	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );

	bSizer33->Add( ImportButton, 0, wxALL, 5 );


	bSizer26->Add( bSizer33, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer26 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::ClearClick ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::ImportClick ), NULL, this );
}

AtomicCoordinatesImportDialogParent::~AtomicCoordinatesImportDialogParent()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::ClearClick ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesImportDialogParent::ImportClick ), NULL, this );

}

MovieImportDialog::MovieImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 0, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );

	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );

	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );

	s = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( s, 33, wxALL, 5 );


	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );

	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText19 = new wxStaticText( this, wxID_ANY, wxT("Voltage (kV) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19->Wrap( -1 );
	bSizer28->Add( m_staticText19, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	VoltageCombo = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	VoltageCombo->Append( wxT("300") );
	VoltageCombo->Append( wxT("200") );
	VoltageCombo->Append( wxT("120") );
	bSizer28->Add( VoltageCombo, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer28, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer291;
	bSizer291 = new wxBoxSizer( wxHORIZONTAL );

	EerSuperResFactorStaticText = new wxStaticText( this, wxID_ANY, wxT("EER super res. factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	EerSuperResFactorStaticText->Wrap( -1 );
	EerSuperResFactorStaticText->Enable( false );

	bSizer291->Add( EerSuperResFactorStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxString EerSuperResFactorChoiceChoices[] = { wxT("1"), wxT("2"), wxT("4") };
	int EerSuperResFactorChoiceNChoices = sizeof( EerSuperResFactorChoiceChoices ) / sizeof( wxString );
	EerSuperResFactorChoice = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, EerSuperResFactorChoiceNChoices, EerSuperResFactorChoiceChoices, 0 );
	EerSuperResFactorChoice->SetSelection( 0 );
	EerSuperResFactorChoice->Enable( false );

	bSizer291->Add( EerSuperResFactorChoice, 50, wxALL, 5 );


	bSizer26->Add( bSizer291, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );

	PixelSizeStaticText = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	bSizer29->Add( PixelSizeStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer2911;
	bSizer2911 = new wxBoxSizer( wxHORIZONTAL );

	EerNumberOfFramesStaticText = new wxStaticText( this, wxID_ANY, wxT("EER number of frames to average :"), wxDefaultPosition, wxDefaultSize, 0 );
	EerNumberOfFramesStaticText->Wrap( -1 );
	EerNumberOfFramesStaticText->Enable( false );

	bSizer2911->Add( EerNumberOfFramesStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	EerNumberOfFramesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100000, 25 );
	EerNumberOfFramesSpinCtrl->Enable( false );

	bSizer2911->Add( EerNumberOfFramesSpinCtrl, 50, wxALL, 5 );


	bSizer26->Add( bSizer2911, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxHORIZONTAL );

	ExposurePerFrameStaticText = new wxStaticText( this, wxID_ANY, wxT("Exposure per frame (e¯/Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	ExposurePerFrameStaticText->Wrap( -1 );
	bSizer32->Add( ExposurePerFrameStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DoseText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32->Add( DoseText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer32, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer2292;
	bSizer2292 = new wxBoxSizer( wxHORIZONTAL );

	ApplyDarkImageCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Apply dark image :"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2292->Add( ApplyDarkImageCheckbox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DarkFilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("Any supported type (*.mrc;*.tif;*.dm;*.dm3;*.dm4)|*.mrc;*.tif;*.dm;*.dm3;*.dm4|MRC files (*.mrc) | *.mrc |TIFF files (*.tif) | *.tif | DM files (*.dm;*.dm3;*.dm4)|*.dm;*.dm3;*.dm4"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	DarkFilePicker->Enable( false );

	bSizer2292->Add( DarkFilePicker, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer2292, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer229;
	bSizer229 = new wxBoxSizer( wxHORIZONTAL );

	ApplyGainImageCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Apply gain image :"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer229->Add( ApplyGainImageCheckbox, 50, wxALL, 5 );

	GainFilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("Any supported type (*.mrc;*.tif;*.dm;*.dm3;*.dm4)|*.mrc;*.tif;*.dm;*.dm3;*.dm4|MRC files (*.mrc) | *.mrc |TIFF files (*.tif) | *.tif | DM files (*.dm;*.dm3;*.dm4)|*.dm;*.dm3;*.dm4"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	GainFilePicker->Enable( false );

	bSizer229->Add( GainFilePicker, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer229, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer3211;
	bSizer3211 = new wxBoxSizer( wxHORIZONTAL );

	ResampleMoviesCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Resample movies during processing"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3211->Add( ResampleMoviesCheckBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer3211->Add( 0, 0, 50, wxEXPAND, 5 );


	bSizer26->Add( bSizer3211, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer2291;
	bSizer2291 = new wxBoxSizer( wxHORIZONTAL );

	DesiredPixelSizeStaticText = new wxStaticText( this, wxID_ANY, wxT("        Desired pixel size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DesiredPixelSizeStaticText->Wrap( -1 );
	DesiredPixelSizeStaticText->Enable( false );

	bSizer2291->Add( DesiredPixelSizeStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DesiredPixelSizeTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DesiredPixelSizeTextCtrl->Enable( false );

	bSizer2291->Add( DesiredPixelSizeTextCtrl, 50, wxALL, 5 );


	bSizer26->Add( bSizer2291, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer282;
	bSizer282 = new wxBoxSizer( wxVERTICAL );

	CorrectMagDistortionCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Correct magnification distortion"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer282->Add( CorrectMagDistortionCheckBox, 0, wxALL, 5 );

	wxBoxSizer* bSizer283;
	bSizer283 = new wxBoxSizer( wxHORIZONTAL );

	DistortionAngleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Distortion Angle (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DistortionAngleStaticText->Wrap( -1 );
	DistortionAngleStaticText->Enable( false );

	bSizer283->Add( DistortionAngleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DistortionAngleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DistortionAngleTextCtrl->Enable( false );

	bSizer283->Add( DistortionAngleTextCtrl, 50, wxALL, 5 );


	bSizer282->Add( bSizer283, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer2831;
	bSizer2831 = new wxBoxSizer( wxHORIZONTAL );

	MajorScaleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Major Scale  :"), wxDefaultPosition, wxDefaultSize, 0 );
	MajorScaleStaticText->Wrap( -1 );
	MajorScaleStaticText->Enable( false );

	bSizer2831->Add( MajorScaleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MajorScaleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.000"), wxDefaultPosition, wxDefaultSize, 0 );
	MajorScaleTextCtrl->Enable( false );

	bSizer2831->Add( MajorScaleTextCtrl, 50, wxALL, 5 );


	bSizer282->Add( bSizer2831, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer28311;
	bSizer28311 = new wxBoxSizer( wxHORIZONTAL );

	MinorScaleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Minor Scale  :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinorScaleStaticText->Wrap( -1 );
	MinorScaleStaticText->Enable( false );

	bSizer28311->Add( MinorScaleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MinorScaleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.000"), wxDefaultPosition, wxDefaultSize, 0 );
	MinorScaleTextCtrl->Enable( false );

	bSizer28311->Add( MinorScaleTextCtrl, 50, wxALL, 5 );


	bSizer282->Add( bSizer28311, 1, wxEXPAND, 5 );


	bSizer26->Add( bSizer282, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer3212;
	bSizer3212 = new wxBoxSizer( wxHORIZONTAL );

	MoviesHaveInvertedContrast = new wxCheckBox( this, wxID_ANY, wxT("Particles are white"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3212->Add( MoviesHaveInvertedContrast, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer3212->Add( 0, 0, 50, wxEXPAND, 5 );


	bSizer26->Add( bSizer3212, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer32121;
	bSizer32121 = new wxBoxSizer( wxHORIZONTAL );

	SkipFullIntegrityCheck = new wxCheckBox( this, wxID_ANY, wxT("Skip full integrity check of frames"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32121->Add( SkipFullIntegrityCheck, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer32121->Add( 0, 0, 50, wxEXPAND, 5 );


	bSizer26->Add( bSizer32121, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer32122;
	bSizer32122 = new wxBoxSizer( wxHORIZONTAL );

	ImportMetadataCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Import Metadata"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32122->Add( ImportMetadataCheckbox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer32122->Add( 0, 0, 50, wxEXPAND, 5 );


	bSizer26->Add( bSizer32122, 1, wxEXPAND, 5 );

	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );

	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );

	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );

	bSizer33->Add( ImportButton, 0, wxALL, 5 );


	bSizer26->Add( bSizer33, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer26 );
	this->Layout();
	bSizer26->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	s->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	ApplyDarkImageCheckbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	DarkFilePicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ApplyGainImageCheckbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	GainFilePicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ResampleMoviesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnResampleMoviesCheckBox ), NULL, this );
	DesiredPixelSizeTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CorrectMagDistortionCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnCorrectMagDistortionCheckBox ), NULL, this );
	DistortionAngleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MajorScaleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MinorScaleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MoviesHaveInvertedContrast->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesHaveInvertedContrastCheckBox ), NULL, this );
	SkipFullIntegrityCheck->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnSkipFullIntegrityCheckCheckBox ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ImportClick ), NULL, this );
}

MovieImportDialog::~MovieImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	s->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	ApplyDarkImageCheckbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	DarkFilePicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ApplyGainImageCheckbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	GainFilePicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ResampleMoviesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnResampleMoviesCheckBox ), NULL, this );
	DesiredPixelSizeTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CorrectMagDistortionCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnCorrectMagDistortionCheckBox ), NULL, this );
	DistortionAngleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MajorScaleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MinorScaleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MoviesHaveInvertedContrast->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesHaveInvertedContrastCheckBox ), NULL, this );
	SkipFullIntegrityCheck->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnSkipFullIntegrityCheckCheckBox ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ImportClick ), NULL, this );

}

ImageImportDialog::ImageImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 100, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );

	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );

	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );

	ClearButton = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( ClearButton, 33, wxALL, 5 );


	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );

	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText19 = new wxStaticText( this, wxID_ANY, wxT("Voltage (kV) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19->Wrap( -1 );
	bSizer28->Add( m_staticText19, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	VoltageCombo = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	VoltageCombo->Append( wxT("300") );
	VoltageCombo->Append( wxT("200") );
	VoltageCombo->Append( wxT("120") );
	bSizer28->Add( VoltageCombo, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer28, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer3212;
	bSizer3212 = new wxBoxSizer( wxVERTICAL );

	SaveScaledSumCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Create Scaled Version to Speed up Visualisation?"), wxDefaultPosition, wxDefaultSize, 0 );
	SaveScaledSumCheckbox->SetValue(true);
	bSizer3212->Add( SaveScaledSumCheckbox, 0, wxALL, 5 );

	ImagesHaveInvertedContrast = new wxCheckBox( this, wxID_ANY, wxT("Particles are white"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3212->Add( ImagesHaveInvertedContrast, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer26->Add( bSizer3212, 1, wxEXPAND, 5 );

	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );

	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );

	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );

	bSizer33->Add( ImportButton, 0, wxALL, 5 );


	bSizer26->Add( bSizer33, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer26 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	ImagesHaveInvertedContrast->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( ImageImportDialog::OnImagesHaveInvertedContrastCheckBox ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ImportClick ), NULL, this );
}

ImageImportDialog::~ImageImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	ImagesHaveInvertedContrast->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( ImageImportDialog::OnImagesHaveInvertedContrastCheckBox ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ImportClick ), NULL, this );

}

TemplateMatchesPackageAssetPanelParent::TemplateMatchesPackageAssetPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer187;
	bSizer187 = new wxBoxSizer( wxVERTICAL );

	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );

	m_splitter11 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter11->Connect( wxEVT_IDLE, wxIdleEventHandler( TemplateMatchesPackageAssetPanelParent::m_splitter11OnIdle ), NULL, this );

	m_panel50 = new wxPanel( m_splitter11, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer188;
	bSizer188 = new wxBoxSizer( wxVERTICAL );

	m_staticText313 = new wxStaticText( m_panel50, wxID_ANY, wxT("Template Matches Packages:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText313->Wrap( -1 );
	bSizer188->Add( m_staticText313, 0, wxALL, 5 );

	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer193;
	bSizer193 = new wxBoxSizer( wxVERTICAL );

	CreateButton = new wxButton( m_panel50, wxID_ANY, wxT("Create"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( CreateButton, 0, wxALL, 5 );

	RenameButton = new wxButton( m_panel50, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( RenameButton, 0, wxALL, 5 );

	DeleteButton = new wxButton( m_panel50, wxID_ANY, wxT("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( DeleteButton, 0, wxALL, 5 );

	m_staticline122 = new wxStaticLine( m_panel50, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer193->Add( m_staticline122, 0, wxEXPAND | wxALL, 5 );

	ImportButton = new wxButton( m_panel50, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( ImportButton, 0, wxALL, 5 );

	ExportButton = new wxButton( m_panel50, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );
	ExportButton->Hide();

	bSizer193->Add( ExportButton, 0, wxALL, 5 );

	CombineButton = new wxButton( m_panel50, wxID_ANY, wxT("Combine"), wxDefaultPosition, wxDefaultSize, 0 );
	CombineButton->Hide();

	bSizer193->Add( CombineButton, 0, wxALL, 5 );


	bSizer145->Add( bSizer193, 0, wxEXPAND, 5 );

	RefinementPackageListCtrl = new TemplateMatchesPackageListControl( m_panel50, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer145->Add( RefinementPackageListCtrl, 1, wxALL|wxEXPAND, 5 );


	bSizer188->Add( bSizer145, 1, wxEXPAND, 5 );


	m_panel50->SetSizer( bSizer188 );
	m_panel50->Layout();
	bSizer188->Fit( m_panel50 );
	m_panel51 = new wxPanel( m_splitter11, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer191;
	bSizer191 = new wxBoxSizer( wxVERTICAL );

	ContainedParticlesStaticText = new wxStaticText( m_panel51, wxID_ANY, wxT("Contained Particles :"), wxDefaultPosition, wxDefaultSize, 0 );
	ContainedParticlesStaticText->Wrap( -1 );
	bSizer191->Add( ContainedParticlesStaticText, 0, wxALL, 5 );

	ContainedParticlesListCtrl = new ContainedParticleListControl( m_panel51, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer191->Add( ContainedParticlesListCtrl, 1, wxALL|wxEXPAND, 5 );


	m_panel51->SetSizer( bSizer191 );
	m_panel51->Layout();
	bSizer191->Fit( m_panel51 );
	m_splitter11->SplitVertically( m_panel50, m_panel51, 600 );
	bSizer187->Add( m_splitter11, 1, wxEXPAND, 5 );

	m_staticline53 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline53, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer192;
	bSizer192 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText319 = new wxStaticText( this, wxID_ANY, wxT("Starfile Filename :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText319->Wrap( -1 );
	bSizer192->Add( m_staticText319, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	StarFileNameText = new wxTextCtrl( this, wxID_ANY, wxT("None"), wxDefaultPosition, wxDefaultSize, wxTE_READONLY );
	StarFileNameText->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_WINDOWTEXT ) );
	StarFileNameText->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_WINDOWFRAME ) );

	bSizer192->Add( StarFileNameText, 1, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );


	bSizer187->Add( bSizer192, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer187 );
	this->Layout();
	bSizer187->Fit( this );

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( TemplateMatchesPackageAssetPanelParent::OnUpdateUI ) );
	CreateButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnCreateClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnRenameClick ), NULL, this );
	DeleteButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnDeleteClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnImportClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnExportClick ), NULL, this );
	CombineButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnCombineClick ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
}

TemplateMatchesPackageAssetPanelParent::~TemplateMatchesPackageAssetPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( TemplateMatchesPackageAssetPanelParent::OnUpdateUI ) );
	CreateButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnCreateClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnRenameClick ), NULL, this );
	DeleteButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnDeleteClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnImportClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnExportClick ), NULL, this );
	CombineButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( TemplateMatchesPackageAssetPanelParent::OnCombineClick ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( TemplateMatchesPackageAssetPanelParent::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( TemplateMatchesPackageAssetPanelParent::MouseVeto ), NULL, this );

}

AssetPanelParent::AssetPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	this->SetMinSize( wxSize( 680,400 ) );

	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );

	m_staticline5 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer15->Add( m_staticline5, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );

	SplitterWindow = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	SplitterWindow->SetSashGravity( 0.2 );
	SplitterWindow->Connect( wxEVT_IDLE, wxIdleEventHandler( AssetPanelParent::SplitterWindowOnIdle ), NULL, this );

	LeftPanel = new wxPanel( SplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );

	m_staticText18 = new wxStaticText( LeftPanel, wxID_ANY, wxT("Groups:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText18->Wrap( -1 );
	bSizer27->Add( m_staticText18, 0, wxALL, 5 );

	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );

	GroupListBox = new wxListCtrl( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer221->Add( GroupListBox, 1, wxALL|wxEXPAND, 5 );


	bSizer27->Add( bSizer221, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxHORIZONTAL );

	AddGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( AddGroupButton, 0, wxALL, 5 );

	RenameGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	RenameGroupButton->Hide();

	bSizer49->Add( RenameGroupButton, 0, wxALL, 5 );

	RemoveGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( RemoveGroupButton, 0, wxALL, 5 );

	InvertGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Invert"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( InvertGroupButton, 0, wxALL, 5 );

	NewFromParentButton = new wxButton( LeftPanel, wxID_ANY, wxT("New From Parent"), wxDefaultPosition, wxDefaultSize, 0 );
	NewFromParentButton->Enable( false );

	bSizer49->Add( NewFromParentButton, 0, wxALL, 5 );


	bSizer27->Add( bSizer49, 0, wxEXPAND, 5 );


	LeftPanel->SetSizer( bSizer27 );
	LeftPanel->Layout();
	bSizer27->Fit( LeftPanel );
	m_panel3 = new wxPanel( SplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxVERTICAL );

	AssetTypeText = new wxStaticText( m_panel3, wxID_ANY, wxT("Asset Type :"), wxDefaultPosition, wxDefaultSize, 0 );
	AssetTypeText->Wrap( -1 );
	bSizer30->Add( AssetTypeText, 0, wxALL, 5 );

	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );

	ContentsListBox = new ContentsList( m_panel3, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_ALIGN_LEFT|wxLC_NO_SORT_HEADER|wxLC_REPORT|wxLC_VIRTUAL|wxLC_VRULES );
	bSizer25->Add( ContentsListBox, 100, wxALL|wxEXPAND, 5 );

	bSizer28 = new wxBoxSizer( wxHORIZONTAL );

	ImportAsset = new wxButton( m_panel3, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer28->Add( ImportAsset, 0, wxALL|wxEXPAND, 5 );

	RemoveSelectedAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	RemoveSelectedAssetButton->Enable( false );

	bSizer28->Add( RemoveSelectedAssetButton, 0, wxALL|wxEXPAND, 5 );

	RemoveAllAssetsButton = new wxButton( m_panel3, wxID_ANY, wxT("Remove All"), wxDefaultPosition, wxDefaultSize, 0 );
	RemoveAllAssetsButton->Enable( false );

	bSizer28->Add( RemoveAllAssetsButton, 0, wxALL|wxEXPAND, 5 );

	RenameAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	RenameAssetButton->Enable( false );

	bSizer28->Add( RenameAssetButton, 0, wxALL, 5 );

	AddSelectedAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Add To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	AddSelectedAssetButton->Enable( false );

	bSizer28->Add( AddSelectedAssetButton, 0, wxALL|wxEXPAND, 5 );

	DisplayButton = new wxButton( m_panel3, wxID_ANY, wxT("Display"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer28->Add( DisplayButton, 0, wxALL, 5 );

	ResampleButton = new wxButton( m_panel3, wxID_ANY, wxT("Resample"), wxDefaultPosition, wxDefaultSize, 0 );
	ResampleButton->Enable( false );

	bSizer28->Add( ResampleButton, 0, wxALL|wxEXPAND, 5 );


	bSizer25->Add( bSizer28, 0, wxEXPAND, 5 );


	bSizer30->Add( bSizer25, 100, wxEXPAND, 5 );


	m_panel3->SetSizer( bSizer30 );
	m_panel3->Layout();
	bSizer30->Fit( m_panel3 );
	SplitterWindow->SplitVertically( LeftPanel, m_panel3, 405 );
	bSizer20->Add( SplitterWindow, 100, wxEXPAND, 5 );


	bSizer15->Add( bSizer20, 1, wxEXPAND, 5 );

	m_staticline6 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer15->Add( m_staticline6, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );

	wxGridSizer* gSizer1;
	gSizer1 = new wxGridSizer( 4, 6, 0, 0 );

	Label0Title = new wxStaticText( this, wxID_ANY, wxT("Label 0 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label0Title->Wrap( -1 );
	Label0Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label0Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label0Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label0Text->Wrap( -1 );
	gSizer1->Add( Label0Text, 0, wxALIGN_LEFT|wxALL, 5 );


	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );


	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );


	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );


	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	Label1Title = new wxStaticText( this, wxID_ANY, wxT("Label 1 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label1Title->Wrap( -1 );
	Label1Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label1Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label1Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label1Text->Wrap( -1 );
	gSizer1->Add( Label1Text, 0, wxALL, 5 );

	Label2Title = new wxStaticText( this, wxID_ANY, wxT("Label 2 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label2Title->Wrap( -1 );
	Label2Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label2Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label2Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label2Text->Wrap( -1 );
	gSizer1->Add( Label2Text, 0, wxALIGN_LEFT|wxALL, 5 );

	Label3Title = new wxStaticText( this, wxID_ANY, wxT("Label 3 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label3Title->Wrap( -1 );
	Label3Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label3Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label3Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label3Text->Wrap( -1 );
	gSizer1->Add( Label3Text, 0, wxALIGN_LEFT|wxALL, 5 );

	Label4Title = new wxStaticText( this, wxID_ANY, wxT("Label 4 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label4Title->Wrap( -1 );
	Label4Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label4Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label4Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label4Text->Wrap( -1 );
	gSizer1->Add( Label4Text, 0, wxALIGN_LEFT|wxALL, 5 );

	Label5Title = new wxStaticText( this, wxID_ANY, wxT("Label 5 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label5Title->Wrap( -1 );
	Label5Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label5Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label5Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label5Text->Wrap( -1 );
	gSizer1->Add( Label5Text, 0, wxALL, 5 );

	Label6Title = new wxStaticText( this, wxID_ANY, wxT("Label 6 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label6Title->Wrap( -1 );
	Label6Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label6Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label6Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label6Text->Wrap( -1 );
	gSizer1->Add( Label6Text, 0, wxALIGN_LEFT|wxALL, 5 );

	Label7Title = new wxStaticText( this, wxID_ANY, wxT("Label 7 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label7Title->Wrap( -1 );
	Label7Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label7Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label7Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label7Text->Wrap( -1 );
	gSizer1->Add( Label7Text, 0, wxALIGN_LEFT|wxALL, 5 );

	Label8Title = new wxStaticText( this, wxID_ANY, wxT("Label 8 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label8Title->Wrap( -1 );
	Label8Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label8Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label8Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label8Text->Wrap( -1 );
	gSizer1->Add( Label8Text, 0, wxALL, 5 );

	Label9Title = new wxStaticText( this, wxID_ANY, wxT("Label 9 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label9Title->Wrap( -1 );
	Label9Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	gSizer1->Add( Label9Title, 0, wxALIGN_RIGHT|wxALL, 5 );

	Label9Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label9Text->Wrap( -1 );
	gSizer1->Add( Label9Text, 0, wxALL, 5 );


	bSizer34->Add( gSizer1, 90, wxEXPAND, 5 );


	bSizer15->Add( bSizer34, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer15 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetPanelParent::OnUpdateUI ) );
	GroupListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( AssetPanelParent::OnBeginEdit ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( AssetPanelParent::OnEndEdit ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetPanelParent::OnGroupActivated ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( AssetPanelParent::OnGroupFocusChange ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	AddGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::NewGroupClick ), NULL, this );
	RenameGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RenameGroupClick ), NULL, this );
	RemoveGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RemoveGroupClick ), NULL, this );
	InvertGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::InvertGroupClick ), NULL, this );
	NewFromParentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::NewFromParentClick ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetPanelParent::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetPanelParent::OnAssetActivated ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( AssetPanelParent::OnContentsSelected ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( AssetPanelParent::OnMotion ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ImportAsset->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::ImportAssetClick ), NULL, this );
	RemoveSelectedAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RemoveAssetClick ), NULL, this );
	RemoveAllAssetsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RemoveAllAssetsClick ), NULL, this );
	RenameAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RenameAssetClick ), NULL, this );
	AddSelectedAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::AddSelectedAssetClick ), NULL, this );
	DisplayButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::OnDisplayButtonClick ), NULL, this );
	ResampleButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::OnResampleClick ), NULL, this );
}

AssetPanelParent::~AssetPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetPanelParent::OnUpdateUI ) );
	GroupListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( AssetPanelParent::OnBeginEdit ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( AssetPanelParent::OnEndEdit ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetPanelParent::OnGroupActivated ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( AssetPanelParent::OnGroupFocusChange ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	AddGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::NewGroupClick ), NULL, this );
	RenameGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RenameGroupClick ), NULL, this );
	RemoveGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RemoveGroupClick ), NULL, this );
	InvertGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::InvertGroupClick ), NULL, this );
	NewFromParentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::NewFromParentClick ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetPanelParent::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetPanelParent::OnAssetActivated ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( AssetPanelParent::OnContentsSelected ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( AssetPanelParent::OnMotion ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetPanelParent::MouseVeto ), NULL, this );
	ImportAsset->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::ImportAssetClick ), NULL, this );
	RemoveSelectedAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RemoveAssetClick ), NULL, this );
	RemoveAllAssetsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RemoveAllAssetsClick ), NULL, this );
	RenameAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::RenameAssetClick ), NULL, this );
	AddSelectedAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::AddSelectedAssetClick ), NULL, this );
	DisplayButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::OnDisplayButtonClick ), NULL, this );
	ResampleButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPanelParent::OnResampleClick ), NULL, this );

}

RenameDialog::RenameDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );

	MainBoxSizer = new wxBoxSizer( wxVERTICAL );

	m_staticText246 = new wxStaticText( this, wxID_ANY, wxT("Set Asset Names :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText246->Wrap( -1 );
	MainBoxSizer->Add( m_staticText246, 0, wxALL, 5 );

	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );

	RenameScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	RenameScrollPanel->SetScrollRate( 5, 5 );
	RenameBoxSizer = new wxBoxSizer( wxVERTICAL );


	RenameScrollPanel->SetSizer( RenameBoxSizer );
	RenameScrollPanel->Layout();
	RenameBoxSizer->Fit( RenameScrollPanel );
	MainBoxSizer->Add( RenameScrollPanel, 1, wxEXPAND, 5 );

	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );

	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );

	RenameButton = new wxButton( this, wxID_ANY, wxT("Set Name"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( RenameButton, 0, wxALL, 5 );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );


	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );


	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnCancelClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnRenameClick ), NULL, this );
}

RenameDialog::~RenameDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnCancelClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnRenameClick ), NULL, this );

}

VolumeChooserDialog::VolumeChooserDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );

	MainBoxSizer = new wxBoxSizer( wxVERTICAL );

	m_staticText246 = new wxStaticText( this, wxID_ANY, wxT("Select New Volume :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText246->Wrap( -1 );
	MainBoxSizer->Add( m_staticText246, 0, wxALL, 5 );

	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );

	ComboBox = new VolumeAssetPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ComboBox->SetMinSize( wxSize( 350,-1 ) );

	MainBoxSizer->Add( ComboBox, 1, wxEXPAND | wxALL, 5 );

	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );

	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );

	SetButton = new wxButton( this, wxID_ANY, wxT("Set Reference"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( SetButton, 0, wxALL, 5 );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );


	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );


	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnCancelClick ), NULL, this );
	SetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnRenameClick ), NULL, this );
}

VolumeChooserDialog::~VolumeChooserDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnCancelClick ), NULL, this );
	SetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnRenameClick ), NULL, this );

}

ParticlePositionExportDialog::ParticlePositionExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer133;
	bSizer133 = new wxBoxSizer( wxVERTICAL );

	m_panel38 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer135;
	bSizer135 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer3;
	sbSizer3 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Export from these images") ), wxVERTICAL );

	GroupComboBox = new wxComboBox( sbSizer3->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );


	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );

	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Destination directory") ), wxVERTICAL );

	DestinationDirectoryPickerCtrl = new wxDirPickerCtrl( sbSizer4->GetStaticBox(), wxID_ANY, wxEmptyString, wxT("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	sbSizer4->Add( DestinationDirectoryPickerCtrl, 0, wxALL|wxEXPAND, 5 );


	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );


	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );

	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	WarningText->Wrap( -1 );
	WarningText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	WarningText->Hide();

	bSizer135->Add( WarningText, 0, wxALL, 5 );

	wxBoxSizer* bSizer137;
	bSizer137 = new wxBoxSizer( wxHORIZONTAL );


	bSizer137->Add( 0, 0, 1, wxEXPAND, 5 );

	CancelButton = new wxButton( m_panel38, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer137->Add( CancelButton, 0, wxALL, 5 );

	ExportButton = new wxButton( m_panel38, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );

	ExportButton->SetDefault();
	ExportButton->Enable( false );

	bSizer137->Add( ExportButton, 0, wxALL, 5 );


	bSizer135->Add( bSizer137, 0, wxEXPAND, 5 );


	m_panel38->SetSizer( bSizer135 );
	m_panel38->Layout();
	bSizer135->Fit( m_panel38 );
	bSizer133->Add( m_panel38, 0, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer133 );
	this->Layout();
	bSizer133->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	DestinationDirectoryPickerCtrl->Connect( wxEVT_COMMAND_DIRPICKER_CHANGED, wxFileDirPickerEventHandler( ParticlePositionExportDialog::OnDirChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnExportButtonClick ), NULL, this );
}

ParticlePositionExportDialog::~ParticlePositionExportDialog()
{
	// Disconnect Events
	DestinationDirectoryPickerCtrl->Disconnect( wxEVT_COMMAND_DIRPICKER_CHANGED, wxFileDirPickerEventHandler( ParticlePositionExportDialog::OnDirChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnExportButtonClick ), NULL, this );

}

AtomicCoordinatesChooserDialogParent::AtomicCoordinatesChooserDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );

	MainBoxSizer = new wxBoxSizer( wxVERTICAL );

	m_staticText246 = new wxStaticText( this, wxID_ANY, wxT("Select New Volume :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText246->Wrap( -1 );
	MainBoxSizer->Add( m_staticText246, 0, wxALL, 5 );

	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );

	ComboBox = new VolumeAssetPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ComboBox->SetMinSize( wxSize( 350,-1 ) );

	MainBoxSizer->Add( ComboBox, 1, wxEXPAND | wxALL, 5 );

	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );

	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );

	SetButton = new wxButton( this, wxID_ANY, wxT("Set Reference"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( SetButton, 0, wxALL, 5 );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );


	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );


	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesChooserDialogParent::OnCancelClick ), NULL, this );
	SetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesChooserDialogParent::OnRenameClick ), NULL, this );
}

AtomicCoordinatesChooserDialogParent::~AtomicCoordinatesChooserDialogParent()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesChooserDialogParent::OnCancelClick ), NULL, this );
	SetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AtomicCoordinatesChooserDialogParent::OnRenameClick ), NULL, this );

}
