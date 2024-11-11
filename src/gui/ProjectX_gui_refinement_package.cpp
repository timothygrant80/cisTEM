///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "my_controls.h"

#include "ProjectX_gui_refinement_package.h"

///////////////////////////////////////////////////////////////////////////

FrealignExportDialog::FrealignExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
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

	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer142;
	bSizer142 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText202 = new wxStaticText( m_panel38, wxID_ANY, wxT("Downsampling factor : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	bSizer142->Add( m_staticText202, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DownsamplingFactorSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 1 );
	bSizer142->Add( DownsamplingFactorSpinCtrl, 0, wxALL, 5 );


	bSizer141->Add( bSizer142, 0, wxEXPAND, 5 );


	bSizer135->Add( bSizer141, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer143;
	bSizer143 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText203 = new wxStaticText( m_panel38, wxID_ANY, wxT("Box size after downsampling : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText203->Wrap( -1 );
	bSizer143->Add( m_staticText203, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	BoxSizeSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 512 );
	bSizer143->Add( BoxSizeSpinCtrl, 0, wxALL, 5 );


	bSizer145->Add( bSizer143, 1, wxEXPAND, 5 );


	bSizer135->Add( bSizer145, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer140;
	bSizer140 = new wxBoxSizer( wxVERTICAL );

	NormalizeCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Normalize images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer140->Add( NormalizeCheckBox, 0, wxALL, 5 );


	bSizer135->Add( bSizer140, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer213;
	bSizer213 = new wxBoxSizer( wxVERTICAL );

	FlipCTFCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Flip CTF"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer213->Add( FlipCTFCheckBox, 0, wxALL, 5 );


	bSizer135->Add( bSizer213, 1, wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Output image stack") ), wxVERTICAL );

	OutputImageStackPicker = new wxFilePickerCtrl( sbSizer4->GetStaticBox(), wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("MRC files (*.mrc, *.mrcs)|*.mrc;*.mrcs"), wxDefaultPosition, wxDefaultSize, wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	sbSizer4->Add( OutputImageStackPicker, 0, wxALL|wxEXPAND, 5 );


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
	FlipCTFCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( FrealignExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnExportButtonClick ), NULL, this );
}

FrealignExportDialog::~FrealignExportDialog()
{
	// Disconnect Events
	FlipCTFCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( FrealignExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnExportButtonClick ), NULL, this );

}

RelionExportDialog::RelionExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
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

	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer142;
	bSizer142 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText202 = new wxStaticText( m_panel38, wxID_ANY, wxT("Downsampling factor : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	bSizer142->Add( m_staticText202, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DownsamplingFactorSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 1 );
	bSizer142->Add( DownsamplingFactorSpinCtrl, 0, wxALL|wxEXPAND, 5 );


	bSizer141->Add( bSizer142, 0, wxEXPAND, 5 );


	bSizer135->Add( bSizer141, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer143;
	bSizer143 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText203 = new wxStaticText( m_panel38, wxID_ANY, wxT("Box size after downsampling : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText203->Wrap( -1 );
	bSizer143->Add( m_staticText203, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	BoxSizeSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 512 );
	bSizer143->Add( BoxSizeSpinCtrl, 0, wxALL|wxEXPAND, 5 );


	bSizer145->Add( bSizer143, 1, wxEXPAND, 5 );


	bSizer135->Add( bSizer145, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer140;
	bSizer140 = new wxBoxSizer( wxVERTICAL );

	NormalizeCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Normalize images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer140->Add( NormalizeCheckBox, 0, wxALL, 5 );


	bSizer135->Add( bSizer140, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer1451;
	bSizer1451 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer1431;
	bSizer1431 = new wxBoxSizer( wxHORIZONTAL );

	particleRadiusStaticText = new wxStaticText( m_panel38, wxID_ANY, wxT("Particle radius (A) : "), wxDefaultPosition, wxDefaultSize, 0 );
	particleRadiusStaticText->Wrap( -1 );
	particleRadiusStaticText->Enable( false );

	bSizer1431->Add( particleRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	particleRadiusTextCtrl = new wxTextCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	particleRadiusTextCtrl->Enable( false );

	bSizer1431->Add( particleRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );


	bSizer1451->Add( bSizer1431, 1, wxEXPAND, 5 );


	bSizer135->Add( bSizer1451, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer213;
	bSizer213 = new wxBoxSizer( wxVERTICAL );

	FlipCTFCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Flip CTF"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer213->Add( FlipCTFCheckBox, 0, wxALL, 5 );


	bSizer135->Add( bSizer213, 1, wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Output image stack") ), wxVERTICAL );

	wxBoxSizer* bSizer229;
	bSizer229 = new wxBoxSizer( wxHORIZONTAL );

	OutputImageStackPicker = new wxFilePickerCtrl( sbSizer4->GetStaticBox(), wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("MRC files (*.mrcs)|*.mrcs"), wxDefaultPosition, wxDefaultSize, wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE );
	bSizer229->Add( OutputImageStackPicker, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	FileNameStaticText = new wxStaticText( sbSizer4->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FileNameStaticText->Wrap( -1 );
	bSizer229->Add( FileNameStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	sbSizer4->Add( bSizer229, 1, wxEXPAND, 5 );


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
	NormalizeCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnNormalizeCheckBox ), NULL, this );
	FlipCTFCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( RelionExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnExportButtonClick ), NULL, this );
}

RelionExportDialog::~RelionExportDialog()
{
	// Disconnect Events
	NormalizeCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnNormalizeCheckBox ), NULL, this );
	FlipCTFCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( RelionExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnExportButtonClick ), NULL, this );

}

RefinementPackageAssetPanel::RefinementPackageAssetPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer187;
	bSizer187 = new wxBoxSizer( wxVERTICAL );

	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );

	m_splitter11 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter11->Connect( wxEVT_IDLE, wxIdleEventHandler( RefinementPackageAssetPanel::m_splitter11OnIdle ), NULL, this );

	m_panel50 = new wxPanel( m_splitter11, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer188;
	bSizer188 = new wxBoxSizer( wxVERTICAL );

	m_staticText313 = new wxStaticText( m_panel50, wxID_ANY, wxT("Refinement Packages :"), wxDefaultPosition, wxDefaultSize, 0 );
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
	bSizer193->Add( ExportButton, 0, wxALL, 5 );

	CombineButton = new wxButton( m_panel50, wxID_ANY, wxT("Combine"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( CombineButton, 0, wxALL, 5 );

	BinButton = new wxButton( m_panel50, wxID_ANY, wxT("Resample"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( BinButton, 0, wxALL, 5 );


	bSizer145->Add( bSizer193, 0, wxEXPAND, 5 );

	RefinementPackageListCtrl = new RefinementPackageListControl( m_panel50, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
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

	wxBoxSizer* bSizer199;
	bSizer199 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText230 = new wxStaticText( m_panel51, wxID_ANY, wxT("Active 3D References :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText230->Wrap( -1 );
	bSizer199->Add( m_staticText230, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer199->Add( 0, 0, 1, wxEXPAND, 5 );

	DisplayStackButton = new wxButton( m_panel51, wxID_ANY, wxT("Display Stack"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer199->Add( DisplayStackButton, 0, wxALL, 5 );


	bSizer191->Add( bSizer199, 0, wxEXPAND, 5 );

	Active3DReferencesListCtrl = new ReferenceVolumesListControl( m_panel51, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer191->Add( Active3DReferencesListCtrl, 0, wxALL|wxEXPAND, 5 );


	m_panel51->SetSizer( bSizer191 );
	m_panel51->Layout();
	bSizer191->Fit( m_panel51 );
	m_splitter11->SplitVertically( m_panel50, m_panel51, 600 );
	bSizer187->Add( m_splitter11, 1, wxEXPAND, 5 );

	m_staticline53 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline53, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer192;
	bSizer192 = new wxBoxSizer( wxHORIZONTAL );

	wxGridSizer* gSizer12;
	gSizer12 = new wxGridSizer( 0, 6, 0, 0 );

	m_staticText319 = new wxStaticText( this, wxID_ANY, wxT("Stack Filename :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText319->Wrap( -1 );
	gSizer12->Add( m_staticText319, 0, wxALIGN_RIGHT|wxALL, 5 );

	StackFileNameText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StackFileNameText->Wrap( -1 );
	gSizer12->Add( StackFileNameText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );


	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );


	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );


	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText210 = new wxStaticText( this, wxID_ANY, wxT("Stack Box Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText210->Wrap( -1 );
	gSizer12->Add( m_staticText210, 0, wxALIGN_RIGHT|wxALL, 5 );

	StackBoxSizeText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StackBoxSizeText->Wrap( -1 );
	gSizer12->Add( StackBoxSizeText, 0, wxALL, 5 );

	m_staticText315 = new wxStaticText( this, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText315->Wrap( -1 );
	gSizer12->Add( m_staticText315, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	NumberofClassesText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberofClassesText->Wrap( -1 );
	gSizer12->Add( NumberofClassesText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticText279 = new wxStaticText( this, wxID_ANY, wxT("Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText279->Wrap( -1 );
	gSizer12->Add( m_staticText279, 0, wxALIGN_RIGHT|wxALL, 5 );

	SymmetryText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SymmetryText->Wrap( -1 );
	gSizer12->Add( SymmetryText, 0, wxALL, 5 );

	m_staticText281 = new wxStaticText( this, wxID_ANY, wxT("M.W. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText281->Wrap( -1 );
	gSizer12->Add( m_staticText281, 0, wxALIGN_RIGHT|wxALL, 5 );

	MolecularWeightText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MolecularWeightText->Wrap( -1 );
	gSizer12->Add( MolecularWeightText, 0, wxALL, 5 );

	m_staticText283 = new wxStaticText( this, wxID_ANY, wxT("Largest Dimension :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText283->Wrap( -1 );
	gSizer12->Add( m_staticText283, 0, wxALIGN_RIGHT|wxALL, 5 );

	LargestDimensionText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LargestDimensionText->Wrap( -1 );
	gSizer12->Add( LargestDimensionText, 0, wxALL, 5 );

	m_staticText317 = new wxStaticText( this, wxID_ANY, wxT("No. Refinements :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	gSizer12->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	NumberofRefinementsText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberofRefinementsText->Wrap( -1 );
	gSizer12->Add( NumberofRefinementsText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticText212 = new wxStaticText( this, wxID_ANY, wxT("Last Refinement ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText212->Wrap( -1 );
	gSizer12->Add( m_staticText212, 0, wxALIGN_RIGHT|wxALL, 5 );

	LastRefinementIDText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LastRefinementIDText->Wrap( -1 );
	gSizer12->Add( LastRefinementIDText, 0, wxALL, 5 );


	bSizer192->Add( gSizer12, 1, wxEXPAND, 5 );


	bSizer187->Add( bSizer192, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer187 );
	this->Layout();
	bSizer187->Fit( this );

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementPackageAssetPanel::OnUpdateUI ) );
	CreateButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCreateClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnRenameClick ), NULL, this );
	DeleteButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDeleteClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnImportClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnExportClick ), NULL, this );
	CombineButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCombineClick ), NULL, this );
	BinButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnResampleClick ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	DisplayStackButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDisplayStackButton ), NULL, this );
	Active3DReferencesListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnVolumeListItemActivated ), NULL, this );
}

RefinementPackageAssetPanel::~RefinementPackageAssetPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementPackageAssetPanel::OnUpdateUI ) );
	CreateButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCreateClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnRenameClick ), NULL, this );
	DeleteButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDeleteClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnImportClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnExportClick ), NULL, this );
	CombineButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCombineClick ), NULL, this );
	BinButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnResampleClick ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	DisplayStackButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDisplayStackButton ), NULL, this );
	Active3DReferencesListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnVolumeListItemActivated ), NULL, this );

}

ClassumSelectionCopyFromDialogParent::ClassumSelectionCopyFromDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer279;
	bSizer279 = new wxBoxSizer( wxVERTICAL );

	m_staticText414 = new wxStaticText( this, wxID_ANY, wxT("Please select a valid (same no. classums) selection :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText414->Wrap( -1 );
	bSizer279->Add( m_staticText414, 0, wxALL, 5 );

	m_staticline72 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer279->Add( m_staticline72, 0, wxEXPAND | wxALL, 5 );

	SelectionListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxSize( -1,100 ), wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer279->Add( SelectionListCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticline71 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer279->Add( m_staticline71, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer280;
	bSizer280 = new wxBoxSizer( wxHORIZONTAL );


	bSizer280->Add( 0, 0, 1, wxEXPAND, 5 );

	OkButton = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer280->Add( OkButton, 0, wxALL, 5 );

	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer280->Add( CancelButton, 0, wxALL, 5 );


	bSizer280->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer279->Add( bSizer280, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer279 );
	this->Layout();
	bSizer279->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	OkButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnOKButtonClick ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnCancelButtonClick ), NULL, this );
}

ClassumSelectionCopyFromDialogParent::~ClassumSelectionCopyFromDialogParent()
{
	// Disconnect Events
	OkButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnOKButtonClick ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnCancelButtonClick ), NULL, this );

}
