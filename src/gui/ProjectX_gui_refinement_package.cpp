///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "AssetPickerComboPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_refinement_package.h"

///////////////////////////////////////////////////////////////////////////

CombineRefinementPackagesWizardParent::CombineRefinementPackagesWizardParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style )
{
	this->Create( parent, id, title, bitmap, pos, style );

	this->SetSizeHints( wxSize( -1,-1 ), wxSize( -1,-1 ) );


	this->Centre( wxBOTH );


	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( CombineRefinementPackagesWizardParent::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( CombineRefinementPackagesWizardParent::PageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( CombineRefinementPackagesWizardParent::PageChanging ) );
}

CombineRefinementPackagesWizardParent::~CombineRefinementPackagesWizardParent()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( CombineRefinementPackagesWizardParent::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( CombineRefinementPackagesWizardParent::PageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( CombineRefinementPackagesWizardParent::PageChanging ) );

	m_pages.Clear();
}

ExportRefinementPackageWizardParent::ExportRefinementPackageWizardParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style )
{
	this->Create( parent, id, title, bitmap, pos, style );

	this->SetSizeHints( wxSize( 700,400 ), wxDefaultSize );

	wxWizardPageSimple* ChooseParamsPage = new wxWizardPageSimple( this );
	m_pages.Add( ChooseParamsPage );

	wxBoxSizer* bSizer39311;
	bSizer39311 = new wxBoxSizer( wxVERTICAL );

	m_staticText46511 = new wxStaticText( ChooseParamsPage, wxID_ANY, wxT("Export which parameters? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46511->Wrap( -1 );
	bSizer39311->Add( m_staticText46511, 0, wxALL, 5 );

	m_staticline10511 = new wxStaticLine( ChooseParamsPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer39311->Add( m_staticline10511, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer419;
	bSizer419 = new wxBoxSizer( wxHORIZONTAL );

	ParameterSelectPanel = new RefinementPickerComboPanel( ChooseParamsPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer419->Add( ParameterSelectPanel, 1, wxEXPAND | wxALL, 5 );

	m_staticline123 = new wxStaticLine( ChooseParamsPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer419->Add( m_staticline123, 0, wxEXPAND | wxALL, 5 );

	ClassComboBox = new wxComboBox( ChooseParamsPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	ClassComboBox->SetMinSize( wxSize( 150,-1 ) );

	bSizer419->Add( ClassComboBox, 0, wxALL, 5 );


	bSizer39311->Add( bSizer419, 0, wxEXPAND, 5 );


	ChooseParamsPage->SetSizer( bSizer39311 );
	ChooseParamsPage->Layout();
	bSizer39311->Fit( ChooseParamsPage );
	wxWizardPageSimple* ExportTypePage = new wxWizardPageSimple( this );
	m_pages.Add( ExportTypePage );

	wxBoxSizer* bSizer3931;
	bSizer3931 = new wxBoxSizer( wxVERTICAL );

	m_staticText4651 = new wxStaticText( ExportTypePage, wxID_ANY, wxT("Export to what format? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4651->Wrap( -1 );
	bSizer3931->Add( m_staticText4651, 0, wxALL, 5 );

	m_staticline1051 = new wxStaticLine( ExportTypePage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer3931->Add( m_staticline1051, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer3941;
	bSizer3941 = new wxBoxSizer( wxVERTICAL );

	FrealignRadioButton = new wxRadioButton( ExportTypePage, wxID_ANY, wxT("Frealign"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( FrealignRadioButton, 0, wxALL, 5 );

	RelionRadioButton = new wxRadioButton( ExportTypePage, wxID_ANY, wxT("Relion (Legacy)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( RelionRadioButton, 0, wxALL, 5 );

	Relion3RadioButton = new wxRadioButton( ExportTypePage, wxID_ANY, wxT("Relion-3.1 (Currently only supports single optics group)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( Relion3RadioButton, 0, wxALL, 5 );


	bSizer3931->Add( bSizer3941, 1, wxEXPAND, 5 );


	ExportTypePage->SetSizer( bSizer3931 );
	ExportTypePage->Layout();
	bSizer3931->Fit( ExportTypePage );
	wxWizardPageSimple* GetPathPage = new wxWizardPageSimple( this );
	m_pages.Add( GetPathPage );

	GetPathPage->SetMinSize( wxSize( 600,300 ) );

	wxBoxSizer* bSizer471;
	bSizer471 = new wxBoxSizer( wxVERTICAL );

	m_staticText4741 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Please provide the stack and metadata files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4741->Wrap( -1 );
	bSizer471->Add( m_staticText4741, 0, wxALL, 5 );

	m_staticline1061 = new wxStaticLine( GetPathPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer471->Add( m_staticline1061, 0, wxEXPAND | wxALL, 5 );

	m_staticText411 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Output Particle Stack Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText411->Wrap( -1 );
	bSizer471->Add( m_staticText411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer5011;
	bSizer5011 = new wxBoxSizer( wxHORIZONTAL );

	ParticleStackFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer5011->Add( ParticleStackFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_button2411 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer5011->Add( m_button2411, 0, wxALL, 5 );


	bSizer471->Add( bSizer5011, 0, wxEXPAND, 5 );

	MetaFilenameStaticText = new wxStaticText( GetPathPage, wxID_ANY, wxT("Output PAR / STAR Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	MetaFilenameStaticText->Wrap( -1 );
	bSizer471->Add( MetaFilenameStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer502;
	bSizer502 = new wxBoxSizer( wxHORIZONTAL );

	MetaDataFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer502->Add( MetaDataFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_button242 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer502->Add( m_button242, 0, wxALL, 5 );


	bSizer471->Add( bSizer502, 0, wxEXPAND, 5 );


	GetPathPage->SetSizer( bSizer471 );
	GetPathPage->Layout();
	bSizer471->Fit( GetPathPage );

	this->Centre( wxBOTH );

	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ExportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button2411->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button242->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );
}

ExportRefinementPackageWizardParent::~ExportRefinementPackageWizardParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ExportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button2411->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button242->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );

	m_pages.Clear();
}

ImportRefinementPackageWizardParent::ImportRefinementPackageWizardParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style )
{
	this->Create( parent, id, title, bitmap, pos, style );

	this->SetSizeHints( wxSize( 700,400 ), wxDefaultSize );

	wxWizardPageSimple* ImportTypePage = new wxWizardPageSimple( this );
	m_pages.Add( ImportTypePage );

	wxBoxSizer* bSizer3931;
	bSizer3931 = new wxBoxSizer( wxVERTICAL );

	m_staticText4651 = new wxStaticText( ImportTypePage, wxID_ANY, wxT("Import from what source? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4651->Wrap( -1 );
	bSizer3931->Add( m_staticText4651, 0, wxALL, 5 );

	m_staticline1051 = new wxStaticLine( ImportTypePage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer3931->Add( m_staticline1051, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer3941;
	bSizer3941 = new wxBoxSizer( wxVERTICAL );

	cisTEMRadioButton = new wxRadioButton( ImportTypePage, wxID_ANY, wxT("cisTEM (Requires particle stack and STAR file)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( cisTEMRadioButton, 0, wxALL, 5 );

	RelionRadioButton = new wxRadioButton( ImportTypePage, wxID_ANY, wxT("Relion (Requires particle stack and STAR file)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( RelionRadioButton, 0, wxALL, 5 );

	FrealignRadioButton = new wxRadioButton( ImportTypePage, wxID_ANY, wxT("Frealign (Requires particle stack and PAR file)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( FrealignRadioButton, 0, wxALL, 5 );


	bSizer3931->Add( bSizer3941, 1, wxEXPAND, 5 );


	ImportTypePage->SetSizer( bSizer3931 );
	ImportTypePage->Layout();
	bSizer3931->Fit( ImportTypePage );
	wxWizardPageSimple* GetPathPage = new wxWizardPageSimple( this );
	m_pages.Add( GetPathPage );

	GetPathPage->SetMinSize( wxSize( 600,300 ) );

	wxBoxSizer* bSizer47;
	bSizer47 = new wxBoxSizer( wxVERTICAL );

	m_staticText474 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Please provide the stack and metadata files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText474->Wrap( -1 );
	bSizer47->Add( m_staticText474, 0, wxALL, 5 );

	m_staticline106 = new wxStaticLine( GetPathPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer47->Add( m_staticline106, 0, wxEXPAND | wxALL, 5 );

	m_staticText41 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Particle Stack Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText41->Wrap( -1 );
	bSizer47->Add( m_staticText41, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer501;
	bSizer501 = new wxBoxSizer( wxHORIZONTAL );

	ParticleStackFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer501->Add( ParticleStackFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_button241 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer501->Add( m_button241, 0, wxALL, 5 );


	bSizer47->Add( bSizer501, 0, wxEXPAND, 5 );

	MetaFilenameStaticText = new wxStaticText( GetPathPage, wxID_ANY, wxT("PAR / STAR Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	MetaFilenameStaticText->Wrap( -1 );
	bSizer47->Add( MetaFilenameStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer50;
	bSizer50 = new wxBoxSizer( wxHORIZONTAL );

	MetaDataFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( MetaDataFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_button24 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( m_button24, 0, wxALL, 5 );


	bSizer47->Add( bSizer50, 0, wxEXPAND, 5 );


	GetPathPage->SetSizer( bSizer47 );
	GetPathPage->Layout();
	bSizer47->Fit( GetPathPage );
	wxWizardPageSimple* GetParametersPage = new wxWizardPageSimple( this );
	m_pages.Add( GetParametersPage );

	wxBoxSizer* bSizer402;
	bSizer402 = new wxBoxSizer( wxVERTICAL );

	m_staticText476 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Please provide the following information :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText476->Wrap( -1 );
	bSizer402->Add( m_staticText476, 0, wxALL, 5 );

	m_staticline107 = new wxStaticLine( GetParametersPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer402->Add( m_staticline107, 0, wxEXPAND | wxALL, 5 );

	wxFlexGridSizer* fgSizer23;
	fgSizer23 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer23->SetFlexibleDirection( wxBOTH );
	fgSizer23->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	PixelSizeTextCtrlLabel = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Pixel Size (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeTextCtrlLabel->Wrap( -1 );
	fgSizer23->Add( PixelSizeTextCtrlLabel, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	PixelSizeTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeTextCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer23->Add( PixelSizeTextCtrl, 0, wxALL, 5 );

	MicroscopeVoltageTextCtrlLabel = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Microscope Voltage (kV) : "), wxDefaultPosition, wxDefaultSize, 0 );
	MicroscopeVoltageTextCtrlLabel->Wrap( -1 );
	fgSizer23->Add( MicroscopeVoltageTextCtrlLabel, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	MicroscopeVoltageTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	MicroscopeVoltageTextCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer23->Add( MicroscopeVoltageTextCtrl, 0, wxALL, 5 );

	m_staticText479 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Microscope Cs (mm) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText479->Wrap( -1 );
	fgSizer23->Add( m_staticText479, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	SphericalAberrationTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("2.70"), wxDefaultPosition, wxDefaultSize, 0 );
	SphericalAberrationTextCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer23->Add( SphericalAberrationTextCtrl, 0, wxALL, 5 );

	AmplitudeContrastTextCtrlLabel = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Amplitude Contrast : "), wxDefaultPosition, wxDefaultSize, 0 );
	AmplitudeContrastTextCtrlLabel->Wrap( -1 );
	fgSizer23->Add( AmplitudeContrastTextCtrlLabel, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	AmplitudeContrastTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("0.07"), wxDefaultPosition, wxDefaultSize, 0 );
	AmplitudeContrastTextCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer23->Add( AmplitudeContrastTextCtrl, 0, wxALL, 5 );

	m_staticText459 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Pointgroup Symmetry : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText459->Wrap( -1 );
	fgSizer23->Add( m_staticText459, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	SymmetryComboBox = new wxComboBox( GetParametersPage, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	SymmetryComboBox->SetMinSize( wxSize( 100,-1 ) );

	fgSizer23->Add( SymmetryComboBox, 0, wxALL, 5 );

	m_staticText460 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Estimated Molecular Weight (kDa) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText460->Wrap( -1 );
	fgSizer23->Add( m_staticText460, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	MolecularWeightTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("400.00"), wxDefaultPosition, wxDefaultSize, 0 );
	MolecularWeightTextCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer23->Add( MolecularWeightTextCtrl, 0, wxALL, 5 );

	m_staticText214 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Estimated Largest Dimension (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	fgSizer23->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	LargestDimensionTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("150.00"), wxDefaultPosition, wxDefaultSize, 0 );
	LargestDimensionTextCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer23->Add( LargestDimensionTextCtrl, 0, wxALL, 5 );

	m_staticText462 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Protein Density in Stack is : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText462->Wrap( -1 );
	fgSizer23->Add( m_staticText462, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	wxBoxSizer* bSizer379;
	bSizer379 = new wxBoxSizer( wxHORIZONTAL );

	BlackProteinRadioButton = new wxRadioButton( GetParametersPage, wxID_ANY, wxT("Black (cisTEM Default)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer379->Add( BlackProteinRadioButton, 0, wxALL, 5 );

	WhiteProteinRadioButton = new wxRadioButton( GetParametersPage, wxID_ANY, wxT("White (Relion Default)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer379->Add( WhiteProteinRadioButton, 0, wxALL, 5 );


	fgSizer23->Add( bSizer379, 1, wxEXPAND, 5 );


	bSizer402->Add( fgSizer23, 1, wxEXPAND, 5 );


	GetParametersPage->SetSizer( bSizer402 );
	GetParametersPage->Layout();
	bSizer402->Fit( GetParametersPage );

	this->Centre( wxBOTH );

	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ImportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button241->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button24->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );
}

ImportRefinementPackageWizardParent::~ImportRefinementPackageWizardParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ImportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button241->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button24->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );

	m_pages.Clear();
}

NewRefinementPackageWizard::NewRefinementPackageWizard( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style )
{
	this->Create( parent, id, title, bitmap, pos, style );

	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );


	this->Centre( wxBOTH );


	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewRefinementPackageWizard::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( NewRefinementPackageWizard::PageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( NewRefinementPackageWizard::PageChanging ) );
}

NewRefinementPackageWizard::~NewRefinementPackageWizard()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewRefinementPackageWizard::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( NewRefinementPackageWizard::PageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( NewRefinementPackageWizard::PageChanging ) );

	m_pages.Clear();
}

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

ClassesSetupWizardPanelA::ClassesSetupWizardPanelA( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Carry over all Particles? : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );

	CarryOverYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( CarryOverYesButton, 0, wxALL, 5 );

	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );


	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );


	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Do you want to carry over all particles from the template refinement package to the new refinement package?  If No, you will be able to select the classes from which particles should be carried over (based on which class has the highest occupancy for that particle), and a new particle stack will be created. If yes all the particles will be included in the new refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelA::~ClassesSetupWizardPanelA()
{
}

PackageSelectionPanel::PackageSelectionPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* mainSizer;
	mainSizer = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* PackageSelectionSizer;
	PackageSelectionSizer = new wxBoxSizer( wxVERTICAL );

	wxArrayString RefinementPackagesCheckListBoxChoices;
	RefinementPackagesCheckListBox = new wxCheckListBox( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, RefinementPackagesCheckListBoxChoices, wxLB_MULTIPLE|wxLB_NEEDED_SB );
	PackageSelectionSizer->Add( RefinementPackagesCheckListBox, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	mainSizer->Add( PackageSelectionSizer, 1, wxEXPAND, 5 );

	ErrorStaticText = new wxStaticText( this, wxID_ANY, wxT("Oops! - Command must contain \"$command\""), wxDefaultPosition, wxDefaultSize, 0 );
	ErrorStaticText->Wrap( -1 );
	ErrorStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	ErrorStaticText->Hide();

	mainSizer->Add( ErrorStaticText, 0, wxALIGN_CENTER|wxALL, 5 );

	RemoveDuplicatesCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Remove Duplicate Particles"), wxDefaultPosition, wxDefaultSize, 0 );
	mainSizer->Add( RemoveDuplicatesCheckbox, 0, wxALL|wxEXPAND, 5 );

	ImportedParamsWarning = new wxStaticText( this, wxID_ANY, wxT("*Removing duplicates from packages containing imported parameters is not possible."), wxDefaultPosition, wxDefaultSize, 0 );
	ImportedParamsWarning->Wrap( -1 );
	ImportedParamsWarning->SetForegroundColour( wxColour( 251, 8, 8 ) );

	mainSizer->Add( ImportedParamsWarning, 0, wxALL|wxEXPAND, 5 );

	m_staticline158 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	mainSizer->Add( m_staticline158, 0, wxEXPAND | wxALL, 5 );

	m_staticText798 = new wxStaticText( this, wxID_ANY, wxT("Specify Combined Package Parameters"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText798->Wrap( -1 );
	mainSizer->Add( m_staticText798, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

	wxBoxSizer* bSizer587;
	bSizer587 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer584;
	bSizer584 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* symmetrySizer;
	symmetrySizer = new wxBoxSizer( wxHORIZONTAL );

	m_staticText792 = new wxStaticText( this, wxID_ANY, wxT("Wanted Pointgroup Symmetry:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText792->Wrap( -1 );
	symmetrySizer->Add( m_staticText792, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer584->Add( symmetrySizer, 1, wxALIGN_CENTER_VERTICAL, 5 );

	wxBoxSizer* bSizer582;
	bSizer582 = new wxBoxSizer( wxHORIZONTAL );

	SymmetryComboBox = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	SymmetryComboBox->Append( wxT("C1") );
	SymmetryComboBox->Append( wxT("C2") );
	SymmetryComboBox->Append( wxT("C3") );
	SymmetryComboBox->Append( wxT("C4") );
	SymmetryComboBox->Append( wxT("D2") );
	SymmetryComboBox->Append( wxT("D3") );
	SymmetryComboBox->Append( wxT("D4") );
	SymmetryComboBox->Append( wxT("I") );
	SymmetryComboBox->Append( wxT("I2") );
	SymmetryComboBox->Append( wxT("O") );
	SymmetryComboBox->Append( wxT("T") );
	SymmetryComboBox->Append( wxT("T2") );
	SymmetryComboBox->SetSelection( 0 );
	bSizer582->Add( SymmetryComboBox, 1, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL|wxFIXED_MINSIZE, 0 );


	bSizer584->Add( bSizer582, 1, wxEXPAND, 5 );


	bSizer587->Add( bSizer584, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer585;
	bSizer585 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* molecularWeightSizer;
	molecularWeightSizer = new wxBoxSizer( wxHORIZONTAL );

	molecularWeightSizer->SetMinSize( wxSize( 1,-1 ) );
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Estimated Molecular Weight (kDa) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	molecularWeightSizer->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer585->Add( molecularWeightSizer, 1, wxALIGN_CENTER_HORIZONTAL|wxEXPAND, 5 );

	wxBoxSizer* bSizer580;
	bSizer580 = new wxBoxSizer( wxHORIZONTAL );

	MolecularWeightTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer580->Add( MolecularWeightTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALIGN_RIGHT|wxALL|wxFIXED_MINSIZE, 5 );


	bSizer585->Add( bSizer580, 1, wxEXPAND, 5 );


	bSizer587->Add( bSizer585, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer586;
	bSizer586 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* largestDimensionSizer;
	largestDimensionSizer = new wxBoxSizer( wxHORIZONTAL );

	m_staticText2141 = new wxStaticText( this, wxID_ANY, wxT("Estimated Largest Dimension / Diameter (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText2141->Wrap( -1 );
	largestDimensionSizer->Add( m_staticText2141, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer586->Add( largestDimensionSizer, 1, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	wxBoxSizer* bSizer581;
	bSizer581 = new wxBoxSizer( wxHORIZONTAL );

	LargestDimensionTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer581->Add( LargestDimensionTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALIGN_RIGHT|wxALL, 5 );


	bSizer586->Add( bSizer581, 1, wxEXPAND, 5 );


	bSizer587->Add( bSizer586, 1, wxEXPAND, 5 );


	mainSizer->Add( bSizer587, 1, wxEXPAND, 0 );


	this->SetSizer( mainSizer );
	this->Layout();

	// Connect Events
	RefinementPackagesCheckListBox->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( PackageSelectionPanel::PackageClassSelection ), NULL, this );
}

PackageSelectionPanel::~PackageSelectionPanel()
{
	// Disconnect Events
	RefinementPackagesCheckListBox->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( PackageSelectionPanel::PackageClassSelection ), NULL, this );

}

CombinedClassSelectionPanel::CombinedClassSelectionPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer686;
	bSizer686 = new wxBoxSizer( wxVERTICAL );

	m_staticText883 = new wxStaticText( this, wxID_ANY, wxT("Select a class from each package:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText883->Wrap( -1 );
	bSizer686->Add( m_staticText883, 0, wxALL, 5 );

	m_staticline187 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer686->Add( m_staticline187, 0, wxEXPAND | wxALL, 5 );

	CombinedClassScrollWindow = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	CombinedClassScrollWindow->SetScrollRate( 5, 5 );
	CombinedClassScrollSizer = new wxBoxSizer( wxVERTICAL );


	CombinedClassScrollWindow->SetSizer( CombinedClassScrollSizer );
	CombinedClassScrollWindow->Layout();
	CombinedClassScrollSizer->Fit( CombinedClassScrollWindow );
	bSizer686->Add( CombinedClassScrollWindow, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer686 );
	this->Layout();
}

CombinedClassSelectionPanel::~CombinedClassSelectionPanel()
{
}

CombinedPackageRefinementPanel::CombinedPackageRefinementPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer686;
	bSizer686 = new wxBoxSizer( wxVERTICAL );

	SelectRefinementText = new wxStaticText( this, wxID_ANY, wxT("Select a previously used Refinement to apply, or select Random Parameters to generate from random parameters:"), wxDefaultPosition, wxDefaultSize, 0 );
	SelectRefinementText->Wrap( -1 );
	bSizer686->Add( SelectRefinementText, 0, wxALL, 5 );

	m_staticline187 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer686->Add( m_staticline187, 0, wxEXPAND | wxALL, 5 );

	CombinedRefinementScrollWindow = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	CombinedRefinementScrollWindow->SetScrollRate( 5, 5 );
	CombinedRefinementScrollSizer = new wxBoxSizer( wxVERTICAL );


	CombinedRefinementScrollWindow->SetSizer( CombinedRefinementScrollSizer );
	CombinedRefinementScrollWindow->Layout();
	CombinedRefinementScrollSizer->Fit( CombinedRefinementScrollWindow );
	bSizer686->Add( CombinedRefinementScrollWindow, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer686 );
	this->Layout();
}

CombinedPackageRefinementPanel::~CombinedPackageRefinementPanel()
{
}

TemplateWizardPanel::TemplateWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Template Refinement Package :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	GroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer147->Add( GroupComboBox, 1, wxALL, 5 );


	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );


	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select \"New Refinement Package\" to create a package from scratch.  Select \"Create From 2D Class Average Selection\" to create a new package based on one or more selections of 2D class averages. Alternatively, an existing refinement package can be used as a template.  Template based refinement packages will have the same particle stack as their template, but you will be able to change the other parameters, you may wish to do this to change the number of classes or symmetry for example."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

TemplateWizardPanel::~TemplateWizardPanel()
{
}

InputParameterWizardPanel::InputParameterWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Input Parameters :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	GroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer147->Add( GroupComboBox, 1, wxALL, 5 );


	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );


	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select the parameters to use to create the new refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

InputParameterWizardPanel::~InputParameterWizardPanel()
{
}

ClassSelectionWizardPanel::ClassSelectionWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	m_staticText416 = new wxStaticText( this, wxID_ANY, wxT("Select class average selections to use :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText416->Wrap( -1 );
	bSizer153->Add( m_staticText416, 0, wxALL, 5 );

	wxBoxSizer* bSizer278;
	bSizer278 = new wxBoxSizer( wxHORIZONTAL );

	SelectionListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_REPORT );
	bSizer278->Add( SelectionListCtrl, 1, wxALL|wxEXPAND, 5 );


	bSizer153->Add( bSizer278, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("If you have performed any 2D classifications, then you can use one or more selections of class averages (selected from the classification results panel) to copy over ONLY the particles in the selected class averages to the new refinement package.  This may be done to clean the dataset for example."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

ClassSelectionWizardPanel::~ClassSelectionWizardPanel()
{
}

SymmetryWizardPanel::SymmetryWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Wanted Pointgroup Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SymmetryComboBox = new wxComboBox( this, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	bSizer147->Add( SymmetryComboBox, 1, wxALL, 5 );


	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );


	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select the pointgroup symmetry to use for the refinement.  Whilst a number of symmetries can be selected from the menu for convenience, you may enter any valid pointgroup symmetry (in Schoenflies notation)."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

SymmetryWizardPanel::~SymmetryWizardPanel()
{
}

MolecularWeightWizardPanel::MolecularWeightWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Estimated Molecular Weight (kDa) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MolecularWeightTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( MolecularWeightTextCtrl, 1, wxALL, 5 );


	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );


	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the estimated molecular weight for the refinement.  In general, this should be the molecular weight of the coherent parts (e.g. micelle would not be included)."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

MolecularWeightWizardPanel::~MolecularWeightWizardPanel()
{
}

InitialReferenceSelectWizardPanel::InitialReferenceSelectWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	MainSizer = new wxBoxSizer( wxVERTICAL );

	TitleText = new wxStaticText( this, wxID_ANY, wxT("Initial class references :-"), wxDefaultPosition, wxDefaultSize, 0 );
	TitleText->Wrap( -1 );
	MainSizer->Add( TitleText, 0, wxALL, 5 );

	m_staticline108 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline108, 0, wxEXPAND | wxALL, 5 );

	ScrollWindow = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( 100,-1 ), wxHSCROLL|wxVSCROLL );
	ScrollWindow->SetScrollRate( 5, 5 );
	ScrollSizer = new wxBoxSizer( wxVERTICAL );


	ScrollWindow->SetSizer( ScrollSizer );
	ScrollWindow->Layout();
	MainSizer->Add( ScrollWindow, 1, wxALL|wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the volume asset to use for the initial reference for each of the classes.  If \"Generate from params.\" is selected, a new volume asset will be calculated from the starting parameters at the start of the next refinement. In general, you will provide a reference for a new refinement package, and generate one when using a template."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	MainSizer->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( MainSizer );
	this->Layout();
}

InitialReferenceSelectWizardPanel::~InitialReferenceSelectWizardPanel()
{
}

LargestDimensionWizardPanel::LargestDimensionWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Estimated Largest Dimension / Diameter (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LargestDimensionTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( LargestDimensionTextCtrl, 1, wxALL, 5 );


	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );


	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the estimated largest dimension of the particle. This will be used for autosizing masks and shift limits, it need not be exact, and it is better to err on the large size. Please note, this is the largest dimension / diameter - NOT RADIUS."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

LargestDimensionWizardPanel::~LargestDimensionWizardPanel()
{
}

OutputPixelSizeWizardPanel::OutputPixelSizeWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Wanted Output Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	OutputPixelSizeTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( OutputPixelSizeTextCtrl, 1, wxALL, 5 );


	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );


	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the pixel size for outputs.  In most cases this will be the pixel size of your images, however in the case where you have data of mixed pixel sizes you must enter a specific pixel size here.  All images will be scaled to this pixel size during process, and all outputs will be this pixel size."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

OutputPixelSizeWizardPanel::~OutputPixelSizeWizardPanel()
{
}

ParticleGroupWizardPanel::ParticleGroupWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer1531;
	bSizer1531 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer1471;
	bSizer1471 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText2141 = new wxStaticText( this, wxID_ANY, wxT("Particle Position Group :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText2141->Wrap( -1 );
	bSizer1471->Add( m_staticText2141, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ParticlePositionsGroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer1471->Add( ParticlePositionsGroupComboBox, 1, wxALL, 5 );


	bSizer1531->Add( bSizer1471, 0, wxEXPAND, 5 );


	bSizer1531->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select a group of particle positions to use for the refinement package.  The particles will be \"cut\" from the active image assets using the active CTF estimation parameters.  From this point on the particle images and defocus parameters will be fixed, and changing the active results will not change the refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer1531->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer1531 );
	this->Layout();
}

ParticleGroupWizardPanel::~ParticleGroupWizardPanel()
{
}

BoxSizeWizardPanel::BoxSizeWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Box size for particles (pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	BoxSizeSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 2048, 1 );
	bSizer14711->Add( BoxSizeSpinCtrl, 1, wxALL, 5 );


	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the box size that should be used to extract the particles.  This box size should be big enough to not only contain the full particle, but also to take into account the delocalisation of information due to the CTF, as well as any aliasing of the CTF."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

BoxSizeWizardPanel::~BoxSizeWizardPanel()
{
}

NumberofClassesWizardPanel::NumberofClassesWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153111;
	bSizer153111 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147111;
	bSizer147111 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214111 = new wxStaticText( this, wxID_ANY, wxT("Number of classes for 3D refinement :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214111->Wrap( -1 );
	bSizer147111->Add( m_staticText214111, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	NumberOfClassesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100, 0 );
	bSizer147111->Add( NumberOfClassesSpinCtrl, 1, wxALL, 5 );


	bSizer153111->Add( bSizer147111, 0, wxEXPAND, 5 );


	bSizer153111->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the number 3D of classes to use during 3D refinement.  This number only affects 3D refinements, and will have no effect during 2D classification.  This number is fixed for a refinement package, but can be changed by creating a new refinement package and using an old refinement package as a template.  In general you will start with 1 class, and split into more classes later in the refinement.If this package is based on a previous package, you will be given options about how to split/merge classes in the following steps.  "), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153111->Add( InfoText, 0, wxALL|wxEXPAND, 10 );


	this->SetSizer( bSizer153111 );
	this->Layout();
}

NumberofClassesWizardPanel::~NumberofClassesWizardPanel()
{
}

RecentrePicksWizardPanel::RecentrePicksWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Re-Centre Based on Classification? : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );

	ReCentreYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( ReCentreYesButton, 0, wxALL, 5 );

	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );


	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );


	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Do you want to use the shifts estimated for the class averages to re-center the picking co-ordinates prior to cutting out the images?  This can be especially helpful when the centreing on the picking was not very good, and a subsequent 2D classification with centre averages set to yes was run.   Note : This will only apply if cisTEM has images and co-ordinates for the particles, it will do nothing if the images come from an imported stack."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

RecentrePicksWizardPanel::~RecentrePicksWizardPanel()
{
}

RemoveDuplicatesWizardPanel::RemoveDuplicatesWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Remove Duplicate Particle Positions? : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );

	RemoveDuplicateYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( RemoveDuplicateYesButton, 0, wxALL, 5 );

	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );


	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );


	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Do you want to remove duplicate (within a certain distance) picks?  After re-centring based on 2D classification it can happen that multiple different picks are moved onto the same point. In this case, selecting yes here will keep only one of these picks and help reduce having multiple copies of the same particle."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

RemoveDuplicatesWizardPanel::~RemoveDuplicatesWizardPanel()
{
}

RemoveDuplicateThresholdWizardPanel::RemoveDuplicateThresholdWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Duplicate Pick Threshold Distance (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DuplicatePickThresholdTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( DuplicatePickThresholdTextCtrl, 1, wxALL, 5 );


	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );


	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("When removing duplicate picks, picks within this distance will be counted as duplicate and thus removed.  Typically this would be around 1/3rd to 1/2 the size of your particle."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer153 );
	this->Layout();
}

RemoveDuplicateThresholdWizardPanel::~RemoveDuplicateThresholdWizardPanel()
{
}

ClassesSetupWizardPanelB::ClassesSetupWizardPanelB( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Carry over particles from which classes? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticline103 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer14711->Add( m_staticline103, 0, wxEXPAND | wxALL, 5 );

	ClassListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT );
	bSizer14711->Add( ClassListCtrl, 10, wxALL|wxEXPAND, 5 );


	bSizer15311->Add( bSizer14711, 100, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Only particles from the selected classes will be carried over to the new refinement package.  At least one class must be selected, multiple selections are allowed (with CTRL/SHIFT)."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelB::~ClassesSetupWizardPanelB()
{
}

ClassesSetupWizardPanelC::ClassesSetupWizardPanelC( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("How should the new classes be created? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticline104 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer14711->Add( m_staticline104, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer378;
	bSizer378 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer391;
	bSizer391 = new wxBoxSizer( wxVERTICAL );

	NewClassListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL );
	NewClassListCtrl->SetMinSize( wxSize( 10,-1 ) );

	bSizer391->Add( NewClassListCtrl, 100, wxALL|wxEXPAND, 5 );


	bSizer378->Add( bSizer391, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer392;
	bSizer392 = new wxBoxSizer( wxVERTICAL );

	OldClassListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT );
	OldClassListCtrl->SetMinSize( wxSize( 10,-1 ) );

	bSizer392->Add( OldClassListCtrl, 1, wxALL|wxEXPAND, 5 );


	bSizer378->Add( bSizer392, 1, wxEXPAND, 5 );


	bSizer14711->Add( bSizer378, 1, wxEXPAND, 5 );


	bSizer15311->Add( bSizer14711, 100, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("For each new class (Left) select class(es) to copy parameters from.  If two or more classes are selected (e.g. to merge similar classes), parameters can be taken either from the class with the highest occupancy, or from a random class. At least one class must be selected, multiple selections are allowed (with CTRL/SHIFT).  Note : You are selecting which parameters to copy over, all particles are represented in each class. "), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelC::~ClassesSetupWizardPanelC()
{
}

ClassesSetupWizardPanelD::ClassesSetupWizardPanelD( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Which parameters should be used when combining classes? :- "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );

	BestOccupancyRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Best Occupancy"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( BestOccupancyRadioButton, 0, wxALL, 5 );

	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("Random"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );


	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );


	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("In the case where a new class will be created by merging parameters from one or more classes, do you want to take the parameters from the class with the highest occupancy, or the parameters from a random class."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelD::~ClassesSetupWizardPanelD()
{
}

ClassesSetupWizardPanelE::ClassesSetupWizardPanelE( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );

	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("How should the new occupancies be set? :- "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );

	RandomiseOccupanciesRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Random occupancies"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( RandomiseOccupanciesRadioButton, 0, wxALL, 5 );

	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("Keep input occupancies"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );


	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );


	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );


	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );

	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Should the new occupancies be randomised, or should the input occupancies be kept. Randomising the occupanices is a way of creating new \"seed\" references. For example, a refinement done with 1 class, could be classified into multiple classes by creating the new classes with parameters from the single class and randomising the occupancies.  This will lead to small differences which will hopefully converge to an accurate 3D classification after a number of rounds of additional refinement."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );


	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelE::~ClassesSetupWizardPanelE()
{
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
