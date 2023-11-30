///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "AssetPickerComboPanel.h"
#include "DisplayRefinementResultsPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_autorefine.h"

///////////////////////////////////////////////////////////////////////////

AutoRefine3DPanelParent::AutoRefine3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer364;
	bSizer364 = new wxBoxSizer( wxVERTICAL );

	InputParamsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxHORIZONTAL );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText262 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	RefinementPackageSelectPanel = new RefinementPackagePickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageSelectPanel->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( RefinementPackageSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticText478 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Starting Reference :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText478->Wrap( -1 );
	fgSizer15->Add( m_staticText478, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	ReferenceSelectPanel = new VolumeAssetPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ReferenceSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	ReferenceSelectPanel->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( ReferenceSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	UseMaskCheckBox = new wxCheckBox( InputParamsPanel, wxID_ANY, wxT("Use a Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( UseMaskCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	MaskSelectPanel = new VolumeAssetPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	MaskSelectPanel->SetMinSize( wxSize( 350,-1 ) );

	fgSizer15->Add( MaskSelectPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer200->Add( fgSizer15, 0, wxEXPAND, 5 );

	m_staticline54 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline54, 0, wxEXPAND | wxALL, 5 );

	wxGridSizer* gSizer11;
	gSizer11 = new wxGridSizer( 1, 1, 0, 0 );

	wxBoxSizer* bSizer214;
	bSizer214 = new wxBoxSizer( wxHORIZONTAL );

	InitialResLimitStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Initial Res. Limit (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	InitialResLimitStaticText->Wrap( -1 );
	bSizer214->Add( InitialResLimitStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	HighResolutionLimitTextCtrl = new NumericTextCtrl( InputParamsPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer214->Add( HighResolutionLimitTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	gSizer11->Add( bSizer214, 0, 0, 5 );


	bSizer200->Add( gSizer11, 0, wxEXPAND, 5 );


	bSizer200->Add( 0, 0, 1, wxEXPAND, 5 );

	ExpertToggleButton = new wxToggleButton( InputParamsPanel, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer200->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer441->Add( bSizer200, 1, wxEXPAND, 5 );

	PleaseCreateRefinementPackageText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Please create a refinement package (in the assets panel) in order to perform a 3D refinement."), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseCreateRefinementPackageText->Wrap( -1 );
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	PleaseCreateRefinementPackageText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	PleaseCreateRefinementPackageText->Hide();

	bSizer441->Add( PleaseCreateRefinementPackageText, 0, wxALL, 5 );

	m_staticline105 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer441->Add( m_staticline105, 0, wxEXPAND | wxALL, 5 );


	InputParamsPanel->SetSizer( bSizer441 );
	InputParamsPanel->Layout();
	bSizer441->Fit( InputParamsPanel );
	bSizer364->Add( InputParamsPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer451->Add( bSizer364, 1, wxEXPAND, 5 );


	bSizer43->Add( bSizer451, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();

	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer258;
	bSizer258 = new wxBoxSizer( wxHORIZONTAL );


	bSizer258->Add( 0, 0, 1, wxEXPAND, 5 );

	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer258->Add( ResetAllDefaultsButton, 0, wxALL, 5 );


	bSizer258->Add( 5, 0, 0, wxEXPAND, 5 );


	InputSizer->Add( bSizer258, 0, wxEXPAND, 5 );


	InputSizer->Add( 0, 5, 0, wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("General Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText202, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( LowResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Outer Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText330 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Inner Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText330->Wrap( -1 );
	fgSizer1->Add( m_staticText330, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	InnerMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( InnerMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Global Search"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText201, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	GlobalMaskRadiusStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Global Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	GlobalMaskRadiusStaticText->Wrap( -1 );
	fgSizer1->Add( GlobalMaskRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	GlobalMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( GlobalMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	NumberToRefineStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Number of Results to Refine :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberToRefineStaticText->Wrap( -1 );
	fgSizer1->Add( NumberToRefineStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	NumberToRefineSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, -100000, 100000, 16 );
	fgSizer1->Add( NumberToRefineSpinCtrl, 0, wxALL, 5 );

	SearchRangeXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in X (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SearchRangeXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeXTextCtrl, 0, wxALL|wxEXPAND, 5 );

	SearchRangeYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in Y (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SearchRangeYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeYTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText329 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText329->Wrap( -1 );
	m_staticText329->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText329, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText336 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Autocrop Images?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText336->Wrap( -1 );
	fgSizer1->Add( m_staticText336, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer266;
	bSizer266 = new wxBoxSizer( wxHORIZONTAL );

	AutoCropYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer266->Add( AutoCropYesRadioButton, 0, wxALL, 5 );

	AutoCropNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer266->Add( AutoCropNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer266, 1, wxEXPAND, 5 );

	m_staticText363 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Apply Likelihood Blurring?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText363->Wrap( -1 );
	fgSizer1->Add( m_staticText363, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2661;
	bSizer2661 = new wxBoxSizer( wxHORIZONTAL );

	ApplyBlurringYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2661->Add( ApplyBlurringYesRadioButton, 0, wxALL, 5 );

	ApplyBlurringNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2661->Add( ApplyBlurringNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2661, 1, wxEXPAND, 5 );

	SmoothingFactorStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSmoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	SmoothingFactorStaticText->Wrap( -1 );
	fgSizer1->Add( SmoothingFactorStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SmoothingFactorTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SmoothingFactorTextCtrl, 0, wxALL|wxEXPAND, 5 );

	AutoCenterStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Autocenter reconstruction?"), wxDefaultPosition, wxDefaultSize, 0 );
	AutoCenterStaticText->Wrap( -1 );
	fgSizer1->Add( AutoCenterStaticText, 0, wxALL, 5 );

	wxBoxSizer* bSizer26612;
	bSizer26612 = new wxBoxSizer( wxHORIZONTAL );

	AutoCenterYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26612->Add( AutoCenterYesRadioButton, 0, wxALL, 5 );

	AutoCenterNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26612->Add( AutoCenterNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26612, 1, wxEXPAND, 5 );

	m_staticText405 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Masking"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText405->Wrap( -1 );
	m_staticText405->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText405, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	AutoMaskStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Use Auto-Masking?"), wxDefaultPosition, wxDefaultSize, 0 );
	AutoMaskStaticText->Wrap( -1 );
	fgSizer1->Add( AutoMaskStaticText, 0, wxALL, 5 );

	wxBoxSizer* bSizer2641;
	bSizer2641 = new wxBoxSizer( wxHORIZONTAL );

	AutoMaskYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2641->Add( AutoMaskYesRadioButton, 0, wxALL, 5 );

	AutoMaskNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2641->Add( AutoMaskNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2641, 1, wxEXPAND, 5 );

	MaskEdgeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Edge Width (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaskEdgeStaticText->Wrap( -1 );
	fgSizer1->Add( MaskEdgeStaticText, 0, wxALL, 5 );

	MaskEdgeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskEdgeTextCtrl, 0, wxALL, 5 );

	MaskWeightStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Outside Weight :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaskWeightStaticText->Wrap( -1 );
	fgSizer1->Add( MaskWeightStaticText, 0, wxALL, 5 );

	MaskWeightTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskWeightTextCtrl, 0, wxALL, 5 );

	LowPassYesNoStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Pass Filter Outside Mask?"), wxDefaultPosition, wxDefaultSize, 0 );
	LowPassYesNoStaticText->Wrap( -1 );
	fgSizer1->Add( LowPassYesNoStaticText, 0, wxALL, 5 );

	wxBoxSizer* bSizer26611;
	bSizer26611 = new wxBoxSizer( wxHORIZONTAL );

	LowPassMaskYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26611->Add( LowPassMaskYesRadio, 0, wxALL, 5 );

	LowPassMaskNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26611->Add( LowPassMaskNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26611, 1, wxEXPAND, 5 );

	FilterResolutionStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tFilter Resolution (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterResolutionStaticText->Wrap( -1 );
	fgSizer1->Add( FilterResolutionStaticText, 0, wxALL, 5 );

	MaskFilterResolutionText = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("20.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskFilterResolutionText, 0, wxALL, 5 );


	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );


	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );

	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();

	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );

	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );


	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer46->Add( OutputTextPanel, 20, wxEXPAND | wxALL, 5 );

	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );

	ShowRefinementResultsPanel = new DisplayRefinementResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ShowRefinementResultsPanel->Hide();

	bSizer46->Add( ShowRefinementResultsPanel, 80, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );

	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer71;
	bSizer71 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxHORIZONTAL );


	bSizer71->Add( bSizer55, 0, wxALIGN_CENTER_HORIZONTAL, 5 );


	bSizer70->Add( bSizer71, 1, wxALIGN_CENTER_VERTICAL, 5 );

	wxBoxSizer* bSizer74;
	bSizer74 = new wxBoxSizer( wxVERTICAL );


	bSizer70->Add( bSizer74, 0, wxALIGN_CENTER_VERTICAL, 5 );

	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();

	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxHORIZONTAL );

	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("o"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer59->Add( NumberConnectedText, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 );
	bSizer59->Add( ProgressBar, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );

	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();

	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );


	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer70->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );

	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer267;
	bSizer267 = new wxBoxSizer( wxHORIZONTAL );

	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Refinement Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer267->Add( RunProfileText, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	RefinementRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer267->Add( RefinementRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer268->Add( bSizer267, 50, wxEXPAND, 5 );

	wxBoxSizer* bSizer2671;
	bSizer2671 = new wxBoxSizer( wxHORIZONTAL );

	RunProfileText1 = new wxStaticText( StartPanel, wxID_ANY, wxT("Reconstruction Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText1->Wrap( -1 );
	bSizer2671->Add( RunProfileText1, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ReconstructionRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer2671->Add( ReconstructionRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer268->Add( bSizer2671, 0, wxEXPAND, 5 );


	bSizer58->Add( bSizer268, 50, wxEXPAND, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );


	bSizer60->Add( 0, 0, 1, wxEXPAND, 5 );

	StartRefinementButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartRefinementButton, 0, wxALL, 5 );


	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );


	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer43 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AutoRefine3DPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnUseMaskCheckBox ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AutoRefine3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::StartRefinementClick ), NULL, this );
}

AutoRefine3DPanelParent::~AutoRefine3DPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AutoRefine3DPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnUseMaskCheckBox ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AutoRefine3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::StartRefinementClick ), NULL, this );

}
