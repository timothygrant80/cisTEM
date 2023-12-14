///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "AbInitioPlotPanel.h"
#include "AssetPickerComboPanel.h"
#include "DisplayPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_abinitio.h"

///////////////////////////////////////////////////////////////////////////

AbInitio3DPanelParent::AbInitio3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	TopStaticLine = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( TopStaticLine, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer504;
	bSizer504 = new wxBoxSizer( wxHORIZONTAL );

	ImageOrClassAverageStaticText = new wxStaticText( this, wxID_ANY, wxT("Use Images or Class Averages? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ImageOrClassAverageStaticText->Wrap( -1 );
	bSizer504->Add( ImageOrClassAverageStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	wxBoxSizer* bSizer502;
	bSizer502 = new wxBoxSizer( wxHORIZONTAL );

	ImageInputRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Images"), wxDefaultPosition, wxDefaultSize, 0 );
	ImageInputRadioButton->SetValue( true );
	bSizer502->Add( ImageInputRadioButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ClassAverageInputRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Class Averages"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer502->Add( ClassAverageInputRadioButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer504->Add( bSizer502, 1, wxEXPAND, 5 );


	bSizer43->Add( bSizer504, 0, wxEXPAND, 5 );

	BottomStaticLine = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( BottomStaticLine, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );

	InputParamsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	InputRefinementPackageStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	InputRefinementPackageStaticText->Wrap( -1 );
	fgSizer15->Add( InputRefinementPackageStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	RefinementPackageComboBox = new RefinementPackagePickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageComboBox->Enable( false );
	RefinementPackageComboBox->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( RefinementPackageComboBox, 1, wxEXPAND | wxALL, 5 );

	InputClassificationSelectionStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Classification Selection :"), wxDefaultPosition, wxDefaultSize, 0 );
	InputClassificationSelectionStaticText->Wrap( -1 );
	InputClassificationSelectionStaticText->Enable( false );

	fgSizer15->Add( InputClassificationSelectionStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	ClassSelectionComboBox = new ClassSelectionPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ClassSelectionComboBox->Enable( false );
	ClassSelectionComboBox->SetMinSize( wxSize( 350,-1 ) );
	ClassSelectionComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( ClassSelectionComboBox, 1, wxEXPAND | wxALL, 5 );


	bSizer200->Add( fgSizer15, 0, wxALIGN_CENTER_VERTICAL, 5 );

	m_staticline52 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );

	wxFlexGridSizer* fgSizer35;
	fgSizer35 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer35->SetFlexibleDirection( wxBOTH );
	fgSizer35->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	NoClassesStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoClassesStaticText->Wrap( -1 );
	NoClassesStaticText->Enable( false );

	fgSizer35->Add( NoClassesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	NumberClassesSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 50, 1 );
	NumberClassesSpinCtrl->Enable( false );
	NumberClassesSpinCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer35->Add( NumberClassesSpinCtrl, 0, wxALL, 5 );

	m_staticText415 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Starts :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText415->Wrap( -1 );
	fgSizer35->Add( m_staticText415, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	NumberStartsSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxT("2"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 1000, 0 );
	NumberStartsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer35->Add( NumberStartsSpinCtrl, 0, wxALL, 5 );


	bSizer200->Add( fgSizer35, 0, wxALIGN_CENTER_VERTICAL, 5 );

	m_staticline144 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline144, 0, wxEXPAND | wxALL, 5 );

	wxFlexGridSizer* fgSizer36;
	fgSizer36 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer36->SetFlexibleDirection( wxBOTH );
	fgSizer36->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	SymmetryStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Symmetry : "), wxDefaultPosition, wxDefaultSize, 0 );
	SymmetryStaticText->Wrap( -1 );
	SymmetryStaticText->Enable( false );

	fgSizer36->Add( SymmetryStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	SymmetryComboBox = new wxComboBox( InputParamsPanel, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	SymmetryComboBox->Enable( false );
	SymmetryComboBox->SetMinSize( wxSize( 100,-1 ) );

	fgSizer36->Add( SymmetryComboBox, 0, wxALL, 5 );

	m_staticText264 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Cycles :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText264->Wrap( -1 );
	fgSizer36->Add( m_staticText264, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	NumberRoundsSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 5, 100, 40 );
	NumberRoundsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer36->Add( NumberRoundsSpinCtrl, 1, wxALL|wxEXPAND, 5 );


	bSizer200->Add( fgSizer36, 0, wxALIGN_CENTER_VERTICAL, 5 );


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


	InputParamsPanel->SetSizer( bSizer441 );
	InputParamsPanel->Layout();
	bSizer441->Fit( InputParamsPanel );
	bSizer451->Add( InputParamsPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer451, 0, wxEXPAND, 5 );

	m_staticline141 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline141, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();

	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer258;
	bSizer258 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText531 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText531->Wrap( -1 );
	m_staticText531->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	bSizer258->Add( m_staticText531, 0, wxALIGN_BOTTOM|wxALL, 5 );


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

	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Initial Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	InitialResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("40.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( InitialResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Final Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	FinalResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("9.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( FinalResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );

	GlobalMaskRadiusStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	GlobalMaskRadiusStaticText->Wrap( -1 );
	fgSizer1->Add( GlobalMaskRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	GlobalMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( GlobalMaskRadiusTextCtrl, 0, wxALL, 5 );

	InnerMaskRadiusStaticText1 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Inner Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	InnerMaskRadiusStaticText1->Wrap( -1 );
	fgSizer1->Add( InnerMaskRadiusStaticText1, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	InnerMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( InnerMaskRadiusTextCtrl, 0, wxALL, 5 );

	SearchRangeXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in X (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SearchRangeXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeXTextCtrl, 0, wxALL, 5 );

	SearchRangeYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in Y (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SearchRangeYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeYTextCtrl, 0, wxALL, 5 );

	UseAutoMaskingStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Use Auto-Masking?"), wxDefaultPosition, wxDefaultSize, 0 );
	UseAutoMaskingStaticText->Wrap( -1 );
	fgSizer1->Add( UseAutoMaskingStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer264;
	bSizer264 = new wxBoxSizer( wxHORIZONTAL );

	AutoMaskYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer264->Add( AutoMaskYesRadio, 0, wxALL, 5 );

	AutoMaskNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( AutoMaskNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer264, 1, wxEXPAND, 5 );

	m_staticText4151 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Auto Percent Used?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4151->Wrap( -1 );
	fgSizer1->Add( m_staticText4151, 0, wxALL, 5 );

	wxBoxSizer* bSizer2641;
	bSizer2641 = new wxBoxSizer( wxHORIZONTAL );

	AutoPercentUsedYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2641->Add( AutoPercentUsedYesRadio, 0, wxALL, 5 );

	AutoPercentUsedNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2641->Add( AutoPercentUsedNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2641, 1, wxEXPAND, 5 );

	InitialPercentUsedStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tInitial % Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	InitialPercentUsedStaticText->Wrap( -1 );
	fgSizer1->Add( InitialPercentUsedStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	StartPercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("9.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( StartPercentUsedTextCtrl, 0, wxALL, 5 );

	FinalPercentUsedStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tFinal % Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	FinalPercentUsedStaticText->Wrap( -1 );
	fgSizer1->Add( FinalPercentUsedStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	EndPercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("9.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( EndPercentUsedTextCtrl, 0, wxALL, 5 );

	AlwaysApplySymmetryStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Always Apply Symmetry?"), wxDefaultPosition, wxDefaultSize, 0 );
	AlwaysApplySymmetryStaticText->Wrap( -1 );
	fgSizer1->Add( AlwaysApplySymmetryStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer26611;
	bSizer26611 = new wxBoxSizer( wxHORIZONTAL );

	AlwaysApplySymmetryYesButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26611->Add( AlwaysApplySymmetryYesButton, 0, wxALL, 5 );

	AlwaysApplySymmetryNoButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26611->Add( AlwaysApplySymmetryNoButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26611, 1, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	m_staticText532 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("3D Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText532->Wrap( -1 );
	m_staticText532->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText532, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

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

	m_staticText662 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Class Averages"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText662->Wrap( -1 );
	m_staticText662->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText662, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText663 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Images per Class :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText663->Wrap( -1 );
	fgSizer1->Add( m_staticText663, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ImagesPerClassSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 500, 5 );
	ImagesPerClassSpinCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer1->Add( ImagesPerClassSpinCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );


	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer325;
	bSizer325 = new wxBoxSizer( wxVERTICAL );

	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();

	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );

	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );


	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer325->Add( OutputTextPanel, 40, wxEXPAND | wxALL, 5 );

	PlotPanel = new AbInitioPlotPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	PlotPanel->Hide();

	bSizer325->Add( PlotPanel, 60, wxEXPAND | wxALL, 5 );


	bSizer46->Add( bSizer325, 40, wxEXPAND, 5 );

	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );

	OrthResultsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OrthResultsPanel->Hide();

	wxBoxSizer* bSizer326;
	bSizer326 = new wxBoxSizer( wxVERTICAL );

	m_staticText377 = new wxStaticText( OrthResultsPanel, wxID_ANY, wxT("Orthogonal Slices / Projections"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText377->Wrap( -1 );
	m_staticText377->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer326->Add( m_staticText377, 0, wxALL, 5 );

	m_staticline91 = new wxStaticLine( OrthResultsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer326->Add( m_staticline91, 0, wxEXPAND | wxALL, 5 );

	ShowOrthDisplayPanel = new DisplayPanel( OrthResultsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer326->Add( ShowOrthDisplayPanel, 1, wxEXPAND | wxALL, 5 );


	OrthResultsPanel->SetSizer( bSizer326 );
	OrthResultsPanel->Layout();
	bSizer326->Fit( OrthResultsPanel );
	bSizer46->Add( OrthResultsPanel, 80, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );

	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );

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

	CurrentLineOne = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( CurrentLineOne, 0, wxEXPAND | wxALL, 5 );

	TakeCurrentResultButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Take Current"), wxDefaultPosition, wxDefaultSize, 0 );
	TakeCurrentResultButton->SetToolTip( wxT("This will terminate the job") );

	bSizer59->Add( TakeCurrentResultButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	CurrentLineTwo = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( CurrentLineTwo, 0, wxEXPAND | wxALL, 5 );

	TakeLastStartResultButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Take Last Start"), wxDefaultPosition, wxDefaultSize, 0 );
	TakeLastStartResultButton->SetToolTip( wxT("This will terminate the job") );

	bSizer59->Add( TakeLastStartResultButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );


	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer48->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );

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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AbInitio3DPanelParent::OnUpdateUI ) );
	ImageInputRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ClassAverageInputRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AbInitio3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TerminateButtonClick ), NULL, this );
	TakeCurrentResultButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeCurrentClicked ), NULL, this );
	TakeLastStartResultButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeLastStartClicked ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::StartRefinementClick ), NULL, this );
}

AbInitio3DPanelParent::~AbInitio3DPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AbInitio3DPanelParent::OnUpdateUI ) );
	ImageInputRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ClassAverageInputRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AbInitio3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TerminateButtonClick ), NULL, this );
	TakeCurrentResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeCurrentClicked ), NULL, this );
	TakeLastStartResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeLastStartClicked ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::StartRefinementClick ), NULL, this );

}

AbInitioPlotPanelParent::AbInitioPlotPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer278;
	bSizer278 = new wxBoxSizer( wxVERTICAL );

	my_notebook = new wxAuiNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	SigmaPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( SigmaPanel, wxT("Sigma"), false, wxNullBitmap );

	bSizer278->Add( my_notebook, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer278 );
	this->Layout();
}

AbInitioPlotPanelParent::~AbInitioPlotPanelParent()
{
}
