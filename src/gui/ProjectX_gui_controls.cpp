///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "AngularDistributionPlotPanel.h"
#include "AssetPickerComboPanel.h"
#include "PlotCurvePanel.h"
#include "PlotFSCPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_controls.h"

///////////////////////////////////////////////////////////////////////////

AssetPickerComboPanelParent::AssetPickerComboPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	bSizer436 = new wxBoxSizer( wxHORIZONTAL );

	AssetComboBox = new MemoryComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer436->Add( AssetComboBox, 100, wxALIGN_CENTER_VERTICAL|wxEXPAND, 0 );

	bSizer494 = new wxBoxSizer( wxVERTICAL );

	PreviousButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	PreviousButton->SetDefault();
	bSizer494->Add( PreviousButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0 );

	NextButton = new NoFocusBitmapButton( this, wxID_ADD, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	NextButton->SetDefault();
	bSizer494->Add( NextButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0 );


	bSizer436->Add( bSizer494, 0, wxALIGN_CENTER_VERTICAL, 5 );

	WindowSelectButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	WindowSelectButton->SetDefault();
	bSizer436->Add( WindowSelectButton, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, 5 );


	this->SetSizer( bSizer436 );
	this->Layout();
	bSizer436->Fit( this );

	// Connect Events
	this->Connect( wxEVT_SIZE, wxSizeEventHandler( AssetPickerComboPanelParent::OnSize ) );
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetPickerComboPanelParent::OnUpdateUI ) );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnPreviousButtonClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnNextButtonClick ), NULL, this );
}

AssetPickerComboPanelParent::~AssetPickerComboPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_SIZE, wxSizeEventHandler( AssetPickerComboPanelParent::OnSize ) );
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetPickerComboPanelParent::OnUpdateUI ) );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnPreviousButtonClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnNextButtonClick ), NULL, this );

}

FilterDialog::FilterDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );

	MainBoxSizer = new wxBoxSizer( wxVERTICAL );

	m_staticText64 = new wxStaticText( this, wxID_ANY, wxT("Filter By :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText64->Wrap( -1 );
	m_staticText64->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	MainBoxSizer->Add( m_staticText64, 0, wxALL, 5 );

	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );

	FilterScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL|wxBORDER_SUNKEN );
	FilterScrollPanel->SetScrollRate( 5, 5 );
	FilterBoxSizer = new wxBoxSizer( wxVERTICAL );


	FilterScrollPanel->SetSizer( FilterBoxSizer );
	FilterScrollPanel->Layout();
	FilterBoxSizer->Fit( FilterScrollPanel );
	MainBoxSizer->Add( FilterScrollPanel, 1, wxALL|wxEXPAND, 5 );

	m_staticText81 = new wxStaticText( this, wxID_ANY, wxT("\nSort By :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText81->Wrap( -1 );
	m_staticText81->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	MainBoxSizer->Add( m_staticText81, 0, wxALL, 5 );

	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );

	SortScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	SortScrollPanel->SetScrollRate( 5, 5 );
	SortSizer = new wxGridSizer( 0, 3, 0, 0 );


	SortScrollPanel->SetSizer( SortSizer );
	SortScrollPanel->Layout();
	SortSizer->Fit( SortScrollPanel );
	MainBoxSizer->Add( SortScrollPanel, 0, wxEXPAND | wxALL, 5 );

	m_staticline21 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline21, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );

	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );

	FilterButton = new wxButton( this, wxID_ANY, wxT("Filter/Sort"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( FilterButton, 0, wxALL, 5 );


	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );


	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );


	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnCancelClick ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnFilterClick ), NULL, this );
}

FilterDialog::~FilterDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnCancelClick ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnFilterClick ), NULL, this );

}

FSCPanel::FSCPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxVERTICAL );

	TitleSizer = new wxBoxSizer( wxHORIZONTAL );

	TitleStaticText = new wxStaticText( this, wxID_ANY, wxT("Current FSC"), wxDefaultPosition, wxDefaultSize, 0 );
	TitleStaticText->Wrap( -1 );
	TitleStaticText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	TitleSizer->Add( TitleStaticText, 0, wxALIGN_BOTTOM|wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	TitleSizer->Add( 0, 0, 1, wxEXPAND, 5 );

	EstimatedResolutionLabel = new wxStaticText( this, wxID_ANY, wxT("Best Est. Resolution : "), wxDefaultPosition, wxDefaultSize, 0 );
	EstimatedResolutionLabel->Wrap( -1 );
	TitleSizer->Add( EstimatedResolutionLabel, 0, wxALIGN_BOTTOM|wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	EstimatedResolutionText = new wxStaticText( this, wxID_ANY, wxT("00.00  ‎Å"), wxDefaultPosition, wxDefaultSize, 0 );
	EstimatedResolutionText->Wrap( -1 );
	TitleSizer->Add( EstimatedResolutionText, 0, wxALIGN_BOTTOM|wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticline104 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	TitleSizer->Add( m_staticline104, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer505;
	bSizer505 = new wxBoxSizer( wxHORIZONTAL );

	SaveButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	SaveButton->SetDefault();
	bSizer505->Add( SaveButton, 1, wxEXPAND|wxLEFT, 5 );

	FSCDetailsButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	FSCDetailsButton->SetDefault();
	bSizer505->Add( FSCDetailsButton, 1, wxEXPAND|wxRIGHT, 5 );


	TitleSizer->Add( bSizer505, 0, wxEXPAND, 5 );


	bSizer200->Add( TitleSizer, 0, wxEXPAND, 5 );

	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer200->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );

	PlotPanel = new PlotFSCPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer200->Add( PlotPanel, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer200 );
	this->Layout();

	// Connect Events
	SaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::SaveImageClick ), NULL, this );
	FSCDetailsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::PopupTextClick ), NULL, this );
}

FSCPanel::~FSCPanel()
{
	// Disconnect Events
	SaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::SaveImageClick ), NULL, this );
	FSCDetailsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::PopupTextClick ), NULL, this );

}

DisplayPanelParent::DisplayPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	MainSizer = new wxBoxSizer( wxVERTICAL );

	Toolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_FLAT|wxTB_HORIZONTAL );
	Toolbar->Realize();

	MainSizer->Add( Toolbar, 0, wxEXPAND, 5 );


	this->SetSizer( MainSizer );
	this->Layout();
}

DisplayPanelParent::~DisplayPanelParent()
{
}

DisplayManualDialogParent::DisplayManualDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	MainSizer = new wxBoxSizer( wxVERTICAL );


	MainSizer->Add( 400, 200, 0, wxFIXED_MINSIZE, 5 );

	m_staticline58 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline58, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer262;
	bSizer262 = new wxBoxSizer( wxHORIZONTAL );


	bSizer262->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText315 = new wxStaticText( this, wxID_ANY, wxT("Min : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText315->Wrap( -1 );
	bSizer262->Add( m_staticText315, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	minimum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER|wxTE_CENTER );
	bSizer262->Add( minimum_text_ctrl, 0, wxALL, 5 );

	m_staticText316 = new wxStaticText( this, wxID_ANY, wxT("/"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText316->Wrap( -1 );
	bSizer262->Add( m_staticText316, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	maximum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER|wxTE_CENTER );
	bSizer262->Add( maximum_text_ctrl, 0, wxALL, 5 );

	m_staticText317 = new wxStaticText( this, wxID_ANY, wxT(": Max"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	bSizer262->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer262->Add( 0, 0, 1, wxEXPAND, 5 );


	MainSizer->Add( bSizer262, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );


	bSizer265->Add( 0, 0, 1, wxEXPAND, 5 );

	Toolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL );
	Toolbar->Realize();

	bSizer265->Add( Toolbar, 0, 0, 5 );


	bSizer265->Add( 0, 0, 1, wxEXPAND, 5 );


	MainSizer->Add( bSizer265, 0, wxEXPAND, 5 );

	m_staticline61 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline61, 0, wxEXPAND | wxALL, 5 );

	wxGridSizer* gSizer13;
	gSizer13 = new wxGridSizer( 0, 2, 0, 0 );

	histogram_checkbox = new wxCheckBox( this, wxID_ANY, wxT("Use Entire File For Histogram"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer13->Add( histogram_checkbox, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );

	live_checkbox = new wxCheckBox( this, wxID_ANY, wxT("Live Update of Display"), wxDefaultPosition, wxDefaultSize, 0 );
	live_checkbox->SetValue(true);
	gSizer13->Add( live_checkbox, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );


	MainSizer->Add( gSizer13, 0, wxEXPAND, 5 );

	m_staticline63 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline63, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer264;
	bSizer264 = new wxBoxSizer( wxHORIZONTAL );


	bSizer264->Add( 0, 0, 1, wxEXPAND, 5 );

	m_button94 = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( m_button94, 0, wxALL, 5 );

	m_button95 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( m_button95, 0, wxALL, 5 );


	bSizer264->Add( 0, 0, 1, wxEXPAND, 5 );


	MainSizer->Add( bSizer264, 0, wxEXPAND, 5 );


	this->SetSizer( MainSizer );
	this->Layout();
	MainSizer->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	this->Connect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( DisplayManualDialogParent::OnClose ) );
	this->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnLeftDown ) );
	this->Connect( wxEVT_MOTION, wxMouseEventHandler( DisplayManualDialogParent::OnMotion ) );
	this->Connect( wxEVT_PAINT, wxPaintEventHandler( DisplayManualDialogParent::OnPaint ) );
	this->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnRightDown ) );
	minimum_text_ctrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnLowChange ), NULL, this );
	maximum_text_ctrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnHighChange ), NULL, this );
	histogram_checkbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnHistogramCheck ), NULL, this );
	live_checkbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnRealtimeCheck ), NULL, this );
	m_button94->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonOK ), NULL, this );
	m_button95->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonCancel ), NULL, this );
}

DisplayManualDialogParent::~DisplayManualDialogParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( DisplayManualDialogParent::OnClose ) );
	this->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnLeftDown ) );
	this->Disconnect( wxEVT_MOTION, wxMouseEventHandler( DisplayManualDialogParent::OnMotion ) );
	this->Disconnect( wxEVT_PAINT, wxPaintEventHandler( DisplayManualDialogParent::OnPaint ) );
	this->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnRightDown ) );
	minimum_text_ctrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnLowChange ), NULL, this );
	maximum_text_ctrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnHighChange ), NULL, this );
	histogram_checkbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnHistogramCheck ), NULL, this );
	live_checkbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnRealtimeCheck ), NULL, this );
	m_button94->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonOK ), NULL, this );
	m_button95->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonCancel ), NULL, this );

}

ErrorDialog::ErrorDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,500 ), wxDefaultSize );

	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer36;
	bSizer36 = new wxBoxSizer( wxHORIZONTAL );

	m_bitmap1 = new wxStaticBitmap( this, wxID_ANY, wxArtProvider::GetBitmap( wxART_ERROR, wxART_OTHER ), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer36->Add( m_bitmap1, 0, wxALL, 5 );

	m_staticText25 = new wxStaticText( this, wxID_ANY, wxT(": ERROR"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText25->Wrap( -1 );
	bSizer36->Add( m_staticText25, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer35->Add( bSizer36, 0, wxEXPAND, 5 );

	ErrorText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer35->Add( ErrorText, 100, wxALL|wxEXPAND, 5 );

	m_button23 = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer35->Add( m_button23, 0, wxALIGN_RIGHT|wxALL, 5 );


	this->SetSizer( bSizer35 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	m_button23->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ErrorDialog::OnClickOK ), NULL, this );
}

ErrorDialog::~ErrorDialog()
{
	// Disconnect Events
	m_button23->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ErrorDialog::OnClickOK ), NULL, this );

}

ListCtrlDialog::ListCtrlDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer321;
	bSizer321 = new wxBoxSizer( wxVERTICAL );

	MyListCtrl = new AssetPickerListCtrl( this, wxID_ANY, wxDefaultPosition, wxSize( 500,300 ), wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer321->Add( MyListCtrl, 0, wxALL, 5 );

	wxBoxSizer* bSizer322;
	bSizer322 = new wxBoxSizer( wxHORIZONTAL );


	bSizer322->Add( 0, 0, 1, wxEXPAND, 5 );

	m_button108 = new wxButton( this, wxID_OK, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer322->Add( m_button108, 0, wxALL, 5 );

	m_button109 = new wxButton( this, wxID_CANCEL, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer322->Add( m_button109, 0, wxALL, 5 );


	bSizer322->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer321->Add( bSizer322, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer321 );
	this->Layout();
	bSizer321->Fit( this );

	this->Centre( wxBOTH );
}

ListCtrlDialog::~ListCtrlDialog()
{
}

PopupTextDialogParent::PopupTextDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,600 ), wxDefaultSize );

	wxBoxSizer* bSizer363;
	bSizer363 = new wxBoxSizer( wxVERTICAL );

	OutputTextCtrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxTE_MULTILINE|wxTE_READONLY );
	OutputTextCtrl->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxT("Fixed") ) );

	bSizer363->Add( OutputTextCtrl, 1, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer380;
	bSizer380 = new wxBoxSizer( wxHORIZONTAL );

	CloseButton = new wxButton( this, wxID_CLOSE, wxT("Close"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( CloseButton, 0, wxALL, 5 );


	bSizer380->Add( 0, 0, 1, wxEXPAND, 5 );

	ClipBoardButton = new wxButton( this, wxID_COPY, wxT("Copy All"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( ClipBoardButton, 0, wxALL, 5 );

	m_button146 = new wxButton( this, wxID_SAVE, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( m_button146, 0, wxALL, 5 );


	bSizer363->Add( bSizer380, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer363 );
	this->Layout();
	bSizer363->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	CloseButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCopyToClipboardClick ), NULL, this );
	m_button146->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnSaveButtonClick ), NULL, this );
}

PopupTextDialogParent::~PopupTextDialogParent()
{
	// Disconnect Events
	CloseButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCopyToClipboardClick ), NULL, this );
	m_button146->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnSaveButtonClick ), NULL, this );

}

LargeAngularPlotDialogParent::LargeAngularPlotDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,800 ), wxDefaultSize );

	wxBoxSizer* bSizer363;
	bSizer363 = new wxBoxSizer( wxVERTICAL );

	AngularPlotPanel = new AngularDistributionPlotPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer363->Add( AngularPlotPanel, 1, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer380;
	bSizer380 = new wxBoxSizer( wxHORIZONTAL );

	CloseButton = new wxButton( this, wxID_CLOSE, wxT("Close"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( CloseButton, 0, wxALL, 5 );


	bSizer380->Add( 0, 0, 1, wxEXPAND, 5 );

	ClipBoardButton = new wxButton( this, wxID_COPY, wxT("Copy"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( ClipBoardButton, 0, wxALL, 5 );

	SaveButton = new wxButton( this, wxID_SAVE, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( SaveButton, 0, wxALL, 5 );


	bSizer363->Add( bSizer380, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer363 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	CloseButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCopyToClipboardClick ), NULL, this );
	SaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnSaveButtonClick ), NULL, this );
}

LargeAngularPlotDialogParent::~LargeAngularPlotDialogParent()
{
	// Disconnect Events
	CloseButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCopyToClipboardClick ), NULL, this );
	SaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnSaveButtonClick ), NULL, this );

}

RefinementParametersDialogParent::RefinementParametersDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,800 ), wxDefaultSize );

	wxBoxSizer* bSizer363;
	bSizer363 = new wxBoxSizer( wxVERTICAL );

	ClassToolBar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NOICONS|wxTB_TEXT );
	m_staticText831 = new wxStaticText( ClassToolBar, wxID_ANY, wxT("Select Class : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText831->Wrap( -1 );
	ClassToolBar->AddControl( m_staticText831 );
	ClassToolBar->Realize();

	bSizer363->Add( ClassToolBar, 0, wxEXPAND, 5 );

	m_staticline137 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer363->Add( m_staticline137, 0, wxEXPAND | wxALL, 5 );

	ParameterListCtrl = new RefinementParametersListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_VIRTUAL|wxLC_VRULES );
	bSizer363->Add( ParameterListCtrl, 1, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer380;
	bSizer380 = new wxBoxSizer( wxHORIZONTAL );

	CloseButton = new wxButton( this, wxID_CLOSE, wxT("Close"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( CloseButton, 0, wxALL, 5 );


	bSizer380->Add( 0, 0, 1, wxEXPAND, 5 );

	SaveButton = new wxButton( this, wxID_SAVE, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( SaveButton, 0, wxALL, 5 );


	bSizer363->Add( bSizer380, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer363 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	CloseButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnCloseButtonClick ), NULL, this );
	SaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnSaveButtonClick ), NULL, this );
}

RefinementParametersDialogParent::~RefinementParametersDialogParent()
{
	// Disconnect Events
	CloseButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnCloseButtonClick ), NULL, this );
	SaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnSaveButtonClick ), NULL, this );

}

DistributionPlotDialogParent::DistributionPlotDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer489;
	bSizer489 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer493;
	bSizer493 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer497;
	bSizer497 = new wxBoxSizer( wxVERTICAL );


	bSizer497->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText64711 = new wxStaticText( this, wxID_ANY, wxT("Max"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText64711->Wrap( -1 );
	bSizer497->Add( m_staticText64711, 0, wxALL, 5 );

	UpperBoundYNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer497->Add( UpperBoundYNumericCtrl, 0, wxALL, 5 );

	m_staticText6472 = new wxStaticText( this, wxID_ANY, wxT("Min"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6472->Wrap( -1 );
	bSizer497->Add( m_staticText6472, 0, wxALL, 5 );

	LowerBoundYNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer497->Add( LowerBoundYNumericCtrl, 0, wxALL, 5 );


	bSizer493->Add( bSizer497, 0, wxEXPAND, 5 );

	PlotCurvePanelInstance = new PlotCurvePanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer493->Add( PlotCurvePanelInstance, 1, wxEXPAND | wxALL, 5 );


	bSizer489->Add( bSizer493, 1, wxEXPAND, 5 );

	wxBoxSizer* DistributionPlotButtonsSizer;
	DistributionPlotButtonsSizer = new wxBoxSizer( wxHORIZONTAL );

	wxArrayString DataSeriesToPlotChoiceChoices;
	DataSeriesToPlotChoice = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, DataSeriesToPlotChoiceChoices, 0 );
	DataSeriesToPlotChoice->SetSelection( 0 );
	DistributionPlotButtonsSizer->Add( DataSeriesToPlotChoice, 0, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer492;
	bSizer492 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText647 = new wxStaticText( this, wxID_ANY, wxT("Min"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText647->Wrap( -1 );
	bSizer492->Add( m_staticText647, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowerBoundXNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer492->Add( LowerBoundXNumericCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticText6471 = new wxStaticText( this, wxID_ANY, wxT("Max"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6471->Wrap( -1 );
	bSizer492->Add( m_staticText6471, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	UpperBoundXNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer492->Add( UpperBoundXNumericCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	DistributionPlotButtonsSizer->Add( bSizer492, 1, wxEXPAND, 5 );


	DistributionPlotButtonsSizer->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer489->Add( DistributionPlotButtonsSizer, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer491;
	bSizer491 = new wxBoxSizer( wxHORIZONTAL );


	bSizer491->Add( 0, 0, 1, wxEXPAND, 5 );

	SaveTXTButton = new wxButton( this, wxID_ANY, wxT("Save TXT"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer491->Add( SaveTXTButton, 0, wxALL, 5 );

	SavePNGButton = new wxButton( this, wxID_ANY, wxT("Save PNG"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer491->Add( SavePNGButton, 0, wxALL, 5 );


	bSizer489->Add( bSizer491, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer489 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	UpperBoundYNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYKillFocus ), NULL, this );
	UpperBoundYNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYSetFocus ), NULL, this );
	UpperBoundYNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundYTextEnter ), NULL, this );
	LowerBoundYNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYKillFocus ), NULL, this );
	LowerBoundYNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYSetFocus ), NULL, this );
	LowerBoundYNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundYTextEnter ), NULL, this );
	DataSeriesToPlotChoice->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( DistributionPlotDialogParent::OnDataSeriesToPlotChoice ), NULL, this );
	LowerBoundXNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXKillFocus ), NULL, this );
	LowerBoundXNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXSetFocus ), NULL, this );
	LowerBoundXNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundXTextEnter ), NULL, this );
	UpperBoundXNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXKillFocus ), NULL, this );
	UpperBoundXNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXSetFocus ), NULL, this );
	UpperBoundXNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundXTextEnter ), NULL, this );
	SaveTXTButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSaveTXTButtonClick ), NULL, this );
	SavePNGButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSavePNGButtonClick ), NULL, this );
}

DistributionPlotDialogParent::~DistributionPlotDialogParent()
{
	// Disconnect Events
	UpperBoundYNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYKillFocus ), NULL, this );
	UpperBoundYNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYSetFocus ), NULL, this );
	UpperBoundYNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundYTextEnter ), NULL, this );
	LowerBoundYNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYKillFocus ), NULL, this );
	LowerBoundYNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYSetFocus ), NULL, this );
	LowerBoundYNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundYTextEnter ), NULL, this );
	DataSeriesToPlotChoice->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( DistributionPlotDialogParent::OnDataSeriesToPlotChoice ), NULL, this );
	LowerBoundXNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXKillFocus ), NULL, this );
	LowerBoundXNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXSetFocus ), NULL, this );
	LowerBoundXNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundXTextEnter ), NULL, this );
	UpperBoundXNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXKillFocus ), NULL, this );
	UpperBoundXNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXSetFocus ), NULL, this );
	UpperBoundXNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundXTextEnter ), NULL, this );
	SaveTXTButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSaveTXTButtonClick ), NULL, this );
	SavePNGButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSavePNGButtonClick ), NULL, this );

}
