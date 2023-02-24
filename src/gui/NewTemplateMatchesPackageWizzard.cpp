//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyImageAssetPanel*  image_asset_panel;
extern MyVolumeAssetPanel* volume_asset_panel;

NewTemplateMatchesPackageWizard::NewTemplateMatchesPackageWizard(wxWindow* parent)
    : NewTemplateMatchesPackageWizardParent(parent) {
    page = new TemplateMatchesWizardPage(this);

    GetPageAreaSizer( )->Add(page);
    //GetPageAreaSizer()->Add(particle_group_page);
    //GetPageAreaSizer()->Add(number_of_classes_page);
    //GetPageAreaSizer()->Add(box_size_page);

    Bind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(NewTemplateMatchesPackageWizard::OnUpdateUI), this);
}

NewTemplateMatchesPackageWizard::~NewTemplateMatchesPackageWizard( ) {
    /*
	delete template_page;
	delete particle_group_page;
	delete number_of_classes_page;
	delete box_size_page;
	delete class_setup_page;
	delete initial_reference_page;
	delete symmetry_page;
	delete molecular_weight_page;
	delete largest_dimension_page;
	delete particle_group_page;
	delete class_selection_page;
*/
    Unbind(wxEVT_UPDATE_UI, wxUpdateUIEventHandler(NewTemplateMatchesPackageWizard::OnUpdateUI), this);
}

void NewTemplateMatchesPackageWizard::OnUpdateUI(wxUpdateUIEvent& event) {
    EnableNextButton( );
}

void NewTemplateMatchesPackageWizard::DisableNextButton( ) {
    wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
    if ( win )
        win->Enable(false);
}

void NewTemplateMatchesPackageWizard::EnableNextButton( ) {
    wxWindow* win = wxWindow::FindWindowById(wxID_FORWARD);
    if ( win )
        win->Enable(true);
}

void NewTemplateMatchesPackageWizard::PageChanging(wxWizardEvent& event) {
}

void NewTemplateMatchesPackageWizard::PageChanged(wxWizardEvent& event) {

    if ( event.GetPage( ) == page ) {
        if ( page->my_panel->InfoText->has_autowrapped == false ) {
            page->Freeze( );
            page->my_panel->InfoText->AutoWrap( );
            page->Layout( );
            page->Thaw( );
        }
    }
}

void NewTemplateMatchesPackageWizard::OnFinished(wxWizardEvent& event) {
}

////////////////

// TEMPLATE MATCHES PAGE

/////////////////

TemplateMatchesWizardPage::TemplateMatchesWizardPage(NewTemplateMatchesPackageWizard* parent, const wxBitmap& bitmap)
    : wxWizardPage(parent, bitmap) {
    Freeze( );
    wizard_pointer = parent;
    wxBoxSizer* main_sizer;
    my_panel = new TemplateMatchesWizardPanel(this);

    main_sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(main_sizer);
    main_sizer->Fit(this);
    main_sizer->Add(my_panel);

    Thaw( );

    //my_panel->GroupComboBox->Bind(wxEVT_COMBOBOX, wxCommandEventHandler(TemplateWizardPage::RefinementPackageChanged), this);
}

TemplateMatchesWizardPage::~TemplateMatchesWizardPage( ) {
    //my_panel->GroupComboBox->Unbind(wxEVT_COMBOBOX, wxComboEventHandler( TemplateWizardPage::RefinementPackageChanged), this);
}
