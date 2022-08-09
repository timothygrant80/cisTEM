#include "../core/gui_core_headers.h"

extern MyRefinementResultsPanel* refinement_results_panel;

RefinementParametersDialog::RefinementParametersDialog(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style)
    : RefinementParametersDialogParent(parent, id, title, pos, size, style) {
    int frame_width;
    int frame_height;
    int frame_position_x;
    int frame_position_y;
    int columns_width = 0;

    wxToolBarToolBase* current_button_pointer;

    Bind(wxEVT_CHAR_HOOK, &RefinementParametersDialog::OnCharHook, this);

    current_class = 0;
    ParameterListCtrl->SetParent(this);

    ParameterListCtrl->Freeze( );
    ParameterListCtrl->ClearAll( );
    ParameterListCtrl->InsertColumn(0, wxT("Position In Stack."), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(1, wxT("Psi Angle (°)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(2, wxT("Theta Angle (°)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(3, wxT("Phi Angle (°)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(4, wxT("X-Shift (Å)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(5, wxT("Y-Shift (Å)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(6, wxT("Defocus 1 (Å)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(7, wxT("Defocus 2 (Å)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(8, wxT("Defocus Angle (°)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(9, wxT("Phase Shift (°)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(10, wxT("Beam Tilt X (mrad)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(11, wxT("Beam Tilt Y (mrad)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(12, wxT("Image Shift X (Å)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(13, wxT("Image Shift Y (Å)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(14, wxT("Voltage (kV)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(15, wxT("Cs (mm)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(16, wxT("Amplitude Contrast"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(17, wxT("Pixel Size (Å)"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(18, wxT("Occupancy"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(19, wxT("logP"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(20, wxT("Sigma"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(21, wxT("Score"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);
    ParameterListCtrl->InsertColumn(22, wxT("Image Active?"), wxLIST_FORMAT_CENTRE, wxLIST_AUTOSIZE_USEHEADER);

    ParameterListCtrl->SetItemCount(refinement_results_panel->buffered_full_refinement->number_of_particles);
    ParameterListCtrl->RefreshItems(0, ParameterListCtrl->GetItemCount( ) - 1);

    for ( int counter = 0; counter < ParameterListCtrl->GetColumnCount( ); counter++ ) {
        ParameterListCtrl->SetColumnWidth(counter, ParameterListCtrl->ReturnGuessAtColumnTextWidth(counter));
    }

    ParameterListCtrl->EnableAlternateRowColours( );
    ParameterListCtrl->Thaw( );

    for ( int class_counter = 1; class_counter <= refinement_results_panel->buffered_full_refinement->number_of_classes; class_counter++ ) {
        if ( class_counter < 10 )
            current_button_pointer = ClassToolBar->AddTool(wxID_ANY, wxString::Format(wxT(" %i "), class_counter), wxNullBitmap, wxNullBitmap, wxITEM_RADIO, wxEmptyString, wxEmptyString, NULL);
        else
            current_button_pointer = ClassToolBar->AddTool(wxID_ANY, wxString::Format(wxT("%i"), class_counter), wxNullBitmap, wxNullBitmap, wxITEM_RADIO, wxEmptyString, wxEmptyString, NULL);

        class_button_ids.Add(current_button_pointer->GetId( ));
    }

    ClassToolBar->Realize( );
    ClassToolBar->Layout( );

    ClassToolBar->Bind(wxEVT_TOOL, &RefinementParametersDialog::OnSelectionChange, this);

    for ( int column_counter = 0; column_counter < ParameterListCtrl->GetColumnCount( ); column_counter++ ) {
        columns_width += ParameterListCtrl->GetColumnWidth(column_counter);
    }

    columns_width += 100;

    main_frame->GetClientSize(&frame_width, &frame_height);
    main_frame->GetPosition(&frame_position_x, &frame_position_y);

    SetClientSize(wxSize(columns_width, myroundint(float(frame_height * 0.95f))));

    // ok so how big is this dialog now?

    int new_x_pos = (frame_position_x + (frame_width / 2) - (columns_width / 2));
    int new_y_pos = (frame_position_y + (frame_height / 2) - myroundint(float(frame_height) * 0.95f / 2.0f));

    Move(new_x_pos, new_y_pos);

    ParameterListCtrl->SetFocus( );
}

void RefinementParametersDialog::OnSelectionChange(wxCommandEvent& event) {
    wxToolBarToolBase* tool;
    tool = ClassToolBar->FindById(event.GetId( ));

    if ( tool != NULL ) {
        long button_number;

        wxString button_label = tool->GetLabel( );
        button_label          = button_label.Trim(true);
        button_label          = button_label.Trim(false);

        if ( button_label.ToLong(&button_number) == true ) {
            //	wxPrintf("clicked on button %li\n", button_number);
            current_class = button_number - 1;
            ParameterListCtrl->RefreshItems(0, ParameterListCtrl->GetItemCount( ) - 1);
        }
    }
}

void RefinementParametersDialog::OnCharHook(wxKeyEvent& event) {

    if ( event.GetKeyCode( ) == WXK_LEFT ) {
        current_class--;
        if ( current_class < 0 ) {
            current_class = refinement_results_panel->buffered_full_refinement->number_of_classes - 1;
        }

        SetActiveClassButton(current_class);
        ParameterListCtrl->RefreshItems(0, ParameterListCtrl->GetItemCount( ) - 1);
    }
    else if ( event.GetKeyCode( ) == WXK_RIGHT ) {
        current_class++;

        if ( current_class >= refinement_results_panel->buffered_full_refinement->number_of_classes ) {
            current_class = 0;
        }

        SetActiveClassButton(current_class);
        ParameterListCtrl->RefreshItems(0, ParameterListCtrl->GetItemCount( ) - 1);
    }

    event.Skip( );
}

void RefinementParametersDialog::SetActiveClassButton(int wanted_class) {
    ClassToolBar->ToggleTool(class_button_ids[wanted_class], true);
}

void RefinementParametersDialog::OnSaveButtonClick(wxCommandEvent& event) {

    ProperOverwriteCheckSaveDialog* saveFileDialog;
    saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save par file"), "par files (*.par)|*.par", ".par");
    if ( saveFileDialog->ShowModal( ) == wxID_CANCEL ) {
        saveFileDialog->Destroy( );
        return;
    }

    // save the file then..
    refinement_results_panel->buffered_full_refinement->WriteSingleClassFrealignParameterFile(saveFileDialog->ReturnProperPath( ), current_class);
    saveFileDialog->Destroy( );
}

void RefinementParametersDialog::OnCloseButtonClick(wxCommandEvent& event) {
    EndModal(0);
}
