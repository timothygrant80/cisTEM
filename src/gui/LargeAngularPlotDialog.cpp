#include "../core/gui_core_headers.h"

LargeAngularPlotDialog::LargeAngularPlotDialog(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style)
    : LargeAngularPlotDialogParent(parent, id, title, pos, size, style) {
    int frame_width;
    int frame_height;
    int frame_position_x;
    int frame_position_y;

    main_frame->GetClientSize(&frame_width, &frame_height);
    main_frame->GetPosition(&frame_position_x, &frame_position_y);

    SetSize(wxSize(frame_height, myroundint(float(frame_height * 0.9f))));

    // ok so how big is this dialog now?

    int new_x_pos = (frame_position_x + (frame_width / 2) - (frame_height / 2));
    int new_y_pos = (frame_position_y + (frame_height / 2) - myroundint(float(frame_height) * 0.9f / 2.0f));

    Move(new_x_pos, new_y_pos);
}

void LargeAngularPlotDialog::OnCopyToClipboardClick(wxCommandEvent& event) {
    if ( wxTheClipboard->Open( ) ) {
        //  wxTheClipboard->SetData( new wxTextDataObject(OutputTextCtrl->GetValue()) );
        wxTheClipboard->SetData(new wxBitmapDataObject(AngularPlotPanel->buffer_bitmap));
        wxTheClipboard->Close( );
    }
}

void LargeAngularPlotDialog::OnSaveButtonClick(wxCommandEvent& event) {
    ProperOverwriteCheckSaveDialog* saveFileDialog;
    saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save png image"), "PNG files (*.png)|*.png", ".png");
    if ( saveFileDialog->ShowModal( ) == wxID_CANCEL ) {
        saveFileDialog->Destroy( );
        return;
    }

    // save the file then..

    AngularPlotPanel->buffer_bitmap.SaveFile(saveFileDialog->ReturnProperPath( ), wxBITMAP_TYPE_PNG);
    saveFileDialog->Destroy( );
}

void LargeAngularPlotDialog::OnCloseButtonClick(wxCommandEvent& event) {
    EndModal(0);
}
