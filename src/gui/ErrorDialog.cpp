//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyErrorDialog::MyErrorDialog(wxWindow* parent)
    : ErrorDialog(parent) {
}

void MyErrorDialog::OnClickOK(wxCommandEvent& event) {
    Destroy( );
}
