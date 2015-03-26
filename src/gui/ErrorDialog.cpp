#include "ErrorDialog.h"

MyErrorDialog::MyErrorDialog( wxWindow* parent )
:
ErrorDialog( parent )
{

}

void MyErrorDialog::OnClickOK( wxCommandEvent& event )
{
	Destroy();
}
