#ifndef __MyRelionExportDialog__
#define __MyRelionExportDialog__

class MyRelionExportDialog : public RelionExportDialog
{
public:
	MyRelionExportDialog( wxWindow* parent );

	void OnCancelButtonClick( wxCommandEvent & event );
	void OnExportButtonClick( wxCommandEvent & event );
	void OnOutputImageStackFileChanged( wxFileDirPickerEvent & event );
	void OnNormalizeCheckBox( wxCommandEvent & event );


};

#endif
