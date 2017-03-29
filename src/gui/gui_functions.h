void ConvertImageToBitmap(Image *input_image, wxBitmap *output_bitmap, bool auto_contrast = false);
void GetMultilineTextExtent	(wxDC *wanted_dc, const wxString & string, int &width, int &height);
void FillGroupComboBoxSlave( wxComboBox *GroupComboBox, bool include_all_images_group = true );
void FillParticlePositionsGroupComboBox(wxComboBox *GroupComboBox, bool include_all_particle_positions_group = true);

void AppendVolumeAssetsToComboBox(wxComboBox *GroupComboBox);
void AppendRefinementPackagesToComboBox(wxComboBox *GroupComboBox);
wxArrayString GetRecentProjectsFromSettings();

