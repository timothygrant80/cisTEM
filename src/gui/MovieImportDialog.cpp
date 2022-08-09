//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel* movie_asset_panel;
extern MyMainFrame*       main_frame;

MyMovieImportDialog::MyMovieImportDialog(wxWindow* parent)
    : MovieImportDialog(parent) {
    int list_height;
    int list_width;

    PathListCtrl->GetClientSize(&list_width, &list_height);
    PathListCtrl->InsertColumn(0, "Files", wxLIST_FORMAT_LEFT, list_width);

    DesiredPixelSizeTextCtrl->SetMinMaxValue(0, 100);
    DesiredPixelSizeTextCtrl->SetPrecision(4);
    MajorScaleTextCtrl->SetMinMaxValue(0, FLT_MAX);
    MinorScaleTextCtrl->SetMinMaxValue(0, FLT_MAX);
    MajorScaleTextCtrl->SetPrecision(3);
    MinorScaleTextCtrl->SetPrecision(3);

    // do we have defaults?

    if ( main_frame->current_project.database.DoesTableExist("MOVIE_IMPORT_DEFAULTS") == true ) {
        float    default_voltage;
        float    default_spherical_aberration;
        float    default_pixel_size;
        float    default_exposure_per_frame;
        bool     default_movies_are_gain_corrected;
        wxString default_gain_reference_filename;
        bool     default_movies_are_dark_corrected;
        wxString default_dark_reference_filename;
        bool     default_resample_movies;
        float    default_desired_pixel_size;
        bool     default_correct_mag_distortion;
        float    default_mag_distortion_angle;
        float    default_mag_distortion_major_scale;
        float    default_mag_distortion_minor_scale;
        bool     default_protein_is_white;
        int      default_eer_super_res_factor;
        int      default_eer_frames_per_image;

        main_frame->current_project.database.GetMovieImportDefaults(default_voltage, default_spherical_aberration, default_pixel_size, default_exposure_per_frame, default_movies_are_gain_corrected, default_gain_reference_filename, default_movies_are_dark_corrected, default_dark_reference_filename, default_resample_movies, default_desired_pixel_size, default_correct_mag_distortion, default_mag_distortion_angle, default_mag_distortion_major_scale, default_mag_distortion_minor_scale, default_protein_is_white, default_eer_super_res_factor, default_eer_frames_per_image);

        VoltageCombo->ChangeValue(wxString::Format("%.0f", default_voltage));
        CsText->ChangeValue(wxString::Format("%.2f", default_spherical_aberration));
        PixelSizeText->ChangeValue(wxString::Format("%.4f", default_pixel_size));
        DoseText->ChangeValue(wxString::Format("%.2f", default_exposure_per_frame));

        EerNumberOfFramesSpinCtrl->SetValue(default_eer_frames_per_image);
        EerSuperResFactorChoice->SetStringSelection(wxString::Format("%i", default_eer_super_res_factor));

        if ( default_movies_are_gain_corrected == true ) {
            ApplyGainImageCheckbox->SetValue(false);
        }
        else {
            ApplyGainImageCheckbox->SetValue(true);
        }

        if ( default_movies_are_dark_corrected == true ) {
            ApplyDarkImageCheckbox->SetValue(false);
        }
        else {
            ApplyDarkImageCheckbox->SetValue(true);
        }

        // don't set the gain/dark  references as these is likely to be different, and i want to force users to have to pick the correct one.

        if ( default_resample_movies == true ) {
            ResampleMoviesCheckBox->SetValue(true);
        }
        else {
            ResampleMoviesCheckBox->SetValue(false);
        }

        DesiredPixelSizeTextCtrl->ChangeValueFloat(default_desired_pixel_size);

        if ( default_correct_mag_distortion == true ) {
            CorrectMagDistortionCheckBox->SetValue(true);
        }
        else {
            CorrectMagDistortionCheckBox->SetValue(false);
        }

        MoviesHaveInvertedContrast->SetValue(default_protein_is_white);

        GainFilePicker->Enable(ApplyGainImageCheckbox->IsChecked( ));
        DarkFilePicker->Enable(ApplyDarkImageCheckbox->IsChecked( ));

        DesiredPixelSizeStaticText->Enable(ResampleMoviesCheckBox->IsChecked( ));
        DesiredPixelSizeTextCtrl->Enable(ResampleMoviesCheckBox->IsChecked( ));
        DistortionAngleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
        MajorScaleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
        MinorScaleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
        DistortionAngleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
        MajorScaleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
        MinorScaleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked( ));

        DistortionAngleTextCtrl->ChangeValueFloat(default_mag_distortion_angle);
        MajorScaleTextCtrl->ChangeValueFloat(default_mag_distortion_major_scale);
        MinorScaleTextCtrl->ChangeValueFloat(default_mag_distortion_minor_scale);
    }
}

void MyMovieImportDialog::AddFilesClick(wxCommandEvent& event) {
    wxFileDialog openFileDialog(this, _("Select movie files - basic wildcards are allowed"), "", "", "MRC, TIFF, or EER files (*.mrc;*.mrcs;*.tif;*.tiff;*.eer)|*.mrc;*.mrcs;*.tif;*.tiff;*.eer;*.MRC;*.MRCS;*.TIF;*.TIFF;*.EER", wxFD_OPEN | wxFD_MULTIPLE);

    bool     at_least_one_eer_file = false;
    wxString current_extension;

    if ( openFileDialog.ShowModal( ) == wxID_OK ) {
        wxArrayString selected_paths;
        wxArrayString final_paths;
        openFileDialog.GetPaths(selected_paths);

        PathListCtrl->Freeze( );

        for ( unsigned long counter = 0; counter < selected_paths.GetCount( ); counter++ ) {
            // is this an actual filename, that exists - in which case add it.

            if ( DoesFileExist(selected_paths.Item(counter)) == true )
                final_paths.Add(selected_paths.Item(counter));
            else {
                // perhaps it is a wildcard..
                int           wildcard_counter;
                wxArrayString wildcard_files;
                wxString      directory_string;
                wxString      file_string;

                SplitFileIntoDirectoryAndFile(selected_paths.Item(counter), directory_string, file_string);
                wxDir::GetAllFiles(directory_string, &wildcard_files, file_string, wxDIR_FILES);

                for ( int wildcard_counter = 0; wildcard_counter < wildcard_files.GetCount( ); wildcard_counter++ ) {
                    current_extension = wxFileName(wildcard_files.Item(wildcard_counter)).GetExt( );
                    current_extension = current_extension.MakeLower( );

                    if ( current_extension == "mrc" || current_extension == "mrcs" || current_extension == "tif" || current_extension == "eer" )
                        final_paths.Add(wildcard_files.Item(wildcard_counter));
                }
            }
        }

        final_paths.Sort( );

        for ( int file_counter = 0; file_counter < final_paths.GetCount( ); file_counter++ ) {
            PathListCtrl->InsertItem(PathListCtrl->GetItemCount( ), final_paths.Item(file_counter), PathListCtrl->GetItemCount( ));
            current_extension = wxFileName(final_paths.Item(file_counter)).GetExt( );
            current_extension = current_extension.MakeLower( );
            if ( current_extension == "eer" )
                at_least_one_eer_file = true;
        }

        PathListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);
        PathListCtrl->Thaw( );

        GainFilePicker->SetInitialDirectory(wxFileName(final_paths.Last( )).GetPath( ));
        DarkFilePicker->SetInitialDirectory(wxFileName(final_paths.Last( )).GetPath( ));

        CheckImportButtonStatus( );

        if ( at_least_one_eer_file ) {
            PixelSizeStaticText->SetLabel(wxT("Pixel Size (post EER sampling) (Å) :"));
            ExposurePerFrameStaticText->SetLabel(wxT("Exp. per frame (post EER avg.) (e¯/Å²) :"));
            EerNumberOfFramesStaticText->Enable( );
            EerNumberOfFramesSpinCtrl->Enable( );
            EerSuperResFactorStaticText->Enable( );
            EerSuperResFactorChoice->Enable( );
        }
        else {
            PixelSizeStaticText->SetLabel(wxT("Pixel Size (Å) :"));
            ExposurePerFrameStaticText->SetLabel(wxT("Exposure per frame (e¯/Å²) :"));
            EerNumberOfFramesStaticText->Disable( );
            EerNumberOfFramesSpinCtrl->Disable( );
            EerSuperResFactorStaticText->Disable( );
            EerSuperResFactorChoice->Disable( );
        }
    }
}

void MyMovieImportDialog::ClearClick(wxCommandEvent& event) {
    PathListCtrl->DeleteAllItems( );
    CheckImportButtonStatus( );
    PixelSizeStaticText->SetLabel(wxT("Pixel Size (Å) :"));
    ExposurePerFrameStaticText->SetLabel(wxT("Exposure per frame (e¯/Å²) :"));
    EerNumberOfFramesStaticText->Disable( );
    EerNumberOfFramesSpinCtrl->Disable( );
    EerSuperResFactorStaticText->Disable( );
    EerSuperResFactorChoice->Disable( );
}

void MyMovieImportDialog::CancelClick(wxCommandEvent& event) {
    Destroy( );
}

void MyMovieImportDialog::AddDirectoryClick(wxCommandEvent& event) {
    wxDirDialog dlg(NULL, "Choose import directory", "", wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

    wxArrayString all_files;

    bool     at_least_one_eer_file = false;
    wxString current_extension;

    if ( dlg.ShowModal( ) == wxID_OK ) {
        wxDir::GetAllFiles(dlg.GetPath( ), &all_files, "*.mrc", wxDIR_FILES);
        wxDir::GetAllFiles(dlg.GetPath( ), &all_files, "*.mrcs", wxDIR_FILES);
        wxDir::GetAllFiles(dlg.GetPath( ), &all_files, "*.tif", wxDIR_FILES);

        all_files.Sort( );

        PathListCtrl->Freeze( );

        for ( unsigned long counter = 0; counter < all_files.GetCount( ); counter++ ) {
            PathListCtrl->InsertItem(PathListCtrl->GetItemCount( ), all_files.Item(counter), PathListCtrl->GetItemCount( ));
        }

        PathListCtrl->SetColumnWidth(0, wxLIST_AUTOSIZE);
        PathListCtrl->Thaw( );

        GainFilePicker->SetInitialDirectory(dlg.GetPath( ));
        DarkFilePicker->SetInitialDirectory(dlg.GetPath( ));

        CheckImportButtonStatus( );

        if ( at_least_one_eer_file ) {
            PixelSizeStaticText->SetLabel(wxT("Pixel Size (post EER sampling) (Å) :"));
            ExposurePerFrameStaticText->SetLabel(wxT("Exp. per frame (post EER avg.) (e¯/Å²) :"));
            EerNumberOfFramesStaticText->Enable( );
            EerNumberOfFramesSpinCtrl->Enable( );
            EerSuperResFactorStaticText->Enable( );
            EerSuperResFactorChoice->Enable( );
        }
        else {
            PixelSizeStaticText->SetLabel(wxT("Pixel Size (Å) :"));
            ExposurePerFrameStaticText->SetLabel(wxT("Exposure per frame (e¯/Å²) :"));
            EerNumberOfFramesStaticText->Disable( );
            EerNumberOfFramesSpinCtrl->Disable( );
            EerSuperResFactorStaticText->Disable( );
            EerSuperResFactorChoice->Disable( );
        }
    }
}

void MyMovieImportDialog::OnTextKeyPress(wxKeyEvent& event) {

    int  keycode      = event.GetKeyCode( );
    bool is_valid_key = false;

    if ( keycode > 31 && keycode < 127 ) {
        if ( keycode > 47 && keycode < 58 )
            is_valid_key = true;
        else if ( keycode > 44 && keycode < 47 )
            is_valid_key = true;
    }
    else
        is_valid_key = true;

    if ( is_valid_key == true )
        event.Skip( );
}

void MyMovieImportDialog::TextChanged(wxCommandEvent& event) {
    CheckImportButtonStatus( );
}

void MyMovieImportDialog::CheckImportButtonStatus( ) {
    bool   enable_import_box = true;
    double temp_double;
    double current_pixel_size;

    if ( PathListCtrl->GetItemCount( ) < 1 )
        enable_import_box = false;

    if ( VoltageCombo->IsTextEmpty( ) == true || PixelSizeText->GetLineLength(0) == 0 || CsText->GetLineLength(0) == 0 || DoseText->GetLineLength(0) == 0 )
        enable_import_box = false;

    if ( (ApplyGainImageCheckbox->IsChecked( )) && (! GainFilePicker->GetFileName( ).Exists( )) )
        enable_import_box = false;
    if ( (ApplyDarkImageCheckbox->IsChecked( )) && (! DarkFilePicker->GetFileName( ).Exists( )) )
        enable_import_box = false;

    //if ((! MoviesAreGainCorrectedCheckBox->IsChecked())) wxPrintf("Movies are not gain corrected\n");
    //if (GainFilePicker->GetFileName().Exists()) wxPrintf("Gain file exists\n");

    if ( ResampleMoviesCheckBox->IsChecked( ) == true ) {
        if ( PixelSizeText->GetLineText(0).ToDouble(&current_pixel_size) == false )
            enable_import_box = false;
        else if ( current_pixel_size >= DesiredPixelSizeTextCtrl->ReturnValue( ) )
            enable_import_box = false;
    }

    if ( CorrectMagDistortionCheckBox->IsChecked( ) == true ) {
        if ( MajorScaleTextCtrl->ReturnValue( ) < MinorScaleTextCtrl->ReturnValue( ) )
            enable_import_box = false;
    }

    ImportButton->Enable(enable_import_box);

    Update( );
    Refresh( );
}

void MyMovieImportDialog::OnGainFilePickerChanged(wxFileDirPickerEvent& event) {
    CheckImportButtonStatus( );
}

void MyMovieImportDialog::OnSkipFullIntegrityCheckCheckBox(wxCommandEvent& event) {
    CheckImportButtonStatus( );
}

void MyMovieImportDialog::OnMoviesAreGainCorrectedCheckBox(wxCommandEvent& event) {
    GainFilePicker->Enable(ApplyGainImageCheckbox->IsChecked( ));
    DarkFilePicker->Enable(ApplyDarkImageCheckbox->IsChecked( ));
    CheckImportButtonStatus( );
}

void MyMovieImportDialog::OnResampleMoviesCheckBox(wxCommandEvent& event) {
    DesiredPixelSizeStaticText->Enable(ResampleMoviesCheckBox->IsChecked( ));
    DesiredPixelSizeTextCtrl->Enable(ResampleMoviesCheckBox->IsChecked( ));
    //DesiredPixelSizeTextCtrl->ChangeValue("1.00");
    CheckImportButtonStatus( );
}

void MyMovieImportDialog::OnCorrectMagDistortionCheckBox(wxCommandEvent& event) {
    DistortionAngleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
    MajorScaleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
    MinorScaleTextCtrl->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
    DistortionAngleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
    MajorScaleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked( ));
    MinorScaleStaticText->Enable(CorrectMagDistortionCheckBox->IsChecked( ));

    CheckImportButtonStatus( );
}

void MyMovieImportDialog::ImportClick(wxCommandEvent& event) {
    // Get the microscope values

    double microscope_voltage;
    double pixel_size;
    double spherical_aberration;
    double dose_per_frame;

    bool have_errors = false;

    int gain_ref_x_size = -1;
    int gain_ref_y_size = -1;

    int dark_ref_x_size = -1;
    int dark_ref_y_size = -1;

    int movies_are_gain_corrected;
    int movies_are_dark_corrected;

    int resample_movies;
    int correct_mag_distortion;

    wxString gain_ref_filename;
    wxString dark_ref_filename;

    /*
	 * To save time, set this to true.
	 * When true, we will not check the number of frames and every frame for every movie. Most of the time,
	 * we will just assume we already know the correct number of frames, and that all frames are correct,
	 * and that the file is not corrupt. This may not be a safe assumption, so you should only
	 * set this to "true" if you are confident that all programs that will use the movies
	 * as input will handle bad/weird input files gracefully.
	 */
    const bool skip_full_check_of_tiff_movies = SkipFullIntegrityCheck->GetValue( );

    VoltageCombo->GetValue( ).ToDouble(&microscope_voltage);
    //VoltageCombo->GetStringSelection().ToDouble(&microscope_voltage);
    PixelSizeText->GetLineText(0).ToDouble(&pixel_size);
    CsText->GetLineText(0).ToDouble(&spherical_aberration);
    DoseText->GetLineText(0).ToDouble(&dose_per_frame);

    //  In case we need it make an error dialog..

    MyErrorDialog* my_error = new MyErrorDialog(this);

    if ( PathListCtrl->GetItemCount( ) > 0 ) {
        MovieAsset temp_asset;

        temp_asset.microscope_voltage   = microscope_voltage;
        temp_asset.pixel_size           = pixel_size;
        temp_asset.dose_per_frame       = dose_per_frame;
        temp_asset.spherical_aberration = spherical_aberration;
        temp_asset.protein_is_white     = MoviesHaveInvertedContrast->IsChecked( );
        long temp_long;
        EerSuperResFactorChoice->GetStringSelection( ).ToLong(&temp_long);
        temp_asset.eer_super_res_factor = temp_long;
        temp_asset.eer_frames_per_image = EerNumberOfFramesSpinCtrl->GetValue( );

        movies_are_gain_corrected = ! ApplyGainImageCheckbox->IsChecked( );
        movies_are_dark_corrected = ! ApplyDarkImageCheckbox->IsChecked( );

        if ( ApplyGainImageCheckbox->IsChecked( ) == false ) {
            temp_asset.gain_filename = "";
            gain_ref_filename        = "";
        }
        else {
            temp_asset.gain_filename = GainFilePicker->GetFileName( ).GetFullPath( );
            gain_ref_filename        = GainFilePicker->GetFileName( ).GetFullPath( );

            ImageFile gain_ref;
            if ( gain_ref.OpenFile(temp_asset.gain_filename.ToStdString( ), false) == false ) {
                my_error->ErrorText->AppendText(wxString::Format(wxT("cannot open gain reference file (%s)\n"), temp_asset.gain_filename));
                have_errors = true;
            }
            else {
                gain_ref_x_size = gain_ref.ReturnXSize( );
                gain_ref_y_size = gain_ref.ReturnYSize( );
            }
        }

        if ( ApplyDarkImageCheckbox->IsChecked( ) == false ) {
            temp_asset.dark_filename = "";
            dark_ref_filename        = "";
        }
        else {
            temp_asset.dark_filename = DarkFilePicker->GetFileName( ).GetFullPath( );
            dark_ref_filename        = DarkFilePicker->GetFileName( ).GetFullPath( );

            ImageFile dark_ref;
            if ( dark_ref.OpenFile(temp_asset.dark_filename.ToStdString( ), false) == false ) {
                my_error->ErrorText->AppendText(wxString::Format(wxT("cannot open dark reference file (%s)\n"), temp_asset.dark_filename));
                have_errors = true;
            }
            else {
                dark_ref_x_size = dark_ref.ReturnXSize( );
                dark_ref_y_size = dark_ref.ReturnYSize( );
            }
        }

        resample_movies = ResampleMoviesCheckBox->IsChecked( );
        if ( ResampleMoviesCheckBox->IsChecked( ) == false ) {
            temp_asset.output_binning_factor = 1;
        }
        else {
            float desired_pixel_size = DesiredPixelSizeTextCtrl->ReturnValue( );

            // if we are correcting for a mag distortion - calculate the binning factor off the corrected pixel size

            if ( CorrectMagDistortionCheckBox->IsChecked( ) == false ) {
                temp_asset.output_binning_factor = desired_pixel_size / temp_asset.pixel_size;
            }
            else {
                temp_asset.output_binning_factor = desired_pixel_size / ReturnMagDistortionCorrectedPixelSize(temp_asset.pixel_size, MajorScaleTextCtrl->ReturnValue( ), MinorScaleTextCtrl->ReturnValue( ));
            }
        }

        correct_mag_distortion = CorrectMagDistortionCheckBox->IsChecked( );
        if ( CorrectMagDistortionCheckBox->IsChecked( ) == false ) {
            temp_asset.correct_mag_distortion     = false;
            temp_asset.mag_distortion_angle       = 0.0;
            temp_asset.mag_distortion_major_scale = 1.0;
            temp_asset.mag_distortion_minor_scale = 1.0;
        }
        else {
            temp_asset.correct_mag_distortion     = true;
            temp_asset.mag_distortion_angle       = DistortionAngleTextCtrl->ReturnValue( );
            temp_asset.mag_distortion_major_scale = MajorScaleTextCtrl->ReturnValue( );
            temp_asset.mag_distortion_minor_scale = MinorScaleTextCtrl->ReturnValue( );
        }

        // ProgressBar..

        OneSecondProgressDialog* my_progress_dialog = new OneSecondProgressDialog("Import Movie", "Importing Movies...", PathListCtrl->GetItemCount( ), this, wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_REMAINING_TIME);

        // loop through all the files and add them as assets..

        main_frame->current_project.database.Begin( );
        // for adding to the database..
        main_frame->current_project.database.BeginMovieAssetInsert( );

        // list of added movies for subsequent metadata insertion
        //
        wxArrayInt    list_asset_id;
        wxArrayString list_filenames;

        for ( long counter = 0; counter < PathListCtrl->GetItemCount( ); counter++ ) {

            wxDateTime start = wxDateTime::UNow( );
            temp_asset.Update(PathListCtrl->GetItemText(counter), skip_full_check_of_tiff_movies);
            wxDateTime end = wxDateTime::UNow( );
            //wxPrintf("File parsing (disk access) took %li milliseconds\n", (end-start).GetMilliseconds());
            start = wxDateTime::UNow( );

            // check everything ok with gain_ref

            if ( ApplyGainImageCheckbox->IsChecked( ) == true && (temp_asset.x_size != gain_ref_x_size || temp_asset.y_size != gain_ref_y_size) ) {
                my_error->ErrorText->AppendText(wxString::Format(wxT("%s does not have the same size as the provided gain reference, skipping\n"), temp_asset.ReturnFullPathString( )));
                have_errors = true;
            }
            else if ( ApplyDarkImageCheckbox->IsChecked( ) == true && (temp_asset.x_size != dark_ref_x_size || temp_asset.y_size != dark_ref_y_size) ) {
                my_error->ErrorText->AppendText(wxString::Format(wxT("%s does not have the same size as the provided dark reference, skipping\n"), temp_asset.ReturnFullPathString( )));
                have_errors = true;
            }
            else if ( movie_asset_panel->IsFileAnAsset(temp_asset.filename) == false ) // Check this movie is not already an asset..
            {
                end = wxDateTime::UNow( );

                //wxPrintf("File searching took %li milliseconds\n", (end-start).GetMilliseconds());

                if ( temp_asset.is_valid == true ) {
                    if ( temp_asset.number_of_frames < 3 && ! skip_full_check_of_tiff_movies ) //if we didnt' check the full files, the count of frames is not accurate anyway (it's probably 0)
                    {
                        my_error->ErrorText->AppendText(wxString::Format(wxT("%s contains less than 3 frames, skipping\n"), temp_asset.ReturnFullPathString( )));
                        have_errors = true;
                    }
                    else {

                        temp_asset.asset_id   = movie_asset_panel->current_asset_number;
                        temp_asset.asset_name = temp_asset.filename.GetName( );
                        temp_asset.total_dose = double(temp_asset.number_of_frames) * dose_per_frame;
                        movie_asset_panel->AddAsset(&temp_asset);

                        main_frame->current_project.database.AddNextMovieAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath( ), 1, temp_asset.x_size, temp_asset.y_size, temp_asset.number_of_frames, temp_asset.microscope_voltage, temp_asset.pixel_size, temp_asset.dose_per_frame, temp_asset.spherical_aberration, temp_asset.gain_filename, temp_asset.dark_filename, temp_asset.output_binning_factor, temp_asset.correct_mag_distortion, temp_asset.mag_distortion_angle, temp_asset.mag_distortion_major_scale, temp_asset.mag_distortion_minor_scale, temp_asset.protein_is_white, temp_asset.eer_super_res_factor, temp_asset.eer_frames_per_image);
                    }
                    // Since we cannot bulk insert into two tables, save info in these lists to loop through them later
                    if ( ImportMetadataCheckbox->IsChecked( ) == true ) {
                        list_asset_id.Add(temp_asset.asset_id);
                        list_filenames.Add(temp_asset.filename.GetFullPath( ));
                    }
                }
                else {
                    my_error->ErrorText->AppendText(wxString::Format(wxT("%s is not a valid image file, skipping\n"), temp_asset.ReturnFullPathString( )));
                    have_errors = true;
                }
            }
            else {
                my_error->ErrorText->AppendText(wxString::Format(wxT("%s is already an asset, skipping\n"), temp_asset.ReturnFullPathString( )));
                have_errors = true;
            }

            my_progress_dialog->Update(counter);
        }

        // do database write..

        main_frame->current_project.database.EndMovieAssetInsert( );

        // Do Metadata import
        if ( ImportMetadataCheckbox->IsChecked( ) == true ) {
            wxTextFile   temp_file;
            wxString     temp_string;
            wxJSONValue  root;
            wxString     temp_json_string;
            wxJSONWriter json_writer(wxJSONWRITER_NONE);
            main_frame->current_project.database.BeginMovieAssetMetadataInsert( );
            double             temp_double;
            MovieMetadataAsset temp_metadata_asset;
            for ( long counter = 0; counter < list_asset_id.GetCount( ); counter++ ) {
                root.Clear( );
                if ( wxFileExists(list_filenames[counter] + ".mdoc") ) {
                    temp_file.Open(list_filenames[counter] + ".mdoc");

                    while ( ! temp_file.Eof( ) ) {
                        temp_string = temp_file.GetNextLine( );
                        if ( ! temp_string.StartsWith("[") & temp_string.Contains(" = ") ) {
                            root[temp_string.BeforeFirst('=').Trim( ).Trim(false)] = temp_string.AfterFirst('=').Trim( ).Trim(false);
                        }
                    }
                }
                json_writer.Write(root, temp_json_string);
                temp_metadata_asset                 = MovieMetadataAsset( );
                temp_metadata_asset.movie_asset_id  = list_asset_id[counter];
                temp_metadata_asset.metadata_source = "serialem_frames_mdoc";
                temp_metadata_asset.content_json    = temp_json_string;
                if ( root.HasMember("TiltAngle") & root["TiltAngle"].AsString( ).ToDouble(&temp_double) ) {
                    temp_metadata_asset.tilt_angle = temp_double;
                }
                if ( root.HasMember("StagePosition") ) {
                    if ( root["StagePosition"].AsString( ).BeforeFirst(' ').ToDouble(&temp_double) ) {
                        temp_metadata_asset.stage_position_x = temp_double;
                    }
                    if ( root["StagePosition"].AsString( ).AfterFirst(' ').ToDouble(&temp_double) ) {
                        temp_metadata_asset.stage_position_y = temp_double;
                    }
                }
                if ( root.HasMember("StageZ") & root["StageZ"].AsString( ).ToDouble(&temp_double) ) {
                    temp_metadata_asset.stage_position_z = temp_double;
                }
                if ( root.HasMember("ImageShift") ) {
                    if ( root["ImageShift"].AsString( ).BeforeFirst(' ').ToDouble(&temp_double) ) {
                        temp_metadata_asset.image_shift_x = temp_double;
                    }
                    if ( root["ImageShift"].AsString( ).AfterFirst(' ').ToDouble(&temp_double) ) {
                        temp_metadata_asset.image_shift_y = temp_double;
                    }
                }
                if ( root.HasMember("ExposureDose") & root["ExposureDose"].AsString( ).ToDouble(&temp_double) ) {
                    temp_metadata_asset.exposure_dose = temp_double;
                }
                if ( root.HasMember("DateTime") ) {
                    temp_metadata_asset.acquisition_time.ParseFormat(root["DateTime"].AsString( ), "%d-%b-%y %H:%M:%S");
                }
                main_frame->current_project.database.AddNextMovieAssetMetadata(temp_metadata_asset);
            }
            main_frame->current_project.database.EndMovieAssetMetadataInsert( );
        }

        my_progress_dialog->Destroy( );

        // write these values as future defaults..

        main_frame->current_project.database.DeleteTable("MOVIE_IMPORT_DEFAULTS");
        main_frame->current_project.database.CreateMovieImportDefaultsTable( );
        main_frame->current_project.database.InsertOrReplace("MOVIE_IMPORT_DEFAULTS", "prrrrititirirrriii", "NUMBER", "VOLTAGE", "SPHERICAL_ABERRATION", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "MOVIES_ARE_GAIN_CORRECTED", "GAIN_REFERENCE_FILENAME", "MOVIES_ARE_DARK_CORRECTED", "DARK_REFERENCE_FILENAME", "RESAMPLE_MOVIES", "DESIRED_PIXEL_SIZE", "CORRECT_MAG_DISTORTION", "MAG_DISTORTION_ANGLE", "MAG_DISTORTION_MAJOR_SCALE", "MAG_DISTORTION_MINOR_SCALE", "PROTEIN_IS_WHITE", "EER_SUPER_RES_FACTOR", "EER_FRAMES_PER_IMAGE", 1, microscope_voltage, spherical_aberration, pixel_size, dose_per_frame, movies_are_gain_corrected, gain_ref_filename.ToUTF8( ).data( ), movies_are_dark_corrected, dark_ref_filename.ToUTF8( ).data( ), resample_movies, double(DesiredPixelSizeTextCtrl->ReturnValue( )), correct_mag_distortion, double(DistortionAngleTextCtrl->ReturnValue( )), double(MajorScaleTextCtrl->ReturnValue( )), double(MinorScaleTextCtrl->ReturnValue( )), int(MoviesHaveInvertedContrast->IsChecked( )), temp_asset.eer_super_res_factor, temp_asset.eer_frames_per_image);
        main_frame->current_project.database.Commit( );

        main_frame->DirtyMovieGroups( );
        //		movie_asset_panel->SetSelectedGroup(0);
        //	movie_asset_panel->FillGroupList();
        //movie_asset_panel->FillContentsList();
        //main_frame->RecalculateAssetBrowser();
    }

    if ( have_errors == true ) {
        Hide( );
        my_error->ShowModal( );
    }

    my_error->Destroy( );
    EndModal(0);
    Destroy( );
}
