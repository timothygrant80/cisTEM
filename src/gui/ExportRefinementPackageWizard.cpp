//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyImageAssetPanel*             image_asset_panel;
extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;

ExportRefinementPackageWizard::ExportRefinementPackageWizard(wxWindow* parent)
    : ExportRefinementPackageWizardParent(parent) {
    SetPageSize(wxSize(600, 400));

    current_package = &refinement_package_asset_panel->all_refinement_packages.Item(refinement_package_asset_panel->selected_refinement_package);

    ParameterSelectPanel->FillComboBox(refinement_package_asset_panel->selected_refinement_package, true);
    //ParameterSelectPanel->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &ExportRefinementPackageWizard::OnParamsComboBox, this);

    ClassComboBox->Freeze( );

    for ( int class_counter = 0; class_counter < current_package->number_of_classes; class_counter++ ) {
        ClassComboBox->Append(wxString::Format("Class #%i", class_counter + 1));
    }

    ClassComboBox->SetSelection(0);
}

ExportRefinementPackageWizard::~ExportRefinementPackageWizard( ) {
    //ParameterSelectPanel->AssetComboBox->Unbind(wxEVT_COMMAND_COMBOBOX_SELECTED, &ExportRefinementPackageWizard::OnParamsComboBox, this);
}

void ExportRefinementPackageWizard::OnParamsComboBox(wxCommandEvent& event) {
}

void ExportRefinementPackageWizard::CheckPaths( ) {
    if ( GetCurrentPage( ) == m_pages.Item(2) ) {
        Freeze( );

        EnableNextButton( );

        // if the stack file ends in .mrc and we have relion, change to mrcs and vice versa for frealign
        wxFileName current_stack_filename = ParticleStackFileTextCtrl->GetLineText(0);
        wxFileName current_meta_filename  = MetaDataFileTextCtrl->GetLineText(0);

        if ( ParticleStackFileTextCtrl->GetLineText(0).IsEmpty( ) )
            DisableNextButton( );
        else {
            if ( FrealignRadioButton->GetValue( ) ) {
                if ( current_stack_filename.GetExt( ) != "mrc" ) {
                    current_stack_filename.SetExt("mrc");
                    ParticleStackFileTextCtrl->SetValue(current_stack_filename.GetFullPath( ));
                }
            }
            else if ( RelionRadioButton->GetValue( ) ) {
                if ( current_stack_filename.GetExt( ) != "mrcs" ) {
                    current_stack_filename.SetExt("mrcs");
                    ParticleStackFileTextCtrl->SetValue(current_stack_filename.GetFullPath( ));
                }
            }
            else if ( Relion3RadioButton->GetValue( ) ) {
                if ( current_stack_filename.GetExt( ) != "mrcs" ) {
                    current_stack_filename.SetExt("mrcs");
                    ParticleStackFileTextCtrl->SetValue(current_stack_filename.GetFullPath( ));
                }
            }
        }

        if ( MetaDataFileTextCtrl->GetLineText(0).IsEmpty( ) )
            DisableNextButton( );
        else {
            if ( FrealignRadioButton->GetValue( ) ) {
                if ( current_meta_filename.GetExt( ) != "par" ) {
                    current_meta_filename.SetExt("par");
                    MetaDataFileTextCtrl->SetValue(current_meta_filename.GetFullPath( ));
                }
            }
            else if ( RelionRadioButton->GetValue( ) ) {
                if ( current_meta_filename.GetExt( ) != "star" ) {
                    current_meta_filename.SetExt("star");
                    MetaDataFileTextCtrl->SetValue(current_meta_filename.GetFullPath( ));
                }
            }
            else if ( Relion3RadioButton->GetValue( ) ) {
                if ( current_meta_filename.GetExt( ) != "star" ) {
                    current_meta_filename.SetExt("star");
                    MetaDataFileTextCtrl->SetValue(current_meta_filename.GetFullPath( ));
                }
            }
        }

        Thaw( );
    }
}

void ExportRefinementPackageWizard::OnStackBrowseButtonClick(wxCommandEvent& event) {
    ProperOverwriteCheckSaveDialog* saveFileDialog;

    if ( FrealignRadioButton->GetValue( ) ) {
        saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save MRC stack file"), "MRC files (*.mrc)|*.mrc", ".mrc");
    }
    else if ( RelionRadioButton->GetValue( ) ) {
        saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save MRC stack file"), "MRC files (*.mrcs)|*.mrcs", ".mrcs");
    }
    else if ( Relion3RadioButton->GetValue( ) ) {
        saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save MRC stack file"), "MRC files (*.mrcs)|*.mrcs", ".mrcs");
    }

    if ( saveFileDialog->ShowModal( ) == wxID_OK ) {
        ParticleStackFileTextCtrl->SetValue(saveFileDialog->ReturnProperPath( ));
    }

    //	CheckPaths();
    saveFileDialog->Destroy( );
}

void ExportRefinementPackageWizard::OnMetaBrowseButtonClick(wxCommandEvent& event) {
    ProperOverwriteCheckSaveDialog* saveFileDialog;

    if ( FrealignRadioButton->GetValue( ) ) {
        saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save PAR file"), "PAR files (*.par)|*.par", ".par");
    }
    else if ( RelionRadioButton->GetValue( ) ) {
        saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save STAR file"), "STAR files (*.star)|*.star", ".star");
    }
    else if ( Relion3RadioButton->GetValue( ) ) {
        saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save STAR file"), "STAR files (*.star)|*.star", ".star");
    }

    if ( saveFileDialog->ShowModal( ) == wxID_OK ) {
        MetaDataFileTextCtrl->SetValue(saveFileDialog->ReturnProperPath( ));
    }

    CheckPaths( );
    saveFileDialog->Destroy( );
}

void ExportRefinementPackageWizard::OnPageChanged(wxWizardEvent& event) {
    if ( event.GetPage( ) == m_pages.Item(0) ) {
        EnableNextButton( );
    }
    else if ( event.GetPage( ) == m_pages.Item(2) ) {
        if ( FrealignRadioButton->GetValue( ) ) {
            MetaFilenameStaticText->SetLabel("Output PAR Filename :-    ");
        }
        else if ( RelionRadioButton->GetValue( ) ) {
            MetaFilenameStaticText->SetLabel("Output STAR Filename :-   ");
        }
        else if ( Relion3RadioButton->GetValue( ) ) {
            MetaFilenameStaticText->SetLabel("Output STAR Filename :-   ");
        }

        CheckPaths( );
    }
}

void ExportRefinementPackageWizard::OnPathChange(wxCommandEvent& event) {
    Freeze( );
    EnableNextButton( );
    if ( ParticleStackFileTextCtrl->GetLineText(0).IsEmpty( ) )
        DisableNextButton( );
    if ( MetaDataFileTextCtrl->GetLineText(0).IsEmpty( ) )
        DisableNextButton( );
    Thaw( );
}

void ExportRefinementPackageWizard::OnUpdateUI(wxUpdateUIEvent& event) {
}

void ExportRefinementPackageWizard::OnFinished(wxWizardEvent& event) {
    // get the current refinement package..

    long particle_counter;

    // Are we doing frealign, or relion?

    if ( FrealignRadioButton->GetValue( ) ) // Frealign
    {
        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Export To Frealign", "Writing PAR file...", 2, this, wxPD_AUTO_HIDE | wxPD_APP_MODAL);

        // get the refinement..

        Refinement* current_refinement = main_frame->current_project.database.GetRefinementByID(current_package->refinement_ids[ParameterSelectPanel->GetSelection( )]);

        // write the parameter file..

        current_refinement->WriteSingleClassFrealignParameterFile(MetaDataFileTextCtrl->GetLineText(0), ClassComboBox->GetSelection( ));

        my_dialog->Update(1, "Copying Stack...");
        // copy the stack..

        wxCopyFile(current_package->stack_filename, ParticleStackFileTextCtrl->GetLineText(0));

        my_dialog->Destroy( );
        delete current_refinement;
    }
    else
        // Relion
        if ( RelionRadioButton->GetValue( ) || Relion3RadioButton->GetValue( ) ) {
            wxFileName output_stack_filename                 = ParticleStackFileTextCtrl->GetLineText(0);
            wxFileName relion_star_filename                  = MetaDataFileTextCtrl->GetLineText(0); // one line per particle
            wxFileName relion_corrected_micrographs_filename = MetaDataFileTextCtrl->GetLineText(0); // one line per motion-corrected movie
            wxFileName relion_motioncor_star_base_filename   = MetaDataFileTextCtrl->GetLineText(0); // one star file per movie, with one line per frame
            wxFileName relion_motioncor_star_current_filename;

            wxFileName gain_ref_filename;
            bool       should_convert_gain_ref;

            // Check to ensure that a filename extension is there, and if not add "mrcs"
            if ( output_stack_filename.GetExt( ).IsEmpty( ) ) {
                output_stack_filename.SetExt("mrcs");
            }

            relion_star_filename.SetExt("star");
            relion_corrected_micrographs_filename.ClearExt( );
            relion_corrected_micrographs_filename.SetName(relion_corrected_micrographs_filename.GetName( ) + wxT("_corrected_micrographs.star"));
            relion_motioncor_star_base_filename.ClearExt( );
            relion_motioncor_star_base_filename.SetName(relion_motioncor_star_base_filename.GetName( ) + wxT("_motioncorr"));

            wxTextFile*                   relion_star_file = new wxTextFile(relion_star_filename.GetFullPath( ));
            RefinementPackageParticleInfo first_particle   = current_package->ReturnParticleInfoByPositionInStack(1);
            wxTextFile*                   relion_corrected_micrographs_file;

            // Let's check that we have usable images and movies for writing out additional Relion info;
            // when we don't, we won't bother with the corrected micrographs file since we don't have that info anyway.
            // Currently only using the first particle to get information for the optics groups;
            // will cause issues if datasets are merged together with different microscopy parameters
            bool is_valid_parent_image = first_particle.parent_image_id > 0;
            bool is_valid_parent_movie;
            if ( is_valid_parent_image )
                is_valid_parent_movie = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(first_particle.parent_image_id))->parent_id > 0;
            else
                is_valid_parent_movie = false;
            bool have_valid_images_and_movies = is_valid_parent_image && is_valid_parent_movie;

            if ( have_valid_images_and_movies )
                relion_corrected_micrographs_file = new wxTextFile(relion_corrected_micrographs_filename.GetFullPath( ));
            wxTextFile* relion_motioncor_star_current_file;

            wxArrayInt array_of_unique_parent_image_asset_ids;
            bool       current_particle_is_first_from_this_image;

            MRCFile input_stack(current_package->stack_filename.ToStdString( ), false);
            MRCFile output_stack(output_stack_filename.GetFullPath( ).ToStdString( ), true);
            output_stack.SetPixelSize(current_package->contained_particles[0].pixel_size);
            Image particle_image;

            double particle_radius = current_package->estimated_particle_size_in_angstroms / 2;

            wxString micrograph_filename;
            float    original_movie_pixel_size;

            int random_subset;

            RefinementPackageParticleInfo current_particle;
            RefinementResult              current_refinement_result;
            ImageAsset*                   current_image_asset;
            MovieAsset*                   current_movie_asset;

            if ( relion_star_file->Exists( ) ) {
                relion_star_file->Open( );
                relion_star_file->Clear( );
            }
            else {
                relion_star_file->Create( );
            }

            if ( have_valid_images_and_movies ) {
                if ( relion_corrected_micrographs_file->Exists( ) ) {
                    relion_corrected_micrographs_file->Open( );
                    relion_corrected_micrographs_file->Clear( );
                }
                else {
                    relion_corrected_micrographs_file->Create( );
                }
            }
            //write optics data block for relion3.1 format
            if ( Relion3RadioButton->GetValue( ) ) {
                relion_star_file->AddLine(wxString(" "));
                relion_star_file->AddLine(wxString("data_optics"));
                relion_star_file->AddLine(wxString(" "));
                relion_star_file->AddLine(wxString("loop_"));
                relion_star_file->AddLine(wxString("_rlnOpticsGroup #1"));
                relion_star_file->AddLine(wxString("_rlnOpticsGroupName #2"));
                relion_star_file->AddLine(wxString("_rlnAmplitudeContrast #3"));
                relion_star_file->AddLine(wxString("_rlnSphericalAberration #4"));
                relion_star_file->AddLine(wxString("_rlnVoltage #5"));
                relion_star_file->AddLine(wxString("_rlnImagePixelSize #6"));
                relion_star_file->AddLine(wxString("_rlnImageSize #7"));
                relion_star_file->AddLine(wxString("_rlnImageDimensionality #8"));

                // We only need to load image/movie assets if the ID is valid; otherwise, they won't be used yet
                if ( is_valid_parent_image ) {
                    current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(first_particle.parent_image_id));

                    // In case we imported images and don't have parent movies, we'll create a default
                    if ( current_image_asset->parent_id != -1 )
                        current_movie_asset = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnArrayPositionFromAssetID(current_image_asset->parent_id));
                    else
                        current_movie_asset = new MovieAsset( );
                }

                relion_star_file->AddLine(wxString::Format("%i %s %f %f %f %f %i %i", 1,
                                                           "opticsGroup1",
                                                           first_particle.amplitude_contrast,
                                                           first_particle.spherical_aberration,
                                                           first_particle.microscope_voltage,
                                                           first_particle.pixel_size,
                                                           current_package->stack_box_size,
                                                           2));

                // The corrected micrographs star file
                if ( have_valid_images_and_movies ) {
                    relion_corrected_micrographs_file->AddLine(wxString(" "));
                    relion_corrected_micrographs_file->AddLine(wxString("data_optics"));
                    relion_corrected_micrographs_file->AddLine(wxString(" "));
                    relion_corrected_micrographs_file->AddLine(wxString("loop_"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnOpticsGroupName #1"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnOpticsGroup #2"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnMicrographOriginalPixelSize #3"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnVoltage #4"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnSphericalAberration #5"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnAmplitudeContrast #6"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnMicrographPixelSize #7"));

                    relion_corrected_micrographs_file->AddLine(wxString::Format("%s %i %f %f %f %f %f", "opticsGroup1", 1,
                                                                                current_movie_asset->pixel_size,
                                                                                first_particle.microscope_voltage,
                                                                                first_particle.spherical_aberration,
                                                                                first_particle.amplitude_contrast,
                                                                                current_image_asset->pixel_size));
                }
            }

            //write particle data block for either relion2 or relion3.1 format
            if ( RelionRadioButton->GetValue( ) ) {
                relion_star_file->AddLine(wxString(" "));
                relion_star_file->AddLine(wxString("data_"));
                relion_star_file->AddLine(wxString(" "));
                relion_star_file->AddLine(wxString("loop_"));
                relion_star_file->AddLine(wxString("_rlnMicrographName #1"));
                relion_star_file->AddLine(wxString("_rlnCoordinateX #2"));
                relion_star_file->AddLine(wxString("_rlnCoordinateY #3"));
                relion_star_file->AddLine(wxString("_rlnImageName #4"));
                relion_star_file->AddLine(wxString("_rlnDefocusU #5"));
                relion_star_file->AddLine(wxString("_rlnDefocusV #6"));
                relion_star_file->AddLine(wxString("_rlnDefocusAngle #7"));
                relion_star_file->AddLine(wxString("_rlnPhaseShift #8"));
                relion_star_file->AddLine(wxString("_rlnVoltage #9"));
                relion_star_file->AddLine(wxString("_rlnSphericalAberration #10"));
                relion_star_file->AddLine(wxString("_rlnAmplitudeContrast #11"));
                relion_star_file->AddLine(wxString("_rlnMagnification #12"));
                relion_star_file->AddLine(wxString("_rlnDetectorPixelSize #13"));
                relion_star_file->AddLine(wxString("_rlnAngleRot #14"));
                relion_star_file->AddLine(wxString("_rlnAngleTilt #15"));
                relion_star_file->AddLine(wxString("_rlnAnglePsi #16"));
                relion_star_file->AddLine(wxString("_rlnOriginX #17"));
                relion_star_file->AddLine(wxString("_rlnOriginY #18"));
            }
            else if ( Relion3RadioButton->GetValue( ) ) {
                relion_star_file->AddLine(wxString(" "));
                relion_star_file->AddLine(wxString("data_particles"));
                relion_star_file->AddLine(wxString(" "));
                relion_star_file->AddLine(wxString("loop_"));
                relion_star_file->AddLine(wxString("_rlnMicrographName #1"));
                relion_star_file->AddLine(wxString("_rlnCoordinateX #2"));
                relion_star_file->AddLine(wxString("_rlnCoordinateY #3"));
                relion_star_file->AddLine(wxString("_rlnImageName #4"));
                relion_star_file->AddLine(wxString("_rlnDefocusU #5"));
                relion_star_file->AddLine(wxString("_rlnDefocusV #6"));
                relion_star_file->AddLine(wxString("_rlnDefocusAngle #7"));
                relion_star_file->AddLine(wxString("_rlnPhaseShift #8"));
                relion_star_file->AddLine(wxString("_rlnVoltage #9"));
                relion_star_file->AddLine(wxString("_rlnSphericalAberration #10"));
                relion_star_file->AddLine(wxString("_rlnAmplitudeContrast #11"));
                relion_star_file->AddLine(wxString("_rlnMagnification #12"));
                relion_star_file->AddLine(wxString("_rlnDetectorPixelSize #13"));
                relion_star_file->AddLine(wxString("_rlnAngleRot #14"));
                relion_star_file->AddLine(wxString("_rlnAngleTilt #15"));
                relion_star_file->AddLine(wxString("_rlnAnglePsi #16"));
                relion_star_file->AddLine(wxString("_rlnOriginXAngst #17"));
                relion_star_file->AddLine(wxString("_rlnOriginYAngst #18"));
                relion_star_file->AddLine(wxString("_rlnOpticsGroup #19"));
                relion_star_file->AddLine(wxString("_rlnRandomSubset #20"));

                if ( have_valid_images_and_movies ) {
                    relion_corrected_micrographs_file->AddLine(wxString(" "));
                    relion_corrected_micrographs_file->AddLine(wxString("data_micrographs"));
                    relion_corrected_micrographs_file->AddLine(wxString(" "));
                    relion_corrected_micrographs_file->AddLine(wxString("loop_"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnMicrographName #1"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnMicrographMetadata #2"));
                    relion_corrected_micrographs_file->AddLine(wxString("_rlnOpticsGroup #3"));
                }
            }

            OneSecondProgressDialog* my_dialog          = new OneSecondProgressDialog("Export To Relion", "Exporting...", current_package->contained_particles.GetCount( ), this);
            Refinement*              current_refinement = main_frame->current_project.database.GetRefinementByID(current_package->refinement_ids[ParameterSelectPanel->GetSelection( )]);

            for ( particle_counter = 0; particle_counter < current_package->contained_particles.GetCount( ); particle_counter++ ) {
                current_particle          = current_package->ReturnParticleInfoByPositionInStack(particle_counter + 1);
                current_refinement_result = current_refinement->ReturnRefinementResultByClassAndPositionInStack(ClassComboBox->GetSelection( ), particle_counter + 1);
                is_valid_parent_image     = current_particle.parent_image_id >= 0;
                if ( is_valid_parent_image )
                    current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(current_particle.parent_image_id));
                else
                    current_image_asset = new ImageAsset( );

                // In case we imported images and don't have parent movies, we'll create a default
                is_valid_parent_movie = current_image_asset->parent_id >= 0;
                if ( is_valid_parent_movie )
                    current_movie_asset = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnArrayPositionFromAssetID(current_image_asset->parent_id));
                else
                    current_movie_asset = new MovieAsset( );

                // Check whether we have already written out the motioncor results for the parent image asset. If not,
                current_particle_is_first_from_this_image = (particle_counter == 0) || (array_of_unique_parent_image_asset_ids.Index(current_particle.parent_image_id, true) == wxNOT_FOUND);
                if ( current_particle_is_first_from_this_image ) {
                    //wxPrintf("Particle %li, from image asset %li. If you see this more than once per image asset, there is a bug.\n",particle_counter,current_particle.parent_image_id);
                    array_of_unique_parent_image_asset_ids.Add(int(current_particle.parent_image_id));
                }

                particle_image.ReadSlice(&input_stack, particle_counter + 1);
                if ( ! current_package->stack_has_white_protein )
                    particle_image.InvertRealValues( );
                particle_image.ZeroFloatAndNormalize(1.0, particle_radius / current_particle.pixel_size, true);
                particle_image.WriteSlice(&output_stack, particle_counter + 1);

                // if we have micrograph info, may as well include it..

                if ( current_particle.parent_image_id >= 0 ) {
                    micrograph_filename = image_asset_panel->ReturnAssetLongFilename(image_asset_panel->ReturnArrayPositionFromAssetID(current_particle.parent_image_id));
                }
                else {
                    micrograph_filename = "unknown.mrc";
                }

                //write particle data for either legacy relion format or relion3.1
                if ( RelionRadioButton->GetValue( ) ) {
                    relion_star_file->AddLine(wxString::Format("%s %f %f %06li@%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f", micrograph_filename,
                                                               current_particle.x_pos / current_particle.pixel_size,
                                                               current_particle.y_pos / current_particle.pixel_size,
                                                               particle_counter + 1,
                                                               output_stack_filename.GetFullPath( ),
                                                               current_refinement_result.defocus1,
                                                               current_refinement_result.defocus2,
                                                               current_refinement_result.defocus_angle,
                                                               current_refinement_result.phase_shift,
                                                               current_particle.microscope_voltage,
                                                               current_particle.spherical_aberration,
                                                               current_particle.amplitude_contrast,
                                                               10000.0f,
                                                               current_particle.pixel_size,
                                                               current_refinement_result.phi,
                                                               current_refinement_result.theta,
                                                               current_refinement_result.psi,
                                                               -current_refinement_result.xshift / current_particle.pixel_size,
                                                               -current_refinement_result.yshift / current_particle.pixel_size));
                }
                else if ( Relion3RadioButton->GetValue( ) ) {

                    //  if for some reason the assigned_subset doesn't have a reasonable value, we will go even/odd
                    if ( current_refinement_result.assigned_subset >= 1 ) {
                        random_subset = current_refinement_result.assigned_subset;
                    }
                    else {
                        if ( random_subset == 2 ) {
                            random_subset = 1;
                        }
                        else {
                            random_subset = 2;
                        }
                    }
                    relion_star_file->AddLine(wxString::Format("%s %f %f %06li@%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %i %i", micrograph_filename,
                                                               current_particle.x_pos / current_particle.pixel_size,
                                                               current_particle.y_pos / current_particle.pixel_size,
                                                               particle_counter + 1,
                                                               output_stack_filename.GetFullName( ),
                                                               current_refinement_result.defocus1,
                                                               current_refinement_result.defocus2,
                                                               current_refinement_result.defocus_angle,
                                                               current_refinement_result.phase_shift,
                                                               current_particle.microscope_voltage,
                                                               current_particle.spherical_aberration,
                                                               current_particle.amplitude_contrast,
                                                               10000.0f,
                                                               current_particle.pixel_size,
                                                               current_refinement_result.phi,
                                                               current_refinement_result.theta,
                                                               current_refinement_result.psi,
                                                               -current_refinement_result.xshift,
                                                               -current_refinement_result.yshift,
                                                               1,
                                                               random_subset));

                    if ( current_particle_is_first_from_this_image ) {
                        // We will need a star file to store results of motion correction
                        relion_motioncor_star_current_filename.SetPath(relion_motioncor_star_base_filename.GetPath( ));
                        relion_motioncor_star_current_filename.SetName(wxString::Format("%s_%06i.star", relion_motioncor_star_base_filename.GetName( ), current_image_asset->asset_id));

                        // Add this micrograph to the star file listing all micrographs
                        if ( have_valid_images_and_movies )
                            relion_corrected_micrographs_file->AddLine(wxString::Format("%s %s %i", micrograph_filename.ToStdString( ), relion_motioncor_star_current_filename.GetFullPath( ), 1));

                        // Let's grab the motion correction job details from the database...
                        // but only if we didn't import images -- if we did, we don't have motion correction info.
                        if ( current_movie_asset->asset_id != -1 ) {
                            bool should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT PRE_EXPOSURE_AMOUNT,FIRST_FRAME_TO_SUM FROM MOVIE_ALIGNMENT_LIST WHERE MOVIE_ASSET_ID=%i AND ALIGNMENT_ID=%i", current_movie_asset->asset_id, current_image_asset->alignment_id));
                            if ( ! should_continue ) {
                                MyPrintWithDetails("Error getting information about movie alignment!");
                                DEBUG_ABORT;
                            }
                            float pre_exposure_amount;
                            int   first_frame_to_sum;
                            main_frame->current_project.database.GetFromBatchSelect("si", &pre_exposure_amount, &first_frame_to_sum);
                            main_frame->current_project.database.EndBatchSelect( );

                            // Write out a star file with the results of the whole-frame motion correction

                            relion_motioncor_star_current_file = new wxTextFile(relion_motioncor_star_current_filename.GetFullPath( ));
                            if ( relion_motioncor_star_current_file->Exists( ) ) {
                                relion_motioncor_star_current_file->Open( );
                                relion_motioncor_star_current_file->Clear( );
                            }
                            else {
                                relion_motioncor_star_current_file->Create( );
                            }

                            // Unfortunately Relion does not support reading .dm4 files. To make users' lives easier,
                            // let's convert the gain ref from dm4 to mrc

                            gain_ref_filename = current_movie_asset->gain_filename;
                            if ( gain_ref_filename.GetExt( ) == "dm4" ) {
                                gain_ref_filename.SetExt("mrc");
                                gain_ref_filename.SetPath(output_stack_filename.GetPath( ));

                                if ( ! wxFileExists(gain_ref_filename.GetFullPath( )) ) {
                                    ImageFile input_gain_ref_file(current_movie_asset->gain_filename.ToStdString( ), false);
                                    MRCFile   output_gain_ref_file(gain_ref_filename.GetFullPath( ).ToStdString( ), true);
                                    Image     gain_image;
                                    gain_image.ReadSlice(&input_gain_ref_file, 1);
                                    gain_image.WriteSlice(&output_gain_ref_file, 1);
                                }
                            }

                            // Start writing the star file
                            relion_motioncor_star_current_file->AddLine(wxString(" "));
                            relion_motioncor_star_current_file->AddLine(wxString("data_general"));
                            relion_motioncor_star_current_file->AddLine(wxString(" "));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %i", "_rlnImageSizeX", current_movie_asset->x_size));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %i", "_rlnImageSizeY", current_movie_asset->y_size));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %i", "_rlnImageSizeZ", current_movie_asset->number_of_frames));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %s", "_rlnMicrographMovieName", current_movie_asset->filename.GetFullPath( )));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %s", "_rlnMicrographGainName", gain_ref_filename.GetFullPath( )));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %f", "_rlnMicrographBinning", current_movie_asset->output_binning_factor));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %f", "_rlnMicrographOriginalPixelSize", current_movie_asset->pixel_size));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %f", "_rlnMicrographDoseRate", current_movie_asset->dose_per_frame));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %f", "_rlnMicrographPreExposure", pre_exposure_amount));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %f", "_rlnMicrographVoltage", current_movie_asset->microscope_voltage));
                            relion_motioncor_star_current_file->AddLine(wxString::Format("%s %i", "_rlnMicrographStartFrame", first_frame_to_sum));

                            // Write out the section with the whole-frame shifts
                            relion_motioncor_star_current_file->AddLine(wxString(" "));
                            relion_motioncor_star_current_file->AddLine(wxString("data_global_shift"));
                            relion_motioncor_star_current_file->AddLine(wxString(" "));
                            relion_motioncor_star_current_file->AddLine(wxString("loop_"));
                            relion_motioncor_star_current_file->AddLine(wxString("_rlnMicrographFrameNumber #1"));
                            relion_motioncor_star_current_file->AddLine(wxString("_rlnMicrographShiftX #2"));
                            relion_motioncor_star_current_file->AddLine(wxString("_rlnMicrographShiftY #3"));

                            // Grab the actual alignment results from the database
                            should_continue = main_frame->current_project.database.BeginBatchSelect(wxString::Format("SELECT * FROM MOVIE_ALIGNMENT_PARAMETERS_%i", current_image_asset->alignment_id));
                            int   frame_counter;
                            float current_x_shift;
                            float current_y_shift;
                            float first_x_shift;
                            float first_y_shift;
                            if ( ! should_continue ) {
                                MyPrintWithDetails("Error getting alignment result!");
                                DEBUG_ABORT;
                            }
                            while ( should_continue ) {
                                should_continue = main_frame->current_project.database.GetFromBatchSelect("iss", &frame_counter, &current_x_shift, &current_y_shift);
                                if ( frame_counter == 1 ) {
                                    first_x_shift = current_x_shift;
                                    first_y_shift = current_y_shift;
                                }

                                //   Relion expects shifts to be in pixels, but unblur returns them in Angstroms
                                //   Specifically, Relion expects them in the original input movie pixels before binning
                                //   even with --early_binning. For EER, it depends on --eer_upsampling. If it is 2,
                                //   it is 8K pixels (= 2x superresolution), if 1, it is 4K pixels (= hardware pixels)

                                relion_motioncor_star_current_file->AddLine(wxString::Format("%i %f %f", frame_counter, (current_x_shift - first_x_shift) / float(current_movie_asset->pixel_size), (current_y_shift - first_y_shift) / float(current_movie_asset->pixel_size)));
                            }
                            main_frame->current_project.database.EndBatchSelect( );

                            if ( is_valid_parent_movie )
                                relion_motioncor_star_current_file->Write( );
                            relion_motioncor_star_current_file->Close( );
                            delete relion_motioncor_star_current_file;
                        }

                    } // first particle from this image

                } // relion3

                my_dialog->Update(particle_counter + 1);
            } // loop over particles

            relion_star_file->Write( );
            relion_star_file->Close( );

            if ( Relion3RadioButton->GetValue( ) && have_valid_images_and_movies ) {
                relion_corrected_micrographs_file->Write( );
                relion_corrected_micrographs_file->Close( );
                delete relion_corrected_micrographs_file;
            }

            input_stack.CloseFile( );
            output_stack.CloseFile( );

            delete relion_star_file;
            delete current_refinement;

            my_dialog->Destroy( );
        }
}
