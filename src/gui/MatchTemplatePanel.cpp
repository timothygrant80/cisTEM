

// BATCH_HIGH_RES_EXPERIMENT: Uncomment to enable batch iteration over high-res limit values
// When enabled, clicking StartEstimation will automatically cycle through values:
// - First run: GUI value as-is
// - Subsequent runs: 0.5A steps landing on half/whole numbers up to end_value
// - NOTE: there is no check that the user has supplied the --max-search-size which will change the resolution to be the highest for a given size. There
//         is no plan to fix this directly, as we will make that CLI option a radio in this GUI and THEN we can fix it. For now, if you aren't BAH, you shouldn not be building with this hack anyway.
// #define BATCH_HIGH_RES_EXPERIMENT
// #define BATCH_ALL_TEMPLATES

// Mutually exclusive hacks
#if defined(BATCH_HIGH_RES_EXPERIMENT) && defined(BATCH_ALL_TEMPLATES)
#error "BATCH_HIGH_RES_EXPERIMENT && BATCH_ALL_TEMPLATES cannot be defined together"
#endif

#if defined(BATCH_HIGH_RES_EXPERIMENT) || defined(BATCH_ALL_TEMPLATES)
#include <cmath>
// File-static variables for batch experiment state (confined to this translation unit)
static bool  s_batch_experiment_active    = false;
static float s_batch_experiment_end_value = 8.0f; // Stop after this value
static float s_batch_experiment_step      = 0.5f; // Step size in Angstroms
static int   s_first_volume_asset_idx     = 0;
static int   s_number_of_volume_asset_idx = 0;
static int   s_current_volume_asset_idx   = 0;
#endif

//#include "../core/core_headers.h"
#include "../constants/constants.h"
#include "../core/gui_core_headers.h"

// extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel*         image_asset_panel;
extern MyVolumeAssetPanel*        volume_asset_panel;
extern MyRunProfilesPanel*        run_profiles_panel;
extern MyMainFrame*               main_frame;
extern MatchTemplateResultsPanel* match_template_results_panel;

constexpr std::array<int, 7> kAllowedSearchSizes = {4096, 2048, 1024, 512, 256, 128, 64};

MatchTemplatePanel::MatchTemplatePanel(wxWindow* parent)
    : MatchTemplatePanelParent(parent) {
    // Set variables

    my_job_id   = -1;
    running_job = false;

    group_combo_is_dirty   = false;
    run_profiles_are_dirty = false;
    set_up_to_resume_job   = false;

#ifndef SHOW_CISTEM_GPU_OPTIONS
    UseGPURadioYes->Enable(false);
    UseGPURadioNo->Enable(false);
    UseFastFFTRadioYes->Enable(false);
    UseFastFFTRadioNo->Enable(false);
#endif

#ifndef cisTEM_USING_FastFFT
    UseFastFFTRadioYes->Enable(false);
    UseFastFFTRadioNo->Enable(false);
#endif

    // We need to allow a higher precision, otherwise, the option to resample will almost always be taken
    HighResolutionLimitNumericCtrl->SetPrecision(4);

    SetInfo( );
    FillGroupComboBox( );
    FillRunProfileComboBox( );

    wxSize input_size = InputSizer->GetMinSize( );
    input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
    input_size.y = -1;
    ExpertPanel->SetMinSize(input_size);
    ExpertPanel->SetSize(input_size);

    ResetDefaults( );
    //	EnableMovieProcessingIfAppropriate();

    result_bitmap.Create(1, 1, 24);
    time_of_last_result_update = time(NULL);

    // Restrict the angular search values to what is allowed on the CLI input
    OutofPlaneStepNumericCtrl->SetMinMaxValue(0.1f, 360.f);
    InPlaneStepNumericCtrl->SetMinMaxValue(0.1f, 360.f);
    DefocusSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
    DefocusSearchStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
    PixelSizeSearchRangeNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
    PixelSizeSearchStepNumericCtrl->SetMinMaxValue(0.01f, FLT_MAX);
    HighResolutionLimitNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);

    SymmetryComboBox->Clear( );
    SymmetryComboBox->Append("C1");
    SymmetryComboBox->Append("C2");
    SymmetryComboBox->Append("C3");
    SymmetryComboBox->Append("C4");
    SymmetryComboBox->Append("D2");
    SymmetryComboBox->Append("D3");
    SymmetryComboBox->Append("D4");
    SymmetryComboBox->Append("I");
    SymmetryComboBox->Append("I2");
    SymmetryComboBox->Append("O");
    SymmetryComboBox->Append("T");
    SymmetryComboBox->Append("T2");
    SymmetryComboBox->SetSelection(0);

    SearchSizeComboBox->Clear( );
    SearchSizeComboBox->Append("high-res limit");
    for ( int size : kAllowedSearchSizes ) {
        SearchSizeComboBox->Append(wxString::Format("%d", size));
    }
    SearchSizeComboBox->SetSelection(0);
    SearchSizeComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, [this](wxCommandEvent&) {
        HighResolutionLimitNumericCtrl->Enable(SearchSizeComboBox->GetSelection( ) == 0);
    });

    GroupComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MatchTemplatePanel::OnGroupComboBox, this);

#ifdef BATCH_ALL_TEMPLATES
    s_number_of_volume_asset_idx = volume_asset_panel->all_assets_list->number_of_assets;
#endif
}

/*
void MatchTemplatePanel::EnableMovieProcessingIfAppropriate()
{
	// Check whether all members of the group have movie parents. If not, make sure we only allow image processing
	MovieRadioButton->Enable(true);
	NoMovieFramesStaticText->Enable(true);
	NoFramesToAverageSpinCtrl->Enable(true);
	for (int counter = 0; counter < image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()); counter ++ )
	{
		if (image_asset_panel->all_assets_list->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(),counter))->parent_id < 0)
		{
			MovieRadioButton->SetValue(false);
			MovieRadioButton->Enable(false);
			NoMovieFramesStaticText->Enable(false);
			NoFramesToAverageSpinCtrl->Enable(false);
			ImageRadioButton->SetValue(true);
		}
	}
}
*/

void MatchTemplatePanel::ResetAllDefaultsClick(wxCommandEvent& event) {
    ResetDefaults( );
}

void MatchTemplatePanel::OnInfoURL(wxTextUrlEvent& event) {
    const wxMouseEvent& ev = event.GetMouseEvent( );

    // filter out mouse moves, too many of them
    if ( ev.Moving( ) )
        return;

    long start = event.GetURLStart( );

    wxTextAttr my_style;

    InfoText->GetStyle(start, my_style);

    // Launch the URL

    wxLaunchDefaultBrowser(my_style.GetURL( ));
}

void MatchTemplatePanel::Reset( ) {
    ProgressBar->SetValue(0);
    TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
    FinishButton->Show(false);

    ProgressPanel->Show(false);
    StartPanel->Show(true);
    OutputTextPanel->Show(false);
    output_textctrl->Clear( );
    ResultsPanel->Show(false);
    InputPanel->Show(true);
    //graph_is_hidden = true;
    InfoPanel->Show(true);

    ResultsPanel->Clear( );

    if ( running_job == true ) {
        main_frame->job_controller.KillJob(my_job_id);
        cached_results.Clear( );

        running_job = false;
    }

    ResetDefaults( );
    Layout( );
}

void MatchTemplatePanel::ResetDefaults( ) {
    OutofPlaneStepNumericCtrl->ChangeValueFloat(2.5);
    InPlaneStepNumericCtrl->ChangeValueFloat(1.5);
    MinPeakRadiusNumericCtrl->ChangeValueFloat(10.0f);

    DefocusSearchYesRadio->SetValue(true);
    PixelSizeSearchNoRadio->SetValue(true);
    set_up_to_resume_job = false;

    SymmetryComboBox->SetValue("C1");

    if ( main_frame->current_project.is_open ) {
        ResumeRunCheckBox->SetValue(false);
        if ( match_template_results_panel->template_match_job_ids.empty( ) )
            ResumeRunCheckBox->Enable(true);
        else
            ResumeRunCheckBox->Enable(false);
    }
    else {
        ResumeRunCheckBox->Enable(false);
    }

#ifdef SHOW_CISTEM_GPU_OPTIONS
    UseGPURadioYes->SetValue(true);
#ifdef cisTEM_USING_FastFFT
    UseFastFFTRadioYes->SetValue(true);
#endif
#else
    UseGPURadioNo->SetValue(true);
    UseFastFFTRadioNo->SetValue(true);
#endif

    UsePeakSamplingCorrectionRadioYes->SetValue(true);

    DefocusSearchRangeNumericCtrl->ChangeValueFloat(1200.0f);
    DefocusSearchStepNumericCtrl->ChangeValueFloat(200.0f);
    PixelSizeSearchRangeNumericCtrl->ChangeValueFloat(0.05f);
    PixelSizeSearchStepNumericCtrl->ChangeValueFloat(0.01f);

    //	AssetGroup active_group;
    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);
    if ( active_group.number_of_members > 0 ) {
        ImageAsset* current_image;
        current_image = image_asset_panel->ReturnAssetPointer(GroupComboBox->GetSelection( ));
        HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_image->pixel_size);
    }
}

void MatchTemplatePanel::OnGroupComboBox(wxCommandEvent& event) {
    //	ResetDefaults();
    //	AssetGroup active_group;

    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);

    if ( active_group.number_of_members > 0 ) {

        ImageAsset* current_image;
        current_image = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
        HighResolutionLimitNumericCtrl->ChangeValueFloat(2.0f * current_image->pixel_size);
    }

    if ( GroupComboBox->GetCount( ) > 0 && main_frame->current_project.is_open == true )
        all_images_have_defocus_values = CheckGroupHasDefocusValues( );

    if ( all_images_have_defocus_values == true && PleaseEstimateCTFStaticText->IsShown( ) == true ) {

        PleaseEstimateCTFStaticText->Show(false);
        Layout( );
    }
    else if ( all_images_have_defocus_values == false && PleaseEstimateCTFStaticText->IsShown( ) == false ) {
        PleaseEstimateCTFStaticText->Show(true);
        Layout( );
    }
}

void MatchTemplatePanel::SetInfo( ) {
    /*	#include "icons/ctffind_definitions.cpp"
	#include "icons/ctffind_diagnostic_image.cpp"
	#include "icons/ctffind_example_1dfit.cpp"

	wxLogNull *suppress_png_warnings = new wxLogNull;
	wxBitmap definitions_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_definitions);
	wxBitmap diagnostic_image_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_diagnostic_image);
	wxBitmap example_1dfit_bmp = wxBITMAP_PNG_FROM_DATA(ctffind_example_1dfit);
	delete suppress_png_warnings;*/

    InfoText->GetCaret( )->Hide( );

    InfoText->BeginSuppressUndo( );
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->BeginFontSize(14);
    InfoText->WriteText(wxT("Match Templates"));
    InfoText->EndFontSize( );
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->WriteText(wxT("In template matching, we localize macromolecular complexes in cryo-EM images using a matched filter, the optimal detector for a known signal in Gaussian noise. Compared to single-particle cryo-EM this approach produces a scoring metric that is meaningful on an absolute scale, allowing for automation and the ability to perform statistical inference using the results, while the trade-off is computational expense (Rickgauer, Grigorieff, Denk 2017). In addition to visual proteomics, our template matching implementation allows for high-fidelity detection of bound ligands without template bias while using substantially fewer images than single-particle cryo-EM (Lucas, Himes, Grigorieff 2023).\n\nThe quality of detection depends critically on how faithfully the reference volume reproduces the contrast of the target as it appears in the raw image. Templates are therefore best generated by physics-based simulation rather than simple projection of an atomic model. The cisTEM simulator uses multislice wave propagation through a fully atomistic specimen model, including a coarse-grained solvent representation and an explicit treatment of inelastic scattering via the Frozen-Plasmon method (Himes & Grigorieff, 2021).\n\nThe approach requires enough computation that GPU computation is essential for practical use, especially when searching multiple defocus planes for in situ samples (Lucas, Himes et al. 2021). In order to make the algorithm accessible to users without large GPU clusters, we have introduced a custom FFT library, highly optimized task level parallelsim and optional on-the-fly downsampling with a correction for correlation peak undersampling (Himes, Grant 2026)."));
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("Program Options"));
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Input Group : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The group of image assets to look for templates in"));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Reference Volume : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The volume that will used for the template search. NOTE: if you are intend to compare results from different templates it is essential that they have the same average power spectrum, which can be achieved using the scale_with_mask program. Otherwise, differences in local image quality can mask real (or worse yet, create false) differences in template matching score between templates."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("High-ResolutionLimit : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Defaults to Nyquist limit (2x pixel size) but can be increased to speed up processing, especially for samples that likely have less high-resolution information"));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Limit Search Size : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Alternative to setting the resolution limit, especially useful for data sets with images of different sizes. Uses the maximum resolution that fits in a square window of the specified size. For images > 4096 you MUST downsample to use the FastFFT option."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Point Group Symmetry : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Restrict the search space by the given symmetry operator."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Perform Defocus Search : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Repeat the full search at different nominal defocus values. The defocus offset is saved per-pixel in the results. Particularly useful for thicker in situ samples."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Perform Pixel Size Search : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Repeat the full search, including defocus if specified, for different magnification changes. Generally, this should NOT be used and instead a refined pixel size is better determined for the data set using a few images at the outset of an experiment."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Peak Selection : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Exclusion radius around each peak in x/y."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Run Profile : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The selected run profile will be used to run the job. For TM, prefer one process with 2-4 threads per GPU. Each process must see only one GPU, which can be accomplished by modifying the run command with \" CUDA_VISIBLE_DEVICES = N <path and command>\".\nRun profiles are set in the Run Profile panel, located under settings."));
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("Expert Options"));
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    // TODO: add section on scaling based on pixel size. Note that the input is a target that may slightly differ. Note limits on sizing and power of two (ref to fast FFT bit)
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Out of Plane Angular Step : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The angular step that should be used for the out of plane search.  Smaller values may increase accuracy, but will significantly increase the required processing time."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("In Plane Angular Step : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("The angular step that should be used for the in plane search.  As with the out of plane angle, smaller values may increase accuracy, but will significantly increase the required processing time."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Use GPU : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Debug option. Generally should not be changed."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Use FastFFT Library : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Debug option. Generally should not be changed unless you must search an image > 4096 in any dimension w/o downsampling."));
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Use Peak Sampling Correction : "));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("Debug option. Corrects for scalloping loss when peak intensity is split over pixels due to undersampling. Generally should not be changed."));
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxT("References"));
    InfoText->EndBold( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    // NOTE: created these via export bibliography in Zotero using the Elsevier style.
    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Rickgauer J.P., Grigorieff N., Denk W."));
    InfoText->EndBold( );
    InfoText->WriteText(wxT(" 2017. Single-protein detection in crowded molecular environments in cryo-EM images. Elife 6, e25648.. "));
    InfoText->BeginURL("http://doi.org/10.7554/eLife.25648");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("doi:10.7554/eLife.25648"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Lucas, B.A., Himes, B.A., Grigorieff, N."));
    InfoText->EndBold( );
    InfoText->WriteText(wxT("2023. Baited reconstruction with 2D template matching for high-resolution structure determination in vitro and in vivo without template bias. eLife 12, RP90486. "));
    InfoText->BeginURL("https://doi.org/10.7554/eLife.90486");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("doi:10.7554/eLife.90486"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Himes, B., Grigorieff, N."));
    InfoText->EndBold( );
    InfoText->WriteText(wxT(" 2021. Cryo-TEM simulations of amorphous radiation-sensitive samples using multislice wave propagation. IUCrJ 8, 943–953."));
    InfoText->BeginURL("https://doi.org/10.1107/S2052252521008538");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("doi:10.1107/S2052252521008538"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Lucas, B.A., Himes, B.A., Xue, L., Grant, T., Mahamid, J., Grigorieff, N."));
    InfoText->EndBold( );
    InfoText->WriteText(wxT(" 2021. Locating macromolecular assemblies in cells by 2D template matching with cisTEM. eLife 10, e68946. "));
    InfoText->BeginURL("https://doi.org/10.7554/eLife.68946");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("doi:10.7554/eLife.68946"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );

    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Himes B.A., Grant, T."));
    InfoText->EndBold( );
    InfoText->WriteText(wxT(" 2026. To Be Published."));
    InfoText->BeginURL("http://example.com");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("doi:tbp"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->BeginBold( );

    InfoText->Newline( );

    InfoText->EndSuppressUndo( );
}

void MatchTemplatePanel::FillGroupComboBox( ) {
    // Called from constructor (when panel is created) or OnUpdateUI (when groups change)
    // Return early if no project is open (can happen during workflow switching)
    if ( ! main_frame->current_project.is_open ) {
        return;
    }

    GroupComboBox->FillComboBox(true);

    if ( GroupComboBox->GetCount( ) > 0 && main_frame->current_project.is_open == true )
        all_images_have_defocus_values = CheckGroupHasDefocusValues( );

    if ( all_images_have_defocus_values == true && PleaseEstimateCTFStaticText->IsShown( ) == true ) {
        PleaseEstimateCTFStaticText->Show(false);
        Layout( );
    }
    else if ( all_images_have_defocus_values == false && PleaseEstimateCTFStaticText->IsShown( ) == false ) {
        PleaseEstimateCTFStaticText->Show(true);
        Layout( );
    }
}

void MatchTemplatePanel::FillRunProfileComboBox( ) {
    RunProfileComboBox->FillWithRunProfiles( );
}

bool MatchTemplatePanel::CheckGroupHasDefocusValues( ) {
    wxArrayLong images_with_defocus_values = main_frame->current_project.database.ReturnLongArrayFromSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");
    long        current_image_id;
    int         images_with_defocus_counter;
    bool        image_was_found;

    for ( int image_in_group_counter = 0; image_in_group_counter < image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection( )); image_in_group_counter++ ) {
        current_image_id = image_asset_panel->all_assets_list->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection( ), image_in_group_counter))->asset_id;
        image_was_found  = false;

        for ( images_with_defocus_counter = 0; images_with_defocus_counter < images_with_defocus_values.GetCount( ); images_with_defocus_counter++ ) {
            if ( images_with_defocus_values[images_with_defocus_counter] == current_image_id ) {
                image_was_found = true;
                break;
            }
        }

        if ( image_was_found == false )
            return false;
    }

    return true;
}

/**
 * @brief Checks the currently selected image group for over-focus (negative defocus) images
 *        and presents the user with options to handle them.
 *
 * Queries the database to find images in the active group whose CTF estimates have
 * defocus1 > 0 AND defocus2 > 0 (under-focus). Any images not meeting this criterion
 * are considered over-focus. If over-focus images are found, a dialog is shown offering:
 *
 *   - YES: Create a new image group containing only the under-focus images, select it
 *     in the GroupComboBox, and return true. Because this is called before
 *     active_group.CopyFrom(), the caller will naturally pick up the new group.
 *   - NO: Proceed anyway. Sets @p append_allow_over_focus to true so the caller can
 *     pass --allow-over-focus to the worker processes.
 *   - CANCEL: Abort the run.
 *
 * @param[out] append_allow_over_focus  Set to true if the user chose to proceed despite
 *             over-focus images (NO), meaning --allow-over-focus should be appended to
 *             the run profile. Unchanged otherwise.
 * @return true if the caller should proceed with the job, false to abort.
 */
bool MatchTemplatePanel::CheckForOverFocus(bool& append_allow_over_focus) {
    wxArrayLong underfocus_asset_ids;
    int         group_list_id = image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )].id;
    int         total_members = image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )].number_of_members;
    bool        has_overfocus = main_frame->current_project.database.ReturnAllAssetIdsWithUnderfocus(
                   group_list_id, total_members, underfocus_asset_ids);

    if ( ! has_overfocus )
        return true;

    int overfocus_count = total_members - underfocus_asset_ids.GetCount( );

    wxString message = wxString::Format(
            "%d of %d images in this group have over-focus (negative defocus values), "
            "which may produce unreliable template matching results.\n\n"
            "Would you like to create a new image group containing only the %zu "
            "under-focus images?\n\n"
            "YES = Create a new filtered group and select it\n"
            "NO = Ignore over-focus and proceed anyway (not recommended)\n"
            "CANCEL = Abort",
            overfocus_count, total_members,
            underfocus_asset_ids.GetCount( ));

    wxMessageDialog overfocus_dialog(this, message, "Over-Focus Detected",
                                     wxYES_NO | wxCANCEL | wxYES_DEFAULT | wxICON_WARNING);

    int result = overfocus_dialog.ShowModal( );

    if ( result == wxID_CANCEL )
        return false;

    if ( result == wxID_NO ) {
        append_allow_over_focus = true;
        return true;
    }

    // wxID_YES — create a new filtered group
    wxTextEntryDialog name_dialog(this, "Enter a name for the new image group:",
                                  "New Image Group", "under_focus");
    if ( name_dialog.ShowModal( ) != wxID_OK )
        return false;

    wxString new_group_name = name_dialog.GetValue( );

    // Calling AddGroup increments all_groups_list->number_of_groups
    image_asset_panel->all_groups_list->AddGroup(new_group_name);
    // This is used in dynamic table naming in the DB so is indexed from 1 not 0
    long new_group_id = image_asset_panel->all_groups_list->ReturnNumberOfGroups( );

    // current_group_number in the image_asset_panel SEEMs to server the same function as number_of_groups (whereas a second var selected_group would map to current_group in my (BAH) mind)
    image_asset_panel->current_group_number = new_group_id;

    image_asset_panel->AddGroupToDatabase(new_group_id, new_group_name.ToUTF8( ).data( ), new_group_id);

    // Man, group is a funny looking word at this point.
    AssetGroup& new_group = image_asset_panel->all_groups_list->groups[image_asset_panel->all_groups_list->number_of_groups - 1];
    new_group.id          = new_group_id;

    // FIXME: this should probably be a database class method. (Or maybe it is and we should be calling it.)
    main_frame->current_project.database.Begin( );
    main_frame->current_project.database.BeginBatchInsert(
            wxString::Format("IMAGE_GROUP_%ld", new_group_id), 2, "MEMBER_NUMBER", "IMAGE_ASSET_ID");

    for ( size_t i = 0; i < underfocus_asset_ids.GetCount( ); i++ ) {
        long asset_id = underfocus_asset_ids[i];
        main_frame->current_project.database.AddToBatchInsert("ii", (int)i, (int)asset_id);
        int array_pos = image_asset_panel->ReturnArrayPositionFromAssetID(asset_id);
        new_group.AddMember(array_pos);
    }

    main_frame->current_project.database.EndBatchInsert( );
    main_frame->current_project.database.Commit( );

    // The method for image_asset_panel->DirtyGroups() is protected and I don't want to mess with it (BAH)
    main_frame->DirtyImageGroups( );
    FillGroupComboBox( );

    // Presumably we always want to select the most recent addition which will be at the end so this (should) be safe
    GroupComboBox->SetSelection(GroupComboBox->GetCount( ) - 1);

    // Not sure why this is here, FIXME.
    // wxCommandEvent dummy_event;
    // OnGroupComboBox(dummy_event);

    return true;
}

void MatchTemplatePanel::OnUpdateUI(wxUpdateUIEvent& event) {

    // We want things to be greyed out if the user is re-running the job.
    if ( set_up_to_resume_job ) {
        return;
    }

    // are there enough members in the selected group.
    if ( main_frame->current_project.is_open == false ) {
        RunProfileComboBox->Enable(false);
        GroupComboBox->Enable(false);
        StartEstimationButton->Enable(false);
        ReferenceSelectPanel->Enable(false);
        ResumeRunCheckBox->Enable(false);
    }
    else {
        if ( match_template_results_panel->template_match_job_ids.empty( ) ) {
            ResumeRunCheckBox->Enable(true);
        }
        else {
            ResumeRunCheckBox->Enable(false);
        }

        if ( running_job == false ) {
            RunProfileComboBox->Enable(true);
            GroupComboBox->Enable(true);
            ReferenceSelectPanel->Enable(true);

            if ( RunProfileComboBox->GetCount( ) > 0 ) {
                if ( image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection( )) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection( )) > 0 && all_images_have_defocus_values == true ) {
                    StartEstimationButton->Enable(true);
                }
                else
                    StartEstimationButton->Enable(false);
            }
            else {
                StartEstimationButton->Enable(false);
            }

            if ( DefocusSearchYesRadio->GetValue( ) == true ) {
                DefocusRangeStaticText->Enable(true);
                DefocusSearchRangeNumericCtrl->Enable(true);
                DefocusStepStaticText->Enable(true);
                DefocusSearchStepNumericCtrl->Enable(true);
            }
            else {
                DefocusRangeStaticText->Enable(false);
                DefocusSearchRangeNumericCtrl->Enable(false);
                DefocusStepStaticText->Enable(false);
                DefocusSearchStepNumericCtrl->Enable(false);
            }

            if ( PixelSizeSearchYesRadio->GetValue( ) == true ) {
                PixelSizeRangeStaticText->Enable(true);
                PixelSizeSearchRangeNumericCtrl->Enable(true);
                PixelSizeStepStaticText->Enable(true);
                PixelSizeSearchStepNumericCtrl->Enable(true);
            }
            else {
                PixelSizeRangeStaticText->Enable(false);
                PixelSizeSearchRangeNumericCtrl->Enable(false);
                PixelSizeStepStaticText->Enable(false);
                PixelSizeSearchStepNumericCtrl->Enable(false);
            }
        }
        else {
            GroupComboBox->Enable(false);
            ReferenceSelectPanel->Enable(false);
            RunProfileComboBox->Enable(false);
        }

        if ( group_combo_is_dirty == true ) {
            FillGroupComboBox( );
            group_combo_is_dirty = false;
        }

        if ( run_profiles_are_dirty == true ) {
            FillRunProfileComboBox( );
            run_profiles_are_dirty = false;
        }

        if ( volumes_are_dirty == true ) {
            ReferenceSelectPanel->FillComboBox( );
            volumes_are_dirty = false;
        }
    }
}

/** 
 * Disables argument controls if set_up_to_resume_job is true.
 * In that case a TemplateMatchResult must be supplied to set
 * the arguments identical to the to be resumed job.
*/

void MatchTemplatePanel::SetInputsForPossibleReRun(bool set_up_to_resume_job, TemplateMatchJobResults* job_to_resume) {

    this->set_up_to_resume_job = set_up_to_resume_job;
    bool enable_value;

    // FIXME: If the project is moved, the reference volume will have anew path and we aren't allowing updates to the reference.
    // Probably the right solution is to be able to update the path to the reference when updating the project. Or alternatively
    // to have a dialog pop up when a reference is NOT found and ask if the user wants to update the reference.
    if ( set_up_to_resume_job ) {
        // We want to disable user inputs so the job run matches the intial state.
        enable_value = false;
        ResumeRunCheckBox->Show(true);
        ResumeRunCheckBox->SetValue(true);
        ResumeRunCheckBox->Enable(true);
        ResetAllDefaultsButton->Enable(false);

        // We want to set the controls to the values of the job to be resumed.

        // SetSelection requires the array position in the volume asset panel,
        // which needs to be calculated from the volume asset id.
        ReferenceSelectPanel->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(job_to_resume->ref_volume_asset_id));
        OutofPlaneStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->out_of_plane_step));
        InPlaneStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->in_plane_step));
        MinPeakRadiusNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->min_peak_radius));
        HighResolutionLimitNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->high_res_limit));
        SymmetryComboBox->SetValue(job_to_resume->symmetry);
        // If either range or step are 0 no search will be perfomed.
        DefocusSearchYesRadio->SetValue(job_to_resume->defocus_search_range != 0.0f && job_to_resume->defocus_step != 0.0f);
        DefocusSearchNoRadio->SetValue(job_to_resume->defocus_search_range == 0.0f || job_to_resume->defocus_step == 0.0f);
        PixelSizeSearchYesRadio->SetValue(job_to_resume->pixel_size_search_range != 0.0f && job_to_resume->pixel_size_step != 0.0f);
        PixelSizeSearchNoRadio->SetValue(job_to_resume->pixel_size_search_range == 0.0f || job_to_resume->pixel_size_step == 0.0f);
        DefocusSearchRangeNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->defocus_search_range));
        DefocusSearchStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->defocus_step));
        PixelSizeSearchRangeNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->pixel_size_search_range));
        PixelSizeSearchStepNumericCtrl->SetValue(wxString::Format(wxT("%f"), job_to_resume->pixel_size_step));
    }
    else {
        // We want to allow the user to not re-run the job if the disable the ReRun radio button.
        // The state remembered in "was_enabled..." is meaningless here.
        enable_value = true;
        ResetAllDefaultsButton->Enable(true);
    }

    //SetAndRememberEnableState(GroupComboBox, was_enabled_GroupComboBox, enable_value);
    SetAndRememberEnableState(ReferenceSelectPanel, was_enabled_ReferenceSelectPanel, enable_value);

    SetAndRememberEnableState(OutofPlaneStepNumericCtrl, was_enabled_OutofPlaneStepNumericCtrl, enable_value);
    SetAndRememberEnableState(InPlaneStepNumericCtrl, was_enabled_InPlaneStepNumericCtrl, enable_value);
    SetAndRememberEnableState(MinPeakRadiusNumericCtrl, was_enabled_MinPeakRadiusNumericCtrl, enable_value);

    SetAndRememberEnableState(DefocusSearchYesRadio, was_enabled_DefocusSearchYesRadio, enable_value);
    SetAndRememberEnableState(DefocusSearchNoRadio, was_enabled_DefocusSearchNoRadio, enable_value);
    SetAndRememberEnableState(PixelSizeSearchYesRadio, was_enabled_PixelSizeSearchYesRadio, enable_value);
    SetAndRememberEnableState(PixelSizeSearchNoRadio, was_enabled_PixelSizeSearchNoRadio, enable_value);

    SetAndRememberEnableState(SymmetryComboBox, was_enabled_SymmetryComboBox, enable_value);
    SetAndRememberEnableState(HighResolutionLimitNumericCtrl, was_enabled_HighResolutionLimitNumericCtrl, enable_value);
    SetAndRememberEnableState(DefocusSearchRangeNumericCtrl, was_enabled_DefocusSearchRangeNumericCtrl, enable_value);
    SetAndRememberEnableState(DefocusSearchStepNumericCtrl, was_enabled_DefocusSearchStepNumericCtrl, enable_value);

    // It is okay to change the run profile for a rerun
    if ( RunProfileComboBox->GetCount( ) > 0 ) {
        if ( image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection( )) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection( )) > 0 && all_images_have_defocus_values == true ) {
            StartEstimationButton->Enable(true);
        }
        else
            StartEstimationButton->Enable(false);
    }
    else {
        StartEstimationButton->Enable(false);
    }

    if ( group_combo_is_dirty == true ) {
        FillGroupComboBox( );
        group_combo_is_dirty = false;
    }

    if ( run_profiles_are_dirty == true ) {
        FillRunProfileComboBox( );
        run_profiles_are_dirty = false;
    }

    if ( volumes_are_dirty == true ) {
        ReferenceSelectPanel->FillComboBox( );
        volumes_are_dirty = false;
    }
}

void MatchTemplatePanel::StartEstimationClick(wxCommandEvent& event) {

#if defined(BATCH_HIGH_RES_EXPERIMENT) || defined(BATCH_ALL_TEMPLATES)

    // We are running already, print the update
    if ( s_batch_experiment_active ) {
#ifdef BATCH_ALL_TEMPLATES
        // Log what we're running (value was already set in ProcessAllJobsFinished)
        WriteInfoText(wxString::Format("BATCH EXPERIMENT: Running ref index %d/%d: %s\n",
                                       s_current_volume_asset_idx,
                                       s_number_of_volume_asset_idx,
                                       volume_asset_panel->ReturnAssetShortFilename(s_current_volume_asset_idx).ToUTF8( ).data( )));
#else
        // Log what we're running (value was already set in ProcessAllJobsFinished)
        WriteInfoText(wxString::Format("BATCH EXPERIMENT: Running high-res limit = %.2f A",
                                       HighResolutionLimitNumericCtrl->ReturnValue( )));
#endif
    }
    else {
        // First click - activate batch mode
        // Print starting message
        s_batch_experiment_active  = true;
        s_current_volume_asset_idx = ReferenceSelectPanel->GetSelection( );
        VolumeAsset* temp_volume   = volume_asset_panel->ReturnAssetPointer(ReferenceSelectPanel->GetSelection( ));
#ifdef BATCH_ALL_TEMPLATES
        WriteInfoText(wxString::Format("BATCH EXPERIMENT: Starting. Will run all templates in dropdown starting from %d (%s)",
                                       s_current_volume_asset_idx,
                                       temp_volume->filename.GetName( )));
#else
        WriteInfoText(wxString::Format("BATCH EXPERIMENT: Starting. Will run from %.2f to %.2f in %.1fA steps",
                                       HighResolutionLimitNumericCtrl->ReturnValue( ),
                                       s_batch_experiment_end_value,
                                       s_batch_experiment_step));
#endif
    }
#endif // print info block

    // Over-focus check MUST be called before active_group.CopyFrom() below.
    // If the user creates a new filtered group, CheckForOverFocus changes the
    // GroupComboBox selection, so the subsequent CopyFrom picks up the new group.
    bool append_allow_over_focus = false;
    if ( ! CheckForOverFocus(append_allow_over_focus) )
        return;

    active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);

    // Check if this is a resume job. If yes, get the job id and set the active
    // group to the remaining images
    bool        resume = ResumeRunCheckBox->GetValue( );
    int         job_id_to_resume;
    wxArrayLong images_to_resume;
    if ( resume ) {
        images_to_resume = CheckForUnfinishedWork(true, true);
        job_id_to_resume = match_template_results_panel->ResultDataView->ReturnActiveJobID( );
        active_group.RemoveAll( );
        for ( long counter = 0; counter < images_to_resume.GetCount( ); counter++ ) {
            // The active group contains array positions in the image asset
            // panel, which need to be calculate from the image asset id.
            long image_index = image_asset_panel->ReturnArrayPositionFromAssetID(images_to_resume[counter]);
            active_group.AddMember(image_index);
        }
    }

    float resolution_limit;
    float orientations_per_process;
    float current_orientation_counter;

    int job_counter;
    int number_of_rotations = 0;
    int number_of_defocus_positions;
    int number_of_pixel_size_positions;

    bool use_gpu;
    bool use_fast_fft;
    int  max_threads = 1; // Only used for the GPU code. For GUI this comes from the run profile -> command line override as in other programs.

    int image_number_for_gui;
    int number_of_jobs_per_image_in_gui;
    int number_of_jobs;

    double voltage_kV;
    double spherical_aberration_mm;
    double amplitude_contrast;
    double defocus1;
    double defocus2;
    double defocus_angle;
    double phase_shift;
    double iciness;

    input_image_filenames.Clear( );
    cached_results.Clear( );

    ResultsPanel->Clear( );

    // Package the job details..

    EulerSearch* current_image_euler_search;
    ImageAsset*  current_image;
    VolumeAsset* current_volume;

    current_volume         = volume_asset_panel->ReturnAssetPointer(ReferenceSelectPanel->GetSelection( ));
    ref_box_size_in_pixels = current_volume->x_size / current_volume->pixel_size;

    ParameterMap parameter_map;
    parameter_map.SetAllTrue( );

    float wanted_out_of_plane_angular_step = OutofPlaneStepNumericCtrl->ReturnValue( );
    float wanted_in_plane_angular_step     = InPlaneStepNumericCtrl->ReturnValue( );

    float defocus_search_range;
    float defocus_step;
    float pixel_size_search_range;
    float pixel_size_step;

    if ( DefocusSearchYesRadio->GetValue( ) == true ) {
        defocus_search_range = DefocusSearchRangeNumericCtrl->ReturnValue( );
        defocus_step         = DefocusSearchStepNumericCtrl->ReturnValue( );
    }
    else {
        defocus_search_range = 0.0f;
        defocus_step         = 0.0f;
    }

    if ( PixelSizeSearchYesRadio->GetValue( ) == true ) {

        pixel_size_search_range = PixelSizeSearchRangeNumericCtrl->ReturnValue( );
        pixel_size_step         = PixelSizeSearchStepNumericCtrl->ReturnValue( );
    }
    else {
        pixel_size_search_range = 0.0f;
        pixel_size_step         = 0.0f;
    }

    float min_peak_radius = MinPeakRadiusNumericCtrl->ReturnValue( );

    use_gpu                           = UseGPURadioYes->GetValue( ) ? true : false;
    use_fast_fft                      = UseFastFFTRadioYes->GetValue( ) ? true : false;
    bool use_peak_sampling_correction = UsePeakSamplingCorrectionRadioYes->GetValue( ) ? true : false;

    wxString wanted_symmetry = SymmetryComboBox->GetValue( ).Upper( );

    float    high_resolution_limit;
    wxString search_size_str   = SearchSizeComboBox->GetStringSelection( );
    long     search_size_value = 0;

    if ( search_size_str != "high-res limit" && search_size_str.ToLong(&search_size_value) ) {
        bool is_valid = false;
        for ( int allowed : kAllowedSearchSizes ) {
            if ( search_size_value == allowed ) {
                is_valid = true;
                break;
            }
        }
        if ( is_valid ) {
            high_resolution_limit = static_cast<float>(search_size_value);
        }
        else {
            high_resolution_limit = HighResolutionLimitNumericCtrl->ReturnValue( );
        }
    }
    else {
        high_resolution_limit = HighResolutionLimitNumericCtrl->ReturnValue( );
    }

    wxPrintf("\n\nWanted symmetry %s, Defocus Range %3.3f, Defocus Step %3.3f\n", wanted_symmetry, defocus_search_range, defocus_step);

    RunProfile active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection( )];

    if ( append_allow_over_focus ) {
        active_refinement_run_profile.AppendCLIArgument("--allow-over-focus");
    }

    int number_of_processes = active_refinement_run_profile.ReturnTotalJobs( );

    // how many jobs are there going to be..

    // get first image to make decisions about how many jobs.. .we assume this is representative.

    current_image              = image_asset_panel->ReturnAssetPointer(active_group.members[0]);
    current_image_euler_search = new EulerSearch;
    // NOTE: resolution limit is not actually used here
    resolution_limit = 1.f;
    current_image_euler_search->InitGrid(wanted_symmetry, wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);

    if ( wanted_symmetry.StartsWith("C") ) {
        if ( current_image_euler_search->test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
        {
            current_image_euler_search->theta_max = 180.0f;
        }
    }

    // Normally this is called in EulerSearch::InitGrid, but we need to re-call it here to get the search positions WITHOUT the default randomization to phi (azimuthal angle.)
    current_image_euler_search->CalculateGridSearchPositions(false);

    // Optionally split each image over multiple jobs (processes)
    // The coordinating thread needs to process all the worker's results, so we can only process 1 image at a time, i.e.
    // the min number of jobs per image is number_of_processes
    if ( use_gpu ) {
        number_of_jobs_per_image_in_gui = number_of_processes; // Using two threads in each job

        number_of_jobs = number_of_jobs_per_image_in_gui * active_group.number_of_members;

        wxPrintf("In USEGPU:\n There are %d search positions\nThere are %d jobs per image\n", current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
        delete current_image_euler_search;
    }
    else {
        if ( active_group.number_of_members >= 5 || current_image_euler_search->number_of_search_positions < number_of_processes * 20 )
            number_of_jobs_per_image_in_gui = number_of_processes;
        else if ( current_image_euler_search->number_of_search_positions > number_of_processes * 250 )
            number_of_jobs_per_image_in_gui = number_of_processes * 10;
        else
            number_of_jobs_per_image_in_gui = number_of_processes * 5;

        number_of_jobs = number_of_jobs_per_image_in_gui * active_group.number_of_members;

        delete current_image_euler_search;
    }

    // Some settings for testing
    //	float defocus_search_range = 1200.0f;
    //	float defocus_step = 200.0f;

    // number of rotations

    for ( float current_psi = 0.0f; current_psi <= 360.0f; current_psi += wanted_in_plane_angular_step ) {
        number_of_rotations++;
    }

    // CPU match_template can probably be DEPRECATED
    if ( use_gpu )
        current_job_package.Reset(active_refinement_run_profile, "match_template_gpu", number_of_jobs);
    else
        current_job_package.Reset(active_refinement_run_profile, "match_template", number_of_jobs);

    expected_number_of_results = 0;
    number_of_received_results = 0;

    // loop over all images..

    OneSecondProgressDialog* my_progress_dialog = new OneSecondProgressDialog("Preparing Job", "Preparing Job...", active_group.number_of_members, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_APP_MODAL);

    TemplateMatchJobResults temp_result;
    temp_result.input_job_id               = -1;
    temp_result.job_type                   = cistem::job_type::template_match_full_search;
    temp_result.mask_radius                = 0.0f;
    temp_result.min_peak_radius            = min_peak_radius;
    temp_result.exclude_above_xy_threshold = false;
    temp_result.xy_change_threshold        = 0.0f;

    for ( int image_counter = 0; image_counter < active_group.number_of_members; image_counter++ ) {
        image_number_for_gui = image_counter + 1;

        // current image asset

        current_image = image_asset_panel->ReturnAssetPointer(active_group.members[image_counter]);

        // setup the euler search for this image..
        // this needs to be changed when more parameters are added.
        // right now, the resolution is always Nyquist.

        resolution_limit           = current_image->pixel_size * 2.0f;
        current_image_euler_search = new EulerSearch;
        current_image_euler_search->InitGrid(wanted_symmetry, wanted_out_of_plane_angular_step, 0.0, 0.0, 360.0, wanted_in_plane_angular_step, 0.0, current_image->pixel_size / resolution_limit, parameter_map, 1);
        if ( wanted_symmetry.StartsWith("C") ) {
            if ( current_image_euler_search->test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
            {
                current_image_euler_search->theta_max = 180.0f;
            }
        }
        current_image_euler_search->CalculateGridSearchPositions(false);

        if ( DefocusSearchYesRadio->GetValue( ) == true )
            number_of_defocus_positions = 2 * myround(float(defocus_search_range) / float(defocus_step)) + 1;
        else
            number_of_defocus_positions = 1;

        if ( PixelSizeSearchYesRadio->GetValue( ) == true )
            number_of_pixel_size_positions = 2 * myround(float(pixel_size_search_range) / float(pixel_size_step)) + 1;
        else
            number_of_pixel_size_positions = 1;

        wxPrintf("For Image %li\nThere are %i search positions\nThere are %i jobs per image\n", active_group.members[image_counter], current_image_euler_search->number_of_search_positions, number_of_jobs_per_image_in_gui);
        wxPrintf("Calculating %i correlation maps\n", current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions);
        // how many orientations will each process do for this image..
        expected_number_of_results += current_image_euler_search->number_of_search_positions * number_of_rotations * number_of_defocus_positions * number_of_pixel_size_positions;
        orientations_per_process = float(current_image_euler_search->number_of_search_positions) / float(number_of_jobs_per_image_in_gui);
        if ( orientations_per_process < 1 )
            orientations_per_process = 1;

        int number_of_previous_template_matches = main_frame->current_project.database.ReturnNumberOfPreviousTemplateMatchesByAssetID(current_image->asset_id);
        main_frame->current_project.database.GetCTFParameters(current_image->ctf_estimation_id, voltage_kV, spherical_aberration_mm, amplitude_contrast, defocus1, defocus2, defocus_angle, phase_shift, iciness);

        wxString mip_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        mip_output_file += wxString::Format("/%s_mip_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_psi_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_psi_output_file += wxString::Format("/%s_psi_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_theta_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_theta_output_file += wxString::Format("/%s_theta_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_phi_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_phi_output_file += wxString::Format("/%s_phi_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_defocus_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_defocus_output_file += wxString::Format("/%s_defocus_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString best_pixel_size_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        best_pixel_size_output_file += wxString::Format("/%s_pixel_size_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString scaled_mip_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        scaled_mip_output_file += wxString::Format("/%s_scaled_mip_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString output_histogram_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        output_histogram_file += wxString::Format("/%s_histogram_%i_%i.txt", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString output_result_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        output_result_file += wxString::Format("/%s_plotted_result_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString correlation_avg_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        correlation_avg_output_file += wxString::Format("/%s_avg_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        wxString correlation_std_output_file = main_frame->current_project.template_matching_asset_directory.GetFullPath( );
        correlation_std_output_file += wxString::Format("/%s_std_%i_%i.mrc", current_image->filename.GetName( ), current_image->asset_id, number_of_previous_template_matches);

        //		wxString correlation_std_output_file = "/dev/null";
        current_orientation_counter = 0;

        wxString input_search_image   = current_image->filename.GetFullPath( );
        wxString input_reconstruction = current_volume->filename.GetFullPath( );
        float    pixel_size           = current_image->pixel_size;

        input_image_filenames.Add(input_search_image);

        float low_resolution_limit = 300.0f; // FIXME set this somehwere that is not buried in the code!

        temp_result.image_asset_id                  = current_image->asset_id;
        temp_result.job_name                        = wxString::Format("Full search with %s", current_volume->filename.GetName( ));
        temp_result.ref_volume_asset_id             = current_volume->asset_id;
        wxDateTime now                              = wxDateTime::Now( );
        temp_result.datetime_of_run                 = (long int)now.GetAsDOS( );
        temp_result.symmetry                        = wanted_symmetry;
        temp_result.pixel_size                      = pixel_size;
        temp_result.voltage                         = voltage_kV;
        temp_result.spherical_aberration            = spherical_aberration_mm;
        temp_result.amplitude_contrast              = amplitude_contrast;
        temp_result.defocus1                        = defocus1;
        temp_result.defocus2                        = defocus2;
        temp_result.defocus_angle                   = defocus_angle;
        temp_result.phase_shift                     = phase_shift;
        temp_result.low_res_limit                   = low_resolution_limit;
        temp_result.high_res_limit                  = high_resolution_limit;
        temp_result.out_of_plane_step               = wanted_out_of_plane_angular_step;
        temp_result.in_plane_step                   = wanted_in_plane_angular_step;
        temp_result.defocus_search_range            = defocus_search_range;
        temp_result.defocus_step                    = defocus_step;
        temp_result.pixel_size_search_range         = pixel_size_search_range;
        temp_result.pixel_size_step                 = pixel_size_step;
        temp_result.reference_box_size_in_angstroms = ref_box_size_in_pixels * pixel_size;
        temp_result.mip_filename                    = mip_output_file;
        temp_result.scaled_mip_filename             = scaled_mip_output_file;
        temp_result.psi_filename                    = best_psi_output_file;
        temp_result.theta_filename                  = best_theta_output_file;
        temp_result.phi_filename                    = best_phi_output_file;
        temp_result.defocus_filename                = best_defocus_output_file;
        temp_result.pixel_size_filename             = best_pixel_size_output_file;
        temp_result.histogram_filename              = output_histogram_file;
        temp_result.projection_result_filename      = output_result_file;
        temp_result.avg_filename                    = correlation_avg_output_file;
        temp_result.std_filename                    = correlation_std_output_file;

        cached_results.Add(temp_result);

        for ( job_counter = 0; job_counter < number_of_jobs_per_image_in_gui; job_counter++ ) {

            //			float high_resolution_limit = resolution_limit;
            int best_parameters_to_keep = 1;
            //			float defocus_search_range = 0.0f;
            //			float defocus_step = 0.0f;
            float padding            = 1;
            bool  ctf_refinement     = false;
            float mask_radius_search = 0.0f; //current_volume->x_size; // this is actually not really used...

            wxPrintf("\n\tFor image %i, current_orientation_counter is %f\n", image_number_for_gui, current_orientation_counter);
            if ( current_orientation_counter >= current_image_euler_search->number_of_search_positions )
                current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
            int first_search_position = myroundint(current_orientation_counter);
            current_orientation_counter += orientations_per_process;
            if ( current_orientation_counter >= current_image_euler_search->number_of_search_positions || job_counter == number_of_jobs_per_image_in_gui - 1 )
                current_orientation_counter = current_image_euler_search->number_of_search_positions - 1;
            int last_search_position = myroundint(current_orientation_counter);
            current_orientation_counter++;

            wxString directory_for_results = main_frame->current_project.image_asset_directory.GetFullPath( );
            //			wxString directory_for_results = main_frame->ReturnScratchDirectory();

            //wxPrintf("%i = %i - %i\n", job_counter, first_search_position, last_search_position);
            // These are accessed directly via index in MatchTemplateApp::MasterHandleProgramDefinedResult
            // any changes here MUST be propagated there, e.g. jobs[0].arguments[37].ReturnStringArgument( );
            // NOTE: also, please keep in sync with the manual command line arguments.
            // TODO: this is a bit of a mess.

            current_job_package.AddJob("ttffffffffffifffffbfftttttttttftiiiitttfbbbi",
                                       input_search_image.ToUTF8( ).data( ),
                                       input_reconstruction.ToUTF8( ).data( ),
                                       pixel_size,
                                       voltage_kV,
                                       spherical_aberration_mm,
                                       amplitude_contrast,
                                       defocus1,
                                       defocus2,
                                       defocus_angle,
                                       low_resolution_limit,
                                       high_resolution_limit,
                                       wanted_out_of_plane_angular_step,
                                       best_parameters_to_keep,
                                       defocus_search_range,
                                       defocus_step,
                                       pixel_size_search_range,
                                       pixel_size_step,
                                       padding,
                                       ctf_refinement,
                                       mask_radius_search,
                                       phase_shift,
                                       mip_output_file.ToUTF8( ).data( ),
                                       best_psi_output_file.ToUTF8( ).data( ),
                                       best_theta_output_file.ToUTF8( ).data( ),
                                       best_phi_output_file.ToUTF8( ).data( ),
                                       best_defocus_output_file.ToUTF8( ).data( ),
                                       best_pixel_size_output_file.ToUTF8( ).data( ),
                                       scaled_mip_output_file.ToUTF8( ).data( ),
                                       correlation_std_output_file.ToUTF8( ).data( ),
                                       wanted_symmetry.ToUTF8( ).data( ),
                                       wanted_in_plane_angular_step,
                                       output_histogram_file.ToUTF8( ).data( ),
                                       first_search_position,
                                       last_search_position,
                                       image_number_for_gui,
                                       number_of_jobs_per_image_in_gui,
                                       correlation_avg_output_file.ToUTF8( ).data( ),
                                       directory_for_results.ToUTF8( ).data( ),
                                       output_result_file.ToUTF8( ).data( ),
                                       min_peak_radius,
                                       use_gpu,
                                       use_fast_fft,
                                       use_peak_sampling_correction,
                                       max_threads);
        }

        delete current_image_euler_search;
        my_progress_dialog->Update(image_counter + 1);
    }

    my_progress_dialog->Destroy( );

    // Get ID's from database for writing results as they come in..

    template_match_id = main_frame->current_project.database.ReturnHighestTemplateMatchID( ) + 1;

    // If we resume, reuse the job id of the job we want to resume
    if ( resume ) {
        template_match_job_id = job_id_to_resume;
    }
    else {
        template_match_job_id = main_frame->current_project.database.ReturnHighestTemplateMatchJobID( ) + 1;
    }
    // launch a controller

    my_job_id = main_frame->job_controller.AddJob(this, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection( )].manager_command, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection( )].gui_address);

    if ( my_job_id != -1 ) {
        SetNumberConnectedTextToZeroAndStartTracking( );

        StartPanel->Show(false);
        ProgressPanel->Show(true);
        InputPanel->Show(false);

        ExpertPanel->Show(false);
        InfoPanel->Show(false);
        OutputTextPanel->Show(true);
        ResultsPanel->Show(true);

        GroupComboBox->Enable(false);
        Layout( );
    }

    ProgressBar->Pulse( );
}

void MatchTemplatePanel::HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& high_res_limit_used, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes) {
    // result is available for an image..

    cached_results[image_number - 1].found_peaks.Clear( );
    cached_results[image_number - 1].found_peaks    = peak_infos;
    cached_results[image_number - 1].used_threshold = threshold_used;
    cached_results[image_number - 1].high_res_limit = high_res_limit_used;

    ResultsPanel->SetActiveResult(cached_results[image_number - 1]);

    // write to database..

    main_frame->current_project.database.Begin( );

    cached_results[image_number - 1].job_id = template_match_job_id;
    main_frame->current_project.database.AddTemplateMatchingResult(template_match_id, cached_results[image_number - 1]);
    template_match_id++;

    main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(cached_results[image_number - 1].image_asset_id, template_match_job_id);
    main_frame->current_project.database.Commit( );
    match_template_results_panel->is_dirty = true;
}

void MatchTemplatePanel::FinishButtonClick(wxCommandEvent& event) {
    ProgressBar->SetValue(0);
    TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
    FinishButton->Show(false);

    ProgressPanel->Show(false);
    StartPanel->Show(true);
    OutputTextPanel->Show(false);
    output_textctrl->Clear( );
    ResultsPanel->Show(false);
    //graph_is_hidden = true;
    InfoPanel->Show(true);
    InputPanel->Show(true);

    ExpertPanel->Show(true);

    running_job = false;
    Layout( );
}

void MatchTemplatePanel::TerminateButtonClick(wxCommandEvent& event) {
    // kill the job, this will kill the socket to terminate downstream processes
    // - this will have to be improved when clever network failure is incorporated

    main_frame->job_controller.KillJob(my_job_id);

    WriteInfoText("Terminated Job");
    TimeRemainingText->SetLabel("Time Remaining : Terminated");
    CancelAlignmentButton->Show(false);
    FinishButton->Show(true);
    ProgressPanel->Layout( );
    cached_results.Clear( );

#if defined(BATCH_HIGH_RES_EXPERIMENT) || defined(BATCH_ALL_TEMPLATES)
    // Cancel batch experiment on user termination
    if ( s_batch_experiment_active ) {
        s_batch_experiment_active = false;
        WriteInfoText("BATCH EXPERIMENT: Cancelled by user");
    }
#endif

    //running_job = false;
}

void MatchTemplatePanel::WriteInfoText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
    output_textctrl->AppendText(text_to_write);

    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void MatchTemplatePanel::WriteErrorText(wxString text_to_write) {
    output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
    output_textctrl->AppendText(text_to_write);

    if ( text_to_write.EndsWith("\n") == false )
        output_textctrl->AppendText("\n");
}

void MatchTemplatePanel::OnSocketJobResultMsg(JobResult& received_result) {
    if ( received_result.result_size > 0 ) {
        ProcessResult(&received_result);
    }
}

void MatchTemplatePanel::OnSocketJobResultQueueMsg(ArrayofJobResults& received_queue) {
    for ( int counter = 0; counter < received_queue.GetCount( ); counter++ ) {
        ProcessResult(&received_queue.Item(counter));
    }
}

void MatchTemplatePanel::SetNumberConnectedText(wxString wanted_text) {
    NumberConnectedText->SetLabel(wanted_text);
}

void MatchTemplatePanel::SetTimeRemainingText(wxString wanted_text) {
    TimeRemainingText->SetLabel(wanted_text);
}

void MatchTemplatePanel::OnSocketAllJobsFinished( ) {
    ProcessAllJobsFinished( );
}

void MatchTemplatePanel::ProcessResult(JobResult* result_to_process) // this will have to be overidden in the parent clas when i make it.
{

    long     current_time = time(NULL);
    wxString bitmap_string;
    wxString plot_string;

    number_of_received_results++;

    if ( number_of_received_results == 1 ) {
        current_job_starttime = current_time;
        time_of_last_update   = 0;
    }
    else if ( current_time != time_of_last_update ) {
        int current_percentage;
        current_percentage = myroundint(float(number_of_received_results) / float(expected_number_of_results) * 100.0f);

        time_of_last_update = current_time;
        if ( current_percentage > 100 )
            current_percentage = 100;
        ProgressBar->SetValue(current_percentage);

        long  job_time        = current_time - current_job_starttime;
        float seconds_per_job = float(job_time) / float(number_of_received_results - 1);

        long seconds_remaining;
        seconds_remaining = float(expected_number_of_results - number_of_received_results) * seconds_per_job;

        wxTimeSpan time_remaining = wxTimeSpan(0, 0, seconds_remaining);
        TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));
    }

    // results should be ..

    // Defocus 1 (Angstroms)
    // Defocus 2 (Angstroms)
    // Astigmatism Angle (degrees)
    // Additional phase shift (e.g. from phase plate) radians
    // Score
    // Resolution (Angstroms) to which Thon rings are well fit by the CTF
    // Reolution (Angstroms) at which aliasing was detected

    /*

	if (current_time - time_of_last_result_update > 5)
	{
		// we need the filename of the image..

		wxString image_filename = image_asset_panel->ReturnAssetPointer(active_group.members[result_to_process->job_number])->filename.GetFullPath();

		ResultsPanel->Draw(my_job_package.jobs[result_to_process->job_number].arguments[3].ReturnStringArgument(), my_job_package.jobs[result_to_process->job_number].arguments[16].ReturnBoolArgument(), result_to_process->result_data[0], result_to_process->result_data[1], result_to_process->result_data[2], result_to_process->result_data[3], result_to_process->result_data[4], result_to_process->result_data[5], result_to_process->result_data[6], image_filename);
		time_of_last_result_update = time(NULL);
	}
*/

    //	my_job_tracker.MarkJobFinished();
    //	if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

    // store the results..
    //buffered_results[result_to_process->job_number] = result_to_process;
}

void MatchTemplatePanel::ProcessAllJobsFinished( ) {

    MyDebugAssertTrue(my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs, "In ProcessAllJobsFinished, but total_number_of_finished_jobs != total_number_of_jobs. Oops.");

    // Update the GUI with project timings
    extern MyOverviewPanel* overview_panel;
    overview_panel->SetProjectInfo( );

    //
    WriteResultToDataBase( );
    match_template_results_panel->is_dirty = true;

    // let the FindParticles panel check whether any of the groups are now ready to be picked
    //extern MyFindParticlesPanel *findparticles_panel;
    //findparticles_panel->CheckWhetherGroupsCanBePicked();

    cached_results.Clear( );

    // Kill the job (in case it isn't already dead)
    main_frame->job_controller.KillJob(my_job_id);

// Key section to advance to the next experiment in the batch
#if defined(BATCH_HIGH_RES_EXPERIMENT) || defined(BATCH_ALL_TEMPLATES)
#ifdef BATCH_HIGH_RES_EXPERIMENT
    if ( s_batch_experiment_active ) {
        float current_value = HighResolutionLimitNumericCtrl->ReturnValue( );

        // Calculate next value: round up to next 0.5 boundary
        float next_value = std::ceil(current_value * 2.0f) / 2.0f;
        if ( next_value <= current_value ) {
            next_value = current_value + s_batch_experiment_step;
        }
        // Ensure it lands on 0.5 boundary
        next_value = std::round(next_value * 2.0f) / 2.0f;

        if ( next_value <= s_batch_experiment_end_value ) {
            WriteInfoText(wxString::Format("\nBATCH EXPERIMENT: Completed %.2f, next = %.2f",
                                           current_value, next_value));

            // Update the GUI so the Call after has the updated state
            HighResolutionLimitNumericCtrl->ChangeValueFloat(next_value);

            // Use CallAfter for safe event loop handling
            CallAfter([this]( ) {
                if ( s_batch_experiment_active ) {
                    wxCommandEvent dummy_event;
                    StartEstimationClick(dummy_event);
                }
            });
            return; // Don't show Finish button yet
        }
        else {
            // Done with all values
            s_batch_experiment_active = false;
            WriteInfoText(wxString::Format("BATCH EXPERIMENT: All values completed (ended at %.2f)!",
                                           current_value));
        }
    }
#endif

#ifdef BATCH_ALL_TEMPLATES
    if ( s_batch_experiment_active ) {

        if ( s_current_volume_asset_idx + 1 < s_number_of_volume_asset_idx ) {
            WriteInfoText(wxString::Format("BATCH EXPERIMENT: Completed template idx %d, next = %d/%d",
                                           s_current_volume_asset_idx,
                                           s_current_volume_asset_idx + 1,
                                           s_number_of_volume_asset_idx));

            // Update the GUI so the Call after has the updated state
            s_current_volume_asset_idx++;
            ReferenceSelectPanel->SetSelection(s_current_volume_asset_idx);

            // Use CallAfter for safe event loop handling
            CallAfter([this]( ) {
                if ( s_batch_experiment_active ) {
                    wxCommandEvent dummy_event;
                    StartEstimationClick(dummy_event);
                }
            });
            return; // Don't show Finish button yet
        }
        else {
            // Done with all values
            s_batch_experiment_active = false;
            WriteInfoText(wxString::Format("BATCH EXPERIMENT: All templates are completed!"));
        }
    }
#endif
#endif // batch block

    WriteInfoText("All Jobs have finished.");
    ProgressBar->SetValue(100);
    TimeRemainingText->SetLabel("Time Remaining : All Done!");
    CancelAlignmentButton->Show(false);
    FinishButton->Show(true);
    ProgressPanel->Layout( );
}

void MatchTemplatePanel::WriteResultToDataBase( ) {
    // I have moved this to HandleSocketTemplateMatchResultReady so that things are done one result at at time.
    /*
	// find the current highest template match numbers in the database, then increment by one

	int template_match_id = main_frame->current_project.database.ReturnHighestTemplateMatchID() + 1;
	int template_match_job_id =  main_frame->current_project.database.ReturnHighestTemplateMatchJobID() + 1;
	main_frame->current_project.database.Begin();


	for (int counter = 0; counter < cached_results.GetCount(); counter++)
	{
		cached_results[counter].job_id = template_match_job_id;
		main_frame->current_project.database.AddTemplateMatchingResult(template_match_id, cached_results[counter]);
		template_match_id++;
	}

	for (int counter = 0; counter < cached_results.GetCount(); counter++)
	{
		main_frame->current_project.database.SetActiveTemplateMatchJobForGivenImageAssetID(cached_results[counter].image_asset_id, template_match_job_id);
	}

	main_frame->current_project.database.Commit();

	match_template_results_panel->is_dirty = true;
*/
}

void MatchTemplatePanel::UpdateProgressBar( ) {
    ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted( ));
    TimeRemainingText->SetLabel(my_job_tracker.ReturnRemainingTime( ).Format("Time Remaining : %Hh:%Mm:%Ss"));
}

void MatchTemplatePanel::ResumeRunCheckBoxOnCheckBox(wxCommandEvent& event) {
    if ( event.IsChecked( ) ) {
        CheckForUnfinishedWork(true, true);
    }
    else {
        CheckForUnfinishedWork(false, true);
    }
}

/** 
 * This may be called when the user clicks the resume run checkbox.
 * OR
 * When the header in the results panel is changed.
*/

wxArrayLong MatchTemplatePanel::CheckForUnfinishedWork(bool is_checked, bool is_from_check_box) {
    wxArrayLong unfinished_match_template_ids;
    if ( is_checked ) {
        // active group might have been overriden when resuming a run
        active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection( )]);

        int active_job_id                 = match_template_results_panel->ResultDataView->ReturnActiveJobID( );
        int images_total                  = active_group.number_of_members;
        int images_to_be_processed        = 0;
        int images_successfully_processed = 0;

        // Get a list of unfinished images by performing a left join between all
        // image assets or the image assets in the desired group and the results
        // stored for this job. The image assets that don't match are what we
        // want.

        // All images
        if ( active_group.id == -1 ) {
            unfinished_match_template_ids = main_frame->current_project.database.ReturnLongArrayFromSelectCommand(
                    wxString::Format("select IMAGE_ASSETS.IMAGE_ASSET_ID, COMP.IMAGE_ASSET_ID as CID FROM IMAGE_ASSETS "
                                     "LEFT JOIN (SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID = %i ) COMP "
                                     "ON IMAGE_ASSETS.IMAGE_ASSET_ID = COMP.IMAGE_ASSET_ID "
                                     "WHERE CID IS NULL",
                                     active_job_id));
        }
        // An Image group
        else {
            unfinished_match_template_ids = main_frame->current_project.database.ReturnLongArrayFromSelectCommand(
                    wxString::Format("select IMAGE_ASSETS.IMAGE_ASSET_ID, COMP.IMAGE_ASSET_ID as CID FROM IMAGE_GROUP_%i AS IMAGE_ASSETS "
                                     "LEFT JOIN (SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID = %i ) COMP "
                                     "ON IMAGE_ASSETS.IMAGE_ASSET_ID = COMP.IMAGE_ASSET_ID "
                                     "WHERE CID IS NULL",
                                     active_group.id, active_job_id));
        }
        images_to_be_processed        = unfinished_match_template_ids.GetCount( );
        images_successfully_processed = images_total - images_to_be_processed;
        no_unfinished_jobs            = (images_to_be_processed == 0);

        if ( no_unfinished_jobs ) {
            wxPrintf("No unfinished jobs.\n");
            // Only create the dialog if triggered by the checkbox, not by the header change.
            if ( is_from_check_box ) {
                wxMessageDialog* check_dialog = new wxMessageDialog(this,
                                                                    wxString::Format("There is no unfinished work for job %d.\n\nYou may select another job by clicking the header in the TM results panel.",
                                                                                     active_job_id, "Please Confirm", wxOK));
                check_dialog->ShowModal( );
            }
            ResumeRunCheckBox->SetValue(false);
            SetInputsForPossibleReRun(false);
        }
        else {
            wxPrintf("Checking for unfinished work for job %d\n", active_job_id);
            // Only create the dialog if triggered by the checkbox, not by the header change.
            if ( is_from_check_box ) {
                wxMessageDialog* check_dialog = new wxMessageDialog(this,
                                                                    wxString::Format("Resuming work for job %d, which has %d/%d images completed.\n",
                                                                                     active_job_id, images_successfully_processed, images_total, "Please Confirm", wxOK));
                check_dialog->ShowModal( );
            }

            int job_id_to_resume = match_template_results_panel->ResultDataView->ReturnActiveJobID( );
            // This returns just one of the finished jobs, so we can get the arguments from it.
            long                    template_match_id     = main_frame->current_project.database.GetTemplateMatchIdForGivenJobId(job_id_to_resume);
            TemplateMatchJobResults job_results_to_resume = main_frame->current_project.database.GetTemplateMatchingResultByID(template_match_id);
            SetInputsForPossibleReRun(true, &job_results_to_resume);
        }
    }
    else {
        wxPrintf("ELSE\n");
        SetInputsForPossibleReRun(false);
    }
    return unfinished_match_template_ids;
}
