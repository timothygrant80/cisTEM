#include "../../core/core_headers.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <pugixml/pugixml.hpp>

using json = nlohmann::json;

// Warp's XML is read with pugixml, which hands back UTF-8 C strings; wrap them
// in std::string so the comparisons and ToDouble() calls below stay as they were.
static std::string XmlName(const pugi::xml_node& node) {
    return std::string(node.name( ));
}

// Returns "" when the attribute is missing, as the wxWidgets XML reader this replaces did.
static std::string XmlAttribute(const pugi::xml_node& node, const char* name) {
    return std::string(node.attribute(name).value( ));
}

// Live2D's latest_run.json fields are read through these so that a value
// written either as a JSON string ("1234") or as a JSON number (1234) is
// accepted; the JSON reader this replaces converted numbers to strings
// before parsing them and so accepted both.
static std::string JsonToString(const json& value) {
    if ( value.is_string( ) )
        return std::string(value.get<std::string>( ).c_str( ));
    if ( value.is_null( ) )
        return std::string( );
    return std::string(value.dump( ).c_str( ));
}

static bool JsonToLong(const json& value, long& out) {
    if ( value.is_number_integer( ) ) {
        out = value.get<long>( );
        return true;
    }
    if ( value.is_number_float( ) ) {
        out = long(value.get<double>( ));
        return true;
    }
    if ( value.is_string( ) )
        return StringToLong(JsonToString(value), out);
    return false;
}

static bool JsonToDouble(const json& value, double& out) {
    if ( value.is_number( ) ) {
        out = value.get<double>( );
        return true;
    }
    if ( value.is_string( ) )
        return StringToDouble(JsonToString(value), out);
    return false;
}

class
        WarpToCistemApp : public MyApp {

  public:
    bool                          DoCalculation( );
    bool                          GetSettingsFromWarp(const pugi::xml_document& warp_settings_doc, std::string& boxnet_name, double& warp_picking_radius, double& warp_picking_threshold, double& warp_minimum_distance_from_exclusions);
    MovieAsset                    LoadMovieFromWarp(const pugi::xml_document& warp_doc, std::string warp_folder, std::string movie_filename, unsigned long count, float wanted_binned_pixel_size);
    ImageAsset                    LoadImageFromWarp(std::string image_filename, unsigned long parent_asset_id, double parent_voltage, double parent_cs, bool parent_are_white);
    ArrayOfParticlePositionAssets LoadParticlePositionsFromWarp(std::string star_filename, ImageAsset new_image_asset, int starting_id);
    CTF                           LoadCTFFromWarp(const pugi::xml_document& warp_doc, float pixel_size, float voltage, float spherical_aberration, std::string wanted_avrot_filename);
    RefinementPackage*            LoadRefinementPackageFromLive2D(std::string star_filename, std::string stack_filename, double pixel_size, double particle_mass, double voltage, double spherical_aberration, double amplitude_contrast, double picking_radius, Database& database, MovieAssetList& movie_list, Refinement& refinement);
    ArrayofClassifications        LoadClassificationsFromLive2D(std::string live_2d_path, std::string latest_settings_filename);
    void                          DoInteractiveUserInput( );

    std::string warp_directory;
    std::string cistem_parent_directory;
    std::string project_name;
    float    wanted_binned_pixel_size;
    bool     do_import_images;
    bool     do_import_ctf_results;
    bool     do_import_particle_coordinates;
    bool     do_import_refinement_package;
    std::string live_2d_directory;
    bool     do_import_classification_results;

    Project new_project;

  private:
};

IMPLEMENT_APP(WarpToCistemApp)

void WarpToCistemApp::DoInteractiveUserInput( ) {
    std::string   warp_directory                   = "";
    std::string   cistem_parent_directory          = "";
    std::string   project_name                     = "";
    float      wanted_binned_pixel_size         = 1.0;
    bool       do_import_images                 = false;
    bool       do_scale_images_and_make_spectra = false;
    bool       do_import_ctf_results            = false;
    bool       do_import_particle_coordinates   = false;
    bool       do_import_refinement_package     = false;
    std::string   live_2d_directory                = "";
    bool       do_import_classification_results = false;
    float      particle_mass                    = 100.0;
    UserInput* my_input                         = new UserInput("Warp to Cistem", 1.0);

    warp_directory           = my_input->GetFilenameFromUser("Input Warp Directory", "The folder in which Warp processed movies", "./Data", false);
    cistem_parent_directory  = my_input->GetFilenameFromUser("Cistem Project Parent Directory", "The parent directory for the new cistem project", "~/", false);
    project_name             = my_input->GetFilenameFromUser("Project Name", "Name for new cisTEM2 project", "New_Project", false);
    wanted_binned_pixel_size = my_input->GetFloatFromUser("Binned Pixel Size", "Pixel size to resample movies to after import.", "1.0", 0.0);
    do_import_images         = my_input->GetYesNoFromUser("Import Images?", "Should we import aligned averaged images from WARP (using a different motion correction system than cisTEM)?", "Yes");
    if ( do_import_images ) {
        do_scale_images_and_make_spectra = my_input->GetYesNoFromUser("Generate Scaled Images and Spectra?", "Should we make scaled images and spectra? Scaled images are very slow to generate but accelerate normal cisTEM operations", "No");
        do_import_ctf_results            = my_input->GetYesNoFromUser("Import CTF Estimates?", "Should we import results of CTF estimation from Warp?", "Yes");
    }
    else {
        do_scale_images_and_make_spectra = false;
        do_import_ctf_results            = false;
    }
    if ( do_import_ctf_results ) {
        do_import_particle_coordinates = my_input->GetYesNoFromUser("Import Particle Coordinates?", "Should we import coordinates of particles picked by WARP (using BoxNet)?", "Yes");
    }
    else
        do_import_particle_coordinates = false;

    if ( do_import_particle_coordinates ) {
        do_import_refinement_package = my_input->GetYesNoFromUser("Import Refinement Package from Live2D?", "Should we import the refinement package used by Live2D for classification?", "No");
    }
    else
        do_import_refinement_package = false;

    if ( do_import_refinement_package ) {
        live_2d_directory                = my_input->GetFilenameFromUser("Input Live2D Directory", "The folder in which live2d output combined stacks and classification results.", "./Live2D", false);
        particle_mass                    = my_input->GetFloatFromUser("Approximate Particle Mass (kDa)", "The approximate molar mass of the boxes particles", "100.0", 0.0);
        do_import_classification_results = my_input->GetYesNoFromUser("Import Classification Results from Live2D?", "Should we import 2D Classification results from Live2D?", "No");
    }
    else
        do_import_classification_results = false;

    delete my_input;

    my_current_job.ManualSetArguments("tttbbfbbbtfb", warp_directory.c_str(), cistem_parent_directory.c_str(), project_name.c_str(), do_import_images, do_scale_images_and_make_spectra, wanted_binned_pixel_size, do_import_ctf_results, do_import_particle_coordinates, do_import_refinement_package, live_2d_directory.c_str(), particle_mass, do_import_classification_results);
}

bool WarpToCistemApp::GetSettingsFromWarp(const pugi::xml_document& warp_settings_doc, std::string& boxnet_name, double& warp_picking_radius, double& warp_picking_threshold, double& warp_minimum_distance_from_exclusions) {
    pugi::xml_node child_1 = warp_settings_doc.document_element( ).first_child( );
    std::string       str_warp_picking_radius;
    std::string       str_warp_picking_threshold;
    std::string       str_distance;
    while ( child_1 ) {
        if ( XmlName(child_1) == "Picking" ) {
            pugi::xml_node child_2 = child_1.first_child( );
            while ( child_2 ) {
                if ( XmlAttribute(child_2, "Name") == "Diameter" ) {
                    str_warp_picking_radius = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_warp_picking_radius, warp_picking_radius) )
                        SendErrorAndCrash("Couldn't convert Radius into a double");
                    warp_picking_radius = warp_picking_radius / 2; // Radius vs Diameter
                }
                if ( XmlAttribute(child_2, "Name") == "MinimumScore" ) {
                    str_warp_picking_threshold = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_warp_picking_threshold, warp_picking_threshold) )
                        SendErrorAndCrash("Couldn't convert Threshold into a double");
                }
                if ( XmlAttribute(child_2, "Name") == "MinimumDistance" ) {
                    str_distance = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_distance, warp_minimum_distance_from_exclusions) )
                        SendErrorAndCrash("Couldn't convert Minimum Distance into a double");
                }
                if ( XmlAttribute(child_2, "Name") == "ModelPath" ) {
                    boxnet_name = XmlAttribute(child_2, "Value");
                }

                child_2 = child_2.next_sibling( );
            }
        }
        child_1 = child_1.next_sibling( );
    }
    if ( boxnet_name == "" ) {
        SendErrorAndCrash("Could not parse boxnet particle picker from warp settings");
    }
    return true;
}

MovieAsset WarpToCistemApp::LoadMovieFromWarp(const pugi::xml_document& warp_doc, std::string warp_folder, std::string movie_filename, unsigned long count, float wanted_binned_pixel_size) {
    MovieAsset new_asset            = MovieAsset( );
    new_asset.filename              = movie_filename;
    new_asset.asset_name            = new_asset.filename.stem().string();
    new_asset.asset_id              = count + 1;
    new_asset.dark_filename         = "";
    new_asset.output_binning_factor = 1.0;
    /* TODO: Replace this with logic to handle mag distortion info from Warp */
    new_asset.correct_mag_distortion     = false;
    new_asset.mag_distortion_angle       = 0.0;
    new_asset.mag_distortion_major_scale = 1.0;
    new_asset.mag_distortion_minor_scale = 1.0;
    std::string   dimension_string          = "";
    double     pixel_size                = 1.0;
    double     cs                        = 2.7;
    double     voltage                   = 300;
    double     dose_rate                 = 1.0;
    bool       is_valid                  = true;
    pugi::xml_node child_1               = warp_doc.document_element( ).first_child( );
    while ( child_1 ) {
        if ( XmlName(child_1) == "OptionsCTF" ) {
            pugi::xml_node child_2 = child_1.first_child( );
            while ( child_2 ) {
                if ( XmlAttribute(child_2, "Name") == "PixelSizeX" ) {
                    std::string str_pixel_size = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_pixel_size, pixel_size) ) {
                        SendInfo("Couldn't convert Pixel Size to a double");
                        is_valid = false;
                    }
                    new_asset.pixel_size  = pixel_size;
                    double binning_factor = wanted_binned_pixel_size / pixel_size;
                    if ( binning_factor >= 1.0 ) {
                        new_asset.output_binning_factor = binning_factor;
                    }
                }
                if ( XmlAttribute(child_2, "Name") == "GainPath" ) {
                    std::string gain_filename     = AfterLast(XmlAttribute(child_2, "Value"), '\\'); // a Windows path, so the file name is whatever follows the last backslash
                    std::string adjusted_filename = warp_folder + gain_filename; // This requires that the gain filename be in the warp folder! todo locate gain file more flexibly... user selected?
                    new_asset.gain_filename      = adjusted_filename;
                }
                if ( XmlAttribute(child_2, "Name") == "Cs" ) {
                    std::string str_cs = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_cs, cs) ) {
                        SendInfo("Couldn't convert Spherical Aberration to a double");
                        is_valid = false;
                    }
                    new_asset.spherical_aberration = cs;
                }
                if ( XmlAttribute(child_2, "Name") == "Voltage" ) {
                    std::string str_voltage = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_voltage, voltage) ) {
                        SendInfo("Couldn't convert Voltage to a double");
                        is_valid = false;
                    }
                    new_asset.microscope_voltage = voltage;
                }
                if ( XmlAttribute(child_2, "Name") == "Dimensions" ) {
                    dimension_string = XmlAttribute(child_2, "Value");
                }
                child_2 = child_2.next_sibling( );
            }
        }
        else if ( XmlName(child_1) == "OptionsMovieExport" ) {
            pugi::xml_node child_2 = child_1.first_child( );
            while ( child_2 ) {
                if ( XmlAttribute(child_2, "Name") == "DosePerAngstromFrame" ) {
                    std::string str_dose_rate = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_dose_rate, dose_rate) ) {
                        SendInfo("Couldn't convert Dose Rate to a double");
                        is_valid = false;
                    }
                    new_asset.dose_per_frame = dose_rate;
                }
                child_2 = child_2.next_sibling( );
            }
        }
        child_1 = child_1.next_sibling( );
    }
    double x_size_angstroms = 0.0;
    double y_size_angstroms = 0.0;

    std::vector<std::string> tokens               = SplitString(dimension_string, ",", false);
    std::string              str_x_size           = tokens.size( ) > 0 ? tokens[0] : std::string( );
    std::string              str_y_size           = tokens.size( ) > 1 ? tokens[1] : std::string( );
    std::string              str_number_of_frames = tokens.size( ) > 2 ? tokens[2] : std::string( );
    if ( ! StringToDouble(str_x_size, x_size_angstroms) ) {
        SendInfo("Couldn't convert x size to a double");
        is_valid = false;
    }
    int x_size = myroundint(x_size_angstroms / pixel_size);
    if ( ! StringToDouble(str_y_size, y_size_angstroms) ) {
        SendInfo("Couldn't convert y size to a double");
        is_valid = false;
    }
    int y_size                 = myroundint(y_size_angstroms / pixel_size);
    int number_of_frames       = atoi(str_number_of_frames.c_str( ));
    new_asset.x_size           = x_size;
    new_asset.y_size           = y_size;
    new_asset.number_of_frames = number_of_frames;
    new_asset.total_dose       = number_of_frames * new_asset.dose_per_frame;
    new_asset.protein_is_white = false;
    new_asset.is_valid         = is_valid;
    return new_asset;
}

ImageAsset WarpToCistemApp::LoadImageFromWarp(std::string image_filename, unsigned long parent_asset_id, double parent_voltage, double parent_cs, bool parent_is_white) {
    ImageAsset new_asset           = ImageAsset( );
    new_asset.filename             = image_filename;
    new_asset.asset_name           = new_asset.filename.stem().string();
    new_asset.parent_id            = parent_asset_id;
    new_asset.asset_id             = parent_asset_id;
    new_asset.alignment_id         = parent_asset_id;
    new_asset.microscope_voltage   = parent_voltage;
    new_asset.spherical_aberration = parent_cs;
    new_asset.protein_is_white     = parent_is_white;
    ImageFile img_file(image_filename, false);
    new_asset.x_size     = img_file.ReturnXSize( );
    new_asset.y_size     = img_file.ReturnYSize( );
    new_asset.pixel_size = img_file.ReturnPixelSize( );
    new_asset.is_valid   = true;
    return new_asset;
}

CTF WarpToCistemApp::LoadCTFFromWarp(const pugi::xml_document& warp_doc, float pixel_size, float voltage, float spherical_aberration, std::string wanted_avrot_filename) {
    double   defocus = 0.0;
    std::string str_defocus;
    double   defocus_delta = 0.0;
    std::string str_defocus_delta;
    double   defocus_angle = 0.0;
    std::string str_defocus_angle;
    double   defocus_1 = 0.0;
    double   defocus_2 = 0.0;
    //	double defocus_max = 2.0;
    //	std::string str_defocus_max;
    //	double defocus_min = 0.0;
    //	std::string str_defocus_min;
    double   amplitude_contrast = 0.07;
    std::string str_amplitude_contrast;
    double   minimum_range     = .1;
    double   minimum_frequency = .1;
    std::string str_minimum_range;
    double   maximum_range     = 0.6;
    double   maximum_frequency = 0.6;
    std::string str_maximum_range;

    double   phase_shift = 0.0;
    std::string str_phase_shift;
    double   resolution_estimate = 0.0;
    std::string str_resolution_estimate;

    str_resolution_estimate = XmlAttribute(warp_doc.document_element( ), "CTFResolutionEstimate");
    if ( ! StringToDouble(str_resolution_estimate, resolution_estimate) ) {
        SendErrorAndCrash("Couldn't convert resolution estimate to a double");
    }
    pugi::xml_node child_1 = warp_doc.document_element( ).first_child( );
    while ( child_1 ) {
        // TODO - parse `PS1D` eventually... but not yet.
        if ( XmlName(child_1) == "OptionsCTF" ) {
            pugi::xml_node child_2 = child_1.first_child( );
            while ( child_2 ) {
                if ( XmlAttribute(child_2, "Name") == "Amplitude" ) {
                    str_amplitude_contrast = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_amplitude_contrast, amplitude_contrast) ) {
                        SendErrorAndCrash("Couldn't convert Amplitude to a double");
                    }
                }
                if ( XmlAttribute(child_2, "Name") == "RangeMax" ) {
                    str_maximum_range = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_maximum_range, maximum_range) ) {
                        SendErrorAndCrash("Couldn't convert RangeMax to a double");
                    }
                    maximum_frequency = maximum_range / (2 * pixel_size);
                }
                if ( XmlAttribute(child_2, "Name") == "RangeMin" ) {
                    str_minimum_range = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_minimum_range, minimum_range) ) {
                        SendErrorAndCrash("Couldn't convert RangeMin to a double");
                    }
                    minimum_frequency = minimum_range / (2 * pixel_size);
                }
                child_2 = child_2.next_sibling( );
            }
        }
        else if ( XmlName(child_1) == "CTF" ) {
            pugi::xml_node child_2 = child_1.first_child( );
            while ( child_2 ) {
                if ( XmlAttribute(child_2, "Name") == "DefocusAngle" ) {
                    str_defocus_angle = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_defocus_angle, defocus_angle) ) {
                        SendErrorAndCrash("Couldn't convert DefocusAngle to a double");
                    }
                }
                if ( XmlAttribute(child_2, "Name") == "Defocus" ) {
                    str_defocus = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_defocus, defocus) ) {
                        SendErrorAndCrash("Couldn't convert Defocus to a double");
                    }
                }
                if ( XmlAttribute(child_2, "Name") == "DefocusDelta" ) {
                    str_defocus_delta = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_defocus_delta, defocus_delta) ) {
                        SendErrorAndCrash("Couldn't convert DefocusDelta to a double");
                    }
                }
                if ( XmlAttribute(child_2, "Name") == "PhaseShift" ) {
                    str_phase_shift = XmlAttribute(child_2, "Value");
                    if ( ! StringToDouble(str_phase_shift, phase_shift) ) {
                        SendErrorAndCrash("Couldn't convert Phaseshift to a double");
                    }
                }
                child_2 = child_2.next_sibling( );
            }
        }
        child_1 = child_1.next_sibling( );
    }

    defocus_1 = (defocus + defocus_delta / 2.0) * 10000; // convert to Å, warp uses um
    defocus_2 = (defocus - defocus_delta / 2.0) * 10000; // convert to Å, warp uses um
    if ( defocus_angle > 90 )
        defocus_angle += -180; // convert from 0 to 180 to -90 to 90
    CTF return_ctf = CTF(voltage,
                         spherical_aberration,
                         (float)amplitude_contrast,
                         (float)defocus_1,
                         (float)defocus_2,
                         (float)defocus_angle,
                         (float)minimum_frequency,
                         (float)maximum_frequency,
                         -1.0f,
                         pixel_size,
                         (float)phase_shift,
                         0.0f,
                         0.0f,
                         0.0f,
                         0.0f); //A
    return_ctf.SetHighestFrequencyWithGoodFit(pixel_size / resolution_estimate); // send in inverse pixels

    // Write out the avrot file here so we have it in the same scope as the xmldoc for when we want to actually scrape the plot.
    std::ofstream avrot_file(wanted_avrot_filename);
    // Stub file - all 1 all the time
    avrot_file << "# This file is a stub that does not reflect the data from Warp" << std::endl;
    for ( long counter = 0; counter < 6; counter++ ) {
        avrot_file << "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" << std::endl;
    }
    avrot_file.close( );

    return return_ctf;
}

ArrayOfParticlePositionAssets WarpToCistemApp::LoadParticlePositionsFromWarp(std::string star_filename, ImageAsset new_image_asset, int starting_id) {
    ArrayOfParticlePositionAssets loaded_positions;
    ParticlePositionAsset         temp_asset;
    std::ifstream                 input_file(star_filename);
    std::string                   current_line;
    std::string                   str_x_pos;
    double                        x_pos;
    std::string                   str_y_pos;
    double                        y_pos;
    std::string                   str_figure_of_merit;
    double                        figure_of_merit;
    int                           current_id = starting_id;

    MyDebugAssertTrue(input_file.is_open( ), "File not open");

    while ( std::getline(input_file, current_line) ) {
        TrimRight(current_line);
        TrimLeft(current_line);
        if ( current_line.empty() == true )
            continue;
        if ( current_line[0] == '#' || current_line[0] == '\0' || current_line[0] == ';' || current_line[0] == 'd' || current_line[0] == 'l' || current_line[0] == '_' )
            continue; //Added a catch for data and loop
        std::vector<std::string> tokens = SplitString(current_line);
        if ( tokens.size( ) != 4 )
            continue;
        str_x_pos = tokens[0];
        if ( ! StringToDouble(str_x_pos, x_pos) ) {
            SendErrorAndCrash("Couldn't convert X position to a double");
        }
        x_pos     = new_image_asset.pixel_size * x_pos;
        str_y_pos = tokens[1];
        if ( ! StringToDouble(str_y_pos, y_pos) ) {
            SendErrorAndCrash("Couldn't convert Y position to a double");
        }
        y_pos = new_image_asset.pixel_size * y_pos;
        // tokens[2] is the micrograph name, which is not used
        str_figure_of_merit = tokens[3];
        if ( ! StringToDouble(str_figure_of_merit, figure_of_merit) ) {
            SendErrorAndCrash("Couldn't convert Figure of Merit to a double");
        }
        current_id += 1;
        temp_asset.filename    = new_image_asset.filename;
        temp_asset.asset_name  = new_image_asset.asset_name;
        temp_asset.asset_id    = current_id;
        temp_asset.picking_id  = new_image_asset.asset_id;
        temp_asset.parent_id   = new_image_asset.asset_id;
        temp_asset.pick_job_id = 1;
        temp_asset.peak_height = figure_of_merit;
        temp_asset.x_position  = x_pos;
        temp_asset.y_position  = y_pos;
        loaded_positions.push_back(temp_asset);
    }
    return loaded_positions;
}

RefinementPackage* WarpToCistemApp::LoadRefinementPackageFromLive2D(std::string star_filename, std::string stack_filename, double pixel_size, double particle_weight, double voltage, double spherical_aberration, double amplitude_contrast, double particle_radius, Database& database, MovieAssetList& movie_list, Refinement& refinement) {
    RefinementPackage*            refinement_package;
    RefinementPackageParticleInfo temp_particle_info;
    int                           stack_x_size;
    int                           stack_y_size;
    int                           stack_number_of_images;
    bool                          stack_is_ok = GetMRCDetails(stack_filename.c_str( ), stack_x_size, stack_y_size, stack_number_of_images);
    if ( stack_is_ok == false )
        SendErrorAndCrash(Format("Could not load the Stack file: %s\n", stack_filename));
    MyDebugAssertTrue(stack_x_size == stack_y_size, "Particles are not square");
    BasicStarFileReader input_star_file;
    std::string            star_error_text;
    if ( input_star_file.ReadFile(star_filename, &star_error_text) == false ) {
        SendErrorAndCrash(Format("Error: Encountered the following error - aborting :-\n%s", star_error_text));
    }
    MyDebugAssertTrue(stack_number_of_images >= input_star_file.cached_parameters.size(), "Number of Particles in star is larger than in stack");
    refinement_package = new RefinementPackage;
    refinement.SizeAndFillWithEmpty(input_star_file.cached_parameters.size(), 1);
    refinement_package->asset_id                 = 1;
    refinement_package->name                     = "Live2D Refinement Package";
    refinement_package->number_of_classes        = 1;
    refinement_package->number_of_run_refinments = 0; // Adjust later when importing classification results.
    refinement_package->stack_has_white_protein  = true;
    refinement_package->output_pixel_size        = pixel_size;

    refinement.number_of_classes                = refinement_package->number_of_classes;
    refinement.number_of_particles              = input_star_file.cached_parameters.size();
    refinement.name                             = "Imported Parameters";
    refinement.resolution_statistics_box_size   = stack_x_size;
    refinement.resolution_statistics_pixel_size = pixel_size;
    refinement.refinement_package_asset_id      = 1;

    refinement_package->stack_box_size                       = stack_x_size;
    refinement_package->stack_filename                       = stack_filename;
    refinement_package->symmetry                             = "C1";
    refinement_package->estimated_particle_weight_in_kda     = particle_weight;
    refinement_package->estimated_particle_size_in_angstroms = 2.0 * particle_radius;

    refinement_package->refinement_ids.push_back(1);
    refinement_package->references_for_next_refinement.push_back(-1);

    refinement.refinement_id                       = 1;
    refinement.resolution_statistics_are_generated = true;

    temp_particle_info.spherical_aberration = spherical_aberration;
    temp_particle_info.microscope_voltage   = voltage;
    temp_particle_info.pixel_size           = pixel_size;
    temp_particle_info.amplitude_contrast   = amplitude_contrast;
    temp_particle_info.x_pos                = 0; // I am f
    temp_particle_info.y_pos                = 0; // This isn't by default read by star_reader
    refinement.class_refinement_results[0].class_resolution_statistics.Init(temp_particle_info.pixel_size, refinement.resolution_statistics_box_size);
    refinement.class_refinement_results[0].class_resolution_statistics.GenerateDefaultStatistics(refinement_package->estimated_particle_weight_in_kda);
    // Loop over all particles
    std::string                      movie_name                          = "";
    int                           particle_coordinate_index_per_image = 0;
    int                           parent_asset_id                     = 0;
    ArrayOfParticlePositionAssets particle_assets_by_parent;
    ParticlePositionAsset         particle_asset;
    ProgressBar*                  my_progress = new ProgressBar(input_star_file.cached_parameters.size());
    for ( int particle_counter = 0; particle_counter < stack_number_of_images; particle_counter++ ) {
        //		 Count number of particles in a given micrograph to associate back to relevant particle ID.
        //		 Micrograph ID = Image ID in this script always, so I can get away with querying movie_list instead of image_list to skip some filename wrangling.
        //		 This is all necessary because Warp's processing order is very different from cisTEM's import order (and is not consistent, depending on when symlinks are written).
        if ( movie_name != input_star_file.ReturnMicrographName(particle_counter) ) {
            movie_name                          = input_star_file.ReturnMicrographName(particle_counter);
            particle_coordinate_index_per_image = 0;
            parent_asset_id                     = movie_list.FindFile(movie_name, true, -1) + 1;
            particle_assets_by_parent           = database.ReturnArrayOfParticlePositionAssetsFromAssetsTable(parent_asset_id);
        }
        else
            particle_coordinate_index_per_image += 1;
        particle_asset                                         = particle_assets_by_parent[particle_coordinate_index_per_image];
        temp_particle_info.parent_image_id                     = parent_asset_id;
        temp_particle_info.original_particle_position_asset_id = particle_asset.asset_id;
        temp_particle_info.x_pos                               = particle_asset.x_position;
        temp_particle_info.y_pos                               = particle_asset.y_position;
        temp_particle_info.position_in_stack                   = input_star_file.ReturnPositionInStack(particle_counter);
        temp_particle_info.defocus_1                           = input_star_file.ReturnDefocus1(particle_counter);
        temp_particle_info.defocus_2                           = input_star_file.ReturnDefocus2(particle_counter);
        temp_particle_info.defocus_angle                       = input_star_file.ReturnDefocusAngle(particle_counter);
        temp_particle_info.phase_shift                         = input_star_file.ReturnPhaseShift(particle_counter);

        refinement_package->contained_particles.push_back(temp_particle_info);

        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].position_in_stack = input_star_file.ReturnPositionInStack(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus1          = input_star_file.ReturnDefocus1(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus2          = input_star_file.ReturnDefocus2(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].defocus_angle     = input_star_file.ReturnDefocusAngle(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phase_shift       = input_star_file.ReturnPhaseShift(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].logp              = 0;

        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].occupancy       = 100.0;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].phi             = input_star_file.ReturnPhi(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].theta           = input_star_file.ReturnTheta(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].psi             = input_star_file.ReturnPsi(particle_counter);
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].xshift          = -input_star_file.ReturnXShift(particle_counter) * pixel_size;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].yshift          = -input_star_file.ReturnYShift(particle_counter) * pixel_size;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].score           = 0.0;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].image_is_active = 1;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].sigma           = 10.0;

        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].pixel_size                         = pixel_size;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_voltage_kv              = voltage;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].microscope_spherical_aberration_mm = spherical_aberration;
        refinement.class_refinement_results[0].particle_refinement_results[particle_counter].amplitude_contrast                 = amplitude_contrast;
        my_progress->Update(particle_counter + 1);
    }
    delete my_progress;
    return refinement_package;
}

ArrayofClassifications WarpToCistemApp::LoadClassificationsFromLive2D(std::string live_2d_path, std::string latest_settings_filename) {
    ArrayofClassifications classification_list;
    std::ifstream          latest_settings(latest_settings_filename.c_str());
    if ( ! latest_settings.is_open( ) )
        SendErrorAndCrash("Couldn't open " + latest_settings_filename);
    json root = json::parse(latest_settings, nullptr, false);
    if ( root.is_discarded( ) )
        SendErrorAndCrash("Couldn't parse " + latest_settings_filename + " as JSON");
    json   cycles = root.is_object( ) && root.contains("cycles") && root["cycles"].is_array( ) ? root["cycles"] : json::array( );
    long   particle_count;
    long   number;
    long   class_count;
    double high_res_lim;
    double mask_radius;
    int    cycle_count                 = int(cycles.size( ));
    int    parent_id                   = -1;
    int    refinement_package_asset_id = 1;
    float  low_resolution_limit        = 300.0; // Currently hardcoded in live2d
    float  angular_search_step         = 15.0; // Currently hardcoded in live2d
    float  search_range                = 49.5; // Currently hardcoded in live2d
    float  smoothing_factor            = 1.0; // Currently hardcoded in live2d
    bool   include_blank_edges         = true; // Currently hardcoded in live2d
    bool   auto_percent_used           = true; // Currently hardcoded in live2d
    // iterate over classifications
    ProgressBar* my_progress = new ProgressBar(cycle_count);
    for ( int i = 0; i < cycle_count; i++ ) {
        Classification*  classification = new Classification( );
        cisTEMParameters parameters;
        json             cycle = cycles[i];
        if ( ! JsonToLong(cycle["particle_count"], particle_count) )
            SendErrorAndCrash("Couldn't read particle_count of a Live2D cycle");
        classification->SizeAndFillWithEmpty(particle_count);
        if ( ! JsonToLong(cycle["number"], number) )
            SendErrorAndCrash("Couldn't read number of a Live2D cycle");
        classification->classification_id                        = number + 1; // Start from 1, not 0.
        classification->refinement_package_asset_id              = refinement_package_asset_id;
        classification->name                                     = Format("Live2D Cycle #%ld", number);
        classification->class_average_file                       = live_2d_path + JsonToString(cycle["name"]) + ".mrc";
        classification->classification_was_imported_or_generated = true;
        classification->datetime_of_run                          = DateTime::Now( );
        classification->starting_classification_id               = parent_id;
        classification->number_of_particles                      = particle_count;
        if ( ! JsonToLong(cycle["settings"]["class_number"], class_count) )
            SendErrorAndCrash("Couldn't read settings.class_number of a Live2D cycle");
        classification->number_of_classes    = class_count;
        classification->low_resolution_limit = low_resolution_limit;
        if ( ! JsonToDouble(cycle["high_res_limit"], high_res_lim) )
            SendErrorAndCrash("Couldn't read high_res_limit of a Live2D cycle");
        classification->high_resolution_limit = high_res_lim;
        if ( ! JsonToDouble(cycle["settings"]["mask_radius"], mask_radius) )
            SendErrorAndCrash("Couldn't read settings.mask_radius of a Live2D cycle");
        classification->mask_radius         = mask_radius;
        classification->angular_search_step = angular_search_step;
        classification->search_range_x      = search_range;
        classification->search_range_y      = search_range;
        classification->smoothing_factor    = smoothing_factor;
        classification->auto_percent_used   = auto_percent_used;
        classification->percent_used        = 100; // This isn't getting reported in the json file... yet. I made the change, but it hasn't gone live.
        // classification->percent_used = cycle["fraction_used"].AsDouble() * 100;

        parent_id = classification->classification_id; // Live2d classes are linear - no branching allowed.

        // Load in data from the class file and move it to an ArrayOfClassificationResults.
        // This may eventually be worth moving into a class method for classification... I have to imagine this will be used elsewhere...
        parameters.ReadFromcisTEMStarFile(live_2d_path + JsonToString(cycle["name"]) + ".star");
        MyDebugAssertTrue(particle_count == parameters.ReturnNumberofLines( ), "Wrong number of particles in Star File");
        for ( int line_number = 0; line_number < particle_count; line_number++ ) {
            classification->classification_results[line_number].position_in_stack                  = parameters.ReturnPositionInStack(line_number);
            classification->classification_results[line_number].psi                                = parameters.ReturnPsi(line_number);
            classification->classification_results[line_number].xshift                             = parameters.ReturnXShift(line_number);
            classification->classification_results[line_number].yshift                             = parameters.ReturnYShift(line_number);
            classification->classification_results[line_number].best_class                         = parameters.ReturnBest2DClass(line_number);
            classification->classification_results[line_number].sigma                              = parameters.ReturnSigma(line_number);
            classification->classification_results[line_number].logp                               = parameters.ReturnLogP(line_number);
            classification->classification_results[line_number].defocus_1                          = parameters.ReturnDefocus1(line_number);
            classification->classification_results[line_number].defocus_2                          = parameters.ReturnDefocus2(line_number);
            classification->classification_results[line_number].defocus_angle                      = parameters.ReturnDefocusAngle(line_number);
            classification->classification_results[line_number].phase_shift                        = parameters.ReturnPhaseShift(line_number);
            classification->classification_results[line_number].beam_tilt_x                        = parameters.ReturnBeamTiltX(line_number);
            classification->classification_results[line_number].beam_tilt_y                        = parameters.ReturnBeamTiltY(line_number);
            classification->classification_results[line_number].image_shift_x                      = parameters.ReturnImageShiftX(line_number);
            classification->classification_results[line_number].image_shift_y                      = parameters.ReturnImageShiftY(line_number);
            classification->classification_results[line_number].pixel_size                         = parameters.ReturnPixelSize(line_number);
            classification->classification_results[line_number].microscope_voltage_kv              = parameters.ReturnMicroscopekV(line_number);
            classification->classification_results[line_number].microscope_spherical_aberration_mm = parameters.ReturnMicroscopeCs(line_number);
            classification->classification_results[line_number].amplitude_contrast                 = parameters.ReturnAmplitudeContrast(line_number);
        }
        my_progress->Update(i + 1);
        classification_list.push_back(*classification);
        delete classification;
    }
    delete my_progress;
    return classification_list;
}

bool WarpToCistemApp::DoCalculation( ) {
    std::string warp_directory                   = my_current_job.arguments[0].ReturnStringArgument( );
    std::string cistem_parent_directory          = my_current_job.arguments[1].ReturnStringArgument( );
    std::string project_name                     = my_current_job.arguments[2].ReturnStringArgument( );
    bool     do_import_images                 = my_current_job.arguments[3].ReturnBoolArgument( );
    bool     do_scale_images_and_make_spectra = my_current_job.arguments[4].ReturnBoolArgument( );
    float    wanted_binned_pixel_size         = my_current_job.arguments[5].ReturnFloatArgument( );
    bool     do_import_ctf_results            = my_current_job.arguments[6].ReturnBoolArgument( );
    bool     do_import_particle_coordinates   = my_current_job.arguments[7].ReturnBoolArgument( );
    bool     do_import_refinement_package     = my_current_job.arguments[8].ReturnBoolArgument( );
    std::string live_2d_directory                = my_current_job.arguments[9].ReturnStringArgument( );
    float    particle_mass                    = my_current_job.arguments[10].ReturnFloatArgument( );
    bool     do_import_classification_results = my_current_job.arguments[11].ReturnBoolArgument( );

    ProgressBar* my_progress;
    long         counter;

    Printf("\nGenerating New cisTEM Project...\n\n");

    if ( EndsWith(warp_directory, "/") == false )
        warp_directory += "/";
    if ( live_2d_directory != "" && EndsWith(live_2d_directory, "/") == false )
        live_2d_directory += "/";

    if ( EndsWith(cistem_parent_directory, "/") == false )
        cistem_parent_directory += "/";
    std::string wanted_folder_name = cistem_parent_directory + project_name;
    if ( PathExists(wanted_folder_name) ) {
        SendErrorAndCrash("Database directory should not already exist, and does!\n");
    }
    else
        MakeDirectory(wanted_folder_name);
    std::filesystem::path wanted_database_file = wanted_folder_name + "/" + project_name + ".db";

    Project new_project = Project( );
    new_project.CreateNewProject(wanted_database_file, wanted_folder_name, project_name);
    Printf(wanted_database_file.string() + "\n");
    Printf("\nSuccessfully made project database\n\n");

    Printf("\nImporting files from Warp...\n\n");
    std::string      boxnet_name                           = "";
    double        warp_picking_radius                   = 0.0;
    double        warp_picking_threshold                = 0.0;
    double        warp_minimum_distance_from_exclusions = 0.0;
    std::string      warp_settings_file                    = warp_directory + "previous.settings";
    pugi::xml_document settings_doc;
    if ( PathExists(warp_settings_file) && settings_doc.load_file(warp_settings_file.c_str()) && settings_doc.document_element( ) ) {
        GetSettingsFromWarp(settings_doc, boxnet_name, warp_picking_radius, warp_picking_threshold, warp_minimum_distance_from_exclusions);
    }
    else
        SendErrorAndCrash("Couldn't load warp settings");

    std::vector<std::string> all_files;
    for ( const char* wanted_extension : {"*.mrc", "*.mrcs", "*.tif"} ) {
        std::vector<std::string> matching_files = ReturnAllFilesInDirectory(warp_directory, wanted_extension);
        all_files.insert(all_files.end( ), matching_files.begin( ), matching_files.end( ));
    }
    std::sort(all_files.begin( ), all_files.end( ));
    pugi::xml_document            doc;
    size_t                        number_of_files = all_files.size();
    std::string                      xml_filename;
    std::string                      image_filename;
    std::string                      star_filename;
    MovieAssetList                movie_list = MovieAssetList( );
    MovieAsset                    new_movie_asset;
    ImageAssetList                image_list = ImageAssetList( );
    ArrayOfParticlePositionAssets particle_list;
    ArrayOfParticlePositionAssets temp_particle_list;
    ImageAsset                    new_image_asset;
    std::vector<CTF>              ctf_list;
    CTF                           new_ctf_asset;
    int                           starting_id = 0;

    if ( all_files.empty() == true ) {
        SendErrorAndCrash("No movies were detected in the warp directory.");
    }
    my_progress = new ProgressBar(number_of_files);

    // Load filenames into memory first, then we'll do batch database inserts later.
    for ( counter = 0; counter < all_files.size(); counter++ ) {
        size_t split_point = all_files[counter].find_last_of(".");
        xml_filename       = all_files[counter];
        xml_filename       = xml_filename.substr(0, split_point);
        xml_filename += ".xml";
        // Check if warp xml exists before trying to do any more inserts
        if ( PathExists(xml_filename) && doc.load_file(xml_filename.c_str()) && doc.document_element( ) ) {
            new_movie_asset = LoadMovieFromWarp(doc, warp_directory, all_files[counter], counter, wanted_binned_pixel_size);
            if ( new_movie_asset.is_valid ) {
                movie_list.AddAsset(&new_movie_asset);

                image_filename = warp_directory + "average/" + std::filesystem::path(xml_filename).stem( ).string( ) + ".mrc";
                if ( do_import_images && image_filename.empty( ) == false && FileExists(image_filename) == true ) {
                    new_image_asset = LoadImageFromWarp(image_filename, new_movie_asset.asset_id, new_movie_asset.microscope_voltage, new_movie_asset.spherical_aberration, new_movie_asset.protein_is_white);
                    // Need to do the ctf first to adjust the ctf_estimation_id if necessary
                    if ( do_import_ctf_results ) {
                        new_image_asset.ctf_estimation_id = new_image_asset.asset_id;
                        std::string wanted_avrot_filename    = wanted_folder_name + Format("/Assets/CTF/%s_CTF_0_avrot.txt", new_image_asset.asset_name);

                        CTF new_ctf_asset = LoadCTFFromWarp(doc, new_image_asset.pixel_size, new_movie_asset.microscope_voltage, new_movie_asset.spherical_aberration, wanted_avrot_filename);
                        ctf_list.push_back(new_ctf_asset);
                    }
                    if ( do_import_particle_coordinates ) {
                        star_filename = warp_directory + "matching/" + new_image_asset.asset_name + "_" + boxnet_name + ".star";
                        if ( PathExists(star_filename) ) {
                            temp_particle_list = LoadParticlePositionsFromWarp(star_filename, new_image_asset, starting_id);
                            particle_list.insert(particle_list.end( ), temp_particle_list.begin( ), temp_particle_list.end( ));
                            starting_id += temp_particle_list.size();
                        }
                    }
                    image_list.AddAsset(&new_image_asset);
                }
                else if ( do_import_images )
                    Printf("Couldn't find averaged image: %s\n", image_filename);
            }
            else
                Printf("Corrupt warp xml output for movie " + all_files[counter] + "\n");
        }
        else
            Printf("Couldn't find a valid warp xml output for movie " + all_files[counter] + "\n");
        my_progress->Update(counter + 1);
    }

    // DB write operations and moving/writing files.

    delete my_progress;
    Printf("\nSuccessfully imported files\n\n");

    Printf("\nWriting movies to database\n\n");
    new_project.database.Begin( );
    my_progress = new ProgressBar(movie_list.number_of_assets);
    new_project.database.BeginMovieAssetInsert( );
    for ( counter = 0; counter < movie_list.number_of_assets; counter++ ) {
        new_movie_asset = reinterpret_cast<MovieAsset*>(movie_list.assets)[counter];
        // NOTE: EER not yet supported... setting eer_frames_per_image to 0
        new_project.database.AddNextMovieAsset(new_movie_asset.asset_id, new_movie_asset.asset_name, new_movie_asset.filename.string(), 1, new_movie_asset.x_size, new_movie_asset.y_size, new_movie_asset.number_of_frames, new_movie_asset.microscope_voltage, new_movie_asset.pixel_size, new_movie_asset.dose_per_frame, new_movie_asset.spherical_aberration, new_movie_asset.gain_filename, new_movie_asset.dark_filename, new_movie_asset.output_binning_factor, new_movie_asset.correct_mag_distortion, new_movie_asset.mag_distortion_angle, new_movie_asset.mag_distortion_major_scale, new_movie_asset.mag_distortion_minor_scale, new_movie_asset.protein_is_white, 1, 0);
        my_progress->Update(counter + 1);
    }
    new_project.database.EndMovieAssetInsert( );
    new_project.database.Commit( );
    delete my_progress;

    Printf("\nSuccessfully imported movies\n\n");

    if ( do_import_images ) {
        Printf("\nWriting motion-corrected images to database and writing scaled images and spectra.\n\n");
        DateTime now = DateTime::Now( );
        new_project.database.Begin( );
        my_progress = new ProgressBar(image_list.number_of_assets);
        new_project.database.BeginImageAssetInsert( );
        for ( counter = 0; counter < image_list.number_of_assets; counter++ ) {
            new_image_asset = reinterpret_cast<ImageAsset*>(image_list.assets)[counter];
            new_project.database.AddNextImageAsset(new_image_asset.asset_id, new_image_asset.asset_name, new_image_asset.filename.string(), new_image_asset.position_in_stack, new_image_asset.parent_id, new_image_asset.alignment_id, new_image_asset.ctf_estimation_id, new_image_asset.x_size, new_image_asset.y_size, new_image_asset.microscope_voltage, new_image_asset.pixel_size, new_image_asset.spherical_aberration, new_image_asset.protein_is_white);
            my_progress->Update(counter + 1);
        }
        new_project.database.EndImageAssetInsert( );
        delete my_progress;

        Printf("\nDone with database insert of images\n\n");

        Printf("\nInserting Image Alignment Jobs into Database\n\n");
        my_progress = new ProgressBar(image_list.number_of_assets);
        new_project.database.BeginBatchInsert("MOVIE_ALIGNMENT_LIST", 22, "ALIGNMENT_ID", "DATETIME_OF_RUN", "ALIGNMENT_JOB_ID", "MOVIE_ASSET_ID", "OUTPUT_FILE", "VOLTAGE", "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "PRE_EXPOSURE_AMOUNT", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER", "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS", "HORIZONTAL_MASK", "VERTICAL_MASK", "SHOULD_INCLUDE_ALL_FRAMES_IN_SUM", "FIRST_FRAME_TO_SUM", "LAST_FRAME_TO_SUM");
        for ( counter = 0; counter < image_list.number_of_assets; counter++ ) {
            new_image_asset = reinterpret_cast<ImageAsset*>(image_list.assets)[counter];
            new_project.database.AddToBatchInsert("iliitrrrrrriiriiiiiiii", new_image_asset.alignment_id,
                                                  (long int)now.GetAsDOS( ),
                                                  1, //alignment_job_id - set to 1
                                                  new_image_asset.asset_id,
                                                  new_image_asset.filename.string().c_str( ),
                                                  new_image_asset.microscope_voltage,
                                                  new_image_asset.pixel_size,
                                                  0.0, // exposure per frame
                                                  0.0, //current_pre_exposure
                                                  0.0, // min shift
                                                  0.0, // max shift
                                                  1, // should dose filter
                                                  1, // should restore power
                                                  1.0, // termination threshold
                                                  20, // max iterations
                                                  1500, //b factor
                                                  1, //mask central cross
                                                  1, // horizontal mask
                                                  1, // vertical cross
                                                  1, // include all frames
                                                  1, //first frame
                                                  0); //last frames
            my_progress->Update(counter + 1);
        }
        new_project.database.EndBatchInsert( );
        delete my_progress;

        my_progress = new ProgressBar(image_list.number_of_assets);
        std::string current_table_name;
        long     frame_counter;
        for ( counter = 0; counter < image_list.number_of_assets; counter++ ) {
            new_image_asset    = reinterpret_cast<ImageAsset*>(image_list.assets)[counter];
            current_table_name = Format("MOVIE_ALIGNMENT_PARAMETERS_%i", new_image_asset.alignment_id);
            new_project.database.CreateTable(current_table_name, "prr", "FRAME_NUMBER", "X_SHIFT", "Y_SHIFT");
            new_project.database.BeginBatchInsert(current_table_name, 3, "FRAME_NUMBER", "X_SHIFT", "Y_SHIFT");
            for ( frame_counter = 0; frame_counter < 40; frame_counter++ ) // 5 is totally arbitrary
            {
                new_project.database.AddToBatchInsert("irr", frame_counter + 1, 0.0, 0.0);
            }
            new_project.database.EndBatchInsert( );
            my_progress->Update(counter + 1);
        }
        delete my_progress;
        new_project.database.Commit( );
        Printf("\nDone with Image Alignment Jobs\n\n");
    }

    if ( do_scale_images_and_make_spectra ) {
        Printf("\nPreparing scaled images and spectra\n\n");
        my_progress = new ProgressBar(image_list.number_of_assets);
        Image large_image;
        Image buffer_image;
        float average;
        float sigma;
        for ( counter = 0; counter < image_list.number_of_assets; counter++ ) {
            new_image_asset = reinterpret_cast<ImageAsset*>(image_list.assets)[counter];
            //Spectrum
            std::string wanted_spectrum_filename  = wanted_folder_name + Format("/Assets/Images/Spectra/%s.mrc", new_image_asset.asset_name); //, new_image_asset.asset_id, 0); _%i_%i
            std::string current_spectrum_filename = warp_directory + Format("powerspectrum/%s.mrc", new_image_asset.asset_name);

            // Warp spectra are halved and unthresholded, so this fixes that.
            large_image.QuickAndDirtyReadSlice(current_spectrum_filename, 1); // reusing large image and buffer_image here
            buffer_image.Allocate(large_image.logical_x_dimension, large_image.logical_y_dimension * 2, 1);
            for ( int output_address = 0; output_address < large_image.real_memory_allocated; output_address++ ) {
                buffer_image.real_values[output_address] = large_image.real_values[output_address];
            }
            int input_address = buffer_image.real_memory_allocated / 2;
            for ( int address = large_image.real_memory_allocated - 1; address >= 0; address-- ) {
                buffer_image.real_values[input_address] = large_image.real_values[address];
                input_address++;
            }

            buffer_image.CosineRingMask(0, buffer_image.logical_x_dimension / 2, 5.0f);
            buffer_image.ForwardFFT( );
            buffer_image.CosineMask(0, 0.05, true);
            buffer_image.BackwardFFT( );
            buffer_image.ComputeAverageAndSigmaOfValuesInSpectrum(float(buffer_image.logical_x_dimension) * 0.5, float(buffer_image.logical_x_dimension), average, sigma, 12);
            buffer_image.DivideByConstant(sigma);
            buffer_image.SetMaximumValueOnCentralCross(average / sigma + 10.0);

            buffer_image.ComputeAverageAndSigmaOfValuesInSpectrum(float(buffer_image.logical_x_dimension) * 0.5, float(buffer_image.logical_x_dimension), average, sigma, 12);
            buffer_image.SetMinimumAndMaximumValues(average - 15.0, average + 15.0);
            buffer_image.QuickAndDirtyWriteSlice(wanted_spectrum_filename, 1);
            // Also write fake diagnostic ctf spectrum in this scope to reuse the work done above.
            if ( do_import_ctf_results ) {
                std::string wanted_ctf_filename = wanted_folder_name + Format("/Assets/CTF/%s_CTF_0.mrc", new_image_asset.asset_name);
                buffer_image.QuickAndDirtyWriteSlice(wanted_ctf_filename, 1, true);
            }

            //Scaled Image - borrowed the logic heavily from Unblur
            std::string wanted_scaled_filename = wanted_folder_name + Format("/Assets/Images/Scaled/%s.mrc", new_image_asset.asset_name); //, new_image_asset.asset_id, 0); _%i_%i
            //			std::string current_scaled_filename = warp_directory + Format("thumbnails/%s.png");
            //			png_image = wxImage(current_scaled_filename);
            //			png_data = png_image.GetData();
            //			buffer_image.Allocate(png_image.GetWidth(), png_image.GetHeight(), true);
            //			for (int output_address = 0; output_address < large_image.real_memory_allocated; output_address++) {
            //				buffer_image.real_values[output_address] = (float)(png_data[output_address*3]);
            //			}
            //			buffer_image.AddFFTWPadding();
            int   largest_dimension = std::max(new_image_asset.x_size, new_image_asset.y_size);
            float scale_factor      = float(SCALED_IMAGE_SIZE) / float(largest_dimension);
            if ( scale_factor > 1 )
                scale_factor = 1.0;
            large_image.QuickAndDirtyReadSlice(new_image_asset.filename.string(), 1);
            large_image.ForwardFFT( );
            buffer_image.Allocate(myroundint(new_image_asset.x_size * scale_factor), myroundint(new_image_asset.y_size * scale_factor), false);
            large_image.ClipInto(&buffer_image);
            buffer_image.BackwardFFT( );
            buffer_image.QuickAndDirtyWriteSlice(wanted_scaled_filename, 1, true);

            my_progress->Update(counter + 1);
        }
        delete my_progress;
        Printf("\nDone writing spectra and scaled images\n\n");
    }

    if ( do_import_ctf_results ) {
        Printf("\nInserting CTF results in database\n\n");
        DateTime now = DateTime::Now( );
        new_project.database.Begin( );
        my_progress = new ProgressBar(image_list.number_of_assets);
        new_project.database.BeginBatchInsert("ESTIMATED_CTF_PARAMETERS", 32,
                                              "CTF_ESTIMATION_ID",
                                              "CTF_ESTIMATION_JOB_ID",
                                              "DATETIME_OF_RUN",
                                              "IMAGE_ASSET_ID",
                                              "ESTIMATED_ON_MOVIE_FRAMES",
                                              "VOLTAGE",
                                              "SPHERICAL_ABERRATION",
                                              "PIXEL_SIZE",
                                              "AMPLITUDE_CONTRAST",
                                              "BOX_SIZE",
                                              "MIN_RESOLUTION",
                                              "MAX_RESOLUTION",
                                              "MIN_DEFOCUS",
                                              "MAX_DEFOCUS",
                                              "DEFOCUS_STEP",
                                              "RESTRAIN_ASTIGMATISM",
                                              "TOLERATED_ASTIGMATISM",
                                              "FIND_ADDITIONAL_PHASE_SHIFT",
                                              "MIN_PHASE_SHIFT",
                                              "MAX_PHASE_SHIFT",
                                              "PHASE_SHIFT_STEP",
                                              "DEFOCUS1",
                                              "DEFOCUS2",
                                              "DEFOCUS_ANGLE",
                                              "ADDITIONAL_PHASE_SHIFT",
                                              "SCORE",
                                              "DETECTED_RING_RESOLUTION",
                                              "DETECTED_ALIAS_RESOLUTION",
                                              "OUTPUT_DIAGNOSTIC_FILE",
                                              "NUMBER_OF_FRAMES_AVERAGED",
                                              "LARGE_ASTIGMATISM_EXPECTED",
                                              "ICINESS");
        for ( counter = 0; counter < image_list.number_of_assets; counter++ ) {
            new_image_asset              = reinterpret_cast<ImageAsset*>(image_list.assets)[counter];
            CTF      new_ctf             = ctf_list.at(counter);
            std::string wanted_ctf_filename = wanted_folder_name + Format("/Assets/CTF/%s_CTF_0.mrc", new_image_asset.asset_name); // This logic is repeated before - its the only case of straight repeating code I've resorted to, but calculating it twice is easier than storing the data in an object somewhere for multiple iterations.
            new_project.database.AddToBatchInsert("iiliirrrrirrrrririrrrrrrrrrrtiir",
                                                  new_image_asset.ctf_estimation_id, // Image asset join column reference
                                                  1, // CTF Job id
                                                  (long int)now.GetAsDOS( ), // datetime
                                                  new_image_asset.asset_id, // Image asset parent
                                                  1, // Estimated on movie frames - I am not scraping this from Warp because it doesn't fit in the CTF object easily, and doesn't add enough to be worth adding to the object definition.
                                                  new_image_asset.microscope_voltage, // I could get this from the wavelength of the CTF object, but it doesn't seem necessary.
                                                  new_ctf.GetSphericalAberration( ) / 10000000.0 * new_image_asset.pixel_size, // convert back to mm
                                                  new_image_asset.pixel_size,
                                                  new_ctf.GetAmplitudeContrast( ),
                                                  512, // box size - I think this is the box size of the spectrum and am filling on that assumption.
                                                  new_image_asset.pixel_size / new_ctf.GetLowestFrequencyForFitting( ), // Invert the frequency
                                                  new_image_asset.pixel_size / new_ctf.GetHighestFrequencyForFitting( ), // Invert the frequency
                                                  10000 * 0.0, // Min defocus - I am not scraping this from warp (same as ESTIMATED_ON_MOVIE_FRAMES) but these are the defaults as of 190815
                                                  10000 * 2.0, // Max defocus - I am not scraping this from warp (same as above).
                                                  -1, // Defocus Step is not reported by warp anywhere... this is negative for safety.
                                                  0, // Restrain Astigmatism is not selectable by warp.
                                                  -1, //Tolerated Astigmatism is not selectable by warp.
                                                  0, // Not scraping find additional phase shift - See above, I don't want to add all this to CTF unless I have to.
                                                  0.0, // Min Phase
                                                  0.0, // Max Phase
                                                  0.0, // Phase shift step
                                                  new_ctf.GetDefocus1( ) * new_image_asset.pixel_size,
                                                  new_ctf.GetDefocus2( ) * new_image_asset.pixel_size,
                                                  new_ctf.GetAstigmatismAzimuth( ) * 180 / PI, // Convert back to degrees
                                                  new_ctf.GetAdditionalPhaseShift( ),
                                                  -1.0, // Score
                                                  new_image_asset.pixel_size / new_ctf.GetHighestFrequencyWithGoodFit( ), //Resolution of fit.
                                                  0.0, // Alias Resolution
                                                  wanted_ctf_filename.c_str( ),
                                                  -1, // Number of frames averaged
                                                  0, // Large Astigmatism Expected is not used by warp
                                                  -1.0); // Iciness is not calculated by warp

            my_progress->Update(counter + 1);
        }
        delete my_progress;
        new_project.database.EndBatchInsert( );
        new_project.database.Commit( );
        Printf("\nDone with CTF database insertions");
    }

    if ( do_import_particle_coordinates ) {
        Printf("\nInserting Particle Coordinates into database\n\n");
        new_project.database.Begin( );
        new_project.database.CreateParticlePickingResultsTable(1); // hardcode 1
        new_project.database.AddArrayOfParticlePositionAssetsToResultsTable(1, &particle_list);
        new_project.database.AddArrayOfParticlePositionAssetsToAssetsTable(&particle_list);
        Printf("\nDone with Particle Coordinates insert\n\n");

        Printf("\nInserting Particle Picking Job Metadata into database\n\n");
        new_project.database.BeginBatchInsert("PARTICLE_PICKING_LIST", 14,
                                              "PICKING_ID",
                                              "PICKING_JOB_ID",
                                              "DATETIME_OF_RUN",
                                              "PARENT_IMAGE_ASSET_ID",
                                              "PICKING_ALGORITHM",
                                              "CHARACTERISTIC_RADIUS",
                                              "MAXIMUM_RADIUS",
                                              "THRESHOLD_PEAK_HEIGHT",
                                              "HIGHEST_RESOLUTION_USED_IN_PICKING",
                                              "MIN_DIST_FROM_EDGES",
                                              "AVOID_HIGH_VARIANCE",
                                              "AVOID_HIGH_LOW_MEAN",
                                              "NUM_BACKGROUND_BOXES",
                                              "MANUAL_EDIT");
        DateTime now = DateTime::Now( );
        my_progress    = new ProgressBar(image_list.number_of_assets);
        for ( counter = 0; counter < image_list.number_of_assets; counter++ ) {
            new_image_asset = reinterpret_cast<ImageAsset*>(image_list.assets)[counter];
            new_project.database.AddToBatchInsert("iiliirrrriiiii", new_image_asset.asset_id,
                                                  1,
                                                  (long int)now.GetAsDOS( ),
                                                  new_image_asset.asset_id,
                                                  -1, // Algorithm
                                                  warp_picking_radius,
                                                  warp_picking_radius, // Only one radius is used by warp
                                                  warp_picking_threshold, // This is actually a FOM threshold, not peak height, but its roughly parallel
                                                  -1, // No resolution Filter
                                                  warp_minimum_distance_from_exclusions,
                                                  0,
                                                  0,
                                                  -1, // Background boxes is nonsense here.
                                                  0);
            my_progress->Update(counter + 1);
        }
        new_project.database.EndBatchInsert( );
        delete my_progress;
        new_project.database.Commit( );
        Printf("\nDone with Particle Coordinate Metadata insert\n\n");
    }

    Printf("\nDone with database operations for Warp import\n\n");

    if ( do_import_refinement_package ) {
        Printf("\nImporting Refinement Package from Live2D\n\n");
        double             pixel_size           = new_image_asset.pixel_size;
        double             voltage              = new_image_asset.microscope_voltage;
        double             spherical_aberration = new_image_asset.spherical_aberration;
        double             amplitude_contrast   = ctf_list.at(0).GetAmplitudeContrast( );
        std::string           star_filename        = warp_directory + "allparticles_" + boxnet_name + ".star";
        std::string           stack_filename       = live_2d_directory + "combined_stack.mrcs";
        Refinement         refinement;
        RefinementPackage* refinement_package = LoadRefinementPackageFromLive2D(star_filename, stack_filename, pixel_size, particle_mass, voltage, spherical_aberration, amplitude_contrast, warp_picking_radius, new_project.database, movie_list, refinement);
        new_project.database.Begin( );
        Printf("\nRefinement Package Loading Complete\nInserting Into Database\n\n");
        new_project.database.AddRefinementPackageAsset(refinement_package);
        new_project.database.AddRefinement(&refinement);
        ArrayofAngularDistributionHistograms all_histograms;
        all_histograms = refinement.ReturnAngularDistributions(refinement_package->symmetry);
        for ( int class_counter = 0; class_counter < refinement.number_of_classes; class_counter++ ) {
            new_project.database.AddRefinementAngularDistribution(all_histograms[class_counter], refinement.refinement_id, class_counter + 1);
        }
        new_project.database.Commit( );
        Printf("\nDone Inserting Refinement Package\n\n");
    }

    if ( do_import_classification_results ) {
        Printf("\nImporting Classification Results from Live2D\n\n");
        std::string               latest_settings_filename = live_2d_directory + "latest_run.json";
        ArrayofClassifications classification_list      = LoadClassificationsFromLive2D(live_2d_directory, latest_settings_filename);
        Classification*        classification;
        Printf("\nDone Importing Classification Results\nInserting Into Database\n\n");
        new_project.database.Begin( );
        my_progress = new ProgressBar(classification_list.size());
        for ( counter = 0; counter < classification_list.size(); counter++ ) {
            classification = &classification_list[counter];
            new_project.database.AddClassification(classification);
            long refinement_package_id = 1; // hardcoded above.
            long classification_id     = classification->classification_id;
            new_project.database.ExecuteSQL(Format("INSERT INTO REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li (CLASSIFICATION_NUMBER, CLASSIFICATION_ID) VALUES (%li, %li);", refinement_package_id, classification_id, classification_id));
            my_progress->Update(counter + 1);
        }
        delete my_progress;
        new_project.database.Commit( );
        Printf("\nDone Inserting Classification Results\n\n");
    }

    Printf("\ncisTEM project ready to be loaded by GUI.\n");
    return true;
}
