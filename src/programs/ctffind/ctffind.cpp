#include "../../core/core_headers.h"
#include "./ctffind.h"

//#define threshold_spectrum

// The timing that ctffind originally tracks is always on, by direct reference to cistem_timer::StopWatch
// The profiling for development is under conrtol of --enable-profiling.
#ifdef PROFILING
using namespace cistem_timer;
#else
using namespace cistem_timer_noop;
#endif

const std::string ctffind_version = "4.1.14";

/*
 * Changelog
 * - 4.1.14
 * -- bug fixes (memory, equiphase averaging)
 * -- bug fixes from David Mastronarde (fixed/known phase shift)
 * - 4.1.13
 * -- EPA bug fixed (affected 4.1.12 only)
 * - 4.1.12
 * -- diagnostic image includes radial "equi-phase" average of experimental spectrum in bottom right quadrant
 * -- new "equi-phase averaging" code will probably replace zero-counting eventually
 * -- bug fix (affected diagnostics for all 4.x): at very high resolution, Cs dominates and the phase aberration decreases
 * -- bug fix (affected 4.1.11): fitting was only done up to 5Å
 * -- slow, exhaustive search is no longer the default (since astigmatism-related bugs appear fixed)
 * -- Number of OpenMP threads defaults to 1 and can be set by:
 * --- using interactive user input (under expert options)
 * --- using the -j command-line option (overrides interactive user input)
 * -- printout timing information
 * - 4.1.11
 * -- speed-ups from David Mastronarde, including OpenMP threading of the exhaustive search
 * -- score is now a normalized cross-correlation coefficient (David Mastronarde)
 * - 4.1.10
 * -- astimatism-related bug fixes from David Mastronarde
 * - 4.1.9
 * -- FRC is between a re-normalized version of the amplitude spectrum, to emphasize phase of the Thon rings over their relative amplitudes
 * -- tweaked criteria for "fit resolution"
 * -- tweaked FRC computation
 * -- astigmatism restraint is off by default
 * -- fixed bug affecting astigmatism 
 * -- tweaked background subtraction (thanks Niko!) - helps with noisy VPP spectra
 */

class
        CtffindApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void AddCommandLineOptions( );

  private:
};

float PixelSizeForFitting(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                          int box_size, Image* current_power_spectrum, Image* resampled_power_spectrum, bool do_resampling, float stretch_factor) {
    int   temporary_box_size;
    int   stretched_dimension;
    float pixel_size_for_fitting;
    bool  resampling_is_necessary;

    Image temp_image;

    // Resample the amplitude spectrum
    if ( resample_if_pixel_too_small && pixel_size_of_input_image < target_pixel_size_after_resampling ) {
        // The input pixel was too small, so let's resample the amplitude spectrum into a large temporary box, before clipping the center out for fitting
        temporary_box_size = round(float(box_size) / pixel_size_of_input_image * target_pixel_size_after_resampling);
        if ( IsOdd(temporary_box_size) )
            temporary_box_size++;
        resampling_is_necessary = current_power_spectrum->logical_x_dimension != box_size || current_power_spectrum->logical_y_dimension != box_size;
        if ( do_resampling ) {
            if ( resampling_is_necessary || stretch_factor != 1.0f ) {
                stretched_dimension = myroundint(temporary_box_size * stretch_factor);
                if ( IsOdd(stretched_dimension) )
                    stretched_dimension++;
                if ( fabsf(stretched_dimension - temporary_box_size * stretch_factor) > fabsf(stretched_dimension - 2 - temporary_box_size * stretch_factor) )
                    stretched_dimension -= 2;

                current_power_spectrum->ForwardFFT(false);
                resampled_power_spectrum->Allocate(stretched_dimension, stretched_dimension, 1, false);
                current_power_spectrum->ClipInto(resampled_power_spectrum);
                resampled_power_spectrum->BackwardFFT( );
                temp_image.Allocate(box_size, box_size, 1, true);
                temp_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
                resampled_power_spectrum->ClipInto(&temp_image);
                resampled_power_spectrum->Consume(&temp_image);
            }
            else {
                resampled_power_spectrum->CopyFrom(current_power_spectrum);
            }
        }
        pixel_size_for_fitting = pixel_size_of_input_image * float(temporary_box_size) / float(box_size);
    }
    else {
        // The regular way (the input pixel size was large enough)
        resampling_is_necessary = current_power_spectrum->logical_x_dimension != box_size || current_power_spectrum->logical_y_dimension != box_size;
        if ( do_resampling ) {
            if ( resampling_is_necessary || stretch_factor != 1.0f ) {
                stretched_dimension = myroundint(box_size * stretch_factor);
                if ( IsOdd(stretched_dimension) )
                    stretched_dimension++;
                if ( fabsf(stretched_dimension - box_size * stretch_factor) > fabsf(stretched_dimension - 2 - box_size * stretch_factor) )
                    stretched_dimension -= 2;

                current_power_spectrum->ForwardFFT(false);
                resampled_power_spectrum->Allocate(stretched_dimension, stretched_dimension, 1, false);
                current_power_spectrum->ClipInto(resampled_power_spectrum);
                resampled_power_spectrum->BackwardFFT( );
                temp_image.Allocate(box_size, box_size, 1, true);
                temp_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
                resampled_power_spectrum->ClipInto(&temp_image);
                resampled_power_spectrum->Consume(&temp_image);
            }
            else {
                resampled_power_spectrum->CopyFrom(current_power_spectrum);
            }
        }
        pixel_size_for_fitting = pixel_size_of_input_image;
    }

    return pixel_size_for_fitting;
}

//#pragma GCC pop_options

IMPLEMENT_APP(CtffindApp)

// override the DoInteractiveUserInput

void CtffindApp::DoInteractiveUserInput( ) {

    float lowest_allowed_minimum_resolution = 50.0;

    std::string input_filename                     = "/dev/null";
    bool        input_is_a_movie                   = false;
    int         number_of_frames_to_average        = 0;
    std::string output_diagnostic_filename         = "/dev/null";
    float       pixel_size                         = 0.0;
    float       acceleration_voltage               = 0.0;
    float       spherical_aberration               = 0.0;
    float       amplitude_contrast                 = 0.0;
    int         box_size                           = 0;
    float       minimum_resolution                 = 0.0;
    float       maximum_resolution                 = 0.0;
    float       minimum_defocus                    = 0.0;
    float       maximum_defocus                    = 0.0;
    float       defocus_search_step                = 0.0;
    bool        astigmatism_is_known               = false;
    float       known_astigmatism                  = 0.0;
    float       known_astigmatism_angle            = 0.0;
    bool        slower_search                      = false;
    bool        should_restrain_astigmatism        = false;
    float       astigmatism_tolerance              = 0.0;
    bool        find_additional_phase_shift        = false;
    bool        determine_tilt                     = false;
    float       minimum_additional_phase_shift     = 0.0;
    float       maximum_additional_phase_shift     = 0.0;
    float       additional_phase_shift_search_step = 0.0;
    bool        give_expert_options                = false;
    bool        resample_if_pixel_too_small        = false;
    bool        movie_is_gain_corrected            = false;
    bool        movie_is_dark_corrected;
    wxString    dark_filename;
    wxString    gain_filename                    = "/dev/null";
    bool        correct_movie_mag_distortion     = false;
    float       movie_mag_distortion_angle       = 0.0;
    float       movie_mag_distortion_major_scale = 1.0;
    float       movie_mag_distortion_minor_scale = 1.0;
    bool        defocus_is_known                 = false;
    float       known_defocus_1                  = 0.0;
    float       known_defocus_2                  = 0.0;
    float       known_phase_shift                = 0.0;
    int         desired_number_of_threads        = 1;
    int         eer_frames_per_image             = 0;
    int         eer_super_res_factor             = 1;

    // Things we need for old school input
    double     temp_double               = -1.0;
    long       temp_long                 = -1;
    float      xmag                      = -1;
    float      dstep                     = -1.0;
    int        token_counter             = -1;
    const bool old_school_input          = command_line_parser.FoundSwitch("old-school-input");
    const bool old_school_input_ctffind4 = command_line_parser.FoundSwitch("old-school-input-ctffind4");

    if ( old_school_input || old_school_input_ctffind4 ) {

        astigmatism_is_known        = false;
        known_astigmatism           = 0.0;
        known_astigmatism_angle     = 0.0;
        resample_if_pixel_too_small = true;
        movie_is_gain_corrected     = true;
        gain_filename               = "";
        movie_is_dark_corrected     = true;
        dark_filename               = "";

        char     buf[4096];
        wxString my_string;

        // Line 1
        std::cin.getline(buf, 4096);
        input_filename = buf;

        // Line 2
        std::cin.getline(buf, 4096);
        output_diagnostic_filename = buf;

        // Line 3
        std::cin.getline(buf, 4096);
        my_string = buf;
        wxStringTokenizer tokenizer(my_string, ",");
        if ( tokenizer.CountTokens( ) != 5 ) {
            SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 3 of input\n", tokenizer.CountTokens( ), 5));
            exit(-1);
        }
        token_counter = -1;
        while ( tokenizer.HasMoreTokens( ) ) {
            token_counter++;
            tokenizer.GetNextToken( ).ToDouble(&temp_double);
            switch ( token_counter ) {
                case 0:
                    spherical_aberration = float(temp_double);
                    break;
                case 1:
                    acceleration_voltage = float(temp_double);
                    break;
                case 2:
                    amplitude_contrast = float(temp_double);
                    break;
                case 3:
                    xmag = float(temp_double);
                    break;
                case 4:
                    dstep = float(temp_double);
                    break;
                default:
                    wxPrintf("Ooops - bad token number: %li\n", tokenizer.GetPosition( ));
                    MyDebugAssertTrue(false, "oops\n");
            }
        }
        pixel_size = dstep * 10000.0 / xmag;

        // Line 4
        std::cin.getline(buf, 4096);
        my_string = buf;
        tokenizer.SetString(my_string, ",");
        if ( tokenizer.CountTokens( ) != 7 ) {
            SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 4 of input\n", tokenizer.CountTokens( ), 7));
            exit(-1);
        }
        token_counter = -1;
        while ( tokenizer.HasMoreTokens( ) ) {
            token_counter++;
            switch ( token_counter ) {
                case 0:
                    tokenizer.GetNextToken( ).ToLong(&temp_long);
                    box_size = int(temp_long);
                    break;
                case 1:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    minimum_resolution = float(temp_double);
                    break;
                case 2:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    maximum_resolution = float(temp_double);
                    break;
                case 3:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    minimum_defocus = float(temp_double);
                    break;
                case 4:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    maximum_defocus = float(temp_double);
                    break;
                case 5:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    defocus_search_step = float(temp_double);
                    break;
                case 6:
                    tokenizer.GetNextToken( ).ToDouble(&temp_double);
                    astigmatism_tolerance = float(temp_double);
                    break;
            }
        }
        // If we are getting dAst = 0.0, which is the default in Relion, the user probably
        // expects the ctffind3 behaviour, which is no restraint on astigmatism
        if ( astigmatism_tolerance == 0.0 )
            astigmatism_tolerance = -100.0;

        // Output for old-school users
        if ( is_running_locally ) {
            wxPrintf("\n CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]\n");
            wxPrintf("%5.1f%9.1f%8.2f%10.1f%9.3f\n\n", spherical_aberration, acceleration_voltage, amplitude_contrast, xmag, dstep);
        }

        // Extra lines of input
        if ( old_school_input_ctffind4 ) {
            // Line 5
            std::cin.getline(buf, 4096);
            my_string = buf;
            tokenizer.SetString(my_string, ",");
            if ( tokenizer.CountTokens( ) != 2 ) {
                SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 5 of input\n", tokenizer.CountTokens( ), 2));
                exit(-1);
            }
            while ( tokenizer.HasMoreTokens( ) ) {
                switch ( tokenizer.GetPosition( ) ) {
                    case 0:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        if ( int(temp_double) != 0 ) {
                            input_is_a_movie = true;
                        }
                        else {
                            input_is_a_movie = false;
                        }
                        break;
                    case 1:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        number_of_frames_to_average = 1;
                        if ( input_is_a_movie ) {
                            number_of_frames_to_average = int(temp_double);
                        }
                        break;
                }
            }

            // Line 6
            std::cin.getline(buf, 4096);
            my_string = buf;
            tokenizer.SetString(my_string, ",");
            if ( tokenizer.CountTokens( ) != 4 ) {
                SendError(wxString::Format("Bad number of arguments (%i, expected %i) in line 6 of input\n", tokenizer.CountTokens( ), 4));
                exit(-1);
            }
            while ( tokenizer.HasMoreTokens( ) ) {
                switch ( tokenizer.GetPosition( ) ) {
                    case 0:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        if ( int(temp_double) != 0 ) {
                            find_additional_phase_shift = true;
                        }
                        else {
                            find_additional_phase_shift = false;
                        }
                        break;
                    case 1:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        minimum_additional_phase_shift = 0.0;
                        if ( find_additional_phase_shift ) {
                            minimum_additional_phase_shift = float(temp_double);
                        }
                        break;
                    case 2:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        maximum_additional_phase_shift = 0.0;
                        if ( find_additional_phase_shift ) {
                            maximum_additional_phase_shift = float(temp_double);
                        }
                        break;
                    case 3:
                        tokenizer.GetNextToken( ).ToDouble(&temp_double);
                        additional_phase_shift_search_step = 0.0;
                        if ( find_additional_phase_shift ) {
                            additional_phase_shift_search_step = float(temp_double);
                        }
                        break;
                }
            }
        } // end of old school ctffind4 input
        else {
            input_is_a_movie                   = false;
            find_additional_phase_shift        = false;
            minimum_additional_phase_shift     = 0.0;
            maximum_additional_phase_shift     = 0.0;
            additional_phase_shift_search_step = 0.0;
            number_of_frames_to_average        = 1;
        }

        // Do some argument checking on movie processing option
        MRCFile input_file(input_filename, false);
        if ( input_is_a_movie ) {
            if ( input_file.ReturnZSize( ) < number_of_frames_to_average ) {
                SendError(wxString::Format("Input stack has %i images, so you cannot average %i frames together\n", input_file.ReturnZSize( ), number_of_frames_to_average));
                ExitMainLoop( );
            }
        }
        else {
            // We're not doing movie processing
            if ( input_file.ReturnZSize( ) > 1 ) {
                SendError("Input stacks are only supported --old-school-input-ctffind4 if doing movie processing\n");
                ExitMainLoop( );
            }
        }

        if ( find_additional_phase_shift ) {
            if ( minimum_additional_phase_shift > maximum_additional_phase_shift ) {
                SendError(wxString::Format("Minimum phase shift (%f) cannot be greater than maximum phase shift (%f)\n", minimum_additional_phase_shift, maximum_additional_phase_shift));
                ExitMainLoop( );
            }
        }

        desired_number_of_threads = 1;

    } // end of test for old-school-input or old-school-input-ctffind4
    else {

        UserInput* my_input = new UserInput("Ctffind", ctffind_version);

        input_filename = my_input->GetFilenameFromUser("Input image file name", "Filename of input image", "input.mrc", true);

        ImageFile input_file;
        if ( FilenameExtensionMatches(input_filename, "eer") ) {
            // If the input file is EER format, we will find out how many EER frames to average per image only later, so for now we have no way of knowing the logical Z dimension, but we know it's a movie
            input_is_a_movie = true;
        }
        else {
            input_file.OpenFile(input_filename, false);
            if ( input_file.ReturnZSize( ) > 1 ) {
                input_is_a_movie = my_input->GetYesNoFromUser("Input is a movie (stack of frames)", "Answer yes if the input file is a stack of frames from a dose-fractionated movie. If not, each image will be processed separately", "No");
            }
            else {
                input_is_a_movie = false;
            }
        }

        if ( input_is_a_movie ) {
            number_of_frames_to_average = my_input->GetIntFromUser("Number of frames to average together", "If the number of electrons per frame is too low, there may be strong artefacts in the estimated power spectrum. This can be alleviated by averaging frames with each other in real space before computing their Fourier transforms", "1");
        }
        else {
            number_of_frames_to_average = 1;
        }

        output_diagnostic_filename = my_input->GetFilenameFromUser("Output diagnostic image file name", "Will contain the experimental power spectrum and the best CTF fit", "diagnostic_output.mrc", false);
        pixel_size                 = my_input->GetFloatFromUser("Pixel size", "In Angstroms", "1.0", 0.0);
        acceleration_voltage       = my_input->GetFloatFromUser("Acceleration voltage", "in kV", "300.0", 0.0);
        spherical_aberration       = my_input->GetFloatFromUser("Spherical aberration", "in mm", "2.70", 0.0);
        amplitude_contrast         = my_input->GetFloatFromUser("Amplitude contrast", "Fraction of amplitude contrast", "0.07", 0.0, 1.0);
        box_size                   = my_input->GetIntFromUser("Size of amplitude spectrum to compute", "in pixels", "512", 128);
        minimum_resolution         = my_input->GetFloatFromUser("Minimum resolution", "Lowest resolution used for fitting CTF (Angstroms)", "30.0", 0.0, lowest_allowed_minimum_resolution);
        maximum_resolution         = my_input->GetFloatFromUser("Maximum resolution", "Highest resolution used for fitting CTF (Angstroms)", "5.0", 0.0, minimum_resolution);
        minimum_defocus            = my_input->GetFloatFromUser("Minimum defocus", "Positive values for underfocus. Lowest value to search over (Angstroms)", "5000.0");
        maximum_defocus            = my_input->GetFloatFromUser("Maximum defocus", "Positive values for underfocus. Highest value to search over (Angstroms)", "50000.0", minimum_defocus);
        defocus_search_step        = my_input->GetFloatFromUser("Defocus search step", "Step size for defocus search (Angstroms)", "100.0", 1.0);
        astigmatism_is_known       = my_input->GetYesNoFromUser("Do you know what astigmatism is present?", "Answer yes if you already know how much astigmatism was present. If you answer no, the program will search for the astigmatism and astigmatism angle", "No");
        if ( astigmatism_is_known ) {
            slower_search = my_input->GetYesNoFromUser("Slower, more exhaustive search?", "Answer yes to use a slower exhaustive search against 2D spectra (rather than 1D radial averages) for the initial search", "No");
            ;
            should_restrain_astigmatism = false;
            astigmatism_tolerance       = -100.0;
            known_astigmatism           = my_input->GetFloatFromUser("Known astigmatism", "In Angstroms, the amount of astigmatism, defined as the difference between the defocus along the major and minor axes", "0.0", 0.0);
            known_astigmatism_angle     = my_input->GetFloatFromUser("Known astigmatism angle", "In degrees, the angle of astigmatism", "0.0");
        }
        else {
            slower_search               = my_input->GetYesNoFromUser("Slower, more exhaustive search?", "Answer yes if you expect very high astigmatism (say, greater than 1000A) or in tricky cases. In that case, a slower exhaustive search against 2D spectra (rather than 1D radial averages) will be used for the initial search", "No");
            should_restrain_astigmatism = my_input->GetYesNoFromUser("Use a restraint on astigmatism?", "If you answer yes, the CTF parameter search and refinement will penalise large astigmatism. You will specify the astigmatism tolerance in the next question. If you answer no, no such restraint will apply", "No");
            if ( should_restrain_astigmatism ) {
                astigmatism_tolerance = my_input->GetFloatFromUser("Expected (tolerated) astigmatism", "Astigmatism values much larger than this will be penalised (Angstroms). Give a negative value to turn off this restraint.", "200.0");
            }
            else {
                astigmatism_tolerance = -100.0; // a negative value here signals that we don't want any restraint on astigmatism
            }
        }

        find_additional_phase_shift = my_input->GetYesNoFromUser("Find additional phase shift?", "Input micrograph was recorded using a phase plate with variable phase shift, which you want to find", "No");

        if ( find_additional_phase_shift ) {
            minimum_additional_phase_shift     = my_input->GetFloatFromUser("Minimum phase shift (rad)", "Lower bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians", "0.0", -3.15, 3.15);
            maximum_additional_phase_shift     = my_input->GetFloatFromUser("Maximum phase shift (rad)", "Upper bound of the search for additional phase shift. Phase shift is of scattered electrons relative to unscattered electrons. In radians", "3.15", minimum_additional_phase_shift, 3.15);
            additional_phase_shift_search_step = my_input->GetFloatFromUser("Phase shift search step", "Step size for phase shift search (radians)", "0.5", 0.0, maximum_additional_phase_shift - minimum_additional_phase_shift);
        }
        else {
            minimum_additional_phase_shift     = 0.0;
            maximum_additional_phase_shift     = 0.0;
            additional_phase_shift_search_step = 0.0;
        }

        // Currently, tilt determination only works when there is no additional phase shift
        if ( ! find_additional_phase_shift )
            determine_tilt = my_input->GetYesNoFromUser("Determine sample tilt?", "Answer yes if you tilted the sample and need to determine tilt axis and angle", "No");

        give_expert_options = my_input->GetYesNoFromUser("Do you want to set expert options?", "There are options which normally not changed, but can be accessed by answering yes here", "No");
        if ( give_expert_options ) {
            resample_if_pixel_too_small = my_input->GetYesNoFromUser("Resample micrograph if pixel size too small?", "When the pixel is too small, Thon rings appear very thin and near the origin of the spectrum, which can lead to suboptimal fitting. This options resamples micrographs to a more reasonable pixel size if needed", "Yes");
            if ( input_is_a_movie ) {
                movie_is_dark_corrected = my_input->GetYesNoFromUser("Movie is dark-subtracted?", "If the movie is not dark-subtracted you will need to provide a dark reference image", "Yes");
                if ( movie_is_dark_corrected ) {
                    dark_filename = "";
                }
                else {
                    dark_filename = my_input->GetFilenameFromUser("Dark image filename", "The filename of the dark reference image for the detector/camera", "dark.dm4", true);
                }

                movie_is_gain_corrected = my_input->GetYesNoFromUser("Movie is gain-corrected?", "If the movie is not gain-corrected, you will need to provide a gain reference image", "Yes");
                if ( movie_is_gain_corrected ) {
                    gain_filename = "";
                }
                else {
                    gain_filename = my_input->GetFilenameFromUser("Gain image filename", "The filename of the gain reference image for the detector/camera", "gain.dm4", true);
                }

                correct_movie_mag_distortion = my_input->GetYesNoFromUser("Correct Movie Mag. Distortion?", "If the movie has a mag distortion you can specify the parameters to correct it prior to estimation", "No");

                if ( correct_movie_mag_distortion == true ) {
                    movie_mag_distortion_angle       = my_input->GetFloatFromUser("Mag. distortion angle", "The angle of the distortion", "0.0");
                    movie_mag_distortion_major_scale = my_input->GetFloatFromUser("Mag. distortion major scale", "The scale factor along the major axis", "1.0");
                    movie_mag_distortion_minor_scale = my_input->GetFloatFromUser("Mag. distortion minor scale", "The scale factor along the minor axis", "1.0");
                    ;
                }
                else {
                    movie_mag_distortion_angle       = 0.0;
                    movie_mag_distortion_major_scale = 1.0;
                    movie_mag_distortion_minor_scale = 1.0;
                }

                if ( FilenameExtensionMatches(input_filename, "eer") ) {
                    eer_frames_per_image = my_input->GetIntFromUser("Number of EER frames per image", "If the input movie is in EER format, we will average EER frames together so that each frame image for alignment has a reasonable exposure", "25", 1);
                    eer_super_res_factor = my_input->GetIntFromUser("EER super resolution factor", "Choose between 1 (no supersampling), 2, or 4 (image pixel size will be 4 times smaller that the camera phyiscal pixel)", "1", 1, 4);
                }
                else {
                    eer_frames_per_image = 0;
                    eer_super_res_factor = 1;
                }
            }
            else {
                movie_is_gain_corrected = true;
                movie_is_dark_corrected = true;
                gain_filename           = "";
                dark_filename           = "";
                eer_frames_per_image    = 0;
                eer_super_res_factor    = 1;
            }
            defocus_is_known = my_input->GetYesNoFromUser("Do you already know the defocus?", "Answer yes if you already know the defocus and you just want to know the score or fit resolution. If you answer yes, the known astigmatism parameter specified eariler will be ignored", "No");
            if ( defocus_is_known ) {
                /*
				 * Right now, we don't support phase plate data for this. The proper way to do this would be to also ask whether phase shift is known.
				 * Another acceptable solution might be to say that if you know the defocus you must also know the phase shift (in other words, this would
				 * only be used to test the ctffind scoring function / diagnostics using given defocus parameters). Neither are implemented right now,
				 * because I don't need either.
				 */
                known_defocus_1         = my_input->GetFloatFromUser("Known defocus 1", "In Angstroms, the defocus along the first axis", "0.0");
                known_defocus_2         = my_input->GetFloatFromUser("Known defocus 2", "In Angstroms, the defocus along the second axis", "0.0");
                known_astigmatism_angle = my_input->GetFloatFromUser("Known astigmatism angle", "In degrees, the angle of astigmatism", "0.0");
                if ( find_additional_phase_shift ) {
                    known_phase_shift = my_input->GetFloatFromUser("Known phase shift (radians)", "In radians, the phase shift (from a phase plate presumably)", "0.0");
                }
            }
            else {
                known_defocus_1         = 0.0;
                known_defocus_2         = 0.0;
                known_astigmatism_angle = 0.0;
                known_phase_shift       = 0.0;
            }
            desired_number_of_threads = my_input->GetIntFromUser("Desired number of parallel threads", "The command-line option -j will override this", "1", 1);
        }
        else // expert options not supplied by user
        {
            resample_if_pixel_too_small = true;
            movie_is_gain_corrected     = true;
            movie_is_dark_corrected     = true;
            gain_filename               = "";
            dark_filename               = "";
            defocus_is_known            = false;
            desired_number_of_threads   = 1;
            eer_frames_per_image        = 0;
            eer_super_res_factor        = 1;
        }

        delete my_input;
    }

    //	my_current_job.Reset(39);
    my_current_job.ManualSetArguments("tbitffffifffffbfbfffbffbbsbsbfffbfffbiiibbbfffbb", input_filename.c_str( ), //1
                                      input_is_a_movie,
                                      number_of_frames_to_average,
                                      output_diagnostic_filename.c_str( ),
                                      pixel_size,
                                      acceleration_voltage,
                                      spherical_aberration,
                                      amplitude_contrast,
                                      box_size,
                                      minimum_resolution, //10
                                      maximum_resolution,
                                      minimum_defocus,
                                      maximum_defocus,
                                      defocus_search_step,
                                      slower_search,
                                      astigmatism_tolerance,
                                      find_additional_phase_shift,
                                      minimum_additional_phase_shift,
                                      maximum_additional_phase_shift,
                                      additional_phase_shift_search_step, //20
                                      astigmatism_is_known,
                                      known_astigmatism,
                                      known_astigmatism_angle,
                                      resample_if_pixel_too_small,
                                      movie_is_gain_corrected,
                                      gain_filename.ToStdString( ).c_str( ),
                                      movie_is_dark_corrected,
                                      dark_filename.ToStdString( ).c_str( ),
                                      correct_movie_mag_distortion,
                                      movie_mag_distortion_angle, //30
                                      movie_mag_distortion_major_scale,
                                      movie_mag_distortion_minor_scale,
                                      defocus_is_known,
                                      known_defocus_1,
                                      known_defocus_2,
                                      known_phase_shift,
                                      determine_tilt,
                                      desired_number_of_threads,
                                      eer_frames_per_image,
                                      eer_super_res_factor,
                                      false,
                                      false,
                                      false,
                                      10.0,
                                      3.0,
                                      1.4,
                                      false,
                                      false);
}

// Optional command-line stuff
void CtffindApp::AddCommandLineOptions( ) {
    command_line_parser.AddLongSwitch("old-school-input", "Pretend this is ctffind3 (for compatibility with old scripts and programs)");
    command_line_parser.AddLongSwitch("old-school-input-ctffind4", "Accept parameters from stdin, like ctffind3, but with extra lines for ctffind4-specific options (movie processing and phase shift estimation");
    command_line_parser.AddLongSwitch("amplitude-spectrum-input", "The input image is an amplitude spectrum, not a real-space image");
    command_line_parser.AddLongSwitch("filtered-amplitude-spectrum-input", "The input image is filtered (background-subtracted) amplitude spectrum");
    command_line_parser.AddLongSwitch("fast", "Skip computation of fit statistics as well as spectrum contrast enhancement");
    command_line_parser.AddOption("j", "", "Desired number of threads. Overrides interactive user input. Is overriden by env var OMP_NUM_THREADS", wxCMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongSwitch("debug", "Write debug information to disk");
}

// override the do calculation method which will be what is actually run..

bool CtffindApp::DoCalculation( ) {

    cistem_timer::StopWatch ctffind_timing;
    ctffind_timing.start("Initialization");
    StopWatch profile_timing;
    // Arguments for this job

    const std::string input_filename                     = my_current_job.arguments[0].ReturnStringArgument( );
    const bool        input_is_a_movie                   = my_current_job.arguments[1].ReturnBoolArgument( );
    const int         number_of_frames_to_average        = my_current_job.arguments[2].ReturnIntegerArgument( );
    const std::string output_diagnostic_filename         = my_current_job.arguments[3].ReturnStringArgument( );
    float             pixel_size_of_input_image          = my_current_job.arguments[4].ReturnFloatArgument( ); // no longer const, as the mag distortion can change it.
    const float       acceleration_voltage               = my_current_job.arguments[5].ReturnFloatArgument( );
    const float       spherical_aberration               = my_current_job.arguments[6].ReturnFloatArgument( );
    const float       amplitude_contrast                 = my_current_job.arguments[7].ReturnFloatArgument( );
    const int         box_size                           = my_current_job.arguments[8].ReturnIntegerArgument( );
    const float       minimum_resolution                 = my_current_job.arguments[9].ReturnFloatArgument( );
    const float       maximum_resolution                 = my_current_job.arguments[10].ReturnFloatArgument( );
    const float       minimum_defocus                    = my_current_job.arguments[11].ReturnFloatArgument( );
    const float       maximum_defocus                    = my_current_job.arguments[12].ReturnFloatArgument( );
    const float       defocus_search_step                = my_current_job.arguments[13].ReturnFloatArgument( );
    const bool        slower_search                      = my_current_job.arguments[14].ReturnBoolArgument( );
    const float       astigmatism_tolerance              = my_current_job.arguments[15].ReturnFloatArgument( );
    const bool        find_additional_phase_shift        = my_current_job.arguments[16].ReturnBoolArgument( );
    const float       minimum_additional_phase_shift     = my_current_job.arguments[17].ReturnFloatArgument( );
    const float       maximum_additional_phase_shift     = my_current_job.arguments[18].ReturnFloatArgument( );
    const float       additional_phase_shift_search_step = my_current_job.arguments[19].ReturnFloatArgument( );
    const bool        astigmatism_is_known               = my_current_job.arguments[20].ReturnBoolArgument( );
    const float       known_astigmatism                  = my_current_job.arguments[21].ReturnFloatArgument( );
    const float       known_astigmatism_angle            = my_current_job.arguments[22].ReturnFloatArgument( );
    const bool        resample_if_pixel_too_small        = my_current_job.arguments[23].ReturnBoolArgument( );
    const bool        movie_is_gain_corrected            = my_current_job.arguments[24].ReturnBoolArgument( );
    const wxString    gain_filename                      = my_current_job.arguments[25].ReturnStringArgument( );
    const bool        movie_is_dark_corrected            = my_current_job.arguments[26].ReturnBoolArgument( );
    const wxString    dark_filename                      = my_current_job.arguments[27].ReturnStringArgument( );
    const bool        correct_movie_mag_distortion       = my_current_job.arguments[28].ReturnBoolArgument( );
    const float       movie_mag_distortion_angle         = my_current_job.arguments[29].ReturnFloatArgument( );
    const float       movie_mag_distortion_major_scale   = my_current_job.arguments[30].ReturnFloatArgument( );
    const float       movie_mag_distortion_minor_scale   = my_current_job.arguments[31].ReturnFloatArgument( );
    const bool        defocus_is_known                   = my_current_job.arguments[32].ReturnBoolArgument( );
    const float       known_defocus_1                    = my_current_job.arguments[33].ReturnFloatArgument( );
    const float       known_defocus_2                    = my_current_job.arguments[34].ReturnFloatArgument( );
    const float       known_phase_shift                  = my_current_job.arguments[35].ReturnFloatArgument( );
    const bool        determine_tilt                     = my_current_job.arguments[36].ReturnBoolArgument( );
    int               desired_number_of_threads          = my_current_job.arguments[37].ReturnIntegerArgument( );
    int               eer_frames_per_image               = my_current_job.arguments[38].ReturnIntegerArgument( );
    int               eer_super_res_factor               = my_current_job.arguments[39].ReturnIntegerArgument( );
    bool              fit_nodes                          = my_current_job.arguments[40].ReturnBoolArgument( );
    bool              fit_nodes_1D_brute_force           = my_current_job.arguments[41].ReturnBoolArgument( );
    bool              fit_nodes_2D_refine                = my_current_job.arguments[42].ReturnBoolArgument( );
    float             fit_nodes_low_resolution_limit     = my_current_job.arguments[43].ReturnFloatArgument( );
    float             fit_nodes_high_resolution_limit    = my_current_job.arguments[44].ReturnFloatArgument( );
    float             target_pixel_size_after_resampling = my_current_job.arguments[45].ReturnFloatArgument( );
    bool              fit_nodes_use_rounded_square       = my_current_job.arguments[46].ReturnBoolArgument( );
    MyDebugPrint("fit_nodes_use_rounded_square = %i", fit_nodes_use_rounded_square);
    bool fit_nodes_downweight_nodes = my_current_job.arguments[47].ReturnBoolArgument( );
    // if we are applying a mag distortion, it can change the pixel size, so do that here to make sure it is used forever onwards..

    if ( input_is_a_movie && correct_movie_mag_distortion ) {
        pixel_size_of_input_image = ReturnMagDistortionCorrectedPixelSize(pixel_size_of_input_image, movie_mag_distortion_major_scale, movie_mag_distortion_minor_scale);
    }

    // These variables will be set by command-line options
    const bool old_school_input                  = command_line_parser.FoundSwitch("old-school-input") || command_line_parser.FoundSwitch("old-school-input-ctffind4");
    const bool amplitude_spectrum_input          = command_line_parser.FoundSwitch("amplitude-spectrum-input");
    const bool filtered_amplitude_spectrum_input = command_line_parser.FoundSwitch("filtered-amplitude-spectrum-input");
    const bool compute_extra_stats               = ! command_line_parser.FoundSwitch("fast");
    const bool boost_ring_contrast               = ! command_line_parser.FoundSwitch("fast");
    long       command_line_desired_number_of_threads;
    if ( command_line_parser.Found("j", &command_line_desired_number_of_threads) ) {
        // Command-line argument overrides
        desired_number_of_threads = command_line_desired_number_of_threads;
    }

    // Resampling of input images to ensure that the pixel size isn't too small
    if ( target_pixel_size_after_resampling <= 0.0 ) {
        target_pixel_size_after_resampling = 1.4f;
    }
    const float target_nyquist_after_resampling = 2 * target_pixel_size_after_resampling; // Angstroms
    // const float target_pixel_size_after_resampling = 0.5 * target_nyquist_after_resampling;
    float pixel_size_for_fitting = pixel_size_of_input_image;
    int   temporary_box_size;

    // Maybe the user wants to hold the phase shift value (which they can do by giving the same value for min and max)
    const bool fixed_additional_phase_shift = fabs(maximum_additional_phase_shift - minimum_additional_phase_shift) < 0.01;

    // This could become a user-supplied parameter later - for now only for developers / expert users
    const bool follow_1d_search_with_local_2D_brute_force = false;

    // Initial search should be done only using up to that resolution, to improve radius of convergence
    const float maximum_resolution_for_initial_search = 5.0;

    // Debugging
    const bool dump_debug_files = command_line_parser.FoundSwitch("debug");
    if ( dump_debug_files ) {
        MyDebugPrint("Print debug info\n");
    }
    else {
        MyDebugPrint("No info here\n");
    }
    std::string debug_file_prefix = output_diagnostic_filename.substr(0, output_diagnostic_filename.find_last_of('.')) + "_debug_";

    /*
	 *  Scoring function
	 */
    float MyFunction(float[]);

    // Other variables
    int              number_of_movie_frames;
    int              number_of_micrographs;
    ImageFile        input_file;
    Image*           average_spectrum        = new Image( );
    Image*           average_spectrum_masked = new Image( );
    wxString         output_text_fn;
    ProgressBar*     my_progress_bar;
    NumericTextFile* output_text;
    NumericTextFile* output_text_avrot;
    int              current_micrograph_number;
    int              number_of_tiles_used;
    Image*           current_power_spectrum = new Image( );
    int              current_first_frame_within_average;
    int              current_frame_within_average;
    int              current_input_location;
    Image*           current_input_image        = new Image( );
    Image*           current_input_image_square = new Image( );
    int              micrograph_square_dimension;
    Image*           temp_image               = new Image( );
    Image*           sum_image                = new Image( );
    Image*           resampled_power_spectrum = new Image( );
    bool             resampling_is_necessary;
    CTF*             current_ctf = new CTF( );
    float            average, sigma;
    int              convolution_box_size;
    // ImageCTFComparison  comparison_object_2D;
    CurveCTFComparison comparison_object_1D;
    float              estimated_astigmatism_angle;
    float              bf_halfrange[4];
    float              bf_midpoint[4];
    float              bf_stepsize[4];
    float              cg_starting_point[4];
    float              cg_accuracy[4];
    int                number_of_search_dimensions;
    BruteForceSearch*  brute_force_search;
    int                counter;
    ConjugateGradient* conjugate_gradient_minimizer;
    int                current_output_location;
    int                number_of_bins_in_1d_spectra;
    Curve*             number_of_averaged_pixels                 = new Curve( );
    Curve*             rotational_average                        = new Curve( );
    Image*             number_of_extrema_image                   = new Image( );
    Image*             ctf_values_image                          = new Image( );
    double*            rotational_average_astig                  = NULL;
    double*            rotational_average_astig_renormalized     = NULL;
    double*            spatial_frequency                         = NULL;
    double*            spatial_frequency_in_reciprocal_angstroms = NULL;
    double*            rotational_average_astig_fit              = NULL;
    float*             number_of_extrema_profile                 = NULL;
    float*             ctf_values_profile                        = NULL;
    double*            fit_frc                                   = NULL;
    double*            fit_frc_sigma                             = NULL;
    MRCFile            output_diagnostic_file(output_diagnostic_filename, true);
    int                last_bin_with_good_fit;
    double*            values_to_write_out = new double[7];
    float              best_score_after_initial_phase;
    int                last_bin_without_aliasing;
    ImageFile          gain_file;
    Image*             gain = new Image( );
    ImageFile          dark_file;
    Image*             dark = new Image( );
    float              final_score;
    float              tilt_axis;
    float              tilt_angle;

    // Open the input file
    bool input_file_is_valid = input_file.OpenFile(input_filename, false, false, false, eer_super_res_factor, eer_frames_per_image);
    if ( ! input_file_is_valid ) {
        SendInfo(wxString::Format("Input movie %s seems to be corrupt. Ctffind results may not be meaningful.\n", input_filename));
    }
    else {
        wxPrintf("Input file looks OK, proceeding\n");
    }

    // Some argument checking
    if ( determine_tilt && find_additional_phase_shift ) {
        SendError(wxString::Format("Error: Finding additional phase shift and determining sample tilt cannot be active at the same time. Terminating."));
        ExitMainLoop( );
    }
    if ( determine_tilt && (amplitude_spectrum_input || filtered_amplitude_spectrum_input) ) {
        SendError(wxString::Format("Error: Determining sample tilt cannot be run with either amplitude-spectrum-input or filtered-amplitude-spectrum-input. Terminating."));
        DEBUG_ABORT; // THis applies to CLI not GUI
    }
    if ( minimum_resolution < maximum_resolution ) {
        SendError(wxString::Format("Error: Minimum resolution (%f) higher than maximum resolution (%f). Terminating.", minimum_resolution, maximum_resolution));
        ExitMainLoop( );
    }
    if ( minimum_defocus > maximum_defocus ) {
        SendError(wxString::Format("Minimum defocus must be less than maximum defocus. Terminating."));
        ExitMainLoop( );
    }

    // How many micrographs are we dealing with
    if ( input_is_a_movie ) {
        // We only support 1 movie per file
        number_of_movie_frames = input_file.ReturnZSize( );
        number_of_micrographs  = 1;
    }
    else {
        number_of_movie_frames = 1;
        number_of_micrographs  = input_file.ReturnZSize( );
    }

    if ( is_running_locally ) {
        // Print out information about input file
        input_file.PrintInfo( );
    }

    // Prepare the output text file
    output_text_fn = FilenameReplaceExtension(output_diagnostic_filename, "txt");

    if ( is_running_locally ) {
        output_text = new NumericTextFile(output_text_fn, OPEN_TO_WRITE, 7);

        // Print header to the output text file
        output_text->WriteCommentLine("# Output from CTFFind version %s, run on %s\n", ctffind_version.c_str( ), wxDateTime::Now( ).FormatISOCombined(' ').ToStdString( ).c_str( ));
        output_text->WriteCommentLine("# Input file: %s ; Number of micrographs: %i\n", input_filename.c_str( ), number_of_micrographs);
        output_text->WriteCommentLine("# Pixel size: %0.3f Angstroms ; acceleration voltage: %0.1f keV ; spherical aberration: %0.2f mm ; amplitude contrast: %0.2f\n", pixel_size_of_input_image, acceleration_voltage, spherical_aberration, amplitude_contrast);
        output_text->WriteCommentLine("# Box size: %i pixels ; min. res.: %0.1f Angstroms ; max. res.: %0.1f Angstroms ; min. def.: %0.1f um; max. def. %0.1f um\n", box_size, minimum_resolution, maximum_resolution, minimum_defocus, maximum_defocus);
        output_text->WriteCommentLine("# Columns: #1 - micrograph number; #2 - defocus 1 [Angstroms]; #3 - defocus 2; #4 - azimuth of astigmatism; #5 - additional phase shift [radians]; #6 - cross correlation; #7 - spacing (in Angstroms) up to which CTF rings were fit successfully\n");
    }

    // Prepare a text file with 1D rotational average spectra
    output_text_fn = FilenameAddSuffix(output_text_fn.ToStdString( ), "_avrot");

    if ( ! old_school_input && number_of_micrographs > 1 && is_running_locally ) {
        wxPrintf("Will estimate the CTF parameters for %i micrographs.\n", number_of_micrographs);
        wxPrintf("Results will be written to this file: %s\n", output_text->ReturnFilename( ));
        wxPrintf("\nEstimating CTF parameters...\n\n");
        my_progress_bar = new ProgressBar(number_of_micrographs);
    }

    ctffind_timing.lap("Initialization");
    ctffind_timing.start("Spectrum computation");

    // Prepare the dark/gain_reference
    if ( input_is_a_movie && ! movie_is_gain_corrected ) {
        profile_timing.start("Read gain reference");
        gain_file.OpenFile(gain_filename.ToStdString( ), false);
        gain->ReadSlice(&gain_file, 1);
        profile_timing.lap("Read gain reference");
    }

    if ( input_is_a_movie && ! movie_is_dark_corrected ) {
        profile_timing.start("Read dark reference");
        dark_file.OpenFile(dark_filename.ToStdString( ), false);
        dark->ReadSlice(&dark_file, 1);
        profile_timing.lap("Read dark reference");
    }

    // Prepare the average spectrum image
    average_spectrum->Allocate(box_size, box_size, true);

    // Loop over micrographs
    for ( current_micrograph_number = 1; current_micrograph_number <= number_of_micrographs; current_micrograph_number++ ) {
        if ( is_running_locally && (old_school_input || number_of_micrographs == 1) )
            wxPrintf("Working on micrograph %i of %i\n", current_micrograph_number, number_of_micrographs);

        number_of_tiles_used = 0;
        average_spectrum->SetToConstant(0.0);
        average_spectrum->is_in_real_space = true;

        if ( amplitude_spectrum_input || filtered_amplitude_spectrum_input ) {
            current_power_spectrum->ReadSlice(&input_file, current_micrograph_number);
            current_power_spectrum->ForwardFFT( );
            average_spectrum->Allocate(box_size, box_size, 1, false);
            current_power_spectrum->ClipInto(average_spectrum);
            average_spectrum->BackwardFFT( );
            average_spectrum_masked->CopyFrom(average_spectrum);
        }
        else {
            CTFTilt tilt_scorer(input_file, 5.0f, 10.0f, minimum_defocus, maximum_defocus, pixel_size_of_input_image, acceleration_voltage, spherical_aberration, amplitude_contrast, 0.0f, dump_debug_files, debug_file_prefix + "_tilt.json");

            for ( current_first_frame_within_average = 1; current_first_frame_within_average <= number_of_movie_frames; current_first_frame_within_average += number_of_frames_to_average ) {
                for ( current_frame_within_average = 1; current_frame_within_average <= number_of_frames_to_average; current_frame_within_average++ ) {
                    current_input_location = current_first_frame_within_average + number_of_movie_frames * (current_micrograph_number - 1) + (current_frame_within_average - 1);
                    if ( current_input_location > number_of_movie_frames * current_micrograph_number )
                        continue;
                    // Read the image in
                    profile_timing.start("Read and check image");
                    current_input_image->ReadSlice(&input_file, current_input_location);
                    if ( current_input_image->IsConstant( ) ) {

                        if ( is_running_locally == false ) {
                            // don't crash, as this will lead to the gui job never finishing, instead send a blank result..
                            SendError(wxString::Format("Error: location %i of input file %s is blank, defocus parameters will be set to 0", current_input_location, input_filename));

                            float results_array[10];
                            results_array[0] = 0.0; // Defocus 1 (Angstroms)
                            results_array[1] = 0.0; // Defocus 2 (Angstroms)
                            results_array[2] = 0.0; // Astigmatism angle (degrees)
                            results_array[3] = 0.0; // Additional phase shift (e.g. from phase plate) (radians)
                            results_array[4] = 0.0; // CTFFIND score
                            results_array[5] = 0.0;
                            results_array[6] = 0.0;
                            results_array[7] = 0.0;
                            results_array[8] = 0.0;
                            results_array[9] = 0.0;

                            my_result.SetResult(10, results_array);

                            delete average_spectrum;
                            delete average_spectrum_masked;
                            delete current_power_spectrum;
                            delete current_input_image;
                            delete current_input_image_square;
                            delete temp_image;
                            delete sum_image;
                            delete resampled_power_spectrum;
                            delete number_of_extrema_image;
                            delete ctf_values_image;
                            delete gain;
                            delete dark;
                            delete[] values_to_write_out;

                            return true;
                        }
                        else {
                            SendError(wxString::Format("Error: location %i of input file %s is blank", current_input_location, input_filename));
                            ExitMainLoop( );
                        }
                    }
                    profile_timing.lap("Read and check image");

                    // Apply dark reference
                    if ( input_is_a_movie && ! movie_is_dark_corrected ) {
                        profile_timing.start("Apply dark");
                        if ( ! current_input_image->HasSameDimensionsAs(dark) ) {
                            SendError(wxString::Format("Error: location %i of input file %s does not have same dimensions as the dark image", current_input_location, input_filename));
                            ExitMainLoop( );
                        }

                        current_input_image->SubtractImage(dark);
                        profile_timing.lap("Apply dark");
                    }

                    // Apply gain reference
                    if ( input_is_a_movie && ! movie_is_gain_corrected ) {
                        profile_timing.start("Apply gain");
                        if ( ! current_input_image->HasSameDimensionsAs(gain) ) {
                            SendError(wxString::Format("Error: location %i of input file %s does not have same dimensions as the gain image", current_input_location, input_filename));
                            ExitMainLoop( );
                        }
                        current_input_image->MultiplyPixelWise(*gain);
                        profile_timing.lap("Apply gain");
                    }
                    // correct for mag distortion
                    if ( input_is_a_movie && correct_movie_mag_distortion ) {
                        profile_timing.start("Correct mag distortion");
                        current_input_image->CorrectMagnificationDistortion(movie_mag_distortion_angle, movie_mag_distortion_major_scale, movie_mag_distortion_minor_scale);
                        profile_timing.lap("Correct mag distortion");
                    }
                    // Make the image square
                    profile_timing.start("Crop image to shortest dimension");
                    micrograph_square_dimension = std::max(current_input_image->logical_x_dimension, current_input_image->logical_y_dimension);
                    if ( IsOdd((micrograph_square_dimension)) )
                        micrograph_square_dimension++;
                    if ( current_input_image->logical_x_dimension != micrograph_square_dimension || current_input_image->logical_y_dimension != micrograph_square_dimension ) {
                        current_input_image_square->Allocate(micrograph_square_dimension, micrograph_square_dimension, true);
                        //current_input_image->ClipInto(current_input_image_square,current_input_image->ReturnAverageOfRealValues());
                        current_input_image->ClipIntoLargerRealSpace2D(current_input_image_square, current_input_image->ReturnAverageOfRealValues( ));
                        current_input_image->Consume(current_input_image_square);
                    }
                    profile_timing.lap("Crop image to shortest dimension");
                    //
                    profile_timing.start("Average frames");
                    if ( current_frame_within_average == 1 ) {
                        sum_image->Allocate(current_input_image->logical_x_dimension, current_input_image->logical_y_dimension, true);
                        sum_image->SetToConstant(0.0);
                    }
                    sum_image->AddImage(current_input_image);
                    profile_timing.lap("Average frames");
                } // end of loop over frames to average together
                current_input_image->Consume(sum_image);

                // Taper the edges of the micrograph in real space, to lessen Gibbs artefacts
                // Introduces an artefact of its own, so it's not clear on balance whether tapering helps, especially with modern micrographs from good detectors
                //current_input_image->TaperEdges();

                number_of_tiles_used++;
                if ( determine_tilt ) {
                    //					wxPrintf("Read frame = %i\n", number_of_tiles_used);
                    tilt_scorer.UpdateInputImage(current_input_image);
                    if ( current_first_frame_within_average + number_of_frames_to_average > number_of_movie_frames )
                        tilt_scorer.CalculatePowerSpectra(true);
                    else
                        tilt_scorer.CalculatePowerSpectra( );
                }
                else {
                    profile_timing.start("Compute amplitude spectrum");
                    // Compute the amplitude spectrum
                    current_power_spectrum->Allocate(current_input_image->logical_x_dimension, current_input_image->logical_y_dimension, true);
                    current_input_image->ForwardFFT(false);
                    current_input_image->ComputeAmplitudeSpectrumFull2D(current_power_spectrum);

                    //current_power_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_resampling.mrc",1);

                    // Set origin of amplitude spectrum to 0.0
                    current_power_spectrum->real_values[current_power_spectrum->ReturnReal1DAddressFromPhysicalCoord(current_power_spectrum->physical_address_of_box_center_x, current_power_spectrum->physical_address_of_box_center_y, current_power_spectrum->physical_address_of_box_center_z)] = 0.0;

                    // Resample the amplitude spectrum
                    pixel_size_for_fitting = PixelSizeForFitting(resample_if_pixel_too_small, pixel_size_of_input_image, target_pixel_size_after_resampling, box_size, current_power_spectrum, resampled_power_spectrum);

                    average_spectrum->AddImage(resampled_power_spectrum);
                    profile_timing.lap("Compute amplitude spectrum");
                }

            } // end of loop over movie frames

            if ( determine_tilt ) {
                profile_timing.start("Tilt estimation");
                // Find rough defocus
                tilt_scorer.FindRoughDefocus( );
                //	wxPrintf("\nFindRoughDefocus values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                // Find astigmatism
                tilt_scorer.FindDefocusAstigmatism( );
                //	wxPrintf("\nFindDefocusAstigmatism values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                // Search tilt axis and angle
                tilt_scorer.SearchTiltAxisAndAngle( );
                //	wxPrintf("\nSearchTiltAxisAndAngle values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                // Refine tilt axis and angle
                tilt_scorer.RefineTiltAxisAndAngle( );
                //	wxPrintf("\nRefineTiltAxisAndAngle values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);

                tilt_axis  = tilt_scorer.best_tilt_axis;
                tilt_angle = tilt_scorer.best_tilt_angle;
                if ( tilt_angle < 0.0f ) {
                    tilt_axis += 180.0f;
                    tilt_angle = -tilt_angle;
                }
                if ( tilt_axis > 360.0f ) {
                    tilt_axis -= 360.0f;
                }
                MyDebugPrint("Final values: defocus_1, defocus_2, astig_angle, tilt_axis, tilt_angle = %g %g %g || %g %g\n\n", tilt_scorer.defocus_1, tilt_scorer.defocus_2, tilt_scorer.astigmatic_angle, tilt_scorer.best_tilt_axis, tilt_scorer.best_tilt_angle);
                profile_timing.lap("Tilt estimation");
                profile_timing.start("Tilt correction");
                pixel_size_for_fitting = tilt_scorer.CalculateTiltCorrectedSpectra(resample_if_pixel_too_small, pixel_size_of_input_image, target_pixel_size_after_resampling, box_size, average_spectrum);
                average_spectrum->MultiplyByConstant(float(number_of_tiles_used));
                profile_timing.lap("Tilt correction");
            }
            else {
                tilt_angle = 0.0f;
                tilt_axis  = 0.0f;
            }

            // We need to take care of the scaling of the FFTs, as well as the averaging of tiles
            if ( resampling_is_necessary ) {
                average_spectrum->MultiplyByConstant(1.0 / (float(number_of_tiles_used) * current_input_image->logical_x_dimension * current_input_image->logical_y_dimension * current_power_spectrum->logical_x_dimension * current_power_spectrum->logical_y_dimension));
            }
            else {
                average_spectrum->MultiplyByConstant(1.0 / (float(number_of_tiles_used) * current_input_image->logical_x_dimension * current_input_image->logical_y_dimension));
            }

        } // end of test of whether we were given amplitude spectra on input

        //average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_bg_sub.mrc",1);

        // Filter the amplitude spectrum, remove background
        if ( ! filtered_amplitude_spectrum_input ) {
            profile_timing.start("Filter spectrum");
            average_spectrum->ComputeFilteredAmplitudeSpectrumFull2D(average_spectrum_masked, current_power_spectrum, average, sigma, minimum_resolution, maximum_resolution, pixel_size_for_fitting);
            profile_timing.lap("Filter spectrum");
        }

        /*
		 *
		 *
		 * We now have a spectrum which we can use to fit CTFs
		 *
		 *
		 */
        ctffind_timing.lap("Spectrum computation");
        ctffind_timing.start("Parameter search");

        if ( dump_debug_files ) {
            average_spectrum->WriteSlicesAndFillHeader(debug_file_prefix + "dbg_spectrum_for_fitting.mrc", pixel_size_for_fitting);
            average_spectrum_masked->WriteSlicesAndFillHeader(debug_file_prefix + "dbg_spectrum_for_fitting_masked.mrc", pixel_size_for_fitting);
        }

#ifdef threshold_spectrum
        wxPrintf("DEBUG: thresholding spectrum\n");
        for ( counter = 0; counter < average_spectrum->real_memory_allocated; counter++ ) {
            average_spectrum->real_values[counter] = std::max(average_spectrum->real_values[counter], -0.0f);
            average_spectrum->real_values[counter] = std::min(average_spectrum->real_values[counter], 1.0f);
        }
        average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_thr.mrc", 1);
#endif

        // Set up the CTF object, initially limited to at most maximum_resolution_for_initial_search
        current_ctf->Init(acceleration_voltage, spherical_aberration, amplitude_contrast, minimum_defocus, minimum_defocus, 0.0, 1.0 / minimum_resolution, 1.0 / std::max(maximum_resolution, maximum_resolution_for_initial_search), astigmatism_tolerance, pixel_size_for_fitting, minimum_additional_phase_shift);
        current_ctf->SetDefocus(minimum_defocus / pixel_size_for_fitting, minimum_defocus / pixel_size_for_fitting, 0.0);
        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);

        // Set up the comparison object
        ImageCTFComparison comparison_object_2D = ImageCTFComparison(1, *current_ctf, pixel_size_for_fitting, find_additional_phase_shift && ! fixed_additional_phase_shift, astigmatism_is_known, known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf, false);
        comparison_object_2D.SetImage(0, average_spectrum_masked);
        comparison_object_2D.SetupQuickCorrelation( );

        if ( defocus_is_known ) {
            profile_timing.start("Calculate Score for known defocus");
            current_ctf->SetDefocus(known_defocus_1 / pixel_size_for_fitting, known_defocus_2 / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf);
            current_ctf->SetAdditionalPhaseShift(known_phase_shift);
            current_ctf->SetHighestFrequencyForFitting(pixel_size_for_fitting / maximum_resolution);
            comparison_object_2D.SetCTF(*current_ctf);
            comparison_object_2D.SetupQuickCorrelation( );
            final_score = 0.0;
            final_score = comparison_object_2D.img[0].QuickCorrelationWithCTF(*current_ctf, comparison_object_2D.number_to_correlate, comparison_object_2D.norm_image, comparison_object_2D.image_mean, comparison_object_2D.addresses,
                                                                              comparison_object_2D.spatial_frequency_squared, comparison_object_2D.azimuths);
            profile_timing.lap("Calculate Score for known defocus");
        }
        else {

            if ( is_running_locally && old_school_input ) {
                wxPrintf("\nSEARCHING CTF PARAMETERS...\n");
            }

            // Let's look for the astigmatism angle first
            if ( astigmatism_is_known ) {
                estimated_astigmatism_angle = known_astigmatism_angle;
            }
            else {
                profile_timing.start("Estimate initial astigmatism");
                temp_image->CopyFrom(average_spectrum);
                temp_image->ApplyMirrorAlongY( );
                //temp_image.QuickAndDirtyWriteSlice("dbg_spec_y.mrc",1);
                estimated_astigmatism_angle = 0.5 * FindRotationalAlignmentBetweenTwoStacksOfImages(average_spectrum, temp_image, 1, 90.0, 5.0, pixel_size_for_fitting / minimum_resolution, pixel_size_for_fitting / std::max(maximum_resolution, maximum_resolution_for_initial_search));
                profile_timing.lap("Estimate initial astigmatism");
            }

            //MyDebugPrint ("Estimated astigmatism angle = %f degrees\n", estimated_astigmatism_angle);

            /*
			 * Initial brute-force search, in 1D (fast, but not as accurate)
			 */
            if ( ! slower_search ) {
                profile_timing.start("Setup 1D search");
                // 1D rotational average
                number_of_bins_in_1d_spectra = int(ceil(average_spectrum_masked->ReturnMaximumDiagonalRadius( )));
                rotational_average->SetupXAxis(0.0, float(number_of_bins_in_1d_spectra) * average_spectrum_masked->fourier_voxel_size_x, number_of_bins_in_1d_spectra);
                number_of_averaged_pixels->CopyFrom(rotational_average);
                average_spectrum_masked->Compute1DRotationalAverage(*rotational_average, *number_of_averaged_pixels, true);

                comparison_object_1D.ctf   = *current_ctf;
                comparison_object_1D.curve = new float[number_of_bins_in_1d_spectra];
                for ( counter = 0; counter < number_of_bins_in_1d_spectra; counter++ ) {
                    comparison_object_1D.curve[counter] = rotational_average->data_y[counter];
                }
                comparison_object_1D.find_phase_shift      = find_additional_phase_shift && ! fixed_additional_phase_shift;
                comparison_object_1D.number_of_bins        = number_of_bins_in_1d_spectra;
                comparison_object_1D.reciprocal_pixel_size = average_spectrum_masked->fourier_voxel_size_x;

                // We can now look for the defocus value
                bf_halfrange[0] = 0.5 * (maximum_defocus - minimum_defocus) / pixel_size_for_fitting;
                bf_halfrange[1] = 0.5 * (maximum_additional_phase_shift - minimum_additional_phase_shift);

                bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[1];

                bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                bf_stepsize[1] = additional_phase_shift_search_step;

                if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                    number_of_search_dimensions = 2;
                }
                else {
                    number_of_search_dimensions = 1;
                }

                // DNM: Do one-time set of phase shift for fixed value
                if ( find_additional_phase_shift && fixed_additional_phase_shift ) {
                    current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                }
                profile_timing.lap("Setup 1D search");
                profile_timing.start("Perform 1D search");
                // Actually run the BF search
                brute_force_search = new BruteForceSearch( );
                brute_force_search->Init(&CtffindCurveObjectiveFunction, &comparison_object_1D, number_of_search_dimensions, bf_midpoint, bf_halfrange, bf_stepsize, false, false, desired_number_of_threads);
                brute_force_search->Run( );

                /*
				wxPrintf("After 1D brute\n");
				wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
				wxPrintf("%12.2f%12.2f%12.2f%12.5f\n",brute_force_search->GetBestValue(0),brute_force_search->GetBestValue(0),0.0,brute_force_search->GetBestScore());
				wxPrintf("%12.2f%12.2f%12.2f%12.5f\n",brute_force_search->GetBestValue(0)*pixel_size_for_fitting,brute_force_search->GetBestValue(0)*pixel_size_for_fitting,0.0,brute_force_search->GetBestScore());
				*/

                /*
				 * We can now do a local optimization.
				 * The end point of the BF search is the beginning of the CG search, but we will want to use
				 * the full resolution range
				 */
                profile_timing.lap("Perform 1D search");
                profile_timing.start("Optimize 1D search");
                current_ctf->SetHighestFrequencyForFitting(pixel_size_for_fitting / maximum_resolution);
                comparison_object_1D.ctf = *current_ctf;
                for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                    cg_starting_point[counter] = brute_force_search->GetBestValue(counter);
                }
                cg_accuracy[0]               = 100.0;
                cg_accuracy[1]               = 0.05;
                conjugate_gradient_minimizer = new ConjugateGradient( );
                conjugate_gradient_minimizer->Init(&CtffindCurveObjectiveFunction, &comparison_object_1D, number_of_search_dimensions, cg_starting_point, cg_accuracy);
                conjugate_gradient_minimizer->Run( );
                for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                    cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
                }
                current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[0], estimated_astigmatism_angle / 180.0 * PIf);
                if ( find_additional_phase_shift ) {
                    if ( fixed_additional_phase_shift ) {
                        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                    }
                    else {
                        current_ctf->SetAdditionalPhaseShift(cg_starting_point[1]);
                    }
                }

                // Remember the best score so far
                best_score_after_initial_phase = -conjugate_gradient_minimizer->GetBestScore( );

                // Cleanup
                delete conjugate_gradient_minimizer;
                delete brute_force_search;
                delete[] comparison_object_1D.curve;
                profile_timing.lap("Optimize 1D search");
            } // end of the fast search over the 1D function

            /*
			 * Brute-force search over the 2D scoring function.
			 * This is either the first search we are doing, or just a refinement
			 * starting from the result of the 1D search
			 */
            if ( slower_search || (! slower_search && follow_1d_search_with_local_2D_brute_force) ) {
                // Setup the parameters for the brute force search
                profile_timing.start("Setup 2D search");
                if ( slower_search ) // This is the first search we are doing - scan the entire range the user specified
                {

                    if ( astigmatism_is_known ) {
                        bf_halfrange[0] = 0.5 * (maximum_defocus - minimum_defocus) / pixel_size_for_fitting;
                        bf_halfrange[1] = 0.5 * (maximum_additional_phase_shift - minimum_additional_phase_shift);

                        bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                        bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[3];

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = additional_phase_shift_search_step;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 2;
                        }
                        else {
                            number_of_search_dimensions = 1;
                        }
                    }
                    else {
                        bf_halfrange[0] = 0.5 * (maximum_defocus - minimum_defocus) / pixel_size_for_fitting;
                        bf_halfrange[1] = bf_halfrange[0];
                        bf_halfrange[2] = 0.0;
                        bf_halfrange[3] = 0.5 * (maximum_additional_phase_shift - minimum_additional_phase_shift);

                        bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                        bf_midpoint[1] = bf_midpoint[0];
                        bf_midpoint[2] = estimated_astigmatism_angle / 180.0 * PIf;
                        bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = bf_stepsize[0];
                        bf_stepsize[2] = 0.0;
                        bf_stepsize[3] = additional_phase_shift_search_step;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 4;
                        }
                        else {
                            number_of_search_dimensions = 3;
                        }
                    }
                }
                else // we will do a brute-force search near the result of the search over the 1D objective function
                {
                    if ( astigmatism_is_known ) {

                        bf_midpoint[0] = current_ctf->GetDefocus1( );
                        bf_midpoint[1] = current_ctf->GetAdditionalPhaseShift( );

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = additional_phase_shift_search_step;

                        bf_halfrange[0] = 2.0 * defocus_search_step / pixel_size_for_fitting + 0.1;
                        bf_halfrange[1] = 2.0 * additional_phase_shift_search_step + 0.01;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 2;
                        }
                        else {
                            number_of_search_dimensions = 1;
                        }
                    }
                    else {

                        bf_midpoint[0] = current_ctf->GetDefocus1( );
                        bf_midpoint[1] = current_ctf->GetDefocus2( );
                        bf_midpoint[2] = current_ctf->GetAstigmatismAzimuth( );
                        bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

                        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                        bf_stepsize[1] = bf_stepsize[0];
                        bf_stepsize[2] = 0.0;
                        bf_stepsize[3] = additional_phase_shift_search_step;

                        if ( astigmatism_tolerance > 0 ) {
                            bf_halfrange[0] = 2.0 * astigmatism_tolerance / pixel_size_for_fitting + 0.1;
                        }
                        else {
                            bf_halfrange[0] = 2.0 * defocus_search_step / pixel_size_for_fitting + 0.1;
                        }
                        bf_halfrange[1] = bf_halfrange[0];
                        bf_halfrange[2] = 0.0;
                        bf_halfrange[3] = 2.0 * additional_phase_shift_search_step + 0.01;

                        if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                            number_of_search_dimensions = 4;
                        }
                        else {
                            number_of_search_dimensions = 3;
                        }
                    }
                }
                profile_timing.lap("Setup 2D search");
                profile_timing.start("Perform 2D search");
                // Actually run the BF search (we run a local minimizer at every grid point only if this is a refinement search following 1D search (otherwise the full brute-force search would get too long)
                brute_force_search = new BruteForceSearch( );
                brute_force_search->Init(&CtffindObjectiveFunction, &comparison_object_2D, number_of_search_dimensions, bf_midpoint, bf_halfrange, bf_stepsize, ! slower_search, is_running_locally, desired_number_of_threads);
                brute_force_search->Run( );

                profile_timing.lap("Perform 2D search");
                profile_timing.start("Setup 2D search optimization");
                // If we did the slower exhaustive search as our first search, we want to update the max fit resolution from maximum_resolution_for_initial_search -> maximum_resolution.
                if ( slower_search ) {
                    current_ctf->SetHighestFrequencyForFitting(pixel_size_for_fitting / maximum_resolution);
                }

                // The end point of the BF search is the beginning of the CG search
                for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                    cg_starting_point[counter] = brute_force_search->GetBestValue(counter);
                }

                //
                if ( astigmatism_is_known ) {
                    current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf);
                    if ( find_additional_phase_shift ) {
                        if ( fixed_additional_phase_shift ) {
                            current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                        }
                        else {
                            current_ctf->SetAdditionalPhaseShift(cg_starting_point[1]);
                        }
                    }
                }
                else {
                    current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[1], cg_starting_point[2]);
                    if ( find_additional_phase_shift ) {
                        if ( fixed_additional_phase_shift ) {
                            current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                        }
                        else {
                            current_ctf->SetAdditionalPhaseShift(cg_starting_point[3]);
                        }
                    }
                }
                current_ctf->EnforceConvention( );

                // Remember the best score so far
                best_score_after_initial_phase = -brute_force_search->GetBestScore( );

                delete brute_force_search;
                profile_timing.lap("Setup 2D search optimization");
            }

            // Print out the results of brute force search
            //if (is_running_locally && old_school_input)
            {
                wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
                wxPrintf("%12.2f%12.2f%12.2f%12.5f\n", current_ctf->GetDefocus1( ) * pixel_size_for_fitting, current_ctf->GetDefocus2( ) * pixel_size_for_fitting, current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf, best_score_after_initial_phase);
            }

            // Now we refine in the neighbourhood by using Powell's conjugate gradient algorithm
            if ( is_running_locally && old_school_input ) {
                wxPrintf("\nREFINING CTF PARAMETERS...\n");
                wxPrintf("      DFMID1      DFMID2      ANGAST          CC\n");
            }

            profile_timing.start("Setup 2D search optimization");
            /*
			 * Set up the conjugate gradient minimization of the 2D scoring function
			 */
            if ( astigmatism_is_known ) {
                cg_starting_point[0] = current_ctf->GetDefocus1( );
                if ( find_additional_phase_shift )
                    cg_starting_point[1] = current_ctf->GetAdditionalPhaseShift( );
                if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                    number_of_search_dimensions = 2;
                }
                else {
                    number_of_search_dimensions = 1;
                }
                cg_accuracy[0] = 100.0;
                cg_accuracy[1] = 0.05;
            }
            else {
                cg_accuracy[0]       = 100.0; //TODO: try defocus_search_step  / pix_size_for_fitting / 10.0
                cg_accuracy[1]       = 100.0;
                cg_accuracy[2]       = 0.025;
                cg_accuracy[3]       = 0.05;
                cg_starting_point[0] = current_ctf->GetDefocus1( );
                cg_starting_point[1] = current_ctf->GetDefocus2( );
                if ( slower_search || (! slower_search && follow_1d_search_with_local_2D_brute_force) ) {
                    // we did a search against the 2D power spectrum so we have a better estimate
                    // of the astigmatism angle in the CTF object
                    cg_starting_point[2] = current_ctf->GetAstigmatismAzimuth( );
                }
                else {
                    // all we have right now is the guessed astigmatism angle from the mirror
                    // trick before any CTF fitting was even tried
                    cg_starting_point[2] = estimated_astigmatism_angle / 180.0 * PIf;
                }

                if ( find_additional_phase_shift )
                    cg_starting_point[3] = current_ctf->GetAdditionalPhaseShift( );
                if ( find_additional_phase_shift && ! fixed_additional_phase_shift ) {
                    number_of_search_dimensions = 4;
                }
                else {
                    number_of_search_dimensions = 3;
                }
            }
            profile_timing.lap("Setup 2D search optimization");
            // CG minimization
            profile_timing.start("Peform 2D search optimization");
            comparison_object_2D.SetCTF(*current_ctf);
            conjugate_gradient_minimizer = new ConjugateGradient( );
            conjugate_gradient_minimizer->Init(&CtffindObjectiveFunction, &comparison_object_2D, number_of_search_dimensions, cg_starting_point, cg_accuracy);
            conjugate_gradient_minimizer->Run( );
            profile_timing.lap("Peform 2D search optimization");
            // Remember the results of the refinement
            for ( counter = 0; counter < number_of_search_dimensions; counter++ ) {
                cg_starting_point[counter] = conjugate_gradient_minimizer->GetBestValue(counter);
            }
            if ( astigmatism_is_known ) {
                current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting, known_astigmatism_angle / 180.0 * PIf);
                if ( find_additional_phase_shift ) {
                    if ( fixed_additional_phase_shift ) {
                        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                    }
                    else {
                        current_ctf->SetAdditionalPhaseShift(cg_starting_point[1]);
                    }
                }
            }
            else {
                current_ctf->SetDefocus(cg_starting_point[0], cg_starting_point[1], cg_starting_point[2]);
                if ( find_additional_phase_shift ) {
                    if ( fixed_additional_phase_shift ) {
                        current_ctf->SetAdditionalPhaseShift(minimum_additional_phase_shift);
                    }
                    else {
                        current_ctf->SetAdditionalPhaseShift(cg_starting_point[3]);
                    }
                }
            }
            current_ctf->EnforceConvention( );

            // Print results to the terminal
            if ( is_running_locally && old_school_input ) {
                wxPrintf("%12.2f%12.2f%12.2f%12.5f   Final Values\n", current_ctf->GetDefocus1( ) * pixel_size_for_fitting, current_ctf->GetDefocus2( ) * pixel_size_for_fitting, current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf, -conjugate_gradient_minimizer->GetBestScore( ));
                if ( find_additional_phase_shift ) {
                    wxPrintf("Final phase shift = %0.3f radians\n", current_ctf->GetAdditionalPhaseShift( ));
                }
            }

            final_score = -conjugate_gradient_minimizer->GetBestScore( );
        } // End of test for defocus_is_known

        /*
		 * We're all done with our search & refinement of defocus and phase shift parameter values.
		 * Now onto diagnostics.
		 */
        ctffind_timing.lap("Parameter search");
        ctffind_timing.start("Diagnostics");
        profile_timing.start("Renormalize spectrum");
        // Generate diagnostic image
        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice(debug_file_prefix + "dbg_spec_diag_start.mrc", 1);
        current_output_location = current_micrograph_number;
        average_spectrum->AddConstant(-1.0 * average_spectrum->ReturnAverageOfRealValuesOnEdges( ));

        /*
		 *  Attempt some renormalisations - we want to do this over a range not affected by the central peak or strong Thon rings,
		 *  so as to emphasize the "regular" Thon rings
		 */
        float start_zero               = sqrtf(current_ctf->ReturnSquaredSpatialFrequencyOfAZero(3, current_ctf->GetAstigmatismAzimuth( ), true));
        float finish_zero              = sqrtf(current_ctf->ReturnSquaredSpatialFrequencyOfAZero(4, current_ctf->GetAstigmatismAzimuth( ), true));
        float normalization_radius_min = start_zero * average_spectrum->logical_x_dimension;
        float normalization_radius_max = finish_zero * average_spectrum->logical_x_dimension;

        if ( start_zero > current_ctf->GetHighestFrequencyForFitting( ) || start_zero < current_ctf->GetLowestFrequencyForFitting( ) || finish_zero > current_ctf->GetHighestFrequencyForFitting( ) || finish_zero < current_ctf->GetLowestFrequencyForFitting( ) ) {
            normalization_radius_max = current_ctf->GetHighestFrequencyForFitting( ) * average_spectrum->logical_x_dimension;
            normalization_radius_min = std::max(0.5f * normalization_radius_max, current_ctf->GetLowestFrequencyForFitting( ) * average_spectrum->logical_x_dimension);
        }

        MyDebugAssertTrue(normalization_radius_max > normalization_radius_min, "Bad values for min (%f) and max (%f) normalization radii\n");

        if ( normalization_radius_max - normalization_radius_min > 2.0 ) {
            average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(normalization_radius_min,
                                                                       normalization_radius_max,
                                                                       average, sigma);
            average_spectrum->CircleMask(5.0, true);
            average_spectrum->SetMaximumValueOnCentralCross(average);
            average_spectrum->SetMinimumAndMaximumValues(average - 4.0 * sigma, average + 4.0 * sigma);
            average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(normalization_radius_min,
                                                                       normalization_radius_max,
                                                                       average, sigma);
            average_spectrum->AddConstant(-1.0 * average);
            average_spectrum->MultiplyByConstant(1.0 / sigma);
            average_spectrum->AddConstant(average);
        }

        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice(debug_file_prefix + "dbg_spec_diag_1.mrc", 1);
        profile_timing.lap("Renormalize spectrum");
        profile_timing.start("Compute final 1D spectrum");
        // 1D rotational average
        number_of_bins_in_1d_spectra = int(ceil(average_spectrum->ReturnMaximumDiagonalRadius( )));
        rotational_average->SetupXAxis(0.0, float(number_of_bins_in_1d_spectra) * average_spectrum->fourier_voxel_size_x, number_of_bins_in_1d_spectra);
        rotational_average->ZeroYData( );
        //number_of_averaged_pixels.ZeroYData();
        number_of_averaged_pixels->CopyFrom(rotational_average);
        average_spectrum->Compute1DRotationalAverage(*rotational_average, *number_of_averaged_pixels, true);
        profile_timing.lap("Compute final 1D spectrum");
        // Rotational average, taking astigmatism into account
        Curve equiphase_average_pre_max;
        Curve equiphase_average_post_max;
        if ( compute_extra_stats ) {
            profile_timing.start("Compute EPA");
            number_of_extrema_image->Allocate(average_spectrum->logical_x_dimension, average_spectrum->logical_y_dimension, true);
            ctf_values_image->Allocate(average_spectrum->logical_x_dimension, average_spectrum->logical_y_dimension, true);
            spatial_frequency                     = new double[number_of_bins_in_1d_spectra];
            rotational_average_astig              = new double[number_of_bins_in_1d_spectra];
            rotational_average_astig_renormalized = new double[number_of_bins_in_1d_spectra];
            rotational_average_astig_fit          = new double[number_of_bins_in_1d_spectra];
            number_of_extrema_profile             = new float[number_of_bins_in_1d_spectra];
            ctf_values_profile                    = new float[number_of_bins_in_1d_spectra];
            fit_frc                               = new double[number_of_bins_in_1d_spectra];
            fit_frc_sigma                         = new double[number_of_bins_in_1d_spectra];
            ComputeImagesWithNumberOfExtremaAndCTFValues(current_ctf, number_of_extrema_image, ctf_values_image);
            //ctf_values_image.QuickAndDirtyWriteSlice("dbg_ctf_values.mrc",1);
            if ( dump_debug_files ) {
                average_spectrum->QuickAndDirtyWriteSlice(debug_file_prefix + "dbg_spectrum_before_1dave.mrc", 1);
                number_of_extrema_image->QuickAndDirtyWriteSlice(debug_file_prefix + "dbg_num_extrema.mrc", 1);
            }
            ComputeRotationalAverageOfPowerSpectrum(average_spectrum, current_ctf, number_of_extrema_image, ctf_values_image, number_of_bins_in_1d_spectra, spatial_frequency, rotational_average_astig, rotational_average_astig_fit, rotational_average_astig_renormalized, number_of_extrema_profile, ctf_values_profile);
#ifdef use_epa_rather_than_zero_counting
            ComputeEquiPhaseAverageOfPowerSpectrum(average_spectrum, current_ctf, &equiphase_average_pre_max, &equiphase_average_post_max);
            // Replace the old curve with EPA values
            {
                float current_sq_sf;
                float azimuth_for_1d_plots         = ReturnAzimuthToUseFor1DPlots(current_ctf);
                float defocus_for_1d_plots         = current_ctf->DefocusGivenAzimuth(azimuth_for_1d_plots);
                float sq_sf_of_phase_shift_maximum = current_ctf->ReturnSquaredSpatialFrequencyOfPhaseShiftExtremumGivenDefocus(defocus_for_1d_plots);
                for ( counter = 1; counter < number_of_bins_in_1d_spectra; counter++ ) {
                    current_sq_sf = powf(spatial_frequency[counter], 2);
                    if ( current_sq_sf <= sq_sf_of_phase_shift_maximum ) {
                        rotational_average_astig[counter] = equiphase_average_pre_max.ReturnLinearInterpolationFromX(current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
                    }
                    else {
                        rotational_average_astig[counter] = equiphase_average_post_max.ReturnLinearInterpolationFromX(current_ctf->PhaseShiftGivenSquaredSpatialFrequencyAndDefocus(current_sq_sf, defocus_for_1d_plots));
                    }
                    rotational_average_astig_renormalized[counter] = rotational_average_astig[counter];
                }
                Renormalize1DSpectrumForFRC(number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, rotational_average_astig_fit, number_of_extrema_profile);
            }
#endif
            // Here, do FRC
            int first_fit_bin = 0;
            for ( int bin_counter = number_of_bins_in_1d_spectra - 1; bin_counter >= 0; bin_counter-- ) {
                if ( spatial_frequency[bin_counter] >= current_ctf->GetLowestFrequencyForFitting( ) )
                    first_fit_bin = bin_counter;
            }
            ComputeFRCBetween1DSpectrumAndFit(number_of_bins_in_1d_spectra, rotational_average_astig_renormalized, rotational_average_astig_fit, number_of_extrema_profile, fit_frc, fit_frc_sigma, first_fit_bin);
            profile_timing.lap("Compute EPA");
            profile_timing.start("Detect antialiasing");
            // At what bin does CTF aliasing become problematic?
            last_bin_without_aliasing         = 0;
            int location_of_previous_extremum = 0;
            for ( counter = 1; counter < number_of_bins_in_1d_spectra; counter++ ) {
                if ( number_of_extrema_profile[counter] - number_of_extrema_profile[counter - 1] >= 0.9 ) {
                    // We just reached a new extremum
                    if ( counter - location_of_previous_extremum < 4 ) {
                        last_bin_without_aliasing = location_of_previous_extremum;
                        break;
                    }
                    location_of_previous_extremum = counter;
                }
            }
            if ( is_running_locally && old_school_input && last_bin_without_aliasing != 0 ) {
                wxPrintf("CTF aliasing apparent from %0.1f Angstroms\n", pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]);
            }
            profile_timing.lap("Detect antialiasing");
        }

        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice(debug_file_prefix + "dbg_spec_diag_2.mrc", 1);

        // Until what frequency were CTF rings detected?
        if ( compute_extra_stats ) {
            profile_timing.start("Compute resolution cutoff");
            static float low_threshold              = 0.1;
            static float frc_significance_threshold = 0.5; // In analogy to the usual criterion when comparing experimental results to the atomic model
            static float high_threshold             = 0.66;
            bool         at_last_bin_with_good_fit;
            int          number_of_bins_above_low_threshold          = 0;
            int          number_of_bins_above_significance_threshold = 0;
            int          number_of_bins_above_high_threshold         = 0;
            int          first_bin_to_check                          = 0.1 * number_of_bins_in_1d_spectra;
            MyDebugAssertTrue(first_bin_to_check >= 0 && first_bin_to_check < number_of_bins_in_1d_spectra, "Bad first bin to check\n");
            //wxPrintf("Will only check from bin %i of %i onwards\n", first_bin_to_check, number_of_bins_in_1d_spectra);
            last_bin_with_good_fit = -1;
            for ( counter = first_bin_to_check; counter < number_of_bins_in_1d_spectra; counter++ ) {
                //wxPrintf("On bin %i, fit_frc = %f, rot averate astig = %f\n", counter, fit_frc[counter], rotational_average_astig[counter]);
                at_last_bin_with_good_fit = ((number_of_bins_above_low_threshold > 3) && (fit_frc[counter] < low_threshold)) ||
                                            ((number_of_bins_above_high_threshold > 3) && (fit_frc[counter] < frc_significance_threshold));
                if ( at_last_bin_with_good_fit ) {
                    last_bin_with_good_fit = counter;
                    break;
                }
                // Count number of bins above given thresholds
                if ( fit_frc[counter] > low_threshold )
                    number_of_bins_above_low_threshold++;
                if ( fit_frc[counter] > frc_significance_threshold )
                    number_of_bins_above_significance_threshold++;
                if ( fit_frc[counter] > high_threshold )
                    number_of_bins_above_high_threshold++;
            }
            //wxPrintf("%i bins out of %i checked were above significance threshold\n",number_of_bins_above_significance_threshold,number_of_bins_in_1d_spectra-first_bin_to_check);
            if ( number_of_bins_above_significance_threshold == number_of_bins_in_1d_spectra - first_bin_to_check )
                last_bin_with_good_fit = number_of_bins_in_1d_spectra - 1;
            if ( number_of_bins_above_significance_threshold == 0 )
                last_bin_with_good_fit = 1;
            last_bin_with_good_fit = std::min(last_bin_with_good_fit, number_of_bins_in_1d_spectra);
            profile_timing.lap("Compute resolution cutoff");
        }
        else {
            last_bin_with_good_fit = 1;
        }
#ifdef DEBUG
        //MyDebugAssertTrue(last_bin_with_good_fit >= 0 && last_bin_with_good_fit < number_of_bins_in_1d_spectra,"Did not find last bin with good fit: %i", last_bin_with_good_fit);
        if ( ! (last_bin_with_good_fit >= 0 && last_bin_with_good_fit < number_of_bins_in_1d_spectra) ) {
            wxPrintf("WARNING: Did not find last bin with good fit: %i\n", last_bin_with_good_fit);
        }
#else
        if ( last_bin_with_good_fit < 1 && last_bin_with_good_fit >= number_of_bins_in_1d_spectra ) {
            last_bin_with_good_fit = 1;
        }
#endif
        // Start of Node fitting
        CTFNodeFitOuput node_output;
        if ( fit_nodes ) {
            MyDebugPrint("Estimating thickness: ");
            profile_timing.start("Thickness estimation");
            CTFNodeFitInput node_fit_input = {
                    current_ctf,
                    last_bin_with_good_fit,
                    number_of_bins_in_1d_spectra,
                    pixel_size_for_fitting,
                    spatial_frequency,
                    rotational_average_astig,
                    rotational_average_astig_fit,
                    equiphase_average_pre_max,
                    equiphase_average_post_max,
                    &comparison_object_1D,
                    &comparison_object_2D,
                    fit_nodes_1D_brute_force,
                    fit_nodes_2D_refine,
                    fit_nodes_low_resolution_limit,
                    fit_nodes_high_resolution_limit,
                    fit_frc,
                    fit_frc_sigma,
                    average_spectrum,
                    dump_debug_files,
                    debug_file_prefix,
                    fit_nodes_use_rounded_square,
                    fit_nodes_downweight_nodes};

            node_output = fit_thickness_nodes(&node_fit_input);
            MyDebugPrint("Got out of the function\n");
            last_bin_with_good_fit = node_output.last_bin_with_good_fit;
            profile_timing.lap("Thickness estimation");
            MyDebugPrint("Done!\n");
        }
        wxPrintf("Or here\n");

        // Prepare output diagnostic image
        //average_spectrum->AddConstant(- average_spectrum->ReturnAverageOfRealValuesOnEdges()); // this used to be done in OverlayCTF / CTFOperation in the Fortran code
        //average_spectrum.QuickAndDirtyWriteSlice("dbg_spec_diag_3.mrc",1);
        if ( dump_debug_files )
            average_spectrum->QuickAndDirtyWriteSlice(debug_file_prefix + "dbg_spec_before_rescaling.mrc", 1);
        profile_timing.start("Write diagnostic image");
        if ( compute_extra_stats && ! fit_nodes ) {
            RescaleSpectrumAndRotationalAverage(average_spectrum, number_of_extrema_image, ctf_values_image, number_of_bins_in_1d_spectra, spatial_frequency, rotational_average_astig, rotational_average_astig_fit, number_of_extrema_profile, ctf_values_profile, last_bin_without_aliasing, last_bin_with_good_fit);
        }
        //average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_thresholding.mrc",1);

        normalization_radius_max = std::max(normalization_radius_max, float(average_spectrum->logical_x_dimension * spatial_frequency[last_bin_with_good_fit]));
        average_spectrum->ComputeAverageAndSigmaOfValuesInSpectrum(normalization_radius_min,
                                                                   normalization_radius_max,
                                                                   average, sigma);

        average_spectrum->SetMinimumAndMaximumValues(average - sigma, average + 2.0 * sigma);
        //average_spectrum->QuickAndDirtyWriteSlice("dbg_spec_before_overlay.mrc",1);
        OverlayCTF(average_spectrum, current_ctf, number_of_extrema_image, ctf_values_image, number_of_bins_in_1d_spectra, spatial_frequency, rotational_average_astig, number_of_extrema_profile, ctf_values_profile, &equiphase_average_pre_max, &equiphase_average_post_max, fit_nodes);

        average_spectrum->WriteSlice(&output_diagnostic_file, current_output_location);
        output_diagnostic_file.SetDensityStatistics(average_spectrum->ReturnMinimumValue( ), average_spectrum->ReturnMaximumValue( ), average_spectrum->ReturnAverageOfRealValues( ), 0.1);
        profile_timing.lap("Write diagnostic image");
        // Keep track of time
        ctffind_timing.lap("Diagnostics");
        ctffind_timing.print_times( );
        profile_timing.print_times( );
        // Print more detailed results to terminal
        if ( is_running_locally && number_of_micrographs == 1 ) {

            wxPrintf("\n\nEstimated defocus values        : %0.2f , %0.2f Angstroms\nEstimated azimuth of astigmatism: %0.2f degrees\n", current_ctf->GetDefocus1( ) * pixel_size_for_fitting, current_ctf->GetDefocus2( ) * pixel_size_for_fitting, current_ctf->GetAstigmatismAzimuth( ) / PIf * 180.0);
            if ( find_additional_phase_shift ) {
                wxPrintf("Additional phase shift          : %0.3f degrees (%0.3f radians) (%0.3f PIf)\n", current_ctf->GetAdditionalPhaseShift( ) / PIf * 180.0, current_ctf->GetAdditionalPhaseShift( ), current_ctf->GetAdditionalPhaseShift( ) / PIf);
            }
            if ( determine_tilt )
                wxPrintf("Tilt_axis, tilt angle           : %0.2f , %0.2f degrees\n", tilt_axis, tilt_angle);
            wxPrintf("Score                           : %0.5f\n", final_score);
            wxPrintf("Pixel size for fitting          : %0.3f Angstroms\n", pixel_size_for_fitting);
            if ( compute_extra_stats ) {
                wxPrintf("Thon rings with good fit up to  : %0.1f Angstroms\n", pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit]);
                if ( last_bin_without_aliasing != 0 ) {
                    wxPrintf("CTF aliasing apparent from      : %0.1f Angstroms\n", pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]);
                }
                else {
                    wxPrintf("Did not detect CTF aliasing\n");
                }
            }
        }
        // Warn the user if significant aliasing occurred within the fit range
        if ( compute_extra_stats && last_bin_without_aliasing != 0 && spatial_frequency[last_bin_without_aliasing] < current_ctf->GetHighestFrequencyForFitting( ) ) {
            if ( is_running_locally && number_of_micrographs == 1 ) {
                MyPrintfRed("Warning: CTF aliasing occurred within your CTF fitting range. Consider computing a larger spectrum (current size = %i).\n", box_size);
            }
            else {
                //SendInfo(wxString::Format("Warning: for image %s (location %i of %i), CTF aliasing occurred within the CTF fitting range. Consider computing a larger spectrum (current size = %i)\n",input_filename,current_micrograph_number, number_of_micrographs,box_size));
            }
        }
        if ( is_running_locally ) {
            // Write out results to a summary file
            values_to_write_out[0] = current_micrograph_number;
            values_to_write_out[1] = current_ctf->GetDefocus1( ) * pixel_size_for_fitting;
            values_to_write_out[2] = current_ctf->GetDefocus2( ) * pixel_size_for_fitting;
            values_to_write_out[3] = current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf;
            values_to_write_out[4] = current_ctf->GetAdditionalPhaseShift( );
            values_to_write_out[5] = final_score;
            if ( compute_extra_stats ) {
                values_to_write_out[6] = pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit];
            }
            else {
                values_to_write_out[6] = 0.0;
            }
            output_text->WriteLine(values_to_write_out);

            if ( (! old_school_input) && number_of_micrographs > 1 && is_running_locally )
                my_progress_bar->Update(current_micrograph_number);
        }
        // Write out avrot
        // TODO: add to the output a line with non-normalized avrot, so that users can check for things like ice crystal reflections
        if ( compute_extra_stats ) {
            if ( current_micrograph_number == 1 ) {
                output_text_avrot = new NumericTextFile(output_text_fn, OPEN_TO_WRITE, number_of_bins_in_1d_spectra);
                output_text_avrot->WriteCommentLine("# Output from CTFFind version %s, run on %s\n", ctffind_version.c_str( ), wxDateTime::Now( ).FormatISOCombined(' ').ToUTF8( ).data( ));
                output_text_avrot->WriteCommentLine("# Input file: %s ; Number of micrographs: %i\n", input_filename.c_str( ), number_of_micrographs);
                output_text_avrot->WriteCommentLine("# Pixel size: %0.3f Angstroms ; acceleration voltage: %0.1f keV ; spherical aberration: %0.2f mm ; amplitude contrast: %0.2f\n", pixel_size_of_input_image, acceleration_voltage, spherical_aberration, amplitude_contrast);
                output_text_avrot->WriteCommentLine("# Box size: %i pixels ; min. res.: %0.1f Angstroms ; max. res.: %0.1f Angstroms ; min. def.: %0.1f um; max. def. %0.1f um; num. frames averaged: %i\n", box_size, minimum_resolution, maximum_resolution, minimum_defocus, maximum_defocus, number_of_frames_to_average);
                output_text_avrot->WriteCommentLine("# 6 lines per micrograph: #1 - spatial frequency (1/Angstroms); #2 - 1D rotational average of spectrum (assuming no astigmatism); #3 - 1D rotational average of spectrum; #4 - CTF fit; #5 - cross-correlation between spectrum and CTF fit; #6 - 2sigma of expected cross correlation of noise\n");
            }
            spatial_frequency_in_reciprocal_angstroms = new double[number_of_bins_in_1d_spectra];
            for ( counter = 0; counter < number_of_bins_in_1d_spectra; counter++ ) {
                spatial_frequency_in_reciprocal_angstroms[counter] = spatial_frequency[counter] / pixel_size_for_fitting;
            }
            output_text_avrot->WriteLine(spatial_frequency_in_reciprocal_angstroms);
            output_text_avrot->WriteLine(rotational_average->data_y);
            output_text_avrot->WriteLine(rotational_average_astig);
            output_text_avrot->WriteLine(rotational_average_astig_fit);
            output_text_avrot->WriteLine(fit_frc);
            output_text_avrot->WriteLine(fit_frc_sigma);
            delete[] spatial_frequency_in_reciprocal_angstroms;
        }
        wxPrintf("Cleaning up comparison object... do not\n");
        //delete comparison_object_2D;
    } // End of loop over micrographs
    wxPrintf("I'm back...\n");
    if ( is_running_locally && (! old_school_input) && number_of_micrographs > 1 ) {
        delete my_progress_bar;
        wxPrintf("\n");
    }

    // Tell the user where the outputs are
    if ( is_running_locally ) {

        wxPrintf("\n\nSummary of results                          : %s\n", output_text->ReturnFilename( ));
        wxPrintf("Diagnostic images                           : %s\n", output_diagnostic_filename);
        if ( compute_extra_stats ) {
            wxPrintf("Detailed results, including 1D fit profiles : %s\n", output_text_avrot->ReturnFilename( ));
            wxPrintf("Use this command to plot 1D fit profiles    : ctffind_plot_results.sh %s\n", output_text_avrot->ReturnFilename( ));
        }

        wxPrintf("\n\n");
    }

    // Send results back
    wxPrintf("Sending results back...\n");
    float results_array[11];
    results_array[0] = current_ctf->GetDefocus1( ) * pixel_size_for_fitting; // Defocus 1 (Angstroms)
    results_array[1] = current_ctf->GetDefocus2( ) * pixel_size_for_fitting; // Defocus 2 (Angstroms)
    results_array[2] = current_ctf->GetAstigmatismAzimuth( ) * 180.0 / PIf; // Astigmatism angle (degrees)
    results_array[3] = current_ctf->GetAdditionalPhaseShift( ); // Additional phase shift (e.g. from phase plate) (radians)
    results_array[4] = final_score; // CTFFIND score

    if ( last_bin_with_good_fit == 0 ) {
        results_array[5] = 0.0; //	A value of 0.0 indicates that the calculation to determine the goodness of fit failed for some reason
    }
    else {
        results_array[5] = pixel_size_for_fitting / spatial_frequency[last_bin_with_good_fit]; //	The resolution (Angstroms) up to which Thon rings are well fit by the CTF
    }
    if ( last_bin_without_aliasing == 0 ) {
        results_array[6] = 0.0; // 	A value of 0.0 indicates that no aliasing was detected
    }
    else {
        results_array[6] = pixel_size_for_fitting / spatial_frequency[last_bin_without_aliasing]; //	The resolution (Angstroms) at which aliasing was just detected
    }

    results_array[7]  = average_spectrum->ReturnIcinessOfSpectrum(pixel_size_for_fitting);
    results_array[8]  = tilt_angle;
    results_array[9]  = tilt_axis;
    results_array[10] = current_ctf->GetSampleThickness( ) * pixel_size_for_fitting; // Sample thickness (Angstroms)
    my_result.SetResult(11, results_array);
    // Cleanup
    wxPrintf("Cleaning up memory...\n");
    delete current_ctf;
    delete average_spectrum;
    delete average_spectrum_masked;
    delete current_power_spectrum;
    delete current_input_image;
    delete current_input_image_square;
    delete temp_image;
    delete sum_image;
    delete resampled_power_spectrum;
    delete number_of_extrema_image;
    delete ctf_values_image;
    delete gain;
    delete dark;
    delete[] values_to_write_out;
    if ( is_running_locally )
        delete output_text;
    if ( compute_extra_stats ) {
        delete[] spatial_frequency;
        delete[] rotational_average_astig;
        delete[] rotational_average_astig_renormalized;
        delete[] rotational_average_astig_fit;
        delete[] number_of_extrema_profile;
        delete[] ctf_values_profile;
        delete[] fit_frc;
        delete[] fit_frc_sigma;
        delete output_text_avrot;
    }
    if ( ! defocus_is_known )
        delete conjugate_gradient_minimizer;

    delete number_of_averaged_pixels;
    delete rotational_average;

    // Return
    return true;
}
