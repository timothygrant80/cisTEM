# Additional programs configuration for cisTEM
#
# This file defines optional programs that can be built with cisTEM.
# By default, all programs are built unless --disable-build-all is specified.
#
# Configuration flags:
#   --disable-build-all              : Only build essential programs (GUI requirements)
#   --enable-build-<program-name>    : Build specific program when --disable-build-all is set
#
# To add a new optional program:
#   1. Add a line in the main macro below:
#      CISTEM_OPTIONAL_PROGRAM([program_internal_name], [ENABLE_PROGRAMNAME], [display_name])
#   2. Add corresponding AM_CONDITIONAL block in src/Makefile.am:
#      if ENABLE_PROGRAMNAME_AM
#      bin_PROGRAMS += your_program
#      endif
#
# Example: To add a program called "my_filter":
#   1. Add to this file:
#      CISTEM_OPTIONAL_PROGRAM([my_filter], [ENABLE_MYFILTER], [my_filter])
#   2. User can then configure with:
#      ./configure --disable-build-all --enable-build-my-filter

# Define a reusable macro for optional programs
# Usage: CISTEM_OPTIONAL_PROGRAM([program_name], [CONDITIONAL_NAME], [display_name])
AC_DEFUN([CISTEM_OPTIONAL_PROGRAM], [
    AS_IF([test "x$build_all" = "xyes"],
          [build_$1="yes"],
          [build_$1="no"])

    AC_ARG_ENABLE([build-$1],
        AS_HELP_STRING([--enable-build-$1], [build $3 @<:@default="no"@:>@]),
        [AS_IF([test "x$enableval" = "xyes"],
               [build_$1=yes
                AC_MSG_NOTICE([Building $3])])])

    AM_CONDITIONAL([$2_AM], [test "x$build_$1" = "xyes"])
])

# Main macro for configuring all optional programs
AC_DEFUN([NON_ESSENTIAL_PROGRAMS_TO_BE_COMPILED],
[
    AC_MSG_NOTICE([Checking for additional programs])

    # Check if we should build all programs
    build_all="yes"
    AC_ARG_ENABLE(build-all,
        AS_HELP_STRING([--disable-build-all], [only build essential and requested programs]),
        [AS_IF([test "x$enable_build_all" = "xno"],
               [build_all="no"
                AC_MSG_NOTICE([Building only essential and requested programs])])])

    AC_MSG_NOTICE([Checking for additional programs 1])

    # Define all optional programs using the macro
    # Format: CISTEM_OPTIONAL_PROGRAM([internal_name], [CONDITIONAL_NAME], [display_name])

    CISTEM_OPTIONAL_PROGRAM([applyctf], [ENABLE_APPLYCTF], [applyctf])
    CISTEM_OPTIONAL_PROGRAM([project3d], [ENABLE_PROJECT3D], [project3D])
    CISTEM_OPTIONAL_PROGRAM([calc_occ], [ENABLE_CALCOCC], [calc_occ])
    CISTEM_OPTIONAL_PROGRAM([remove_outlier_pixels], [ENABLE_REMOVEOUTLIERPIXELS], [remove_outlier_pixels])
    CISTEM_OPTIONAL_PROGRAM([resize], [ENABLE_RESIZE], [resize])
    CISTEM_OPTIONAL_PROGRAM([resample], [ENABLE_RESAMPLE], [resample])
    CISTEM_OPTIONAL_PROGRAM([reset_mrc_header], [ENABLE_RESETMRCHEADER], [reset_mrc_header])
    CISTEM_OPTIONAL_PROGRAM([estimate_dataset_ssnr], [ENABLE_ESTIMATEDATASETSSNR], [estimate_dataset_ssnr])
    CISTEM_OPTIONAL_PROGRAM([montage], [ENABLE_MONTAGE], [montage])
    CISTEM_OPTIONAL_PROGRAM([extract_particles], [ENABLE_EXTRACTPARTICLES], [extract_particles])
    CISTEM_OPTIONAL_PROGRAM([sum_all_mrc_files], [ENABLE_SUMALLMRCFILES], [sum_all_mrc_files])
    CISTEM_OPTIONAL_PROGRAM([sum_all_tif_files], [ENABLE_SUMALLTIFFILES], [sum_all_tif_files])
    CISTEM_OPTIONAL_PROGRAM([sum_all_eer_files], [ENABLE_SUMALLEERFILES], [sum_all_eer_files])
    CISTEM_OPTIONAL_PROGRAM([apply_gain_ref], [ENABLE_APPLYGAINREF], [apply_gain_ref])
    CISTEM_OPTIONAL_PROGRAM([scale_with_mask], [ENABLE_SCALEWITHMASK], [scale_with_mask])
    CISTEM_OPTIONAL_PROGRAM([mag_distortion_correct], [ENABLE_MAGDISTORTIONCORRECT], [mag_distortion_correct])
    CISTEM_OPTIONAL_PROGRAM([apply_mask], [ENABLE_APPLYMASK], [apply_mask])
    CISTEM_OPTIONAL_PROGRAM([convert_tif_to_mrc], [ENABLE_CONVERTTIFTOMRC], [convert_tif_to_mrc])
    CISTEM_OPTIONAL_PROGRAM([remove_inf_and_nan], [ENABLE_REMOVEINFANDNAN], [remove_inf_and_nan])
    CISTEM_OPTIONAL_PROGRAM([make_orth_views], [ENABLE_MAKEORTHVIEWS], [make_orth_views])
    CISTEM_OPTIONAL_PROGRAM([sharpen_map], [ENABLE_SHARPENMAP], [sharpen_map])
    CISTEM_OPTIONAL_PROGRAM([calculate_fsc], [ENABLE_CALCULATEFSC], [calculate_fsc])
    CISTEM_OPTIONAL_PROGRAM([make_size_map], [ENABLE_MAKESIZEMAP], [make_size_map])
    CISTEM_OPTIONAL_PROGRAM([convert_par_to_star], [ENABLE_CONVERTPARTOSTAR], [convert_par_to_star])
    CISTEM_OPTIONAL_PROGRAM([subtract_from_stack], [ENABLE_SUBTRACTFROMSTACK], [subtract_from_stack])
    CISTEM_OPTIONAL_PROGRAM([binarize], [ENABLE_BINARIZE], [binarize])
    CISTEM_OPTIONAL_PROGRAM([move_volume_xyz], [ENABLE_MOVEVOLUMEXYZ], [move_volume_xyz])
    CISTEM_OPTIONAL_PROGRAM([symmetry_expand_stack_and_par], [ENABLE_SYMMETRYEXPANDSTACKANDPAR], [symmetry_expand_stack_and_par])
    CISTEM_OPTIONAL_PROGRAM([subtract_two_stacks], [ENABLE_SUBTRACTTWOSTACKS], [subtract_two_stacks])
    CISTEM_OPTIONAL_PROGRAM([add_two_stacks], [ENABLE_ADDTWOSTACKS], [add_two_stacks])
    CISTEM_OPTIONAL_PROGRAM([multiply_two_stacks], [ENABLE_MULTIPLYTWOSTACKS], [multiply_two_stacks])
    CISTEM_OPTIONAL_PROGRAM([divide_two_stacks], [ENABLE_DIVIDETWOSTACKS], [divide_two_stacks])
    CISTEM_OPTIONAL_PROGRAM([invert_stack], [ENABLE_INVERTSTACK], [invert_stack])
    CISTEM_OPTIONAL_PROGRAM([align_coordinates], [ENABLE_ALIGNCOORDINATES], [align_coordinates])
    CISTEM_OPTIONAL_PROGRAM([align_symmetry], [ENABLE_ALIGNSYMMETRY], [align_symmetry])
    CISTEM_OPTIONAL_PROGRAM([find_dqe], [ENABLE_FINDDQE], [find_dqe])
    CISTEM_OPTIONAL_PROGRAM([combine_via_max], [ENABLE_COMBINEVIAMAX], [combine_via_max])
    CISTEM_OPTIONAL_PROGRAM([remove_relion_stripes], [ENABLE_REMOVERELIONSTRIPES], [remove_relion_stripes])
    CISTEM_OPTIONAL_PROGRAM([create_mask], [ENABLE_CREATEMASK], [create_mask])
    CISTEM_OPTIONAL_PROGRAM([invert_hand], [ENABLE_INVERTHAND], [invert_hand])
    CISTEM_OPTIONAL_PROGRAM([append_stacks], [ENABLE_APPENDSTACKS], [append_stacks])
    CISTEM_OPTIONAL_PROGRAM([convert_star_to_binary], [ENABLE_CONVERTSTARTOBINARY], [convert_star_to_binary])
    CISTEM_OPTIONAL_PROGRAM([convert_binary_to_star], [ENABLE_CONVERTBINARYTOSTAR], [convert_binary_to_star])
    CISTEM_OPTIONAL_PROGRAM([convert_eer_to_mrc], [ENABLE_CONVERTEERTOMRC], [convert_eer_to_mrc])
    CISTEM_OPTIONAL_PROGRAM([azimuthal_average], [ENABLE_AZIMUTHALAVERAGE], [azimuthal_average])
    CISTEM_OPTIONAL_PROGRAM([normalize_stack], [ENABLE_NORMALIZESTACK], [normalize_stack])
    CISTEM_OPTIONAL_PROGRAM([print_stack_statistics], [ENABLE_PRINTSTACKSTATISTICS], [print_stack_statistics])
    CISTEM_OPTIONAL_PROGRAM([combine_stacks_by_star], [ENABLE_COMBINESTACKSBYSTAR], [combine_stacks_by_star])
    CISTEM_OPTIONAL_PROGRAM([measure_template_bias], [ENABLE_MEASURETEMPLATEBIAS], [measure_template_bias])
    CISTEM_OPTIONAL_PROGRAM([align_nmr_spectra], [ENABLE_ALIGNNMRSPECTRA], [align_nmr_spectra])
    CISTEM_OPTIONAL_PROGRAM([correlate_nmr_spectra], [ENABLE_CORRELATENMRSPECTRA], [correlate_nmr_spectra])
    CISTEM_OPTIONAL_PROGRAM([filter_images], [ENABLE_FILTERIMAGES], [filter_images])

    # Special case: calculate_template_pvalue needs Eigen library check
    use_Eigen="no"
    want_calculate_template_pvalue="no"

    AS_IF([test "x$build_all" = "xyes"],
          [want_calculate_template_pvalue="yes"],
          [want_calculate_template_pvalue="no"])

    AC_ARG_ENABLE(build-calculate-template-pvalue,
        AS_HELP_STRING([--enable-build-calculate-template-pvalue], [build calculate_template_pvalue @<:@default="no"@:>@]),
        [AS_IF([test "x$enableval" = "xyes"],
               [want_calculate_template_pvalue="yes"])])

    # Only check for Eigen if we actually want to build calculate_template_pvalue
    AS_IF([test "x$want_calculate_template_pvalue" = "xyes"],
          [AC_MSG_NOTICE([Checking for Eigen v3.4.0 or later])
           AC_CHECK_FILE("$TOPSRCDIR/include/Eigen/Dense",
                         [use_Eigen="yes"],
                         [use_Eigen="no"])

           AS_IF([test "x$use_Eigen" = "xyes"],
                 [build_calculate_template_pvalue="yes"
                  AC_MSG_NOTICE([Building calculate_template_pvalue])],
                 [build_calculate_template_pvalue="no"
                  AC_MSG_NOTICE([Eigen is required to build calculate_template_pvalue. Please install Eigen v3.4.0 or configure without --enable-build-calculate-template-pvalue])])],
          [build_calculate_template_pvalue="no"])

    AM_CONDITIONAL([ENABLE_CALCULATETEMPLATEPVALUE_AM], [test "x$build_calculate_template_pvalue" = "xyes"])
])

dnl COMMENTED EXAMPLE OF ORIGINAL PATTERN (DO NOT USE - SHOWN FOR REFERENCE ONLY)
dnl This shows how each program was originally defined with ~12 lines of repetitive code.
dnl The new CISTEM_OPTIONAL_PROGRAM macro above replaces all this boilerplate.
dnl
dnl AS_IF([test "x$build_all" = "xyes"], [build_apply_ctf="yes"], [build_apply_ctf="no"])
dnl AC_ARG_ENABLE(build-applyctf, AS_HELP_STRING([--enable-build-applyctf],[build applyctf  [default="no"]]),[
dnl   if test "x$enableval" = "xyes"; then
dnl     build_apply_ctf=yes
dnl     AC_MSG_NOTICE([Building applyctf])
dnl   fi
dnl   ])
dnl AM_CONDITIONAL([ENABLE_APPLYCTF_AM], [test "x$build_apply_ctf" = "xyes"])
dnl
dnl NOTE: one newline is needed at the end of this file for autoconf to work correctly
