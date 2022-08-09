
class TemplateMatchFoundPeakInfo {
  public:
    int   peak_number;
    float x_pos;
    float y_pos;
    float psi;
    float theta;
    float phi;
    float defocus;
    float pixel_size;
    float peak_height;
    int   original_peak_number;
    int   new_peak_number;
};

WX_DECLARE_OBJARRAY(TemplateMatchFoundPeakInfo, ArrayOfTemplateMatchFoundPeakInfos);

class TemplateMatchJobResults {
  public:
    TemplateMatchJobResults( );

    wxString job_name;
    int      job_type;
    long     input_job_id;
    long     job_id;
    long     datetime_of_run;
    long     image_asset_id;
    long     ref_volume_asset_id;
    wxString symmetry;
    float    pixel_size;
    float    voltage;
    float    spherical_aberration;
    float    amplitude_contrast;
    float    defocus1;
    float    defocus2;
    float    defocus_angle;
    float    phase_shift;
    float    low_res_limit;
    float    high_res_limit;
    float    out_of_plane_step;
    float    in_plane_step;
    float    defocus_search_range;
    float    defocus_step;
    float    pixel_size_search_range;
    float    pixel_size_step;
    float    mask_radius;
    float    min_peak_radius;
    float    xy_change_threshold;
    bool     exclude_above_xy_threshold;

    wxString mip_filename;
    wxString scaled_mip_filename;
    wxString psi_filename;
    wxString theta_filename;
    wxString phi_filename;
    wxString defocus_filename;
    wxString pixel_size_filename;
    wxString histogram_filename;
    wxString projection_result_filename;
    wxString avg_filename;
    wxString std_filename;

    float refinement_threshold;
    float used_threshold;
    float reference_box_size_in_angstroms;

    ArrayOfTemplateMatchFoundPeakInfos found_peaks;
    ArrayOfTemplateMatchFoundPeakInfos peak_changes;
};

WX_DECLARE_OBJARRAY(TemplateMatchJobResults, ArrayOfTemplateMatchJobResults);
