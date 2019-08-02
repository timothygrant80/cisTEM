#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfcisTEMParameterLines);

cisTEMParameterLine::cisTEMParameterLine()
{
	SetAllToZero();
}

cisTEMParameterMask::cisTEMParameterMask()
{
	SetAllToTrue();
}

void cisTEMParameterMask::SetAllToTrue()
{
	position_in_stack = true;
	image_is_active = true;
	psi = true;
	theta = true;
	phi = true;
	x_shift = true;
	y_shift = true;
	defocus_1 = true;
	defocus_2 = true;
	defocus_angle = true;
	phase_shift = true;
	occupancy = true;
	logp = true;
	sigma = true;
	score = true;
	score_change= true;
	pixel_size = true;
	microscope_voltage_kv = true;
	microscope_spherical_aberration_mm = true;
	amplitude_contrast = true;
	beam_tilt_x = true;
	beam_tilt_y = true;
	image_shift_x = true;
	image_shift_y = true;
	stack_filename = true;
	original_image_filename = true;
	reference_3d_filename = true;
	best_2d_class = true;
	beam_tilt_group = true;
}

void cisTEMParameterMask::SetAllToFalse()
{
	position_in_stack = false;
	image_is_active = false;
	psi = false;
	theta = false;
	phi = false;
	x_shift = false;
	y_shift = false;
	defocus_1 = false;
	defocus_2 = false;
	defocus_angle = false;
	phase_shift = false;
	occupancy = false;
	logp = false;
	sigma = false;
	score = false;
	score_change= false;
	pixel_size = false;
	microscope_voltage_kv = false;
	microscope_spherical_aberration_mm = false;
	amplitude_contrast = false;
	beam_tilt_x = false;
	beam_tilt_y = false;
	image_shift_x = false;
	image_shift_y = false;
	stack_filename = false;
	original_image_filename = false;
	reference_3d_filename = false;
	best_2d_class = false;
	beam_tilt_group = false;
}

void cisTEMParameterMask::SetActiveParameters(long parameters_to_set)
{
	position_in_stack = ((parameters_to_set & POSITION_IN_STACK) == POSITION_IN_STACK);
	image_is_active = ((parameters_to_set & IMAGE_IS_ACTIVE) == IMAGE_IS_ACTIVE);
	psi = ((parameters_to_set & PSI) == PSI);
	theta = ((parameters_to_set & THETA) == THETA);
	phi = ((parameters_to_set & PHI) == PHI);
	x_shift = ((parameters_to_set & X_SHIFT) == X_SHIFT);
	y_shift = ((parameters_to_set & Y_SHIFT) == Y_SHIFT);
	defocus_1 = ((parameters_to_set & DEFOCUS_1) == DEFOCUS_1);
	defocus_2 = ((parameters_to_set & DEFOCUS_2) == DEFOCUS_2);
	defocus_angle = ((parameters_to_set & DEFOCUS_ANGLE) == DEFOCUS_ANGLE);
	phase_shift = ((parameters_to_set & PHASE_SHIFT) == PHASE_SHIFT);
	occupancy = ((parameters_to_set & OCCUPANCY) == OCCUPANCY);
	logp = ((parameters_to_set & LOGP) == LOGP);
	sigma = ((parameters_to_set & SIGMA) == SIGMA);
	score = ((parameters_to_set & SCORE) == SCORE);
	score_change= ((parameters_to_set & SCORE_CHANGE) == SCORE_CHANGE);
	pixel_size = ((parameters_to_set & PIXEL_SIZE) == PIXEL_SIZE);
	microscope_voltage_kv = ((parameters_to_set & MICROSCOPE_VOLTAGE) == MICROSCOPE_VOLTAGE);
	microscope_spherical_aberration_mm = ((parameters_to_set & MICROSCOPE_CS) == MICROSCOPE_CS);
	amplitude_contrast = ((parameters_to_set & AMPLITUDE_CONTRAST) == AMPLITUDE_CONTRAST);
	beam_tilt_x = ((parameters_to_set & BEAM_TILT_X) == BEAM_TILT_X);
	beam_tilt_y = ((parameters_to_set & BEAM_TILT_Y) == BEAM_TILT_Y);
	image_shift_x = ((parameters_to_set & IMAGE_SHIFT_X) == IMAGE_SHIFT_X);
	image_shift_y = ((parameters_to_set & IMAGE_SHIFT_Y) == IMAGE_SHIFT_Y);
	stack_filename = ((parameters_to_set & STACK_FILENAME) == STACK_FILENAME);
	original_image_filename = ((parameters_to_set & ORIGINAL_IMAGE_FILENAME) == ORIGINAL_IMAGE_FILENAME);
	reference_3d_filename = ((parameters_to_set & REFERENCE_3D_FILENAME) == REFERENCE_3D_FILENAME);
	best_2d_class = ((parameters_to_set & BEST_2D_CLASS) == BEST_2D_CLASS);
	beam_tilt_group = ((parameters_to_set & BEAM_TILT_GROUP) == BEAM_TILT_GROUP);
}



/* Should never be needed actually
void cisTEMParameterLine::SwapPsiAndPhi()
{
	float temp_float = psi;
	psi = phi;
	phi = temp_float;
}*/

void cisTEMParameterLine::Add(cisTEMParameterLine &line_to_add)
{
	position_in_stack += line_to_add.position_in_stack;
	image_is_active += line_to_add.image_is_active;
	psi += line_to_add.psi;
	theta += line_to_add.theta;
	phi += line_to_add.phi;
	x_shift += line_to_add.x_shift;
	y_shift += line_to_add.y_shift;
	defocus_1 += line_to_add.defocus_1;
	defocus_2 += line_to_add.defocus_2;
	defocus_angle += line_to_add.defocus_angle;
	phase_shift += line_to_add.phase_shift;
	occupancy += line_to_add.occupancy;
	logp += line_to_add.logp;
	sigma += line_to_add.sigma;
	score += line_to_add.score;
	score_change += line_to_add.score_change;
	pixel_size += line_to_add.pixel_size;
	microscope_voltage_kv += line_to_add.microscope_voltage_kv;
	microscope_spherical_aberration_mm += line_to_add.microscope_spherical_aberration_mm;
	amplitude_contrast += line_to_add.amplitude_contrast;
	beam_tilt_x += line_to_add.beam_tilt_x;
	beam_tilt_y += line_to_add.beam_tilt_y;
	image_shift_x += line_to_add.image_shift_x;
	image_shift_y += line_to_add.image_shift_y;

	// not adding filenames or groups as it doesn't make sense
}

void cisTEMParameterLine::Subtract(cisTEMParameterLine &line_to_add)
{
	position_in_stack -= line_to_add.position_in_stack;
	image_is_active -= line_to_add.image_is_active;
	psi -= line_to_add.psi;
	theta -= line_to_add.theta;
	phi -= line_to_add.phi;
	x_shift -= line_to_add.x_shift;
	y_shift -= line_to_add.y_shift;
	defocus_1 -= line_to_add.defocus_1;
	defocus_2 -= line_to_add.defocus_2;
	defocus_angle -= line_to_add.defocus_angle;
	phase_shift -= line_to_add.phase_shift;
	occupancy -= line_to_add.occupancy;
	logp -= line_to_add.logp;
	sigma -= line_to_add.sigma;
	score -= line_to_add.score;
	score_change -= line_to_add.score_change;
	pixel_size -= line_to_add.pixel_size;
	microscope_voltage_kv -= line_to_add.microscope_voltage_kv;
	microscope_spherical_aberration_mm -= line_to_add.microscope_spherical_aberration_mm;
	amplitude_contrast -= line_to_add.amplitude_contrast;
	beam_tilt_x -= line_to_add.beam_tilt_x;
	beam_tilt_y -= line_to_add.beam_tilt_y;
	image_shift_x -= line_to_add.image_shift_x;
	image_shift_y -= line_to_add.image_shift_y;

	// not adding filenames or groups as it doesn't make sense
}

void cisTEMParameterLine::AddSquare(cisTEMParameterLine &line_to_add)
{
	position_in_stack += powf(line_to_add.position_in_stack, 2);
	image_is_active += powf(line_to_add.image_is_active, 2);
	psi += powf(line_to_add.psi, 2);
	theta += powf(line_to_add.theta, 2);
	phi += powf(line_to_add.phi, 2);
	x_shift += powf(line_to_add.x_shift, 2);
	y_shift += powf(line_to_add.y_shift, 2);
	defocus_1 += powf(line_to_add.defocus_1, 2);
	defocus_2 += powf(line_to_add.defocus_2, 2);
	defocus_angle += powf(line_to_add.defocus_angle, 2);
	phase_shift += powf(line_to_add.phase_shift, 2);
	occupancy += powf(line_to_add.occupancy, 2);
	logp += powf(line_to_add.logp, 2);
	sigma += powf(line_to_add.sigma, 2);
	score += powf(line_to_add.score, 2);
	score_change += powf(line_to_add.score_change, 2);
	pixel_size += powf(line_to_add.pixel_size, 2);
	microscope_voltage_kv += powf(line_to_add.microscope_voltage_kv, 2);
	microscope_spherical_aberration_mm += powf(line_to_add.microscope_spherical_aberration_mm, 2);
	amplitude_contrast += powf(line_to_add.amplitude_contrast, 2);
	beam_tilt_x += powf(line_to_add.beam_tilt_x, 2);
	beam_tilt_y = powf(line_to_add.beam_tilt_y, 2);
	image_shift_x += powf(line_to_add.image_shift_x, 2);
	image_shift_y += powf(line_to_add.image_shift_y, 2);

	// not adding filenames or groups as it doesn't make sense

}

void cisTEMParameterLine::SetAllToZero()
{
	position_in_stack = 0;
	image_is_active = 0;
	psi = 0.0f;
	theta = 0.0f;
	phi = 0.0f;
	x_shift = 0.0f;
	y_shift = 0.0f;
	defocus_1 = 0.0f;
	defocus_2 = 0.0f;
	defocus_angle = 0.0f;
	phase_shift = 0.0f;
	occupancy = 0.0f;
	logp = 0.0f;
	sigma = 0.0f;
	score = 0.0f;
	score_change = 0.0f;
	pixel_size = 0.0f;
	microscope_voltage_kv = 0.0f;
	microscope_spherical_aberration_mm = 0.0f;
	amplitude_contrast = 0.0f;
	beam_tilt_x = 0.0f;
	beam_tilt_y = 0.0f;
	image_shift_x = 0.0f;
	image_shift_y = 0.0f;
	stack_filename = wxEmptyString;
	original_image_filename = wxEmptyString;
	reference_3d_filename = wxEmptyString;
	best_2d_class = 0;
	beam_tilt_group = 0;

}

void cisTEMParameterLine::ReplaceNanAndInfWithOther(cisTEMParameterLine &other_params)
{
	if (isnan(psi) || isinf(psi)) psi = other_params.psi;
	if (isnan(theta) || isinf(theta)) theta = other_params.theta;
	if (isnan(phi) || isinf(phi)) phi = other_params.phi;
	if (isnan(x_shift) || isinf(x_shift)) x_shift = other_params.x_shift;
	if (isnan(y_shift) || isinf(y_shift)) y_shift = other_params.y_shift;
	if (isnan(defocus_1) || isinf(defocus_1)) defocus_1 = other_params.defocus_1;
	if (isnan(defocus_2) || isinf(defocus_2)) defocus_2 = other_params.defocus_2;
	if (isnan(defocus_angle) || isinf(defocus_angle)) defocus_angle = other_params.defocus_angle;
	if (isnan(phase_shift) || isinf(phase_shift)) phase_shift = other_params.phase_shift;
	if (isnan(occupancy) || isinf(occupancy)) occupancy = other_params.occupancy;
	if (isnan(logp) || isinf(logp)) logp = other_params.logp;
	if (isnan(sigma) || isinf(sigma)) sigma = other_params.sigma;
	if (isnan(score) || isinf(score)) score = other_params.score;
	if (isnan(score_change) || isinf(score_change)) score_change = other_params.score_change;
	if (isnan(pixel_size) || isinf(pixel_size)) pixel_size = other_params.pixel_size;
	if (isnan(microscope_voltage_kv) || isinf(microscope_voltage_kv)) microscope_voltage_kv = other_params.microscope_voltage_kv;
	if (isnan(microscope_spherical_aberration_mm) || isinf(microscope_spherical_aberration_mm)) microscope_spherical_aberration_mm = other_params.microscope_spherical_aberration_mm;
	if (isnan(amplitude_contrast) || isinf(amplitude_contrast)) amplitude_contrast = other_params.amplitude_contrast;
	if (isnan(beam_tilt_x) || isinf(beam_tilt_x)) beam_tilt_x = other_params.beam_tilt_x;
	if (isnan(beam_tilt_y) || isinf(beam_tilt_y)) beam_tilt_y = other_params.beam_tilt_y;
	if (isnan(image_shift_x) || isinf(image_shift_x)) image_shift_x = other_params.image_shift_x;
	if (isnan(image_shift_y) || isinf(image_shift_y)) image_shift_y = other_params.image_shift_y;
	if (isnan(image_shift_y) || isinf(image_shift_y)) image_shift_y = other_params.image_shift_y;
}

cisTEMParameterLine::~cisTEMParameterLine()
{

}

cisTEMParameters::cisTEMParameters()
{
	parameters_that_were_read.SetAllToFalse();
}

cisTEMParameters::~cisTEMParameters()
{

}

void cisTEMParameters::PreallocateMemoryAndBlank(int number_to_allocate)
{
	ClearAll();
	cisTEMParameterLine temp_line;
	all_parameters.Add(temp_line, number_to_allocate);
}

// THIS IS NOT BEING UPDATED...

void cisTEMParameters::ReadFromFrealignParFile(wxString wanted_filename, float wanted_pixel_size, float wanted_microscope_voltage, float wanted_microscope_cs, float wanted_amplitude_contrast, float wanted_beam_tilt_x, float wanted_beam_tilt_y, float wanted_image_shift_x, float wanted_image_shift_y)
{
	ClearAll();
	float input_parameters[17];

	FrealignParameterFile input_par_file(wanted_filename, OPEN_TO_READ);
	input_par_file.ReadFile(false, -1);

	// pre-allocate the stack...

	PreallocateMemoryAndBlank(input_par_file.number_of_lines);

	// fill the array..

	for (long counter = 0; counter < input_par_file.number_of_lines; counter++)
	{
		input_par_file.ReadLine(input_parameters);

		all_parameters[counter].position_in_stack = input_parameters[0];
		all_parameters[counter].psi = input_parameters[1];
		all_parameters[counter].theta = input_parameters[2];
		all_parameters[counter].phi = input_parameters[3];
		all_parameters[counter].x_shift = input_parameters[4];
		all_parameters[counter].y_shift = input_parameters[5];
		all_parameters[counter].image_is_active = int(input_parameters[7]);
		all_parameters[counter].defocus_1 = input_parameters[8];
		all_parameters[counter].defocus_2 = input_parameters[9];
		all_parameters[counter].defocus_angle = input_parameters[10];
		all_parameters[counter].phase_shift = input_parameters[11];
		all_parameters[counter].occupancy = input_parameters[12];
		all_parameters[counter].logp = input_parameters[13];
		all_parameters[counter].sigma = input_parameters[14];
		all_parameters[counter].score = input_parameters[15];
		all_parameters[counter].score_change = input_parameters[16];
		all_parameters[counter].pixel_size = wanted_pixel_size; // not there
		all_parameters[counter].microscope_voltage_kv = wanted_microscope_voltage; // not there
		all_parameters[counter].microscope_spherical_aberration_mm = wanted_microscope_cs; // not there
		all_parameters[counter].amplitude_contrast = wanted_amplitude_contrast; // not there
		all_parameters[counter].beam_tilt_x = wanted_beam_tilt_x; // not there
		all_parameters[counter].beam_tilt_y = wanted_beam_tilt_y; // not there
		all_parameters[counter].image_shift_x = wanted_image_shift_x; // not there
		all_parameters[counter].image_shift_y = wanted_image_shift_y; // not there
	}
}

void cisTEMParameters::ReadFromcisTEMStarFile(wxString wanted_filename, bool exclude_negative_film_numbers)
{
	cisTEMParameterLine temp_line;
	all_parameters.Clear();
	cisTEMStarFileReader star_reader(wanted_filename, &all_parameters, exclude_negative_film_numbers);
	parameters_that_were_read = star_reader.parameters_that_were_read;
}

void cisTEMParameters::AddCommentToHeader(wxString comment_to_add)
{
	if (comment_to_add.StartsWith("#") == false)
	{
		comment_to_add = "# " + comment_to_add;
	}

	comment_to_add.Trim(true);
	header_comments.Add(comment_to_add);
}

void cisTEMParameters::ClearAll()
{
	header_comments.Clear();
	all_parameters.Clear();
}

void cisTEMParameters::SetAllReference3DFilename(wxString wanted_filename)
{
	for (long counter = 0; counter < all_parameters.GetCount(); counter++)
	{
		all_parameters[counter].reference_3d_filename = wanted_filename;
	}
}

void cisTEMParameters::WriteTocisTEMStarFile(wxString wanted_filename, int first_line_to_write, int last_line_to_write, int first_image_to_write, int last_image_to_write)
{

	wxFileName cisTEM_star_filename = wanted_filename;
	cisTEM_star_filename.SetExt("star");
	long particle_counter;

	wxTextFile *cisTEM_star_file = new wxTextFile(cisTEM_star_filename.GetFullPath());

	if (first_line_to_write == -1) first_line_to_write = 0;
	else
	if (first_line_to_write < 0 || first_line_to_write >= all_parameters.GetCount()) first_line_to_write = 0;

	if (last_line_to_write == -1) last_line_to_write = all_parameters.GetCount() - 1;
	else
	if (last_line_to_write < 0 || last_line_to_write >= all_parameters.GetCount()) last_line_to_write = all_parameters.GetCount() - 1;

	if (cisTEM_star_file->Exists())
	{
		cisTEM_star_file->Open();
		cisTEM_star_file->Clear();
	}
	else
	{
		cisTEM_star_file->Create();
	}

	cisTEM_star_file->AddLine(wxString::Format("# Written by cisTEM Version %s on %s", CISTEM_VERSION_TEXT, wxDateTime::Now().FormatISOCombined(' ')));

	for (int counter = 0; counter < header_comments.GetCount(); counter++)
	{
		cisTEM_star_file->AddLine(header_comments[counter]);
	}
	if (first_image_to_write == -1) first_image_to_write = 1;
	if (last_image_to_write == -1) last_image_to_write = INT_MAX;

	int column_counter = 1;

	// Write headers
	cisTEM_star_file->AddLine(wxString(" "));
	cisTEM_star_file->AddLine(wxString("data_"));
	cisTEM_star_file->AddLine(wxString(" "));
	cisTEM_star_file->AddLine(wxString("loop_"));

	// write headers depending on parameter mask...

	if (parameters_to_write.position_in_stack == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMPositionInStack #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.psi == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMAnglePsi #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.theta == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMAngleTheta #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.phi == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMAnglePhi #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.x_shift == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMXShift #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.y_shift == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMYShift #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.defocus_1 == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMDefocus1 #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.defocus_2 == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMDefocus2 #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.defocus_angle == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMDefocusAngle #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.phase_shift == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMPhaseShift #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.image_is_active == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMImageActivity #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.occupancy == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMOccupancy #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.logp == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMLogP #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.sigma == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMSigma #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.score == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMScore #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.score_change == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMScoreChange #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.pixel_size == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMPixelSize #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.microscope_voltage_kv == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMMicroscopeVoltagekV #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.microscope_spherical_aberration_mm == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMMicroscopeCsMM #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.amplitude_contrast == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMAmplitudeContrast #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.beam_tilt_x == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMBeamTiltX #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.beam_tilt_y == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMBeamTiltY #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.image_shift_x == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMImageShiftX #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.image_shift_y == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMImageShiftY #%i", column_counter));
		column_counter++;
	}


	if (parameters_to_write.best_2d_class == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMBest2DClass #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.beam_tilt_group == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMBeamTiltGroup #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.stack_filename == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMStackFilename #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.original_image_filename == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMOriginalImageFilename #%i", column_counter));
		column_counter++;
	}

	if (parameters_to_write.reference_3d_filename == true)
	{
		cisTEM_star_file->AddLine(wxString::Format("_cisTEMReference3DFilename #%i", column_counter));
		column_counter++;
	}

	wxString data_line = "";

	// header...

	if (parameters_to_write.position_in_stack == true) 					data_line += "     POS ";
	if (parameters_to_write.psi == true) 								data_line += "    PSI ";
	if (parameters_to_write.theta == true) 								data_line += "  THETA ";
	if (parameters_to_write.phi == true) 								data_line += "    PHI ";
	if (parameters_to_write.x_shift == true) 							data_line += "      SHX ";
	if (parameters_to_write.y_shift == true) 							data_line += "      SHY ";
	if (parameters_to_write.defocus_1 == true) 							data_line += "     DF1 ";
	if (parameters_to_write.defocus_2 == true) 							data_line += "     DF2 ";
	if (parameters_to_write.defocus_angle == true)  					data_line += " ANGAST ";
	if (parameters_to_write.phase_shift == true) 						data_line += " PSHIFT ";
	if (parameters_to_write.image_is_active == true) 					data_line += " STAT ";
	if (parameters_to_write.occupancy == true) 							data_line += "    OCC ";
	if (parameters_to_write.logp == true) 								data_line += "     LogP ";
	if (parameters_to_write.sigma == true)								data_line += "     SIGMA ";
	if (parameters_to_write.score == true) 								data_line += "  SCORE ";
	if (parameters_to_write.score_change == true)						data_line += " CHANGE ";
	if (parameters_to_write.pixel_size == true) 						data_line += "   PSIZE ";
	if (parameters_to_write.microscope_voltage_kv == true)  			data_line += "   VOLT ";
	if (parameters_to_write.microscope_spherical_aberration_mm == true) data_line += "     Cs ";
	if (parameters_to_write.amplitude_contrast == true)  				data_line += "   AmpC ";
	if (parameters_to_write.beam_tilt_x == true) 						data_line += " BTILTX ";
	if (parameters_to_write.beam_tilt_y == true) 						data_line += " BTILTY ";
	if (parameters_to_write.image_shift_x == true) 						data_line += " ISHFTX ";
	if (parameters_to_write.image_shift_y == true) 						data_line += " ISHFTY ";
	if (parameters_to_write.best_2d_class == true) 						data_line += "2DCLS ";
	if (parameters_to_write.beam_tilt_group == true) 					data_line += " TGRP ";
	if (parameters_to_write.stack_filename == true)  					data_line += "                                     STACK_FILENAME ";
	if (parameters_to_write.original_image_filename == true) 			data_line += "                            ORIGINAL_IMAGE_FILENAME ";
	if (parameters_to_write.reference_3d_filename == true) 				data_line += "                              REFERENCE_3D_FILENAME ";

	data_line[0] = '#';
	cisTEM_star_file->AddLine(data_line);



	// write the data..



	for (particle_counter = first_line_to_write; particle_counter <= last_line_to_write; particle_counter ++ )
	{
		if (all_parameters[particle_counter].position_in_stack < first_image_to_write || all_parameters[particle_counter].position_in_stack > last_image_to_write) continue;
		data_line = "";

		if (parameters_to_write.position_in_stack == true) data_line += wxString::Format("%8u ", all_parameters[particle_counter].position_in_stack);
		if (parameters_to_write.psi == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].psi);
		if (parameters_to_write.theta == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].theta);
		if (parameters_to_write.phi == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].phi);
		if (parameters_to_write.x_shift == true) data_line += wxString::Format("%9.2f ", all_parameters[particle_counter].x_shift);
		if (parameters_to_write.y_shift == true) data_line += wxString::Format("%9.2f ", all_parameters[particle_counter].y_shift);
		if (parameters_to_write.defocus_1 == true) data_line += wxString::Format("%8.1f ", all_parameters[particle_counter].defocus_1);
		if (parameters_to_write.defocus_2 == true) data_line += wxString::Format("%8.1f ", all_parameters[particle_counter].defocus_2);
		if (parameters_to_write.defocus_angle == true)  data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].defocus_angle);
		if (parameters_to_write.phase_shift == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].phase_shift);
		if (parameters_to_write.image_is_active == true) data_line += wxString::Format("%5i ", all_parameters[particle_counter].image_is_active);
		if (parameters_to_write.occupancy == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].occupancy);
		if (parameters_to_write.logp == true) data_line += wxString::Format("%9i ", myroundint(all_parameters[particle_counter].logp));
		if (parameters_to_write.sigma == true) data_line += wxString::Format("%10.4f ", all_parameters[particle_counter].sigma);
		if (parameters_to_write.score == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].score);
		if (parameters_to_write.score_change == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].score_change);
		if (parameters_to_write.pixel_size == true) data_line += wxString::Format("%8.5f ", all_parameters[particle_counter].pixel_size);
		if (parameters_to_write.microscope_voltage_kv == true)  data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].microscope_voltage_kv);
		if (parameters_to_write.microscope_spherical_aberration_mm == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].microscope_spherical_aberration_mm);
		if (parameters_to_write.amplitude_contrast == true) data_line += wxString::Format("%7.2f ", all_parameters[particle_counter].amplitude_contrast);
		if (parameters_to_write.beam_tilt_x == true) data_line += wxString::Format("%7.3f ", all_parameters[particle_counter].beam_tilt_x);
		if (parameters_to_write.beam_tilt_y == true) data_line += wxString::Format("%7.3f ", all_parameters[particle_counter].beam_tilt_y);
		if (parameters_to_write.image_shift_x == true) data_line += wxString::Format("%7.3f ", all_parameters[particle_counter].image_shift_x);
		if (parameters_to_write.image_shift_y == true) data_line += wxString::Format("%7.3f ", all_parameters[particle_counter].image_shift_y);
		if (parameters_to_write.best_2d_class == true) data_line += wxString::Format("%5i ", all_parameters[particle_counter].best_2d_class);
		if (parameters_to_write.beam_tilt_group == true) data_line += wxString::Format("%5i ", all_parameters[particle_counter].beam_tilt_group);
		if (parameters_to_write.stack_filename == true)  data_line += wxString::Format("%50s ", wxString::Format("'%s'", all_parameters[particle_counter].stack_filename));
		if (parameters_to_write.original_image_filename == true) data_line += wxString::Format("%50s ", wxString::Format("'%s'", all_parameters[particle_counter].original_image_filename));
		if (parameters_to_write.reference_3d_filename == true) data_line += wxString::Format("%50s ", wxString::Format("'%s'", all_parameters[particle_counter].reference_3d_filename));
		cisTEM_star_file->AddLine(data_line);

	}

	cisTEM_star_file->Write();
	cisTEM_star_file->Close();

	delete  cisTEM_star_file;

}

cisTEMParameterLine cisTEMParameters::ReturnParameterAverages(bool only_average_active)
{
	cisTEMParameterLine average_values;

	long 			average_position_in_stack = 0;
	long			average_image_is_active = 0;
	double		 	average_psi = 0.0;
	double			average_theta = 0.0;
	double			average_phi = 0.0;
	double			average_x_shift = 0.0;
	double			average_y_shift = 0.0;
	double			average_defocus_1 = 0.0;
	double			average_defocus_2 = 0.0;
	double			average_defocus_angle = 0.0;
	double			average_phase_shift = 0.0;
	double			average_occupancy = 0.0;
	double			average_logp = 0;
	double			average_sigma = 0.0;
	double 			average_score = 0.0;
	double			average_score_change = 0.0;
	double			average_pixel_size = 0.0;
	double			average_microscope_voltage_kv = 0.0;
	double			average_microscope_spherical_aberration_mm = 0.0;
	double			average_amplitude_contrast = 0.0;
	double			average_beam_tilt_x = 0.0;
	double			average_beam_tilt_y = 0.0;
	double			average_image_shift_x = 0.0;
	double			average_image_shift_y = 0.0;

	long number_summed = 0;

	for (long counter = 0; counter < all_parameters.GetCount(); counter++)
	{
		if (ReturnImageIsActive(counter) >= 0 || only_average_active == false)
		{
			average_position_in_stack += ReturnPositionInStack(counter);
			average_image_is_active += ReturnImageIsActive(counter);
			average_psi += ReturnPsi(counter);
			average_theta += ReturnTheta(counter);
			average_phi += ReturnPhi(counter);
			average_x_shift += ReturnXShift(counter);
			average_y_shift += ReturnYShift(counter);
			average_defocus_1 += ReturnDefocus1(counter);
			average_defocus_2 += ReturnDefocus2(counter);
			average_defocus_angle += ReturnDefocusAngle(counter);
			average_phase_shift += ReturnPhaseShift(counter);
			average_occupancy += ReturnOccupancy(counter);
			average_logp += ReturnLogP(counter);
			average_sigma += ReturnSigma(counter);
			average_score += ReturnScore(counter);
			average_score_change += ReturnScoreChange(counter);
			average_pixel_size += ReturnPixelSize(counter);
			average_microscope_voltage_kv += ReturnMicroscopekV(counter);
			average_microscope_spherical_aberration_mm += ReturnMicroscopeCs(counter);
			average_amplitude_contrast += ReturnAmplitudeContrast(counter);
			average_beam_tilt_x += ReturnBeamTiltX(counter);
			average_beam_tilt_y += ReturnBeamTiltY(counter);
			average_image_shift_x += ReturnImageShiftX(counter);
			average_image_shift_y += ReturnImageShiftY(counter);

			number_summed++;


		}

	}

	if (number_summed > 0)
	{
		average_values.position_in_stack = average_position_in_stack / double(number_summed);
		average_values.image_is_active = average_image_is_active / double(number_summed);
		average_values.psi = average_psi  / double(number_summed);
		average_values.theta = average_theta  / double(number_summed);
		average_values.phi = average_phi  / double(number_summed);
		average_values.x_shift = average_x_shift  / double(number_summed);
		average_values.y_shift = average_y_shift  / double(number_summed);
		average_values.defocus_1 = average_defocus_1  / double(number_summed);
		average_values.defocus_2 = average_defocus_2  / double(number_summed);
		average_values.defocus_angle = average_defocus_angle  / double(number_summed);
		average_values.phase_shift = average_phase_shift  / double(number_summed);
		average_values.occupancy = average_occupancy  / double(number_summed);
		average_values.logp = average_logp  / double(number_summed);
		average_values.sigma = average_sigma  / double(number_summed);
		average_values.score = average_score  / double(number_summed);
		average_values.score_change = average_score_change  / double(number_summed);
		average_values.pixel_size = average_pixel_size  / double(number_summed);
		average_values.microscope_voltage_kv = average_microscope_voltage_kv  / double(number_summed);
		average_values.microscope_spherical_aberration_mm = average_microscope_spherical_aberration_mm  / double(number_summed);
		average_values.amplitude_contrast = average_amplitude_contrast  / double(number_summed);
		average_values.beam_tilt_x = average_beam_tilt_x  / double(number_summed);
		average_values.beam_tilt_y = average_beam_tilt_y  / double(number_summed);
		average_values.image_shift_x = average_image_shift_x  / double(number_summed);
		average_values.image_shift_y = average_image_shift_y  / double(number_summed);
	}

	return average_values;
}

cisTEMParameterLine cisTEMParameters::ReturnParameterVariances(bool only_average_active)
{

	cisTEMParameterLine average_values;
	average_values = ReturnParameterAverages(only_average_active);

	cisTEMParameterLine variance_values;

	long 			variance_position_in_stack = 0;
	long			variance_image_is_active = 0;
	double		 	variance_psi = 0.0;
	double			variance_theta = 0.0;
	double			variance_phi = 0.0;
	double			variance_x_shift = 0.0;
	double			variance_y_shift = 0.0;
	double			variance_defocus_1 = 0.0;
	double			variance_defocus_2 = 0.0;
	double			variance_defocus_angle = 0.0;
	double			variance_phase_shift = 0.0;
	double			variance_occupancy = 0.0;
	double			variance_logp = 0;
	double			variance_sigma = 0.0;
	double 			variance_score = 0.0;
	double			variance_score_change = 0.0;
	double			variance_pixel_size = 0.0;
	double			variance_microscope_voltage_kv = 0.0;
	double			variance_microscope_spherical_aberration_mm = 0.0;
	double			variance_amplitude_contrast = 0.0;
	double			variance_beam_tilt_x = 0.0;
	double			variance_beam_tilt_y = 0.0;
	double			variance_image_shift_x = 0.0;
	double			variance_image_shift_y = 0.0;

	long number_summed = 0;

	for (long counter = 0; counter < all_parameters.GetCount(); counter++)
	{
		if (ReturnImageIsActive(counter) >= 0 || only_average_active == false)
		{
			variance_position_in_stack += powf(ReturnPositionInStack(counter), 2);
			variance_image_is_active += powf(ReturnImageIsActive(counter), 2);
			variance_psi += powf(ReturnPsi(counter), 2);
			variance_theta += powf(ReturnTheta(counter), 2);
			variance_phi += powf(ReturnPhi(counter), 2);
			variance_x_shift += powf(ReturnXShift(counter), 2);
			variance_y_shift += powf(ReturnYShift(counter), 2);
			variance_defocus_1 += powf(ReturnDefocus1(counter), 2);
			variance_defocus_2 += powf(ReturnDefocus2(counter), 2);
			variance_defocus_angle += powf(ReturnDefocusAngle(counter), 2);
			variance_phase_shift += powf(ReturnPhaseShift(counter), 2);
			variance_occupancy += powf(ReturnOccupancy(counter), 2);
			variance_logp += powf(ReturnLogP(counter), 2);
			variance_sigma += powf(ReturnSigma(counter), 2);
			variance_score += powf(ReturnScore(counter), 2);
			variance_score_change += powf(ReturnScoreChange(counter), 2);
			variance_pixel_size += powf(ReturnPixelSize(counter), 2);
			variance_microscope_voltage_kv += powf(ReturnMicroscopekV(counter), 2);
			variance_microscope_spherical_aberration_mm += powf(ReturnMicroscopeCs(counter), 2);
			variance_amplitude_contrast += powf(ReturnAmplitudeContrast(counter), 2);
			variance_beam_tilt_x += powf(ReturnBeamTiltX(counter), 2);
			variance_beam_tilt_y += powf(ReturnBeamTiltY(counter), 2);
			variance_image_shift_x += powf(ReturnImageShiftX(counter), 2);
			variance_image_shift_y += powf(ReturnImageShiftY(counter), 2);

			number_summed++;


		}

	}

	if (number_summed > 0)
	{
		variance_values.position_in_stack = variance_position_in_stack / double(number_summed);
		variance_values.image_is_active = variance_image_is_active / double(number_summed);
		variance_values.psi = variance_psi  / double(number_summed);
		variance_values.theta = variance_theta  / double(number_summed);
		variance_values.phi = variance_phi  / double(number_summed);
		variance_values.x_shift = variance_x_shift  / double(number_summed);
		variance_values.y_shift = variance_y_shift  / double(number_summed);
		variance_values.defocus_1 = variance_defocus_1  / double(number_summed);
		variance_values.defocus_2 = variance_defocus_2  / double(number_summed);
		variance_values.defocus_angle = variance_defocus_angle  / double(number_summed);
		variance_values.phase_shift = variance_phase_shift  / double(number_summed);
		variance_values.occupancy = variance_occupancy  / double(number_summed);
		variance_values.logp = variance_logp  / double(number_summed);
		variance_values.sigma = variance_sigma  / double(number_summed);
		variance_values.score = variance_score  / double(number_summed);
		variance_values.score_change = variance_score_change  / double(number_summed);
		variance_values.pixel_size = variance_pixel_size  / double(number_summed);
		variance_values.microscope_voltage_kv = variance_microscope_voltage_kv  / double(number_summed);
		variance_values.microscope_spherical_aberration_mm = variance_microscope_spherical_aberration_mm  / double(number_summed);
		variance_values.amplitude_contrast = variance_amplitude_contrast  / double(number_summed);
		variance_values.beam_tilt_x = variance_beam_tilt_x  / double(number_summed);
		variance_values.beam_tilt_y = variance_beam_tilt_y  / double(number_summed);
		variance_values.image_shift_x = variance_image_shift_x  / double(number_summed);
		variance_values.image_shift_y = variance_image_shift_y  / double(number_summed);
	}

	variance_values.position_in_stack -= powf(average_values.position_in_stack, 2);
	variance_values.image_is_active -= powf(average_values.image_is_active, 2);
	variance_values.psi -= powf(average_values.psi, 2);
	variance_values.theta -= powf(average_values.theta, 2);
	variance_values.phi -= powf(average_values.phi, 2);
	variance_values.x_shift -= powf(average_values.x_shift, 2);
	variance_values.y_shift -= powf(average_values.y_shift, 2);
	variance_values.defocus_1 -= powf(average_values.defocus_1, 2);
	variance_values.defocus_2 -= powf(average_values.defocus_2, 2);
	variance_values.defocus_angle -= powf(average_values.defocus_angle, 2);
	variance_values.phase_shift -= powf(average_values.phase_shift, 2);
	variance_values.occupancy -= powf(average_values.occupancy, 2);
	variance_values.logp -= powf(average_values.logp, 2);
	variance_values.sigma -= powf(average_values.sigma, 2);
	variance_values.score -= powf(average_values.score, 2);
	variance_values.score_change -= powf(average_values.score_change, 2);
	variance_values.pixel_size -= powf(average_values.pixel_size, 2);
	variance_values.microscope_voltage_kv -= powf(average_values.microscope_voltage_kv, 2);
	variance_values.microscope_spherical_aberration_mm -= powf(average_values.microscope_spherical_aberration_mm, 2);
	variance_values.amplitude_contrast -= powf(average_values.amplitude_contrast, 2);
	variance_values.beam_tilt_x -= powf(average_values.beam_tilt_x, 2);
	variance_values.beam_tilt_y -= powf(average_values.beam_tilt_y, 2);
	variance_values.image_shift_x -= powf(average_values.image_shift_x, 2);
	variance_values.image_shift_y -= powf(average_values.image_shift_y, 2);

	return variance_values;
}

float cisTEMParameters::ReturnAverageSigma(bool exclude_negative_film_numbers)
{
	double sum = 0;
	long number_summed = 0;

	for (long counter = 0; counter < all_parameters.GetCount(); counter++)
	{
		if (ReturnImageIsActive(counter) >= 0 || ! exclude_negative_film_numbers)
		{
			sum += ReturnSigma(counter);
			number_summed++;
		}
	}

	if (number_summed > 0) return sum / double(number_summed);
	else return 0.0;

}

float cisTEMParameters::ReturnAverageOccupancy(bool exclude_negative_film_numbers)
{
	double sum = 0;
	long number_summed = 0;

	for (long counter = 0; counter < all_parameters.GetCount(); counter++)
	{
		if (ReturnImageIsActive(counter) >= 0 || ! exclude_negative_film_numbers)
		{
			sum += ReturnOccupancy(counter);
			number_summed++;
		}
	}

	if (number_summed > 0) return float(sum / double(number_summed));
	else return 0.0f;

}

float cisTEMParameters::ReturnAverageScore(bool exclude_negative_film_numbers)
{
	double sum = 0;
	long number_summed = 0;

	for (long counter = 0; counter < all_parameters.GetCount(); counter++)
	{
		if (ReturnImageIsActive(counter) >= 0 || ! exclude_negative_film_numbers)
		{
			sum += ReturnScore(counter);
			number_summed++;
		}
	}

	if (number_summed > 0) return sum / double(number_summed);
	else return 0.0;

}

void cisTEMParameters::RemoveSigmaOutliers(float wanted_standard_deviation, bool exclude_negative_film_numbers, bool reciprocal_square)
{
	MyDebugAssertTrue(wanted_standard_deviation > 0.0, "Invalid standard deviation");

	int line;
	int sum_i;
	double average = 0.0;
	double sum2 = 0.0;
	float std;
	float upper_threshold;
	float lower_threshold;
	float temp_float;

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			temp_float = ReturnSigma(line);
			if (reciprocal_square && temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
			average += temp_float;
			sum2 += powf(temp_float, 2);
			sum_i++;
		}
	}

	if (sum_i > 0)
	{
		average /= sum_i;
		std = sum2 / sum_i - powf(average / sum_i, 2);
	}

	if (std > 0.0)
	{
		// Remove extreme outliers and recalculate std
		std = sqrtf(std);
		upper_threshold = average + 2.0 * wanted_standard_deviation * std;
		lower_threshold = average - 2.0 * wanted_standard_deviation * std;
//		wxPrintf("0: average, std, upper, lower = %g %g %g %g\n", float(average), std, upper_threshold, lower_threshold);
		average = 0.0;
		sum2 = 0.0;
		sum_i = 0;

		for (line = 0; line < all_parameters.GetCount(); line++)
		{
			if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
			{
				temp_float = ReturnSigma(line);
				if (reciprocal_square && temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
				if (temp_float <= upper_threshold && temp_float >= lower_threshold)
				{
					average += temp_float;
					sum2 += powf(temp_float, 2);
					sum_i++;
				}
			}
		}

		if (sum_i > 0)
		{
			average /= sum_i;
			std = sum2 / sum_i - powf(average / sum_i, 2);
		}

		// Now remove outliers according to (hopefully) more reasonable std
		std = sqrtf(std);

		upper_threshold = average + wanted_standard_deviation * std;
		lower_threshold = average - wanted_standard_deviation * std;
//		wxPrintf("1: average, std, upper, lower = %g %g %g %g\n", float(average), std, upper_threshold, lower_threshold);

		for (line = 0; line < all_parameters.GetCount(); line++)
		{
			if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
			{
				temp_float = ReturnSigma(line);

				if (reciprocal_square)
				{
					if (temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
					else temp_float = average;
					if (temp_float > upper_threshold) temp_float = upper_threshold;
					if (temp_float < lower_threshold) temp_float = lower_threshold;
					temp_float = sqrtf(1.0 / temp_float);
				}
				else
				{
					if (temp_float > upper_threshold) temp_float = upper_threshold;
					if (temp_float < lower_threshold) temp_float = lower_threshold;
				}

				all_parameters[line].sigma = temp_float;
			}
		}
	}


}

void cisTEMParameters::RemoveScoreOutliers(float wanted_standard_deviation, bool exclude_negative_film_numbers, bool reciprocal_square)
{
	MyDebugAssertTrue(wanted_standard_deviation > 0.0, "Invalid standard deviation");

	int line;
	int sum_i;
	double average = 0.0;
	double sum2 = 0.0;
	float std;
	float upper_threshold;
	float lower_threshold;
	float temp_float;

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			temp_float = ReturnScore(line);
			if (reciprocal_square && temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
			average += temp_float;
			sum2 += powf(temp_float, 2);
			sum_i++;
		}
	}

	if (sum_i > 0)
	{
		average /= sum_i;
		std = sum2 / sum_i - powf(average / sum_i, 2);
	}

	if (std > 0.0)
	{
		// Remove extreme outliers and recalculate std
		std = sqrtf(std);
		upper_threshold = average + 2.0 * wanted_standard_deviation * std;
		lower_threshold = average - 2.0 * wanted_standard_deviation * std;
//		wxPrintf("0: average, std, upper, lower = %g %g %g %g\n", float(average), std, upper_threshold, lower_threshold);
		average = 0.0;
		sum2 = 0.0;
		sum_i = 0;

		for (line = 0; line < all_parameters.GetCount(); line++)
		{
			if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
			{
				temp_float = ReturnScore(line);
				if (reciprocal_square && temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
				if (temp_float <= upper_threshold && temp_float >= lower_threshold)
				{
					average += temp_float;
					sum2 += powf(temp_float, 2);
					sum_i++;
				}
			}
		}

		if (sum_i > 0)
		{
			average /= sum_i;
			std = sum2 / sum_i - powf(average / sum_i, 2);
		}

		// Now remove outliers according to (hopefully) more reasonable std
		std = sqrtf(std);

		upper_threshold = average + wanted_standard_deviation * std;
		lower_threshold = average - wanted_standard_deviation * std;
//		wxPrintf("1: average, std, upper, lower = %g %g %g %g\n", float(average), std, upper_threshold, lower_threshold);

		for (line = 0; line < all_parameters.GetCount(); line++)
		{
			if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
			{
				temp_float = ReturnScore(line);

				if (reciprocal_square)
				{
					if (temp_float > 0.0) temp_float = 1.0 / powf(temp_float, 2);
					else temp_float = average;
					if (temp_float > upper_threshold) temp_float = upper_threshold;
					if (temp_float < lower_threshold) temp_float = lower_threshold;
					temp_float = sqrtf(1.0 / temp_float);
				}
				else
				{
					if (temp_float > upper_threshold) temp_float = upper_threshold;
					if (temp_float < lower_threshold) temp_float = lower_threshold;
				}

				all_parameters[line].score = temp_float;
			}
		}
	}


}


void cisTEMParameters::CalculateDefocusDependence(bool exclude_negative_film_numbers)
{
	int line;
	double s = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
	double delta;

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			average_defocus = (ReturnDefocus1(line) + ReturnDefocus2(line)) / 2.0;
			s += ReturnOccupancy(line);
			sx += average_defocus * ReturnOccupancy(line);
			sy += ReturnScore(line) * ReturnOccupancy(line);
			sxx += powf(average_defocus,2) * ReturnOccupancy(line);
			sxy += average_defocus * ReturnScore(line) * ReturnOccupancy(line);
		}
	}
	average_defocus = sx / s;
	delta = s * sxx - powf(sx,2);
	defocus_coeff_a = (sxx * sy - sx * sxy) / delta;
	defocus_coeff_b = (s * sxy - sx * sy) / delta;
//	wxPrintf("average_defocus = %g, defocus_coeff_a = %g, defocus_coeff_b = %g\n", average_defocus, defocus_coeff_a, defocus_coeff_b);
}

void cisTEMParameters::AdjustScores(bool exclude_negative_film_numbers)
{
	int line;
	float defocus;

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			defocus = (ReturnDefocus1(line) + ReturnDefocus2(line)) / 2.0;
			if (defocus != 0.0f) all_parameters[line].score -= ReturnScoreAdjustment(defocus); // added 0 check for defocus sweep
		}
	}
}

float cisTEMParameters::ReturnScoreAdjustment(float defocus)
{
	MyDebugAssertTrue(average_defocus != 0.0 || defocus_coeff_b != 0.0, "Defocus coefficients not determined");

	return (defocus - average_defocus) * defocus_coeff_b;
}

float cisTEMParameters::ReturnScoreThreshold(float wanted_percentage, bool exclude_negative_film_numbers)
{

	int i;
	int line;
	int number_of_bins = 10000;
	float average_occ = 0.0;
	float sum_occ;
	float increment;
	float threshold;
	float percentage;
	float min, max;

	min = ReturnMinScore(exclude_negative_film_numbers);
	max = ReturnMaxScore(exclude_negative_film_numbers);

	average_occ = ReturnAverageOccupancy(exclude_negative_film_numbers);
	increment = (min - max) / (number_of_bins - 1);
	if (increment == 0.0) return min;

	//wxPrintf("min = %f, max = %f, increment = %f\n", min, max, increment);
	for (i = 0; i < number_of_bins; i++)
	{
		sum_occ = 0.0;
		threshold = float(i) * increment + max;

		for (line = 0; line < all_parameters.GetCount(); line++)
		{
			if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
			{
	 			if (ReturnScore(line) >= threshold) sum_occ += ReturnOccupancy(line);
			}
		}
		percentage = sum_occ / all_parameters.GetCount() / average_occ;

	//	wxPrintf("sum_occ = %f : threshold = %f\n", sum_occ, threshold);
		if (percentage >= wanted_percentage) break;
	}

	if (sum_occ == 0.0)
	{
		MyPrintWithDetails("Error: Number of particles selected = 0; please change score threshold\n");
		DEBUG_ABORT;
	}

	return threshold;
}

float cisTEMParameters::ReturnMinScore(bool exclude_negative_film_numbers)
{
	int line;
	float min;
	float temp_float;

	min = std::numeric_limits<float>::max();

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			temp_float = ReturnScore(line);
			if (min > temp_float) min = temp_float;
		}
	}

	return min;
}

float cisTEMParameters::ReturnMaxScore(bool exclude_negative_film_numbers)
{
	int line;
	float max;
	float temp_float;

	max = std::numeric_limits<float>::min();

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			temp_float = ReturnScore(line);
			if (max < temp_float) max = temp_float;
		}
	}

	return max;
}


int cisTEMParameters::ReturnMinPositionInStack(bool exclude_negative_film_numbers)
{
	int line;
	int min;
	int temp_int;

	min = std::numeric_limits<int>::max();

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			temp_int = ReturnPositionInStack(line);
			if (min > temp_int) min = temp_int;
		}
	}

	return min;
}

int cisTEMParameters::ReturnMaxPositionInStack(bool exclude_negative_film_numbers)
{
	int line;
	int max;
	int temp_int;

	max = std::numeric_limits<int>::min();

	for (line = 0; line < all_parameters.GetCount(); line++)
	{
		if (ReturnImageIsActive(line) >= 0 || ! exclude_negative_film_numbers)
		{
			temp_int = ReturnPositionInStack(line);
			if (max < temp_int) max = temp_int;
		}
	}

	return max;
}


static int wxCMPFUNC_CONV SortByReference3DFilenameCompareFunction( cisTEMParameterLine **a, cisTEMParameterLine **b) // function for sorting the classum selections by parent_image_id - this makes cutting them out more efficient
{
	return wxStringSortAscending(&(*a)->reference_3d_filename, &(*b)->reference_3d_filename);
};

void cisTEMParameters::SortByReference3DFilename()
{
	all_parameters.Sort(SortByReference3DFilenameCompareFunction);
}

