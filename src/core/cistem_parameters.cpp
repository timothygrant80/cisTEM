#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfcisTEMParameterLines);

cisTEMParameterLine::cisTEMParameterLine()
{
	position_in_stack = -1;
	image_is_active = -1;
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
	beam_tilt_x = 0.0f;
	beam_tilt_y = 0.0f;


}

cisTEMParameterLine::~cisTEMParameterLine()
{

}

cisTEMParameters::cisTEMParameters()
{

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

void cisTEMParameters::ReadFromFrealignParFile(wxString wanted_filename, float wanted_pixel_size, float wanted_microscope_voltage, float wanted_microscope_cs, float wanted_beam_tilt_x, float wanted_beam_tilt_y)
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
		all_parameters[counter].logp = int(input_parameters[13]);
		all_parameters[counter].sigma = input_parameters[14];
		all_parameters[counter].score = input_parameters[15];
		all_parameters[counter].score_change = input_parameters[16];
		all_parameters[counter].pixel_size = wanted_pixel_size; // not there
		all_parameters[counter].microscope_voltage_kv = wanted_microscope_voltage; // not there
		all_parameters[counter].microscope_spherical_aberration_mm = wanted_microscope_cs; // not there
		all_parameters[counter].beam_tilt_x = wanted_beam_tilt_x; // not there
		all_parameters[counter].beam_tilt_y = wanted_beam_tilt_y; // not there
	}
}

void cisTEMParameters::ReadFromcisTEMStarFile(wxString wanted_filename)
{
	cisTEMParameterLine temp_line;
	all_parameters.Clear();
	cisTEMStarFileReader star_reader(wanted_filename, &all_parameters);
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

void cisTEMParameters::WriteTocisTEMStarFile(wxString wanted_filename)
{

	wxFileName cisTEM_star_filename = wanted_filename;
	cisTEM_star_filename.SetExt("star");
	long particle_counter;

	wxTextFile *cisTEM_star_file = new wxTextFile(cisTEM_star_filename.GetFullPath());

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


	// Write headers
	cisTEM_star_file->AddLine(wxString(" "));
	cisTEM_star_file->AddLine(wxString("data_"));
	cisTEM_star_file->AddLine(wxString(" "));
	cisTEM_star_file->AddLine(wxString("loop_"));
	cisTEM_star_file->AddLine(wxString("_cisTEMPositionInStack #1"));
	cisTEM_star_file->AddLine(wxString("_cisTEMAnglePsi #2"));
	cisTEM_star_file->AddLine(wxString("_cisTEMAngleTheta #3"));
	cisTEM_star_file->AddLine(wxString("_cisTEMAnglePhi #4"));
	cisTEM_star_file->AddLine(wxString("_cisTEMXShift #5"));
	cisTEM_star_file->AddLine(wxString("_cisTEMYShift #6"));
	cisTEM_star_file->AddLine(wxString("_cisTEMDefocus1 #7"));
	cisTEM_star_file->AddLine(wxString("_cisTEMDefocus2 #8"));
	cisTEM_star_file->AddLine(wxString("_cisTEMDefocusAngle #9"));
	cisTEM_star_file->AddLine(wxString("_cisTEMPhaseShift #10"));
	cisTEM_star_file->AddLine(wxString("_cisTEMImageActivity #11"));
	cisTEM_star_file->AddLine(wxString("_cisTEMOccupancy #12"));
	cisTEM_star_file->AddLine(wxString("_cisTEMLogP #13"));
	cisTEM_star_file->AddLine(wxString("_cisTEMSigma #14"));
	cisTEM_star_file->AddLine(wxString("_cisTEMScore #15"));
	cisTEM_star_file->AddLine(wxString("_cisTEMScoreChange #16"));
	cisTEM_star_file->AddLine(wxString("_cisTEMPixelSize #17"));
	cisTEM_star_file->AddLine(wxString("_cisTEMMicroscopeVoltagekV #18"));
	cisTEM_star_file->AddLine(wxString("_cisTEMMicroscopeCsMM #19"));
	cisTEM_star_file->AddLine(wxString("_cisTEMBeamTiltX #20"));
	cisTEM_star_file->AddLine(wxString("_cisTEMBeamTiltY #21"));


	cisTEM_star_file->AddLine("#            PSI   THETA     PHI       SHX       SHY      DF1      DF2  ANGAST  PSHIFT  STAT     OCC      LogP      SIGMA   SCORE  CHANGE    PSIZE    VOLT      CS  BTILTX  BTILTY");


	for (particle_counter = 0; particle_counter < all_parameters.GetCount(); particle_counter ++ )
	{
		cisTEM_star_file->AddLine(wxString::Format("%8u %7.2f %7.2f %7.2f %9.2f %9.2f %8.1f %8.1f %7.2f %7.2f %5i %7.2f %9i %10.4f %7.2f %7.2f %8.5f %7.2f %7.2f %7.2f %7.2f",	all_parameters[particle_counter].position_in_stack,
																														all_parameters[particle_counter].psi,
																														all_parameters[particle_counter].theta,
																														all_parameters[particle_counter].phi,
																														all_parameters[particle_counter].x_shift,
																														all_parameters[particle_counter].y_shift,
																														all_parameters[particle_counter].defocus_1,
																														all_parameters[particle_counter].defocus_2,
																														all_parameters[particle_counter].defocus_angle,
																														all_parameters[particle_counter].phase_shift,
																														all_parameters[particle_counter].image_is_active,
																														all_parameters[particle_counter].occupancy,
																														all_parameters[particle_counter].logp,
																														all_parameters[particle_counter].sigma,
																														all_parameters[particle_counter].score,
																														all_parameters[particle_counter].score_change,
																														all_parameters[particle_counter].pixel_size,
																														all_parameters[particle_counter].microscope_voltage_kv,
																														all_parameters[particle_counter].microscope_spherical_aberration_mm,
																														all_parameters[particle_counter].beam_tilt_x,
																														all_parameters[particle_counter].beam_tilt_y));



	}

	cisTEM_star_file->Write();
	cisTEM_star_file->Close();

	delete  cisTEM_star_file;

}
