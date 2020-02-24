#include "../../core/core_headers.h"

class
AlignCoordinatesApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};



IMPLEMENT_APP(AlignCoordinatesApp)

// override the DoInteractiveUserInput

void AlignCoordinatesApp::DoInteractiveUserInput()
{
	UserInput *my_input = new UserInput("AlignCoordinatesApp", 1.0);

	std::string input_ref_coordinates = my_input->GetFilenameFromUser("Reference coordinates (A)", "File with x,y,z coordinates to align to", "input_ref_coords.txt", true );
	std::string input_align_coordinates	= my_input->GetFilenameFromUser("Coordinates to be aligned (A)", "File with x,y,z coordinates to be aligned", "input_align_coords.txt", true );
	std::string outut_chimera_cmd = my_input->GetFilenameFromUser("Output Chimera cmd file", "File with alignment commands for UCSF Chimera", "output_chimera.cmd", false );
	float tolerance = my_input->GetFloatFromUser("Distance tolerance (A)", "Distance of points to count as being close (in projection)", "100.0", 0.0);
	float margin = my_input->GetFloatFromUser("Margin (A)", "Margin of aligned coordinates that should be excluded for comparison", "200.0", 0.0);
	float x_dimension = my_input->GetFloatFromUser("X-dimension (A)", "The X-dimension of the field of view used for HRTM", "5800.0", 0.0);
	float y_dimension = my_input->GetFloatFromUser("Y-dimension (A)", "The Y-dimension of the field of view used for HRTM", "4100.0", 0.0);
	float initial_x = my_input->GetFloatFromUser("Initial X offset (A)", "The X-offset from a previous search", "0.0");
	float initial_y = my_input->GetFloatFromUser("Initial Y offset (A)", "The Y-offset from a previous search", "0.0");
	float initial_z = my_input->GetFloatFromUser("Initial Z offset (A)", "The Z-offset from a previous search", "0.0");
	float initial_angle = my_input->GetFloatFromUser("Initial angle (deg)", "The in-plane alignment angle from a previous search", "0.0");
	bool perform_search = my_input->GetYesNoFromUser("Perform search", "Should a search & refinement be performed?", "Yes");

	delete my_input;

//	my_current_job.Reset(12);
	my_current_job.ManualSetArguments("tttffffffffb", input_ref_coordinates.c_str(), input_align_coordinates.c_str(), outut_chimera_cmd.c_str(), tolerance, margin,
			x_dimension, y_dimension, initial_x, initial_y, initial_z, initial_angle, perform_search);
}

// override the do calculation method which will be what is actually run..

bool AlignCoordinatesApp::DoCalculation()
{
	wxString input_ref_coordinates = my_current_job.arguments[0].ReturnStringArgument();
	wxString input_align_coordinates = my_current_job.arguments[1].ReturnStringArgument();
	wxString outut_chimera_cmd = my_current_job.arguments[2].ReturnStringArgument();
	float tolerance = my_current_job.arguments[3].ReturnFloatArgument();
	float margin = my_current_job.arguments[4].ReturnFloatArgument();
	float x_dimension = my_current_job.arguments[5].ReturnFloatArgument();
	float y_dimension = my_current_job.arguments[6].ReturnFloatArgument();
	float initial_x = my_current_job.arguments[7].ReturnFloatArgument();
	float initial_y = my_current_job.arguments[8].ReturnFloatArgument();
	float initial_z = my_current_job.arguments[9].ReturnFloatArgument();
	float initial_angle = my_current_job.arguments[10].ReturnFloatArgument();
	bool perform_search = my_current_job.arguments[11].ReturnBoolArgument();

	float angular_step = 0.5f;
	float xy_search_range = 1000.0f;
	float z_search_range = 20000.0f;
	float xyz_refine_range = 100.0f;
	float x_step = 1.0f;
	float y_step = 1.0f;
	float z_step = 1.0f;
	float psi_step = 0.1f;
	float theta_step = 0.1f;
	float phi_step = 0.1f;
	float euler_range = 10.0f;

	int i, j, k;
	int ii, jj, kk;
	int number_of_matches = 0;
	int number_ref_in_view = 0;
	float score;
	float best_score;
	int best_i, best_ii;
	int best_j;
	float difference, smallest_diff;
	float angle, best_search_angle, best_refine_angle;
	float x_coordinate_2d, y_coordinate_2d;
	float z_coordinate_2d = 0.0f;
	float x_coordinate_3d, y_coordinate_3d, z_coordinate_3d;
	float **ref_coordinates;
	float **align_coordinates;
	int angle_i, number_of_angles;
	int x_i, number_of_x_steps;
	int y_i, number_of_y_steps;
	int z_i, number_of_z_steps;
	float x, y, z;
	float psi, theta, phi;
	float best_psi, best_theta, best_phi;
	int psi_i, number_of_psi_angles;
	int theta_i, number_of_theta_angles;
	int phi_i, number_of_phi_angles;
	float best_x, best_y;
	float xyz_offset[3];
	float ref_center_of_mass[3];
	float align_center_of_mass[3];
	float anchor_align_x, anchor_align_y;
	float anchor_ref_x, anchor_ref_y;
	float std_x = 0.0f, std_y = 0.0f, std_z = 0.0f;
	float precision_ref, recall_ref;
	float precision_align, recall_align;
	int tp;
//	int fp;

	NumericTextFile input_ref_file(input_ref_coordinates, OPEN_TO_READ);
	NumericTextFile input_align_file(input_align_coordinates, OPEN_TO_READ);

	AnglesAndShifts coordinate_transformation;

//	float ref_distances[input_ref_file.number_of_lines - 1];
//	float align_distances[input_align_file.number_of_lines - 1];
	int align_pairs[input_align_file.number_of_lines];
	for (i = 0; i < input_align_file.number_of_lines; i++) align_pairs[i] = -1;
	int ref_pairs[input_ref_file.number_of_lines];
	for (i = 0; i < input_ref_file.number_of_lines; i++) ref_pairs[i] = -1;
	float ref_distance[input_ref_file.number_of_lines];
	ZeroFloatArray(ref_distance, input_ref_file.number_of_lines);

	Allocate2DFloatArray(ref_coordinates, input_ref_file.number_of_lines, 4);
	Allocate2DFloatArray(align_coordinates, input_align_file.number_of_lines, 5);

	for (i = 0; i < input_ref_file.number_of_lines; i++) input_ref_file.ReadLine(ref_coordinates[i]);
	wxPrintf("\nRead %3i reference coordinates\n\n", input_ref_file.number_of_lines);
	ii = 0;
	for (i = 0; i < input_align_file.number_of_lines; i++)
	{
		input_align_file.ReadLine(align_coordinates[ii]);
		align_coordinates[ii][4] = i;
		if (align_coordinates[ii][0] < margin || align_coordinates[ii][0] > x_dimension - margin || align_coordinates[ii][1] < margin || align_coordinates[ii][1] > y_dimension - margin)
		{
			wxPrintf("Align coordinates %3i excluded\n", i + 1);
//			wxPrintf("%g %g %g %g %g\n", margin, align_coordinates[ii][0], align_coordinates[ii][1], x_dimension, y_dimension);
		}
		else
		{
			ii++;
		}
	}
	input_align_file.number_of_lines = ii;
	wxPrintf("\nRead %3i align coordinates\n", input_align_file.number_of_lines);

//	for (i = 0; i < input_ref_file.number_of_lines; i++) wxPrintf("ref   x,y,z = %10.2f %10.2f %10.2f\n", ref_coordinates[i][0], ref_coordinates[i][1], ref_coordinates[i][2]);
//	for (i = 0; i < input_align_file.number_of_lines; i++) wxPrintf("align x,y,z = %10.2f %10.2f %10.2f\n", align_coordinates[i][0], align_coordinates[i][1], align_coordinates[i][2]);

	ZeroFloatArray(ref_center_of_mass, 3);
	ZeroFloatArray(align_center_of_mass, 3);
	for (i = 0; i < input_align_file.number_of_lines; i++) {align_center_of_mass[0] += align_coordinates[i][0]; align_center_of_mass[1] += align_coordinates[i][1]; align_center_of_mass[2] += align_coordinates[i][2];}
	for (i = 0; i < input_align_file.number_of_lines; i++) {ref_center_of_mass[0] += ref_coordinates[i][0]; ref_center_of_mass[1] += ref_coordinates[i][1]; ref_center_of_mass[2] += ref_coordinates[i][2];}
	for (i = 0; i < 3; i++) {align_center_of_mass[i] /= input_align_file.number_of_lines; ref_center_of_mass[i] /= input_align_file.number_of_lines;}
	align_center_of_mass[0] = x_dimension / 2.0f;
	align_center_of_mass[1] = y_dimension / 2.0f;

	FILE *cmd_file;
	cmd_file = fopen(outut_chimera_cmd, "w");
	if (cmd_file == NULL)
	{
		MyPrintWithDetails("Error: Cannot open Chimera cmd file (%s) for write\n", outut_chimera_cmd);
		DEBUG_ABORT;
	}

	if (perform_search)
	{
		wxPrintf("\nSearching...\n\n");

		number_of_angles = myroundint(360.0f/angular_step);
		ProgressBar *search_progress = new ProgressBar(number_of_angles);

		best_score = 0.0f;
		for (angle_i = 0; angle_i < number_of_angles; angle_i++)
	//	for (angle_i = number_of_angles/2 - 100; angle_i < number_of_angles/2 + 100; angle_i++)
		{
			angle = angle_i * angular_step;
			coordinate_transformation.GenerateRotationMatrix2D(angle);
			for (i = 0; i < input_ref_file.number_of_lines; i++)
			{
				for (ii = 0; ii < input_align_file.number_of_lines; ii++)
				{
					score = 0.0f;
					for (jj = ii + 1; jj < input_align_file.number_of_lines; jj++)
					{
	//					if (ii != jj)
	//					{
							smallest_diff = tolerance;
							for (j = i + 1; j < input_ref_file.number_of_lines; j++)
							{
	//							if (i != j)
	//							{
									x_coordinate_2d = align_coordinates[jj][0] - align_coordinates[ii][0];
									y_coordinate_2d = align_coordinates[jj][1] - align_coordinates[ii][1];
									coordinate_transformation.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
									difference = sqrtf(powf(x_coordinate_3d - ref_coordinates[j][0] + ref_coordinates[i][0], 2) + powf(y_coordinate_3d - ref_coordinates[j][1] + ref_coordinates[i][1], 2));
									if (difference < smallest_diff) smallest_diff = difference;
	//							}
							}
							score += powf(tolerance - smallest_diff, 2);
	//					}
					}
					if (score > best_score)
					{
						best_score = score;
						best_i = i;
						best_ii = ii;
						best_search_angle = angle;
	//					wxPrintf("Score, angle = %g %g\n", sqrtf(best_score), best_angle);
					}
				}
			}
			search_progress->Update(angle_i + 1);
		}
		delete search_progress;
		anchor_align_x = align_coordinates[best_ii][0];
		anchor_align_y = align_coordinates[best_ii][1];
		anchor_ref_x = ref_coordinates[best_i][0];
		anchor_ref_y = ref_coordinates[best_i][1];
	}
	else
	{
		best_i = -1;
		best_ii = -1;
		best_search_angle = initial_angle;
		anchor_align_x = align_center_of_mass[0];
		anchor_align_y = align_center_of_mass[1];
		anchor_ref_x = xyz_offset[0] + align_center_of_mass[0];
		anchor_ref_y = xyz_offset[1] + align_center_of_mass[1];
// score += powf((x_coordinate_3d + xyz_offset[0] + align_center_of_mass[0] - ref_coordinates[align_pairs[ii]][0]), 2) + powf((y_coordinate_3d + ref_center_of_mass[1] + best_y + y - ref_coordinates[align_pairs[ii]][1]), 2);
// xyz_offset[0] = ref_center_of_mass[0] - align_center_of_mass[0] + best_x + x;
// xyz_offset[0] + align_center_of_mass[0] = ref_center_of_mass[0] + best_x + x;
	}

//	xyz_offset[0] = anchor_ref_x;
//	xyz_offset[1] = anchor_ref_y;

//	wxPrintf("\nIn-plane angle (deg) = %g\n", best_angle);

	wxPrintf("\nMatching coordinates (A)\n\n");

	if (perform_search)
	{
		number_of_matches++;
		wxPrintf("%3i ref %3i xy = %8.2f %8.2f   align %3i xy = %8.2f %8.2f   in-plane distance = %8.2f\n", number_of_matches, best_i + 1, anchor_ref_x, anchor_ref_y, best_ii + 1, anchor_ref_x, anchor_ref_y, smallest_diff);
		align_pairs[best_ii] = best_i;
		ref_pairs[best_i] = best_ii;
	}
//	wxPrintf("best_i = %i\n", best_i);
	coordinate_transformation.GenerateRotationMatrix2D(best_search_angle);
	for (jj = 0; jj < input_align_file.number_of_lines; jj++)
	{
		if (best_ii != jj)
		{
//			align_distances[0] = sqrtf(powf((anchor_align_x - align_coordinates[jj][0]), 2) + powf((anchor_align_y - align_coordinates[jj][1]), 2));
			smallest_diff = tolerance;
			for (j = 0; j < input_ref_file.number_of_lines; j++)
			{
				if (best_i != j)
				{
//					ref_distances[0] = sqrtf(powf((anchor_ref_x - ref_coordinates[j][0]), 2) + powf((anchor_ref_y - ref_coordinates[j][1]), 2));
					x_coordinate_2d = align_coordinates[jj][0] - anchor_align_x;
					y_coordinate_2d = align_coordinates[jj][1] - anchor_align_y;
					coordinate_transformation.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
					// x_coordinate_3d + anchor_align_x - anchor_ref_x
					difference = sqrtf(powf(x_coordinate_3d + anchor_ref_x - ref_coordinates[j][0], 2) + powf(y_coordinate_3d + anchor_ref_y - ref_coordinates[j][1], 2));
//					difference = fabsf(align_distances[0] - ref_distances[0]);
					if (difference < smallest_diff) {smallest_diff = difference; best_j = j;}
				}
			}
			if (smallest_diff < tolerance)
			{
				if (ref_pairs[best_j] >= 0)
				{
					if (ref_distance[best_j] > smallest_diff)
					{
						wxPrintf("%3i ref %3i peak = %8.4f xy = %8.2f %8.2f   align %3i xy = %8.2f %8.2f   in-plane distance = %8.2f   updated\n", number_of_matches, best_j + 1, ref_coordinates[best_j][3], ref_coordinates[best_j][0], ref_coordinates[best_j][1], jj + 1, x_coordinate_3d + anchor_ref_x, y_coordinate_3d + anchor_ref_y, smallest_diff);
						ref_distance[best_j] = smallest_diff;
						for (ii = 0; ii < input_align_file.number_of_lines; ii++) if (align_pairs[ii] == best_j) align_pairs[ii] = -1;
						align_pairs[jj] = best_j;
					}
					else
					{
						wxPrintf("%3i ref %3i peak = %8.4f xy = %8.2f %8.2f   align %3i xy = %8.2f %8.2f   in-plane distance = %8.2f   discarded\n", number_of_matches, best_j + 1, ref_coordinates[best_j][3], ref_coordinates[best_j][0], ref_coordinates[best_j][1], jj + 1, x_coordinate_3d + anchor_ref_x, y_coordinate_3d + anchor_ref_y, smallest_diff);
//						align_pairs[jj] = -1;
					}
				}
				else
				{
					number_of_matches++;
					ref_distance[best_j] = smallest_diff;
					wxPrintf("%3i ref %3i peak = %8.4f xy = %8.2f %8.2f   align %3i xy = %8.2f %8.2f   in-plane distance = %8.2f\n", number_of_matches, best_j + 1, ref_coordinates[best_j][3], ref_coordinates[best_j][0], ref_coordinates[best_j][1], jj + 1, x_coordinate_3d + anchor_ref_x, y_coordinate_3d + anchor_ref_y, smallest_diff);
					align_pairs[jj] = best_j;
					ref_pairs[best_j] = jj;
				}
//				wxPrintf("ref xyz = %8.2f %8.2f %8.2f   align xyz = %8.2f %8.2f %8.2f   distance = %8.2f\n", ref_coordinates[best_j][0], ref_coordinates[best_j][1], ref_coordinates[best_j][2], x_coordinate_3d + anchor_ref_x, y_coordinate_3d + anchor_ref_y, 1300.0f + align_coordinates[jj][2], smallest_diff);
//				wxPrintf("%3i ref xy = %8.2f %8.2f   align xy = %8.2f %8.2f   in-plane distance = %8.2f\n", number_of_matches, ref_coordinates[best_j][0], ref_coordinates[best_j][1], x_coordinate_3d + anchor_ref_x, y_coordinate_3d + anchor_ref_y, smallest_diff);
//				wxPrintf("best_j = %i\n", best_j);
			}
		}
	}

	number_of_z_steps = myroundint(z_search_range/z_step);

	best_score = 0.0f;
	for (z_i = -number_of_z_steps / 2; z_i < number_of_z_steps / 2; z_i++)
	{
		z = z_i * z_step;
		score = 0.0f;
		for (ii = 0; ii < input_align_file.number_of_lines; ii++)
		{
			if (align_pairs[ii] >= 0)
			{
				difference = fabsf(align_coordinates[ii][2] + z - ref_coordinates[align_pairs[ii]][2]);
				if (difference < 3.0f * tolerance) score += powf(3.0f * tolerance - difference, 2);
			}
		}
		if (score > best_score)
		{
			best_score = score;
			xyz_offset[2] = z;
//			wxPrintf("Score, z offset = %g %g\n", sqrtf(best_score), xyz_offset[2]);
		}
	}
//	for (i = 0; i < input_align_file.number_of_lines; i++) if (align_pairs[i] >= 0) wxPrintf("%g %g\n", align_coordinates[i][2] + z, ref_coordinates[align_pairs[i]][2]);

//	for (i = 0; i < input_align_file.number_of_lines; i++) if (align_pairs[i] >= 0) {align_center_of_mass[0] += align_coordinates[i][0]; align_center_of_mass[1] += align_coordinates[i][1]; align_center_of_mass[2] += align_coordinates[i][2];}
//	for (i = 0; i < input_align_file.number_of_lines; i++) if (align_pairs[i] >= 0) {ref_center_of_mass[0] += ref_coordinates[align_pairs[i]][0]; ref_center_of_mass[1] += ref_coordinates[align_pairs[i]][1]; ref_center_of_mass[2] += ref_coordinates[align_pairs[i]][2];}
//	for (i = 0; i < 3; i++) {align_center_of_mass[i] /= number_of_matches; ref_center_of_mass[i] /= number_of_matches;}

	if (perform_search)
	{
		x_coordinate_2d = anchor_align_x - align_center_of_mass[0];
		y_coordinate_2d = anchor_align_y - align_center_of_mass[1];
		coordinate_transformation.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
		best_x = -x_coordinate_3d - ref_center_of_mass[0] + anchor_ref_x;
		best_y = -y_coordinate_3d - ref_center_of_mass[1] + anchor_ref_y;
	//	wxPrintf("\nalign_center    = %8.2f %8.2f\n", align_center_of_mass[0], align_center_of_mass[1]);
	//	wxPrintf("\nref_center      = %8.2f %8.2f\n", ref_center_of_mass[0], ref_center_of_mass[1]);
	//	wxPrintf("\nref_coordinates = %8.2f %8.2f\n", anchor_ref_x, anchor_ref_y);
	//	wxPrintf("\ncoordinate_3d   = %8.2f %8.2f\n", x_coordinate_3d, y_coordinate_3d);
	//	best_x = ref_center_of_mass[0] - anchor_ref_x + align_center_of_mass[0];
	//	best_y = ref_center_of_mass[1] - anchor_ref_y + align_center_of_mass[1];

		wxPrintf("\nRough xyz offset (A), in-plane rotation (deg) = %8.2f %8.2f %8.2f %8.2f\n", ref_center_of_mass[0] - align_center_of_mass[0] + best_x, ref_center_of_mass[1] - align_center_of_mass[1] + best_y, xyz_offset[2], best_search_angle);

		wxPrintf("\nRefining parameters...\n\n");

		number_of_x_steps = myroundint(xy_search_range / x_step);
		number_of_y_steps = myroundint(xy_search_range / y_step);
		number_of_angles = 100;
		angular_step /= 50.0;

		best_score = FLT_MAX;

		ProgressBar *refine_progress = new ProgressBar(number_of_angles + 1);

		for (angle_i = - number_of_angles / 2; angle_i < number_of_angles / 2; angle_i++)
		{
			angle = best_search_angle + angle_i * angular_step;
			coordinate_transformation.GenerateRotationMatrix2D(angle);

			for (y_i = -number_of_y_steps / 2; y_i < number_of_y_steps / 2; y_i++)
			{
				y = y_i * y_step;
				for (x_i = -number_of_x_steps / 2; x_i < number_of_x_steps / 2; x_i++)
				{
					x = x_i * x_step;
					score = 0.0f;
					for (ii = 0; ii < input_align_file.number_of_lines; ii++)
					{
						if (align_pairs[ii] >= 0)
						{
							x_coordinate_2d = align_coordinates[ii][0] - align_center_of_mass[0];
							y_coordinate_2d = align_coordinates[ii][1] - align_center_of_mass[1];
							coordinate_transformation.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
							score += powf(x_coordinate_3d + ref_center_of_mass[0] + best_x + x - ref_coordinates[align_pairs[ii]][0], 2) + powf(y_coordinate_3d + ref_center_of_mass[1] + best_y + y - ref_coordinates[align_pairs[ii]][1], 2);
						}
					}
					if (score < best_score)
					{
						best_score = score;
						xyz_offset[0] = ref_center_of_mass[0] - align_center_of_mass[0] + best_x + x;
						xyz_offset[1] = ref_center_of_mass[1] - align_center_of_mass[1] + best_y + y;
						best_refine_angle = angle;
		//				wxPrintf("Score, x,y offset = %g %g %g\n", sqrtf(best_score), x, y);
					}
				}
			}
			refine_progress->Update(angle_i + number_of_angles / 2 + 1);
		}
		delete refine_progress;
	}
	else
	{
		xyz_offset[0] = initial_x;
		xyz_offset[1] = initial_y;
		xyz_offset[2] = initial_z;
		best_refine_angle = initial_angle;
	}

	wxPrintf("\nxyz offset (A), in-plane rotation (deg) = %8.2f %8.2f %8.2f %8.2f\n", xyz_offset[0], xyz_offset[1], xyz_offset[2], best_refine_angle);

	fprintf(cmd_file, "turn z %g center %g,%g,%g coord #0 model #1\n", best_refine_angle, align_center_of_mass[0], align_center_of_mass[1], align_center_of_mass[2]);
	fprintf(cmd_file, "move x %g coord #0 model #1\n", xyz_offset[0]);
	fprintf(cmd_file, "move y %g coord #0 model #1\n", xyz_offset[1]);
	fprintf(cmd_file, "move z %g coord #0 model #1\n", xyz_offset[2]);
	fclose(cmd_file);

//	wxPrintf("\nReference coordinates in HRTM field of view, x min /max, y min /max (A) = %8.2f/%8.2f %8.2f/%8.2f\n\n", xyz_offset[0], xyz_offset[0] + x_dimension, xyz_offset[1], xyz_offset[1] + y_dimension);
	wxPrintf("\nReference coordinates in HRTM field of view\n\n");

	tp = 0;
//	fp = 0;
	coordinate_transformation.GenerateRotationMatrix2D(-best_refine_angle);
	for (i = 0; i < input_ref_file.number_of_lines; i++)
	{
		x_coordinate_2d = ref_coordinates[i][0] - align_center_of_mass[0] - xyz_offset[0];
		y_coordinate_2d = ref_coordinates[i][1] - align_center_of_mass[1] - xyz_offset[1];
		coordinate_transformation.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
		x_coordinate_3d += align_center_of_mass[0];
		y_coordinate_3d += align_center_of_mass[1];

		if (x_coordinate_3d >= margin && x_coordinate_3d <= x_dimension - margin && y_coordinate_3d >= margin && y_coordinate_3d <= y_dimension - margin)
		{
			number_ref_in_view++;
			if (ref_pairs[i] >= 0)
			{
//				wxPrintf("%3i  xyz (A) = %8.2f %8.2f %8.2f   xy_rotated = %8.2f %8.2f %8.2f   align pair %3i\n", i + 1, ref_coordinates[i][0], ref_coordinates[i][1], ref_coordinates[i][2], x_coordinate_3d, y_coordinate_3d, ref_pairs[i] + 1);
				wxPrintf("%3i  xyz (A) = %8.2f %8.2f %8.2f   ref peak = %8.4f   xy_rotated = %8.2f %8.2f   align peak = %8.4f   align pair %3i\n", i + 1, ref_coordinates[i][0], ref_coordinates[i][1], ref_coordinates[i][2], \
						ref_coordinates[i][3], x_coordinate_3d, y_coordinate_3d, align_coordinates[ref_pairs[i]][3], myroundint(align_coordinates[ref_pairs[i]][4]) + 1);
				tp++;
			}
			else
			{
				wxPrintf("%3i  xyz (A) = %8.2f %8.2f %8.2f   ref peak = %8.4f   xy_rotated = %8.2f %8.2f\n", i + 1, ref_coordinates[i][0], ref_coordinates[i][1], ref_coordinates[i][2], ref_coordinates[i][3], x_coordinate_3d, y_coordinate_3d);
//				fp++;
			}
		}
		else
		{
			if (ref_pairs[i] >= 0)
			{
				align_pairs[ref_pairs[i]] = -1;
				ref_pairs[i] = -1;
			}
		}
	}
	precision_align = float(tp)/number_ref_in_view;
	recall_align = float(tp)/input_align_file.number_of_lines;
//	wxPrintf("\nTotal, matched = %3i  %3i\n", number_ref_in_view, number_of_matches);
	wxPrintf("\nTotal, matched = %3i  %3i\n", number_ref_in_view, tp);
	wxPrintf("\nPrecision, recall, F1 score = %8.4f %8.4f %8.4f\n", precision_align, recall_align, 2.0f * precision_align * recall_align / (precision_align + recall_align));

	wxPrintf("\nAligned coordinates\n\n");

	tp = 0;
//	fp = 0;
	coordinate_transformation.GenerateRotationMatrix2D(best_refine_angle);
	for (ii = 0; ii < input_align_file.number_of_lines; ii++)
	{
		x_coordinate_2d = align_coordinates[ii][0] - align_center_of_mass[0];
		y_coordinate_2d = align_coordinates[ii][1] - align_center_of_mass[1];
		coordinate_transformation.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
		if (align_pairs[ii] >= 0 && ref_pairs[align_pairs[ii]] >= 0)
		{
			wxPrintf("%3i  xyz (A) = %8.2f %8.2f %8.2f   align peak = %8.4f   xyz_rotated = %8.2f %8.2f %8.2f   ref peak = %8.4f   ref pair %3i\n", myroundint(align_coordinates[ii][4]) + 1, align_coordinates[ii][0], align_coordinates[ii][1], align_coordinates[ii][2], align_coordinates[ii][3], \
					x_coordinate_3d + xyz_offset[0] + align_center_of_mass[0], y_coordinate_3d + xyz_offset[1] + align_center_of_mass[1], align_coordinates[ii][2] + xyz_offset[2], ref_coordinates[align_pairs[ii]][3], align_pairs[ii] + 1);
//			wxPrintf("%3i  xyz (A) = %8.2f %8.2f %8.2f   ref pair  %3i\n", myroundint(align_coordinates[ii][3]) + 1, x_coordinate_3d + xyz_offset[0] + align_center_of_mass[0], y_coordinate_3d + xyz_offset[1] + align_center_of_mass[1], align_coordinates[ii][2] + xyz_offset[2], align_pairs[ii] + 1);
			std_x += powf(x_coordinate_3d + xyz_offset[0] + align_center_of_mass[0] - ref_coordinates[align_pairs[ii]][0], 2);
			std_y += powf(y_coordinate_3d + xyz_offset[1] + align_center_of_mass[1] - ref_coordinates[align_pairs[ii]][1], 2);
			std_z += powf(align_coordinates[ii][2] + xyz_offset[2] - ref_coordinates[align_pairs[ii]][2], 2);
			tp++;
		}
		else
		{
			wxPrintf("%3i  xyz (A) = %8.2f %8.2f %8.2f   align peak = %8.4f   xyz_rotated = %8.2f %8.2f %8.2f\n", myroundint(align_coordinates[ii][4]) + 1, align_coordinates[ii][0], align_coordinates[ii][1], align_coordinates[ii][2],  align_coordinates[ii][3], \
					x_coordinate_3d + xyz_offset[0] + align_center_of_mass[0], y_coordinate_3d + xyz_offset[1] + align_center_of_mass[1], align_coordinates[ii][2] + xyz_offset[2]);
//			wxPrintf("%3i  xyz (A) = %8.2f %8.2f %8.2f\n", myroundint(align_coordinates[ii][3]) + 1, x_coordinate_3d + xyz_offset[0] + align_center_of_mass[0], y_coordinate_3d + xyz_offset[1] + align_center_of_mass[1], align_coordinates[ii][2] + xyz_offset[2]);
//			fp++;
		}
	}
	precision_ref = float(tp)/input_align_file.number_of_lines;
	recall_ref = float(tp)/number_ref_in_view;
//	wxPrintf("\nTotal, matched = %3i  %3i\n", input_align_file.number_of_lines, number_of_matches);
	wxPrintf("\nTotal, matched = %3i  %3i\n", input_align_file.number_of_lines, tp);
	wxPrintf("\nPrecision, recall, F1 score = %8.4f %8.4f %8.4f\n", precision_ref, recall_ref, 2.0f * precision_ref * recall_ref / (precision_ref + recall_ref));

	wxPrintf("\nSTD xy, z = %8.2f %8.2f\n\n", sqrtf((std_x + std_y) / number_of_matches), sqrtf(std_z / number_of_matches));
/*
	number_of_psi_angles = myroundint(10.0f * angular_step/psi_step);
	number_of_theta_angles = myroundint(euler_range/theta_step);
	number_of_phi_angles = myroundint(euler_range/phi_step);
	number_of_z_steps = myroundint(xyz_refine_range/z_step);

	ProgressBar *refine_progress = new ProgressBar(number_of_z_steps + 1);

	best_score = 0;
	for (z_i = -number_of_z_steps / 2; z_i < number_of_z_steps / 2; z_i++)
	{
		z = z_i * z_step;
		for (phi_i = -number_of_phi_angles / 2; phi_i < number_of_phi_angles / 2; phi_i++)
		{
			phi = phi_i * phi_step;
			for (theta_i = -number_of_theta_angles / 2; theta_i < number_of_theta_angles / 2; theta_i++)
			{
				theta = theta_i * theta_step;
				for (psi_i = -number_of_psi_angles / 2; psi_i < number_of_psi_angles / 2; psi_i++)
				{
					psi = psi_i * psi_step;
					coordinate_transformation.GenerateEulerMatrices(phi, theta, best_angle + psi);
					score = 0.0f;
					for (ii = 0; ii < input_align_file.number_of_lines; ii++)
					{
						if (align_pairs[ii] >= 0) score += powf(align_coordinates[ii][2] + z - ref_coordinates[align_pairs[ii]][2], 2);
					}
					if (score > best_score)
					{
						best_score = score;
						xyz_offset[2] = z;
						best_psi = best_angle + psi;
						best_theta = theta;
						best_phi = phi;
			//			wxPrintf("Score, z offset = %g %g\n", sqrtf(best_score), best_z);
					}
				}
			}
		}
		refine_progress->Update(z_i + number_of_z_steps / 2 + 1);
	}
	delete refine_progress;
*/

/*
	// First find best relative distance match
	best_score = 0.0f;
	for (i = 0; i < input_ref_file.number_of_lines; i++)
	{
		k = 0;
		for (j = 0; j < input_ref_file.number_of_lines; j++)
		{
			if (i != j) {ref_distances[k] = sqrtf(powf((ref_coordinates[i][0] - ref_coordinates[j][0]), 2) + powf((ref_coordinates[i][1] - ref_coordinates[j][1]), 2) + powf((ref_coordinates[i][2] - ref_coordinates[j][2]), 2)); k++;}
		}
		for (ii = 0; ii < input_align_file.number_of_lines; ii++)
		{
			kk = 0;
			for (jj = 0; jj < input_align_file.number_of_lines; jj++)
			{
				if (ii != jj) {align_distances[kk] = sqrtf(powf((align_coordinates[ii][0] - align_coordinates[jj][0]), 2) + powf((align_coordinates[ii][1] - align_coordinates[jj][1]), 2) + powf((align_coordinates[ii][2] - align_coordinates[jj][2]), 2)); kk++;}
			}
			score = 0.0f;
			for (jj = 0; jj < kk; jj++)
			{
				smallest_diff = tolerance;
				for (j = 0; j < k; j++)
				{
					difference = fabsf(align_distances[jj] - ref_distances[j]);
					if (difference < smallest_diff) smallest_diff = difference;
//					wxPrintf("diff = %g\n", difference);
				}
				if (smallest_diff < tolerance) score += powf(tolerance - smallest_diff, 2);
			}
			if (score > best_score)
			{
				best_score = score;
				wxPrintf("Score = %g\n", best_score);
				best_i = i;
				best_ii = ii;
			}
		}
	}

	// Next, find the in-plane rotation
	best_score = 0.0f;
	for (angle = 0.0f; angle < 360.0f; angle += 0.1f)
	{
		coordinate_transformation.GenerateRotationMatrix2D(angle);
		score = 0.0f;
		for (jj = 0; jj < input_align_file.number_of_lines; jj++)
		{
			if (best_ii != jj)
			{
				smallest_diff = tolerance;
				for (j = 0; j < input_ref_file.number_of_lines; j++)
				{
					if (best_i != j)
					{
						x_coordinate_2d = align_coordinates[jj][0] - align_coordinates[best_ii][0];
						y_coordinate_2d = align_coordinates[jj][1] - align_coordinates[best_ii][1];
						coordinate_transformation.euler_matrix.RotateCoords2D(x_coordinate_2d, y_coordinate_2d, x_coordinate_3d, y_coordinate_3d);
						difference = sqrtf(powf((x_coordinate_3d + ref_coordinates[best_i][0] - ref_coordinates[j][0]), 2) + powf((y_coordinate_3d + ref_coordinates[best_i][1] - ref_coordinates[j][1]), 2));
						if (difference < smallest_diff) smallest_diff = difference;
					}
				}
				if (smallest_diff < tolerance) score += powf(tolerance - smallest_diff, 2);
			}
		}
		if (score > best_score)
		{
			best_score = score;
//			wxPrintf("Score = %i\n", best_score);
			best_angle = angle;
		}
	}
*/

	return true;
}
