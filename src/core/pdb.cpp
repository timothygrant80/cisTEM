#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfAtoms);
WX_DEFINE_OBJARRAY(ArrayOfParticleTrajectories);
//WX_DEFINE_OBJARRAY(ArrayOfParticleInstances);



const double MIN_PADDING_Z    = 4;
const int MAX_XY_DIMENSION = 4096;

// Define fixed width limits for PDB reading
#define NAMESTART	12  //Char
#define NAMELENGTH	 4
#define RESIDUESTART 16
#define RESIDUELENGTH 5
#define XSTART 		30  // Float
#define YSTART		38
#define ZSTART		46
#define XYZLENGTH	 8
#define OCCUPANCYSTART 54 //Float
#define OCCUPANCYLENGTH 6
#define BFACTORSTART   60 //Float
#define BFACTORLENGTH 	6
#define WATERSTART	   72 // Char, where VMD put WTAA for a TIP3 Water. This is a deprecated segment id spot also used still by Chimera for pdbSegment attribute
#define WATERLENGTH     4
#define ELEMENTSTART   76 // Char, sometimes length = 3, e.g. N1+ or O1- :: I expect not to see O1+ s.t. the charge may be inferred from the extra "1" and element name
#define ELEMENTLENGTH   2
#define CHARGESTART	   78 //Char, not enabled, set to 0.0
#define CHARGELENGTH	2
#define REMARKSTART		7 // Remark ID 351 used to hold particle instances.
#define REMARKLENGTH		4
#define XPARTICLE		11
#define YPARTICLE		19
#define ZPARTICLE		27
#define E1PARTICLE		35
#define E2PARTICLE		43
#define E3PARTICLE		51
#define PARTICLELENGTH	8



PDB::PDB()
{
	input_file_stream = NULL;
	input_text_stream = NULL;
	output_file_stream = NULL;
	output_text_stream = NULL;
	Atom dummy_atom;
	my_atoms.Alloc(1);
	my_atoms.Add(dummy_atom,1);

    SetEmpty();
}


PDB::PDB(long number_of_non_water_atoms, float cubic_size, float wanted_pixel_size, int minimum_paddeding_x_and_y, double minimum_thickness_z)
{

	input_file_stream = NULL;
	input_text_stream = NULL;
	output_file_stream = NULL;
	output_text_stream = NULL;
	// Create a total PDB object to hold all the atoms in a specimen at a given time in the trajectory
	Atom dummy_atom;
	my_atoms.Alloc(number_of_non_water_atoms);
	my_atoms.Add(dummy_atom,number_of_non_water_atoms);
	this->cubic_size = cubic_size;

	this->use_provided_com = false;


	this->MIN_PADDING_XY = minimum_paddeding_x_and_y;
	this->MIN_THICKNESS  = minimum_thickness_z;

    SetEmpty();
	this->pixel_size = wanted_pixel_size;

}

PDB::PDB(wxString Filename, long wanted_access_type, float wanted_pixel_size, long wanted_records_per_line, int minimum_paddeding_x_and_y, double minimum_thickness_z)
{
	input_file_stream = NULL;
	input_text_stream = NULL;
	output_file_stream = NULL;
	output_text_stream = NULL;

	this->use_provided_com = false;

	this->MIN_PADDING_XY = minimum_paddeding_x_and_y;
	this->MIN_THICKNESS  = minimum_thickness_z;


    SetEmpty();
	this->pixel_size = wanted_pixel_size;
	Open(Filename, wanted_access_type, wanted_records_per_line);
}

PDB::PDB(wxString Filename, long wanted_access_type, float wanted_pixel_size, long wanted_records_per_line, int minimum_paddeding_x_and_y, double minimum_thickness_z, double *COM)
{
	input_file_stream = NULL;
	input_text_stream = NULL;
	output_file_stream = NULL;
	output_text_stream = NULL;

	this->MIN_PADDING_XY = minimum_paddeding_x_and_y;
	this->MIN_THICKNESS  = minimum_thickness_z;

	this->use_provided_com = true;

	for (int iCOM = 0; iCOM < 3; iCOM++)
	{
		this->center_of_mass[iCOM] = COM[iCOM];
		wxPrintf("Using provided center of mass %d %3.3f\n",iCOM,this->center_of_mass[iCOM]);
	}


    SetEmpty();
	this->pixel_size = wanted_pixel_size;
	Open(Filename, wanted_access_type, wanted_records_per_line);
}


PDB::~PDB()
{
	Close();
}

void PDB::Open(wxString Filename, long wanted_access_type, long wanted_records_per_line)
{
	access_type = wanted_access_type;
	records_per_line = wanted_records_per_line;
	text_filename = Filename;

	if (access_type == OPEN_TO_READ)
	{
		if (input_file_stream != NULL)
		{
			if (input_file_stream->GetFile()->IsOpened() == true)
			{
				MyPrintWithDetails("File already Open\n");
				DEBUG_ABORT;
			}

		}
	}
	else
	if (access_type == OPEN_TO_WRITE)
	{
		records_per_line = wanted_records_per_line;

		if (records_per_line <= 0)
		{
			MyPrintWithDetails("PDB asked to OPEN_TO_WRITE, but with erroneous records per line\n");
			DEBUG_ABORT;
		}

		if (output_file_stream != NULL)
		{
			if (output_file_stream->GetFile()->IsOpened() == true)
			{
				MyPrintWithDetails("File already Open\n");
				DEBUG_ABORT;
			}

		}


	}
	else
	{
		MyPrintWithDetails("Unknown access type!\n");
		DEBUG_ABORT;
	}

    SetEmpty();
	Init();
}

void PDB::Close()
{
	if (input_text_stream != NULL) delete input_text_stream;
	if (output_text_stream != NULL) delete output_text_stream;



	if (output_file_stream != NULL)
	{
		if (output_file_stream->GetFile()->IsOpened() == true) output_file_stream->GetFile()->Close();
		delete output_file_stream;
	}

	if (input_file_stream != NULL)
	{
		if (input_file_stream->GetFile()->IsOpened() == true) input_file_stream->GetFile()->Close();
		delete input_file_stream;
	}

	input_file_stream = NULL;
	input_text_stream = NULL;
	output_file_stream = NULL;
	output_text_stream = NULL;
}

void PDB::SetEmpty()
{
	this->vol_nX = 0;
	this->vol_nY = 0;
	this->vol_nZ = 0;
	this->vol_oX = 0;
	this->vol_oY = 0;
	this->vol_oZ = 0;
	this->vol_angX = 0;
	this->vol_angY = 0;
	this->vol_angZ = 0;
	this->atomic_volume = 0;
	this->pixel_size = 0;
	this->number_of_particles_initialized = 0;
	if (! use_provided_com)
	{
		this->center_of_mass[0] = 0;
		this->center_of_mass[1] = 0;
		this->center_of_mass[2] = 0;
	}
}

void PDB::Init()
{



	if (access_type == OPEN_TO_READ)
	{
		wxString current_line;
		wxString token;
		double temp_double;
		int current_records_per_line;
		int old_records_per_line = -1;
		long current_atom_number = 0;
		float wanted_origin_x;
		float wanted_origin_y;
		float wanted_origin_z;
		float euler1;
		float euler2;
		float euler3; // from pdb




		input_file_stream = new wxFileInputStream(text_filename);
		input_text_stream = new wxTextInputStream(*input_file_stream);

		if (input_file_stream->IsOk() == false)
		{
			MyPrintWithDetails("Attempt to access %s for reading failed\n",text_filename);
			DEBUG_ABORT;
		}

		// work out the records per line and how many lines

		this->number_of_lines = 0;
		this->number_of_atoms = 0;



		while (input_file_stream->Eof() == false)
		{
			current_line = input_text_stream->ReadLine();
			current_line.Trim(false);

			if (current_line.StartsWith("#") != true && current_line.StartsWith("C") != true && current_line.Length() > 0)
			{
				number_of_lines++;

				if (current_line.StartsWith("ATOM") == true)
				{
					number_of_atoms++;

				}


			}
		}


		records_per_line = current_records_per_line;

		// rewind the file..

		Rewind();

		// Create the atom array, then loop back over the pdb to get the desired info.
		Atom dummy_atom;
		my_atoms.Alloc(number_of_atoms);
		my_atoms.Add(dummy_atom,number_of_atoms);



		while (input_file_stream->Eof() == false)
		{
			current_line = input_text_stream->ReadLine();
			current_line.Trim(false);

			if (current_line.StartsWith("ATOM") == true && current_line.Length() > 0)
			{


				wxString residue_name = current_line.Mid(RESIDUESTART,RESIDUELENGTH).Trim(true).Trim(false);

				my_atoms.Item(current_atom_number).name =  current_line.Mid(NAMESTART,NAMELENGTH).Trim(true).Trim(false);


				// Set all charge to zero - only override for Asp/Glu O that may be charged. Neutral at first in the simulation
				my_atoms.Item(current_atom_number).charge = 0;

				if (this->IsAcidicOxygen(my_atoms.Item(current_atom_number).name))
				{
					my_atoms.Item(current_atom_number).charge = 1;
				}


//				wxPrintf("%2.2f\n",my_atoms.Item(current_atom_number).relative_bfactor);

				if (current_line.Mid(XSTART,XYZLENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
				{
					MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(XSTART,XYZLENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
					DEBUG_ABORT;
				}
				my_atoms.Item(current_atom_number).x_coordinate = temp_double;

				if (current_line.Mid(YSTART,XYZLENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
				{
					MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(YSTART,XYZLENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
					DEBUG_ABORT;
				}
				my_atoms.Item(current_atom_number).y_coordinate = temp_double;

				if (current_line.Mid(ZSTART,XYZLENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
				{
					MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(ZSTART,XYZLENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
					DEBUG_ABORT;
				}

				my_atoms.Item(current_atom_number).z_coordinate = temp_double;

				if (current_line.Mid(OCCUPANCYSTART,OCCUPANCYLENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
				{
					MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(OCCUPANCYSTART,OCCUPANCYLENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
					DEBUG_ABORT;
				}

				my_atoms.Item(current_atom_number).occupancy = temp_double;

				if (current_line.Mid(BFACTORSTART,BFACTORLENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
				{
					MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(BFACTORSTART,BFACTORLENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
					DEBUG_ABORT;
				}

				my_atoms.Item(current_atom_number).bfactor = temp_double;



				// H(0),C(1),N(2),O(3),F(4),Na(5),Mg(6),P(7),S(8),Cl(9),K(10),Ca(11),Mn(12),Fe(13),Zn(14)
				wxString temp_name;
				temp_name = my_atoms.Item(current_atom_number).name;
				temp_name.Trim(true).Trim(false);


//				temp_name = current_line.Mid(WATERSTART,WATERLENGTH).Trim(true).Trim(false);
				// First, check to see if it is a water, if not check to see if the element name exists.
				if (temp_name.StartsWith("WTAA"))
				{
					// Set to oxygen for now - TODO FIXME
					my_atoms.Item(current_atom_number).atom_type =  water;
				}
				else
				{

					// Maybe this should be a switch statement
					if (temp_name.StartsWith("O") ) {
						my_atoms.Item(current_atom_number).atom_type = oxygen;
					} else if (temp_name.StartsWith("C")  ) {
						my_atoms.Item(current_atom_number).atom_type = carbon;
					} else if (temp_name.StartsWith("N")  ) {
						my_atoms.Item(current_atom_number).atom_type = nitrogen;
					} else if (temp_name.StartsWith("H")  ) {
						my_atoms.Item(current_atom_number).atom_type = hydrogen;
					} else if (temp_name.StartsWith("F")  ) {
						my_atoms.Item(current_atom_number).atom_type = fluorine;
					} else if (temp_name.StartsWith("Na")  ) {
						my_atoms.Item(current_atom_number).atom_type = sodium;
					} else if (temp_name.StartsWith("Mg")  ) {
						my_atoms.Item(current_atom_number).atom_type = magnesium;
					} else if (temp_name.StartsWith("MG")  ) {
						my_atoms.Item(current_atom_number).atom_type = magnesium;
					} else if (temp_name.StartsWith("P")  ) {
						my_atoms.Item(current_atom_number).atom_type = phosphorus;
					} else if (temp_name.StartsWith("S")  ) {
						my_atoms.Item(current_atom_number).atom_type = sulfur;
					} else if (temp_name.StartsWith("Cl")  ) {
						my_atoms.Item(current_atom_number).atom_type = chlorine;
					} else if (temp_name.StartsWith("K")  ) {
						my_atoms.Item(current_atom_number).atom_type = potassium;
					} else if (temp_name.StartsWith("Ca") ) {
						my_atoms.Item(current_atom_number).atom_type = calcium;
					} else if (temp_name.StartsWith("Mn")  ) {
						my_atoms.Item(current_atom_number).atom_type = manganese;
					} else if (temp_name.StartsWith("Fe")  ) {
						my_atoms.Item(current_atom_number).atom_type = iron;
					} else if (temp_name.StartsWith("Zn")  ) {
						my_atoms.Item(current_atom_number).atom_type = zinc;
					} else {
						MyPrintWithDetails("Failed to match the element name %s\n",my_atoms.Item(current_atom_number).name);
						DEBUG_ABORT;
					};


				}


				current_atom_number++;

			}
			else if (current_line.StartsWith("REMARK") == true && current_line.Length() > 0)
			{
				// code for copies
				if (current_line.Mid(REMARKSTART,REMARKLENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
				{
					MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(REMARKSTART,REMARKLENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
					DEBUG_ABORT;
				}

				if (temp_double == 351)
				{


					if (current_line.Mid(XPARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
					{
						MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(XPARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
						DEBUG_ABORT;
					}
					wanted_origin_x = temp_double;

					if (current_line.Mid(YPARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
					{
						MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(YPARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
						DEBUG_ABORT;
					}
					wanted_origin_y = temp_double;

					if (current_line.Mid(ZPARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
					{
						MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(ZPARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
						DEBUG_ABORT;
					}
					wanted_origin_z = temp_double;

					if (current_line.Mid(E1PARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
					{
						MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(E1PARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
						DEBUG_ABORT;
					}
					euler1 = temp_double;

					if (current_line.Mid(E2PARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
					{
						MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(E2PARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
						DEBUG_ABORT;
					}
					euler2 = temp_double;

					if (current_line.Mid(E3PARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToDouble(&temp_double) == false)
					{
						MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", current_line.Mid(E3PARTICLE,PARTICLELENGTH).Trim(true).Trim(false).ToUTF8().data(), current_line.ToUTF8().data());
						DEBUG_ABORT;
					}

					euler3 = temp_double;

					// Set up the initial trajectory for this particle instance. This doesn't actually change the coordinates.
					this->TransformBaseCoordinates(wanted_origin_x,wanted_origin_y,wanted_origin_z,euler1,euler2,euler3);
					//this->TransformBaseCoordinates(wanted_origin_x,wanted_origin_y,wanted_origin_z,0,0,0);
				}

			}




		}

		// In case there is only one particle, ie  no 351 in pdb
		if (this->number_of_particles_initialized == 0)
		{
			// Set up the initial trajectory for this particle instance.
			this->TransformBaseCoordinates(0,0,0,0,0,0);
		}

		// rewind the file..

		Rewind();

		// Finally, calculate the center of mass of the PDB object.
		if (! use_provided_com)
		{
			for (current_atom_number = 0; current_atom_number < number_of_atoms; current_atom_number++)
			{

				center_of_mass[0] += my_atoms.Item(current_atom_number).x_coordinate;
				center_of_mass[1] += my_atoms.Item(current_atom_number).y_coordinate;
				center_of_mass[2] += my_atoms.Item(current_atom_number).z_coordinate;


			}


			for (current_atom_number = 0; current_atom_number < 3; current_atom_number++)
			{
				center_of_mass[current_atom_number] /= number_of_atoms;
				if (std::isnan(center_of_mass[current_atom_number]))
				{
					wxPrintf("NaN in center of mass calc from PDB for coordinate %ld, 0=x,1=y,2=z",current_atom_number);
					throw;
				}
			}
		}

		wxPrintf("\n\nPDB center of mass at %f %f %f (x,y,z Angstrom)\n\nSetting origin there.\n\n",center_of_mass[0],center_of_mass[1],center_of_mass[2]);

		// Set the coordinate origin to the center of mass
		for (current_atom_number = 0; current_atom_number < number_of_atoms; current_atom_number++)
		{
			my_atoms.Item(current_atom_number).x_coordinate -= center_of_mass[0];
			my_atoms.Item(current_atom_number).y_coordinate -= center_of_mass[1];
			my_atoms.Item(current_atom_number).z_coordinate -= center_of_mass[2];
		}

	}

	else
	if (access_type == OPEN_TO_WRITE)
	{
		// check if the file exists..

		if (DoesFileExist(text_filename) == true)
		{
			if (wxRemoveFile(text_filename) == false)
			{
				MyDebugPrintWithDetails("Cannot remove already existing text file");
			}
		}

		output_file_stream = new wxFileOutputStream(text_filename);
		output_text_stream = new wxTextOutputStream(*output_file_stream);
	}



}

void PDB::Rewind()
{

	if (access_type == OPEN_TO_READ)
	{
		delete input_file_stream;
		delete input_text_stream;

		input_file_stream = new wxFileInputStream(text_filename);
		input_text_stream = new wxTextInputStream(*input_file_stream);

	}
	else
	output_file_stream->GetFile()->Seek(0);

}

void PDB::Flush()
{
	if (access_type == OPEN_TO_READ) input_file_stream->GetFile()->Flush();
	else
	output_file_stream->GetFile()->Flush();
}

void PDB::ReadLine(float *data_array)
{
	if (access_type != OPEN_TO_READ)
	{
		MyPrintWithDetails("Attempt to read from %s however access type is not READ\n",text_filename);
		DEBUG_ABORT;
	}

	wxString current_line;
	wxString token;
	double temp_double;

	while(input_file_stream->Eof() == false)
	{
		current_line = input_text_stream->ReadLine();
		current_line.Trim(false);

		if (current_line.StartsWith("C") == false && current_line.StartsWith("#") == false && current_line.Length() != 0) break;
	}

	wxStringTokenizer tokenizer(current_line);

	for (int counter = 0; counter < records_per_line; counter++ )
	{
		token = tokenizer.GetNextToken();

		if (token.ToDouble(&temp_double) == false)
		{
			MyPrintWithDetails("Failed on the following record : %s\nFrom Line  : %s\n", token.ToUTF8().data(), current_line.ToUTF8().data());
	    	DEBUG_ABORT;
		}
		else
		{
			data_array[counter] = temp_double;

		}
	}
}

void PDB::WriteLine(float *data_array)
{
	if (access_type != OPEN_TO_WRITE)
	{
		MyPrintWithDetails("Attempt to read from %s however access type is not WRITE\n",text_filename);
		DEBUG_ABORT;
	}

	for (int counter = 0; counter < records_per_line; counter++ )
	{
//		output_text_stream->WriteDouble(data_array[counter]);
		output_text_stream->WriteString(wxString::Format("%14.5f",data_array[counter]));
		if (counter != records_per_line - 1) output_text_stream->WriteString(" ");
	}

	output_text_stream->WriteString("\n");
}

void PDB::WriteLine(double *data_array)
{
	if (access_type != OPEN_TO_WRITE)
	{
		MyPrintWithDetails("Attempt to read from %s however access type is not WRITE\n",text_filename);
		DEBUG_ABORT;
	}

	for (int counter = 0; counter < records_per_line; counter++ )
	{
		output_text_stream->WriteDouble(data_array[counter]);
		if (counter != records_per_line - 1) output_text_stream->WriteString(" ");
	}

	output_text_stream->WriteString("\n");
}

void PDB::WriteCommentLine(const char * format, ...)
{
	va_list args;
	va_start(args, format);

	wxString comment_string;
	wxString buffer;


	comment_string.PrintfV(format, args);

	buffer = comment_string;
	buffer.Trim(false);

	if (buffer.StartsWith("#") == false && buffer.StartsWith("C") == false)
	{
		comment_string = "# " + comment_string;
	}

	output_text_stream->WriteString(comment_string);

	if (comment_string.EndsWith("\n") == false) output_text_stream->WriteString("\n");

	va_end(args);
}

wxString PDB::ReturnFilename()
{
	return text_filename;
}


void PDB::TransformBaseCoordinates(float wanted_origin_x,float wanted_origin_y,float wanted_origin_z, float euler1, float euler2, float euler3 )
// Sets the initial position and orientation of the particle (my_ensemble.my_trajectories.Item(0)...)
{

	// Initialize a new trajectory which represents an individual instance of a particle
	ParticleTrajectory dummy_trajectory;
	my_trajectory.Add(dummy_trajectory,1);


	RotationMatrix rotmat;
	rotmat.SetToRotation(euler1,euler2,euler3);


    // Is it safe to increment like this?
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][0]  = wanted_origin_x;
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][1]  = wanted_origin_y;
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][2]  = wanted_origin_z;
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][3]  = rotmat.m[0][0];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][4]  = rotmat.m[1][0];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][5]  = rotmat.m[2][0];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][6]  = rotmat.m[0][1];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][7]  = rotmat.m[1][1];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][8]  = rotmat.m[2][1];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][9]  = rotmat.m[0][2];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][10] = rotmat.m[1][2];
	my_trajectory.Item(this->number_of_particles_initialized).current_orientation[0][11] = rotmat.m[2][2];


	// Storing the update for reference. On the intial round this matches the orientation.
	my_trajectory.Item(this->number_of_particles_initialized).current_update[0][0]  = wanted_origin_x;
	my_trajectory.Item(this->number_of_particles_initialized).current_update[0][1]  = wanted_origin_y;
	my_trajectory.Item(this->number_of_particles_initialized).current_update[0][2]  = wanted_origin_z;
	my_trajectory.Item(this->number_of_particles_initialized).current_update[0][3]  = euler1;
	my_trajectory.Item(this->number_of_particles_initialized).current_update[0][4]  = euler2;
	my_trajectory.Item(this->number_of_particles_initialized).current_update[0][5]  = euler3;



	// Now that a new member is added to the ensemble, increment the counter
	this->number_of_particles_initialized++;
	wxPrintf("\n\nNumber of particles initialized %d\n",this->number_of_particles_initialized);
}

void PDB::TransformLocalAndCombine(PDB *pdb_ensemble, int number_of_pdbs, long number_of_non_water_atoms, int time_step, RotationMatrix particle_rot, float shift_z)
{
	/*
	 * Take an array of PDB objects and create a single array of atoms transformed according to the timestep
    */

	int current_pdb = 0;
	int current_particle = 0;
	int current_atom = 0;
	long current_total_atom = 0;
    float ox, oy, oz; // origin for the current particle
    float ix, iy, iz; // input coords for current atom
    float tx, ty, tz; // transformed coords for current atom

    // Assuming some a distributino around 0,0,0
	this->min_z   =  0;
	this->max_z   =  0;
	float min_x   =  0; // These are then constant, only the z-dimension will change on tilting, so keep them local.
	float min_y   =  0;
	float max_x   =  0;
	float max_y   =  0;
	this->average_bFactor = 0;
	RotationMatrix rotmat;


	// PDB current_specimen(number_of_non_water_atoms);

	for (int iAtom = 0; iAtom < NUMBER_OF_ATOM_TYPES; iAtom++)
	{
		number_of_each_atom[iAtom]=0;
	}

	for (current_pdb = 0; current_pdb < number_of_pdbs ; current_pdb++)
	{
		//wxPrintf("Checking %ld %ld\n",pdb_ensemble[current_pdb].my_atoms.GetCount(),current_specimen.my_atoms.GetCount());


		for (current_particle = 0; current_particle < pdb_ensemble[current_pdb].number_of_particles_initialized; current_particle++)
		{
			ox = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][0];
			oy = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][1];
			oz = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][2];
			rotmat.SetToValues(pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][3],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][4],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][5],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][6],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][7],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][8],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][9],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][10],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[time_step][11]);

			rotmat *= particle_rot;

			for (current_atom = 0; current_atom < pdb_ensemble[current_pdb].number_of_atoms ; current_atom++)
			{


				this->my_atoms.Item(current_total_atom) = CopyAtom(pdb_ensemble[current_pdb].my_atoms.Item(current_atom));


				ix =  my_atoms.Item(current_total_atom).x_coordinate;
				iy =  my_atoms.Item(current_total_atom).y_coordinate;
				iz =  my_atoms.Item(current_total_atom).z_coordinate;
/*				wxPrintf("xyz at %f %f %f\n",pdb_ensemble[current_pdb].my_atoms.Item(current_atom).x_coordinate,
											 pdb_ensemble[current_pdb].my_atoms.Item(current_atom).y_coordinate,
											 pdb_ensemble[current_pdb].my_atoms.Item(current_atom).z_coordinate);
*/
				rotmat.RotateCoords(ix,iy,iz,tx,ty,tz); // Why can't I just put the shift operation above inline to the function?
				// Update the specimen with the transformed coords


				this->my_atoms.Item(current_total_atom).x_coordinate = tx+ox;
				this->my_atoms.Item(current_total_atom).y_coordinate = ty+oy;
				this->my_atoms.Item(current_total_atom).z_coordinate = tz+oz;

				// minMax X
				if (this->my_atoms.Item(current_total_atom).x_coordinate < min_x)
				{
					min_x = this->my_atoms.Item(current_total_atom).x_coordinate;
				}
				else if (this->my_atoms.Item(current_total_atom).x_coordinate > max_x)
				{
					max_x = this->my_atoms.Item(current_total_atom).x_coordinate;
				}
				// minMax Y
				if (this->my_atoms.Item(current_total_atom).y_coordinate < min_y)
				{
					min_y = this->my_atoms.Item(current_total_atom).y_coordinate;
				}

				else if (this->my_atoms.Item(current_total_atom).y_coordinate > max_y )
				{
					max_y = this->my_atoms.Item(current_total_atom).y_coordinate;
				}
				// minMax Z
				if (this->my_atoms.Item(current_total_atom).z_coordinate < this->min_z)
				{
				  this->min_z = this->my_atoms.Item(current_total_atom).z_coordinate;
				}

				else if (this->my_atoms.Item(current_total_atom).z_coordinate > this->max_z )
				{
				   this->max_z = this->my_atoms.Item(current_total_atom).z_coordinate;
				}

				this->number_of_each_atom[pdb_ensemble[current_pdb].my_atoms.Item(current_atom).atom_type]++;
				current_total_atom++;


			}


		} // end of the loop on particles

	}

	this->average_bFactor /= current_total_atom;

	if (current_total_atom > 2) // for single atom test
	{
		// Again, need a check to make sure all sizes are consistent
		if (max_x - min_x <= 0)
		{
			MyPrintWithDetails("The measured X dimension is invalid max - min = X, %d - %d = %d\n",max_x,min_x, max_x-min_x);
			DEBUG_ABORT;
		}
		if (max_y - min_y <= 0)
		{
			MyPrintWithDetails("The measured Y dimension is invalid max - min = Y, %d - %d = %d\n",max_y,min_y, max_y-min_y);
			DEBUG_ABORT;
		}
		if (this->max_z - this->min_z <= 0)
		{
			MyPrintWithDetails("The measured Z dimension is invalid max - min = Z, %d - %d = %d\n",this->max_z,this->min_z, this->max_z-this->min_z);
			DEBUG_ABORT;
		}

		wxPrintf("max/min xyz, %f,%f  %f,%f  %f,%f\n",max_x,min_x,max_y,min_y,max_z,min_z);
	}



	this->vol_angX = max_x-min_x+ MIN_PADDING_XY;
	this->vol_angY = max_y-min_y+ MIN_PADDING_XY;
	// Shifting all atoms in the ensemble by some offset to keep them centered may be preferrable. This could lead to too many waters. TODO
//	this->vol_angZ = std::max((double)300,(1.50*std::abs(this->max_z-this->min_z))); // take the larger of 20 nm + range and 1.5x the specimen diameter. Look closer at Nobles paper.
	this->vol_angZ = std::max(MIN_THICKNESS,(2*(MIN_PADDING_Z+fabsf(shift_z)) + 1.05*fabsf(this->max_z-this->min_z))); // take the larger of 20 nm + range and 1.5x the specimen diameter. Look closer at Nobles paper.

	if (this->cubic_size > 1)
	{
		// Override the dimensions
		this->vol_nX = cubic_size;
		this->vol_nY = cubic_size;
		this->vol_nZ = cubic_size;
	}
	else
	{
		this->vol_nX = myroundint(this->vol_angX / pixel_size);
		this->vol_nY = myroundint(this->vol_angY / pixel_size);
		this->vol_nZ = myroundint(this->vol_angZ / pixel_size);
		if (IsEven(this->vol_nZ) == false) this->vol_nZ += 1;
	}


	// Adjust the angstrom dimension to match the extended values
	this->vol_angX = this->vol_nX * pixel_size;
	this->vol_angY = this->vol_nY * pixel_size;
	this->vol_angZ = this->vol_nZ * pixel_size;



	this->vol_oX = floor(this->vol_nX / 2);
	this->vol_oY = floor(this->vol_nY / 2);
	this->vol_oZ = floor(this->vol_nZ / 2);



	wxPrintf("vol_angZ = %3.3e\n angstroms", this->vol_angZ);




}

void PDB::TransformGlobalAndSortOnZ(long number_of_non_water_atoms, float shift_x, float shift_y, float shift_z, RotationMatrix rotmat)
{


	long current_atom;
    float ix, iy, iz; // input coords for current atom
    float tx, ty, tz; // transformed coords for current atom



//    // May need more than one to specify a complete transformation?
//    RotationMatrix rotmat;
//    rotmat.SetToRotation(0,tilt_specimen,0);




	for (current_atom = 0; current_atom < number_of_non_water_atoms ; current_atom++)
	{
		ix = shift_x + my_atoms.Item(current_atom).x_coordinate;
		iy = shift_y + my_atoms.Item(current_atom).y_coordinate;
		iz = shift_z + my_atoms.Item(current_atom).z_coordinate;

		rotmat.RotateCoords(ix,iy,iz,tx,ty,tz);

		my_atoms.Item(current_atom).x_coordinate = tx;
		my_atoms.Item(current_atom).y_coordinate = ty;
		my_atoms.Item(current_atom).z_coordinate = tz;

		//wxPrintf("xyz at %f %f %f\n",tx,ty,iz);

	}

	// I am not sure if putting the conditionals in the last loop would prevent vectorization, so this would be good to look at.

	for (current_atom = 0; current_atom < number_of_non_water_atoms; current_atom++)
	{
		if (my_atoms.Item(current_atom).z_coordinate < this->min_z)
		{
		   min_z = my_atoms.Item(current_atom).z_coordinate;
		}

		if (my_atoms.Item(current_atom).z_coordinate > this->max_z)
		{
		   max_z = my_atoms.Item(current_atom).z_coordinate;
		}
	}

	return ;

}








