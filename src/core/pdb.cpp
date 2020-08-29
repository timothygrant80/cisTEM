#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfAtoms);
WX_DEFINE_OBJARRAY(ArrayOfParticleTrajectories);
//WX_DEFINE_OBJARRAY(ArrayOfParticleInstances);



const double MIN_PADDING_Z    = 4;
const int MAX_XY_DIMENSION = 4096*2;


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


PDB::PDB(long number_of_non_water_atoms, float cubic_size, float wanted_pixel_size, int minimum_paddeding_x_and_y, double minimum_thickness_z,
		int max_number_of_noise_particles,
		float wanted_noise_particle_radius_as_mutliple_of_particle_radius,
		float wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
		float wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
		float wanted_tilt_angle_to_emulate)
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

	// The default is to not generate neighboring noise particles. This should probably be switched.
	if ( max_number_of_noise_particles > 0 )
	{
		this->generate_noise_atoms = true;
	}



	this->max_number_of_noise_particles = max_number_of_noise_particles;
	this->noise_particle_radius_as_mutliple_of_particle_radius = wanted_noise_particle_radius_as_mutliple_of_particle_radius;
	this->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius;
	this->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius;
	this->emulate_tilt_angle = wanted_tilt_angle_to_emulate;

}

PDB::PDB(wxString Filename, long wanted_access_type, float wanted_pixel_size, long wanted_records_per_line, int minimum_paddeding_x_and_y, double minimum_thickness_z,
		int max_number_of_noise_particles,
		float wanted_noise_particle_radius_as_mutliple_of_particle_radius,
		float wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
		float wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
		float wanted_tilt_angle_to_emulate)
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
	// The default is to not generate neighboring noise particles. This should probably be switched.
	if ( max_number_of_noise_particles > 0 )
	{
		this->generate_noise_atoms = true;
	}



	this->max_number_of_noise_particles = max_number_of_noise_particles;
	this->noise_particle_radius_as_mutliple_of_particle_radius = wanted_noise_particle_radius_as_mutliple_of_particle_radius;
	this->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius;
	this->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius = wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius;
	this->emulate_tilt_angle = wanted_tilt_angle_to_emulate;

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


		// After a phenix ADP refinement + Chimera selection and split, all ATOM --> HETATM. Quick hack to set this until I can figure out why. Generally speaking, this should be left as ATOM
		wxString pdb_atom = "ATOM";

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

				if (current_line.StartsWith(pdb_atom) == true)
				{
					number_of_atoms++;

				}


			}
		}


		// Only those atoms that are part of the target molecule
		number_of_real_atoms = number_of_atoms;

		// Copy every real atom into a noise atom
		if (generate_noise_atoms)
		{
			number_of_atoms += number_of_atoms * this->max_number_of_noise_particles;
		}

		wxPrintf("Max particles is %d here\n",max_number_of_noise_particles);
		// If noie noise atoms, this will always equal number of atoms. Otherwise, number of atoms is at most this, and changes from particle to particle in the stack
		number_of_real_and_noise_atoms = number_of_atoms;

		wxPrintf("\nIn constructor real total current %ld %ld %ld\n", number_of_real_atoms,number_of_real_and_noise_atoms, number_of_atoms);

		// records_per_line = current_records_per_line;

		// rewind the file..

		Rewind();

		// Create the atom array, then loop back over the pdb to get the desired info.
		Atom dummy_atom;
		my_atoms.Alloc(number_of_real_and_noise_atoms);
		my_atoms.Add(dummy_atom,number_of_real_and_noise_atoms);




		while (input_file_stream->Eof() == false)
		{
			current_line = input_text_stream->ReadLine();
			current_line.Trim(false);

			if (current_line.StartsWith(pdb_atom) == true && current_line.Length() > 0)
			{

				wxString residue_name = current_line.Mid(RESIDUESTART,RESIDUELENGTH).Trim(true).Trim(false);

				my_atoms.Item(current_atom_number).name =  current_line.Mid(NAMESTART,NAMELENGTH).Trim(true).Trim(false);
				my_atoms.Item(current_atom_number).is_real_particle = true;

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
					} else if (temp_name.StartsWith("AU")  ) {
						my_atoms.Item(current_atom_number).atom_type = gold;
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
					this->TransformBaseCoordinates(wanted_origin_x,wanted_origin_y,wanted_origin_z,euler1,euler2,euler3, number_of_particles_initialized, 0);
					//this->TransformBaseCoordinates(wanted_origin_x,wanted_origin_y,wanted_origin_z,0,0,0);
				}

			}




		}

		// In case there is only one particle, ie  no 351 in pdb
		if (this->number_of_particles_initialized == 0)
		{
			// Set up the initial trajectory for this particle instance.
			this->TransformBaseCoordinates(0,0,0,0,0,0,0,0);
		}

		// rewind the file..

		Rewind();

		// Finally, calculate the center of mass of the PDB object.
		if (! use_provided_com)
		{
			for (current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++)
			{

				center_of_mass[0] += my_atoms.Item(current_atom_number).x_coordinate;
				center_of_mass[1] += my_atoms.Item(current_atom_number).y_coordinate;
				center_of_mass[2] += my_atoms.Item(current_atom_number).z_coordinate;


			}


			for (current_atom_number = 0; current_atom_number < 3; current_atom_number++)
			{
				center_of_mass[current_atom_number] /= number_of_real_atoms;
				if (std::isnan(center_of_mass[current_atom_number]))
				{
					wxPrintf("NaN in center of mass calc from PDB for coordinate %ld, 0=x,1=y,2=z",current_atom_number);
					throw;
				}
			}
		}

		wxPrintf("\n\nPDB center of mass at %f %f %f (x,y,z Angstrom)\n\nSetting origin there.\n\n",center_of_mass[0],center_of_mass[1],center_of_mass[2]);

		// Set the coordinate origin to the center of mass
		for (current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++)
		{
			my_atoms.Item(current_atom_number).x_coordinate -= center_of_mass[0];
			my_atoms.Item(current_atom_number).y_coordinate -= center_of_mass[1];
			my_atoms.Item(current_atom_number).z_coordinate -= center_of_mass[2];
		}

		if (generate_noise_atoms)
		{
			// First get the largest molecular radius of the target molecule (centered)
			for (current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++)
			{
				// FIXME number of noise molecules!!
				if (fabsf(my_atoms.Item(current_atom_number).x_coordinate) > max_radius) max_radius = fabsf(my_atoms.Item(current_atom_number).x_coordinate);
				if (fabsf(my_atoms.Item(current_atom_number).y_coordinate) > max_radius) max_radius = fabsf(my_atoms.Item(current_atom_number).y_coordinate);
				if (fabsf(my_atoms.Item(current_atom_number ).z_coordinate) > max_radius) max_radius = fabsf(my_atoms.Item(current_atom_number).z_coordinate);

			}
			wxPrintf("Max particles is %d in spot 2\n",max_number_of_noise_particles);


			for (int iPart = 0; iPart < this->max_number_of_noise_particles; iPart++)
			{

				for (current_atom_number = 0; current_atom_number < number_of_real_atoms; current_atom_number++)
				{

					my_atoms.Item(current_atom_number + number_of_real_atoms*(1+iPart)) = CopyAtom(my_atoms.Item(current_atom_number));
					my_atoms.Item(current_atom_number + number_of_real_atoms*(1+iPart)).is_real_particle = false;


				}
			}

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


void PDB::TransformBaseCoordinates(float wanted_origin_x,float wanted_origin_y,float wanted_origin_z, float euler1, float euler2, float euler3, int particle_idx, int frame_number )
// Sets the initial position and orientation of the particle (my_ensemble.my_trajectories.Item(0)...)
{

	// Initialize a new trajectory which represents an individual instance of a particle
	ParticleTrajectory dummy_trajectory;
	my_trajectory.Add(dummy_trajectory,1);


	RotationMatrix rotmat;
	rotmat.SetToRotation(euler1,euler2,euler3);


    // Is it safe to increment like this?
	my_trajectory.Item(particle_idx).current_orientation[frame_number][0]  = wanted_origin_x;
	my_trajectory.Item(particle_idx).current_orientation[frame_number][1]  = wanted_origin_y;
	my_trajectory.Item(particle_idx).current_orientation[frame_number][2]  = wanted_origin_z;
	my_trajectory.Item(particle_idx).current_orientation[frame_number][3]  = rotmat.m[0][0];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][4]  = rotmat.m[1][0];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][5]  = rotmat.m[2][0];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][6]  = rotmat.m[0][1];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][7]  = rotmat.m[1][1];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][8]  = rotmat.m[2][1];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][9]  = rotmat.m[0][2];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][10] = rotmat.m[1][2];
	my_trajectory.Item(particle_idx).current_orientation[frame_number][11] = rotmat.m[2][2];


	// Storing the update for reference. On the intial round this matches the orientation.
	my_trajectory.Item(particle_idx).current_update[frame_number][0]  = wanted_origin_x;
	my_trajectory.Item(particle_idx).current_update[frame_number][1]  = wanted_origin_y;
	my_trajectory.Item(particle_idx).current_update[frame_number][2]  = wanted_origin_z;
	my_trajectory.Item(particle_idx).current_update[frame_number][3]  = euler1;
	my_trajectory.Item(particle_idx).current_update[frame_number][4]  = euler2;
	my_trajectory.Item(particle_idx).current_update[frame_number][5]  = euler3;




	// Now that a new member is added to the ensemble, increment the counter
	this->number_of_particles_initialized++;
	wxPrintf("\n\nNumber of particles initialized %d\n",this->number_of_particles_initialized);
}

void PDB::TransformLocalAndCombine(PDB *pdb_ensemble, int number_of_pdbs, int frame_number, RotationMatrix particle_rot, float shift_z, bool is_single_particle )
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


	for (int iAtom = 0; iAtom < NUMBER_OF_ATOM_TYPES; iAtom++)
	{
		number_of_each_atom[iAtom]=0;
	}

	for (current_pdb = 0; current_pdb < number_of_pdbs ; current_pdb++)
	{
		//wxPrintf("Checking %ld %ld\n",pdb_ensemble[current_pdb].my_atoms.GetCount(),current_specimen.my_atoms.GetCount());


		for (current_particle = 0; current_particle < pdb_ensemble[current_pdb].number_of_particles_initialized; current_particle++)
		{
//			ox = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][0];
//			oy = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][1];
//			oz = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][2];
//			rotmat.SetToValues(pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][3],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][4],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][5],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][6],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][7],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][8],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][9],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][10],
//							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[frame_number][11]);
			ox = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][0];
			oy = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][1];
			oz = pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][2];
			rotmat.SetToValues(pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][3],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][4],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][5],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][6],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][7],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][8],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][9],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][10],
							   pdb_ensemble[current_pdb].my_trajectory.Item(current_particle).current_orientation[0][11]);

			rotmat *= particle_rot;


			// Randomly set noise atoms before copying in. Initially, there is a copy of every real atom, at the same x,y,z coords. For frame zero we
			// want to randomly rotate each noise particle, and offset each particle by a random amount that ensures no overlap with the real particle.
			// The first two noise particles are set to not overlap. Additional particles are able to overlap, which I think is more likely with high crowding (multiple layers of particles.)
			// It might be that local motion should also be applied, but I think it is fair for now that the neighbors move identically frame to frame as the target particles. They should
			// at the minimum be highly correlated based on empirical observation.


			for (current_atom = 0; current_atom < pdb_ensemble[current_pdb].number_of_atoms ; current_atom++)
			{
				this->my_atoms.Item(current_atom) = CopyAtom(pdb_ensemble[current_pdb].my_atoms.Item(current_atom));
			}


			if (pdb_ensemble[current_pdb].generate_noise_atoms && frame_number == 0)
			{

				RandomNumberGenerator my_rand(PIf);
				// Set the number of noise particles for this given particle in the stack
				pdb_ensemble[current_pdb].number_of_noise_particles = myroundint(my_rand.GetUniformRandomSTD(std::max(0,this->max_number_of_noise_particles-2),this->max_number_of_noise_particles));
				wxPrintf("\n\n\tSetting pdb %d to %d noise particles of max %d\n\n", current_pdb, pdb_ensemble[current_pdb].number_of_noise_particles,max_number_of_noise_particles);

				// Angular sector size such that noise particles do not overlap. This could be a method.
				float sector_size = 1.1f; //2.0*PIf / pdb_ensemble[current_pdb].number_of_noise_particles;
				float occupied_sectors[this->max_number_of_noise_particles];
				int current_sector = 0;
				bool non_overlaping_particle_found;

				for (int iPart = 0; iPart < this->max_number_of_noise_particles; iPart++)
				{

					non_overlaping_particle_found = false;
					float offset_radius;
					// Rather than mess with the allocations, simply send ignored particles off into space.
					if (iPart >= pdb_ensemble[current_pdb].number_of_noise_particles)
					{
						offset_radius = 1e8;
					}
					else
					{
//						offset_radius = my_rand.GetNormalRandomSTD(0.0f, 0.1*pdb_ensemble[current_pdb].max_radius);
						// These numbers (-0.5, 1.8, 0.1 are not at all thought out - please FIXME)
						offset_radius = my_rand.GetUniformRandomSTD(this->noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius*pdb_ensemble[current_pdb].max_radius,
																	this->noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius*pdb_ensemble[current_pdb].max_radius);
						offset_radius += this->noise_particle_radius_as_mutliple_of_particle_radius*pdb_ensemble[current_pdb].max_radius;

					}



					float offset_angle;
					// If we have more than one noise particle, enforce non-overlap
					if (iPart == 0)
					{
						offset_angle = clamp_angular_range(my_rand.GetUniformRandomSTD(-PIf,PIf));
						non_overlaping_particle_found = true;
						occupied_sectors[0] = offset_angle;
					}
					else
					{
						int max_tries = 5000;
						int iTry = 0;
						bool is_too_close = true;
						while (is_too_close && iTry < max_tries)
						{
							iTry += 1;
							offset_angle = clamp_angular_range(my_rand.GetUniformRandomSTD(-PIf,PIf));
							float dx = cosf(offset_angle);
							float dy = sinf(offset_angle);
							float ang_diff;
							int jPart;
//							wxPrintf("offset angle is %3.3f\n", rad_2_deg(offset_angle));

							// now check the angle against each previous
							for (jPart = 0; jPart < iPart; jPart ++)
							{
								ang_diff = cosf(occupied_sectors[jPart])*dx + sinf(occupied_sectors[jPart])*dy;
								ang_diff = acosf(ang_diff);


								if (fabsf(ang_diff) >= sector_size)
								{
									// We need to keep track of whether or not a goo dmatch was found
									is_too_close = false;
								}
								else
								{
									// if any are too close, want to break out
									is_too_close = true;
									break;
								}
							}
//							//
//							// For trouble shooting overlaps: We found a good spot, run through theoptions again and print out.
//							if (! is_too_close)
//							{
//								for (jPart = 0; jPart < iPart; jPart ++)
//								{
//									ang_diff = cosf(occupied_sectors[jPart])*dx + sinf(occupied_sectors[jPart])*dy;
//									ang_diff = acosf(ang_diff);
//									wxPrintf("on iPart %d comparing %3.3f against %3.3f (jPart %d) found a diff of %3.3f\n", iPart,rad_2_deg(offset_angle),rad_2_deg(occupied_sectors[jPart]), jPart,rad_2_deg(ang_diff));
//
//								}
//
//							}
						}
						if (is_too_close)
						{
							wxPrintf("Error, did not find a well separated noise particle\n");
//							exit(-1);
						}
						else
						{
							non_overlaping_particle_found = true;
							occupied_sectors[iPart] = offset_angle;
						}
					}



					if ( ! non_overlaping_particle_found)
					{
						// fix double negative
						offset_radius = 1e8;
					}


						float offset_X = offset_radius * cosf(offset_angle) * cosf(deg_2_rad(emulate_tilt_angle));
						float offset_Y = offset_radius * sinf(offset_angle);


	//					RotationMatrix randmat;
						// FIXME this will not sample the euler sphere propertly
	//					randmat.SetToEulerRotation(my_rand.GetUniformRandomSTD(0,360),my_rand.GetUniformRandomSTD(0,180),my_rand.GetUniformRandomSTD(0,360));

						pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].Init(my_rand.GetUniformRandomSTD(0,360),my_rand.GetUniformRandomSTD(0,360),my_rand.GetUniformRandomSTD(0,360),offset_X, offset_Y);


	//					wxPrintf("\n\t\tFor pdb %d particle %d np %d, eulers are %3.3e %3.3e %3.3e, %3.3e %3.3e\n", current_pdb, current_particle, iPart,
	//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnPhiAngle(),
	//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnThetaAngle(),
	//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnPsiAngle(),
	//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftX(),
	//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftY());

						wxPrintf("\nreal total current %ld %ld %ld\n", pdb_ensemble[current_pdb].number_of_real_atoms,pdb_ensemble[current_pdb].number_of_real_and_noise_atoms, pdb_ensemble[current_pdb].number_of_atoms);

						for (int current_atom_number = 0; current_atom_number < pdb_ensemble[current_pdb].number_of_real_atoms; current_atom_number++)
						{

							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].euler_matrix.RotateCoords(
										my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).x_coordinate,
										my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).y_coordinate,
										my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).z_coordinate,
										tx,ty,tz);

							my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).x_coordinate = tx + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftX();
							my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).y_coordinate = ty + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftY();
							my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).z_coordinate = tz;
						}

//					} // if condition on non-overlaping particles
				} // end of loop over noise particles
			} // if condition on noise particles and frame 0

			else if (pdb_ensemble[current_pdb].generate_noise_atoms)
			{
				for (int iPart = 0; iPart < this->max_number_of_noise_particles; iPart++)
				{
//					wxPrintf("\n\t\tFor pdb %d particle %d np %d, eulers are %3.3e %3.3e %3.3e, %3.3e %3.3e\n", current_pdb, current_particle, iPart,
//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnPhiAngle(),
//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnThetaAngle(),
//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnPsiAngle(),
//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftX(),
//							pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftY());
					for (int current_atom_number = 0; current_atom_number < pdb_ensemble[current_pdb].number_of_real_atoms; current_atom_number++)
					{

						pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].euler_matrix.RotateCoords(
									my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).x_coordinate,
									my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).y_coordinate,
									my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).z_coordinate,
									tx,ty,tz);

						my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).x_coordinate = tx + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftX();
						my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).y_coordinate = ty + pdb_ensemble[current_pdb].my_angles_and_shifts[iPart].ReturnShiftY();
						my_atoms.Item(current_atom_number + pdb_ensemble[current_pdb].number_of_real_atoms*(1+iPart)).z_coordinate = tz;
					}
				}
			}


			for (current_atom = 0; current_atom < pdb_ensemble[current_pdb].number_of_atoms ; current_atom++)
			{

				if (my_atoms.Item(current_atom).is_real_particle)
				{

					ix =  my_atoms.Item(current_atom).x_coordinate;
					iy =  my_atoms.Item(current_atom).y_coordinate;
					iz =  my_atoms.Item(current_atom).z_coordinate;
	/*				wxPrintf("xyz at %f %f %f\n",pdb_ensemble[current_pdb].my_atoms.Item(current_atom).x_coordinate,
												 pdb_ensemble[current_pdb].my_atoms.Item(current_atom).y_coordinate,
												 pdb_ensemble[current_pdb].my_atoms.Item(current_atom).z_coordinate);
	*/
					rotmat.RotateCoords(ix,iy,iz,tx,ty,tz); // Why can't I just put the shift operation above inline to the function?
					// Update the specimen with the transformed coords


					this->my_atoms.Item(current_atom).x_coordinate = tx+ox;
					this->my_atoms.Item(current_atom).y_coordinate = ty+oy;
					this->my_atoms.Item(current_atom).z_coordinate = tz+oz;

					// minMax X
					if (this->my_atoms.Item(current_atom).x_coordinate < min_x)
					{
						min_x = this->my_atoms.Item(current_atom).x_coordinate;
					}
					if (this->my_atoms.Item(current_atom).x_coordinate > max_x)
					{
						max_x = this->my_atoms.Item(current_atom).x_coordinate;
					}
					// minMax Y
					if (this->my_atoms.Item(current_atom).y_coordinate < min_y)
					{
						min_y = this->my_atoms.Item(current_atom).y_coordinate;
					}

					if (this->my_atoms.Item(current_atom).y_coordinate > max_y )
					{
						max_y = this->my_atoms.Item(current_atom).y_coordinate;
					}
					// minMax Z
					if (this->my_atoms.Item(current_atom).z_coordinate < this->min_z)
					{
					  this->min_z = this->my_atoms.Item(current_atom).z_coordinate;
					}

					if (this->my_atoms.Item(current_atom).z_coordinate > this->max_z )
					{
					   this->max_z = this->my_atoms.Item(current_atom).z_coordinate;
					}

					this->number_of_each_atom[pdb_ensemble[current_pdb].my_atoms.Item(current_atom).atom_type]++;

					average_bFactor += my_atoms.Item(current_atom).bfactor;

					current_total_atom++;
				} // if on real particles

			}


		} // end of the loop on particles

	}

	// This is used in the simulator to determine how large a window should be used for the calculation of the atoms.
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

	float max_depth = 0.0f;
	if (is_single_particle)
	{
		// Keep the thickness of the ice mostly constant, which may not happen if the particle is non globular and randomly oriented.
		max_depth = std::max(max_x-min_x, std::max(max_y - min_y, fabsf(max_z-min_z)));

	}
	else
	{
		max_depth = max_z-min_z;
	}
	// Shifting all atoms in the ensemble by some offset to keep them centered may be preferrable. This could lead to too many waters. TODO
//	this->vol_angZ = std::max((double)300,(1.50*std::abs(this->max_z-this->min_z))); // take the larger of 20 nm + range and 1.5x the specimen diameter. Look closer at Nobles paper.
	wxPrintf("min is %f, shift is %d, max_depth is %d\n", MIN_THICKNESS, (int)fabsf(shift_z), (int)fabsf(max_depth));
	this->vol_angZ = std::max(MIN_THICKNESS,(2*(MIN_PADDING_Z+fabsf(shift_z)) + (MIN_PADDING_Z+fabsf(max_depth)))); // take the larger of 20 nm + range and 1.5x the specimen diameter. Look closer at Nobles paper.


	this->vol_angZ /= cosf(deg_2_rad(emulate_tilt_angle));

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


//	// Adjust the angstrom dimension to match the extended values
//	this->vol_angX = (float)this->vol_nX * pixel_size;
//	this->vol_angY = (float)this->vol_nY * pixel_size;
//	this->vol_angZ = (float)this->vol_nZ * pixel_size;
//
//
//
//	this->vol_oX = floor(this->vol_nX / 2);
//	this->vol_oY = floor(this->vol_nY / 2);
//	this->vol_oZ = floor(this->vol_nZ / 2);
//
//
//
//	wxPrintf("vol_angZ = %3.3e\n angstroms", this->vol_angZ);




}

void PDB::TransformGlobalAndSortOnZ(long number_of_non_water_atoms, float shift_x, float shift_y, float shift_z, RotationMatrix rotmat)
{


	long current_atom;
    float ix, iy, iz; // input coords for current atom
    float tx, ty, tz; // transformed coords for current atom



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








