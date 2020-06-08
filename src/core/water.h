typedef
struct __attribute__((packed)) __attribute__((aligned(16))) _AtomPos
{
	float x;
	float y;
	float z;

} AtomPos;

class Water {

	// Track xyz coords for waters and provide some basic additions to the Image class.

	private:


	public:


		// Constructors
		Water(bool do_carbon = false);
		Water(const PDB *current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, RotationMatrix max_rotation,float in_plane_rotation, int *padX, int *padY, int nThreads,  bool is_single_particle, bool do_carbon = false);
		~Water();

		// data

		long number_of_waters = 0;
		int size_neighborhood;
		float pixel_size;
		float dose_per_frame;
		int records_per_line;
		int number_of_time_steps;
		bool simulate_phase_plate;
		bool keep_time_steps;
		double center_of_mass[3];
		long number_of_each_atom[NUMBER_OF_ATOM_TYPES];
		float atomic_volume;
		float vol_angX, vol_angY, vol_angZ;
		int vol_nX, vol_nY, vol_nZ;
		float vol_oX, vol_oY, vol_oZ;
		float offset_z;
		float min_z;
		float max_z;
		AtomPos* water_coords;
		bool is_allocated_water_coords = false;
		int nThreads;



		//	void set_initial_trajectories(PDB *pdb_ensemble);


		void Init(const PDB *current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, RotationMatrix max_rotation, float in_plane_rotation,int *padX, int *padY, int nThreads, bool is_single_particle);
		void SeedWaters3d();
		void ShakeWaters3d(int number_of_threads);
		void ReturnPadding(RotationMatrix max_rotation, float in_plane_rotation, int current_thickness, int current_nX, int current_nY, int* padX, int* padY, int *padZ);


		inline float Return_x_Coordinate(long current_atom) { return water_coords[current_atom].x ; }


		inline void ReturnCenteredCoordinates(long current_atom, float &dx, float &dy,float &dz)
		{

			// Find the water's coordinates in the rotated slab
			// Shift those coordinates to a centered origin
			dx = water_coords[current_atom].x - vol_oX;
			dy = water_coords[current_atom].y - vol_oY;
			dz = water_coords[current_atom].z - vol_oZ;


		}



};

