


class Water : Image {

	// Track xyz coords for waters and provide some basic additions to the Image class.

	private:


	public:


		// Constructors
		Water();
		Water(const PDB *current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, float max_tilt);
		~Water();

		// data

		long number_of_waters;
		int size_neighborhood;
		float pixel_size;
		float dose_per_frame;
		int records_per_line;
		int number_of_time_steps;
		bool keep_time_steps;
		double center_of_mass[3];
		long number_of_each_atom[NUMBER_OF_ATOM_TYPES];
		float atomic_volume;
		float vol_angX, vol_angY, vol_angZ;
		long vol_nX, vol_nY, vol_nZ;
		long vol_oX, vol_oY, vol_oZ;
		float offset_z;
		float min_z;
		float max_z;

		//	void set_initial_trajectories(PDB *pdb_ensemble);


		void Init(const PDB *current_specimen, int wanted_size_neighborhood, float wanted_pixel_size, float wanted_dose_per_frame, float max_tilt);
		void SeedWaters3d();
		void ShakeWaters3d(int number_of_threads);
		int ReturnPaddingForTilt(float max_tilt, int current_nX);

		inline float Return_x_Coordinate(long current_atom) { return real_values[current_atom*3] ; }

		inline void ReturnCenteredCoordinates(long current_atom, float &dx, float &dy,float &dz)
		{

			// Find the water's coordinates in the rotated slab
			// Shift those coordinates to a centered origin
			dx = real_values[current_atom*3 + 0] - vol_oX;
			dy = real_values[current_atom*3 + 1] - vol_oY;
			dz = real_values[current_atom*3 + 2] - vol_oZ;


		}



};

