/*  \brief  Reconstruct3D class (derived from Frealign Pinsert) */

class Particle;

class Reconstruct3D {

public:

	float						pixel_size;
	SymmetryMatrix				symmetry_matrices;
	bool						edge_terms_were_added;

	Image						image_reconstruction;
	float						*ctf_reconstruction;
//	float						*weights_reconstruction;
//	int							*number_of_measurements;

	Image						current_ctf_image;
	CTF							current_ctf;

	int 						logical_x_dimension;		// !< Logical (X) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 						logical_y_dimension;		// !< Logical (Y) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 						logical_z_dimension;		// !< Logical (Z) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).

	Reconstruct3D(float wanted_pixel_size);
	Reconstruct3D(float wanted_pixel_size, wxString wanted_symmetry);
	Reconstruct3D(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, wxString wanted_symmetry);	// constructor with size
	~Reconstruct3D();							// destructor

	void Init(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size);
	void InsertSliceWithCTF(Particle &particle_to_insert, float &average_score, float &score_bfactor_conversion);
	void InsertSliceNoCTF(Particle &particle_to_insert, float &average_score, float &score_bfactor_conversion);
//	void InsertSlice(Image &image_to_insert, AnglesAndShifts &angles_and_shifts_of_image, float &particle_weight, float &particle_score, float &average_score, float &score_bfactor_conversion);
//	void InsertSlice(Image &image_to_insert, CTF &ctf_of_image, AnglesAndShifts &angles_and_shifts_of_image, float &particle_weight, float &particle_score, float &average_score, float &score_bfactor_conversion);
	void AddByLinearInterpolation(float &wanted_x_coordinate, float &wanted_y_coordinate, float &wanted_z_coordinate, fftwf_complex &wanted_value, fftwf_complex &ctf_value, float wanted_weight);
	void CompleteEdges();
	Reconstruct3D operator + (const Reconstruct3D &other);
	Reconstruct3D &operator = (const Reconstruct3D &other);
	Reconstruct3D &operator = (const Reconstruct3D *other);
	Reconstruct3D &operator += (const Reconstruct3D &other);
	Reconstruct3D &operator += (const Reconstruct3D *other);
};
