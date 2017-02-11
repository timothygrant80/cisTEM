/*  \brief  Reconstruct3D class (derived from Frealign Pinsert) */

class Particle;

class Reconstruct3D {

public:

	float						pixel_size;
	float						original_pixel_size;
	float						average_occupancy;
	float						average_sigma;
	float						sigma_bfactor_conversion;
	SymmetryMatrix				symmetry_matrices;
	bool						edge_terms_were_added;

	Image						image_reconstruction;
	float						*ctf_reconstruction;

//	Image						*current_ctf_image;
//	CTF							current_ctf;

	int 						logical_x_dimension;		// !< Logical (X) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 						logical_y_dimension;		// !< Logical (Y) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 						logical_z_dimension;		// !< Logical (Z) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 						original_x_dimension;
	int 						original_y_dimension;
	int 						original_z_dimension;

	Reconstruct3D(float wanted_pixel_size = 0.0, float wanted_average_occupancy = 0.0, float wanted_average_sigma = 0.0, float wanted_sigma_bfactor_conversion = 0.0);
	Reconstruct3D(float wanted_pixel_size, float wanted_average_occupancy, float wanted_average_sigma, float wanted_sigma_bfactor_conversion, wxString wanted_symmetry);
	Reconstruct3D(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, float wanted_average_occupancy, float wanted_average_sigma, float wanted_sigma_bfactor_conversion, wxString wanted_symmetry);	// constructor with size
	~Reconstruct3D();							// destructor

	void Init(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, float wanted_average_occupancy, float wanted_average_sigma, float wanted_sigma_bfactor_conversion);
	void InsertSliceWithCTF(Particle &particle_to_insert);
	void InsertSliceNoCTF(Particle &particle_to_insert);
	void AddByLinearInterpolation(float &wanted_x_coordinate, float &wanted_y_coordinate, float &wanted_z_coordinate, std::complex<float> &wanted_value, std::complex<float> &ctf_value, float wanted_weight);
	void CompleteEdges();
	float Correct3DCTF(Image &buffer3d);
	void DumpArrays(wxString filename, bool insert_even);
	void ReadArrayHeader(wxString filename, int &logical_x_dimension, int &logical_y_dimension, int &logical_z_dimension,
			int &original_x_dimension, int &original_y_dimension, int &original_z_dimension, float &pixel_size, float &original_pixel_size,
			float &average_occupancy, float &average_sigma, float &sigma_bfactor_conversion, wxString &symmetry_symbol, bool &insert_even);
	void ReadArrays(wxString filename);
	Reconstruct3D operator + (const Reconstruct3D &other);
	Reconstruct3D &operator = (const Reconstruct3D &other);
	Reconstruct3D &operator = (const Reconstruct3D *other);
	Reconstruct3D &operator += (const Reconstruct3D &other);
	Reconstruct3D &operator += (const Reconstruct3D *other);
};
