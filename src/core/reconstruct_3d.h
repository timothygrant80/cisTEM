/*  \brief  Reconstruct3d class (derived from Frealign Pinsert) */

class Reconstruct3d {

public:

	Image		image_reconstruction;
	float		*ctf_reconstruction;

	Image		current_ctf_image;
	CTF			current_ctf;

	int 		logical_x_dimension;							// !< Logical (X) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 		logical_y_dimension;							// !< Logical (Y) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).
	int 		logical_z_dimension;							// !< Logical (Z) dimensions of the image., Note that this does not necessarily correspond to memory allocation dimensions (ie physical dimensions).

	Reconstruct3d();
	Reconstruct3d(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension);	// constructor with size
	~Reconstruct3d();							// destructor

	void Initialize(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension);
	void InsertSlice(Image &image_to_insert, AnglesAndShifts &angles_and_shifts_of_image);
	void InsertSlice(Image &image_to_insert, CTF &ctf_of_image, AnglesAndShifts &angles_and_shifts_of_image);
	void AddByLinearInterpolation(float &wanted_x_coordinate, float &wanted_y_coordinate, float &wanted_z_coordinate, fftwf_complex &wanted_value, fftwf_complex &ctf_value);
	void FinalizeSimple(Image &final3d);
	void FinalizeOptimal(Image &final3d);
};
