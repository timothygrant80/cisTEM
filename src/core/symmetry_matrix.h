/*  \brief  SymmetryMatrix class */

// Matrices generated using Frealign code, implemented in tool generate_symmetry_matrices.exe

class SymmetryMatrix {

public:

	wxString			symmetry_symbol;
	int					number_of_matrices;
	RotationMatrix		*rot_mat;                /* 3D rotation matrix array*/

	SymmetryMatrix();
	SymmetryMatrix(wxString wanted_symmetry_symbol);
	~SymmetryMatrix();

	SymmetryMatrix & operator = (const SymmetryMatrix &other_matrix);
	SymmetryMatrix & operator = (const SymmetryMatrix *other_matrix);

	void Init(wxString wanted_symmetry_symbol);

	void PrintMatrices();
};
