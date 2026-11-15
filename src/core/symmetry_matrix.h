/*  \brief  SymmetryMatrix class */

// Matrices generated using Frealign code, implemented in tool generate_symmetry_matrices.exe

// Point groups (Cn, Dn, T, O, I) are lists of rotations. A helix, "H<rise>,<twist>[,<copies>][,C<n>]" (rise in A along
// z, twist in degrees about z, right-handed for positive values of both; the operators are n = 0, +-1, +-2, ...
// times the screw), adds a translation along z to each rotation: z_shift[k] in A, zero for every point group.
// A helix has no finite group, so the number of operators is a choice: the third number, or, when it is
// absent, as many as span 30% of the reconstruction box (RELION symmetrises the same central fraction), which
// Reconstruct3D::Init decides because it knows the box; until then the symbol builds the identity alone. The
// count is rounded up to an odd number so the operators stay centred on the identity, and the resolved symbol
// ("H6,50,7") is written back so a dump file hands merge3d the same operators. A multi-start helix adds a Cn
// about the axis ("H15,12,C3": three strands): every screw step is then inserted n times, rotated by multiples of
// 360/n, so the operators number copies x n, the identity still first.
class SymmetryMatrix {

  public:
    std::string     symmetry_symbol;
    int             number_of_matrices;
    RotationMatrix* rot_mat; /* 3D rotation matrix array*/
    float*          z_shift; /* A along z per operator; NULL unless helical */
    bool            is_helical;
    float           helical_rise; /* A */
    float           helical_twist; /* degrees */
    int             helical_copies; /* operators built; 0 until the box decides the default */
    int             helical_cyclic_order; /* the Cn about the axis a multi-start helix adds (1: none); the operators are screw steps x this */

    SymmetryMatrix( );
    SymmetryMatrix(std::string wanted_symmetry_symbol);
    ~SymmetryMatrix( );

    SymmetryMatrix& operator=(const SymmetryMatrix& other_matrix);
    SymmetryMatrix& operator=(const SymmetryMatrix* other_matrix);

    void Init(std::string wanted_symmetry_symbol);

    static bool IsHelicalSymbol(const std::string& symbol);
    static bool ParseHelicalSymbol(const std::string& symbol, float& rise, float& twist, int& copies, int& cyclic_order); // copies = 0 when absent, cyclic_order = 1
    int         ReturnDefaultHelicalCopies(float box_size_in_angstroms);
    void        SetHelicalCopies(int wanted_copies);

    void PrintMatrices( );
};
