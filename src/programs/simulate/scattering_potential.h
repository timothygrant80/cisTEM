/*
 * scattering_potential.h
 *
 *  Created on: Oct 3, 2019
 *      Author: himesb
 */

#ifndef PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_
#define PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_

#define MAX_NUMBER_PDBS 64 // This probably doesn't need to be limited - check use TODO.


class ScatteringPotential
{

public:

	ScatteringPotential();
	virtual ~ScatteringPotential();

	PDB  		*pdb_ensemble;
	wxString	pdb_file_names[MAX_NUMBER_PDBS];
    int 	 	number_of_pdbs = 1;
    bool 		is_allocated_pdb_ensemble = false;



	__inline__ float ReturnScatteringParamtersA( AtomType id, int term_number) { return SCATTERING_PARAMETERS_A[id][term_number]; }
	__inline__ float ReturnScatteringParamtersB( AtomType id, int term_number) { return  SCATTERING_PARAMETERS_B[id][term_number]; }
	__inline__ float ReturnAtomicNumber( AtomType id) { return ATOMIC_NUMBER[id]; }

	void InitPdbEnsemble(	float wanted_pixel_size, float do3d, int minimum_padding_x_and_y, int minimum_thickness_z,
							int max_number_of_noise_particles,
							float wanted_noise_particle_radius_as_mutliple_of_particle_radius,
							float wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
							float wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
							float wanted_tilt_angle_to_emulate);
	long ReturnTotalNumberOfNonWaterAtoms();


private:


	const float WN = 0.8045*0.79; // sum netOxy A / sum water (A) = 0.8045 and ratio of total elastic cross section water/oxygen 0.67-0.92 Using average 0.79 (there is no fixed estimate)

	// The name is to an index matching here in the PDB class. If you change this, you MUST change that. This is probably a bad idea.
	// H(0),C(1),N(2),O(3),F(4),Na(5),Mg(6),P(7),S(8),Cl(9),K(10),Ca(11),Mn(12),Fe(13),Zn(14),H20(15),0-(16)
	const float  ATOMIC_NUMBER[NUMBER_OF_ATOM_TYPES] = {1.0f, // 0
														6.0f,
														7.0f,
														8.0f,
														9.0f, // 4
														11.0f,
														12.0f,
														15.0f,
														16.0f,
														17.0f, // 9
														19.0f,
														20.0f,
														25.0f,
														26.0f,
														30.0f, // 14
														10.0f,
														8.0f,
														14.0f,
														27.0f,
														34.0f, // 19
														79.0f,
														7.35f,
														};


	// TODO complete for those higher than 16
	const float   SCATTERING_PARAMETERS_A[17][5] = {
		{ 0.0349,  0.1201, 0.1970, 0.0573, 0.1195}, //0
		{ 0.0893,  0.2563, 0.7570, 1.0487, 0.3575},
		{ 0.1022,  0.3219, 0.7982, 0.8197, 0.1715},
		{ 0.0974,  0.2921, 0.6910, 0.6990, 0.2039},
		{ 0.1083,  0.3175, 0.6487, 0.5846, 0.1421}, // 4
		{ 0.2142,  0.6853, 0.7692, 1.6589, 1.4482},
		{ 0.2314,  0.6866, 0.9677, 2.1882, 1.1339},
		{ 0.2548,  0.6106, 1.4541, 2.3204, 0.8477},
		{ 0.2497,  0.5628, 1.3899, 2.1865, 0.7715},
		{ 0.2443,  0.5397, 1.3919, 2.0197, 0.6621}, // 9
		{ 0.4115, -1.4031, 2.2784, 2.6742, 2.2162},
		{ 0.4054,  1.3880, 2.1602, 3.7532, 2.2063},
		{ 0.3796,  1.2094, 1.7815, 2.5420, 1.5937},
		{ 0.3946,  1.2725, 1.7031, 2.3140, 1.4795},
		{ 0.4288,  1.2646, 1.4472, 1.8294, 1.0934}, // 14
		{WN*0.07967, WN*0.1053, WN* 0.2933, WN*0.6831, WN*1.304},
		{ 0.2050,  0.6280, 1.1700, 1.0300, 0.290 }, // Peng 1998
	};

	// 12.5664 ~ (4*pi)
	// -39.47841760685077 ~ -4*pi^2
	const float SCATTERING_PARAMETERS_B[17][5] = {
			{0.5347, 3.5867, 12.347, 18.9525, 38.6269},
			{0.2465, 1.7100, 6.4094, 18.6113, 50.2523},
			{0.2451, 1.7481, 6.1925, 17.3894, 48.1431},
			{0.2067, 1.3815, 4.6943, 12.7105, 32.4726},
			{0.2057, 1.3439, 4.2788, 11.3932, 28.7881},
			{0.3334, 2.3446, 10.083, 48.3037, 138.270},
			{0.3278, 2.2720, 10.924, 39.2898, 101.9748},
			{0.2908, 1.8740, 8.5176, 24.3434, 63.2996},
			{0.2681, 1.6711, 7.0267, 19.5377, 50.3888},
			{0.2468, 1.5242, 6.1537, 16.6687, 42.3086},
			{0.3703, 3.3874, 13.1029, 68.9592, 194.4329},
			{0.3499, 3.0991, 11.9608, 53.9353, 142.3892},
			{0.2699, 2.0455, 7.4726, 31.0604, 91.5622},
			{0.2717, 2.0443, 7.6007, 29.9714, 86.2265},
			{0.2593, 1.7998, 6.7500, 25.5860, 73.5284},
			{WN*4.718 , WN* 16.75,WN* 0.4524,  WN* 13.43, WN* 4.4480},
			{ 0.397, 2.6400, 8.8000, 27.1, 91.8}, //Peng 98

	};
};

#endif /* PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_ */
