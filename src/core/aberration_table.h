#ifndef _SRC_CORE_ABERRATION_TABLE_H_
#define _SRC_CORE_ABERRATION_TABLE_H_

/*
 * Higher-order aberrations per optics group, as Zernike coefficient sets in RELION's convention (zernike.h,
 * CTF::SetZernikeCoefficients). A program receives them as a small text file, one group per line:
 *
 *     # comments start with a hash
 *     <group> <number of odd coefficients> <odd...> <number of even coefficients> <even...>
 *
 * Group 0 is the default set: it applies to every particle whose own group has no line. The file name "none"
 * (or an empty one) means no aberrations, which is what every caller passed before this existed.
 */

class AberrationSet {
  public:
    int                group = 0;
    std::vector<float> odd_zernike;
    std::vector<float> even_zernike;

    bool IsEmpty( ) const;
};

class AberrationTable {
  public:
    std::vector<AberrationSet> sets;

    // "none" or "" gives an empty table and returns true; a missing or malformed file returns false
    bool ReadFromFile(const std::string& filename);
    bool WriteToFile(const std::string& filename) const;

    inline bool IsEmpty( ) const { return sets.empty( ); };

    // The set for the group, else the group-0 set, else nullptr
    const AberrationSet* ForGroup(int group) const;

    // Sets the CTF's coefficients to the group's (cleared when there are none). negate flips the odd coefficients, for
    // simulating rather than correcting the aberrations: the phase factor that simulates is the conjugate of the one
    // that corrects (project3d passes a negated beam tilt in the same way), while the even terms are part of the CTF
    // itself, which is the same for both.
    void ApplyTo(CTF& ctf, int group, bool negate = false) const;
};

#endif
