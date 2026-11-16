#include "core_headers.h"

#include <fstream>
#include <sstream>

bool AberrationSet::IsEmpty( ) const {
    for ( float coefficient : odd_zernike )
        if ( coefficient != 0.0f )
            return false;
    for ( float coefficient : even_zernike )
        if ( coefficient != 0.0f )
            return false;
    return true;
}

bool AberrationTable::ReadFromFile(const std::string& filename) {
    sets.clear( );
    if ( filename.empty( ) || filename == "none" )
        return true;

    std::ifstream file(filename);
    if ( ! file.is_open( ) )
        return false;

    std::string line;
    while ( std::getline(file, line) ) {
        const size_t first = line.find_first_not_of(" \t\r");
        if ( first == std::string::npos || line[first] == '#' )
            continue;

        std::istringstream fields(line);
        AberrationSet      set;
        int                count;
        if ( ! (fields >> set.group >> count) || count < 0 )
            return false;
        set.odd_zernike.resize(count);
        for ( int i = 0; i < count; i++ )
            if ( ! (fields >> set.odd_zernike[i]) )
                return false;
        if ( ! (fields >> count) || count < 0 )
            return false;
        set.even_zernike.resize(count);
        for ( int i = 0; i < count; i++ )
            if ( ! (fields >> set.even_zernike[i]) )
                return false;
        sets.push_back(set);
    }
    return true;
}

bool AberrationTable::WriteToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if ( ! file.is_open( ) )
        return false;
    file << "# group  n_odd  odd Zernike coefficients (rad, RELION order)  n_even  even Zernike coefficients\n";
    for ( const AberrationSet& set : sets ) {
        file << set.group << " " << set.odd_zernike.size( );
        for ( float coefficient : set.odd_zernike )
            file << " " << Format("%.8g", coefficient);
        file << " " << set.even_zernike.size( );
        for ( float coefficient : set.even_zernike )
            file << " " << Format("%.8g", coefficient);
        file << "\n";
    }
    return bool(file);
}

const AberrationSet* AberrationTable::ForGroup(int group) const {
    const AberrationSet* default_set = nullptr;
    for ( const AberrationSet& set : sets ) {
        if ( set.group == group )
            return &set;
        if ( set.group == 0 )
            default_set = &set;
    }
    return default_set;
}

void AberrationTable::ApplyTo(CTF& ctf, int group, bool negate) const {
    const AberrationSet* set = ForGroup(group);
    if ( set == nullptr ) {
        ctf.ClearZernikeCoefficients( );
        return;
    }
    if ( negate ) {
        std::vector<float> odd(set->odd_zernike);
        for ( float& coefficient : odd )
            coefficient = -coefficient;
        ctf.SetZernikeCoefficients(odd, set->even_zernike);
    }
    else
        ctf.SetZernikeCoefficients(set->odd_zernike, set->even_zernike);
}
