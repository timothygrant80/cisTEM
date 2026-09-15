#ifndef _SRC_CORE_STRING_FUNCTIONS_H_
#define _SRC_CORE_STRING_FUNCTIONS_H_

/*
 * String, printf-style formatting and sleeping helpers that replace the pieces of wxWidgets
 * (wxString, wxPrintf, wxString::Format, wxStringTokenizer, wxSleep...) cisTEM used to lean on.
 *
 * Printf()/Format() are printf-compatible, but std::string (and std::filesystem::path) arguments
 * may be passed straight to a "%s" conversion, as they could with wxPrintf. Every other argument
 * is forwarded to vsnprintf untouched, so the usual printf type rules apply.
 */

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace cistem {
namespace detail {

// Map an argument to the type vsnprintf needs to see for it.
inline const char* PrintfArgument(const std::string& value) { return value.c_str( ); }

inline const char* PrintfArgument(const std::filesystem::path& value) { return value.c_str( ); }

inline const char* PrintfArgument(const char* value) { return value; }

inline const char* PrintfArgument(char* value) { return value; }

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> || std::is_pointer_v<std::decay_t<T>> || std::is_enum_v<std::decay_t<T>>>>
inline T PrintfArgument(T value) { return value; }

inline std::string VFormat(const char* format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char stack_buffer[512];
    int  needed = vsnprintf(stack_buffer, sizeof(stack_buffer), format, args_copy);
    va_end(args_copy);
    if ( needed < 0 )
        return std::string( );
    if ( size_t(needed) < sizeof(stack_buffer) )
        return std::string(stack_buffer, size_t(needed));
    std::string result(size_t(needed), '\0');
    vsnprintf(&result[0], size_t(needed) + 1, format, args);
    return result;
}

inline std::string FormatCString(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::string result = VFormat(format, args);
    va_end(args);
    return result;
}

inline void PrintfCString(FILE* stream, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stream, format, args);
    va_end(args);
}

} // namespace detail
} // namespace cistem

// ---------------------------------------------------------------------------------------------
// Formatting / printing
// ---------------------------------------------------------------------------------------------

/// printf to stdout; std::string arguments are accepted by "%s". Replacement for wxPrintf.
template <typename... Args>
inline void Printf(const char* format, const Args&... args) {
    cistem::detail::PrintfCString(stdout, format, cistem::detail::PrintfArgument(args)...);
}

template <typename... Args>
inline void Printf(const std::string& format, const Args&... args) {
    Printf(format.c_str( ), args...);
}

/// printf to an arbitrary stream (stderr, an open file); std::string arguments are accepted by "%s".
template <typename... Args>
inline void FPrintf(FILE* stream, const char* format, const Args&... args) {
    cistem::detail::PrintfCString(stream, format, cistem::detail::PrintfArgument(args)...);
}

/// Build a std::string printf-style; std::string arguments are accepted by "%s". Replacement for wxString::Format.
template <typename... Args>
inline std::string Format(const char* format, const Args&... args) {
    return cistem::detail::FormatCString(format, cistem::detail::PrintfArgument(args)...);
}

template <typename... Args>
inline std::string Format(const std::string& format, const Args&... args) {
    return Format(format.c_str( ), args...);
}

// ---------------------------------------------------------------------------------------------
// Whitespace, case and prefix/suffix handling
// ---------------------------------------------------------------------------------------------

inline bool IsSpaceCharacter(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

/// Remove trailing whitespace in place (wxString::Trim()).
inline std::string& TrimRight(std::string& s) {
    size_t end = s.size( );
    while ( end > 0 && IsSpaceCharacter(s[end - 1]) )
        end--;
    s.erase(end);
    return s;
}

/// Remove leading whitespace in place (wxString::Trim(false)).
inline std::string& TrimLeft(std::string& s) {
    size_t start = 0;
    while ( start < s.size( ) && IsSpaceCharacter(s[start]) )
        start++;
    s.erase(0, start);
    return s;
}

/// Remove leading and trailing whitespace in place.
inline std::string& Trim(std::string& s) {
    TrimRight(s);
    return TrimLeft(s);
}

/// Return a copy with leading and trailing whitespace removed.
inline std::string Trimmed(std::string s) {
    return Trim(s);
}

inline bool StartsWith(const std::string& s, const std::string& prefix) {
    return s.size( ) >= prefix.size( ) && s.compare(0, prefix.size( ), prefix) == 0;
}

inline bool StartsWith(const std::string& s, char prefix) {
    return ! s.empty( ) && s[0] == prefix;
}

inline bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size( ) >= suffix.size( ) && s.compare(s.size( ) - suffix.size( ), suffix.size( ), suffix) == 0;
}

inline bool EndsWith(const std::string& s, char suffix) {
    return ! s.empty( ) && s.back( ) == suffix;
}

inline std::string ToLower(std::string s) {
    for ( char& c : s )
        c = char(tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string ToUpper(std::string s) {
    for ( char& c : s )
        c = char(toupper(static_cast<unsigned char>(c)));
    return s;
}

/// strcasecmp-style comparison (wxString::CmpNoCase).
inline int CompareNoCase(const std::string& a, const std::string& b) {
    return strcasecmp(a.c_str( ), b.c_str( ));
}

/// Case-insensitive equality (wxString::IsSameAs(other, false)).
inline bool EqualsNoCase(const std::string& a, const std::string& b) {
    return a.size( ) == b.size( ) && CompareNoCase(a, b) == 0;
}

inline bool Contains(const std::string& s, const std::string& what) {
    return s.find(what) != std::string::npos;
}

/// Replace every occurrence of `from` with `to`, returning the number of replacements (wxString::Replace).
inline size_t ReplaceAll(std::string& s, const std::string& from, const std::string& to) {
    if ( from.empty( ) )
        return 0;
    size_t count = 0;
    size_t pos   = 0;
    while ( (pos = s.find(from, pos)) != std::string::npos ) {
        s.replace(pos, from.size( ), to);
        pos += to.size( );
        count++;
    }
    return count;
}

/// Shorten the string to at most `length` characters (wxString::Truncate).
inline std::string& Truncate(std::string& s, size_t length) {
    if ( s.size( ) > length )
        s.erase(length);
    return s;
}

// ---------------------------------------------------------------------------------------------
// Splitting on a character (same semantics as the wxString members of the same name)
// ---------------------------------------------------------------------------------------------

/// Everything before the first `separator`; the whole string if it does not occur.
inline std::string BeforeFirst(const std::string& s, char separator) {
    size_t pos = s.find(separator);
    return pos == std::string::npos ? s : s.substr(0, pos);
}

/// Everything after the first `separator`; empty if it does not occur.
inline std::string AfterFirst(const std::string& s, char separator) {
    size_t pos = s.find(separator);
    return pos == std::string::npos ? std::string( ) : s.substr(pos + 1);
}

/// Everything before the last `separator`; empty if it does not occur.
inline std::string BeforeLast(const std::string& s, char separator) {
    size_t pos = s.rfind(separator);
    return pos == std::string::npos ? std::string( ) : s.substr(0, pos);
}

/// Everything after the last `separator`; the whole string if it does not occur.
inline std::string AfterLast(const std::string& s, char separator) {
    size_t pos = s.rfind(separator);
    return pos == std::string::npos ? s : s.substr(pos + 1);
}

/**
 * Split on any character in `delimiters`. With `skip_empty` (the default, matching
 * wxStringTokenizer's default whitespace mode) runs of delimiters produce no empty tokens.
 */
inline std::vector<std::string> SplitString(const std::string& s, const std::string& delimiters = " \t\r\n", bool skip_empty = true) {
    std::vector<std::string> tokens;
    size_t                   start = 0;
    while ( start <= s.size( ) ) {
        size_t end = s.find_first_of(delimiters, start);
        if ( end == std::string::npos )
            end = s.size( );
        if ( end > start || ! skip_empty )
            tokens.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    if ( ! skip_empty && ! s.empty( ) && delimiters.find(s.back( )) != std::string::npos )
        tokens.pop_back( ); // a trailing delimiter does not create a final empty token (wxTOKEN_RET_EMPTY)
    return tokens;
}

inline std::string JoinStrings(const std::vector<std::string>& parts, const std::string& separator) {
    std::string result;
    for ( size_t i = 0; i < parts.size( ); i++ ) {
        if ( i > 0 )
            result += separator;
        result += parts[i];
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// Number parsing. Like wxString::ToLong/ToDouble these succeed only if the whole string
// (ignoring surrounding whitespace) is a number.
// ---------------------------------------------------------------------------------------------

inline bool StringToLong(const std::string& s, long& value, int base = 10) {
    if ( s.empty( ) )
        return false;
    errno       = 0;
    char* end   = nullptr;
    long  local = strtol(s.c_str( ), &end, base);
    if ( end == s.c_str( ) || errno == ERANGE )
        return false;
    while ( *end != '\0' && IsSpaceCharacter(*end) )
        end++;
    if ( *end != '\0' )
        return false;
    value = local;
    return true;
}

inline bool StringToLongLong(const std::string& s, long long& value, int base = 10) {
    if ( s.empty( ) )
        return false;
    errno           = 0;
    char*     end   = nullptr;
    long long local = strtoll(s.c_str( ), &end, base);
    if ( end == s.c_str( ) || errno == ERANGE )
        return false;
    while ( *end != '\0' && IsSpaceCharacter(*end) )
        end++;
    if ( *end != '\0' )
        return false;
    value = local;
    return true;
}

inline bool StringToULong(const std::string& s, unsigned long& value, int base = 10) {
    if ( s.empty( ) )
        return false;
    errno               = 0;
    char*         end   = nullptr;
    unsigned long local = strtoul(s.c_str( ), &end, base);
    if ( end == s.c_str( ) || errno == ERANGE )
        return false;
    while ( *end != '\0' && IsSpaceCharacter(*end) )
        end++;
    if ( *end != '\0' )
        return false;
    value = local;
    return true;
}

inline bool StringToInt(const std::string& s, int& value) {
    long local;
    if ( ! StringToLong(s, local) || local < INT_MIN || local > INT_MAX )
        return false;
    value = int(local);
    return true;
}

inline bool StringToDouble(const std::string& s, double& value) {
    if ( s.empty( ) )
        return false;
    errno        = 0;
    char*  end   = nullptr;
    double local = strtod(s.c_str( ), &end);
    if ( end == s.c_str( ) || errno == ERANGE )
        return false;
    while ( *end != '\0' && IsSpaceCharacter(*end) )
        end++;
    if ( *end != '\0' )
        return false;
    value = local;
    return true;
}

inline bool StringToFloat(const std::string& s, float& value) {
    double local;
    if ( ! StringToDouble(s, local) )
        return false;
    value = float(local);
    return true;
}

/// True if the whole string parses as an integer.
inline bool StringIsInteger(const std::string& s) {
    long junk;
    return StringToLong(s, junk);
}

/// True if the whole string parses as a number.
inline bool StringIsNumber(const std::string& s) {
    double junk;
    return StringToDouble(s, junk);
}

// ---------------------------------------------------------------------------------------------
// Sleeping (wxSleep / wxMilliSleep)
// ---------------------------------------------------------------------------------------------

inline void SleepForSeconds(double seconds) {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

inline void SleepForMilliseconds(long milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

#endif // _SRC_CORE_STRING_FUNCTIONS_H_
