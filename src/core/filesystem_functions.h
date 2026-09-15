#ifndef _SRC_CORE_FILESYSTEM_FUNCTIONS_H_
#define _SRC_CORE_FILESYSTEM_FUNCTIONS_H_

/*
 * File and directory helpers built on std::filesystem, replacing the wxFileName / wxDir
 * static functions cisTEM used. Paths are plain std::string; use std::filesystem::path
 * directly when you need to decompose a path more than these helpers allow.
 */

#include <algorithm>
#include <cstdlib>
#include <fnmatch.h>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

/// True for an existing regular file (follows symlinks). wxFileName::FileExists.
inline bool FileExists(const std::string& filename) {
    std::error_code error;
    return std::filesystem::is_regular_file(filename, error);
}

/// True for an existing directory. wxDir::Exists / wxFileName::DirExists.
inline bool DirectoryExists(const std::string& directory) {
    std::error_code error;
    return std::filesystem::is_directory(directory, error);
}

/// True if anything exists at the path. wxFileName::Exists.
inline bool PathExists(const std::string& path) {
    std::error_code error;
    return std::filesystem::exists(path, error);
}

/**
 * Create a directory and any missing parents (wxFileName::Mkdir with wxPATH_MKDIR_FULL).
 * Returns true if the directory exists afterwards.
 */
inline bool MakeDirectory(const std::string& directory) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    return DirectoryExists(directory);
}

/// Delete a file; true on success (wxRemoveFile).
inline bool RemoveFile(const std::string& filename) {
    std::error_code error;
    return std::filesystem::remove(filename, error) && ! error;
}

/// Delete a directory and everything below it; returns the number of entries removed.
inline std::uintmax_t RemoveDirectoryRecursively(const std::string& directory) {
    std::error_code error;
    return std::filesystem::remove_all(directory, error);
}

/// Copy a file, overwriting the destination (wxCopyFile).
inline bool CopyFile(const std::string& source, const std::string& destination) {
    std::error_code error;
    return std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing, error);
}

/// Rename / move a file (wxRenameFile).
inline bool RenameFile(const std::string& source, const std::string& destination) {
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    return ! error;
}

/// The file name including its extension ("dir/a.mrc" -> "a.mrc"). wxFileName::GetFullName.
inline std::string ReturnFileName(const std::string& path) {
    return std::filesystem::path(path).filename( ).string( );
}

/// The file name without its last extension ("dir/a.b.mrc" -> "a.b"). wxFileName::GetName.
inline std::string ReturnFileStem(const std::string& path) {
    return std::filesystem::path(path).stem( ).string( );
}

/// The last extension without the dot ("a.mrc" -> "mrc", "a" -> ""). wxFileName::GetExt.
inline std::string ReturnFileExtension(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension( ).string( );
    if ( ! extension.empty( ) && extension[0] == '.' )
        extension.erase(0, 1);
    return extension;
}

/// The directory part without a trailing separator ("dir/a.mrc" -> "dir", "a.mrc" -> ""). wxFileName::GetPath.
inline std::string ReturnDirectory(const std::string& path) {
    return std::filesystem::path(path).parent_path( ).string( );
}

/// The directory part with a trailing separator ("dir/a.mrc" -> "dir/", "a.mrc" -> ""). wxFileName::GetPathWithSep.
inline std::string ReturnDirectoryWithSeparator(const std::string& path) {
    std::string directory = ReturnDirectory(path);
    if ( ! directory.empty( ) && directory.back( ) != '/' )
        directory += '/';
    return directory;
}

/// Remove the last extension, keeping the directory ("dir/a.b.mrc" -> "dir/a.b"). wxFileName::StripExtension.
inline std::string StripExtension(const std::string& path) {
    std::filesystem::path p(path);
    if ( ! p.has_extension( ) )
        return path;
    return p.replace_extension( ).string( );
}

/// Replace (or add) the last extension; `new_extension` is given without the dot. wxFileName::SetExt.
inline std::string ReplaceExtension(const std::string& path, const std::string& new_extension) {
    std::filesystem::path p(path);
    if ( new_extension.empty( ) )
        return p.replace_extension( ).string( );
    return p.replace_extension("." + new_extension).string( );
}

inline bool PathIsAbsolute(const std::string& path) {
    return std::filesystem::path(path).is_absolute( );
}

inline std::string ReturnHomeDirectory( ) {
    const char* home = getenv("HOME");
    return home != nullptr ? std::string(home) : std::string( );
}

inline std::string ReturnTempDirectory( ) {
    std::error_code error;
    std::string     directory = std::filesystem::temp_directory_path(error).string( );
    return error ? std::string("/tmp") : directory;
}

inline std::string ReturnCurrentWorkingDirectory( ) {
    std::error_code error;
    return std::filesystem::current_path(error).string( );
}

/// Expand a leading "~" or "~/" to the home directory.
inline std::string ExpandTilde(const std::string& path) {
    if ( path == "~" )
        return ReturnHomeDirectory( );
    if ( path.size( ) > 1 && path[0] == '~' && path[1] == '/' )
        return ReturnHomeDirectory( ) + path.substr(1);
    return path;
}

/**
 * Absolute, tilde-expanded, "."/".." collapsed form of a path without a trailing separator
 * (wxFileName::Normalize with the default flags, then GetFullPath). Does not resolve symlinks
 * and does not require the path to exist.
 */
inline std::string NormalizePath(const std::string& path) {
    if ( path.empty( ) )
        return path;
    std::filesystem::path p = std::filesystem::absolute(ExpandTilde(path)).lexically_normal( );
    std::string           result = p.string( );
    while ( result.size( ) > 1 && result.back( ) == '/' )
        result.pop_back( );
    return result;
}

/// Absolute form of a path relative to the current directory, without collapsing dots (wxFileName::MakeAbsolute).
inline std::string MakeAbsolutePath(const std::string& path) {
    return std::filesystem::absolute(path).string( );
}

/// Size in bytes, or -1 if the file cannot be read.
inline long long ReturnFileSize(const std::string& filename) {
    std::error_code error;
    auto            size = std::filesystem::file_size(filename, error);
    return error ? -1 : (long long)size;
}

/**
 * Every regular file in `directory` whose name matches the shell wildcard `pattern`
 * ("*.mrc", "*" for everything), returned as full paths and sorted. Recurses only when asked.
 * wxDir::GetAllFiles(directory, &files, pattern, wxDIR_FILES).
 */
inline std::vector<std::string> ReturnAllFilesInDirectory(const std::string& directory, const std::string& pattern = "*", bool recursive = false) {
    std::vector<std::string> files;
    std::error_code          error;

    auto consider = [&](const std::filesystem::directory_entry& entry) {
        std::error_code entry_error;
        if ( ! entry.is_regular_file(entry_error) )
            return;
        std::string name = entry.path( ).filename( ).string( );
        if ( fnmatch(pattern.c_str( ), name.c_str( ), 0) == 0 )
            files.push_back(entry.path( ).string( ));
    };

    if ( recursive ) {
        for ( const auto& entry : std::filesystem::recursive_directory_iterator(directory, std::filesystem::directory_options::skip_permission_denied, error) )
            consider(entry);
    }
    else {
        for ( const auto& entry : std::filesystem::directory_iterator(directory, std::filesystem::directory_options::skip_permission_denied, error) )
            consider(entry);
    }
    std::sort(files.begin( ), files.end( ));
    return files;
}

/// Names (not paths) of the sub-directories of `directory`, sorted.
inline std::vector<std::string> ReturnAllSubdirectories(const std::string& directory) {
    std::vector<std::string> directories;
    std::error_code          error;
    for ( const auto& entry : std::filesystem::directory_iterator(directory, std::filesystem::directory_options::skip_permission_denied, error) ) {
        std::error_code entry_error;
        if ( entry.is_directory(entry_error) )
            directories.push_back(entry.path( ).filename( ).string( ));
    }
    std::sort(directories.begin( ), directories.end( ));
    return directories;
}

#endif // _SRC_CORE_FILESYSTEM_FUNCTIONS_H_
