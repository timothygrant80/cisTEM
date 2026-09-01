#ifndef cisTEM_PROGRAMS_BLUSH_REFINEMENT_BLUSH_LOGGER_H_
#define cisTEM_PROGRAMS_BLUSH_REFINEMENT_BLUSH_LOGGER_H_

#include <string>
#include <cstdio>
#include <ctime>
#include <cstdarg>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

// #define BLUSH_DEBUG_LOGGING

/**
 * @brief Production logging system for Blush inference
 *
 * Provides structured logging with two modes:
 * - GUI mode: Logs to <ProjectDir>/Scratch/BlushLogs/ (passed as parameter)
 * - CLI mode: Logs to ~/.cisTEM/logs/blush/ (fallback when no path provided)
 *
 * Features:
 * - Per-process log files (no conflicts)
 * - Timestamp + PID tracking
 * - Log levels (LOG_DEBUG, INFO, WARN, ERROR)
 * - Printf-style formatting
 * - Zero wxWidgets dependency (works in ipc_blush subprocess)
 * - Thread-safe (each process has own file handle)
 *
 * Usage:
 *   BlushLogger::InitializeLogger("/path/to/project/Scratch/BlushLogs");
 *   BLUSH_LOG_INFO("Processing volume: box_size=%d", box_size);
 *   BLUSH_LOG_ERROR("Model loading failed: %s", error_msg);
 */

namespace BlushLogger {

enum class Level {
    LOG_DEBUG = 0,
    INFO      = 1,
    WARN      = 2,
    ERROR     = 3
};

class Logger {
  private:
    std::string log_file_path;
    FILE*       log_file;
    Level       min_level;
    bool        initialized;

    const char* LevelToString(Level level) const {
        switch ( level ) {
            case Level::LOG_DEBUG: return "LOG_DEBUG";
            case Level::INFO: return "INFO ";
            case Level::WARN: return "WARN ";
            case Level::ERROR: return "ERROR";
            default: return "?????";
        }
    }

    void CreateDirectoryRecursive(const std::string& path) const {
        if ( path.empty( ) )
            return;

        // Create all parent directories recursively
        size_t pos = 0;
        while ( (pos = path.find('/', pos + 1)) != std::string::npos ) {
            std::string partial = path.substr(0, pos);
            mkdir(partial.c_str( ), 0755); // Ignore errors - directory may already exist
        }
        // Create the final directory
        mkdir(path.c_str( ), 0755);
    }

    std::string GetUserLogDirectory( ) const {
        const char* home = getenv("HOME");
        if ( ! home )
            home = "/tmp"; // Fallback if HOME not set

        std::string log_dir = std::string(home) + "/.cisTEM/logs/blush";

        // Create directory structure recursively
        CreateDirectoryRecursive(log_dir);

        return log_dir;
    }

    std::string GenerateTimestamp( ) const {
        time_t     now     = time(nullptr);
        struct tm* tm_info = localtime(&now);
        char       timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);
        return std::string(timestamp);
    }

    std::string GenerateLogFilename(const std::string& log_dir) const {
        std::string timestamp = GenerateTimestamp( );
        char        filename[512];
        snprintf(filename, sizeof(filename), "%s/blush_%s_%d.log",
                 log_dir.c_str( ), timestamp.c_str( ), getpid( ));
        return std::string(filename);
    }

    void UpdateSymlink(const std::string& log_dir) const {
        // Extract just the filename from the full path for relative symlink
        size_t      last_slash = log_file_path.rfind('/');
        std::string filename   = (last_slash != std::string::npos) ? log_file_path.substr(last_slash + 1) : log_file_path;

        std::string symlink_path = log_dir + "/blush_latest.log";
        unlink(symlink_path.c_str( )); // Remove old symlink (ignore errors)

        // Create RELATIVE symlink (works across container/host boundary)
        symlink(filename.c_str( ), symlink_path.c_str( )); // Points to filename in same directory
    }

  public:
    Logger( )
        : log_file(nullptr), min_level(Level::INFO), initialized(false) {
    }

    ~Logger( ) {
        if ( log_file ) {
            Log(Level::INFO, "=== Blush session ended ===");
            fclose(log_file);
            log_file = nullptr;
        }
    }

    /**
     * @brief Initialize logger with optional project log directory
     *
     * @param project_log_dir Project-specific log directory (GUI mode), or empty for CLI mode
     * @param level Minimum logging level
     */
    void Initialize(const std::string& project_log_dir = "", Level level = Level::INFO) {
        if ( initialized )
            return;

        min_level = level;

        // Determine log directory: project-specific or user home
        std::string log_dir;
        if ( ! project_log_dir.empty( ) ) {
            // GUI mode: Use project scratch directory
            log_dir = project_log_dir;
            // Create directory recursively (handles missing parent directories)
            CreateDirectoryRecursive(log_dir);
        }
        else {
            // CLI mode: Use ~/.cisTEM/logs/blush/
            log_dir = GetUserLogDirectory( );
        }

        log_file_path = GenerateLogFilename(log_dir);
        log_file      = fopen(log_file_path.c_str( ), "a");

        if ( log_file ) {
            UpdateSymlink(log_dir);
            initialized = true;

            // Write session header
            Log(Level::INFO, "=== Blush session started ===");
            char info[512];
            snprintf(info, sizeof(info), "PID: %d, Log: %s", getpid( ), log_file_path.c_str( ));
            Log(Level::INFO, info);
        }
    }

    void Log(Level level, const char* message) {
        if ( ! initialized || ! log_file || level < min_level )
            return;

        time_t     now     = time(nullptr);
        struct tm* tm_info = localtime(&now);
        char       timestamp[32];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

        fprintf(log_file, "[%s] [%s] %s\n",
                timestamp, LevelToString(level), message);
        fflush(log_file);
    }

    void Debug(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        if ( min_level > Level::LOG_DEBUG )
            return;

        char    buffer[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        Log(Level::LOG_DEBUG, buffer);
    }

    void Info(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        if ( min_level > Level::INFO )
            return;

        char    buffer[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        Log(Level::INFO, buffer);
    }

    void Warn(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        if ( min_level > Level::WARN )
            return;

        char    buffer[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        Log(Level::WARN, buffer);
    }

    void Error(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char    buffer[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        Log(Level::ERROR, buffer);
    }

    const std::string& GetLogPath( ) const { return log_file_path; }

    bool IsInitialized( ) const { return initialized; }

    /**
     * @brief Clean up old log files to prevent disk clutter
     *
     * @param log_dir Directory to clean (project or user logs)
     * @param days_to_keep Keep logs newer than this many days
     */
    static void CleanupOldLogs(const std::string& log_dir, int days_to_keep = 7) {
        if ( log_dir.empty( ) )
            return;

        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                 "find '%s' -name 'blush_*.log' -type f -mtime +%d -delete 2>/dev/null",
                 log_dir.c_str( ), days_to_keep);

        int result = system(cmd);
        (void)result; // Suppress unused variable warning - cleanup failure is non-critical
    }
};

// Global logger singleton
inline Logger& GetGlobalLogger( ) {
    static Logger instance;
    return instance;
}

/**
 * @brief Initialize the global logger (call once at program startup)
 *
 * @param project_log_dir Optional project log directory (GUI mode), empty for CLI mode
 * @param level Minimum logging level
 */
inline void InitializeLogger(const std::string& project_log_dir = "", Level level = Level::INFO) {
    GetGlobalLogger( ).Initialize(project_log_dir, level);
}

// Convenience macros for logging
#define BLUSH_LOG_DEBUG(...) BlushLogger::GetGlobalLogger( ).Debug(__VA_ARGS__)
#define BLUSH_LOG_INFO(...) BlushLogger::GetGlobalLogger( ).Info(__VA_ARGS__)
#define BLUSH_LOG_WARN(...) BlushLogger::GetGlobalLogger( ).Warn(__VA_ARGS__)
#define BLUSH_LOG_ERROR(...) BlushLogger::GetGlobalLogger( ).Error(__VA_ARGS__)

} // namespace BlushLogger

#endif // cisTEM_PROGRAMS_BLUSH_REFINEMENT_BLUSH_LOGGER_H_
