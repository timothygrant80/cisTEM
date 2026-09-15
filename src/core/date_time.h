#ifndef _SRC_CORE_DATE_TIME_H_
#define _SRC_CORE_DATE_TIME_H_

/*
 * Small date/time value types replacing wxDateTime, wxTimeSpan and wxStopWatch.
 *
 * DateTime wraps a system_clock time point with millisecond resolution. The DOS-timestamp
 * conversions (GetAsDOS / SetFromDOS) are bit-for-bit those of wxDateTime because cisTEM
 * project databases store dates in that format.
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

class TimeSpan {
  public:
    TimeSpan( ) : milliseconds_(0) {}

    TimeSpan(long hours, long minutes = 0, long long seconds = 0, long long milliseconds = 0)
        : milliseconds_(((static_cast<long long>(hours) * 60 + minutes) * 60 + seconds) * 1000 + milliseconds) {}

    static TimeSpan Milliseconds(long long milliseconds) {
        TimeSpan span;
        span.milliseconds_ = milliseconds;
        return span;
    }

    static TimeSpan Seconds(long long seconds) { return Milliseconds(seconds * 1000); }

    long long GetValue( ) const { return milliseconds_; }

    long long GetMilliseconds( ) const { return milliseconds_; }

    long long GetSeconds( ) const { return milliseconds_ / 1000; }

    long GetMinutes( ) const { return long(GetSeconds( ) / 60); }

    long GetHours( ) const { return GetMinutes( ) / 60; }

    long GetDays( ) const { return GetHours( ) / 24; }

    long GetWeeks( ) const { return GetDays( ) / 7; }

    bool IsNegative( ) const { return milliseconds_ < 0; }

    bool IsNull( ) const { return milliseconds_ == 0; }

    TimeSpan operator+(const TimeSpan& other) const { return Milliseconds(milliseconds_ + other.milliseconds_); }

    TimeSpan operator-(const TimeSpan& other) const { return Milliseconds(milliseconds_ - other.milliseconds_); }

    bool operator==(const TimeSpan& other) const { return milliseconds_ == other.milliseconds_; }

    bool operator!=(const TimeSpan& other) const { return milliseconds_ != other.milliseconds_; }

    bool operator<(const TimeSpan& other) const { return milliseconds_ < other.milliseconds_; }

    bool operator>(const TimeSpan& other) const { return milliseconds_ > other.milliseconds_; }

    /**
     * wxTimeSpan::Format semantics: %E weeks, %D days, %H hours, %M minutes, %S seconds, %l milliseconds,
     * %% a percent sign. The largest unit mentioned is printed in full; smaller units are reduced
     * modulo the next larger unit that appears before them. H, M and S are zero padded to two digits,
     * l to three. A negative span gets a leading '-'.
     */
    std::string Format(const char* format = "%H:%M:%S") const {
        enum Part { PART_NONE = 0,
                    PART_MSEC,
                    PART_SEC,
                    PART_MIN,
                    PART_HOUR,
                    PART_DAY,
                    PART_WEEK };

        long long   magnitude = milliseconds_ < 0 ? -milliseconds_ : milliseconds_;
        TimeSpan    absolute  = Milliseconds(magnitude);
        std::string result;
        if ( milliseconds_ < 0 )
            result += '-';

        Part biggest = PART_NONE;
        for ( const char* p = format; *p != '\0'; p++ ) {
            if ( *p != '%' ) {
                result += *p;
                continue;
            }
            p++;
            if ( *p == '\0' )
                break;

            long long n;
            int       width = 2;
            switch ( *p ) {
                case 'E':
                    n       = absolute.GetWeeks( );
                    biggest = PART_WEEK;
                    width   = 0;
                    break;
                case 'D':
                    n = absolute.GetDays( );
                    if ( biggest < PART_DAY )
                        biggest = PART_DAY;
                    else
                        n %= 7;
                    width = 0;
                    break;
                case 'H':
                    n = absolute.GetHours( );
                    if ( biggest < PART_HOUR )
                        biggest = PART_HOUR;
                    else
                        n %= 24;
                    break;
                case 'M':
                    n = absolute.GetMinutes( );
                    if ( biggest < PART_MIN )
                        biggest = PART_MIN;
                    else
                        n %= 60;
                    break;
                case 'S':
                    n = absolute.GetSeconds( );
                    if ( biggest < PART_SEC )
                        biggest = PART_SEC;
                    else
                        n %= 60;
                    break;
                case 'l':
                    n = absolute.GetMilliseconds( );
                    if ( biggest < PART_MSEC )
                        biggest = PART_MSEC;
                    else
                        n %= 1000;
                    width = 3;
                    break;
                case '%':
                    result += '%';
                    continue;
                default:
                    result += '%';
                    result += *p;
                    continue;
            }
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "%0*lld", width, n);
            result += buffer;
        }
        return result;
    }

  private:
    long long milliseconds_;
};

class DateTime {
  public:
    using Clock = std::chrono::system_clock;

    DateTime( ) : time_point_( ), valid_(false) {}

    explicit DateTime(time_t ticks) : time_point_(Clock::from_time_t(ticks)), valid_(true) {}

    explicit DateTime(Clock::time_point time_point) : time_point_(time_point), valid_(true) {}

    /// Current local time, whole seconds (wxDateTime::Now).
    static DateTime Now( ) {
        return DateTime(std::chrono::time_point_cast<std::chrono::seconds>(Clock::now( )));
    }

    /// Current local time, millisecond resolution (wxDateTime::UNow).
    static DateTime UNow( ) {
        return DateTime(std::chrono::time_point_cast<std::chrono::milliseconds>(Clock::now( )));
    }

    bool IsValid( ) const { return valid_; }

    void Invalidate( ) { valid_ = false; }

    Clock::time_point GetTimePoint( ) const { return time_point_; }

    time_t GetTicks( ) const { return Clock::to_time_t(time_point_); }

    long GetMillisecond( ) const {
        auto since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(time_point_.time_since_epoch( )).count( );
        return long(since_epoch % 1000);
    }

    DateTime& Set(time_t ticks) {
        time_point_ = Clock::from_time_t(ticks);
        valid_      = true;
        return *this;
    }

    /// strftime in local time; "%c" is the wxDateTime::Format default.
    std::string Format(const char* format = "%c") const {
        if ( ! valid_ )
            return std::string( );
        time_t    ticks = GetTicks( );
        struct tm local_time;
        localtime_r(&ticks, &local_time);
        char buffer[256];
        size_t written = strftime(buffer, sizeof(buffer), format, &local_time);
        return std::string(buffer, written);
    }

    std::string FormatISODate( ) const { return Format("%Y-%m-%d"); }

    std::string FormatISOTime( ) const { return Format("%H:%M:%S"); }

    std::string FormatISOCombined(char separator = 'T') const {
        return FormatISODate( ) + separator + FormatISOTime( );
    }

    /// DOS timestamp, identical to wxDateTime::GetAsDOS (what the project database stores).
    unsigned long GetAsDOS( ) const {
        time_t    ticks = GetTicks( );
        struct tm local_time;
        localtime_r(&ticks, &local_time);

        long year   = local_time.tm_year - 80;
        long month  = local_time.tm_mon + 1;
        long day    = local_time.tm_mday;
        long hour   = local_time.tm_hour;
        long minute = local_time.tm_min;
        long second = local_time.tm_sec / 2;

        return (unsigned long)((year << 25) | (month << 21) | (day << 16) | (hour << 11) | (minute << 5) | second);
    }

    /// Inverse of GetAsDOS, identical to wxDateTime::SetFromDOS.
    DateTime& SetFromDOS(unsigned long dos_timestamp) {
        struct tm local_time;
        memset(&local_time, 0, sizeof(local_time));
        local_time.tm_year  = int(((dos_timestamp & 0xFE000000) >> 25) + 80);
        local_time.tm_mon   = int(((dos_timestamp & 0x01E00000) >> 21) - 1);
        local_time.tm_mday  = int((dos_timestamp & 0x001F0000) >> 16);
        local_time.tm_hour  = int((dos_timestamp & 0x0000F800) >> 11);
        local_time.tm_min   = int((dos_timestamp & 0x000007E0) >> 5);
        local_time.tm_sec   = int((dos_timestamp & 0x0000001F) * 2);
        local_time.tm_isdst = -1;
        return Set(mktime(&local_time));
    }

    TimeSpan Subtract(const DateTime& other) const {
        return TimeSpan::Milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(time_point_ - other.time_point_).count( ));
    }

    TimeSpan operator-(const DateTime& other) const { return Subtract(other); }

    DateTime operator+(const TimeSpan& span) const { return DateTime(time_point_ + std::chrono::milliseconds(span.GetMilliseconds( ))); }

    DateTime operator-(const TimeSpan& span) const { return DateTime(time_point_ - std::chrono::milliseconds(span.GetMilliseconds( ))); }

    bool operator==(const DateTime& other) const { return valid_ == other.valid_ && time_point_ == other.time_point_; }

    bool operator!=(const DateTime& other) const { return ! (*this == other); }

    bool operator<(const DateTime& other) const { return time_point_ < other.time_point_; }

    bool operator>(const DateTime& other) const { return time_point_ > other.time_point_; }

    bool operator<=(const DateTime& other) const { return time_point_ <= other.time_point_; }

    bool operator>=(const DateTime& other) const { return time_point_ >= other.time_point_; }

    bool IsEarlierThan(const DateTime& other) const { return *this < other; }

    bool IsLaterThan(const DateTime& other) const { return *this > other; }

  private:
    Clock::time_point time_point_;
    bool              valid_;
};

/// Millisecond wall-clock timer (wxStopWatch).
class ElapsedTimer {
  public:
    ElapsedTimer( ) { Start( ); }

    void Start( ) {
        start_  = std::chrono::steady_clock::now( );
        paused_ = false;
        offset_ = std::chrono::steady_clock::duration::zero( );
    }

    void Pause( ) {
        if ( ! paused_ ) {
            offset_ += std::chrono::steady_clock::now( ) - start_;
            paused_ = true;
        }
    }

    void Resume( ) {
        if ( paused_ ) {
            start_  = std::chrono::steady_clock::now( );
            paused_ = false;
        }
    }

    /// Elapsed milliseconds.
    long Time( ) const {
        return long(std::chrono::duration_cast<std::chrono::milliseconds>(Elapsed( )).count( ));
    }

    /// Elapsed microseconds.
    long long TimeInMicro( ) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(Elapsed( )).count( );
    }

  private:
    std::chrono::steady_clock::duration Elapsed( ) const {
        return paused_ ? offset_ : offset_ + (std::chrono::steady_clock::now( ) - start_);
    }

    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::duration   offset_;
    bool                                  paused_;
};

#endif // _SRC_CORE_DATE_TIME_H_
