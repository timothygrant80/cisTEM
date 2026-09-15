#ifndef _SRC_CORE_SOCKET_COMMUNICATION_UTILS_TCP_SOCKET_H_
#define _SRC_CORE_SOCKET_COMMUNICATION_UTILS_TCP_SOCKET_H_

/*
 * Blocking TCP sockets on the POSIX API, replacing wxSocketClient / wxSocketBase / wxSocketServer.
 *
 * Read() and Write() transfer the whole buffer or fail (the wxSOCKET_WAITALL | wxSOCKET_BLOCK
 * behaviour every cisTEM socket relied on). A failed transfer, a closed peer or an error marks the
 * socket disconnected; IsConnected() then returns false and the owner is expected to Close() and
 * delete it. Writes never raise SIGPIPE.
 */

#include <cstddef>
#include <string>
#include <vector>

class TcpSocket {
  public:
    TcpSocket( );
    /// Adopt an already-connected descriptor (from TcpServer::Accept).
    explicit TcpSocket(int descriptor);
    ~TcpSocket( );

    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    /**
     * Resolve `host` (a name or dotted IPv4 address) and connect to `port`, giving up after
     * `timeout_seconds`. On failure the socket is left closed and IsConnected() is false.
     */
    bool Connect(const std::string& host, int port, int timeout_seconds = 30);

    void Close( );

    /// True while a descriptor is open.
    bool IsOk( ) const { return descriptor_ >= 0; }

    /// True while connected and no transfer has failed.
    bool IsConnected( ) const { return descriptor_ >= 0 && connected_; }

    /// True after a failed transfer (a closed peer counts as an error, as it did with wxSocket).
    bool Error( ) const { return error_; }

    /// errno of the last failure, 0 for a peer that closed the connection cleanly.
    int LastError( ) const { return last_errno_; }

    std::string LastErrorText( ) const;

    /// Write exactly `nbytes`; false (and disconnected) on failure.
    bool Write(const void* buffer, size_t nbytes);

    /// Read exactly `nbytes`; false (and disconnected) on EOF or failure.
    bool Read(void* buffer, size_t nbytes);

    /// Read up to `nbytes`, returning the count (0 = peer closed, -1 = error).
    long ReadSome(void* buffer, size_t nbytes);

    /// True if data (or EOF / an error) is waiting to be read within the timeout; 0 polls without waiting.
    bool WaitForRead(long timeout_milliseconds = 0);

    /// True if the socket can be written to within the timeout.
    bool WaitForWrite(long timeout_milliseconds = 0);

    /// Blocking read and write timeout in seconds (default 600, wxSocket's default). 0 waits forever.
    void SetTimeout(long seconds);

    int GetDescriptor( ) const { return descriptor_; }

    std::string ReturnLocalIPAddress( ) const;
    int         ReturnLocalPort( ) const;
    std::string ReturnPeerIPAddress( ) const;
    int         ReturnPeerPort( ) const;

  private:
    void ApplyTimeout( );
    void MarkFailed(int error_number);

    int  descriptor_;
    bool connected_;
    bool error_;
    int  last_errno_;
    long timeout_seconds_;
};

class TcpServer {
  public:
    TcpServer( );
    ~TcpServer( );

    TcpServer(const TcpServer&) = delete;
    TcpServer& operator=(const TcpServer&) = delete;

    /// Listen on every interface at `port`. False if the port cannot be bound.
    bool Listen(int port, int backlog = 128);

    /**
     * Accept a pending connection, returning a new TcpSocket the caller owns, or nullptr when none
     * is pending (with wait == false) or on error. With wait == true blocks until one arrives.
     */
    TcpSocket* Accept(bool wait = false);

    void Close( );

    bool IsOk( ) const { return descriptor_ >= 0; }

    int ReturnPort( ) const { return port_; }

  private:
    int descriptor_;
    int port_;
};

/// Resolve a host name (or dotted address) to a dotted IPv4 address. False if it cannot be resolved.
bool ResolveHostnameToIPv4(const std::string& host, std::string& ip_address);

/// True if `host` is a dotted IPv4 address or a name that resolves (wxIPV4address::Hostname).
bool HostnameIsValid(const std::string& host);

/// Dotted IPv4 addresses of every interface that is up, loopback last.
std::vector<std::string> ReturnAllLocalIPv4Addresses( );

std::string ReturnLocalHostName( );

#endif // _SRC_CORE_SOCKET_COMMUNICATION_UTILS_TCP_SOCKET_H_
