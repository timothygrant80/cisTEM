#include "tcp_socket.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

const long kDefaultTimeoutSeconds = 600; // wxSocketBase's default

std::string AddressToString(const sockaddr_in& address) {
    char text[INET_ADDRSTRLEN];
    if ( inet_ntop(AF_INET, &address.sin_addr, text, sizeof(text)) == nullptr )
        return std::string( );
    return std::string(text);
}

bool SetBlocking(int descriptor, bool blocking) {
    int flags = fcntl(descriptor, F_GETFL, 0);
    if ( flags < 0 )
        return false;
    flags = blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK);
    return fcntl(descriptor, F_SETFL, flags) == 0;
}

} // namespace

// ------------------------------------------------------------------------------------------ TcpSocket

TcpSocket::TcpSocket( ) : descriptor_(-1), connected_(false), error_(false), last_errno_(0), timeout_seconds_(kDefaultTimeoutSeconds) {
}

TcpSocket::TcpSocket(int descriptor) : descriptor_(descriptor), connected_(descriptor >= 0), error_(false), last_errno_(0), timeout_seconds_(kDefaultTimeoutSeconds) {
    if ( descriptor_ >= 0 ) {
        SetBlocking(descriptor_, true);
        ApplyTimeout( );
    }
}

TcpSocket::~TcpSocket( ) {
    Close( );
}

void TcpSocket::Close( ) {
    if ( descriptor_ >= 0 ) {
        ::shutdown(descriptor_, SHUT_RDWR);
        ::close(descriptor_);
        descriptor_ = -1;
    }
    connected_ = false;
}

void TcpSocket::MarkFailed(int error_number) {
    error_      = true;
    last_errno_ = error_number;
    connected_  = false;
}

void TcpSocket::ApplyTimeout( ) {
    if ( descriptor_ < 0 )
        return;
    struct timeval timeout;
    timeout.tv_sec  = timeout_seconds_;
    timeout.tv_usec = 0;
    setsockopt(descriptor_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(descriptor_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

void TcpSocket::SetTimeout(long seconds) {
    timeout_seconds_ = seconds < 0 ? 0 : seconds;
    ApplyTimeout( );
}

bool TcpSocket::Connect(const std::string& host, int port, int timeout_seconds) {
    Close( );
    error_      = false;
    last_errno_ = 0;

    std::string ip_address;
    if ( ! ResolveHostnameToIPv4(host, ip_address) ) {
        MarkFailed(EHOSTUNREACH);
        return false;
    }

    sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port   = htons(uint16_t(port));
    if ( inet_pton(AF_INET, ip_address.c_str( ), &address.sin_addr) != 1 ) {
        MarkFailed(EINVAL);
        return false;
    }

    descriptor_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if ( descriptor_ < 0 ) {
        MarkFailed(errno);
        return false;
    }

    SetBlocking(descriptor_, false);
    int result = ::connect(descriptor_, reinterpret_cast<sockaddr*>(&address), sizeof(address));
    if ( result < 0 && errno != EINPROGRESS ) {
        int saved = errno;
        Close( );
        MarkFailed(saved);
        return false;
    }

    if ( result < 0 ) {
        pollfd poll_descriptor;
        poll_descriptor.fd     = descriptor_;
        poll_descriptor.events = POLLOUT;
        int ready              = ::poll(&poll_descriptor, 1, timeout_seconds * 1000);
        if ( ready <= 0 ) {
            Close( );
            MarkFailed(ready == 0 ? ETIMEDOUT : errno);
            return false;
        }
        int       socket_error = 0;
        socklen_t length       = sizeof(socket_error);
        if ( getsockopt(descriptor_, SOL_SOCKET, SO_ERROR, &socket_error, &length) < 0 || socket_error != 0 ) {
            Close( );
            MarkFailed(socket_error != 0 ? socket_error : errno);
            return false;
        }
    }

    SetBlocking(descriptor_, true);
    int one = 1;
    setsockopt(descriptor_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ApplyTimeout( );
    connected_ = true;
    return true;
}

bool TcpSocket::Write(const void* buffer, size_t nbytes) {
    if ( ! IsConnected( ) )
        return false;
    const char* data = static_cast<const char*>(buffer);
    size_t      sent = 0;
    while ( sent < nbytes ) {
        ssize_t count = ::send(descriptor_, data + sent, nbytes - sent, MSG_NOSIGNAL);
        if ( count < 0 ) {
            if ( errno == EINTR )
                continue;
            MarkFailed(errno);
            return false;
        }
        sent += size_t(count);
    }
    return true;
}

bool TcpSocket::Read(void* buffer, size_t nbytes) {
    if ( ! IsConnected( ) )
        return false;
    char*  data     = static_cast<char*>(buffer);
    size_t received = 0;
    while ( received < nbytes ) {
        ssize_t count = ::recv(descriptor_, data + received, nbytes - received, 0);
        if ( count == 0 ) {
            MarkFailed(0); // peer closed
            return false;
        }
        if ( count < 0 ) {
            if ( errno == EINTR )
                continue;
            MarkFailed(errno);
            return false;
        }
        received += size_t(count);
    }
    return true;
}

long TcpSocket::ReadSome(void* buffer, size_t nbytes) {
    if ( ! IsConnected( ) )
        return -1;
    while ( true ) {
        ssize_t count = ::recv(descriptor_, buffer, nbytes, 0);
        if ( count == 0 ) {
            MarkFailed(0);
            return 0;
        }
        if ( count < 0 ) {
            if ( errno == EINTR )
                continue;
            MarkFailed(errno);
            return -1;
        }
        return long(count);
    }
}

bool TcpSocket::WaitForRead(long timeout_milliseconds) {
    if ( descriptor_ < 0 )
        return false;
    pollfd poll_descriptor;
    poll_descriptor.fd     = descriptor_;
    poll_descriptor.events = POLLIN;
    int ready              = ::poll(&poll_descriptor, 1, int(timeout_milliseconds));
    if ( ready < 0 )
        return errno != EINTR; // treat a real poll error as "readable" so the caller's read reports it
    return ready > 0;
}

bool TcpSocket::WaitForWrite(long timeout_milliseconds) {
    if ( descriptor_ < 0 )
        return false;
    pollfd poll_descriptor;
    poll_descriptor.fd     = descriptor_;
    poll_descriptor.events = POLLOUT;
    int ready              = ::poll(&poll_descriptor, 1, int(timeout_milliseconds));
    return ready > 0 && (poll_descriptor.revents & POLLOUT);
}

std::string TcpSocket::LastErrorText( ) const {
    if ( ! error_ )
        return "No error";
    if ( last_errno_ == 0 )
        return "Connection closed by peer";
    return std::string(strerror(last_errno_));
}

std::string TcpSocket::ReturnLocalIPAddress( ) const {
    sockaddr_in address;
    socklen_t   length = sizeof(address);
    if ( descriptor_ < 0 || getsockname(descriptor_, reinterpret_cast<sockaddr*>(&address), &length) < 0 )
        return std::string( );
    return AddressToString(address);
}

int TcpSocket::ReturnLocalPort( ) const {
    sockaddr_in address;
    socklen_t   length = sizeof(address);
    if ( descriptor_ < 0 || getsockname(descriptor_, reinterpret_cast<sockaddr*>(&address), &length) < 0 )
        return -1;
    return ntohs(address.sin_port);
}

std::string TcpSocket::ReturnPeerIPAddress( ) const {
    sockaddr_in address;
    socklen_t   length = sizeof(address);
    if ( descriptor_ < 0 || getpeername(descriptor_, reinterpret_cast<sockaddr*>(&address), &length) < 0 )
        return std::string( );
    return AddressToString(address);
}

int TcpSocket::ReturnPeerPort( ) const {
    sockaddr_in address;
    socklen_t   length = sizeof(address);
    if ( descriptor_ < 0 || getpeername(descriptor_, reinterpret_cast<sockaddr*>(&address), &length) < 0 )
        return -1;
    return ntohs(address.sin_port);
}

// ------------------------------------------------------------------------------------------ TcpServer

TcpServer::TcpServer( ) : descriptor_(-1), port_(-1) {
}

TcpServer::~TcpServer( ) {
    Close( );
}

bool TcpServer::Listen(int port, int backlog) {
    Close( );

    descriptor_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if ( descriptor_ < 0 )
        return false;

    int one = 1;
    setsockopt(descriptor_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port        = htons(uint16_t(port));

    if ( ::bind(descriptor_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 || ::listen(descriptor_, backlog) < 0 ) {
        Close( );
        return false;
    }

    SetBlocking(descriptor_, false);
    port_ = port;
    return true;
}

TcpSocket* TcpServer::Accept(bool wait) {
    if ( descriptor_ < 0 )
        return nullptr;

    if ( wait ) {
        pollfd poll_descriptor;
        poll_descriptor.fd     = descriptor_;
        poll_descriptor.events = POLLIN;
        while ( ::poll(&poll_descriptor, 1, -1) < 0 ) {
            if ( errno != EINTR )
                return nullptr;
        }
    }

    sockaddr_in address;
    socklen_t   length = sizeof(address);
    int         client = ::accept(descriptor_, reinterpret_cast<sockaddr*>(&address), &length);
    if ( client < 0 )
        return nullptr; // EAGAIN / EWOULDBLOCK when nothing is pending

    int one = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return new TcpSocket(client);
}

void TcpServer::Close( ) {
    if ( descriptor_ >= 0 ) {
        ::close(descriptor_);
        descriptor_ = -1;
    }
    port_ = -1;
}

// ------------------------------------------------------------------------------------------ free functions

bool ResolveHostnameToIPv4(const std::string& host, std::string& ip_address) {
    if ( host.empty( ) )
        return false;

    in_addr binary;
    if ( inet_pton(AF_INET, host.c_str( ), &binary) == 1 ) {
        ip_address = host;
        return true;
    }

    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    if ( getaddrinfo(host.c_str( ), nullptr, &hints, &results) != 0 || results == nullptr )
        return false;

    bool found = false;
    for ( addrinfo* entry = results; entry != nullptr; entry = entry->ai_next ) {
        if ( entry->ai_family == AF_INET ) {
            ip_address = AddressToString(*reinterpret_cast<sockaddr_in*>(entry->ai_addr));
            found      = ! ip_address.empty( );
            if ( found )
                break;
        }
    }
    freeaddrinfo(results);
    return found;
}

bool HostnameIsValid(const std::string& host) {
    std::string junk;
    return ResolveHostnameToIPv4(host, junk);
}

std::vector<std::string> ReturnAllLocalIPv4Addresses( ) {
    std::vector<std::string> addresses;
    std::vector<std::string> loopback;

    ifaddrs* interfaces = nullptr;
    if ( getifaddrs(&interfaces) != 0 )
        return addresses;

    for ( ifaddrs* entry = interfaces; entry != nullptr; entry = entry->ifa_next ) {
        if ( entry->ifa_addr == nullptr || entry->ifa_addr->sa_family != AF_INET )
            continue;
        if ( ! (entry->ifa_flags & IFF_UP) )
            continue;
        std::string address = AddressToString(*reinterpret_cast<sockaddr_in*>(entry->ifa_addr));
        if ( address.empty( ) )
            continue;
        if ( entry->ifa_flags & IFF_LOOPBACK )
            loopback.push_back(address);
        else
            addresses.push_back(address);
    }
    freeifaddrs(interfaces);

    addresses.insert(addresses.end( ), loopback.begin( ), loopback.end( ));
    return addresses;
}

std::string ReturnLocalHostName( ) {
    char buffer[256];
    if ( gethostname(buffer, sizeof(buffer)) != 0 )
        return std::string( );
    buffer[sizeof(buffer) - 1] = '\0';
    return std::string(buffer);
}
