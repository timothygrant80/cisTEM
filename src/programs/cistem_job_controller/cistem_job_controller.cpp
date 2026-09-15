/*
 * cistem_job_controller -- the per-job controller that speaks cisTEM Job
 * Protocol v1 to a job server and the legacy socket protocol to the workers.
 *
 *     cistem_job_controller <hosts> <port> <token> [--reconnect-window S] [--worker-timeout S]
 *
 * --worker-timeout (default 300 s) is how long to wait for the *first* worker
 * to connect after the run commands are launched. The legacy controller waits
 * forever, so a run command that silently fails (executable not on PATH on the
 * remote host, a rejected sbatch) leaves a job "running" until someone notices.
 * Once any worker has connected the timeout is off: the rest may be queued in
 * a scheduler, and the master dispatches to whoever is there. Set it in the
 * run profile's manager command, e.g. "$command --worker-timeout 3600", when
 * a cluster queue can legitimately take longer than the default.
 *
 * This is guix_job_control.cpp with its GUI-facing half replaced. Everything
 * on the worker side is unchanged and still comes from SocketCommunicator:
 * the socket server the workers dial into, master election (first to
 * connect), the run-command fan-out with delays and thread counts, and the
 * relaying of results. What used to be WriteToSocket(gui_socket, <magic
 * code>, ...) is now a JSON frame to the server:
 *
 *     legacy (from master)                 v1 (to server)
 *     socket_i_have_info / _an_error   ->  log {level}
 *     socket_number_of_connections     ->  workers {connected, expected}
 *     socket_job_result                ->  task_done {status:"ok", result:{kind:"floats"}}
 *     socket_job_finished              ->  task_done {status:"ok"}       (a task with no result)
 *     socket_job_result_queue          ->  task_progress                  (intermediate results)
 *     socket_all_jobs_finished + ms    ->  job_done {cpu_ms}
 *     socket_time_to_die (from GUI)    <-  cancel
 *
 * The master sends either job_result or job_finished for a task, never
 * both (MyApp::HandleSocketSendNextJob), so each maps to exactly one
 * task_done.
 *
 * Reconnection (spec section 7): if the server connection drops, the workers
 * keep running; a link thread reconnects with backoff, says hello again, and
 * resends every frame the server has not acknowledged. The server's `ack`
 * lets us drop frames from that buffer. The controller exits only when the
 * server has acknowledged job_done, or the reconnect window runs out, or the
 * server rejects it.
 *
 * Threads: the main thread runs the EventLoop and owns nothing but
 * shutdown. SocketCommunicator's monitor thread delivers worker-side events
 * (the HandleSocket* overrides run on it, as in the legacy controller). The
 * ServerLinkThread reads frames from the server. All protocol state shared
 * between them -- sequence counter, unacked buffer, task bookkeeping, the
 * server socket itself -- is under `link_mutex`; every send goes through
 * SendToServer(), which takes it.
 *
 * See docs/job-protocol.md in the cisTEM3 web-app repository for the spec;
 * tools/fake_controller.py there is the Python reference this follows.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "../../core/core_headers.h"
#include "../../core/socket_communication_utils/socket_codes.h"

using json = nlohmann::json;

// Tolerant accessors: a value of the wrong JSON type never throws, it yields
// the conversion's natural default (0, false, ""), so a malformed field from
// the server is handled by the protocol checks below rather than by an
// uncaught exception. Integers are read through the widest signed type, so a
// particle range in a big stack or a beam-tilt search position is never
// truncated before the final int().
static long long JsonToLongLong(const json& value) {
    if ( value.is_number_integer( ) )
        return value.get<long long>( );
    if ( value.is_number_float( ) )
        return (long long)value.get<double>( );
    if ( value.is_boolean( ) )
        return value.get<bool>( ) ? 1 : 0;
    if ( value.is_string( ) )
        return atoll(value.get<std::string>( ).c_str( ));
    return 0;
}

static int JsonToInt(const json& value) {
    return int(JsonToLongLong(value));
}

static long JsonToLong(const json& value) {
    return long(JsonToLongLong(value));
}

static double JsonToDouble(const json& value) {
    if ( value.is_number( ) )
        return value.get<double>( );
    if ( value.is_boolean( ) )
        return value.get<bool>( ) ? 1.0 : 0.0;
    if ( value.is_string( ) )
        return atof(value.get<std::string>( ).c_str( ));
    return 0.0;
}

static bool JsonToBool(const json& value) {
    if ( value.is_boolean( ) )
        return value.get<bool>( );
    if ( value.is_number( ) )
        return value.get<double>( ) != 0.0;
    return false;
}

// A JSON string as UTF-8; other scalars/containers as their JSON text, null as "".
static std::string JsonToStdString(const json& value) {
    if ( value.is_string( ) )
        return value.get<std::string>( );
    if ( value.is_null( ) )
        return std::string( );
    return value.dump( );
}

static wxString JsonToWxString(const json& value) {
    std::string utf8 = JsonToStdString(value);
    return wxString::FromUTF8(utf8.data( ), utf8.size( ));
}

// wxString -> JSON string value (UTF-8).
static std::string ToUtf8(const wxString& text) {
    return text.utf8_string( );
}

// Member lookup that yields null for a missing key or a non-object, so nested
// lookups like JsonMember(message["program"], "executable") are always safe.
static const json& JsonMember(const json& value, const char* key) {
    static const json null_value;
    if ( value.is_object( ) ) {
        json::const_iterator it = value.find(key);
        if ( it != value.end( ) )
            return *it;
    }
    return null_value;
}

SETUP_SOCKET_CODES

// ---------------------------------------------------------------------------
// Exit codes (spec section 12)
// ---------------------------------------------------------------------------
enum {
    EXIT_JOB_OK          = 0,
    EXIT_OTHER           = 1,
    EXIT_REJECTED        = 2,
    EXIT_RECONNECT_EXPIRED = 3,
    EXIT_PROTOCOL_ERROR  = 4,
    EXIT_LAUNCH_FAILED   = 5,
};

// ---------------------------------------------------------------------------
// Timings (spec section 7)
// ---------------------------------------------------------------------------
static const int    kProtocolVersion       = 1;
static const long   kConnectTimeoutSeconds = 5;
static const long   kPingAfterIdleSeconds  = 15;
static const long   kDeadAfterSilenceSeconds = 60;
static const long   kJobDoneAckWaitSeconds = 30;
static const double kMaxReconnectBackoff   = 30.0;
static const uint32_t kMaxPayload          = 64u * 1024u * 1024u;

static const uint8_t kKindJson   = 0x01;
static const uint8_t kKindBinary = 0x02;

#ifndef CISTEM_JOB_CONTROLLER_VERSION
#define CISTEM_JOB_CONTROLLER_VERSION "phase1"
#endif

// ---------------------------------------------------------------------------
// Framing helpers (spec section 3). Big-endian length, kind byte, payload.
// ---------------------------------------------------------------------------

enum ReadStatus { READ_OK, READ_IDLE, READ_CLOSED, READ_ERROR, READ_OVERSIZE };

// Read exactly `n` bytes. The socket is polled in kPingAfterIdleSeconds slices so
// that a quiet link reports READ_IDLE (the caller then sends a ping) while a
// partial header is never dropped: once something has arrived we keep waiting.
static ReadStatus ReadExact(TcpSocket* sock, unsigned char* buffer, uint32_t n) {
    uint32_t got = 0;
    while ( got < n ) {
        if ( ! sock->WaitForRead(kPingAfterIdleSeconds * 1000) ) {
            if ( got == 0 )
                return READ_IDLE;
            continue;
        }
        long count = sock->ReadSome(buffer + got, n - got);
        if ( count == 0 )
            return READ_CLOSED;
        if ( count < 0 )
            return READ_ERROR;
        got += uint32_t(count);
    }
    return READ_OK;
}

static ReadStatus ReadFrame(TcpSocket* sock, uint8_t& kind, std::vector<unsigned char>& payload) {
    unsigned char header[5];
    ReadStatus    status = ReadExact(sock, header, 5);
    if ( status != READ_OK )
        return status;
    uint32_t length = (uint32_t(header[0]) << 24) | (uint32_t(header[1]) << 16) | (uint32_t(header[2]) << 8) | uint32_t(header[3]);
    kind            = header[4];
    if ( length > kMaxPayload )
        return READ_OVERSIZE;
    payload.resize(length);
    if ( length == 0 )
        return READ_OK;
    // The payload follows immediately; a timeout here is a stall, not idleness.
    uint32_t got = 0;
    while ( got < length ) {
        if ( ! sock->WaitForRead(kPingAfterIdleSeconds * 1000) )
            continue;
        long count = sock->ReadSome(&payload[got], length - got);
        if ( count == 0 )
            return READ_CLOSED;
        if ( count < 0 )
            return READ_ERROR;
        got += uint32_t(count);
    }
    return READ_OK;
}

static bool WriteFrame(TcpSocket* sock, uint8_t kind, const std::string& payload) {
    if ( sock == NULL || ! sock->IsConnected( ) )
        return false;
    if ( payload.size( ) > kMaxPayload )
        return false;
    unsigned char header[5];
    uint32_t      length = uint32_t(payload.size( ));
    header[0]            = (length >> 24) & 0xff;
    header[1]            = (length >> 16) & 0xff;
    header[2]            = (length >> 8) & 0xff;
    header[3]            = length & 0xff;
    header[4]            = kind;
    if ( ! sock->Write(header, 5) )
        return false;
    if ( length > 0 && ! sock->Write(payload.data( ), length) )
        return false;
    return true;
}

// Compact, single-line JSON text. An invalid UTF-8 byte in a string (a
// worker's log line in a non-UTF-8 locale, say) is replaced with U+FFFD
// rather than aborting the frame.
static std::string JsonToString(const json& value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

static bool ParseJson(const std::vector<unsigned char>& payload, json& out) {
    out = json::parse(payload.begin( ), payload.end( ), nullptr, /*allow_exceptions=*/false);
    return ! out.is_discarded( ) && out.is_object( );
}

static long NowMs( ) {
    return long(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now( ).time_since_epoch( )).count( ));
}

// ---------------------------------------------------------------------------
// The app
// ---------------------------------------------------------------------------

class ServerLinkThread;

class JobControllerApp : public EventLoop, public SocketCommunicator {
  public:
    // ---- command line ----
    std::vector<std::string> server_hosts;
    long                     server_port;
    wxString                 token;
    double        reconnect_window_seconds;
    double        worker_timeout_seconds;
    // 0 = not waiting; otherwise the NowMs() by which the first worker must
    // have connected. Written on the main thread, polled by the link thread.
    std::atomic<long> worker_connect_deadline_ms;

    // ---- link to the server (guarded by link_mutex) ----
    std::mutex         link_mutex;
    TcpSocket*         server_socket;
    long               our_seq;                 // our envelope counter, never reset
    long               last_server_seq;         // highest seq seen from the server
    long               last_tx_ms;
    std::deque<std::pair<long, std::string>> unacked;  // (seq, frame bytes)
    long               job_done_seq;            // -1 until job_done is sent
    long               job_done_sent_ms;
    bool               ever_welcomed;
    ServerLinkThread*  link_thread;

    // ---- the package (assembled from package / tasks / package_end) ----
    bool          package_started;
    bool          package_complete;
    int           expected_task_count;
    int           tasks_received;
    json          task_refs;                // array: index -> ref (or null)
    std::vector<int> progress_counts;       // per task, for task_progress.result_number
    bool             forward_progress;      // package.forward_progress: relay intermediate results at all?

    // ---- worker side (as in guix_job_control) ----
    bool          have_assigned_master;
    TcpSocket*    master_socket;
    wxString      master_ip_address;
    wxString      master_port;
    long          number_of_workers_already_connected;
    bool          all_jobs_are_finished;
    bool          cancel_in_progress;
    std::vector<char> task_reported;
    int           tasks_ok;
    int           tasks_failed;

    JobControllerApp( );

    /// main(): parse the command line, start the link thread, run the event loop.
    int  Main(int argc, char** argv);
    void Start(int argc, char** argv);

    // ---- v1: sending ----
    long SendToServer(const wxString& type, json fields, bool buffer_for_resend = true);
    void SendLog(const wxString& level, const wxString& text);
    void SendWorkers( );
    void SendTaskDone(int task, const JobResult* result);
    void SendTaskProgress(const JobResult& result);
    void SendJobDone(const wxString& status, long cpu_ms, const wxString& error = wxEmptyString);
    int  ExpectedWorkers( );

    // ---- v1: receiving. The link thread handles ack/ping/pong itself and
    // hands everything else to the main thread as JSON text -- worker-side
    // state (and the worker sockets) live there, exactly as SocketCommunicator
    // CallAfter()s every legacy handler onto it. ----
    void HandleServerMessageText(std::string payload);
    void HandleServerMessage(json message);
    void HandlePackage(json message);
    void HandleTasks(json message);
    void HandlePackageEnd( );
    void HandleCancel(const wxString& reason);
    void ProtocolFailure(const wxString& reason);

    // ---- worker side: legacy SocketCommunicator overrides ----
    void HandleNewSocketConnection(TcpSocket* new_connection, unsigned char* identification_code);
    void HandleSocketIHaveAnError(TcpSocket* connected_socket, wxString error_message);
    void HandleSocketIHaveInfo(TcpSocket* connected_socket, wxString info_message);
    void HandleSocketJobResult(TcpSocket* connected_socket, JobResult* received_result);
    void HandleSocketJobResultQueue(TcpSocket* connected_socket, ArrayofJobResults* received_queue);
    void HandleSocketJobFinished(TcpSocket* connected_socket, int finished_job_number);
    void HandleSocketAllJobsFinished(TcpSocket* connected_socket, long received_timing_in_milliseconds);
    void HandleSocketDisconnect(TcpSocket* connected_socket) override;
    void HandleSocketTemplateMatchResultReady(TcpSocket* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes) override;

    void LaunchWorkers( );
    void CheckWorkerTimeout( );   // link thread: poll the deadline
    void WorkerTimeoutExpired( ); // main thread: fail the job
    void KillWorkers( );
    // Safe from any thread: schedules the real shutdown on the main thread.
    void Shutdown(int exit_code, bool kill_workers = false);
    void DoShutdown(int exit_code, bool kill_workers);
};

// ---------------------------------------------------------------------------
// Worker launch thread -- unchanged in substance from guix_job_control.cpp:
// substitutes $command / $program_name in each run command and executes it
// N times with the profile's delay.
// ---------------------------------------------------------------------------

class LaunchJobThread {
  public:
    LaunchJobThread(JobControllerApp* handler, RunProfile wanted_run_profile, wxString wanted_ip_address, wxString wanted_port, const unsigned char* wanted_job_code, long wanted_actual_number_of_jobs) {
        main_thread_pointer   = handler;
        current_run_profile   = wanted_run_profile;
        ip_address            = wanted_ip_address;
        port_number           = wanted_port;
        actual_number_of_jobs = wanted_actual_number_of_jobs;
        for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ )
            job_code[counter] = wanted_job_code[counter];
    }

    /// Run Entry() on a detached thread that owns this object and deletes it when done.
    bool Run( ) {
        try {
            std::thread([this]( ) {
                Entry( );
                delete this;
            }).detach( );
        }
        catch ( const std::system_error& ) {
            return false;
        }
        return true;
    }

  protected:
    JobControllerApp* main_thread_pointer;
    RunProfile        current_run_profile;
    wxString          ip_address;
    wxString          port_number;
    long              actual_number_of_jobs;
    unsigned char     job_code[SOCKET_CODE_SIZE];

    void Entry( ) {
        wxString executable;
        if ( current_run_profile.controller_address == "" )
            executable = current_run_profile.executable_name + " " + ip_address + " " + port_number + " ";
        else
            executable = current_run_profile.executable_name + " " + current_run_profile.controller_address + " " + port_number + " ";
        for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ )
            executable += job_code[counter];

        SleepForMilliseconds(2000);

        long number_of_commands_to_run = std::min(actual_number_of_jobs, current_run_profile.ReturnTotalJobs( ));
        long number_of_commands_run    = 0;

        for ( long command_counter = 0; command_counter < current_run_profile.number_of_run_commands; command_counter++ ) {
            long number_to_run_for_this_command = std::min(long(current_run_profile.run_commands[command_counter].number_of_copies),
                                                           number_of_commands_to_run - number_of_commands_run);
            wxString execution_command       = current_run_profile.run_commands[command_counter].command_to_run;
            wxString executable_with_threads = executable + wxString::Format(" %i", current_run_profile.run_commands[command_counter].number_of_threads_per_copy);
            execution_command.Replace("$command", executable_with_threads);
            execution_command.Replace("$program_name", current_run_profile.executable_name);
            execution_command += "&";

            for ( long process_counter = 0; process_counter < number_to_run_for_this_command; process_counter++ ) {
                SleepForMilliseconds(current_run_profile.run_commands[command_counter].delay_time_in_ms);
                if ( process_counter == 0 ) {
                    // Blank the job code before it reaches any log.
                    wxString shown = execution_command;
                    shown.Replace(wxString::From8BitData(reinterpret_cast<const char*>(job_code), SOCKET_CODE_SIZE), "<job code>");
                    main_thread_pointer->SendLog("info", wxString::Format("Job Control : Executing '%s' %li times.", shown, number_to_run_for_this_command));
                }
                system(execution_command.ToUTF8( ).data( ));
                number_of_commands_run++;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Server link thread: connect, hello, read frames, reconnect. Owns the only
// blocking reads on the server socket.
// ---------------------------------------------------------------------------

class ServerLinkThread {
  public:
    ServerLinkThread(JobControllerApp* app) : app(app) {}

    /// Start Entry() on a detached thread; the object lives as long as the process.
    bool Run( ) {
        try {
            std::thread([this]( ) { Entry( ); }).detach( );
        }
        catch ( const std::system_error& ) {
            return false;
        }
        return true;
    }

  protected:
    JobControllerApp* app;

    TcpSocket* ConnectOnce( ) {
        for ( size_t counter = 0; counter < app->server_hosts.size( ); counter++ ) {
            TcpSocket* sock = new TcpSocket( );
            if ( sock->Connect(app->server_hosts[counter], int(app->server_port), int(kConnectTimeoutSeconds)) )
                return sock;
            delete sock;
        }
        return NULL;
    }

    // Returns the connected socket or NULL when the window ran out.
    TcpSocket* ConnectWithBackoff( ) {
        long   deadline_ms = NowMs( ) + long(app->reconnect_window_seconds * 1000.0);
        double delay       = 1.0;
        while ( NowMs( ) < deadline_ms ) {
            TcpSocket* sock = ConnectOnce( );
            if ( sock != NULL )
                return sock;
            long remaining = deadline_ms - NowMs( );
            if ( remaining <= 0 )
                break;
            SleepForMilliseconds(long(std::min(delay * 1000.0, double(remaining))));
            delay = std::min(delay * 2.0, kMaxReconnectBackoff);
        }
        return NULL;
    }

    bool SendHello(TcpSocket* sock, bool resume) {
        json hello;
        hello["type"] = "hello";
        json versions = json::array( );
        versions.push_back(kProtocolVersion);
        hello["protocol_versions"] = versions;
        hello["token"]             = ToUtf8(app->token);
        json controller;
        controller["name"]    = "cistem_job_controller";
        controller["version"] = CISTEM_JOB_CONTROLLER_VERSION;
        controller["host"]    = ReturnLocalHostName( );
        controller["pid"]     = (long)getpid( );
        hello["controller"]   = controller;
        {
            std::lock_guard<std::mutex> lock(app->link_mutex);
            app->our_seq++;
            hello["seq"] = app->our_seq;
            hello["t"]   = NowMs( );
            if ( resume )
                hello["last_seq_sent"] = app->our_seq - 1;
            app->last_tx_ms = NowMs( );
        }
        return WriteFrame(sock, kKindJson, JsonToString(hello));
    }

    void Entry( ) {
        bool resume = false;
        while ( true ) {
            TcpSocket* sock = ConnectWithBackoff( );
            if ( sock == NULL ) {
                fprintf(stderr, "cistem_job_controller: could not (re)connect to the server within %.0f s\n", app->reconnect_window_seconds);
                app->Shutdown(EXIT_RECONNECT_EXPIRED, true);
                return;
            }
            if ( ! SendHello(sock, resume) ) {
                delete sock;
                continue;
            }

            long last_rx_ms = NowMs( );
            bool welcomed   = false;
            bool lost       = false;

            while ( ! lost ) {
                app->CheckWorkerTimeout( );
                uint8_t                    kind = 0;
                std::vector<unsigned char> payload;
                ReadStatus                 status = ReadFrame(sock, kind, payload);

                if ( status == READ_IDLE ) {
                    long now = NowMs( );
                    if ( now - last_rx_ms > kDeadAfterSilenceSeconds * 1000 ) {
                        lost = true;
                        break;
                    }
                    {
                        std::lock_guard<std::mutex> lock(app->link_mutex);
                        if ( app->job_done_seq >= 0 && now - app->job_done_sent_ms > kJobDoneAckWaitSeconds * 1000 ) {
                            fprintf(stderr, "cistem_job_controller: no ack for job_done within %ld s\n", kJobDoneAckWaitSeconds);
                            app->Shutdown(EXIT_OTHER);
                            return;
                        }
                        if ( now - app->last_tx_ms > kPingAfterIdleSeconds * 1000 ) {
                            json ping;
                            ping["type"] = "ping";
                            app->our_seq++;
                            ping["seq"] = app->our_seq;
                            ping["t"]   = now;
                            WriteFrame(sock, kKindJson, JsonToString(ping));
                            app->last_tx_ms = now;
                        }
                    }
                    continue;
                }
                if ( status == READ_CLOSED || status == READ_ERROR ) {
                    lost = true;
                    break;
                }
                if ( status == READ_OVERSIZE ) {
                    app->ProtocolFailure("frame from server exceeds the 64 MiB limit");
                    return;
                }
                last_rx_ms = NowMs( );
                if ( kind == kKindBinary )
                    continue; // no v1 message uses one; skip (spec section 3)
                if ( kind != kKindJson ) {
                    app->ProtocolFailure(wxString::Format("unknown frame kind 0x%02x", kind));
                    return;
                }
                json message;
                if ( ! ParseJson(payload, message) ) {
                    app->ProtocolFailure("frame from server is not a JSON object");
                    return;
                }
                wxString type = JsonToWxString(JsonMember(message, "type"));

                if ( ! welcomed ) {
                    if ( type == "reject" ) {
                        wxString code = message.contains("code") ? JsonToWxString(message["code"]) : "?";
                        fprintf(stderr, "cistem_job_controller: rejected by server: %s %s\n", code.ToUTF8( ).data( ),
                                JsonToStdString(JsonMember(message, "reason")).c_str( ));
                        if ( code == "already_connected" ) {
                            // Our previous connection hasn't been declared dead yet; wait and retry.
                            SleepForSeconds(2);
                            lost = true;
                            break;
                        }
                        app->Shutdown(EXIT_REJECTED, true);
                        return;
                    }
                    if ( type != "welcome" ) {
                        wxString detail = message.contains("reason") ? " (" + JsonToWxString(message["reason"]) + ")" : wxString( );
                        app->ProtocolFailure("expected welcome from server, got " + type + detail);
                        return;
                    }
                    welcomed = true;
                    {
                        std::lock_guard<std::mutex> lock(app->link_mutex);
                        app->server_socket = sock;
                        app->ever_welcomed = true;
                        if ( JsonToBool(JsonMember(message, "resume")) ) {
                            long resume_from = JsonToLong(JsonMember(message, "resume_from_seq"));
                            int  resent      = 0;
                            for ( std::deque<std::pair<long, std::string>>::iterator it = app->unacked.begin( ); it != app->unacked.end( ); ++it ) {
                                if ( it->first > resume_from ) {
                                    WriteFrame(sock, kKindJson, it->second);
                                    resent++;
                                }
                            }
                            app->last_tx_ms = NowMs( );
                            if ( resent > 0 )
                                fprintf(stderr, "cistem_job_controller: resumed; resent %d frame(s) after seq %ld\n", resent, resume_from);
                        }
                    }
                    resume = true;
                    continue;
                }

                if ( type == "ack" ) {
                    long upto = JsonToLong(JsonMember(message, "upto"));
                    std::lock_guard<std::mutex> lock(app->link_mutex);
                    while ( ! app->unacked.empty( ) && app->unacked.front( ).first <= upto )
                        app->unacked.pop_front( );
                }
                else if ( type == "ping" ) {
                    app->SendToServer("pong", json::object( ), false);
                }
                else if ( type == "pong" ) {
                }
                else {
                    app->CallAfter(std::bind(&JobControllerApp::HandleServerMessageText, app,
                                             std::string(reinterpret_cast<const char*>(payload.data( )), payload.size( ))));
                }
                if ( app->job_done_seq >= 0 ) {
                    std::lock_guard<std::mutex> lock(app->link_mutex);
                    bool acked = true;
                    for ( std::deque<std::pair<long, std::string>>::iterator it = app->unacked.begin( ); it != app->unacked.end( ); ++it )
                        if ( it->first == app->job_done_seq )
                            acked = false;
                    if ( acked ) {
                        sock->Close( );
                        app->server_socket = NULL;
                        app->Shutdown(EXIT_JOB_OK);
                        return;
                    }
                }
            }

            // Connection lost: forget the socket, keep everything else, and go round again.
            {
                std::lock_guard<std::mutex> lock(app->link_mutex);
                if ( app->server_socket == sock )
                    app->server_socket = NULL;
            }
            delete sock;
            fprintf(stderr, "cistem_job_controller: connection to the server lost; reconnecting\n");
        }
    }
};

// ---------------------------------------------------------------------------
// JobControllerApp
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    JobControllerApp* app = new JobControllerApp( );
    return app->Main(argc, argv);
}

JobControllerApp::JobControllerApp( ) {
    server_port                         = 0;
    reconnect_window_seconds            = 600.0;
    worker_timeout_seconds              = 300.0;
    worker_connect_deadline_ms          = 0;
    server_socket                       = NULL;
    our_seq                             = 0;
    last_server_seq                     = 0;
    last_tx_ms                          = 0;
    job_done_seq                        = -1;
    job_done_sent_ms                    = 0;
    ever_welcomed                       = false;
    link_thread                         = NULL;
    package_started                     = false;
    forward_progress                    = true;
    package_complete                    = false;
    expected_task_count                 = 0;
    tasks_received                      = 0;
    task_refs                           = json::array( );
    have_assigned_master                = false;
    master_socket                       = NULL;
    number_of_workers_already_connected = 0;
    all_jobs_are_finished               = false;
    cancel_in_progress                  = false;
    tasks_ok                            = 0;
    tasks_failed                        = 0;
}

int JobControllerApp::Main(int argc, char** argv) {
    Start(argc, argv);
    return Run( ); // every exit path calls exit() from DoShutdown; this returns only if the loop is asked to stop
}

void JobControllerApp::Start(int argc, char** argv) {
    brother_event_handler = this; // required by SocketCommunicator

    CommandLineParser command_line_parser(argc, argv);
    command_line_parser.AddParam("hosts", CMD_LINE_VAL_STRING);
    command_line_parser.AddParam("port", CMD_LINE_VAL_NUMBER);
    command_line_parser.AddParam("token", CMD_LINE_VAL_STRING);
    command_line_parser.AddLongOption("reconnect-window", "seconds to keep trying to reach the server", CMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongOption("worker-timeout", "seconds to wait for the first worker to connect (0 = forever)", CMD_LINE_VAL_NUMBER);
    command_line_parser.AddLongSwitch("verbose", "print protocol traffic to stderr");

    if ( command_line_parser.Parse(true) != 0 ) {
        exit(EXIT_OTHER);
    }

    for ( const std::string& host : SplitString(command_line_parser.GetParam(0), ",") ) {
        std::string trimmed = Trimmed(host);
        if ( ! trimmed.empty( ) )
            server_hosts.push_back(trimmed);
    }
    if ( server_hosts.empty( ) || ! StringToLong(command_line_parser.GetParam(1), server_port) ) {
        fprintf(stderr, "cistem_job_controller: bad hosts or port\n");
        exit(EXIT_OTHER);
    }
    token = command_line_parser.GetParam(2);
    if ( token.IsEmpty( ) ) {
        fprintf(stderr, "cistem_job_controller: empty token\n");
        exit(EXIT_OTHER);
    }
    long window;
    if ( command_line_parser.Found("reconnect-window", &window) )
        reconnect_window_seconds = double(window);
    long worker_timeout;
    if ( command_line_parser.Found("worker-timeout", &worker_timeout) )
        worker_timeout_seconds = double(worker_timeout);

    // The legacy worker protocol identifies connections by a 16-byte job
    // code. Derive one from the token so it is per job and unguessable
    // enough for its purpose (it only has to tell this job's workers from
    // a stale one's); the token itself never goes to the workers.
    for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ )
        current_job_code[counter] = (unsigned char)token.GetChar(counter % token.Length( ));

    link_thread = new ServerLinkThread(this);
    if ( link_thread->Run( ) == false ) {
        fprintf(stderr, "cistem_job_controller: can't start the server link thread\n");
        exit(EXIT_OTHER);
    }
}

// ---------------------------------------------------------------------------
// v1 sending
// ---------------------------------------------------------------------------

long JobControllerApp::SendToServer(const wxString& type, json fields, bool buffer_for_resend) {
    std::lock_guard<std::mutex> lock(link_mutex);
    our_seq++;
    fields["type"] = ToUtf8(type);
    fields["seq"]  = our_seq;
    fields["t"]    = NowMs( );
    std::string frame = JsonToString(fields);
    if ( buffer_for_resend )
        unacked.push_back(std::make_pair(our_seq, frame));
    if ( server_socket != NULL ) {
        WriteFrame(server_socket, kKindJson, frame);
        last_tx_ms = NowMs( );
    }
    return our_seq;
}

void JobControllerApp::SendLog(const wxString& level, const wxString& text) {
    json fields;
    fields["level"] = ToUtf8(level);
    fields["text"]  = ToUtf8(text);
    SendToServer("log", fields);
}

int JobControllerApp::ExpectedWorkers( ) {
    long total = current_job_package.my_profile.ReturnTotalJobs( );
    return int(std::min(total, long(current_job_package.number_of_jobs)));
}

void JobControllerApp::SendWorkers( ) {
    json fields;
    fields["connected"] = number_of_workers_already_connected;
    fields["expected"]  = ExpectedWorkers( );
    SendToServer("workers", fields);
}

void JobControllerApp::SendTaskDone(int task, const JobResult* result) {
    if ( task < 0 || task >= int(task_reported.size( )) ) {
        SendLog("error", wxString::Format("master reported task %i, outside 0..%i", task, int(task_reported.size( )) - 1));
        return;
    }
    if ( task_reported[task] )
        return;
    task_reported[task] = 1;
    tasks_ok++;

    json fields;
    fields["task"]   = task;
    fields["status"] = "ok";
    if ( unsigned(task) < task_refs.size( ) && ! task_refs[unsigned(task)].is_null( ) )
        fields["ref"] = task_refs[unsigned(task)];
    if ( result != NULL && result->result_size > 0 ) {
        json data = json::array( );
        for ( int counter = 0; counter < result->result_size; counter++ )
            data.push_back(double(result->result_data[counter]));
        json res;
        res["kind"]      = "floats";
        res["data"]      = data;
        fields["result"] = res;
    }
    SendToServer("task_done", fields);
}

void JobControllerApp::SendTaskProgress(const JobResult& result) {
    // The server said it has no use for intermediate results (package.forward_progress
    // false): estimate_beamtilt sends one per search position, 290 880 per job.
    if ( ! forward_progress )
        return;
    int task = result.job_number;
    if ( task < 0 || task >= int(progress_counts.size( )) )
        return;
    progress_counts[task]++;
    json data = json::array( );
    for ( int counter = 0; counter < result.result_size; counter++ )
        data.push_back(double(result.result_data[counter]));
    json res;
    res["kind"] = "floats";
    res["data"] = data;
    json fields;
    fields["task"]          = task;
    fields["result_number"] = progress_counts[task];
    fields["expected"]      = 0; // the legacy queue carries no total; 0 = unknown
    fields["result"]        = res;
    if ( unsigned(task) < task_refs.size( ) && ! task_refs[unsigned(task)].is_null( ) )
        fields["ref"] = task_refs[unsigned(task)];
    SendToServer("task_progress", fields);
}

void JobControllerApp::SendJobDone(const wxString& status, long cpu_ms, const wxString& error) {
    json fields;
    fields["status"]       = ToUtf8(status);
    fields["cpu_ms"]       = cpu_ms;
    fields["tasks_ok"]     = tasks_ok;
    fields["tasks_failed"] = tasks_failed;
    if ( ! error.IsEmpty( ) )
        fields["error"] = ToUtf8(error);
    long seq = SendToServer("job_done", fields);
    std::lock_guard<std::mutex> lock(link_mutex);
    job_done_seq     = seq;
    job_done_sent_ms = NowMs( );
}

// ---------------------------------------------------------------------------
// v1 receiving
// ---------------------------------------------------------------------------

void JobControllerApp::HandleServerMessageText(std::string payload) {
    json                       message;
    std::vector<unsigned char> bytes(payload.begin( ), payload.end( ));
    if ( ! ParseJson(bytes, message) ) {
        ProtocolFailure("frame from server is not a JSON object");
        return;
    }
    HandleServerMessage(message);
}

void JobControllerApp::HandleServerMessage(json message) {
    wxString type = JsonToWxString(JsonMember(message, "type"));
    long     seq  = JsonToLong(JsonMember(message, "seq"));
    if ( seq > 0 )
        last_server_seq = std::max(last_server_seq, seq);

    if ( type == "ack" ) {
        long upto = JsonToLong(JsonMember(message, "upto"));
        std::lock_guard<std::mutex> lock(link_mutex);
        while ( ! unacked.empty( ) && unacked.front( ).first <= upto )
            unacked.pop_front( );
    }
    else if ( type == "ping" ) {
        SendToServer("pong", json::object( ), false);
    }
    else if ( type == "pong" ) {
    }
    else if ( type == "package" ) {
        HandlePackage(message);
    }
    else if ( type == "tasks" ) {
        HandleTasks(message);
    }
    else if ( type == "package_end" ) {
        HandlePackageEnd( );
    }
    else if ( type == "cancel" ) {
        HandleCancel(JsonToWxString(JsonMember(message, "reason")));
    }
    else if ( type == "protocol_error" ) {
        fprintf(stderr, "cistem_job_controller: server reported a protocol error: %s\n",
                JsonToStdString(JsonMember(message, "reason")).c_str( ));
        Shutdown(EXIT_PROTOCOL_ERROR, true);
    }
    else if ( type == "welcome" || type == "reject" ) {
        ProtocolFailure("unexpected " + type + " mid-session");
    }
    else {
        fprintf(stderr, "cistem_job_controller: ignoring unknown message type '%s'\n", type.ToUTF8( ).data( ));
    }
}

void JobControllerApp::HandlePackage(json message) {
    if ( package_started ) {
        ProtocolFailure("second package for the same job");
        return;
    }
    if ( ! message.contains("program") || ! message.contains("profile") || ! message.contains("task_count") ) {
        ProtocolFailure("package is missing program, profile or task_count");
        return;
    }
    package_started     = true;
    expected_task_count = JsonToInt(message["task_count"]);
    if ( expected_task_count <= 0 ) {
        ProtocolFailure("package has no tasks");
        return;
    }

    const json& profile_json = message["profile"];
    RunProfile  profile;
    profile.name               = profile_json.contains("name") ? JsonToWxString(profile_json["name"]) : "unnamed";
    profile.manager_command    = "$command";
    profile.gui_address        = "";
    profile.controller_address = profile_json.contains("controller_address") ? JsonToWxString(profile_json["controller_address"]) : "";
    if ( profile_json.contains("run_commands") && profile_json["run_commands"].is_array( ) ) {
        const json& commands = profile_json["run_commands"];
        for ( unsigned counter = 0; counter < unsigned(commands.size( )); counter++ ) {
            const json& c = commands[counter];
            profile.AddCommand(c.contains("command") ? JsonToWxString(c["command"]) : "$command",
                               c.contains("copies") ? JsonToInt(c["copies"]) : 1,
                               c.contains("threads_per_copy") ? JsonToInt(c["threads_per_copy"]) : 1,
                               c.contains("override_total_copies") ? JsonToBool(c["override_total_copies"]) : false,
                               c.contains("overridden_total_copies") ? JsonToInt(c["overridden_total_copies"]) : 0,
                               c.contains("delay_ms") ? JsonToInt(c["delay_ms"]) : 0);
        }
    }
    const json& program    = message["program"];
    wxString    executable = program.contains("executable") ? JsonToWxString(program["executable"])
                                                            : JsonToWxString(JsonMember(program, "name"));

    forward_progress = ! message.contains("forward_progress") || JsonToBool(message["forward_progress"]);
    current_job_package.Reset(profile, executable, expected_task_count);
    task_refs = json::array( );
    task_reported.assign(expected_task_count, 0);
    progress_counts.assign(expected_task_count, 0);
    tasks_received = 0;
}

void JobControllerApp::HandleTasks(json message) {
    if ( ! package_started || package_complete ) {
        ProtocolFailure("tasks outside a package");
        return;
    }
    int first_index = message.contains("first_index") ? JsonToInt(message["first_index"]) : -1;
    if ( first_index != tasks_received ) {
        ProtocolFailure(wxString::Format("tasks out of order: expected first_index %i, got %i", tasks_received, first_index));
        return;
    }
    if ( ! message.contains("tasks") || ! message["tasks"].is_array( ) ) {
        ProtocolFailure("tasks message without a tasks array");
        return;
    }
    const json& tasks = message["tasks"];
    for ( unsigned counter = 0; counter < unsigned(tasks.size( )); counter++ ) {
        const json& task  = tasks[counter];
        int         index = task.contains("index") ? JsonToInt(task["index"]) : -1;
        if ( index != tasks_received || index >= expected_task_count ) {
            ProtocolFailure(wxString::Format("task index %i where %i was expected", index, tasks_received));
            return;
        }
        if ( ! task.contains("args") || ! task["args"].is_array( ) ) {
            ProtocolFailure(wxString::Format("task %i has no args array", index));
            return;
        }
        const json& args = task["args"];
        RunJob&     job  = current_job_package.jobs[index];
        job.Reset(int(args.size( )));
        job.job_number = index;
        for ( unsigned a = 0; a < unsigned(args.size( )); a++ ) {
            const json& arg   = args[a];
            wxString    atype = JsonToWxString(JsonMember(arg, "type"));
            if ( ! arg.contains("value") ) {
                ProtocolFailure(wxString::Format("task %i argument %u has no value", index, a));
                return;
            }
            if ( atype == "text" )
                job.arguments[a].SetStringArgument(JsonToStdString(arg["value"]).c_str( ));
            else if ( atype == "int" )
                job.arguments[a].SetIntArgument(JsonToInt(arg["value"]));
            else if ( atype == "float" )
                job.arguments[a].SetFloatArgument(float(JsonToDouble(arg["value"])));
            else if ( atype == "bool" )
                job.arguments[a].SetBoolArgument(JsonToBool(arg["value"]));
            else {
                ProtocolFailure(wxString::Format("task %i argument %u has unknown type '%s'", index, a, atype));
                return;
            }
        }
        task_refs.push_back(task.contains("ref") ? task["ref"] : json( ));
        tasks_received++;
    }
}

void JobControllerApp::HandlePackageEnd( ) {
    if ( ! package_started || tasks_received != expected_task_count ) {
        ProtocolFailure(wxString::Format("package_end with %i of %i tasks", tasks_received, expected_task_count));
        return;
    }
    package_complete                         = true;
    current_job_package.number_of_added_jobs = expected_task_count;
    LaunchWorkers( );
}

void JobControllerApp::HandleCancel(const wxString& reason) {
    if ( cancel_in_progress || job_done_seq >= 0 )
        return;
    cancel_in_progress = true;
    SendLog("info", "cancelled by the server" + (reason.IsEmpty( ) ? wxString( ) : ": " + reason));
    KillWorkers( );
    SendJobDone("cancelled", 0);
}

void JobControllerApp::ProtocolFailure(const wxString& reason) {
    fprintf(stderr, "cistem_job_controller: protocol error: %s\n", reason.ToUTF8( ).data( ));
    json fields;
    fields["reason"] = ToUtf8(reason);
    SendToServer("protocol_error", fields, false);
    Shutdown(EXIT_PROTOCOL_ERROR, true);
}

// ---------------------------------------------------------------------------
// Worker side -- guix_job_control.cpp, with the GUI writes swapped for v1
// ---------------------------------------------------------------------------

void JobControllerApp::LaunchWorkers( ) {
    SetupServer( );
    wxString my_port_string = ReturnServerPortString( );

    // Prefer the address the server reached us on (it's the one most likely
    // to be routable), then everything else we have.
    wxString      current_address_according_to_server;
    wxArrayString my_possible_ip_addresses;
    {
        std::lock_guard<std::mutex> lock(link_mutex);
        if ( server_socket != NULL )
            current_address_according_to_server = ReturnIPAddressFromSocket(server_socket);
    }
    if ( ! current_address_according_to_server.IsEmpty( ) )
        my_possible_ip_addresses.Add(current_address_according_to_server);
    wxArrayString buffer_addresses = ReturnServerAllIpAddresses( );
    for ( size_t counter = 0; counter < buffer_addresses.GetCount( ); counter++ )
        if ( buffer_addresses.Item(counter) != current_address_according_to_server )
            my_possible_ip_addresses.Add(buffer_addresses.Item(counter));

    wxString ip_address_string;
    for ( size_t counter = 0; counter < my_possible_ip_addresses.GetCount( ); counter++ ) {
        if ( counter != 0 )
            ip_address_string += ",";
        ip_address_string += my_possible_ip_addresses.Item(counter);
    }

    SendLog("info", wxString::Format("Launching %i worker process(es) for %s", ExpectedWorkers( ), current_job_package.my_profile.executable_name));
    SendWorkers( );
    if ( worker_timeout_seconds > 0 )
        worker_connect_deadline_ms = NowMs( ) + long(worker_timeout_seconds * 1000.0);

    LaunchJobThread* launch_thread = new LaunchJobThread(this, current_job_package.my_profile, ip_address_string, my_port_string, current_job_code, current_job_package.number_of_jobs);
    if ( launch_thread->Run( ) == false ) {
        delete launch_thread;
        SendLog("error", "could not start the worker launch thread");
        SendJobDone("failed", 0, "could not start the worker launch thread");
    }
}

void JobControllerApp::CheckWorkerTimeout( ) {
    long deadline = worker_connect_deadline_ms.load( );
    if ( deadline != 0 && NowMs( ) > deadline ) {
        worker_connect_deadline_ms = 0; // fire once
        CallAfter(std::bind(&JobControllerApp::WorkerTimeoutExpired, this));
    }
}

void JobControllerApp::WorkerTimeoutExpired( ) {
    if ( have_assigned_master || cancel_in_progress || job_done_seq >= 0 )
        return;
    wxString why = wxString::Format(
            "no worker process connected within %.0f s of launching the run commands -- check that '%s' is on the PATH "
            "where they run, and that the commands themselves succeed (their output is in this job's controller log)",
            worker_timeout_seconds, current_job_package.my_profile.executable_name);
    SendLog("error", why);
    KillWorkers( );
    SendJobDone("failed", 0, why);
}

void JobControllerApp::KillWorkers( ) {
    if ( have_assigned_master && master_socket != NULL ) {
        WriteToSocket(master_socket, socket_time_to_die, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        StopMonitoringAndDestroySocket(master_socket);
        master_socket = NULL;
    }
    ShutDownServer( );
}

void JobControllerApp::Shutdown(int exit_code, bool kill_workers) {
    CallAfter(std::bind(&JobControllerApp::DoShutdown, this, exit_code, kill_workers));
}

void JobControllerApp::DoShutdown(int exit_code, bool kill_workers) {
    if ( kill_workers )
        KillWorkers( );
    ShutDownSocketMonitor( );
    fflush(stderr);
    exit(exit_code);
}

void JobControllerApp::HandleNewSocketConnection(TcpSocket* new_connection, unsigned char* identification_code) {
    if ( new_connection == NULL )
        return;

    if ( memcmp(identification_code, current_job_code, SOCKET_CODE_SIZE) != 0 ) {
        SendLog("error", "a process with an unknown job code connected (leftover from a previous job?) - closing it");
        delete new_connection;
    }
    else if ( ! have_assigned_master ) {
        worker_connect_deadline_ms = 0; // somebody made it; the rest may still be queued
        master_socket        = new_connection;
        have_assigned_master = true;
        WriteToSocket(new_connection, socket_you_are_the_master, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        current_job_package.SendJobPackage(new_connection);
        bool no_error;
        master_ip_address = ReceivewxStringFromSocket(new_connection, no_error);
        master_port       = ReceivewxStringFromSocket(new_connection, no_error);
        MonitorSocket(new_connection);
        number_of_workers_already_connected++;
        SendWorkers( );
    }
    else {
        WriteToSocket(new_connection, socket_you_are_a_worker, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        SendwxStringToSocket(&master_ip_address, new_connection);
        SendwxStringToSocket(&master_port, new_connection);
        MonitorSocket(new_connection);
        number_of_workers_already_connected++;
        SendWorkers( );
    }

    if ( number_of_workers_already_connected == ExpectedWorkers( ) ) {
        SendLog("info", wxString::Format("All %ld processes are connected.", number_of_workers_already_connected));
        ShutDownServer( ); // nobody else is expected
    }

    delete[] identification_code;
}

void JobControllerApp::HandleSocketIHaveAnError(TcpSocket* connected_socket, wxString error_message) {
    SendLog("error", error_message);
}

void JobControllerApp::HandleSocketIHaveInfo(TcpSocket* connected_socket, wxString info_message) {
    SendLog("info", info_message);
}

void JobControllerApp::HandleSocketJobResult(TcpSocket* connected_socket, JobResult* received_result) {
    SendTaskDone(received_result->job_number, received_result);
    delete received_result;
}

void JobControllerApp::HandleSocketJobResultQueue(TcpSocket* connected_socket, ArrayofJobResults* received_queue) {
    for ( size_t counter = 0; counter < received_queue->size( ); counter++ )
        SendTaskProgress((*received_queue)[counter]);
    delete received_queue;
}

void JobControllerApp::HandleSocketJobFinished(TcpSocket* connected_socket, int finished_job_number) {
    SendTaskDone(finished_job_number, NULL);
}

void JobControllerApp::HandleSocketAllJobsFinished(TcpSocket* connected_socket, long received_timing_in_milliseconds) {
    all_jobs_are_finished = true;
    // Anything the master never reported on is a failure as far as the
    // server's bookkeeping goes; say so explicitly so every task has a row.
    for ( int task = 0; task < int(task_reported.size( )); task++ ) {
        if ( ! task_reported[task] ) {
            task_reported[task] = 1;
            tasks_failed++;
            json fields;
            fields["task"]   = task;
            fields["status"] = "failed";
            fields["error"]  = "the master finished without reporting a result for this task";
            if ( unsigned(task) < task_refs.size( ) && ! task_refs[unsigned(task)].is_null( ) )
                fields["ref"] = task_refs[unsigned(task)];
            SendToServer("task_done", fields);
        }
    }
    SendJobDone(tasks_failed == 0 ? "completed" : "failed", received_timing_in_milliseconds,
                tasks_failed == 0 ? wxString( ) : wxString::Format("%i task(s) reported no result", tasks_failed));
    // Don't exit yet: the link thread does, once the server acks job_done.
}

void JobControllerApp::HandleSocketTemplateMatchResultReady(TcpSocket* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes) {
    // Phase 2 material: becomes a result.kind. Note it rather than lose it silently.
    SendLog("info", wxString::Format("template match result for image %i (%zu peaks) received; not relayed in protocol v1 phase 1", image_number, size_t(peak_infos.GetCount( ))));
}

void JobControllerApp::HandleSocketDisconnect(TcpSocket* connected_socket) {
    if ( connected_socket == master_socket ) {
        if ( ! all_jobs_are_finished && ! cancel_in_progress ) {
            SendLog("error", "the master process disconnected before the job was finished");
            master_socket = NULL;
            have_assigned_master = false;
            ShutDownServer( );
            SendJobDone("failed", 0, "the master process disconnected before the job was finished");
        }
        // otherwise: it's done, and we're waiting on the server's ack
    }
    else {
        // A worker dropping us to connect to the master, as designed.
        StopMonitoringAndDestroySocket(connected_socket);
    }
}
