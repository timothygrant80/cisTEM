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
 * Threads: the main thread runs the wx event loop and owns nothing but
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

#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <wx/evtloop.h>
#include <wx/socket.h>
#include <wx/tokenzr.h>
#include <wx/utils.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <vector>
#include <atomic>

#include "../../core/core_headers.h"
#include "../../core/socket_codes.h"

// wxJSON's AsInt() only accepts values it stored as SHORT (or as LONG on a
// 32-bit build), so on 64-bit Linux any integer above 32767 -- a particle
// range in a big stack, a beam-tilt search position -- trips its assertion.
// Read every integer through the widest signed accessor instead.
static int JsonToInt(const wxJSONValue& value) {
    if ( value.IsInt( ) )
        return value.AsInt( );
    if ( value.IsInt64( ) )
        return int(value.AsInt64( ));
    if ( value.IsUInt64( ) )
        return int(value.AsUInt64( ));
    if ( value.IsDouble( ) )
        return int(value.AsDouble( ));
    return value.AsInt( );
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
static const wxUint32 kMaxPayload          = 64u * 1024u * 1024u;

static const wxUint8 kKindJson   = 0x01;
static const wxUint8 kKindBinary = 0x02;

#ifndef CISTEM_JOB_CONTROLLER_VERSION
#define CISTEM_JOB_CONTROLLER_VERSION "phase1"
#endif

// ---------------------------------------------------------------------------
// Framing helpers (spec section 3). Big-endian length, kind byte, payload.
// ---------------------------------------------------------------------------

enum ReadStatus { READ_OK, READ_IDLE, READ_CLOSED, READ_ERROR, READ_OVERSIZE };

// Read exactly `n` bytes. With wxSOCKET_WAITALL|wxSOCKET_BLOCK a Read either
// completes or times out; on a timeout that delivered nothing we report
// READ_IDLE so the caller can send a ping, otherwise we keep going -- a
// partial header must never be dropped.
static ReadStatus ReadExact(wxSocketBase* sock, unsigned char* buffer, wxUint32 n) {
    wxUint32 got = 0;
    while ( got < n ) {
        sock->Read(buffer + got, n - got);
        wxUint32 count = sock->LastReadCount( );
        got += count;
        if ( sock->Error( ) ) {
            if ( sock->LastError( ) == wxSOCKET_TIMEDOUT ) {
                if ( got == 0 )
                    return READ_IDLE;
                continue;
            }
            return READ_ERROR;
        }
        if ( count == 0 )
            return READ_CLOSED;
    }
    return READ_OK;
}

static ReadStatus ReadFrame(wxSocketBase* sock, wxUint8& kind, std::vector<unsigned char>& payload) {
    unsigned char header[5];
    ReadStatus    status = ReadExact(sock, header, 5);
    if ( status != READ_OK )
        return status;
    wxUint32 length = (wxUint32(header[0]) << 24) | (wxUint32(header[1]) << 16) | (wxUint32(header[2]) << 8) | wxUint32(header[3]);
    kind            = header[4];
    if ( length > kMaxPayload )
        return READ_OVERSIZE;
    payload.resize(length);
    if ( length == 0 )
        return READ_OK;
    // The payload follows immediately; a timeout here is a stall, not idleness.
    wxUint32 got = 0;
    while ( got < length ) {
        sock->Read(&payload[got], length - got);
        wxUint32 count = sock->LastReadCount( );
        got += count;
        if ( sock->Error( ) && sock->LastError( ) != wxSOCKET_TIMEDOUT )
            return READ_ERROR;
        if ( count == 0 && ! sock->Error( ) )
            return READ_CLOSED;
    }
    return READ_OK;
}

static bool WriteFrame(wxSocketBase* sock, wxUint8 kind, const std::string& payload) {
    if ( sock == NULL || ! sock->IsConnected( ) )
        return false;
    if ( payload.size( ) > kMaxPayload )
        return false;
    unsigned char header[5];
    wxUint32      length = wxUint32(payload.size( ));
    header[0]            = (length >> 24) & 0xff;
    header[1]            = (length >> 16) & 0xff;
    header[2]            = (length >> 8) & 0xff;
    header[3]            = length & 0xff;
    header[4]            = kind;
    sock->Write(header, 5);
    if ( sock->Error( ) )
        return false;
    if ( length > 0 ) {
        sock->Write(payload.data( ), length);
        if ( sock->Error( ) )
            return false;
    }
    return true;
}

static std::string JsonToString(wxJSONValue value) {
    wxJSONWriter writer(wxJSONWRITER_NONE);
    writer.SetDoubleFmtString("%.9g");
    wxString text;
    writer.Write(value, text);
    return std::string(text.ToUTF8( ).data( ));
}

static bool ParseJson(const std::vector<unsigned char>& payload, wxJSONValue& out) {
    wxString     text = wxString::FromUTF8(reinterpret_cast<const char*>(payload.data( )), payload.size( ));
    wxJSONReader reader(wxJSONREADER_STRICT);
    int          errors = reader.Parse(text, &out);
    return errors == 0 && out.IsObject( );
}

static long NowMs( ) {
    return wxGetUTCTimeMillis( ).GetValue( );
}

// ---------------------------------------------------------------------------
// The app
// ---------------------------------------------------------------------------

class ServerLinkThread;

class JobControllerApp : public wxAppConsole, public SocketCommunicator {
  public:
    // ---- command line ----
    wxArrayString server_hosts;
    long          server_port;
    wxString      token;
    double        reconnect_window_seconds;
    double        worker_timeout_seconds;
    // 0 = not waiting; otherwise the NowMs() by which the first worker must
    // have connected. Written on the main thread, polled by the link thread.
    std::atomic<long> worker_connect_deadline_ms;

    // ---- link to the server (guarded by link_mutex) ----
    wxMutex            link_mutex;
    wxSocketClient*    server_socket;
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
    wxJSONValue   task_refs;                // array: index -> ref (or null)
    std::vector<int> progress_counts;       // per task, for task_progress.result_number

    // ---- worker side (as in guix_job_control) ----
    bool          have_assigned_master;
    wxSocketBase* master_socket;
    wxString      master_ip_address;
    wxString      master_port;
    long          number_of_workers_already_connected;
    bool          all_jobs_are_finished;
    bool          cancel_in_progress;
    std::vector<char> task_reported;
    int           tasks_ok;
    int           tasks_failed;

    JobControllerApp( );

    virtual bool OnInit( );
    void         OnEventLoopEnter(wxEventLoopBase* loop);

    // ---- v1: sending ----
    long SendToServer(const wxString& type, wxJSONValue fields, bool buffer_for_resend = true);
    void SendLog(const wxString& level, const wxString& text);
    void SendWorkers( );
    void SendTaskDone(int task, const JobResult* result);
    void SendTaskProgress(const JobResult& result);
    void SendJobDone(const wxString& status, long cpu_ms, const wxString& error = wxEmptyString);
    int  ExpectedWorkers( );

    // ---- v1: receiving. The link thread handles ack/ping/pong itself and
    // hands everything else to the main thread as JSON text -- worker-side
    // state (and wx sockets) live there, exactly as SocketCommunicator
    // CallAfter()s every legacy handler onto it. ----
    void HandleServerMessageText(std::string payload);
    void HandleServerMessage(wxJSONValue message);
    void HandlePackage(wxJSONValue message);
    void HandleTasks(wxJSONValue message);
    void HandlePackageEnd( );
    void HandleCancel(const wxString& reason);
    void ProtocolFailure(const wxString& reason);

    // ---- worker side: legacy SocketCommunicator overrides ----
    void HandleNewSocketConnection(wxSocketBase* new_connection, unsigned char* identification_code);
    void HandleSocketIHaveAnError(wxSocketBase* connected_socket, wxString error_message);
    void HandleSocketIHaveInfo(wxSocketBase* connected_socket, wxString info_message);
    void HandleSocketJobResult(wxSocketBase* connected_socket, JobResult* received_result);
    void HandleSocketJobResultQueue(wxSocketBase* connected_socket, ArrayofJobResults* received_queue);
    void HandleSocketJobFinished(wxSocketBase* connected_socket, int finished_job_number);
    void HandleSocketAllJobsFinished(wxSocketBase* connected_socket, long received_timing_in_milliseconds);
    void HandleSocketDisconnect(wxSocketBase* connected_socket);
    void HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes);

    void LaunchWorkers( );
    void CheckWorkerTimeout( );   // link thread: poll the deadline
    void WorkerTimeoutExpired( ); // main thread: fail the job
    void KillWorkers( );
    // Safe from any thread: schedules the real shutdown on the main thread.
    void Shutdown(int exit_code, bool kill_workers = false);
    void DoShutdown(int exit_code, bool kill_workers);
};

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_SENDINFO, wxThreadEvent);

// ---------------------------------------------------------------------------
// Worker launch thread -- unchanged in substance from guix_job_control.cpp:
// substitutes $command / $program_name in each run command and executes it
// N times with the profile's delay.
// ---------------------------------------------------------------------------

class LaunchJobThread : public wxThread {
  public:
    LaunchJobThread(JobControllerApp* handler, RunProfile wanted_run_profile, wxString wanted_ip_address, wxString wanted_port, const unsigned char* wanted_job_code, long wanted_actual_number_of_jobs) : wxThread(wxTHREAD_DETACHED) {
        main_thread_pointer   = handler;
        current_run_profile   = wanted_run_profile;
        ip_address            = wanted_ip_address;
        port_number           = wanted_port;
        actual_number_of_jobs = wanted_actual_number_of_jobs;
        for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ )
            job_code[counter] = wanted_job_code[counter];
    }

  protected:
    JobControllerApp* main_thread_pointer;
    RunProfile        current_run_profile;
    wxString          ip_address;
    wxString          port_number;
    long              actual_number_of_jobs;
    unsigned char     job_code[SOCKET_CODE_SIZE];

    virtual ExitCode Entry( ) {
        wxString executable;
        if ( current_run_profile.controller_address == "" )
            executable = current_run_profile.executable_name + " " + ip_address + " " + port_number + " ";
        else
            executable = current_run_profile.executable_name + " " + current_run_profile.controller_address + " " + port_number + " ";
        for ( int counter = 0; counter < SOCKET_CODE_SIZE; counter++ )
            executable += job_code[counter];

        wxMilliSleep(2000);

        long number_of_commands_to_run = wxMin(actual_number_of_jobs, current_run_profile.ReturnTotalJobs( ));
        long number_of_commands_run    = 0;

        for ( long command_counter = 0; command_counter < current_run_profile.number_of_run_commands; command_counter++ ) {
            long number_to_run_for_this_command = wxMin(long(current_run_profile.run_commands[command_counter].number_of_copies),
                                                        number_of_commands_to_run - number_of_commands_run);
            wxString execution_command       = current_run_profile.run_commands[command_counter].command_to_run;
            wxString executable_with_threads = executable + wxString::Format(" %i", current_run_profile.run_commands[command_counter].number_of_threads_per_copy);
            execution_command.Replace("$command", executable_with_threads);
            execution_command.Replace("$program_name", current_run_profile.executable_name);
            execution_command += "&";

            for ( long process_counter = 0; process_counter < number_to_run_for_this_command; process_counter++ ) {
                wxMilliSleep(current_run_profile.run_commands[command_counter].delay_time_in_ms);
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
        return (wxThread::ExitCode)0;
    }
};

// ---------------------------------------------------------------------------
// Server link thread: connect, hello, read frames, reconnect. Owns the only
// blocking reads on the server socket.
// ---------------------------------------------------------------------------

class ServerLinkThread : public wxThread {
  public:
    ServerLinkThread(JobControllerApp* app) : wxThread(wxTHREAD_DETACHED), app(app) {}

  protected:
    JobControllerApp* app;

    wxSocketClient* ConnectOnce( ) {
        for ( size_t counter = 0; counter < app->server_hosts.GetCount( ); counter++ ) {
            wxIPV4address address;
            if ( ! address.Hostname(app->server_hosts.Item(counter)) )
                continue;
            address.Service(app->server_port);
            wxSocketClient* sock = new wxSocketClient(SOCKET_FLAGS);
            sock->Notify(false);
            sock->SetTimeout(kConnectTimeoutSeconds);
            sock->Connect(address, false);
            sock->WaitOnConnect(kConnectTimeoutSeconds);
            if ( sock->IsConnected( ) ) {
                sock->SetFlags(SOCKET_FLAGS);
                sock->SetTimeout(kPingAfterIdleSeconds);
                return sock;
            }
            sock->Close( );
            sock->Destroy( );
        }
        return NULL;
    }

    // Returns the connected socket or NULL when the window ran out.
    wxSocketClient* ConnectWithBackoff( ) {
        long   deadline_ms = NowMs( ) + long(app->reconnect_window_seconds * 1000.0);
        double delay       = 1.0;
        while ( NowMs( ) < deadline_ms ) {
            wxSocketClient* sock = ConnectOnce( );
            if ( sock != NULL )
                return sock;
            long remaining = deadline_ms - NowMs( );
            if ( remaining <= 0 )
                break;
            wxMilliSleep(long(wxMin(delay * 1000.0, double(remaining))));
            delay = wxMin(delay * 2.0, kMaxReconnectBackoff);
        }
        return NULL;
    }

    bool SendHello(wxSocketClient* sock, bool resume) {
        wxJSONValue hello;
        hello["type"] = wxString("hello");
        wxJSONValue versions(wxJSONTYPE_ARRAY);
        versions.Append(kProtocolVersion);
        hello["protocol_versions"] = versions;
        hello["token"]             = app->token;
        wxJSONValue controller;
        controller["name"]    = wxString("cistem_job_controller");
        controller["version"] = wxString(CISTEM_JOB_CONTROLLER_VERSION);
        controller["host"]    = wxGetHostName( );
        controller["pid"]     = (long)wxGetProcessId( );
        hello["controller"]   = controller;
        {
            wxMutexLocker lock(app->link_mutex);
            app->our_seq++;
            hello["seq"] = app->our_seq;
            hello["t"]   = NowMs( );
            if ( resume )
                hello["last_seq_sent"] = app->our_seq - 1;
            app->last_tx_ms = NowMs( );
        }
        return WriteFrame(sock, kKindJson, JsonToString(hello));
    }

    virtual ExitCode Entry( ) {
        bool resume = false;
        while ( true ) {
            wxSocketClient* sock = ConnectWithBackoff( );
            if ( sock == NULL ) {
                fprintf(stderr, "cistem_job_controller: could not (re)connect to the server within %.0f s\n", app->reconnect_window_seconds);
                app->Shutdown(EXIT_RECONNECT_EXPIRED, true);
                return (wxThread::ExitCode)0;
            }
            if ( ! SendHello(sock, resume) ) {
                sock->Destroy( );
                continue;
            }

            long last_rx_ms = NowMs( );
            bool welcomed   = false;
            bool lost       = false;

            while ( ! lost ) {
                app->CheckWorkerTimeout( );
                wxUint8                    kind = 0;
                std::vector<unsigned char> payload;
                ReadStatus                 status = ReadFrame(sock, kind, payload);

                if ( status == READ_IDLE ) {
                    long now = NowMs( );
                    if ( now - last_rx_ms > kDeadAfterSilenceSeconds * 1000 ) {
                        lost = true;
                        break;
                    }
                    {
                        wxMutexLocker lock(app->link_mutex);
                        if ( app->job_done_seq >= 0 && now - app->job_done_sent_ms > kJobDoneAckWaitSeconds * 1000 ) {
                            fprintf(stderr, "cistem_job_controller: no ack for job_done within %ld s\n", kJobDoneAckWaitSeconds);
                            app->Shutdown(EXIT_OTHER);
                            return (wxThread::ExitCode)0;
                        }
                        if ( now - app->last_tx_ms > kPingAfterIdleSeconds * 1000 ) {
                            wxJSONValue ping;
                            ping["type"] = wxString("ping");
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
                    return (wxThread::ExitCode)0;
                }
                last_rx_ms = NowMs( );
                if ( kind == kKindBinary )
                    continue; // no v1 message uses one; skip (spec section 3)
                if ( kind != kKindJson ) {
                    app->ProtocolFailure(wxString::Format("unknown frame kind 0x%02x", kind));
                    return (wxThread::ExitCode)0;
                }
                wxJSONValue message;
                if ( ! ParseJson(payload, message) ) {
                    app->ProtocolFailure("frame from server is not a JSON object");
                    return (wxThread::ExitCode)0;
                }
                wxString type = message.HasMember("type") ? message["type"].AsString( ) : wxString( );

                if ( ! welcomed ) {
                    if ( type == "reject" ) {
                        wxString code = message.HasMember("code") ? message["code"].AsString( ) : "?";
                        fprintf(stderr, "cistem_job_controller: rejected by server: %s %s\n", code.ToUTF8( ).data( ),
                                message.HasMember("reason") ? message["reason"].AsString( ).ToUTF8( ).data( ) : "");
                        if ( code == "already_connected" ) {
                            // Our previous connection hasn't been declared dead yet; wait and retry.
                            sock->Destroy( );
                            wxSleep(2);
                            lost = true;
                            break;
                        }
                        app->Shutdown(EXIT_REJECTED, true);
                        return (wxThread::ExitCode)0;
                    }
                    if ( type != "welcome" ) {
                        wxString detail = message.HasMember("reason") ? " (" + message["reason"].AsString( ) + ")" : wxString( );
                        app->ProtocolFailure("expected welcome from server, got " + type + detail);
                        return (wxThread::ExitCode)0;
                    }
                    welcomed = true;
                    {
                        wxMutexLocker lock(app->link_mutex);
                        app->server_socket = sock;
                        app->ever_welcomed = true;
                        if ( message.HasMember("resume") && message["resume"].AsBool( ) ) {
                            long resume_from = message.HasMember("resume_from_seq") ? message["resume_from_seq"].AsLong( ) : 0;
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
                    long upto = message.HasMember("upto") ? message["upto"].AsLong( ) : 0;
                    wxMutexLocker lock(app->link_mutex);
                    while ( ! app->unacked.empty( ) && app->unacked.front( ).first <= upto )
                        app->unacked.pop_front( );
                }
                else if ( type == "ping" ) {
                    app->SendToServer("pong", wxJSONValue( ), false);
                }
                else if ( type == "pong" ) {
                }
                else {
                    app->CallAfter(std::bind(&JobControllerApp::HandleServerMessageText, app,
                                             std::string(reinterpret_cast<const char*>(payload.data( )), payload.size( ))));
                }
                if ( app->job_done_seq >= 0 ) {
                    wxMutexLocker lock(app->link_mutex);
                    bool acked = true;
                    for ( std::deque<std::pair<long, std::string>>::iterator it = app->unacked.begin( ); it != app->unacked.end( ); ++it )
                        if ( it->first == app->job_done_seq )
                            acked = false;
                    if ( acked ) {
                        sock->Close( );
                        app->server_socket = NULL;
                        app->Shutdown(EXIT_JOB_OK);
                        return (wxThread::ExitCode)0;
                    }
                }
            }

            // Connection lost: forget the socket, keep everything else, and go round again.
            {
                wxMutexLocker lock(app->link_mutex);
                if ( app->server_socket == sock )
                    app->server_socket = NULL;
            }
            sock->Destroy( );
            fprintf(stderr, "cistem_job_controller: connection to the server lost; reconnecting\n");
        }
    }
};

// ---------------------------------------------------------------------------
// JobControllerApp
// ---------------------------------------------------------------------------

IMPLEMENT_APP(JobControllerApp)

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
    package_complete                    = false;
    expected_task_count                 = 0;
    tasks_received                      = 0;
    task_refs                           = wxJSONValue(wxJSONTYPE_ARRAY);
    have_assigned_master                = false;
    master_socket                       = NULL;
    number_of_workers_already_connected = 0;
    all_jobs_are_finished               = false;
    cancel_in_progress                  = false;
    tasks_ok                            = 0;
    tasks_failed                        = 0;
}

bool JobControllerApp::OnInit( ) {
    wxSocketBase::Initialize( );
    return true;
}

void JobControllerApp::OnEventLoopEnter(wxEventLoopBase* loop) {
    if ( ! loop->IsMain( ) )
        return;

    brother_event_handler = this; // required by SocketCommunicator, see gui_job_controller.cpp

    static const wxCmdLineEntryDesc command_line_descriptor[] = {
            {wxCMD_LINE_PARAM, NULL, NULL, "hosts", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY},
            {wxCMD_LINE_PARAM, NULL, NULL, "port", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_OPTION_MANDATORY},
            {wxCMD_LINE_PARAM, NULL, NULL, "token", wxCMD_LINE_VAL_STRING, wxCMD_LINE_OPTION_MANDATORY},
            {wxCMD_LINE_OPTION, NULL, "reconnect-window", "seconds to keep trying to reach the server", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL},
            {wxCMD_LINE_OPTION, NULL, "worker-timeout", "seconds to wait for the first worker to connect (0 = forever)", wxCMD_LINE_VAL_NUMBER, wxCMD_LINE_PARAM_OPTIONAL},
            {wxCMD_LINE_SWITCH, NULL, "verbose", "print protocol traffic to stderr", wxCMD_LINE_VAL_NONE, wxCMD_LINE_PARAM_OPTIONAL},
            {wxCMD_LINE_NONE}};

    wxCmdLineParser command_line_parser(command_line_descriptor, argc, argv);
    if ( command_line_parser.Parse(true) != 0 ) {
        exit(EXIT_OTHER);
    }

    wxStringTokenizer host_tokens(command_line_parser.GetParam(0), ",");
    while ( host_tokens.HasMoreTokens( ) ) {
        wxString host = host_tokens.GetNextToken( ).Trim( ).Trim(false);
        if ( ! host.IsEmpty( ) )
            server_hosts.Add(host);
    }
    if ( server_hosts.IsEmpty( ) || ! command_line_parser.GetParam(1).ToLong(&server_port) ) {
        fprintf(stderr, "cistem_job_controller: bad hosts or port\n");
        exit(EXIT_OTHER);
    }
    token = command_line_parser.GetParam(2);
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
    if ( link_thread->Run( ) != wxTHREAD_NO_ERROR ) {
        fprintf(stderr, "cistem_job_controller: can't start the server link thread\n");
        exit(EXIT_OTHER);
    }
}

// ---------------------------------------------------------------------------
// v1 sending
// ---------------------------------------------------------------------------

long JobControllerApp::SendToServer(const wxString& type, wxJSONValue fields, bool buffer_for_resend) {
    wxMutexLocker lock(link_mutex);
    our_seq++;
    fields["type"] = type;
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
    wxJSONValue fields;
    fields["level"] = level;
    fields["text"]  = text;
    SendToServer("log", fields);
}

int JobControllerApp::ExpectedWorkers( ) {
    long total = current_job_package.my_profile.ReturnTotalJobs( );
    return int(wxMin(total, long(current_job_package.number_of_jobs)));
}

void JobControllerApp::SendWorkers( ) {
    wxJSONValue fields;
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

    wxJSONValue fields;
    fields["task"]   = task;
    fields["status"] = wxString("ok");
    if ( task_refs.HasMember(unsigned(task)) && ! task_refs[unsigned(task)].IsNull( ) )
        fields["ref"] = task_refs[unsigned(task)];
    if ( result != NULL && result->result_size > 0 ) {
        wxJSONValue data(wxJSONTYPE_ARRAY);
        for ( int counter = 0; counter < result->result_size; counter++ )
            data.Append(double(result->result_data[counter]));
        wxJSONValue res;
        res["kind"]      = wxString("floats");
        res["data"]      = data;
        fields["result"] = res;
    }
    SendToServer("task_done", fields);
}

void JobControllerApp::SendTaskProgress(const JobResult& result) {
    int task = result.job_number;
    if ( task < 0 || task >= int(progress_counts.size( )) )
        return;
    progress_counts[task]++;
    wxJSONValue data(wxJSONTYPE_ARRAY);
    for ( int counter = 0; counter < result.result_size; counter++ )
        data.Append(double(result.result_data[counter]));
    wxJSONValue res;
    res["kind"] = wxString("floats");
    res["data"] = data;
    wxJSONValue fields;
    fields["task"]          = task;
    fields["result_number"] = progress_counts[task];
    fields["expected"]      = 0; // the legacy queue carries no total; 0 = unknown
    fields["result"]        = res;
    if ( task_refs.HasMember(unsigned(task)) && ! task_refs[unsigned(task)].IsNull( ) )
        fields["ref"] = task_refs[unsigned(task)];
    SendToServer("task_progress", fields);
}

void JobControllerApp::SendJobDone(const wxString& status, long cpu_ms, const wxString& error) {
    wxJSONValue fields;
    fields["status"]       = status;
    fields["cpu_ms"]       = cpu_ms;
    fields["tasks_ok"]     = tasks_ok;
    fields["tasks_failed"] = tasks_failed;
    if ( ! error.IsEmpty( ) )
        fields["error"] = error;
    long seq = SendToServer("job_done", fields);
    wxMutexLocker lock(link_mutex);
    job_done_seq     = seq;
    job_done_sent_ms = NowMs( );
}

// ---------------------------------------------------------------------------
// v1 receiving
// ---------------------------------------------------------------------------

void JobControllerApp::HandleServerMessageText(std::string payload) {
    wxJSONValue                message;
    std::vector<unsigned char> bytes(payload.begin( ), payload.end( ));
    if ( ! ParseJson(bytes, message) ) {
        ProtocolFailure("frame from server is not a JSON object");
        return;
    }
    HandleServerMessage(message);
}

void JobControllerApp::HandleServerMessage(wxJSONValue message) {
    wxString type = message.HasMember("type") ? message["type"].AsString( ) : wxString( );
    long     seq  = message.HasMember("seq") ? message["seq"].AsLong( ) : 0;
    if ( seq > 0 )
        last_server_seq = wxMax(last_server_seq, seq);

    if ( type == "ack" ) {
        long upto = message.HasMember("upto") ? message["upto"].AsLong( ) : 0;
        wxMutexLocker lock(link_mutex);
        while ( ! unacked.empty( ) && unacked.front( ).first <= upto )
            unacked.pop_front( );
    }
    else if ( type == "ping" ) {
        SendToServer("pong", wxJSONValue( ), false);
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
        HandleCancel(message.HasMember("reason") ? message["reason"].AsString( ) : wxString( ));
    }
    else if ( type == "protocol_error" ) {
        fprintf(stderr, "cistem_job_controller: server reported a protocol error: %s\n",
                message.HasMember("reason") ? message["reason"].AsString( ).ToUTF8( ).data( ) : "");
        Shutdown(EXIT_PROTOCOL_ERROR, true);
    }
    else if ( type == "welcome" || type == "reject" ) {
        ProtocolFailure("unexpected " + type + " mid-session");
    }
    else {
        fprintf(stderr, "cistem_job_controller: ignoring unknown message type '%s'\n", type.ToUTF8( ).data( ));
    }
}

void JobControllerApp::HandlePackage(wxJSONValue message) {
    if ( package_started ) {
        ProtocolFailure("second package for the same job");
        return;
    }
    if ( ! message.HasMember("program") || ! message.HasMember("profile") || ! message.HasMember("task_count") ) {
        ProtocolFailure("package is missing program, profile or task_count");
        return;
    }
    package_started     = true;
    expected_task_count = JsonToInt(message["task_count"]);
    if ( expected_task_count <= 0 ) {
        ProtocolFailure("package has no tasks");
        return;
    }

    wxJSONValue profile_json = message["profile"];
    RunProfile  profile;
    profile.name               = profile_json.HasMember("name") ? profile_json["name"].AsString( ) : "unnamed";
    profile.manager_command    = "$command";
    profile.gui_address        = "";
    profile.controller_address = profile_json.HasMember("controller_address") ? profile_json["controller_address"].AsString( ) : "";
    if ( profile_json.HasMember("run_commands") && profile_json["run_commands"].IsArray( ) ) {
        wxJSONValue commands = profile_json["run_commands"];
        for ( unsigned counter = 0; counter < unsigned(commands.Size( )); counter++ ) {
            wxJSONValue c = commands[counter];
            profile.AddCommand(c.HasMember("command") ? c["command"].AsString( ) : "$command",
                               c.HasMember("copies") ? JsonToInt(c["copies"]) : 1,
                               c.HasMember("threads_per_copy") ? JsonToInt(c["threads_per_copy"]) : 1,
                               c.HasMember("override_total_copies") ? c["override_total_copies"].AsBool( ) : false,
                               c.HasMember("overridden_total_copies") ? JsonToInt(c["overridden_total_copies"]) : 0,
                               c.HasMember("delay_ms") ? JsonToInt(c["delay_ms"]) : 0);
        }
    }
    wxString executable = message["program"].HasMember("executable") ? message["program"]["executable"].AsString( )
                                                                       : message["program"]["name"].AsString( );

    current_job_package.Reset(profile, executable, expected_task_count);
    task_refs = wxJSONValue(wxJSONTYPE_ARRAY);
    task_reported.assign(expected_task_count, 0);
    progress_counts.assign(expected_task_count, 0);
    tasks_received = 0;
}

void JobControllerApp::HandleTasks(wxJSONValue message) {
    if ( ! package_started || package_complete ) {
        ProtocolFailure("tasks outside a package");
        return;
    }
    int first_index = message.HasMember("first_index") ? JsonToInt(message["first_index"]) : -1;
    if ( first_index != tasks_received ) {
        ProtocolFailure(wxString::Format("tasks out of order: expected first_index %i, got %i", tasks_received, first_index));
        return;
    }
    if ( ! message.HasMember("tasks") || ! message["tasks"].IsArray( ) ) {
        ProtocolFailure("tasks message without a tasks array");
        return;
    }
    wxJSONValue tasks = message["tasks"];
    for ( unsigned counter = 0; counter < unsigned(tasks.Size( )); counter++ ) {
        wxJSONValue task  = tasks[counter];
        int         index = task.HasMember("index") ? JsonToInt(task["index"]) : -1;
        if ( index != tasks_received || index >= expected_task_count ) {
            ProtocolFailure(wxString::Format("task index %i where %i was expected", index, tasks_received));
            return;
        }
        if ( ! task.HasMember("args") || ! task["args"].IsArray( ) ) {
            ProtocolFailure(wxString::Format("task %i has no args array", index));
            return;
        }
        wxJSONValue args = task["args"];
        RunJob&     job  = current_job_package.jobs[index];
        job.Reset(int(args.Size( )));
        job.job_number = index;
        for ( unsigned a = 0; a < unsigned(args.Size( )); a++ ) {
            wxJSONValue arg   = args[a];
            wxString    atype = arg.HasMember("type") ? arg["type"].AsString( ) : wxString( );
            if ( ! arg.HasMember("value") ) {
                ProtocolFailure(wxString::Format("task %i argument %u has no value", index, a));
                return;
            }
            if ( atype == "text" )
                job.arguments[a].SetStringArgument(arg["value"].AsString( ).ToUTF8( ).data( ));
            else if ( atype == "int" )
                job.arguments[a].SetIntArgument(JsonToInt(arg["value"]));
            else if ( atype == "float" )
                job.arguments[a].SetFloatArgument(float(arg["value"].AsDouble( )));
            else if ( atype == "bool" )
                job.arguments[a].SetBoolArgument(arg["value"].AsBool( ));
            else {
                ProtocolFailure(wxString::Format("task %i argument %u has unknown type '%s'", index, a, atype));
                return;
            }
        }
        task_refs.Append(task.HasMember("ref") ? task["ref"] : wxJSONValue(wxJSONTYPE_NULL));
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
    wxJSONValue fields;
    fields["reason"] = reason;
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
        wxMutexLocker lock(link_mutex);
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
    if ( launch_thread->Run( ) != wxTHREAD_NO_ERROR ) {
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

void JobControllerApp::HandleNewSocketConnection(wxSocketBase* new_connection, unsigned char* identification_code) {
    if ( new_connection == NULL )
        return;

    if ( memcmp(identification_code, current_job_code, SOCKET_CODE_SIZE) != 0 ) {
        SendLog("error", "a process with an unknown job code connected (leftover from a previous job?) - closing it");
        new_connection->Destroy( );
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

void JobControllerApp::HandleSocketIHaveAnError(wxSocketBase* connected_socket, wxString error_message) {
    SendLog("error", error_message);
}

void JobControllerApp::HandleSocketIHaveInfo(wxSocketBase* connected_socket, wxString info_message) {
    SendLog("info", info_message);
}

void JobControllerApp::HandleSocketJobResult(wxSocketBase* connected_socket, JobResult* received_result) {
    SendTaskDone(received_result->job_number, received_result);
    delete received_result;
}

void JobControllerApp::HandleSocketJobResultQueue(wxSocketBase* connected_socket, ArrayofJobResults* received_queue) {
    for ( size_t counter = 0; counter < received_queue->GetCount( ); counter++ )
        SendTaskProgress(received_queue->Item(counter));
    delete received_queue;
}

void JobControllerApp::HandleSocketJobFinished(wxSocketBase* connected_socket, int finished_job_number) {
    SendTaskDone(finished_job_number, NULL);
}

void JobControllerApp::HandleSocketAllJobsFinished(wxSocketBase* connected_socket, long received_timing_in_milliseconds) {
    all_jobs_are_finished = true;
    // Anything the master never reported on is a failure as far as the
    // server's bookkeeping goes; say so explicitly so every task has a row.
    for ( int task = 0; task < int(task_reported.size( )); task++ ) {
        if ( ! task_reported[task] ) {
            task_reported[task] = 1;
            tasks_failed++;
            wxJSONValue fields;
            fields["task"]   = task;
            fields["status"] = wxString("failed");
            fields["error"]  = wxString("the master finished without reporting a result for this task");
            if ( task_refs.HasMember(unsigned(task)) && ! task_refs[unsigned(task)].IsNull( ) )
                fields["ref"] = task_refs[unsigned(task)];
            SendToServer("task_done", fields);
        }
    }
    SendJobDone(tasks_failed == 0 ? "completed" : "failed", received_timing_in_milliseconds,
                tasks_failed == 0 ? wxString( ) : wxString::Format("%i task(s) reported no result", tasks_failed));
    // Don't exit yet: the link thread does, once the server acks job_done.
}

void JobControllerApp::HandleSocketTemplateMatchResultReady(wxSocketBase* connected_socket, int& image_number, float& threshold_used, ArrayOfTemplateMatchFoundPeakInfos& peak_infos, ArrayOfTemplateMatchFoundPeakInfos& peak_changes) {
    // Phase 2 material: becomes a result.kind. Note it rather than lose it silently.
    SendLog("info", wxString::Format("template match result for image %i (%zu peaks) received; not relayed in protocol v1 phase 1", image_number, size_t(peak_infos.GetCount( ))));
}

void JobControllerApp::HandleSocketDisconnect(wxSocketBase* connected_socket) {
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
