"""The job server's half of the cisTEM Job Protocol (docs/job-protocol.md).

One JobRunner owns a listening TCP socket and every controller connection to
it, on a single event-loop thread. For each submitted job it launches the run
profile's manager command (which starts a `cistem_job_controller` that dials
back in), authenticates that controller by its per-job token, hands it the
package, and turns what comes back -- worker counts, log lines, per-task
results, the final job_done -- into calls on a Sink.

The Sink is the only way anything leaves this module. It is how the reference
server writes job state into the project database, and it is deliberately
the whole of the coupling: the runner opens no SQLite connections, knows no
table names, and could be lifted into its own process without change (which
is what would have to happen if the web server ever moved to a multi-worker
WSGI setup -- the listener has to be exactly one process).

Threading model: `start()` spawns the loop thread. `submit()`, `cancel()` and
`restore()` may be called from any thread; they take `_lock`, mutate the
session table, and rely on the loop's one-second tick to notice. Everything
that touches a socket happens on the loop thread only.

What the loop does on each tick, per spec section:
  - 7.1  send `ping` after 15 s of silence on our side; drop a connection
         after 60 s of silence on theirs (the session goes to awaiting
         reconnect, the workers keep running)
  - 7.2  fail a session whose controller never said hello within the launch
         timeout, or didn't reconnect within the reconnect window
  - 6.6  send `ack` at least every 5 s while unacknowledged frames exist,
         and immediately after `job_done`
  - 5    kill the manager process of a cancelled job that hasn't wound down
"""

import logging
import os
import selectors
import socket
import subprocess
import tempfile
import threading
import time

import job_protocol as jp

log = logging.getLogger("job_runner")

# Timings from the spec. Seconds.
PING_AFTER_IDLE = 15.0
DEAD_AFTER_SILENCE = 60.0
ACK_INTERVAL = 5.0
ACK_EVERY_N_FRAMES = 100        # or sooner, if the controller is chatty
LAUNCH_TIMEOUT = 120.0          # section 5: no hello -> failed
RECONNECT_WINDOW = 600.0        # section 7.2
FINISH_GRACE = 30.0             # after job_done + ack, how long to wait for the controller to hang up
CANCEL_GRACE = 60.0             # after cancel, how long before we kill the manager process
FINISHED_TOKEN_MEMORY = 3600.0  # how long a finished token still gets `reject job_finished` rather than `bad_token`

# Job status values the Sink sees. `awaiting_reconnect` is a sub-state of
# running as far as the API is concerned; the sink decides how to surface it.
LAUNCHING = "launching"
RUNNING = "running"
AWAITING_RECONNECT = "awaiting_reconnect"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"


class Sink:
    """Everything the runner reports, as no-op defaults. Subclass and
    override what you need. All calls arrive on the loop thread, so keep
    them quick -- a slow sink stalls every job's connection."""

    def on_status(self, job_id, status, error=None):
        """Job-level status changed. `error` accompanies FAILED."""

    def on_log(self, job_id, text, level="info"):
        """A line for the job log -- from the controller, or from the runner
        about the controller."""

    def on_workers(self, job_id, connected, expected):
        pass

    def on_task_progress(self, job_id, task, ref, result_number, expected, result):
        pass

    def on_task_done(self, job_id, task, ref, status, result, error, cpu_ms, done_count, task_count):
        """One task finished. `done_count`/`task_count` are for progress."""

    def on_job_done(self, job_id, status, cpu_ms, tasks_ok, tasks_failed, error=None):
        """The controller's final word. Called before on_status(terminal)."""

    def on_controller_seq(self, job_id, seq):
        """The highest controller `seq` processed so far -- persist it if you
        want reconnection to survive a server restart (see restore())."""


class JobSpec:
    """What submit() needs. `tasks` is the list of {"index", "ref"?, "args"}
    objects exactly as they go on the wire (job_protocol.arg builds args)."""

    def __init__(self, job_id, job_info, program, profile, tasks, manager_command, token=None,
                 controller_log=None, forward_progress=True):
        self.job_id = job_id
        # package.forward_progress: whether the controller should relay the
        # workers' intermediate results (task_progress). Off unless the
        # stage's adapter wants them -- a program like estimate_beamtilt
        # sends one per search position, hundreds of thousands per job.
        self.forward_progress = forward_progress
        self.job_info = job_info          # package.job
        self.program = program            # package.program: {"name", "executable"}
        self.profile = profile            # package.profile (db.load_run_profiles shape)
        self.tasks = tasks
        self.manager_command = manager_command
        self.token = token or jp.new_token()
        # Where the manager process's stdout/stderr go. A file, not a pipe
        # back to us: the controller must outlive a server restart (section
        # 7.2), and a process writing to a pipe whose reader has died gets
        # SIGPIPE. None -> a temp file.
        self.controller_log = controller_log


class _Session:
    """Server-side state for one job, living from submit() to a terminal
    status. Outlives any single connection -- that is the whole point."""

    def __init__(self, spec):
        self.spec = spec
        self.job_id = spec.job_id
        self.token = spec.token
        self.status = LAUNCHING
        self.seq = jp.Sequencer()               # our seq, one per job, never reset
        self.last_controller_seq = 0            # highest received *and processed*
        self.unacked = 0                        # frames since our last ack
        self.ack_due = None
        self.tasks_done = set()
        self.conn = None                        # _Conn or None
        self.ever_connected = False
        self.launch_deadline = None
        self.reconnect_deadline = None
        self.finish_deadline = None
        self.cancel_requested = False
        self.cancel_deadline = None
        self.manager = None                     # subprocess.Popen or None
        self.manager_log = None                 # path the manager's output is written to
        self.log_relayed = False
        self.terminal = None                    # COMPLETED/FAILED/CANCELLED once decided

    @property
    def task_count(self):
        return len(self.spec.tasks)


class _Conn:
    """One accepted TCP connection. Not bound to a session until hello."""

    def __init__(self, sock, peer):
        self.sock = sock
        self.peer = peer
        self.reader = jp.FrameReader()
        self.outbuf = bytearray()
        self.session = None
        self.last_rx = time.monotonic()
        self.last_tx = time.monotonic()
        self.closing = False


class JobRunner:
    def __init__(self, sink, bind_host="0.0.0.0", port=8010, advertise_hosts=None,
                 controller_executable="cistem_job_controller",
                 reconnect_window=RECONNECT_WINDOW, launch_timeout=LAUNCH_TIMEOUT,
                 server_info=None):
        self.sink = sink
        self.bind_host = bind_host
        self.port = port
        self.advertise_hosts = list(advertise_hosts) if advertise_hosts else None
        self.controller_executable = controller_executable
        self.reconnect_window = reconnect_window
        self.launch_timeout = launch_timeout
        self.server_info = server_info or {"name": "cistem3-server"}

        self._lock = threading.RLock()
        self._sessions = {}           # job_id -> _Session
        self._finished_tokens = {}    # token -> monotonic time of finish
        self._exiting = []            # manager Popens of forgotten sessions, polled until they exit
        self._conns = set()
        self._sel = None
        self._listener = None
        self._thread = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.bind_host, self.port))
        self._listener.listen(64)
        self._listener.setblocking(False)
        self.port = self._listener.getsockname()[1]  # in case 0 was asked for
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._listener, selectors.EVENT_READ, data=None)
        self._thread = threading.Thread(target=self._loop, name="job-runner", daemon=True)
        self._thread.start()
        log.info("job runner listening on %s:%d, advertising %s", self.bind_host, self.port,
                 ",".join(self.hosts_to_advertise()))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Whatever controllers are still out there belong to this server;
        # don't leave them dialling a port nobody listens on.
        with self._lock:
            procs = [s.manager for s in self._sessions.values() if s.manager] + list(self._exiting)
            self._sessions.clear()
            self._exiting.clear()
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def hosts_to_advertise(self):
        """The addresses the controller is told to try, in order -- the
        machine's real addresses first, loopback last, like the GUI's
        all_my_ip_addresses. Overridable for NAT/multi-homed setups."""
        if self.advertise_hosts:
            return self.advertise_hosts
        hosts = []
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                addr = info[4][0]
                if addr not in hosts and not addr.startswith("127."):
                    hosts.append(addr)
        except socket.gaierror:
            pass
        hosts.append("127.0.0.1")
        return hosts

    # ------------------------------------------------------------------
    # API for the rest of the server (any thread)
    # ------------------------------------------------------------------

    def submit(self, spec):
        """Register the job and launch its manager command. Returns the
        session token (also spec.token) so the caller can persist it."""
        session = _Session(spec)
        with self._lock:
            if spec.job_id in self._sessions:
                raise ValueError("job {} is already registered".format(spec.job_id))
            self._sessions[spec.job_id] = session
        self.sink.on_status(spec.job_id, LAUNCHING)
        self._launch_manager(session)
        return session.token

    def restore(self, spec, last_controller_seq, tasks_done):
        """Re-register a job that was running when the server last stopped
        (section 7.2): it goes straight to awaiting reconnect, and the
        controller -- if it's still out there -- picks up where it left off.
        `tasks_done` is the set of task indices already recorded."""
        session = _Session(spec)
        session.status = AWAITING_RECONNECT
        session.ever_connected = True
        session.last_controller_seq = last_controller_seq
        session.tasks_done = set(tasks_done)
        session.reconnect_deadline = time.monotonic() + self.reconnect_window
        with self._lock:
            self._sessions[spec.job_id] = session
        self.sink.on_status(spec.job_id, AWAITING_RECONNECT)
        self.sink.on_log(spec.job_id, "server restarted; waiting up to {:.0f} s for the controller to reconnect".format(self.reconnect_window))

    def cancel(self, job_id):
        with self._lock:
            session = self._sessions.get(job_id)
            if session is None or session.terminal:
                return False
            session.cancel_requested = True
            session.cancel_deadline = time.monotonic() + CANCEL_GRACE
        return True

    def is_active(self, job_id):
        with self._lock:
            s = self._sessions.get(job_id)
            return s is not None and s.terminal is None

    # ------------------------------------------------------------------
    # launching the controller
    # ------------------------------------------------------------------

    def controller_command_line(self, session):
        # A profile may pin the address the controller dials (cisTEM's
        # gui_address, "Specify" on the Run Profiles panel); otherwise the
        # controller is told every address this machine answers on.
        pinned = (session.spec.profile.get("gui_address") or "").strip()
        hosts = pinned if pinned else ",".join(self.hosts_to_advertise())
        return "{} {} {} {}".format(self.controller_executable_for(session.spec.profile), hosts, self.port,
                                    session.token)

    def controller_executable_for(self, profile):
        """The controller a profile launches: its own `controller_command`
        when it names one (so profiles can point at different cisTEM builds,
        or at one on a remote machine the manager command reaches), else the
        runner's default."""
        return (profile.get("controller_command") or "").strip() or self.controller_executable

    def _launch_manager(self, session):
        command = session.spec.manager_command or "$command"
        command = command.replace("$command", self.controller_command_line(session))
        command = command.replace("$program_name", self.controller_executable_for(session.spec.profile))
        # Log the command with the token blanked -- section 9: tokens never appear in logs.
        self.sink.on_log(session.job_id, "launching controller: " + command.replace(session.token, "<token>"))
        log_path = session.spec.controller_log
        try:
            if log_path is None:
                fd, log_path = tempfile.mkstemp(prefix="controller-{}-".format(session.job_id), suffix=".log")
                os.close(fd)
            else:
                os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            session.manager_log = log_path
            with open(log_path, "ab") as out:
                session.manager = subprocess.Popen(
                    command, shell=True, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                    start_new_session=True,  # a signal to the server must not take the controller with it
                )
        except OSError as exc:
            self._finish(session, FAILED, error="could not launch the controller: {}".format(exc))
            return
        session.launch_deadline = time.monotonic() + self.launch_timeout
        self.sink.on_log(session.job_id, "controller output: " + log_path)

    def _relay_manager_log(self, session, max_lines=40):
        """Copy the tail of the manager's output file into the job log --
        once, when it matters: the process died before connecting (this is
        where 'command not found' and a failing sbatch show up), or the job
        is over. While running, the controller talks to us over the socket."""
        if session.log_relayed or not session.manager_log:
            return
        session.log_relayed = True
        try:
            with open(session.manager_log, "r", errors="replace") as f:
                lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]
        except OSError:
            return
        if len(lines) > max_lines:
            self.sink.on_log(session.job_id, "[controller] ... {} earlier line(s) in {}".format(
                len(lines) - max_lines, session.manager_log))
            lines = lines[-max_lines:]
        for line in lines:
            self.sink.on_log(session.job_id, "[controller] " + line.replace(session.token, "<token>"))

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def _loop(self):
        while not self._stop.is_set():
            try:
                events = self._sel.select(timeout=1.0)
            except OSError:
                break
            for key, mask in events:
                if key.data is None:
                    self._accept()
                    continue
                conn = key.data
                if mask & selectors.EVENT_READ:
                    self._read(conn)
                if mask & selectors.EVENT_WRITE and conn in self._conns:
                    self._flush(conn)
            self._tick()
        for conn in list(self._conns):
            self._close_conn(conn, "server shutting down")
        try:
            self._sel.unregister(self._listener)
        except Exception:  # noqa: BLE001
            pass
        self._listener.close()

    def _accept(self):
        try:
            sock, peer = self._listener.accept()
        except OSError:
            return
        sock.setblocking(False)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        conn = _Conn(sock, peer)
        self._conns.add(conn)
        self._sel.register(sock, selectors.EVENT_READ, data=conn)

    def _read(self, conn):
        try:
            data = conn.sock.recv(65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError as exc:
            self._connection_lost(conn, "read error: {}".format(exc))
            return
        if not data:
            self._connection_lost(conn, "controller closed the connection")
            return
        conn.last_rx = time.monotonic()
        conn.reader.feed(data)
        try:
            while True:
                frame = conn.reader.pop()
                if frame is None:
                    break
                kind, payload = frame
                if kind == jp.KIND_BINARY:
                    continue  # no v1 message uses one; skip per section 3
                if kind != jp.KIND_JSON:
                    raise jp.ProtocolError("unknown frame kind 0x{:02x}".format(kind))
                self._handle(conn, jp.decode_json_payload(payload))
                if conn.closing:
                    break
        except jp.ProtocolError as exc:
            self._protocol_error(conn, exc)

    def _send(self, conn, message):
        conn.outbuf += jp.encode_json_frame(message)
        conn.last_tx = time.monotonic()
        self._flush(conn)

    def _flush(self, conn):
        if conn not in self._conns:
            return
        try:
            while conn.outbuf:
                n = conn.sock.send(conn.outbuf)
                del conn.outbuf[:n]
        except (BlockingIOError, InterruptedError):
            pass
        except OSError as exc:
            self._connection_lost(conn, "write error: {}".format(exc))
            return
        want = selectors.EVENT_READ | (selectors.EVENT_WRITE if conn.outbuf else 0)
        try:
            self._sel.modify(conn.sock, want, data=conn)
        except (KeyError, ValueError):
            pass
        if conn.closing and not conn.outbuf:
            self._close_conn(conn, None)

    def _close_conn(self, conn, why):
        if conn not in self._conns:
            return
        self._conns.discard(conn)
        try:
            self._sel.unregister(conn.sock)
        except (KeyError, ValueError):
            pass
        try:
            conn.sock.close()
        except OSError:
            pass
        session = conn.session
        if session is not None and session.conn is conn:
            session.conn = None
            if why and session.terminal is None:
                self.sink.on_log(session.job_id, why)

    def _close_after_flush(self, conn):
        conn.closing = True
        if not conn.outbuf:
            self._close_conn(conn, None)

    def _connection_lost(self, conn, why):
        session = conn.session
        self._close_conn(conn, None)
        if session is None or session.terminal is not None:
            return
        # Section 7.2: the workers keep running; the controller will be back.
        session.status = AWAITING_RECONNECT
        session.reconnect_deadline = time.monotonic() + self.reconnect_window
        self.sink.on_status(session.job_id, AWAITING_RECONNECT)
        self.sink.on_log(session.job_id, "{}; waiting up to {:.0f} s for it to reconnect".format(why, self.reconnect_window))

    def _protocol_error(self, conn, exc):
        """Section 6.7: send protocol_error, close, and -- because we sent
        it -- mark the job failed."""
        session = conn.session
        seq = session.seq if session else jp.Sequencer()
        try:
            self._send(conn, jp.protocol_error(seq, exc.reason, exc.offending_seq))
        except Exception:  # noqa: BLE001
            pass
        self._close_after_flush(conn)
        if session is not None and session.terminal is None:
            self._finish(session, FAILED, error="protocol error: {}".format(exc.reason))
        elif session is None:
            log.warning("protocol error from %s before hello: %s", conn.peer, exc.reason)

    # ------------------------------------------------------------------
    # message handling
    # ------------------------------------------------------------------

    def _handle(self, conn, message):
        mtype = jp.validate(message)
        if conn.session is None:
            if mtype != "hello":
                raise jp.ProtocolError("first frame must be hello, got {}".format(mtype), message["seq"])
            self._hello(conn, message)
            return

        session = conn.session
        seq = message["seq"]
        if seq <= session.last_controller_seq:
            # A resend after reconnect of something we already processed
            # (section 7.3), or a controller repeating itself. Either way,
            # done already. ping still deserves a pong though -- liveness
            # isn't idempotency.
            if mtype == "ping":
                self._send(conn, jp.pong(session.seq))
            return

        if mtype == "ping":
            self._send(conn, jp.pong(session.seq))
        elif mtype == "pong":
            pass
        elif mtype == "workers":
            self.sink.on_workers(session.job_id, message["connected"], message["expected"])
        elif mtype == "log":
            self.sink.on_log(session.job_id, message["text"], level=message["level"])
        elif mtype == "task_progress":
            self.sink.on_task_progress(session.job_id, message["task"], message.get("ref"),
                                       message["result_number"], message["expected"], message["result"])
        elif mtype == "task_done":
            self._task_done(session, message)
        elif mtype == "job_done":
            self._job_done(session, message)
        elif mtype == "protocol_error":
            self.sink.on_log(session.job_id, "controller reported a protocol error: " + message["reason"], level="error")
            self._close_conn(conn, None)
            self._finish(session, FAILED, error="controller reported a protocol error: " + message["reason"])
            return
        else:
            # Known-to-the-spec-but-not-to-us, or a newer controller's
            # addition: section 4 says ignore.
            log.info("job %s: ignoring message type %r", session.job_id, mtype)

        session.last_controller_seq = seq
        self.sink.on_controller_seq(session.job_id, seq)
        session.unacked += 1
        now = time.monotonic()
        if session.ack_due is None:
            session.ack_due = now + ACK_INTERVAL
        if mtype == "job_done" or session.unacked >= ACK_EVERY_N_FRAMES:
            self._ack(session)

    def _hello(self, conn, message):
        versions = message["protocol_versions"]
        token = message["token"]
        seq = jp.Sequencer()  # reject frames use a throwaway counter; no session yet

        session = self._session_for_token(token)
        if session is None:
            code = "job_finished" if self._token_recently_finished(token) else "bad_token"
            self._send(conn, jp.reject(seq, code))
            self._close_after_flush(conn)
            log.warning("rejected controller from %s: %s", conn.peer, code)
            return
        version = jp.choose_version(versions)
        if version is None:
            self._send(conn, jp.reject(seq, "no_common_version",
                                       "server speaks {}".format(list(jp.SUPPORTED_VERSIONS))))
            self._close_after_flush(conn)
            return
        if session.conn is not None and session.conn in self._conns:
            # Section 6.1 already_connected: the old connection stands until
            # the heartbeat proves it dead; the new one retries later.
            self._send(conn, jp.reject(seq, "already_connected"))
            self._close_after_flush(conn)
            return

        conn.session = session
        session.conn = conn
        resume = session.ever_connected
        session.ever_connected = True
        session.launch_deadline = None
        session.reconnect_deadline = None
        info = message.get("controller") or {}
        self.sink.on_log(session.job_id, "controller connected from {} ({} {} on {}, pid {}){}".format(
            conn.peer[0], info.get("name", "?"), info.get("version", "?"), info.get("host", "?"),
            info.get("pid", "?"), " -- resuming" if resume else ""))

        if resume:
            last_sent = message.get("last_seq_sent")
            if isinstance(last_sent, int) and last_sent < session.last_controller_seq:
                raise jp.ProtocolError(
                    "resume: server has processed seq {} but controller says it only sent {}".format(
                        session.last_controller_seq, last_sent), message["seq"])
            self._send(conn, jp.welcome(session.seq, protocol=version, resume=True,
                                        resume_from_seq=session.last_controller_seq, server=self.server_info))
        else:
            self._send(conn, jp.welcome(session.seq, protocol=version, resume=False, server=self.server_info))
            self._send_package(session)

        if session.status != RUNNING:
            session.status = RUNNING
            self.sink.on_status(session.job_id, RUNNING)
        if session.cancel_requested:
            self._send(conn, jp.cancel(session.seq, "cancelled by user"))

    def _send_package(self, session):
        spec = session.spec
        conn = session.conn
        self._send(conn, jp.package(session.seq, spec.job_info, spec.program, self._wire_profile(spec.profile),
                                    len(spec.tasks), forward_progress=spec.forward_progress))
        # Section 6.2: chunk so a frame stays well under the 64 MiB limit.
        # A few thousand 38-argument tasks per frame is a few MB.
        chunk = 2000
        for start in range(0, len(spec.tasks), chunk):
            self._send(conn, jp.tasks(session.seq, start, spec.tasks[start:start + chunk]))
        self._send(conn, jp.package_end(session.seq))
        self.sink.on_log(session.job_id, "sent {} task{} to the controller".format(
            len(spec.tasks), "" if len(spec.tasks) == 1 else "s"))

    @staticmethod
    def _wire_profile(profile):
        """db.load_run_profiles() shape -> package.profile (section 6.2)."""
        return {
            "name": profile["name"],
            "controller_address": profile.get("controller_address", "") or "",
            "run_commands": [
                {
                    "command": c["command"],
                    "copies": int(c["copies"]),
                    "threads_per_copy": int(c["threads_per_copy"]),
                    "override_total_copies": bool(c["override_total_copies"]),
                    "overridden_total_copies": int(c["overridden_total_copies"]),
                    "delay_ms": int(c["delay_ms"]),
                }
                for c in profile["run_commands"]
            ],
        }

    def _task_done(self, session, message):
        task = message["task"]
        if task in session.tasks_done:
            return  # duplicate, section 6.5
        if not (0 <= task < session.task_count):
            raise jp.ProtocolError("task_done for task {} but the job has {} tasks".format(task, session.task_count),
                                   message["seq"])
        session.tasks_done.add(task)
        self.sink.on_task_done(
            session.job_id, task, message.get("ref"), message["status"], message.get("result"),
            message.get("error"), message.get("cpu_ms"), len(session.tasks_done), session.task_count,
        )

    def _job_done(self, session, message):
        status = message["status"]
        self.sink.on_job_done(session.job_id, status, message["cpu_ms"], message["tasks_ok"],
                              message["tasks_failed"], message.get("error"))
        terminal = {"completed": COMPLETED, "failed": FAILED, "cancelled": CANCELLED}[status]
        # Ack is sent by _handle right after this (job_done forces one); the
        # session then lingers until the controller hangs up or the grace
        # period passes, so that ack actually reaches it.
        session.finish_deadline = time.monotonic() + FINISH_GRACE
        self._finish(session, terminal, error=message.get("error"), keep_conn=True)

    def _ack(self, session):
        if session.conn is None or session.unacked == 0:
            return
        self._send(session.conn, jp.ack(session.seq, session.last_controller_seq))
        session.unacked = 0
        session.ack_due = None

    # ------------------------------------------------------------------
    # finishing
    # ------------------------------------------------------------------

    def _finish(self, session, terminal, error=None, keep_conn=False):
        if session.terminal is not None:
            return
        session.terminal = terminal
        session.status = terminal
        if terminal == FAILED:
            self._relay_manager_log(session)
        with self._lock:
            self._finished_tokens[session.token] = time.monotonic()
        self.sink.on_status(session.job_id, terminal, error=error)
        if not keep_conn and session.conn is not None:
            self._close_after_flush(session.conn)
        if not keep_conn:
            self._reap_manager(session)
            self._forget(session)

    def _forget(self, session):
        with self._lock:
            self._sessions.pop(session.job_id, None)
            if session.manager is not None and session.manager.poll() is None:
                # Still winding down (a controller exits a moment after our
                # last ack). Keep the handle so the tick can reap it.
                self._exiting.append(session.manager)

    def _reap_manager(self, session, force=False):
        proc = session.manager
        if proc is None:
            return
        if proc.poll() is None and force:
            try:
                proc.terminate()
            except OSError:
                pass

    def _reap_exited(self):
        with self._lock:
            self._exiting = [p for p in self._exiting if p.poll() is None]

    # ------------------------------------------------------------------
    # the tick
    # ------------------------------------------------------------------

    def _tick(self):
        now = time.monotonic()

        # Heartbeats, per connection (section 7.1).
        for conn in list(self._conns):
            if now - conn.last_rx > DEAD_AFTER_SILENCE:
                self._connection_lost(conn, "no frame from the controller for {:.0f} s".format(DEAD_AFTER_SILENCE))
            elif conn.session is not None and now - conn.last_tx > PING_AFTER_IDLE:
                self._send(conn, jp.ping(conn.session.seq))

        with self._lock:
            sessions = list(self._sessions.values())

        for session in sessions:
            if session.terminal is not None:
                # Finished; waiting for the controller to hang up after our ack.
                if session.conn is None or now > (session.finish_deadline or 0):
                    if session.conn is not None:
                        self._close_conn(session.conn, None)
                    self._reap_manager(session)
                    self._forget(session)
                continue

            if session.ack_due is not None and now >= session.ack_due:
                self._ack(session)

            if session.status == LAUNCHING:
                proc = session.manager
                if proc is not None and proc.poll() is not None:
                    self._relay_manager_log(session)
                    self._finish(session, FAILED, error="controller exited with code {} before connecting".format(
                        proc.returncode))
                    continue
                if session.launch_deadline is not None and now > session.launch_deadline:
                    self._reap_manager(session, force=True)
                    self._finish(session, FAILED, error="controller did not connect within {:.0f} s".format(
                        self.launch_timeout))
                    continue

            if session.status == AWAITING_RECONNECT and session.reconnect_deadline is not None \
                    and now > session.reconnect_deadline:
                self._reap_manager(session, force=True)
                self._finish(session, FAILED, error="controller did not reconnect within {:.0f} s".format(
                    self.reconnect_window))
                continue

            if session.cancel_requested:
                if session.status == LAUNCHING or session.conn is None:
                    # Nothing to talk to: kill what we started and call it cancelled.
                    self._reap_manager(session, force=True)
                    self._finish(session, CANCELLED)
                    continue
                if session.cancel_deadline is not None and now > session.cancel_deadline:
                    self.sink.on_log(session.job_id, "controller did not wind down after cancel; killing it", level="error")
                    self._reap_manager(session, force=True)
                    self._finish(session, CANCELLED)
                    continue
                if not getattr(session, "_cancel_sent", False):
                    self._send(session.conn, jp.cancel(session.seq, "cancelled by user"))
                    session._cancel_sent = True

        self._reap_exited()

        # Forget tokens of long-finished jobs (they only exist to say
        # `job_finished` instead of `bad_token` to a straggling controller).
        with self._lock:
            stale = [t for t, at in self._finished_tokens.items() if now - at > FINISHED_TOKEN_MEMORY]
            for t in stale:
                del self._finished_tokens[t]

    # ------------------------------------------------------------------
    # token lookup (section 9: constant-time compare, every candidate)
    # ------------------------------------------------------------------

    def _session_for_token(self, token):
        with self._lock:
            found = None
            for session in self._sessions.values():
                # Compare against every live session rather than returning
                # early, so timing doesn't reveal how far down the list a
                # near-miss got.
                if jp.token_matches(token, session.token) and session.terminal is None:
                    found = session
            return found

    def _token_recently_finished(self, token):
        with self._lock:
            return any(jp.token_matches(token, t) for t in self._finished_tokens)
