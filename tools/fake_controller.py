#!/usr/bin/env python3
"""A controller that speaks cisTEM Job Protocol v1 to a job server and fakes
the workers.

Two jobs. First, it is the test double for server/job_runner.py: the runner
can launch it exactly as it would launch the real cistem_job_controller --
same command line, same handshake, same messages -- so the whole server side
is exercisable end to end before the C++ exists. Second, it is the reference
for whoever writes that C++: every message the controller must send or
handle is here, in order, with the spec section noted.

    fake_controller.py <hosts> <port> <token> [options]

<hosts> is comma-separated, tried in order, as in docs/job-protocol.md
section 12. Options make it misbehave on purpose:

    --task-delay S        seconds per task (default 0)
    --fail-task I         report task I as failed (repeatable)
    --drop-after N        after sending N frames, close the socket without a
                          word and reconnect -- exercises resume (section 7)
    --exit-before-hello   connect, then exit without saying hello
    --reconnect-delay S   backoff base between reconnect attempts (default 1)
    --workers N           how many fake workers to "connect" (default: the
                          profile's total copies, capped at the task count)

Exit codes follow section 12.
"""

import argparse
import os
import select
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import job_protocol as jp  # noqa: E402

EXIT_OK, EXIT_OTHER, EXIT_REJECTED, EXIT_RECONNECT_EXPIRED, EXIT_PROTOCOL_ERROR, EXIT_LAUNCH = 0, 1, 2, 3, 4, 5

CONNECT_TIMEOUT = 5.0
PING_AFTER_IDLE = 15.0
DEAD_AFTER_SILENCE = 60.0
JOB_DONE_ACK_WAIT = 30.0


class Rejected(Exception):
    def __init__(self, code, reason):
        super().__init__(code)
        self.code, self.reason = code, reason


class ConnectionDropped(Exception):
    pass


class FakeController:
    def __init__(self, hosts, port, token, task_delay=0.0, fail_tasks=(), drop_after=None,
                 exit_before_hello=False, reconnect_delay=1.0, reconnect_window=600.0, workers=None,
                 task_runner=None, log=None):
        self.hosts = hosts
        self.port = port
        self.token = token
        self.task_delay = task_delay
        self.fail_tasks = set(fail_tasks)
        self.drop_after = drop_after
        self.exit_before_hello = exit_before_hello
        self.reconnect_delay = reconnect_delay
        self.reconnect_window = reconnect_window
        self.workers_override = workers
        self.task_runner = task_runner or self._default_task_runner
        self.log = log or (lambda text: print(text, file=sys.stderr, flush=True))

        # Protocol state that must survive reconnects (section 4, 7.3):
        self.seq = jp.Sequencer()          # our counter, never reset
        self.unacked = []                  # [(seq, frame bytes)] since the last ack
        self.frames_sent = 0
        self.package = None                # the `package` message
        self.tasks = []                    # ordered task objects
        self.next_task = 0
        self.tasks_ok = 0
        self.tasks_failed = 0
        self.cpu_ms = 0
        self.job_done_seq = None
        self.cancelled = False
        self.sock = None
        self.reader = None
        self.last_rx = self.last_tx = 0.0

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    def _connect_once(self):
        for host in self.hosts:
            try:
                s = socket.create_connection((host, self.port), timeout=CONNECT_TIMEOUT)
            except OSError as exc:
                self.log("connect to {}:{} failed: {}".format(host, self.port, exc))
                continue
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(None)
            self.sock = s
            self.reader = jp.FrameReader()
            self.last_rx = self.last_tx = time.monotonic()
            return True
        return False

    def _connect_with_backoff(self):
        deadline = time.monotonic() + self.reconnect_window
        delay = self.reconnect_delay
        while time.monotonic() < deadline:
            if self._connect_once():
                return True
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, 30.0)
        return False

    def _hello(self, resume):
        msg = jp.make(self.seq, "hello", protocol_versions=list(jp.SUPPORTED_VERSIONS), token=self.token,
                      controller={"name": "fake_controller", "version": "0.1", "host": socket.gethostname(),
                                  "pid": os.getpid()})
        if resume:
            msg["last_seq_sent"] = self.seq.last - 1  # everything before this hello
        # hello itself is never resent (a reconnect sends a fresh one), so it
        # goes straight out rather than through the unacked buffer.
        self._write(jp.encode_json_frame(msg))

    # ------------------------------------------------------------------
    # sending
    # ------------------------------------------------------------------

    def _write(self, frame):
        self.sock.sendall(frame)
        self.last_tx = time.monotonic()
        self.frames_sent += 1
        if self.drop_after is not None and self.frames_sent == self.drop_after:
            self.log("--drop-after {}: closing the socket without a word".format(self.drop_after))
            self.drop_after = None
            self.sock.close()
            raise ConnectionDropped()

    def send(self, mtype, **fields):
        """A message that must reach the server: buffered until acked so it
        can be resent after a reconnect (section 7.3)."""
        msg = jp.make(self.seq, mtype, **fields)
        frame = jp.encode_json_frame(msg)
        self.unacked.append((msg["seq"], frame))
        self._write(frame)
        return msg["seq"]

    def _resend_after(self, resume_from_seq):
        pending = [(s, f) for s, f in self.unacked if s > resume_from_seq]
        if pending:
            self.log("resuming: resending {} frame(s) after seq {}".format(len(pending), resume_from_seq))
        for _s, frame in pending:
            self._write(frame)

    # ------------------------------------------------------------------
    # receiving
    # ------------------------------------------------------------------

    def _read_available(self, timeout):
        """Feed whatever arrives within `timeout` to the reader; return the
        decoded messages. Raises ConnectionDropped on EOF."""
        r, _, _ = select.select([self.sock], [], [], timeout)
        if not r:
            return []
        data = self.sock.recv(65536)
        if not data:
            raise ConnectionDropped()
        self.last_rx = time.monotonic()
        self.reader.feed(data)
        out = []
        while True:
            frame = self.reader.pop()
            if frame is None:
                return out
            kind, payload = frame
            if kind != jp.KIND_JSON:
                continue
            msg = jp.decode_json_payload(payload)
            jp.validate(msg)
            out.append(msg)

    def _handle(self, msg):
        t = msg["type"]
        if t == "ack":
            self.unacked = [(s, f) for s, f in self.unacked if s > msg["upto"]]
        elif t == "ping":
            self._write(jp.encode_json_frame(jp.make(self.seq, "pong")))
        elif t == "pong":
            pass
        elif t == "package":
            self.package = msg
            self.tasks = []
        elif t == "tasks":
            if msg["first_index"] != len(self.tasks):
                raise jp.ProtocolError("tasks out of order: expected first_index {}, got {}".format(
                    len(self.tasks), msg["first_index"]), msg["seq"])
            for task in msg["tasks"]:
                jp.validate_task(task)
                self.tasks.append(task)
        elif t == "package_end":
            if self.package is None or len(self.tasks) != self.package["task_count"]:
                raise jp.ProtocolError("package_end with {} of {} tasks".format(
                    len(self.tasks), self.package["task_count"] if self.package else "?"), msg["seq"])
            self.package_complete = True
        elif t == "cancel":
            self.log("server cancelled the job: {}".format(msg.get("reason", "")))
            self.cancelled = True
        elif t == "reject":
            raise Rejected(msg["code"], msg.get("reason"))
        elif t == "protocol_error":
            self.log("server reported a protocol error: " + msg["reason"])
            raise SystemExit(EXIT_PROTOCOL_ERROR)
        elif t == "welcome":
            raise jp.ProtocolError("unexpected welcome mid-session", msg["seq"])
        else:
            self.log("ignoring unknown message type {!r}".format(t))

    def _expect_welcome(self):
        """Return (welcome, following) -- the server sends welcome, package,
        tasks and package_end in one burst, so whatever else came in the
        same read has to be handed on, not dropped."""
        deadline = time.monotonic() + CONNECT_TIMEOUT
        while time.monotonic() < deadline:
            msgs = self._read_available(0.5)
            if not msgs:
                continue
            first = msgs[0]
            if first["type"] == "reject":
                raise Rejected(first["code"], first.get("reason"))
            if first["type"] != "welcome":
                raise jp.ProtocolError("expected welcome, got {}".format(first["type"]), first["seq"])
            return first, msgs[1:]
        raise ConnectionDropped()

    # ------------------------------------------------------------------
    # the job
    # ------------------------------------------------------------------

    @staticmethod
    def _default_task_runner(task):
        """Stand-in for a worker: a legacy-style float array. A real worker
        running unblur would return x shifts then y shifts, in Å."""
        n_frames = 4
        return {"kind": "floats", "data": [0.0] * n_frames + [0.0] * n_frames}, 50

    def _expected_workers(self):
        if self.workers_override is not None:
            return self.workers_override
        total = 0
        for c in self.package["profile"]["run_commands"]:
            total += c["overridden_total_copies"] if c["override_total_copies"] else c["copies"]
        return max(1, min(total, len(self.tasks)))

    def _launch_fake_workers(self):
        n = self._expected_workers()
        self.send("log", level="info", text="Launching {} fake worker{}".format(n, "" if n == 1 else "s"))
        for connected in range(1, n + 1):
            self.send("workers", connected=connected, expected=n)
        self.send("log", level="info", text="All {} processes are connected.".format(n))

    def _run_one_task(self):
        task = self.tasks[self.next_task]
        self.next_task += 1
        if self.task_delay:
            time.sleep(self.task_delay)
        fields = {"task": task["index"]}
        if "ref" in task:
            fields["ref"] = task["ref"]
        if task["index"] in self.fail_tasks:
            self.tasks_failed += 1
            self.send("task_done", status="failed", error="fake failure requested for task {}".format(task["index"]),
                      **fields)
            return
        result, cpu_ms = self.task_runner(task)
        self.tasks_ok += 1
        self.cpu_ms += cpu_ms
        self.send("task_done", status="ok", cpu_ms=cpu_ms, result=result, **fields)

    def _send_job_done(self, status):
        self.job_done_seq = self.send("job_done", status=status, cpu_ms=self.cpu_ms,
                                      tasks_ok=self.tasks_ok, tasks_failed=self.tasks_failed)

    def run(self):
        resume = False
        launched = False
        self.package_complete = False
        while True:
            try:
                if not self._connect_with_backoff():
                    self.log("could not (re)connect within the window")
                    return EXIT_RECONNECT_EXPIRED
                if self.exit_before_hello:
                    self.log("--exit-before-hello: leaving")
                    return EXIT_OTHER
                self._hello(resume)
                welcome, following = self._expect_welcome()
                if welcome["resume"]:
                    self._resend_after(welcome["resume_from_seq"])
                resume = True
                for msg in following:
                    self._handle(msg)

                # Main loop: read what arrived, then advance the job one step.
                while True:
                    for msg in self._read_available(0.05 if self.package_complete else 0.5):
                        self._handle(msg)

                    now = time.monotonic()
                    if now - self.last_rx > DEAD_AFTER_SILENCE:
                        raise ConnectionDropped()
                    if now - self.last_tx > PING_AFTER_IDLE:
                        self._write(jp.encode_json_frame(jp.make(self.seq, "ping")))

                    if self.job_done_seq is not None:
                        if not any(s == self.job_done_seq for s, _ in self.unacked):
                            self.log("job_done acknowledged; exiting")
                            self.sock.close()
                            return EXIT_OK
                        if now - self.job_done_sent_at > JOB_DONE_ACK_WAIT:
                            self.log("no ack for job_done within {} s".format(JOB_DONE_ACK_WAIT))
                            return EXIT_OTHER
                        continue

                    if self.cancelled:
                        self._send_job_done("cancelled")
                        self.job_done_sent_at = time.monotonic()
                        continue
                    if not self.package_complete:
                        continue
                    if not launched:
                        self._launch_fake_workers()
                        launched = True
                    if self.next_task < len(self.tasks):
                        self._run_one_task()
                    else:
                        self._send_job_done("completed" if self.tasks_failed == 0 else "failed")
                        self.job_done_sent_at = time.monotonic()

            except ConnectionDropped:
                self.log("connection lost; reconnecting")
                try:
                    self.sock.close()
                except OSError:
                    pass
                continue
            except Rejected as exc:
                self.log("rejected: {} {}".format(exc.code, exc.reason or ""))
                if exc.code == "already_connected":
                    time.sleep(self.reconnect_delay)
                    continue
                return EXIT_REJECTED
            except jp.ProtocolError as exc:
                self.log("protocol error: {}".format(exc.reason))
                try:
                    self._write(jp.encode_json_frame(jp.make(self.seq, "protocol_error", reason=exc.reason)))
                except Exception:  # noqa: BLE001
                    pass
                return EXIT_PROTOCOL_ERROR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hosts")
    ap.add_argument("port", type=int)
    ap.add_argument("token")
    ap.add_argument("--task-delay", type=float, default=0.0)
    ap.add_argument("--fail-task", type=int, action="append", default=[])
    ap.add_argument("--drop-after", type=int, default=None)
    ap.add_argument("--exit-before-hello", action="store_true")
    ap.add_argument("--reconnect-delay", type=float, default=1.0)
    ap.add_argument("--reconnect-window", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    ctl = FakeController(a.hosts.split(","), a.port, a.token, task_delay=a.task_delay, fail_tasks=a.fail_task,
                         drop_after=a.drop_after, exit_before_hello=a.exit_before_hello,
                         reconnect_delay=a.reconnect_delay, reconnect_window=a.reconnect_window, workers=a.workers)
    return ctl.run()


if __name__ == "__main__":
    sys.exit(main())
