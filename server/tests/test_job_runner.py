"""Integration tests for job_runner.py, driven by tools/fake_controller.py
launched for real through a manager command -- the same path a
cistem_job_controller will take.

    python -m unittest discover -s server/tests
"""

import os
import socket
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
FAKE = os.path.join(HERE, "..", "..", "tools", "fake_controller.py")

import job_protocol as jp  # noqa: E402
import job_runner as jr  # noqa: E402

PROFILE = {
    "name": "test",
    "controller_address": "",
    "run_commands": [{"command": "$command", "copies": 2, "threads_per_copy": 1,
                      "override_total_copies": False, "overridden_total_copies": 0, "delay_ms": 0}],
}


def make_tasks(n):
    return [{"index": i, "ref": 100 + i, "args": [jp.arg("text", "/m{}.mrc".format(i)), jp.arg("float", 1.5),
                                                 jp.arg("int", 20), jp.arg("bool", True)]} for i in range(n)]


class RecordingSink(jr.Sink):
    def __init__(self):
        self.events = []
        self.terminal = threading.Event()
        self.terminal_status = None
        self.error = None
        self._lock = threading.Lock()

    def _rec(self, *e):
        with self._lock:
            self.events.append(e)

    def on_status(self, job_id, status, error=None):
        self._rec("status", status, error)
        if status in (jr.COMPLETED, jr.FAILED, jr.CANCELLED):
            self.terminal_status, self.error = status, error
            self.terminal.set()

    def on_log(self, job_id, text, level="info"):
        self._rec("log", level, text)

    def on_workers(self, job_id, connected, expected):
        self._rec("workers", connected, expected)

    def on_task_done(self, job_id, task, ref, status, result, error, cpu_ms, done_count, task_count):
        self._rec("task_done", task, ref, status, result, error, done_count, task_count)

    def on_job_done(self, job_id, status, cpu_ms, tasks_ok, tasks_failed, error=None):
        self._rec("job_done", status, cpu_ms, tasks_ok, tasks_failed, error)

    def on_controller_seq(self, job_id, seq):
        self._rec("seq", seq)

    def of(self, kind):
        with self._lock:
            return [e for e in self.events if e[0] == kind]


class RunnerHarness:
    def __init__(self, extra_args=""):
        self.sink = RecordingSink()
        self.runner = jr.JobRunner(self.sink, bind_host="127.0.0.1", port=0, advertise_hosts=["127.0.0.1"],
                                   controller_executable="{} {}".format(sys.executable, FAKE),
                                   reconnect_window=20.0, launch_timeout=15.0)
        self.runner.start()
        self.extra_args = extra_args

    def submit(self, n_tasks=3, manager="$command"):
        spec = jr.JobSpec("job-1", {"id": "job-1", "number": 1, "name": "Job 1", "project": "p"},
                          {"name": "unblur", "executable": "unblur"}, PROFILE, make_tasks(n_tasks),
                          manager + (" " + self.extra_args if self.extra_args else ""))
        return self.runner.submit(spec)

    def wait(self, timeout=30):
        self.sink.terminal.wait(timeout)
        return self.sink.terminal_status

    def stop(self):
        self.runner.stop()


class HappyPathTests(unittest.TestCase):
    def test_three_tasks_complete(self):
        h = RunnerHarness()
        try:
            h.submit(3)
            self.assertEqual(h.wait(), jr.COMPLETED, h.sink.events)
            statuses = [e[1] for e in h.sink.of("status")]
            self.assertEqual(statuses, [jr.LAUNCHING, jr.RUNNING, jr.COMPLETED])
            done = h.sink.of("task_done")
            self.assertEqual(sorted(e[1] for e in done), [0, 1, 2])
            self.assertEqual([e[2] for e in sorted(done, key=lambda e: e[1])], [100, 101, 102])  # refs echoed
            self.assertTrue(all(e[3] == "ok" and e[4]["kind"] == "floats" for e in done))
            self.assertEqual([e[6] for e in done], [1, 2, 3])  # progress counts up
            jd = h.sink.of("job_done")[0]
            self.assertEqual(jd[1:5], ("completed", 150, 3, 0))
            self.assertEqual(h.sink.of("workers")[-1][1:], (2, 2))
            self.assertTrue(any("controller connected" in e[2] for e in h.sink.of("log")))
        finally:
            h.stop()

    def test_failed_task_fails_job(self):
        h = RunnerHarness(extra_args="--fail-task 1")
        try:
            h.submit(3)
            self.assertEqual(h.wait(), jr.FAILED)
            failed = [e for e in h.sink.of("task_done") if e[3] == "failed"]
            self.assertEqual([e[1] for e in failed], [1])
            self.assertIn("fake failure", failed[0][5])
            self.assertEqual(h.sink.of("job_done")[0][1:5], ("failed", 100, 2, 1))
        finally:
            h.stop()

    def test_reconnect_resumes_without_duplicating_results(self):
        # Drop the socket after 6 frames (hello, log, 2x workers, log, task_done 0)
        # and come back; the server must end up with each task exactly once.
        h = RunnerHarness(extra_args="--drop-after 6 --reconnect-delay 0.2 --task-delay 0.3")
        try:
            h.submit(4)
            self.assertEqual(h.wait(), jr.COMPLETED, h.sink.events)
            statuses = [e[1] for e in h.sink.of("status")]
            self.assertIn(jr.AWAITING_RECONNECT, statuses)
            self.assertEqual(statuses[-1], jr.COMPLETED)
            done = h.sink.of("task_done")
            self.assertEqual(sorted(e[1] for e in done), [0, 1, 2, 3])
            self.assertTrue(any("resuming" in e[2] for e in h.sink.of("log")))
        finally:
            h.stop()


class FailureTests(unittest.TestCase):
    def test_controller_that_exits_before_hello_fails_fast(self):
        sink = RecordingSink()
        runner = jr.JobRunner(sink, bind_host="127.0.0.1", port=0, advertise_hosts=["127.0.0.1"],
                              controller_executable="false", launch_timeout=60.0)
        runner.start()
        try:
            runner.submit(jr.JobSpec("j", {"id": "j"}, {"name": "x", "executable": "x"}, PROFILE, make_tasks(1),
                                     "$command"))
            t0 = time.monotonic()
            self.assertEqual(sink.terminal.wait(10) and sink.terminal_status, jr.FAILED)
            self.assertLess(time.monotonic() - t0, 5.0)  # far short of launch_timeout
            self.assertIn("exited with code 1 before connecting", sink.error)
        finally:
            runner.stop()

    def test_missing_controller_binary_fails_with_its_output(self):
        sink = RecordingSink()
        runner = jr.JobRunner(sink, bind_host="127.0.0.1", port=0, advertise_hosts=["127.0.0.1"],
                              controller_executable="definitely_not_a_real_binary_xyz")
        runner.start()
        try:
            runner.submit(jr.JobSpec("j", {"id": "j"}, {"name": "x", "executable": "x"}, PROFILE, make_tasks(1),
                                     "$command"))
            sink.terminal.wait(10)
            self.assertEqual(sink.terminal_status, jr.FAILED)
            self.assertTrue(any("[controller]" in e[2] and "not found" in e[2] for e in sink.of("log")), sink.events)
        finally:
            runner.stop()

    def test_bad_token_rejected(self):
        h = RunnerHarness()
        try:
            h.submit(1)  # a real session exists; we present a different token
            s = socket.create_connection(("127.0.0.1", h.runner.port), timeout=5)
            seq = jp.Sequencer()
            s.sendall(jp.encode_json_frame(jp.make(seq, "hello", protocol_versions=[1], token="f" * 32,
                                                   controller={"name": "t"})))
            reader = jp.FrameReader()
            deadline = time.monotonic() + 5
            msg = None
            while msg is None and time.monotonic() < deadline:
                data = s.recv(4096)
                if not data:
                    break
                reader.feed(data)
                f = reader.pop()
                if f:
                    msg = jp.decode_json_payload(f[1])
            self.assertIsNotNone(msg)
            self.assertEqual(msg["type"], "reject")
            self.assertEqual(msg["code"], "bad_token")
            s.close()
            self.assertEqual(h.wait(), jr.COMPLETED)  # the real job was unaffected
        finally:
            h.stop()

    def test_no_common_version_rejected(self):
        h = RunnerHarness()
        try:
            h.submit(1)
            s = socket.create_connection(("127.0.0.1", h.runner.port), timeout=5)
            # steal the real token from the session table -- test-only reach-in
            token = next(iter(h.runner._sessions.values())).token
            seq = jp.Sequencer()
            s.sendall(jp.encode_json_frame(jp.make(seq, "hello", protocol_versions=[99], token=token,
                                                   controller={"name": "t"})))
            reader = jp.FrameReader()
            s.settimeout(5)
            data = s.recv(4096)
            reader.feed(data)
            msg = jp.decode_json_payload(reader.pop()[1])
            self.assertEqual((msg["type"], msg["code"]), ("reject", "no_common_version"))
            s.close()
        finally:
            h.stop()


class CancelTests(unittest.TestCase):
    def test_cancel_running_job(self):
        h = RunnerHarness(extra_args="--task-delay 0.5")
        try:
            h.submit(20)
            # wait until it is running and has produced at least one result
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not h.sink.of("task_done"):
                time.sleep(0.05)
            self.assertTrue(h.runner.cancel("job-1"))
            self.assertEqual(h.wait(), jr.CANCELLED, h.sink.events)
            jd = h.sink.of("job_done")[0]
            self.assertEqual(jd[1], "cancelled")
            self.assertLess(len(h.sink.of("task_done")), 20)
        finally:
            h.stop()

    def test_cancel_before_connect_kills_manager(self):
        sink = RecordingSink()
        # `sleep 30` never dials in; cancel must terminate it rather than wait.
        runner = jr.JobRunner(sink, bind_host="127.0.0.1", port=0, advertise_hosts=["127.0.0.1"],
                              controller_executable="sleep 30 #", launch_timeout=60.0)
        runner.start()
        try:
            runner.submit(jr.JobSpec("j", {"id": "j"}, {"name": "x", "executable": "x"}, PROFILE, make_tasks(1),
                                     "$command"))
            time.sleep(0.5)
            self.assertTrue(runner.cancel("j"))
            sink.terminal.wait(10)
            self.assertEqual(sink.terminal_status, jr.CANCELLED)
        finally:
            runner.stop()


if __name__ == "__main__":
    unittest.main()
