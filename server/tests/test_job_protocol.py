"""Codec-level tests for job_protocol.py. Stdlib unittest, no fixtures:

    python -m unittest discover -s server/tests
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import job_protocol as jp  # noqa: E402


class FramingTests(unittest.TestCase):
    def test_round_trip_one_frame(self):
        msg = {"type": "ping", "seq": 1}
        frame = jp.encode_json_frame(msg)
        self.assertEqual(frame[:5], struct.pack("!IB", len(frame) - 5, jp.KIND_JSON))
        reader = jp.FrameReader()
        reader.feed(frame)
        kind, payload = reader.pop()
        self.assertEqual(kind, jp.KIND_JSON)
        self.assertEqual(jp.decode_json_payload(payload), msg)
        self.assertIsNone(reader.pop())

    def test_frames_split_and_coalesced_arbitrarily(self):
        frames = b"".join(jp.encode_json_frame({"type": "ping", "seq": i}) for i in range(1, 6))
        reader = jp.FrameReader()
        seen = []
        # feed one byte at a time -- header and payload boundaries fall anywhere
        for b in frames:
            reader.feed(bytes([b]))
            while True:
                got = reader.pop()
                if got is None:
                    break
                seen.append(jp.decode_json_payload(got[1])["seq"])
        self.assertEqual(seen, [1, 2, 3, 4, 5])

    def test_header_is_big_endian(self):
        frame = jp.encode_frame(jp.KIND_BINARY, b"abc")
        self.assertEqual(frame, b"\x00\x00\x00\x03\x02abc")

    def test_oversize_length_rejected_before_bytes_arrive(self):
        reader = jp.FrameReader()
        reader.feed(struct.pack("!IB", jp.MAX_PAYLOAD + 1, jp.KIND_JSON))
        with self.assertRaises(jp.ProtocolError):
            reader.pop()

    def test_encode_refuses_oversize(self):
        with self.assertRaises(jp.ProtocolError):
            jp.encode_frame(jp.KIND_BINARY, b"\0" * (jp.MAX_PAYLOAD + 1))

    def test_payload_must_be_an_object(self):
        with self.assertRaises(jp.ProtocolError):
            jp.decode_json_payload(b"[1,2,3]")
        with self.assertRaises(jp.ProtocolError):
            jp.decode_json_payload(b"{not json")
        with self.assertRaises(jp.ProtocolError):
            jp.decode_json_payload(b"\xff\xfe")

    def test_non_ascii_survives(self):
        msg = {"type": "log", "seq": 3, "level": "info", "text": "1.5 Å — done"}
        payload = jp.FrameReader()
        payload.feed(jp.encode_json_frame(msg))
        self.assertEqual(jp.decode_json_payload(payload.pop()[1])["text"], "1.5 Å — done")


class ValidationTests(unittest.TestCase):
    def test_envelope_required(self):
        with self.assertRaises(jp.ProtocolError):
            jp.validate({"seq": 1})
        with self.assertRaises(jp.ProtocolError):
            jp.validate({"type": "ping"})
        with self.assertRaises(jp.ProtocolError):
            jp.validate({"type": "ping", "seq": 0})
        with self.assertRaises(jp.ProtocolError):
            jp.validate({"type": "ping", "seq": True})

    def test_unknown_type_is_not_an_error(self):
        self.assertEqual(jp.validate({"type": "something_new", "seq": 7, "x": 1}), "something_new")

    def test_unknown_members_ignored(self):
        self.assertEqual(jp.validate({"type": "ping", "seq": 1, "extra": {"a": 1}}), "ping")

    def test_hello(self):
        good = {"type": "hello", "seq": 1, "protocol_versions": [1], "token": "ab" * 16,
                "controller": {"name": "x"}}
        self.assertEqual(jp.validate(good), "hello")
        bad = dict(good, protocol_versions=["1"])
        with self.assertRaises(jp.ProtocolError):
            jp.validate(bad)
        del good["token"]
        with self.assertRaises(jp.ProtocolError):
            jp.validate(good)

    def test_welcome_resume_requires_resume_from_seq(self):
        self.assertEqual(jp.validate({"type": "welcome", "seq": 1, "protocol": 1, "resume": False}), "welcome")
        with self.assertRaises(jp.ProtocolError):
            jp.validate({"type": "welcome", "seq": 1, "protocol": 1, "resume": True})
        self.assertEqual(jp.validate({"type": "welcome", "seq": 1, "protocol": 1, "resume": True,
                                      "resume_from_seq": 42}), "welcome")

    def test_task_done_conditionals(self):
        ok = {"type": "task_done", "seq": 5, "task": 0, "status": "ok",
              "result": {"kind": "floats", "data": [0.0, 1.0]}}
        self.assertEqual(jp.validate(ok), "task_done")
        failed = {"type": "task_done", "seq": 6, "task": 1, "status": "failed"}
        with self.assertRaises(jp.ProtocolError):  # failed needs error
            jp.validate(failed)
        failed["error"] = "boom"
        self.assertEqual(jp.validate(failed), "task_done")
        with self.assertRaises(jp.ProtocolError):
            jp.validate(dict(ok, result="not an object"))
        with self.assertRaises(jp.ProtocolError):
            jp.validate(dict(ok, ref=1.5))
        self.assertEqual(jp.validate(dict(ok, ref=17)), "task_done")
        self.assertEqual(jp.validate(dict(ok, ref="movie-17")), "task_done")

    def test_log_level_enum(self):
        with self.assertRaises(jp.ProtocolError):
            jp.validate({"type": "log", "seq": 1, "level": "warn", "text": "x"})

    def test_job_done(self):
        msg = {"type": "job_done", "seq": 9, "status": "completed", "cpu_ms": 1000,
               "tasks_ok": 2, "tasks_failed": 0}
        self.assertEqual(jp.validate(msg), "job_done")
        with self.assertRaises(jp.ProtocolError):
            jp.validate(dict(msg, status="done"))

    def test_task_entries(self):
        t = {"index": 0, "ref": 17, "args": [
            {"type": "text", "value": "/a.mrc"}, {"type": "float", "value": 1.5},
            {"type": "int", "value": 20}, {"type": "bool", "value": True}]}
        self.assertIs(jp.validate_task(t), t)
        # an integral JSON number is fine for a float arg
        jp.validate_task({"index": 1, "args": [{"type": "float", "value": 2}]})
        with self.assertRaises(jp.ProtocolError):  # but a float is not an int
            jp.validate_task({"index": 1, "args": [{"type": "int", "value": 2.5}]})
        with self.assertRaises(jp.ProtocolError):  # bool is not int
            jp.validate_task({"index": 1, "args": [{"type": "int", "value": True}]})
        with self.assertRaises(jp.ProtocolError):
            jp.validate_task({"index": 1, "args": [{"type": "double", "value": 2.5}]})
        with self.assertRaises(jp.ProtocolError):
            jp.validate_task({"index": 1})

    def test_arg_builder_coerces(self):
        self.assertEqual(jp.arg("float", 2), {"type": "float", "value": 2.0})
        self.assertIsInstance(jp.arg("float", 2)["value"], float)
        self.assertEqual(jp.arg("int", 3.0), {"type": "int", "value": 3})
        self.assertEqual(jp.arg("bool", 1), {"type": "bool", "value": True})
        self.assertEqual(jp.arg("text", 5), {"type": "text", "value": "5"})
        with self.assertRaises(ValueError):
            jp.arg("string", "x")


class BuildersTests(unittest.TestCase):
    def test_sequencer_counts_up_and_survives_reconnect(self):
        seq = jp.Sequencer()
        self.assertEqual(jp.make(seq, "ping")["seq"], 1)
        self.assertEqual(jp.make(seq, "ping")["seq"], 2)
        resumed = jp.Sequencer(start_after=seq.last)
        self.assertEqual(jp.make(resumed, "ping")["seq"], 3)

    def test_every_server_builder_validates(self):
        seq = jp.Sequencer()
        for msg in (
            jp.welcome(seq),
            jp.welcome(seq, resume=True, resume_from_seq=12, server={"name": "t"}),
            jp.reject(seq, "bad_token", "nope"),
            jp.package(seq, {"id": "j"}, {"name": "unblur", "executable": "unblur"},
                       {"name": "p", "controller_address": "", "run_commands": []}, 2),
            jp.tasks(seq, 0, [{"index": 0, "args": []}]),
            jp.package_end(seq),
            jp.ack(seq, 6),
            jp.cancel(seq),
            jp.cancel(seq, "user asked"),
            jp.ping(seq),
            jp.pong(seq),
            jp.protocol_error(seq, "bad", 3),
        ):
            self.assertEqual(jp.validate(msg), msg["type"])
            self.assertIn("t", msg)
            # and it round-trips through a frame unchanged
            r = jp.FrameReader()
            r.feed(jp.encode_json_frame(msg))
            self.assertEqual(jp.decode_json_payload(r.pop()[1]), msg)

    def test_choose_version(self):
        self.assertEqual(jp.choose_version([1]), 1)
        self.assertEqual(jp.choose_version([1, 2, 7], supported=(1, 2)), 2)
        self.assertIsNone(jp.choose_version([3], supported=(1, 2)))


class TokenTests(unittest.TestCase):
    def test_token_shape(self):
        t = jp.new_token()
        self.assertEqual(len(t), 32)
        int(t, 16)  # all hex
        self.assertEqual(t, t.lower())
        self.assertNotEqual(t, jp.new_token())

    def test_token_matches(self):
        t = jp.new_token()
        self.assertTrue(jp.token_matches(t, t))
        self.assertFalse(jp.token_matches(t[:-1] + ("0" if t[-1] != "0" else "1"), t))
        self.assertFalse(jp.token_matches(None, t))
        self.assertFalse(jp.token_matches(123, t))


if __name__ == "__main__":
    unittest.main()
