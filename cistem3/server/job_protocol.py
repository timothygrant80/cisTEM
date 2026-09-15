"""cisTEM Job Protocol v1 -- framing, message schema and tokens.

The wire format between the job server and a per-job controller, as written
up in docs/job-protocol.md. This module is the codec only: it turns messages
into frames and frames back into validated messages, and knows nothing about
sockets, jobs or the database. job_runner.py is the side that owns
connections and state; keeping the two apart is what makes this half
testable without a peer and reusable by tools/fake_controller.py, which
speaks the controller's end of the same protocol.

Frame layout (docs/job-protocol.md section 3):

    offset  size  field
    0       4     payload_length  uint32, big-endian, payload bytes only
    4       1     kind            0x01 JSON (one UTF-8 object), 0x02 binary
    5       n     payload

Every JSON payload is an object carrying at least `type` and `seq` (section
4). validate() checks the required members for the types this side knows and
leaves the rest alone -- unknown types and unknown members are, by the spec,
to be ignored rather than rejected.
"""

import hmac
import json
import os
import struct
import time

PROTOCOL_VERSION = 1
SUPPORTED_VERSIONS = (1,)

KIND_JSON = 0x01
KIND_BINARY = 0x02

# Section 3: a receiver that sees a larger length MUST send protocol_error
# and close. 64 MiB.
MAX_PAYLOAD = 64 * 1024 * 1024

_HEADER = struct.Struct("!IB")  # big-endian uint32 length, uint8 kind
HEADER_SIZE = _HEADER.size

# Section 6.3: the four RunArgument types, by their wire names.
ARG_TYPES = ("text", "int", "float", "bool")


class ProtocolError(Exception):
    """A frame or message the spec says to reject. The `reason` is what goes
    in the protocol_error message; `offending_seq` is filled in when the
    caller knows it."""

    def __init__(self, reason, offending_seq=None):
        super().__init__(reason)
        self.reason = reason
        self.offending_seq = offending_seq


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def encode_frame(kind, payload):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes")
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError("frame payload of {} bytes exceeds the {} byte limit".format(len(payload), MAX_PAYLOAD))
    return _HEADER.pack(len(payload), kind) + bytes(payload)


def encode_json_frame(message):
    """One JSON object -> one frame. Compact separators: nobody reads the
    wire directly often enough to pay for the whitespace on every frame."""
    if not isinstance(message, dict):
        raise TypeError("a JSON frame carries exactly one object")
    text = json.dumps(message, separators=(",", ":"), ensure_ascii=False)
    return encode_frame(KIND_JSON, text.encode("utf-8"))


def encode_binary_frame(data):
    return encode_frame(KIND_BINARY, data)


class FrameReader:
    """Incremental frame parser. feed() it whatever the socket produced,
    then pop() frames until it returns None. Raises ProtocolError on an
    oversize length so the caller can answer per section 3 and close.

    Deliberately holds at most one frame's worth of buffered bytes beyond
    the header: the reader never allocates for a frame until the header has
    been read and checked, so a hostile or broken peer can't make it reserve
    64 MiB by lying about the length before the bytes arrive -- it just sits
    waiting for bytes that either come (and are bounded) or don't.
    """

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data):
        self._buf += data

    def pop(self):
        """Return (kind, payload) for the next complete frame, or None."""
        if len(self._buf) < HEADER_SIZE:
            return None
        length, kind = _HEADER.unpack_from(self._buf, 0)
        if length > MAX_PAYLOAD:
            raise ProtocolError("frame payload of {} bytes exceeds the {} byte limit".format(length, MAX_PAYLOAD))
        if len(self._buf) < HEADER_SIZE + length:
            return None
        payload = bytes(self._buf[HEADER_SIZE:HEADER_SIZE + length])
        del self._buf[:HEADER_SIZE + length]
        return kind, payload

    def pending_bytes(self):
        return len(self._buf)


def decode_json_payload(payload):
    """Bytes of a JSON frame -> the message object. Any failure -- bad UTF-8,
    bad JSON, a JSON value that isn't an object -- is a ProtocolError."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("JSON frame is not valid UTF-8: {}".format(exc))
    try:
        message = json.loads(text)
    except ValueError as exc:
        raise ProtocolError("JSON frame does not parse: {}".format(exc))
    if not isinstance(message, dict):
        raise ProtocolError("JSON frame must be a single object, got {}".format(type(message).__name__))
    return message


# ---------------------------------------------------------------------------
# Message schema (section 6)
# ---------------------------------------------------------------------------

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_int_list(v):
    return isinstance(v, list) and all(_is_int(x) for x in v)


def _is_str_or_int(v):
    return isinstance(v, str) or _is_int(v)


# type -> {field: predicate}. Only *required* members are listed; optional
# ones are checked by the handler that uses them. Conditionally-required
# members (e.g. welcome.resume_from_seq when resume is true) are checked in
# _CONDITIONAL below.
_REQUIRED = {
    # 6.1 handshake
    "hello": {
        "protocol_versions": _is_int_list,
        "token": lambda v: isinstance(v, str),
        "controller": lambda v: isinstance(v, dict),
    },
    "welcome": {
        "protocol": _is_int,
        "resume": lambda v: isinstance(v, bool),
    },
    "reject": {
        "code": lambda v: isinstance(v, str),
    },
    # 6.2 package
    "package": {
        "job": lambda v: isinstance(v, dict),
        "program": lambda v: isinstance(v, dict),
        "profile": lambda v: isinstance(v, dict),
        "task_count": _is_int,
    },
    "tasks": {
        "first_index": _is_int,
        "tasks": lambda v: isinstance(v, list),
    },
    "package_end": {},
    # 6.4 progress and logging
    "workers": {
        "connected": _is_int,
        "expected": _is_int,
    },
    "log": {
        "level": lambda v: v in ("info", "error"),
        "text": lambda v: isinstance(v, str),
    },
    "task_progress": {
        "task": _is_int,
        "result_number": _is_int,
        "expected": _is_int,
        "result": lambda v: isinstance(v, dict),
    },
    # 6.5 results
    "task_done": {
        "task": _is_int,
        "status": lambda v: v in ("ok", "failed"),
    },
    "job_done": {
        "status": lambda v: v in ("completed", "failed", "cancelled"),
        "cpu_ms": _is_int,
        "tasks_ok": _is_int,
        "tasks_failed": _is_int,
    },
    # 6.6 control and flow
    "ack": {"upto": _is_int},
    "cancel": {},
    "ping": {},
    "pong": {},
    # 6.7
    "protocol_error": {"reason": lambda v: isinstance(v, str)},
}

# (type, field, predicate, condition) -- field is required when condition(msg)
_CONDITIONAL = [
    ("welcome", "resume_from_seq", _is_int, lambda m: m.get("resume") is True),
    ("task_done", "error", lambda v: isinstance(v, str), lambda m: m.get("status") == "failed"),
    ("task_done", "result", lambda v: isinstance(v, dict), lambda m: "result" in m),
    ("task_done", "ref", _is_str_or_int, lambda m: "ref" in m),
    ("task_progress", "ref", _is_str_or_int, lambda m: "ref" in m),
]

KNOWN_TYPES = frozenset(_REQUIRED)


def validate(message):
    """Check the envelope and, for a known type, its required members.

    Returns the message type. For an unknown type the envelope is still
    checked but nothing else is -- the caller is expected to log and ignore
    it, per section 4. Raises ProtocolError otherwise.
    """
    if not isinstance(message, dict):
        raise ProtocolError("message is not an object")
    mtype = message.get("type")
    if not isinstance(mtype, str):
        raise ProtocolError("message has no string `type`")
    seq = message.get("seq")
    if not _is_int(seq) or seq < 1:
        raise ProtocolError("message `seq` must be a positive integer", None)
    if "t" in message and not _is_int(message["t"]):
        raise ProtocolError("message `t` must be an integer", seq)

    required = _REQUIRED.get(mtype)
    if required is None:
        return mtype  # unknown type: ignore, don't reject
    for field, ok in required.items():
        if field not in message:
            raise ProtocolError("{} is missing required field `{}`".format(mtype, field), seq)
        if not ok(message[field]):
            raise ProtocolError("{} field `{}` has the wrong type".format(mtype, field), seq)
    for ctype, field, ok, when in _CONDITIONAL:
        if ctype != mtype or not when(message):
            continue
        if field not in message:
            raise ProtocolError("{} is missing `{}`, required here".format(mtype, field), seq)
        if not ok(message[field]):
            raise ProtocolError("{} field `{}` has the wrong type".format(mtype, field), seq)
    return mtype


def validate_task(task):
    """A `tasks[]` entry: {"index": int, "ref"?: str|int, "args": [typed]}."""
    if not isinstance(task, dict):
        raise ProtocolError("task entry is not an object")
    if not _is_int(task.get("index")):
        raise ProtocolError("task entry has no integer `index`")
    if "ref" in task and not _is_str_or_int(task["ref"]):
        raise ProtocolError("task {} `ref` must be a string or integer".format(task["index"]))
    args = task.get("args")
    if not isinstance(args, list):
        raise ProtocolError("task {} has no `args` array".format(task["index"]))
    for i, a in enumerate(args):
        if not isinstance(a, dict) or a.get("type") not in ARG_TYPES or "value" not in a:
            raise ProtocolError("task {} argument {} is not a typed value".format(task["index"], i))
        t, v = a["type"], a["value"]
        ok = (
            (t == "text" and isinstance(v, str))
            or (t == "int" and _is_int(v))
            or (t == "float" and isinstance(v, (int, float)) and not isinstance(v, bool))
            or (t == "bool" and isinstance(v, bool))
        )
        if not ok:
            raise ProtocolError("task {} argument {} value does not match its type `{}`".format(task["index"], i, t))
    return task


def arg(type_, value):
    """Build one typed argument for a task's `args`. Coerces the Python value
    to the declared type so a float that happens to be integral goes on the
    wire as a float, and an int never silently becomes one."""
    if type_ == "text":
        return {"type": "text", "value": str(value)}
    if type_ == "int":
        return {"type": "int", "value": int(value)}
    if type_ == "float":
        return {"type": "float", "value": float(value)}
    if type_ == "bool":
        return {"type": "bool", "value": bool(value)}
    raise ValueError("unknown argument type {!r}".format(type_))


# ---------------------------------------------------------------------------
# Building outgoing messages
# ---------------------------------------------------------------------------

class Sequencer:
    """Hands out the per-sender `seq` (section 4): starts at 1 on a job's
    first frame, strictly increasing, and -- the important part -- never
    reset on reconnect. One Sequencer per job per side, not per connection."""

    def __init__(self, start_after=0):
        self._last = start_after

    @property
    def last(self):
        return self._last

    def next(self):
        self._last += 1
        return self._last


def now_ms():
    return int(time.time() * 1000)


def make(sequencer, mtype, **fields):
    """Envelope + fields. `t` is always attached; it's cheap and it's what
    makes two machines' logs line up."""
    message = {"type": mtype, "seq": sequencer.next(), "t": now_ms()}
    message.update(fields)
    return message


# Server-side builders, one per message the server sends (sections 6.1-6.7).

def welcome(seq, protocol=PROTOCOL_VERSION, resume=False, resume_from_seq=None, server=None):
    fields = {"protocol": protocol, "resume": resume}
    if resume:
        fields["resume_from_seq"] = int(resume_from_seq)
    if server:
        fields["server"] = server
    return make(seq, "welcome", **fields)


def reject(seq, code, reason=None):
    fields = {"code": code}
    if reason:
        fields["reason"] = reason
    return make(seq, "reject", **fields)


def package(seq, job, program, profile, task_count, forward_progress=True):
    return make(seq, "package", job=job, program=program, profile=profile, task_count=int(task_count),
                forward_progress=bool(forward_progress))


def tasks(seq, first_index, task_list):
    return make(seq, "tasks", first_index=int(first_index), tasks=list(task_list))


def package_end(seq):
    return make(seq, "package_end")


def ack(seq, upto):
    return make(seq, "ack", upto=int(upto))


def cancel(seq, reason=None):
    return make(seq, "cancel", **({"reason": reason} if reason else {}))


def ping(seq):
    return make(seq, "ping")


def pong(seq):
    return make(seq, "pong")


def protocol_error(seq, reason, offending_seq=None):
    fields = {"reason": reason}
    if offending_seq is not None:
        fields["offending_seq"] = int(offending_seq)
    return make(seq, "protocol_error", **fields)


def choose_version(offered, supported=SUPPORTED_VERSIONS):
    """The highest version both sides speak, or None (-> reject
    no_common_version)."""
    common = set(offered) & set(supported)
    return max(common) if common else None


# ---------------------------------------------------------------------------
# Tokens (section 9)
# ---------------------------------------------------------------------------

TOKEN_BYTES = 16  # 128 bits -> 32 lowercase hex characters


def new_token():
    return os.urandom(TOKEN_BYTES).hex()


def token_matches(presented, expected):
    """Constant-time comparison. Both must be str; anything else is False
    rather than an exception, since `presented` came off the wire."""
    if not isinstance(presented, str) or not isinstance(expected, str):
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
