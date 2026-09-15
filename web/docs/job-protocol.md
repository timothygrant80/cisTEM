# cisTEM Job Protocol, version 1 — DRAFT

The wire protocol between a **job server** (the cisTEM3 web server today; the
desktop GUI later) and a **job controller** (the per-job process that launches
and supervises the workers running a cisTEM program). It replaces the
`socket_codes.h` / `job_packager.cpp` encoding on this one leg of the system and
changes nothing else: the controller still launches workers through the run
profile's commands, one worker still acts as master and dispatches units of
work, and programs still see `my_current_job.arguments[i]` and
`my_result.SetResult()`.

Status: draft, implemented end to end. Server side: `server/job_protocol.py` (codec) and
`server/job_runner.py` (the server's state machine). Controller side: `cistem_job_controller`
in the cisTEM tree (`src/programs/cistem_job_controller/`), speaking v1 to the server and the
legacy protocol to unmodified workers -- verified running real `unblur` jobs. `tools/fake_controller.py`
is a Python controller used as the server's test double and as the readable reference.

---

## 1. Scope and phasing

```
                 protocol v1 (this document)          legacy protocol (unchanged in phase 1)
  job server  <───────────────────────────>  controller  <──────────────────────────────>  master / workers
  (Python or desktop GUI)                    (C++, per job)                                 (C++ programs)
```

- **Phase 1** — the controller speaks v1 to the server and the existing legacy
  protocol to the workers. `MyApp` and every program are untouched. The
  desktop GUI keeps using the old controller. The legacy encoding survives in
  exactly one program, which already contains it.
- **Phase 2** — workers move to v1 too (the `RunJob`/`JobResult` classes keep
  their in-memory shape; only their serialization changes). Section 10 sketches
  the controller↔worker messages so the same framing and envelope carry over.
- **Phase 3** (independent, per program) — programs declare their parameters
  once and both the prompt sequence and the job arguments are generated from
  the declaration. Section 6.3 reserves the field for it.

Out of scope for v1: transport security (see §9), scheduler integration beyond
what run-profile command templates already do, and any change to how workers
find the controller.

## 2. Design goals

1. **Readable and language-neutral.** JSON frames a Python server, a C++
   controller, a wx desktop GUI and a person with a packet capture can all read.
   Message rates here are tens per second at most; encoding cost is irrelevant.
2. **Explicitly framed and byte-order independent.** No native `struct` dumps,
   no same-architecture requirement, no compile-time framing variants.
3. **Versioned from the first frame,** with a rule for what does and does not
   require a version bump (§8).
4. **Survives a server restart.** The connection may drop without the job
   dying; the controller reconnects and resends what wasn't acknowledged (§7).
5. **Authenticated per job** with a real random token (§9).
6. **Typed, named results** are possible, with a passthrough for legacy float
   arrays so phase 1 needs no program changes (§6.5).
7. **Same topology as today.** Only the controller connects to the server, and
   it initiates the connection — the single inbound edge from the cluster is
   unchanged.

## 3. Transport and framing

TCP. The controller connects to the server; the server never connects out.
Exactly one connection per controller at a time; one controller per job.

Plain TCP is the v1 carrier, chosen deliberately: the server and the compute
live on the same network, wxSocket already gives the C++ side everything the
framing below needs, and encryption when wanted is a tunnel around the
connection (§9) rather than a library inside the controller. §3.1 defines how
a second carrier can be added later without touching a message.

Every frame is:

```
 offset  size  field
 0       4     payload_length   uint32, big-endian (network byte order); length of payload only
 4       1     kind             0x01 = JSON (UTF-8 text, one object)
                                0x02 = binary (raw bytes; meaning given by the immediately
                                       preceding JSON frame — see below)
 5       n     payload
```

- `payload_length` MUST be ≤ **64 MiB** (`67108864`). A receiver that sees a
  larger length MUST send `protocol_error` and close.
- A JSON payload MUST be a single UTF-8 encoded JSON object (RFC 8259). No
  trailing newline is required or forbidden.
- Binary frames exist so v1 never needs a v2 for a bulk payload. **No v1
  message uses one.** If one is introduced, the JSON frame before it carries
  `"binary_follows": true` and describes the bytes; a receiver that does not
  understand that message skips the binary frame that follows it.
- Sockets SHOULD be opened with `TCP_NODELAY`. Application-level keepalive is
  defined in §7.1; TCP keepalive is optional on top.

### 3.1 Carriers

A *carrier* is what delivers frames; the frame and everything above it are
carrier-independent. v1 defines one carrier and reserves a second:

| carrier | address form | frame delivery | status |
|---|---|---|---|
| **TCP** | `host:port` (or the `<hosts> <port>` pair on the command line, §12) | the byte stream is a sequence of frames as laid out above | **required** — every v1 server and controller speaks it |
| WebSocket | `ws://host[:port]/path` or `wss://…` | each WebSocket *binary* message carries exactly one frame, **including its 5-byte header** | reserved — an optional addition under §8, not part of v1 conformance |

The header is kept inside a WebSocket message on purpose, even though the
message boundary makes `payload_length` redundant: the same encode/decode code
then serves both carriers, and the `kind` byte survives. WebSocket's own
ping/pong may be used in place of §7.1's `ping`/`pong` but does not replace the
60 s silence rule.

The controller selects the carrier from the form of the address it is given.
A server that offers the WebSocket carrier does so in addition to TCP, never
instead of it. This is the path to take if the server ever moves behind an
institutional edge that only passes HTTPS; on one network it is not needed.

## 4. Envelope

Every JSON frame is an object with these members:

| member | type   | required | meaning |
|--------|--------|----------|---------|
| `type` | string | yes      | message type (§6) |
| `seq`  | int    | yes      | per-sender counter, starts at 1 on the first frame of a *job*, never resets across reconnects, strictly increasing |
| `t`    | int    | no       | sender's wall clock, milliseconds since the Unix epoch. For logs and ordering across machines; never used for protocol decisions |

Because one connection carries one job, and the job is fixed by the token in
`hello`, no per-frame job identifier is needed.

Receivers MUST ignore unknown members. Receivers MUST ignore unknown `type`
values (logging them is encouraged). A known type with a missing or wrongly
typed required member is a protocol error (§6.7).

## 5. Lifecycle

```
 server                                          controller
 ──────                                          ──────────
 create job row (status: launching)
 token = 32 hex chars from os.urandom(16)
 run manager_command with $command →
   cistem_job_controller <hosts> <port> <token>
                                                 connect to first reachable host:port
                                     ◄── hello   {protocol_versions:[1], token, controller:{...}}
 validate token (constant time)
 welcome ──►                                     {protocol:1, resume:false}
 package ──►                                     job / program / profile / task_count
 tasks ──►  (one or more)                        {first_index, tasks:[...]}
 package_end ──►
 status: running                                 launch run_commands → workers connect (legacy)
                                     ◄── workers {connected:1, expected:4}
                                     ◄── log     {level:"info", text:"..."}
                                     ◄── workers {connected:4, expected:4}
                                     ◄── task_done {task:0, status:"ok", result:{kind:"floats",...}}
 ack ──►  (periodically)                         {upto: <last controller seq received>}
                                     ◄── task_done ...
                                     ◄── job_done {status:"completed", cpu_ms:...}
 ack ──►
 status: completed                               close, exit 0
```

States, as the **server** sees the job:

`launching` → (`hello` accepted) `running` → (`job_done`) `completed` |
`failed` | `cancelled`.

`running` → (connection lost) `running, awaiting reconnect` → (`hello` with
`resume`) `running` — or, after the reconnect window (§7.2), `failed`.

`launching` → (no `hello` within the launch timeout, default 120 s) `failed`.

States, as the **controller** sees itself:

`connecting` → `handshake` → `receiving package` → `launching workers` →
`running` → `finishing` → exit. From any connected state a lost connection
enters `reconnecting` (§7.2) without touching the workers.

### Cancellation

The server sends `cancel`. The controller stops dispatching, terminates its
workers (legacy `socket_time_to_die` in phase 1), then sends
`job_done {status:"cancelled"}` and exits. A controller that is already in
`finishing` ignores `cancel`. If the server wants a job gone and the controller
is unreachable, the server kills the manager process it launched and marks the
job `cancelled` itself — the protocol does not need to cover that path.

## 6. Messages

Fields marked *req* are required. Types are JSON types; `int` means a JSON
number with no fractional part.

### 6.1 Handshake

**`hello`** — controller → server, first frame on every connection.

| field | type | req | |
|---|---|---|---|
| `protocol_versions` | int[] | yes | versions the controller can speak, e.g. `[1]` |
| `token` | string | yes | the job token from the command line |
| `controller` | object | yes | `{"name": "cistem_job_controller", "version": "<cisTEM version string>", "host": "<hostname>", "pid": 1234}` — informational, for the job log |
| `last_seq_sent` | int | no | on reconnect: the highest `seq` this controller has sent so far, so the server can sanity-check its `resume_from_seq` |

**`welcome`** — server → controller, in reply to an accepted `hello`.

| field | type | req | |
|---|---|---|---|
| `protocol` | int | yes | the version chosen: the highest in `protocol_versions` the server also speaks |
| `resume` | bool | yes | `false` on the first connection of a job, `true` on a reconnect |
| `resume_from_seq` | int | when `resume` | highest controller `seq` the server has already received and processed; the controller resends everything above it (§7.3) |
| `server` | object | no | `{"name": "cistem3-server", "version": "..."}` |

**`reject`** — server → controller, in reply to a `hello` it will not accept.
The server closes the connection after sending it. The controller MUST NOT
retry a `reject` whose `code` is `bad_token`, `job_not_found` or `job_finished`;
it terminates any workers it has and exits non-zero.

| field | type | req | |
|---|---|---|---|
| `code` | string | yes | `bad_token` · `job_not_found` · `job_finished` · `no_common_version` · `already_connected` |
| `reason` | string | no | human-readable detail |

`already_connected` means another controller connection for this token is
live. The new connection is rejected; the existing one is untouched. (A
controller that lost its connection and reconnected before the server noticed
the loss will hit this; it MUST wait and retry — the server will detect the
dead connection through §7.1 within the heartbeat window.)

### 6.2 Package

The server sends the package immediately after `welcome` when `resume` is
`false`. On a resume it MUST NOT resend the package.

**`package`** — server → controller.

| field | type | req | |
|---|---|---|---|
| `job` | object | yes | `{"id": "<opaque>", "number": 3, "name": "Job 3", "project": "<opaque>"}` |
| `program` | object | yes | `{"name": "unblur", "executable": "unblur"}` — `executable` is what the run commands launch and may differ from `name` (e.g. `unblur_gpu`) |
| `profile` | object | yes | see below |
| `task_count` | int | yes | total number of tasks that will follow in `tasks` frames |
| `forward_progress` | bool | no | whether the controller should forward the workers' intermediate results as `task_progress` frames (default `true`). A server sets it `false` for programs whose intermediate results are progress ticks it does not need -- `estimate_beamtilt` sends one per search position, 290 880 of them -- and `true` where they carry data (`refine_ctf`'s per-particle defocus). A controller that does not know the field forwards everything, as before. |

`profile` mirrors cisTEM's `RunProfile`, minus anything the controller does not
need:

```json
{
  "name": "Local (multi-threaded)",
  "controller_address": "",
  "run_commands": [
    {
      "command": "$command",
      "copies": 4,
      "threads_per_copy": 1,
      "override_total_copies": false,
      "overridden_total_copies": 0,
      "delay_ms": 0
    }
  ]
}
```

`controller_address` is the address workers use to reach the controller;
empty means the controller advertises its own addresses, as today. Each
`command` is a shell template in which the controller substitutes `$command`
with `<executable> <controller_address> <port> <job_code> <threads>` and
`$program_name` with `program.executable` — identical to
`guix_job_control.cpp` today, so existing run profiles keep working unchanged.

**`tasks`** — server → controller, one or more frames.

| field | type | req | |
|---|---|---|---|
| `first_index` | int | yes | index of `tasks[0]`; tasks are numbered `0 … task_count-1` and MUST arrive in order with no gaps |
| `tasks` | object[] | yes | each `{"index": n, "ref": <opaque>, "args": [...]}` — `ref` is optional, a string or int the server attaches (e.g. the movie asset id) and gets echoed back on `task_done` / `task_progress` so it needn't keep its own index→asset map |

Chunking is the sender's choice; a frame MUST stay under the 64 MiB limit. A
few thousand tasks per frame is a reasonable default.

**`package_end`** — server → controller, after the last `tasks` frame. No
fields. The controller MUST NOT launch workers before receiving it, and MUST
treat a `task_count` mismatch as a protocol error.

### 6.3 Task arguments

`args` is an **ordered array of typed values**, in the order the program's
`DoCalculation()` reads `my_current_job.arguments[i]`:

```json
[
  {"type": "text",  "value": "/data/May08_03.05.02.bin.mrc"},
  {"type": "float", "value": 1.5},
  {"type": "int",   "value": 20},
  {"type": "bool",  "value": true}
]
```

`type` is one of `text` · `int` · `float` · `bool`, the four legacy
`RunArgument` types. The object form is deliberate: JSON has one number type,
and `int` vs `float` matters to the program. In phase 1 the controller maps
these 1:1 onto a legacy `RunJob`.

**Reserved for phase 3:** `params` — an object of named arguments
(`{"input_filename": "...", "pixel_size": 1.5}`) for programs that declare
their parameters. A v1 task carries `args`, `params`, or both; `args` is
required until every program a server targets has moved. A controller that
does not understand `params` ignores it.

### 6.4 Progress and logging — controller → server

**`workers`**

| field | type | req | |
|---|---|---|---|
| `connected` | int | yes | workers that have connected to the controller so far |
| `expected` | int | yes | `min(task_count, total copies across run_commands)` — the same number the GUI's `JobPanel` computes |

Sent whenever `connected` changes.

**`log`**

| field | type | req | |
|---|---|---|---|
| `level` | string | yes | `info` · `error` |
| `text` | string | yes | |
| `task` | int | no | task index, when the message is about one task |

Maps legacy `socket_i_have_info` / `socket_i_have_an_error`. The controller's
own messages ("Launching 4 workers…") use the same channel.

**`task_progress`** — optional partial results from a running task; maps
legacy `socket_program_defined_result`.

| field | type | req | |
|---|---|---|---|
| `task` | int | yes | |
| `ref` | string/int | when the task had one | echoed verbatim from `tasks` |
| `result_number` | int | yes | 1-based index of this partial result |
| `expected` | int | yes | how many partials this task will send |
| `result` | object | yes | same shape as `task_done.result` (§6.5) |

### 6.5 Results — controller → server

**`task_done`** — exactly one per task. Replaces the legacy pair
`socket_job_result` + `socket_job_finished`.

| field | type | req | |
|---|---|---|---|
| `task` | int | yes | task index |
| `ref` | string/int | when the task had one | echoed verbatim from `tasks` |
| `status` | string | yes | `ok` · `failed` |
| `cpu_ms` | int | no | thread time the worker reported for this task |
| `error` | string | when `failed` | |
| `result` | object | when `ok` and the program returns one | see below |

`result` is an object with a required `kind` string; everything else depends
on `kind`:

- **`"floats"`** — the phase-1 passthrough of a legacy `JobResult`:
  `{"kind": "floats", "data": [ ... ]}`. The server interprets `data` the way
  the desktop GUI's `ProcessResult()` does for that program (for unblur: the
  first half is x shifts, the second half y shifts, in Å at the output pixel
  size).
- **Program-specific kinds** are namespaced and versioned, e.g.
  `"unblur.shifts.1"` → `{"kind": "unblur.shifts.1", "unit": "angstrom",
  "x": [...], "y": [...]}`. They are introduced by the program that emits them
  (phase 2 onwards) and documented alongside it. A server that does not know a
  `kind` MUST still record the task as done and keep the raw object.

The server MUST treat a `task_done` for a task it has already recorded as a
duplicate and ignore it (this is what makes resend after reconnect safe).

**`job_done`** — last message of the job. The controller waits for the
server's `ack` covering it (or the ack timeout, §7.3), then closes and exits.

| field | type | req | |
|---|---|---|---|
| `status` | string | yes | `completed` · `failed` · `cancelled` |
| `cpu_ms` | int | yes | total thread time across all workers — what `JobPanel::HandleSocketAllJobsFinished` adds to the project's CPU hours today |
| `tasks_ok` | int | yes | |
| `tasks_failed` | int | yes | |
| `error` | string | no | why the job as a whole failed, when `status` is `failed` |

`completed` means every task reported `ok`. Any `failed` task makes the job
`failed`; the per-task detail is in the `task_done` messages.

### 6.6 Control and flow — either direction

**`ack`** — server → controller.

| field | type | req | |
|---|---|---|---|
| `upto` | int | yes | every controller frame with `seq` ≤ this has been durably processed |

The server MUST send an `ack` at least every 5 seconds while unacknowledged
frames exist, and immediately after processing `job_done`. The controller may
discard its resend buffer up to `upto`. (Named `upto`, not `seq`, so it can
never be confused with the envelope's own counter.)

**`cancel`** — server → controller. `reason` (string) optional. See §5.

**`ping`** / **`pong`** — either direction. A `ping` MUST be answered with a
`pong` promptly. No fields. Used by §7.1.

### 6.7 Errors

**`protocol_error`** — either direction, then close.

| field | type | req | |
|---|---|---|---|
| `reason` | string | yes | |
| `offending_seq` | int | no | the `seq` of the frame that caused it, if known |

Sent for: oversize frame, invalid JSON, a known `type` with a bad required
field, `tasks` out of order, `task_count` mismatch, a first frame that is not
`hello`/`welcome`/`reject`. It is **not** a reconnect trigger: a controller
that receives one exits non-zero; a server that sends one marks the job
`failed`.

## 7. Liveness and reconnection

### 7.1 Heartbeat

If a side has sent nothing for **15 s**, it sends `ping`. If a side has
received nothing (any frame, including `pong`) for **60 s**, it considers the
connection dead and closes it. The server then moves the job to *awaiting
reconnect*; the controller enters *reconnecting*.

### 7.2 Reconnection window

On a lost connection the controller **keeps its workers running** and retries
the connection with exponential backoff: 1, 2, 4, … capped at 30 s between
attempts, cycling through every host it was given, for a total window of
**10 minutes** (configurable on the controller command line). Each attempt is
a fresh `hello` with the same token and `last_seq_sent`.

If the window expires, the controller terminates its workers and exits
non-zero. The server, independently, marks a job *awaiting reconnect* as
`failed` once its own copy of the window expires. Both sides default to the
same value so that in the normal case they agree.

A server that **restarts** MUST NOT fail jobs that were `running` at
shutdown; it MUST put them in *awaiting reconnect* with the window measured
from startup, so a redeploy no longer kills every in-flight job. (This
replaces the current reference server's behaviour of failing them on start.)

### 7.3 Resend

The controller keeps every frame it has sent since the last `ack` in a buffer.
On `welcome {resume: true, resume_from_seq: N}` it resends, in order, every
buffered frame with `seq > N`, then continues normally. It MUST NOT renumber
them. If `N` is greater than `last_seq_sent`, the server has state the
controller doesn't recognise; the controller treats it as a protocol error.

The buffer is bounded only by memory; if it exceeds **64 MiB** the controller
MUST give up (terminate workers, exit non-zero) rather than drop frames
silently. In practice `ack` every 5 s keeps it tiny.

After `job_done` the controller waits up to **30 s** for the `ack` that covers
it. If none arrives and reconnection also fails within that time, it exits
non-zero; the server will notice the job never completed via §7.2.

## 8. Versioning

- The version is negotiated once per connection in `hello`/`welcome`; frames
  do not carry it.
- **Does not bump the version:** adding a message type; adding an optional
  field to any message; adding a `result.kind`; adding a `reject.code`.
  Receivers already ignore what they don't know.
- **Bumps the version:** removing or renaming a message or field; changing a
  field's type or meaning; making an optional field required; changing the
  framing in §3.
- A server SHOULD keep speaking every version any deployed controller might
  send for at least one cisTEM release cycle.

## 9. Security

- **Token:** 32 lowercase hex characters, 128 bits from a cryptographic RNG
  (`os.urandom(16)` / `getrandom`). One token per job. The server compares it
  in constant time, accepts it only for a job in `launching` or *awaiting
  reconnect*, and invalidates it when the job reaches a terminal state.
- The token travels on the controller's command line, as the job code does
  today. That is visible in `ps` on the launching host; it is not a secret
  from other users of that machine. It exists to stop the wrong controller
  attaching to the wrong job and to make blind connections useless, not to
  defend against a local attacker.
- **Tokens MUST NOT appear in logs** on either side. Log the job id instead.
- **Transport encryption is out of scope for v1.** The framing is
  byte-transparent, so wrapping the TCP connection in TLS (`stunnel`, an SSH
  tunnel, or native TLS in a later version) changes nothing above §3. On an
  untrusted network, do that. The server SHOULD bind to a specific interface
  rather than `0.0.0.0` when it can.
- The server executes nothing on the controller's say-so. The only command
  execution in the system is the server running `manager_command` (already
  the case) and the controller running `run_commands` it received in the
  package from the server it authenticated to.

## 10. Phase 2: the same protocol between controller and workers

Not part of v1's conformance requirements; recorded so phase-1 choices don't
paint phase 2 into a corner. Same framing (§3) and envelope (§4).

- Worker → controller: `hello {protocol_versions, job_code, worker:{host,pid,threads}}`.
- Controller → worker: `role {role: "master" | "worker", master_address, master_port}` — replaces `socket_you_are_the_master` / `socket_you_are_a_worker`. A master then receives `package` / `tasks` / `package_end` exactly as the controller did.
- Master → worker: `task {index, args}` — one `RunJob`. Worker → master: `task_done` (§6.5), `log`, `task_progress`. Master → controller: `task_done`, `log`, `job_done`, forwarded unchanged (the master may batch several `task_done` frames, which replaces `socket_job_result_queue`).
- Controller → master/worker: `cancel` — replaces `socket_time_to_die`.
- `socket_result_with_image_to_write` is not carried forward: it was already disabled in cisTEM in favour of writing to the shared filesystem. `socket_template_match_result_ready` becomes a `result.kind` (`"match_template.peaks.1"`).

## 11. Legacy code → v1 message

| `socket_codes.h` | v1 |
|---|---|
| `socket_please_identify` / `socket_sending_identification` / `socket_you_are_connected` | `hello` / `welcome` (the token replaces the job code on this leg) |
| `socket_send_job_details` / `socket_sending_job_package` | `package` + `tasks`… + `package_end`, sent unprompted after `welcome` |
| `socket_number_of_connections` | `workers` |
| `socket_i_have_info` / `socket_i_have_an_error` | `log {level}` |
| `socket_job_result` / `socket_job_result_queue` / `socket_job_finished` | `task_done` (one per task; the queue form is just several frames) |
| `socket_program_defined_result` | `task_progress` |
| `socket_all_jobs_finished` + timing | `job_done {cpu_ms}` |
| `socket_time_to_die` | `cancel` |
| `socket_send_thread_timing` | folded into `task_done.cpu_ms` and `job_done.cpu_ms` |
| `socket_you_are_the_master` / `socket_you_are_a_worker` / `socket_send_next_job` / `socket_ready_to_send_single_job` | controller↔worker only — unchanged in phase 1, see §10 for phase 2 |
| `socket_result_with_image_to_write` | dropped (already disabled upstream) |
| `socket_template_match_result_ready` | a `result.kind` |

## 12. Controller command line

```
cistem_job_controller <hosts> <port> <token> [--reconnect-window SECONDS] [--verbose]
```

`<hosts>` is a comma-separated list of hostnames or IP addresses to try in
order — the server passes every address it is reachable on, as the GUI passes
`all_my_ip_addresses` today. Substituted into `manager_command` via
`$command`; `$program_name` becomes `cistem_job_controller`.

Exit codes: `0` job reached a terminal state and the server acknowledged it ·
`2` rejected by server · `3` reconnect window expired · `4` protocol error ·
`5` failed to launch workers · `1` anything else.

## 13. Worked example

A two-movie Align Movies job. Envelope members `t` omitted for brevity; each
line is one JSON frame.

```jsonc
// controller → server
{"type":"hello","seq":1,"protocol_versions":[1],"token":"9f1c…e2","controller":{"name":"cistem_job_controller","version":"2.0.0-alpha","host":"node07","pid":48213}}

// server → controller
{"type":"welcome","seq":1,"protocol":1,"resume":false,"server":{"name":"cistem3-server","version":"0.1"}}
{"type":"package","seq":2,"job":{"id":"adb6fdf303","number":2,"name":"Job 2","project":"betagal-65d89e"},
 "program":{"name":"unblur","executable":"unblur"},
 "profile":{"name":"Local (multi-threaded)","controller_address":"","run_commands":[{"command":"$command","copies":2,"threads_per_copy":4,"override_total_copies":false,"overridden_total_copies":0,"delay_ms":0}]},
 "task_count":2}
{"type":"tasks","seq":3,"first_index":0,"tasks":[
  {"index":0,"ref":17,"args":[{"type":"text","value":"/data/May08_03.05.02.bin.mrc"},{"type":"text","value":"/proj/Assets/Images/May08_03.05.02.bin_aligned.mrc"},{"type":"float","value":1.5}, /* …35 more, in unblur's DoCalculation order… */ ]},
  {"index":1,"ref":18,"args":[ /* … */ ]}]}
{"type":"package_end","seq":4}

// controller → server
{"type":"log","seq":2,"level":"info","text":"Launching 2 workers"}
{"type":"workers","seq":3,"connected":1,"expected":2}
{"type":"workers","seq":4,"connected":2,"expected":2}
{"type":"log","seq":5,"level":"info","text":"All 2 processes are connected."}
{"type":"task_done","seq":6,"task":1,"ref":18,"status":"ok","cpu_ms":41230,"result":{"kind":"floats","data":[0.0,-0.31,-0.58, /*…x…*/ 0.0,0.12,0.27 /*…y…*/]}}

// server → controller
{"type":"ack","seq":5,"upto":6}

// controller → server
{"type":"task_done","seq":7,"task":0,"ref":17,"status":"ok","cpu_ms":40870,"result":{"kind":"floats","data":[ /*…*/ ]}}
{"type":"job_done","seq":8,"status":"completed","cpu_ms":82100,"tasks_ok":2,"tasks_failed":0}

// server → controller
{"type":"ack","seq":6,"upto":8}
// controller closes, exits 0
```

Note the two independent counters: each side's envelope `seq` counts its own
frames, and `ack.upto` refers to the *controller's* counter.

## 14. Open questions

1. **Reconnect window default.** 10 minutes is a guess. Too short fails jobs
   on a slow redeploy; too long leaves workers burning cluster allocation on a
   server that isn't coming back.
2. **Compression.** A 100k-task package is ~150 MB of JSON. Chunking bounds the
   frame size but not the total. Add `kind = 0x03` (deflate-compressed JSON)
   now, or wait until it hurts? Leaning wait; it's a non-breaking addition.
3. **Where the spec lives long-term.** It is in this repo because the Python
   side is; the C++ controller in the cisTEM tree will be written against it
   too. One copy, referenced from cisTEM by URL/commit, or vendored into both?

### Resolved

- **Raw TCP vs WebSocket** → TCP is the required v1 carrier; WebSocket is
  reserved as an optional second carrier (§3.1). Decided on the basis that
  the server and the compute share a network, so the proxy/TLS advantages of
  WebSocket don't apply, while its cost — a WebSocket and TLS stack in the
  cisTEM build — does.
