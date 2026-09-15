# Socket Communication Utilities for cisTEM

This directory contains the core infrastructure for socket-based job distribution and result collection in cisTEM. These components enable distributed processing across multiple worker nodes in a cluster environment.

## Architecture Overview

cisTEM uses a hierarchical socket-based architecture for job distribution:

```
┌─────────────┐
│   GUI       │
│ (job_panel) │
└──────┬──────┘
       │ SendJobPackage
       ↓
┌──────────────────┐
│   Controller     │
│ (guix_job_control)
└────┬───────┬─────┘
     │       │
     │       └─────→ SendJobResult/Queue (back to GUI)
     │
     │ SendJobPackage (socket_you_are_the_master)
     ↓
┌──────────────────┐
│   Master Worker  │
│   (myapp.cpp)    │
└────┬─────────────┘
     │
     │ SendJob (individual jobs)
     ↓
┌──────────────────┐
│  Worker Nodes    │
│   (myapp.cpp)    │
└────┬─────────────┘
     │
     └─────→ SendJobResult (back to master)
              ↓
          Master aggregates
              ↓
          SendJobResultQueue (to controller/GUI)
```

## Key Components

### JobPackage
Contains the complete job specification including:
- `RunProfile` - execution parameters (executable_name, gui_address, controller_address, run_commands[])
- `RunJob[]` - array of individual jobs with arguments

### RunJob
Represents a single computational task:
- `job_number` - unique identifier
- `number_of_arguments` - argument count
- `RunArgument[]` - typed arguments (int, float, bool, string)

### JobResult
Contains results from a completed job:
- `job_number` - matches the RunJob
- `result_size` - number of result values
- `result_data[]` - float array of results

### RunProfile
Defines the execution environment:
- `executable_name` - program to run
- `gui_address`, `controller_address` - network addresses
- `run_commands[]` - shell commands to launch workers

## Socket Communication Protocol

### Signal Codes
All socket communications use predefined signal codes before data transfer:
- `socket_sending_job_package` - Sending complete job package
- `socket_you_are_the_master` - Designating master worker
- `socket_ready_to_send_single_job` - Individual job transmission
- `socket_job_result` - Single result transmission
- `socket_job_result_queue` - Batch result transmission
- `socket_job_finished` - Job completion notification

### Buffer Transfer Pattern
All data transfers follow this protocol:
```cpp
// Sender
long transfer_size = ReturnEncodedByteTransferSize();
WriteToSocket(socket, &transfer_size, sizeof(long), ...);
WriteToSocket(socket, transfer_buffer, transfer_size, ...);

// Receiver
long transfer_size;
ReadFromSocket(socket, &transfer_size, sizeof(long), ...);
unsigned char* buffer = new unsigned char[transfer_size];
ReadFromSocket(socket, buffer, transfer_size, ...);
```

**Critical:** Sender and receiver MUST use the same encoding format. Buffer size is always sent as a `long` first, followed by the actual data buffer.

## Data Flow Patterns

### Job Distribution
1. **GUI → Controller:** User submits job via GUI
   - Calls `JobPackage::SendJobPackage(socket)`

2. **Controller → Master Worker:** Controller assigns first worker as master
   - Sends `socket_you_are_the_master` signal
   - Sends complete job package via `JobPackage::SendJobPackage(socket)`

3. **Master → Workers:** Master distributes individual jobs
   - Sends `socket_ready_to_send_single_job` signal
   - Sends job via `RunJob::SendJob(socket)`

### Result Collection
1. **Worker → Master:** Worker completes job
   - Sends `socket_job_result` signal
   - Sends result via `JobResult::SendToSocket(socket)`

2. **Master → Controller:** Master aggregates and forwards
   - Option A: Individual results via `JobResult::SendToSocket(socket)`
   - Option B: Batch results via `SendResultQueueToSocket(socket, array)`

3. **Controller → GUI:** Controller forwards to GUI
   - Same pattern as Master → Controller

## File Descriptions

### job_packager.h / job_packager.cpp
Core data structures and serialization methods:
- `JobPackage` - Complete job specification
- `RunJob` - Individual job with typed arguments
- `RunArgument` - Type-safe job argument container
- `JobResult` - Job result container
- `SendJobPackage()`, `ReceiveJobPackage()` - Job package transfer
- `SendJob()`, `RecieveJob()` - Individual job transfer
- `SendToSocket()`, `ReceiveFromSocket()` - Result transfer
- `SendResultQueueToSocket()`, `ReceiveResultQueueFromSocket()` - Batch transfer

### socket_communicator.h / socket_communicator.cpp
Socket management and monitoring:
- `SocketCommunicator` - Base class for socket-based communication
- `SocketServerThread` - Accepts incoming connections (a std::thread over `TcpServer`)
- `SocketClientMonitorThread` - Monitors active connections (a std::thread polling `TcpSocket`s and posting results to the owner's `EventLoop` with `CallAfter`)
- Virtual handlers for all socket events (must be overridden)

### socket_codes.h
Protocol signal definitions:
- `SOCKET_CODE_SIZE` - Size of signal codes
- Signal code constants for all message types
- `SETUP_SOCKET_CODES` macro for initialization

### run_profile.h / run_profile.cpp
Execution environment specification:
- `RunProfile` - Launch configuration
- `RunCommand` - Shell command specification
- Methods for adding/removing commands
- Command substitution (e.g., `$command`, `$program_name`)

### run_profile_manager.h / run_profile_manager.cpp
Management of multiple run profiles:
- `RunProfileManager` - Collection of RunProfile objects
- Database persistence
- Profile selection and retrieval

## Encoding and Decoding

### Current Implementation (Legacy)
Manual byte-by-byte encoding:
```cpp
// Example: Encoding a std::string
for (counter = 0; counter < str.length(); counter++) {
    transfer_buffer[byte_counter] = str[counter];
    byte_counter++;
}
```

**Character Encoding:** strings are copied byte by byte (`std::string`), no re-encoding.

**Type Descriptors:** Each encoded value includes type information from `cistem::fundamental_type::Enum`.

### Future Enhancement: ByteEncoder/ByteDecoder

A new template-based encoding system is planned (see `/workspaces/cisTEM/.claude/cache/byte_encoder_plan.md`):

**Goals:**
- Type-safe encoding/decoding with templates
- Automatic type deduction
- Self-describing format (header + data + footer)
- Backward compatibility via conditional compilation

**Migration Strategy:**
1. Implement `src/core/byte_encoding.h` (header-only)
2. Guard new code with `#ifdef cisTEM_using_new_byteencoder`
3. Keep existing code in `#else` blocks
4. All nodes in cluster MUST use same encoding (critical!)
5. Gradual migration after thorough testing

**8 Methods Requiring Updates:**
- `JobPackage::SendJobPackage()` / `ReceiveJobPackage()`
- `RunJob::SendJob()` / `RecieveJob()`
- `JobResult::SendToSocket()` / `ReceiveFromSocket()`
- `SendResultQueueToSocket()` / `ReceiveResultQueueFromSocket()`

## Documentation Requirements

### Encoding Order Documentation

**All send/receive method pairs MUST include Doxygen documentation specifying the encoding order.**

#### Pattern: Send Method (Full Specification)
The send method contains the complete encoding specification:
```cpp
/**
 * @brief Encodes and sends a JobPackage over a socket
 *
 * @param socket The socket to send the package to
 * @return true on success, false on failure
 *
 * @note Encoding order:
 * 1. my_profile.executable_name (std::string → text_t)
 * 2. my_profile.gui_address (std::string → text_t)
 * 3. my_profile.number_of_run_commands (long → long_t)
 * 4. For each run_command [i=0..number_of_run_commands-1]:
 *    a. run_commands[i].command_to_run (std::string → text_t)
 *    b. run_commands[i].number_of_copies (int → integer_t)
 * 5. For each job [j=0..number_of_jobs-1]:
 *    - See RunJob::SendJob() for nested encoding
 *
 * @see ReceiveJobPackage() for decoder counterpart
 * @see RunJob::SendJob() for nested job encoding specification
 */
bool JobPackage::SendJobPackage(TcpSocket* socket);
```

#### Pattern: Receive Method (Reference Only)
The receive method simply references the send method:
```cpp
/**
 * @brief Receives and decodes a JobPackage from a socket
 *
 * @param socket The socket to receive from
 * @return true on success, false on failure
 *
 * @see SendJobPackage() for encoding order specification
 */
bool JobPackage::ReceiveJobPackage(TcpSocket* socket);
```

### Documentation Principles

1. **Single source of truth**: Send method contains complete encoding spec
2. **No per-line comments**: Implementation matches docstring, no redundant comments
3. **Type annotations**: Use format `C++ type → fundamental_type::Enum`
4. **Loop bounds**: Clearly specify iteration ranges `[i=0..count-1]`
5. **Nested structures**: Reference other methods for sub-encodings
6. **Cross-references**: Use `@see` to link encoder/decoder pairs

### Required Method Pairs

**Job Distribution:**
- `JobPackage::SendJobPackage()` ↔ `ReceiveJobPackage()`
- `RunJob::SendJob()` ↔ `RecieveJob()`

**Result Collection:**
- `JobResult::SendToSocket()` ↔ `ReceiveFromSocket()`
- `SendResultQueueToSocket()` ↔ `ReceiveResultQueueFromSocket()`

### Benefits

- **Compile-time contract**: Order and types explicitly documented
- **Easy verification**: Read doc, check implementation matches
- **Version control**: Changes to encoding visible in code review
- **Maintainability**: Future developers understand encoding format
- **No drift**: Documentation lives with the code it describes

## Best Practices

### Socket Communication
- **Always send signal codes first** before any data
- **Always send buffer size** as `long` before buffer data
- **Never block the main thread** - use monitor threads
- **Handle disconnections gracefully** - workers may fail
- **Validate job codes** - prevent cross-job contamination

### Error Handling
- Use `SendError(std::string)` to propagate errors to the controller
- Use `SendInfo(std::string)` for status updates
- Clean up sockets on disconnection
- Shut down gracefully on fatal errors

### Thread Safety
- Socket monitoring runs in separate threads
- Use mutexes for shared data access
- Never read from sockets outside monitor thread
- Writing to sockets is thread-safe with proper synchronization

### Cluster Deployment
- **All nodes must use same binary** (same encoding)
- **All nodes must have same endianness** (assumed, not checked)
- Configure firewall to allow socket connections
- Use consistent naming for executable paths
- Test single-node before multi-node deployment

## Testing

### Single-Node Testing
Test complete workflow on one machine:
```bash
# Terminal 1: Launch GUI
./cisTEM

# Terminal 2: Monitor controller
ps aux | grep cisTEM_job_control

# Verify: Jobs run, results return, no crashes
```

### Multi-Node Testing
Test on actual cluster:
1. Ensure all nodes have same cisTEM build
2. Configure run profile with correct addresses
3. Start with small job package (2-3 jobs)
4. Monitor all nodes for errors
5. Verify result correctness and completeness

### Common Issues
- **Connection refused:** Check firewall, verify addresses
- **Mismatched encoding:** All nodes must be same version
- **Hanging jobs:** Check worker logs, verify executable exists
- **Incomplete results:** Check for worker crashes, network issues

## Integration Points

### GUI Integration
- `gui/job_panel.cpp` - Creates and sends JobPackage
- `gui/MyRunProfilesPanel.cpp` - Manages run profiles
- Result handlers update database and display

### Program Integration
- `core/myapp.cpp` - Base class for all worker programs
- Programs inherit job handling infrastructure
- Automatic socket setup and monitoring

### Database Integration
- Run profiles stored in project database
- GUI updates database with results
- Programs do NOT access database directly

## Future Enhancements

### Planned Improvements
1. **ByteEncoder/ByteDecoder** - Modern type-safe encoding
2. **Protocol versioning** - Handle mixed version clusters
3. **Encryption** - Secure socket communication
4. **Compression** - Reduce network bandwidth
5. **Checksum validation** - Detect corruption
6. **Job priorities** - Weighted scheduling
7. **Fault tolerance** - Automatic retry on failure

### Backward Compatibility
All enhancements must maintain compatibility:
- Old GUI should work with new workers (within reason)
- Graceful degradation when features unavailable
- Clear error messages for version mismatches

## References

- Socket protocol details: `socket_codes.h`
- Encoding format details: `.claude/cache/byte_encoder_impact_map.md`
- ByteEncoder design: `.claude/cache/byte_encoder_plan.md`
- Core library guide: `src/core/CLAUDE.md`
- GUI integration: `src/gui/CLAUDE.md`
