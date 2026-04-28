# Windows Native UDP Probe Server Usage

This document explains how to build, enable, configure, and validate the Windows native UDP server used by the PTR network probe.

## 1. Purpose

The Python UDP probe server records packet arrival time inside `asyncio.DatagramProtocol.datagram_received()`. On Windows, Python can process queued UDP datagrams in event-loop batches, which can distort packet-train arrival gaps.

The native helper in `server/winsock_timestamp_udp_server.cpp` keeps the UDP probe port at `9999`, but records arrival time immediately after `recvfrom()` returns using `QueryPerformanceCounter`.

Current timestamp source:

```text
timestamp_source=app_qpc
```

Important limitation: this is still an app-level timestamp. It is earlier and lighter than the Python callback timestamp, but it is not a NIC/kernel timestamp.

## 2. Files

```text
server/winsock_timestamp_udp_server.cpp
tools/build_winsock_timestamp_udp_server.ps1
server/inference_worker.py
```

Generated binary:

```text
server/winsock_timestamp_udp_server.exe
```

FastAPI automatically uses the generated `.exe` on Windows if it exists and `NETWORK_PROBE_USE_WINSOCK_TIMESTAMP` is not disabled. The helper now supports only:

```text
latency
ptr
```

## 3. Requirements

Use native Windows, not WSL.

Required:

```text
Visual Studio Build Tools with MSVC C++ compiler
Python environment used to run GroundingDINO on Windows
```

Open one of these from the Start Menu:

```text
x64 Native Tools Command Prompt for VS
x64 Native Tools PowerShell for VS
```

Confirm MSVC is available:

```powershell
cl
```

If `cl` is not recognized, the Visual Studio build environment is not initialized.

## 4. Build

From the Windows native tools terminal:

```powershell
cd "C:\Users\Jan\Downloads\GroundingDINO"
powershell -ExecutionPolicy Bypass -File .\tools\build_winsock_timestamp_udp_server.ps1
```

Expected output:

```text
Built C:\Users\Jan\Downloads\GroundingDINO\server\winsock_timestamp_udp_server.exe
```

Confirm the binary exists:

```powershell
Test-Path .\server\winsock_timestamp_udp_server.exe
```

Expected:

```text
True
```

## 5. Recommended Start Command

Use the native helper with app-level QPC receive timestamps:

```powershell
cd "C:\Users\Jan\Downloads\GroundingDINO"

$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="1"
$env:NETWORK_PROBE_HIGH_PRIORITY="1"

uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

Expected startup log:

```text
[NetworkProbe] started Windows native UDP probe server: ...
[NetworkProbe][Winsock] listening host=0.0.0.0 port=9999 timestamp_source=app_qpc protocols=latency,ptr high_priority=1
```

If you do not see `[NetworkProbe][Winsock]`, the helper is not running.

## 6. Environment Variables

```text
NETWORK_PROBE_USE_WINSOCK_TIMESTAMP
Default: 1 on Windows.
Set to 0 to disable the native helper and use the Python asyncio UDP server.
```

```text
NETWORK_PROBE_HIGH_PRIORITY
Default: 0.
Set to 1 to run the helper process as HIGH_PRIORITY_CLASS and its receive thread as THREAD_PRIORITY_HIGHEST. This reduces app-level scheduler stalls, but it cannot replace kernel/NIC timestamps.
```

## 7. Validation Logs

PTR phase summary:

```text
[PTR][Winsock][Summary] round=... phase=... payload_bytes=700 received_train=60/60 PTR=...Mbps first_half_rate=...Mbps second_half_rate=...Mbps gap_count=59 gap_mean=...us loss=0.000 missing=0 non_positive_gap=0 reordered=0 timestamp_source=app_qpc
[PTR][Winsock][Detail] round=... phase=... gaps=1->2:...us,2->3:...us,...
```

The Swift client also prints the target train send gap and measured send completion gap:

```text
[NetworkProbe] PTR=...Mbps PTR_corrected=...Mbps phase=... completed_phases=... loss=... dst_gap=...ms src_gap=...ms turning_gap=...ms gap_diff_ratio=... receiver_rate=...Mbps receiver_rate_first_half=...Mbps receiver_rate_second_half=...Mbps actual_send_gap_mean=...ms ...
```

Key fields:

```text
PTR
Receiver-side packet-train rate for the completed phase.
```

```text
first_half_rate / second_half_rate
Used to diagnose burst effects inside a train.
```

```text
missing
Number of expected train packets not observed by the server.
```

```text
non_positive_gap
Number of adjacent train sequence pairs whose receive timestamp gap is <= 0.
Good: 0.
```

```text
reordered
Number of adjacent sequence pairs whose receive timestamp moved backwards.
Good: 0.
```

## 8. Disable And Fallback

To force the Python UDP server:

```powershell
$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="0"
uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

You can also remove or rename:

```text
server/winsock_timestamp_udp_server.exe
```

If the `.exe` is missing, `server/inference_worker.py` automatically falls back to `server/udp_echo_server.py`.

## 9. Troubleshooting

### No `[NetworkProbe][Winsock]` logs

Likely causes:

```text
The service was started from WSL.
server/winsock_timestamp_udp_server.exe was not built.
NETWORK_PROBE_USE_WINSOCK_TIMESTAMP=0.
The wrong Python/GroundingDINO directory was started.
```

### `bind failed`

UDP port `9999` is already in use.

Check:

```powershell
netstat -ano -p udp | findstr :9999
```

Stop the existing process or restart the terminal/session.

### PTR is unstable

Recommended checks:

```text
Use server wired Ethernet instead of server Wi-Fi when possible.
Reduce competing traffic during controlled experiments.
Check missing, non_positive_gap, reordered, first_half_rate, and second_half_rate.
Use the Python fallback to compare behavior if the native helper appears stale or was not rebuilt.
```

## 10. Recommended Test Procedure

1. Build the helper.
2. Start the server with `NETWORK_PROBE_HIGH_PRIORITY=1`.
3. Run the app for at least 5-10 probe rounds.
4. Check `missing`, `non_positive_gap`, `reordered`, `PTR`, `first_half_rate`, and `second_half_rate`.
