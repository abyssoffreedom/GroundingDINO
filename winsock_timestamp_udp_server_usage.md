# Windows Winsock Timestamp UDP Server Usage

This document explains how to build, enable, configure, and validate the Windows Winsock timestamp UDP server used by the WBest network probe.

## 1. Purpose

The original Python UDP probe server records packet arrival time inside `asyncio.DatagramProtocol.datagram_received()`. That timestamp is too late for WBest packet-pair measurement because Windows can queue several UDP datagrams before Python processes them. This can compress packet-pair gaps to tens of microseconds and severely overestimate `Ce`.

The Winsock helper in `server/winsock_timestamp_udp_server.cpp` receives the same UDP probe protocol on port `9999`, but it uses:

```text
SIO_TIMESTAMPING + WSARecvMsg + SO_TIMESTAMP
```

When supported by Windows, the returned receive timestamp is generated below the Python application layer. The helper preserves the existing UDP protocol, so the Swift client does not need to change.

Important limitation: this improves the timestamp source, but it does not guarantee accurate `Ce`. Windows, the NIC driver, or Wi-Fi stack can still batch packets or return identical timestamps.

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

FastAPI automatically uses the generated `.exe` on Windows if it exists and `NETWORK_PROBE_USE_WINSOCK_TIMESTAMP` is not disabled.

## 3. Requirements

Use native Windows, not WSL.

Required:

```text
Windows 10 build 20348 or later, or Windows 11
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
cd "C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO"
powershell -ExecutionPolicy Bypass -File .\tools\build_winsock_timestamp_udp_server.ps1
```

Expected output:

```text
Built C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO\server\winsock_timestamp_udp_server.exe
```

Confirm the binary exists:

```powershell
Test-Path .\server\winsock_timestamp_udp_server.exe
```

Expected:

```text
True
```

## 5. Start With Raw Winsock Timestamp Mode

First run without filtering. This tells you whether Winsock timestamping actually improves the packet-pair gaps.

```powershell
cd "C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO"

$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="1"
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="0"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="1"
$env:NETWORK_PROBE_TIMESTAMP_SOURCE="auto"
$env:NETWORK_PROBE_TRAIN_GAP_AGGREGATION="mean"
$env:NETWORK_PROBE_HIGH_PRIORITY="0"

uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

Expected startup log:

```text
[NetworkProbe] started Winsock timestamp UDP server: ...
[WBest][Winsock] SIO_TIMESTAMPING RX enabled.
[WBest][Winsock] listening host=0.0.0.0 port=9999 timestamping_enabled=1 ...
```

If you do not see `[WBest][Winsock]`, the helper is not running.

## 6. Start With Protective Filtering

If raw mode still produces extremely small or zero gaps, use filtering:

```powershell
cd "C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO"

$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="1"
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="50"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="15"
$env:NETWORK_PROBE_TIMESTAMP_SOURCE="auto"
$env:NETWORK_PROBE_TRAIN_GAP_AGGREGATION="mean"
$env:NETWORK_PROBE_HIGH_PRIORITY="0"

uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

Meaning:

```text
NETWORK_PROBE_MIN_PAIR_GAP_US=50
Discard packet pairs whose gap is <= 50 us for Ce median calculation.

NETWORK_PROBE_MIN_VALID_PAIRS=15
Require at least 15 valid packet pairs out of 30. Otherwise packet-pair summary fails.
```

Use raw mode first. Use filtering only after verifying the timestamp quality.

## 7. Environment Variables

```text
NETWORK_PROBE_USE_WINSOCK_TIMESTAMP
Default: 1 on Windows.
Set to 0 to disable the helper and use the Python asyncio UDP server.
```

```text
NETWORK_PROBE_MIN_PAIR_GAP_US
Default: 0.
Minimum positive packet-pair gap used for Ce samples. A value of 50 rejects gaps <= 50 us.
```

```text
NETWORK_PROBE_MIN_VALID_PAIRS
Default: 1.
Minimum number of valid packet-pair samples required for a successful packet-pair summary.
```

```text
NETWORK_PROBE_TIMESTAMP_SOURCE
Default: auto.
Controls which receive timestamp the Winsock helper uses.

auto
Use Winsock SO_TIMESTAMP when it is present and non-zero; otherwise fall back to app-level QPC receive time.

socket
Prefer Winsock SO_TIMESTAMP. A zero SO_TIMESTAMP is still treated as invalid and falls back to app-level QPC receive time.

app
Ignore SO_TIMESTAMP for WBest timing and use the helper's app-level QPC receive time.
This is not kernel timestamping, but it is useful when SO_TIMESTAMP returns identical timestamps for every packet pair.
```

```text
NETWORK_PROBE_TRAIN_GAP_AGGREGATION
Default: mean.
Controls how packet-train receive gaps are aggregated for R.

mean
Use the original arithmetic mean of all positive train gaps.

trimmed_mean
Sort train gaps and remove both tails before averaging. This is useful when app-level receive timestamps contain a few large scheduler outliers and a few tiny socket-buffer drain gaps.

median
Use the median train gap. This is very robust but can overestimate R when many small drain gaps are present.
```

```text
NETWORK_PROBE_TRAIN_GAP_TRIM_RATIO
Default: 0.10.
Used only by trimmed_mean. With 29 train gaps and 0.10, the helper trims the 2 smallest and 2 largest gaps before averaging.
```

```text
NETWORK_PROBE_HIGH_PRIORITY
Default: 0.
Set to 1 to run the Winsock helper process as HIGH_PRIORITY_CLASS and its receive thread as THREAD_PRIORITY_HIGHEST. This reduces app-level timestamp scheduler stalls, but it cannot replace kernel/NIC timestamps.
```

Examples:

```powershell
$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="0"
```

```powershell
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="50"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="15"
```

```powershell
$env:NETWORK_PROBE_TIMESTAMP_SOURCE="app"
```

Recommended app-level timestamp configuration when the NIC does not support kernel timestamps:

```powershell
$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="1"
$env:NETWORK_PROBE_TIMESTAMP_SOURCE="app"
$env:NETWORK_PROBE_TRAIN_GAP_AGGREGATION="trimmed_mean"
$env:NETWORK_PROBE_TRAIN_GAP_TRIM_RATIO="0.10"
$env:NETWORK_PROBE_HIGH_PRIORITY="1"
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="0"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="1"
```

## 8. Validation Logs

After the app performs probing, inspect lines like:

```text
[WBest][Winsock][Summary] round=... payload_bytes=1400 valid_pairs=.../30 Ce=...Mbps gap_lt_50us=... non_positive_gap=... ci_300_1000=... missing=... reordered=... seq_mismatch=... socket_timestamped_pairs=... fallback_timestamp_pairs=... min_pair_gap_us=...
```

Packet train diagnostics are also printed:

```text
[WBest][Winsock][Train][Summary] round=... payload_bytes=1400 received_train=30/30 Ce=...Mbps R=...Mbps aggregation=trimmed_mean trim_ratio=0.10 gap_count=29 gap_min=...us gap_median=...us gap_mean=...us gap_calc=...us gap_p90=...us gap_max=...us timestamp_source=...
[WBest][Winsock][Train][Detail] round=... gaps=1->2:...us,2->3:...us,...
```

The Swift client also prints the target train send gap and measured send completion gap:

```text
[NetworkProbe] ... send_gap=...ms actual_send_gap_mean=...ms actual_send_gap_min=...ms actual_send_gap_median=...ms actual_send_gap_p90=...ms actual_send_gap_max=...ms ...
```

Key fields:

```text
socket_timestamped_pairs
Number of packet pairs where both packets had Winsock SO_TIMESTAMP values.
Good: close to 30.
Bad: 0 or very low.
```

```text
fallback_timestamp_pairs
Number of packet pairs where at least one packet had no Winsock timestamp and the helper used app-level QPC time.
Good: 0.
Bad: high values.
```

```text
non_positive_gap
Number of packet pairs where packet 2 timestamp <= packet 1 timestamp.
Good: 0 or very low.
Bad: large values, because Ce cannot be trusted.
```

```text
gap_lt_50us
Number of positive but suspiciously tiny packet-pair gaps.
Good: low.
Bad: high values, because these can inflate Ce.
```

```text
ci_300_1000
Number of packet-pair capacity samples in 300-1000 Mbps range.
High values usually indicate timestamp compression or an unrealistic Ce estimate.
```

```text
missing, reordered, seq_mismatch
Should be 0. If not, packet matching or packet delivery is broken.
```

```text
Train gap mean vs median
If train gap median is close to the target send gap but train gap mean is much larger, a few large receive gaps are pulling R down.
If client actual_send_gap_mean is already much larger than send_gap, the client pacing path cannot send at the target Ce rate.
If using trimmed_mean, R is calculated from gap_calc, while gap_mean remains the raw arithmetic mean for diagnostics.
```

## 9. Interpreting Results

Good result:

```text
socket_timestamped_pairs close to 30
fallback_timestamp_pairs = 0
non_positive_gap = 0 or very low
gap_lt_50us = 0 or very low
ci_300_1000 = 0 or very low
```

This means Winsock timestamping is usable for WBest on this machine.

Bad result:

```text
socket_timestamped_pairs close to 30
non_positive_gap still high
```

This means Winsock is returning timestamps, but the Windows/NIC/Wi-Fi receive path is still batching packets or assigning identical timestamps. Ce is still not reliable.

If this happens, run one diagnostic pass with:

```powershell
$env:NETWORK_PROBE_TIMESTAMP_SOURCE="app"
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="0"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="1"
```

If UI data returns in `app` mode, the protocol path is working and the problem is specifically the Winsock `SO_TIMESTAMP` values.

Unsupported result:

```text
[WBest][Winsock] SIO_TIMESTAMPING failed error=...
fallback_timestamp_pairs high
```

This means the socket timestamp feature is unavailable or not active for this network path. The helper falls back to C++ app-level QPC timestamps, which are earlier than Python callback timestamps but are still not kernel timestamps.

## 10. Disable And Fallback

To force the original Python UDP server:

```powershell
$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="0"
uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

You can also remove or rename:

```text
server/winsock_timestamp_udp_server.exe
```

If the `.exe` is missing, `server/inference_worker.py` automatically falls back to `server/udp_echo_server.py`.

## 11. Troubleshooting

### No `[WBest][Winsock]` logs

Likely causes:

```text
The service was started from WSL.
server/winsock_timestamp_udp_server.exe was not built.
NETWORK_PROBE_USE_WINSOCK_TIMESTAMP=0.
The wrong Python/GroundingDINO directory was started.
```

### `SIO_TIMESTAMPING failed error=10045`

`10045` is `WSAEOPNOTSUPP`. The Windows network stack, NIC, or driver does not support this timestamping mode for the socket.

Possible mitigations:

```text
Try a wired Ethernet adapter.
Update the NIC driver.
Disable Wi-Fi/VPN/virtual adapters for the test path.
Use protective filtering and fallback logic.
```

### `bind failed`

UDP port `9999` is already in use.

Check:

```powershell
netstat -ano -p udp | findstr :9999
```

Stop the existing process or restart the terminal/session.

### `non_positive_gap` remains high

Winsock timestamps are being returned, but packet-pair timestamps are still not usable.

Recommended actions:

```text
Use server wired Ethernet instead of server Wi-Fi.
Try another NIC/driver.
Enable NETWORK_PROBE_MIN_PAIR_GAP_US=50 and NETWORK_PROBE_MIN_VALID_PAIRS=15.
Avoid using Ce when valid pair count is low.
```

### App bandwidth becomes unavailable or zero

If filtering rejects too many packet pairs, the pair summary fails and the client may treat the bandwidth probe as unavailable.

Use raw mode to diagnose:

```powershell
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="0"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="1"
```

Then decide whether `50 us / 15 pairs` is too strict for the current network.

## 12. Recommended Test Procedure

1. Build the helper.
2. Start server in raw mode:

```powershell
$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="1"
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="0"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="1"
$env:NETWORK_PROBE_TIMESTAMP_SOURCE="auto"
uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

3. Run the app for at least 5-10 probe rounds.
4. Save the server console logs.
5. Check `socket_timestamped_pairs`, `non_positive_gap`, `gap_lt_50us`, and `Ce`.
6. If raw timestamp quality is acceptable, keep raw mode or use a small filter.
7. If raw timestamp quality is bad, test with:

```powershell
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="50"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="15"
```

8. If filtering rejects most rounds, do not trust packet-pair `Ce` on this Windows Wi-Fi path. Use fallback throughput or change the network adapter/path.

## 13. Reference

Microsoft Winsock timestamping:

```text
https://learn.microsoft.com/en-us/windows/win32/winsock/winsock-timestamping
```

The API uses `SIO_TIMESTAMPING` to enable receive timestamping. Timestamps are returned by `WSARecvMsg` in an `SO_TIMESTAMP` control message.
