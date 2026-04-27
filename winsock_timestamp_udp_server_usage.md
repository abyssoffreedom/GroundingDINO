# Windows Native UDP Probe Server Usage

This document explains how to build, enable, configure, and validate the Windows native UDP server used by the WBest network probe.

## 1. Purpose

The original Python UDP probe server records packet arrival time inside `asyncio.DatagramProtocol.datagram_received()`. On Windows, Python can process several queued UDP datagrams in one event-loop batch, which can compress packet-pair gaps and overestimate `Ce`.

The native helper in `server/winsock_timestamp_udp_server.cpp` keeps the same UDP probe protocol on port `9999`, but records arrival time immediately after `recvfrom()` returns using `QueryPerformanceCounter`.

The helper records app-level receive timestamps using `QueryPerformanceCounter`.

Current timestamp source:

```text
timestamp_source=app_qpc
```

Important limitation: this is still an app-level timestamp. It is earlier and lighter than the Python callback timestamp, but it is not a NIC/kernel timestamp. Packet-pair `Ce` can still be inflated by Wi-Fi, driver batching, OS scheduling, or token-bucket traffic shaping.

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

## 5. Recommended Start Command

Use the native helper with app-level QPC receive timestamps:

```powershell
cd "C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO"

$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="1"
$env:NETWORK_PROBE_MIN_PAIR_GAP_US="0"
$env:NETWORK_PROBE_MIN_VALID_PAIRS="1"
$env:NETWORK_PROBE_TRAIN_GAP_AGGREGATION="trimmed_mean"
$env:NETWORK_PROBE_TRAIN_GAP_TRIM_RATIO="0.10"
$env:NETWORK_PROBE_HIGH_PRIORITY="1"

uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

Expected startup log:

```text
[NetworkProbe] started Windows native UDP probe server: ...
[WBest][Winsock] listening host=0.0.0.0 port=9999 timestamp_source=app_qpc ...
```

If you do not see `[WBest][Winsock]`, the helper is not running.

## 6. Environment Variables

```text
NETWORK_PROBE_USE_WINSOCK_TIMESTAMP
Default: 1 on Windows.
Set to 0 to disable the native helper and use the Python asyncio UDP server.
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
NETWORK_PROBE_TRAIN_GAP_AGGREGATION
Default: trimmed_mean.
Controls how packet-train receive gaps are aggregated for R.

mean
Use the arithmetic mean of all positive train gaps.

trimmed_mean
Sort train gaps and remove both tails before averaging. This is useful when app-level receive timestamps contain a few large scheduler outliers and a few tiny socket-buffer drain gaps.

median
Use the median train gap. This is very robust but can overestimate R when many small burst gaps are present.
```

```text
NETWORK_PROBE_TRAIN_GAP_TRIM_RATIO
Default: 0.10.
Used only by trimmed_mean. With 29 train gaps and 0.10, the helper trims the 2 smallest and 2 largest gaps before averaging.
```

```text
NETWORK_PROBE_HIGH_PRIORITY
Default: 0.
Set to 1 to run the helper process as HIGH_PRIORITY_CLASS and its receive thread as THREAD_PRIORITY_HIGHEST. This reduces app-level scheduler stalls, but it cannot replace kernel/NIC timestamps.
```

## 7. Validation Logs

Packet-pair summary:

```text
[WBest][Winsock][Summary] round=... payload_bytes=1400 valid_pairs=.../30 Ce=...Mbps gap_lt_50us=... non_positive_gap=... ci_300_1000=... missing=... reordered=... seq_mismatch=... min_pair_gap_us=... timestamp_source=app_qpc
```

Packet-train summary:

```text
[WBest][Winsock][Train][Summary] round=... payload_bytes=1400 received_train=30/30 Ce=...Mbps Ce_raw=...Mbps C_shaper=...Mbps R=...Mbps shaper_detected=... shaper_bic_delta=... aggregation=trimmed_mean trim_ratio=0.10 gap_count=29 gap_min=...us gap_median=...us gap_mean=...us gap_calc=...us gap_p90=...us gap_max=...us timestamp_source=app_qpc
[WBest][Winsock][Train][Detail] round=... gaps=1->2:...us,2->3:...us,...
```

The Swift client also prints the target train send gap and measured send completion gap:

```text
[NetworkProbe] ... send_gap=...ms actual_send_gap_mean=...ms actual_send_gap_min=...ms actual_send_gap_median=...ms actual_send_gap_p90=...ms actual_send_gap_max=...ms ...
```

Key fields:

```text
non_positive_gap
Number of packet pairs where packet 2 timestamp <= packet 1 timestamp.
Good: 0 or very low.
Bad: large values, because Ce cannot be trusted.
```

```text
gap_lt_50us
Number of positive but suspiciously tiny packet-pair gaps.
High values can inflate Ce.
```

```text
ci_300_1000
Number of packet-pair capacity samples in the 300-1000 Mbps range.
High values usually indicate timestamp compression or an unrealistic Ce estimate.
```

```text
missing, reordered, seq_mismatch
Should be 0. If not, packet matching or delivery is broken.
```

```text
gap_mean vs gap_calc
gap_mean is the raw arithmetic mean.
gap_calc is the aggregation actually used to calculate R.
```

```text
Ce_raw / C_shaper / Ce
Ce_raw is the packet-pair burst capacity.
C_shaper is the robust train-derived shaping capacity.
Ce is the capacity used by the WBest A formula after shaper correction.
```

```text
shaper_detected
Set to 1 when the train gap distribution fits a two-component log-normal model better than a single-component model by BIC, and the train-derived C_shaper would otherwise trigger WBest's R < Ce_raw / 2 zero threshold.
```

## 8. Interpreting Results

Normal high-bandwidth path:

```text
Ce and R are close.
threshold_zero=false.
Most train gaps are concentrated, with only a few outliers.
```

Token-bucket or Network Link Conditioner shaping:

```text
Many tiny gaps and many long pause gaps appear in the same train.
R is close to the configured long-term limit.
Ce can be much larger than R because packet pairs land inside bursts.
The helper keeps Ce_raw for diagnostics, estimates C_shaper from the train, and uses Ce=min(Ce_raw, C_shaper) for the WBest A formula.
```

In that shaper case, `Ce_raw` remains a burst-capacity diagnostic while corrected `Ce` is the effective capacity used for `A` and `A_corrected`.

## 9. Disable And Fallback

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

## 10. Troubleshooting

### No `[WBest][Winsock]` logs

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

### `non_positive_gap` remains high

The app-level timestamp path is still not clean enough for packet-pair `Ce`.

Recommended actions:

```text
Use server wired Ethernet instead of server Wi-Fi.
Try another NIC/driver.
Enable NETWORK_PROBE_MIN_PAIR_GAP_US=50 and NETWORK_PROBE_MIN_VALID_PAIRS=15.
Avoid using Ce when valid pair count is low.
Prefer R for UI/adaptation if the train distribution clearly shows token-bucket shaping.
```

### Bandwidth UI becomes zero under limited uplink

If `R < Ce_raw / 2` and `shaper_detected=0`, the WBest available-bandwidth formula still sets `A=0`.

This does not necessarily mean upload throughput is zero. Check `Ce_raw`, `C_shaper`, `R`, and the train gap distribution in the client/server logs.

## 11. Recommended Test Procedure

1. Build the helper.
2. Start the server with `NETWORK_PROBE_TRAIN_GAP_AGGREGATION=trimmed_mean` and `NETWORK_PROBE_HIGH_PRIORITY=1`.
3. Run the app for at least 5-10 probe rounds.
4. Check `non_positive_gap`, `gap_lt_50us`, `ci_300_1000`, `Ce`, `R`, and the train gap distribution.
5. If `Ce` is inflated but `R` tracks the known NLC limit, treat the path as token-bucket shaped and use `R` as the throughput signal.
