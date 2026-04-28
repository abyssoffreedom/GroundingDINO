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

---

# Windows 原生 UDP 探测服务器使用说明

本文档说明如何构建、启用、配置和验证 PTR 网络探测使用的 Windows 原生 UDP 服务器。

## 1. 目的

Python UDP 探测服务器会在 `asyncio.DatagramProtocol.datagram_received()` 中记录数据包到达时间。在 Windows 上，Python 可能会按事件循环批次处理排队的 UDP 数据报，这会扭曲包列内部的数据包到达间隔。

`server/winsock_timestamp_udp_server.cpp` 中的原生辅助程序仍然使用 UDP 探测端口 `9999`，但会在 `recvfrom()` 返回后立即使用 `QueryPerformanceCounter` 记录到达时间。

当前时间戳来源：

```text
timestamp_source=app_qpc
```

重要限制：这仍然是应用层时间戳。它比 Python 回调里的时间戳更早、开销更低，但不是网卡或内核时间戳。

## 2. 文件

```text
server/winsock_timestamp_udp_server.cpp
tools/build_winsock_timestamp_udp_server.ps1
server/inference_worker.py
```

生成的二进制文件：

```text
server/winsock_timestamp_udp_server.exe
```

在 Windows 上，如果生成的 `.exe` 存在且 `NETWORK_PROBE_USE_WINSOCK_TIMESTAMP` 没有被禁用，FastAPI 会自动使用它。该辅助程序目前仅支持：

```text
latency
ptr
```

## 3. 环境要求

请使用 Windows 原生环境，不要使用 WSL。

需要：

```text
带有 MSVC C++ 编译器的 Visual Studio Build Tools
用于在 Windows 上运行 GroundingDINO 的 Python 环境
```

从开始菜单打开以下任意一个终端：

```text
x64 Native Tools Command Prompt for VS
x64 Native Tools PowerShell for VS
```

确认 MSVC 可用：

```powershell
cl
```

如果无法识别 `cl`，说明 Visual Studio 构建环境尚未初始化。

## 4. 构建

在 Windows 原生工具终端中运行：

```powershell
cd "C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO"
powershell -ExecutionPolicy Bypass -File .\tools\build_winsock_timestamp_udp_server.ps1
```

预期输出：

```text
Built C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO\server\winsock_timestamp_udp_server.exe
```

确认二进制文件存在：

```powershell
Test-Path .\server\winsock_timestamp_udp_server.exe
```

预期结果：

```text
True
```

## 5. 推荐启动命令

使用带有应用层 QPC 接收时间戳的原生辅助程序：

```powershell
cd "C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO"

$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="1"
$env:NETWORK_PROBE_HIGH_PRIORITY="1"

uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

预期启动日志：

```text
[NetworkProbe] started Windows native UDP probe server: ...
[NetworkProbe][Winsock] listening host=0.0.0.0 port=9999 timestamp_source=app_qpc protocols=latency,ptr high_priority=1
```

如果没有看到 `[NetworkProbe][Winsock]`，说明该辅助程序没有运行。

## 6. 环境变量

```text
NETWORK_PROBE_USE_WINSOCK_TIMESTAMP
默认值：Windows 上默认为 1。
设置为 0 可禁用原生辅助程序，并改用 Python asyncio UDP 服务器。
```

```text
NETWORK_PROBE_HIGH_PRIORITY
默认值：0。
设置为 1 会让辅助进程以 HIGH_PRIORITY_CLASS 运行，并让接收线程使用 THREAD_PRIORITY_HIGHEST。这样可以减少应用层调度停顿，但不能替代内核或网卡时间戳。
```

## 7. 验证日志

PTR 阶段摘要：

```text
[PTR][Winsock][Summary] round=... phase=... payload_bytes=700 received_train=60/60 PTR=...Mbps first_half_rate=...Mbps second_half_rate=...Mbps gap_count=59 gap_mean=...us loss=0.000 missing=0 non_positive_gap=0 reordered=0 timestamp_source=app_qpc
[PTR][Winsock][Detail] round=... phase=... gaps=1->2:...us,2->3:...us,...
```

Swift 客户端也会打印目标包列发送间隔和实际发送完成间隔：

```text
[NetworkProbe] PTR=...Mbps PTR_corrected=...Mbps phase=... completed_phases=... loss=... dst_gap=...ms src_gap=...ms turning_gap=...ms gap_diff_ratio=... receiver_rate=...Mbps receiver_rate_first_half=...Mbps receiver_rate_second_half=...Mbps actual_send_gap_mean=...ms ...
```

关键字段：

```text
PTR
已完成阶段的接收端包列速率。
```

```text
first_half_rate / second_half_rate
用于诊断单个包列内部的突发效应。
```

```text
missing
服务器未观察到的预期包列数据包数量。
```

```text
non_positive_gap
相邻包列序号对中，接收时间戳间隔 <= 0 的数量。
理想值：0。
```

```text
reordered
相邻序号对中，接收时间戳向后移动的数量。
理想值：0。
```

## 8. 禁用与回退

强制使用 Python UDP 服务器：

```powershell
$env:NETWORK_PROBE_USE_WINSOCK_TIMESTAMP="0"
uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
```

也可以删除或重命名：

```text
server/winsock_timestamp_udp_server.exe
```

如果缺少 `.exe`，`server/inference_worker.py` 会自动回退到 `server/udp_echo_server.py`。

## 9. 故障排查

### 没有 `[NetworkProbe][Winsock]` 日志

可能原因：

```text
服务是从 WSL 启动的。
server/winsock_timestamp_udp_server.exe 尚未构建。
NETWORK_PROBE_USE_WINSOCK_TIMESTAMP=0。
启动时使用了错误的 Python/GroundingDINO 目录。
```

### `bind failed`

UDP 端口 `9999` 已被占用。

检查：

```powershell
netstat -ano -p udp | findstr :9999
```

停止现有进程，或重启终端/会话。

### PTR 不稳定

推荐检查：

```text
条件允许时，服务器使用有线以太网而不是 Wi-Fi。
在受控实验期间减少其他网络流量。
检查 missing、non_positive_gap、reordered、first_half_rate 和 second_half_rate。
如果原生辅助程序疑似未更新或没有重新构建，可使用 Python 回退版本对比行为。
```

## 10. 推荐测试流程

1. 构建辅助程序。
2. 使用 `NETWORK_PROBE_HIGH_PRIORITY=1` 启动服务器。
3. 运行 app 至少 5-10 轮探测。
4. 检查 `missing`、`non_positive_gap`、`reordered`、`PTR`、`first_half_rate` 和 `second_half_rate`。
