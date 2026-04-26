**目标**
用 Npcap/Wireshark 抓到真正到达 Windows 服务器网卡的 UDP 包时间戳，然后重新计算每个 packet pair 的 `gap` 和 `Ci`，对比 Python 服务端日志，判断 `Ce` 高估是不是 Python `asyncio` 用户态时间戳造成的。

**1. 安装工具**
安装 Wireshark，并确认安装时包含 Npcap 和命令行工具：

```powershell
C:\Program Files\Wireshark\dumpcap.exe
C:\Program Files\Wireshark\tshark.exe
```

如果 Npcap 安装时勾了“Only administrators can use Npcap”，后面的 PowerShell 要用管理员权限打开。

**2. 找到正确网卡**
在 Windows 管理员 PowerShell 里运行：

```powershell
$WIRESHARK = "C:\Program Files\Wireshark"
& "$WIRESHARK\dumpcap.exe" -D
```

你会看到类似：

```text
1. \Device\NPF_{...} (Wi-Fi)
2. \Device\NPF_{...} (Ethernet)
3. \Device\NPF_Loopback (Adapter for loopback traffic capture)
```

如果 Vision Pro/手机/客户端通过 Wi-Fi 访问 Windows 服务器，选 `Wi-Fi` 那个编号。如果客户端和服务端都在同一台 Windows 机器上，选 `Npcap Loopback Adapter`。

**3. 查看支持的时间戳类型**
把 `<interface_id>` 换成上一步的编号：

```powershell
& "$WIRESHARK\dumpcap.exe" -i <interface_id> --list-time-stamp-types
```

优先用：

```text
host_hiprec
```

如果不支持，试：

```text
host_hiprec_unsynced
```

再不行就不指定 timestamp type。

**4. 开始抓包**
服务端当前 WBest UDP 端口是 `9999`。开始抓客户端到服务器方向的包：

```powershell
& "$WIRESHARK\dumpcap.exe" -i <interface_id> --time-stamp-type host_hiprec -f "udp dst port 9999" -s 2000 -B 64 -w "$env:USERPROFILE\Desktop\wbest.pcapng"
```

如果 `host_hiprec` 报错，改成：

```powershell
& "$WIRESHARK\dumpcap.exe" -i <interface_id> --time-stamp-type host_hiprec_unsynced -f "udp dst port 9999" -s 2000 -B 64 -w "$env:USERPROFILE\Desktop\wbest.pcapng"
```

如果还报错，就用：

```powershell
& "$WIRESHARK\dumpcap.exe" -i <interface_id> -f "udp dst port 9999" -s 2000 -B 64 -w "$env:USERPROFILE\Desktop\wbest.pcapng"
```

然后运行你的 app，让它完成几轮 network probing。完成后在 PowerShell 里按 `Ctrl+C` 停止抓包。

**5. 导出 TSV**
把 pcapng 转成脚本可读的 TSV：

```powershell
& "$WIRESHARK\tshark.exe" -r "$env:USERPROFILE\Desktop\wbest.pcapng" -Y "udp.dstport == 9999" -T fields -E header=y -E separator=/t -e frame.number -e frame.time_epoch -e ip.src -e udp.srcport -e ip.dst -e udp.dstport -e udp.length -e udp.payload -e data.data | Set-Content -Encoding utf8 "$env:USERPROFILE\Desktop\wbest_udp.tsv"
```

**6. 用我加的脚本分析**
在 WSL 或你的终端里运行：

```bash
python3 /mnt/c/Users/Jan/Downloads/GroundingDINO/tools/parse_wbest_capture_tsv.py /mnt/c/Users/Jan/Desktop/wbest_udp.tsv --packet-bytes 1500
```

如果你用 Windows Python，也可以：

```powershell
python C:\Users\Jan\Downloads\GroundingDINO\tools\parse_wbest_capture_tsv.py C:\Users\Jan\Desktop\wbest_udp.tsv --packet-bytes 1500
```

输出会类似：

```text
[WBest][Capture][Summary] round=... valid_pairs=30/30 Ce=520.33Mbps gap_lt_50us=27 ci_300_1000=24 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Detail] round=... pair=01 frame=(10,11) seq=(1,2) expected=(1,2) gap=24.10us Ci=497.93Mbps flags=gap_lt_50us,ci_300_1000
```

**判读方式**
如果 Npcap 输出也显示大量 `gap_lt_50us` 和 `Ci=300–1000Mbps`，说明高 `Ce` 不是 Python `asyncio` 独有问题，可能是 Windows/Npcap/NIC 层批量时间戳或真实接收间隔就很短。

如果 Npcap 输出的 gap 明显更大，而服务端日志仍是大量 `20–30us`，就基本确认 Python `datagram_received()` 的 `time.perf_counter_ns()` 太晚，受 socket buffer/event loop 批处理影响，不能用于 WBest packet-pair 时间戳。

如果 `reordered > 0` 或 `seq_mismatch > 0`，说明 packet pair 的顺序或解析有问题，需要先修包序列/分组逻辑。

参考文档：Npcap timestamp 类型说明 https://npcap.com/guide/npcap-api.html ，dumpcap 参数说明 https://www.wireshark.org/docs/man-pages/dumpcap.html ，tshark 字段导出说明 https://www.wireshark.org/docs/man-pages/tshark.html 。