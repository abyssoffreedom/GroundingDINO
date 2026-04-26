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

**5. 分析 pcapng**
推荐直接分析 pcapng。当前 WBest 的 `--packet-bytes 1500` 会让 UDP datagram 超过普通 Ethernet MTU，从而发生 IP fragmentation；直接读 pcapng 可以从 UDP 首片里解析 WBest header，不需要重新导出 TSV。

在 WSL 或你的终端里运行：

```bash
python3 tools/parse_wbest_capture_tsv.py /mnt/c/Users/EmVis/Desktop/wbest.pcapng --packet-bytes 1500
```

只看每轮摘要：

```bash
python3 tools/parse_wbest_capture_tsv.py /mnt/c/Users/EmVis/Desktop/wbest.pcapng --packet-bytes 1500 --summary-only
```

如果你用 Windows Python，也可以：

```powershell
python "C:\Users\EmVis\Shuyang's minor thesis\GroundingDINO\tools\parse_wbest_capture_tsv.py" "$env:USERPROFILE\Desktop\wbest.pcapng" --packet-bytes 1500
```

**6. 可选：导出 TSV**
只有 packet-pair 没有发生 IP fragmentation 时，TSV 导出才适合直接分析。比如把 probe payload 调小到不超过 1472 bytes 后，可以把 pcapng 转成 TSV：

```powershell
& "$WIRESHARK\tshark.exe" -r "$env:USERPROFILE\Desktop\wbest.pcapng" -Y "udp.dstport == 9999" -T fields -E header=y -E separator=/t -e frame.number -e frame.time_epoch -e ip.src -e udp.srcport -e ip.dst -e udp.dstport -e udp.length -e udp.payload -e data.data | Set-Content -Encoding utf8 "$env:USERPROFILE\Desktop\wbest_udp.tsv"
```

然后运行：

```bash
python3 tools/parse_wbest_capture_tsv.py /mnt/c/Users/EmVis/Desktop/wbest_udp.tsv --packet-bytes 1500
```

输出会类似：

```text
[WBest][Capture][Summary] round=... valid_pairs=30/30 Ce=520.33Mbps gap_lt_50us=27 non_positive_gap=0 ci_300_1000=24 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Detail] round=... pair=01 frame=(10,11) seq=(1,2) expected=(1,2) gap=24.10us Ci=497.93Mbps flags=gap_lt_50us,ci_300_1000
```

**判读方式**
如果 Npcap 输出也显示大量 `gap_lt_50us` 和 `Ci=300–1000Mbps`，说明高 `Ce` 不是 Python `asyncio` 独有问题，可能是 Windows/Npcap/NIC 层批量时间戳或真实接收间隔就很短。

如果 Npcap 输出的 gap 明显更大，而服务端日志仍是大量 `20–30us`，就基本确认 Python `datagram_received()` 的 `time.perf_counter_ns()` 太晚，受 socket buffer/event loop 批处理影响，不能用于 WBest packet-pair 时间戳。

如果 `reordered > 0` 或 `seq_mismatch > 0`，说明 packet pair 的顺序或解析有问题，需要先修包序列/分组逻辑。

参考文档：Npcap timestamp 类型说明 https://npcap.com/guide/npcap-api.html ，dumpcap 参数说明 https://www.wireshark.org/docs/man-pages/dumpcap.html ，tshark 字段导出说明 https://www.wireshark.org/docs/man-pages/tshark.html 。





[WBest][Capture][Summary] round=02cda593-f21b-4cdf-b02e-245e3ce3bd96 valid_pairs=9/30 Ce=8.02Mbps gap_lt_50us=0 non_positive_gap=21 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=0b563902-7a3d-49e9-ac24-6f6bb12aa5af valid_pairs=7/30 Ce=10.43Mbps gap_lt_50us=0 non_positive_gap=23 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=11246406-1e97-44e0-881a-a598d71ee9d4 valid_pairs=5/30 Ce=8.27Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=1788f891-7478-49b5-b74f-5a4815a7934c valid_pairs=5/30 Ce=10.06Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=1ce7459c-3dbe-42d0-a5c9-3ad7a708ba12 valid_pairs=6/30 Ce=7.36Mbps gap_lt_50us=0 non_positive_gap=24 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=217cb591-7b93-4b4d-8cd8-bc9db9f8e287 valid_pairs=10/30 Ce=4.89Mbps gap_lt_50us=0 non_positive_gap=20 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=248051fb-badd-4b79-96f9-3555c6bad99f valid_pairs=3/30 Ce=8.97Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=29c2b229-df7c-4607-93bc-cfbd52998d3c valid_pairs=5/30 Ce=10.34Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=3c402e92-1525-41ea-b6ef-f42fb19e99a5 valid_pairs=1/30 Ce=9.92Mbps gap_lt_50us=0 non_positive_gap=29 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=3ce5f5fb-9818-4d66-8201-c82a4be0ec94 valid_pairs=6/30 Ce=7.47Mbps gap_lt_50us=0 non_positive_gap=24 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=3ed7982c-c05e-4bd4-ac5c-59dbc726220c valid_pairs=3/30 Ce=8.05Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=4bb9b22b-56d7-415a-bf29-6e0def30de07 valid_pairs=8/30 Ce=10.06Mbps gap_lt_50us=0 non_positive_gap=22 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=507daf8a-0e75-4fde-bbe2-e5db9da669d4 valid_pairs=9/30 Ce=10.15Mbps gap_lt_50us=0 non_positive_gap=21 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=5187c730-19e2-426f-8ca4-8cee71ccc560 valid_pairs=4/30 Ce=7.09Mbps gap_lt_50us=0 non_positive_gap=26 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=532810fd-cfe1-4a39-a11d-16473c923d4c valid_pairs=7/30 Ce=8.67Mbps gap_lt_50us=0 non_positive_gap=23 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=6e762630-a74f-488e-baf4-b2dd1218d1a8 valid_pairs=3/30 Ce=1.54Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=711f652c-0eb3-4ca8-816b-4eada5b4580f valid_pairs=3/30 Ce=7.23Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=7419b9f4-b385-483d-a496-c7c5c6682d0e valid_pairs=10/30 Ce=7.22Mbps gap_lt_50us=0 non_positive_gap=20 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=81c46062-3fd6-439f-a1f3-11e9841a12b9 valid_pairs=7/30 Ce=8.03Mbps gap_lt_50us=0 non_positive_gap=23 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=8cfed090-364b-4366-8648-2d829b6bb74a valid_pairs=3/30 Ce=8.79Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=8d496292-fe45-46ba-aa90-202b66ceb3a1 valid_pairs=7/30 Ce=6.39Mbps gap_lt_50us=0 non_positive_gap=23 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=8d4e7c81-6f26-4c13-9492-dd5cd3f17363 valid_pairs=2/30 Ce=10.64Mbps gap_lt_50us=0 non_positive_gap=28 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=90f4945d-9b3f-4815-8e7c-d6b51b0e7024 valid_pairs=6/30 Ce=8.96Mbps gap_lt_50us=0 non_positive_gap=24 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=9c480254-1523-4f5e-954e-94fd57c88033 valid_pairs=5/30 Ce=7.73Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=9c9b6cd6-0e21-4a13-aa7b-82f747481cef valid_pairs=5/30 Ce=9.72Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=9fba4e26-78ba-4868-be1f-bdfc8101809d valid_pairs=2/30 Ce=9.98Mbps gap_lt_50us=0 non_positive_gap=28 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=a0dfb8c6-ba5a-4820-ad52-03eec5f6db9a valid_pairs=2/30 Ce=8.37Mbps gap_lt_50us=0 non_positive_gap=28 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=a661a0f8-58e8-4fae-b502-f666cea4b861 valid_pairs=6/30 Ce=8.93Mbps gap_lt_50us=0 non_positive_gap=24 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=a6b48c8a-c900-4654-8a8f-8ca9a4a441d4 valid_pairs=6/30 Ce=8.08Mbps gap_lt_50us=0 non_positive_gap=24 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=a8c3a292-adff-46d1-b32a-ce1bbd736939 valid_pairs=4/30 Ce=8.21Mbps gap_lt_50us=0 non_positive_gap=26 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=aa790382-500b-4fd3-9e54-d5740c953403 valid_pairs=4/30 Ce=6.41Mbps gap_lt_50us=0 non_positive_gap=26 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=ab9ca8eb-26bb-4f4f-bfb5-6a332a970ce4 valid_pairs=3/30 Ce=4.09Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=b30f67ef-e47c-4901-b46a-7c4da466f19c valid_pairs=6/30 Ce=9.31Mbps gap_lt_50us=0 non_positive_gap=24 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=b6a013ac-ddb2-4adc-9a23-e154dff8b944 valid_pairs=3/30 Ce=7.03Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=be749c94-f375-4e12-b5cc-5e85bb42a113 valid_pairs=5/30 Ce=10.34Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=c1a0fd87-53a1-451e-98f9-718797a5af9f valid_pairs=4/30 Ce=9.10Mbps gap_lt_50us=0 non_positive_gap=26 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=d1a5344c-f06d-49a1-a5f1-2d667046165f valid_pairs=4/30 Ce=9.05Mbps gap_lt_50us=0 non_positive_gap=26 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=d2ef3bd6-bb76-4fb8-8458-5ec02037bf82 valid_pairs=5/30 Ce=10.29Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=d56e87db-1736-4b56-a207-339f2546de08 valid_pairs=2/30 Ce=10.60Mbps gap_lt_50us=0 non_positive_gap=28 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=db4ac96c-d22e-40e1-bee4-16f051996bfd valid_pairs=3/30 Ce=10.38Mbps gap_lt_50us=0 non_positive_gap=27 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=dc007cf6-33d4-4943-b2c3-bdd9d2a82e92 valid_pairs=7/30 Ce=9.49Mbps gap_lt_50us=0 non_positive_gap=23 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=e3abb1aa-d856-4336-9d28-8ebf4f424d34 valid_pairs=5/30 Ce=3.94Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=eb3aafb9-b829-4ac1-bd55-dae8cba4d553 valid_pairs=5/30 Ce=8.38Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=ec2e8c00-743e-40e3-8d86-41dc1c3430e0 valid_pairs=10/30 Ce=8.92Mbps gap_lt_50us=0 non_positive_gap=20 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=f6ca012e-d98d-4897-9abd-929e3be60d3d valid_pairs=6/30 Ce=7.30Mbps gap_lt_50us=0 non_positive_gap=24 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=f8be3ee2-a785-4f0d-ad50-a85aa69028ab valid_pairs=2/30 Ce=4.93Mbps gap_lt_50us=0 non_positive_gap=28 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=fbff75a5-1999-4f16-8d71-0960f86ba6da valid_pairs=8/30 Ce=10.07Mbps gap_lt_50us=0 non_positive_gap=22 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0
[WBest][Capture][Summary] round=fe818265-ec1a-4972-b410-c803d5e05321 valid_pairs=5/30 Ce=10.32Mbps gap_lt_50us=0 non_positive_gap=25 ci_300_1000=0 missing=0 reordered=0 seq_mismatch=0