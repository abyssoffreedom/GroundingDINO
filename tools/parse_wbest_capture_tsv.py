#!/usr/bin/env python3
"""Analyze WBest packet-pair timing exported from tshark or pcapng.

Expected tshark fields:
frame.number, frame.time_epoch, ip.src, udp.srcport, ip.dst, udp.dstport,
udp.length, udp.payload, data.data
"""

from __future__ import annotations

import argparse
import csv
import statistics
import struct
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


WBEST_MESSAGE_TYPE = 2
WBEST_STAGE_PACKET_PAIR = 1
WBEST_STAGE_NAMES = {
    1: "packet_pair",
    2: "packet_pair_summary",
    3: "packet_train",
    4: "final_summary",
}
WBEST_HEADER_BYTES = 31
PCAPNG_SECTION_HEADER_BLOCK = 0x0A0D0D0A
PCAPNG_INTERFACE_DESCRIPTION_BLOCK = 0x00000001
PCAPNG_ENHANCED_PACKET_BLOCK = 0x00000006
LINKTYPE_ETHERNET = 1
DEFAULT_UDP_DST_PORT = 9999


@dataclass(frozen=True)
class Packet:
    frame: int
    time_epoch: float
    sequence: int


@dataclass(frozen=True)
class PcapInterface:
    linktype: int
    timestamp_resolution: float
    timestamp_offset: float


@dataclass(frozen=True)
class PairSample:
    round_id: str
    pair_id: int
    first: Packet | None
    second: Packet | None
    expected_seq1: int
    expected_seq2: int
    gap_us: float | None
    ci_mbps: float | None
    reordered: bool
    seq_mismatch: bool

    @property
    def complete(self) -> bool:
        return self.first is not None and self.second is not None

    @property
    def gap_under_50us(self) -> bool:
        return self.gap_us is not None and 0.0 < self.gap_us < 50.0

    @property
    def non_positive_gap(self) -> bool:
        return self.complete and self.gap_us is not None and self.gap_us <= 0.0

    @property
    def ci_300_to_1000(self) -> bool:
        return self.ci_mbps is not None and 300.0 <= self.ci_mbps <= 1000.0


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_payload_hex(row: dict[str, str]) -> bytes | None:
    payload = (row.get("udp.payload") or row.get("data.data") or "").strip()
    if not payload:
        return None

    payload = payload.replace(":", "").replace(" ", "")
    if len(payload) < WBEST_HEADER_BYTES * 2:
        return None

    try:
        return bytes.fromhex(payload)
    except ValueError:
        return None


def parse_wbest_packet(
    payload: bytes | None,
    frame: int,
    time_epoch: float,
) -> tuple[str, int, int, int, Packet] | None:
    if payload is None or len(payload) < WBEST_HEADER_BYTES:
        return None

    if payload[0] != WBEST_MESSAGE_TYPE or payload[1] != WBEST_STAGE_PACKET_PAIR:
        return None

    round_id = str(uuid.UUID(bytes=payload[2:18]))
    sequence, pair_id, total_count = struct.unpack("!III", payload[18:30])
    index_in_pair = payload[30]
    if index_in_pair not in (1, 2):
        return None

    packet = Packet(
        frame=frame,
        time_epoch=time_epoch,
        sequence=sequence,
    )
    return round_id, pair_id, total_count, index_in_pair, packet


def parse_packet(row: dict[str, str]) -> tuple[str, int, int, int, Packet] | None:
    payload = parse_payload_hex(row)
    frame_text = row.get("frame.number", "0") or "0"
    time_text = row.get("frame.time_epoch", "") or ""
    try:
        frame = int(frame_text)
        time_epoch = float(time_text)
    except ValueError:
        return None

    return parse_wbest_packet(payload, frame, time_epoch)


def build_pair_samples_from_parsed_packets(
    parsed_packets,
    packet_bytes: int,
) -> tuple[dict[str, list[PairSample]], int]:
    pairs: dict[tuple[str, int], dict[int, Packet]] = defaultdict(dict)
    expected_counts: dict[str, int] = {}
    parsed_packet_count = 0

    for parsed in parsed_packets:
        if parsed is None:
            continue

        round_id, pair_id, total_count, index_in_pair, packet = parsed
        parsed_packet_count += 1
        expected_counts[round_id] = max(expected_counts.get(round_id, 0), total_count)
        pairs[(round_id, pair_id)][index_in_pair] = packet

    samples_by_round: dict[str, list[PairSample]] = defaultdict(list)
    pair_ids_by_round: dict[str, set[int]] = defaultdict(set)
    for round_id, pair_id in pairs:
        pair_ids_by_round[round_id].add(pair_id)

    for round_id, pair_ids in pair_ids_by_round.items():
        max_pair_id = max(max(pair_ids), expected_counts.get(round_id, 0))
        for pair_id in range(1, max_pair_id + 1):
            pair = pairs.get((round_id, pair_id), {})
            first = pair.get(1)
            second = pair.get(2)
            expected_seq1 = (2 * pair_id) - 1
            expected_seq2 = 2 * pair_id
            reordered = False
            gap_us = None
            ci_mbps = None

            if first is not None and second is not None:
                reordered = second.time_epoch < first.time_epoch or second.frame < first.frame
                gap_us = (second.time_epoch - first.time_epoch) * 1_000_000.0
                if gap_us > 0.0:
                    ci_mbps = (packet_bytes * 8.0) / gap_us

            seq_mismatch = (
                first is not None
                and second is not None
                and (
                    first.sequence != expected_seq1
                    or second.sequence != expected_seq2
                )
            )
            samples_by_round[round_id].append(
                PairSample(
                    round_id=round_id,
                    pair_id=pair_id,
                    first=first,
                    second=second,
                    expected_seq1=expected_seq1,
                    expected_seq2=expected_seq2,
                    gap_us=gap_us,
                    ci_mbps=ci_mbps,
                    reordered=reordered,
                    seq_mismatch=seq_mismatch,
                )
            )

    return samples_by_round, parsed_packet_count


def build_pair_samples(
    rows: list[dict[str, str]],
    packet_bytes: int,
) -> tuple[dict[str, list[PairSample]], int]:
    return build_pair_samples_from_parsed_packets(
        (parse_packet(row) for row in rows),
        packet_bytes,
    )


def iter_pcapng_options(body: bytes, offset: int, endian: str):
    while offset + 4 <= len(body):
        code, length = struct.unpack(endian + "HH", body[offset : offset + 4])
        offset += 4
        value = body[offset : offset + length]
        offset += (length + 3) & ~3
        if code == 0:
            break
        yield code, value


def parse_timestamp_resolution(value: bytes) -> float:
    if not value:
        return 1e-6

    resolution = value[0]
    if resolution & 0x80:
        return 2.0 ** -(resolution & 0x7F)
    return 10.0 ** -resolution


def iter_pcapng_frames(path: Path):
    data = path.read_bytes()
    offset = 0
    endian = "<"
    interfaces: list[PcapInterface] = []
    frame_number = 0

    while offset + 12 <= len(data):
        block_type = struct.unpack(endian + "I", data[offset : offset + 4])[0]
        if block_type == PCAPNG_SECTION_HEADER_BLOCK:
            byte_order_magic = data[offset + 8 : offset + 12]
            if byte_order_magic == b"\x4d\x3c\x2b\x1a":
                endian = "<"
            elif byte_order_magic == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            else:
                raise ValueError(f"Invalid pcapng byte-order magic at offset {offset}")
            block_total_length = struct.unpack(
                endian + "I", data[offset + 4 : offset + 8]
            )[0]
        else:
            block_total_length = struct.unpack(
                endian + "I", data[offset + 4 : offset + 8]
            )[0]

        if block_total_length < 12 or offset + block_total_length > len(data):
            raise ValueError(f"Invalid pcapng block length at offset {offset}")

        body = data[offset + 8 : offset + block_total_length - 4]

        if block_type == PCAPNG_INTERFACE_DESCRIPTION_BLOCK:
            if len(body) < 8:
                offset += block_total_length
                continue

            linktype, _reserved, _snaplen = struct.unpack(endian + "HHI", body[:8])
            timestamp_resolution = 1e-6
            timestamp_offset = 0.0
            for code, value in iter_pcapng_options(body, 8, endian):
                if code == 9:
                    timestamp_resolution = parse_timestamp_resolution(value)
                elif code == 14 and len(value) == 8:
                    timestamp_offset = float(struct.unpack(endian + "q", value)[0])

            interfaces.append(
                PcapInterface(
                    linktype=linktype,
                    timestamp_resolution=timestamp_resolution,
                    timestamp_offset=timestamp_offset,
                )
            )

        elif block_type == PCAPNG_ENHANCED_PACKET_BLOCK:
            if len(body) < 20:
                offset += block_total_length
                continue

            interface_id, timestamp_high, timestamp_low, captured_len, _packet_len = (
                struct.unpack(endian + "IIIII", body[:20])
            )
            if interface_id >= len(interfaces):
                offset += block_total_length
                continue

            packet_start = 20
            packet_end = packet_start + captured_len
            packet_data = body[packet_start:packet_end]
            interface = interfaces[interface_id]
            timestamp_raw = (timestamp_high << 32) | timestamp_low
            time_epoch = (
                timestamp_raw * interface.timestamp_resolution
                + interface.timestamp_offset
            )
            frame_number += 1
            yield frame_number, time_epoch, interface.linktype, packet_data

        offset += block_total_length


def extract_udp_payload_from_ethernet(
    packet_data: bytes,
    udp_dst_port: int | None,
) -> bytes | None:
    if len(packet_data) < 14:
        return None

    ether_type_offset = 12
    ether_type = struct.unpack("!H", packet_data[ether_type_offset : ether_type_offset + 2])[0]
    payload_offset = 14

    while ether_type in (0x8100, 0x88A8, 0x9100):
        if len(packet_data) < payload_offset + 4:
            return None
        ether_type = struct.unpack("!H", packet_data[payload_offset + 2 : payload_offset + 4])[0]
        payload_offset += 4

    if ether_type != 0x0800:
        return None

    return extract_udp_payload_from_ipv4(packet_data, payload_offset, udp_dst_port)


def extract_udp_payload_from_ipv4(
    packet_data: bytes,
    ip_offset: int,
    udp_dst_port: int | None,
) -> bytes | None:
    if len(packet_data) < ip_offset + 20:
        return None

    version = packet_data[ip_offset] >> 4
    ihl = (packet_data[ip_offset] & 0x0F) * 4
    if version != 4 or ihl < 20 or len(packet_data) < ip_offset + ihl:
        return None

    total_length = struct.unpack("!H", packet_data[ip_offset + 2 : ip_offset + 4])[0]
    protocol = packet_data[ip_offset + 9]
    flags_fragment = struct.unpack("!H", packet_data[ip_offset + 6 : ip_offset + 8])[0]
    fragment_offset = flags_fragment & 0x1FFF
    if protocol != 17 or fragment_offset != 0:
        return None

    udp_offset = ip_offset + ihl
    if len(packet_data) < udp_offset + 8:
        return None

    _src_port, dst_port, udp_length, _checksum = struct.unpack(
        "!HHHH",
        packet_data[udp_offset : udp_offset + 8],
    )
    if udp_dst_port is not None and dst_port != udp_dst_port:
        return None

    ip_payload_end = min(len(packet_data), ip_offset + total_length)
    udp_payload_end = min(ip_payload_end, udp_offset + udp_length)
    if udp_payload_end < udp_offset + 8:
        return None

    return packet_data[udp_offset + 8 : udp_payload_end]


def parse_pcapng_wbest_packets(
    path: Path,
    udp_dst_port: int,
) -> tuple[list[tuple[str, int, int, int, Packet] | None], Counter[int]]:
    parsed_packets = []
    stage_counts: Counter[int] = Counter()

    for frame, time_epoch, linktype, packet_data in iter_pcapng_frames(path):
        if linktype != LINKTYPE_ETHERNET:
            continue

        payload = extract_udp_payload_from_ethernet(packet_data, udp_dst_port)
        if payload is None or len(payload) < 2 or payload[0] != WBEST_MESSAGE_TYPE:
            continue

        stage_counts[payload[1]] += 1
        parsed_packets.append(parse_wbest_packet(payload, frame, time_epoch))

    return parsed_packets, stage_counts


def is_pcapng(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(4) == b"\x0a\x0d\x0d\x0a"
    except OSError:
        return False


def summarize_tsv_wbest_stages(rows: list[dict[str, str]]) -> Counter[int]:
    stage_counts: Counter[int] = Counter()
    for row in rows:
        payload = parse_payload_hex(row)
        if payload is not None and len(payload) >= 2 and payload[0] == WBEST_MESSAGE_TYPE:
            stage_counts[payload[1]] += 1
    return stage_counts


def format_stage_counts(stage_counts: Counter[int]) -> str:
    if not stage_counts:
        return "none"
    return ", ".join(
        f"{stage}({WBEST_STAGE_NAMES.get(stage, 'unknown')})={count}"
        for stage, count in sorted(stage_counts.items())
    )


def format_optional(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{unit}"


def print_report(samples_by_round: dict[str, list[PairSample]], summary_only: bool) -> None:
    for round_id in sorted(samples_by_round):
        samples = samples_by_round[round_id]
        capacity_samples = [sample.ci_mbps for sample in samples if sample.ci_mbps is not None]
        ce_text = f"{statistics.median(capacity_samples):.2f}Mbps" if capacity_samples else "n/a"
        print(
            "[WBest][Capture][Summary] "
            f"round={round_id} "
            f"valid_pairs={len(capacity_samples)}/{len(samples)} "
            f"Ce={ce_text} "
            f"gap_lt_50us={sum(sample.gap_under_50us for sample in samples)} "
            f"non_positive_gap={sum(sample.non_positive_gap for sample in samples)} "
            f"ci_300_1000={sum(sample.ci_300_to_1000 for sample in samples)} "
            f"missing={sum(not sample.complete for sample in samples)} "
            f"reordered={sum(sample.reordered for sample in samples)} "
            f"seq_mismatch={sum(sample.seq_mismatch for sample in samples)}"
        )

        if summary_only:
            continue

        for sample in samples:
            seq1 = sample.first.sequence if sample.first is not None else None
            seq2 = sample.second.sequence if sample.second is not None else None
            frame1 = sample.first.frame if sample.first is not None else None
            frame2 = sample.second.frame if sample.second is not None else None
            flags = []
            if sample.gap_under_50us:
                flags.append("gap_lt_50us")
            if sample.non_positive_gap:
                flags.append("non_positive_gap")
            if sample.ci_300_to_1000:
                flags.append("ci_300_1000")
            if sample.reordered:
                flags.append("reordered")
            if sample.seq_mismatch:
                flags.append("seq_mismatch")
            if not sample.complete:
                flags.append("incomplete")

            print(
                "[WBest][Capture][Detail] "
                f"round={round_id} "
                f"pair={sample.pair_id:02d} "
                f"frame=({frame1},{frame2}) "
                f"seq=({seq1},{seq2}) "
                f"expected=({sample.expected_seq1},{sample.expected_seq2}) "
                f"gap={format_optional(sample.gap_us, 'us')} "
                f"Ci={format_optional(sample.ci_mbps, 'Mbps')} "
                f"flags={','.join(flags) if flags else 'ok'}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze WBest packet-pair gaps from tshark TSV export or pcapng."
    )
    parser.add_argument("capture", type=Path, help="TSV exported by tshark or pcapng")
    parser.add_argument(
        "--packet-bytes",
        type=int,
        default=1500,
        help="UDP payload bytes used by the WBest probe, default: 1500",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print one summary line per round without all pair details",
    )
    parser.add_argument(
        "--udp-dst-port",
        type=int,
        default=DEFAULT_UDP_DST_PORT,
        help=f"UDP destination port used by WBest probes, default: {DEFAULT_UDP_DST_PORT}",
    )
    args = parser.parse_args()

    stage_counts: Counter[int] = Counter()
    if is_pcapng(args.capture):
        parsed_packets, stage_counts = parse_pcapng_wbest_packets(
            args.capture,
            args.udp_dst_port,
        )
        samples_by_round, parsed_packet_count = build_pair_samples_from_parsed_packets(
            parsed_packets,
            args.packet_bytes,
        )
    else:
        text = read_text(args.capture)
        rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
        stage_counts = summarize_tsv_wbest_stages(rows)
        samples_by_round, parsed_packet_count = build_pair_samples(rows, args.packet_bytes)

    if parsed_packet_count == 0:
        print(
            "No WBest packet-pair packets found. Check that tshark exported "
            "udp.payload or data.data and that the capture contains "
            f"udp.dstport == {args.udp_dst_port}.",
            file=sys.stderr,
        )
        print(
            f"Observed WBest stages: {format_stage_counts(stage_counts)}.",
            file=sys.stderr,
        )
        if stage_counts and WBEST_STAGE_PACKET_PAIR not in stage_counts:
            print(
                "The capture/export has WBest control packets but no stage=1 packet-pair "
                "packets. If --packet-bytes is 1500, those UDP datagrams are likely IP "
                "fragmented; parse the .pcapng file directly with this script or recapture "
                "with a smaller packet size.",
                file=sys.stderr,
            )
        return 1

    print_report(samples_by_round, args.summary_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
