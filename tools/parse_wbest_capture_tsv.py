#!/usr/bin/env python3
"""Analyze WBest packet-pair timing exported from tshark.

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


WBEST_MESSAGE_TYPE = 2
WBEST_STAGE_PACKET_PAIR = 1
WBEST_HEADER_BYTES = 31


@dataclass(frozen=True)
class Packet:
    frame: int
    time_epoch: float
    sequence: int


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


def parse_packet(row: dict[str, str]) -> tuple[str, int, int, int, Packet] | None:
    payload = parse_payload_hex(row)
    if payload is None or len(payload) < WBEST_HEADER_BYTES:
        return None

    if payload[0] != WBEST_MESSAGE_TYPE or payload[1] != WBEST_STAGE_PACKET_PAIR:
        return None

    round_id = str(uuid.UUID(bytes=payload[2:18]))
    sequence, pair_id, total_count = struct.unpack("!III", payload[18:30])
    index_in_pair = payload[30]
    if index_in_pair not in (1, 2):
        return None

    frame_text = row.get("frame.number", "0") or "0"
    time_text = row.get("frame.time_epoch", "") or ""
    packet = Packet(
        frame=int(frame_text),
        time_epoch=float(time_text),
        sequence=sequence,
    )
    return round_id, pair_id, total_count, index_in_pair, packet


def build_pair_samples(
    rows: list[dict[str, str]],
    packet_bytes: int,
) -> tuple[dict[str, list[PairSample]], int]:
    pairs: dict[tuple[str, int], dict[int, Packet]] = defaultdict(dict)
    expected_counts: dict[str, int] = {}
    parsed_packet_count = 0

    for row in rows:
        parsed = parse_packet(row)
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
        description="Analyze WBest packet-pair gaps from tshark TSV export."
    )
    parser.add_argument("tsv", type=Path, help="TSV exported by tshark")
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
    args = parser.parse_args()

    text = read_text(args.tsv)
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    samples_by_round, parsed_packet_count = build_pair_samples(rows, args.packet_bytes)
    if parsed_packet_count == 0:
        print(
            "No WBest packet-pair packets found. Check that tshark exported "
            "udp.payload or data.data and that the capture contains udp.dstport == 9999.",
            file=sys.stderr,
        )
        return 1

    print_report(samples_by_round, args.summary_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
