import asyncio
import statistics
import struct
import time
import uuid

LATENCY_MESSAGE_TYPE = 1
WBEST_MESSAGE_TYPE = 2
PATHMON_MESSAGE_TYPE = 3
WBEST_ROUND_STATE_TTL_NS = 10_000_000_000
PATHMON_ROUND_STATE_TTL_NS = 10_000_000_000
SUSPICIOUS_PAIR_GAP_US = 50.0
SUSPICIOUS_CI_LOW_MBPS = 300.0
SUSPICIOUS_CI_HIGH_MBPS = 1000.0

LATENCY_REQUEST_STRUCT = struct.Struct("!BIQ")
LATENCY_RESPONSE_STRUCT = struct.Struct("!BIQQ")
WBEST_REQUEST_STRUCT = struct.Struct("!BB16sIIIB")
WBEST_PAIR_SUMMARY_RESPONSE_STRUCT = struct.Struct("!BB16sBId")
WBEST_FINAL_SUMMARY_RESPONSE_STRUCT = struct.Struct("!BB16sBIIQddddd")
PATHMON_REQUEST_STRUCT = struct.Struct("!BB16sIIIB")
PATHMON_SUMMARY_RESPONSE_HEADER_STRUCT = struct.Struct("!BB16sBIIQd")
PATHMON_SUMMARY_RECORD_STRUCT = struct.Struct("!IQ")

WBEST_STAGE_PACKET_PAIR = 1
WBEST_STAGE_PACKET_PAIR_SUMMARY = 2
WBEST_STAGE_PACKET_TRAIN = 3
WBEST_STAGE_FINAL_SUMMARY = 4

PATHMON_STAGE_JITTER_PACKET = 1
PATHMON_STAGE_JITTER_SUMMARY = 2
PATHMON_STAGE_BANDWIDTH_PACKET = 3
PATHMON_STAGE_BANDWIDTH_SUMMARY = 4


class WBestRoundState:
    def __init__(self, created_perf_ns: int):
        self.packet_size_bytes = 0
        self.packet_pair_records = {}
        self.train_records = {}
        self.valid_pair_sample_count = 0
        self.effective_capacity_mbps = None
        self.expected_train_packets = 0
        self.last_updated_perf_ns = created_perf_ns

    def touch(self, now_perf_ns: int):
        self.last_updated_perf_ns = now_perf_ns


class PathMonRoundState:
    def __init__(self, created_perf_ns: int):
        self.packet_size_bytes = 0
        self.jitter_records = {}
        self.bandwidth_records = {}
        self.expected_jitter_packets = 0
        self.expected_bandwidth_packets = 0
        self.last_updated_perf_ns = created_perf_ns

    def touch(self, now_perf_ns: int):
        self.last_updated_perf_ns = now_perf_ns


class UDPEchoProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None
        self.wbest_round_states = {}
        self.pathmon_round_states = {}

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if not data:
            return

        message_type = data[0]
        if message_type == LATENCY_MESSAGE_TYPE:
            self._handle_latency_probe(data, addr)
            return

        if message_type == WBEST_MESSAGE_TYPE:
            self._handle_wbest_probe(data, addr)
            return

        if message_type == PATHMON_MESSAGE_TYPE:
            self._handle_pathmon_probe(data, addr)

    def _handle_latency_probe(self, data, addr):
        try:
            version, sequence, client_send_time_ns = LATENCY_REQUEST_STRUCT.unpack(data)
        except struct.error:
            return

        server_receive_time_ns = time.perf_counter_ns()
        response = LATENCY_RESPONSE_STRUCT.pack(
            version,
            sequence,
            client_send_time_ns,
            server_receive_time_ns,
        )
        self.transport.sendto(response, addr)

    def _handle_wbest_probe(self, data, addr):
        try:
            message_type, stage, round_id_bytes, sequence, group_id, total_count, index_in_group = (
                WBEST_REQUEST_STRUCT.unpack_from(data)
            )
        except struct.error:
            return

        arrival_perf_ns = time.perf_counter_ns()
        self._cleanup_stale_wbest_round_states(arrival_perf_ns)

        if stage == WBEST_STAGE_PACKET_PAIR:
            self._record_packet_pair(
                round_id_bytes=round_id_bytes,
                pair_id=group_id,
                sequence=sequence,
                index_in_pair=index_in_group,
                packet_size_bytes=len(data),
                arrival_perf_ns=arrival_perf_ns,
            )
            return

        if stage == WBEST_STAGE_PACKET_PAIR_SUMMARY:
            self._respond_with_packet_pair_summary(
                message_type=message_type,
                round_id_bytes=round_id_bytes,
                addr=addr,
            )
            return

        if stage == WBEST_STAGE_PACKET_TRAIN:
            self._record_packet_train_packet(
                round_id_bytes=round_id_bytes,
                sequence=sequence,
                packet_count=total_count,
                packet_size_bytes=len(data),
                arrival_perf_ns=arrival_perf_ns,
            )
            return

        if stage == WBEST_STAGE_FINAL_SUMMARY:
            self._respond_with_final_summary(
                message_type=message_type,
                round_id_bytes=round_id_bytes,
                packet_count=total_count,
                addr=addr,
            )

    def _handle_pathmon_probe(self, data, addr):
        try:
            message_type, stage, round_id_bytes, sequence, _group_id, total_count, _index_in_group = (
                PATHMON_REQUEST_STRUCT.unpack_from(data)
            )
        except struct.error:
            return

        arrival_perf_ns = time.perf_counter_ns()
        self._cleanup_stale_pathmon_round_states(arrival_perf_ns)

        if stage == PATHMON_STAGE_JITTER_PACKET:
            self._record_pathmon_packet(
                round_id_bytes=round_id_bytes,
                stage=stage,
                sequence=sequence,
                packet_count=total_count,
                packet_size_bytes=len(data),
                arrival_perf_ns=arrival_perf_ns,
            )
            return

        if stage == PATHMON_STAGE_JITTER_SUMMARY:
            self._respond_with_pathmon_summary(
                message_type=message_type,
                stage=stage,
                round_id_bytes=round_id_bytes,
                packet_count=total_count,
                addr=addr,
            )
            return

        if stage == PATHMON_STAGE_BANDWIDTH_PACKET:
            self._record_pathmon_packet(
                round_id_bytes=round_id_bytes,
                stage=stage,
                sequence=sequence,
                packet_count=total_count,
                packet_size_bytes=len(data),
                arrival_perf_ns=arrival_perf_ns,
            )
            return

        if stage == PATHMON_STAGE_BANDWIDTH_SUMMARY:
            self._respond_with_pathmon_summary(
                message_type=message_type,
                stage=stage,
                round_id_bytes=round_id_bytes,
                packet_count=total_count,
                addr=addr,
            )

    def _record_packet_pair(
        self,
        round_id_bytes: bytes,
        pair_id: int,
        sequence: int,
        index_in_pair: int,
        packet_size_bytes: int,
        arrival_perf_ns: int,
    ):
        if index_in_pair not in (1, 2):
            return

        state = self._get_or_create_round_state(round_id_bytes, arrival_perf_ns)
        state.packet_size_bytes = max(state.packet_size_bytes, packet_size_bytes)
        state.touch(arrival_perf_ns)
        pair_record = state.packet_pair_records.setdefault(pair_id, {})
        pair_record[f"seq{index_in_pair}"] = sequence
        pair_record[f"recv{index_in_pair}"] = arrival_perf_ns

    def _record_pathmon_packet(
        self,
        round_id_bytes: bytes,
        stage: int,
        sequence: int,
        packet_count: int,
        packet_size_bytes: int,
        arrival_perf_ns: int,
    ):
        if sequence <= 0:
            return

        state = self._get_or_create_pathmon_round_state(round_id_bytes, arrival_perf_ns)
        state.packet_size_bytes = max(state.packet_size_bytes, packet_size_bytes)
        state.touch(arrival_perf_ns)

        if stage == PATHMON_STAGE_JITTER_PACKET:
            state.expected_jitter_packets = max(state.expected_jitter_packets, packet_count)
            state.jitter_records[sequence] = arrival_perf_ns
            return

        if stage == PATHMON_STAGE_BANDWIDTH_PACKET:
            state.expected_bandwidth_packets = max(state.expected_bandwidth_packets, packet_count)
            state.bandwidth_records[sequence] = arrival_perf_ns

    def _respond_with_pathmon_summary(
        self,
        message_type: int,
        stage: int,
        round_id_bytes: bytes,
        packet_count: int,
        addr,
    ):
        state = self.pathmon_round_states.get(round_id_bytes)
        if state is not None:
            state.touch(time.perf_counter_ns())

        if stage == PATHMON_STAGE_JITTER_SUMMARY:
            records = state.jitter_records if state is not None else {}
            expected_count = max(
                packet_count,
                state.expected_jitter_packets if state is not None else 0,
                len(records),
            )
        else:
            records = state.bandwidth_records if state is not None else {}
            expected_count = max(
                packet_count,
                state.expected_bandwidth_packets if state is not None else 0,
                len(records),
            )

        sorted_records = sorted(records.items(), key=lambda item: item[0])
        received_count = len(sorted_records)
        first_receive_time_ns = sorted_records[0][1] if sorted_records else 0
        loss_rate = (
            max(expected_count - received_count, 0) / expected_count
            if expected_count > 0
            else 0.0
        )
        success = received_count > 0

        response = bytearray(
            PATHMON_SUMMARY_RESPONSE_HEADER_STRUCT.pack(
                message_type,
                stage,
                round_id_bytes,
                int(success),
                received_count,
                expected_count,
                first_receive_time_ns,
                float(loss_rate),
            )
        )
        for sequence, receive_time_ns in sorted_records:
            response.extend(
                PATHMON_SUMMARY_RECORD_STRUCT.pack(
                    sequence,
                    max(receive_time_ns - first_receive_time_ns, 0),
                )
            )

        self.transport.sendto(bytes(response), addr)
        self._log_pathmon_summary(
            round_id_bytes=round_id_bytes,
            stage=stage,
            packet_size_bytes=state.packet_size_bytes if state is not None else 0,
            received_count=received_count,
            expected_count=expected_count,
            loss_rate=loss_rate,
            sorted_records=sorted_records,
        )

        if stage == PATHMON_STAGE_BANDWIDTH_SUMMARY:
            self.pathmon_round_states.pop(round_id_bytes, None)

    def _respond_with_packet_pair_summary(self, message_type: int, round_id_bytes: bytes, addr):
        state = self.wbest_round_states.get(round_id_bytes)
        success = False
        valid_pair_sample_count = 0
        effective_capacity_mbps = 0.0

        if state is not None:
            state.touch(time.perf_counter_ns())
            effective_capacity_mbps, valid_pair_sample_count = self._compute_effective_capacity_mbps(state)
            success = effective_capacity_mbps is not None
            self._log_packet_pair_analysis(round_id_bytes, state)
            if success:
                state.effective_capacity_mbps = effective_capacity_mbps
                state.valid_pair_sample_count = valid_pair_sample_count
            else:
                effective_capacity_mbps = 0.0

        response = WBEST_PAIR_SUMMARY_RESPONSE_STRUCT.pack(
            message_type,
            WBEST_STAGE_PACKET_PAIR_SUMMARY,
            round_id_bytes,
            int(success),
            valid_pair_sample_count,
            float(effective_capacity_mbps),
        )
        self.transport.sendto(response, addr)

    def _record_packet_train_packet(
        self,
        round_id_bytes: bytes,
        sequence: int,
        packet_count: int,
        packet_size_bytes: int,
        arrival_perf_ns: int,
    ):
        state = self._get_or_create_round_state(round_id_bytes, arrival_perf_ns)
        state.packet_size_bytes = max(state.packet_size_bytes, packet_size_bytes)
        state.expected_train_packets = max(state.expected_train_packets, packet_count)
        state.touch(arrival_perf_ns)

        if sequence > 0:
            state.train_records[sequence] = arrival_perf_ns

    def _respond_with_final_summary(
        self,
        message_type: int,
        round_id_bytes: bytes,
        packet_count: int,
        addr,
    ):
        state = self.wbest_round_states.get(round_id_bytes)
        success = False
        received_train_packets = 0
        sent_train_packets = 0
        mean_train_gap_ns = 0
        effective_capacity_mbps = 0.0
        achievable_throughput_mbps = 0.0
        available_bandwidth_mbps = 0.0
        corrected_available_bandwidth_mbps = 0.0
        loss_rate = 0.0

        if state is not None:
            state.touch(time.perf_counter_ns())
            state.expected_train_packets = max(state.expected_train_packets, packet_count)

            if state.effective_capacity_mbps is None:
                effective_capacity_mbps_candidate, valid_pair_sample_count = (
                    self._compute_effective_capacity_mbps(state)
                )
                if effective_capacity_mbps_candidate is not None:
                    state.effective_capacity_mbps = effective_capacity_mbps_candidate
                    state.valid_pair_sample_count = valid_pair_sample_count

            summary = self._compute_final_summary(state)
            if summary is not None:
                success = True
                received_train_packets = summary["received_train_packets"]
                sent_train_packets = summary["sent_train_packets"]
                mean_train_gap_ns = summary["mean_train_gap_ns"]
                effective_capacity_mbps = summary["effective_capacity_mbps"]
                achievable_throughput_mbps = summary["achievable_throughput_mbps"]
                available_bandwidth_mbps = summary["available_bandwidth_mbps"]
                corrected_available_bandwidth_mbps = summary[
                    "corrected_available_bandwidth_mbps"
                ]
                loss_rate = summary["loss_rate"]

        response = WBEST_FINAL_SUMMARY_RESPONSE_STRUCT.pack(
            message_type,
            WBEST_STAGE_FINAL_SUMMARY,
            round_id_bytes,
            int(success),
            received_train_packets,
            sent_train_packets,
            mean_train_gap_ns,
            float(effective_capacity_mbps),
            float(achievable_throughput_mbps),
            float(available_bandwidth_mbps),
            float(corrected_available_bandwidth_mbps),
            float(loss_rate),
        )
        self.transport.sendto(response, addr)
        self.wbest_round_states.pop(round_id_bytes, None)

    def _compute_effective_capacity_mbps(self, state: WBestRoundState):
        analysis = self._build_packet_pair_analysis(state)
        capacity_samples_mbps = analysis["capacity_samples_mbps"]
        if not capacity_samples_mbps:
            return None, 0

        return float(statistics.median(capacity_samples_mbps)), len(capacity_samples_mbps)

    def _compute_final_summary(self, state: WBestRoundState):
        packet_size_bytes = state.packet_size_bytes
        effective_capacity_mbps = state.effective_capacity_mbps

        if packet_size_bytes <= 0 or effective_capacity_mbps is None or effective_capacity_mbps <= 0:
            return None

        train_records = sorted(state.train_records.items(), key=lambda item: item[0])
        received_train_packets = len(train_records)
        if received_train_packets < 2:
            return None

        train_gaps_ns = []
        for index in range(received_train_packets - 1):
            current_time = train_records[index][1]
            next_time = train_records[index + 1][1]
            gap_ns = next_time - current_time
            if gap_ns > 0:
                train_gaps_ns.append(gap_ns)

        if not train_gaps_ns:
            return None

        mean_train_gap_ns_float = statistics.fmean(train_gaps_ns)
        mean_train_gap_seconds = mean_train_gap_ns_float / 1_000_000_000.0
        achievable_throughput_mbps = (
            (packet_size_bytes * 8.0) / mean_train_gap_seconds / 1_000_000.0
        )
        achievable_throughput_mbps = min(achievable_throughput_mbps, effective_capacity_mbps)

        if achievable_throughput_mbps >= (effective_capacity_mbps / 2.0):
            available_bandwidth_mbps = effective_capacity_mbps * (
                2.0 - (effective_capacity_mbps / achievable_throughput_mbps)
            )
        else:
            available_bandwidth_mbps = 0.0

        available_bandwidth_mbps = max(0.0, available_bandwidth_mbps)
        sent_train_packets = max(state.expected_train_packets, received_train_packets)
        loss_rate = (
            max(sent_train_packets - received_train_packets, 0) / sent_train_packets
            if sent_train_packets > 0
            else 0.0
        )
        corrected_available_bandwidth_mbps = available_bandwidth_mbps * (1.0 - loss_rate)

        return {
            "received_train_packets": received_train_packets,
            "sent_train_packets": sent_train_packets,
            "mean_train_gap_ns": int(round(mean_train_gap_ns_float)),
            "effective_capacity_mbps": float(effective_capacity_mbps),
            "achievable_throughput_mbps": float(achievable_throughput_mbps),
            "available_bandwidth_mbps": float(available_bandwidth_mbps),
            "corrected_available_bandwidth_mbps": float(corrected_available_bandwidth_mbps),
            "loss_rate": float(loss_rate),
        }

    def _get_or_create_round_state(self, round_id_bytes: bytes, now_perf_ns: int):
        state = self.wbest_round_states.get(round_id_bytes)
        if state is None:
            state = WBestRoundState(now_perf_ns)
            self.wbest_round_states[round_id_bytes] = state
        return state

    def _get_or_create_pathmon_round_state(self, round_id_bytes: bytes, now_perf_ns: int):
        state = self.pathmon_round_states.get(round_id_bytes)
        if state is None:
            state = PathMonRoundState(now_perf_ns)
            self.pathmon_round_states[round_id_bytes] = state
        return state

    def _build_packet_pair_analysis(self, state: WBestRoundState):
        details = []
        capacity_samples_mbps = []
        pair_ids = sorted(state.packet_pair_records.keys())

        for pair_id in pair_ids:
            pair_record = state.packet_pair_records[pair_id]
            expected_seq1 = (2 * pair_id) - 1
            expected_seq2 = 2 * pair_id
            seq1 = pair_record.get("seq1")
            seq2 = pair_record.get("seq2")
            recv_time_1 = pair_record.get("recv1")
            recv_time_2 = pair_record.get("recv2")

            has_both = recv_time_1 is not None and recv_time_2 is not None
            reordered = has_both and recv_time_2 < recv_time_1
            seq_mismatch = (
                seq1 is not None
                and seq2 is not None
                and (seq1 != expected_seq1 or seq2 != expected_seq2)
            )

            gap_ns = None
            gap_us = None
            ci_mbps = None
            if has_both:
                gap_ns = recv_time_2 - recv_time_1
                gap_us = gap_ns / 1_000.0
                if gap_ns > 0:
                    gap_seconds = gap_ns / 1_000_000_000.0
                    ci_mbps = (state.packet_size_bytes * 8.0) / gap_seconds / 1_000_000.0
                    capacity_samples_mbps.append(ci_mbps)

            gap_under_50us = gap_us is not None and 0.0 < gap_us < SUSPICIOUS_PAIR_GAP_US
            ci_in_suspicious_range = (
                ci_mbps is not None
                and SUSPICIOUS_CI_LOW_MBPS <= ci_mbps <= SUSPICIOUS_CI_HIGH_MBPS
            )

            details.append(
                {
                    "pair_id": pair_id,
                    "expected_seq1": expected_seq1,
                    "expected_seq2": expected_seq2,
                    "seq1": seq1,
                    "seq2": seq2,
                    "recv_time_1": recv_time_1,
                    "recv_time_2": recv_time_2,
                    "has_both": has_both,
                    "reordered": reordered,
                    "seq_mismatch": seq_mismatch,
                    "gap_ns": gap_ns,
                    "gap_us": gap_us,
                    "ci_mbps": ci_mbps,
                    "gap_under_50us": gap_under_50us,
                    "ci_in_suspicious_range": ci_in_suspicious_range,
                }
            )

        return {
            "details": details,
            "capacity_samples_mbps": capacity_samples_mbps,
            "gap_under_50us_count": sum(1 for detail in details if detail["gap_under_50us"]),
            "ci_300_to_1000_count": sum(1 for detail in details if detail["ci_in_suspicious_range"]),
            "missing_pair_count": sum(1 for detail in details if not detail["has_both"]),
            "reordered_pair_count": sum(1 for detail in details if detail["reordered"]),
            "seq_mismatch_count": sum(1 for detail in details if detail["seq_mismatch"]),
        }

    def _log_packet_pair_analysis(self, round_id_bytes: bytes, state: WBestRoundState):
        analysis = self._build_packet_pair_analysis(state)
        details = analysis["details"]
        capacity_samples_mbps = analysis["capacity_samples_mbps"]
        round_id = self._format_round_id(round_id_bytes)
        ce_text = (
            f"{statistics.median(capacity_samples_mbps):.2f}Mbps"
            if capacity_samples_mbps
            else "n/a"
        )

        print(
            "[WBest][Pairs][Summary] "
            f"round={round_id} "
            f"payload_bytes={state.packet_size_bytes} "
            f"valid_pairs={len(capacity_samples_mbps)}/{len(details)} "
            f"Ce={ce_text} "
            f"gap_lt_50us={analysis['gap_under_50us_count']} "
            f"ci_300_1000={analysis['ci_300_to_1000_count']} "
            f"missing={analysis['missing_pair_count']} "
            f"reordered={analysis['reordered_pair_count']} "
            f"seq_mismatch={analysis['seq_mismatch_count']}"
        )

        for detail in details:
            gap_text = (
                f"{detail['gap_us']:.2f}us"
                if detail["gap_us"] is not None
                else "n/a"
            )
            ci_text = (
                f"{detail['ci_mbps']:.2f}Mbps"
                if detail["ci_mbps"] is not None
                else "n/a"
            )

            flags = []
            if detail["gap_under_50us"]:
                flags.append("gap_lt_50us")
            if detail["ci_in_suspicious_range"]:
                flags.append("ci_300_1000")
            if detail["reordered"]:
                flags.append("reordered")
            if detail["seq_mismatch"]:
                flags.append("seq_mismatch")
            if not detail["has_both"]:
                flags.append("incomplete")

            flag_text = ",".join(flags) if flags else "ok"
            print(
                "[WBest][Pairs][Detail] "
                f"round={round_id} "
                f"pair={detail['pair_id']:02d} "
                f"seq=({detail['seq1']},{detail['seq2']}) "
                f"expected=({detail['expected_seq1']},{detail['expected_seq2']}) "
                f"gap={gap_text} "
                f"Ci={ci_text} "
                f"flags={flag_text}"
            )

    def _format_round_id(self, round_id_bytes: bytes) -> str:
        try:
            return str(uuid.UUID(bytes=round_id_bytes))
        except ValueError:
            return round_id_bytes.hex()

    def _log_pathmon_summary(
        self,
        round_id_bytes: bytes,
        stage: int,
        packet_size_bytes: int,
        received_count: int,
        expected_count: int,
        loss_rate: float,
        sorted_records,
    ):
        stage_name = (
            "jitter"
            if stage == PATHMON_STAGE_JITTER_SUMMARY
            else "bandwidth"
        )
        gaps_ns = [
            sorted_records[index][1] - sorted_records[index - 1][1]
            for index in range(1, len(sorted_records))
        ]
        gap_fields = ""
        if gaps_ns:
            sorted_gaps_ns = sorted(gaps_ns)
            gap_fields = (
                f" gap_count={len(gaps_ns)}"
                f" gap_min={min(gaps_ns) / 1000.0:.2f}us"
                f" gap_median={statistics.median(gaps_ns) / 1000.0:.2f}us"
                f" gap_mean={statistics.mean(gaps_ns) / 1000.0:.2f}us"
                f" gap_p90={self._percentile_ns_to_us(sorted_gaps_ns, 0.9):.2f}us"
                f" gap_max={max(gaps_ns) / 1000.0:.2f}us"
            )

        print(
            "[PathMon][Summary] "
            f"round={self._format_round_id(round_id_bytes)} "
            f"stage={stage_name} "
            f"payload_bytes={packet_size_bytes} "
            f"received={received_count}/{expected_count} "
            f"loss={loss_rate:.3f} "
            f"timestamp_source=python_perf_counter{gap_fields}"
        )
        if gaps_ns:
            tail_start = max(len(gaps_ns) - 8, 0)
            tail_items = []
            for index in range(tail_start, len(gaps_ns)):
                sequence = sorted_records[index][0]
                tail_items.append(
                    f"{sequence}->{sequence + 1}:{gaps_ns[index] / 1000.0:.2f}us"
                )
            print(
                "[PathMon][Detail] "
                f"round={self._format_round_id(round_id_bytes)} "
                f"stage={stage_name} "
                f"tail_gaps={','.join(tail_items)}"
            )

    @staticmethod
    def _percentile_ns_to_us(sorted_values_ns, percentile: float):
        if not sorted_values_ns:
            return 0.0
        clamped = min(max(percentile, 0.0), 1.0)
        raw_index = clamped * (len(sorted_values_ns) - 1)
        lower_index = int(raw_index)
        upper_index = min(lower_index + 1, len(sorted_values_ns) - 1)
        if lower_index == upper_index:
            return sorted_values_ns[lower_index] / 1000.0
        fraction = raw_index - lower_index
        return (
            sorted_values_ns[lower_index]
            + ((sorted_values_ns[upper_index] - sorted_values_ns[lower_index]) * fraction)
        ) / 1000.0

    def _cleanup_stale_wbest_round_states(self, now_perf_ns: int):
        stale_round_ids = [
            round_id_bytes
            for round_id_bytes, state in self.wbest_round_states.items()
            if (now_perf_ns - state.last_updated_perf_ns) > WBEST_ROUND_STATE_TTL_NS
        ]
        for round_id_bytes in stale_round_ids:
            self.wbest_round_states.pop(round_id_bytes, None)

    def _cleanup_stale_pathmon_round_states(self, now_perf_ns: int):
        stale_round_ids = [
            round_id_bytes
            for round_id_bytes, state in self.pathmon_round_states.items()
            if (now_perf_ns - state.last_updated_perf_ns) > PATHMON_ROUND_STATE_TTL_NS
        ]
        for round_id_bytes in stale_round_ids:
            self.pathmon_round_states.pop(round_id_bytes, None)


async def start_udp_echo_server(host: str = "0.0.0.0", port: int = 9999):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: UDPEchoProtocol(),
        local_addr=(host, port),
    )
    return transport
