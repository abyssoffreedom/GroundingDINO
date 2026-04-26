import asyncio
import statistics
import struct
import time

LATENCY_MESSAGE_TYPE = 1
WBEST_MESSAGE_TYPE = 2
WBEST_ROUND_STATE_TTL_NS = 10_000_000_000

LATENCY_REQUEST_STRUCT = struct.Struct("!BIQ")
LATENCY_RESPONSE_STRUCT = struct.Struct("!BIQQ")
WBEST_REQUEST_STRUCT = struct.Struct("!BB16sIIIB")
WBEST_PAIR_SUMMARY_RESPONSE_STRUCT = struct.Struct("!BB16sBId")
WBEST_FINAL_SUMMARY_RESPONSE_STRUCT = struct.Struct("!BB16sBIIQddddd")

WBEST_STAGE_PACKET_PAIR = 1
WBEST_STAGE_PACKET_PAIR_SUMMARY = 2
WBEST_STAGE_PACKET_TRAIN = 3
WBEST_STAGE_FINAL_SUMMARY = 4


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


class UDPEchoProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None
        self.wbest_round_states = {}

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

    def _record_packet_pair(
        self,
        round_id_bytes: bytes,
        pair_id: int,
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
        pair_record[index_in_pair] = arrival_perf_ns

    def _respond_with_packet_pair_summary(self, message_type: int, round_id_bytes: bytes, addr):
        state = self.wbest_round_states.get(round_id_bytes)
        success = False
        valid_pair_sample_count = 0
        effective_capacity_mbps = 0.0

        if state is not None:
            state.touch(time.perf_counter_ns())
            effective_capacity_mbps, valid_pair_sample_count = self._compute_effective_capacity_mbps(state)
            success = effective_capacity_mbps is not None
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
        if state.packet_size_bytes <= 0:
            return None, 0

        capacity_samples_mbps = []
        for pair_record in state.packet_pair_records.values():
            recv_time_1 = pair_record.get(1)
            recv_time_2 = pair_record.get(2)
            if recv_time_1 is None or recv_time_2 is None:
                continue

            gap_ns = recv_time_2 - recv_time_1
            if gap_ns <= 0:
                continue

            gap_seconds = gap_ns / 1_000_000_000.0
            capacity_mbps = (state.packet_size_bytes * 8.0) / gap_seconds / 1_000_000.0
            capacity_samples_mbps.append(capacity_mbps)

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

    def _cleanup_stale_wbest_round_states(self, now_perf_ns: int):
        stale_round_ids = [
            round_id_bytes
            for round_id_bytes, state in self.wbest_round_states.items()
            if (now_perf_ns - state.last_updated_perf_ns) > WBEST_ROUND_STATE_TTL_NS
        ]
        for round_id_bytes in stale_round_ids:
            self.wbest_round_states.pop(round_id_bytes, None)


async def start_udp_echo_server(host: str = "0.0.0.0", port: int = 9999):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: UDPEchoProtocol(),
        local_addr=(host, port),
    )
    return transport
