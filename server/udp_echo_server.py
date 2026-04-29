import asyncio
import json
import statistics
import struct
import uuid

from server.monotonic_clock import server_now_ns

LATENCY_MESSAGE_TYPE = 1
TIME_SYNC_MESSAGE_TYPE = 4
PTR_MESSAGE_TYPE = 3
PTR_PHASE_STATE_TTL_NS = 10_000_000_000

LATENCY_REQUEST_STRUCT = struct.Struct("!BIQ")
LATENCY_RESPONSE_STRUCT = struct.Struct("!BIQQ")
TIME_SYNC_REQUEST_STRUCT = struct.Struct("!BIQ")
TIME_SYNC_RESPONSE_STRUCT = struct.Struct("!BIQQQ")
PTR_REQUEST_STRUCT = struct.Struct("!BB16sIII")
PTR_PHASE_SUMMARY_RESPONSE_STRUCT = struct.Struct("!BB16sBIIIQQddddIIII")

PTR_STAGE_PACKET_TRAIN = 1
PTR_STAGE_PHASE_SUMMARY = 2


class PTRPhaseState:
    def __init__(self, created_perf_ns: int):
        self.packet_size_bytes = 0
        self.expected_packets = 0
        self.train_records = {}
        self.last_updated_perf_ns = created_perf_ns

    def touch(self, now_perf_ns: int):
        self.last_updated_perf_ns = now_perf_ns


class UDPEchoProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None
        self.ptr_phase_states = {}

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if not data:
            return

        arrival_perf_ns = server_now_ns()
        if data[:1] == b"{":
            self._handle_json_time_sync_probe(data, addr, arrival_perf_ns)
            return

        message_type = data[0]
        if message_type == LATENCY_MESSAGE_TYPE:
            self._handle_latency_probe(data, addr, arrival_perf_ns)
            return

        if message_type == TIME_SYNC_MESSAGE_TYPE:
            self._handle_binary_time_sync_probe(data, addr, arrival_perf_ns)
            return

        if message_type == PTR_MESSAGE_TYPE:
            self._handle_ptr_probe(data, addr, arrival_perf_ns)

    def _handle_latency_probe(self, data, addr, server_receive_time_ns: int):
        try:
            version, sequence, client_send_time_ns = LATENCY_REQUEST_STRUCT.unpack(data)
        except struct.error:
            return

        response = LATENCY_RESPONSE_STRUCT.pack(
            version,
            sequence,
            client_send_time_ns,
            server_receive_time_ns,
        )
        self.transport.sendto(response, addr)

    def _handle_binary_time_sync_probe(self, data, addr, server_receive_time_ns: int):
        try:
            message_type, sequence, client_send_time_ns = TIME_SYNC_REQUEST_STRUCT.unpack(data)
        except struct.error:
            return

        response = TIME_SYNC_RESPONSE_STRUCT.pack(
            message_type,
            sequence,
            client_send_time_ns,
            server_receive_time_ns,
            server_now_ns(),
        )
        self.transport.sendto(response, addr)

    def _handle_json_time_sync_probe(self, data, addr, server_receive_time_ns: int):
        try:
            message = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        if message.get("type") != "time_sync":
            return

        sequence = message.get("seq")
        client_send_time_ns = message.get("client_send_ns")
        if not isinstance(sequence, int) or not isinstance(client_send_time_ns, int):
            return

        response = {
            "type": "time_sync_response",
            "seq": sequence,
            "client_send_ns": client_send_time_ns,
            "server_receive_ns": server_receive_time_ns,
            "server_send_ns": server_now_ns(),
        }
        self.transport.sendto(json.dumps(response, separators=(",", ":")).encode("utf-8"), addr)

    def _handle_ptr_probe(self, data, addr, arrival_perf_ns: int):
        try:
            message_type, stage, round_id_bytes, phase_id, sequence, total_count = (
                PTR_REQUEST_STRUCT.unpack_from(data)
            )
        except struct.error:
            return

        self._cleanup_stale_ptr_phase_states(arrival_perf_ns)

        if stage == PTR_STAGE_PACKET_TRAIN:
            self._record_ptr_train_packet(
                round_id_bytes=round_id_bytes,
                phase_id=phase_id,
                sequence=sequence,
                packet_count=total_count,
                packet_size_bytes=len(data),
                arrival_perf_ns=arrival_perf_ns,
            )
            return

        if stage == PTR_STAGE_PHASE_SUMMARY:
            self._respond_with_ptr_phase_summary(
                message_type=message_type,
                round_id_bytes=round_id_bytes,
                phase_id=phase_id,
                packet_count=total_count,
                addr=addr,
            )

    def _record_ptr_train_packet(
        self,
        round_id_bytes: bytes,
        phase_id: int,
        sequence: int,
        packet_count: int,
        packet_size_bytes: int,
        arrival_perf_ns: int,
    ):
        if sequence <= 0:
            return

        state = self._get_or_create_ptr_phase_state(round_id_bytes, phase_id, arrival_perf_ns)
        state.packet_size_bytes = max(state.packet_size_bytes, packet_size_bytes)
        state.expected_packets = max(state.expected_packets, packet_count)
        state.train_records[sequence] = arrival_perf_ns
        state.touch(arrival_perf_ns)

    def _respond_with_ptr_phase_summary(
        self,
        message_type: int,
        round_id_bytes: bytes,
        phase_id: int,
        packet_count: int,
        addr,
    ):
        key = (round_id_bytes, phase_id)
        state = self.ptr_phase_states.get(key)
        success = False
        summary = {
            "received_packets": 0,
            "sent_packets": packet_count,
            "mean_gap_ns": 0,
            "recv_duration_ns": 0,
            "loss_rate": 0.0,
            "overall_rate_mbps": 0.0,
            "first_half_rate_mbps": 0.0,
            "second_half_rate_mbps": 0.0,
            "gap_count": 0,
            "non_positive_gap_count": 0,
            "missing_packet_count": 0,
            "reordered_packet_count": 0,
        }

        if state is not None:
            state.touch(server_now_ns())
            state.expected_packets = max(state.expected_packets, packet_count)
            computed_summary = self._compute_ptr_phase_summary(state)
            if computed_summary is not None:
                success = True
                summary = computed_summary
                self._log_ptr_phase_summary(round_id_bytes, phase_id, state, summary)

        response = PTR_PHASE_SUMMARY_RESPONSE_STRUCT.pack(
            message_type,
            PTR_STAGE_PHASE_SUMMARY,
            round_id_bytes,
            int(success),
            phase_id,
            summary["received_packets"],
            summary["sent_packets"],
            summary["mean_gap_ns"],
            summary["recv_duration_ns"],
            float(summary["loss_rate"]),
            float(summary["overall_rate_mbps"]),
            float(summary["first_half_rate_mbps"]),
            float(summary["second_half_rate_mbps"]),
            summary["gap_count"],
            summary["non_positive_gap_count"],
            summary["missing_packet_count"],
            summary["reordered_packet_count"],
        )
        self.transport.sendto(response, addr)
        self.ptr_phase_states.pop(key, None)

    def _get_or_create_ptr_phase_state(
        self,
        round_id_bytes: bytes,
        phase_id: int,
        now_perf_ns: int,
    ):
        key = (round_id_bytes, phase_id)
        state = self.ptr_phase_states.get(key)
        if state is None:
            state = PTRPhaseState(now_perf_ns)
            self.ptr_phase_states[key] = state
        return state

    def _compute_ptr_phase_summary(self, state: PTRPhaseState):
        packet_size_bytes = state.packet_size_bytes
        if packet_size_bytes <= 0:
            return None

        train_records = sorted(state.train_records.items(), key=lambda item: item[0])
        received_packets = len(train_records)
        sent_packets = max(state.expected_packets, received_packets)
        missing_packet_count = (
            max(sent_packets - received_packets, 0)
            if sent_packets > 0
            else 0
        )
        if received_packets < 2:
            return None

        positive_gaps_ns = []
        gap_details = []
        non_positive_gap_count = 0
        reordered_packet_count = 0

        for index in range(received_packets - 1):
            current_sequence, current_time = train_records[index]
            next_sequence, next_time = train_records[index + 1]
            gap_ns = next_time - current_time
            gap_details.append((current_sequence, next_sequence, gap_ns))

            if gap_ns <= 0:
                non_positive_gap_count += 1
                if gap_ns < 0:
                    reordered_packet_count += 1
                continue

            positive_gaps_ns.append(gap_ns)

        if not positive_gaps_ns:
            return None

        recv_duration_ns = train_records[-1][1] - train_records[0][1]
        if recv_duration_ns <= 0:
            return None

        mean_gap_ns_float = statistics.fmean(positive_gaps_ns)
        overall_rate_mbps = (
            ((received_packets - 1) * packet_size_bytes * 8.0)
            / (recv_duration_ns / 1_000_000_000.0)
            / 1_000_000.0
        )

        split_index = received_packets // 2
        first_half_rate_mbps = self._compute_ptr_segment_rate_mbps(
            train_records[:split_index],
            packet_size_bytes,
        )
        second_half_rate_mbps = self._compute_ptr_segment_rate_mbps(
            train_records[split_index:],
            packet_size_bytes,
        )
        loss_rate = (
            missing_packet_count / sent_packets
            if sent_packets > 0
            else 0.0
        )

        return {
            "received_packets": received_packets,
            "sent_packets": sent_packets,
            "mean_gap_ns": int(round(mean_gap_ns_float)),
            "recv_duration_ns": recv_duration_ns,
            "loss_rate": float(loss_rate),
            "overall_rate_mbps": float(overall_rate_mbps),
            "first_half_rate_mbps": float(first_half_rate_mbps),
            "second_half_rate_mbps": float(second_half_rate_mbps),
            "gap_count": len(positive_gaps_ns),
            "non_positive_gap_count": non_positive_gap_count,
            "missing_packet_count": missing_packet_count,
            "reordered_packet_count": reordered_packet_count,
            "gap_details": gap_details,
        }

    def _compute_ptr_segment_rate_mbps(self, records, packet_size_bytes: int) -> float:
        if len(records) < 2:
            return 0.0

        duration_ns = records[-1][1] - records[0][1]
        if duration_ns <= 0:
            return 0.0

        return (
            ((len(records) - 1) * packet_size_bytes * 8.0)
            / (duration_ns / 1_000_000_000.0)
            / 1_000_000.0
        )

    def _log_ptr_phase_summary(
        self,
        round_id_bytes: bytes,
        phase_id: int,
        state: PTRPhaseState,
        summary,
    ):
        round_id = self._format_round_id(round_id_bytes)
        gap_details = summary.get("gap_details", [])
        gap_text = ",".join(
            f"{current}->{next_}:{gap_ns / 1_000.0:.2f}us"
            for current, next_, gap_ns in gap_details
        )

        print(
            "[PTR][Summary] "
            f"round={round_id} "
            f"phase={phase_id} "
            f"payload_bytes={state.packet_size_bytes} "
            f"received_train={summary['received_packets']}/{summary['sent_packets']} "
            f"PTR={summary['overall_rate_mbps']:.2f}Mbps "
            f"first_half_rate={summary['first_half_rate_mbps']:.2f}Mbps "
            f"second_half_rate={summary['second_half_rate_mbps']:.2f}Mbps "
            f"gap_count={summary['gap_count']} "
            f"gap_mean={summary['mean_gap_ns'] / 1_000.0:.2f}us "
            f"loss={summary['loss_rate']:.3f} "
            f"missing={summary['missing_packet_count']} "
            f"non_positive_gap={summary['non_positive_gap_count']} "
            f"reordered={summary['reordered_packet_count']} "
            "timestamp_source=app_perf_counter"
        )
        print(
            "[PTR][Detail] "
            f"round={round_id} "
            f"phase={phase_id} "
            f"gaps={gap_text}"
        )

    def _format_round_id(self, round_id_bytes: bytes) -> str:
        try:
            return str(uuid.UUID(bytes=round_id_bytes))
        except ValueError:
            return round_id_bytes.hex()

    def _cleanup_stale_ptr_phase_states(self, now_perf_ns: int):
        stale_keys = [
            key
            for key, state in self.ptr_phase_states.items()
            if (now_perf_ns - state.last_updated_perf_ns) > PTR_PHASE_STATE_TTL_NS
        ]
        for key in stale_keys:
            self.ptr_phase_states.pop(key, None)


async def start_udp_echo_server(host: str = "0.0.0.0", port: int = 9999):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: UDPEchoProtocol(),
        local_addr=(host, port),
    )
    return transport
