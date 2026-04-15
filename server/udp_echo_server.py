import asyncio
import struct
import time

LATENCY_MESSAGE_TYPE = 1
BANDWIDTH_MESSAGE_TYPE = 2
PACKET_TRAIN_STATE_TTL_NS = 10_000_000_000

LATENCY_REQUEST_STRUCT = struct.Struct("!BIQ")
LATENCY_RESPONSE_STRUCT = struct.Struct("!BIQQ")
BANDWIDTH_REQUEST_HEADER_STRUCT = struct.Struct("!B16sII")
BANDWIDTH_RESPONSE_STRUCT = struct.Struct("!B16sIIQ")


class PacketTrainState:
    def __init__(self, last_arrival_perf_ns: int):
        self.last_arrival_perf_ns = last_arrival_perf_ns
        self.arrival_count = 1
        self.last_updated_perf_ns = last_arrival_perf_ns


class UDPEchoProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None
        self.packet_train_states = {}

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if not data:
            return

        message_type = data[0]
        if message_type == LATENCY_MESSAGE_TYPE:
            self._handle_latency_probe(data, addr)
            return

        if message_type == BANDWIDTH_MESSAGE_TYPE:
            self._handle_bandwidth_probe(data, addr)

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

    def _handle_bandwidth_probe(self, data, addr):
        try:
            message_type, train_id_bytes, packet_index, packet_count = (
                BANDWIDTH_REQUEST_HEADER_STRUCT.unpack_from(data)
            )
        except struct.error:
            return

        if packet_count < 2:
            return

        arrival_perf_ns = time.perf_counter_ns()
        arrival_index, inter_packet_gap_ns = self._resolve_packet_train_gap(
            train_id_bytes=train_id_bytes,
            packet_count=packet_count,
            arrival_perf_ns=arrival_perf_ns,
        )
        response = BANDWIDTH_RESPONSE_STRUCT.pack(
            message_type,
            train_id_bytes,
            packet_index,
            arrival_index,
            inter_packet_gap_ns,
        )
        self.transport.sendto(response, addr)

    def _resolve_packet_train_gap(self, train_id_bytes: bytes, packet_count: int, arrival_perf_ns: int):
        self._cleanup_stale_packet_train_states(arrival_perf_ns)
        state = self.packet_train_states.get(train_id_bytes)

        if state is None:
            self.packet_train_states[train_id_bytes] = PacketTrainState(arrival_perf_ns)
            return 1, 0

        state.arrival_count += 1
        state.last_updated_perf_ns = arrival_perf_ns
        arrival_index = state.arrival_count
        inter_packet_gap_ns = max(arrival_perf_ns - state.last_arrival_perf_ns, 0)
        state.last_arrival_perf_ns = arrival_perf_ns

        if arrival_index >= packet_count:
            self.packet_train_states.pop(train_id_bytes, None)

        return arrival_index, inter_packet_gap_ns

    def _cleanup_stale_packet_train_states(self, now_perf_ns: int):
        stale_train_ids = [
            train_id_bytes
            for train_id_bytes, state in self.packet_train_states.items()
            if (now_perf_ns - state.last_updated_perf_ns) > PACKET_TRAIN_STATE_TTL_NS
        ]
        for train_id_bytes in stale_train_ids:
            self.packet_train_states.pop(train_id_bytes, None)


async def start_udp_echo_server(host: str = "0.0.0.0", port: int = 9999):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: UDPEchoProtocol(),
        local_addr=(host, port),
    )
    return transport
