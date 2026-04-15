import asyncio
import struct
import time

LATENCY_MESSAGE_TYPE = 1
BANDWIDTH_MESSAGE_TYPE = 2
PACKET_PAIR_STATE_TTL_NS = 10_000_000_000

LATENCY_REQUEST_STRUCT = struct.Struct("!BIQ")
LATENCY_RESPONSE_STRUCT = struct.Struct("!BIQQ")
BANDWIDTH_REQUEST_HEADER_STRUCT = struct.Struct("!B16sIQ")
BANDWIDTH_RESPONSE_STRUCT = struct.Struct("!B16sIIQ")


class PacketPairState:
    def __init__(self, first_arrival_perf_ns: int):
        self.first_arrival_perf_ns = first_arrival_perf_ns
        self.arrival_count = 1
        self.last_updated_perf_ns = first_arrival_perf_ns


class UDPEchoProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None
        self.packet_pair_states = {}

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
            message_type, pair_id_bytes, packet_index, client_send_time_ns = (
                BANDWIDTH_REQUEST_HEADER_STRUCT.unpack_from(data)
            )
        except struct.error:
            return

        arrival_perf_ns = time.perf_counter_ns()
        arrival_index, packet_gap_ns = self._resolve_packet_pair_gap(
            pair_id_bytes=pair_id_bytes,
            arrival_perf_ns=arrival_perf_ns,
        )
        response = BANDWIDTH_RESPONSE_STRUCT.pack(
            message_type,
            pair_id_bytes,
            packet_index,
            arrival_index,
            packet_gap_ns,
        )
        self.transport.sendto(response, addr)

    def _resolve_packet_pair_gap(self, pair_id_bytes: bytes, arrival_perf_ns: int):
        self._cleanup_stale_packet_pair_states(arrival_perf_ns)
        state = self.packet_pair_states.get(pair_id_bytes)

        if state is None:
            self.packet_pair_states[pair_id_bytes] = PacketPairState(arrival_perf_ns)
            return 1, 0

        state.arrival_count += 1
        state.last_updated_perf_ns = arrival_perf_ns
        arrival_index = state.arrival_count
        packet_gap_ns = max(arrival_perf_ns - state.first_arrival_perf_ns, 0)

        if arrival_index >= 2:
            self.packet_pair_states.pop(pair_id_bytes, None)

        return arrival_index, packet_gap_ns

    def _cleanup_stale_packet_pair_states(self, now_perf_ns: int):
        stale_pair_ids = [
            pair_id_bytes
            for pair_id_bytes, state in self.packet_pair_states.items()
            if (now_perf_ns - state.last_updated_perf_ns) > PACKET_PAIR_STATE_TTL_NS
        ]
        for pair_id_bytes in stale_pair_ids:
            self.packet_pair_states.pop(pair_id_bytes, None)


async def start_udp_echo_server(host: str = "0.0.0.0", port: int = 9999):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: UDPEchoProtocol(),
        local_addr=(host, port),
    )
    return transport
