import asyncio
import struct
import time

REQUEST_STRUCT = struct.Struct("!BIQ")
RESPONSE_STRUCT = struct.Struct("!BIQQ")


class UDPEchoProtocol(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            version, sequence, client_send_time_ns = REQUEST_STRUCT.unpack(data)
        except struct.error:
            return

        server_receive_time_ns = time.perf_counter_ns()
        response = RESPONSE_STRUCT.pack(
            version,
            sequence,
            client_send_time_ns,
            server_receive_time_ns,
        )
        self.transport.sendto(response, addr)


async def start_udp_echo_server(host: str = "0.0.0.0", port: int = 9999):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: UDPEchoProtocol(),
        local_addr=(host, port),
    )
    return transport
