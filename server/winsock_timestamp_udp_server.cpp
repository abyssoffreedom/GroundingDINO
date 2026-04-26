#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <mstcpip.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma comment(lib, "Ws2_32.lib")

#ifndef SIO_TIMESTAMPING
#define SIO_TIMESTAMPING _WSAIOW(IOC_VENDOR, 235)
#endif

#ifndef TIMESTAMPING_FLAG_RX
#define TIMESTAMPING_FLAG_RX 0x1
#endif

#ifndef SO_TIMESTAMP
#define SO_TIMESTAMP 0x300A
#endif

namespace {

constexpr uint8_t kLatencyMessageType = 1;
constexpr uint8_t kWBestMessageType = 2;
constexpr uint8_t kWBestStagePacketPair = 1;
constexpr uint8_t kWBestStagePacketPairSummary = 2;
constexpr uint8_t kWBestStagePacketTrain = 3;
constexpr uint8_t kWBestStageFinalSummary = 4;
constexpr int64_t kRoundStateTtlNs = 10'000'000'000LL;
constexpr double kSuspiciousPairGapUs = 50.0;
constexpr double kSuspiciousCiLowMbps = 300.0;
constexpr double kSuspiciousCiHighMbps = 1000.0;
constexpr size_t kMaxDatagramBytes = 4096;

struct TimestampingConfigWire {
    ULONG Flags;
    USHORT TxTimestampsBuffered;
    USHORT Reserved;
};

struct ProgramOptions {
    std::string host = "0.0.0.0";
    uint16_t port = 9999;
    double minPairGapUs = 0.0;
    int minValidPairs = 1;
};

struct RoundId {
    std::array<uint8_t, 16> bytes{};

    bool operator==(const RoundId& other) const {
        return bytes == other.bytes;
    }
};

struct RoundIdHash {
    size_t operator()(const RoundId& id) const {
        uint64_t first = 0;
        uint64_t second = 0;
        std::memcpy(&first, id.bytes.data(), sizeof(first));
        std::memcpy(&second, id.bytes.data() + sizeof(first), sizeof(second));
        return static_cast<size_t>(first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6) + (first >> 2)));
    }
};

struct PairRecord {
    bool hasSeq1 = false;
    bool hasSeq2 = false;
    bool hasRecv1 = false;
    bool hasRecv2 = false;
    bool recv1Timestamped = false;
    bool recv2Timestamped = false;
    uint32_t seq1 = 0;
    uint32_t seq2 = 0;
    int64_t recv1Ns = 0;
    int64_t recv2Ns = 0;
};

struct TrainRecord {
    int64_t recvNs = 0;
    bool timestamped = false;
};

struct RoundState {
    int packetSizeBytes = 0;
    std::unordered_map<uint32_t, PairRecord> packetPairRecords;
    std::unordered_map<uint32_t, TrainRecord> trainRecords;
    int validPairSampleCount = 0;
    bool hasEffectiveCapacity = false;
    double effectiveCapacityMbps = 0.0;
    int expectedTrainPackets = 0;
    int64_t lastUpdatedNs = 0;
};

struct ReceivedDatagram {
    std::vector<uint8_t> data;
    sockaddr_storage remote{};
    int remoteLength = 0;
    int64_t arrivalNs = 0;
    bool hasSocketTimestamp = false;
};

struct PairDetail {
    uint32_t pairId = 0;
    uint32_t expectedSeq1 = 0;
    uint32_t expectedSeq2 = 0;
    bool hasBoth = false;
    bool reordered = false;
    bool seqMismatch = false;
    bool nonPositiveGap = false;
    bool gapUnder50Us = false;
    bool ciInSuspiciousRange = false;
    bool usedTimestamp = false;
    bool fallbackTimestamp = false;
    int64_t gapNs = 0;
    double gapUs = 0.0;
    double ciMbps = 0.0;
    bool hasCi = false;
    PairRecord record;
};

struct PairAnalysis {
    std::vector<PairDetail> details;
    std::vector<double> capacitySamplesMbps;
    int gapUnder50UsCount = 0;
    int nonPositiveGapCount = 0;
    int ci300To1000Count = 0;
    int missingPairCount = 0;
    int reorderedPairCount = 0;
    int seqMismatchCount = 0;
    int socketTimestampedPairCount = 0;
    int fallbackTimestampPairCount = 0;
};

struct FinalSummary {
    int receivedTrainPackets = 0;
    int sentTrainPackets = 0;
    uint64_t meanTrainGapNs = 0;
    double effectiveCapacityMbps = 0.0;
    double achievableThroughputMbps = 0.0;
    double availableBandwidthMbps = 0.0;
    double correctedAvailableBandwidthMbps = 0.0;
    double lossRate = 0.0;
};

uint32_t readU32BE(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

uint64_t readU64BE(const uint8_t* data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | static_cast<uint64_t>(data[i]);
    }
    return value;
}

void appendU32BE(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(value & 0xff));
}

void appendU64BE(std::vector<uint8_t>& out, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
    }
}

void appendDoubleBE(std::vector<uint8_t>& out, double value) {
    static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit IEEE-754");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU64BE(out, bits);
}

std::string formatRoundId(const RoundId& roundId) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t i = 0; i < roundId.bytes.size(); ++i) {
        stream << std::setw(2) << static_cast<int>(roundId.bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            stream << '-';
        }
    }
    return stream.str();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() % 2) == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
}

int64_t qpcToNs(uint64_t qpcValue, int64_t qpcFrequency) {
    const long double ns =
        (static_cast<long double>(qpcValue) * 1'000'000'000.0L) /
        static_cast<long double>(qpcFrequency);
    if (ns > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(ns);
}

int64_t currentQpcNs(int64_t qpcFrequency) {
    LARGE_INTEGER counter{};
    QueryPerformanceCounter(&counter);
    return qpcToNs(static_cast<uint64_t>(counter.QuadPart), qpcFrequency);
}

bool parseOptions(int argc, char** argv, ProgramOptions& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: winsock_timestamp_udp_server.exe "
                << "[--host 0.0.0.0] [--port 9999] "
                << "[--min-pair-gap-us 0] [--min-valid-pairs 1]\n";
            return false;
        }
        if (arg == "--host" && i + 1 < argc) {
            options.host = argv[++i];
            continue;
        }
        if (arg == "--port" && i + 1 < argc) {
            const int port = std::stoi(argv[++i]);
            if (port <= 0 || port > 65535) {
                throw std::runtime_error("Invalid port");
            }
            options.port = static_cast<uint16_t>(port);
            continue;
        }
        if (arg == "--min-pair-gap-us" && i + 1 < argc) {
            options.minPairGapUs = std::stod(argv[++i]);
            if (options.minPairGapUs < 0.0) {
                throw std::runtime_error("Invalid min pair gap");
            }
            continue;
        }
        if (arg == "--min-valid-pairs" && i + 1 < argc) {
            options.minValidPairs = std::stoi(argv[++i]);
            if (options.minValidPairs < 1) {
                throw std::runtime_error("Invalid minimum valid pair count");
            }
            continue;
        }
        throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
    return true;
}

SOCKET makeBoundSocket(const ProgramOptions& options) {
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        throw std::runtime_error("socket failed: " + std::to_string(WSAGetLastError()));
    }

    int recvBufferBytes = 4 * 1024 * 1024;
    setsockopt(
        sock,
        SOL_SOCKET,
        SO_RCVBUF,
        reinterpret_cast<const char*>(&recvBufferBytes),
        sizeof(recvBufferBytes)
    );

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(options.port);
    if (options.host == "0.0.0.0" || options.host == "*") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, options.host.c_str(), &addr.sin_addr) != 1) {
        closesocket(sock);
        throw std::runtime_error("Only IPv4 host addresses are supported by this helper");
    }

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        const int error = WSAGetLastError();
        closesocket(sock);
        throw std::runtime_error("bind failed: " + std::to_string(error));
    }

    return sock;
}

LPFN_WSARECVMSG getRecvMsgFunction(SOCKET sock) {
    GUID recvMsgGuid = WSAID_WSARECVMSG;
    LPFN_WSARECVMSG recvMsg = nullptr;
    DWORD bytesReturned = 0;
    const int result = WSAIoctl(
        sock,
        SIO_GET_EXTENSION_FUNCTION_POINTER,
        &recvMsgGuid,
        sizeof(recvMsgGuid),
        &recvMsg,
        sizeof(recvMsg),
        &bytesReturned,
        nullptr,
        nullptr
    );
    if (result == SOCKET_ERROR || recvMsg == nullptr) {
        throw std::runtime_error("SIO_GET_EXTENSION_FUNCTION_POINTER failed: " + std::to_string(WSAGetLastError()));
    }
    return recvMsg;
}

bool enableReceiveTimestamping(SOCKET sock) {
    TimestampingConfigWire config{};
    config.Flags = TIMESTAMPING_FLAG_RX;
    DWORD bytesReturned = 0;
    const int result = WSAIoctl(
        sock,
        SIO_TIMESTAMPING,
        &config,
        sizeof(config),
        nullptr,
        0,
        &bytesReturned,
        nullptr,
        nullptr
    );
    if (result == SOCKET_ERROR) {
        std::cerr
            << "[WBest][Winsock] SIO_TIMESTAMPING failed error="
            << WSAGetLastError()
            << "; falling back to app-level QPC receive timestamps.\n";
        return false;
    }
    std::cout << "[WBest][Winsock] SIO_TIMESTAMPING RX enabled.\n";
    return true;
}

ReceivedDatagram receiveDatagram(
    SOCKET sock,
    LPFN_WSARECVMSG recvMsg,
    int64_t qpcFrequency
) {
    ReceivedDatagram received;
    received.data.resize(kMaxDatagramBytes);
    std::array<char, 128> control{};

    WSABUF dataBuf{};
    dataBuf.buf = reinterpret_cast<char*>(received.data.data());
    dataBuf.len = static_cast<ULONG>(received.data.size());

    WSABUF controlBuf{};
    controlBuf.buf = control.data();
    controlBuf.len = static_cast<ULONG>(control.size());

    received.remoteLength = sizeof(received.remote);

    WSAMSG message{};
    message.name = reinterpret_cast<sockaddr*>(&received.remote);
    message.namelen = received.remoteLength;
    message.lpBuffers = &dataBuf;
    message.dwBufferCount = 1;
    message.Control = controlBuf;
    message.dwFlags = 0;

    DWORD bytesReceived = 0;
    const int result = recvMsg(sock, &message, &bytesReceived, nullptr, nullptr);
    const int64_t appReceiveNs = currentQpcNs(qpcFrequency);
    if (result == SOCKET_ERROR) {
        throw std::runtime_error("WSARecvMsg failed: " + std::to_string(WSAGetLastError()));
    }

    received.remoteLength = message.namelen;
    received.data.resize(bytesReceived);
    received.arrivalNs = appReceiveNs;

    for (WSACMSGHDR* cmsg = WSA_CMSG_FIRSTHDR(&message);
         cmsg != nullptr;
         cmsg = WSA_CMSG_NXTHDR(&message, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SO_TIMESTAMP) {
            if (cmsg->cmsg_len >= WSA_CMSG_LEN(sizeof(uint64_t))) {
                uint64_t socketTimestamp = 0;
                std::memcpy(&socketTimestamp, WSA_CMSG_DATA(cmsg), sizeof(socketTimestamp));
                received.arrivalNs = qpcToNs(socketTimestamp, qpcFrequency);
                received.hasSocketTimestamp = true;
                break;
            }
        }
    }

    return received;
}

class WBestServer {
public:
    WBestServer(SOCKET sock, ProgramOptions options, int64_t qpcFrequency)
        : sock_(sock), options_(std::move(options)), qpcFrequency_(qpcFrequency) {}

    void handleDatagram(const ReceivedDatagram& datagram) {
        if (datagram.data.empty()) {
            return;
        }

        const uint8_t messageType = datagram.data[0];
        if (messageType == kLatencyMessageType) {
            handleLatency(datagram);
            return;
        }

        if (messageType == kWBestMessageType) {
            handleWBest(datagram);
        }
    }

private:
    void sendResponse(const std::vector<uint8_t>& response, const ReceivedDatagram& datagram) {
        sendto(
            sock_,
            reinterpret_cast<const char*>(response.data()),
            static_cast<int>(response.size()),
            0,
            reinterpret_cast<const sockaddr*>(&datagram.remote),
            datagram.remoteLength
        );
    }

    void handleLatency(const ReceivedDatagram& datagram) {
        if (datagram.data.size() != 13) {
            return;
        }

        const uint32_t sequence = readU32BE(datagram.data.data() + 1);
        const uint64_t clientSendTimeNs = readU64BE(datagram.data.data() + 5);

        std::vector<uint8_t> response;
        response.reserve(21);
        response.push_back(kLatencyMessageType);
        appendU32BE(response, sequence);
        appendU64BE(response, clientSendTimeNs);
        appendU64BE(response, static_cast<uint64_t>(std::max<int64_t>(datagram.arrivalNs, 0)));
        sendResponse(response, datagram);
    }

    void handleWBest(const ReceivedDatagram& datagram) {
        if (datagram.data.size() < 31) {
            return;
        }

        cleanupStaleRoundStates(datagram.arrivalNs);

        const uint8_t stage = datagram.data[1];
        RoundId roundId{};
        std::memcpy(roundId.bytes.data(), datagram.data.data() + 2, roundId.bytes.size());
        const uint32_t sequence = readU32BE(datagram.data.data() + 18);
        const uint32_t groupId = readU32BE(datagram.data.data() + 22);
        const uint32_t totalCount = readU32BE(datagram.data.data() + 26);
        const uint8_t indexInGroup = datagram.data[30];

        if (stage == kWBestStagePacketPair) {
            recordPacketPair(
                roundId,
                groupId,
                sequence,
                indexInGroup,
                static_cast<int>(datagram.data.size()),
                datagram.arrivalNs,
                datagram.hasSocketTimestamp
            );
            return;
        }

        if (stage == kWBestStagePacketPairSummary) {
            respondWithPairSummary(roundId, datagram);
            return;
        }

        if (stage == kWBestStagePacketTrain) {
            recordPacketTrain(
                roundId,
                sequence,
                totalCount,
                static_cast<int>(datagram.data.size()),
                datagram.arrivalNs,
                datagram.hasSocketTimestamp
            );
            return;
        }

        if (stage == kWBestStageFinalSummary) {
            respondWithFinalSummary(roundId, totalCount, datagram);
        }
    }

    RoundState& getOrCreateRoundState(const RoundId& roundId, int64_t nowNs) {
        auto iter = roundStates_.find(roundId);
        if (iter == roundStates_.end()) {
            RoundState state;
            state.lastUpdatedNs = nowNs;
            iter = roundStates_.emplace(roundId, std::move(state)).first;
        }
        return iter->second;
    }

    void recordPacketPair(
        const RoundId& roundId,
        uint32_t pairId,
        uint32_t sequence,
        uint8_t indexInPair,
        int packetSizeBytes,
        int64_t arrivalNs,
        bool hasSocketTimestamp
    ) {
        if (indexInPair != 1 && indexInPair != 2) {
            return;
        }

        RoundState& state = getOrCreateRoundState(roundId, arrivalNs);
        state.packetSizeBytes = std::max(state.packetSizeBytes, packetSizeBytes);
        state.lastUpdatedNs = arrivalNs;

        PairRecord& record = state.packetPairRecords[pairId];
        if (indexInPair == 1) {
            record.hasSeq1 = true;
            record.hasRecv1 = true;
            record.seq1 = sequence;
            record.recv1Ns = arrivalNs;
            record.recv1Timestamped = hasSocketTimestamp;
        } else {
            record.hasSeq2 = true;
            record.hasRecv2 = true;
            record.seq2 = sequence;
            record.recv2Ns = arrivalNs;
            record.recv2Timestamped = hasSocketTimestamp;
        }
    }

    void respondWithPairSummary(const RoundId& roundId, const ReceivedDatagram& datagram) {
        bool success = false;
        uint32_t validPairSampleCount = 0;
        double effectiveCapacityMbps = 0.0;

        auto iter = roundStates_.find(roundId);
        if (iter != roundStates_.end()) {
            RoundState& state = iter->second;
            state.lastUpdatedNs = datagram.arrivalNs;
            const PairAnalysis analysis = buildPairAnalysis(state);
            logPairAnalysis(roundId, state, analysis);
            if (static_cast<int>(analysis.capacitySamplesMbps.size()) >= options_.minValidPairs) {
                success = true;
                validPairSampleCount = static_cast<uint32_t>(analysis.capacitySamplesMbps.size());
                effectiveCapacityMbps = median(analysis.capacitySamplesMbps);
                state.hasEffectiveCapacity = true;
                state.effectiveCapacityMbps = effectiveCapacityMbps;
                state.validPairSampleCount = static_cast<int>(validPairSampleCount);
            }
        }

        std::vector<uint8_t> response;
        response.reserve(31);
        response.push_back(kWBestMessageType);
        response.push_back(kWBestStagePacketPairSummary);
        response.insert(response.end(), roundId.bytes.begin(), roundId.bytes.end());
        response.push_back(success ? 1 : 0);
        appendU32BE(response, validPairSampleCount);
        appendDoubleBE(response, effectiveCapacityMbps);
        sendResponse(response, datagram);
    }

    void recordPacketTrain(
        const RoundId& roundId,
        uint32_t sequence,
        uint32_t packetCount,
        int packetSizeBytes,
        int64_t arrivalNs,
        bool hasSocketTimestamp
    ) {
        RoundState& state = getOrCreateRoundState(roundId, arrivalNs);
        state.packetSizeBytes = std::max(state.packetSizeBytes, packetSizeBytes);
        state.expectedTrainPackets = std::max<int>(state.expectedTrainPackets, static_cast<int>(packetCount));
        state.lastUpdatedNs = arrivalNs;
        if (sequence > 0) {
            state.trainRecords[sequence] = TrainRecord{arrivalNs, hasSocketTimestamp};
        }
    }

    void respondWithFinalSummary(
        const RoundId& roundId,
        uint32_t packetCount,
        const ReceivedDatagram& datagram
    ) {
        bool success = false;
        FinalSummary summary;

        auto iter = roundStates_.find(roundId);
        if (iter != roundStates_.end()) {
            RoundState& state = iter->second;
            state.lastUpdatedNs = datagram.arrivalNs;
            state.expectedTrainPackets = std::max<int>(state.expectedTrainPackets, static_cast<int>(packetCount));

            if (!state.hasEffectiveCapacity) {
                const PairAnalysis analysis = buildPairAnalysis(state);
                if (static_cast<int>(analysis.capacitySamplesMbps.size()) >= options_.minValidPairs) {
                    state.hasEffectiveCapacity = true;
                    state.effectiveCapacityMbps = median(analysis.capacitySamplesMbps);
                    state.validPairSampleCount = static_cast<int>(analysis.capacitySamplesMbps.size());
                }
            }

            if (computeFinalSummary(state, summary)) {
                success = true;
            }
        }

        std::vector<uint8_t> response;
        response.reserve(75);
        response.push_back(kWBestMessageType);
        response.push_back(kWBestStageFinalSummary);
        response.insert(response.end(), roundId.bytes.begin(), roundId.bytes.end());
        response.push_back(success ? 1 : 0);
        appendU32BE(response, static_cast<uint32_t>(summary.receivedTrainPackets));
        appendU32BE(response, static_cast<uint32_t>(summary.sentTrainPackets));
        appendU64BE(response, summary.meanTrainGapNs);
        appendDoubleBE(response, summary.effectiveCapacityMbps);
        appendDoubleBE(response, summary.achievableThroughputMbps);
        appendDoubleBE(response, summary.availableBandwidthMbps);
        appendDoubleBE(response, summary.correctedAvailableBandwidthMbps);
        appendDoubleBE(response, summary.lossRate);
        sendResponse(response, datagram);
        roundStates_.erase(roundId);
    }

    PairAnalysis buildPairAnalysis(const RoundState& state) const {
        PairAnalysis analysis;
        std::vector<uint32_t> pairIds;
        pairIds.reserve(state.packetPairRecords.size());
        for (const auto& entry : state.packetPairRecords) {
            pairIds.push_back(entry.first);
        }
        std::sort(pairIds.begin(), pairIds.end());

        const int64_t minGapNs = static_cast<int64_t>(options_.minPairGapUs * 1000.0);
        for (const uint32_t pairId : pairIds) {
            const PairRecord& record = state.packetPairRecords.at(pairId);
            PairDetail detail;
            detail.pairId = pairId;
            detail.expectedSeq1 = (2 * pairId) - 1;
            detail.expectedSeq2 = 2 * pairId;
            detail.record = record;
            detail.hasBoth = record.hasRecv1 && record.hasRecv2;
            detail.usedTimestamp = detail.hasBoth && record.recv1Timestamped && record.recv2Timestamped;
            detail.fallbackTimestamp = detail.hasBoth && (!record.recv1Timestamped || !record.recv2Timestamped);

            if (detail.hasBoth) {
                detail.reordered = record.recv2Ns < record.recv1Ns;
                detail.seqMismatch =
                    record.hasSeq1 && record.hasSeq2 &&
                    (record.seq1 != detail.expectedSeq1 || record.seq2 != detail.expectedSeq2);
                detail.gapNs = record.recv2Ns - record.recv1Ns;
                detail.gapUs = static_cast<double>(detail.gapNs) / 1000.0;
                detail.nonPositiveGap = detail.gapNs <= 0;
                detail.gapUnder50Us = detail.gapNs > 0 && detail.gapUs < kSuspiciousPairGapUs;
                if (detail.gapNs > minGapNs) {
                    const double gapSeconds = static_cast<double>(detail.gapNs) / 1'000'000'000.0;
                    detail.ciMbps =
                        (static_cast<double>(state.packetSizeBytes) * 8.0) /
                        gapSeconds /
                        1'000'000.0;
                    detail.hasCi = true;
                    analysis.capacitySamplesMbps.push_back(detail.ciMbps);
                    detail.ciInSuspiciousRange =
                        detail.ciMbps >= kSuspiciousCiLowMbps &&
                        detail.ciMbps <= kSuspiciousCiHighMbps;
                }
            }

            if (!detail.hasBoth) {
                ++analysis.missingPairCount;
            }
            if (detail.reordered) {
                ++analysis.reorderedPairCount;
            }
            if (detail.seqMismatch) {
                ++analysis.seqMismatchCount;
            }
            if (detail.nonPositiveGap) {
                ++analysis.nonPositiveGapCount;
            }
            if (detail.gapUnder50Us) {
                ++analysis.gapUnder50UsCount;
            }
            if (detail.ciInSuspiciousRange) {
                ++analysis.ci300To1000Count;
            }
            if (detail.usedTimestamp) {
                ++analysis.socketTimestampedPairCount;
            }
            if (detail.fallbackTimestamp) {
                ++analysis.fallbackTimestampPairCount;
            }
            analysis.details.push_back(detail);
        }

        return analysis;
    }

    bool computeFinalSummary(const RoundState& state, FinalSummary& summary) const {
        if (state.packetSizeBytes <= 0 || !state.hasEffectiveCapacity || state.effectiveCapacityMbps <= 0.0) {
            return false;
        }

        std::vector<std::pair<uint32_t, TrainRecord>> trainRecords;
        trainRecords.reserve(state.trainRecords.size());
        for (const auto& entry : state.trainRecords) {
            trainRecords.push_back(entry);
        }
        std::sort(
            trainRecords.begin(),
            trainRecords.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }
        );

        summary.receivedTrainPackets = static_cast<int>(trainRecords.size());
        if (summary.receivedTrainPackets < 2) {
            return false;
        }

        std::vector<int64_t> gapsNs;
        for (size_t i = 0; i + 1 < trainRecords.size(); ++i) {
            const int64_t gapNs = trainRecords[i + 1].second.recvNs - trainRecords[i].second.recvNs;
            if (gapNs > 0) {
                gapsNs.push_back(gapNs);
            }
        }
        if (gapsNs.empty()) {
            return false;
        }

        long double gapSum = 0.0;
        for (const int64_t gapNs : gapsNs) {
            gapSum += static_cast<long double>(gapNs);
        }
        const long double meanGapNs = gapSum / static_cast<long double>(gapsNs.size());
        const double meanGapSeconds = static_cast<double>(meanGapNs / 1'000'000'000.0L);

        summary.meanTrainGapNs = static_cast<uint64_t>(meanGapNs + 0.5L);
        summary.effectiveCapacityMbps = state.effectiveCapacityMbps;
        summary.achievableThroughputMbps =
            (static_cast<double>(state.packetSizeBytes) * 8.0) /
            meanGapSeconds /
            1'000'000.0;
        summary.achievableThroughputMbps =
            std::min(summary.achievableThroughputMbps, summary.effectiveCapacityMbps);

        if (summary.achievableThroughputMbps >= (summary.effectiveCapacityMbps / 2.0)) {
            summary.availableBandwidthMbps =
                summary.effectiveCapacityMbps *
                (2.0 - (summary.effectiveCapacityMbps / summary.achievableThroughputMbps));
        } else {
            summary.availableBandwidthMbps = 0.0;
        }
        summary.availableBandwidthMbps = std::max(0.0, summary.availableBandwidthMbps);

        summary.sentTrainPackets = std::max(state.expectedTrainPackets, summary.receivedTrainPackets);
        if (summary.sentTrainPackets > 0) {
            summary.lossRate =
                static_cast<double>(std::max(summary.sentTrainPackets - summary.receivedTrainPackets, 0)) /
                static_cast<double>(summary.sentTrainPackets);
        }
        summary.correctedAvailableBandwidthMbps =
            summary.availableBandwidthMbps * (1.0 - summary.lossRate);
        return true;
    }

    void logPairAnalysis(const RoundId& roundId, const RoundState& state, const PairAnalysis& analysis) const {
        const std::string roundText = formatRoundId(roundId);
        const std::string ceText =
            analysis.capacitySamplesMbps.empty()
                ? "n/a"
                : ([&]() {
                      std::ostringstream stream;
                      stream << std::fixed << std::setprecision(2)
                             << median(analysis.capacitySamplesMbps) << "Mbps";
                      return stream.str();
                  })();

        std::cout
            << "[WBest][Winsock][Summary] "
            << "round=" << roundText << ' '
            << "payload_bytes=" << state.packetSizeBytes << ' '
            << "valid_pairs=" << analysis.capacitySamplesMbps.size() << '/' << analysis.details.size() << ' '
            << "Ce=" << ceText << ' '
            << "gap_lt_50us=" << analysis.gapUnder50UsCount << ' '
            << "non_positive_gap=" << analysis.nonPositiveGapCount << ' '
            << "ci_300_1000=" << analysis.ci300To1000Count << ' '
            << "missing=" << analysis.missingPairCount << ' '
            << "reordered=" << analysis.reorderedPairCount << ' '
            << "seq_mismatch=" << analysis.seqMismatchCount << ' '
            << "socket_timestamped_pairs=" << analysis.socketTimestampedPairCount << ' '
            << "fallback_timestamp_pairs=" << analysis.fallbackTimestampPairCount << ' '
            << "min_pair_gap_us=" << options_.minPairGapUs
            << '\n';

        for (const PairDetail& detail : analysis.details) {
            std::vector<std::string> flags;
            if (detail.gapUnder50Us) {
                flags.emplace_back("gap_lt_50us");
            }
            if (detail.nonPositiveGap) {
                flags.emplace_back("non_positive_gap");
            }
            if (detail.ciInSuspiciousRange) {
                flags.emplace_back("ci_300_1000");
            }
            if (detail.reordered) {
                flags.emplace_back("reordered");
            }
            if (detail.seqMismatch) {
                flags.emplace_back("seq_mismatch");
            }
            if (!detail.hasBoth) {
                flags.emplace_back("incomplete");
            }
            if (detail.fallbackTimestamp) {
                flags.emplace_back("fallback_timestamp");
            }

            std::ostringstream flagStream;
            if (flags.empty()) {
                flagStream << "ok";
            } else {
                for (size_t i = 0; i < flags.size(); ++i) {
                    if (i > 0) {
                        flagStream << ',';
                    }
                    flagStream << flags[i];
                }
            }

            std::cout
                << "[WBest][Winsock][Detail] "
                << "round=" << roundText << ' '
                << "pair=" << std::setw(2) << std::setfill('0') << detail.pairId << std::setfill(' ') << ' '
                << "seq=("
                << (detail.record.hasSeq1 ? std::to_string(detail.record.seq1) : "None") << ','
                << (detail.record.hasSeq2 ? std::to_string(detail.record.seq2) : "None") << ") "
                << "expected=(" << detail.expectedSeq1 << ',' << detail.expectedSeq2 << ") ";

            if (detail.hasBoth) {
                std::cout << "gap=" << std::fixed << std::setprecision(2) << detail.gapUs << "us ";
            } else {
                std::cout << "gap=n/a ";
            }
            if (detail.hasCi) {
                std::cout << "Ci=" << std::fixed << std::setprecision(2) << detail.ciMbps << "Mbps ";
            } else {
                std::cout << "Ci=n/a ";
            }
            std::cout
                << "timestamped=(" << (detail.record.recv1Timestamped ? 1 : 0)
                << ',' << (detail.record.recv2Timestamped ? 1 : 0) << ") "
                << "flags=" << flagStream.str()
                << '\n';
        }
    }

    void cleanupStaleRoundStates(int64_t nowNs) {
        std::vector<RoundId> staleRoundIds;
        for (const auto& entry : roundStates_) {
            if ((nowNs - entry.second.lastUpdatedNs) > kRoundStateTtlNs) {
                staleRoundIds.push_back(entry.first);
            }
        }
        for (const RoundId& roundId : staleRoundIds) {
            roundStates_.erase(roundId);
        }
    }

    SOCKET sock_;
    ProgramOptions options_;
    int64_t qpcFrequency_;
    std::unordered_map<RoundId, RoundState, RoundIdHash> roundStates_;
};

} // namespace

int main(int argc, char** argv) {
    ProgramOptions options;
    try {
        if (!parseOptions(argc, argv, options)) {
            return 0;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }

    WSADATA wsaData{};
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed\n";
        return 1;
    }

    SOCKET sock = INVALID_SOCKET;
    try {
        LARGE_INTEGER frequency{};
        QueryPerformanceFrequency(&frequency);
        if (frequency.QuadPart <= 0) {
            throw std::runtime_error("QueryPerformanceFrequency failed");
        }

        sock = makeBoundSocket(options);
        LPFN_WSARECVMSG recvMsg = getRecvMsgFunction(sock);
        const bool timestampingEnabled = enableReceiveTimestamping(sock);

        std::cout
            << "[WBest][Winsock] listening host=" << options.host
            << " port=" << options.port
            << " timestamping_enabled=" << (timestampingEnabled ? 1 : 0)
            << " min_pair_gap_us=" << options.minPairGapUs
            << " min_valid_pairs=" << options.minValidPairs
            << '\n';

        WBestServer server(sock, options, frequency.QuadPart);
        while (true) {
            ReceivedDatagram datagram = receiveDatagram(sock, recvMsg, frequency.QuadPart);
            server.handleDatagram(datagram);
        }
    } catch (const std::exception& error) {
        std::cerr << "[WBest][Winsock] fatal: " << error.what() << '\n';
        if (sock != INVALID_SOCKET) {
            closesocket(sock);
        }
        WSACleanup();
        return 1;
    }
}
