#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "Ws2_32.lib")

namespace {

constexpr uint8_t kLatencyMessageType = 1;
constexpr uint8_t kPTRMessageType = 3;
constexpr uint8_t kPTRStagePacketTrain = 1;
constexpr uint8_t kPTRStagePhaseSummary = 2;
constexpr int64_t kPTRPhaseStateTtlNs = 10'000'000'000LL;
constexpr size_t kMaxDatagramBytes = 4096;

struct ProgramOptions {
    std::string host = "0.0.0.0";
    uint16_t port = 9999;
    bool highPriority = false;
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

struct PTRPhaseKey {
    RoundId roundId;
    uint32_t phaseId = 0;

    bool operator==(const PTRPhaseKey& other) const {
        return roundId == other.roundId && phaseId == other.phaseId;
    }
};

struct PTRPhaseKeyHash {
    size_t operator()(const PTRPhaseKey& key) const {
        const size_t roundHash = RoundIdHash{}(key.roundId);
        const size_t phaseHash = std::hash<uint32_t>{}(key.phaseId);
        return roundHash ^ (phaseHash + 0x9e3779b97f4a7c15ULL + (roundHash << 6) + (roundHash >> 2));
    }
};

struct PTRTrainRecord {
    int64_t recvNs = 0;
};

struct PTRPhaseState {
    int packetSizeBytes = 0;
    int expectedPackets = 0;
    std::unordered_map<uint32_t, PTRTrainRecord> trainRecords;
    int64_t lastUpdatedNs = 0;
};

struct ReceivedDatagram {
    std::vector<uint8_t> data;
    sockaddr_storage remote{};
    int remoteLength = 0;
    int64_t arrivalNs = 0;
};

struct PTRPhaseSummary {
    int receivedPackets = 0;
    int sentPackets = 0;
    uint64_t meanGapNs = 0;
    uint64_t recvDurationNs = 0;
    double lossRate = 0.0;
    double overallRateMbps = 0.0;
    double firstHalfRateMbps = 0.0;
    double secondHalfRateMbps = 0.0;
    uint32_t gapCount = 0;
    uint32_t nonPositiveGapCount = 0;
    uint32_t missingPacketCount = 0;
    uint32_t reorderedPacketCount = 0;
    std::vector<int64_t> gapDetailsNs;
    std::vector<uint32_t> gapStartSequences;
};

uint32_t readU32BE(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
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

bool parseBool(const std::string& value) {
    if (value == "1" || value == "true" || value == "True" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "False" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("Invalid boolean value. Use 0 or 1.");
}

std::string formatGapListUs(
    const std::vector<uint32_t>& startSequences,
    const std::vector<int64_t>& gapsNs
) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < gapsNs.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        if (i < startSequences.size()) {
            stream << startSequences[i] << "->" << (startSequences[i] + 1) << ':';
        }
        stream << (static_cast<double>(gapsNs[i]) / 1000.0) << "us";
    }
    return stream.str();
}

long double meanNs(const std::vector<int64_t>& values) {
    if (values.empty()) {
        return 0.0L;
    }

    const long double sum = std::accumulate(
        values.begin(),
        values.end(),
        0.0L,
        [](long double partial, int64_t value) {
            return partial + static_cast<long double>(value);
        }
    );
    return sum / static_cast<long double>(values.size());
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
                << "[--high-priority 0|1]\n";
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
        if (arg == "--high-priority" && i + 1 < argc) {
            options.highPriority = parseBool(argv[++i]);
            continue;
        }
        throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
    return true;
}

void configureProcessPriority(bool highPriority) {
    if (!highPriority) {
        return;
    }

    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        std::cerr
            << "[NetworkProbe][Winsock] failed to set process priority error="
            << GetLastError()
            << '\n';
    }

    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        std::cerr
            << "[NetworkProbe][Winsock] failed to set thread priority error="
            << GetLastError()
            << '\n';
    }
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

ReceivedDatagram receiveDatagram(SOCKET sock, int64_t qpcFrequency) {
    ReceivedDatagram received;
    received.data.resize(kMaxDatagramBytes);
    received.remoteLength = sizeof(received.remote);

    const int bytesReceived = recvfrom(
        sock,
        reinterpret_cast<char*>(received.data.data()),
        static_cast<int>(received.data.size()),
        0,
        reinterpret_cast<sockaddr*>(&received.remote),
        &received.remoteLength
    );
    received.arrivalNs = currentQpcNs(qpcFrequency);
    if (bytesReceived == SOCKET_ERROR) {
        throw std::runtime_error("recvfrom failed: " + std::to_string(WSAGetLastError()));
    }

    received.data.resize(static_cast<size_t>(bytesReceived));
    return received;
}

class NetworkProbeServer {
public:
    explicit NetworkProbeServer(SOCKET sock)
        : sock_(sock) {}

    void handleDatagram(const ReceivedDatagram& datagram) {
        if (datagram.data.empty()) {
            return;
        }

        const uint8_t messageType = datagram.data[0];
        if (messageType == kLatencyMessageType) {
            handleLatency(datagram);
            return;
        }

        if (messageType == kPTRMessageType) {
            handlePTR(datagram);
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
        uint64_t clientSendTimeNs = 0;
        for (int i = 0; i < 8; ++i) {
            clientSendTimeNs = (clientSendTimeNs << 8) | static_cast<uint64_t>(datagram.data[5 + i]);
        }

        std::vector<uint8_t> response;
        response.reserve(21);
        response.push_back(kLatencyMessageType);
        appendU32BE(response, sequence);
        appendU64BE(response, clientSendTimeNs);
        appendU64BE(response, static_cast<uint64_t>(std::max<int64_t>(datagram.arrivalNs, 0)));
        sendResponse(response, datagram);
    }

    void handlePTR(const ReceivedDatagram& datagram) {
        if (datagram.data.size() < 30) {
            return;
        }

        cleanupStalePTRPhaseStates(datagram.arrivalNs);

        const uint8_t stage = datagram.data[1];
        RoundId roundId{};
        std::memcpy(roundId.bytes.data(), datagram.data.data() + 2, roundId.bytes.size());
        const uint32_t phaseId = readU32BE(datagram.data.data() + 18);
        const uint32_t sequence = readU32BE(datagram.data.data() + 22);
        const uint32_t totalCount = readU32BE(datagram.data.data() + 26);

        if (stage == kPTRStagePacketTrain) {
            recordPTRTrainPacket(
                roundId,
                phaseId,
                sequence,
                totalCount,
                static_cast<int>(datagram.data.size()),
                datagram.arrivalNs
            );
            return;
        }

        if (stage == kPTRStagePhaseSummary) {
            respondWithPTRPhaseSummary(
                roundId,
                phaseId,
                totalCount,
                datagram
            );
        }
    }

    PTRPhaseState& getOrCreatePTRPhaseState(const RoundId& roundId, uint32_t phaseId, int64_t nowNs) {
        PTRPhaseKey key{roundId, phaseId};
        auto iter = ptrPhaseStates_.find(key);
        if (iter == ptrPhaseStates_.end()) {
            PTRPhaseState state;
            state.lastUpdatedNs = nowNs;
            iter = ptrPhaseStates_.emplace(key, std::move(state)).first;
        }
        return iter->second;
    }

    void recordPTRTrainPacket(
        const RoundId& roundId,
        uint32_t phaseId,
        uint32_t sequence,
        uint32_t packetCount,
        int packetSizeBytes,
        int64_t arrivalNs
    ) {
        if (sequence == 0) {
            return;
        }

        PTRPhaseState& state = getOrCreatePTRPhaseState(roundId, phaseId, arrivalNs);
        state.packetSizeBytes = std::max(state.packetSizeBytes, packetSizeBytes);
        state.expectedPackets = std::max<int>(state.expectedPackets, static_cast<int>(packetCount));
        state.trainRecords[sequence] = PTRTrainRecord{arrivalNs};
        state.lastUpdatedNs = arrivalNs;
    }

    void respondWithPTRPhaseSummary(
        const RoundId& roundId,
        uint32_t phaseId,
        uint32_t packetCount,
        const ReceivedDatagram& datagram
    ) {
        bool success = false;
        PTRPhaseSummary summary;
        summary.sentPackets = static_cast<int>(packetCount);

        const PTRPhaseKey key{roundId, phaseId};
        auto iter = ptrPhaseStates_.find(key);
        if (iter != ptrPhaseStates_.end()) {
            PTRPhaseState& state = iter->second;
            state.lastUpdatedNs = datagram.arrivalNs;
            state.expectedPackets = std::max<int>(state.expectedPackets, static_cast<int>(packetCount));
            if (computePTRPhaseSummary(state, summary)) {
                success = true;
                logPTRPhaseSummary(roundId, phaseId, state, summary);
            }
        }

        std::vector<uint8_t> response;
        response.reserve(95);
        response.push_back(kPTRMessageType);
        response.push_back(kPTRStagePhaseSummary);
        response.insert(response.end(), roundId.bytes.begin(), roundId.bytes.end());
        response.push_back(success ? 1 : 0);
        appendU32BE(response, phaseId);
        appendU32BE(response, static_cast<uint32_t>(std::max(summary.receivedPackets, 0)));
        appendU32BE(response, static_cast<uint32_t>(std::max(summary.sentPackets, 0)));
        appendU64BE(response, summary.meanGapNs);
        appendU64BE(response, summary.recvDurationNs);
        appendDoubleBE(response, summary.lossRate);
        appendDoubleBE(response, summary.overallRateMbps);
        appendDoubleBE(response, summary.firstHalfRateMbps);
        appendDoubleBE(response, summary.secondHalfRateMbps);
        appendU32BE(response, summary.gapCount);
        appendU32BE(response, summary.nonPositiveGapCount);
        appendU32BE(response, summary.missingPacketCount);
        appendU32BE(response, summary.reorderedPacketCount);
        sendResponse(response, datagram);
        ptrPhaseStates_.erase(key);
    }

    double computePTRSegmentRateMbps(
        const std::vector<std::pair<uint32_t, PTRTrainRecord>>& records,
        size_t begin,
        size_t end,
        int packetSizeBytes
    ) const {
        if (end <= begin || (end - begin) < 2 || packetSizeBytes <= 0) {
            return 0.0;
        }

        const int64_t durationNs = records[end - 1].second.recvNs - records[begin].second.recvNs;
        if (durationNs <= 0) {
            return 0.0;
        }

        return (
            (static_cast<double>((end - begin) - 1) * static_cast<double>(packetSizeBytes) * 8.0) /
            (static_cast<double>(durationNs) / 1'000'000'000.0) /
            1'000'000.0
        );
    }

    bool computePTRPhaseSummary(const PTRPhaseState& state, PTRPhaseSummary& summary) const {
        if (state.packetSizeBytes <= 0) {
            return false;
        }

        std::vector<std::pair<uint32_t, PTRTrainRecord>> trainRecords;
        trainRecords.reserve(state.trainRecords.size());
        for (const auto& entry : state.trainRecords) {
            trainRecords.push_back(entry);
        }
        std::sort(
            trainRecords.begin(),
            trainRecords.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }
        );

        summary.receivedPackets = static_cast<int>(trainRecords.size());
        summary.sentPackets = std::max(state.expectedPackets, summary.receivedPackets);
        summary.missingPacketCount = summary.sentPackets > 0
            ? static_cast<uint32_t>(std::max(summary.sentPackets - summary.receivedPackets, 0))
            : 0;

        if (summary.receivedPackets < 2) {
            return false;
        }

        std::vector<int64_t> positiveGapsNs;
        std::vector<int64_t> gapDetailsNs;
        std::vector<uint32_t> gapStartSequences;
        for (size_t i = 0; i + 1 < trainRecords.size(); ++i) {
            const int64_t gapNs = trainRecords[i + 1].second.recvNs - trainRecords[i].second.recvNs;
            gapDetailsNs.push_back(gapNs);
            gapStartSequences.push_back(trainRecords[i].first);

            if (gapNs <= 0) {
                ++summary.nonPositiveGapCount;
                if (gapNs < 0) {
                    ++summary.reorderedPacketCount;
                }
                continue;
            }

            positiveGapsNs.push_back(gapNs);
        }

        if (positiveGapsNs.empty()) {
            return false;
        }

        const int64_t recvDurationNs = trainRecords.back().second.recvNs - trainRecords.front().second.recvNs;
        if (recvDurationNs <= 0) {
            return false;
        }

        const long double meanGap = meanNs(positiveGapsNs);
        summary.meanGapNs = static_cast<uint64_t>(meanGap + 0.5L);
        summary.recvDurationNs = static_cast<uint64_t>(recvDurationNs);
        summary.gapCount = static_cast<uint32_t>(positiveGapsNs.size());
        summary.lossRate = summary.sentPackets > 0
            ? static_cast<double>(summary.missingPacketCount) / static_cast<double>(summary.sentPackets)
            : 0.0;
        summary.overallRateMbps =
            (static_cast<double>(summary.receivedPackets - 1) *
             static_cast<double>(state.packetSizeBytes) *
             8.0) /
            (static_cast<double>(recvDurationNs) / 1'000'000'000.0) /
            1'000'000.0;

        const size_t splitIndex = trainRecords.size() / 2;
        summary.firstHalfRateMbps = computePTRSegmentRateMbps(
            trainRecords,
            0,
            splitIndex,
            state.packetSizeBytes
        );
        summary.secondHalfRateMbps = computePTRSegmentRateMbps(
            trainRecords,
            splitIndex,
            trainRecords.size(),
            state.packetSizeBytes
        );
        summary.gapDetailsNs = std::move(gapDetailsNs);
        summary.gapStartSequences = std::move(gapStartSequences);
        return true;
    }

    void logPTRPhaseSummary(
        const RoundId& roundId,
        uint32_t phaseId,
        const PTRPhaseState& state,
        const PTRPhaseSummary& summary
    ) const {
        const std::string roundText = formatRoundId(roundId);
        const std::vector<int64_t>& gapsNs = summary.gapDetailsNs;
        const double meanGapUs = static_cast<double>(summary.meanGapNs) / 1000.0;

        std::cout
            << "[PTR][Winsock][Summary] "
            << "round=" << roundText << ' '
            << "phase=" << phaseId << ' '
            << "payload_bytes=" << state.packetSizeBytes << ' '
            << "received_train=" << summary.receivedPackets << '/' << summary.sentPackets << ' '
            << "PTR=" << std::fixed << std::setprecision(2) << summary.overallRateMbps << "Mbps "
            << "first_half_rate=" << std::fixed << std::setprecision(2) << summary.firstHalfRateMbps << "Mbps "
            << "second_half_rate=" << std::fixed << std::setprecision(2) << summary.secondHalfRateMbps << "Mbps "
            << "gap_count=" << summary.gapCount << ' '
            << "gap_mean=" << meanGapUs << "us "
            << "loss=" << std::fixed << std::setprecision(3) << summary.lossRate << ' '
            << "missing=" << summary.missingPacketCount << ' '
            << "non_positive_gap=" << summary.nonPositiveGapCount << ' '
            << "reordered=" << summary.reorderedPacketCount << ' '
            << "timestamp_source=app_qpc"
            << '\n';

        std::cout
            << "[PTR][Winsock][Detail] "
            << "round=" << roundText << ' '
            << "phase=" << phaseId << ' '
            << "gaps=" << formatGapListUs(summary.gapStartSequences, gapsNs)
            << '\n';
    }

    void cleanupStalePTRPhaseStates(int64_t nowNs) {
        std::vector<PTRPhaseKey> staleKeys;
        for (const auto& entry : ptrPhaseStates_) {
            if ((nowNs - entry.second.lastUpdatedNs) > kPTRPhaseStateTtlNs) {
                staleKeys.push_back(entry.first);
            }
        }
        for (const PTRPhaseKey& key : staleKeys) {
            ptrPhaseStates_.erase(key);
        }
    }

    SOCKET sock_;
    std::unordered_map<PTRPhaseKey, PTRPhaseState, PTRPhaseKeyHash> ptrPhaseStates_;
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

    configureProcessPriority(options.highPriority);

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

        std::cout
            << "[NetworkProbe][Winsock] listening host=" << options.host
            << " port=" << options.port
            << " timestamp_source=app_qpc"
            << " protocols=latency,ptr"
            << " high_priority=" << (options.highPriority ? 1 : 0)
            << '\n';

        NetworkProbeServer server(sock);
        while (true) {
            ReceivedDatagram datagram = receiveDatagram(sock, frequency.QuadPart);
            server.handleDatagram(datagram);
        }
    } catch (const std::exception& error) {
        std::cerr << "[NetworkProbe][Winsock] fatal: " << error.what() << '\n';
        if (sock != INVALID_SOCKET) {
            closesocket(sock);
        }
        WSACleanup();
        return 1;
    }
}
