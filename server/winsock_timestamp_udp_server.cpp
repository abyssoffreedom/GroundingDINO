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
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma comment(lib, "Ws2_32.lib")

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
constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kMinGaussianVariance = 1e-6;

enum class TrainGapAggregation {
    Mean,
    TrimmedMean,
    Median,
};

struct ProgramOptions {
    std::string host = "0.0.0.0";
    uint16_t port = 9999;
    double minPairGapUs = 0.0;
    int minValidPairs = 1;
    TrainGapAggregation trainGapAggregation = TrainGapAggregation::TrimmedMean;
    double trainGapTrimRatio = 0.10;
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

struct PairRecord {
    bool hasSeq1 = false;
    bool hasSeq2 = false;
    bool hasRecv1 = false;
    bool hasRecv2 = false;
    uint32_t seq1 = 0;
    uint32_t seq2 = 0;
    int64_t recv1Ns = 0;
    int64_t recv2Ns = 0;
};

struct TrainRecord {
    int64_t recvNs = 0;
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
};

struct FinalSummary {
    int receivedTrainPackets = 0;
    int sentTrainPackets = 0;
    uint64_t meanTrainGapNs = 0;
    double rawEffectiveCapacityMbps = 0.0;
    double shaperCapacityMbps = 0.0;
    bool shaperDetected = false;
    bool receiveQueueCompressionDetected = false;
    double receiveQueueCompressionRatio = 0.0;
    double clientActualSendRateMbps = 0.0;
    double shaperSingleModelBic = 0.0;
    double shaperTokenBucketModelBic = 0.0;
    double shaperBicImprovement = 0.0;
    double effectiveCapacityMbps = 0.0;
    double achievableThroughputMbps = 0.0;
    double availableBandwidthMbps = 0.0;
    double correctedAvailableBandwidthMbps = 0.0;
    double lossRate = 0.0;
    uint64_t calculationTrainGapNs = 0;
    double rawMeanTrainGapNs = 0.0;
    std::vector<int64_t> trainGapsNs;
    std::vector<uint32_t> trainGapStartSequences;
};

struct FinalSummaryRequestDiagnostics {
    uint64_t clientActualMeanSendGapNs = 0;
    uint64_t clientActualMedianSendGapNs = 0;
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

std::string trainGapAggregationName(TrainGapAggregation aggregation) {
    switch (aggregation) {
    case TrainGapAggregation::Mean:
        return "mean";
    case TrainGapAggregation::TrimmedMean:
        return "trimmed_mean";
    case TrainGapAggregation::Median:
        return "median";
    }
    return "unknown";
}

TrainGapAggregation parseTrainGapAggregation(const std::string& value) {
    if (value == "mean") {
        return TrainGapAggregation::Mean;
    }
    if (value == "trimmed_mean" || value == "trimmed-mean") {
        return TrainGapAggregation::TrimmedMean;
    }
    if (value == "median") {
        return TrainGapAggregation::Median;
    }
    throw std::runtime_error("Invalid train gap aggregation. Use mean, trimmed_mean, or median.");
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

double percentileNsToUs(std::vector<int64_t> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const double clampedP = std::min(std::max(p, 0.0), 1.0);
    const double rawIndex = clampedP * static_cast<double>(values.size() - 1);
    const size_t lowerIndex = static_cast<size_t>(std::floor(rawIndex));
    const size_t upperIndex = static_cast<size_t>(std::ceil(rawIndex));
    if (lowerIndex == upperIndex) {
        return static_cast<double>(values[lowerIndex]) / 1000.0;
    }

    const double fraction = rawIndex - static_cast<double>(lowerIndex);
    const double lowerValue = static_cast<double>(values[lowerIndex]);
    const double upperValue = static_cast<double>(values[upperIndex]);
    return (lowerValue + ((upperValue - lowerValue) * fraction)) / 1000.0;
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

long double aggregateTrainGapNs(
    std::vector<int64_t> values,
    TrainGapAggregation aggregation,
    double trimRatio
) {
    if (values.empty()) {
        return 0.0L;
    }

    if (aggregation == TrainGapAggregation::Mean) {
        return meanNs(values);
    }

    std::sort(values.begin(), values.end());

    if (aggregation == TrainGapAggregation::Median) {
        const size_t mid = values.size() / 2;
        if ((values.size() % 2) == 0) {
            return (static_cast<long double>(values[mid - 1]) + static_cast<long double>(values[mid])) / 2.0L;
        }
        return static_cast<long double>(values[mid]);
    }

    const double clampedTrimRatio = std::min(std::max(trimRatio, 0.0), 0.45);
    const size_t trimCount = std::min(
        static_cast<size_t>(std::floor(static_cast<double>(values.size()) * clampedTrimRatio)),
        (values.size() - 1) / 2
    );
    const auto begin = values.begin() + static_cast<std::ptrdiff_t>(trimCount);
    const auto end = values.end() - static_cast<std::ptrdiff_t>(trimCount);
    const std::vector<int64_t> trimmed(begin, end);
    return meanNs(trimmed);
}

double gaussianLogPdf(double value, double mean, double variance) {
    const double safeVariance = std::max(variance, kMinGaussianVariance);
    const double diff = value - mean;
    return -0.5 * (std::log(kTwoPi * safeVariance) + ((diff * diff) / safeVariance));
}

double logSumExp(double left, double right) {
    const double maxValue = std::max(left, right);
    return maxValue + std::log(std::exp(left - maxValue) + std::exp(right - maxValue));
}

double bicFromLogLikelihood(size_t sampleCount, int parameterCount, double logLikelihood) {
    return (static_cast<double>(parameterCount) * std::log(static_cast<double>(sampleCount))) -
           (2.0 * logLikelihood);
}

bool singleLogNormalGapBic(const std::vector<int64_t>& gapsNs, double& bic) {
    if (gapsNs.size() < 6) {
        return false;
    }

    std::vector<double> logGaps;
    logGaps.reserve(gapsNs.size());
    for (int64_t gapNs : gapsNs) {
        if (gapNs > 0) {
            logGaps.push_back(std::log(static_cast<double>(gapNs)));
        }
    }
    if (logGaps.size() < 6) {
        return false;
    }

    const double mean =
        std::accumulate(logGaps.begin(), logGaps.end(), 0.0) /
        static_cast<double>(logGaps.size());
    double variance = 0.0;
    for (double value : logGaps) {
        const double diff = value - mean;
        variance += diff * diff;
    }
    variance = std::max(variance / static_cast<double>(logGaps.size()), kMinGaussianVariance);

    double logLikelihood = 0.0;
    for (double value : logGaps) {
        logLikelihood += gaussianLogPdf(value, mean, variance);
    }

    bic = bicFromLogLikelihood(logGaps.size(), 2, logLikelihood);
    return true;
}

bool twoComponentLogNormalGapBic(const std::vector<int64_t>& gapsNs, double& bic) {
    if (gapsNs.size() < 6) {
        return false;
    }

    std::vector<double> logGaps;
    logGaps.reserve(gapsNs.size());
    for (int64_t gapNs : gapsNs) {
        if (gapNs > 0) {
            logGaps.push_back(std::log(static_cast<double>(gapNs)));
        }
    }
    if (logGaps.size() < 6) {
        return false;
    }

    std::vector<double> sorted = logGaps;
    std::sort(sorted.begin(), sorted.end());

    const size_t firstIndex = sorted.size() / 3;
    const size_t secondIndex = (2 * sorted.size()) / 3;
    double mean1 = sorted[firstIndex];
    double mean2 = sorted[secondIndex];
    if (mean1 > mean2) {
        std::swap(mean1, mean2);
    }

    const double globalMean =
        std::accumulate(logGaps.begin(), logGaps.end(), 0.0) /
        static_cast<double>(logGaps.size());
    double globalVariance = 0.0;
    for (double value : logGaps) {
        const double diff = value - globalMean;
        globalVariance += diff * diff;
    }
    globalVariance = std::max(
        globalVariance / static_cast<double>(logGaps.size()),
        kMinGaussianVariance
    );

    double variance1 = globalVariance;
    double variance2 = globalVariance;
    double weight1 = 0.5;

    for (int iteration = 0; iteration < 80; ++iteration) {
        std::vector<double> responsibilities;
        responsibilities.reserve(logGaps.size());
        double responsibilitySum = 0.0;

        for (double value : logGaps) {
            const double logP1 = std::log(weight1) + gaussianLogPdf(value, mean1, variance1);
            const double logP2 = std::log(1.0 - weight1) + gaussianLogPdf(value, mean2, variance2);
            const double responsibility = std::exp(logP1 - logSumExp(logP1, logP2));
            responsibilities.push_back(responsibility);
            responsibilitySum += responsibility;
        }

        const double responsibilitySum2 = static_cast<double>(logGaps.size()) - responsibilitySum;
        if (responsibilitySum <= 1e-9 || responsibilitySum2 <= 1e-9) {
            return false;
        }

        double newMean1 = 0.0;
        double newMean2 = 0.0;
        for (size_t i = 0; i < logGaps.size(); ++i) {
            newMean1 += responsibilities[i] * logGaps[i];
            newMean2 += (1.0 - responsibilities[i]) * logGaps[i];
        }
        newMean1 /= responsibilitySum;
        newMean2 /= responsibilitySum2;

        double newVariance1 = 0.0;
        double newVariance2 = 0.0;
        for (size_t i = 0; i < logGaps.size(); ++i) {
            const double diff1 = logGaps[i] - newMean1;
            const double diff2 = logGaps[i] - newMean2;
            newVariance1 += responsibilities[i] * diff1 * diff1;
            newVariance2 += (1.0 - responsibilities[i]) * diff2 * diff2;
        }
        newVariance1 = std::max(newVariance1 / responsibilitySum, kMinGaussianVariance);
        newVariance2 = std::max(newVariance2 / responsibilitySum2, kMinGaussianVariance);

        mean1 = newMean1;
        mean2 = newMean2;
        variance1 = newVariance1;
        variance2 = newVariance2;
        weight1 = std::min(std::max(responsibilitySum / static_cast<double>(logGaps.size()), 1e-6), 1.0 - 1e-6);
    }

    double logLikelihood = 0.0;
    for (double value : logGaps) {
        const double logP1 = std::log(weight1) + gaussianLogPdf(value, mean1, variance1);
        const double logP2 = std::log(1.0 - weight1) + gaussianLogPdf(value, mean2, variance2);
        logLikelihood += logSumExp(logP1, logP2);
    }

    bic = bicFromLogLikelihood(logGaps.size(), 5, logLikelihood);
    return true;
}

bool detectTokenBucketShaper(
    const std::vector<int64_t>& trainGapsNs,
    double rawEffectiveCapacityMbps,
    double shaperCapacityMbps,
    int packetSizeBytes,
    uint64_t clientActualMeanSendGapNs,
    double& singleModelBic,
    double& tokenBucketModelBic,
    double& bicImprovement,
    bool& receiveQueueCompressionDetected,
    double& receiveQueueCompressionRatio
) {
    singleModelBic = 0.0;
    tokenBucketModelBic = 0.0;
    bicImprovement = 0.0;
    receiveQueueCompressionDetected = false;
    receiveQueueCompressionRatio = 0.0;

    if (rawEffectiveCapacityMbps <= 0.0 || shaperCapacityMbps <= 0.0) {
        return false;
    }

    std::vector<int64_t> sortedGaps = trainGapsNs;
    std::sort(sortedGaps.begin(), sortedGaps.end());
    if (sortedGaps.empty()) {
        return false;
    }

    const int64_t medianGapNs = sortedGaps[sortedGaps.size() / 2];
    const double rawPacingGapNs =
        (static_cast<double>(packetSizeBytes) * 8.0 * 1'000.0) /
        rawEffectiveCapacityMbps;
    const double referenceGapNs = clientActualMeanSendGapNs > 0
        ? static_cast<double>(clientActualMeanSendGapNs)
        : rawPacingGapNs;

    if (referenceGapNs > 0.0) {
        receiveQueueCompressionRatio = static_cast<double>(medianGapNs) / referenceGapNs;
        // If most received gaps are far below the sender pacing interval, the
        // server is observing socket-queue drain timing rather than path service.
        receiveQueueCompressionDetected = receiveQueueCompressionRatio < 0.25;
        if (receiveQueueCompressionDetected) {
            return false;
        }
    }

    // Use WBest's own zero-threshold condition as the rate-separation gate.
    if (shaperCapacityMbps >= (rawEffectiveCapacityMbps / 2.0)) {
        return false;
    }

    if (!singleLogNormalGapBic(trainGapsNs, singleModelBic) ||
        !twoComponentLogNormalGapBic(trainGapsNs, tokenBucketModelBic)) {
        return false;
    }

    bicImprovement = singleModelBic - tokenBucketModelBic;
    return tokenBucketModelBic < singleModelBic;
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
                << "[--min-pair-gap-us 0] [--min-valid-pairs 1] "
                << "[--train-gap-aggregation mean|trimmed_mean|median] "
                << "[--train-gap-trim-ratio 0.10] "
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
        if (arg == "--train-gap-aggregation" && i + 1 < argc) {
            options.trainGapAggregation = parseTrainGapAggregation(argv[++i]);
            continue;
        }
        if (arg == "--train-gap-trim-ratio" && i + 1 < argc) {
            options.trainGapTrimRatio = std::stod(argv[++i]);
            if (options.trainGapTrimRatio < 0.0 || options.trainGapTrimRatio >= 0.5) {
                throw std::runtime_error("Invalid train gap trim ratio. Use [0, 0.5).");
            }
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
            << "[WBest][Winsock] failed to set process priority error="
            << GetLastError()
            << '\n';
    }

    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        std::cerr
            << "[WBest][Winsock] failed to set thread priority error="
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

class WBestServer {
public:
    WBestServer(SOCKET sock, ProgramOptions options)
        : sock_(sock), options_(std::move(options)) {}

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
                datagram.arrivalNs
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
                datagram.arrivalNs
            );
            return;
        }

        if (stage == kWBestStageFinalSummary) {
            respondWithFinalSummary(
                roundId,
                totalCount,
                parseFinalSummaryRequestDiagnostics(datagram),
                datagram
            );
        }
    }

    FinalSummaryRequestDiagnostics parseFinalSummaryRequestDiagnostics(const ReceivedDatagram& datagram) const {
        FinalSummaryRequestDiagnostics diagnostics;
        constexpr size_t diagnosticsOffset = 31;
        if (datagram.data.size() < diagnosticsOffset + 16) {
            return diagnostics;
        }

        diagnostics.clientActualMeanSendGapNs = readU64BE(datagram.data.data() + diagnosticsOffset);
        diagnostics.clientActualMedianSendGapNs = readU64BE(datagram.data.data() + diagnosticsOffset + 8);
        return diagnostics;
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
        int64_t arrivalNs
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
        } else {
            record.hasSeq2 = true;
            record.hasRecv2 = true;
            record.seq2 = sequence;
            record.recv2Ns = arrivalNs;
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
        int64_t arrivalNs
    ) {
        RoundState& state = getOrCreateRoundState(roundId, arrivalNs);
        state.packetSizeBytes = std::max(state.packetSizeBytes, packetSizeBytes);
        state.expectedTrainPackets = std::max<int>(state.expectedTrainPackets, static_cast<int>(packetCount));
        state.lastUpdatedNs = arrivalNs;
        if (sequence > 0) {
            state.trainRecords[sequence] = TrainRecord{arrivalNs};
        }
    }

    void respondWithFinalSummary(
        const RoundId& roundId,
        uint32_t packetCount,
        const FinalSummaryRequestDiagnostics& requestDiagnostics,
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

            if (computeFinalSummary(state, requestDiagnostics, summary)) {
                success = true;
                logTrainAnalysis(roundId, state, summary);
            }
        }

        std::vector<uint8_t> response;
        response.reserve(92);
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
        appendDoubleBE(response, summary.rawEffectiveCapacityMbps);
        appendDoubleBE(response, summary.shaperCapacityMbps);
        response.push_back(summary.shaperDetected ? 1 : 0);
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
            analysis.details.push_back(detail);
        }

        return analysis;
    }

    bool computeFinalSummary(
        const RoundState& state,
        const FinalSummaryRequestDiagnostics& requestDiagnostics,
        FinalSummary& summary
    ) const {
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
        std::vector<uint32_t> gapStartSequences;
        for (size_t i = 0; i + 1 < trainRecords.size(); ++i) {
            const int64_t gapNs = trainRecords[i + 1].second.recvNs - trainRecords[i].second.recvNs;
            if (gapNs > 0) {
                gapsNs.push_back(gapNs);
                gapStartSequences.push_back(trainRecords[i].first);
            }
        }
        if (gapsNs.empty()) {
            return false;
        }

        const long double rawMeanGapNs = meanNs(gapsNs);
        const long double calculationGapNs = aggregateTrainGapNs(
            gapsNs,
            options_.trainGapAggregation,
            options_.trainGapTrimRatio
        );
        const double calculationGapSeconds = static_cast<double>(calculationGapNs / 1'000'000'000.0L);
        if (calculationGapSeconds <= 0.0) {
            return false;
        }

        summary.meanTrainGapNs = static_cast<uint64_t>(calculationGapNs + 0.5L);
        summary.calculationTrainGapNs = summary.meanTrainGapNs;
        summary.rawMeanTrainGapNs = static_cast<double>(rawMeanGapNs);
        summary.rawEffectiveCapacityMbps = state.effectiveCapacityMbps;
        summary.shaperCapacityMbps =
            (static_cast<double>(state.packetSizeBytes) * 8.0) /
            calculationGapSeconds /
            1'000'000.0;
        if (requestDiagnostics.clientActualMeanSendGapNs > 0) {
            const double clientActualSendGapSeconds =
                static_cast<double>(requestDiagnostics.clientActualMeanSendGapNs) /
                1'000'000'000.0;
            if (clientActualSendGapSeconds > 0.0) {
                summary.clientActualSendRateMbps =
                    (static_cast<double>(state.packetSizeBytes) * 8.0) /
                    clientActualSendGapSeconds /
                    1'000'000.0;
                summary.shaperCapacityMbps = std::min(
                    summary.shaperCapacityMbps,
                    summary.clientActualSendRateMbps
                );
            }
        }
        summary.shaperDetected = detectTokenBucketShaper(
            gapsNs,
            summary.rawEffectiveCapacityMbps,
            summary.shaperCapacityMbps,
            state.packetSizeBytes,
            requestDiagnostics.clientActualMeanSendGapNs,
            summary.shaperSingleModelBic,
            summary.shaperTokenBucketModelBic,
            summary.shaperBicImprovement,
            summary.receiveQueueCompressionDetected,
            summary.receiveQueueCompressionRatio
        );
        summary.effectiveCapacityMbps = summary.shaperDetected
            ? std::min(summary.rawEffectiveCapacityMbps, summary.shaperCapacityMbps)
            : summary.rawEffectiveCapacityMbps;
        summary.achievableThroughputMbps = summary.shaperCapacityMbps;
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
        summary.trainGapsNs = std::move(gapsNs);
        summary.trainGapStartSequences = std::move(gapStartSequences);
        if (summary.sentTrainPackets > 0) {
            summary.lossRate =
                static_cast<double>(std::max(summary.sentTrainPackets - summary.receivedTrainPackets, 0)) /
                static_cast<double>(summary.sentTrainPackets);
        }
        summary.correctedAvailableBandwidthMbps =
            summary.availableBandwidthMbps * (1.0 - summary.lossRate);
        return true;
    }

    void logTrainAnalysis(
        const RoundId& roundId,
        const RoundState& state,
        const FinalSummary& summary
    ) const {
        const std::string roundText = formatRoundId(roundId);
        const std::vector<int64_t>& gapsNs = summary.trainGapsNs;
        if (gapsNs.empty()) {
            std::cout
                << "[WBest][Winsock][Train][Summary] "
                << "round=" << roundText
                << " gaps=0\n";
            return;
        }

        const auto minMax = std::minmax_element(gapsNs.begin(), gapsNs.end());
        const double rawMeanUs = summary.rawMeanTrainGapNs / 1000.0;
        const double calculationGapUs = static_cast<double>(summary.calculationTrainGapNs) / 1000.0;

        std::cout
            << "[WBest][Winsock][Train][Summary] "
            << "round=" << roundText << ' '
            << "payload_bytes=" << state.packetSizeBytes << ' '
            << "received_train=" << summary.receivedTrainPackets << '/' << summary.sentTrainPackets << ' '
            << "Ce=" << std::fixed << std::setprecision(2) << summary.effectiveCapacityMbps << "Mbps "
            << "Ce_raw=" << std::fixed << std::setprecision(2) << summary.rawEffectiveCapacityMbps << "Mbps "
            << "C_shaper=" << std::fixed << std::setprecision(2) << summary.shaperCapacityMbps << "Mbps "
            << "R=" << std::fixed << std::setprecision(2) << summary.achievableThroughputMbps << "Mbps "
            << "shaper_detected=" << (summary.shaperDetected ? 1 : 0) << ' '
            << "queue_compression=" << (summary.receiveQueueCompressionDetected ? 1 : 0) << ' '
            << "queue_compression_ratio=" << std::fixed << std::setprecision(3)
            << summary.receiveQueueCompressionRatio << ' '
            << "client_send_rate=" << std::fixed << std::setprecision(2)
            << summary.clientActualSendRateMbps << "Mbps "
            << "shaper_bic_delta=" << std::fixed << std::setprecision(2) << summary.shaperBicImprovement << ' '
            << "aggregation=" << trainGapAggregationName(options_.trainGapAggregation) << ' '
            << "trim_ratio=" << options_.trainGapTrimRatio << ' '
            << "gap_count=" << gapsNs.size() << ' '
            << "gap_min=" << (static_cast<double>(*minMax.first) / 1000.0) << "us "
            << "gap_median=" << percentileNsToUs(gapsNs, 0.5) << "us "
            << "gap_mean=" << rawMeanUs << "us "
            << "gap_calc=" << calculationGapUs << "us "
            << "gap_p90=" << percentileNsToUs(gapsNs, 0.9) << "us "
            << "gap_max=" << (static_cast<double>(*minMax.second) / 1000.0) << "us "
            << "timestamp_source=app_qpc"
            << '\n';

        std::cout
            << "[WBest][Winsock][Train][Detail] "
            << "round=" << roundText << ' '
            << "gaps=" << formatGapListUs(summary.trainGapStartSequences, gapsNs)
            << '\n';
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
            << "min_pair_gap_us=" << options_.minPairGapUs
            << " timestamp_source=app_qpc"
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
                << "timestamp_source=app_qpc "
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
            << "[WBest][Winsock] listening host=" << options.host
            << " port=" << options.port
            << " timestamp_source=app_qpc"
            << " min_pair_gap_us=" << options.minPairGapUs
            << " min_valid_pairs=" << options.minValidPairs
            << " train_gap_aggregation=" << trainGapAggregationName(options.trainGapAggregation)
            << " train_gap_trim_ratio=" << options.trainGapTrimRatio
            << " high_priority=" << (options.highPriority ? 1 : 0)
            << '\n';

        WBestServer server(sock, options);
        while (true) {
            ReceivedDatagram datagram = receiveDatagram(sock, frequency.QuadPart);
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
