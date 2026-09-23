#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define AE_HD __host__ __device__
#else
#define AE_HD
#endif

// Transpose lane/warp ownership without changing the covered pages.
AE_HD inline uint64_t ae_worker(uint64_t worker, uint64_t workers, bool transpose) {
    return transpose && workers >= 32 ? (worker % 32) * (workers / 32) + worker / 32 : worker;
}

// Equal per-worker tiles cover the range once; tile=1 matches the archived striped producer.
AE_HD inline uint64_t ae_index(uint64_t worker, uint64_t iteration, uint64_t pages,
                               uint64_t workers, uint64_t tile) {
    const uint64_t per_worker = pages / workers;
    if (!tile || tile > per_worker) tile = per_worker;
    return (iteration / tile * workers + worker) * tile + iteration % tile;
}

// A bijection on a power-of-two page range: no repeated or omitted pages.
AE_HD inline uint64_t ae_page(uint64_t index, uint64_t pages, uint64_t seed, bool random) {
    if (!random) return index;
    const uint64_t mask = pages - 1;
    uint64_t x = index ^ (index >> 7);
    x = (x * 0x9e3779b97f4a7c15ULL + seed) & mask;
    x ^= x >> 13;
    x = (x * 0xbf58476d1ce4e5b9ULL + seed) & mask;
    return x ^ (x >> 11);
}

struct Options {
    std::map<std::string, std::string> values;
    Options(int argc, char** argv, const std::vector<std::string>& allowed) {
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end() || i + 1 >= argc)
                throw std::runtime_error("Unknown option or missing value: " + key);
            if (!values.emplace(key, argv[i + 1]).second)
                throw std::runtime_error("Duplicate option: " + key);
        }
    }
    std::string text(const std::string& key, const std::string& fallback = "") const {
        const auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    }
    uint64_t number(const std::string& key, uint64_t fallback) const {
        const auto value = text(key, std::to_string(fallback));
        if (value.empty() || value[0] == '-') throw std::runtime_error("Invalid integer: " + key);
        size_t end = 0;
        const uint64_t n = std::stoull(value, &end);
        if (end != value.size()) throw std::runtime_error("Use integer bytes/counts: " + key);
        return n;
    }
};

inline void validate_workload(uint64_t bytes, uint64_t io, uint64_t threads, uint64_t offset) {
    if ((io != 4096 && io != 65536) || bytes == 0 || bytes % io || threads == 0 || threads > 4096 ||
        (bytes / io) % threads || offset % io || offset > UINT64_MAX - bytes)
        throw std::runtime_error("Require 4/64KiB I/O, aligned range, and whole I/Os per worker (1..4096)");
    const uint64_t pages = bytes / io;
    if (pages & (pages - 1)) throw std::runtime_error("Page count must be a power of two");
}

inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) throw std::runtime_error("Empty latency sample");
    return sorted[static_cast<size_t>(p * (sorted.size() - 1))];
}
