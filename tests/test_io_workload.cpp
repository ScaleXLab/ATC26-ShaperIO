#include "../benchmarks/io/io_workload.h"
#include <cassert>
#include <set>

int main() {
    for (uint64_t workers : {1ULL, 8ULL, 64ULL, 4096ULL}) {
        const uint64_t pages = 4096 * 32;
        std::set<uint64_t> worker_ids;
        for (uint64_t t = 0; t < workers; ++t) {
            const auto mapped = ae_worker(t, workers, true);
            assert(mapped < workers);
            worker_ids.insert(mapped);
        }
        assert(worker_ids.size() == workers);
        for (uint64_t tile : {0ULL, 1ULL, 4ULL, 16ULL, 32ULL, 65536ULL}) {
            std::set<uint64_t> seen;
            for (uint64_t t = 0; t < workers; ++t)
                for (uint64_t i = 0; i < pages / workers; ++i) {
                    const auto index = ae_index(t, i, pages, workers, tile);
                    assert(index < pages);
                    seen.insert(index);
                    if (!tile) assert(index == t * (pages / workers) + i);
                    if (tile == 1) assert(index == i * workers + t);
                }
            assert(seen.size() == pages);
        }
    }
    for (uint64_t pages : {1ULL, 32ULL, 4096ULL, 16384ULL}) {
        for (uint64_t seed : {0ULL, 1ULL, 133ULL, 987654321ULL}) {
            std::set<uint64_t> seen;
            for (uint64_t i = 0; i < pages; ++i) {
                const auto address = ae_page(i, pages, seed, true);
                assert(address < pages);
                seen.insert(address);
                assert(ae_page(i, pages, seed, false) == i);
            }
            assert(seen.size() == pages);
        }
    }
    validate_workload(1ULL << 30, 65536, 4096, 0);
    bool failed = false;
    try { validate_workload(1ULL << 30, 65536, 4096, UINT64_MAX - 65535); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed);
}
