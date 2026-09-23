#include "io_workload.h"
#include <cuda_runtime.h>
#include <cufile.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

static void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
static void file_check(CUfileError_t code, const char* where) {
    if (code.err != CU_FILE_SUCCESS)
        throw std::runtime_error(std::string(where) + ": " + std::to_string(code.err));
}

int main(int argc, char** argv) {
    try {
        Options o(argc, argv, {"--file", "--op", "--pattern", "--threads", "--bytes", "--offset",
                              "--io", "--gpu", "--seed", "--allow-write", "--latency", "--verify", "--tile-pages",
                              "--worker-order"});
        const auto path = o.text("--file");
        const auto op = o.text("--op", "read");
        const auto pattern = o.text("--pattern", "sequential");
        const bool write = op == "write", random = pattern == "random";
        const uint64_t bytes = o.number("--bytes", 1ULL << 30), io = o.number("--io", 65536);
        const uint64_t offset = o.number("--offset", 0), seed = o.number("--seed", 1);
        const uint64_t tile = o.number("--tile-pages", 0);
        const auto worker_order = o.text("--worker-order", "linear");
        const unsigned workers = o.number("--threads", 1), gpu = o.number("--gpu", 0);
        const bool latency = o.number("--latency", 0) != 0;
        const bool verify = o.number("--verify", 0) != 0;
        if (path.empty() || (op != "read" && op != "write") ||
            (pattern != "sequential" && pattern != "random") || workers > 64)
            throw std::runtime_error("Specify --file, read/write, sequential/random, 1..64 CPU workers");
        validate_workload(bytes, io, workers, offset);
        if ((worker_order != "linear" && worker_order != "warp-transpose") ||
            (worker_order == "warp-transpose" && workers >= 32 && workers % 32))
            throw std::runtime_error("Worker order must be linear or warp-transpose (whole warps)");
        if (tile && ((tile & (tile - 1)) || (tile < bytes / io / workers && (bytes / io / workers) % tile)))
            throw std::runtime_error("Tile must be a power of two dividing the per-worker request count");
        if (write && o.number("--allow-write", 0) != 1)
            throw std::runtime_error("Writes require --allow-write 1");
        struct stat info;
        if (stat(path.c_str(), &info) || !S_ISREG(info.st_mode) ||
            static_cast<uint64_t>(info.st_size) < offset + bytes)
            throw std::runtime_error("Use an existing, fully prepared regular file covering the requested range");
        cuda_check(cudaSetDevice(gpu));
        file_check(cuFileDriverOpen(), "cuFileDriverOpen");
        int fd = open(path.c_str(), (write ? O_RDWR : O_RDONLY) | O_DIRECT);
        if (fd < 0) throw std::runtime_error(std::strerror(errno));
        CUfileDescr_t descriptor = {};
        descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descriptor.handle.fd = fd;
        CUfileHandle_t handle;
        file_check(cuFileHandleRegister(&handle, &descriptor), "cuFileHandleRegister");
        void* buffer = nullptr;
        cuda_check(cudaMalloc(&buffer, workers * io));
        cuda_check(cudaMemset(buffer, 0xa5, workers * io));
        cuda_check(cudaDeviceSynchronize());
        file_check(cuFileBufRegister(buffer, workers * io, 0), "cuFileBufRegister");
        const uint64_t pages = bytes / io, per_worker = pages / workers;
        std::vector<std::vector<double>> samples(workers);
        std::vector<std::thread> threads;
        std::vector<std::string> errors(workers);
        std::mutex mutex;
        std::condition_variable cv;
        unsigned ready = 0;
        bool go = false;
        std::atomic<uint64_t> completed(0);
        using Clock = std::chrono::steady_clock;
        for (unsigned t = 0; t < workers; ++t) {
            threads.emplace_back([&, t] {
                const auto status = cudaSetDevice(gpu);
                std::vector<unsigned char> host(verify ? io : 0);
                if (latency) samples[t].reserve(per_worker);
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ++ready;
                    cv.notify_all();
                    cv.wait(lock, [&] { return go; });
                }
                if (status != cudaSuccess) { errors[t] = cudaGetErrorString(status); return; }
                for (uint64_t i = 0; i < per_worker; ++i) {
                    const auto worker = ae_worker(t, workers, worker_order == "warp-transpose");
                    const auto page = ae_page(ae_index(worker, i, pages, workers, tile), pages, seed, random);
                    const auto start = Clock::now();
                    const ssize_t n = write
                        ? cuFileWrite(handle, buffer, io, offset + page * io, t * io)
                        : cuFileRead(handle, buffer, io, offset + page * io, t * io);
                    if (n != static_cast<ssize_t>(io)) {
                        errors[t] = "Short or failed cuFile I/O: " + std::to_string(n);
                        break;
                    }
                    if (latency) samples[t].push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
                    if (verify && !write) {
                        const auto copy = cudaMemcpy(host.data(), static_cast<char*>(buffer) + t * io,
                                                     io, cudaMemcpyDeviceToHost);
                        if (copy != cudaSuccess ||
                            std::any_of(host.begin(), host.end(), [](unsigned char x) { return x != 0xa5; })) {
                            errors[t] = "Readback verification failed";
                            break;
                        }
                    }
                    completed.fetch_add(io, std::memory_order_relaxed);
                }
            });
        }
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return ready == workers; });
        const auto start = Clock::now();
        go = true;
        lock.unlock();
        cv.notify_all();
        for (auto& thread : threads) thread.join();
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (const auto& error : errors) if (!error.empty()) throw std::runtime_error(error);
        if (completed.load() != bytes) throw std::runtime_error("Incomplete transfer");
        // Report transfer completion and durable write completion separately.
        const auto sync_start = Clock::now();
        if (write && fdatasync(fd)) throw std::runtime_error("fdatasync failed");
        const double sync_seconds = std::chrono::duration<double>(Clock::now() - sync_start).count();
        std::vector<double> all;
        for (const auto& values : samples) all.insert(all.end(), values.begin(), values.end());
        std::sort(all.begin(), all.end());
        std::printf("{\"backend\":\"gds\",\"op\":\"%s\",\"pattern\":\"%s\",\"threads\":%u,"
                    "\"bytes\":%llu,\"io_bytes\":%llu,\"seed\":%llu,\"seconds\":%.9f,"
                    "\"bandwidth_gib_s\":%.9f,\"fdatasync_seconds\":%.9f,\"latency_samples\":%zu",
                    op.c_str(), pattern.c_str(), workers, (unsigned long long)bytes,
                    (unsigned long long)io, (unsigned long long)seed, seconds,
                    bytes / 1073741824.0 / seconds, sync_seconds, all.size());
        if (latency) std::printf(",\"p50_us\":%.6f,\"p99_us\":%.6f", percentile(all, 0.50), percentile(all, 0.99));
        std::printf(",\"verify\":%s,\"tile_pages\":%llu,\"worker_order\":\"%s\"}\n",
                    verify ? "true" : "false", (unsigned long long)tile, worker_order.c_str());
        file_check(cuFileBufDeregister(buffer), "cuFileBufDeregister");
        cuFileHandleDeregister(handle);
        close(fd);
        cuFileDriverClose();
        cuda_check(cudaFree(buffer));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
