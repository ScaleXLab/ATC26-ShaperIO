#include "io_workload.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <ctrl.h>
#include <page_cache.h>
#ifndef AE_CORE_MAX_WINDOW
#define AE_CORE_MAX_WINDOW 16
#endif
#ifdef AE_PAPER_WAIT_BOUNDS
#include "paper_wait_scheduler_impl.cuh"
constexpr unsigned kAeWaitInitial = 64, kAeWaitMax = 512;
#else
#include "../io_scheduler/io_scheduler_impl.cuh"
constexpr unsigned kAeWaitInitial = 8192, kAeWaitMax = 8192;
#endif

__device__ uint64_t global_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ void check_page(page_cache_d_t* pc, uint64_t entry, uint64_t io,
                           unsigned long long* errors) {
    volatile uint32_t* data = reinterpret_cast<volatile uint32_t*>(pc->base_addr + entry * io);
    for (uint64_t j = 0; j < io / sizeof(uint32_t); ++j)
        if (data[j] != 0xa5a5a5a5u) { atomicAdd(errors, 1ULL); break; }
}

__global__ void transfer(Controller** controllers, page_cache_d_t* pc, io_queue_t* queue,
                         uint64_t base, uint64_t pages, uint64_t io, unsigned workers,
                         unsigned queues, unsigned window, bool write, bool random,
                         uint64_t seed, uint64_t tile, bool transpose, uint64_t* latency, bool verify,
                         unsigned long long* errors) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= workers) return;
    const unsigned worker = ae_worker(tid, workers, transpose);
    const uint64_t count = pages / workers;
    if (!queue) {
        QueuePair* qp = &controllers[0]->d_qps[tid % queues];
        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t index = ae_index(worker, i, pages, workers, tile);
            const uint64_t address = base + ae_page(index, pages, seed, random) * io;
            const uint64_t start = latency ? global_ns() : 0;
            if (write) write_data(pc, qp, address >> qp->block_size_log, io >> qp->block_size_log, tid);
            else read_data_no_second(pc, qp, address >> qp->block_size_log, io >> qp->block_size_log, tid);
            if (latency) latency[index] = global_ns() - start;
            if (verify && !write) check_page(pc, tid, io, errors);
        }
        return;
    }
    uint32_t slots[AE_CORE_MAX_WINDOW];
    uint64_t starts[AE_CORE_MAX_WINDOW];
    unsigned head = 0, pending = 0;
    for (uint64_t i = 0; i < count + window; ++i) {
        if (pending && (pending == window || i >= count)) {
            const auto ticket = slots[head];
            while (queue->done_seq[ticket & queue->capacity_mask] < ticket + 1u) __nanosleep(64);
            const uint64_t finished = i < count ? i - window : count - pending;
            if (latency) latency[tid * count + finished] = global_ns() - starts[head];
            if (verify && !write) check_page(pc, tid * window + head, io, errors);
            head = (head + 1) % window;
            --pending;
        }
        if (i < count) {
            const unsigned slot = (head + pending) % window;
            const auto address = base + ae_page(ae_index(worker, i, pages, workers, tile), pages, seed, random) * io;
            starts[slot] = latency ? global_ns() : 0;
            slots[slot] = io_sched_enqueue(queue, address, io, tid * window + slot,
                                           write ? NVM_IO_WRITE : NVM_IO_READ);
            ++pending;
        }
    }
}

__global__ void shutdown_queue(io_queue_t* queue) {
    if (threadIdx.x == 0) io_sched_shutdown(queue);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        Options o(argc, argv, {"--controller", "--backend", "--op", "--pattern", "--threads",
                              "--bytes", "--offset", "--io", "--gpu", "--seed", "--allow-write",
                              "--latency", "--verify", "--queues", "--qd", "--nsid", "--ring",
                              "--window", "--buffer-bytes", "--batch-bytes", "--tile-pages", "--worker-order"});
        const auto device = o.text("--controller");
        const auto backend = o.text("--backend", "bam");
        const auto op = o.text("--op", "read"), pattern = o.text("--pattern", "sequential");
        const bool write = op == "write", shaped = backend == "shaperio", random = pattern == "random";
        const uint64_t bytes = o.number("--bytes", 1ULL << 30), io = o.number("--io", 65536);
        const uint64_t batch_bytes = o.number("--batch-bytes", bytes);
        const uint64_t tile = o.number("--tile-pages", 0);
        const auto worker_order = o.text("--worker-order", "linear");
        const uint64_t offset = o.number("--offset", 0), seed = o.number("--seed", 1);
        const unsigned workers = o.number("--threads", 32), gpu = o.number("--gpu", 0);
        const unsigned queues = o.number("--queues", 32), qd = o.number("--qd", 256);
        const unsigned nsid = o.number("--nsid", 1), ring = o.number("--ring", 4096);
        const unsigned window = o.number("--window", 1);
        const uint64_t buffer_bytes = o.number("--buffer-bytes", 4ULL << 30);
        const bool latency = o.number("--latency", 0) != 0, verify = o.number("--verify", 0) != 0;
        if (device.empty() || (backend != "bam" && backend != "shaperio") ||
            (op != "read" && op != "write") || (pattern != "sequential" && pattern != "random"))
            throw std::runtime_error("Specify controller, bam/shaperio, read/write, sequential/random");
        validate_workload(bytes, io, workers, offset);
        validate_workload(batch_bytes, io, workers, offset);
        if ((worker_order != "linear" && worker_order != "warp-transpose") ||
            (worker_order == "warp-transpose" && workers >= 32 && workers % 32))
            throw std::runtime_error("Worker order must be linear or warp-transpose (whole warps)");
        if (tile && ((tile & (tile - 1)) || (tile < batch_bytes / io / workers && (batch_bytes / io / workers) % tile)))
            throw std::runtime_error("Tile must be a power of two dividing the per-worker request count");
        if (batch_bytes > bytes || bytes % batch_bytes || (batch_bytes != bytes && (random || latency)))
            throw std::runtime_error("Batches must divide the sequential, untimed-latency transfer");
        if (window < 1 || window > AE_CORE_MAX_WINDOW || (!shaped && window != 1) || !queues || queues > 128 ||
            qd < 32 || (qd & (qd - 1)) || ring < 128 || (ring & (ring - 1)))
            throw std::runtime_error("Invalid queue, window, or ring configuration");
        if (shaped && write && IO_SCHED_WRITE_BATCH && qd <= kSchedKneeLimit)
            throw std::runtime_error("Batched write queue depth must exceed the compiled knee limit");
        if (shaped && !write && queues < kSchedReadSubmitThreads)
            throw std::runtime_error("Read scheduler requires at least 32 queues");
        if (buffer_bytes % io || buffer_bytes / io < uint64_t(workers) * window)
            throw std::runtime_error("DMA buffer budget does not cover all outstanding requests");
        if (write && o.number("--allow-write", 0) != 1)
            throw std::runtime_error("Writes require --allow-write 1");
        cuda_err_chk(cudaSetDevice(gpu));
        std::unique_ptr<Controller> controller(new Controller(device.c_str(), nsid, gpu, qd,
                                                              queues + (shaped && write ? 1 : 0)));
        if (controller->n_qps < queues + (shaped && write ? 1 : 0))
            throw std::runtime_error("Controller did not allocate the requested queue pairs");
        if (offset / controller->blk_size > controller->ns.size ||
            bytes / controller->blk_size > controller->ns.size - offset / controller->blk_size)
            throw std::runtime_error("Experiment exceeds namespace capacity");
        std::vector<Controller*> controllers{controller.get()};
        page_cache_t pc(io, buffer_bytes / io, gpu, *controller, 64, controllers);
        auto* d_pc = reinterpret_cast<page_cache_d_t*>(pc.d_pc_ptr);
        cuda_err_chk(cudaMemset(pc.pdt.base_addr, write ? 0xa5 : 0, buffer_bytes));
        uint64_t* d_latency = nullptr;
        unsigned long long* d_errors = nullptr;
        if (latency) cuda_err_chk(cudaMalloc(&d_latency, bytes / io * sizeof(uint64_t)));
        cuda_err_chk(cudaMalloc(&d_errors, sizeof(unsigned long long)));
        cuda_err_chk(cudaMemset(d_errors, 0, sizeof(unsigned long long)));
        io_queue_t* queue = shaped ? io_sched_create(ring, gpu, write ? queues : 0, controller->ctrl) : nullptr;
        if (shaped && !queue) throw std::runtime_error("Cannot allocate scheduler queue");
        cudaStream_t service, producer;
        cudaEvent_t begin, end;
        cuda_err_chk(cudaStreamCreateWithFlags(&service, cudaStreamNonBlocking));
        cuda_err_chk(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
        cuda_err_chk(cudaEventCreate(&begin));
        cuda_err_chk(cudaEventCreate(&end));
        // Load every producer kernel before starting a persistent service kernel.
        cudaFuncAttributes attributes;
        cuda_err_chk(cudaFuncGetAttributes(&attributes, transfer));
        cuda_err_chk(cudaFuncGetAttributes(&attributes, shutdown_queue));
        cuda_err_chk(cudaFuncGetAttributes(&attributes, io_scheduler_kernel));
        cuda_err_chk(cudaFuncGetAttributes(&attributes, io_read_scheduler_kernel));
        // Materialize producer local memory before launching a persistent scheduler.
        // Zero workers returns before touching storage or DMA buffers.
        transfer<<<(workers + 127) / 128, 128>>>(pc.pdt.d_ctrls, d_pc, queue,
            offset, batch_bytes / io, io, 0, queues, window, write, random, seed, tile,
            worker_order == "warp-transpose", nullptr, false, d_errors);
        cuda_err_chk(cudaGetLastError());
        cuda_err_chk(cudaDeviceSynchronize());
        if (shaped) {
            if (write) io_scheduler_kernel<<<1, kSchedThreads, 0, service>>>(pc.pdt.d_ctrls, d_pc, queue);
            else io_read_scheduler_kernel<<<1, kSchedThreads, 0, service>>>(pc.pdt.d_ctrls, d_pc, queue);
            cuda_err_chk(cudaGetLastError());
        }
        cuda_err_chk(cudaEventRecord(begin, producer));
        for (uint64_t batch_offset = 0; batch_offset < bytes; batch_offset += batch_bytes) {
            transfer<<<(workers + 127) / 128, 128, 0, producer>>>(pc.pdt.d_ctrls, d_pc, queue,
                offset + batch_offset, batch_bytes / io, io, workers, queues, window,
                write, random, seed, tile, worker_order == "warp-transpose", d_latency, verify, d_errors);
            cuda_err_chk(cudaGetLastError());
        }
        cuda_err_chk(cudaEventRecord(end, producer));
        cuda_err_chk(cudaEventSynchronize(end));
        float ms;
        cuda_err_chk(cudaEventElapsedTime(&ms, begin, end));
        if (shaped) {
            shutdown_queue<<<1, 1, 0, producer>>>(queue);
            cuda_err_chk(cudaGetLastError());
            cuda_err_chk(cudaStreamSynchronize(service));
        }
        unsigned long long errors = 0;
        cuda_err_chk(cudaMemcpy(&errors, d_errors, sizeof(errors), cudaMemcpyDeviceToHost));
        if (errors) throw std::runtime_error("Data verification failed for " + std::to_string(errors) + " requests");
        std::vector<double> samples;
        if (latency) {
            std::vector<uint64_t> ns(bytes / io);
            cuda_err_chk(cudaMemcpy(ns.data(), d_latency, ns.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            for (auto value : ns) samples.push_back(value / 1000.0);
            std::sort(samples.begin(), samples.end());
        }
        std::printf("{\"backend\":\"%s\",\"op\":\"%s\",\"pattern\":\"%s\",\"threads\":%u,"
                    "\"bytes\":%llu,\"io_bytes\":%llu,\"seed\":%llu,\"seconds\":%.9f,"
                    "\"bandwidth_gib_s\":%.9f,\"buffer_bytes\":%llu,\"window\":%u,"
                    "\"ring\":%u,\"knee_limit\":%u,\"latency_samples\":%zu,\"verify\":%s",
                    backend.c_str(), op.c_str(), pattern.c_str(), workers, (unsigned long long)bytes,
                    (unsigned long long)io, (unsigned long long)seed, ms / 1000.0,
                    bytes / 1073741824.0 / (ms / 1000.0), (unsigned long long)buffer_bytes,
                    window, ring, kSchedKneeLimit, samples.size(), verify ? "true" : "false");
        if (latency) std::printf(",\"p50_us\":%.6f,\"p99_us\":%.6f", percentile(samples, .50), percentile(samples, .99));
        if (shaped) std::printf(",\"submitted_ios\":%llu,\"merged_ios\":%llu,\"scheduler_batches\":%llu",
            (unsigned long long)queue->submitted_ios.load(simt::memory_order_relaxed),
            (unsigned long long)queue->merged_ios.load(simt::memory_order_relaxed),
            (unsigned long long)queue->sched_batch_count);
        std::printf(",\"batch_bytes\":%llu,\"tile_pages\":%llu,\"drain_wait_initial_ns\":%u,"
                    "\"drain_wait_max_ns\":%u,\"accumulation_policy\":\"original-ready-probe\","
                    "\"worker_order\":\"%s\",\"write_submit_mode\":\"%s\"}\n", (unsigned long long)batch_bytes,
                    (unsigned long long)tile, kAeWaitInitial, kAeWaitMax, worker_order.c_str(),
                    IO_SCHED_WRITE_BATCH ? "batch" : "serial");
        cuda_err_chk(cudaStreamSynchronize(producer));
        if (queue) io_sched_destroy(queue);
        cuda_err_chk(cudaFree(d_errors));
        if (d_latency) cuda_err_chk(cudaFree(d_latency));
        cuda_err_chk(cudaEventDestroy(begin));
        cuda_err_chk(cudaEventDestroy(end));
        cuda_err_chk(cudaStreamDestroy(producer));
        cuda_err_chk(cudaStreamDestroy(service));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
