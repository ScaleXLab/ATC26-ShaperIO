#ifndef __IO_SCHEDULER_CUH__
#define __IO_SCHEDULER_CUH__

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "page_cache.h"
#include "ctrl.h"
#include "queue.h"
#include "nvm_cmd.h"

// Global sort mode for write scheduler (must be defined before io_queue_t):
//   1 = best-effort global sort: drain all available → radix sort → chunked merge+submit
//   0 = per-batch sort (bitonic sort within kSchedMaxBatch window)
#ifndef IO_SCHED_GLOBAL_SORT
#define IO_SCHED_GLOBAL_SORT 1
#endif

// IO_SCHED_NOSORT: skip sort and merge, submit each entry as individual NVMe command.
// drain → submit in FIFO order (no sorting, no merging).
// Used in macrobenchmark to isolate victim buffer effect from write ordering.
#ifndef IO_SCHED_NOSORT
#define IO_SCHED_NOSORT 0
#endif

// IO_SCHED_VICTIM_BUFFER and IO_SCHED_VICTIM_SWAP are defined in page_cache.h
// (which is included above) because page_cache.h is parsed first and needs
// these values for page_cache_d_t / find_slot(). Do NOT redefine them here —
// duplicate #ifndef guards would silently shadow -D overrides.

// Parallel scan_ready + drain fusion (optimization 1):
//   1 = warp ballot parallel scan + fused drain (new)
//   0 = thread 0 serial io_sched_scan_ready + parallel drain (original)
#ifndef IO_SCHED_PARALLEL_DRAIN
#define IO_SCHED_PARALLEL_DRAIN 1
#endif

// Parallel merge (optimization 3):
//   1 = parallel boundary detection + parallel PRP construction (new)
//   0 = thread 0 serial merge + PRP (original)
#ifndef IO_SCHED_PARALLEL_MERGE
#define IO_SCHED_PARALLEL_MERGE 1
#endif

// Read scheduler: parallel scan_ready + drain fusion
//   1 = warp ballot parallel scan + fused drain (new)
//   0 = thread 0 serial io_sched_scan_ready + parallel drain (original)
#ifndef IO_SCHED_READ_PARALLEL_DRAIN
#define IO_SCHED_READ_PARALLEL_DRAIN 1
#endif

// Read scheduler: parallel merge
//   1 = parallel boundary detection + parallel PRP construction (new)
//   0 = thread 0 serial merge + PRP (original)
#ifndef IO_SCHED_READ_PARALLEL_MERGE
#define IO_SCHED_READ_PARALLEL_MERGE 1
#endif

// Read scheduler: number of parallel submit threads (each uses its own QP).
// Default 32 for 128-queue SSDs; set to 8 or 16 for 16-queue SSDs.
#ifndef IO_SCHED_READ_SUBMIT_THREADS
#define IO_SCHED_READ_SUBMIT_THREADS 32
#endif

// ---------------------------------------------------------------------------
// Victim Buffer (decouples eviction writes from burst reads)
// ---------------------------------------------------------------------------
#if IO_SCHED_VICTIM_BUFFER

#define VICTIM_FREE     0   // Entry is free, can be claimed by eviction thread
#define VICTIM_COPYING  1   // Eviction thread is memcpy-ing data into this entry
#define VICTIM_OCCUPIED 2   // Data ready, awaiting scheduler flush (or reader hit)

struct victim_buffer_t {
    uint8_t*  data;           // DMA-mapped GPU memory, [n_entries * page_size]
    uint64_t* data_ioaddrs;   // DMA IO addresses, [n_entries * pages_per_entry]
    uint64_t* byte_offsets;   // SSD byte offset tag per entry, [n_entries]
#if IO_SCHED_VICTIM_SWAP
    uint64_t* entry_vaddrs;   // [n_entries] current virtual address of each entry's data buffer
#endif

    simt::atomic<uint32_t, simt::thread_scope_device>* states;        // [n_entries]
    simt::atomic<uint32_t, simt::thread_scope_device>* reader_counts; // [n_entries]

    uint32_t n_entries;       // Must be power of 2
    uint32_t n_entries_mask;  // n_entries - 1
    uint32_t page_size;
    uint32_t pages_per_entry; // page_size / ctrl_page_size (4KB sub-pages per entry)

    simt::atomic<uint32_t, simt::thread_scope_device> alloc_ticket; // round-robin allocation

    // Profiling counters
    simt::atomic<uint64_t, simt::thread_scope_device> prof_fast_count;     // successful victim claims
    simt::atomic<uint64_t, simt::thread_scope_device> prof_fallback_count; // fallback to blocking write
    simt::atomic<uint64_t, simt::thread_scope_device> prof_claim_cycles;   // total CAS claim loop cycles
    simt::atomic<uint64_t, simt::thread_scope_device> prof_memcpy_cycles;  // total memcpy cycles
    simt::atomic<uint64_t, simt::thread_scope_device> prof_enqueue_cycles; // total enqueue_forget cycles

    // DMA handle for cleanup
    void*      data_raw;      // cudaMalloc raw pointer (before alignment)
    nvm_dma_t* data_dma;      // DMA mapping handle
};

// Mark pc_entry as coming from victim buffer (bit 63 set)
#define VICTIM_PC_ENTRY_BIT  (1ULL << 63)
#define VICTIM_PC_ENTRY_MASK (~VICTIM_PC_ENTRY_BIT)

#endif // IO_SCHED_VICTIM_BUFFER

// ---------------------------------------------------------------------------
// IO Request Entry (32 bytes, naturally aligned)
// ---------------------------------------------------------------------------
struct __align__(32) io_request_t {
    uint64_t byte_offset;          // Byte offset on SSD
    uint64_t pc_entry;             // page_cache entry index for PRP lookup
    uint8_t  opcode;               // NVM_IO_READ or NVM_IO_WRITE
    uint8_t  n_bytes_log2;         // log2(io_size): valid 9 (512B) to 21 (2MB)
    uint8_t  pad[14];
};

// ---------------------------------------------------------------------------
// IO Request Queue (GPU global-memory ring buffer)
// ---------------------------------------------------------------------------
struct io_queue_t {
    io_request_t* entries;      // Ring buffer of capacity entries
    volatile uint32_t* done_flags;  // Per-slot completion flags (device memory)
    volatile uint32_t* done_seq;    // Per-slot completion sequence (monotonic, never cleared)
    volatile uint32_t* prod_seq;    // Per-slot production sequence: prod_seq[idx]==slot+1 ⇒ entry ready
    uint32_t capacity;          // Must be power of 2
    uint32_t capacity_mask;     // capacity - 1

    // Producer claim: next slot to claim (atomic, device scope)
    // Producers fetch_add this to reserve a slot, then write data,
    // then increment prod_ready.
    simt::atomic<uint32_t, simt::thread_scope_device> prod_tail;
    // Producer ready: number of entries fully written (atomic, device scope)
    // Scheduler reads this (acquire) to know how many entries are safe to read.
    // Invariant: cons_head <= prod_ready <= prod_tail.
    simt::atomic<uint32_t, simt::thread_scope_device> prod_ready;
    // Consumer head: next slot to read (atomic, device scope)
    simt::atomic<uint32_t, simt::thread_scope_device> cons_head;
    // Shutdown flag: 1 = scheduler should exit
    simt::atomic<uint32_t, simt::thread_scope_device> shutdown;
    // Pause flag: 1 = scheduler skips draining (entries queue up but are not processed)
    simt::atomic<uint32_t, simt::thread_scope_device> sched_paused;
    // Counters for verification / stats
    simt::atomic<uint64_t, simt::thread_scope_device> submitted_ios;
    simt::atomic<uint64_t, simt::thread_scope_device> merged_ios;

    // Enqueue profiling counters
    simt::atomic<uint64_t, simt::thread_scope_device> prof_qfull_spins;     // queue-full spin iterations
    simt::atomic<uint64_t, simt::thread_scope_device> prof_prodready_spins; // prod_ready spin iterations
    simt::atomic<uint64_t, simt::thread_scope_device> prof_enqueue_cycles;  // total enqueue time (all callers)

    // Base QP index for scheduler threads. Write scheduler uses qp[sched_qp_base].
    // Read scheduler uses qp[sched_qp_base + tid] for tid < kSchedReadSubmitThreads.
    // These QPs must NOT be shared with application kernels (batch IO breaks QP state).
    uint32_t sched_qp_base;

    // Timing breakdown (nanoseconds). Written by scheduler thread 0.
    uint64_t sched_drain_ns;    // Phase 1: drain
    uint64_t sched_sort_ns;     // Phase 2: radix/bitonic sort
    uint64_t sched_merge_ns;    // Phase 3: merge (boundary detection + PRP construction)
    uint64_t sched_nvme_ns;     // Phase 4a: NVMe submit (sched_write_batch_merged)
    uint64_t sched_submit_ns;   // Phase 4b: notify (done_seq/done_flags/cons_head)
    uint64_t sched_batch_count; // Number of batches processed
    uint64_t sched_2mb_splits;  // Number of merge groups split due to >2MB

    // PRP list pool for cross-pc_entry merged IO.
    // Each slot is one 4KB page holding up to 512 PRP entries.
    // prp_pool_vaddr[i] = GPU virtual address of slot i (for writing PRP entries)
    // prp_pool_ioaddr[i] = physical/IO address of slot i (for NVMe PRP2 field)
    uint64_t* prp_pool_vaddr;     // device array, nullptr if not allocated
    uint64_t* prp_pool_ioaddr;    // device array, nullptr if not allocated
    uint32_t  prp_pool_n_slots;
    void*     prp_pool_raw;       // cudaMalloc raw pointer (for cleanup)
    nvm_dma_t* prp_pool_dma; // DMA mapping handle (for cleanup)

#if IO_SCHED_GLOBAL_SORT
    // Global sort buffers (allocated when ctrl != nullptr).
    // Used by write scheduler to accumulate all available entries into global
    // memory, radix-sort by byte_offset, then merge+submit in chunks.
    uint64_t* gsort_keys;       // [capacity] sort key = byte_offset
    uint64_t* gsort_keys_alt;   // [capacity] radix sort double-buffer
    uint32_t* gsort_vals;       // [capacity] original accumulation index (0..N-1)
    uint32_t* gsort_vals_alt;   // [capacity] radix sort double-buffer
    uint64_t* gsort_pc_entries; // [capacity] pc_entry (for PRP lookup)
    uint8_t*  gsort_nbytes;     // [capacity] n_bytes_log2
#if IO_SCHED_PARALLEL_MERGE
    uint32_t* gsort_group_starts; // [capacity] parallel merge boundary positions
#endif
#endif
};

// ---------------------------------------------------------------------------
// Scheduler constants
// ---------------------------------------------------------------------------
static constexpr uint32_t kSchedThreads           = 128;  // 4 warps (fixed)
// Max entries per drain/sort batch.  Override at compile time:
//   -DIO_SCHED_MAX_BATCH=128  to revert to original window.
// Must be a power of 2 and a multiple of kSchedThreads.
#ifndef IO_SCHED_MAX_BATCH
#define IO_SCHED_MAX_BATCH 512
#endif
static constexpr uint32_t kSchedMaxBatch          = IO_SCHED_MAX_BATCH;
static_assert(kSchedMaxBatch % kSchedThreads == 0,
              "kSchedMaxBatch must be a multiple of kSchedThreads");
static_assert((kSchedMaxBatch & (kSchedMaxBatch - 1)) == 0,
              "kSchedMaxBatch must be a power of 2");
static constexpr uint32_t kSchedElemsPerThread    = kSchedMaxBatch / kSchedThreads;
static constexpr uint64_t kSchedMergeTarget       = 2ULL * 1024ULL * 1024ULL; // 2 MB
static constexpr uint32_t kSchedReadSubmitThreads = IO_SCHED_READ_SUBMIT_THREADS;

// Knee-point throttle: max NVMe write commands per submit round.
// Limits in-flight writes to avoid saturating SSD and spiking read latency.
// PM9A3 knee point is ~32-64 outstanding commands.
#ifndef IO_SCHED_KNEE_LIMIT
#define IO_SCHED_KNEE_LIMIT 32
#endif
static constexpr uint32_t kSchedKneeLimit = IO_SCHED_KNEE_LIMIT;

// ---------------------------------------------------------------------------
// Sort key encoding
// ---------------------------------------------------------------------------
// Pack (byte_offset, batch_index) into uint64_t for sorting:
//   bits [63:B] = byte_offset         (strict byte/LBA order)
//   bits [B-1:0]= batch index tie-breaker (up to kSchedMaxBatch entries)
//
// B = kSortKeyIndexBits = ceil(log2(kSchedMaxBatch)).

static constexpr uint32_t kSortKeyIndexBits =
    (kSchedMaxBatch <= 128)  ? 7 :
    (kSchedMaxBatch <= 256)  ? 8 :
    (kSchedMaxBatch <= 512)  ? 9 : 10;
static constexpr uint64_t kSortKeyIndexMask = (1ULL << kSortKeyIndexBits) - 1;

__device__ __host__ __forceinline__
uint64_t io_sched_make_sort_key(uint64_t byte_offset, uint32_t batch_index)
{
    return (byte_offset << kSortKeyIndexBits)
           | ((uint64_t)batch_index & kSortKeyIndexMask);
}

__device__ __host__ __forceinline__
uint64_t io_sched_key_byte_offset(uint64_t key)
{
    return key >> kSortKeyIndexBits;
}

__device__ __host__ __forceinline__
uint32_t io_sched_key_batch_index(uint64_t key)
{
    return (uint32_t)(key & kSortKeyIndexMask);
}

// ---------------------------------------------------------------------------
// IO size encoding helpers
// ---------------------------------------------------------------------------
// Encode: n_bytes must be a power of 2 in [512, 2MB].
__device__ __forceinline__
uint8_t io_sched_encode_nbytes(uint64_t n_bytes)
{
    return (uint8_t)(__ffsll((long long)n_bytes) - 1);
}

// Decode: reconstruct byte size from log2.
__device__ __host__ __forceinline__
uint64_t io_sched_decode_nbytes(uint8_t n_bytes_log2)
{
    return 1ULL << n_bytes_log2;
}

// Write scheduler submit mode toggle:
//   1 = batch submit (single doorbell per batch, in-flight = n_groups)
//   0 = serial submit (one doorbell per merged IO, in-flight = 1, original)
#ifndef IO_SCHED_WRITE_BATCH
#define IO_SCHED_WRITE_BATCH 1
#endif

// ---------------------------------------------------------------------------
// Device-side Producer API
// ---------------------------------------------------------------------------

// Enqueue an IO request into the scheduler queue.
// Returns the slot index written. Spins if the queue is full.
// The caller can poll q->done_flags[slot & capacity_mask] for completion.
__device__ __forceinline__
uint32_t io_sched_enqueue(io_queue_t* q,
                          uint64_t byte_offset,
                          uint64_t n_bytes,
                          uint64_t pc_entry,
                          uint8_t  opcode)
{
    uint64_t enq_t0 = clock64();

    // Claim a slot via atomic increment of prod_tail.
    uint32_t slot = q->prod_tail.fetch_add(1u, simt::memory_order_relaxed);
    uint32_t idx  = slot & q->capacity_mask;

    // Spin-wait if the queue is full (producer has lapped consumer).
    // Full condition: slot - cons_head >= capacity.
    // After this check passes, the scheduler has consumed the previous
    // occupant at this slot index, so done_flags can be safely cleared.
    uint64_t qfull_spins = 0;
    while ((slot - q->cons_head.load(simt::memory_order_acquire)) >= q->capacity) {
        ++qfull_spins;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }

    // Clear any stale completion flag from previous occupant.
    q->done_flags[idx] = 0;

    // Write the entry.
    io_request_t* e = &q->entries[idx];
    e->byte_offset  = byte_offset;
    e->pc_entry     = pc_entry;
    e->opcode       = opcode;
    e->n_bytes_log2 = io_sched_encode_nbytes(n_bytes);

    // Ensure the entry is visible before marking it ready.
    __threadfence();

    // Mark this slot as ready via per-slot sequence number.
    // No spin-wait needed — each producer independently flags its own slot.
    // The scheduler scans prod_seq to find the contiguous ready range.
    q->prod_seq[idx] = slot + 1u;

    uint64_t enq_t1 = clock64();
    q->prof_qfull_spins.fetch_add(qfull_spins, simt::memory_order_relaxed);
    q->prof_enqueue_cycles.fetch_add(enq_t1 - enq_t0, simt::memory_order_relaxed);

    return slot;
}

// Scan prod_seq from 'start' forward, return count of consecutive ready entries.
// Scheduler thread 0 calls this instead of reading prod_ready.
__device__ __forceinline__
uint32_t io_sched_scan_ready(const io_queue_t* q, uint32_t start, uint32_t max_count)
{
    uint32_t tail = q->prod_tail.load(simt::memory_order_acquire);
    uint32_t pending = tail - start;
    if (pending > max_count) pending = max_count;
    for (uint32_t i = 0; i < pending; i++) {
        if (q->prod_seq[(start + i) & q->capacity_mask] != (start + i + 1u))
            return i;
    }
    return pending;
}

// Signal the scheduler to shut down after draining remaining requests.
__device__ __forceinline__
void io_sched_shutdown(io_queue_t* q)
{
    __threadfence();
    q->shutdown.store(1u, simt::memory_order_release);
}

// ---------------------------------------------------------------------------
// Device-side Synchronous IO API (submit + wait for completion)
// ---------------------------------------------------------------------------

// Read data via the IO scheduler. Blocks until the NVMe read completes.
// byte_offset: SSD byte offset; n_bytes: IO size (power of 2, 512B–2MB).
__device__ __forceinline__
void io_sched_read(io_queue_t* q,
                   uint64_t byte_offset,
                   uint64_t n_bytes,
                   uint64_t pc_entry)
{
    uint32_t slot = io_sched_enqueue(q, byte_offset, n_bytes, pc_entry, NVM_IO_READ);
    uint32_t idx  = slot & q->capacity_mask;
    uint32_t ticket = slot + 1u;
    while (q->done_seq[idx] < ticket) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
}

// Write data via the IO scheduler. Blocks until the NVMe write completes.
// byte_offset: SSD byte offset; n_bytes: IO size (power of 2, 512B–2MB).
__device__ __forceinline__
void io_sched_write(io_queue_t* q,
                    uint64_t byte_offset,
                    uint64_t n_bytes,
                    uint64_t pc_entry)
{
    uint32_t slot = io_sched_enqueue(q, byte_offset, n_bytes, pc_entry, NVM_IO_WRITE);
    uint32_t idx  = slot & q->capacity_mask;
    uint32_t ticket = slot + 1u;
    while (q->done_seq[idx] < ticket) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
}

// ---------------------------------------------------------------------------
// Fire-and-forget write enqueue (victim buffer path)
// ---------------------------------------------------------------------------
#if IO_SCHED_VICTIM_BUFFER
// Enqueue a write request without waiting for completion.
// Data lives in victim buffer; the cache slot is already released.
// done_flags will be cleared by the next io_sched_enqueue on the same slot.
__device__ __forceinline__
void io_sched_enqueue_forget(io_queue_t* q,
                              uint64_t byte_offset,
                              uint64_t n_bytes,
                              uint64_t pc_entry)
{
    io_sched_enqueue(q, byte_offset, n_bytes, pc_entry, NVM_IO_WRITE);
}
#endif // IO_SCHED_VICTIM_BUFFER

// Dispatch read or write via the IO scheduler based on opcode.
// Blocks until the NVMe IO completes.
__device__ __forceinline__
void io_sched_access(io_queue_t* q,
                     uint64_t byte_offset,
                     uint64_t n_bytes,
                     uint64_t pc_entry,
                     uint8_t  opcode)
{
    uint32_t slot = io_sched_enqueue(q, byte_offset, n_bytes, pc_entry, opcode);
    uint32_t idx  = slot & q->capacity_mask;
    uint32_t ticket = slot + 1u;
    while (q->done_seq[idx] < ticket) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
}

// ---------------------------------------------------------------------------
// Device-side Asynchronous IO API (submit returns done_flag for later wait)
// ---------------------------------------------------------------------------

// Submit a read request to the IO scheduler. Returns a pointer to the
// done_flag for this request. The caller can continue other work and later
// call io_sched_wait() on the returned pointer.
__device__ __forceinline__
volatile uint32_t* io_sched_submit_read(io_queue_t* q,
                                        uint64_t byte_offset,
                                        uint64_t n_bytes,
                                        uint64_t pc_entry)
{
    uint32_t slot = io_sched_enqueue(q, byte_offset, n_bytes, pc_entry, NVM_IO_READ);
    return &q->done_flags[slot & q->capacity_mask];
}

// Submit a write request to the IO scheduler. Returns a pointer to the
// done_flag for this request.
__device__ __forceinline__
volatile uint32_t* io_sched_submit_write(io_queue_t* q,
                                         uint64_t byte_offset,
                                         uint64_t n_bytes,
                                         uint64_t pc_entry)
{
    uint32_t slot = io_sched_enqueue(q, byte_offset, n_bytes, pc_entry, NVM_IO_WRITE);
    return &q->done_flags[slot & q->capacity_mask];
}

// Submit a read/write request to the IO scheduler based on opcode.
// Returns a pointer to the done_flag for this request.
__device__ __forceinline__
volatile uint32_t* io_sched_submit(io_queue_t* q,
                                   uint64_t byte_offset,
                                   uint64_t n_bytes,
                                   uint64_t pc_entry,
                                   uint8_t  opcode)
{
    uint32_t slot = io_sched_enqueue(q, byte_offset, n_bytes, pc_entry, opcode);
    return &q->done_flags[slot & q->capacity_mask];
}

// Wait for a previously submitted async IO to complete, then release the slot.
__device__ __forceinline__
void io_sched_wait(volatile uint32_t* done_flag)
{
    while (*done_flag == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
    *done_flag = 0;  // Release slot for reuse (AGILE-style)
}

// ---------------------------------------------------------------------------
// Host-side API
// ---------------------------------------------------------------------------

// Allocate and initialize an io_queue_t on the GPU.
// capacity must be a power of 2.
inline io_queue_t* io_sched_create(uint32_t capacity, int cuda_device,
                                    uint32_t sched_qp_base = 0,
                                    const nvm_ctrl_t* ctrl = nullptr)
{
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
        fprintf(stderr, "io_sched_create: capacity must be a power of 2 (got %u)\n", capacity);
        return nullptr;
    }

    cudaSetDevice(cuda_device);

    // Allocate the queue struct in managed memory so both host and device can access.
    io_queue_t* q = nullptr;
    cudaMallocManaged(&q, sizeof(io_queue_t));
    if (!q) { fprintf(stderr, "io_sched_create: failed to allocate queue struct\n"); return nullptr; }

    // Allocate the ring buffer entries in device memory.
    io_request_t* entries = nullptr;
    cudaMalloc(&entries, (size_t)capacity * sizeof(io_request_t));
    if (!entries) { fprintf(stderr, "io_sched_create: failed to allocate entries\n"); cudaFree(q); return nullptr; }
    cudaMemset(entries, 0, (size_t)capacity * sizeof(io_request_t));

    // Allocate per-slot done flags in device memory.
    volatile uint32_t* done_flags = nullptr;
    cudaMalloc((void**)&done_flags, (size_t)capacity * sizeof(uint32_t));
    if (!done_flags) { fprintf(stderr, "io_sched_create: failed to allocate done_flags\n"); cudaFree(entries); cudaFree(q); return nullptr; }
    cudaMemset((void*)done_flags, 0, (size_t)capacity * sizeof(uint32_t));

    // Allocate per-slot done sequence numbers for synchronous waiters.
    volatile uint32_t* done_seq = nullptr;
    cudaMalloc((void**)&done_seq, (size_t)capacity * sizeof(uint32_t));
    if (!done_seq) {
        fprintf(stderr, "io_sched_create: failed to allocate done_seq\n");
        cudaFree((void*)done_flags);
        cudaFree(entries);
        cudaFree(q);
        return nullptr;
    }
    cudaMemset((void*)done_seq, 0, (size_t)capacity * sizeof(uint32_t));

    // Allocate per-slot production sequence numbers (replaces prod_ready).
    volatile uint32_t* prod_seq = nullptr;
    cudaMalloc((void**)&prod_seq, (size_t)capacity * sizeof(uint32_t));
    if (!prod_seq) {
        fprintf(stderr, "io_sched_create: failed to allocate prod_seq\n");
        cudaFree((void*)done_seq);
        cudaFree((void*)done_flags);
        cudaFree(entries);
        cudaFree(q);
        return nullptr;
    }
    cudaMemset((void*)prod_seq, 0, (size_t)capacity * sizeof(uint32_t));

    q->entries       = entries;
    q->done_flags    = done_flags;
    q->done_seq      = done_seq;
    q->prod_seq      = prod_seq;
    q->capacity      = capacity;
    q->capacity_mask = capacity - 1;

    // Zero-initialize atomics via memset (they start at 0).
    // simt::atomic is trivially constructible on CUDA managed memory.
    cudaMemset(&q->prod_tail,     0, sizeof(q->prod_tail));
    cudaMemset(&q->prod_ready,    0, sizeof(q->prod_ready));
    cudaMemset(&q->cons_head,     0, sizeof(q->cons_head));
    cudaMemset(&q->shutdown,      0, sizeof(q->shutdown));
    cudaMemset(&q->sched_paused,  0, sizeof(q->sched_paused));
    cudaMemset(&q->submitted_ios, 0, sizeof(q->submitted_ios));
    cudaMemset(&q->merged_ios,    0, sizeof(q->merged_ios));
    cudaMemset(&q->prof_qfull_spins,     0, sizeof(q->prof_qfull_spins));
    cudaMemset(&q->prof_prodready_spins, 0, sizeof(q->prof_prodready_spins));
    cudaMemset(&q->prof_enqueue_cycles,  0, sizeof(q->prof_enqueue_cycles));
    q->sched_qp_base     = sched_qp_base;
    q->sched_drain_ns    = 0;
    q->sched_sort_ns     = 0;
    q->sched_merge_ns    = 0;
    q->sched_nvme_ns     = 0;
    q->sched_submit_ns   = 0;
    q->sched_batch_count = 0;
    q->sched_2mb_splits  = 0;

    // Allocate PRP list pool for cross-pc_entry merge (if ctrl provided).
    if (ctrl != nullptr) {
        const uint32_t n_slots = kSchedMaxBatch;
        const size_t pool_bytes = (size_t)n_slots * 4096;
        const uint64_t align = 64ULL * 1024ULL;
        void* raw = nullptr;
        cudaMalloc(&raw, pool_bytes + align);
        if (!raw) {
            fprintf(stderr, "io_sched_create: failed to allocate PRP pool\n");
        } else {
            uint64_t raw_u64 = (uint64_t)raw;
            uint64_t aligned_u64 = (raw_u64 + align) & ~(align - 1ULL);
            void* aligned = (void*)aligned_u64;
            cudaMemset(aligned, 0, pool_bytes);
            nvm_dma_t* dma = nullptr;
            int err = nvm_dma_map_device(&dma, ctrl, aligned, pool_bytes);
            if (err != 0 || !dma) {
                fprintf(stderr, "io_sched_create: failed to DMA-map PRP pool (err=%d)\n", err);
                cudaFree(raw);
                raw = nullptr;
                dma = nullptr;
            }
            if (dma) {
                uint64_t *h_va = new uint64_t[n_slots];
                uint64_t *h_io = new uint64_t[n_slots];
                for (uint32_t i = 0; i < n_slots; i++) {
                    h_va[i] = aligned_u64 + (uint64_t)i * 4096;
                    h_io[i] = dma->ioaddrs[i];
                }
                uint64_t *d_va = nullptr, *d_io = nullptr;
                cudaMalloc(&d_va, n_slots * sizeof(uint64_t));
                cudaMalloc(&d_io, n_slots * sizeof(uint64_t));
                cudaMemcpy(d_va, h_va, n_slots * sizeof(uint64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_io, h_io, n_slots * sizeof(uint64_t), cudaMemcpyHostToDevice);
                delete[] h_va;
                delete[] h_io;
                q->prp_pool_vaddr   = d_va;
                q->prp_pool_ioaddr  = d_io;
                q->prp_pool_n_slots = n_slots;
                q->prp_pool_raw     = raw;
                q->prp_pool_dma     = dma;
            } else {
                q->prp_pool_vaddr   = nullptr;
                q->prp_pool_ioaddr  = nullptr;
                q->prp_pool_n_slots = 0;
                q->prp_pool_raw     = nullptr;
                q->prp_pool_dma     = nullptr;
            }
        }
    } else {
        q->prp_pool_vaddr   = nullptr;
        q->prp_pool_ioaddr  = nullptr;
        q->prp_pool_n_slots = 0;
        q->prp_pool_raw     = nullptr;
        q->prp_pool_dma     = nullptr;
    }

#if IO_SCHED_GLOBAL_SORT
    // Allocate global sort buffers (33 bytes/entry × capacity ≈ 1 MB for 32768).
    if (ctrl != nullptr) {
        cudaMalloc(&q->gsort_keys,       (size_t)capacity * sizeof(uint64_t));
        cudaMalloc(&q->gsort_keys_alt,   (size_t)capacity * sizeof(uint64_t));
        cudaMalloc(&q->gsort_vals,       (size_t)capacity * sizeof(uint32_t));
        cudaMalloc(&q->gsort_vals_alt,   (size_t)capacity * sizeof(uint32_t));
        cudaMalloc(&q->gsort_pc_entries, (size_t)capacity * sizeof(uint64_t));
        cudaMalloc(&q->gsort_nbytes,     (size_t)capacity * sizeof(uint8_t));
#if IO_SCHED_PARALLEL_MERGE
        cudaMalloc(&q->gsort_group_starts, (size_t)capacity * sizeof(uint32_t));
#endif
    } else {
        q->gsort_keys       = nullptr;
        q->gsort_keys_alt   = nullptr;
        q->gsort_vals       = nullptr;
        q->gsort_vals_alt   = nullptr;
        q->gsort_pc_entries = nullptr;
        q->gsort_nbytes     = nullptr;
#if IO_SCHED_PARALLEL_MERGE
        q->gsort_group_starts = nullptr;
#endif
    }
#endif

    cudaDeviceSynchronize();
    return q;
}

// Free an io_queue_t and its ring buffer.
inline void io_sched_destroy(io_queue_t* q)
{
    if (!q) return;
    if (q->entries) cudaFree(q->entries);
    if (q->done_flags) cudaFree((void*)q->done_flags);
    if (q->done_seq) cudaFree((void*)q->done_seq);
    if (q->prod_seq) cudaFree((void*)q->prod_seq);
    if (q->prp_pool_vaddr)  cudaFree(q->prp_pool_vaddr);
    if (q->prp_pool_ioaddr) cudaFree(q->prp_pool_ioaddr);
    if (q->prp_pool_dma)    nvm_dma_unmap(q->prp_pool_dma);
    if (q->prp_pool_raw)    cudaFree(q->prp_pool_raw);
#if IO_SCHED_GLOBAL_SORT
    if (q->gsort_keys)       cudaFree(q->gsort_keys);
    if (q->gsort_keys_alt)   cudaFree(q->gsort_keys_alt);
    if (q->gsort_vals)       cudaFree(q->gsort_vals);
    if (q->gsort_vals_alt)   cudaFree(q->gsort_vals_alt);
    if (q->gsort_pc_entries) cudaFree(q->gsort_pc_entries);
    if (q->gsort_nbytes)     cudaFree(q->gsort_nbytes);
#if IO_SCHED_PARALLEL_MERGE
    if (q->gsort_group_starts) cudaFree(q->gsort_group_starts);
#endif
#endif
    cudaFree(q);
}

// ---------------------------------------------------------------------------
// Forward declarations of scheduler kernels (defined in io_scheduler_impl.cuh)
// ---------------------------------------------------------------------------
// Write scheduler: sort + merge + single-thread serial submit (preserves LBA order).
__global__ void io_scheduler_kernel(Controller** ctrls, page_cache_d_t* pc,
                                    io_queue_t* q);

// Read scheduler: sort + merge + 32-thread parallel submit via 32 QueuePairs.
// Requires Controller with num_queues >= kSchedReadSubmitThreads and
// page_cache with n_pages >= kSchedReadSubmitThreads.
__global__ void io_read_scheduler_kernel(Controller** ctrls, page_cache_d_t* pc,
                                         io_queue_t* q);

// ---------------------------------------------------------------------------
// Victim buffer host-side API
// ---------------------------------------------------------------------------
#if IO_SCHED_VICTIM_BUFFER

// Allocate and initialize a victim buffer on the GPU.
// n_entries must be a power of 2. page_size is the page cache page size.
// pages_per_entry = page_size / ctrl->page_size (4KB sub-pages per entry).
inline victim_buffer_t* victim_create(uint32_t n_entries, uint64_t page_size,
                                       uint32_t pages_per_entry, int cuda_device,
                                       const nvm_ctrl_t* ctrl)
{
    if (n_entries == 0 || (n_entries & (n_entries - 1)) != 0) {
        fprintf(stderr, "victim_create: n_entries must be a power of 2 (got %u)\n", n_entries);
        return nullptr;
    }

    cudaSetDevice(cuda_device);

    victim_buffer_t* vb = nullptr;
    cudaMallocManaged(&vb, sizeof(victim_buffer_t));
    if (!vb) { fprintf(stderr, "victim_create: failed to allocate struct\n"); return nullptr; }

    vb->n_entries       = n_entries;
    vb->n_entries_mask  = n_entries - 1;
    vb->page_size       = (uint32_t)page_size;
    vb->pages_per_entry = pages_per_entry;

    // Allocate DMA-mapped data buffer (64KB aligned for NVMe).
    const uint64_t align = 64ULL * 1024ULL;
    const size_t data_bytes = (size_t)n_entries * page_size;
    void* raw = nullptr;
    cudaMalloc(&raw, data_bytes + align);
    if (!raw) { fprintf(stderr, "victim_create: failed to allocate data buffer\n"); cudaFree(vb); return nullptr; }
    uint64_t raw_u64 = (uint64_t)raw;
    uint64_t aligned_u64 = (raw_u64 + align) & ~(align - 1ULL);
    void* aligned = (void*)aligned_u64;
    cudaMemset(aligned, 0, data_bytes);

    vb->data     = (uint8_t*)aligned;
    vb->data_raw = raw;

    // DMA-map the data buffer.
    nvm_dma_t* dma = nullptr;
    int err = nvm_dma_map_device(&dma, ctrl, aligned, data_bytes);
    if (err != 0 || !dma) {
        fprintf(stderr, "victim_create: failed to DMA-map data (err=%d)\n", err);
        cudaFree(raw);
        cudaFree(vb);
        return nullptr;
    }
    vb->data_dma = dma;

    // Build data_ioaddrs[v * ppe + j] = physical address of j-th 4KB sub-page of entry v.
    const uint32_t ppe = pages_per_entry;
    const uint32_t n_sub = n_entries * ppe;
    uint64_t* h_io = new uint64_t[n_sub];
    // DMA page_size is typically 4KB (ctrl->page_size). Each victim entry spans ppe DMA pages.
    // dma->ioaddrs[k] is the physical address of the k-th DMA page in the data buffer.
    // Entry v occupies DMA pages [v*ppe .. v*ppe+ppe-1].
    for (uint32_t v = 0; v < n_entries; v++) {
        for (uint32_t j = 0; j < ppe; j++) {
            uint32_t dma_page_idx = v * ppe + j;
            if (dma_page_idx < dma->n_ioaddrs) {
                h_io[v * ppe + j] = (uint64_t)dma->ioaddrs[dma_page_idx];
            } else {
                // Fallback: compute from base + offset
                h_io[v * ppe + j] = (uint64_t)dma->ioaddrs[0] + (uint64_t)dma_page_idx * ctrl->page_size;
            }
        }
    }
    uint64_t* d_io = nullptr;
    cudaMalloc(&d_io, n_sub * sizeof(uint64_t));
    cudaMemcpy(d_io, h_io, n_sub * sizeof(uint64_t), cudaMemcpyHostToDevice);
    delete[] h_io;
    vb->data_ioaddrs = d_io;

    // Allocate metadata arrays.
    cudaMalloc(&vb->byte_offsets,  (size_t)n_entries * sizeof(uint64_t));
    cudaMalloc((void**)&vb->states,        (size_t)n_entries * sizeof(uint32_t));
    cudaMalloc((void**)&vb->reader_counts, (size_t)n_entries * sizeof(uint32_t));

#if IO_SCHED_VICTIM_SWAP
    // Build entry_vaddrs[v] = virtual address of victim entry v's data buffer.
    {
        uint64_t* h_va = new uint64_t[n_entries];
        for (uint32_t v = 0; v < n_entries; v++)
            h_va[v] = (uint64_t)(vb->data + (uint64_t)v * page_size);
        cudaMalloc(&vb->entry_vaddrs, (size_t)n_entries * sizeof(uint64_t));
        cudaMemcpy(vb->entry_vaddrs, h_va, (size_t)n_entries * sizeof(uint64_t), cudaMemcpyHostToDevice);
        delete[] h_va;
    }
#endif

    // Initialize: states = FREE, byte_offsets = UINT64_MAX, reader_counts = 0.
    cudaMemset((void*)vb->states,        0, (size_t)n_entries * sizeof(uint32_t));
    cudaMemset((void*)vb->reader_counts, 0, (size_t)n_entries * sizeof(uint32_t));
    {
        uint64_t* h_bo = new uint64_t[n_entries];
        for (uint32_t i = 0; i < n_entries; i++) h_bo[i] = UINT64_MAX;
        cudaMemcpy(vb->byte_offsets, h_bo, n_entries * sizeof(uint64_t), cudaMemcpyHostToDevice);
        delete[] h_bo;
    }

    // Initialize alloc_ticket.
    cudaMemset(&vb->alloc_ticket, 0, sizeof(vb->alloc_ticket));
    // Initialize profiling counters.
    cudaMemset(&vb->prof_fast_count,     0, sizeof(vb->prof_fast_count));
    cudaMemset(&vb->prof_fallback_count, 0, sizeof(vb->prof_fallback_count));
    cudaMemset(&vb->prof_claim_cycles,   0, sizeof(vb->prof_claim_cycles));
    cudaMemset(&vb->prof_memcpy_cycles,  0, sizeof(vb->prof_memcpy_cycles));
    cudaMemset(&vb->prof_enqueue_cycles, 0, sizeof(vb->prof_enqueue_cycles));

    cudaDeviceSynchronize();
    printf("victim_create: %u entries, page_size=%u, ppe=%u, data=%.2f MiB\n",
           n_entries, (uint32_t)page_size, ppe,
           (double)data_bytes / (1024.0 * 1024.0));
    return vb;
}

// Free a victim buffer and all its allocations.
inline void victim_destroy(victim_buffer_t* vb)
{
    if (!vb) return;
    if (vb->data_ioaddrs)   cudaFree(vb->data_ioaddrs);
    if (vb->byte_offsets)   cudaFree(vb->byte_offsets);
#if IO_SCHED_VICTIM_SWAP
    if (vb->entry_vaddrs)   cudaFree(vb->entry_vaddrs);
#endif
    if (vb->states)         cudaFree((void*)vb->states);
    if (vb->reader_counts)  cudaFree((void*)vb->reader_counts);
    if (vb->data_dma)       nvm_dma_unmap(vb->data_dma);
    if (vb->data_raw)       cudaFree(vb->data_raw);
    cudaFree(vb);
}

#endif // IO_SCHED_VICTIM_BUFFER

#endif // __IO_SCHEDULER_CUH__
