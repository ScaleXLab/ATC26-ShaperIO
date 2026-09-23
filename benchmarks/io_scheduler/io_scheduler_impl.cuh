#ifndef __IO_SCHEDULER_IMPL_CUH__
#define __IO_SCHEDULER_IMPL_CUH__

// ---------------------------------------------------------------------------
// IO Scheduler Kernel Implementation
//
// Persistent thread block (128 threads / 4 warps) that drains an io_queue_t,
// sorts requests by LBA, merges adjacent IOs up to 2 MB, and submits them
// to the NVMe controller via the minimal no-enqueue-second path.
// ---------------------------------------------------------------------------

#include <io_scheduler.cuh>

// ---------------------------------------------------------------------------
// globaltimer (nanosecond GPU clock)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long io_sched_globaltimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

// ---------------------------------------------------------------------------
// Bitonic sort for kSchedMaxBatch uint64_t keys using kSchedThreads threads.
//
// kSchedMaxBatch must be a power of 2 and a multiple of kSchedThreads.
// Each thread handles kSchedElemsPerThread elements in strided layout
// (tid, tid+T, tid+2T, ...) for coalesced shared-memory access.
// When kSchedElemsPerThread == 1 this is equivalent to the classic
// 1-element-per-thread bitonic sort.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void io_sched_bitonic_sort(volatile uint64_t* keys)
{
    const uint32_t tid = (uint32_t)threadIdx.x;

    for (uint32_t k = 2; k <= kSchedMaxBatch; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (uint32_t e = 0; e < kSchedElemsPerThread; e++) {
                uint32_t ix  = tid + e * kSchedThreads;
                uint32_t ixj = ix ^ j;
                if (ixj > ix) {
                    const bool up = ((ix & k) == 0);
                    uint64_t a = keys[ix];
                    uint64_t b = keys[ixj];
                    if ((a > b) == up) {
                        keys[ix]  = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// Radix sort for global-memory key-value pairs.
//
// LSB-first, 8-bit radix (256 buckets), kSchedThreads cooperating threads.
// Sorts keys[0..N-1] ascending; vals[0..N-1] permuted accordingly.
// keys_a/vals_a are scratch double-buffers of the same capacity.
// n_passes must be EVEN so the result ends up in the original keys/vals.
// ---------------------------------------------------------------------------
#if IO_SCHED_GLOBAL_SORT
__device__ __noinline__
void io_sched_radix_sort(
    uint64_t* __restrict__ keys,   uint32_t* __restrict__ vals,
    uint64_t* __restrict__ keys_a, uint32_t* __restrict__ vals_a,
    uint32_t N, uint32_t n_passes)
{
    __shared__ uint32_t s_hist[256];
    const uint32_t tid = threadIdx.x;

    for (uint32_t pass = 0; pass < n_passes; pass++) {
        const uint32_t shift = pass * 8;

        // 1. Clear histogram
        for (uint32_t i = tid; i < 256; i += kSchedThreads)
            s_hist[i] = 0;
        __syncthreads();

        // 2. Build histogram
        for (uint32_t i = tid; i < N; i += kSchedThreads)
            atomicAdd_block(&s_hist[(keys[i] >> shift) & 0xFF], 1);
        __syncthreads();

        // 3. Exclusive prefix sum (thread 0, serial over 256 entries)
        if (tid == 0) {
            uint32_t s = 0;
            for (int d = 0; d < 256; d++) {
                uint32_t c = s_hist[d];
                s_hist[d] = s;
                s += c;
            }
        }
        __syncthreads();

        // 4. Scatter to alt buffer (STABLE: thread 0 serial to preserve input order.
        //    Parallel atomicAdd would be unstable within each bucket, breaking
        //    the LSB-first radix sort invariant.)
        if (tid == 0) {
            for (uint32_t i = 0; i < N; i++) {
                uint32_t d = (keys[i] >> shift) & 0xFF;
                uint32_t p = s_hist[d]++;
                keys_a[p] = keys[i];
                vals_a[p] = vals[i];
            }
        }
        __syncthreads();

        // 5. Swap src ↔ alt (local pointer swap, all threads agree)
        { uint64_t* t = keys; keys = keys_a; keys_a = t; }
        { uint32_t* t = vals; vals = vals_a; vals_a = t; }
    }
    // n_passes is even → result is in original keys/vals buffer.
}
#endif  // IO_SCHED_GLOBAL_SORT (v1 radix sort)

#if IO_SCHED_GLOBAL_SORT >= 2
// ---------------------------------------------------------------------------
// Radix sort with PARALLEL scatter (v2).
//
// Steps 1-3 (histogram, prefix sum) are identical to v1.
// Step 4 replaces thread-0 serial scatter with 128-thread warp-sequential
// stable scatter using __match_any_sync + __popc(lane_mask_lt).
//
// Stability guarantee:
//   - Within a warp: lower lane = lower input index → lower output position
//     (via lane_mask_lt ranking).
//   - Across warps: warp 0 claims positions before warp 1, etc.
//     (__syncthreads barrier between each warp's s_hist update).
//   - Across chunks: processed in ascending order of input index.
//
// Requires SM70+ for __match_any_sync.
// ---------------------------------------------------------------------------
__device__ __noinline__
void io_sched_radix_sort_v2(
    uint64_t* __restrict__ keys,   uint32_t* __restrict__ vals,
    uint64_t* __restrict__ keys_a, uint32_t* __restrict__ vals_a,
    uint32_t N, uint32_t n_passes)
{
    __shared__ uint32_t s_hist[256];
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane    = tid & 31;
    const uint32_t lane_mask_lt = (1u << lane) - 1u;

    for (uint32_t pass = 0; pass < n_passes; pass++) {
        const uint32_t shift = pass * 8;

        // 1. Clear histogram
        for (uint32_t i = tid; i < 256; i += kSchedThreads)
            s_hist[i] = 0;
        __syncthreads();

        // 2. Build histogram (128 threads)
        for (uint32_t i = tid; i < N; i += kSchedThreads)
            atomicAdd_block(&s_hist[(keys[i] >> shift) & 0xFF], 1);
        __syncthreads();

        // 3. Exclusive prefix sum (thread 0, 256 entries — very fast)
        if (tid == 0) {
            uint32_t s = 0;
            for (int d = 0; d < 256; d++) {
                uint32_t c = s_hist[d];
                s_hist[d] = s;
                s += c;
            }
        }
        __syncthreads();

        // 4. Parallel stable scatter (128 threads, warp-sequential)
        for (uint32_t chunk = 0; chunk < N; chunk += kSchedThreads) {
            uint32_t idx = chunk + tid;
            bool valid = (idx < N);

            uint32_t digit  = 0;
            uint64_t my_key = 0;
            uint32_t my_val = 0;
            if (valid) {
                my_key = keys[idx];
                my_val = vals[idx];
                digit  = (my_key >> shift) & 0xFF;
            }

            // Warp-level: find peers with the same digit.
            uint32_t valid_mask = __ballot_sync(0xFFFFFFFF, valid);
            uint32_t peer_mask  = 0;
            if (valid) {
                peer_mask = __match_any_sync(valid_mask, digit);
            }
            uint32_t local_rank = __popc(peer_mask & lane_mask_lt);
            uint32_t count      = __popc(peer_mask);

            // Warp-sequential: warp 0 claims output positions first,
            // then warp 1, etc.  Guarantees cross-warp stability.
            uint32_t my_pos = 0;
            for (uint32_t w = 0; w < 4; w++) {
                if (warp_id == w && valid && local_rank == 0) {
                    my_pos = s_hist[digit];
                    s_hist[digit] += count;
                }
                __syncthreads();
            }

            // Broadcast base position from rank-0 lane to all peers.
            if (valid) {
                uint32_t src_lane = __ffs(peer_mask) - 1u;
                my_pos = __shfl_sync(peer_mask, my_pos, src_lane);
                keys_a[my_pos + local_rank] = my_key;
                vals_a[my_pos + local_rank] = my_val;
            }
        }
        __syncthreads();

        // 5. Swap src ↔ alt
        { uint64_t* t = keys; keys = keys_a; keys_a = t; }
        { uint32_t* t = vals; vals = vals_a; vals_a = t; }
    }
}
#endif  // IO_SCHED_GLOBAL_SORT >= 2

// ---------------------------------------------------------------------------
// IO Scheduler Persistent Kernel (Write)
//
// Launch with: <<<1, 128>>> on a dedicated CUDA stream.
// ---------------------------------------------------------------------------
__global__ void io_scheduler_kernel(Controller** ctrls, page_cache_d_t* pc,
                                    io_queue_t* q)
{
    if (blockIdx.x != 0) return;
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (tid >= kSchedThreads) return;

    QueuePair* qp = &ctrls[0]->d_qps[q->sched_qp_base];
    const uint64_t blk_log = qp->block_size_log;

    // Thread 0 timing accumulators (nanoseconds).
    uint64_t acc_drain = 0, acc_sort = 0, acc_merge = 0, acc_nvme = 0, acc_submit = 0;
    uint64_t batch_count = 0;
    uint64_t acc_2mb_splits = 0;

#if IO_SCHED_GLOBAL_SORT
    // ========== Global sort mode ==========
    // Best-effort: drain all available → radix sort → chunked merge+submit → notify.
    __shared__ struct {
        uint64_t lbas[kSchedMaxBatch];
        uint32_t nblks[kSchedMaxBatch];
        uint64_t prp1s[kSchedMaxBatch];
        uint64_t prp2s[kSchedMaxBatch];
    } sh_merged;
    __shared__ uint32_t sh_base_head;
    __shared__ uint32_t sh_accumulated;
    __shared__ uint32_t sh_avail;
    __shared__ uint32_t sh_n_groups;
    __shared__ uint32_t sh_drain_ns;  // adaptive drain wait (64..8192 ns)
    __shared__ uint32_t sh_wmask[4]; // warp ballot masks (used by parallel drain/merge)

    uint64_t* g_keys     = q->gsort_keys;
    uint64_t* g_keys_alt = q->gsort_keys_alt;
    uint32_t* g_vals     = q->gsort_vals;
    uint32_t* g_vals_alt = q->gsort_vals_alt;
    uint64_t* g_pc       = q->gsort_pc_entries;
    uint8_t*  g_nb       = q->gsort_nbytes;

    for (;;) {
        unsigned long long t0 = 0, t1 = 0, t2 = 0, t2b = 0, t3 = 0;
        if (tid == 0) t0 = io_sched_globaltimer();

        // ── Phase 1: ACCUMULATE (with drain-wait) ──
        if (tid == 0) {
            sh_base_head   = q->cons_head.load(simt::memory_order_relaxed);
            sh_accumulated = 0;
            if (batch_count == 0) sh_drain_ns = 8192;  // init on first batch
        }
        __syncthreads();

        for (;;) {
#if IO_SCHED_PARALLEL_DRAIN
            // ── Parallel scan + drain (warp ballot, 128 entries/round) ──
            {
                const uint32_t lane    = tid & 31;
                const uint32_t warp_id = tid >> 5;

                // Thread 0 computes how many slots could possibly be ready.
                if (tid == 0) {
                    uint32_t pending = q->prod_tail.load(simt::memory_order_acquire)
                                     - (sh_base_head + sh_accumulated);
                    uint32_t room    = q->capacity - sh_accumulated;
                    sh_avail = (pending < room) ? pending : room;
                    if (sh_avail > kSchedThreads) sh_avail = kSchedThreads;
                }
                __syncthreads();
                uint32_t round_max = sh_avail;

                // Each thread checks its own slot.
                uint32_t my_slot = sh_base_head + sh_accumulated + tid;
                uint32_t my_idx  = my_slot & q->capacity_mask;
                bool ready = (tid < round_max) &&
                             (q->prod_seq[my_idx] == my_slot + 1u);

                uint32_t wmask = __ballot_sync(0xFFFFFFFF, ready);
                if (lane == 0) sh_wmask[warp_id] = wmask;
                __syncthreads();

                // Thread 0: count consecutive ready entries across 4 warps.
                if (tid == 0) {
                    uint32_t a = 0;
                    for (int w = 0; w < 4; w++) {
                        if (sh_wmask[w] == 0xFFFFFFFF) { a += 32; }
                        else { a += __ffs(~sh_wmask[w]) - 1; break; }
                    }
                    if (a > round_max) a = round_max;
                    sh_avail = a;
                }
                __syncthreads();

                uint32_t avail = sh_avail;

                if (avail > 0) {
                    // Fused drain: ready threads directly load to gsort.
                    if (tid < avail) {
                        io_request_t e = q->entries[my_idx];
                        uint32_t acc = sh_accumulated;
                        g_keys[acc + tid] = e.byte_offset;
                        g_pc[acc + tid]   = e.pc_entry;
                        g_nb[acc + tid]   = e.n_bytes_log2;
                        g_vals[acc + tid] = acc + tid;
                    }
                    __syncthreads();
                    if (tid == 0) sh_accumulated += avail;
                    __syncthreads();
                    if (avail == kSchedThreads) continue;  // full round → try more
                    // partial round → fall through to adaptive wait / sort
                }
            }

            // avail == 0 or partial → handle shutdown / adaptive wait
            {
                uint32_t avail = sh_avail;  // re-read from the block above

                if (avail == 0 && sh_accumulated == 0) {
                    // Empty queue — check shutdown.
                    if (q->shutdown.load(simt::memory_order_acquire)) {
                        if (tid == 0) {
                            sh_avail = io_sched_scan_ready(q, sh_base_head, q->capacity);
                        }
                        __syncthreads();
                        if (sh_avail == 0) {
                            if (tid == 0) {
                                q->sched_drain_ns    = acc_drain;
                                q->sched_sort_ns     = acc_sort;
                                q->sched_merge_ns    = acc_merge;
                                q->sched_nvme_ns     = acc_nvme;
                                q->sched_submit_ns   = acc_submit;
                                q->sched_batch_count = batch_count;
                                q->sched_2mb_splits  = acc_2mb_splits;
                            }
                            return;
                        }
                        continue;  // avail > 0, drain at top of loop
                    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    if (tid == 0) __nanosleep(64);
#endif
                    __syncthreads();
                    continue;
                }

                if (sh_accumulated == 0) {
                    // partial round, nothing accumulated yet — keep trying
                    continue;
                }

                // accumulated > 0, no new entries → adaptive wait before sort.
                bool drain_more = false;
                uint32_t ns = sh_drain_ns;
                while (ns >= 64) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    if (tid == 0) __nanosleep(ns);
#endif
                    __syncthreads();

                    if (tid == 0) {
                        sh_avail = io_sched_scan_ready(q,
                            sh_base_head + sh_accumulated,
                            q->capacity - sh_accumulated);
                    }
                    __syncthreads();

                    if (sh_avail > 0) {
                        drain_more = true;
                        break;
                    }
                    ns /= 2;
                }
                if (drain_more) {
                    if (tid == 0 && sh_drain_ns < 8192) sh_drain_ns *= 2;
                    continue;
                }
                if (tid == 0 && sh_drain_ns > 64) sh_drain_ns /= 2;
                break;  // proceed to sort
            }
#else
            // ── Original serial scan + parallel drain ──
            if (tid == 0) {
                uint32_t new_avail = io_sched_scan_ready(q,
                    sh_base_head + sh_accumulated,
                    q->capacity - sh_accumulated);
                sh_avail = new_avail;
            }
            __syncthreads();

            uint32_t avail = sh_avail;

            if (avail > 0) {
                // Drain entries (all threads cooperate).
                uint32_t acc  = sh_accumulated;
                uint32_t base = sh_base_head;
                for (uint32_t i = tid; i < avail; i += kSchedThreads) {
                    uint32_t ring_idx = (base + acc + i) & q->capacity_mask;
                    io_request_t e = q->entries[ring_idx];
                    g_keys[acc + i] = e.byte_offset;
                    g_pc[acc + i]   = e.pc_entry;
                    g_nb[acc + i]   = e.n_bytes_log2;
                    g_vals[acc + i] = acc + i;
                }
                __syncthreads();

                if (tid == 0) sh_accumulated += avail;
                __syncthreads();
                continue;
            }

            // avail == 0
            if (sh_accumulated == 0) {
                // Empty queue — check shutdown.
                if (q->shutdown.load(simt::memory_order_acquire)) {
                    if (tid == 0) {
                        sh_avail = io_sched_scan_ready(q, sh_base_head, q->capacity);
                    }
                    __syncthreads();
                    if (sh_avail == 0) {
                        if (tid == 0) {
                            q->sched_drain_ns    = acc_drain;
                            q->sched_sort_ns     = acc_sort;
                            q->sched_merge_ns    = acc_merge;
                            q->sched_nvme_ns     = acc_nvme;
                            q->sched_submit_ns   = acc_submit;
                            q->sched_batch_count = batch_count;
                            q->sched_2mb_splits  = acc_2mb_splits;
                        }
                        return;
                    }
                    continue;  // avail > 0, drain at top of loop
                }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                if (tid == 0) __nanosleep(64);
#endif
                __syncthreads();
                continue;
            }

            // accumulated > 0 but avail == 0 → adaptive wait before sort.
            {
                bool drain_more = false;
                uint32_t ns = sh_drain_ns;
                while (ns >= 64) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    if (tid == 0) __nanosleep(ns);
#endif
                    __syncthreads();

                    if (tid == 0) {
                        sh_avail = io_sched_scan_ready(q,
                            sh_base_head + sh_accumulated,
                            q->capacity - sh_accumulated);
                    }
                    __syncthreads();

                    if (sh_avail > 0) {
                        drain_more = true;
                        break;
                    }
                    ns /= 2;
                }
                if (drain_more) {
                    if (tid == 0 && sh_drain_ns < 8192) sh_drain_ns *= 2;
                    continue;
                }
                if (tid == 0 && sh_drain_ns > 64) sh_drain_ns /= 2;
                break;  // proceed to sort
            }
#endif  // IO_SCHED_PARALLEL_DRAIN
        }

        uint32_t accumulated = sh_accumulated;
        uint32_t base_head   = sh_base_head;
        if (tid == 0) { t1 = io_sched_globaltimer(); acc_drain += (t1 - t0); }

#if IO_SCHED_NOSORT
        // ── NOSORT: skip sort and merge, submit each entry individually ──
        {
            if (tid == 0) {
                t2 = io_sched_globaltimer();  // sort time = 0
                QueuePair* qp = &ctrls[0]->d_qps[q->sched_qp_base];
                const uint32_t ppe = pc->pages_per_entry;

                // Read-aware pause check (same as normal path).
                if (q->sched_paused.load(simt::memory_order_acquire)) {
                    uint32_t n_read_qps = q->sched_qp_base;
                    for (;;) {
                        uint32_t snap_sub[128], snap_cmp[128];
                        for (uint32_t i = 0; i < n_read_qps; i++) {
                            snap_sub[i] = ctrls[0]->d_qps[i].sq.in_ticket;
                            snap_cmp[i] = ctrls[0]->d_qps[i].cq.tail;
                        }
                        __nanosleep(10000);
                        bool changed = false;
                        for (uint32_t i = 0; i < n_read_qps; i++) {
                            if (ctrls[0]->d_qps[i].sq.in_ticket != snap_sub[i] ||
                                ctrls[0]->d_qps[i].cq.tail != snap_cmp[i]) {
                                changed = true; break;
                            }
                        }
                        if (!changed) break;
                    }
                }

                unsigned long long t_nvme_start = io_sched_globaltimer();
                for (uint32_t i = 0; i < accumulated; i++) {
                    uint64_t pe = g_pc[i];
                    uint64_t byte_off = g_keys[i];
                    uint64_t n_bytes = io_sched_decode_nbytes(g_nb[i]);
                    uint64_t lba = byte_off >> qp->block_size_log;
                    uint64_t n_blocks = n_bytes >> qp->block_size_log;

                    // Resolve PRP from pc_entry or victim entry.
                    uint64_t prp1, prp2;
#if IO_SCHED_VICTIM_BUFFER
                    if (pe & VICTIM_PC_ENTRY_BIT) {
                        uint32_t v_idx = (uint32_t)(pe & VICTIM_PC_ENTRY_MASK);
                        victim_buffer_t* vb = pc->victim;
                        prp1 = vb->data_ioaddrs[v_idx * ppe];
                        prp2 = (ppe > 1) ? vb->data_ioaddrs[v_idx * ppe + 1] : 0;
                    } else
#endif
                    {
                        uint32_t pc_idx = (uint32_t)pe;
                        prp1 = pc->page_ioaddrs[pc_idx * ppe];
                        prp2 = (ppe > 1) ? pc->page_ioaddrs[pc_idx * ppe + 1] : 0;
                    }
                    write_data_merged_no_second(qp, lba, n_blocks, prp1, prp2);
                }
                unsigned long long t_nvme_end = io_sched_globaltimer();
                acc_nvme += (t_nvme_end - t_nvme_start);
                t2b = t_nvme_end;
                sh_n_groups = accumulated;  // no merge: 1 group per entry
            }
            __syncthreads();
        }
#else  // !IO_SCHED_NOSORT
        // ── Phase 2: RADIX SORT (6 passes = 48 bits, even → result in original buf) ──
#if IO_SCHED_GLOBAL_SORT >= 2
        io_sched_radix_sort_v2(g_keys, g_vals, g_keys_alt, g_vals_alt,
                               accumulated, 6);
#else
        io_sched_radix_sort(g_keys, g_vals, g_keys_alt, g_vals_alt,
                            accumulated, 6);
#endif
        __syncthreads();
        if (tid == 0) { t2 = io_sched_globaltimer(); acc_sort += (t2 - t1); }

        // ── Phase 3+4: CHUNKED MERGE + SUBMIT ──
#if IO_SCHED_PARALLEL_MERGE
        // ── Parallel merge: boundary detection + parallel PRP construction ──
        {
            const uint32_t ppe = pc->pages_per_entry;
            const uint32_t lane    = tid & 31;
            const uint32_t warp_id = tid >> 5;
            uint32_t* g_starts = q->gsort_group_starts;

            // Step 1: Parallel boundary detection.
            // 128 threads scan sorted g_keys[] for LBA discontinuities.
            __shared__ uint32_t sh_boundary_count;
            if (tid == 0) sh_boundary_count = 0;
            __syncthreads();

            for (uint32_t bbase = 0; bbase < accumulated; bbase += kSchedThreads) {
                uint32_t pos = bbase + tid;
                bool is_boundary = false;
                if (pos < accumulated) {
                    if (pos == 0) {
                        is_boundary = true;
                    } else {
                        uint32_t orig_prev = g_vals[pos - 1];
                        uint64_t prev_end  = g_keys[pos - 1]
                                           + io_sched_decode_nbytes(g_nb[orig_prev]);
                        is_boundary = (g_keys[pos] != prev_end);
                    }
                }

                uint32_t wmask = __ballot_sync(0xFFFFFFFF, is_boundary);
                if (lane == 0) sh_wmask[warp_id] = wmask;
                __syncthreads();

                // Thread 0: deterministically extract boundary positions in order.
                if (tid == 0) {
                    for (int w = 0; w < 4; w++) {
                        uint32_t m = sh_wmask[w];
                        while (m != 0) {
                            uint32_t bit = __ffs(m) - 1;
                            g_starts[sh_boundary_count++] = bbase + w * 32 + bit;
                            m &= m - 1;
                        }
                    }
                }
                __syncthreads();
            }

            // Step 1b: Detect and split groups exceeding kSchedMergeTarget (2MB).
            // Detection always runs (O(n_groups)); splitting only when needed.
            if (tid == 0) {
                uint32_t n_orig = sh_boundary_count;
                uint32_t n_oversized = 0;
                for (uint32_t i = 0; i < n_orig; i++) {
                    uint32_t start = g_starts[i];
                    uint32_t end = (i + 1 < n_orig) ? g_starts[i + 1] : accumulated;
                    uint64_t last_bytes = io_sched_decode_nbytes(g_nb[g_vals[end - 1]]);
                    uint64_t run_bytes = g_keys[end - 1] - g_keys[start] + last_bytes;
                    if (run_bytes > kSchedMergeTarget) n_oversized++;
                }
                acc_2mb_splits += n_oversized;
                if (n_oversized > 0) {
                    // Write refined list to g_starts[n_orig..], then copy back.
                    uint32_t* dst = g_starts + n_orig;
                    uint32_t refined = 0;
                    for (uint32_t i = 0; i < n_orig; i++) {
                        uint32_t start = g_starts[i];
                        uint32_t end = (i + 1 < n_orig) ? g_starts[i + 1] : accumulated;
                        uint64_t last_bytes = io_sched_decode_nbytes(g_nb[g_vals[end - 1]]);
                        uint64_t run_bytes = g_keys[end - 1] - g_keys[start] + last_bytes;
                        if (run_bytes <= kSchedMergeTarget) {
                            dst[refined++] = start;
                        } else {
                            uint32_t stride = (uint32_t)(kSchedMergeTarget /
                                io_sched_decode_nbytes(g_nb[g_vals[start]]));
                            if (stride == 0) stride = 1;
                            for (uint32_t pos = start; pos < end; pos += stride)
                                dst[refined++] = pos;
                        }
                    }
                    for (uint32_t i = 0; i < refined; i++) g_starts[i] = dst[i];
                    sh_boundary_count = refined;
                }
            }
            __syncthreads();

            uint32_t n_coarse_groups = sh_boundary_count;

            // Step 2: Parallel PRP construction in chunks.
            uint32_t gpos = 0;
            uint32_t total_merged_groups = 0;
            unsigned long long t_nvme_batch = 0;

            while (gpos < n_coarse_groups) {
                uint32_t chunk = n_coarse_groups - gpos;
                if (chunk > q->prp_pool_n_slots) chunk = q->prp_pool_n_slots;
                if (chunk > kSchedMaxBatch) chunk = kSchedMaxBatch;

                // 128 threads parallel PRP construction.
                for (uint32_t g = tid; g < chunk; g += kSchedThreads) {
                    uint32_t grp   = gpos + g;
                    uint32_t start = g_starts[grp];
                    uint32_t end   = (grp + 1 < n_coarse_groups)
                                   ? g_starts[grp + 1] : accumulated;
                    uint32_t run_count = end - start;
                    uint32_t slot = g;  // deterministic position

                    // O(1) run_bytes from sorted + consecutive property.
                    uint64_t start_offset = g_keys[start];
                    uint64_t last_bytes   = io_sched_decode_nbytes(g_nb[g_vals[end - 1]]);
                    uint64_t run_bytes    = g_keys[end - 1] - start_offset + last_bytes;

                    // Build PRP for this merged group.
                    uint32_t orig_0  = g_vals[start];
                    uint64_t first_pe = g_pc[orig_0];
#if IO_SCHED_VICTIM_BUFFER
                    uint64_t prp1;
                    if (first_pe & VICTIM_PC_ENTRY_BIT) {
                        uint32_t v0 = (uint32_t)(first_pe & VICTIM_PC_ENTRY_MASK);
                        prp1 = pc->victim->data_ioaddrs[v0 * ppe];
                    } else {
                        prp1 = pc->page_ioaddrs[first_pe * ppe];
                    }
#else
                    uint64_t prp1 = pc->page_ioaddrs[first_pe * ppe];
#endif
                    uint64_t prp2 = 0;
                    uint32_t total_subpages = run_count * ppe;

                    if (total_subpages <= 1) {
                        prp2 = 0;
                    } else if (total_subpages == 2) {
                        if (ppe >= 2) {
#if IO_SCHED_VICTIM_BUFFER
                            if (first_pe & VICTIM_PC_ENTRY_BIT) {
                                uint32_t v0 = (uint32_t)(first_pe & VICTIM_PC_ENTRY_MASK);
                                prp2 = pc->victim->data_ioaddrs[v0 * ppe + 1];
                            } else
#endif
                            {
                                prp2 = pc->page_ioaddrs[first_pe * ppe + 1];
                            }
                        } else {
                            uint64_t pe2 = g_pc[g_vals[start + 1]];
#if IO_SCHED_VICTIM_BUFFER
                            if (pe2 & VICTIM_PC_ENTRY_BIT) {
                                uint32_t v2 = (uint32_t)(pe2 & VICTIM_PC_ENTRY_MASK);
                                prp2 = pc->victim->data_ioaddrs[v2 * ppe];
                            } else
#endif
                            {
                                prp2 = pc->page_ioaddrs[pe2 * ppe];
                            }
                        }
                    } else if (q->prp_pool_vaddr != nullptr) {
                        volatile uint64_t* plist =
                            (volatile uint64_t*)q->prp_pool_vaddr[slot];
                        uint32_t pi = 0;
                        // First entry's remaining sub-pages.
                        for (uint32_t j = 1; j < ppe; j++) {
#if IO_SCHED_VICTIM_BUFFER
                            if (first_pe & VICTIM_PC_ENTRY_BIT) {
                                uint32_t v0 = (uint32_t)(first_pe & VICTIM_PC_ENTRY_MASK);
                                plist[pi++] = pc->victim->data_ioaddrs[v0 * ppe + j];
                            } else
#endif
                            {
                                plist[pi++] = pc->page_ioaddrs[first_pe * ppe + j];
                            }
                        }
                        // Subsequent entries' all sub-pages.
                        for (uint32_t r = 1; r < run_count; r++) {
                            uint64_t pe = g_pc[g_vals[start + r]];
#if IO_SCHED_VICTIM_BUFFER
                            if (pe & VICTIM_PC_ENTRY_BIT) {
                                uint32_t vr = (uint32_t)(pe & VICTIM_PC_ENTRY_MASK);
                                for (uint32_t j = 0; j < ppe; j++)
                                    plist[pi++] = pc->victim->data_ioaddrs[vr * ppe + j];
                            } else
#endif
                            {
                                for (uint32_t j = 0; j < ppe; j++)
                                    plist[pi++] = pc->page_ioaddrs[pe * ppe + j];
                            }
                        }
                        // prp2 set after threadfence below.
                        prp2 = q->prp_pool_ioaddr[slot];
                    } else {
                        prp2 = pc->prps ? pc->prp2[first_pe] : 0;
                    }

                    sh_merged.lbas[slot]  = start_offset >> blk_log;
                    sh_merged.nblks[slot] = (uint32_t)(run_bytes >> blk_log);
                    sh_merged.prp1s[slot] = prp1;
                    sh_merged.prp2s[slot] = prp2;
                }
                __syncthreads();
                __threadfence_system();  // once for all groups in this chunk

                // Thread 0: submit in kSchedKneeLimit sub-chunks.
                if (tid == 0) {
                    // Read-aware deferral: wait until read QPs are fully idle
                    // (no new submissions AND no pending completions).
                    if (q->sched_paused.load(simt::memory_order_acquire)) {
                        uint32_t n_read_qps = q->sched_qp_base;
                        uint32_t wait_rounds = 0;
                        for (;;) {
                            // Snapshot both in_ticket (submissions) and cq.tail (completions).
                            uint32_t snap_sub[128], snap_cmp[128];
                            for (uint32_t i = 0; i < n_read_qps; i++) {
                                snap_sub[i] = ctrls[0]->d_qps[i].sq.in_ticket.load(simt::memory_order_relaxed);
                                snap_cmp[i] = ctrls[0]->d_qps[i].cq.tail.load(simt::memory_order_relaxed);
                            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                            __nanosleep(10000);  // 10 us
#endif
                            bool changed = false;
                            for (uint32_t i = 0; i < n_read_qps; i++) {
                                if (ctrls[0]->d_qps[i].sq.in_ticket.load(simt::memory_order_relaxed) != snap_sub[i] ||
                                    ctrls[0]->d_qps[i].cq.tail.load(simt::memory_order_relaxed) != snap_cmp[i]) {
                                    changed = true; break;
                                }
                            }
                            if (!changed) break;
                            wait_rounds++;
                        }
                        printf("[wsched] submit-pause: waited %u rounds (%.1f ms) before submit\n",
                               wait_rounds, (float)wait_rounds * 0.01f);
                    }
                    unsigned long long t_sub_start = io_sched_globaltimer();
                    for (uint32_t s = 0; s < chunk; s += kSchedKneeLimit) {
                        uint32_t n = chunk - s;
                        if (n > kSchedKneeLimit) n = kSchedKneeLimit;
#if IO_SCHED_WRITE_BATCH
                        sched_write_batch_merged(pc, qp,
                            &sh_merged.lbas[s], &sh_merged.nblks[s],
                            &sh_merged.prp1s[s], &sh_merged.prp2s[s], n);
#else
                        for (uint32_t gg = 0; gg < n; gg++) {
                            write_data_merged_no_second(qp,
                                sh_merged.lbas[s + gg],
                                (uint64_t)sh_merged.nblks[s + gg],
                                sh_merged.prp1s[s + gg],
                                sh_merged.prp2s[s + gg]);
                        }
#endif
                    }
                    t_nvme_batch += io_sched_globaltimer() - t_sub_start;
                    total_merged_groups += chunk;
                }
                __syncthreads();
                gpos += chunk;
            }

            if (tid == 0) {
                sh_n_groups = total_merged_groups;
                t2b = io_sched_globaltimer();
                acc_nvme += t_nvme_batch;
                acc_merge += (t2b - t2) - t_nvme_batch;
            }
        }
        __syncthreads();
#else
        // ── Original serial merge + submit (thread 0) ──
        if (tid == 0) {
            const uint32_t ppe = pc->pages_per_entry;
            uint32_t scan_pos = 0;
            uint32_t total_merged_groups = 0;
            unsigned long long t_nvme_batch = 0;

            while (scan_pos < accumulated) {
                uint32_t n_groups = 0;

                // Build up to kSchedKneeLimit merge groups (knee-point throttle).
                while (scan_pos < accumulated && n_groups < kSchedKneeLimit) {
                    uint64_t start_offset = g_keys[scan_pos];
                    uint32_t orig_0 = g_vals[scan_pos];
                    uint64_t run_bytes = io_sched_decode_nbytes(g_nb[orig_0]);
                    uint32_t run_count = 1;

                    while (scan_pos + run_count < accumulated &&
                           run_bytes < kSchedMergeTarget) {
                        uint64_t next_offset = g_keys[scan_pos + run_count];
                        uint32_t orig_next   = g_vals[scan_pos + run_count];
                        if (next_offset != start_offset + run_bytes) break;
                        uint64_t next_bytes = io_sched_decode_nbytes(g_nb[orig_next]);
                        if (run_bytes + next_bytes > kSchedMergeTarget) break;
                        run_bytes += next_bytes;
                        run_count++;
                    }

                    // Build PRP for this merged group.
                    uint64_t first_pe = g_pc[orig_0];
#if IO_SCHED_VICTIM_BUFFER
                    uint64_t prp1;
                    if (first_pe & VICTIM_PC_ENTRY_BIT) {
                        uint32_t v0 = (uint32_t)(first_pe & VICTIM_PC_ENTRY_MASK);
                        prp1 = pc->victim->data_ioaddrs[v0 * ppe];
                    } else {
                        prp1 = pc->page_ioaddrs[first_pe * ppe];
                    }
#else
                    uint64_t prp1 = pc->page_ioaddrs[first_pe * ppe];
#endif
                    uint64_t prp2 = 0;
                    uint32_t total_subpages = run_count * ppe;

                    if (total_subpages <= 1) {
                        prp2 = 0;
                    } else if (total_subpages == 2) {
                        if (ppe >= 2) {
#if IO_SCHED_VICTIM_BUFFER
                            if (first_pe & VICTIM_PC_ENTRY_BIT) {
                                uint32_t v0 = (uint32_t)(first_pe & VICTIM_PC_ENTRY_MASK);
                                prp2 = pc->victim->data_ioaddrs[v0 * ppe + 1];
                            } else
#endif
                            {
                                prp2 = pc->page_ioaddrs[first_pe * ppe + 1];
                            }
                        } else {
                            uint64_t pe2 = g_pc[g_vals[scan_pos + 1]];
#if IO_SCHED_VICTIM_BUFFER
                            if (pe2 & VICTIM_PC_ENTRY_BIT) {
                                uint32_t v2 = (uint32_t)(pe2 & VICTIM_PC_ENTRY_MASK);
                                prp2 = pc->victim->data_ioaddrs[v2 * ppe];
                            } else
#endif
                            {
                                prp2 = pc->page_ioaddrs[pe2 * ppe];
                            }
                        }
                    } else if (q->prp_pool_vaddr != nullptr) {
                        volatile uint64_t* plist =
                            (volatile uint64_t*)q->prp_pool_vaddr[n_groups];
                        uint32_t pi = 0;
                        for (uint32_t j = 1; j < ppe; j++) {
#if IO_SCHED_VICTIM_BUFFER
                            if (first_pe & VICTIM_PC_ENTRY_BIT) {
                                uint32_t v0 = (uint32_t)(first_pe & VICTIM_PC_ENTRY_MASK);
                                plist[pi++] = pc->victim->data_ioaddrs[v0 * ppe + j];
                            } else
#endif
                            {
                                plist[pi++] = pc->page_ioaddrs[first_pe * ppe + j];
                            }
                        }
                        for (uint32_t r = 1; r < run_count; r++) {
                            uint64_t pe = g_pc[g_vals[scan_pos + r]];
#if IO_SCHED_VICTIM_BUFFER
                            if (pe & VICTIM_PC_ENTRY_BIT) {
                                uint32_t vr = (uint32_t)(pe & VICTIM_PC_ENTRY_MASK);
                                for (uint32_t j = 0; j < ppe; j++)
                                    plist[pi++] = pc->victim->data_ioaddrs[vr * ppe + j];
                            } else
#endif
                            {
                                for (uint32_t j = 0; j < ppe; j++)
                                    plist[pi++] = pc->page_ioaddrs[pe * ppe + j];
                            }
                        }
                        __threadfence_system();
                        prp2 = q->prp_pool_ioaddr[n_groups];
                    } else {
                        prp2 = pc->prps ? pc->prp2[first_pe] : 0;
                    }

                    sh_merged.lbas[n_groups]  = start_offset >> blk_log;
                    sh_merged.nblks[n_groups] = (uint32_t)(run_bytes >> blk_log);
                    sh_merged.prp1s[n_groups] = prp1;
                    sh_merged.prp2s[n_groups] = prp2;

                    n_groups++;
                    scan_pos += run_count;
                }

                // Batch submit this chunk.
                if (n_groups > 0) {
                    unsigned long long t_sub_start = io_sched_globaltimer();
#if IO_SCHED_WRITE_BATCH
                    sched_write_batch_merged(pc, qp,
                        sh_merged.lbas, sh_merged.nblks,
                        sh_merged.prp1s, sh_merged.prp2s,
                        n_groups);
#else
                    for (uint32_t g = 0; g < n_groups; g++) {
                        write_data_merged_no_second(qp,
                            sh_merged.lbas[g],
                            (uint64_t)sh_merged.nblks[g],
                            sh_merged.prp1s[g],
                            sh_merged.prp2s[g]);
                    }
#endif
                    t_nvme_batch += io_sched_globaltimer() - t_sub_start;
                    total_merged_groups += n_groups;
                }
            }

            sh_n_groups = total_merged_groups;
            t2b = io_sched_globaltimer();
            acc_nvme += t_nvme_batch;
            acc_merge += (t2b - t2) - t_nvme_batch;
        }
        __syncthreads();
#endif  // IO_SCHED_PARALLEL_MERGE
#endif  // IO_SCHED_NOSORT

        // ── Phase 5: NOTIFY (all threads cooperate) ──
        __threadfence();
        for (uint32_t i = tid; i < accumulated; i += kSchedThreads) {
            uint32_t orig     = g_vals[i];
            uint32_t ring_idx = (base_head + orig) & q->capacity_mask;
            q->done_seq[ring_idx]   = base_head + orig + 1;
            q->done_flags[ring_idx] = 1;

#if IO_SCHED_VICTIM_BUFFER
            uint64_t pe = g_pc[orig];
            if (pe & VICTIM_PC_ENTRY_BIT) {
                uint32_t v_idx = (uint32_t)(pe & VICTIM_PC_ENTRY_MASK);
                victim_buffer_t* vb = pc->victim;
                vb->byte_offsets[v_idx] = UINT64_MAX;
                __threadfence();
                if (vb->reader_counts[v_idx].load(simt::memory_order_acquire) == 0) {
                    vb->states[v_idx].store(VICTIM_FREE, simt::memory_order_release);
                }
            }
#endif
        }
        __syncthreads();

        if (tid == 0) {
            q->submitted_ios.fetch_add((uint64_t)accumulated, simt::memory_order_relaxed);
            q->merged_ios.fetch_add((uint64_t)sh_n_groups, simt::memory_order_relaxed);
            q->cons_head.fetch_add(accumulated, simt::memory_order_release);

            t3 = io_sched_globaltimer();
            acc_submit += (t3 - t2b);
            batch_count++;
        }
        __syncthreads();
    }

#else
    // ========== Per-batch sort mode (original) ==========
    __shared__ uint64_t sh_keys[kSchedMaxBatch];
    __shared__ uint8_t  sh_nbytes_log2[kSchedMaxBatch];
    __shared__ uint64_t sh_pc_entries[kSchedMaxBatch];
    __shared__ uint32_t sh_batch_size;
    __shared__ uint32_t sh_head;
    __shared__ uint32_t sh_n_groups;

    // Union: reqs used in Phase 1-2, merged arrays used in Phase 3-4.
    __shared__ union {
        io_request_t reqs[kSchedMaxBatch];
        struct {
            uint64_t lbas[kSchedMaxBatch];
            uint32_t nblks[kSchedMaxBatch];
            uint64_t prp1s[kSchedMaxBatch];
            uint64_t prp2s[kSchedMaxBatch];
            uint16_t first_idx[kSchedMaxBatch];
            uint16_t count[kSchedMaxBatch];
        } merged;
    } sh_buf;

    for (;;) {
        unsigned long long t0 = 0, t1 = 0, t2 = 0, t2b = 0, t3 = 0;
        if (tid == 0) t0 = io_sched_globaltimer();

        // ---- Phase 1: DRAIN ----
        if (tid == 0) {
            uint32_t head = q->cons_head.load(simt::memory_order_relaxed);
            uint32_t avail = io_sched_scan_ready(q, head, kSchedMaxBatch);
            sh_batch_size = avail;
            sh_head = head;
        }
        __syncthreads();

        uint32_t batch = sh_batch_size;

        // Check shutdown when queue is empty.
        if (batch == 0) {
            if (q->shutdown.load(simt::memory_order_acquire)) {
                if (tid == 0) {
                    uint32_t head = q->cons_head.load(simt::memory_order_relaxed);
                    uint32_t avail = io_sched_scan_ready(q, head, kSchedMaxBatch);
                    sh_batch_size = avail;
                    sh_head = head;
                }
                __syncthreads();
                batch = sh_batch_size;
                if (batch == 0) {
                    if (tid == 0) {
                        q->sched_drain_ns    = acc_drain;
                        q->sched_sort_ns     = acc_sort;
                        q->sched_merge_ns    = acc_merge;
                        q->sched_nvme_ns     = acc_nvme;
                        q->sched_submit_ns   = acc_submit;
                        q->sched_batch_count = batch_count;
                        q->sched_2mb_splits  = acc_2mb_splits;
                    }
                    return;
                }
            } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                if (tid == 0) __nanosleep(64);
#endif
                __syncthreads();
                continue;
            }
        }

        // All threads cooperatively load entries + extract nbytes_log2 + pc_entry.
        uint32_t head = sh_head;
        #pragma unroll
        for (uint32_t e = 0; e < kSchedElemsPerThread; e++) {
            uint32_t slot = tid + e * kSchedThreads;
            if (slot < batch) {
                uint32_t idx = (head + slot) & q->capacity_mask;
                sh_buf.reqs[slot] = q->entries[idx];
                sh_keys[slot] = io_sched_make_sort_key(sh_buf.reqs[slot].byte_offset, slot);
                sh_nbytes_log2[slot] = sh_buf.reqs[slot].n_bytes_log2;
                sh_pc_entries[slot]  = sh_buf.reqs[slot].pc_entry;
            } else {
                sh_keys[slot] = UINT64_MAX;
                sh_nbytes_log2[slot] = 0;
                sh_pc_entries[slot]  = 0;
            }
        }
        __syncthreads();
        if (tid == 0) { t1 = io_sched_globaltimer(); acc_drain += (t1 - t0); }

        // ---- Phase 2: SORT ----
        io_sched_bitonic_sort(sh_keys);
        __syncthreads();
        if (tid == 0) { t2 = io_sched_globaltimer(); acc_sort += (t2 - t1); }

        // ---- Phase 3: MERGE (collect into arrays, build PRP) ----
        if (tid == 0) {
            uint32_t n_groups = 0;
            uint64_t total = 0;
            uint32_t i = 0;
            const uint32_t ppe = pc->pages_per_entry;

            while (i < batch) {
                uint64_t key_i = sh_keys[i];
                if (key_i == UINT64_MAX) break;
                uint32_t orig_i = io_sched_key_batch_index(key_i);
                uint64_t start_offset = io_sched_key_byte_offset(key_i);
                uint64_t run_bytes = io_sched_decode_nbytes(sh_nbytes_log2[orig_i]);
                uint32_t run_count = 1;

                while ((i + run_count) < batch && run_bytes < kSchedMergeTarget) {
                    uint64_t key_next = sh_keys[i + run_count];
                    if (key_next == UINT64_MAX) break;
                    uint64_t next_offset = io_sched_key_byte_offset(key_next);
                    if (next_offset != start_offset + run_bytes) break;
                    uint64_t next_bytes = io_sched_decode_nbytes(
                        sh_nbytes_log2[io_sched_key_batch_index(key_next)]);
                    if (run_bytes + next_bytes > kSchedMergeTarget) break;
                    run_bytes += next_bytes;
                    run_count++;
                }

                // Build PRP for this merged group.
                uint64_t first_pe = sh_pc_entries[orig_i];
                uint64_t merged_prp1 = pc->page_ioaddrs[first_pe * ppe];
                uint64_t merged_prp2 = 0;
                uint32_t total_subpages = run_count * ppe;

                if (total_subpages <= 1) {
                    merged_prp2 = 0;
                } else if (total_subpages == 2) {
                    if (ppe >= 2) {
                        merged_prp2 = pc->page_ioaddrs[first_pe * ppe + 1];
                    } else {
                        uint64_t pe2 = sh_pc_entries[io_sched_key_batch_index(sh_keys[i + 1])];
                        merged_prp2 = pc->page_ioaddrs[pe2 * ppe];
                    }
                } else if (q->prp_pool_vaddr != nullptr) {
                    volatile uint64_t* plist =
                        (volatile uint64_t*)q->prp_pool_vaddr[n_groups];
                    uint32_t pi = 0;
                    for (uint32_t j = 1; j < ppe; j++)
                        plist[pi++] = pc->page_ioaddrs[first_pe * ppe + j];
                    for (uint32_t r = 1; r < run_count; r++) {
                        uint64_t pe = sh_pc_entries[
                            io_sched_key_batch_index(sh_keys[i + r])];
                        for (uint32_t j = 0; j < ppe; j++)
                            plist[pi++] = pc->page_ioaddrs[pe * ppe + j];
                    }
                    __threadfence_system();
                    merged_prp2 = q->prp_pool_ioaddr[n_groups];
                } else {
                    merged_prp2 = pc->prps ? pc->prp2[first_pe] : 0;
                }

                sh_buf.merged.lbas[n_groups]      = start_offset >> blk_log;
                sh_buf.merged.nblks[n_groups]     = (uint32_t)(run_bytes >> blk_log);
                sh_buf.merged.prp1s[n_groups]     = merged_prp1;
                sh_buf.merged.prp2s[n_groups]     = merged_prp2;
                sh_buf.merged.first_idx[n_groups] = (uint16_t)i;
                sh_buf.merged.count[n_groups]     = (uint16_t)run_count;

                total += run_count;
                n_groups++;
                i += run_count;
            }

            sh_n_groups = n_groups;
            t2b = io_sched_globaltimer();
            acc_merge += (t2b - t2);
        }
        __syncthreads();

        // ---- Phase 4: SUBMIT ----
        if (tid == 0) {
            const uint32_t n_groups = sh_n_groups;

#if IO_SCHED_WRITE_BATCH
            sched_write_batch_merged(pc, qp,
                sh_buf.merged.lbas,
                sh_buf.merged.nblks,
                sh_buf.merged.prp1s,
                sh_buf.merged.prp2s,
                n_groups);
#else
            for (uint32_t g = 0; g < n_groups; g++) {
                write_data_merged_no_second(qp,
                    sh_buf.merged.lbas[g],
                    (uint64_t)sh_buf.merged.nblks[g],
                    sh_buf.merged.prp1s[g],
                    sh_buf.merged.prp2s[g]);
            }
#endif

            // Notify completion for all original requests.
            __threadfence();
            for (uint32_t g = 0; g < n_groups; g++) {
                const uint16_t first = sh_buf.merged.first_idx[g];
                const uint16_t cnt   = sh_buf.merged.count[g];
                for (uint16_t r = 0; r < cnt; r++) {
                    uint32_t orig = io_sched_key_batch_index(sh_keys[first + r]);
                    uint32_t ring_idx = (head + orig) & q->capacity_mask;
                    const uint32_t done_ticket = head + orig + 1u;
                    q->done_seq[ring_idx] = done_ticket;
                    q->done_flags[ring_idx] = 1;
                }
            }

            // Update counters and advance consumer head.
            uint64_t total_submitted = 0;
            for (uint32_t g = 0; g < n_groups; g++)
                total_submitted += sh_buf.merged.count[g];
            q->submitted_ios.fetch_add(total_submitted, simt::memory_order_relaxed);
            q->merged_ios.fetch_add((uint64_t)n_groups, simt::memory_order_relaxed);
            q->cons_head.fetch_add(batch, simt::memory_order_release);

            t3 = io_sched_globaltimer();
            acc_submit += (t3 - t2b);
            batch_count++;
        }
        __syncthreads();
    }
#endif // IO_SCHED_GLOBAL_SORT
}

// ---------------------------------------------------------------------------
// IO Read Scheduler Persistent Kernel
//
// 32-thread parallel submit variant. Same drain/sort/merge as the write
// scheduler, but Phase 4 fans out merged IOs to 32 threads, each with its
// own QueuePair and page-cache entry.
//
// Launch with: <<<1, 128>>> on a dedicated CUDA stream.
// Requires: Controller with n_qps >= 32, page_cache with n_pages >= 32.
// ---------------------------------------------------------------------------

// Merge-group descriptor stored in shared memory (Phase 3 → Phase 4).
struct io_sched_merge_group_t {
    uint64_t lba;
    uint32_t n_blocks;
    uint16_t first_idx;   // Index into sh_keys where this group starts
    uint16_t count;       // Number of original requests in this group
};

__global__ void io_read_scheduler_kernel(Controller** ctrls, page_cache_d_t* pc,
                                         io_queue_t* q)
{
    if (blockIdx.x != 0) return;
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (tid >= kSchedThreads) return;

    const uint32_t qp_base = q->sched_qp_base;
    const uint64_t blk_log = ctrls[0]->d_qps[qp_base].block_size_log;

    __shared__ uint64_t sh_keys[kSchedMaxBatch];
    __shared__ uint8_t  sh_nbytes_log2[kSchedMaxBatch];
    __shared__ uint64_t sh_pc_entries[kSchedMaxBatch];
    __shared__ uint32_t sh_batch_size;
    __shared__ uint32_t sh_head;
    __shared__ uint32_t sh_n_groups;
    __shared__ uint64_t sh_total_submitted;
#if IO_SCHED_READ_PARALLEL_DRAIN || IO_SCHED_READ_PARALLEL_MERGE
    __shared__ uint32_t sh_wmask[4];
    __shared__ uint32_t sh_avail;
    __shared__ uint32_t sh_accumulated;
#endif
#if IO_SCHED_READ_PARALLEL_MERGE
    __shared__ uint32_t sh_group_starts[kSchedMaxBatch];
    __shared__ uint32_t sh_boundary_count;
#endif

    // Union: reqs used in Phase 1-2 (serial drain only), merged arrays in Phase 3-4.
    __shared__ union {
#if !IO_SCHED_READ_PARALLEL_DRAIN
        io_request_t reqs[kSchedMaxBatch];
#endif
        struct {
            uint64_t lbas[kSchedMaxBatch];
            uint32_t nblks[kSchedMaxBatch];
            uint64_t prp1s[kSchedMaxBatch];
            uint64_t prp2s[kSchedMaxBatch];
            uint16_t first_idx[kSchedMaxBatch];
            uint16_t count[kSchedMaxBatch];
        } merged;
    } sh_buf;

    // Thread 0 timing accumulators.
    uint64_t acc_drain = 0, acc_sort = 0, acc_merge = 0, acc_nvme = 0, acc_submit = 0;
    uint64_t batch_count = 0;
    uint64_t acc_2mb_splits = 0;

    for (;;) {
        unsigned long long t0 = 0, t1 = 0, t2 = 0, t2b = 0, t3 = 0;
        if (tid == 0) t0 = io_sched_globaltimer();

        // ---- Phase 1: DRAIN ----
#if IO_SCHED_READ_PARALLEL_DRAIN
        // ── Parallel drain: warp ballot, multi-round accumulation ──
        if (tid == 0) {
            sh_head = q->cons_head.load(simt::memory_order_relaxed);
            sh_accumulated = 0;
        }
        __syncthreads();

        for (;;) {
            {
                const uint32_t lane    = tid & 31;
                const uint32_t warp_id = tid >> 5;

                if (tid == 0) {
                    uint32_t pending = q->prod_tail.load(simt::memory_order_acquire)
                                     - (sh_head + sh_accumulated);
                    uint32_t room    = kSchedMaxBatch - sh_accumulated;
                    sh_avail = (pending < room) ? pending : room;
                    if (sh_avail > kSchedThreads) sh_avail = kSchedThreads;
                }
                __syncthreads();
                uint32_t round_max = sh_avail;

                uint32_t my_slot = sh_head + sh_accumulated + tid;
                uint32_t my_idx  = my_slot & q->capacity_mask;
                bool ready = (tid < round_max) &&
                             (q->prod_seq[my_idx] == my_slot + 1u);

                uint32_t wmask = __ballot_sync(0xFFFFFFFF, ready);
                if (lane == 0) sh_wmask[warp_id] = wmask;
                __syncthreads();

                if (tid == 0) {
                    uint32_t a = 0;
                    for (int w = 0; w < 4; w++) {
                        if (sh_wmask[w] == 0xFFFFFFFF) { a += 32; }
                        else { a += __ffs(~sh_wmask[w]) - 1; break; }
                    }
                    if (a > round_max) a = round_max;
                    sh_avail = a;
                }
                __syncthreads();

                uint32_t avail = sh_avail;

                if (avail > 0) {
                    if (tid < avail) {
                        io_request_t e = q->entries[my_idx];
                        uint32_t acc = sh_accumulated;
                        sh_keys[acc + tid] = io_sched_make_sort_key(e.byte_offset, acc + tid);
                        sh_nbytes_log2[acc + tid] = e.n_bytes_log2;
                        sh_pc_entries[acc + tid]  = e.pc_entry;
                    }
                    __syncthreads();
                    if (tid == 0) sh_accumulated += avail;
                    __syncthreads();
                    if (avail == kSchedThreads) continue;  // full round → try more
                }
            }

            // avail == 0 or partial → handle shutdown / break
            {
                uint32_t avail = sh_avail;

                if (avail == 0 && sh_accumulated == 0) {
                    if (q->shutdown.load(simt::memory_order_acquire)) {
                        if (tid == 0) {
                            sh_avail = io_sched_scan_ready(q, sh_head, kSchedMaxBatch);
                        }
                        __syncthreads();
                        if (sh_avail == 0) {
                            if (tid == 0) {
                                q->sched_drain_ns    = acc_drain;
                                q->sched_sort_ns     = acc_sort;
                                q->sched_merge_ns    = acc_merge;
                                q->sched_nvme_ns     = acc_nvme;
                                q->sched_submit_ns   = acc_submit;
                                q->sched_batch_count = batch_count;
                                q->sched_2mb_splits  = acc_2mb_splits;
                            }
                            return;
                        }
                        continue;
                    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    if (tid == 0) __nanosleep(64);
#endif
                    __syncthreads();
                    continue;
                }

                if (sh_accumulated == 0) {
                    continue;  // partial round, nothing accumulated yet
                }

                // accumulated > 0, no new entries → proceed to sort (no adaptive wait for reads)
                break;
            }
        }

        uint32_t batch = sh_accumulated;
        uint32_t head  = sh_head;
        // Pad remaining slots with UINT64_MAX for bitonic sort.
        #pragma unroll
        for (uint32_t e = 0; e < kSchedElemsPerThread; e++) {
            uint32_t slot = tid + e * kSchedThreads;
            if (slot >= batch) {
                sh_keys[slot] = UINT64_MAX;
            }
        }
        __syncthreads();
        if (tid == 0) { t1 = io_sched_globaltimer(); acc_drain += (t1 - t0); }

#else
        // ── Original serial drain ──
        if (tid == 0) {
            uint32_t h = q->cons_head.load(simt::memory_order_relaxed);
            uint32_t avail = io_sched_scan_ready(q, h, kSchedMaxBatch);
            sh_batch_size = avail;
            sh_head = h;
        }
        __syncthreads();

        uint32_t batch = sh_batch_size;

        if (batch == 0) {
            if (q->shutdown.load(simt::memory_order_acquire)) {
                if (tid == 0) {
                    uint32_t h = q->cons_head.load(simt::memory_order_relaxed);
                    uint32_t avail = io_sched_scan_ready(q, h, kSchedMaxBatch);
                    sh_batch_size = avail;
                    sh_head = h;
                }
                __syncthreads();
                batch = sh_batch_size;
                if (batch == 0) {
                    if (tid == 0) {
                        q->sched_drain_ns    = acc_drain;
                        q->sched_sort_ns     = acc_sort;
                        q->sched_merge_ns    = acc_merge;
                        q->sched_nvme_ns     = acc_nvme;
                        q->sched_submit_ns   = acc_submit;
                        q->sched_batch_count = batch_count;
                        q->sched_2mb_splits  = acc_2mb_splits;
                    }
                    return;
                }
            } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                if (tid == 0) __nanosleep(64);
#endif
                __syncthreads();
                continue;
            }
        }

        uint32_t head = sh_head;
        #pragma unroll
        for (uint32_t e = 0; e < kSchedElemsPerThread; e++) {
            uint32_t slot = tid + e * kSchedThreads;
            if (slot < batch) {
                uint32_t idx = (head + slot) & q->capacity_mask;
                io_request_t e_req = q->entries[idx];
                sh_keys[slot] = io_sched_make_sort_key(e_req.byte_offset, slot);
                sh_nbytes_log2[slot] = e_req.n_bytes_log2;
                sh_pc_entries[slot]  = e_req.pc_entry;
            } else {
                sh_keys[slot] = UINT64_MAX;
                sh_nbytes_log2[slot] = 0;
                sh_pc_entries[slot]  = 0;
            }
        }
        __syncthreads();
        if (tid == 0) { t1 = io_sched_globaltimer(); acc_drain += (t1 - t0); }
#endif  // IO_SCHED_READ_PARALLEL_DRAIN

        // ---- Phase 2: SORT ----
        io_sched_bitonic_sort(sh_keys);
        __syncthreads();
        if (tid == 0) { t2 = io_sched_globaltimer(); acc_sort += (t2 - t1); }

        // ---- Phase 3+4: MERGE + SUBMIT ----
#if IO_SCHED_READ_PARALLEL_MERGE
        // ── Parallel merge: boundary detection + parallel PRP + knee-point submit ──
        {
            const uint32_t ppe = pc->pages_per_entry;
            const uint32_t lane    = tid & 31;
            const uint32_t warp_id = tid >> 5;

            // Step 1: Parallel boundary detection.
            if (tid == 0) sh_boundary_count = 0;
            __syncthreads();

            for (uint32_t bbase = 0; bbase < batch; bbase += kSchedThreads) {
                uint32_t pos = bbase + tid;
                bool is_boundary = false;
                if (pos < batch) {
                    uint64_t key_pos = sh_keys[pos];
                    if (key_pos == UINT64_MAX) {
                        is_boundary = false;  // padding, not a real entry
                    } else if (pos == 0) {
                        is_boundary = true;
                    } else {
                        uint64_t key_prev = sh_keys[pos - 1];
                        if (key_prev == UINT64_MAX) {
                            is_boundary = true;
                        } else {
                            uint32_t orig_prev = io_sched_key_batch_index(key_prev);
                            uint64_t prev_end  = io_sched_key_byte_offset(key_prev)
                                               + io_sched_decode_nbytes(sh_nbytes_log2[orig_prev]);
                            is_boundary = (io_sched_key_byte_offset(key_pos) != prev_end);
                        }
                    }
                }

                uint32_t wmask = __ballot_sync(0xFFFFFFFF, is_boundary);
                if (lane == 0) sh_wmask[warp_id] = wmask;
                __syncthreads();

                if (tid == 0) {
                    for (int w = 0; w < 4; w++) {
                        uint32_t m = sh_wmask[w];
                        while (m != 0) {
                            uint32_t bit = __ffs(m) - 1;
                            sh_group_starts[sh_boundary_count++] = bbase + w * 32 + bit;
                            m &= m - 1;
                        }
                    }
                }
                __syncthreads();
            }

            // Step 1b: Detect and split groups exceeding kSchedMergeTarget (2MB).
            if (tid == 0) {
                uint32_t n_orig = sh_boundary_count;
                uint32_t n_oversized = 0;
                for (uint32_t gi = 0; gi < n_orig; gi++) {
                    uint32_t start = sh_group_starts[gi];
                    uint32_t end = (gi + 1 < n_orig) ? sh_group_starts[gi + 1] : batch;
                    // Skip if end hits padding.
                    if (sh_keys[end - 1] == UINT64_MAX) {
                        // Find actual end.
                        while (end > start && sh_keys[end - 1] == UINT64_MAX) end--;
                        if (end == start) continue;
                    }
                    uint32_t orig_last = io_sched_key_batch_index(sh_keys[end - 1]);
                    uint64_t last_bytes = io_sched_decode_nbytes(sh_nbytes_log2[orig_last]);
                    uint64_t run_bytes = io_sched_key_byte_offset(sh_keys[end - 1])
                                       - io_sched_key_byte_offset(sh_keys[start]) + last_bytes;
                    if (run_bytes > kSchedMergeTarget) n_oversized++;
                }
                acc_2mb_splits += n_oversized;
                if (n_oversized > 0) {
                    // Use upper half of sh_group_starts as scratch.
                    uint32_t* dst = sh_group_starts + n_orig;
                    uint32_t refined = 0;
                    for (uint32_t gi = 0; gi < n_orig; gi++) {
                        uint32_t start = sh_group_starts[gi];
                        uint32_t end = (gi + 1 < n_orig) ? sh_group_starts[gi + 1] : batch;
                        while (end > start && sh_keys[end - 1] == UINT64_MAX) end--;
                        if (end == start) continue;
                        uint32_t orig_last = io_sched_key_batch_index(sh_keys[end - 1]);
                        uint64_t last_bytes = io_sched_decode_nbytes(sh_nbytes_log2[orig_last]);
                        uint64_t run_bytes = io_sched_key_byte_offset(sh_keys[end - 1])
                                           - io_sched_key_byte_offset(sh_keys[start]) + last_bytes;
                        if (run_bytes <= kSchedMergeTarget) {
                            dst[refined++] = start;
                        } else {
                            uint32_t orig_start = io_sched_key_batch_index(sh_keys[start]);
                            uint32_t stride = (uint32_t)(kSchedMergeTarget /
                                io_sched_decode_nbytes(sh_nbytes_log2[orig_start]));
                            if (stride == 0) stride = 1;
                            for (uint32_t pos = start; pos < end; pos += stride)
                                dst[refined++] = pos;
                        }
                    }
                    for (uint32_t gi = 0; gi < refined; gi++) sh_group_starts[gi] = dst[gi];
                    sh_boundary_count = refined;
                }
            }
            __syncthreads();

            uint32_t n_groups_total = sh_boundary_count;

            // Step 2: Parallel PRP construction in chunks + knee-point submit.
            uint32_t gpos = 0;
            uint32_t total_merged_groups = 0;
            unsigned long long t_nvme_batch = 0;

            while (gpos < n_groups_total) {
                uint32_t chunk = n_groups_total - gpos;
                if (chunk > q->prp_pool_n_slots) chunk = q->prp_pool_n_slots;
                if (chunk > kSchedMaxBatch) chunk = kSchedMaxBatch;

                // 128 threads parallel PRP construction.
                for (uint32_t g = tid; g < chunk; g += kSchedThreads) {
                    uint32_t grp   = gpos + g;
                    uint32_t start = sh_group_starts[grp];
                    uint32_t end   = (grp + 1 < n_groups_total)
                                   ? sh_group_starts[grp + 1] : batch;
                    // Trim padding.
                    while (end > start && sh_keys[end - 1] == UINT64_MAX) end--;
                    uint32_t run_count = end - start;
                    uint32_t slot = g;

                    uint32_t orig_0 = io_sched_key_batch_index(sh_keys[start]);
                    uint64_t start_offset = io_sched_key_byte_offset(sh_keys[start]);
                    uint32_t orig_last = io_sched_key_batch_index(sh_keys[end - 1]);
                    uint64_t last_bytes   = io_sched_decode_nbytes(sh_nbytes_log2[orig_last]);
                    uint64_t run_bytes    = io_sched_key_byte_offset(sh_keys[end - 1])
                                          - start_offset + last_bytes;

                    uint64_t first_pe = sh_pc_entries[orig_0];
                    uint64_t prp1 = pc->page_ioaddrs[first_pe * ppe];
                    uint64_t prp2 = 0;
                    uint32_t total_subpages = run_count * ppe;

                    if (total_subpages <= 1) {
                        prp2 = 0;
                    } else if (total_subpages == 2) {
                        if (ppe >= 2) {
                            prp2 = pc->page_ioaddrs[first_pe * ppe + 1];
                        } else {
                            uint32_t orig_1 = io_sched_key_batch_index(sh_keys[start + 1]);
                            uint64_t pe2 = sh_pc_entries[orig_1];
                            prp2 = pc->page_ioaddrs[pe2 * ppe];
                        }
                    } else if (q->prp_pool_vaddr != nullptr) {
                        volatile uint64_t* plist =
                            (volatile uint64_t*)q->prp_pool_vaddr[slot];
                        uint32_t pi = 0;
                        for (uint32_t j = 1; j < ppe; j++)
                            plist[pi++] = pc->page_ioaddrs[first_pe * ppe + j];
                        for (uint32_t r = 1; r < run_count; r++) {
                            uint32_t orig_r = io_sched_key_batch_index(sh_keys[start + r]);
                            uint64_t pe = sh_pc_entries[orig_r];
                            for (uint32_t j = 0; j < ppe; j++)
                                plist[pi++] = pc->page_ioaddrs[pe * ppe + j];
                        }
                        prp2 = q->prp_pool_ioaddr[slot];
                    } else {
                        prp2 = pc->prps ? pc->prp2[first_pe] : 0;
                    }

                    sh_buf.merged.lbas[slot]  = start_offset >> blk_log;
                    sh_buf.merged.nblks[slot] = (uint32_t)(run_bytes >> blk_log);
                    sh_buf.merged.prp1s[slot] = prp1;
                    sh_buf.merged.prp2s[slot] = prp2;
                }
                __syncthreads();
                __threadfence_system();  // flush PRP lists for all groups

                // Knee-point throttled parallel submit.
                {
                    unsigned long long t_sub_start = 0;
                    if (tid == 0) t_sub_start = io_sched_globaltimer();

                    for (uint32_t s = 0; s < chunk; s += kSchedKneeLimit) {
                        uint32_t n = chunk - s;
                        if (n > kSchedKneeLimit) n = kSchedKneeLimit;
                        if (tid < kSchedReadSubmitThreads) {
                            QueuePair* qp = &ctrls[0]->d_qps[qp_base + tid];
                            for (uint32_t gg = tid; gg < n; gg += kSchedReadSubmitThreads) {
                                read_data_merged_no_second(qp,
                                    sh_buf.merged.lbas[s + gg],
                                    (uint64_t)sh_buf.merged.nblks[s + gg],
                                    sh_buf.merged.prp1s[s + gg],
                                    sh_buf.merged.prp2s[s + gg]);
                            }
                        }
                        __syncthreads();
                    }

                    if (tid == 0) t_nvme_batch += io_sched_globaltimer() - t_sub_start;
                    total_merged_groups += chunk;
                }
                __syncthreads();
                gpos += chunk;
            }

            if (tid == 0) {
                sh_n_groups = total_merged_groups;
                t2b = io_sched_globaltimer();
                acc_nvme += t_nvme_batch;
                acc_merge += (t2b - t2) - t_nvme_batch;
            }
        }
        __syncthreads();

        // ---- Phase 5: NOTIFY (all threads cooperate) ----
        __threadfence();
        for (uint32_t i = tid; i < batch; i += kSchedThreads) {
            uint64_t key_i = sh_keys[i];
            if (key_i == UINT64_MAX) continue;
            uint32_t orig = io_sched_key_batch_index(key_i);
            uint32_t ring_idx = (head + orig) & q->capacity_mask;
            q->done_seq[ring_idx]   = head + orig + 1;
            q->done_flags[ring_idx] = 1;
        }
        __syncthreads();

        if (tid == 0) {
            q->submitted_ios.fetch_add((uint64_t)batch, simt::memory_order_relaxed);
            q->merged_ios.fetch_add((uint64_t)sh_n_groups, simt::memory_order_relaxed);
            q->cons_head.fetch_add(batch, simt::memory_order_release);

            t3 = io_sched_globaltimer();
            acc_submit += (t3 - t2b);
            batch_count++;
        }
        __syncthreads();

#else
        // ── Original serial merge + parallel submit ──
        // Thread 0 scans sorted keys, builds merge groups with proper PRP lists.
        if (tid == 0) {
            uint32_t n_groups = 0;
            uint64_t total = 0;
            uint32_t i = 0;
            const uint32_t ppe = pc->pages_per_entry;

            while (i < batch) {
                uint64_t key_i = sh_keys[i];
                if (key_i == UINT64_MAX) break;
                uint32_t orig_i = io_sched_key_batch_index(key_i);
                uint64_t start_offset = io_sched_key_byte_offset(key_i);
                uint64_t run_bytes = io_sched_decode_nbytes(sh_nbytes_log2[orig_i]);
                uint32_t run_count = 1;

                while ((i + run_count) < batch && run_bytes < kSchedMergeTarget) {
                    uint64_t key_next = sh_keys[i + run_count];
                    if (key_next == UINT64_MAX) break;
                    uint64_t next_offset = io_sched_key_byte_offset(key_next);
                    if (next_offset != start_offset + run_bytes) break;
                    uint64_t next_bytes = io_sched_decode_nbytes(
                        sh_nbytes_log2[io_sched_key_batch_index(key_next)]);
                    if (run_bytes + next_bytes > kSchedMergeTarget) break;
                    run_bytes += next_bytes;
                    run_count++;
                }

                // Build PRP for this merged group.
                uint64_t first_pe = sh_pc_entries[orig_i];
                uint64_t merged_prp1 = pc->page_ioaddrs[first_pe * ppe];
                uint64_t merged_prp2 = 0;
                uint32_t total_subpages = run_count * ppe;

                if (total_subpages <= 1) {
                    merged_prp2 = 0;
                } else if (total_subpages == 2) {
                    if (ppe >= 2) {
                        merged_prp2 = pc->page_ioaddrs[first_pe * ppe + 1];
                    } else {
                        uint64_t pe2 = sh_pc_entries[
                            io_sched_key_batch_index(sh_keys[i + 1])];
                        merged_prp2 = pc->page_ioaddrs[pe2 * ppe];
                    }
                } else if (q->prp_pool_vaddr != nullptr) {
                    volatile uint64_t* plist =
                        (volatile uint64_t*)q->prp_pool_vaddr[n_groups];
                    uint32_t pi = 0;
                    for (uint32_t j = 1; j < ppe; j++)
                        plist[pi++] = pc->page_ioaddrs[first_pe * ppe + j];
                    for (uint32_t r = 1; r < run_count; r++) {
                        uint64_t pe = sh_pc_entries[
                            io_sched_key_batch_index(sh_keys[i + r])];
                        for (uint32_t j = 0; j < ppe; j++)
                            plist[pi++] = pc->page_ioaddrs[pe * ppe + j];
                    }
                    __threadfence_system();
                    merged_prp2 = q->prp_pool_ioaddr[n_groups];
                } else {
                    merged_prp2 = pc->prps ? pc->prp2[first_pe] : 0;
                }

                sh_buf.merged.lbas[n_groups]      = start_offset >> blk_log;
                sh_buf.merged.nblks[n_groups]     = (uint32_t)(run_bytes >> blk_log);
                sh_buf.merged.prp1s[n_groups]     = merged_prp1;
                sh_buf.merged.prp2s[n_groups]     = merged_prp2;
                sh_buf.merged.first_idx[n_groups] = (uint16_t)i;
                sh_buf.merged.count[n_groups]     = (uint16_t)run_count;

                total += run_count;
                n_groups++;
                i += run_count;
            }

            sh_n_groups = n_groups;
            sh_total_submitted = total;
            t2b = io_sched_globaltimer();
            acc_merge += (t2b - t2);
        }
        __syncthreads();

        // Parallel submit with knee-point throttle.
        {
            const uint32_t n_groups = sh_n_groups;
            for (uint32_t s = 0; s < n_groups; s += kSchedKneeLimit) {
                uint32_t n = n_groups - s;
                if (n > kSchedKneeLimit) n = kSchedKneeLimit;
                if (tid < kSchedReadSubmitThreads) {
                    QueuePair* qp = &ctrls[0]->d_qps[qp_base + tid];
                    for (uint32_t gg = tid; gg < n; gg += kSchedReadSubmitThreads) {
                        read_data_merged_no_second(qp,
                            sh_buf.merged.lbas[s + gg],
                            (uint64_t)sh_buf.merged.nblks[s + gg],
                            sh_buf.merged.prp1s[s + gg],
                            sh_buf.merged.prp2s[s + gg]);

                        __threadfence();
                        const uint16_t first = sh_buf.merged.first_idx[s + gg];
                        const uint16_t cnt   = sh_buf.merged.count[s + gg];
                        for (uint16_t r = 0; r < cnt; r++) {
                            uint32_t orig = io_sched_key_batch_index(sh_keys[first + r]);
                            uint32_t ring_idx = (head + orig) & q->capacity_mask;
                            const uint32_t done_ticket = head + orig + 1u;
                            q->done_seq[ring_idx] = done_ticket;
                            q->done_flags[ring_idx] = 1;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Thread 0: update counters and advance consumer head.
        if (tid == 0) {
            q->submitted_ios.fetch_add(sh_total_submitted, simt::memory_order_relaxed);
            q->merged_ios.fetch_add((uint64_t)sh_n_groups, simt::memory_order_relaxed);
            q->cons_head.fetch_add(batch, simt::memory_order_release);

            t3 = io_sched_globaltimer();
            acc_submit += (t3 - t2b);
            batch_count++;
        }
        __syncthreads();
#endif  // IO_SCHED_READ_PARALLEL_MERGE
    }
}

#endif // __IO_SCHEDULER_IMPL_CUH__
