# PM9A3 Experiments

## Common Settings

| Parameter | Value |
|---|---|
| SSD | Samsung PM9A3 7.68 TB, firmware GDC5602Q |
| GPU | NVIDIA A100-SXM4-40GB |
| GPU NVMe queues | 128 application queue pairs, depth 256 |
| DMA data allocation | 4 GiB per GPU backend |
| GPU raw-device offset | 300 GiB |
| Namespace | NSID 1, LBAF 0, 512-byte LBAs |
| Address assignment | Linear worker order, contiguous per-worker chunks |
| Condition order | Shuffled using seed 133 |
| Repetitions for new runs | 1 |
| Timeout | 1,800 seconds per command |
| GDS path | cuFile with compatibility mode disabled, ext4, preallocated file |

`configs/device.json` selects the SSD and GPU. The runner combines it with the panel profile and stores the resulting configuration with each campaign. For a second machine, create `configs/device.local.json` and pass `--device configs/device.local.json`.

The GPU benchmark assigns a separate DMA slot to each outstanding request. BaM issues requests directly from GPU workers. ShaperIO producers enqueue requests into the scheduler, which orders them by LBA, merges adjacent requests, and submits writes through a dedicated queue pair. GDS uses CPU workers to issue synchronous cuFile calls to GPU buffers.

## Figure 1(a)

Profile: [`configs/fig1ab.json`](../configs/fig1ab.json).

- BaM writers: 1, 8, 64, 512, 2048, 4096 GPU threads; 128 subsequent GPU readers.
- GDS writers: 1, 8, 64 CPU workers; 64 subsequent CPU readers.
- Write request: 4 KiB. Read request: 64 KiB.
- Read extent: 1 GiB.

For GPU writes, each batch assigns 2 MiB per writer. Up to eight batches cover at least 1 GiB. With 1 or 8 writers, the benchmark uses one 1 GiB batch. At 2048 and 4096 writers, the total written extent is 4 and 8 GiB, respectively. The read phase covers the first 1 GiB. GDS writes and reads a 1 GiB file.

Each write condition begins with NVMe format (`NSID=1`, `LBAF=0`, `SES=0`). For GDS, the helper creates ext4 with eager metadata initialization and preallocates `core.bin`. The benchmark completes writes before starting the read process. Read bandwidth is measured from completed bytes divided by elapsed time.

## Figure 1(b)

Profile: [`configs/fig1ab.json`](../configs/fig1ab.json).

Use 1 and 512 BaM writers, then sweep 1, 4, 16, 64, 128, and 256 BaM readers. Each writer/reader pair receives a separate device reset and write phase. Request sizes, extents, queues, and DMA allocation follow Figure 1(a).

## Figure 1(c)

Profile: [`configs/fig1c.json`](../configs/fig1c.json).

Format the device and sequentially write 8 GiB with one BaM writer using 64 KiB requests. Sweep 1, 16, 256, 1024, and 4096 BaM readers over that extent with 64 KiB requests. The sweep shares this prepared data within each repetition.

The driver times each read using the GPU global timer and reports P50/P99 latency. The figure plots bandwidth on the horizontal axis and latency in milliseconds on a logarithmic vertical axis; annotations give the reader count.

## Figure 3

Profile: [`configs/fig3.json`](../configs/fig3.json).

| Parameter | Value |
|---|---|
| BaM/ShaperIO writers | 1, 8, 64, 128, 256, 512, 1024, 4096 |
| GDS writers | 1, 8, 64 |
| Read concurrency | 32 |
| Write/read requests | 4 KiB / 64 KiB |
| GPU write extent and batching | 2 MiB per writer, up to 8 batches, minimum 1 GiB |
| Read extent | First 1 GiB |
| ShaperIO producer write window | 256 requests per writer |
| Scheduler ring | 1,048,576 entries |
| Maximum compiled producer window | 256 |
| Scheduler write submission | Batched, up to 32 merged requests per submission round |
| Accumulation wait initial/maximum | 8,192 ns / 8,192 ns |
| Default read window | 16 |

BaM and ShaperIO use the same input address assignment, write extent, and 4 GiB DMA allocation. After either GPU write path completes, the same direct BaM reader measures read bandwidth with 32 threads. GDS uses 32 CPU readers.

The per-writer write window exposes up to 1 MiB of adjacent 4 KiB requests. At 4096 writers, outstanding request data fits in the 4 GiB DMA allocation. Ring entries contain request descriptors; data remains in the DMA buffers.

## Files and Aggregation

Each run directory contains:

- `run.json`: resolved configuration, commands, timestamps, binary hashes, and parsed measurements.
- `*-device-reset.log`: SSD identification, formatting, and driver-binding output.
- `*-write.log`, `*-postwrite-read.log`, or `*-read.log`: benchmark output, including JSON measurements.
- `summary.csv`: median, minimum, maximum, and sample count for each metric and condition.

`tools/plot_figures.py` verifies the CSV values against the raw logs before plotting. Split runs for a series must use identical configurations and binary hashes, and their data points must be disjoint. Figures use GiB/s (2^30 bytes per second).

The reference Figure 3 GPU sweep is stored as two disjoint runs per backend. GDS uses its own run metadata and three repetitions per point. The manifest in `provenance/measurements.json` records each source directory and every copied file's SHA256.
