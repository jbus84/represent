# PyTorch Dataloader Performance Report
Generated: 2025-07-29 14:32:08

## Executive Summary

⚠️ **PERFORMANCE TARGET MISSED**: Ultra-Fast mode exceeds 10ms target
- Ultra-Fast Mode: 28.65ms average
- Standard Mode: 16.62ms average
- Performance improvement: 0.6x faster

## Detailed Performance Metrics

### Ultra-Fast Mode
- Average processing time: 28.65ms
- Minimum processing time: 27.10ms
- Maximum processing time: 37.54ms
- Standard deviation: 2.17ms
- Memory usage: 2.3MB
- Output tensor shape: torch.Size([402, 500])

### Standard Mode
- Average processing time: 16.62ms
- Minimum processing time: 15.77ms
- Maximum processing time: 19.62ms
- Standard deviation: 0.80ms
- Memory usage: 50.3MB
- Output tensor shape: torch.Size([402, 500])

## Streaming Performance Analysis

- Data ingestion time: 1.47ms
- Processing time: 28.46ms
- Total cycle time: 29.94ms
- Theoretical throughput: 33.4 updates/second

## Multi-Core Scaling Analysis

### Scaling Summary
- Optimal configuration: Single-threaded (0 workers)
- Best batch processing time: 1.20ms
- Speedup over baseline: 1.0x

### Per-Worker Performance
- single-threaded: 1.20ms (1.0x speedup, 835.3 batches/sec)

## Background Processing Performance

### Background Processing Summary
- Average batch retrieval time: 5.649ms
- Synchronous generation time: 41.29ms
- Background processing speedup: 7.3x
- Performance assessment: ⚠️ NEEDS IMPROVEMENT - Over 5ms access

### Background Producer Statistics
- Batches produced: 20
- Background generation rate: 17.4 batches/second
- Average background generation: 29.28ms
- Queue utilization efficiency: 0.0%

## Recommendations

📊 **Batch Processing**: System best suited for research and analysis
- Recommended for backtesting and research workflows
- Further optimization needed for real-time applications

## Technical Specifications

- Ring buffer capacity: 50,000 samples
- Output tensor dimensions: 402 × 500
- Memory-mapped file support: Yes
- Thread-safe operations: Yes
- PyTorch integration: Native tensor output