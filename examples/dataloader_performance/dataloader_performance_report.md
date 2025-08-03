# PyTorch Dataloader Performance Report
Generated: 2025-08-03 17:03:47

## Executive Summary

⚠️ **PERFORMANCE TARGET MISSED**: Ultra-Fast mode exceeds 10ms target
- Ultra-Fast Mode: 27.93ms average
- Standard Mode: 16.59ms average
- Performance improvement: 0.6x faster

## Detailed Performance Metrics

### Ultra-Fast Mode
- Average processing time: 27.93ms
- Minimum processing time: 27.22ms
- Maximum processing time: 28.73ms
- Standard deviation: 0.46ms
- Memory usage: 27.9MB
- Output tensor shape: torch.Size([402, 500])

### Standard Mode
- Average processing time: 16.59ms
- Minimum processing time: 15.94ms
- Maximum processing time: 17.23ms
- Standard deviation: 0.38ms
- Memory usage: 42.3MB
- Output tensor shape: torch.Size([402, 500])

## Streaming Performance Analysis

- Data ingestion time: 1.50ms
- Processing time: 28.40ms
- Total cycle time: 29.89ms
- Theoretical throughput: 33.5 updates/second

## Multi-Core Scaling Analysis

### Scaling Summary
- Optimal configuration: Single-threaded (0 workers)
- Best batch processing time: 0.22ms
- Speedup over baseline: 1.0x

### Per-Worker Performance
- single-threaded: 0.22ms (1.0x speedup, 4490.7 batches/sec)

## Background Processing Performance

### Background Processing Summary
- Average batch retrieval time: 30.752ms
- Synchronous generation time: 29.04ms
- Background processing speedup: 0.9x
- Performance assessment: ⚠️ NEEDS IMPROVEMENT - Over 5ms access

### Background Producer Statistics
- Test batches processed: 20
- Average processing time: 30.752ms
- Performance assessment: Needs improvement

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