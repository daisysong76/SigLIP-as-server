# CLIP Service Benchmarking Guide

## Overview
This document describes the benchmarking infrastructure for the CLIP service, including both CPU and GPU optimizations.

## CPU Optimizations

### Dynamic Batch Size Scheduling
The `CPUScheduler` in `cpu_scheduler.py` provides:
- Automatic batch size adjustment based on CPU/memory load
- System metrics monitoring (CPU, memory, temperature, power)
- Integration with torch.compile() and other optimizations

### CPU-Specific Features
- TorchInductor optimization via `torch.compile()`
- NumExpr integration for faster numerical operations
- OpenBLAS threading optimization
- Memory-aware batch sizing
- Core affinity management

## Running Benchmarks

### Quick Start
```bash
# Run all benchmarks
python tests/benchmark_runner.py

# CPU-only benchmarks
python tests/benchmark_runner.py --cpu-only

# Custom batch sizes
python tests/benchmark_runner.py --batch-sizes 16 32 64 128
```

### Configuration Options
```bash
--output-dir DIR      # Output directory for results
--batch-sizes N [N..] # Batch sizes to test
--num-runs N          # Number of runs per configuration
--export-format FMT   # Output format (markdown/latex/all)
--cpu-only           # Run only CPU tests
--gpu-only           # Run only GPU tests
```

## Performance Metrics

### CPU Metrics
- Processing time per batch
- Throughput (samples/second)
- Memory usage
- CPU utilization
- Power consumption (where available)
- Temperature monitoring

### Comparison Metrics
| Metric | CPU | GPU |
|--------|-----|-----|
| Batch Processing | Dynamic scheduling | Fixed batching |
| Memory Management | Adaptive | Pre-allocated |
| Optimization | TorchInductor + NumExpr | CUDA optimization |
| Monitoring | Full system metrics | GPU-specific metrics |

## Output Formats

### Metrics Export
- CSV files with detailed metrics
- JSON raw data export
- Markdown tables
- LaTeX tables
- Matplotlib/Seaborn visualizations

### Visualization Types
1. Throughput vs Batch Size
2. Processing Time vs Batch Size
3. Memory Usage Patterns
4. Power Consumption Comparison

## Best Practices

### CPU Optimization
1. Use dynamic batch sizing
2. Enable TorchInductor when possible
3. Configure NumExpr threads
4. Monitor system metrics
5. Adjust based on memory availability

### Benchmarking Tips
1. Run multiple iterations
2. Warm up the system
3. Clear cache between runs
4. Monitor thermal throttling
5. Record environmental conditions

## Example Results

### Sample Performance Table
| Batch Size | CPU Throughput | GPU Throughput | CPU Memory | GPU Memory |
|------------|---------------|----------------|------------|------------|
| 1          | X samples/s   | Y samples/s    | A%        | B%        |
| 32         | X samples/s   | Y samples/s    | A%        | B%        |
| 128        | X samples/s   | Y samples/s    | A%        | B%        |

### Optimization Results
- TorchInductor: Up to X% improvement
- Dynamic Batching: Y% better resource utilization
- NumExpr: Z% faster numerical operations

## Troubleshooting

### Common Issues
1. Memory pressure
2. Thermal throttling
3. System interference
4. Inconsistent results

### Solutions
1. Adjust batch sizes
2. Monitor system load
3. Use warm-up runs
4. Clear system cache

## Future Improvements

### Planned Features
1. More sophisticated scheduling
2. Additional optimization backends
3. Extended metrics collection
4. Automated optimization selection

### Research Areas
1. Advanced batching strategies
2. Hardware-specific optimizations
3. Power/performance tradeoffs
4. Thermal management 