# SigLIP-as-Server

A server implementation for SigLIP (Sigmoid Loss for Language Image Pre-Training) model serving. This repository contains code to run SigLIP as a service for image and text embedding.

## Features

- REST API for image and text embedding
- Performance benchmarks
- Comparison with other models (like CLIP)
- Various use cases demonstrations

## Installation

```bash
# Clone the repository
git clone https://github.com/daisysong76/SigLIP-as-server.git
cd SigLIP-as-server

# Install dependencies
pip install -r requirements.txt
```

## Usage

See the documentation in the `clip-service` directory for detailed usage instructions.

## Structure

- `clip-service/`: Main service implementation
- `scripts/`: Utility scripts
- `performance/`: Performance benchmarks
- `comparison_results/`: Comparison against other models
- `use_cases/`: Example applications

## License

MIT 

Summary of Inference Pipeline Enhancements

Below is a concise overview of four key optimizations that transform a CPU-only LLaVA embedding test into a production-quality multimodal inference pipeline:

Low-Bit Quantization (bitsandbytes + accelerate)

Shrinks model size by 4× (FP16→4-bit), cutting load times from ~30 s to <5 s.

Leverages QNNPACK/FBGEMM CPU kernels for near-GPU throughput and enables memory-constrained edge deployments.

ONNX Runtime with Static Quantization

Exports the vision-encoder + LLM head to ONNX, applies static quant (QOperator), and runs on optimized CPU backends (DNNL/OpenVINO).

Ensures sub-second model loading and consistent, cross-platform performance in development and production.

Asynchronous Micro-Batching of Embeddings

Groups 2–4 embeddings per inference call via ThreadPoolExecutor or asyncio, hiding Python overhead behind kernel execution.

Boosts end-to-end throughput and reduces per-request latency, critical for real-time multimodal applications.

CPU-Only Profiling & CI Time-Gate

Integrates PyTorch’s CPU Profiler for detailed hotspot analysis (operator, I/O, Python wiring).

Implements CI checks that fail early if the end-to-end test exceeds a defined SLA (e.g. 60 s), preventing performance regressions.
