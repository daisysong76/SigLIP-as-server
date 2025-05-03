"""CLI tool for running CPU and GPU benchmarks and comparing results.
benchmark_runner.py:
CLI tool for running benchmarks
Supports both CPU and GPU tests
Multiple output formats (CSV, JSON, Markdown, LaTeX)
Visualization generation
Statistical analysis"""

import argparse
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
from typing import Dict, List, Optional
import json
from datetime import datetime
import torch
from cpu_scheduler import create_default_scheduler, CPUScheduler
import pytest
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Runs and compares CPU and GPU benchmarks."""
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        batch_sizes: Optional[List[int]] = None,
        num_runs: int = 3,
        export_format: str = "all"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_sizes = batch_sizes or [1, 8, 16, 32, 64, 128]
        self.num_runs = num_runs
        self.export_format = export_format
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize CPU scheduler
        self.cpu_scheduler = create_default_scheduler(
            metrics_file=str(self.output_dir / "cpu_metrics.csv")
        )
    
    async def run_cpu_benchmarks(self) -> Dict:
        """Run CPU-specific benchmarks."""
        logger.info("Running CPU benchmarks...")
        
        results = []
        for batch_size in self.batch_sizes:
            for run in range(self.num_runs):
                logger.info(f"Running CPU batch_size={batch_size}, run {run+1}/{self.num_runs}")
                
                # Run pytest with CPU configuration
                pytest_args = [
                    "tests/test_stream_cpu.py",
                    "-v",
                    f"--batch-size={batch_size}",
                    "--cpu-only"
                ]
                
                # Capture pytest output
                exit_code = pytest.main(pytest_args)
                
                if exit_code == 0:
                    # Get metrics from CPU scheduler
                    metrics = self.cpu_scheduler.export_metrics()
                    results.append({
                        "device": "cpu",
                        "batch_size": batch_size,
                        "run": run,
                        **metrics
                    })
                else:
                    logger.error(f"CPU benchmark failed for batch_size={batch_size}, run={run}")
        
        return {"cpu_results": results}
    
    async def run_gpu_benchmarks(self) -> Dict:
        """Run GPU-specific benchmarks if available."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU benchmarks")
            return {"gpu_results": []}
        
        logger.info("Running GPU benchmarks...")
        results = []
        
        for batch_size in self.batch_sizes:
            for run in range(self.num_runs):
                logger.info(f"Running GPU batch_size={batch_size}, run {run+1}/{self.num_runs}")
                
                # Run pytest with GPU configuration
                pytest_args = [
                    "tests/test_stream_gpu.py",
                    "-v",
                    f"--batch-size={batch_size}",
                    "--gpu-only"
                ]
                
                exit_code = pytest.main(pytest_args)
                
                if exit_code == 0:
                    # Get GPU metrics from file
                    gpu_metrics_file = self.output_dir / f"gpu_metrics_{batch_size}_{run}.json"
                    if gpu_metrics_file.exists():
                        with open(gpu_metrics_file) as f:
                            metrics = json.load(f)
                            results.append({
                                "device": "gpu",
                                "batch_size": batch_size,
                                "run": run,
                                **metrics
                            })
                else:
                    logger.error(f"GPU benchmark failed for batch_size={batch_size}, run={run}")
        
        return {"gpu_results": results}
    
    def _create_comparison_plots(self, cpu_results: List[Dict], gpu_results: List[Dict]):
        """Create comparison plots between CPU and GPU performance."""
        df = pd.DataFrame(cpu_results + gpu_results)
        
        # Set up the plot style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Throughput vs Batch Size
        sns.lineplot(
            data=df,
            x="batch_size",
            y="avg_throughput",
            hue="device",
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("Throughput vs Batch Size")
        axes[0, 0].set_ylabel("Samples/second")
        
        # 2. Latency vs Batch Size
        sns.lineplot(
            data=df,
            x="batch_size",
            y="avg_processing_time",
            hue="device",
            ax=axes[0, 1]
        )
        axes[0, 1].set_title("Processing Time vs Batch Size")
        axes[0, 1].set_ylabel("Seconds")
        
        # 3. Memory Usage
        sns.lineplot(
            data=df,
            x="batch_size",
            y="avg_memory_percent",
            hue="device",
            ax=axes[1, 0]
        )
        axes[1, 0].set_title("Memory Usage vs Batch Size")
        axes[1, 0].set_ylabel("Memory %")
        
        # 4. Power Consumption (if available)
        if "avg_power_consumption" in df.columns:
            sns.lineplot(
                data=df,
                x="batch_size",
                y="avg_power_consumption",
                hue="device",
                ax=axes[1, 1]
            )
            axes[1, 1].set_title("Power Consumption vs Batch Size")
            axes[1, 1].set_ylabel("Watts")
        
        plt.tight_layout()
        plot_file = self.output_dir / f"benchmark_comparison_{self.timestamp}.png"
        plt.savefig(plot_file)
        logger.info(f"Saved comparison plots to {plot_file}")
    
    def _create_markdown_table(self, cpu_results: List[Dict], gpu_results: List[Dict]) -> str:
        """Create a markdown table comparing CPU and GPU results."""
        df_cpu = pd.DataFrame(cpu_results).groupby("batch_size").mean()
        df_gpu = pd.DataFrame(gpu_results).groupby("batch_size").mean() if gpu_results else None
        
        rows = []
        headers = ["Batch Size", "CPU Throughput", "GPU Throughput", "CPU Memory %", "GPU Memory %"]
        
        for batch_size in self.batch_sizes:
            row = [batch_size]
            if batch_size in df_cpu.index:
                row.extend([
                    f"{df_cpu.loc[batch_size, 'avg_throughput']:.2f}",
                    f"{df_gpu.loc[batch_size, 'avg_throughput']:.2f}" if df_gpu is not None else "N/A",
                    f"{df_cpu.loc[batch_size, 'avg_memory_percent']:.1f}%",
                    f"{df_gpu.loc[batch_size, 'avg_memory_percent']:.1f}%" if df_gpu is not None else "N/A"
                ])
            rows.append(row)
        
        return tabulate(rows, headers=headers, tablefmt="pipe")
    
    def _create_latex_table(self, cpu_results: List[Dict], gpu_results: List[Dict]) -> str:
        """Create a LaTeX table comparing CPU and GPU results."""
        df_cpu = pd.DataFrame(cpu_results).groupby("batch_size").mean()
        df_gpu = pd.DataFrame(gpu_results).groupby("batch_size").mean() if gpu_results else None
        
        latex = "\\begin{tabular}{|c|cc|cc|}\n\\hline\n"
        latex += "Batch Size & CPU Throughput & GPU Throughput & CPU Memory & GPU Memory \\\\\n\\hline\n"
        
        for batch_size in self.batch_sizes:
            if batch_size in df_cpu.index:
                cpu_throughput = f"{df_cpu.loc[batch_size, 'avg_throughput']:.2f}"
                gpu_throughput = f"{df_gpu.loc[batch_size, 'avg_throughput']:.2f}" if df_gpu is not None else "N/A"
                cpu_memory = f"{df_cpu.loc[batch_size, 'avg_memory_percent']:.1f}\\%"
                gpu_memory = f"{df_gpu.loc[batch_size, 'avg_memory_percent']:.1f}\\%" if df_gpu is not None else "N/A"
                
                latex += f"{batch_size} & {cpu_throughput} & {gpu_throughput} & {cpu_memory} & {gpu_memory} \\\\\n"
        
        latex += "\\hline\n\\end{tabular}"
        return latex
    
    async def run_benchmarks(self):
        """Run all benchmarks and generate comparisons."""
        # Run benchmarks
        cpu_results = await self.run_cpu_benchmarks()
        gpu_results = await self.run_gpu_benchmarks()
        
        # Create comparison plots
        self._create_comparison_plots(
            cpu_results["cpu_results"],
            gpu_results["gpu_results"]
        )
        
        # Export results in requested format
        if self.export_format in ["markdown", "all"]:
            markdown_table = self._create_markdown_table(
                cpu_results["cpu_results"],
                gpu_results["gpu_results"]
            )
            with open(self.output_dir / f"benchmark_results_{self.timestamp}.md", "w") as f:
                f.write("# Benchmark Results\n\n")
                f.write(markdown_table)
        
        if self.export_format in ["latex", "all"]:
            latex_table = self._create_latex_table(
                cpu_results["cpu_results"],
                gpu_results["gpu_results"]
            )
            with open(self.output_dir / f"benchmark_results_{self.timestamp}.tex", "w") as f:
                f.write(latex_table)
        
        # Export raw data
        all_results = {
            "timestamp": self.timestamp,
            "cpu_results": cpu_results,
            "gpu_results": gpu_results
        }
        with open(self.output_dir / f"benchmark_raw_{self.timestamp}.json", "w") as f:
            json.dump(all_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run CPU and GPU benchmarks")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to store benchmark results"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for each configuration"
    )
    parser.add_argument(
        "--export-format",
        choices=["markdown", "latex", "all"],
        default="all",
        help="Format for exporting results"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only CPU benchmarks"
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Run only GPU benchmarks"
    )
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        export_format=args.export_format
    )
    
    asyncio.run(runner.run_benchmarks())

if __name__ == "__main__":
    main() 