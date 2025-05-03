import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import re
import os
import sys
from datetime import datetime

def run_all_tests():
    """Run all performance tests and collect metrics."""
    print("Running comparative performance analysis for CLIP model optimizations...")
    
    # Create results directory
    os.makedirs("performance_results", exist_ok=True)
    
    # Dictionary to store all results
    results = {}
    
    # Test 1: Run baseline test (test_stream_quick.py)
    print("\n1. Running baseline test (test_stream_quick.py)...")
    cmd = "python clip-service/tests/test_stream_quick.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    results["baseline"] = extract_metrics_from_output(result.stdout, "baseline")
    
    # Test 2: Run dynamic batch size test (debug_dataset.py)
    print("\n2. Running dynamic batch size test (debug_dataset.py)...")
    cmd = "python clip-service/tests/debug_dataset.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    results["dynamic"] = extract_metrics_from_output(result.stdout, "dynamic")
    
    return results

def extract_metrics_from_output(output, test_type):
    """Extract performance metrics from test output."""
    metrics = {
        "test_type": test_type,
        "processing_times": [],
        "processing_rates": [],
        "batch_sizes": [],
        "memory_usages": [],
        "gpu_memories": [],
    }
    
    # Extract common metrics
    batch_times = re.findall(r"[Bb]atch (?:processing )?time: (\d+\.\d+)s", output)
    metrics["processing_times"] = [float(t) for t in batch_times]
    
    # Extract processing rates
    if test_type == "baseline":
        rates = re.findall(r"Processing rate: (\d+\.\d+) samples/sec", output)
        metrics["processing_rates"] = [float(r) for r in rates]
    else:
        # For dynamic test, calculate from per-sample times if available
        per_sample_times = re.findall(r"\((\d+\.\d+)s per sample\)", output)
        if per_sample_times:
            metrics["processing_rates"] = [1.0/float(t) for t in per_sample_times]
    
    # Extract batch sizes
    if test_type == "baseline":
        # For baseline, fixed batch size, check the output
        batch_size = 8  # Default from test_stream_quick.py
        # Try to extract from config
        batch_config = re.search(r"batch_size=(\d+)", output)
        if batch_config:
            batch_size = int(batch_config.group(1))
        metrics["batch_sizes"] = [batch_size] * len(metrics["processing_times"])
    else:
        # For dynamic test
        batch_sizes = re.findall(r"Processing batch of size (\d+)", output)
        metrics["batch_sizes"] = [int(s) for s in batch_sizes]
    
    # Extract memory usage
    memory_usages = re.findall(r"Memory usage: (\d+\.\d+)", output)
    metrics["memory_usages"] = [float(m) for m in memory_usages]
    
    # Extract GPU memory if available
    gpu_memories = re.findall(r"GPU memory used for inference: (\d+\.\d+) MB", output)
    metrics["gpu_memories"] = [float(m) for m in gpu_memories]
    
    # Extract batch dimension fixes
    metrics["dimension_fixes"] = len(re.findall(r"Adding missing batch dimension", output))
    
    return metrics

def visualize_comparison(results):
    """Create visualizations comparing the performance metrics."""
    # Set up the plot style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("CLIP Model Performance Optimization Comparison", fontsize=20)
    
    # Combine data from both tests
    baseline = results["baseline"]
    dynamic = results["dynamic"]
    
    # 1. Processing Rate Comparison
    ax1 = fig.add_subplot(231)
    
    # Create dataframes for easier plotting
    baseline_df = pd.DataFrame({
        'Batch Number': range(1, len(baseline["processing_rates"])+1),
        'Processing Rate': baseline["processing_rates"],
        'Type': 'Baseline (Fixed Batch Size)'
    })
    
    dynamic_df = pd.DataFrame({
        'Batch Number': range(1, len(dynamic["processing_rates"])+1),
        'Processing Rate': dynamic["processing_rates"],
        'Type': 'Dynamic Batch Size'
    })
    
    # Combine dataframes
    df = pd.concat([baseline_df, dynamic_df])
    
    # Plot
    sns.lineplot(x='Batch Number', y='Processing Rate', hue='Type', 
                 data=df, markers=True, dashes=False, ax=ax1)
    ax1.set_title("Processing Rate Comparison")
    ax1.set_ylabel("Samples/Second")
    ax1.grid(True)
    
    # 2. Batch Size Comparison
    ax2 = fig.add_subplot(232)
    
    baseline_batch_df = pd.DataFrame({
        'Batch Number': range(1, len(baseline["batch_sizes"])+1),
        'Batch Size': baseline["batch_sizes"],
        'Type': 'Baseline (Fixed)'
    })
    
    dynamic_batch_df = pd.DataFrame({
        'Batch Number': range(1, len(dynamic["batch_sizes"])+1),
        'Batch Size': dynamic["batch_sizes"],
        'Type': 'Dynamic'
    })
    
    batch_df = pd.concat([baseline_batch_df, dynamic_batch_df])
    
    sns.lineplot(x='Batch Number', y='Batch Size', hue='Type',
                 data=batch_df, markers=True, dashes=False, ax=ax2)
    ax2.set_title("Batch Size Comparison")
    ax2.set_ylabel("Batch Size")
    ax2.grid(True)
    
    # 3. Processing Time Comparison
    ax3 = fig.add_subplot(233)
    
    baseline_time_df = pd.DataFrame({
        'Batch Number': range(1, len(baseline["processing_times"])+1),
        'Processing Time': baseline["processing_times"],
        'Type': 'Baseline'
    })
    
    dynamic_time_df = pd.DataFrame({
        'Batch Number': range(1, len(dynamic["processing_times"])+1),
        'Processing Time': dynamic["processing_times"],
        'Type': 'Dynamic'
    })
    
    time_df = pd.concat([baseline_time_df, dynamic_time_df])
    
    sns.lineplot(x='Batch Number', y='Processing Time', hue='Type',
                 data=time_df, markers=True, dashes=False, ax=ax3)
    ax3.set_title("Processing Time Comparison")
    ax3.set_ylabel("Time (seconds)")
    ax3.grid(True)
    
    # 4. Efficiency Metrics
    ax4 = fig.add_subplot(234)
    
    # Calculate efficiency metrics
    baseline_avg_rate = np.mean(baseline["processing_rates"]) if baseline["processing_rates"] else 0
    dynamic_avg_rate = np.mean(dynamic["processing_rates"]) if dynamic["processing_rates"] else 0
    
    baseline_max_rate = np.max(baseline["processing_rates"]) if baseline["processing_rates"] else 0
    dynamic_max_rate = np.max(dynamic["processing_rates"]) if dynamic["processing_rates"] else 0
    
    metrics_df = pd.DataFrame({
        'Metric': ['Average Rate', 'Maximum Rate'],
        'Baseline': [baseline_avg_rate, baseline_max_rate],
        'Dynamic': [dynamic_avg_rate, dynamic_max_rate]
    })
    
    metrics_df = pd.melt(metrics_df, id_vars=['Metric'], 
                         var_name='Test Type', value_name='Samples/Second')
    
    sns.barplot(x='Metric', y='Samples/Second', hue='Test Type', data=metrics_df, ax=ax4)
    ax4.set_title("Efficiency Comparison")
    ax4.grid(True)
    
    # Add percentage improvement labels
    for i, metric in enumerate(['Average Rate', 'Maximum Rate']):
        baseline_val = baseline_avg_rate if i == 0 else baseline_max_rate
        dynamic_val = dynamic_avg_rate if i == 0 else dynamic_max_rate
        
        if baseline_val > 0:
            improvement = ((dynamic_val - baseline_val) / baseline_val) * 100
            color = 'green' if improvement > 0 else 'red'
            ax4.text(i, dynamic_val + 0.5, f"{improvement:.1f}%", 
                     color=color, ha='center', weight='bold')
    
    # 5. Memory Efficiency
    ax5 = fig.add_subplot(235)
    
    # Calculate memory per sample (if available)
    if baseline["batch_sizes"] and baseline["memory_usages"]:
        baseline_mem_per_sample = [m/bs for m, bs in zip(
            baseline["memory_usages"], baseline["batch_sizes"][:len(baseline["memory_usages"])])]
        baseline_avg_mem = np.mean(baseline_mem_per_sample)
    else:
        baseline_avg_mem = 0
        
    if dynamic["batch_sizes"] and dynamic["memory_usages"]:
        dynamic_mem_per_sample = [m/bs for m, bs in zip(
            dynamic["memory_usages"], dynamic["batch_sizes"][:len(dynamic["memory_usages"])])]
        dynamic_avg_mem = np.mean(dynamic_mem_per_sample)
    else:
        dynamic_avg_mem = 0
    
    memory_df = pd.DataFrame({
        'Test Type': ['Baseline', 'Dynamic'],
        'Memory per Sample': [baseline_avg_mem, dynamic_avg_mem]
    })
    
    sns.barplot(x='Test Type', y='Memory per Sample', data=memory_df, ax=ax5)
    ax5.set_title("Memory Efficiency Comparison")
    ax5.set_ylabel("Memory Usage per Sample")
    ax5.grid(True)
    
    # Add percentage improvement label
    if baseline_avg_mem > 0:
        improvement = ((baseline_avg_mem - dynamic_avg_mem) / baseline_avg_mem) * 100
        color = 'green' if improvement > 0 else 'red'
        ax5.text(1, dynamic_avg_mem + 0.01, f"{improvement:.1f}%", 
                 color=color, ha='center', weight='bold')
    
    # 6. Processing Rate to Batch Size Relationship
    ax6 = fig.add_subplot(236)
    
    # Dynamic test data
    if dynamic["batch_sizes"] and dynamic["processing_rates"]:
        scatter_df = pd.DataFrame({
            'Batch Size': dynamic["batch_sizes"][:len(dynamic["processing_rates"])],
            'Processing Rate': dynamic["processing_rates"],
            'Batch Number': range(1, len(dynamic["processing_rates"])+1)
        })
        
        scatter = ax6.scatter(scatter_df['Batch Size'], scatter_df['Processing Rate'], 
                     c=scatter_df['Batch Number'], cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax6, label='Batch Number')
        
        # Add trend line
        if len(scatter_df) > 1:
            z = np.polyfit(scatter_df['Batch Size'], scatter_df['Processing Rate'], 1)
            p = np.poly1d(z)
            ax6.plot(scatter_df['Batch Size'], p(scatter_df['Batch Size']), "r--", 
                     alpha=0.8, label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
            ax6.legend()
        
        ax6.set_title("Processing Rate vs Batch Size")
        ax6.set_xlabel("Batch Size")
        ax6.set_ylabel("Processing Rate (samples/second)")
    else:
        ax6.text(0.5, 0.5, "Insufficient data for this analysis", 
                 ha='center', va='center', transform=ax6.transAxes)
    
    # Add summary stats
    summary = (
        f"Summary Statistics:\n"
        f"Baseline Test:\n"
        f"  - Fixed batch size: {baseline['batch_sizes'][0] if baseline['batch_sizes'] else 'N/A'}\n"
        f"  - Batch dimension fixes: {baseline['dimension_fixes']}\n"
        f"  - Avg processing rate: {baseline_avg_rate:.2f} samples/s\n\n"
        f"Dynamic Batch Test:\n"
        f"  - Batch size range: {min(dynamic['batch_sizes']) if dynamic['batch_sizes'] else 'N/A'} to "
        f"{max(dynamic['batch_sizes']) if dynamic['batch_sizes'] else 'N/A'}\n"
        f"  - Batch dimension fixes: {dynamic['dimension_fixes']}\n"
        f"  - Avg processing rate: {dynamic_avg_rate:.2f} samples/s"
    )
    
    # Add text box with summary
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    fig.text(0.5, 0.01, summary, ha='center', va='bottom', 
             fontsize=10, bbox=props)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', fontsize=8)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig("performance_results/optimization_comparison.png", dpi=300)
    
    return "performance_results/optimization_comparison.png"

def generate_comparison_report(results):
    """Generate a detailed report comparing the performance of both approaches."""
    baseline = results["baseline"]
    dynamic = results["dynamic"]
    
    # Calculate summary statistics
    baseline_avg_rate = np.mean(baseline["processing_rates"]) if baseline["processing_rates"] else 0
    dynamic_avg_rate = np.mean(dynamic["processing_rates"]) if dynamic["processing_rates"] else 0
    
    baseline_max_rate = np.max(baseline["processing_rates"]) if baseline["processing_rates"] else 0
    dynamic_max_rate = np.max(dynamic["processing_rates"]) if dynamic["processing_rates"] else 0
    
    if baseline_avg_rate > 0:
        rate_improvement = ((dynamic_avg_rate - baseline_avg_rate) / baseline_avg_rate) * 100
    else:
        rate_improvement = 0
    
    # Memory efficiency
    if baseline["batch_sizes"] and baseline["memory_usages"]:
        baseline_mem_per_sample = [m/bs for m, bs in zip(
            baseline["memory_usages"], baseline["batch_sizes"][:len(baseline["memory_usages"])])]
        baseline_avg_mem = np.mean(baseline_mem_per_sample)
    else:
        baseline_avg_mem = 0
        
    if dynamic["batch_sizes"] and dynamic["memory_usages"]:
        dynamic_mem_per_sample = [m/bs for m, bs in zip(
            dynamic["memory_usages"], dynamic["batch_sizes"][:len(dynamic["memory_usages"])])]
        dynamic_avg_mem = np.mean(dynamic_mem_per_sample)
    else:
        dynamic_avg_mem = 0
    
    if baseline_avg_mem > 0:
        memory_improvement = ((baseline_avg_mem - dynamic_avg_mem) / baseline_avg_mem) * 100
    else:
        memory_improvement = 0
    
    # Generate report
    report = f"""
    CLIP Model Optimization Comparison Report
    =========================================
    
    Executive Summary:
    -----------------
    This report compares the performance of two approaches to CLIP model batch processing:
    1. Baseline: Fixed batch size with batch dimension fix
    2. Dynamic: Adaptive batch sizing based on memory usage and processing time
    
    Overall Performance Comparison:
    ------------------------------
    - Average Processing Rate:
      * Baseline: {baseline_avg_rate:.2f} samples/second
      * Dynamic:  {dynamic_avg_rate:.2f} samples/second
      * Improvement: {rate_improvement:.1f}%
    
    - Maximum Processing Rate:
      * Baseline: {baseline_max_rate:.2f} samples/second
      * Dynamic:  {dynamic_max_rate:.2f} samples/second
      * Improvement: {((dynamic_max_rate - baseline_max_rate) / baseline_max_rate * 100) if baseline_max_rate > 0 else 0:.1f}%
    
    Batch Size Analysis:
    -------------------
    - Baseline: Fixed at {baseline['batch_sizes'][0] if baseline['batch_sizes'] else 'N/A'}
    - Dynamic: 
      * Average: {np.mean(dynamic['batch_sizes']) if dynamic['batch_sizes'] else 'N/A':.1f}
      * Range: {min(dynamic['batch_sizes']) if dynamic['batch_sizes'] else 'N/A'} to {max(dynamic['batch_sizes']) if dynamic['batch_sizes'] else 'N/A'}
    
    Memory Efficiency:
    -----------------
    - Memory Usage per Sample:
      * Baseline: {baseline_avg_mem:.4f}
      * Dynamic:  {dynamic_avg_mem:.4f}
      * Improvement: {memory_improvement:.1f}%
    
    Processing Time:
    ---------------
    - Average Batch Processing Time:
      * Baseline: {np.mean(baseline['processing_times']) if baseline['processing_times'] else 0:.4f}s
      * Dynamic:  {np.mean(dynamic['processing_times']) if dynamic['processing_times'] else 0:.4f}s
    
    Batch Dimension Fix:
    -------------------
    - Baseline: {baseline['dimension_fixes']} fixes applied
    - Dynamic:  {dynamic['dimension_fixes']} fixes applied
    
    Key Findings:
    ------------
    1. {("Dynamic batch sizing improves processing rate by approximately " + f"{rate_improvement:.1f}% over fixed batch size approach") if rate_improvement > 0 else "Fixed batch size performed better in this test configuration"}
    
    2. {("Memory efficiency is improved by " + f"{memory_improvement:.1f}% with dynamic batch sizing") if memory_improvement > 0 else "No significant memory efficiency improvement was observed"}
    
    3. Batch dimension fix is crucial for both approaches to ensure compatibility with the CLIP model's expected input format
    
    Recommendations:
    --------------
    1. For production environments: {("Implement dynamic batch sizing for optimal performance") if rate_improvement > 0 else "Use fixed batch size with appropriate tuning"}
    
    2. For all deployments: Ensure batch dimension handling is properly implemented
    
    3. For variable workloads: The dynamic approach offers better adaptability to changing resource availability
    
    4. For stable, predictable workloads: Consider a fixed batch size close to the optimal value discovered through dynamic testing
    """
    
    # Write report to file
    with open("performance_results/optimization_comparison_report.txt", "w") as f:
        f.write(report)
    
    return report

if __name__ == "__main__":
    # Create results directory
    os.makedirs("performance_results", exist_ok=True)
    
    # Run all tests and collect metrics
    results = run_all_tests()
    
    # Generate visualization
    visualization_file = visualize_comparison(results)
    
    # Generate detailed report
    report = generate_comparison_report(results)
    
    print(f"\nVisualization saved to: {visualization_file}")
    print("Comparison report saved to: performance_results/optimization_comparison_report.txt")
    
    # Print a condensed summary
    baseline_avg = np.mean(results["baseline"]["processing_rates"]) if results["baseline"]["processing_rates"] else 0
    dynamic_avg = np.mean(results["dynamic"]["processing_rates"]) if results["dynamic"]["processing_rates"] else 0
    
    if baseline_avg > 0:
        improvement = ((dynamic_avg - baseline_avg) / baseline_avg) * 100
        print(f"\nPerformance Summary:")
        print(f"- Baseline average processing rate: {baseline_avg:.2f} samples/second")
        print(f"- Dynamic average processing rate:  {dynamic_avg:.2f} samples/second")
        print(f"- Overall improvement: {improvement:.1f}%")
    else:
        print("\nInsufficient data for comparison summary.")
        
    print("\nSee the full report for detailed analysis and recommendations.") 