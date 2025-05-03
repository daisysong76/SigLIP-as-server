import matplotlib.pyplot as plt
import numpy as np
import re
import os
import subprocess
from datetime import datetime
import pandas as pd
import seaborn as sns

def run_dynamic_batch_test():
    """Run the debug_dataset.py test with dynamic batch sizing and extract metrics."""
    cmd = "python clip-service/tests/debug_dataset.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    
    # Extract batch sizes used during the test
    batch_sizes = re.findall(r"Processing batch of size (\d+)", output)
    batch_sizes = [int(s) for s in batch_sizes]
    
    # Extract any batch size adjustments
    adjustments = re.findall(r"Adjusting batch size from (\d+) to (\d+)", output)
    adjustments = [(int(a), int(b)) for a, b in adjustments]
    
    # Extract memory usage
    memory_usages = re.findall(r"Memory usage: (\d+\.\d+)", output)
    memory_usages = [float(m) for m in memory_usages]
    
    # Extract processing times
    proc_times = re.findall(r"Batch processing time: (\d+\.\d+)s", output)
    proc_times = [float(t) for t in proc_times]
    
    # Extract per-sample times if available
    sample_times = re.findall(r"\((\d+\.\d+)s per sample\)", output)
    sample_times = [float(t) for t in sample_times]
    
    # Extract GPU memory usage if available
    gpu_memories = re.findall(r"GPU memory used for inference: (\d+\.\d+) MB", output)
    gpu_memories = [float(m) for m in gpu_memories]
    
    return {
        "batch_sizes": batch_sizes,
        "adjustments": adjustments,
        "memory_usages": memory_usages,
        "processing_times": proc_times,
        "sample_times": sample_times,
        "gpu_memories": gpu_memories,
        "raw_output": output
    }

def visualize_dynamic_batch_sizing(metrics):
    """Create visualizations for dynamic batch sizing performance."""
    # Create figure with multiple subplots
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Dynamic Batch Sizing Performance", fontsize=16)
    
    # Convert adjustments to a time series of batch sizes
    batch_size_history = []
    memory_history = []
    time_history = []
    
    # Initial values
    current_batch_size = metrics["batch_sizes"][0] if metrics["batch_sizes"] else 8
    
    # Create adjustment history
    batch_idx = 0
    for i in range(len(metrics["processing_times"])):
        # Add current values
        batch_size_history.append(current_batch_size)
        
        # Add memory and time metrics if available
        if i < len(metrics["memory_usages"]):
            memory_history.append(metrics["memory_usages"][i])
        else:
            memory_history.append(None)
            
        if i < len(metrics["processing_times"]):
            time_history.append(metrics["processing_times"][i])
        else:
            time_history.append(None)
        
        # Check for adjustment for next iteration
        if batch_idx < len(metrics["adjustments"]):
            old, new = metrics["adjustments"][batch_idx]
            if old == current_batch_size:
                current_batch_size = new
                batch_idx += 1
    
    # Create a DataFrame for better plotting
    data = pd.DataFrame({
        'Batch Number': range(1, len(batch_size_history) + 1),
        'Batch Size': batch_size_history,
        'Memory Usage': memory_history,
        'Processing Time': time_history
    })
    
    # Plot 1: Batch Size Evolution
    ax1 = fig.add_subplot(221)
    sns.lineplot(x='Batch Number', y='Batch Size', data=data, 
                 marker='o', linewidth=2, ax=ax1)
    ax1.set_title("Dynamic Batch Size Evolution")
    ax1.set_ylabel("Batch Size")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight adjustments
    for idx, (old, new) in enumerate(metrics["adjustments"]):
        x_pos = idx + 1  # Approximate position
        ax1.annotate(f"{old}→{new}", 
                     xy=(x_pos, old),
                     xytext=(x_pos, old + (5 if new > old else -5)),
                     arrowprops=dict(arrowstyle="->", color='red'),
                     color='red')
    
    # Plot 2: Memory Usage vs Batch Size
    ax2 = fig.add_subplot(222)
    if memory_history and not all(m is None for m in memory_history):
        scatter = ax2.scatter(data['Batch Size'], data['Memory Usage'], 
                     c=data['Batch Number'], cmap='viridis', 
                     alpha=0.7, s=80)
        plt.colorbar(scatter, ax=ax2, label='Batch Number')
        
        # Add trend line
        valid_data = data.dropna(subset=['Memory Usage'])
        if len(valid_data) > 1:
            x = valid_data['Batch Size']
            y = valid_data['Memory Usage']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "r--", alpha=0.8, 
                     label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
            ax2.legend()
        
        ax2.set_title("Memory Usage vs Batch Size")
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Memory Usage (0-1 scale)")
    else:
        ax2.text(0.5, 0.5, "No memory usage data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    
    # Plot 3: Processing Time vs Batch Size
    ax3 = fig.add_subplot(223)
    if time_history and not all(t is None for t in time_history):
        scatter = ax3.scatter(data['Batch Size'], data['Processing Time'], 
                     c=data['Batch Number'], cmap='plasma', 
                     alpha=0.7, s=80)
        plt.colorbar(scatter, ax=ax3, label='Batch Number')
        
        # Add trend line
        valid_data = data.dropna(subset=['Processing Time'])
        if len(valid_data) > 1:
            x = valid_data['Batch Size']
            y = valid_data['Processing Time']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax3.plot(x, p(x), "r--", alpha=0.8, 
                     label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
            ax3.legend()
        
        ax3.set_title("Processing Time vs Batch Size")
        ax3.set_xlabel("Batch Size")
        ax3.set_ylabel("Processing Time (seconds)")
    else:
        ax3.text(0.5, 0.5, "No processing time data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)
    
    # Plot 4: Efficiency (samples per second) by Batch
    ax4 = fig.add_subplot(224)
    if metrics["sample_times"]:
        # Calculate samples per second
        samples_per_second = [1.0/t for t in metrics["sample_times"]]
        efficiency_data = pd.DataFrame({
            'Batch Number': range(1, len(samples_per_second) + 1),
            'Efficiency': samples_per_second,
            'Batch Size': batch_size_history[:len(samples_per_second)]
        })
        
        bars = ax4.bar(efficiency_data['Batch Number'], 
                       efficiency_data['Efficiency'],
                       color=plt.cm.viridis(efficiency_data['Batch Size']/max(efficiency_data['Batch Size'])))
        
        # Add batch size labels on top of bars
        for i, (idx, row) in enumerate(efficiency_data.iterrows()):
            ax4.text(i+1, row['Efficiency'] + 0.1, 
                     f"BS: {row['Batch Size']}", 
                     ha='center', rotation=45, fontsize=8)
        
        ax4.set_title("Processing Efficiency by Batch")
        ax4.set_xlabel("Batch Number")
        ax4.set_ylabel("Samples per Second")
    else:
        ax4.text(0.5, 0.5, "No per-sample timing data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax4.transAxes)
    
    # Add summary info
    avg_batch_size = np.mean(batch_size_history) if batch_size_history else 0
    num_adjustments = len(metrics["adjustments"])
    
    summary_text = (
        f"Summary:\n"
        f"Average Batch Size: {avg_batch_size:.1f}\n"
        f"Number of Adjustments: {num_adjustments}\n"
        f"Initial Batch Size: {batch_size_history[0] if batch_size_history else 'N/A'}\n"
        f"Final Batch Size: {batch_size_history[-1] if batch_size_history else 'N/A'}"
    )
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.98, 0.02, f"Generated: {timestamp}", ha='right', fontsize=8)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("dynamic_batch_visualization.png", dpi=300)
    plt.show()
    
    return "dynamic_batch_visualization.png"

def generate_dynamic_batch_report(metrics):
    """Generate a detailed report on dynamic batch sizing performance."""
    # Calculate helpful statistics
    avg_batch_size = np.mean(metrics["batch_sizes"]) if metrics["batch_sizes"] else 0
    min_batch_size = min(metrics["batch_sizes"]) if metrics["batch_sizes"] else 0
    max_batch_size = max(metrics["batch_sizes"]) if metrics["batch_sizes"] else 0
    num_adjustments = len(metrics["adjustments"])
    
    # Calculate efficiency improvements
    if metrics["processing_times"] and len(metrics["processing_times"]) > 1:
        first_time = metrics["processing_times"][0]
        last_time = metrics["processing_times"][-1]
        time_improvement = (first_time - last_time) / first_time * 100 if first_time > 0 else 0
    else:
        time_improvement = 0
    
    # Calculate memory efficiency
    if metrics["memory_usages"] and len(metrics["memory_usages"]) > 1:
        memory_per_sample = [m/(bs if bs > 0 else 1) for m, bs in 
                            zip(metrics["memory_usages"], metrics["batch_sizes"][:len(metrics["memory_usages"])])]
        avg_memory_per_sample = np.mean(memory_per_sample)
    else:
        avg_memory_per_sample = 0
    
    report = f"""
    Dynamic Batch Sizing Performance Report
    ======================================
    
    Batch Size Metrics:
    - Average batch size: {avg_batch_size:.2f}
    - Minimum batch size used: {min_batch_size}
    - Maximum batch size used: {max_batch_size}
    - Number of batch size adjustments: {num_adjustments}
    
    Performance Impact:
    - Processing time improvement: {time_improvement:.1f}%
    - Average memory usage per sample: {avg_memory_per_sample:.4f}
    
    Batch Size Adjustment History:
    """
    
    # Add adjustment history
    for i, (old, new) in enumerate(metrics["adjustments"]):
        change = (new - old) / old * 100 if old > 0 else 0
        direction = "Increased" if new > old else "Decreased"
        report += f"  {i+1}. {direction} from {old} to {new} ({change:.1f}%)\n"
    
    # Add efficiency analysis
    if metrics["sample_times"]:
        avg_sample_time = np.mean(metrics["sample_times"])
        best_sample_time = min(metrics["sample_times"])
        
        report += f"""
    Efficiency Metrics:
    - Average time per sample: {avg_sample_time:.4f}s
    - Best time per sample: {best_sample_time:.4f}s
    - Samples per second: {1/avg_sample_time:.2f}
    """
    
    # Add GPU memory analysis if available
    if metrics["gpu_memories"]:
        avg_gpu_mem = np.mean(metrics["gpu_memories"])
        max_gpu_mem = max(metrics["gpu_memories"])
        
        # Calculate memory per sample
        gpu_mem_per_sample = [m/(bs if bs > 0 else 1) for m, bs in 
                             zip(metrics["gpu_memories"], metrics["batch_sizes"][:len(metrics["gpu_memories"])])]
        avg_gpu_mem_per_sample = np.mean(gpu_mem_per_sample)
        
        report += f"""
    GPU Memory Usage:
    - Average GPU memory: {avg_gpu_mem:.2f} MB
    - Peak GPU memory: {max_gpu_mem:.2f} MB
    - Average GPU memory per sample: {avg_gpu_mem_per_sample:.2f} MB
    """
    
    # Add recommendations based on observations
    report += """
    Recommendations:
    - For consistent workloads: Use a fixed batch size close to the average discovered
    - For variable workloads: Continue using dynamic batch sizing
    - Tune target_memory_usage parameter based on available hardware
    """
    
    # Write report to file
    with open("dynamic_batch_report.txt", "w") as f:
        f.write(report)
    
    return report

if __name__ == "__main__":
    print("Running dynamic batch sizing performance analysis...")
    
    # Run test and extract metrics
    metrics = run_dynamic_batch_test()
    
    # Generate visualization
    visualization_file = visualize_dynamic_batch_sizing(metrics)
    
    # Generate detailed report
    report = generate_dynamic_batch_report(metrics)
    
    print(f"\nVisualization saved to: {visualization_file}")
    print("Performance report saved to: dynamic_batch_report.txt")
    print("\nDynamic Batch Sizing Summary:")
    
    # Print a condensed summary
    avg_batch_size = np.mean(metrics["batch_sizes"]) if metrics["batch_sizes"] else 0
    num_adjustments = len(metrics["adjustments"])
    print(f"- Average batch size: {avg_batch_size:.2f}")
    print(f"- Number of batch size adjustments: {num_adjustments}")
    if metrics["adjustments"]:
        print(f"- Initial batch size: {metrics['batch_sizes'][0]}")
        print(f"- Final batch size: {metrics['batch_sizes'][-1]}")
    print(f"- See the full report for detailed analysis and recommendations.") 