import matplotlib.pyplot as plt
import numpy as np
import re
import os
import subprocess
from datetime import datetime

def run_test_and_extract_metrics(test_file):
    """Run the test and extract performance metrics."""
    # Run the test and capture output
    cmd = f"python {test_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    
    # Extract metrics using regex
    batch_times = re.findall(r"Total batch time: (\d+\.\d+)s", output)
    batch_times = [float(t) for t in batch_times]
    
    processing_rates = re.findall(r"Processing rate: (\d+\.\d+) samples/sec", output)
    processing_rates = [float(r) for r in processing_rates]
    
    inference_times = re.findall(r"Model inference time: (\d+\.\d+)s", output)
    inference_times = [float(t) for t in inference_times]
    
    post_times = re.findall(r"Post-processing time: (\d+\.\d+)s", output)
    post_times = [float(t) for t in post_times]
    
    # Check if batch dimensions were added
    dimension_fixes = len(re.findall(r"Adding missing batch dimension", output))
    
    return {
        "batch_times": batch_times,
        "processing_rates": processing_rates,
        "inference_times": inference_times,
        "post_times": post_times,
        "dimension_fixes": dimension_fixes,
        "raw_output": output
    }

def visualize_performance(metrics, title="CLIP Model Performance"):
    """Create visualizations for the performance metrics."""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Processing Rates by Batch
    ax1 = fig.add_subplot(221)
    batches = range(1, len(metrics["processing_rates"])+1)
    ax1.plot(batches, metrics["processing_rates"], 'o-', color='blue', linewidth=2)
    ax1.set_title("Processing Rates by Batch")
    ax1.set_xlabel("Batch Number")
    ax1.set_ylabel("Samples/Second")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Bar color based on performance (green for better)
    colors = ['green' if rate > 50 else 'orange' if rate > 20 else 'red' 
              for rate in metrics["processing_rates"]]
    
    # Plot 2: Processing Rates as Bar Chart
    ax2 = fig.add_subplot(222)
    ax2.bar(batches, metrics["processing_rates"], color=colors)
    ax2.set_title("Processing Rate Comparison")
    ax2.set_xlabel("Batch Number")
    ax2.set_ylabel("Samples/Second")
    ax2.axhline(y=np.mean(metrics["processing_rates"]), color='red', 
                linestyle='--', label=f"Avg: {np.mean(metrics['processing_rates']):.2f}")
    ax2.legend()
    
    # Plot 3: Time Breakdown per Batch
    ax3 = fig.add_subplot(223)
    width = 0.35
    inference = metrics["inference_times"]
    post = metrics["post_times"]
    other = [t - i - p for t, i, p in zip(metrics["batch_times"], inference, post)]
    
    ax3.bar(batches, inference, width, label='Inference Time')
    ax3.bar(batches, post, width, bottom=inference, label='Post-processing Time')
    ax3.bar(batches, other, width, bottom=[i+p for i, p in zip(inference, post)], label='Other Time')
    
    ax3.set_title("Time Breakdown per Batch")
    ax3.set_xlabel("Batch Number")
    ax3.set_ylabel("Time (seconds)")
    ax3.legend()
    
    # Plot 4: Batch Processing Time Trend
    ax4 = fig.add_subplot(224)
    ax4.plot(batches, metrics["batch_times"], 'o-', color='purple', linewidth=2)
    ax4.set_title("Batch Processing Time Trend")
    ax4.set_xlabel("Batch Number")
    ax4.set_ylabel("Time (seconds)")
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add batch dimension fix information
    annotation = f"Batch dimension fixes applied: {metrics['dimension_fixes']}"
    fig.text(0.5, 0.01, annotation, ha='center', fontsize=12)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.95, 0.01, f"Generated: {timestamp}", ha='right', fontsize=8)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("clip_performance_visualization.png", dpi=300)
    plt.show()
    
    return "clip_performance_visualization.png"

def generate_performance_summary(metrics):
    """Generate a text summary of performance metrics."""
    avg_rate = np.mean(metrics["processing_rates"])
    max_rate = np.max(metrics["processing_rates"])
    total_batches = len(metrics["batch_times"])
    avg_inference = np.mean(metrics["inference_times"])
    total_dimension_fixes = metrics["dimension_fixes"]
    
    summary = f"""
    CLIP Performance Summary
    =======================
    
    Overall Performance:
    - Average processing rate: {avg_rate:.2f} samples/second
    - Peak processing rate: {max_rate:.2f} samples/second
    - Batches processed: {total_batches}
    - Average inference time: {avg_inference:.4f} seconds
    
    Optimization Results:
    - Batch dimension fixes applied: {total_dimension_fixes}
    - Processing rate improvement: {(metrics["processing_rates"][-1] / metrics["processing_rates"][0] - 1) * 100:.1f}%
      (from {metrics["processing_rates"][0]:.2f} to {metrics["processing_rates"][-1]:.2f} samples/second)
    
    Time Efficiency:
    - Inference time makes up {100 * np.sum(metrics["inference_times"]) / np.sum(metrics["batch_times"]):.1f}% of total processing time
    - Post-processing makes up {100 * np.sum(metrics["post_times"]) / np.sum(metrics["batch_times"]):.1f}% of total processing time
    """
    
    # Write summary to file
    with open("clip_performance_summary.txt", "w") as f:
        f.write(summary)
    
    return summary

if __name__ == "__main__":
    print("Running performance visualization script...")
    # Path to the test file
    test_file = "clip-service/tests/test_stream_quick.py"
    
    # Run test and extract metrics
    metrics = run_test_and_extract_metrics(test_file)
    
    # Generate visualization
    visualization_file = visualize_performance(metrics)
    
    # Generate summary
    summary = generate_performance_summary(metrics)
    
    print(f"\nVisualization saved to: {visualization_file}")
    print("Performance summary saved to: clip_performance_summary.txt")
    print("\n" + summary) 