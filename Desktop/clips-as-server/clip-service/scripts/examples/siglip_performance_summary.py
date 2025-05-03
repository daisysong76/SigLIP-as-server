#!/usr/bin/env python3
"""
Generate a visual summary of SigLIP model performance based on saved results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def load_similarity_data(results_dir="./results"):
    """Load similarity data from the results directory."""
    similarity_file = os.path.join(results_dir, "similarity_details.txt")
    
    if not os.path.exists(similarity_file):
        print(f"Error: Similarity file not found at {similarity_file}")
        return None
    
    try:
        # Try the more robust parsing approach
        return parse_similarity_details(similarity_file)
    except Exception as e:
        print(f"Error parsing similarity details: {str(e)}")
        return []

def parse_similarity_details(similarity_file):
    """Correctly parse the similarity details file format."""
    examples = []
    current_example = None
    in_scores_section = False
    
    with open(similarity_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Example "):
                # Start a new example
                if current_example:
                    examples.append(current_example)
                current_example = {"id": int(line.split()[1]), "scores": [], "captions": []}
                in_scores_section = False
            elif line.startswith("Original Caption:") and current_example:
                current_example["original_caption"] = line[len("Original Caption:"):].strip()
            elif line.startswith("Top 5 similarity scores:") and current_example:
                in_scores_section = True
            elif in_scores_section and line.startswith("  ") and current_example:
                # Parse score line
                try:
                    parts = line.split("Score:", 1)
                    if len(parts) == 2:
                        rank_num = parts[0].strip().split('.')[0].strip()
                        score_part = parts[1].split(" ", 2)
                        
                        score = float(score_part[0].strip())
                        is_match = "(MATCH)" in score_part[1]
                        
                        # Get the caption
                        if len(score_part) >= 2:
                            caption_text = score_part[1]
                            if " - " in caption_text:
                                caption_text = caption_text.split(" - ", 1)[1]
                            else:
                                caption_text = caption_text.strip()
                        
                        current_example["scores"].append(score)
                        current_example["captions"].append(caption_text)
                        
                        if is_match:
                            current_example["match_index"] = len(current_example["scores"]) - 1
                            current_example["match_score"] = score
                except Exception as e:
                    print(f"Error parsing line: {line} - {str(e)}")
            elif line.startswith("---------") and in_scores_section:
                in_scores_section = False
    
    # Add the last example
    if current_example:
        examples.append(current_example)
    
    return examples

def generate_score_distribution_chart(examples, output_dir="./results"):
    """Generate a chart showing score distribution between matched and unmatched captions."""
    matched_scores = [ex["match_score"] for ex in examples if "match_score" in ex]
    
    # Collect all unmatched scores
    unmatched_scores = []
    for ex in examples:
        if "match_index" in ex:
            for i, score in enumerate(ex["scores"]):
                if i != ex["match_index"]:
                    unmatched_scores.append(score)
    
    # Skip if no scores found
    if not matched_scores or not unmatched_scores:
        print("Insufficient score data, skipping score distribution chart.")
        return
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        "Score": matched_scores + unmatched_scores,
        "Type": ["Correct Match"] * len(matched_scores) + ["Incorrect Match"] * len(unmatched_scores)
    })
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Type", y="Score", data=df, palette=["#2ecc71", "#e74c3c"])
    ax.set_title("SigLIP Similarity Score Distribution", fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Similarity Score", fontsize=14)
    
    # Add individual points
    sns.stripplot(x="Type", y="Score", data=df, color="black", alpha=0.5, jitter=True)
    
    # Add statistics
    plt.figtext(0.15, 0.01, f"Correct match avg: {np.mean(matched_scores):.4f}", fontsize=12)
    plt.figtext(0.55, 0.01, f"Incorrect match avg: {np.mean(unmatched_scores):.4f}", fontsize=12)
    
    # Save the plot
    output_path = os.path.join(output_dir, "score_distribution.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score distribution chart saved to {output_path}")

def generate_rank_analysis(examples, output_dir="./results"):
    """Generate a chart showing where the correct match ranks."""
    ranks = []
    for ex in examples:
        if "match_index" in ex:
            ranks.append(ex["match_index"] + 1)  # Convert to 1-indexed
    
    # Skip if no ranks found
    if not ranks:
        print("No rank data found, skipping rank analysis chart.")
        return
    
    # Count occurrences of each rank
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    # Create sorted data for plotting
    sorted_ranks = sorted(rank_counts.keys())
    counts = [rank_counts[r] for r in sorted_ranks]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sorted_ranks, counts, color='#3498db')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.0f}', ha='center', va='bottom')
    
    plt.title("Rank of Correct Caption in SigLIP Results", fontsize=16)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Number of Examples", fontsize=14)
    plt.xticks(sorted_ranks)
    plt.ylim(0, max(counts) + 1)
    
    # Add percentage annotations
    total = len(ranks)
    for i, rank in enumerate(sorted_ranks):
        percentage = (rank_counts[rank] / total) * 100
        plt.text(rank, counts[i]/2, f"{percentage:.1f}%", 
                 ha='center', va='center', color='white', fontweight='bold')
    
    # Save the plot
    output_path = os.path.join(output_dir, "rank_analysis.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Rank analysis chart saved to {output_path}")

def generate_similarity_matrix_visualization(examples, output_dir="./results", top_n=5):
    """Generate a heatmap visualization of the top similarity scores."""
    # Filter to examples with match_index
    valid_examples = [ex for ex in examples if "match_index" in ex]
    
    # Skip if insufficient examples
    if len(valid_examples) < 2:
        print("Insufficient examples with match data, skipping similarity matrix visualization.")
        return
    
    # Use at most top_n examples
    examples_to_use = valid_examples[:min(top_n, len(valid_examples))]
    
    # Prepare data
    data = []
    for ex in examples_to_use:
        for i, (score, caption) in enumerate(zip(ex["scores"], ex["captions"])):
            is_match = (i == ex.get("match_index", -1))
            # Truncate long captions
            short_caption = caption[:30] + "..." if len(caption) > 30 else caption
            data.append({
                "Example": f"Ex {ex['id']}",
                "Caption": f"{i+1}. {short_caption}",
                "Score": score,
                "Is Match": is_match
            })
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Create pivot table for heatmap
    pivot = df.pivot_table(index="Example", columns="Caption", values="Score", aggfunc='first')
    
    # Create mask for correct matches
    match_mask = np.zeros_like(pivot, dtype=bool)
    for i, ex in enumerate(examples_to_use):
        match_caption = f"{ex['match_index']+1}. {ex['captions'][ex['match_index']][:30] + '...' if len(ex['captions'][ex['match_index']]) > 30 else ex['captions'][ex['match_index']]}"
        if match_caption in pivot.columns:
            match_mask[i, pivot.columns.get_loc(match_caption)] = True
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
    
    # Highlight correct matches
    for i, j in zip(*np.where(match_mask)):
        heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=3))
    
    plt.title(f"SigLIP Similarity Scores Heatmap (Top {len(examples_to_use)} Examples)", fontsize=16)
    plt.ylabel("Query Example", fontsize=14)
    plt.xlabel("Retrieved Captions", fontsize=14)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot
    output_path = os.path.join(output_dir, "similarity_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Similarity heatmap saved to {output_path}")

def main():
    # Set default results directory
    results_dir = "./results"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    # Load data
    print(f"Loading similarity data from {results_dir}...")
    examples = load_similarity_data(results_dir)
    
    if not examples:
        print("No examples found or error loading data.")
        return 1
    
    print(f"Loaded {len(examples)} examples.")
    
    # Generate visualizations
    generate_score_distribution_chart(examples, results_dir)
    generate_rank_analysis(examples, results_dir)
    generate_similarity_matrix_visualization(examples, results_dir)
    
    print("\nAll visualizations generated successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 